"""
Generation-only and unified self-evolving experiments.

This module implements:
- generation_self_evolving
- unified_self_evolving (alternating understanding + generation)

Design goals:
- Single-codebase reproducibility (metadata, checkpoints, resumability)
- Role-specific adapters (solver / proposer / generator)
- Detailed ablation logs for proposer/spec/reward internals
"""

import dataclasses
import datetime as dt
import gc
import importlib
import inspect
import json
import math
import os
import pathlib
import random
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

from self_evolving.data.image_pool import ImagePool, ImagePoolConfig
import self_evolving.experiments.understanding as _u

DEFAULT_LORA_TARGETS = _u.DEFAULT_LORA_TARGETS
HAS_PEFT = _u.HAS_PEFT
HAS_WANDB = _u.HAS_WANDB
RolePolicyUpdater = _u.RolePolicyUpdater
_build_chat_text = _u._build_chat_text
_clip_grad_norm_multi_device = _u._clip_grad_norm_multi_device
_collect_git_info = _u._collect_git_info
_collect_trainable_params = _u._collect_trainable_params
_decode_tokens = _u._decode_tokens
_infer_primary_device = _u._infer_primary_device
_json_dump = _u._json_dump
_load_model_with_fallback = _u._load_model_with_fallback
_parse_answer = _u._parse_answer
_parse_first_question = _u._parse_first_question
_prepare_mm_inputs = _u._prepare_mm_inputs
_resolve_attn_implementation = _u._resolve_attn_implementation
_safe_dtype = _u._safe_dtype
_set_global_seed = _u._set_global_seed
_unwrap_model = _u._unwrap_model
build_proposer_prompt = _u.build_proposer_prompt
build_solver_prompt = _u.build_solver_prompt
gaussian_reward = _u.gaussian_reward
majority_vote = _u.majority_vote
normalize_answer = _u.normalize_answer
pre_answer_word_count = _u.pre_answer_word_count
shannon_entropy_nats = _u.shannon_entropy_nats
strip_tags = _u.strip_tags
use_adapter = _u.use_adapter
LoraConfig = getattr(_u, "LoraConfig", None)
TaskType = getattr(_u, "TaskType", None)
get_peft_model = getattr(_u, "get_peft_model", None)

if HAS_WANDB:
    import wandb


GEN_PROMPT_TEMPLATE = (
    "You are a generation-spec proposer for self-evolving training.\n"
    "Given the source image, propose one new text-to-image prompt and verification QA pairs.\n"
    "Rules:\n"
    "- Prompt must be image-grounded but not a trivial copy.\n"
    "- QA pairs must be short-answer and visually verifiable.\n"
    "- Expected answers must be concise.\n"
    "Output XML only:\n"
    "<prompt>...</prompt>\n"
    "<spec>\n"
    "  <qa><question>...</question><expected>...</expected></qa>\n"
    "  <qa><question>...</question><expected>...</expected></qa>\n"
    "  <qa><question>...</question><expected>...</expected></qa>\n"
    "</spec>"
)

SOURCE_CAPTION_PROMPT = (
    "Describe this image in one concise sentence with key entities, attributes, and scene context."
)

GEN_CYCLE_CAPTION_PROMPT = (
    "Describe this image in one concise sentence focusing on key visual facts."
)

GENERATOR_PROXY_CAPTION_PROMPT = (
    "Describe this image in one concise sentence with key objects, attributes, and relations."
)


@dataclass
class GenerationQAPair:
    question: str
    expected: str


@dataclass
class GenerationSpec:
    prompt: str
    qa_pairs: Tuple[GenerationQAPair, ...]
    raw_output: str
    fallback_used: bool


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _jaccard_similarity(a: str, b: str) -> float:
    ta = set(_tokenize_words(a))
    tb = set(_tokenize_words(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return float(inter) / float(max(1, union))


def _parse_float_safe(text: str) -> Optional[float]:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _soft_match(pred: str, expected: str) -> float:
    p = normalize_answer(pred)
    e = normalize_answer(expected)
    if not e:
        return 0.5
    if p == e:
        return 1.0
    if p and e and (p in e or e in p):
        return 0.8

    pn = _parse_float_safe(p)
    en = _parse_float_safe(e)
    if pn is not None and en is not None:
        den = max(abs(en), 1.0)
        rel = abs(pn - en) / den
        if rel <= 0.01:
            return 1.0
        if rel <= 0.05:
            return 0.7
        if rel <= 0.20:
            return 0.4
        return 0.0

    return _jaccard_similarity(p, e)


def _yes_no_polarity(text: str) -> int:
    t = normalize_answer(text)
    if t.startswith("yes") or t in {"true", "1", "present"}:
        return 1
    if t.startswith("no") or t in {"false", "0", "absent"}:
        return -1
    return 0


def _image_diversity_score(images: List[Image.Image]) -> float:
    if len(images) <= 1:
        return 0.5
    if not HAS_NUMPY:
        return 0.5

    vectors = []
    for image in images:
        arr = np.asarray(image.convert("RGB").resize((48, 48)), dtype=np.float32) / 255.0
        vectors.append(arr.reshape(-1))

    dists = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            d = float(np.mean(np.abs(vectors[i] - vectors[j])))
            dists.append(d)
    if not dists:
        return 0.5

    # Typical mean abs RGB diff lies near [0, ~0.35]. Scale to [0,1].
    scaled = min(1.0, max(0.0, (sum(dists) / len(dists)) / 0.25))
    return scaled


def _ensure_pil_image(image_obj) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj
    if HAS_NUMPY and isinstance(image_obj, np.ndarray):
        arr = image_obj
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            h, w = int(arr.shape[0]), int(arr.shape[1])
            ratio = max(h, w) / float(max(1, min(h, w)))
            if ratio > 6.0:
                raise TypeError(f"Array has implausible image aspect ratio for direct conversion: shape={arr.shape}")
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
            return Image.fromarray(arr)
    if torch.is_tensor(image_obj):
        tensor = image_obj.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3:
            if tensor.shape[0] in (1, 3, 4):
                h, w = int(tensor.shape[1]), int(tensor.shape[2])
                ratio = max(h, w) / float(max(1, min(h, w)))
                if ratio > 6.0:
                    raise TypeError(
                        f"Tensor looks like latent/features, not an image (shape={tuple(tensor.shape)})."
                    )
                tensor = tensor.permute(1, 2, 0)
            elif tensor.shape[-1] in (1, 3, 4):
                h, w = int(tensor.shape[0]), int(tensor.shape[1])
                ratio = max(h, w) / float(max(1, min(h, w)))
                if ratio > 6.0:
                    raise TypeError(
                        f"Tensor looks like latent/features, not an image (shape={tuple(tensor.shape)})."
                    )
            else:
                raise TypeError(f"Tensor does not have image-like channel layout: shape={tuple(tensor.shape)}")
            if tensor.dtype in (torch.bfloat16, torch.float16):
                tensor = tensor.to(torch.float32)
            arr = tensor.numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
            return Image.fromarray(arr)
    raise TypeError(f"Unsupported generated image type: {type(image_obj)}")


def _latent_tensor_to_pil(image_obj, target_size: Optional[Tuple[int, int]] = None) -> Optional[Image.Image]:
    """
    Deterministic fallback visualization for latent tensors when a decoder pipeline is unavailable.
    This keeps training/evaluation loops running instead of hard-failing.
    """
    if not (HAS_NUMPY and torch.is_tensor(image_obj)):
        return None

    t = image_obj.detach().to(torch.float32).cpu()
    if t.ndim == 4:
        t = t[0]
    if t.ndim == 3 and t.shape[0] == 1:
        t = t[0]

    arr_t: Optional[torch.Tensor] = None

    # Case A: token-latent layout [N, C], e.g. [64, 1792]
    if t.ndim == 2:
        n, c = int(t.shape[0]), int(t.shape[1])
        s = int(round(math.sqrt(float(n))))
        if s * s == n:
            arr_t = t.view(s, s, c)
        else:
            arr_t = t.view(n, 1, c)
    # Case B: channel-first latent [C, H, W]
    elif t.ndim == 3:
        if t.shape[0] in (1, 3, 4):
            arr_t = t.permute(1, 2, 0).contiguous()
        elif t.shape[-1] in (1, 3, 4):
            arr_t = t.contiguous()
        else:
            # Treat as [C, H, W] latent map.
            arr_t = t.permute(1, 2, 0).contiguous()
    else:
        return None

    c = int(arr_t.shape[-1])
    if c == 1:
        arr_t = arr_t.repeat(1, 1, 3)
    elif c == 2:
        arr_t = torch.cat([arr_t, arr_t[..., :1]], dim=-1)
    elif c > 3:
        chunks = torch.chunk(arr_t, 3, dim=-1)
        arr_t = torch.stack([ch.mean(dim=-1) for ch in chunks], dim=-1)

    mn = float(arr_t.min().item())
    mx = float(arr_t.max().item())
    if mx <= mn:
        arr_t = torch.zeros_like(arr_t)
    else:
        arr_t = (arr_t - mn) / (mx - mn)

    arr = np.clip(arr_t.numpy() * 255.0, 0.0, 255.0).astype(np.uint8)
    img = Image.fromarray(arr)
    if target_size is not None:
        try:
            tw, th = int(target_size[0]), int(target_size[1])
            if tw > 0 and th > 0:
                img = img.resize((tw, th), Image.BILINEAR)
        except Exception:
            pass
    return img


def _parse_generation_spec(raw_text: str) -> GenerationSpec:
    prompt_text = (strip_tags(raw_text, "prompt") or "").strip()

    qa_pairs: List[GenerationQAPair] = []
    pattern = re.compile(
        r"<qa>\s*<question>(.*?)</question>\s*<expected>(.*?)</expected>\s*</qa>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for q, e in pattern.findall(raw_text):
        qn = " ".join(q.strip().split())
        en = " ".join(e.strip().split())
        if qn:
            qa_pairs.append(GenerationQAPair(question=qn, expected=en))

    # Fallback parsing for line-based outputs.
    if not qa_pairs:
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        if not prompt_text:
            for line in lines:
                if line.lower().startswith("prompt:"):
                    prompt_text = line.split(":", 1)[1].strip()
                    break

        pending_q: Optional[str] = None
        for line in lines:
            lower = line.lower()
            if lower.startswith("q") and ":" in line:
                pending_q = line.split(":", 1)[1].strip()
            elif lower.startswith("a") and ":" in line and pending_q:
                ans = line.split(":", 1)[1].strip()
                qa_pairs.append(GenerationQAPair(question=pending_q, expected=ans))
                pending_q = None

    fallback = False
    if not prompt_text:
        fallback = True
        prompt_text = "Generate a realistic image with clear salient objects and readable details."

    if not qa_pairs:
        fallback = True

    return GenerationSpec(
        prompt=prompt_text,
        qa_pairs=tuple(qa_pairs[:3]),
        raw_output=raw_text,
        fallback_used=fallback,
    )


def _prepare_text_inputs(processor, device: torch.device, text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    return inputs.to(device)


def _signature_param_names(obj, fn_name: str) -> Tuple[set, bool]:
    fn = getattr(obj, fn_name, None)
    if not callable(fn):
        return set(), False
    try:
        sig = inspect.signature(fn)
    except Exception:
        return set(), False
    names = set()
    has_var_kw = False
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kw = True
            continue
        names.add(p.name)
    return names, has_var_kw


def _supports_kwarg_anywhere(model, kwarg_name: str) -> bool:
    try:
        wrappers = _iter_wrapper_objects(model)
    except Exception:
        wrappers = [("model", model)]
    for _, obj in wrappers:
        for fn_name in ("generate", "prepare_inputs_for_generation", "forward"):
            names, has_var_kw = _signature_param_names(obj, fn_name)
            if has_var_kw or kwarg_name in names:
                return True
    return False


def _adapt_mm_generate_inputs(model, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Normalize multimodal kwargs across BLIP3o/Qwen wrappers.
    """
    out = dict(inputs)
    supports_images = _supports_kwarg_anywhere(model, "images")
    supports_pixel_values = _supports_kwarg_anywhere(model, "pixel_values")
    supports_grid_thw = _supports_kwarg_anywhere(model, "grid_thw")
    supports_image_grid_thw = _supports_kwarg_anywhere(model, "image_grid_thw")

    if "images" in out and (not supports_images) and supports_pixel_values and "pixel_values" not in out:
        out["pixel_values"] = out.pop("images")
    if "pixel_values" in out and (not supports_pixel_values) and supports_images and "images" not in out:
        out["images"] = out.pop("pixel_values")

    if "image_grid_thw" in out and (not supports_image_grid_thw) and supports_grid_thw and "grid_thw" not in out:
        out["grid_thw"] = out.pop("image_grid_thw")
    if "grid_thw" in out and (not supports_grid_thw) and supports_image_grid_thw and "image_grid_thw" not in out:
        out["image_grid_thw"] = out.pop("grid_thw")
    return out


def _parse_unused_model_kwargs_from_error(exc: Exception) -> List[str]:
    msg = str(exc)
    m = re.search(r"model_kwargs` are not used by the model: \[(.*?)\]", msg)
    if not m:
        return []
    raw = m.group(1).strip()
    if not raw:
        return []
    parts = [p.strip().strip("'").strip('"') for p in raw.split(",")]
    return [p for p in parts if p]


def _collect_image_token_ids(model) -> List[int]:
    ids: List[int] = []

    def _collect_from_cfg(cfg):
        if cfg is None:
            return
        for name in ("image_token_id", "image_token_index", "vision_token_id"):
            value = getattr(cfg, name, None)
            if isinstance(value, int):
                ids.append(int(value))
            elif isinstance(value, (list, tuple)):
                for v in value:
                    if isinstance(v, int):
                        ids.append(int(v))

    model_ref = _unwrap_model(model)
    cfg = getattr(model_ref, "config", None)
    _collect_from_cfg(cfg)
    _collect_from_cfg(getattr(cfg, "text_config", None) if cfg is not None else None)
    return sorted(set(ids))


def _count_image_tokens_in_inputs(input_ids: torch.Tensor, image_token_ids: List[int]) -> int:
    if input_ids is None or not torch.is_tensor(input_ids) or not image_token_ids:
        return 0
    total = 0
    for tok_id in image_token_ids:
        total += int((input_ids == int(tok_id)).sum().item())
    return int(total)


def _is_original_blip3o_model_name(model_name: str) -> bool:
    name = (model_name or "").strip().lower()
    return "blip3o-model" in name and "next" not in name


def _is_blip3o_next_model_name(model_name: str) -> bool:
    name = (model_name or "").strip().lower()
    return "blip3o-next" in name


def _looks_like_unregistered_blip3o_arch_error(text: str) -> bool:
    t = (text or "").lower()
    return ("blip3o_qwen" in t) and ("does not recognize this architecture" in t or "unrecognized" in t)


def _is_next_style_blip3o_class(cls) -> bool:
    """
    Detect BLIP3o-NEXT style classes (Qwen3-based) that are incompatible with
    original BLIP3o-Model checkpoints (Qwen2.5-VL based).
    """
    try:
        for base in cls.mro():
            name = getattr(base, "__name__", "").lower()
            if "qwen3" in name:
                return True
    except Exception:
        pass
    module_name = str(getattr(cls, "__module__", "")).lower()
    return "qwen3" in module_name


def _maybe_add_local_blip3o_path(*, allow_implicit_repo: bool = False) -> Optional[str]:
    """
    Add a local BLIP3o source tree to sys.path if present.
    """
    candidates: List[pathlib.Path] = []
    env_repo = os.environ.get("BLIP3O_REPO", "").strip()
    if env_repo:
        candidates.append(pathlib.Path(env_repo).expanduser())
    if allow_implicit_repo:
        # Common in-repo locations (disabled by default to avoid accidentally
        # loading BLIP3o-NEXT local code for original BLIP3o checkpoints).
        try:
            here = pathlib.Path(__file__).resolve()
            repo_root = here.parents[2]
            candidates.append(repo_root / "BLIP3o")
        except Exception:
            pass
        candidates.append(pathlib.Path.cwd() / "BLIP3o")

    for cand in candidates:
        try:
            cand_resolved = cand.resolve()
        except Exception:
            continue
        if not (cand_resolved / "blip3o").is_dir():
            continue
        cand_str = str(cand_resolved)
        if cand_str not in sys.path:
            sys.path.insert(0, cand_str)
        return cand_str
    return None


def _import_blip3o_classes(*, allow_implicit_repo: bool = False):
    """
    Import BLIP3o model classes (inference/grpo/causal) if available.
    Returns (classes_dict, error_or_none, added_path_or_none).
    """
    last_exc = None
    added_path = _maybe_add_local_blip3o_path(allow_implicit_repo=allow_implicit_repo)
    explicit_repo = os.environ.get("BLIP3O_REPO", "").strip()
    explicit_repo_resolved: Optional[pathlib.Path] = None
    if explicit_repo:
        try:
            explicit_repo_resolved = pathlib.Path(explicit_repo).expanduser().resolve()
        except Exception:
            explicit_repo_resolved = None

    def _purge_blip3o_modules():
        for key in list(sys.modules.keys()):
            if key == "blip3o" or key.startswith("blip3o."):
                sys.modules.pop(key, None)

    def _module_matches_explicit_repo(module) -> bool:
        if explicit_repo_resolved is None:
            return True
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return False
        try:
            resolved = pathlib.Path(module_file).resolve()
            resolved_str = str(resolved)
            repo_str = str(explicit_repo_resolved)
            return resolved_str == repo_str or resolved_str.startswith(repo_str + os.sep)
        except Exception:
            return False

    if explicit_repo_resolved is not None:
        # Avoid stale pre-imported `blip3o` modules from another location (e.g. site-packages).
        _purge_blip3o_modules()

    def _patch_gradient_checkpointing_compat():
        try:
            lumina_mod = importlib.import_module("blip3o.model.lumina_nextdit2d")
            lumina_cls = getattr(lumina_mod, "LuminaNextDiT2DModel", None)
            if lumina_cls is None:
                return
            fn = getattr(lumina_cls, "_set_gradient_checkpointing", None)
            if fn is None:
                return
            sig = inspect.signature(fn)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            has_gcf = "gradient_checkpointing_func" in sig.parameters
            if has_var_kw and has_gcf:
                return

            def _compat_set_gradient_checkpointing(
                self,
                module=None,
                value=False,
                enable=None,
                gradient_checkpointing_func=None,
                **kwargs,
            ):
                if enable is not None:
                    value = bool(enable)
                if module is None:
                    module = self
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = bool(value)
                if gradient_checkpointing_func is not None and hasattr(module, "_gradient_checkpointing_func"):
                    module._gradient_checkpointing_func = gradient_checkpointing_func

            setattr(lumina_cls, "_set_gradient_checkpointing", _compat_set_gradient_checkpointing)
        except Exception:
            return

    for _ in range(2):
        try:
            module = importlib.import_module("blip3o.model")
            _patch_gradient_checkpointing_compat()
            if not _module_matches_explicit_repo(module):
                _purge_blip3o_modules()
                added_path = _maybe_add_local_blip3o_path(allow_implicit_repo=allow_implicit_repo)
                module = importlib.import_module("blip3o.model")
                _patch_gradient_checkpointing_compat()
                if not _module_matches_explicit_repo(module):
                    module_path = getattr(module, "__file__", "unknown")
                    raise RuntimeError(
                        "Imported `blip3o.model` from an unexpected location while BLIP3O_REPO is set. "
                        f"BLIP3O_REPO={explicit_repo_resolved}, imported_from={module_path}"
                    )
            return (
                {
                    "inference": getattr(module, "blip3oQwenForInferenceLM", None),
                    "grpo": getattr(module, "blip3oQwenForGRPOLM", None),
                    "causal": getattr(module, "blip3oQwenForCausalLM", None),
                },
                None,
                added_path,
            )
        except Exception as exc:
            last_exc = exc
            added_path = _maybe_add_local_blip3o_path(allow_implicit_repo=allow_implicit_repo)
    return {}, last_exc, added_path


def _load_from_explicit_class(
    cls,
    model_name: str,
    *,
    torch_dtype: torch.dtype,
    device_map,
    attn_implementation: Optional[str],
):
    errors: List[str] = []
    attempts: List[Dict[str, object]] = []
    base_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        attempts.append({**base_kwargs, "attn_implementation": attn_implementation})
    attempts.append(base_kwargs)

    def _maybe_patch_config_hidden_size():
        """
        BLIP3o original classes sometimes expect `config.hidden_size` while newer
        transformers expose size only under `config.text_config.hidden_size`.
        """
        try:
            from transformers import AutoConfig
        except Exception:
            return None
        try:
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            return None
        if hasattr(cfg, "hidden_size"):
            return cfg

        text_cfg = getattr(cfg, "text_config", None)
        hidden_size = None
        if text_cfg is not None:
            hidden_size = getattr(text_cfg, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(text_cfg, "d_model", None)
        if hidden_size is None:
            return None

        try:
            setattr(cfg, "hidden_size", int(hidden_size))
            return cfg
        except Exception:
            return None

    for kwargs in attempts:
        try:
            return cls.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            errors.append(f"{cls.__name__}({kwargs}): {repr(exc)}")
            msg = str(exc).lower()
            if ("hidden_size" in msg) and ("attributeerror" in msg or "no attribute" in msg):
                patched_cfg = _maybe_patch_config_hidden_size()
                if patched_cfg is None:
                    continue
                retry_kwargs = dict(kwargs)
                retry_kwargs["config"] = patched_cfg
                try:
                    return cls.from_pretrained(model_name, **retry_kwargs)
                except Exception as exc_retry:
                    errors.append(
                        f"{cls.__name__}({retry_kwargs}): {repr(exc_retry)}"
                    )
    raise RuntimeError(" | ".join(errors))


def _import_blip3o_mm_utils(*, allow_implicit_repo: bool = False):
    """
    Import BLIP3o multimodal helpers if available.
    """
    last_exc = None
    added_path = _maybe_add_local_blip3o_path(allow_implicit_repo=allow_implicit_repo)
    explicit_repo = os.environ.get("BLIP3O_REPO", "").strip()
    explicit_repo_resolved: Optional[pathlib.Path] = None
    if explicit_repo:
        try:
            explicit_repo_resolved = pathlib.Path(explicit_repo).expanduser().resolve()
        except Exception:
            explicit_repo_resolved = None

    def _purge_blip3o_modules():
        for key in list(sys.modules.keys()):
            if key == "blip3o" or key.startswith("blip3o."):
                sys.modules.pop(key, None)

    def _module_matches_explicit_repo(module) -> bool:
        if explicit_repo_resolved is None:
            return True
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return False
        try:
            resolved = pathlib.Path(module_file).resolve()
            resolved_str = str(resolved)
            repo_str = str(explicit_repo_resolved)
            return resolved_str == repo_str or resolved_str.startswith(repo_str + os.sep)
        except Exception:
            return False

    if explicit_repo_resolved is not None:
        _purge_blip3o_modules()

    for _ in range(2):
        try:
            module = importlib.import_module("blip3o.mm_utils")
            if not _module_matches_explicit_repo(module):
                _purge_blip3o_modules()
                added_path = _maybe_add_local_blip3o_path(allow_implicit_repo=allow_implicit_repo)
                module = importlib.import_module("blip3o.mm_utils")
                if not _module_matches_explicit_repo(module):
                    module_path = getattr(module, "__file__", "unknown")
                    raise RuntimeError(
                        "Imported `blip3o.mm_utils` from an unexpected location while BLIP3O_REPO is set. "
                        f"BLIP3O_REPO={explicit_repo_resolved}, imported_from={module_path}"
                    )
            return (
                {
                    "tokenizer_image_token": getattr(module, "tokenizer_image_token", None),
                },
                None,
                added_path,
            )
        except Exception as exc:
            last_exc = exc
            added_path = _maybe_add_local_blip3o_path(allow_implicit_repo=allow_implicit_repo)
    return {}, last_exc, added_path


def _link_or_copy_file(src: pathlib.Path, dst: pathlib.Path):
    if dst.exists():
        return
    try:
        os.symlink(str(src), str(dst))
        return
    except Exception:
        pass
    try:
        os.link(str(src), str(dst))
        return
    except Exception:
        pass
    shutil.copy2(str(src), str(dst))


def _ensure_diffusers_component_weight_aliases(component_dir: pathlib.Path):
    """
    Some BLIP3o diffusion checkpoints can contain model weights under generic names
    (`pytorch_model.*`) while the installed diffusers loader expects
    `diffusion_pytorch_model.*`.
    Create lightweight aliases when needed.
    """
    if not component_dir.is_dir():
        return
    target_sf = component_dir / "diffusion_pytorch_model.safetensors"
    target_bin = component_dir / "diffusion_pytorch_model.bin"
    if target_sf.exists() or target_bin.exists():
        return

    sf_candidates: List[pathlib.Path] = []
    bin_candidates: List[pathlib.Path] = []

    preferred_sf = component_dir / "pytorch_model.safetensors"
    preferred_bin = component_dir / "pytorch_model.bin"
    if preferred_sf.is_file():
        sf_candidates.append(preferred_sf)
    if preferred_bin.is_file():
        bin_candidates.append(preferred_bin)

    # Fall back to any model-like file in this component directory.
    for p in sorted(component_dir.glob("*.safetensors")):
        if p.is_file():
            sf_candidates.append(p)
    for p in sorted(component_dir.glob("*.bin")):
        if p.is_file():
            bin_candidates.append(p)

    # Deduplicate while preserving order.
    sf_candidates = list(dict.fromkeys(sf_candidates))
    bin_candidates = list(dict.fromkeys(bin_candidates))

    if sf_candidates:
        try:
            _link_or_copy_file(sf_candidates[0], target_sf)
            return
        except Exception:
            pass
    if bin_candidates:
        try:
            _link_or_copy_file(bin_candidates[0], target_bin)
        except Exception:
            pass


def _build_original_blip3o_diffusion_pipeline(
    model_name: str,
    *,
    multimodal_encoder,
    processor,
    torch_dtype: torch.dtype,
    device: torch.device,
):
    """
    Build original BLIP3o diffusion decoder pipeline from HF model repo.

    Original BLIP3o (`BLIP3o/BLIP3o-Model-8B`) uses a diffusion-decoder
    subfolder with custom pipelines (e.g. `pipeline_llava_gen.py`).
    """
    try:
        from diffusers import DiffusionPipeline
    except Exception as exc:
        raise RuntimeError(
            "diffusers is required for original BLIP3o diffusion-decoder backend."
        ) from exc

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None and hasattr(processor, "tokenizer_image_token"):
        tokenizer = processor
    if tokenizer is None:
        raise RuntimeError("Processor does not expose tokenizer required by BLIP3o diffusion pipeline.")

    attempts = (
        "pipeline_llava_gen",
        "pipeline_ar_gen",
    )
    repo_candidates: List[str] = []
    primary_repo = str(model_name or "").strip()
    if primary_repo:
        repo_candidates.append(primary_repo)

    # Original BLIP3o README notes the diffusion decoder under BLIP3o/BLIP3o-Model.
    # BLIP3o-Model-8B checkpoints may expose text+vision model weights without this
    # decoder subtree, so add explicit fallback source.
    lowered = primary_repo.lower()
    if "blip3o-model-8b" in lowered:
        repo_candidates.append("BLIP3o/BLIP3o-Model")

    extra_decoder_repo = os.environ.get("BLIP3O_DIFFUSION_REPO", "").strip()
    if extra_decoder_repo:
        repo_candidates.append(extra_decoder_repo)

    # Preserve order while removing duplicates.
    repo_candidates = list(dict.fromkeys(repo_candidates))

    errors: List[str] = []
    pipe = None
    for repo_id in repo_candidates:
        for custom_pipeline in attempts:
            try:
                try:
                    pipe = DiffusionPipeline.from_pretrained(
                        repo_id,
                        custom_pipeline=custom_pipeline,
                        subfolder="diffusion-decoder",
                        tokenizer=tokenizer,
                        multimodal_encoder=multimodal_encoder,
                        safety_checker=None,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="bf16",
                    )
                except TypeError:
                    pipe = DiffusionPipeline.from_pretrained(
                        repo_id,
                        custom_pipeline=custom_pipeline,
                        subfolder="diffusion-decoder",
                        tokenizer=tokenizer,
                        multimodal_encoder=multimodal_encoder,
                        safety_checker=None,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="bf16",
                    )
                break
            except Exception as exc:
                errors.append(f"{repo_id}:{custom_pipeline}: {repr(exc)}")
        if pipe is not None:
            break

    # Fallback: older diffusers versions can incorrectly resolve `model_index.json`
    # at repo root even when `subfolder` is provided. Mirror BLIP3o official usage by
    # resolving a local `diffusion-decoder` path, then loading from that directory.
    if pipe is None:
        try:
            from huggingface_hub import snapshot_download

            for repo_id in repo_candidates:
                local_repo = pathlib.Path(
                    snapshot_download(
                        repo_id=repo_id,
                        allow_patterns=[
                            "diffusion-decoder/**",
                            "pipeline_llava_gen.py",
                            "pipeline_ar_gen.py",
                        ],
                    )
                )
                diff_dir = local_repo / "diffusion-decoder"
                if diff_dir.is_dir():
                    # Normalize component weight filenames for broader diffusers compatibility.
                    for component in sorted(diff_dir.iterdir()):
                        if component.is_dir():
                            _ensure_diffusers_component_weight_aliases(component)
                    for custom_pipeline in attempts:
                        cp_arg: object = custom_pipeline
                        root_cp = local_repo / f"{custom_pipeline}.py"
                        sub_cp = diff_dir / f"{custom_pipeline}.py"
                        if root_cp.is_file():
                            cp_arg = str(root_cp)
                        elif sub_cp.is_file():
                            cp_arg = str(sub_cp)
                        try:
                            try:
                                pipe = DiffusionPipeline.from_pretrained(
                                    str(diff_dir),
                                    custom_pipeline=cp_arg,
                                    tokenizer=tokenizer,
                                    multimodal_encoder=multimodal_encoder,
                                    safety_checker=None,
                                    trust_remote_code=True,
                                    torch_dtype=torch_dtype,
                                    use_safetensors=True,
                                    variant="bf16",
                                )
                            except TypeError:
                                pipe = DiffusionPipeline.from_pretrained(
                                    str(diff_dir),
                                    custom_pipeline=cp_arg,
                                    tokenizer=tokenizer,
                                    multimodal_encoder=multimodal_encoder,
                                    safety_checker=None,
                                    torch_dtype=torch_dtype,
                                    use_safetensors=True,
                                    variant="bf16",
                                )
                            break
                        except Exception as exc:
                            errors.append(f"local_snapshot:{repo_id}:{custom_pipeline}: {repr(exc)}")
                    if pipe is not None:
                        break
                else:
                    errors.append(
                        f"local_snapshot:{repo_id}: missing diffusion-decoder dir under {local_repo}"
                    )
            # end for repo_id
        except Exception as exc:
            errors.append(f"local_snapshot: {repr(exc)}")

    if pipe is None:
        detail = " | ".join(errors)
        raise RuntimeError(f"Failed to build BLIP3o diffusion pipeline. {detail}")

    try:
        pipe = pipe.to(device)
    except Exception:
        # Some pipeline objects are partially device-managed internally.
        pass
    return pipe


def _resolve_multimodal_encoder_for_pipeline(model):
    """
    Select the object that exposes `generate_image` for original BLIP3o pipelines.
    """
    try:
        for _, obj in _iter_wrapper_objects(model):
            if callable(getattr(obj, "generate_image", None)):
                return obj
    except Exception:
        pass
    return model


class _Blip3oProcessorShim:
    """
    Minimal processor shim for BLIP3o checkpoints that expose tokenizer + image processor separately.
    """

    def __init__(
        self,
        tokenizer,
        image_processor=None,
        model_name: str = "",
        multimodal_processor=None,
        tokenizer_image_token_fn: Optional[Callable[..., torch.Tensor]] = None,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_name = str(model_name or "")
        self.multimodal_processor = multimodal_processor
        self._tokenizer_image_token_fn = tokenizer_image_token_fn
        # Some generic paths look for these attributes.
        self.chat_template = getattr(tokenizer, "chat_template", None)
        if self.chat_template is None and self.multimodal_processor is not None:
            self.chat_template = getattr(self.multimodal_processor, "chat_template", None)

    def _ensure_image_processor(self):
        if self.image_processor is not None:
            return self.image_processor
        if self.multimodal_processor is not None:
            ip = getattr(self.multimodal_processor, "image_processor", None)
            if ip is not None:
                self.image_processor = ip
                return self.image_processor
        self.image_processor = _load_blip3o_image_processor_from_candidates(
            model_name=self.model_name
        )
        return self.image_processor

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def _image_text_placeholder(self) -> str:
        """
        Choose a textual image placeholder compatible with the underlying tokenizer.
        """
        try:
            to_id = getattr(self.tokenizer, "convert_tokens_to_ids", None)
            if callable(to_id):
                img_pad_id = to_id("<|image_pad|>")
                unk_id = getattr(self.tokenizer, "unk_token_id", None)
                if isinstance(img_pad_id, int) and img_pad_id >= 0 and (unk_id is None or img_pad_id != unk_id):
                    return "<|vision_start|><|image_pad|><|vision_end|>"
        except Exception:
            pass
        return "<image>"

    def _normalize_messages_multimodal(self, messages) -> List[Dict[str, object]]:
        normalized: List[Dict[str, object]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts: List[Dict[str, str]] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    typ = item.get("type")
                    if typ == "image":
                        # Keep multimodal structure but drop raw image payload.
                        parts.append({"type": "image"})
                    elif typ == "text":
                        parts.append({"type": "text", "text": str(item.get("text", ""))})
                normalized.append({"role": role, "content": parts})
            else:
                normalized.append({"role": role, "content": str(content)})
        return normalized

    def _normalize_messages_text(self, messages) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        image_token = self._image_text_placeholder()
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    typ = item.get("type")
                    if typ == "image":
                        parts.append(image_token)
                    elif typ == "text":
                        parts.append(str(item.get("text", "")))
                content = "\n".join([p for p in parts if p]).strip()
            normalized.append({"role": role, "content": str(content)})
        return normalized

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if hasattr(self.multimodal_processor, "apply_chat_template"):
            try:
                normalized_mm = self._normalize_messages_multimodal(messages)
                return self.multimodal_processor.apply_chat_template(
                    normalized_mm,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                pass

        if hasattr(self.tokenizer, "apply_chat_template"):
            # First try multimodal message structure so Qwen-VL inserts image tokens.
            try:
                normalized_mm = self._normalize_messages_multimodal(messages)
                return self.tokenizer.apply_chat_template(
                    normalized_mm,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                pass
            # Fall back to text-only template with explicit image placeholder token.
            normalized_text = self._normalize_messages_text(messages)
            return self.tokenizer.apply_chat_template(
                normalized_text,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )

        normalized = self._normalize_messages_text(messages)

        text = ""
        for msg in normalized:
            text += f"{msg['role']}\n{msg['content']}\n"
        if add_generation_prompt:
            text += "assistant\n"
        if tokenize:
            return self.tokenizer(text).input_ids
        return text

    def _tokenize_with_image_placeholder(self, text_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        fn = self._tokenizer_image_token_fn
        if fn is None:
            batch = self.tokenizer(text_list, return_tensors="pt", padding=True)
            return batch["input_ids"], batch["attention_mask"]

        ids_list = []
        for txt in text_list:
            ids = fn(txt, self.tokenizer, return_tensors="pt")
            if ids.ndim == 1:
                ids_list.append(ids)
            else:
                ids_list.append(ids.squeeze(0))

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)
        max_len = max(int(x.numel()) for x in ids_list)
        input_ids = torch.full((len(ids_list), max_len), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
        for i, ids in enumerate(ids_list):
            ln = int(ids.numel())
            input_ids[i, :ln] = ids
            attention_mask[i, :ln] = 1
        return input_ids, attention_mask

    def __call__(self, text=None, images=None, return_tensors="pt", padding=True, **kwargs):
        from transformers import BatchEncoding

        text_list = text if isinstance(text, list) else [str(text or "")]

        if images is not None and self.multimodal_processor is not None:
            img_list = images if isinstance(images, list) else [images]
            try:
                return self.multimodal_processor(
                    text=text_list,
                    images=img_list,
                    return_tensors=return_tensors,
                    padding=padding,
                    **kwargs,
                )
            except Exception:
                # Fall through to lightweight shim path.
                pass

        use_image_placeholder = any("<image>" in t for t in text_list)
        if use_image_placeholder:
            input_ids, attention_mask = self._tokenize_with_image_placeholder(text_list)
            data: Dict[str, torch.Tensor] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        else:
            tok = self.tokenizer(text_list, return_tensors=return_tensors, padding=padding, **kwargs)
            data = {
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
            }

        if images is not None:
            if self.image_processor is None:
                self._ensure_image_processor()
            if self.image_processor is None:
                raise RuntimeError(
                    "BLIP3o processor shim missing image_processor; cannot prepare multimodal inputs. "
                    "Set BLIP3O_IMAGE_PROCESSOR_ID (e.g., Qwen/Qwen2.5-VL-7B-Instruct) or ensure "
                    "the local BLIP3o vision tower exposes `image_processor`."
                )
            img_list = images if isinstance(images, list) else [images]
            prep = self.image_processor.preprocess(img_list, return_tensors="pt")
            if "pixel_values" in prep:
                data["pixel_values"] = prep["pixel_values"]
            elif "images" in prep:
                data["images"] = prep["images"]
            else:
                raise RuntimeError("BLIP3o image_processor preprocess output missing pixel data.")
            if "image_grid_thw" in prep:
                data["image_grid_thw"] = prep["image_grid_thw"]
            if "grid_thw" in prep:
                data["grid_thw"] = prep["grid_thw"]
            if "image_sizes" in prep:
                data["image_sizes"] = prep["image_sizes"]

        return BatchEncoding(data=data, tensor_type=return_tensors)


def _blip3o_processor_repo_candidates(model_name: str = "") -> List[str]:
    override = os.environ.get("BLIP3O_IMAGE_PROCESSOR_ID", "").strip()
    candidates = []
    if override:
        candidates.append(override)
    if model_name:
        candidates.append(model_name)
    candidates.append("Qwen/Qwen2.5-VL-7B-Instruct")
    return candidates


def _load_blip3o_multimodal_processor_from_candidates(model_name: str = ""):
    from transformers import AutoProcessor

    seen = set()
    for repo_id in _blip3o_processor_repo_candidates(model_name=model_name):
        repo_id = (repo_id or "").strip()
        if not repo_id or repo_id in seen:
            continue
        seen.add(repo_id)
        try:
            proc = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        except Exception:
            try:
                proc = AutoProcessor.from_pretrained(repo_id)
            except Exception:
                continue
        if getattr(proc, "image_processor", None) is not None:
            return proc
    return None


def _load_blip3o_image_processor_from_candidates(model_name: str = ""):
    proc = _load_blip3o_multimodal_processor_from_candidates(model_name=model_name)
    if proc is not None:
        ip = getattr(proc, "image_processor", None)
        if ip is not None:
            return ip
    return None


def _resolve_blip3o_image_processor(model, model_name: str = ""):
    image_processor = None
    if model is not None:
        model_ref = _unwrap_model(model)
        get_vt = getattr(model_ref, "get_vision_tower", None)
        if callable(get_vt):
            try:
                vt = get_vt()
                image_processor = getattr(vt, "image_processor", None)
            except Exception:
                image_processor = None
        if image_processor is None:
            # Some wrapper chains expose image_processor deeper than get_vision_tower().
            try:
                for _, obj in _iter_wrapper_objects(model_ref):
                    ip = getattr(obj, "image_processor", None)
                    if ip is not None:
                        image_processor = ip
                        break
            except Exception:
                pass

    if image_processor is not None:
        return image_processor
    return _load_blip3o_image_processor_from_candidates(model_name=model_name)


def _build_blip3o_processor(tokenizer, model, model_name: str = "", tokenizer_image_token_fn=None):
    image_processor = _resolve_blip3o_image_processor(model=model, model_name=model_name)
    multimodal_processor = _load_blip3o_multimodal_processor_from_candidates(model_name=model_name)
    return _Blip3oProcessorShim(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_name=model_name,
        multimodal_processor=multimodal_processor,
        tokenizer_image_token_fn=tokenizer_image_token_fn,
    )


def _iter_wrapper_objects(root_obj) -> List[Tuple[str, object]]:
    """
    Return model wrapper chain candidates where custom generation APIs may live.

    This handles DDP/PEFT/base-model nesting patterns.
    """
    results: List[Tuple[str, object]] = []
    queue: List[Tuple[str, object]] = [("model", root_obj)]
    seen = set()
    attrs = ("module", "base_model", "model", "language_model", "backbone")

    while queue:
        path, obj = queue.pop(0)
        if obj is None:
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        results.append((path, obj))
        for attr in attrs:
            child = getattr(obj, attr, None)
            if child is not None and child is not obj:
                queue.append((f"{path}.{attr}", child))
    return results


def _find_generation_callable(root_obj) -> Tuple[Optional[str], Optional[object], Optional[str], List[str]]:
    """
    Find `generate_images`/`generate_image` across nested wrapper objects.
    """
    inspected: List[str] = []
    for path, obj in _iter_wrapper_objects(root_obj):
        inspected.append(f"{path}:{type(obj).__name__}")
        for name in ("generate_images", "generate_image"):
            fn = getattr(obj, name, None)
            if callable(fn):
                return name, obj, path, inspected
    return None, None, None, inspected


def _find_callable_object(root_obj, callable_name: str) -> Tuple[Optional[object], Optional[str]]:
    try:
        wrappers = _iter_wrapper_objects(root_obj)
    except Exception:
        wrappers = [("model", root_obj)]
    for path, obj in wrappers:
        fn = getattr(obj, callable_name, None)
        if callable(fn):
            return obj, path
    return None, None


def _extract_tokenizer_from_processor(processor):
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        return tok
    mm_proc = getattr(processor, "multimodal_processor", None)
    if mm_proc is not None:
        tok = getattr(mm_proc, "tokenizer", None)
        if tok is not None:
            return tok
    return None


def _decode_blip3o_generate_image_output(api_obj, image_out):
    decode_fn = getattr(api_obj, "decode_latents", None)
    if not callable(decode_fn):
        decode_owner, _ = _find_callable_object(api_obj, "decode_latents")
        if decode_owner is not None:
            api_obj = decode_owner
            decode_fn = getattr(api_obj, "decode_latents", None)

    model_obj_fn = getattr(api_obj, "get_model", None)
    if not callable(model_obj_fn):
        model_owner, _ = _find_callable_object(api_obj, "get_model")
        if model_owner is not None:
            model_obj_fn = getattr(model_owner, "get_model", None)
    model_obj = None
    if callable(model_obj_fn):
        try:
            model_obj = model_obj_fn()
        except Exception:
            model_obj = None
    vae_obj = None
    if model_obj is not None:
        vae_obj = getattr(model_obj, "vae", None)
        if vae_obj is None:
            vae_obj = getattr(model_obj, "sana_vae", None)
    if not callable(decode_fn) and vae_obj is None:
        return None

    latents = image_out
    if torch.is_tensor(latents) and latents.dtype in (torch.bfloat16, torch.float16):
        latents = latents.to(torch.float32)

    decode_device = None
    decode_dtype = None
    if vae_obj is not None:
        try:
            p = next(vae_obj.parameters())
            decode_device = p.device
            decode_dtype = p.dtype
        except Exception:
            pass

    candidates: List[torch.Tensor] = []
    if torch.is_tensor(latents):
        candidates.append(latents)
        try:
            # BLIP3o `generate_image` often returns reshaped latents [B, N, 1792].
            if latents.ndim == 3:
                bsz = int(latents.shape[0])
                n = int(latents.shape[1])
                c = int(latents.shape[2])

                # Generic square-token reshape [B, S^2, C] -> [B, C, S, S]
                s = int(round(math.sqrt(float(n))))
                if s * s == n:
                    candidates.append(latents.permute(0, 2, 1).contiguous().view(bsz, c, s, s))

                # Reverse-layout variant [B, C, S^2] -> [B, C, S, S]
                s2 = int(round(math.sqrt(float(c))))
                if s2 * s2 == c:
                    candidates.append(latents.contiguous().view(bsz, n, s2, s2))

                # Config-driven reshape when available.
                model_obj_getter = getattr(api_obj, "get_model", None)
                if callable(model_obj_getter):
                    m = model_obj_getter()
                    dit = getattr(m, "dit", None)
                    cfg = getattr(dit, "config", None) if dit is not None else None
                    in_channels = int(getattr(cfg, "in_channels", 0) or 0)
                    input_size = int(getattr(cfg, "input_size", 0) or 0)
                    target = in_channels * input_size * input_size
                    if bsz > 0 and target > 0 and int(latents[0].numel()) == target:
                        candidates.append(
                            latents.permute(0, 2, 1).contiguous().view(
                                bsz, in_channels, input_size, input_size
                            )
                        )

                # Heuristic fallback for original BLIP3o flattened outputs where
                # shape is [B, N_query, D_flat] (e.g., [1, 64, 1792]).
                total = n * c
                for side in (64, 56, 48, 40, 32, 24, 16, 8):
                    area = side * side
                    if total % area != 0:
                        continue
                    channels = total // area
                    if channels <= 0 or channels > 128:
                        continue
                    candidates.append(
                        latents.permute(0, 2, 1).contiguous().view(
                            bsz, channels, side, side
                        )
                    )
        except Exception:
            pass

    for cand in candidates:
        cand_for_decode = cand
        if torch.is_tensor(cand_for_decode):
            try:
                if decode_device is not None:
                    tgt_dtype = decode_dtype if decode_dtype is not None else cand_for_decode.dtype
                    if tgt_dtype in (torch.float16, torch.bfloat16):
                        # VAE decode is more stable in fp32.
                        tgt_dtype = torch.float32
                    cand_for_decode = cand_for_decode.to(device=decode_device, dtype=tgt_dtype)
            except Exception:
                cand_for_decode = cand

        try:
            if callable(decode_fn):
                decoded = decode_fn(cand_for_decode, return_tensor=False)
            elif vae_obj is not None:
                lat_for_vae = cand_for_decode
                cfg = getattr(vae_obj, "config", None)
                scaling = getattr(cfg, "scaling_factor", None) if cfg is not None else None
                shift = getattr(cfg, "shift_factor", None) if cfg is not None else None
                if scaling is not None:
                    lat_for_vae = lat_for_vae / float(scaling)
                if shift is not None:
                    lat_for_vae = lat_for_vae + float(shift)
                sample = vae_obj.decode(lat_for_vae)
                decoded = sample.sample if hasattr(sample, "sample") else sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
                decoded = [Image.fromarray(np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)) for x in decoded]
            else:
                continue
        except Exception:
            # Try next candidate reshape.
            continue
        try:
            if isinstance(decoded, list) and decoded:
                return _ensure_pil_image(decoded[0])
            return _ensure_pil_image(decoded)
        except Exception:
            continue
    return None


class TextPolicyUpdater:
    """
    KL-regularized REINFORCE updater for text-only trajectories (generator role).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor,
        config,
        adapter_name: Optional[str],
        reference_model: Optional[torch.nn.Module] = None,
    ):
        self.model = model
        self.processor = processor
        self.config = config
        self.adapter_name = adapter_name
        self.reference_model = reference_model
        self.kl_coef = config.kl_coef
        self.step_id = 0

        params = list(_collect_trainable_params(model, adapter_name))
        if not params:
            raise RuntimeError(f"No trainable parameters found for adapter={adapter_name!r}")
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    def state_dict(self) -> Dict:
        return {
            "optimizer": self.opt.state_dict(),
            "kl_coef": float(self.kl_coef),
            "step_id": int(self.step_id),
        }

    def load_state_dict(self, state: Dict):
        if not isinstance(state, dict):
            return
        if "optimizer" in state:
            self.opt.load_state_dict(state["optimizer"])
        if "kl_coef" in state:
            self.kl_coef = float(state["kl_coef"])
        if "step_id" in state:
            self.step_id = int(state["step_id"])

    def _adapt_beta(self, kl_val: float):
        target = max(self.config.kl_target, 1e-8)
        delta = (kl_val - target) / target
        beta = self.kl_coef * math.exp(self.config.kl_adapt_rate * delta)
        beta = max(self.config.kl_min, min(self.config.kl_max, beta))
        self.kl_coef = float(beta)

    def step(
        self,
        prompt: str,
        completion: str,
        reward: float,
        baseline: float,
        device: torch.device,
        image: Optional[Image.Image] = None,
        completion_token_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        if not completion or not str(completion).strip():
            raise ValueError("Generator update requires non-empty token completion trace.")
        self.step_id += 1

        if image is None:
            text_prompt = prompt
            use_token_ids = bool(completion_token_ids)
            if use_token_ids:
                prompt_inputs = _prepare_text_inputs(self.processor, device, text_prompt)
                prompt_ids = prompt_inputs["input_ids"]
                if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
                    raise RuntimeError("Expected single-example prompt batch for token-trace generator update.")
                comp_ids = torch.tensor(completion_token_ids, dtype=torch.long, device=prompt_ids.device).view(1, -1)
                full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
                full_mask = torch.ones_like(full_ids, dtype=torch.long)
                prompt_mask = prompt_inputs.get("attention_mask")
                if prompt_mask is None:
                    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long)
                inputs_prompt = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
                inputs_full = {"input_ids": full_ids, "attention_mask": full_mask}
            else:
                text_full = prompt + completion
                inputs_prompt = _prepare_text_inputs(self.processor, device, text_prompt)
                inputs_full = _prepare_text_inputs(self.processor, device, text_full)
        else:
            chat_prompt = _build_chat_text(self.processor, image, prompt)
            chat_full = chat_prompt + completion
            inputs_prompt = _prepare_mm_inputs(self.processor, device, image, chat_prompt)
            inputs_full = _prepare_mm_inputs(self.processor, device, image, chat_full)

        input_ids = inputs_full["input_ids"]
        labels = input_ids.clone()
        prompt_len = inputs_prompt["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        valid_mask = labels[:, 1:] != -100

        self.model.train(True)
        policy_inputs = dict(inputs_full)
        policy_inputs["labels"] = labels
        # Disable KV cache during training forwards to reduce memory pressure.
        policy_inputs["use_cache"] = False
        with use_adapter(self.model, self.adapter_name):
            out_pi = self.model(**policy_inputs)
        ce_loss = out_pi.loss
        logp_pi = F.log_softmax(out_pi.logits, dim=-1)

        ref_inputs = dict(inputs_full)
        ref_inputs["use_cache"] = False
        if self.reference_model is not None:
            with torch.no_grad():
                out_ref = self.reference_model(**ref_inputs)
        else:
            with torch.no_grad():
                with use_adapter(self.model, None):
                    out_ref = self.model(**ref_inputs)
        logp_ref = F.log_softmax(out_ref.logits, dim=-1)

        logp_pi_shift = logp_pi[:, :-1, :]
        logp_ref_shift = logp_ref[:, :-1, :]
        p_pi_shift = logp_pi_shift.exp()
        kl_per_tok = (p_pi_shift * (logp_pi_shift - logp_ref_shift)).sum(dim=-1)
        if valid_mask.any():
            kl_loss = kl_per_tok[valid_mask].mean()
        else:
            kl_loss = torch.tensor(0.0, device=ce_loss.device)

        advantage = float(reward - baseline)
        beta_before = float(self.kl_coef)
        total_loss = advantage * ce_loss + beta_before * kl_loss

        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()
        _clip_grad_norm_multi_device(self.params, self.config.grad_clip)
        self.opt.step()
        self.model.train(False)

        kl_val = float(kl_loss.item())
        self._adapt_beta(kl_val)

        try:
            del inputs_prompt, inputs_full, input_ids, labels, policy_inputs
            del out_pi, out_ref, logp_pi, logp_ref, logp_pi_shift, logp_ref_shift
            del p_pi_shift, kl_per_tok, valid_mask
        except Exception:
            pass

        if (
            torch.cuda.is_available()
            and self.config.clear_cache_every > 0
            and self.step_id % self.config.clear_cache_every == 0
        ):
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            gc.collect()

        return {
            "ce_loss": float(ce_loss.item()),
            "kl_loss": kl_val,
            "advantage": advantage,
            "kl_coef_before": beta_before,
            "kl_coef_after": float(self.kl_coef),
            "total_loss": float(total_loss.item()),
        }


class TextPreferenceDPOUpdater:
    """
    Pairwise DPO updater for generator role.

    This updater compares chosen vs rejected completions under a policy adapter
    against a frozen reference policy and optimizes:
      -log sigma(beta * ((logpi_c - logpi_r) - (logref_c - logref_r)))
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor,
        config,
        adapter_name: Optional[str],
        reference_model: Optional[torch.nn.Module] = None,
    ):
        self.model = model
        self.processor = processor
        self.config = config
        self.adapter_name = adapter_name
        self.reference_model = reference_model
        self.step_id = 0
        self.kl_coef = 0.0  # Kept for checkpoint/log schema compatibility.

        self.dpo_beta = float(max(1e-6, getattr(config, "dpo_beta", 0.1)))
        self.dpo_label_smoothing = float(getattr(config, "dpo_label_smoothing", 0.0))
        if not (0.0 <= self.dpo_label_smoothing < 0.5):
            raise ValueError("dpo_label_smoothing must satisfy 0.0 <= value < 0.5")

        params = list(_collect_trainable_params(model, adapter_name))
        if not params:
            raise RuntimeError(f"No trainable parameters found for adapter={adapter_name!r}")
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    def state_dict(self) -> Dict:
        return {
            "optimizer": self.opt.state_dict(),
            "step_id": int(self.step_id),
            "dpo_beta": float(self.dpo_beta),
            "dpo_label_smoothing": float(self.dpo_label_smoothing),
            "kl_coef": float(self.kl_coef),
        }

    def load_state_dict(self, state: Dict):
        if not isinstance(state, dict):
            return
        if "optimizer" in state:
            self.opt.load_state_dict(state["optimizer"])
        if "step_id" in state:
            self.step_id = int(state["step_id"])
        if "dpo_beta" in state:
            self.dpo_beta = float(state["dpo_beta"])
        if "dpo_label_smoothing" in state:
            self.dpo_label_smoothing = float(state["dpo_label_smoothing"])
        if "kl_coef" in state:
            self.kl_coef = float(state["kl_coef"])

    def _build_inputs(
        self,
        *,
        prompt: str,
        completion: str,
        device: torch.device,
        image: Optional[Image.Image] = None,
        completion_token_ids: Optional[List[int]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not completion or not str(completion).strip():
            raise ValueError("DPO update requires non-empty completion text.")

        if image is None:
            text_prompt = prompt
            use_token_ids = bool(completion_token_ids)
            if use_token_ids:
                prompt_inputs = _prepare_text_inputs(self.processor, device, text_prompt)
                prompt_ids = prompt_inputs["input_ids"]
                if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
                    raise RuntimeError("Expected single-example prompt batch for DPO token-trace update.")
                comp_ids = torch.tensor(completion_token_ids, dtype=torch.long, device=prompt_ids.device).view(1, -1)
                full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
                full_mask = torch.ones_like(full_ids, dtype=torch.long)
                prompt_mask = prompt_inputs.get("attention_mask")
                if prompt_mask is None:
                    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long)
                inputs_prompt = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
                inputs_full = {"input_ids": full_ids, "attention_mask": full_mask}
            else:
                text_full = prompt + completion
                inputs_prompt = _prepare_text_inputs(self.processor, device, text_prompt)
                inputs_full = _prepare_text_inputs(self.processor, device, text_full)
        else:
            chat_prompt = _build_chat_text(self.processor, image, prompt)
            chat_full = chat_prompt + completion
            inputs_prompt = _prepare_mm_inputs(self.processor, device, image, chat_prompt)
            inputs_full = _prepare_mm_inputs(self.processor, device, image, chat_full)
        return inputs_prompt, inputs_full

    def _sequence_logp_from_logits(
        self,
        *,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        prompt_len: int,
    ) -> Tuple[torch.Tensor, int]:
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != -100

        logp = F.log_softmax(logits, dim=-1)
        shift_logp = logp[:, :-1, :]
        gathered = shift_logp.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)

        valid_count = int(valid_mask.sum().item())
        if valid_count <= 0:
            seq_logp = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        else:
            # Mean token log-prob for length robustness across variable completions.
            seq_logp = gathered[valid_mask].mean()
        return seq_logp, valid_count

    def _forward_seq_logp(
        self,
        *,
        model: torch.nn.Module,
        adapter_name: Optional[str],
        inputs_prompt: Dict[str, torch.Tensor],
        inputs_full: Dict[str, torch.Tensor],
        no_grad: bool,
    ) -> Tuple[torch.Tensor, int]:
        context = torch.no_grad() if no_grad else torch.enable_grad()
        with context:
            forward_inputs = dict(inputs_full)
            forward_inputs["use_cache"] = False
            with use_adapter(model, adapter_name):
                out = model(**forward_inputs)
            prompt_len = int(inputs_prompt["input_ids"].shape[1])
            seq_logp, token_count = self._sequence_logp_from_logits(
                logits=out.logits,
                input_ids=forward_inputs["input_ids"],
                prompt_len=prompt_len,
            )
        return seq_logp, token_count

    def step(
        self,
        *,
        prompt: str,
        chosen_completion: str,
        rejected_completion: str,
        device: torch.device,
        chosen_image: Optional[Image.Image] = None,
        rejected_image: Optional[Image.Image] = None,
        chosen_completion_token_ids: Optional[List[int]] = None,
        rejected_completion_token_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        self.step_id += 1
        self.model.train(True)

        chosen_prompt_inputs, chosen_full_inputs = self._build_inputs(
            prompt=prompt,
            completion=chosen_completion,
            device=device,
            image=chosen_image,
            completion_token_ids=chosen_completion_token_ids,
        )
        rejected_prompt_inputs, rejected_full_inputs = self._build_inputs(
            prompt=prompt,
            completion=rejected_completion,
            device=device,
            image=rejected_image,
            completion_token_ids=rejected_completion_token_ids,
        )

        pi_logp_chosen, chosen_token_count = self._forward_seq_logp(
            model=self.model,
            adapter_name=self.adapter_name,
            inputs_prompt=chosen_prompt_inputs,
            inputs_full=chosen_full_inputs,
            no_grad=False,
        )
        pi_logp_rejected, rejected_token_count = self._forward_seq_logp(
            model=self.model,
            adapter_name=self.adapter_name,
            inputs_prompt=rejected_prompt_inputs,
            inputs_full=rejected_full_inputs,
            no_grad=False,
        )

        if self.reference_model is not None:
            ref_model = self.reference_model
            ref_adapter_name = None
        else:
            ref_model = self.model
            ref_adapter_name = None

        ref_logp_chosen, _ = self._forward_seq_logp(
            model=ref_model,
            adapter_name=ref_adapter_name,
            inputs_prompt=chosen_prompt_inputs,
            inputs_full=chosen_full_inputs,
            no_grad=True,
        )
        ref_logp_rejected, _ = self._forward_seq_logp(
            model=ref_model,
            adapter_name=ref_adapter_name,
            inputs_prompt=rejected_prompt_inputs,
            inputs_full=rejected_full_inputs,
            no_grad=True,
        )

        pi_gap = pi_logp_chosen - pi_logp_rejected
        ref_gap = ref_logp_chosen - ref_logp_rejected
        preference_margin = pi_gap - ref_gap

        scaled_margin = self.dpo_beta * preference_margin
        pos_term = -F.logsigmoid(scaled_margin)
        if self.dpo_label_smoothing > 0.0:
            neg_term = -F.logsigmoid(-scaled_margin)
            dpo_loss = (1.0 - self.dpo_label_smoothing) * pos_term + self.dpo_label_smoothing * neg_term
        else:
            dpo_loss = pos_term

        self.opt.zero_grad(set_to_none=True)
        dpo_loss.backward()
        _clip_grad_norm_multi_device(self.params, self.config.grad_clip)
        self.opt.step()
        self.model.train(False)

        if (
            torch.cuda.is_available()
            and self.config.clear_cache_every > 0
            and self.step_id % self.config.clear_cache_every == 0
        ):
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            gc.collect()

        return {
            "dpo_loss": float(dpo_loss.item()),
            "dpo_beta": float(self.dpo_beta),
            "label_smoothing": float(self.dpo_label_smoothing),
            "pi_logp_chosen": float(pi_logp_chosen.detach().item()),
            "pi_logp_rejected": float(pi_logp_rejected.detach().item()),
            "ref_logp_chosen": float(ref_logp_chosen.detach().item()),
            "ref_logp_rejected": float(ref_logp_rejected.detach().item()),
            "pi_gap": float(pi_gap.detach().item()),
            "ref_gap": float(ref_gap.detach().item()),
            "preference_margin": float(preference_margin.detach().item()),
            "chosen_token_count": float(chosen_token_count),
            "rejected_token_count": float(rejected_token_count),
            "kl_coef_before": 0.0,
            "kl_coef_after": 0.0,
        }


@dataclass
class GenerationSelfEvolvingConfig:
    experiment_name: str = "generation_self_evolving"
    run_name: Optional[str] = None
    output_dir: str = "./runs"

    # Data
    data_dir: str = ""
    data_split: str = "all"
    include_subfolders: Optional[Tuple[str, ...]] = None
    max_images: Optional[int] = None

    # Model
    model_name: str = "BLIP3o/BLIP3o-Model-8B"
    dtype: str = "bfloat16"
    cuda_device: int = 0
    device_map: str = "single"
    attn_implementation: str = "auto"

    # Optimization
    total_steps: int = 100
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    proposer_update_freq: int = 5
    generator_update_freq: int = 1
    enable_solver_updates: bool = False
    solver_update_freq: int = 0

    # Decoding
    temp: float = 1.0
    top_p: float = 1.0
    max_new_tokens_solver: int = 128
    max_new_tokens_proposer: int = 256
    max_new_tokens_caption: int = 96
    max_new_tokens_generator: int = 768
    num_solver_samples: int = 5
    num_solver_samples_spec: int = 3
    num_generations: int = 4

    # Generation backend
    generation_num_inference_steps: int = 30
    generation_guidance_scale: float = 2.0
    generation_height: int = 1024
    generation_width: int = 1024
    require_decoder_for_blip3o: bool = True
    allow_latent_visualization_fallback: bool = False
    strict_require_generation_tokens: bool = True
    generator_missing_trace_strategy: str = "proxy"  # proxy|skip|error
    verification_use_reference_solver: bool = True
    generator_update_rule: str = "reinforce"  # reinforce|dpo
    dpo_beta: float = 0.1
    dpo_label_smoothing: float = 0.0
    dpo_min_reward_gap: float = 0.0

    # Reward shaping
    solver_soft_gamma: float = 0.7
    len_penalty_weight: float = 0.10
    len_penalty_target_words: int = 6
    prop_entropy_mu: float = 0.90
    prop_entropy_sigma: float = 0.35
    reward_spec_weight: float = 0.65
    reward_cycle_weight: float = 0.20
    reward_diversity_weight: float = 0.10
    reward_contradiction_weight: float = 0.20
    min_spec_quality_for_update: float = 0.35
    min_spec_qa_pairs: int = 2
    max_expected_words: int = 8
    max_question_words: int = 24

    # KL control
    kl_coef: float = 1e-3
    kl_target: float = 0.02
    kl_adapt_rate: float = 0.10
    kl_min: float = 1e-8
    kl_max: float = 1e2

    # Baselines
    baseline_momentum: float = 0.9

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = DEFAULT_LORA_TARGETS

    # Repro + logging
    seed: int = 42
    deterministic: bool = True
    log_every: int = 1
    save_every: int = 50
    max_checkpoints: int = 3
    clear_cache_every: int = 25
    save_generated_images_every: int = 0

    # W&B
    wandb_mode: str = "disabled"
    wandb_project: str = "self-evolving-uug"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_images_every: int = 0

    # Resume
    resume_from: Optional[str] = None
    start_step: int = 0


class GenerationSelfEvolvingTrainer:
    def _setup_distributed(self):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.cfg.cuda_device)))
        self.distributed = self.world_size > 1
        self.is_main_process = self.rank == 0

        if self.distributed:
            if not dist.is_available():
                raise RuntimeError("torch.distributed is not available in this environment.")
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                backend = "nccl"
            else:
                backend = "gloo"
            if not dist.is_initialized():
                init_kwargs = {"backend": backend, "init_method": "env://"}
                if backend == "nccl":
                    init_kwargs["device_id"] = self.local_rank
                try:
                    dist.init_process_group(**init_kwargs)
                except TypeError:
                    init_kwargs.pop("device_id", None)
                    dist.init_process_group(**init_kwargs)
            self.is_main_process = dist.get_rank() == 0
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            print(
                f"[DDP] Initialized rank={self.rank}/{self.world_size} local_rank={self.local_rank} backend={backend}"
            )
        elif torch.cuda.is_available():
            torch.cuda.set_device(self.cfg.cuda_device)

    def _dist_barrier(self):
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def _dist_mean(self, value: float) -> float:
        if not (self.distributed and dist.is_initialized()):
            return float(value)
        dev = torch.device(f"cuda:{self.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor([float(value)], dtype=torch.float64, device=dev)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float((tensor / float(self.world_size)).item())

    def _sync_state_scalars(self):
        if not (self.distributed and dist.is_initialized()):
            return
        self.generator_baseline = self._dist_mean(self.generator_baseline)
        self.proposer_baseline = self._dist_mean(self.proposer_baseline)
        if self.solver_updater is not None:
            self.solver_baseline = self._dist_mean(self.solver_baseline)
            self.solver_updater.kl_coef = self._dist_mean(self.solver_updater.kl_coef)
        self.proposer_updater.kl_coef = self._dist_mean(self.proposer_updater.kl_coef)
        self.generator_updater.kl_coef = self._dist_mean(self.generator_updater.kl_coef)

    def __init__(self, config: GenerationSelfEvolvingConfig):
        self.cfg = config
        self.cfg.generator_update_rule = str(self.cfg.generator_update_rule or "reinforce").strip().lower()
        if self.cfg.generator_update_rule not in {"reinforce", "dpo"}:
            raise ValueError(
                f"Unsupported generator_update_rule={self.cfg.generator_update_rule!r}. "
                "Expected one of: reinforce, dpo."
            )
        self._setup_distributed()
        _set_global_seed(config.seed + self.rank, deterministic=config.deterministic)

        if not config.data_dir:
            raise ValueError("`data_dir` is required for generation self-evolving training")

        self.run_dir = self._build_run_dir()
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir = self.run_dir / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        self.iter_log_path = self.run_dir / "iter_log.jsonl"
        self.prompts_log_path = self.logs_dir / "proposer_prompts.jsonl"
        self.candidates_log_path = self.logs_dir / "generation_candidates.jsonl"
        self.rewards_log_path = self.logs_dir / "rewards.jsonl"
        self.policy_updates_log_path = self.logs_dir / "policy_updates.jsonl"
        self.dpo_pairs_log_path = self.logs_dir / "dpo_pairs.jsonl"
        self.summary_path = self.run_dir / "ablation_summary.json"
        self._save_run_metadata()

        self._blip3o_diffusion_pipe = None
        self._generation_api_name = None
        self._generation_api_obj = None
        self._generation_api_path = None
        self._warned_latent_fallback = False
        self.model, self.processor = self._load_model()
        fallback_dev = self.local_rank if self.distributed else config.cuda_device
        self.device = _infer_primary_device(self.model, fallback_cuda_device=fallback_dev)
        self._generation_api_name, self._generation_api_obj, self._generation_api_path, inspected = _find_generation_callable(
            _unwrap_model(self.model)
        )
        if self._generation_api_name is None and self._blip3o_diffusion_pipe is None:
            inspected_text = "; ".join(inspected[:10]) if inspected else "none"
            raise RuntimeError(
                "Loaded model does not expose a supported image generation API for "
                f"`{config.experiment_name}`.\n"
                f"model_name={config.model_name}\n"
                f"inspected_wrappers={inspected_text}\n"
                "Expected one of: generate_images(...), generate_image(...), or a BLIP3o diffusion-decoder pipeline.\n"
                "Use a generation-capable model (e.g., BLIP3o family) for generation/unified experiments."
            )
        if (
            _is_original_blip3o_model_name(config.model_name)
            and config.require_decoder_for_blip3o
            and self._blip3o_diffusion_pipe is None
        ):
            raise RuntimeError(
                "Original BLIP3o scientific runs require a working diffusion decoder pipeline, "
                "but decoder initialization failed.\n"
                f"model_name={config.model_name}\n"
                "Set BLIP3O_DIFFUSION_REPO to a checkpoint that contains `diffusion-decoder` "
                "(e.g., BLIP3o/BLIP3o-Model), ensure enough HF cache disk, and rerun.\n"
                "If you intentionally want debug-only latent visualization, pass "
                "`--allow_missing_decoder_for_blip3o --allow_latent_visualization_fallback`."
            )
        if self._generation_api_name is not None and self.is_main_process:
            print(
                f"[Generation] Using generation backend `{self._generation_api_name}` "
                f"from `{self._generation_api_path}` ({type(self._generation_api_obj).__name__})"
            )
        elif self._blip3o_diffusion_pipe is not None and self.is_main_process:
            print("[Generation] Using generation backend `diffusion_pipeline` (original BLIP3o decoder).")
            if not self.cfg.strict_require_generation_tokens:
                print(
                    "[Generation] Note: token traces may be unavailable with diffusion pipeline backend; "
                    "generator updates can be skipped when no completion trace is returned."
                )

        pool_cfg = ImagePoolConfig(
            data_dir=config.data_dir,
            include_subfolders=list(config.include_subfolders) if config.include_subfolders else None,
            split=None if config.data_split == "all" else config.data_split,
            prefer_manifest=False,
            max_images=config.max_images,
            seed=config.seed,
        )
        self.pool = ImagePool(pool_cfg)

        reference_model = None
        if not config.use_lora:
            reference_model = _load_model_with_fallback(
                config.model_name,
                torch_dtype=_safe_dtype(config.dtype),
                device_map={"": fallback_dev} if self.device.type == "cuda" else "cpu",
                trust_remote_code=True,
                attn_implementation=_resolve_attn_implementation(config.attn_implementation),
            )
            reference_model.eval()
            for p in reference_model.parameters():
                p.requires_grad_(False)

        self.train_model = self.model
        if self.distributed:
            ddp_kwargs = {"find_unused_parameters": True}
            if torch.cuda.is_available():
                ddp_kwargs["device_ids"] = [self.local_rank]
                ddp_kwargs["output_device"] = self.local_rank
            self.train_model = torch.nn.parallel.DistributedDataParallel(self.model, **ddp_kwargs)

        self.solver_updater: Optional[RolePolicyUpdater] = None
        if config.enable_solver_updates and config.solver_update_freq > 0:
            self.solver_updater = RolePolicyUpdater(
                model=self.train_model,
                processor=self.processor,
                config=config,
                adapter_name="default" if config.use_lora else None,
                reference_model=reference_model,
            )

        self.proposer_updater = RolePolicyUpdater(
            model=self.train_model,
            processor=self.processor,
            config=config,
            adapter_name="proposer" if config.use_lora else None,
            reference_model=reference_model,
        )

        if self.cfg.generator_update_rule == "dpo":
            self.generator_updater = TextPreferenceDPOUpdater(
                model=self.train_model,
                processor=self.processor,
                config=config,
                adapter_name="generator" if config.use_lora else None,
                reference_model=reference_model,
            )
        else:
            self.generator_updater = TextPolicyUpdater(
                model=self.train_model,
                processor=self.processor,
                config=config,
                adapter_name="generator" if config.use_lora else None,
                reference_model=reference_model,
            )

        self.generator_baseline = 0.0
        self.proposer_baseline = 0.0
        self.solver_baseline = 0.0
        self.start_step = max(0, int(config.start_step))

        self._metric_stats: Dict[str, Dict[str, float]] = {}
        self._policy_update_counts: Dict[str, int] = {"solver": 0, "proposer": 0, "generator": 0}
        self._generator_update_mode_counts: Dict[str, int] = {
            "token_trace": 0,
            "proxy_caption": 0,
            "skipped": 0,
        }
        self.wandb_run = self._init_wandb()

        loaded_resume_step = self._maybe_resume_state()
        if loaded_resume_step is not None:
            self.start_step = max(self.start_step, int(loaded_resume_step))

    def _init_wandb(self):
        if not self.is_main_process:
            return None
        mode = (self.cfg.wandb_mode or "disabled").strip().lower()
        if mode == "disabled":
            return None
        if not HAS_WANDB:
            print("[W&B] wandb package not available; disabling W&B logging.")
            return None

        token = os.environ.get("WANDB_API_KEY", "").strip()
        if token:
            try:
                wandb.login(key=token, relogin=False)
            except Exception as exc:
                print(f"[W&B] login failed using WANDB_API_KEY: {exc}")

        run_name = self.cfg.wandb_run_name or self.cfg.run_name or self.run_dir.name
        kwargs = {
            "project": self.cfg.wandb_project,
            "name": run_name,
            "mode": mode,
            "config": dataclasses.asdict(self.cfg),
            "dir": str(self.run_dir),
        }
        if self.cfg.wandb_entity:
            kwargs["entity"] = self.cfg.wandb_entity
        try:
            run = wandb.init(**kwargs)
            print(f"[W&B] Initialized run: {run_name} (mode={mode})")
            return run
        except Exception as exc:
            print(f"[W&B] init failed; continuing without W&B: {exc}")
            return None

    def _build_run_dir(self) -> pathlib.Path:
        base_dir = pathlib.Path(self.cfg.output_dir).expanduser().resolve()
        if self.distributed and dist.is_initialized():
            obj = [None]
            if self.is_main_process:
                timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                run_name = self.cfg.run_name or f"{self.cfg.experiment_name}_{timestamp}"
                run_dir = base_dir / run_name
                if run_dir.exists() and any(run_dir.iterdir()) and not self.cfg.resume_from:
                    run_dir = base_dir / f"{run_name}_{timestamp}"
                run_dir.mkdir(parents=True, exist_ok=True)
                obj[0] = str(run_dir)
            dist.broadcast_object_list(obj, src=0)
            run_dir = pathlib.Path(obj[0]).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            self._dist_barrier()
            return run_dir

        timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_name = self.cfg.run_name or f"{self.cfg.experiment_name}_{timestamp}"
        run_dir = base_dir / run_name
        if run_dir.exists() and any(run_dir.iterdir()) and not self.cfg.resume_from:
            run_dir = base_dir / f"{run_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _save_run_metadata(self):
        if not self.is_main_process:
            self._dist_barrier()
            return
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        _json_dump(self.run_dir / "config.json", dataclasses.asdict(self.cfg))
        _json_dump(self.run_dir / "git_info.json", _collect_git_info(repo_root))
        _json_dump(
            self.run_dir / "environment.json",
            {
                "python": os.sys.version,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "rank": self.rank,
                "world_size": self.world_size,
                "distributed": self.distributed,
            },
        )
        self._dist_barrier()

    def _append_jsonl(self, path: pathlib.Path, record: Dict):
        if not self.is_main_process:
            return
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _update_metric(self, name: str, value: float):
        stat = self._metric_stats.setdefault(
            name,
            {"count": 0.0, "sum": 0.0, "sum_sq": 0.0, "min": value, "max": value},
        )
        stat["count"] += 1.0
        stat["sum"] += value
        stat["sum_sq"] += value * value
        stat["min"] = min(stat["min"], value)
        stat["max"] = max(stat["max"], value)

    def _metrics_summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for name, stat in self._metric_stats.items():
            count = max(1.0, stat["count"])
            mean = stat["sum"] / count
            variance = max(0.0, (stat["sum_sq"] / count) - (mean * mean))
            summary[name] = {
                "count": int(stat["count"]),
                "mean": mean,
                "std": math.sqrt(variance),
                "min": stat["min"],
                "max": stat["max"],
            }
        return summary

    def _resolve_resume_dir(self) -> Optional[pathlib.Path]:
        if not self.cfg.resume_from:
            return None
        candidate = pathlib.Path(self.cfg.resume_from).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"resume_from path does not exist: {candidate}")

        if candidate.is_dir() and (candidate / "trainer_state.pt").exists():
            return candidate
        if candidate.is_file() and candidate.name == "trainer_state.pt":
            return candidate.parent
        if candidate.is_dir() and candidate.name.startswith("step_"):
            return candidate

        step_dirs = [p for p in candidate.glob("step_*") if p.is_dir() and (p / "trainer_state.pt").exists()]
        if not step_dirs:
            raise FileNotFoundError(
                f"No checkpoint with trainer_state.pt found under resume_from path: {candidate}"
            )
        return sorted(step_dirs, key=lambda p: p.name)[-1]

    def _maybe_resume_state(self) -> Optional[int]:
        resume_dir = self._resolve_resume_dir()
        if resume_dir is None:
            return None

        # Load trainable parameter snapshot to restore adapter weights without rebinding modules.
        trainable_path = resume_dir / "trainable_adapters.pt"
        if trainable_path.exists():
            try:
                adapter_state = torch.load(trainable_path, map_location="cpu")
                model_ref = _unwrap_model(self.model)
                missing, unexpected = model_ref.load_state_dict(adapter_state, strict=False)
                if self.is_main_process:
                    print(
                        f"[Generation] Restored trainable adapter weights from {trainable_path} "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})"
                    )
            except Exception as exc:
                if self.is_main_process:
                    print(f"[Generation] WARNING: failed to restore trainable adapter weights: {exc}")

        state_path = resume_dir / "trainer_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"trainer_state.pt not found in resume checkpoint: {resume_dir}")

        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")

        if self.solver_updater is not None:
            if "solver_updater" in state:
                self.solver_updater.load_state_dict(state["solver_updater"])
            elif "solver_opt" in state:
                self.solver_updater.load_state_dict(
                    {
                        "optimizer": state["solver_opt"],
                        "kl_coef": state.get("solver_kl_coef"),
                        "step_id": state.get("solver_updater_step", state.get("step", 0)),
                    }
                )

        if "proposer_updater" in state:
            self.proposer_updater.load_state_dict(state["proposer_updater"])
        elif "proposer_opt" in state:
            self.proposer_updater.load_state_dict(
                {
                    "optimizer": state["proposer_opt"],
                    "kl_coef": state.get("proposer_kl_coef"),
                    "step_id": state.get("proposer_updater_step", state.get("step", 0)),
                }
            )

        if "generator_updater" in state:
            self.generator_updater.load_state_dict(state["generator_updater"])
        elif "generator_opt" in state:
            self.generator_updater.load_state_dict(
                {
                    "optimizer": state["generator_opt"],
                    "kl_coef": state.get("generator_kl_coef"),
                    "step_id": state.get("generator_updater_step", state.get("step", 0)),
                }
            )

        self.solver_baseline = float(state.get("solver_baseline", self.solver_baseline))
        self.proposer_baseline = float(state.get("proposer_baseline", self.proposer_baseline))
        self.generator_baseline = float(state.get("generator_baseline", self.generator_baseline))

        py_state = state.get("py_random_state")
        if py_state is not None:
            random.setstate(py_state)
        torch_state = state.get("torch_rng_state")
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        cuda_states = state.get("torch_cuda_rng_state_all")
        if torch.cuda.is_available() and cuda_states is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass

        restored_step = int(state.get("step", 0))
        if self.is_main_process:
            print(f"[Generation] Resumed trainer state from: {state_path} (step={restored_step})")
            _json_dump(
                self.run_dir / "resume_info.json",
                {
                    "resume_from": str(resume_dir),
                    "restored_step": restored_step,
                    "restored_solver_baseline": self.solver_baseline,
                    "restored_proposer_baseline": self.proposer_baseline,
                    "restored_generator_baseline": self.generator_baseline,
                },
            )
        self._dist_barrier()
        return restored_step

    def _trainer_state_dict(self, step: int) -> Dict:
        state = {
            "step": int(step),
            "proposer_updater": self.proposer_updater.state_dict(),
            "generator_updater": self.generator_updater.state_dict(),
            "proposer_baseline": float(self.proposer_baseline),
            "generator_baseline": float(self.generator_baseline),
            "py_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if self.solver_updater is not None:
            state["solver_updater"] = self.solver_updater.state_dict()
            state["solver_baseline"] = float(self.solver_baseline)
        if torch.cuda.is_available():
            try:
                state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        return state

    def _is_complete_checkpoint(self, step_dir: pathlib.Path) -> bool:
        if not step_dir.is_dir():
            return False
        if not (step_dir / "SAVE_OK").exists():
            return False
        if self.cfg.use_lora:
            return (
                (step_dir / "solver").is_dir()
                and (step_dir / "proposer").is_dir()
                and (step_dir / "generator").is_dir()
            )
        return (step_dir / "model").is_dir()

    def _list_complete_checkpoints(self) -> List[pathlib.Path]:
        checkpoints = [p for p in self.run_dir.glob("step_*") if self._is_complete_checkpoint(p)]
        return sorted(checkpoints, key=lambda p: p.name)

    def _load_model(self):
        if self.cfg.use_lora and (not HAS_PEFT or LoraConfig is None or get_peft_model is None or TaskType is None):
            raise RuntimeError("PEFT is required for role-specific LoRA adapters")

        dtype = _safe_dtype(self.cfg.dtype)
        attn_impl = _resolve_attn_implementation(self.cfg.attn_implementation)

        if self.distributed:
            device_map = {"": self.local_rank} if torch.cuda.is_available() else "cpu"
        elif self.cfg.device_map == "single":
            device_map = {"": self.cfg.cuda_device} if torch.cuda.is_available() else "cpu"
        elif self.cfg.device_map == "cpu":
            device_map = "cpu"
        else:
            device_map = "auto"

        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        processor = None
        model = None
        loader_used = "auto_fallback"
        mm_utils: Dict[str, Any] = {}
        model_name_lower = (self.cfg.model_name or "").lower()
        is_original_blip3o = _is_original_blip3o_model_name(self.cfg.model_name)
        is_blip3o_next = _is_blip3o_next_model_name(self.cfg.model_name)
        explicit_local_repo = os.environ.get("BLIP3O_REPO", "").strip()
        default_local_classes_mode = "false" if is_original_blip3o else "auto"
        local_classes_mode = os.environ.get("BLIP3O_USE_LOCAL_CLASSES", default_local_classes_mode).strip().lower()
        if local_classes_mode in {"1", "true", "yes", "on"}:
            use_local_blip3o_classes = True
        elif local_classes_mode in {"0", "false", "no", "off"}:
            use_local_blip3o_classes = False
        else:
            # Local in-repo classes are BLIP3o-NEXT style; default to local only for NEXT.
            use_local_blip3o_classes = is_blip3o_next
        if explicit_local_repo:
            use_local_blip3o_classes = True

        if is_original_blip3o and use_local_blip3o_classes and not explicit_local_repo:
            raise RuntimeError(
                "Original BLIP3o checkpoint loading with local classes requires an explicit "
                "`BLIP3O_REPO` path to the original BLIP3o (main branch) codebase.\n"
                "This repository includes BLIP3o-NEXT local code which is incompatible with "
                f"'{self.cfg.model_name}'."
            )
        allow_implicit_local_repo = (not explicit_local_repo) and (not is_original_blip3o)

        if is_original_blip3o and use_local_blip3o_classes and self.is_main_process:
            print("[Generation] Using explicit BLIP3O_REPO for original BLIP3o class registration.")

        if "blip3o" in model_name_lower and use_local_blip3o_classes:
            classes, import_err, added_path = _import_blip3o_classes(
                allow_implicit_repo=allow_implicit_local_repo
            )
            mm_utils, mm_import_err, mm_added_path = _import_blip3o_mm_utils(
                allow_implicit_repo=allow_implicit_local_repo
            )
            if self.is_main_process:
                if added_path:
                    print(f"[Generation] Added local BLIP3o path: {added_path}")
                if import_err is not None:
                    print(f"[Generation] BLIP3o import warning: {repr(import_err)}")
                if mm_added_path and mm_added_path != added_path:
                    print(f"[Generation] Added local BLIP3o mm_utils path: {mm_added_path}")
                if mm_import_err is not None:
                    print(f"[Generation] BLIP3o mm_utils import warning: {repr(mm_import_err)}")

            explicit_errors: List[str] = []
            for key in ("inference", "grpo"):
                cls = classes.get(key)
                if cls is None:
                    continue
                try:
                    model = _load_from_explicit_class(
                        cls,
                        self.cfg.model_name,
                        torch_dtype=dtype,
                        device_map=device_map,
                        attn_implementation=attn_impl,
                    )
                    loader_used = f"explicit:{cls.__name__}"
                    break
                except Exception as exc:
                    explicit_errors.append(f"{cls.__name__}: {exc}")
            if self.is_main_process and explicit_errors:
                print(
                    "[Generation] BLIP3o explicit class loading attempts failed; "
                    "falling back to AutoModel loader."
                )
                for err in explicit_errors:
                    print(f"  - {err}")

            if model is not None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, trust_remote_code=True)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
                processor = _build_blip3o_processor(
                    tokenizer=tokenizer,
                    model=model,
                    model_name=self.cfg.model_name,
                    tokenizer_image_token_fn=mm_utils.get("tokenizer_image_token"),
                )
                if self.is_main_process and isinstance(processor, _Blip3oProcessorShim):
                    has_mm = getattr(processor, "multimodal_processor", None) is not None
                    print(
                        "[Generation] BLIP3o processor shim multimodal backend: "
                        f"{'enabled' if has_mm else 'disabled'}"
                    )
                loader_used = f"{loader_used}+blip3o_shim"
        elif "blip3o" in model_name_lower and self.is_main_process:
            print(
                "[Generation] Using remote/original BLIP3o loading path "
                "(without forcing local BLIP3o classes)."
            )

        if model is None:
            tokenizer_for_fallback = None
            try:
                processor = AutoProcessor.from_pretrained(self.cfg.model_name, trust_remote_code=True)
            except Exception as proc_exc:
                if self.is_main_process:
                    print(
                        f"[Generation] AutoProcessor load warning for '{self.cfg.model_name}': {repr(proc_exc)}. "
                        "Attempting tokenizer fallback."
                    )
                try:
                    tokenizer_for_fallback = AutoTokenizer.from_pretrained(
                        self.cfg.model_name,
                        trust_remote_code=True,
                    )
                except Exception:
                    tokenizer_for_fallback = AutoTokenizer.from_pretrained(self.cfg.model_name)

            if is_original_blip3o:
                # Original BLIP3o model card loads via AutoModel + custom diffusion pipeline.
                auto_model_kwargs = {
                    "device_map": device_map,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                if attn_impl:
                    auto_model_kwargs["attn_implementation"] = attn_impl
                try:
                    model = AutoModel.from_pretrained(
                        self.cfg.model_name,
                        dtype=dtype,
                        **auto_model_kwargs,
                    )
                    loader_used = "AutoModel(original_blip3o)"
                except TypeError:
                    model = AutoModel.from_pretrained(
                        self.cfg.model_name,
                        torch_dtype=dtype,
                        **auto_model_kwargs,
                    )
                    loader_used = "AutoModel(original_blip3o)"
                except Exception as auto_exc:
                    if self.is_main_process:
                        print(
                            f"[Generation] AutoModel load failed for original BLIP3o: {repr(auto_exc)}. "
                            "Falling back to generic loaders."
                        )
                if model is not None and not hasattr(model, "generate"):
                    if self.is_main_process:
                        print(
                            "[Generation] AutoModel object has no `.generate`; "
                            "falling back to CausalLM-compatible loaders."
                        )
                    model = None

            if model is None:
                try:
                    model = _load_model_with_fallback(
                        self.cfg.model_name,
                        torch_dtype=dtype,
                        device_map=device_map,
                        trust_remote_code=True,
                        attn_implementation=attn_impl,
                    )
                except Exception as generic_exc:
                    # Common on clusters with transformers versions that do not recognize
                    # custom BLIP3o model_type unless local classes are imported first.
                    if "blip3o" in model_name_lower and _looks_like_unregistered_blip3o_arch_error(str(generic_exc)):
                        retry_allow_implicit_repo = allow_implicit_local_repo
                        if is_original_blip3o and not use_local_blip3o_classes:
                            # For original BLIP3o, try a safe implicit local lookup once.
                            # It succeeds only when local classes are original-compatible;
                            # NEXT-style classes are filtered out below.
                            retry_allow_implicit_repo = True
                            if self.is_main_process:
                                print(
                                    "[Generation] Original BLIP3o architecture is unregistered in current "
                                    "transformers stack. Trying safe implicit local class registration."
                                )
                        if self.is_main_process:
                            print(
                                "[Generation] Detected unregistered BLIP3o architecture in transformers. "
                                "Retrying with local BLIP3o class registration."
                            )
                        classes, import_err, added_path = _import_blip3o_classes(
                            allow_implicit_repo=retry_allow_implicit_repo
                        )
                        mm_utils_retry, mm_import_err, _ = _import_blip3o_mm_utils(
                            allow_implicit_repo=retry_allow_implicit_repo
                        )
                        if self.is_main_process and added_path:
                            print(f"[Generation] Added local BLIP3o path for retry: {added_path}")
                        if self.is_main_process and import_err is not None:
                            print(f"[Generation] BLIP3o local import retry warning: {repr(import_err)}")
                        if self.is_main_process and mm_import_err is not None:
                            print(f"[Generation] BLIP3o mm_utils retry warning: {repr(mm_import_err)}")

                        retry_errors: List[str] = []
                        for key in ("causal", "inference", "grpo"):
                            cls = classes.get(key)
                            if cls is None:
                                continue
                            if is_original_blip3o and _is_next_style_blip3o_class(cls):
                                retry_errors.append(
                                    f"{cls.__name__}: local class is Qwen3/BLIP3o-NEXT style and "
                                    "is incompatible with original BLIP3o-Model-8B checkpoints"
                                )
                                continue
                            try:
                                model = _load_from_explicit_class(
                                    cls,
                                    self.cfg.model_name,
                                    torch_dtype=dtype,
                                    device_map=device_map,
                                    attn_implementation=attn_impl,
                                )
                                loader_used = f"explicit-retry:{cls.__name__}"
                                mm_utils.update(mm_utils_retry)
                                break
                            except Exception as retry_exc:
                                retry_errors.append(f"{cls.__name__}: {repr(retry_exc)}")

                        if model is None:
                            detail = " | ".join(retry_errors) if retry_errors else "no BLIP3o classes available"
                            if is_original_blip3o and not explicit_local_repo:
                                raise RuntimeError(
                                    "Failed to load original BLIP3o checkpoint due to unregistered architecture in "
                                    "the current transformers stack.\n"
                                    f"checkpoint={self.cfg.model_name}\n"
                                    "Tried safe implicit local class registration but could not load an "
                                    "original-compatible BLIP3o class.\n"
                                    "To fix, set:\n"
                                    "  BLIP3O_REPO=/absolute/path/to/original/BLIP3o/main\n"
                                    "  BLIP3O_USE_LOCAL_CLASSES=1\n"
                                    "and rerun.\n"
                                    f"Original loader error: {repr(generic_exc)}\n"
                                    f"Retry details: {detail}"
                                ) from generic_exc
                            if is_original_blip3o and any("BLIP3o-NEXT" in e or "Qwen3" in e for e in retry_errors):
                                raise RuntimeError(
                                    "Detected incompatible local BLIP3o code for original checkpoint "
                                    f"'{self.cfg.model_name}'.\n"
                                    "Discovered local BLIP3o classes are BLIP3o-NEXT (Qwen3/SANA), while "
                                    "BLIP3o-Model-8B expects original BLIP3o classes.\n"
                                    "Set BLIP3O_REPO to an original BLIP3o main-branch checkout and keep "
                                    "BLIP3O_USE_LOCAL_CLASSES=1.\n"
                                    f"Retry details: {detail}"
                                ) from generic_exc
                            raise RuntimeError(
                                "Failed to load BLIP3o after local-class retry. "
                                f"Original error: {repr(generic_exc)}; retry details: {detail}"
                            ) from generic_exc
                    else:
                        raise
            if processor is None and tokenizer_for_fallback is not None:
                processor = _build_blip3o_processor(
                    tokenizer=tokenizer_for_fallback,
                    model=model,
                    model_name=self.cfg.model_name,
                    tokenizer_image_token_fn=mm_utils.get("tokenizer_image_token"),
                )
                if self.is_main_process and isinstance(processor, _Blip3oProcessorShim):
                    has_mm = getattr(processor, "multimodal_processor", None) is not None
                    print(
                        "[Generation] BLIP3o processor shim multimodal backend: "
                        f"{'enabled' if has_mm else 'disabled'}"
                    )
                loader_used = f"{loader_used}+tokenizer_shim"
            if processor is None:
                processor = AutoProcessor.from_pretrained(self.cfg.model_name, trust_remote_code=True)

        if self.is_main_process:
            print(
                f"[Generation] Load options: dtype={dtype}, device_map={device_map}, "
                f"attn_implementation={attn_impl or 'default'}, loader={loader_used}"
            )

        if self.cfg.use_lora:
            lcfg = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=list(self.cfg.lora_target_modules),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lcfg)
            if hasattr(model, "add_adapter"):
                try:
                    model.add_adapter("proposer", lcfg)
                except Exception:
                    pass
                try:
                    model.add_adapter("generator", lcfg)
                except Exception:
                    pass

            for name, param in model.named_parameters():
                if "lora_" in name and (
                    ".default." in name or ".proposer." in name or ".generator." in name
                ):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

            model.print_trainable_parameters()

        if is_original_blip3o:
            try:
                pipeline_device = (
                    torch.device(f"cuda:{self.local_rank if self.distributed else self.cfg.cuda_device}")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                pipe_encoder = _resolve_multimodal_encoder_for_pipeline(model)
                self._blip3o_diffusion_pipe = _build_original_blip3o_diffusion_pipeline(
                    self.cfg.model_name,
                    multimodal_encoder=pipe_encoder,
                    processor=processor,
                    torch_dtype=dtype,
                    device=pipeline_device,
                )
                loader_used = f"{loader_used}+diffusion_decoder"
            except Exception as exc:
                if self.is_main_process:
                    print(
                        "[Generation] WARNING: failed to initialize original BLIP3o diffusion "
                        f"decoder pipeline: {repr(exc)}"
                    )

        model.eval()
        return model, processor

    def _sample_image_for_step(self, step: int) -> Tuple[Image.Image, Dict]:
        if self.distributed:
            global_offset = (step - 1) * self.world_size + self.rank
            shuffled_idx = self.pool.indices[global_offset % len(self.pool.indices)]
        else:
            shuffled_idx = self.pool.indices[(step - 1) % len(self.pool.indices)]
        return self.pool.get_image(shuffled_idx)

    def _generate(
        self,
        image: Image.Image,
        prompt: str,
        adapter_name: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        chat_text = _build_chat_text(self.processor, image, prompt)
        inputs = _prepare_mm_inputs(self.processor, self.device, image, chat_text)

        # Guard against BLIP3o/Qwen processor mismatches where image features are present
        # but no image tokens were inserted in input_ids.
        has_image_feats = ("pixel_values" in inputs) or ("images" in inputs)
        if has_image_feats and "input_ids" in inputs:
            image_token_ids = _collect_image_token_ids(self.model)
            token_count = _count_image_tokens_in_inputs(inputs["input_ids"], image_token_ids)
            if token_count == 0:
                mm_proc = getattr(self.processor, "multimodal_processor", None)
                if mm_proc is not None and hasattr(mm_proc, "apply_chat_template"):
                    try:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ]
                        chat_text_mm = mm_proc.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        inputs = mm_proc(
                            text=[chat_text_mm],
                            images=[image],
                            return_tensors="pt",
                            padding=True,
                        ).to(self.device)
                    except Exception:
                        pass

        gen_inputs = _adapt_mm_generate_inputs(self.model, dict(inputs))

        def _run_generate(curr_inputs: Dict[str, torch.Tensor]):
            return self.model.generate(
                **curr_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None),
            )

        with torch.no_grad():
            with use_adapter(self.model, adapter_name):
                try:
                    outputs = _run_generate(gen_inputs)
                except ValueError as exc:
                    unused = _parse_unused_model_kwargs_from_error(exc)
                    if not unused:
                        raise
                    retry_inputs = dict(gen_inputs)
                    # BLIP3o/Qwen wrapper mismatch: retry with key translation.
                    if "images" in unused and "images" in retry_inputs and "pixel_values" not in retry_inputs:
                        retry_inputs["pixel_values"] = retry_inputs["images"]
                    if "pixel_values" in unused and "pixel_values" in retry_inputs and "images" not in retry_inputs:
                        retry_inputs["images"] = retry_inputs["pixel_values"]
                    if "image_grid_thw" in unused and "image_grid_thw" in retry_inputs and "grid_thw" not in retry_inputs:
                        retry_inputs["grid_thw"] = retry_inputs["image_grid_thw"]
                    if "grid_thw" in unused and "grid_thw" in retry_inputs and "image_grid_thw" not in retry_inputs:
                        retry_inputs["image_grid_thw"] = retry_inputs["grid_thw"]
                    for key in unused:
                        retry_inputs.pop(key, None)
                    if retry_inputs == gen_inputs:
                        raise
                    outputs = _run_generate(retry_inputs)
                    gen_inputs = retry_inputs

        input_len = gen_inputs["input_ids"].shape[1] if "input_ids" in gen_inputs else 0
        completion_ids = outputs[0, input_len:]
        text = _decode_tokens(self.processor, completion_ids)
        return text.strip()

    def _caption_image(self, image: Image.Image) -> str:
        caption = self._generate(
            image=image,
            prompt=SOURCE_CAPTION_PROMPT,
            adapter_name="default" if self.cfg.use_lora else None,
            max_new_tokens=self.cfg.max_new_tokens_caption,
            temperature=max(0.2, min(self.cfg.temp, 0.8)),
            top_p=1.0,
        )
        caption = " ".join(caption.split())
        if not caption:
            caption = "An image with multiple visual elements."
        return caption

    def _propose_generation_spec(self, image: Image.Image) -> GenerationSpec:
        raw = self._generate(
            image=image,
            prompt=GEN_PROMPT_TEMPLATE,
            adapter_name="proposer" if self.cfg.use_lora else None,
            max_new_tokens=self.cfg.max_new_tokens_proposer,
            temperature=self.cfg.temp,
            top_p=self.cfg.top_p,
        )
        spec = _parse_generation_spec(raw)
        return spec

    def _sanitize_and_score_spec(self, spec: GenerationSpec) -> Tuple[GenerationSpec, float, Dict[str, float]]:
        filtered: List[GenerationQAPair] = []
        seen_questions = set()
        valid_count = 0

        for qa in spec.qa_pairs:
            q = " ".join((qa.question or "").split())
            e = " ".join((qa.expected or "").split())
            if q and not q.endswith("?"):
                q = f"{q}?"

            q_words = len(_tokenize_words(q))
            e_words = len(_tokenize_words(e))
            is_valid = bool(q and e and q_words <= self.cfg.max_question_words and 1 <= e_words <= self.cfg.max_expected_words)
            if not is_valid:
                continue

            q_key = normalize_answer(q)
            if q_key in seen_questions:
                continue
            seen_questions.add(q_key)
            valid_count += 1
            filtered.append(GenerationQAPair(question=q, expected=e))

        filtered = filtered[:3]
        qa_count = len(filtered)
        raw_count = len(spec.qa_pairs)

        count_score = min(1.0, qa_count / float(max(1, self.cfg.min_spec_qa_pairs)))
        validity_score = qa_count / float(max(1, raw_count))
        uniqueness_score = len({normalize_answer(qa.question) for qa in filtered}) / float(max(1, qa_count))
        all_yes_no = qa_count > 0 and all(_yes_no_polarity(qa.expected) != 0 for qa in filtered)
        yes_no_penalty = 0.2 if all_yes_no and qa_count >= self.cfg.min_spec_qa_pairs else 0.0

        quality = 0.5 * count_score + 0.3 * validity_score + 0.2 * uniqueness_score - yes_no_penalty
        quality = float(max(0.0, min(1.0, quality)))

        sanitized = GenerationSpec(
            prompt=spec.prompt,
            qa_pairs=tuple(filtered),
            raw_output=spec.raw_output,
            fallback_used=spec.fallback_used or (qa_count < self.cfg.min_spec_qa_pairs),
        )
        details = {
            "raw_qa_count": float(raw_count),
            "filtered_qa_count": float(qa_count),
            "count_score": float(count_score),
            "validity_score": float(validity_score),
            "uniqueness_score": float(uniqueness_score),
            "yes_no_penalty": float(yes_no_penalty),
            "spec_quality": float(quality),
        }
        return sanitized, quality, details

    def _generate_image_candidate(self, prompt: str) -> Dict[str, object]:
        api_name = self._generation_api_name
        api_obj = self._generation_api_obj
        if api_name is None or api_obj is None:
            api_name, api_obj, _api_path, inspected = _find_generation_callable(_unwrap_model(self.model))
            self._generation_api_name = api_name
            self._generation_api_obj = api_obj
            self._generation_api_path = _api_path
            if (api_name is None or api_obj is None) and self._blip3o_diffusion_pipe is None:
                inspected_text = "; ".join(inspected[:10]) if inspected else "none"
                raise RuntimeError(
                    "Model does not expose a supported image generation API. "
                    f"model_name={self.cfg.model_name} inspected_wrappers={inspected_text}. "
                    "Expected `generate_images(...)`, `generate_image(...)`, or BLIP3o diffusion pipeline."
                )

        if (api_name is None or api_obj is None) and self._blip3o_diffusion_pipe is not None:
            with torch.no_grad():
                with use_adapter(self.model, "generator" if self.cfg.use_lora else None):
                    out = self._blip3o_diffusion_pipe(
                        prompt=prompt,
                        guidance_scale=self.cfg.generation_guidance_scale,
                        num_inference_steps=self.cfg.generation_num_inference_steps,
                        height=self.cfg.generation_height,
                        width=self.cfg.generation_width,
                    )
            images = getattr(out, "images", None)
            if not images:
                raise RuntimeError("BLIP3o diffusion pipeline returned no images.")
            return {
                "image": _ensure_pil_image(images[0]),
                "policy_prompt": prompt,
                "policy_completion": "",
                "policy_completion_ids": None,
                "backend": "diffusion_pipeline",
            }

        # Path 1: BLIP3o-style API with token trace.
        if api_name == "generate_images":
            text_inputs = _prepare_text_inputs(self.processor, self.device, prompt)
            with torch.no_grad():
                with use_adapter(self.model, "generator" if self.cfg.use_lora else None):
                    gen_fn = getattr(api_obj, "generate_images")
                    try:
                        first_param = next(iter(inspect.signature(gen_fn).parameters.values()), None)
                        first_name = first_param.name if first_param is not None else ""
                    except Exception:
                        first_name = ""
                    try:
                        out = gen_fn(
                            text_inputs.get("input_ids"),
                            attention_mask=text_inputs.get("attention_mask"),
                            max_new_tokens=self.cfg.max_new_tokens_generator,
                            temperature=self.cfg.temp,
                            top_p=self.cfg.top_p,
                            num_inference_steps=self.cfg.generation_num_inference_steps,
                            guidance_scale=self.cfg.generation_guidance_scale,
                            return_tensor=False,
                            enable_progress_bar=False,
                        )
                    except TypeError:
                        # Fallback for alternate custom signatures.
                        if first_name in {"inputs", "input_ids"}:
                            out = gen_fn(
                                text_inputs.get("input_ids"),
                                text_inputs.get("attention_mask"),
                                max_new_tokens=self.cfg.max_new_tokens_generator,
                                temperature=self.cfg.temp,
                                top_p=self.cfg.top_p,
                                num_inference_steps=self.cfg.generation_num_inference_steps,
                                guidance_scale=self.cfg.generation_guidance_scale,
                            )
                        else:
                            out = gen_fn(
                                prompt=prompt,
                                max_new_tokens=self.cfg.max_new_tokens_generator,
                                temperature=self.cfg.temp,
                                top_p=self.cfg.top_p,
                                num_inference_steps=self.cfg.generation_num_inference_steps,
                                guidance_scale=self.cfg.generation_guidance_scale,
                            )

            token_completion = ""
            token_completion_ids = None
            image_out = None

            if isinstance(out, tuple) and len(out) >= 2:
                gen_ids, images = out[0], out[1]
                if isinstance(images, list) and images:
                    image_out = images[0]
                elif isinstance(images, Image.Image):
                    image_out = images

                try:
                    if isinstance(gen_ids, torch.Tensor) and gen_ids.ndim == 2 and text_inputs.get("input_ids") is not None:
                        prompt_len = text_inputs["input_ids"].shape[1]
                        completion_ids = gen_ids[0, prompt_len:]
                        token_completion_ids = completion_ids.detach().cpu().tolist()
                        token_completion = _decode_tokens(self.processor, completion_ids).strip()
                except Exception:
                    token_completion = ""
                    token_completion_ids = None
            else:
                images = out
                if isinstance(images, list) and images:
                    image_out = images[0]
                elif isinstance(images, Image.Image):
                    image_out = images

            if image_out is None:
                raise RuntimeError("generate_images returned no image output.")

            backend_name = "generate_images"
            try:
                pil_img = _ensure_pil_image(image_out)
            except Exception:
                if not self.cfg.allow_latent_visualization_fallback:
                    out_shape = tuple(image_out.shape) if torch.is_tensor(image_out) else None
                    out_dtype = str(image_out.dtype) if torch.is_tensor(image_out) else None
                    raise RuntimeError(
                        "Generation backend returned non-image output, and latent visualization fallback is disabled. "
                        "For scientific runs, this indicates missing decoder integration.\n"
                        f"type={type(image_out).__name__} shape={out_shape} dtype={out_dtype}"
                    )
                pil_img = _latent_tensor_to_pil(
                    image_out,
                    target_size=(self.cfg.generation_width, self.cfg.generation_height),
                )
                if pil_img is None:
                    raise
                backend_name = "generate_images_latent_vis"
                if self.is_main_process and not self._warned_latent_fallback:
                    print(
                        "[Generation] WARNING: using latent-visualization fallback for generated outputs "
                        "(decoder pipeline unavailable)."
                    )
                    self._warned_latent_fallback = True

            return {
                "image": pil_img,
                "policy_prompt": prompt,
                "policy_completion": token_completion,
                "policy_completion_ids": token_completion_ids,
                "backend": backend_name,
            }

        # Path 2: generic single-image API.
        if api_name == "generate_image":
            if self._blip3o_diffusion_pipe is not None:
                with torch.no_grad():
                    with use_adapter(self.model, "generator" if self.cfg.use_lora else None):
                        pipe_out = self._blip3o_diffusion_pipe(
                            prompt=prompt,
                            guidance_scale=self.cfg.generation_guidance_scale,
                            num_inference_steps=self.cfg.generation_num_inference_steps,
                            height=self.cfg.generation_height,
                            width=self.cfg.generation_width,
                        )
                images = getattr(pipe_out, "images", None)
                if not images:
                    raise RuntimeError("BLIP3o diffusion pipeline returned no images.")
                return {
                    "image": _ensure_pil_image(images[0]),
                    "policy_prompt": prompt,
                    "policy_completion": "",
                    "policy_completion_ids": None,
                    "backend": "diffusion_pipeline",
                }

            with torch.no_grad():
                with use_adapter(self.model, "generator" if self.cfg.use_lora else None):
                    gen_fn = getattr(api_obj, "generate_image")
                    fn_sig = None
                    fn_params = set()
                    has_var_kw = False
                    try:
                        fn_sig = inspect.signature(gen_fn)
                        for p in fn_sig.parameters.values():
                            if p.kind == inspect.Parameter.VAR_KEYWORD:
                                has_var_kw = True
                            else:
                                fn_params.add(p.name)
                    except Exception:
                        fn_sig = None

                    tokenizer = _extract_tokenizer_from_processor(self.processor)
                    try:
                        if (("text" in fn_params) or (fn_sig is None)) and tokenizer is not None:
                            call_kwargs = {"text": [prompt], "tokenizer": tokenizer}
                            if "pixel_values" in fn_params or has_var_kw:
                                call_kwargs["pixel_values"] = None
                            if "image_grid_thw" in fn_params or has_var_kw:
                                call_kwargs["image_grid_thw"] = None
                            image_out = gen_fn(**call_kwargs)
                        elif "prompt" in fn_params or has_var_kw:
                            image_out = gen_fn(
                                prompt=prompt,
                                num_inference_steps=self.cfg.generation_num_inference_steps,
                                guidance_scale=self.cfg.generation_guidance_scale,
                                height=self.cfg.generation_height,
                                width=self.cfg.generation_width,
                            )
                        elif tokenizer is not None:
                            image_out = gen_fn([prompt], tokenizer)
                        else:
                            image_out = gen_fn(prompt)
                    except TypeError:
                        if tokenizer is not None:
                            try:
                                image_out = gen_fn(text=[prompt], tokenizer=tokenizer)
                            except TypeError:
                                image_out = gen_fn([prompt], tokenizer)
                        else:
                            image_out = gen_fn(prompt)
            try:
                pil_image = _ensure_pil_image(image_out)
            except Exception:
                decode_obj = api_obj
                if not callable(getattr(decode_obj, "decode_latents", None)):
                    decode_obj, _ = _find_callable_object(_unwrap_model(self.model), "decode_latents")
                pil_image = _decode_blip3o_generate_image_output(decode_obj, image_out) if decode_obj is not None else None
                if pil_image is None:
                    if not self.cfg.allow_latent_visualization_fallback:
                        out_shape = tuple(image_out.shape) if torch.is_tensor(image_out) else None
                        out_dtype = str(image_out.dtype) if torch.is_tensor(image_out) else None
                        raise RuntimeError(
                            "generate_image returned non-image output and decoder path failed. "
                            "Latent visualization fallback is disabled for scientific runs.\n"
                            f"type={type(image_out).__name__} shape={out_shape} dtype={out_dtype}"
                        )
                    pil_image = _latent_tensor_to_pil(
                        image_out,
                        target_size=(self.cfg.generation_width, self.cfg.generation_height),
                    )
                    if pil_image is not None:
                        if self.is_main_process and not self._warned_latent_fallback:
                            print(
                                "[Generation] WARNING: using latent-visualization fallback for generated outputs "
                                "(decoder pipeline unavailable)."
                            )
                            self._warned_latent_fallback = True
                        return {
                            "image": pil_image,
                            "policy_prompt": prompt,
                            "policy_completion": "",
                            "policy_completion_ids": None,
                            "backend": "generate_image_latent_vis",
                        }
                    out_shape = tuple(image_out.shape) if torch.is_tensor(image_out) else None
                    out_dtype = str(image_out.dtype) if torch.is_tensor(image_out) else None
                    raise RuntimeError(
                        "generate_image returned a non-image output and no diffusion/latent decode path succeeded. "
                        f"type={type(image_out).__name__} shape={out_shape} dtype={out_dtype}"
                    )
            return {
                "image": pil_image,
                "policy_prompt": prompt,
                "policy_completion": "",
                "policy_completion_ids": None,
                "backend": "generate_image",
            }

        raise RuntimeError(f"Unsupported generation API mode resolved: {api_name}")

    def _solve_question_with_rollouts(self, image: Image.Image, question: str) -> Dict[str, object]:
        solver_prompt = build_solver_prompt(question)
        rollouts = []
        answers_norm: List[str] = []
        adapter_name: Optional[str] = None
        if self.cfg.use_lora and not self.cfg.verification_use_reference_solver:
            adapter_name = "default"

        for _ in range(self.cfg.num_solver_samples_spec):
            completion = self._generate(
                image=image,
                prompt=solver_prompt,
                adapter_name=adapter_name,
                max_new_tokens=self.cfg.max_new_tokens_solver,
                temperature=self.cfg.temp,
                top_p=self.cfg.top_p,
            )
            answer_raw = _parse_answer(completion)
            answer_norm = normalize_answer(answer_raw)
            rollouts.append(
                {
                    "completion": completion,
                    "answer_raw": answer_raw,
                    "answer_norm": answer_norm,
                    "pre_answer_word_count": pre_answer_word_count(completion),
                }
            )
            answers_norm.append(answer_norm)

        maj_answer, maj_count = majority_vote(answers_norm)
        maj_frac = maj_count / float(max(1, len(answers_norm)))

        hist: Dict[str, int] = {}
        for a in answers_norm:
            hist[a] = hist.get(a, 0) + 1
        probs = [count / float(max(1, len(answers_norm))) for count in hist.values()]
        entropy_nats = shannon_entropy_nats(probs)

        return {
            "solver_prompt": solver_prompt,
            "verification_adapter": "default" if adapter_name == "default" else "reference",
            "rollouts": rollouts,
            "majority_answer": maj_answer,
            "majority_count": maj_count,
            "majority_fraction": maj_frac,
            "entropy_nats": entropy_nats,
            "histogram": hist,
        }

    def _score_spec(
        self,
        image: Image.Image,
        qa_pairs: Tuple[GenerationQAPair, ...],
        *,
        step: Optional[int] = None,
        candidate_idx: Optional[int] = None,
        candidate_count: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[float, float, List[Dict[str, object]]]:
        if not qa_pairs:
            return 0.5, 0.0, []

        qa_logs: List[Dict[str, object]] = []
        score_values = []
        contradiction_values = []

        for qa_idx, qa in enumerate(qa_pairs):
            qa_t0 = time.perf_counter()
            if verbose and self.is_main_process and step is not None:
                if candidate_idx is not None and candidate_count is not None:
                    print(
                        f"[Step {step:05d}][G] scoring cand {candidate_idx + 1}/{candidate_count} "
                        f"qa {qa_idx + 1}/{len(qa_pairs)}"
                    )
                else:
                    print(f"[Step {step:05d}][G] scoring qa {qa_idx + 1}/{len(qa_pairs)}")
            solved = self._solve_question_with_rollouts(image=image, question=qa.question)
            mode_answer = str(solved["majority_answer"])
            maj_frac = float(solved["majority_fraction"])

            match_score = _soft_match(mode_answer, qa.expected)
            combined = 0.7 * match_score + 0.3 * maj_frac

            epol = _yes_no_polarity(qa.expected)
            apol = _yes_no_polarity(mode_answer)
            contradiction = 1.0 if (epol != 0 and apol != 0 and epol != apol) else 0.0

            score_values.append(combined)
            contradiction_values.append(contradiction)

            qa_logs.append(
                {
                    "question": qa.question,
                    "expected": qa.expected,
                    "majority_answer": mode_answer,
                    "majority_fraction": maj_frac,
                    "match_score": match_score,
                    "combined_score": combined,
                    "contradiction": contradiction,
                    "solver": solved,
                }
            )
            if verbose and self.is_main_process and step is not None:
                qa_dt = time.perf_counter() - qa_t0
                print(
                    f"[Step {step:05d}][G]   qa {qa_idx + 1}/{len(qa_pairs)} done in {qa_dt:.1f}s "
                    f"(maj_frac={maj_frac:.2f}, match={match_score:.2f})"
                )

        spec_score = float(sum(score_values) / max(1, len(score_values)))
        contradiction_score = float(sum(contradiction_values) / max(1, len(contradiction_values)))
        return spec_score, contradiction_score, qa_logs

    def _cycle_reward(self, prompt: str, image: Image.Image) -> Tuple[float, str]:
        caption = self._generate(
            image=image,
            prompt=GEN_CYCLE_CAPTION_PROMPT,
            adapter_name="default" if self.cfg.use_lora else None,
            max_new_tokens=self.cfg.max_new_tokens_caption,
            temperature=max(0.2, min(self.cfg.temp, 0.8)),
            top_p=1.0,
        )
        caption = " ".join(caption.split())
        if not caption:
            caption = ""
        score = _jaccard_similarity(prompt, caption)
        return score, caption

    def _score_candidates(
        self,
        prompt: str,
        qa_pairs: Tuple[GenerationQAPair, ...],
        candidates: List[Dict[str, object]],
        spec_quality: float,
        *,
        step: Optional[int] = None,
        verbose: bool = False,
    ) -> List[Dict[str, object]]:
        images = [cand["image"] for cand in candidates]
        diversity = _image_diversity_score(images)

        scored: List[Dict[str, object]] = []
        for idx, cand in enumerate(candidates):
            cand_t0 = time.perf_counter()
            if verbose and self.is_main_process and step is not None:
                backend = str(cand.get("backend", "unknown"))
                print(
                    f"[Step {step:05d}][G] evaluating candidate {idx + 1}/{len(candidates)} "
                    f"(backend={backend})"
                )
            image = cand["image"]
            spec_score, contradiction_score, qa_logs = self._score_spec(
                image=image,
                qa_pairs=qa_pairs,
                step=step,
                candidate_idx=idx,
                candidate_count=len(candidates),
                verbose=verbose,
            )
            cycle_score, cycle_caption = self._cycle_reward(prompt=prompt, image=image)

            base_reward = (
                self.cfg.reward_spec_weight * spec_score
                + self.cfg.reward_cycle_weight * cycle_score
                + self.cfg.reward_diversity_weight * diversity
                - self.cfg.reward_contradiction_weight * contradiction_score
            )
            total_reward = spec_quality * base_reward
            scored.append(
                {
                    "candidate_idx": idx,
                    "backend": cand.get("backend"),
                    "policy_prompt": cand.get("policy_prompt", prompt),
                    "policy_completion": cand.get("policy_completion", ""),
                    "spec_score": spec_score,
                    "contradiction_score": contradiction_score,
                    "cycle_score": cycle_score,
                    "cycle_caption": cycle_caption,
                    "diversity_score": diversity,
                    "base_reward": base_reward,
                    "spec_quality": spec_quality,
                    "total_reward": total_reward,
                    "qa_logs": qa_logs,
                    "image": image,
                }
            )
            if verbose and self.is_main_process and step is not None:
                cand_dt = time.perf_counter() - cand_t0
                print(
                    f"[Step {step:05d}][G] candidate {idx + 1}/{len(candidates)} done in {cand_dt:.1f}s "
                    f"(spec={spec_score:.3f}, cycle={cycle_score:.3f}, total={total_reward:.3f})"
                )
        return scored

    def _update_baseline(self, which: str, reward: float):
        m = self.cfg.baseline_momentum
        if which == "generator":
            self.generator_baseline = m * self.generator_baseline + (1.0 - m) * reward
        elif which == "proposer":
            self.proposer_baseline = m * self.proposer_baseline + (1.0 - m) * reward
        else:
            self.solver_baseline = m * self.solver_baseline + (1.0 - m) * reward

    def _save_candidate_images(self, step: int, scored: List[Dict[str, object]], best_idx: int):
        if self.cfg.save_generated_images_every <= 0:
            return
        if (step % self.cfg.save_generated_images_every) != 0:
            return
        step_dir = self.generated_dir / f"step_{step:05d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for i, cand in enumerate(scored):
            image = cand.get("image")
            if not isinstance(image, Image.Image):
                continue
            flag = "best" if i == best_idx else "cand"
            reward = cand.get("total_reward", 0.0)
            path = step_dir / f"{flag}_{i:02d}_r{reward:.4f}.png"
            try:
                image.save(path)
            except Exception:
                pass

    def _proxy_generator_completion(self, image: Image.Image) -> str:
        completion = self._generate(
            image=image,
            prompt=GENERATOR_PROXY_CAPTION_PROMPT,
            adapter_name="generator" if self.cfg.use_lora else None,
            max_new_tokens=self.cfg.max_new_tokens_caption,
            temperature=max(0.2, min(self.cfg.temp, 0.8)),
            top_p=1.0,
        )
        return " ".join(completion.split())

    def _select_dpo_pair_indices(self, scored: List[Dict[str, object]], best_idx: int) -> Optional[Tuple[int, int]]:
        if len(scored) < 2:
            return None
        candidate_indices = [i for i in range(len(scored)) if i != best_idx]
        if not candidate_indices:
            return None
        rejected_idx = min(candidate_indices, key=lambda i: float(scored[i]["total_reward"]))
        return int(best_idx), int(rejected_idx)

    def _save_checkpoint(self, step: int):
        if not self.is_main_process:
            return
        step_dir = self.run_dir / f"step_{step:05d}"
        tmp_dir = self.run_dir / f"step_{step:05d}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.use_lora:
            adapter_map = (("default", "solver"), ("proposer", "proposer"), ("generator", "generator"))
            for adapter_name, sub_name in adapter_map:
                subdir = tmp_dir / sub_name
                subdir.mkdir(parents=True, exist_ok=True)
                saved = False
                try:
                    self.model.save_pretrained(subdir, selected_adapters=[adapter_name])
                    saved = True
                except TypeError:
                    saved = False
                except Exception:
                    saved = False
                if not saved:
                    with use_adapter(self.model, adapter_name):
                        self.model.save_pretrained(subdir)
                if sub_name == "solver":
                    try:
                        self.processor.save_pretrained(subdir)
                    except Exception:
                        pass
        else:
            self.model.save_pretrained(tmp_dir / "model")
            try:
                self.processor.save_pretrained(tmp_dir / "model")
            except Exception:
                pass

        model_ref = _unwrap_model(self.model)
        trainable_state = {
            name: param.detach().cpu()
            for name, param in model_ref.named_parameters()
            if param.requires_grad
        }
        torch.save(trainable_state, tmp_dir / "trainable_adapters.pt")

        torch.save(self._trainer_state_dict(step), tmp_dir / "trainer_state.pt")

        _json_dump(
            tmp_dir / "trainer_state.json",
            {
                "step": step,
                "solver_baseline": self.solver_baseline,
                "proposer_baseline": self.proposer_baseline,
                "generator_baseline": self.generator_baseline,
                "solver_kl_coef": self.solver_updater.kl_coef if self.solver_updater is not None else None,
                "proposer_kl_coef": self.proposer_updater.kl_coef,
                "generator_kl_coef": self.generator_updater.kl_coef,
                "solver_updater_step": self.solver_updater.step_id if self.solver_updater is not None else None,
                "proposer_updater_step": self.proposer_updater.step_id,
                "generator_updater_step": self.generator_updater.step_id,
            },
        )
        with (tmp_dir / "SAVE_OK").open("w", encoding="utf-8") as f:
            f.write("ok\n")

        if step_dir.exists():
            shutil.rmtree(step_dir, ignore_errors=True)
        os.replace(str(tmp_dir), str(step_dir))

        self._prune_checkpoints()

    def _prune_checkpoints(self):
        if not self.is_main_process:
            return
        keep = max(1, int(self.cfg.max_checkpoints))
        checkpoints = self._list_complete_checkpoints()
        if len(checkpoints) <= keep:
            return
        for path in checkpoints[:-keep]:
            shutil.rmtree(path, ignore_errors=True)

    def _write_ablation_summary(
        self,
        final_step: int,
        *,
        status: str = "completed",
        interrupted_at_step: Optional[int] = None,
        error: Optional[str] = None,
    ):
        if not self.is_main_process:
            return
        _json_dump(
            self.summary_path,
            {
                "experiment": self.cfg.experiment_name,
                "run_dir": str(self.run_dir),
                "generator_update_rule": self.cfg.generator_update_rule,
                "final_step": int(final_step),
                "start_step": int(self.start_step),
                "status": status,
                "interrupted_at_step": interrupted_at_step,
                "error": error,
                "policy_update_counts": self._policy_update_counts,
                "generator_update_mode_counts": self._generator_update_mode_counts,
                "metrics": self._metrics_summary(),
            },
        )

    def _wandb_log_step(
        self,
        *,
        step: int,
        image_path: Optional[str],
        source_caption: str,
        spec: GenerationSpec,
        scored: List[Dict[str, object]],
        best_idx: int,
        spec_quality: float,
        reward_mean_global: float,
        reward_max_global: float,
        reward_min_global: float,
        best_spec_global: float,
        best_cycle_global: float,
        best_diversity_global: float,
        best_contradiction_global: float,
        generator_skipped_reason: Optional[str],
        generator_update_mode: Optional[str],
        proposer_stats: Optional[Dict[str, float]],
        generator_stats: Optional[Dict[str, float]],
    ):
        if not self.is_main_process or self.wandb_run is None:
            return

        best = scored[best_idx]
        metrics: Dict[str, object] = {
            "train/step": step,
            "train/source_caption": source_caption,
            "train/spec_fallback_used": 1.0 if spec.fallback_used else 0.0,
            "train/spec_quality": float(spec_quality),
            "train/spec_qa_count": float(len(spec.qa_pairs)),
            "train/candidate_reward_mean": float(reward_mean_global),
            "train/candidate_reward_max": float(reward_max_global),
            "train/candidate_reward_min": float(reward_min_global),
            "train/best_spec_score": float(best_spec_global),
            "train/best_cycle_score": float(best_cycle_global),
            "train/best_diversity_score": float(best_diversity_global),
            "train/best_contradiction_score": float(best_contradiction_global),
            "train/generator_baseline": self.generator_baseline,
            "train/proposer_baseline": self.proposer_baseline,
            "train/generator_update_skipped": 1.0 if generator_skipped_reason else 0.0,
            "train/generator_update_mode_token_trace": 1.0 if generator_update_mode == "token_trace" else 0.0,
            "train/generator_update_mode_proxy_caption": 1.0 if generator_update_mode == "proxy_caption" else 0.0,
            "train/generator_update_rule_reinforce": 1.0 if self.cfg.generator_update_rule == "reinforce" else 0.0,
            "train/generator_update_rule_dpo": 1.0 if self.cfg.generator_update_rule == "dpo" else 0.0,
            "kl/generator_beta": self.generator_updater.kl_coef,
            "kl/proposer_beta": self.proposer_updater.kl_coef,
            "text/prompt": spec.prompt,
            "text/proposer_raw": spec.raw_output,
            "text/best_cycle_caption": best.get("cycle_caption", ""),
        }
        metrics["train/generator_update_rule"] = self.cfg.generator_update_rule
        if generator_update_mode:
            metrics["train/generator_update_mode"] = generator_update_mode
        if generator_skipped_reason:
            metrics["train/generator_skip_reason"] = generator_skipped_reason
        if image_path:
            metrics["data/image_path"] = image_path

        if proposer_stats:
            metrics.update(
                {
                    "proposer/ce_loss": proposer_stats.get("ce_loss"),
                    "proposer/kl_loss": proposer_stats.get("kl_loss"),
                    "proposer/advantage": proposer_stats.get("advantage"),
                }
            )

        if generator_stats:
            if "dpo_loss" in generator_stats:
                metrics.update(
                    {
                        "generator/dpo_loss": generator_stats.get("dpo_loss"),
                        "generator/dpo_margin": generator_stats.get("preference_margin"),
                        "generator/dpo_pi_gap": generator_stats.get("pi_gap"),
                        "generator/dpo_ref_gap": generator_stats.get("ref_gap"),
                        "generator/dpo_beta": generator_stats.get("dpo_beta"),
                    }
                )
            else:
                metrics.update(
                    {
                        "generator/ce_loss": generator_stats.get("ce_loss"),
                        "generator/kl_loss": generator_stats.get("kl_loss"),
                        "generator/advantage": generator_stats.get("advantage"),
                    }
                )

        if (
            self.cfg.wandb_log_images_every > 0
            and (step % self.cfg.wandb_log_images_every) == 0
            and isinstance(best.get("image"), Image.Image)
        ):
            try:
                metrics["vis/best_generated_image"] = wandb.Image(best["image"], caption=f"step={step}")
            except Exception:
                pass

        try:
            wandb.log(metrics, step=step)
        except Exception as exc:
            print(f"[W&B] log failed at step {step}: {exc}")

    def _generation_step(self, step: int, image: Image.Image, meta: Dict) -> Dict[str, object]:
        verbose = self.is_main_process and (step % self.cfg.log_every == 0)
        step_t0 = time.perf_counter()
        if verbose:
            print(f"[Step {step:05d}][G] generation phase start")

        source_caption = self._caption_image(image)
        spec = self._propose_generation_spec(image)
        if spec.fallback_used and source_caption:
            # Ground the fallback prompt to source content.
            spec = GenerationSpec(
                prompt=f"Create an image variation of: {source_caption}",
                qa_pairs=spec.qa_pairs,
                raw_output=spec.raw_output,
                fallback_used=True,
            )

        spec, spec_quality, spec_quality_details = self._sanitize_and_score_spec(spec)
        if verbose:
            print(
                f"[Step {step:05d}][G] spec ready: qa_pairs={len(spec.qa_pairs)} "
                f"quality={spec_quality:.3f} fallback={int(spec.fallback_used)}"
            )

        candidates: List[Dict[str, object]] = []
        for cand_idx in range(self.cfg.num_generations):
            cand_t0 = time.perf_counter()
            if verbose:
                print(
                    f"[Step {step:05d}][G] generating candidate {cand_idx + 1}/{self.cfg.num_generations}"
                )
            cand = self._generate_image_candidate(spec.prompt)
            candidates.append(cand)
            if verbose:
                backend = str(cand.get("backend", "unknown"))
                cand_dt = time.perf_counter() - cand_t0
                print(
                    f"[Step {step:05d}][G] generated candidate {cand_idx + 1}/{self.cfg.num_generations} "
                    f"in {cand_dt:.1f}s (backend={backend})"
                )
        scored = self._score_candidates(
            prompt=spec.prompt,
            qa_pairs=spec.qa_pairs,
            candidates=candidates,
            spec_quality=spec_quality,
            step=step,
            verbose=verbose,
        )
        best_idx = max(range(len(scored)), key=lambda i: float(scored[i]["total_reward"]))
        best = scored[best_idx]

        proposer_stats = None
        generator_stats = None
        generator_skipped_reason = None
        generator_update_mode = None

        if step % self.cfg.generator_update_freq == 0 and spec_quality >= self.cfg.min_spec_quality_for_update:
            baseline_before = self.generator_baseline
            generator_reward = float(best["total_reward"])
            update_rule = self.cfg.generator_update_rule

            if update_rule == "dpo":
                pair = self._select_dpo_pair_indices(scored, best_idx)
                if pair is None:
                    generator_skipped_reason = "dpo_requires_two_candidates"
                    self._generator_update_mode_counts["skipped"] = (
                        self._generator_update_mode_counts.get("skipped", 0) + 1
                    )
                    self._append_jsonl(
                        self.policy_updates_log_path,
                        {
                            "step": step,
                            "role": "generator",
                            "update_rule": "dpo",
                            "skipped": True,
                            "reason": generator_skipped_reason,
                            "candidate_count": len(scored),
                            "spec_quality": spec_quality,
                        },
                    )
                    self._append_jsonl(
                        self.dpo_pairs_log_path,
                        {
                            "step": step,
                            "skipped": True,
                            "reason": generator_skipped_reason,
                            "candidate_count": len(scored),
                            "best_idx": int(best_idx),
                        },
                    )
                else:
                    chosen_idx, rejected_idx = pair
                    chosen = scored[chosen_idx]
                    rejected = scored[rejected_idx]
                    chosen_reward = float(chosen["total_reward"])
                    rejected_reward = float(rejected["total_reward"])
                    reward_gap = chosen_reward - rejected_reward

                    if reward_gap < float(self.cfg.dpo_min_reward_gap):
                        generator_skipped_reason = "dpo_reward_gap_too_small"
                        self._generator_update_mode_counts["skipped"] = (
                            self._generator_update_mode_counts.get("skipped", 0) + 1
                        )
                        self._append_jsonl(
                            self.policy_updates_log_path,
                            {
                                "step": step,
                                "role": "generator",
                                "update_rule": "dpo",
                                "skipped": True,
                                "reason": generator_skipped_reason,
                                "chosen_candidate_idx": int(chosen_idx),
                                "rejected_candidate_idx": int(rejected_idx),
                                "chosen_reward": chosen_reward,
                                "rejected_reward": rejected_reward,
                                "reward_gap": reward_gap,
                                "min_reward_gap": float(self.cfg.dpo_min_reward_gap),
                                "spec_quality": spec_quality,
                            },
                        )
                        self._append_jsonl(
                            self.dpo_pairs_log_path,
                            {
                                "step": step,
                                "skipped": True,
                                "reason": generator_skipped_reason,
                                "chosen_candidate_idx": int(chosen_idx),
                                "rejected_candidate_idx": int(rejected_idx),
                                "chosen_reward": chosen_reward,
                                "rejected_reward": rejected_reward,
                                "reward_gap": reward_gap,
                                "min_reward_gap": float(self.cfg.dpo_min_reward_gap),
                            },
                        )
                    else:
                        chosen_completion = str(chosen.get("policy_completion", "")).strip()
                        rejected_completion = str(rejected.get("policy_completion", "")).strip()

                        chosen_token_ids = chosen.get("policy_completion_ids")
                        if not isinstance(chosen_token_ids, list):
                            chosen_token_ids = None
                        rejected_token_ids = rejected.get("policy_completion_ids")
                        if not isinstance(rejected_token_ids, list):
                            rejected_token_ids = None

                        update_prompt = str(chosen.get("policy_prompt", spec.prompt))
                        chosen_image: Optional[Image.Image] = None
                        rejected_image: Optional[Image.Image] = None

                        if chosen_completion and rejected_completion:
                            generator_update_mode = "token_trace"
                        else:
                            strategy = (self.cfg.generator_missing_trace_strategy or "proxy").strip().lower()
                            if self.cfg.strict_require_generation_tokens or strategy == "error":
                                raise RuntimeError(
                                    "No generation token trace was returned by the model backend. "
                                    "Set --strict_require_generation_tokens false and use "
                                    "--generator_missing_trace_strategy proxy|skip, or use a backend exposing token traces."
                                )
                            if strategy == "proxy":
                                chosen_raw_image = chosen.get("image")
                                rejected_raw_image = rejected.get("image")
                                if isinstance(chosen_raw_image, Image.Image) and isinstance(rejected_raw_image, Image.Image):
                                    chosen_proxy = self._proxy_generator_completion(chosen_raw_image)
                                    rejected_proxy = self._proxy_generator_completion(rejected_raw_image)
                                    if chosen_proxy and rejected_proxy:
                                        chosen_completion = chosen_proxy
                                        rejected_completion = rejected_proxy
                                        chosen_token_ids = None
                                        rejected_token_ids = None
                                        update_prompt = GENERATOR_PROXY_CAPTION_PROMPT
                                        chosen_image = chosen_raw_image
                                        rejected_image = rejected_raw_image
                                        generator_update_mode = "proxy_caption"
                                    else:
                                        generator_skipped_reason = "dpo_proxy_empty_completion"
                                else:
                                    generator_skipped_reason = "dpo_proxy_missing_image"
                            elif strategy == "skip":
                                generator_skipped_reason = "missing_generation_token_trace"
                            else:
                                raise ValueError(
                                    "Unsupported generator_missing_trace_strategy="
                                    f"{self.cfg.generator_missing_trace_strategy!r}. Expected one of: proxy, skip, error."
                                )

                        if chosen_completion and rejected_completion and generator_skipped_reason is None:
                            generator_stats = self.generator_updater.step(
                                prompt=update_prompt,
                                chosen_completion=chosen_completion,
                                rejected_completion=rejected_completion,
                                device=self.device,
                                chosen_image=chosen_image,
                                rejected_image=rejected_image,
                                chosen_completion_token_ids=chosen_token_ids,
                                rejected_completion_token_ids=rejected_token_ids,
                            )
                            self._policy_update_counts["generator"] += 1
                            self._generator_update_mode_counts[generator_update_mode] = (
                                self._generator_update_mode_counts.get(generator_update_mode, 0) + 1
                            )
                            self._update_baseline("generator", generator_reward)
                            self._sync_state_scalars()

                            self._append_jsonl(
                                self.policy_updates_log_path,
                                {
                                    "step": step,
                                    "role": "generator",
                                    "update_rule": "dpo",
                                    "reward": generator_reward,
                                    "baseline_before": baseline_before,
                                    "baseline_after": self.generator_baseline,
                                    "stats": generator_stats,
                                    "update_mode": generator_update_mode,
                                    "update_prompt": update_prompt,
                                    "used_image_conditioning": chosen_image is not None and rejected_image is not None,
                                    "chosen_candidate_idx": int(chosen_idx),
                                    "rejected_candidate_idx": int(rejected_idx),
                                    "chosen_reward": chosen_reward,
                                    "rejected_reward": rejected_reward,
                                    "reward_gap": reward_gap,
                                    "spec_quality": spec_quality,
                                },
                            )
                            self._append_jsonl(
                                self.dpo_pairs_log_path,
                                {
                                    "step": step,
                                    "chosen_candidate_idx": int(chosen_idx),
                                    "rejected_candidate_idx": int(rejected_idx),
                                    "chosen_reward": chosen_reward,
                                    "rejected_reward": rejected_reward,
                                    "reward_gap": reward_gap,
                                    "update_mode": generator_update_mode,
                                    "prompt": update_prompt,
                                    "chosen_completion_char_len": len(chosen_completion),
                                    "rejected_completion_char_len": len(rejected_completion),
                                    "chosen_completion_token_count": len(chosen_token_ids) if chosen_token_ids else None,
                                    "rejected_completion_token_count": len(rejected_token_ids) if rejected_token_ids else None,
                                    "stats": generator_stats,
                                },
                            )
                        else:
                            if generator_skipped_reason is None:
                                generator_skipped_reason = "dpo_missing_completion"
                            self._generator_update_mode_counts["skipped"] = (
                                self._generator_update_mode_counts.get("skipped", 0) + 1
                            )
                            self._append_jsonl(
                                self.policy_updates_log_path,
                                {
                                    "step": step,
                                    "role": "generator",
                                    "update_rule": "dpo",
                                    "skipped": True,
                                    "reason": generator_skipped_reason,
                                    "candidate_idx": int(best_idx),
                                    "spec_quality": spec_quality,
                                },
                            )
                            self._append_jsonl(
                                self.dpo_pairs_log_path,
                                {
                                    "step": step,
                                    "skipped": True,
                                    "reason": generator_skipped_reason,
                                    "candidate_count": len(scored),
                                    "best_idx": int(best_idx),
                                },
                            )
            else:
                completion = str(best.get("policy_completion", "")).strip()
                completion_token_ids = best.get("policy_completion_ids")
                if not isinstance(completion_token_ids, list):
                    completion_token_ids = None
                update_prompt = str(best.get("policy_prompt", spec.prompt))
                update_image: Optional[Image.Image] = None

                if not completion:
                    strategy = (self.cfg.generator_missing_trace_strategy or "proxy").strip().lower()
                    if self.cfg.strict_require_generation_tokens or strategy == "error":
                        raise RuntimeError(
                            "No generation token trace was returned by the model backend. "
                            "Set --strict_require_generation_tokens false and use "
                            "--generator_missing_trace_strategy proxy|skip, or use a backend exposing token traces."
                        )

                    if strategy == "proxy":
                        best_image = best.get("image")
                        if isinstance(best_image, Image.Image):
                            proxy_completion = self._proxy_generator_completion(best_image)
                            if proxy_completion:
                                completion = proxy_completion
                                completion_token_ids = None
                                update_prompt = GENERATOR_PROXY_CAPTION_PROMPT
                                update_image = best_image
                                generator_update_mode = "proxy_caption"
                            else:
                                generator_skipped_reason = "missing_trace_proxy_empty_completion"
                        else:
                            generator_skipped_reason = "missing_trace_proxy_missing_image"
                    elif strategy == "skip":
                        generator_skipped_reason = "missing_generation_token_trace"
                    else:
                        raise ValueError(
                            "Unsupported generator_missing_trace_strategy="
                            f"{self.cfg.generator_missing_trace_strategy!r}. Expected one of: proxy, skip, error."
                        )

                if completion:
                    if generator_update_mode is None:
                        generator_update_mode = "token_trace"
                    generator_stats = self.generator_updater.step(
                        prompt=update_prompt,
                        completion=completion,
                        reward=generator_reward,
                        baseline=baseline_before,
                        device=self.device,
                        image=update_image,
                        completion_token_ids=completion_token_ids,
                    )
                    self._policy_update_counts["generator"] += 1
                    self._generator_update_mode_counts[generator_update_mode] = (
                        self._generator_update_mode_counts.get(generator_update_mode, 0) + 1
                    )
                    self._update_baseline("generator", generator_reward)
                    self._sync_state_scalars()

                    self._append_jsonl(
                        self.policy_updates_log_path,
                        {
                            "step": step,
                            "role": "generator",
                            "update_rule": "reinforce",
                            "reward": generator_reward,
                            "baseline_before": baseline_before,
                            "baseline_after": self.generator_baseline,
                            "stats": generator_stats,
                            "update_mode": generator_update_mode,
                            "update_prompt": update_prompt,
                            "used_image_conditioning": update_image is not None,
                            "completion_char_len": len(completion),
                            "completion_token_count": len(completion_token_ids) if completion_token_ids else None,
                            "candidate_idx": int(best_idx),
                            "spec_quality": spec_quality,
                        },
                    )
                else:
                    if generator_skipped_reason is None:
                        generator_skipped_reason = "missing_generation_token_trace"
                    self._generator_update_mode_counts["skipped"] = (
                        self._generator_update_mode_counts.get("skipped", 0) + 1
                    )
                    self._append_jsonl(
                        self.policy_updates_log_path,
                        {
                            "step": step,
                            "role": "generator",
                            "update_rule": "reinforce",
                            "skipped": True,
                            "reason": generator_skipped_reason,
                            "candidate_idx": int(best_idx),
                            "spec_quality": spec_quality,
                        },
                    )
        elif step % self.cfg.generator_update_freq == 0:
            generator_skipped_reason = "low_spec_quality"
            self._generator_update_mode_counts["skipped"] = (
                self._generator_update_mode_counts.get("skipped", 0) + 1
            )
            self._append_jsonl(
                self.policy_updates_log_path,
                {
                    "step": step,
                    "role": "generator",
                    "update_rule": self.cfg.generator_update_rule,
                    "skipped": True,
                    "reason": generator_skipped_reason,
                    "spec_quality": spec_quality,
                    "min_spec_quality_for_update": self.cfg.min_spec_quality_for_update,
                },
            )

        proposer_reward = float(sum(float(c["total_reward"]) for c in scored) / max(1, len(scored)))
        if step % self.cfg.proposer_update_freq == 0:
            baseline_before = self.proposer_baseline
            proposer_stats = self.proposer_updater.step(
                image=image,
                prompt=GEN_PROMPT_TEMPLATE,
                completion=spec.raw_output,
                reward=proposer_reward,
                baseline=baseline_before,
                device=self.device,
            )
            self._policy_update_counts["proposer"] += 1
            self._update_baseline("proposer", proposer_reward)
            self._sync_state_scalars()

            self._append_jsonl(
                self.policy_updates_log_path,
                {
                    "step": step,
                    "role": "proposer",
                    "reward": proposer_reward,
                    "baseline_before": baseline_before,
                    "baseline_after": self.proposer_baseline,
                    "stats": proposer_stats,
                },
            )

        self._save_candidate_images(step=step, scored=scored, best_idx=best_idx)

        if verbose:
            step_dt = time.perf_counter() - step_t0
            print(
                f"[Step {step:05d}][G] generation phase done in {step_dt:.1f}s "
                f"(best_idx={best_idx}, best_reward={float(best['total_reward']):.3f})"
            )

        self._append_jsonl(
            self.prompts_log_path,
            {
                "step": step,
                "image_path": meta.get("path"),
                "source_caption": source_caption,
                "prompt": spec.prompt,
                "qa_pairs": [dataclasses.asdict(qa) for qa in spec.qa_pairs],
                "fallback_used": spec.fallback_used,
                "spec_quality": spec_quality,
                "spec_quality_details": spec_quality_details,
                "raw_output": spec.raw_output,
            },
        )

        for cand in scored:
            qa_logs = cand["qa_logs"]
            self._append_jsonl(
                self.candidates_log_path,
                {
                    "step": step,
                    "image_path": meta.get("path"),
                    "candidate_idx": cand["candidate_idx"],
                    "is_best": cand["candidate_idx"] == best_idx,
                    "backend": cand.get("backend"),
                    "policy_prompt": cand.get("policy_prompt"),
                    "policy_completion": cand.get("policy_completion"),
                    "spec_score": cand["spec_score"],
                    "contradiction_score": cand["contradiction_score"],
                    "cycle_score": cand["cycle_score"],
                    "cycle_caption": cand["cycle_caption"],
                    "diversity_score": cand["diversity_score"],
                    "base_reward": cand["base_reward"],
                    "spec_quality": cand["spec_quality"],
                    "total_reward": cand["total_reward"],
                    "qa_logs": qa_logs,
                },
            )

        self._append_jsonl(
            self.rewards_log_path,
            {
                "step": step,
                "image_path": meta.get("path"),
                "prompt": spec.prompt,
                "reward_components": {
                    "spec_weight": self.cfg.reward_spec_weight,
                    "cycle_weight": self.cfg.reward_cycle_weight,
                    "diversity_weight": self.cfg.reward_diversity_weight,
                    "contradiction_weight": self.cfg.reward_contradiction_weight,
                },
                "spec_quality": spec_quality,
                "spec_quality_details": spec_quality_details,
                "candidate_rewards": [float(c["total_reward"]) for c in scored],
                "best_idx": int(best_idx),
                "best_reward": float(best["total_reward"]),
                "best_spec_score": float(best["spec_score"]),
                "best_cycle_score": float(best["cycle_score"]),
                "best_diversity_score": float(best["diversity_score"]),
                "best_contradiction_score": float(best["contradiction_score"]),
                "generator_baseline": self.generator_baseline,
                "proposer_baseline": self.proposer_baseline,
                "generator_update_rule": self.cfg.generator_update_rule,
                "generator_skipped_reason": generator_skipped_reason,
                "generator_update_mode": generator_update_mode,
                "generator_update_stats": generator_stats,
            },
        )

        return {
            "source_caption": source_caption,
            "spec": spec,
            "spec_quality": spec_quality,
            "spec_quality_details": spec_quality_details,
            "scored": scored,
            "best_idx": best_idx,
            "proposer_stats": proposer_stats,
            "generator_stats": generator_stats,
            "generator_update_rule": self.cfg.generator_update_rule,
            "generator_skipped_reason": generator_skipped_reason,
            "generator_update_mode": generator_update_mode,
        }

    def _solver_synthetic_update_from_best(self, step: int, best: Dict[str, object]):
        if self.solver_updater is None:
            return
        if self.cfg.solver_update_freq <= 0 or step % self.cfg.solver_update_freq != 0:
            return

        image = best.get("image")
        if not isinstance(image, Image.Image):
            return

        qa_logs = best.get("qa_logs", [])
        for qa in qa_logs:
            question = str(qa.get("question", "")).strip()
            if not question:
                continue
            completion = self._generate(
                image=image,
                prompt=build_solver_prompt(question),
                adapter_name="default" if self.cfg.use_lora else None,
                max_new_tokens=self.cfg.max_new_tokens_solver,
                temperature=self.cfg.temp,
                top_p=self.cfg.top_p,
            ).strip()
            if not completion:
                continue
            reward = float(qa.get("combined_score", 0.0))

            baseline_before = self.solver_baseline
            stats = self.solver_updater.step(
                image=image,
                prompt=build_solver_prompt(question),
                completion=completion,
                reward=reward,
                baseline=baseline_before,
                device=self.device,
            )
            self._policy_update_counts["solver"] += 1
            self._update_baseline("solver", reward)
            self._sync_state_scalars()

            self._append_jsonl(
                self.policy_updates_log_path,
                {
                    "step": step,
                    "role": "solver",
                    "source": "synthetic_generation",
                    "question": question,
                    "reward": reward,
                    "baseline_before": baseline_before,
                    "baseline_after": self.solver_baseline,
                    "stats": stats,
                },
            )

    def train(self):
        cfg = self.cfg
        if cfg.total_steps <= self.start_step:
            raise ValueError(
                f"total_steps ({cfg.total_steps}) must be greater than start_step ({self.start_step})."
            )

        if self.is_main_process:
            print(f"[Generation] Starting run at: {self.run_dir}")
            print(f"[Generation] Model: {cfg.model_name}")
            print(f"[Generation] Generator update rule: {cfg.generator_update_rule}")
            print(f"[Generation] Images: {len(self.pool)}")
            print(f"[Generation] Step range: {self.start_step + 1}..{cfg.total_steps}")
            if self.distributed:
                print(
                    f"[Generation] Distributed mode: world_size={self.world_size}, "
                    f"effective_batch_per_step={self.world_size}"
                )

        last_completed_step = self.start_step
        last_attempted_step = self.start_step
        try:
            for step in range(self.start_step + 1, cfg.total_steps + 1):
                last_attempted_step = step
                step_t0 = time.perf_counter()

                image, meta = self._sample_image_for_step(step)
                out = self._generation_step(step=step, image=image, meta=meta)
                spec: GenerationSpec = out["spec"]
                spec_quality = float(out["spec_quality"])
                scored: List[Dict[str, object]] = out["scored"]
                best_idx = int(out["best_idx"])

                if self.solver_updater is not None:
                    self._solver_synthetic_update_from_best(step, scored[best_idx])

                rewards = [float(c["total_reward"]) for c in scored]
                reward_mean = sum(rewards) / max(1, len(rewards))
                reward_max = max(rewards) if rewards else 0.0
                reward_min = min(rewards) if rewards else 0.0
                step_duration_sec = time.perf_counter() - step_t0

                reward_mean_g = self._dist_mean(reward_mean)
                reward_max_g = self._dist_mean(reward_max)
                reward_min_g = self._dist_mean(reward_min)
                step_duration_g = self._dist_mean(step_duration_sec)
                spec_quality_g = self._dist_mean(spec_quality)

                best = scored[best_idx]
                best_spec = float(best["spec_score"])
                best_cycle = float(best["cycle_score"])
                best_div = float(best["diversity_score"])
                best_contra = float(best["contradiction_score"])

                best_spec_g = self._dist_mean(best_spec)
                best_cycle_g = self._dist_mean(best_cycle)
                best_div_g = self._dist_mean(best_div)
                best_contra_g = self._dist_mean(best_contra)

                if self.is_main_process and step % cfg.log_every == 0:
                    print(
                        f"[Step {step:05d}] R_mean={reward_mean_g:.3f} R_max={reward_max_g:.3f} "
                        f"spec={best_spec_g:.3f} cycle={best_cycle_g:.3f} div={best_div_g:.3f} contra={best_contra_g:.3f}"
                    )
                    print(f"  Prompt: {spec.prompt}")

                self._append_jsonl(
                    self.iter_log_path,
                    {
                        "step": step,
                        "phase": "generation",
                        "image_path": meta.get("path"),
                        "prompt": spec.prompt,
                        "qa_pairs": [dataclasses.asdict(qa) for qa in spec.qa_pairs],
                        "fallback_used": spec.fallback_used,
                        "spec_quality": spec_quality,
                        "spec_quality_details": out.get("spec_quality_details"),
                        "candidate_rewards": rewards,
                        "best_idx": best_idx,
                        "best_metrics": {
                            "spec_score": best_spec,
                            "cycle_score": best_cycle,
                            "diversity_score": best_div,
                            "contradiction_score": best_contra,
                            "total_reward": float(best["total_reward"]),
                        },
                        "generator_baseline": self.generator_baseline,
                        "proposer_baseline": self.proposer_baseline,
                        "solver_baseline": self.solver_baseline,
                        "generator_update_rule": self.cfg.generator_update_rule,
                        "generator_kl_coef": self.generator_updater.kl_coef,
                        "proposer_kl_coef": self.proposer_updater.kl_coef,
                        "solver_kl_coef": self.solver_updater.kl_coef if self.solver_updater is not None else None,
                        "generator_skipped_reason": out.get("generator_skipped_reason"),
                        "step_duration_sec": step_duration_sec,
                    },
                )

                self._wandb_log_step(
                    step=step,
                    image_path=meta.get("path"),
                    source_caption=str(out["source_caption"]),
                    spec=spec,
                    scored=scored,
                    best_idx=best_idx,
                    spec_quality=spec_quality_g,
                    reward_mean_global=reward_mean_g,
                    reward_max_global=reward_max_g,
                    reward_min_global=reward_min_g,
                    best_spec_global=best_spec_g,
                    best_cycle_global=best_cycle_g,
                    best_diversity_global=best_div_g,
                    best_contradiction_global=best_contra_g,
                    generator_skipped_reason=out.get("generator_skipped_reason"),
                    generator_update_mode=out.get("generator_update_mode"),
                    proposer_stats=out["proposer_stats"],
                    generator_stats=out["generator_stats"],
                )

                self._update_metric("reward_mean", reward_mean_g)
                self._update_metric("reward_max", reward_max_g)
                self._update_metric("reward_min", reward_min_g)
                self._update_metric("best_spec_score", best_spec_g)
                self._update_metric("best_cycle_score", best_cycle_g)
                self._update_metric("best_diversity_score", best_div_g)
                self._update_metric("best_contradiction_score", best_contra_g)
                self._update_metric("spec_quality", spec_quality_g)
                self._update_metric("generator_kl_coef", float(self.generator_updater.kl_coef))
                self._update_metric("proposer_kl_coef", float(self.proposer_updater.kl_coef))
                self._update_metric("step_duration_sec", step_duration_g)
                self._update_metric("spec_fallback_used", 1.0 if spec.fallback_used else 0.0)

                if cfg.save_every > 0 and step % cfg.save_every == 0:
                    self._dist_barrier()
                    self._save_checkpoint(step)
                    self._dist_barrier()

                if (
                    torch.cuda.is_available()
                    and cfg.clear_cache_every > 0
                    and step % cfg.clear_cache_every == 0
                ):
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    gc.collect()

                last_completed_step = step

            if cfg.save_every <= 0 or (cfg.total_steps % cfg.save_every) != 0:
                self._dist_barrier()
                self._save_checkpoint(cfg.total_steps)
                self._dist_barrier()
            self._write_ablation_summary(cfg.total_steps, status="completed")
            if self.is_main_process:
                print(f"[Generation] Training complete. Final checkpoint at step {cfg.total_steps:05d}.")

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            interrupted_step = int(last_attempted_step)
            tb = traceback.format_exc()
            if self.is_main_process:
                print(f"[Generation] Training interrupted at step {interrupted_step}: {error_text}")
                _json_dump(
                    self.run_dir / "interruption.json",
                    {
                        "status": "interrupted",
                        "interrupted_at_step": interrupted_step,
                        "last_completed_step": int(last_completed_step),
                        "error": error_text,
                        "traceback": tb,
                    },
                )

            emergency_step = max(1, interrupted_step)
            try:
                self._dist_barrier()
                self._save_checkpoint(emergency_step)
                self._dist_barrier()
                if self.is_main_process:
                    print(f"[Generation] Emergency checkpoint saved at step {emergency_step:05d}.")
                    _json_dump(
                        self.run_dir / "resume_hint.json",
                        {
                            "resume_from": str(self.run_dir / f"step_{emergency_step:05d}"),
                            "start_step": emergency_step,
                            "total_steps": cfg.total_steps,
                            "command_example": (
                                "python self_evolving/run_experiment.py "
                                f"--experiment {cfg.experiment_name} --data_dir {cfg.data_dir} "
                                f"--output_dir {cfg.output_dir} --run_name {self.run_dir.name} "
                                f"--resume_from {self.run_dir / f'step_{emergency_step:05d}'} "
                                f"--start_step {emergency_step} --total_steps {cfg.total_steps}"
                            ),
                        },
                    )
            except Exception as save_exc:
                if self.is_main_process:
                    print(f"[Generation] Emergency checkpoint failed: {save_exc}")

            self._write_ablation_summary(
                max(last_completed_step, emergency_step),
                status="interrupted",
                interrupted_at_step=interrupted_step,
                error=error_text,
            )
            raise
        finally:
            if self.wandb_run is not None and HAS_WANDB:
                try:
                    wandb.finish()
                except Exception:
                    pass
            if self.distributed and dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    pass
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass


@dataclass
class UnifiedSelfEvolvingConfig(GenerationSelfEvolvingConfig):
    experiment_name: str = "unified_self_evolving"
    understanding_steps_per_cycle: int = 3
    generation_steps_per_cycle: int = 2
    synthetic_solver_update_freq: int = 1


class UnifiedSelfEvolvingTrainer(GenerationSelfEvolvingTrainer):
    def __init__(self, config: UnifiedSelfEvolvingConfig):
        if config.enable_solver_updates and config.solver_update_freq <= 0:
            config.solver_update_freq = max(1, config.synthetic_solver_update_freq)
        super().__init__(config)
        self.ucfg = config

    def _understanding_step(self, step: int, image: Image.Image, meta: Dict) -> Dict[str, object]:
        proposer_prompt = build_proposer_prompt()
        proposer_out = self._generate(
            image=image,
            prompt=proposer_prompt,
            adapter_name="proposer" if self.cfg.use_lora else None,
            max_new_tokens=self.cfg.max_new_tokens_proposer,
            temperature=self.cfg.temp,
            top_p=self.cfg.top_p,
        )
        parsed_question = _parse_first_question(proposer_out).replace("\n", " ").strip()
        question = parsed_question or "What is the most salient object in the image?"

        solver_prompt = build_solver_prompt(question)
        solver_outputs: List[str] = []
        solver_answers_raw: List[str] = []
        solver_answers_norm: List[str] = []
        pre_words: List[int] = []

        for _ in range(self.cfg.num_solver_samples):
            solver_out = self._generate(
                image=image,
                prompt=solver_prompt,
                adapter_name="default" if self.cfg.use_lora else None,
                max_new_tokens=self.cfg.max_new_tokens_solver,
                temperature=self.cfg.temp,
                top_p=self.cfg.top_p,
            )
            answer_raw = _parse_answer(solver_out)
            solver_outputs.append(solver_out)
            solver_answers_raw.append(answer_raw)
            solver_answers_norm.append(normalize_answer(answer_raw))
            pre_words.append(pre_answer_word_count(solver_out))

        maj_answer, maj_count = majority_vote(solver_answers_norm)
        maj_frac = maj_count / float(self.cfg.num_solver_samples)
        hist: Dict[str, int] = {}
        for ans in solver_answers_norm:
            hist[ans] = hist.get(ans, 0) + 1
        probs = [count / float(self.cfg.num_solver_samples) for count in hist.values()]
        entropy_nats = shannon_entropy_nats(probs)

        solver_rewards_raw = [1.0 if ans == maj_answer else 0.0 for ans in solver_answers_norm]
        target_w = max(1, self.cfg.len_penalty_target_words)
        penalties = [min(1.0, max(0.0, (w - target_w) / float(target_w))) for w in pre_words]
        prob_map = {ans: count / float(self.cfg.num_solver_samples) for ans, count in hist.items()}
        solver_probs = [prob_map[ans] for ans in solver_answers_norm]
        solver_rewards_soft = [
            (prob ** self.cfg.solver_soft_gamma) * (1.0 - self.cfg.len_penalty_weight * pen)
            for prob, pen in zip(solver_probs, penalties)
        ]
        proposer_reward = gaussian_reward(entropy_nats, self.cfg.prop_entropy_mu, self.cfg.prop_entropy_sigma)

        solver_stats_list = []
        solver_update_applied = (
            self.solver_updater is not None
            and self.cfg.solver_update_freq > 0
            and (step % self.cfg.solver_update_freq == 0)
        )
        if solver_update_applied:
            for completion, reward in zip(solver_outputs, solver_rewards_soft):
                baseline_before = self.solver_baseline
                stats = self.solver_updater.step(
                    image=image,
                    prompt=solver_prompt,
                    completion=completion,
                    reward=reward,
                    baseline=baseline_before,
                    device=self.device,
                )
                solver_stats_list.append(stats)
                self._policy_update_counts["solver"] += 1
                self._update_baseline("solver", reward)
                self._sync_state_scalars()

        proposer_stats = None
        if step % self.cfg.proposer_update_freq == 0:
            baseline_before = self.proposer_baseline
            proposer_stats = self.proposer_updater.step(
                image=image,
                prompt=proposer_prompt,
                completion=proposer_out,
                reward=proposer_reward,
                baseline=baseline_before,
                device=self.device,
            )
            self._policy_update_counts["proposer"] += 1
            self._update_baseline("proposer", proposer_reward)
            self._sync_state_scalars()

        record = {
            "step": step,
            "phase": "understanding",
            "image_path": meta.get("path"),
            "question": question,
            "proposer_out": proposer_out,
            "solver_answers_raw": solver_answers_raw,
            "solver_answers_norm": solver_answers_norm,
            "solver_rewards_raw": solver_rewards_raw,
            "solver_rewards_soft": solver_rewards_soft,
            "majority_answer": maj_answer,
            "majority_count": maj_count,
            "majority_fraction": maj_frac,
            "entropy_nats": entropy_nats,
            "proposer_reward": proposer_reward,
            "solver_baseline": self.solver_baseline,
            "proposer_baseline": self.proposer_baseline,
            "solver_update_applied": solver_update_applied,
            "solver_stats": solver_stats_list,
            "proposer_stats": proposer_stats,
        }
        self._append_jsonl(self.iter_log_path, record)

        self._append_jsonl(
            self.rewards_log_path,
            {
                "step": step,
                "phase": "understanding",
                "image_path": meta.get("path"),
                "majority_answer": maj_answer,
                "majority_fraction": maj_frac,
                "entropy_nats": entropy_nats,
                "solver_reward_soft_mean": sum(solver_rewards_soft) / max(1, len(solver_rewards_soft)),
                "proposer_reward": proposer_reward,
            },
        )

        if self.is_main_process and step % self.cfg.log_every == 0:
            print(
                f"[Step {step:05d}][U] maj={maj_count}/{self.cfg.num_solver_samples} "
                f"maj_frac={maj_frac:.2f} H={entropy_nats:.3f} P_R={proposer_reward:.3f}"
            )
            print(f"  Q: {question}")

        self._update_metric("u_majority_fraction", self._dist_mean(maj_frac))
        self._update_metric("u_entropy_nats", self._dist_mean(entropy_nats))
        self._update_metric("u_proposer_reward", self._dist_mean(proposer_reward))

        return record

    def train(self):
        cfg = self.ucfg
        if cfg.total_steps <= self.start_step:
            raise ValueError(
                f"total_steps ({cfg.total_steps}) must be greater than start_step ({self.start_step})."
            )
        cycle = max(1, cfg.understanding_steps_per_cycle + cfg.generation_steps_per_cycle)

        if self.is_main_process:
            print(f"[Unified] Starting run at: {self.run_dir}")
            print(f"[Unified] Model: {cfg.model_name}")
            print(f"[Unified] Generator update rule: {cfg.generator_update_rule}")
            print(f"[Unified] Images: {len(self.pool)}")
            print(f"[Unified] Step range: {self.start_step + 1}..{cfg.total_steps}")
            print(
                f"[Unified] Schedule: Ux{cfg.understanding_steps_per_cycle} + Gx{cfg.generation_steps_per_cycle} (cycle={cycle})"
            )

        last_completed_step = self.start_step
        last_attempted_step = self.start_step
        try:
            for step in range(self.start_step + 1, cfg.total_steps + 1):
                last_attempted_step = step
                image, meta = self._sample_image_for_step(step)

                phase_idx = (step - 1) % cycle
                if phase_idx < cfg.understanding_steps_per_cycle:
                    self._understanding_step(step=step, image=image, meta=meta)
                else:
                    out = self._generation_step(step=step, image=image, meta=meta)
                    source_caption = str(out.get("source_caption", ""))
                    spec: GenerationSpec = out["spec"]
                    scored: List[Dict[str, object]] = out["scored"]
                    spec_quality = float(out.get("spec_quality", 0.0))
                    best_idx = int(out["best_idx"])
                    if cfg.synthetic_solver_update_freq > 0 and step % cfg.synthetic_solver_update_freq == 0:
                        self._solver_synthetic_update_from_best(step, scored[best_idx])

                    rewards = [float(c["total_reward"]) for c in scored]
                    reward_mean = sum(rewards) / max(1, len(rewards))
                    reward_max = max(rewards) if rewards else 0.0
                    reward_min = min(rewards) if rewards else 0.0
                    reward_mean_g = self._dist_mean(reward_mean)
                    reward_max_g = self._dist_mean(reward_max)
                    reward_min_g = self._dist_mean(reward_min)
                    spec_quality_g = self._dist_mean(spec_quality)

                    best = scored[best_idx]
                    best_spec = float(best["spec_score"])
                    best_cycle = float(best["cycle_score"])
                    best_div = float(best["diversity_score"])
                    best_contra = float(best["contradiction_score"])
                    best_spec_g = self._dist_mean(best_spec)
                    best_cycle_g = self._dist_mean(best_cycle)
                    best_div_g = self._dist_mean(best_div)
                    best_contra_g = self._dist_mean(best_contra)

                    self._append_jsonl(
                        self.iter_log_path,
                        {
                            "step": step,
                            "phase": "generation",
                            "image_path": meta.get("path"),
                            "prompt": spec.prompt,
                            "best_idx": best_idx,
                            "best_reward": float(scored[best_idx]["total_reward"]),
                            "spec_quality": spec_quality,
                            "generator_update_rule": self.cfg.generator_update_rule,
                            "generator_update_mode": out.get("generator_update_mode"),
                            "generator_skipped_reason": out.get("generator_skipped_reason"),
                            "generator_baseline": self.generator_baseline,
                            "proposer_baseline": self.proposer_baseline,
                            "solver_baseline": self.solver_baseline,
                        },
                    )

                    self._wandb_log_step(
                        step=step,
                        image_path=meta.get("path"),
                        source_caption=source_caption,
                        spec=spec,
                        scored=scored,
                        best_idx=best_idx,
                        spec_quality=spec_quality_g,
                        reward_mean_global=reward_mean_g,
                        reward_max_global=reward_max_g,
                        reward_min_global=reward_min_g,
                        best_spec_global=best_spec_g,
                        best_cycle_global=best_cycle_g,
                        best_diversity_global=best_div_g,
                        best_contradiction_global=best_contra_g,
                        generator_skipped_reason=out.get("generator_skipped_reason"),
                        generator_update_mode=out.get("generator_update_mode"),
                        proposer_stats=out.get("proposer_stats"),
                        generator_stats=out.get("generator_stats"),
                    )

                if cfg.save_every > 0 and step % cfg.save_every == 0:
                    self._dist_barrier()
                    self._save_checkpoint(step)
                    self._dist_barrier()

                if (
                    torch.cuda.is_available()
                    and cfg.clear_cache_every > 0
                    and step % cfg.clear_cache_every == 0
                ):
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    gc.collect()

                last_completed_step = step

            if cfg.save_every <= 0 or (cfg.total_steps % cfg.save_every) != 0:
                self._dist_barrier()
                self._save_checkpoint(cfg.total_steps)
                self._dist_barrier()
            self._write_ablation_summary(cfg.total_steps, status="completed")
            if self.is_main_process:
                print(f"[Unified] Training complete. Final checkpoint at step {cfg.total_steps:05d}.")

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            interrupted_step = int(last_attempted_step)
            tb = traceback.format_exc()
            if self.is_main_process:
                print(f"[Unified] Training interrupted at step {interrupted_step}: {error_text}")
                _json_dump(
                    self.run_dir / "interruption.json",
                    {
                        "status": "interrupted",
                        "interrupted_at_step": interrupted_step,
                        "last_completed_step": int(last_completed_step),
                        "error": error_text,
                        "traceback": tb,
                    },
                )

            emergency_step = max(1, interrupted_step)
            try:
                self._dist_barrier()
                self._save_checkpoint(emergency_step)
                self._dist_barrier()
            except Exception:
                pass

            self._write_ablation_summary(
                max(last_completed_step, emergency_step),
                status="interrupted",
                interrupted_at_step=interrupted_step,
                error=error_text,
            )
            raise
        finally:
            if self.wandb_run is not None and HAS_WANDB:
                try:
                    wandb.finish()
                except Exception:
                    pass
            if self.distributed and dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    pass
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
