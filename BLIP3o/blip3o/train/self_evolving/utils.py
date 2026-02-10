"""
Shared utilities for the self-evolving training pipeline.
Ported from self_evolving/experiments/understanding.py.
"""

import contextlib
import importlib.util
import json
import math
import pathlib
import random
import re
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# BLIP3o-native helpers — imported lazily to avoid circular deps at module
# level, but cached after first use.
# ---------------------------------------------------------------------------
_BLIP3O_MM_UTILS_LOADED = False
_tokenizer_image_token = None  # type: ignore
_process_images_fn = None  # type: ignore
_conv_templates = None  # type: ignore
_IMAGE_TOKEN_IDX = None  # type: ignore
_qwen_vl_processor = None  # type: ignore — cached Qwen2.5-VL processor for image prep


def _ensure_blip3o_mm_utils():
    """Lazily import BLIP3o multimodal utilities (tokenizer_image_token, etc.)."""
    global _BLIP3O_MM_UTILS_LOADED, _tokenizer_image_token, _process_images_fn
    global _conv_templates, _IMAGE_TOKEN_IDX
    if _BLIP3O_MM_UTILS_LOADED:
        return
    try:
        from blip3o.mm_utils import tokenizer_image_token as _tit
        from blip3o.mm_utils import process_images as _pi
        from blip3o.conversation import conv_templates as _ct
        from blip3o.constants import IMAGE_TOKEN_IDX as _iti
        _tokenizer_image_token = _tit
        _process_images_fn = _pi
        _conv_templates = _ct
        _IMAGE_TOKEN_IDX = _iti
    except ImportError:
        pass
    _BLIP3O_MM_UTILS_LOADED = True


def _get_qwen_vl_processor():
    """Return a cached Qwen2.5-VL processor for image preprocessing.

    The BLIP3o InferenceLM model inherits from Qwen2_5_VLForConditionalGeneration,
    so its forward()/generate() expects Qwen2.5-VL-format pixel_values and
    image_grid_thw.  The official inference.py loads this processor from
    ``Qwen/Qwen2.5-VL-7B-Instruct``.
    """
    global _qwen_vl_processor
    if _qwen_vl_processor is not None:
        return _qwen_vl_processor
    try:
        from transformers import AutoProcessor
        _qwen_vl_processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True,
        )
    except Exception:
        pass
    return _qwen_vl_processor


def _is_bare_tokenizer(processor) -> bool:
    """Return True when ``processor`` is a plain tokenizer (not a multimodal processor).

    BLIP3o's ``AutoProcessor.from_pretrained(...)`` returns a
    ``PreTrainedTokenizerFast`` that does NOT accept ``images=`` kwarg.
    """
    # A "real" multimodal processor (Qwen2VLProcessor, etc.) wraps a tokenizer
    # internally and exposes an ``image_processor`` attribute.
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        return False
    # Check if it has the tokenizer's encode method but NOT the processor's
    # multi-modal __call__ accepting images=.
    cls_name = type(processor).__name__
    if "Tokenizer" in cls_name:
        return True
    # Fallback: try calling with images kwarg — if it raises TypeError it's bare.
    return not hasattr(processor, "image_processor")

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------
try:
    from peft import LoraConfig, TaskType, get_peft_model  # noqa: F401

    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

try:
    import wandb  # noqa: F401

    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

try:
    import numpy as np  # noqa: F401

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Default LoRA targets
# ---------------------------------------------------------------------------
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "mm_projector",
)

# ---------------------------------------------------------------------------
# Dtype / device helpers
# ---------------------------------------------------------------------------


def _safe_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if dtype == "float16" and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _resolve_attn_implementation(requested: str) -> Optional[str]:
    choice = (requested or "auto").strip().lower()
    if choice in {"none", "off", "disable", "disabled"}:
        return None
    if choice in {"sdpa", "eager", "flash_attention_2"}:
        return choice
    if choice != "auto":
        return None

    if not torch.cuda.is_available():
        return None
    if getattr(torch.version, "hip", None):
        # On ROCm, SDPA is the most stable default backend.
        return "sdpa"
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _infer_primary_device(
    model: torch.nn.Module, fallback_cuda_device: int
) -> torch.device:
    model_ref = _unwrap_model(model)
    hf_device_map = getattr(model_ref, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        cuda_devs = [
            value
            for value in hf_device_map.values()
            if isinstance(value, str) and value.startswith("cuda")
        ]
        if cuda_devs:
            try:
                idx = min(int(item.split(":")[1]) for item in cuda_devs)
                return torch.device(f"cuda:{idx}")
            except Exception:
                pass
    if torch.cuda.is_available():
        return torch.device(f"cuda:{fallback_cuda_device}")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Text parsing helpers
# ---------------------------------------------------------------------------


def strip_tags(text: str, tag: str) -> Optional[str]:
    lt = f"<{tag}>"
    rt = f"</{tag}>"
    if lt in text and rt in text:
        return text.split(lt, 1)[1].split(rt, 1)[0].strip()
    return None


def normalize_answer(ans: str) -> str:
    s = ans.strip().lower()
    s = s.replace(",", " ")
    s = " ".join(s.split())
    return s.strip(" .,:;!?\"'")


def majority_vote(answers: List[str]) -> Tuple[str, int]:
    counts: Dict[str, int] = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    return max(counts.items(), key=lambda x: x[1])


def shannon_entropy_nats(probs: List[float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(max(p, eps)) for p in probs if p > 0.0)


def pre_answer_word_count(text: str) -> int:
    idx = text.lower().find("<answer>")
    prefix = text if idx == -1 else text[:idx]
    return len(prefix.strip().split())


def gaussian_reward(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return math.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma))


def _parse_first_question(text: str) -> str:
    tagged = strip_tags(text, "question")
    if tagged:
        return tagged
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines:
        line = re.sub(r"^\d+[\).\-\s]*", "", line).strip()
        if line.endswith("?"):
            return line
    if lines:
        return lines[0]
    return ""


def _parse_answer(text: str) -> str:
    tagged = strip_tags(text, "answer")
    if tagged:
        return tagged
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else "unknown"


# ---------------------------------------------------------------------------
# Seed / reproducibility
# ---------------------------------------------------------------------------


def _set_global_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if HAS_NUMPY:
        import numpy as np

        np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _json_dump(path: pathlib.Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _collect_git_info(repo_root: pathlib.Path) -> Dict[str, Optional[str]]:
    def run_git(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(
                ["git"] + args,
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out
        except Exception:
            return None

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "is_dirty": run_git(["status", "--porcelain"]) not in (None, ""),
    }


# ---------------------------------------------------------------------------
# Adapter context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def use_adapter(model: torch.nn.Module, adapter_name: Optional[str]):
    model_ref = _unwrap_model(model)
    if not hasattr(model_ref, "set_adapter"):
        yield
        return

    if adapter_name is None and hasattr(model_ref, "disable_adapter"):
        with model_ref.disable_adapter():
            yield
        return

    prev_adapter = getattr(model_ref, "active_adapter", None)
    switched = False
    if adapter_name is not None:
        try:
            model_ref.set_adapter(adapter_name)
            switched = True
        except Exception:
            switched = False
    try:
        yield
    finally:
        if switched and prev_adapter is not None:
            try:
                model_ref.set_adapter(prev_adapter)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Token decode / chat helpers
# ---------------------------------------------------------------------------


def _decode_tokens(processor, token_ids: torch.Tensor) -> str:
    if hasattr(processor, "decode"):
        return processor.decode(token_ids, skip_special_tokens=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor does not expose decode/tokenizer.decode")
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _build_chat_text(
    processor, image: Image.Image, prompt: str
) -> str:
    """Build a chat-formatted text string for multimodal generation.

    For BLIP3o (bare tokenizer as processor) we use the official
    ``conv_templates['qwen']`` conversation template which produces
    CHATML-style text with ``<image>`` as a string placeholder.

    For true multimodal processors (e.g. Qwen2VLProcessor) we try the
    multimodal content-list format first, then fall back.
    """
    _ensure_blip3o_mm_utils()

    # ---- BLIP3o path: use conv_templates to build prompt ----
    if _is_bare_tokenizer(processor) and _conv_templates is not None:
        conv = _conv_templates['qwen'].copy()
        conv.append_message(conv.roles[0], f"<image>\n{prompt}")
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    # ---- Multimodal-processor path (Qwen2.5-VL, etc.) ----
    if hasattr(processor, "apply_chat_template"):
        # Try Qwen2.5-VL style multi-modal content list first
        messages_mm = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            return processor.apply_chat_template(
                messages_mm, tokenize=False, add_generation_prompt=True
            )
        except (TypeError, Exception):
            pass
        # Fallback: simple string content with <image> placeholder
        messages_str = [
            {
                "role": "user",
                "content": "<image>\n" + prompt,
            }
        ]
        try:
            return processor.apply_chat_template(
                messages_str, tokenize=False, add_generation_prompt=True
            )
        except (TypeError, Exception):
            pass
    return "<image>\n" + prompt


def _prepare_mm_inputs(
    processor,
    device: torch.device,
    image: Image.Image,
    chat_text: str,
    model=None,
):
    """Prepare multimodal inputs for model.generate() / model.forward().

    For BLIP3o (bare tokenizer): uses ``tokenizer_image_token()`` to create
    input_ids with image-token placeholders and ``process_images()`` to
    produce pixel-value tensors.  The returned dict contains ``input_ids``,
    ``attention_mask``, and ``images`` — matching the BLIP3o model's
    ``generate(inputs=..., images=...)`` signature.

    For true multimodal processors: delegates to
    ``processor(text=..., images=..., ...)``.
    """
    _ensure_blip3o_mm_utils()

    if _is_bare_tokenizer(processor) and _tokenizer_image_token is not None:
        # --- BLIP3o native path ---
        # The BLIP3o InferenceLM inherits from Qwen2_5_VLForConditionalGeneration.
        # Its generate()/forward() expects Qwen2.5-VL-format inputs:
        #   input_ids, attention_mask, pixel_values, image_grid_thw
        #
        # We use:
        #   1. tokenizer_image_token() for input_ids (with <image> token IDs)
        #   2. Qwen2.5-VL processor for pixel_values + image_grid_thw
        input_ids = _tokenizer_image_token(
            chat_text, processor, _IMAGE_TOKEN_IDX, return_tensors="pt"
        ).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Use the Qwen2.5-VL processor to get pixel_values and image_grid_thw.
        # This matches the official inference.py which loads processor from
        # "Qwen/Qwen2.5-VL-7B-Instruct" separately.
        if image is not None:
            qwen_proc = _get_qwen_vl_processor()
            if qwen_proc is not None:
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": "placeholder"},
                            ],
                        }
                    ]
                    # Use qwen_vl_utils.process_vision_info if available,
                    # otherwise fall back to direct processor call.
                    try:
                        from qwen_vl_utils import process_vision_info
                        image_inputs, _ = process_vision_info(messages)
                    except ImportError:
                        image_inputs = [image]
                    img_inputs = qwen_proc.image_processor(
                        images=image_inputs, return_tensors="pt"
                    )
                    if "pixel_values" in img_inputs:
                        result["pixel_values"] = img_inputs["pixel_values"].to(device)
                    if "image_grid_thw" in img_inputs:
                        result["image_grid_thw"] = img_inputs["image_grid_thw"].to(device)
                except Exception:
                    pass

        return result

    # --- Standard multimodal processor path ---
    inputs = processor(
        text=[chat_text], images=[image], return_tensors="pt", padding=True
    )
    return inputs.to(device)


# ---------------------------------------------------------------------------
# Gradient clipping / parameter collection
# ---------------------------------------------------------------------------


def _clip_grad_norm_multi_device(
    params: Iterable[torch.nn.Parameter], max_norm: float
):
    grouped: Dict[torch.device, List[torch.nn.Parameter]] = {}
    for p in params:
        if p.grad is None:
            continue
        grouped.setdefault(p.grad.device, []).append(p)
    for group in grouped.values():
        torch.nn.utils.clip_grad_norm_(group, max_norm)


def _collect_trainable_params(
    model: torch.nn.Module,
    adapter_name: Optional[str],
) -> Iterable[torch.nn.Parameter]:
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if adapter_name is None:
        return [p for _, p in trainable]

    selected = [
        p
        for n, p in trainable
        if (f".{adapter_name}." in n) or (f"{adapter_name}." in n)
    ]
    if not selected:
        preview = [name for name, _ in trainable[:20]]
        raise RuntimeError(
            f"No trainable parameters matched adapter '{adapter_name}'. "
            f"Trainable preview: {preview}"
        )
    return selected
