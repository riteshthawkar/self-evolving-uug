"""
Shared utilities for the self-evolving training pipeline.
Ported from self_evolving/experiments/understanding.py.
"""

import contextlib
import datetime as dt
import importlib.util
import json
import math
import os
import pathlib
import random
import re
import subprocess
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _normalize_vl_image_size(image: Image.Image) -> Image.Image:
    """Resize very large/small inputs to stable bounds for VL preprocessing.

    This reduces sporadic backend failures on extreme resolutions while keeping
    behavior configurable by environment variables:
      - SE_MAX_IMAGE_SIDE (default: 1024, <=0 disables upper-bound resize)
      - SE_MIN_IMAGE_SIDE (default: 56)
      - SE_IMAGE_SIZE_MULTIPLE (default: 28)
    """
    if not isinstance(image, Image.Image):
        return image

    try:
        max_side = int(os.environ.get("SE_MAX_IMAGE_SIDE", "1024"))
    except Exception:
        max_side = 1024
    try:
        min_side = int(os.environ.get("SE_MIN_IMAGE_SIDE", "56"))
    except Exception:
        min_side = 56
    try:
        size_multiple = int(os.environ.get("SE_IMAGE_SIZE_MULTIPLE", "28"))
    except Exception:
        size_multiple = 28

    size_multiple = max(1, size_multiple)
    min_side = max(size_multiple, min_side)

    w, h = image.size
    if w <= 0 or h <= 0:
        return image

    scale = 1.0
    longest = max(w, h)
    shortest = min(w, h)
    if max_side > 0 and longest > max_side:
        scale = min(scale, max_side / float(longest))
    if shortest < min_side:
        scale = max(scale, min_side / float(shortest))

    if scale != 1.0:
        new_w = max(size_multiple, int(round(w * scale)))
        new_h = max(size_multiple, int(round(h * scale)))
    else:
        new_w, new_h = w, h

    # Keep dimensions aligned to patch multiple.
    new_w = max(size_multiple, (new_w // size_multiple) * size_multiple)
    new_h = max(size_multiple, (new_h // size_multiple) * size_multiple)

    if new_w == w and new_h == h:
        return image
    return image.resize((new_w, new_h), Image.BICUBIC)

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
DEFAULT_TEXT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

DEFAULT_LORA_TARGETS = DEFAULT_TEXT_LORA_TARGETS

DEFAULT_LEGACY_LORA_TARGETS = (
    *DEFAULT_TEXT_LORA_TARGETS,
    "mm_projector",
)

DEFAULT_SOLVER_MERGER_LORA_TARGETS = (
    "visual.merger.mlp.0",
    "visual.merger.mlp.2",
)

DEFAULT_DIT_LORA_TARGETS = (
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "caption_projection.linear_1",
    "caption_projection.linear_2",
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


def _flash_attention_2_available() -> bool:
    if importlib.util.find_spec("flash_attn") is None:
        return False
    try:
        __import__("flash_attn")
        return True
    except Exception:
        return False


def _resolve_attn_implementation(requested: str) -> Optional[str]:
    choice = (requested or "auto").strip().lower()
    if choice in {"none", "off", "disable", "disabled"}:
        return None
    if choice in {"sdpa", "eager"}:
        return choice
    if choice == "flash_attention_2":
        if _flash_attention_2_available():
            return choice
        if os.environ.get("SE_STRICT_FLASH_ATTN", "0") == "1":
            return choice
        warnings.warn(
            "ATTN_IMPL=flash_attention_2 was requested, but flash_attn is not "
            "importable in this environment. Falling back to sdpa. Set "
            "SE_STRICT_FLASH_ATTN=1 to fail fast instead.",
            RuntimeWarning,
        )
        return "sdpa" if torch.cuda.is_available() else None
    if choice != "auto":
        return None

    if not torch.cuda.is_available():
        return None
    if getattr(torch.version, "hip", None):
        # On ROCm, SDPA is the most stable default backend.
        return "sdpa"
    if _flash_attention_2_available():
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


def normalize_answer(ans: str, max_words: int = 8) -> str:
    """Normalize and optionally truncate an answer for self-consistency voting.

    Truncating to ``max_words`` prevents phrasing-level variation in verbose
    answers from creating fake disagreement (e.g. two sentences that say the
    same thing differently would match after truncation to their core).
    """
    s = ans.strip().lower()
    s = s.replace(",", " ")
    s = " ".join(s.split())
    s = s.strip(" .,:;!?\"'")
    # Truncate to max_words to avoid phrasing-level fake disagreement.
    if max_words > 0:
        words = s.split()
        if len(words) > max_words:
            s = " ".join(words[:max_words])
    return s


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


def _parse_all_questions(text: str) -> List[str]:
    """Parse all candidate questions from a multi-question proposer response.

    Handles the XML format produced by ``build_proposer_multi_prompt``:

        <questions>
          <question id="1">
            <solver_failure_reasoning>...</solver_failure_reasoning>
            <text>...question text...</text>
            <rationale>...</rationale>
          </question>
          ...
        </questions>

    Falls back gracefully:
    - If <text> tags are present, extracts their contents in order.
    - If <question id="N"> blocks are present (no <text> sub-tag), extracts the
      innermost non-tag content that ends with '?'.
    - If neither pattern matches, returns [_parse_first_question(text)] so the
      caller always receives at least one candidate.

    Returns a list of question strings (may be empty strings for failed parses).
    """
    # Primary: extract <text>...</text> blocks inside each <question id="N"> block
    questions: List[str] = []

    # Try to find all <question id="..."> blocks first
    question_blocks = re.findall(
        r'<question[^>]*>(.*?)</question>',
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if question_blocks:
        for block in question_blocks:
            # Prefer <text>...</text> sub-tag inside the block
            text_match = re.search(
                r'<text>(.*?)</text>',
                block,
                re.DOTALL | re.IGNORECASE,
            )
            if text_match:
                q = text_match.group(1).strip().replace("\n", " ")
                questions.append(q)
                continue

            # Fall back: strip all XML sub-tags and look for a '?' line
            stripped = re.sub(r'<[^>]+>', ' ', block).strip()
            lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
            found = ""
            for ln in lines:
                ln_clean = re.sub(r"^\d+[\).\-\s]*", "", ln).strip()
                if ln_clean.endswith("?"):
                    found = ln_clean
                    break
            if not found and lines:
                found = lines[0]
            questions.append(found)
        # Filter out empty slots but preserve order
        questions = [q for q in questions if q]

    if questions:
        return questions

    # Secondary fallback: plain <question>...</question> (no id attribute) — legacy format
    plain = strip_tags(text, "question")
    if plain:
        return [plain.replace("\n", " ").strip()]

    # Tertiary fallback: look for lines ending with '?'
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates: List[str] = []
    for ln in lines:
        ln_clean = re.sub(r"^\d+[\).\-\s]*", "", ln).strip()
        # Skip lines that are only XML tags
        if re.match(r'^<[^>]+>$', ln_clean):
            continue
        if ln_clean.endswith("?"):
            candidates.append(ln_clean)
    if candidates:
        return candidates

    # Ultimate fallback: return single result from _parse_first_question
    fallback = _parse_first_question(text)
    return [fallback] if fallback else []


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


def _build_text_only_chat(processor, prompt: str) -> str:
    """Build a chat-formatted text string for TEXT-ONLY generation (no image).

    Used by the imageless proposer mode (E5) where the proposer receives only a
    topic description and generates a spec without any visual input.

    For BLIP3o (bare tokenizer as processor) we use the official
    ``conv_templates['qwen']`` conversation template but WITHOUT the ``<image>``
    placeholder, producing a pure text prompt.

    For true multimodal processors we try ``apply_chat_template`` with text-only
    messages.
    """
    _ensure_blip3o_mm_utils()

    # ---- BLIP3o path: use conv_templates to build prompt (no image) ----
    if _is_bare_tokenizer(processor) and _conv_templates is not None:
        conv = _conv_templates['qwen'].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    # ---- Multimodal-processor path (text-only) ----
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        try:
            return processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (TypeError, Exception):
            pass
    return prompt


def _prepare_text_only_inputs(
    processor,
    device: torch.device,
    chat_text: str,
):
    """Prepare text-only inputs for model.generate() / model.forward().

    Like ``_prepare_mm_inputs`` but without any image processing.
    Used by the imageless proposer mode (E5).
    """
    _ensure_blip3o_mm_utils()

    if _is_bare_tokenizer(processor):
        # BLIP3o bare tokenizer: simple tokenization
        input_ids = processor(
            chat_text,
            return_tensors="pt",
            padding=True,
        )["input_ids"].to(device)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    # True multimodal processor: use text-only path
    try:
        inputs = processor(
            text=[chat_text],
            return_tensors="pt",
            padding=True,
        )
    except TypeError:
        inputs = processor(
            chat_text,
            return_tensors="pt",
            padding=True,
        )
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def _extract_qwen_user_assistant_text(chat_text: str) -> Tuple[str, Optional[str]]:
    """Extract user prompt text and optional assistant completion from CHATML text.

    ``chat_text`` in this pipeline is often built via ``conv_templates['qwen']``
    and may include:
      - system/user/assistant role markers
      - an ``<image>`` placeholder
      - optional assistant completion appended to the generation prompt

    We normalize this into:
      1) user text (without role/control tokens)
      2) assistant text if present, else ``None``.
    """
    text = str(chat_text or "")
    user_text = text
    assistant_text: Optional[str] = None

    if "<image>" in user_text:
        user_text = user_text.split("<image>", 1)[1]

    assistant_marker = "<|im_start|>assistant"
    if assistant_marker in user_text:
        user_text, assistant_text = user_text.split(assistant_marker, 1)

    if "<|im_end|>" in user_text:
        user_text = user_text.split("<|im_end|>", 1)[0]
    user_text = user_text.strip()

    if assistant_text is not None:
        assistant_text = assistant_text.strip()
        if "<|im_end|>" in assistant_text:
            assistant_text = assistant_text.split("<|im_end|>", 1)[0]
        if "<|im_start|>" in assistant_text:
            assistant_text = assistant_text.split("<|im_start|>", 1)[0]
        assistant_text = assistant_text.strip() or None

    if not user_text:
        user_text = text.strip() or "Describe the image."
    return user_text, assistant_text


def _prepare_mm_inputs(
    processor,
    device: torch.device,
    image: Image.Image,
    chat_text: str,
    model=None,
):
    """Prepare multimodal inputs for model.generate() / model.forward().

    For BLIP3o (bare tokenizer): prefers a Qwen2.5-VL chat-template path
    (``apply_chat_template`` + ``processor(...)``) so image-token expansion
    matches image features for InferenceLM. If unavailable, falls back to
    legacy ``tokenizer_image_token()`` text-only placeholders.

    For true multimodal processors: delegates to
    ``processor(text=..., images=..., ...)``.
    """
    _ensure_blip3o_mm_utils()
    image = _normalize_vl_image_size(image)

    if _is_bare_tokenizer(processor) and _tokenizer_image_token is not None:
        # --- BLIP3o native path ---
        # The BLIP3o InferenceLM is Qwen2.5-VL based. To avoid image-token /
        # image-feature mismatches, we must build multimodal inputs through a
        # Qwen2.5-VL processor chat-template path (same strategy as official eval).
        if image is not None:
            qwen_proc = _get_qwen_vl_processor()
            if qwen_proc is not None and hasattr(qwen_proc, "apply_chat_template"):
                try:
                    user_text, assistant_text = _extract_qwen_user_assistant_text(chat_text)

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": user_text},
                            ],
                        }
                    ]
                    add_generation_prompt = True
                    if assistant_text is not None:
                        # Preserve appended completion text for training-loss
                        # construction (prompt + completion) instead of dropping it.
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": assistant_text}],
                            }
                        )
                        add_generation_prompt = False
                    text_mm = qwen_proc.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                    )
                    try:
                        from qwen_vl_utils import process_vision_info
                        image_inputs, _ = process_vision_info(messages)
                    except ImportError:
                        image_inputs = [image]
                    mm_inputs = qwen_proc(
                        text=[text_mm],
                        images=image_inputs,
                        return_tensors="pt",
                        padding=True,
                    )
                    return mm_inputs.to(device)
                except Exception:
                    pass

        # Fallback path (legacy BLIP-style <image> token insertion).
        input_ids = _tokenizer_image_token(
            chat_text, processor, _IMAGE_TOKEN_IDX, return_tensors="pt"
        ).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    # --- Standard multimodal processor path ---
    inputs = processor(
        text=[chat_text], images=[image], return_tensors="pt", padding=True
    )
    return inputs.to(device)


# ---------------------------------------------------------------------------
# Gradient clipping / parameter collection
# ---------------------------------------------------------------------------


def _gradients_are_finite(params: Iterable[torch.nn.Parameter]) -> bool:
    """Return False if any materialized gradient contains NaN/Inf values."""
    for p in params:
        if p.grad is None:
            continue
        try:
            if not bool(torch.isfinite(p.grad.detach()).all().item()):
                return False
        except RuntimeError:
            return False
    return True


def _clip_grad_norm_multi_device(
    params: Iterable[torch.nn.Parameter], max_norm: float
) -> bool:
    """Clip gradients grouped by device and report whether they stayed finite.

    The training loop treats ``False`` as a skipped optimizer step.  This keeps
    AdamW moment estimates from being polluted by NaN/Inf gradients, which was a
    common failure mode in unstable early BLIP3o runs.
    """
    params = list(params)
    if not _gradients_are_finite(params):
        return False
    grouped: Dict[torch.device, List[torch.nn.Parameter]] = {}
    for p in params:
        if p.grad is None:
            continue
        grouped.setdefault(p.grad.device, []).append(p)
    for group in grouped.values():
        total_norm = torch.nn.utils.clip_grad_norm_(group, max_norm)
        if not bool(torch.isfinite(torch.as_tensor(total_norm)).all().item()):
            return False
    return _gradients_are_finite(params)


def _strict_jsonable(value: Any) -> Any:
    """Convert records to strict JSON values with non-finite floats as null."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _strict_jsonable(value.detach().cpu().item())
        return _strict_jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _strict_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    try:
        return _strict_jsonable(float(value))
    except Exception:
        return str(value)


TRAINING_MONITOR_COLUMNS: Tuple[str, ...] = (
    "timestamp_utc",
    "step",
    "phase",
    "health",
    "nan_detected",
    "nonfinite_fields",
    "image_path",
    "reward_mean",
    "reward_max",
    "reward_min",
    "best_reward",
    "spec_quality",
    "solver_reward_raw_mean",
    "solver_reward_soft_mean",
    "proposer_reward",
    "entropy_nats",
    "majority_fraction",
    "generator_objective",
    "generator_mode",
    "generator_did_step",
    "generator_skip",
    "generator_ce_loss",
    "generator_kl_loss",
    "generator_total_loss",
    "generator_valid_tokens",
    "generator_kl_coef",
    "dit_did_step",
    "dit_skip",
    "dit_loss",
    "dit_objective",
    "dit_reward",
    "proposer_did_step",
    "proposer_skip",
    "proposer_ce_loss",
    "proposer_kl_loss",
    "proposer_total_loss",
    "proposer_valid_tokens",
    "proposer_kl_coef",
    "solver_did_step",
    "solver_skip",
    "solver_ce_loss",
    "solver_kl_loss",
    "solver_total_loss",
    "solver_valid_tokens",
    "solver_kl_coef",
    "forced_tail_tokens",
    "generator_baseline",
    "proposer_baseline",
    "solver_baseline",
    "step_duration_sec",
)


def _append_training_monitor_record(
    jsonl_path: pathlib.Path,
    tsv_path: pathlib.Path,
    record: Dict[str, Any],
):
    """Append a concise strict-JSONL record and a spreadsheet-friendly TSV row."""
    payload = _strict_jsonable(dict(record))
    payload.setdefault("timestamp_utc", dt.datetime.utcnow().isoformat(timespec="seconds") + "Z")

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")

    columns = TRAINING_MONITOR_COLUMNS
    write_header = not tsv_path.exists() or tsv_path.stat().st_size == 0
    with tsv_path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(columns) + "\n")
        row = []
        for col in columns:
            val = payload.get(col)
            if isinstance(val, (dict, list)):
                cell = json.dumps(val, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            elif val is None:
                cell = ""
            else:
                cell = str(val)
            row.append(cell.replace("\t", " ").replace("\n", " "))
        f.write("\t".join(row) + "\n")


def _watch_fmt(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    try:
        val = float(value)
    except Exception:
        text = str(value).strip().replace("\t", " ").replace("\n", " ")
        return text[:96] if len(text) > 96 else text
    if not math.isfinite(val):
        return "-"
    if val == 0.0:
        return "0"
    if abs(val) >= 1000 or abs(val) < 0.001:
        return f"{val:.3e}"
    return f"{val:.4g}"


def _watch_first(payload: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None and value != "":
            return value
    return None


def _append_training_watch_record(watch_path: pathlib.Path, record: Dict[str, Any]):
    """Append a short, human-readable line for live training monitoring."""
    payload = _strict_jsonable(dict(record))
    payload.setdefault("timestamp_utc", dt.datetime.utcnow().isoformat(timespec="seconds") + "Z")
    phase = str(payload.get("phase") or "").strip().lower()

    parts = [
        str(payload.get("timestamp_utc")),
        f"step={_watch_fmt(payload.get('step'))}",
        f"phase={_watch_fmt(payload.get('phase'))}",
        f"health={_watch_fmt(payload.get('health'))}",
    ]

    if phase.startswith("u") or phase == "understanding":
        parts.extend(
            [
                f"Rraw={_watch_fmt(payload.get('solver_reward_raw_mean'))}",
                f"Rsoft={_watch_fmt(payload.get('solver_reward_soft_mean'))}",
                f"PropR={_watch_fmt(payload.get('proposer_reward'))}",
                f"Pcap={_watch_fmt(payload.get('proposer_easy_reward_cap_applied'))}",
                f"CapR={_watch_fmt(payload.get('proposer_easy_reward_cap_value'))}",
                f"H={_watch_fmt(payload.get('entropy_nats'))}",
                f"maj={_watch_fmt(payload.get('majority_fraction'))}",
                f"SCE={_watch_fmt(payload.get('solver_ce_loss'))}",
                f"SKL={_watch_fmt(payload.get('solver_kl_loss'))}",
                f"Sstep={_watch_fmt(payload.get('solver_did_step'))}",
                f"Sskip={_watch_fmt(payload.get('solver_skip'))}",
                f"PCE={_watch_fmt(payload.get('proposer_ce_loss'))}",
                f"PKL={_watch_fmt(payload.get('proposer_kl_loss'))}",
                f"Pstep={_watch_fmt(payload.get('proposer_did_step'))}",
                f"Pskip={_watch_fmt(payload.get('proposer_skip'))}",
                f"tok={_watch_fmt(_watch_first(payload, 'solver_valid_tokens', 'proposer_valid_tokens'))}",
            ]
        )
    else:
        parts.extend(
            [
                f"Rmean={_watch_fmt(payload.get('reward_mean'))}",
                f"Rmax={_watch_fmt(payload.get('reward_max'))}",
                f"best={_watch_fmt(payload.get('best_reward'))}",
                f"spec={_watch_fmt(payload.get('spec_quality'))}",
                f"obj={_watch_fmt(payload.get('generator_objective'))}",
                f"mode={_watch_fmt(payload.get('generator_mode'))}",
                f"GCE={_watch_fmt(payload.get('generator_ce_loss'))}",
                f"GKL={_watch_fmt(payload.get('generator_kl_loss'))}",
                f"Gstep={_watch_fmt(payload.get('generator_did_step'))}",
                f"Gskip={_watch_fmt(payload.get('generator_skip'))}",
                f"DitLoss={_watch_fmt(payload.get('dit_loss'))}",
                f"DitObj={_watch_fmt(payload.get('dit_objective'))}",
                f"DitR={_watch_fmt(payload.get('dit_reward'))}",
                f"DitStep={_watch_fmt(payload.get('dit_did_step'))}",
                f"DitSkip={_watch_fmt(payload.get('dit_skip'))}",
                f"Pstep={_watch_fmt(payload.get('proposer_did_step'))}",
                f"Pskip={_watch_fmt(payload.get('proposer_skip'))}",
                f"Sstep={_watch_fmt(payload.get('solver_did_step'))}",
                f"Sskip={_watch_fmt(payload.get('solver_skip'))}",
            ]
        )

    parts.extend(
        [
            f"nan={_watch_fmt(payload.get('nan_detected'))}",
            f"bad={_watch_fmt(payload.get('nonfinite_fields'))}",
            f"dt={_watch_fmt(payload.get('step_duration_sec'))}s",
        ]
    )

    watch_path.parent.mkdir(parents=True, exist_ok=True)
    with watch_path.open("a", encoding="utf-8") as f:
        f.write(" ".join(parts) + "\n")


def _default_code_run_registry_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "training_runs"


def _save_code_run_registry(
    *,
    run_dir: pathlib.Path,
    config: Dict[str, Any],
    git_info: Dict[str, Any],
    environment: Dict[str, Any],
    registry_dir: Optional[str] = None,
) -> pathlib.Path:
    """Mirror lightweight run metadata beside the training code.

    Heavy artifacts remain in ``output_dir``.  This registry keeps the exact
    launch configuration, environment, git state, and output directory together
    with the training package so experiments are easier to audit later.
    """
    configured = registry_dir
    if configured is None:
        configured = os.environ.get("SELF_EVOLVING_CODE_RUN_REGISTRY")
    if configured is not None and str(configured).strip().lower() in {"", "0", "false", "none", "disabled"}:
        raise RuntimeError("code run registry disabled")

    root = pathlib.Path(configured).expanduser().resolve() if configured else _default_code_run_registry_dir()
    entry = root / run_dir.name
    entry.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_name": run_dir.name,
        "output_dir": str(run_dir),
        "config_path": str(run_dir / "config.json"),
        "git_info_path": str(run_dir / "git_info.json"),
        "environment_path": str(run_dir / "environment.json"),
    }
    _json_dump(entry / "manifest.json", manifest)
    _json_dump(entry / "config.json", config)
    _json_dump(entry / "git_info.json", git_info)
    _json_dump(entry / "environment.json", environment)
    with (entry / "output_dir.txt").open("w", encoding="utf-8") as f:
        f.write(str(run_dir) + "\n")

    latest = root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            if latest.is_dir() and not latest.is_symlink():
                pass
            else:
                latest.unlink()
        if not latest.exists():
            latest.symlink_to(entry.name, target_is_directory=True)
    except Exception:
        pass
    return entry


def _collect_trainable_params(
    model: torch.nn.Module,
    adapter_name: Optional[str],
) -> Iterable[torch.nn.Parameter]:
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if adapter_name is None:
        return [p for _, p in trainable]

    def _belongs_to_role_adapter(name: str) -> bool:
        if ".dit." in name or name.startswith("dit."):
            return False
        return (f".{adapter_name}." in name) or (f"{adapter_name}." in name)

    selected = [
        p
        for n, p in trainable
        if _belongs_to_role_adapter(n)
    ]
    if not selected:
        preview = [name for name, _ in trainable[:20]]
        raise RuntimeError(
            f"No trainable parameters matched adapter '{adapter_name}'. "
            f"Trainable preview: {preview}"
        )
    return selected
