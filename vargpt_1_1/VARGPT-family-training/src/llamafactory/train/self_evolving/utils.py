"""
Shared utilities for the VARGPT self-evolving training pipeline.

Ported from BLIP3o's utils.py with adaptations for Qwen2-VL:
  - Chat template uses Qwen2-VL format (system/user/assistant)
  - Image preprocessing uses Qwen2VLProcessor
  - Removed BLIP3o-specific conv_templates and tokenizer_image_token
"""

import contextlib
import json
import math
import os
import pathlib
import random
import re
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

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
# Default LoRA targets for Qwen2-VL
# ---------------------------------------------------------------------------
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
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


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP/FSDP wrapper to get the base model."""
    return model.module if hasattr(model, "module") else model


def _infer_primary_device(
    model: torch.nn.Module, fallback_cuda_device: int = 0
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
    """Normalize and optionally truncate an answer for self-consistency voting."""
    s = ans.strip().lower()
    s = s.replace(",", " ")
    s = " ".join(s.split())
    s = s.strip(" .,:;!?\"'")
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
        text_tagged = strip_tags(tagged, "text")
        if text_tagged:
            return text_tagged.replace("\n", " ").strip()
        tagged = re.sub(r"<[^>]+>", " ", tagged).strip()
        if "?" in tagged:
            return tagged[: tagged.find("?") + 1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines:
        line = re.sub(r"^\s*(?:q(?:uestion)?\s*)?\d+\s*[\).:\-\s]*", "", line, flags=re.IGNORECASE).strip()
        line = re.sub(r"<[^>]+>", " ", line).strip()
        if re.match(r"^<[^>]+>$", line):
            continue
        if "?" in line:
            return line[: line.find("?") + 1].strip()
    return ""


def _parse_all_questions(text: str) -> List[str]:
    """Parse all candidate questions from a multi-question proposer response.

    Handles the XML format produced by ``build_proposer_multi_prompt``.
    """
    questions: List[str] = []

    # Recover common partial-XML outputs such as
    # ``<text>What color is the sign?</text>`` even when the surrounding
    # ``<question>`` block is truncated or malformed.
    for match in re.finditer(r'<text[^>]*>(.*?)</text>', text, re.DOTALL | re.IGNORECASE):
        q = match.group(1).strip().replace("\n", " ")
        if q:
            questions.append(q)

    question_blocks = re.findall(
        r'<question[^>]*>(.*?)</question>',
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if question_blocks:
        for block in question_blocks:
            text_match = re.search(
                r'<text>(.*?)</text>',
                block,
                re.DOTALL | re.IGNORECASE,
            )
            if text_match:
                q = text_match.group(1).strip().replace("\n", " ")
                questions.append(q)
                continue

            stripped = re.sub(r'<[^>]+>', ' ', block).strip()
            lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
            found = ""
            for ln in lines:
                ln_clean = re.sub(r"^\d+[\).\-\s]*", "", ln).strip()
                if ln_clean.endswith("?"):
                    found = ln_clean
                    break
            if found:
                questions.append(found)
        questions = [q for q in questions if q]

    if questions:
        return questions

    plain = strip_tags(text, "question")
    if plain:
        return [plain.replace("\n", " ").strip()]

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates: List[str] = []
    for ln in lines:
        ln_clean = re.sub(r"^\s*(?:q(?:uestion)?\s*)?\d+\s*[\).:\-\s]*", "", ln, flags=re.IGNORECASE).strip()
        ln_clean = re.sub(r'<[^>]+>', ' ', ln_clean).strip()
        if re.match(r'^<[^>]+>$', ln_clean):
            continue
        if "?" in ln_clean:
            candidates.append(ln_clean[: ln_clean.find("?") + 1].strip())
    if candidates:
        return candidates

    stripped = re.sub(r'<[^>]+>', ' ', text)
    for match in re.finditer(r'([^?\n]{6,180}\?)', stripped):
        q = " ".join(match.group(1).split())
        if q:
            candidates.append(q)
    if candidates:
        return candidates

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
    """Switch the active LoRA adapter on the model.

    When ``adapter_name`` is None, disables all adapters (uses base model).
    Restores the previous adapter on exit.
    """
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
# Token decode / chat helpers — adapted for Qwen2-VL / VARGPT
# ---------------------------------------------------------------------------


def _decode_tokens(tokenizer, token_ids: torch.Tensor) -> str:
    """Decode token ids to text using the tokenizer."""
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    raise RuntimeError("Tokenizer does not expose decode method")


def _normalize_vl_image_size(image: Image.Image) -> Image.Image:
    """Resize very large/small inputs to stable bounds for VL preprocessing."""
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

    new_w = max(size_multiple, (new_w // size_multiple) * size_multiple)
    new_h = max(size_multiple, (new_h // size_multiple) * size_multiple)

    if new_w == w and new_h == h:
        return image
    return image.resize((new_w, new_h), Image.BICUBIC)


def _build_chat_text(
    processor, image: Image.Image, prompt: str
) -> str:
    """Build a chat-formatted text string for multimodal generation.

    For VARGPT / Qwen2-VL: uses ``apply_chat_template`` with the
    multimodal content-list format.
    """
    if hasattr(processor, "apply_chat_template"):
        # Qwen2-VL multimodal content list
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
        # Fallback: simple string content
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

    Used by the imageless proposer mode where the proposer receives only a
    topic description and generates a spec without any visual input.
    """
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

    Used by the imageless proposer mode.
    """
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


def _prepare_mm_inputs(
    processor,
    device: torch.device,
    image: Image.Image,
    chat_text: str,
    model=None,
):
    """Prepare multimodal inputs for model.generate() / model.forward().

    For VARGPT / Qwen2-VL: uses the processor with text + images.
    """
    image = _normalize_vl_image_size(image)

    # Try standard multimodal processor path
    try:
        inputs = processor(
            text=[chat_text], images=[image], return_tensors="pt", padding=True
        )
        return inputs.to(device)
    except Exception:
        pass

    # Fallback: try using qwen_vl_utils for vision info processing
    try:
        from qwen_vl_utils import process_vision_info
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": chat_text},
                ],
            }
        ]
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )
        return inputs.to(device)
    except Exception:
        pass

    # Last resort: tokenize text only (no image features)
    inputs = processor(
        text=[chat_text], return_tensors="pt", padding=True
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
) -> List[torch.nn.Parameter]:
    """Collect trainable parameters, optionally filtered by adapter name."""
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
