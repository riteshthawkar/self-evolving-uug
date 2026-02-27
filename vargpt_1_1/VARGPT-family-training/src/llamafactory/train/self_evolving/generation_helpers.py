"""Generation-specific helper functions: dataclasses, prompts, text matching, image conversion, spec parsing."""

import inspect
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

from .utils import normalize_answer, strip_tags

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

GEN_PROMPT_TEMPLATE = (
    "You are a generation-spec proposer for self-evolving training.\n"
    "Given the source image, propose one new text-to-image prompt and verification QA pairs.\n"
    "Rules:\n"
    "- Prompt must be image-grounded but not a trivial copy.\n"
    "- Prompt must be declarative (caption/instruction style), not a question.\n"
    "- Do not use a question mark in the prompt.\n"
    "- The prompt must naturally embed all the verifiable details from your QA pairs.\n"
    "  For example, if QA asks 'How many players?' with answer 'Two', the prompt must say\n"
    "  'two players' — not just 'players'. The image generator sees only the prompt,\n"
    "  so every fact the QA checks must be explicitly described in it.\n"
    "- QA pairs must be objective, short-answer, and visually verifiable.\n"
    "- Prefer challenging QA that require reasoning, counting, reading text, or fine-grained detail usage.\n"
    "- Avoid trivial existence questions (e.g. 'is there a person') if the object is obvious.\n"
    "- Avoid subjective wording (why/might/could/likely/opinion).\n"
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

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Generation-specific helper functions
# ---------------------------------------------------------------------------


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
    """Compute a diversity score across candidate images."""
    if len(images) <= 1:
        return 0.5
    if not HAS_NUMPY:
        return 0.5

    # --- structural component (grayscale, 96x96) ---
    struct_vectors = []
    for image in images:
        arr = np.asarray(
            image.convert("L").resize((96, 96)), dtype=np.float32
        ) / 255.0
        struct_vectors.append(arr.reshape(-1))

    struct_dists: List[float] = []
    for i in range(len(struct_vectors)):
        for j in range(i + 1, len(struct_vectors)):
            dot = float(np.dot(struct_vectors[i], struct_vectors[j]))
            norm_i = float(np.linalg.norm(struct_vectors[i]))
            norm_j = float(np.linalg.norm(struct_vectors[j]))
            cos_sim = dot / max(1e-8, norm_i * norm_j)
            struct_dists.append(1.0 - cos_sim)

    struct_score = 0.5
    if struct_dists:
        struct_score = min(1.0, max(0.0, (sum(struct_dists) / len(struct_dists)) / 0.30))

    # --- colour-histogram component (RGB, 32 bins per channel) ---
    BINS = 32
    hist_vectors = []
    for image in images:
        arr = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
        hists = []
        for ch in range(3):
            h, _ = np.histogram(arr[:, :, ch].ravel(), bins=BINS, range=(0.0, 1.0))
            h = h.astype(np.float32)
            h = h / max(1.0, h.sum())
            hists.append(h)
        hist_vectors.append(np.concatenate(hists))

    hist_dists: List[float] = []
    for i in range(len(hist_vectors)):
        for j in range(i + 1, len(hist_vectors)):
            bc = float(np.sum(np.sqrt(hist_vectors[i] * hist_vectors[j])))
            hist_dists.append(1.0 - bc)

    hist_score = 0.5
    if hist_dists:
        hist_score = min(1.0, max(0.0, (sum(hist_dists) / len(hist_dists)) / 0.25))

    return 0.6 * struct_score + 0.4 * hist_score


def _per_candidate_diversity_scores(images: List[Image.Image]) -> List[float]:
    """Compute per-candidate diversity using leave-one-out contribution.

    For each candidate *i*, we measure how much diversity the full set loses
    when candidate *i* is removed:  ``diversity(all) - diversity(all \\ {i})``.
    The result is normalised to [0, 1].  Higher means the candidate adds more
    diversity to the set.  With 1 candidate returns [0.0] (no diversity
    signal).  With 2 candidates both get the same global diversity score
    (leave-one-out is symmetric for pairs).
    """
    n = len(images)
    if n <= 1:
        return [0.0] * n
    if n == 2:
        # Two candidates: leave-one-out is symmetric (removing either leaves
        # a single image with baseline diversity 0.5).  Use global pairwise
        # diversity directly, clamped to [0, 1].
        d = max(0.0, min(1.0, _image_diversity_score(images)))
        return [d, d]

    full_diversity = _image_diversity_score(images)

    raw: List[float] = []
    for i in range(n):
        subset = images[:i] + images[i + 1:]
        subset_diversity = _image_diversity_score(subset)
        # Contribution: how much diversity drops without this candidate.
        # Positive = candidate adds diversity; negative = candidate hurts it.
        raw.append(full_diversity - subset_diversity)

    # Clamp negative contributions to 0: a candidate that *reduces*
    # overall diversity should never receive a positive diversity reward.
    clamped = [max(0.0, c) for c in raw]

    max_c = max(clamped)
    if max_c < 1e-8:
        # All contributions are zero or negative — no candidate helps diversity
        return [0.0] * n
    # Normalise to [0, 1] using the max positive contribution
    return [min(1.0, c / max_c) for c in clamped]


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
    """Deterministic fallback visualization for latent tensors when a decoder pipeline is unavailable."""
    if not (HAS_NUMPY and torch.is_tensor(image_obj)):
        return None

    t = image_obj.detach().to(torch.float32).cpu()
    if t.ndim == 4:
        t = t[0]
    if t.ndim == 3 and t.shape[0] == 1:
        t = t[0]

    arr_t: Optional[torch.Tensor] = None

    if t.ndim == 2:
        n, c = int(t.shape[0]), int(t.shape[1])
        s = int(round(math.sqrt(float(n))))
        if s * s == n:
            arr_t = t.view(s, s, c)
        else:
            arr_t = t.view(n, 1, c)
    elif t.ndim == 3:
        if t.shape[0] in (1, 3, 4):
            arr_t = t.permute(1, 2, 0).contiguous()
        elif t.shape[-1] in (1, 3, 4):
            arr_t = t.contiguous()
        else:
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
