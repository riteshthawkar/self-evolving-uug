# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")

_clip_model = None
_clip_processor = None


def normalize_answer(text: str) -> str:
    val = (text or "").strip().lower()
    val = _PUNCT_RE.sub(" ", val)
    val = _ARTICLES_RE.sub(" ", val)
    val = _WS_RE.sub(" ", val).strip()
    return val


def _tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(str(text or "").lower())


def _jaccard_similarity(a: str, b: str) -> float:
    ta = set(_tokenize_words(a))
    tb = set(_tokenize_words(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return float(inter) / float(max(1, union))


def _parse_float_safe(text: str) -> Optional[float]:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(text or ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def answer_histogram(answers: Iterable[str]) -> Dict[str, int]:
    hist = Counter(normalize_answer(a) for a in answers if str(a or "").strip())
    return dict(hist)


def majority_vote(answers: Iterable[str]) -> Tuple[str, int, Dict[str, int]]:
    hist = answer_histogram(answers)
    if not hist:
        return "", 0, {}
    answer, count = max(hist.items(), key=lambda kv: kv[1])
    return answer, int(count), hist


def shannon_entropy_nats(probs: Iterable[float]) -> float:
    ent = 0.0
    for p in probs:
        p = float(p)
        if p > 0.0:
            ent -= p * math.log(p)
    return ent


def gaussian_reward(value: float, mu: float, sigma: float) -> float:
    sigma = max(1e-8, float(sigma))
    z = (float(value) - float(mu)) / sigma
    return float(math.exp(-0.5 * z * z))


@dataclass
class DualTrackReward:
    reward: float
    reward_raw: float
    entropy_nats: float
    majority_fraction: float
    majority_answer: str
    dual_track_agree: bool
    easy_case: bool
    unsolvable_case: bool


@dataclass
class SuderJointReward:
    reward: float
    entropy_component: float
    quality_component: float
    mean_entropy_nats: float
    entropy_weight_alpha: float


def compute_dual_track_reward(
    *,
    answers: List[str],
    intuitive_answer: str,
    entropy_mu: float,
    entropy_sigma: float,
    unsolvable_maj_threshold: float,
    zero_entropy_eps: float,
) -> DualTrackReward:
    maj_answer, maj_count, hist = majority_vote(answers)
    n = max(1, len([a for a in answers if str(a or "").strip()]))
    maj_frac = float(maj_count) / float(n)
    probs = [float(c) / float(n) for c in hist.values()] if hist else [1.0]
    entropy = shannon_entropy_nats(probs)

    easy_case = bool(entropy <= float(zero_entropy_eps) or maj_frac >= 1.0)
    unsolvable_case = bool((not easy_case) and (maj_frac <= float(unsolvable_maj_threshold)))

    raw = gaussian_reward(entropy, entropy_mu, entropy_sigma)
    reward = raw

    intuitive_norm = normalize_answer(intuitive_answer)
    tracks_agree = bool((not intuitive_norm) or (intuitive_norm == maj_answer))

    if unsolvable_case:
        reward = 0.0
    elif easy_case:
        reward = 0.5 * maj_frac if (not tracks_agree and intuitive_norm) else 0.0

    reward = max(-1.0, min(1.0, float(reward)))
    return DualTrackReward(
        reward=reward,
        reward_raw=float(raw),
        entropy_nats=float(entropy),
        majority_fraction=float(maj_frac),
        majority_answer=maj_answer,
        dual_track_agree=tracks_agree,
        easy_case=easy_case,
        unsolvable_case=unsolvable_case,
    )


def answer_match_score(predicted: str, expected: str) -> float:
    pred = normalize_answer(predicted)
    exp = normalize_answer(expected)
    if not pred or not exp:
        return 0.0
    if pred == exp:
        return 1.0
    p_tokens = pred.split()
    e_tokens = exp.split()
    if not p_tokens or not e_tokens:
        return 0.0
    common = Counter(p_tokens) & Counter(e_tokens)
    overlap = float(sum(common.values()))
    if overlap <= 0.0:
        return 0.0
    precision = overlap / float(len(p_tokens))
    recall = overlap / float(len(e_tokens))
    denom = precision + recall
    return 0.0 if denom <= 0.0 else float(2.0 * precision * recall / denom)


def soft_match_score(predicted: str, expected: str) -> float:
    pred = normalize_answer(predicted)
    exp = normalize_answer(expected)
    if not exp:
        return 0.5
    if pred == exp:
        return 1.0
    if pred and exp and (pred in exp or exp in pred):
        return 0.8

    pred_num = _parse_float_safe(pred)
    exp_num = _parse_float_safe(exp)
    if pred_num is not None and exp_num is not None:
        denom = max(abs(exp_num), 1.0)
        rel = abs(pred_num - exp_num) / denom
        if rel <= 0.01:
            return 1.0
        if rel <= 0.05:
            return 0.7
        if rel <= 0.20:
            return 0.4
        return 0.0

    return _jaccard_similarity(pred, exp)


def yes_no_polarity(text: str) -> int:
    value = normalize_answer(text)
    if value.startswith("yes") or value in {"true", "1", "present"}:
        return 1
    if value.startswith("no") or value in {"false", "0", "absent"}:
        return -1
    return 0


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is not None and _clip_processor is not None:
        return _clip_model, _clip_processor
    try:
        from transformers import CLIPModel, CLIPProcessor

        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if torch.cuda.is_available():
            _clip_model = _clip_model.cuda().eval()
        else:
            _clip_model = _clip_model.eval()
    except Exception:
        _clip_model = None
        _clip_processor = None
    return _clip_model, _clip_processor


def clip_similarity(image: Image.Image, text: str) -> float:
    model, processor = _load_clip()
    if model is None or processor is None:
        return 0.0

    try:
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            similarity = (image_embeds * text_embeds).sum(dim=-1)
        return float(similarity.item())
    except Exception:
        return 0.0


def clip_text_similarity(text_a: str, text_b: str) -> float:
    model, processor = _load_clip()
    if model is None or processor is None:
        return _jaccard_similarity(text_a, text_b)

    try:
        inputs = processor(text=[text_a, text_b], return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            text_embeds = model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            similarity = (text_embeds[0] * text_embeds[1]).sum()
        return float(similarity.item())
    except Exception:
        return _jaccard_similarity(text_a, text_b)


def image_diversity_score(images: List[Image.Image]) -> float:
    if len(images) <= 1:
        return 0.5
    if not HAS_NUMPY:
        return 0.5

    struct_vectors = []
    for image in images:
        arr = np.asarray(image.convert("L").resize((96, 96)), dtype=np.float32) / 255.0
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

    bins = 32
    hist_vectors = []
    for image in images:
        arr = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
        hists = []
        for channel in range(3):
            hist, _ = np.histogram(arr[:, :, channel].ravel(), bins=bins, range=(0.0, 1.0))
            hist = hist.astype(np.float32)
            hist = hist / max(1.0, hist.sum())
            hists.append(hist)
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


def per_candidate_diversity_scores(images: List[Image.Image]) -> List[float]:
    n = len(images)
    if n <= 1:
        return [0.0] * n
    if n == 2:
        diversity = max(0.0, min(1.0, image_diversity_score(images)))
        return [diversity, diversity]

    full_diversity = image_diversity_score(images)
    raw: List[float] = []
    for idx in range(n):
        subset = images[:idx] + images[idx + 1 :]
        subset_diversity = image_diversity_score(subset)
        raw.append(full_diversity - subset_diversity)

    clamped = [max(0.0, value) for value in raw]
    max_value = max(clamped)
    if max_value < 1e-8:
        return [0.0] * n
    return [min(1.0, value / max_value) for value in clamped]


def compute_generation_spec_quality(
    *,
    qa_pairs: Iterable[Any],
    min_spec_qa_pairs: int,
    max_question_words: int,
    max_expected_words: int,
) -> Tuple[float, Dict[str, float]]:
    raw_pairs = list(qa_pairs)
    filtered: List[Tuple[str, str]] = []
    seen_questions = set()
    valid_count = 0

    for qa in raw_pairs:
        question = " ".join(str(getattr(qa, "question", "") or "").split())
        expected = getattr(qa, "expected", None)
        if expected is None:
            expected = getattr(qa, "answer", "")
        expected = " ".join(str(expected or "").split())
        if question and not question.endswith("?"):
            question = f"{question}?"

        q_words = len(_tokenize_words(question))
        e_words = len(_tokenize_words(expected))
        is_valid = bool(
            question
            and expected
            and q_words <= int(max_question_words)
            and 1 <= e_words <= int(max_expected_words)
        )
        if not is_valid:
            continue

        q_key = normalize_answer(question)
        if q_key in seen_questions:
            continue
        seen_questions.add(q_key)
        valid_count += 1
        filtered.append((question, expected))

    filtered = filtered[:3]
    qa_count = len(filtered)
    raw_count = len(raw_pairs)

    count_score = min(1.0, qa_count / float(max(1, int(min_spec_qa_pairs))))
    validity_score = qa_count / float(max(1, raw_count))
    uniqueness_score = len({normalize_answer(question) for question, _ in filtered}) / float(max(1, qa_count))
    all_yes_no = qa_count > 0 and all(yes_no_polarity(expected) != 0 for _, expected in filtered)
    yes_no_penalty = 0.2 if all_yes_no and qa_count >= int(min_spec_qa_pairs) else 0.0

    quality = 0.5 * count_score + 0.3 * validity_score + 0.2 * uniqueness_score - yes_no_penalty
    quality = float(max(0.0, min(1.0, quality)))
    details = {
        "raw_qa_count": float(raw_count),
        "filtered_qa_count": float(qa_count),
        "count_score": float(count_score),
        "validity_score": float(validity_score),
        "uniqueness_score": float(uniqueness_score),
        "yes_no_penalty": float(yes_no_penalty),
        "spec_quality": float(quality),
    }
    return quality, details


def compute_suder_joint_reward(
    *,
    mean_entropy_nats: float,
    quality_component: float,
    entropy_mu: float,
    entropy_sigma: float,
    entropy_weight_alpha: float,
    zero_entropy_eps: float,
    zero_entropy_reward_cap: float,
) -> SuderJointReward:
    entropy_component = gaussian_reward(mean_entropy_nats, entropy_mu, entropy_sigma)
    if float(mean_entropy_nats) <= float(zero_entropy_eps):
        entropy_component = -max(0.0, float(zero_entropy_reward_cap))
    alpha = max(0.0, min(1.0, float(entropy_weight_alpha)))
    quality = max(0.0, min(1.0, float(quality_component)))
    reward = alpha * entropy_component + (1.0 - alpha) * quality
    reward = max(-1.0, min(1.0, float(reward)))
    return SuderJointReward(
        reward=float(reward),
        entropy_component=float(entropy_component),
        quality_component=float(quality),
        mean_entropy_nats=float(mean_entropy_nats),
        entropy_weight_alpha=float(alpha),
    )
