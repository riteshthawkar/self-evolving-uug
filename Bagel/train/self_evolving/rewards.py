# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def normalize_answer(text: str) -> str:
    val = (text or "").strip().lower()
    val = _PUNCT_RE.sub(" ", val)
    val = _ARTICLES_RE.sub(" ", val)
    val = _WS_RE.sub(" ", val).strip()
    return val


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
