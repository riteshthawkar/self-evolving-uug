# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import Optional

_QUESTION_TAG_RE = re.compile(r"<question>\s*(.*?)\s*</question>", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
_NON_OBJECTIVE_RE = re.compile(
    r"\b(why|might|could|likely|opinion|feel|emotion|think|believe|suggest|imply|purpose|reason)\b",
    flags=re.IGNORECASE,
)


def build_proposer_prompt() -> str:
    return (
        "You are an objective visual-question creator.\n"
        "Given the image, produce exactly one question with a concise, factual answer.\n"
        "Rules:\n"
        "1) The question must be answerable from visible content.\n"
        "2) Avoid subjective or open-ended wording.\n"
        "3) Keep it specific and verifiable.\n"
        "Output format:\n"
        "<question>...</question>"
    )


def build_solver_prompt(question: str) -> str:
    q = (question or "").strip()
    return (
        "Answer the following visual question.\n"
        "Return final answer in this exact format:\n"
        "<answer>...</answer>\n"
        f"Question: {q}"
    )


def parse_first_question(text: str) -> str:
    raw = text or ""
    match = _QUESTION_TAG_RE.search(raw)
    if match:
        return " ".join(match.group(1).strip().split())
    for line in raw.splitlines():
        line = line.strip()
        if "?" in line and len(line) > 3:
            return " ".join(line.split())
    return ""


def parse_answer(text: str) -> str:
    raw = text or ""
    match = _ANSWER_TAG_RE.search(raw)
    if match:
        return " ".join(match.group(1).strip().split())
    # Fallback: use the last non-empty line for robustness.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def is_objective_question(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    return _NON_OBJECTIVE_RE.search(q) is None


def maybe_strip_tagged(text: str, tag: str) -> Optional[str]:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", flags=re.IGNORECASE | re.DOTALL)
    m = pattern.search(text or "")
    return m.group(1).strip() if m else None

