# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_QUESTION_TAG_RE = re.compile(
    r"<question(?:\s+[^>]*)?>\s*(.*?)\s*</question>",
    flags=re.IGNORECASE | re.DOTALL,
)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
_PROMPT_TAG_RE = re.compile(r"<prompt>\s*(.*?)\s*</prompt>", flags=re.IGNORECASE | re.DOTALL)
_QA_PAIR_RE = re.compile(
    r"<qa>\s*<question>\s*(.*?)\s*</question>\s*<answer>\s*(.*?)\s*</answer>\s*</qa>",
    flags=re.IGNORECASE | re.DOTALL,
)
_NON_OBJECTIVE_RE = re.compile(
    r"\b(why|might|could|likely|opinion|feel|emotion|think|believe|suggest|imply|purpose|reason)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class GenerationQAPair:
    question: str
    answer: str


@dataclass(frozen=True)
class GenerationSpec:
    prompt: str
    qa_pairs: List[GenerationQAPair]


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


def build_proposer_multi_prompt(num_questions: int = 3) -> str:
    n = max(1, int(num_questions))
    return (
        "You are an objective visual-question creator.\n"
        "Given the image, produce multiple diverse, factual questions.\n"
        "Rules:\n"
        "1) Every question must be answerable from visible content only.\n"
        "2) Avoid subjective or open-ended wording.\n"
        "3) Keep questions specific and verifiable.\n"
        f"4) Output exactly {n} questions.\n"
        "Output format:\n"
        "<questions>\n"
        "  <question>...</question>\n"
        "</questions>"
    )


def build_solver_prompt(question: str) -> str:
    q = (question or "").strip()
    return (
        "Answer the following visual question.\n"
        "Return final answer in this exact format:\n"
        "<answer>...</answer>\n"
        f"Question: {q}"
    )


def build_generation_spec_prompt(min_qa_pairs: int = 2) -> str:
    n = max(1, int(min_qa_pairs))
    return (
        "You are an image-generation spec writer.\n"
        "Given the source image, write one concise generation prompt and objective QA checks.\n"
        "Rules:\n"
        "1) Prompt must describe visible content only.\n"
        "2) QA checks must be factual and verifiable from the generated image.\n"
        f"3) Provide at least {n} QA pairs.\n"
        "Output format:\n"
        "<prompt>...</prompt>\n"
        "<qa_pairs>\n"
        "  <qa><question>...</question><answer>...</answer></qa>\n"
        "</qa_pairs>"
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


def parse_all_questions(text: str) -> List[str]:
    raw = text or ""
    matches = _QUESTION_TAG_RE.findall(raw)
    if matches:
        out: List[str] = []
        seen = set()
        for m in matches:
            q = " ".join(str(m).strip().split())
            if not q:
                continue
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
        if out:
            return out

    out: List[str] = []
    seen = set()
    for line in raw.splitlines():
        val = line.strip()
        if not val:
            continue
        val = re.sub(r"^\d+[\).\-\s]*", "", val).strip()
        if not val:
            continue
        if "?" not in val:
            continue
        q = " ".join(val.split())
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    if out:
        return out

    first = parse_first_question(raw)
    return [first] if first else []


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


def parse_generation_spec(text: str, min_qa_pairs: int = 2) -> Optional[GenerationSpec]:
    raw = text or ""
    prompt_match = _PROMPT_TAG_RE.search(raw)
    prompt = " ".join(prompt_match.group(1).strip().split()) if prompt_match else ""
    qa_pairs: List[GenerationQAPair] = []
    for q_raw, a_raw in _QA_PAIR_RE.findall(raw):
        q = " ".join(str(q_raw).strip().split())
        a = " ".join(str(a_raw).strip().split())
        if q and a:
            qa_pairs.append(GenerationQAPair(question=q, answer=a))
    if not prompt:
        return None
    if len(qa_pairs) < max(1, int(min_qa_pairs)):
        return None
    return GenerationSpec(prompt=prompt, qa_pairs=qa_pairs)


def maybe_strip_tagged(text: str, tag: str) -> Optional[str]:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", flags=re.IGNORECASE | re.DOTALL)
    m = pattern.search(text or "")
    return m.group(1).strip() if m else None
