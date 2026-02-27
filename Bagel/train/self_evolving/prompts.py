# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

_QUESTION_TAG_RE = re.compile(
    r"<question(?:\s+[^>]*)?>\s*(.*?)\s*</question>",
    flags=re.IGNORECASE | re.DOTALL,
)
_QUESTION_BLOCK_RE = re.compile(
    r"<question(?:\s+[^>]*)?>(.*?)</question>",
    flags=re.IGNORECASE | re.DOTALL,
)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
_PROMPT_TAG_RE = re.compile(r"<prompt>\s*(.*?)\s*</prompt>", flags=re.IGNORECASE | re.DOTALL)
_PROMPT_RELAXED_RE = re.compile(
    r"<prompt[^>]*>\s*(.*?)\s*(?:</prompt>|<qa_pairs>|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
_QA_PAIR_RE = re.compile(
    r"<qa>\s*<question>\s*(.*?)\s*</question>\s*<answer>\s*(.*?)\s*</answer>\s*</qa>",
    flags=re.IGNORECASE | re.DOTALL,
)
_QA_TEXT_RE = re.compile(r"<qa>\s*(.*?)\s*</qa>", flags=re.IGNORECASE | re.DOTALL)
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


def build_proposer_prompt(target_difficulty: str = "medium") -> str:
    level = (target_difficulty or "medium").strip().lower()
    if level not in {"easy", "medium", "hard"}:
        level = "medium"
    if level == "hard":
        diff_hint = (
            "Target HARD: require multi-step visual reasoning with at least two grounded constraints "
            "(for example: relation+attribute, occlusion+count, text+location)."
        )
    elif level == "easy":
        diff_hint = "Target EASY-MEDIUM: objective and image-grounded, avoid trivial lookups."
    else:
        diff_hint = "Target MEDIUM: use at least two grounded constraints with one exact answer."
    return (
        "You are a Question Proposer.\n"
        "Given the image, generate exactly one objective, image-grounded question.\n"
        f"{diff_hint}\n"
        "Rules:\n"
        "- Must be answerable from visible evidence only.\n"
        "- Avoid subjective/speculative wording (why/might/could/likely/feel).\n"
        "- Avoid forced-choice binary forms and vague outputs.\n"
        "- Question must end with '?'.\n"
        "Output only:\n"
        "<question>...</question>\n"
        "<rationale>...</rationale>"
    )


def build_proposer_multi_prompt(num_questions: int = 3, target_difficulty: str = "medium") -> str:
    n = max(1, int(num_questions))
    level = (target_difficulty or "medium").strip().lower()
    if level not in {"easy", "medium", "hard"}:
        level = "medium"
    if level == "hard":
        diff_hint = (
            "TARGET HARD: prioritize subtle, grounded questions requiring multi-hop visual reasoning."
        )
    elif level == "easy":
        diff_hint = "TARGET EASY-MEDIUM: objective, concrete, avoid triviality."
    else:
        diff_hint = "TARGET MEDIUM: grounded multi-constraint questions."

    return (
        "You are a reasoning-first visual question proposer.\n"
        f"{diff_hint}\n"
        f"Generate exactly {n} candidate questions.\n"
        "For each candidate, follow this order: task_card -> reasoning_domains -> reasoning_chain -> "
        "strategy_used -> visual_target -> two_answer_test -> text -> rationale.\n"
        "Rules:\n"
        "- Objective and image-grounded only.\n"
        "- Avoid speculative/subjective wording.\n"
        "- Avoid explicit binary forced-choice questions.\n"
        "- Keep answers short and exactly verifiable.\n"
        "Output format:\n"
        "<questions>\n"
        "  <question id=\"1\">\n"
        "    <task_card>C1/C2/C3/C4/C5/C6/C7/C8/C9</task_card>\n"
        "    <reasoning_domains>D1,D2</reasoning_domains>\n"
        "    <reasoning_chain>step1 -> step2 -> step3</reasoning_chain>\n"
        "    <strategy_used>H1/H3/H4/H6/H7/H8/M2/M4</strategy_used>\n"
        "    <visual_target>concrete visual anchor</visual_target>\n"
        "    <two_answer_test>A option_a vs B option_b</two_answer_test>\n"
        "    <text>...</text>\n"
        "    <rationale>...</rationale>\n"
        "  </question>\n"
        "</questions>"
    )


def build_solver_prompt(question_text: str, focus_hint: str = "") -> str:
    q = (question_text or "").strip()
    hint = (focus_hint or "").strip()
    focus_line = (
        f"- Focus mode for this sample: {hint}. Prefer evidence consistent with this focus.\n"
        if hint
        else ""
    )
    return (
        "You are a precise vision-language solver.\n"
        "Answer using only visible evidence from the image.\n"
        "Rules:\n"
        "- Output exactly 1-5 words.\n"
        "- No explanation, no uncertainty phrases.\n"
        "- For count questions, return a concrete integer.\n"
        "- Never answer with vague terms like 'too many', 'several', 'unclear'.\n"
        f"{focus_line}"
        "Return final answer in XML only:\n"
        "<answer>...</answer>\n"
        f"Question: {q}"
    )


_PPS_TEMPLATES = [
    "You are a precise vision-language solver.\nAnswer using only visible evidence from the image.\n",
    "Look at the image carefully and provide a precise answer from visible evidence only.\n",
    "You are a visual analyst. Answer factually using only what is visible.\n",
    "Study the image and answer the question directly from observable details.\n",
    "As an image examiner, provide a concise, concrete answer based on visible evidence.\n",
    "Based on the image, give a brief factual answer that is visually verifiable.\n",
    "Examine the visual evidence and answer with the most concrete supported answer.\n",
]


def build_solver_prompt_pps(question_text: str, template_index: int, focus_hint: str = "") -> str:
    idx = int(template_index) % len(_PPS_TEMPLATES)
    preamble = _PPS_TEMPLATES[idx]
    return preamble + build_solver_prompt(question_text, focus_hint=focus_hint).split("Rules:\n", 1)[1]


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


def strip_tags(text: str, tag: str) -> str:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", flags=re.IGNORECASE | re.DOTALL)
    m = pattern.search(text or "")
    return m.group(1).strip() if m else ""


def parse_proposer_question_candidates(text: str) -> List[Dict[str, str]]:
    raw = str(text or "")
    blocks = _QUESTION_BLOCK_RE.findall(raw)
    candidates: List[Dict[str, str]] = []
    for block in blocks:
        q_text = (strip_tags(block, "text") or parse_first_question(block) or "").strip()
        q_text = " ".join(q_text.replace("\n", " ").split())
        if not q_text:
            continue
        candidates.append(
            {
                "text": q_text,
                "task_card": strip_tags(block, "task_card"),
                "reasoning_domains": strip_tags(block, "reasoning_domains"),
                "reasoning_chain": strip_tags(block, "reasoning_chain"),
                "strategy_used": strip_tags(block, "strategy_used"),
                "visual_target": strip_tags(block, "visual_target"),
                "two_answer_test": strip_tags(block, "two_answer_test"),
                "rationale": strip_tags(block, "rationale"),
            }
        )
    if candidates:
        return candidates
    return [{"text": q} for q in parse_all_questions(raw)]


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
        if not val or "?" not in val:
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
    if prompt_match:
        prompt = " ".join(prompt_match.group(1).strip().split())
    else:
        prompt = ""
        relaxed = _PROMPT_RELAXED_RE.search(raw)
        if relaxed:
            prompt = " ".join(relaxed.group(1).strip().split())
        if not prompt:
            first_prompt_line = re.search(r"^\s*<prompt[^\n]*", raw, flags=re.IGNORECASE | re.MULTILINE)
            if first_prompt_line:
                line = first_prompt_line.group(0).strip()
                tail = line[len("<prompt") :].lstrip(" >:\t")
                prompt = " ".join(tail.split())

    qa_pairs: List[GenerationQAPair] = []
    seen_questions = set()

    def _append_pair(q_raw: str, a_raw: str) -> None:
        q = " ".join(str(q_raw).strip().split())
        a = " ".join(str(a_raw).strip().split())
        if not q or not a:
            return
        q_key = q.lower()
        if q_key in seen_questions:
            return
        seen_questions.add(q_key)
        qa_pairs.append(GenerationQAPair(question=q, answer=a))

    for q_raw, a_raw in _QA_PAIR_RE.findall(raw):
        _append_pair(q_raw, a_raw)

    if len(qa_pairs) < max(1, int(min_qa_pairs)):
        q_tags = _QUESTION_TAG_RE.findall(raw)
        a_tags = _ANSWER_TAG_RE.findall(raw)
        for q_raw, a_raw in zip(q_tags, a_tags):
            _append_pair(q_raw, a_raw)

    if len(qa_pairs) < max(1, int(min_qa_pairs)):
        qa_questions = _QA_TEXT_RE.findall(raw)
        a_tags = _ANSWER_TAG_RE.findall(raw)
        for q_raw, a_raw in zip(qa_questions, a_tags):
            q_candidate = maybe_strip_tagged(str(q_raw), "question")
            _append_pair(q_candidate if q_candidate else q_raw, a_raw)

    if not prompt:
        return None
    if len(qa_pairs) < max(1, int(min_qa_pairs)):
        return None
    return GenerationSpec(prompt=prompt, qa_pairs=qa_pairs)


def maybe_strip_tagged(text: str, tag: str) -> Optional[str]:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", flags=re.IGNORECASE | re.DOTALL)
    m = pattern.search(text or "")
    return m.group(1).strip() if m else None
