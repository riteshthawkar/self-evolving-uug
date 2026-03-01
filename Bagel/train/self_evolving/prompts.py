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
_TEXT_TAG_RE = re.compile(
    r"<text(?:\s+[^>]*)?>\s*(.*?)\s*(?:</text>|$)",
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
_QA_FALLBACK_RE = re.compile(
    r"(?:^|\n)\s*(?:q(?:uestion)?\s*[:\-]\s*)(.+?)\s*(?:\n|\r\n)\s*(?:a(?:nswer)?\s*[:\-]\s*)(.+?)(?=\n\s*(?:q(?:uestion)?\s*[:\-]|$)|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
_NON_OBJECTIVE_RE = re.compile(
    r"\b(why|might|could|likely|opinion|feel|emotion|think|believe|suggest|imply|purpose|reason)\b",
    flags=re.IGNORECASE,
)
_INVALID_QUESTION_ARTIFACT_RE = re.compile(
    r"(<|>|</|/>|\{|\}|\[|\]|\\x|\\u[0-9a-fA-F]{4}|"
    r"\b(task_card|reasoning_domains|reasoning_chain|strategy_used|visual_target|two_answer_test|rationale)\b)",
    flags=re.IGNORECASE,
)
_RUNAWAY_PUNCT_RE = re.compile(r"[!?.,;:|/_\-]{4,}")
_TAG_RE = re.compile(r"<[^>]+>")
_QUESTION_SLICE_RE = re.compile(r"[^?]{5,240}\?")
_INTERROGATIVE_START_RE = re.compile(
    r"^(what|which|who|whom|whose|where|when|why|how|is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\b",
    flags=re.IGNORECASE,
)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_OPEN_RE = re.compile(r"<answer(?:\s+[^>]*)?>\s*(.*)$", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:final\s+answer|answer)\s*[:\-]\s*", flags=re.IGNORECASE)
_CONCLUSION_RE = re.compile(
    r"\b(?:conclusion|final answer|answer|therefore|thus)\s*[:\-]\s*(.+)$",
    flags=re.IGNORECASE,
)
_YES_NO_RE = re.compile(r"^\s*(yes|no)\b", flags=re.IGNORECASE)
_NUMBER_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\b")
_PROMPT_PREFIX_RE = re.compile(
    r"^\s*<?\s*prompt(?:[^A-Za-z0-9_]+|[A-Za-z0-9_]*\s+)\s*[:>\-]?\s*",
    flags=re.IGNORECASE,
)
_PROMPT_QA_MARKERS = (
    "<qa_pairs",
    "<qa>",
    "<qa ",
    "<question>",
    "<answer>",
    "</qa_pairs>",
    "</qa>",
    "</question>",
    "</answer>",
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
        "- Use a concrete, verifiable question with a single correct answer.\n"
        "- Question must end with '?'.\n"
        "- Use English only.\n"
        "Output only this XML and no extra text:\n"
        "<question>...</question>"
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
        "You are a visual question proposer.\n"
        f"{diff_hint}\n"
        f"Generate exactly {n} objective, image-grounded questions.\n"
        "Rules:\n"
        "- Use English only.\n"
        "- Every question must end with '?'.\n"
        "- Avoid speculative/subjective wording.\n"
        "- Avoid forced-choice binary templates.\n"
        "- Keep each question concise and verifiable.\n"
        "Output only this XML and no extra text:\n"
        "<questions>\n"
        "  <question>...</question>\n"
        "  <question>...</question>\n"
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
        "3) Every QA question must end with '?'.\n"
        "4) Keep every answer short (1-8 words).\n"
        f"5) Provide at least {n} QA pairs.\n"
        "Output only this XML and no extra text:\n"
        "<prompt>...</prompt>\n"
        "<qa_pairs>\n"
        "  <qa><question>...</question><answer>...</answer></qa>\n"
        "</qa_pairs>"
    )


def strip_tags(text: str, tag: str) -> str:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", flags=re.IGNORECASE | re.DOTALL)
    m = pattern.search(text or "")
    return m.group(1).strip() if m else ""


def _cleanup_question_text(text: str) -> str:
    raw = str(text or "")
    raw = raw.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    cleaned = _TAG_RE.sub(" ", raw)
    cleaned = " ".join(cleaned.replace("\n", " ").split())
    return cleaned.strip()


def _cleanup_prompt_text(text: str) -> str:
    raw = str(text or "")
    raw = raw.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    cleaned = " ".join(raw.replace("\n", " ").split())
    cleaned = _PROMPT_PREFIX_RE.sub("", cleaned).strip()
    lower = cleaned.lower()
    for marker in _PROMPT_QA_MARKERS:
        idx = lower.find(marker)
        if idx >= 0:
            cleaned = cleaned[:idx].strip()
            lower = cleaned.lower()
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = " ".join(cleaned.split()).strip(" '\"")
    if not cleaned:
        return ""
    if re.match(r"^(q|question|a|answer)\s*[:\-]\s*", cleaned, flags=re.IGNORECASE):
        return ""
    if len(cleaned) < 12:
        return ""
    return cleaned


def salvage_question(text: str) -> str:
    cleaned = _cleanup_question_text(text)
    if not cleaned:
        return ""

    label_match = re.search(r"(?:^|\b)(?:question|q)\s*[:\-]\s*(.+)$", cleaned, flags=re.IGNORECASE)
    if label_match:
        cleaned = " ".join(str(label_match.group(1) or "").split()).strip()
        if not cleaned:
            return ""

    if is_well_formed_question(cleaned):
        return cleaned

    for slice_text in _QUESTION_SLICE_RE.findall(cleaned):
        candidate = " ".join(str(slice_text).split())
        candidate = re.sub(r"^[^A-Za-z0-9]+", "", candidate).strip()
        if not candidate:
            continue
        if candidate.count("?") > 1:
            candidate = candidate.split("?", 1)[0].strip() + "?"
        if is_well_formed_question(candidate):
            return candidate

    # Last resort: turn a single clean sentence into a question.
    if "?" not in cleaned:
        sentence = re.sub(r"^(?:the\s+)?question\s*(?:is|:)?\s*", "", cleaned, flags=re.IGNORECASE)
        sentence = sentence.rstrip(" .!;:,")
        if sentence and _INTERROGATIVE_START_RE.search(sentence):
            candidate = f"{sentence}?"
            if is_well_formed_question(candidate):
                return candidate
        m_embedded = re.search(
            r"(?:^|[.;]\s+)((?:what|which|who|whom|whose|where|when|why|how|is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\b[^?]{5,220})$",
            sentence,
            flags=re.IGNORECASE,
        )
        if m_embedded:
            candidate = f"{str(m_embedded.group(1) or '').strip()}?"
            if is_well_formed_question(candidate):
                return candidate
    return ""


def parse_proposer_question_candidates(text: str) -> List[Dict[str, str]]:
    raw = str(text or "")
    blocks = _QUESTION_BLOCK_RE.findall(raw)
    candidates: List[Dict[str, str]] = []
    for block in blocks:
        q_text = (
            salvage_question(strip_tags(block, "text"))
            or salvage_question(parse_first_question(block))
        )
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

    fallback_questions: List[str] = []
    for text_block in _TEXT_TAG_RE.findall(raw):
        q = salvage_question(text_block)
        if q:
            fallback_questions.append(q)
    for line in raw.splitlines():
        line_s = str(line).strip()
        if not line_s:
            continue
        q = salvage_question(line_s)
        if q:
            fallback_questions.append(q)

    out: List[Dict[str, str]] = []
    seen = set()
    for q in (fallback_questions + parse_all_questions(raw)):
        q_clean = " ".join(str(q or "").split()).strip()
        if not is_well_formed_question(q_clean):
            continue
        key = q_clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": q_clean})
    return out


def parse_first_question(text: str) -> str:
    raw = text or ""
    match = _QUESTION_TAG_RE.search(raw)
    if match:
        block = str(match.group(1) or "")
        if ("<" in block and ">" in block):
            text_inner = strip_tags(block, "text")
            if text_inner:
                q = salvage_question(text_inner)
                if q:
                    return q
        else:
            q = salvage_question(block)
            if q:
                return q
    text_match = _TEXT_TAG_RE.search(raw)
    if text_match:
        q = salvage_question(str(text_match.group(1) or ""))
        if q:
            return q
    for line in raw.splitlines():
        line = line.strip()
        if "?" in line and len(line) > 3:
            q = salvage_question(line)
            if q:
                return q
    return ""


def parse_all_questions(text: str) -> List[str]:
    raw = text or ""
    matches = _QUESTION_TAG_RE.findall(raw)
    if matches:
        out: List[str] = []
        seen = set()
        for m in matches:
            q = salvage_question(m)
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
        q = salvage_question(val)
        if not q:
            continue
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
    raw = str(text or "")
    match = _ANSWER_TAG_RE.search(raw)
    if match:
        val = " ".join(match.group(1).strip().split())
        val = _ANSWER_PREFIX_RE.sub("", val).strip()
        return val

    # Recover malformed "<answer ...>" without a proper closing tag.
    malformed = _ANSWER_OPEN_RE.search(raw)
    if malformed:
        candidate = str(malformed.group(1) or "")
        candidate = candidate.split("</", 1)[0]
        candidate = _TAG_RE.sub(" ", candidate)
        candidate = " ".join(candidate.split()).strip()
        candidate = _ANSWER_PREFIX_RE.sub("", candidate).strip()
        if candidate:
            m_bool = _YES_NO_RE.match(candidate)
            if m_bool:
                return m_bool.group(1).lower()
            m_num = _NUMBER_RE.match(candidate)
            if m_num:
                return m_num.group(0)
            return " ".join(candidate.split()[:8]).strip()

    # Strip reasoning blocks and salvage concise final answer.
    cleaned = _THINK_BLOCK_RE.sub(" ", raw)
    cleaned = cleaned.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        return ""
    cleaned = _ANSWER_PREFIX_RE.sub("", cleaned).strip()
    m_conclusion = _CONCLUSION_RE.search(cleaned)
    if m_conclusion:
        cleaned = str(m_conclusion.group(1) or "").strip()

    m_bool = _YES_NO_RE.match(cleaned)
    if m_bool:
        return m_bool.group(1).lower()
    m_num = _NUMBER_RE.match(cleaned)
    if m_num:
        return m_num.group(0)

    first_segment = re.split(r"[.!?。！？]", cleaned, maxsplit=1)[0].strip()
    if not first_segment:
        first_segment = cleaned
    return " ".join(first_segment.split()[:8]).strip()


def is_objective_question(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    return _NON_OBJECTIVE_RE.search(q) is None


def is_well_formed_question(question: str) -> bool:
    q = " ".join(str(question or "").strip().split())
    if not q:
        return False
    if len(q) < 8 or len(q) > 220:
        return False
    if _INVALID_QUESTION_ARTIFACT_RE.search(q):
        return False
    if _RUNAWAY_PUNCT_RE.search(q):
        return False
    if q.count("?") != 1 or not q.endswith("?"):
        return False

    non_space = [ch for ch in q if not ch.isspace()]
    if not non_space:
        return False
    alpha_count = sum(1 for ch in non_space if ch.isalpha())
    digit_count = sum(1 for ch in non_space if ch.isdigit())
    alpha_ratio = float(alpha_count) / float(len(non_space))
    digit_ratio = float(digit_count) / float(len(non_space))
    if alpha_ratio < 0.45:
        return False
    if digit_ratio > 0.35:
        return False

    if " " in q:
        toks = q.split()
        if len(toks) < 3 or len(toks) > 40:
            return False
    return True


def parse_generation_spec(text: str, min_qa_pairs: int = 2) -> Optional[GenerationSpec]:
    raw = text or ""
    prompt_candidates: List[str] = []
    prompt_match = _PROMPT_TAG_RE.search(raw)
    if prompt_match:
        prompt_candidates.append(str(prompt_match.group(1) or ""))

    relaxed = _PROMPT_RELAXED_RE.search(raw)
    if relaxed:
        prompt_candidates.append(str(relaxed.group(1) or ""))

    for line in raw.splitlines():
        line_s = str(line).strip()
        if not line_s:
            continue
        if line_s.lower().startswith("<prompt"):
            prompt_candidates.append(line_s)
            break

    for line in raw.splitlines():
        candidate = " ".join(str(line).strip().split())
        if not candidate:
            continue
        if candidate.startswith("<") and candidate.endswith(">"):
            continue
        if candidate.lower().startswith(("q:", "question:", "a:", "answer:")):
            continue
        prompt_candidates.append(candidate)
        break

    prompt = ""
    for candidate in prompt_candidates:
        cleaned = _cleanup_prompt_text(candidate)
        if cleaned:
            prompt = cleaned
            break

    qa_pairs: List[GenerationQAPair] = []
    seen_questions = set()

    def _append_pair(q_raw: str, a_raw: str) -> None:
        q = salvage_question(q_raw)
        a = " ".join(str(a_raw).strip().split())
        if not q or not a:
            return
        if len(a) > 120:
            return
        a_tokens = a.split()
        if len(a_tokens) > 16:
            return
        if _INVALID_QUESTION_ARTIFACT_RE.search(a):
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

    if len(qa_pairs) < max(1, int(min_qa_pairs)):
        for q_raw, a_raw in _QA_FALLBACK_RE.findall(raw):
            q_candidate = salvage_question(q_raw)
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
