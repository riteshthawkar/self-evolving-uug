"""
Prompt templates for the self-evolving training pipeline.
Ported from self_evolving/experiments/understanding.py and generation.py.
"""


def build_proposer_prompt(target_difficulty: str = "medium") -> str:
    level = (target_difficulty or "medium").strip().lower()
    if level not in {"easy", "medium", "hard"}:
        level = "medium"
    if level == "hard":
        diff_hint = (
            "Target HARD: question should require multi-step reasoning with at least two visual constraints "
            "(for example: comparison + attribute, relation + count)."
        )
    elif level == "easy":
        diff_hint = (
            "Target EASY-MEDIUM: keep question objective and avoid trivial one-word lookups."
        )
    else:
        diff_hint = (
            "Target MEDIUM: question should require at least two visual constraints "
            "(for example: count + attribute, relation + attribute, or comparison between entities)."
        )
    return (
        "You are a Question Proposer.\n"
        "Given the image, generate exactly one question that can be answered from the image alone.\n"
        f"{diff_hint}\n"
        "Rules:\n"
        "- Ask an objective, image-grounded question with a verifiable short answer.\n"
        "- Prefer counting, comparison, lookup, spatial relation, or attribute questions.\n"
        "- Use a proper interrogative question (must end with '?').\n"
        "- Avoid subjective/speculative wording such as 'why', 'might', 'could', 'likely', 'feel', or 'opinion'.\n"
        "- Avoid open-ended narrative prompts.\n"
        "- Do not output placeholders/templates like '(count + attribute)'. Use concrete objects from the image.\n"
        "- Do not include XML tags like <answer> or <rationale> inside the question text.\n"
        "- The answer should be short (a few words) and directly checkable from image evidence.\n"
        "- Do not require external knowledge beyond what is visible.\n"
        "- Output XML only:\n"
        "<question>...</question>\n"
        "<rationale>...</rationale>"
    )


def build_solver_prompt(question_text: str) -> str:
    return (
        "You are a precise vision-language solver.\n"
        "Answer the question using only the provided image.\n"
        "Rules:\n"
        "- Your answer MUST be 1-5 words only. No full sentences.\n"
        "- Give only the core answer, not an explanation.\n"
        "- Examples of good answers: 'primary producer', '42%', 'increases then decreases', 'red circle'\n"
        "- Return only the final answer inside XML:\n"
        "<answer>...</answer>\n"
        f"Question: {question_text}"
    )


def build_caption_prompt() -> str:
    return "Describe this image in detail."


def build_generator_prompt(prompt: str) -> str:
    return (
        f"Please generate image based on the following caption: {prompt}"
    )


def build_generation_spec_prompt(target_difficulty: str = "medium") -> str:
    level = (target_difficulty or "medium").strip().lower()
    if level not in {"easy", "medium", "hard"}:
        level = "medium"
    if level == "hard":
        diff_hint = (
            "Target HARD verification: each QA should require at least two visual constraints "
            "(e.g., relation + attribute, comparison + count)."
        )
    elif level == "easy":
        diff_hint = (
            "Target EASY-MEDIUM verification: keep QA objective but avoid trivial one-word lookups."
        )
    else:
        diff_hint = (
            "Target MEDIUM verification: each QA should require at least two visual cues "
            "(count + attribute, relation + attribute, or comparison)."
        )
    return (
        "You are a generation-spec proposer for self-evolving training.\n"
        "Given the source image, propose one text-to-image prompt and verification QA pairs.\n"
        f"{diff_hint}\n"
        "Rules:\n"
        "- Prompt must be image-grounded but not a trivial copy.\n"
        "- Prompt must be declarative (caption/instruction style), not a question.\n"
        "- Do not use a question mark in the prompt.\n"
        "- QA pairs must be objective, short-answer, and visually verifiable.\n"
        "- Avoid subjective wording: why, might, could, likely, feel, opinion.\n"
        "- Avoid trivial single-attribute QA; prefer compositional checks.\n"
        "- Expected answers must be concise (1-6 words).\n"
        "Output XML only:\n"
        "<prompt>...</prompt>\n"
        "<spec>\n"
        "  <qa><question>...</question><expected>...</expected></qa>\n"
        "  <qa><question>...</question><expected>...</expected></qa>\n"
        "  <qa><question>...</question><expected>...</expected></qa>\n"
        "</spec>"
    )


def build_generation_spec_retry_prompt(
    previous_prompt: str,
    reason: str,
    target_difficulty: str = "medium",
) -> str:
    prev = (previous_prompt or "").strip()
    why = (reason or "spec quality was too low").strip()
    level = (target_difficulty or "medium").strip().lower()
    if level not in {"easy", "medium", "hard"}:
        level = "medium"
    if level == "hard":
        diff_hint = "Regenerate with HARD verification QA."
    elif level == "easy":
        diff_hint = "Regenerate with EASY-MEDIUM verification QA."
    else:
        diff_hint = "Regenerate with MEDIUM verification QA."
    return (
        "You are a generation-spec proposer for self-evolving training.\n"
        f"{diff_hint}\n"
        "Your previous spec was rejected. Produce a better one.\n"
        "Mandatory rules:\n"
        "- Prompt must be declarative and image-grounded.\n"
        "- QA pairs must be objective and visually verifiable.\n"
        "- Each QA should combine at least two visual signals when possible.\n"
        "- Avoid trivial single-hop QA and subjective wording.\n"
        "Previous prompt:\n"
        f"{prev}\n"
        "Rejection reason:\n"
        f"{why}\n"
        "Output XML only:\n"
        "<prompt>...</prompt>\n"
        "<spec>\n"
        "  <qa><question>...</question><expected>...</expected></qa>\n"
        "  <qa><question>...</question><expected>...</expected></qa>\n"
        "  <qa><question>...</question><expected>...</expected></qa>\n"
        "</spec>"
    )


def build_spec_proposer_prompt() -> str:
    """Prompt for generation-loop proposer: propose a specification
    (questions + expected answers) to verify a generated image."""
    return (
        "You are a Verification Spec Proposer.\n"
        "Given the image, generate a structured specification that can verify "
        "if a generated image accurately represents this scene.\n"
        "Output XML with question-answer pairs:\n"
        "<spec>\n"
        "  <qa>\n"
        "    <question>...</question>\n"
        "    <expected_answer>...</expected_answer>\n"
        "  </qa>\n"
        "  ...\n"
        "</spec>"
    )


def build_proposer_hardening_prompt(previous_question: str, reason: str) -> str:
    """Prompt proposer to rewrite a too-easy/subjective question into a harder objective one."""
    prev_q = (previous_question or "").strip()
    reason_txt = (reason or "too easy").strip()
    return (
        "You are a Question Proposer.\n"
        "Rewrite the previous question into a harder, objective, image-grounded question.\n"
        "Hard constraints:\n"
        "- The answer must be verifiable from the image only.\n"
        "- Use concrete formulations (count, compare, lookup, spatial relation, attribute).\n"
        "- Avoid trivial single-attribute lookups (e.g., only color/name of obvious object).\n"
        "- Prefer multi-constraint questions (count + attribute, relation + attribute, compare + value).\n"
        "- Do NOT use: why, might, could, likely, opinion, feel, emotion.\n"
        "- Keep the expected answer short (1-5 words).\n"
        "- Avoid yes/no unless unavoidable.\n"
        "Previous question:\n"
        f"{prev_q}\n"
        "Why rewrite is needed:\n"
        f"{reason_txt}\n"
        "Output XML only:\n"
        "<question>...</question>\n"
        "<rationale>...</rationale>"
    )


def build_proposer_force_hard_prompt(
    previous_question: str,
    reason: str,
    target_bucket: str = "medium",
) -> str:
    """Emergency hardening prompt used when regular retries failed."""
    prev_q = (previous_question or "").strip()
    reason_txt = (reason or "question remained too easy").strip()
    target = (target_bucket or "medium").strip().lower()
    if target not in {"easy", "medium", "hard"}:
        target = "medium"
    if target == "hard":
        difficulty_hint = (
            "Aim for HARD difficulty: require at least two visual constraints and a non-trivial comparison."
        )
    else:
        difficulty_hint = (
            "Aim for at least MEDIUM difficulty: require at least two visual constraints."
        )
    return (
        "You are a Question Proposer.\n"
        "Create ONE objectively verifiable question from the image.\n"
        f"{difficulty_hint}\n"
        "Mandatory rules:\n"
        "- Do not ask a simple one-hop lookup (single color/name/count of the most obvious object).\n"
        "- Use one of these forms: (count + attribute), (spatial relation + attribute), (comparison across two entities), (table/chart lookup + comparison).\n"
        "- The answer must be 1-5 words and directly grounded in image evidence.\n"
        "- Do NOT use: why, might, could, likely, feel, opinion, purpose, reason.\n"
        "- Avoid yes/no unless unavoidable.\n"
        "Previous question:\n"
        f"{prev_q}\n"
        "Failure reason:\n"
        f"{reason_txt}\n"
        "Output XML only:\n"
        "<question>...</question>\n"
        "<rationale>...</rationale>"
    )


def build_proposer_template_fallback_prompt(
    previous_question: str,
    reason: str,
    target_bucket: str = "medium",
) -> str:
    """Template-constrained fallback when standard hardening retries are exhausted."""
    prev_q = (previous_question or "").strip()
    reason_txt = (reason or "question remained too easy after retries").strip()
    target = (target_bucket or "medium").strip().lower()
    if target not in {"easy", "medium", "hard"}:
        target = "medium"
    if target == "hard":
        diff_hint = (
            "Choose a HARD template that compares two entities and requires at least two constraints."
        )
    else:
        diff_hint = (
            "Choose a MEDIUM template that combines at least two visual constraints."
        )
    return (
        "You are a Question Proposer.\n"
        "Your previous attempts were rejected. Produce one final objective question using a strict template.\n"
        f"{diff_hint}\n"
        "Mandatory rules:\n"
        "- Replace placeholders with concrete objects/values visible in the image.\n"
        "- The question must end with '?'.\n"
        "- The answer must be short (1-5 words).\n"
        "- Do not output placeholders such as '(count + attribute)'.\n"
        "- Do not output statements; output only a question.\n"
        "- Do NOT use: why, might, could, likely, opinion, feel, emotion.\n"
        "Allowed templates (choose one and fill concretely):\n"
        "1) How many [OBJECT A] are [SPATIAL RELATION] the [OBJECT B]?\n"
        "2) Which has more [ATTRIBUTE], [ENTITY A] or [ENTITY B]?\n"
        "3) What is the difference between [VALUE A] and [VALUE B] shown in the chart/table?\n"
        "4) Is the number of [OBJECT A] greater than the number of [OBJECT B]?\n"
        "Previous question:\n"
        f"{prev_q}\n"
        "Failure reason:\n"
        f"{reason_txt}\n"
        "Output XML only:\n"
        "<question>...</question>\n"
        "<rationale>...</rationale>"
    )
