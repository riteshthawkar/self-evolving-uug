"""
Prompt templates for the self-evolving training pipeline.
Ported from self_evolving/experiments/understanding.py and generation.py.
"""


def build_proposer_prompt() -> str:
    return (
        "You are a Question Proposer.\n"
        "Given the image, generate exactly one question that can be answered from the image alone.\n"
        "Rules:\n"
        "- Ask an objective, image-grounded question with a verifiable short answer.\n"
        "- Prefer counting, comparison, lookup, spatial relation, or attribute questions.\n"
        "- Avoid subjective/speculative wording such as 'why', 'might', 'could', 'likely', 'feel', or 'opinion'.\n"
        "- Avoid open-ended narrative prompts.\n"
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
