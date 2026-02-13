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
