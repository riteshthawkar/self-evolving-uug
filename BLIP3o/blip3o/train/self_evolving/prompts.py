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



def build_proposer_multi_prompt(
    target_difficulty: str = "medium",
    num_questions: int = 3,
) -> str:
    """Single-shot multi-question proposer prompt with adversarial solver-failure framing.

    The proposer's explicit goal is to craft questions that a vision-language
    solver (a peer model looking at the same image) would FAIL to answer
    unanimously — i.e. questions where solvers would disagree or make errors.
    This adversarial framing internalises difficulty estimation, eliminating the
    need for a retry loop.

    Returns K=num_questions candidate questions ordered hardest-first.
    Each candidate includes a chain-of-thought block reasoning about WHY the
    solver will fail before committing to the question text.
    """
    level = (target_difficulty or "medium").strip().lower()
    if level not in {"easy", "medium", "hard"}:
        level = "medium"

    if level == "hard":
        diff_hint = (
            "Target HARD difficulty: each question must require at least two visual constraints "
            "(e.g., comparison across entities + attribute, relation + count). "
            "A solver that sees the same image should be uncertain and frequently give DIFFERENT answers."
        )
    elif level == "easy":
        diff_hint = (
            "Target EASY-MEDIUM difficulty: questions should be objective and non-trivial; "
            "avoid one-word lookups. A solver should sometimes disagree on the precise answer."
        )
    else:
        diff_hint = (
            "Target MEDIUM difficulty: each question must require at least two visual cues "
            "(count + attribute, relation + attribute, or comparison). "
            "A solver that sees the same image should frequently give different answers."
        )

    n = max(1, int(num_questions))
    qa_template = "\n".join(
        f'  <question id="{i}">\n'
        f'    <solver_failure_reasoning>...why the solver will fail or disagree here...</solver_failure_reasoning>\n'
        f'    <text>...the question text ending with "?"...</text>\n'
        f'    <rationale>...why this question is objective and image-grounded...</rationale>\n'
        f'  </question>'
        for i in range(1, n + 1)
    )

    return (
        "You are an Adversarial Question Proposer.\n"
        "Your GOAL is to craft questions that a vision-language solver (a peer model "
        "viewing the same image) will FAIL to answer unanimously — questions where "
        "solvers frequently disagree or make errors.\n"
        f"{diff_hint}\n"
        f"Generate exactly {n} candidate questions, ordered HARDEST first (question 1 = hardest).\n"
        "For EACH question you MUST first reason in <solver_failure_reasoning> about:\n"
        "  - What visual ambiguity, complexity, or multi-step reasoning makes this hard?\n"
        "  - What wrong answers is the solver likely to produce and why?\n"
        "  - Why will solvers DISAGREE with each other on this question?\n"
        "Then write the actual question text and your rationale.\n"
        "\n"
        "Hard constraints for every question:\n"
        "- Ask an objective, image-grounded question with a short, verifiable answer (1-5 words).\n"
        "- Do NOT make questions trivially easy (single obvious attribute, color of most prominent object, etc.).\n"
        "- Prefer: counting under occlusion, comparison between similar-looking entities, "
        "attribute + spatial relation combinations, subtle quantity differences, chart/table lookup + comparison.\n"
        "- Use a proper interrogative (must end with '?').\n"
        "- Avoid: why, might, could, likely, feel, opinion, speculative/subjective wording.\n"
        "- Avoid open-ended narrative prompts.\n"
        "- Do not output placeholders like '(count + attribute)' — use concrete objects from the image.\n"
        "- Do not include XML tags inside the question text itself.\n"
        "- Do not require external knowledge beyond what is visible in the image.\n"
        "\n"
        "Output XML only:\n"
        "<questions>\n"
        f"{qa_template}\n"
        "</questions>"
    )
