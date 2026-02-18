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
        "- The prompt must naturally embed all verifiable details from your QA pairs.\n"
        "  Every fact the QA checks (count, color, sport, location, text, etc.) must\n"
        "  appear explicitly in the prompt — the image generator sees only the prompt.\n"
        "  Example: if QA asks 'How many players?' → 'Two', write 'two players' in the prompt.\n"
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
        "- The prompt must embed all verifiable details from your QA pairs.\n"
        "  Every fact the QA checks must appear explicitly in the prompt text.\n"
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
    image_source_hint: str = "",
) -> str:
    """Single-shot multi-question proposer prompt with adversarial solver-failure framing.

    The proposer's explicit goal is to craft questions that a vision-language
    solver (a peer model looking at the same image) would FAIL to answer
    unanimously — i.e. questions where solvers would disagree or make errors.

    Key design principles (researcher notes):
    1. STRATEGY LIBRARY: Give the proposer concrete question-writing strategies
       that empirically cause solver disagreement, not just abstract difficulty levels.
    2. ANTI-PATTERN BLOCKING: Explicitly name the easy-question patterns the model
       defaults to (single salient object, dominant scene action, obvious text) and
       forbid them.
    3. AMBIGUITY IS THE TARGET: The proposer must identify a visual element where
       the *answer is genuinely uncertain* due to occlusion, similarity, scale, or
       boundary conditions — not just "complex to describe".
    4. TWO-ANSWER TEST: Each question must pass the internal test "could a reasonable
       person looking at this image give TWO different answers?" If not, it is easy.

    Returns K=num_questions candidate questions ordered hardest-first.
    Each candidate includes a chain-of-thought block reasoning about WHY each
    question will cause the solver to fail/disagree.
    """
    level = (target_difficulty or "medium").strip().lower()
    if level not in {"easy", "medium", "hard"}:
        level = "medium"

    if level == "hard":
        diff_hint = (
            "TARGET: HARD — the solver should be genuinely uncertain.\n"
            "Required: pick a question strategy from the HARD tier below and apply it to a specific "
            "visual element in this image where the answer is genuinely ambiguous.\n"
            "The correct answer should NOT be immediately obvious from the most salient part of the image."
        )
    elif level == "easy":
        diff_hint = (
            "TARGET: EASY-MEDIUM — avoid single-lookup questions but keep the answer verifiable.\n"
            "Pick a strategy from the MEDIUM tier below that requires noticing at least two objects or "
            "one non-obvious attribute. Solvers should sometimes give different precise answers."
        )
    else:
        diff_hint = (
            "TARGET: MEDIUM — the solver should frequently disagree on the precise answer.\n"
            "Required: pick a question strategy from the MEDIUM or HARD tier below and apply it to "
            "a non-salient or ambiguous visual element in this image.\n"
            "Do NOT ask about the most obvious/dominant element in the scene."
        )

    # Strategy library — compact format to minimise prompt input tokens.
    # Full descriptions are omitted; the model already knows these patterns.
    strategy_block = (
        "STRATEGIES (pick one per question; apply to a SPECIFIC element in THIS image):\n"
        "HARD: H1=occlusion-count  H2=boundary-comparison  H3=occluded-attribute\n"
        "      H4=multi-hop-relation  H5=edge-condition  H6=secondary-count\n"
        "      H7=occluded-text  H8=chart-delta\n"
        "MEDIUM: M1=precise-count  M2=relative-spatial  M3=comparative-attr\n"
        "        M4=secondary-attr  M5=non-obvious-existence\n"
        "NEVER (always easy): main-subject-action | object-type | dominant-text | "
        "single-salient-color | unambiguous-yes-no | obvious-single-word-answer\n"
    )

    # Dataset-specific one-liner hint (kept short to save tokens).
    src = (image_source_hint or "").strip().lower()
    if src == "textvqa":
        dataset_hint = "IMAGE: text-in-scene. Prefer H7, H4, M2. Avoid largest visible text.\n"
    elif src in {"chartqa", "chart"}:
        dataset_hint = "IMAGE: chart/graph. Prefer H8 (small delta), H6, M3 (2nd-highest). Avoid peak-bar or title.\n"
    elif src == "gqa":
        dataset_hint = "IMAGE: relational scene. Prefer H4, H2, H1, M2. Avoid dominant-object type/color.\n"
    else:
        dataset_hint = "IMAGE: natural photo. Prefer H1, H2, H3, H6, M2 on SECONDARY objects. Avoid main-subject action or type.\n"

    n = max(1, int(num_questions))
    qa_template = "\n".join(
        f'  <question id="{i}">\n'
        f'    <strategy_used>...which strategy from the library above (e.g. H2, M3)...</strategy_used>\n'
        f'    <two_answer_test>...two plausible but DIFFERENT answers the solver might give and why...</two_answer_test>\n'
        f'    <text>...the question text ending with "?"...</text>\n'
        f'    <rationale>...why this is objective, image-grounded, and NOT an anti-pattern...</rationale>\n'
        f'  </question>'
        for i in range(1, n + 1)
    )

    return (
        "You are an Adversarial Question Proposer.\n"
        "GOAL: questions a vision-language solver (same model, same image) will FAIL to answer unanimously.\n"
        f"{diff_hint}\n"
        f"{dataset_hint}"
        f"{strategy_block}"
        f"Generate exactly {n} questions, HARDEST first. For each:\n"
        "  1. <strategy_used>: which strategy (e.g. H2).\n"
        "  2. <two_answer_test>: two different plausible solver answers. If you can't — it's easy, pick another strategy.\n"
        "  3. <text>: the question (ends with '?', uses concrete image objects, 1-5 word answer).\n"
        "  4. <rationale>: why it is objective, image-grounded, NOT an anti-pattern.\n"
        "Rules: verifiable short answer, no subjective wording, no XML tags inside question text.\n"
        "Output XML only:\n"
        "<questions>\n"
        f"{qa_template}\n"
        "</questions>"
    )
