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

    # Strategy library — concrete patterns that empirically cause solver disagreement
    strategy_block = (
        "QUESTION STRATEGY LIBRARY (pick one strategy per question, then apply it to a SPECIFIC element in this image):\n"
        "\n"
        "HARD tier — highest solver disagreement:\n"
        "  H1. Occlusion count: Count objects where some are partially hidden or overlapping.\n"
        '      Example structure: "How many [objects] are fully/partially visible [location]?"\n'
        "  H2. Boundary comparison: Compare two objects of similar size/position where the answer is not obvious.\n"
        '      Example structure: "Is [obj A] [larger/closer/higher/left of] [obj B]?"\n'
        "  H3. Attribute under occlusion: Ask about an attribute of an object that is partially hidden.\n"
        '      Example structure: "What color/shape is the [object] partially behind [occluder]?"\n'
        "  H4. Multi-hop relation: Ask about a property of an object identified by its relation to another.\n"
        '      Example structure: "What is the [attribute] of the [object] that is [relation] the [anchor]?"\n'
        "  H5. Boundary/edge condition: Ask whether something is on a boundary, inside, or outside.\n"
        '      Example structure: "Is the [object] on/inside/outside the [boundary]?"\n'
        "  H6. Non-dominant count: Count a SECONDARY category of object (not the main subject).\n"
        '      Example structure: "How many [background/secondary objects] are there?"\n'
        "  H7. Text reading under transformation: Read text that is rotated, small, or partially occluded.\n"
        '      Example structure: "What does the text on the [specific location] say?"\n'
        "  H8. Chart delta: Ask about a difference or change between two close values in a chart/table.\n"
        '      Example structure: "By how much did [metric] change between [year A] and [year B]?"\n'
        "\n"
        "MEDIUM tier — moderate solver disagreement:\n"
        "  M1. Precise count of non-trivial objects (not just '1' or '2').\n"
        '      Example structure: "How many [objects with shared attribute] are in the image?"\n'
        "  M2. Relative spatial relation between two non-dominant objects.\n"
        '      Example structure: "Is [obj A] to the left/right/above/below [obj B]?"\n'
        "  M3. Comparative attribute across two entities.\n"
        '      Example structure: "Which [object] is [attribute: larger/darker/taller]?"\n'
        "  M4. Specific attribute of a secondary (non-salient) object.\n"
        '      Example structure: "What color/shape/material is the [background/secondary object]?"\n'
        "  M5. Existence of a non-obvious element.\n"
        '      Example structure: "Is there a [specific object] in the [specific region]?"\n'
        "\n"
        "ANTI-PATTERNS — these always produce easy questions, NEVER use them:\n"
        "  ✗ Asking what the MAIN SUBJECT/DOMINANT OBJECT is doing (e.g. 'What is the person doing?')\n"
        "  ✗ Asking what TYPE something obviously is (e.g. 'What type of pizza?', 'What sport?')\n"
        "  ✗ Asking about the most prominent text in the image\n"
        "  ✗ Asking about the color/type of the single most salient object\n"
        "  ✗ Yes/No questions where the answer is visually unambiguous\n"
        "  ✗ Asking about something that has an obvious single-word answer (e.g. 'What game is being played?')\n"
        "  ✗ Questions whose answer appears as large visible text in the image\n"
    )

    # Dataset-specific strategy hints: different image sources reward different strategies.
    src = (image_source_hint or "").strip().lower()
    if src == "textvqa":
        dataset_hint = (
            "IMAGE TYPE: Scene with embedded text (signs, labels, storefronts, etc.).\n"
            "Best strategies for this image type:\n"
            "  → H7 (text under transformation): prefer reading text that is rotated, small, or partially occluded.\n"
            "  → H5 (boundary condition): e.g. 'Is the text above or below the logo?'\n"
            "  → H4 (multi-hop relation): 'What is written on the [object] held by [person]?'\n"
            "  → M2 (spatial): 'Is the [sign/text] to the left or right of [landmark]?'\n"
            "  AVOID: Asking about the largest/most prominent text visible — that is always easy.\n"
        )
    elif src in {"chartqa", "chart"}:
        dataset_hint = (
            "IMAGE TYPE: Chart, graph, or table.\n"
            "Best strategies for this image type:\n"
            "  → H8 (chart delta): 'By how much did [metric] change between [year A] and [year B]?' — "
            "pick years where the difference is small and not immediately obvious.\n"
            "  → H6 (non-dominant count): 'How many data points are above/below the average line?'\n"
            "  → M3 (comparative): 'Which [category] had the second highest [metric]?' — NOT the highest.\n"
            "  → M1 (count): 'How many [bars/lines/segments] show a value between X and Y?'\n"
            "  AVOID: Asking which year had the highest value (trivially obvious from bar height).\n"
            "  AVOID: Asking what the chart is titled (single obvious text lookup).\n"
        )
    elif src == "gqa":
        dataset_hint = (
            "IMAGE TYPE: Natural scene with rich object relationships (GQA-style).\n"
            "Best strategies for this image type:\n"
            "  → H4 (multi-hop relation): 'What is the color of the object that is [relation] the [anchor]?'\n"
            "  → H2 (boundary comparison): compare two similar-sized objects (e.g. 'Is the cup taller than the vase?')\n"
            "  → H1 (occlusion count): count objects where some are partially obscured\n"
            "  → M2 (relative spatial): ask about spatial relation of two non-dominant objects\n"
            "  AVOID: Asking about the single most visible object's type or color.\n"
        )
    else:
        # Default: COCO-style natural images
        dataset_hint = (
            "IMAGE TYPE: Natural photograph (COCO-style).\n"
            "Best strategies for this image type:\n"
            "  → H1 (occlusion count): Count background/secondary objects that are partially hidden.\n"
            "  → H2 (boundary comparison): Compare two objects of similar apparent size.\n"
            "  → H3 (attribute under occlusion): Ask about attribute of a partially-hidden object.\n"
            "  → H6 (non-dominant count): Count objects SECONDARY to the main subject.\n"
            "  → M2 (relative spatial): 'Is [minor obj A] to the left/right of [minor obj B]?'\n"
            "  AVOID: Asking what the main person/animal/vehicle is doing — always easy.\n"
            "  AVOID: Asking the type of food/vehicle/sport in the scene — always easy.\n"
        )

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
        "Your GOAL: craft questions that a vision-language solver (same model, same image) will "
        "FAIL to answer unanimously — questions where different solver samples give DIFFERENT answers.\n"
        "\n"
        f"{diff_hint}\n"
        "\n"
        f"{dataset_hint}\n"
        f"{strategy_block}\n"
        f"Generate exactly {n} candidate questions, ordered HARDEST first (question 1 = hardest).\n"
        "For EACH question:\n"
        "  1. State which strategy you used (strategy_used).\n"
        "  2. Pass the TWO-ANSWER TEST: explicitly name two different answers the solver might give.\n"
        "     If you cannot name two plausible answers, the question is EASY — pick a different strategy.\n"
        "  3. Write the question text (must end with '?').\n"
        "  4. Confirm it is NOT an anti-pattern (rationale).\n"
        "\n"
        "Hard constraints:\n"
        "- Short, verifiable answer (1-5 words). Must be determinable from the image alone.\n"
        "- Use a proper interrogative. No speculative/subjective wording.\n"
        "- Use concrete object names from THIS image — never placeholder text.\n"
        "- Do not include XML tags inside the question text itself.\n"
        "\n"
        "Output XML only:\n"
        "<questions>\n"
        f"{qa_template}\n"
        "</questions>"
    )
