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


# ══════════════════════════════════════════════════════════════════════════════
# Imageless proposer (E5): topic list + text-only spec prompt
# ══════════════════════════════════════════════════════════════════════════════

# Diverse topic themes covering visual capabilities the model should learn.
# Each entry is a short topic description; the proposer LLM elaborates it into
# a concrete text-to-image prompt + verification QA pairs.
IMAGELESS_TOPIC_LIST = [
    # ---- Counting & arrangement ----
    "A dining table with several fruits of different colors arranged in a bowl",
    "A parking lot with multiple cars of varying colors and sizes",
    "A classroom with students sitting at desks and a teacher at a whiteboard",
    "A garden with various flowers in different colors and a stone path",
    "A shelf displaying books of different sizes and colors",

    # ---- Spatial relations ----
    "A cat sitting on top of a bookshelf next to a potted plant by a window",
    "A bicycle leaning against a brick wall near a wooden bench",
    "A bridge over a river with trees on both banks and a boat underneath",
    "A child standing between two adults in front of a house with a red door",
    "A lamp on a nightstand beside a bed with a painting on the wall above",

    # ---- Text rendering & signage ----
    "A storefront with a neon sign displaying the shop name and opening hours",
    "A road intersection with multiple street signs and a traffic light",
    "A restaurant menu board listing several dishes with prices",
    "A library entrance with a banner announcing an upcoming book fair event",
    "A sports scoreboard showing team names and current scores",

    # ---- Charts & diagrams ----
    "A bar chart comparing quarterly sales figures for four different products",
    "A pie chart showing the distribution of energy sources in a country",
    "A line graph tracking temperature changes across twelve months of a year",
    "A flow diagram illustrating the steps of a manufacturing process",
    "A Venn diagram showing the overlap between three scientific categories",

    # ---- Color binding & attributes ----
    "A person wearing a blue jacket, red scarf, and holding a yellow umbrella",
    "Three houses in a row: one red with a white fence, one blue with a garden, one green with a balcony",
    "A sports team photo with players in striped jerseys and numbered shirts",
    "A market stall with baskets of red tomatoes, green peppers, and orange carrots",
    "A painting studio with canvases showing different colored abstract art pieces",

    # ---- Multi-object complex scenes ----
    "A busy kitchen with a chef cooking at a stove, pots on shelves, and ingredients on a counter",
    "A beach scene with swimmers, a lifeguard tower, surfboards, and a volleyball net",
    "A construction site with a crane, workers wearing hard hats, and stacked building materials",
    "An office desk with a computer monitor, coffee mug, stack of papers, and a desk plant",
    "A farm scene with a red barn, a tractor, cows grazing, and a wooden fence",

    # ---- Nature & landscapes ----
    "A mountain lake reflecting snow-capped peaks with pine trees on the shore",
    "A desert landscape with sand dunes, a cactus, and a clear blue sky at sunset",
    "An underwater coral reef with colorful fish, sea anemones, and a sea turtle",
    "A rainforest with tall trees, hanging vines, a waterfall, and exotic birds",
    "A snowy village with wooden cabins, smoke from chimneys, and footprints in the snow",

    # ---- Architecture & interiors ----
    "A modern glass skyscraper next to a historical stone cathedral in a city square",
    "A cozy living room with a fireplace, a bookshelf, a sofa, and a coffee table",
    "A Japanese temple with a curved roof, stone lanterns, and a zen garden",
    "A subway station platform with trains, passengers, and electronic display boards",
    "A museum gallery with paintings on white walls, sculptures on pedestals, and visitors",

    # ---- People & activities ----
    "A group of musicians performing on a stage with different instruments",
    "A scientist in a laboratory examining samples under a microscope with equipment around",
    "A family having a picnic in a park with a checkered blanket and a wicker basket",
    "An artist painting a portrait on an easel in a studio with paint tubes and brushes",
    "Athletes competing in a track race at a stadium with spectators in the stands",

    # ---- Food & cooking ----
    "A sushi platter with different types of rolls, nigiri, and garnishes on a wooden board",
    "A bakery display with cupcakes, croissants, and bread loaves behind glass",
    "A breakfast table set with pancakes, orange juice, eggs, and a bowl of fruit",
    "A pizza being prepared with toppings like mushrooms, olives, and bell peppers",
    "A food truck at a festival serving tacos with customers waiting in line",

    # ---- Technology & machines ----
    "A robotics lab with mechanical arms, circuit boards, and computer screens",
    "A vintage car parked next to a modern electric vehicle in a garage",
    "A weather station on a hilltop with instruments measuring wind and temperature",
    "A drone flying over farmland capturing aerial images of crop fields",
    "A space control room with large monitors, satellite imagery, and operators at consoles",

    # ---- Abstract & conceptual ----
    "An infographic explaining the water cycle with labeled arrows and icons",
    "A world map highlighting countries with different colors for climate zones",
    "A timeline diagram showing major historical events across several centuries",
    "A periodic table poster with color-coded element groups and atomic numbers",
    "A blueprint of a house floor plan with labeled rooms and measurements",

    # ---- Grounding & object relations ----
    "A toy store window display with stuffed animals, board games, and action figures",
    "A workshop bench with hand tools, wood pieces, and a project being assembled",
    "A greenhouse with rows of seedlings in pots, watering cans, and hanging baskets",
    "A harbor with fishing boats, nets drying on poles, and seagulls on wooden posts",
    "A winter market with stalls selling ornaments, hot drinks, and handmade crafts",
]


def _sample_imageless_topic(step: int, seed: int = 42) -> str:
    """Deterministically sample a topic from the list for a given training step."""
    import random as _rng
    r = _rng.Random(seed + step)
    return r.choice(IMAGELESS_TOPIC_LIST)


def build_imageless_spec_prompt(topic: str, target_difficulty: str = "medium") -> str:
    """Build a text-only proposer prompt for imageless generation spec creation.

    The proposer receives a TOPIC (no image) and must:
    1. Invent a detailed text-to-image prompt grounded in the topic.
    2. Create verification QA pairs based on what the image SHOULD contain.

    This enables a fully synthetic self-evolving loop (E5) where no external
    images are used at any point in training.
    """
    level = (target_difficulty or "medium").strip().lower()
    if level not in {"easy", "medium", "hard"}:
        level = "medium"
    if level == "hard":
        diff_hint = (
            "Target HARD verification: each QA should require at least two visual constraints "
            "(e.g., relation + attribute, comparison + count, spatial + color)."
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
        "You are a generation-spec proposer for self-evolving image generation training.\n"
        "You are given a TOPIC DESCRIPTION (not an image). Your task is to:\n"
        "1. Create a detailed text-to-image prompt that a generator will use to create an image.\n"
        "2. Create verification QA pairs to check if the generated image is correct.\n"
        "\n"
        f"TOPIC: {topic}\n"
        "\n"
        f"{diff_hint}\n"
        "Rules:\n"
        "- The prompt must be a rich, detailed description suitable for image generation.\n"
        "- Include specific visual details: object counts, colors, spatial positions, sizes, text content.\n"
        "- The prompt must be declarative (caption/instruction style), not a question.\n"
        "- Do not use a question mark in the prompt.\n"
        "- Every fact checked by the QA pairs MUST appear explicitly in the prompt text.\n"
        "  The image generator sees ONLY the prompt — it cannot infer missing details.\n"
        "  Example: if QA asks 'How many apples?' → 'Three', write 'three apples' in the prompt.\n"
        "- QA pairs must be objective, short-answer, and visually verifiable.\n"
        "- Focus on: object counting, color identification, spatial relations, text content,\n"
        "  object attributes, relative sizes, actions, and scene composition.\n"
        "- Expected answers must be concise (1-6 words).\n"
        "- Avoid subjective wording: why, might, could, likely, feel, opinion.\n"
        "- Prefer compositional checks that combine multiple visual elements.\n"
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
