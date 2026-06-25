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


def build_solver_prompt(question_text: str, focus_hint: str = "") -> str:
    hint = (focus_hint or "").strip()
    focus_line = (
        f"- Focus mode for this sample: {hint}. Prefer evidence consistent with this focus.\n"
        if hint
        else ""
    )
    return (
        "You are a precise vision-language solver.\n"
        "Answer the question using only the provided image.\n"
        "Rules:\n"
        "- Your answer MUST be 1-5 words only. No full sentences.\n"
        "- Give only the core answer, not an explanation.\n"
        "- The answer must be concrete and exact, not vague.\n"
        f"{focus_line}"
        "- For count questions, return a concrete integer.\n"
        "- Never output vague count words: 'too many', 'several', 'many', 'a lot', 'multiple', 'few'.\n"
        "- Never output uncertainty phrases: 'unclear', 'unknown', 'cannot tell', 'not visible'.\n"
        "- If unsure, still choose the single most likely concrete answer from visible evidence.\n"
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
    curriculum_arm_hint: str = "",
    replay_anchor_hints: list = None,
) -> str:
    """Compact multi-question proposer prompt.

    VARGPT is much more reliable when asked for short structured candidates
    instead of a long adversarial XML form. Keep the fields needed by the
    trainer's general quality/certificate checks, but avoid placeholder-heavy
    templates that the model may copy verbatim.
    """
    level = (target_difficulty or "medium").strip().lower()
    if level not in {"easy", "medium", "hard"}:
        level = "medium"

    if level == "hard":
        diff_hint = (
            "Target HARD: ask about a small, occluded, overlapping, distant, or visually subtle element. "
            "The answer must still be directly visible."
        )
    elif level == "easy":
        diff_hint = (
            "Target EASY-MEDIUM: ask a concrete visual question that requires more than naming the main object."
        )
    else:
        diff_hint = (
            "Target MEDIUM: ask about a secondary object, relation, count, attribute, text, or state. "
            "Avoid the most obvious object/action."
        )

    strategy_block = (
        "Useful question types: precise count, spatial relation, comparison, secondary attribute, "
        "visible text, object state, occlusion, part-whole detail, depth/overlap."
    )
    reasoning_domains_block = (
        "Use at least two simple domains per question, chosen from: spatial, counting, attribute, text, "
        "state, comparison, part, depth, action."
    )

    # Dataset-specific one-liner hint (kept short to save tokens).
    src = (image_source_hint or "").strip().lower()
    if src == "textvqa":
        dataset_hint = "Image type: text-in-scene. Prefer small/partial text plus location or attribute.\n"
    elif src in {"chartqa", "chart"}:
        dataset_hint = "Image type: chart/graph. Prefer small value differences, trends, or second-highest/lowest marks.\n"
    elif src == "gqa":
        dataset_hint = "Image type: relational scene. Prefer relation, count, comparison, and secondary attributes.\n"
    else:
        dataset_hint = "Image type: natural photo. Prefer secondary objects and concrete visible details.\n"

    arm_hint = (curriculum_arm_hint or "").strip()
    arm_block = ""
    if arm_hint:
        arm_block = (
            "CURRICULUM ARM PRIORITY (high utility for current solver):\n"
            f"- {arm_hint}\n"
            "Try to satisfy this priority in at least one candidate while keeping strict visual grounding.\n"
        )
    anchors = replay_anchor_hints or []
    anchor_block = ""
    if anchors:
        trimmed = [str(x).strip() for x in anchors if str(x).strip()]
        if trimmed:
            anchor_lines = "\n".join(f"  {i+1}. {x}" for i, x in enumerate(trimmed[:3]))
            anchor_block = (
                "HARD QUESTION EXEMPLARS (learn pattern, apply to THIS image):\n"
                f"{anchor_lines}\n"
                "KEY PATTERNS: spatial/depth, material texture, non-dominant actions, object states, "
                "compositional references, part-whole details, precise occluded counts.\n"
            )

    n = max(1, int(num_questions))

    return (
        "You are a vision question proposer. Look at the image and write candidate questions.\n"
        "Questions must be objective, image-grounded, and answerable with a short phrase or number.\n"
        "Do not ask why/opinion/intent questions. Do not include answer options.\n"
        "Use concrete visible objects from this image.\n"
        f"{diff_hint}\n"
        f"{dataset_hint}"
        f"{arm_block}"
        f"{anchor_block}"
        f"{reasoning_domains_block}\n"
        f"{strategy_block}\n"
        f"Generate exactly {n} different questions. Put the best question first.\n"
        "Output only the questions, one per line. Prefix lines as Q1:, Q2:, etc. "
        "Each line must end with '?'."
    )
