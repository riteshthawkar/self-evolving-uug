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
            "Target HARD: question should require multi-step reasoning with at least two visible constraints "
            "(for example: comparison + attribute, relation + count), while keeping a single exact answer."
        )
    elif level == "easy":
        diff_hint = (
            "Target EASY-MEDIUM: keep question objective and avoid trivial one-word lookups."
        )
    else:
        diff_hint = (
            "Target MEDIUM: question should require at least two visible constraints "
            "(for example: count + attribute, relation + attribute, or comparison between entities), "
            "with one exact answer."
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
        "- Avoid easy binary forms and low-information outputs. Do NOT ask questions where likely answers are "
        "'yes/no', 'not visible', 'unclear', 'too many', or the main obvious object/action.\n"
        "- Avoid latent/physical-state questions that are weakly inferable from a still image "
        "(e.g., 'crispy or soft', 'fresh or stale', 'hot or cold').\n"
        "- Do not ask hidden-state intent questions (e.g., 'about to', 'just', 'trying to').\n"
        "- Prefer precise alternatives grounded in visible evidence (small text tokens, subtle color variants, "
        "tight spatial relations, partial counts under occlusion).\n"
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
        "- If the question asks 'how many' or 'number of', answer with a single integer (e.g., 0, 1, 2, 3).\n"
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
) -> str:
    """Single-shot multi-question proposer prompt with reasoning-first structure."""
    level = (target_difficulty or "medium").strip().lower()
    if level not in {"easy", "medium", "hard"}:
        level = "medium"

    if level == "hard":
        diff_hint = (
            "TARGET: HARD — the question should be difficult due to fine-grained visual grounding.\n"
            "Required: reasoning must include at least 2 domains and at least one non-relational domain.\n"
            "The final question must have one exact image-verifiable answer and not target the dominant object."
        )
    elif level == "easy":
        diff_hint = (
            "TARGET: EASY-MEDIUM — avoid single-lookup questions but keep the answer verifiable.\n"
            "Reasoning must still include at least 2 domains and a concrete two-answer ambiguity test."
        )
    else:
        diff_hint = (
            "TARGET: MEDIUM — the question should require careful multi-hop visual reasoning.\n"
            "Required: reasoning must include at least 2 domains and at least one of action/physics/behavior/"
            "commonsense/scientific.\n"
            "Question must target non-salient details grounded in visible evidence and yield one exact answer."
        )

    strategy_block = (
        "STRATEGIES (pick one per question; apply to a SPECIFIC fine-grained element in THIS image):\n"
        "HARD:\n"
        "  H1=occlusion-count: Count objects that overlap or are partially hidden behind others.\n"
        "     Ask about SMALL or BACKGROUND objects (not the main subject).\n"
        "     Example: 'How many chairs are partially visible behind the table?' (NOT 'How many people?')\n"
        "  H2=boundary-comparison: Compare sizes, distances, or positions of two similar objects.\n"
        "     Example: 'Is the left window wider than the right window?' (NOT 'What color is the car?')\n"
        "  H3=occluded-attribute: Ask about an attribute of an object that is partially hidden.\n"
        "     Example: 'What color is the shoe of the person whose legs are behind the counter?'\n"
        "  H4=multi-hop-relation: Chain two spatial or logical relations.\n"
        "     Example: 'What is to the left of the object that is on top of the bookshelf?'\n"
        "  H5=edge-condition: Ask about objects at the very edge of the image (cropped/cut off).\n"
        "     Example: 'Is the object at the right edge of the image a car or a truck?'\n"
        "  H6=secondary-count: Count small or background objects (NOT the main subject).\n"
        "     Example: 'How many bottles are on the shelf in the background?'\n"
        "  H7=occluded-text: Ask about small, partially visible, or distant text.\n"
        "     Example: 'What is the last word on the small sign in the background?' (NOT 'What brand is on the banner?')\n"
        "  H8=chart-delta: Ask about small differences between bars/lines in a chart.\n"
        "MEDIUM:\n"
        "  M1=precise-count: Count non-obvious objects requiring careful scanning.\n"
        "     Example: 'How many buttons are on the shirt?' (NOT 'How many people are in the image?')\n"
        "  M2=relative-spatial: Ask about relative position of non-dominant objects.\n"
        "     Example: 'Is the cup to the left or right of the napkin?'\n"
        "  M3=comparative-attr: Compare attributes of secondary objects.\n"
        "     Example: 'Which is taller, the lamp or the vase on the table?'\n"
        "  M4=secondary-attr: Ask about an attribute of a non-salient object.\n"
        "     Example: 'What pattern is on the tablecloth?' (NOT 'What color is the main object?')\n"
        "  M5=non-obvious-existence: Ask whether a small or non-obvious object exists.\n"
        "     Example: 'Is there a clock visible anywhere in the image?'\n"
        "\n"
        "BANNED PATTERNS (these ALWAYS produce easy/unanimous answers — NEVER use them):\n"
        "  X main-subject-identity: 'What animal is this?' 'What sport is being played?'\n"
        "  X dominant-text-reading: 'What brand name is on the banner?' 'What does the sign say?'\n"
        "  X single-salient-color: 'What color is the car?' 'What color is the helmet?'\n"
        "  X main-subject-count: 'How many people are in the image?' 'How many dogs?'\n"
        "  X obvious-yes-no: 'Is there a person in the image?' 'Is the sky blue?'\n"
        "  X main-action: 'What is the person doing?' 'What sport is being played?'\n"
    )
    reasoning_domains_block = (
        "REASONING DOMAINS (must include >=2 domains per question; include >=1 non-relational domain):\n"
        "  D1=relation/spatial\n"
        "  D2=action/temporal\n"
        "  D3=physics/mechanics\n"
        "  D4=behavior/intent\n"
        "  D5=commonsense/world-knowledge\n"
        "  D6=scientific/material inference\n"
        "  D7=causal/counterfactual\n"
    )
    hard_task_cards = (
        "HARD TASK CARDS (few-shot templates; do NOT copy literally, instantiate on this image):\n"
        "  C1 physics+causal: target=visible support/contact relation under occlusion; discriminate=two concrete contact configurations;\n"
        "     chain=contact cues -> gravity plausibility -> support conclusion.\n"
        "  C2 action+temporal: target=observable action proxy; discriminate=two concrete pose-grounded states;\n"
        "     chain=limb pose -> object trajectory -> temporal phase.\n"
        "  C3 behavior+commonsense: target=observable behavior cue; discriminate=two concrete visible behavior states;\n"
        "     chain=body orientation -> affordance cue -> context consistency.\n"
        "  C4 scientific+material: target=material identity; discriminate=two concrete materials with visible optical differences;\n"
        "     chain=highlight behavior -> edge softness -> texture consistency.\n"
        "  C5 OCR+context: target=partial text token; discriminate=two concrete candidate words from visible glyph fragments;\n"
        "     chain=glyph fragments -> lexical plausibility -> scene compatibility.\n"
        "  C6 causal+count: target=occluded count source; discriminate=two concrete count outcomes (e.g., 4 vs 5);\n"
        "     chain=occlusion map -> boundary ownership -> count consistency.\n"
    )

    # Dataset-specific one-liner hint (kept short to save tokens).
    src = (image_source_hint or "").strip().lower()
    if src == "textvqa":
        dataset_hint = (
            "IMAGE: text-in-scene. Target the SMALLEST or most distant text, "
            "NOT the largest/clearest text. Prefer H7, H4, M2.\n"
        )
    elif src in {"chartqa", "chart"}:
        dataset_hint = (
            "IMAGE: chart/graph. Ask about the SMALLEST bar, the 2nd or 3rd ranked value, "
            "or a close comparison. Prefer H8, H6, M3. Avoid peak-bar or title.\n"
        )
    elif src == "gqa":
        dataset_hint = (
            "IMAGE: relational scene. Ask about BACKGROUND objects, secondary actors, "
            "or partially visible items. Prefer H4, H2, H1, M2. Avoid the main subject.\n"
        )
    else:
        dataset_hint = (
            "IMAGE: natural photo. Ask about BACKGROUND details, secondary objects, "
            "partially visible items, or fine-grained attributes of non-dominant objects. "
            "Prefer H1, H3, H6, M4, M5. NEVER ask about the main subject.\n"
        )

    n = max(1, int(num_questions))
    qa_template = "\n".join(
        f'  <question id="{i}">\n'
        f'    <task_card>...C1/C2/C3/C4/C5/C6...</task_card>\n'
        f'    <reasoning_domains>...comma-separated D-codes, minimum 2...</reasoning_domains>\n'
        f'    <reasoning_chain>...2-4 short steps with visible evidence...</reasoning_chain>\n'
        f'    <strategy_used>...which strategy from the library above (e.g. H2, M3)...</strategy_used>\n'
        f'    <visual_target>...specific small/secondary/background element (must NOT be main subject)...</visual_target>\n'
        f'    <two_answer_test>...two genuinely DIFFERENT, concrete, image-grounded candidate answers '
        f'(never placeholders; never vague words like many/several/unclear)...</two_answer_test>\n'
        f'    <text>...final question text ending with "?" derived from reasoning_chain...</text>\n'
        f'    <rationale>...why this is hard but still objectively and exactly verifiable from visible evidence...</rationale>\n'
        f'  </question>'
        for i in range(1, n + 1)
    )

    return (
        "You are an Adversarial Fine-Grained Question Proposer.\n"
        "GOAL: produce hard, visually grounded, objective questions using reasoning-first construction.\n"
        "\n"
        "CRITICAL RULES:\n"
        "- NEVER ask about the main/dominant subject, object, or largest text.\n"
        "- Question must be derived AFTER reasoning_chain is defined.\n"
        "- Each question must use >=2 reasoning domains and include >=1 non-relational domain.\n"
        "- Each question must include a valid two-answer precision test with distinct concrete alternatives.\n"
        "- Final question must have one exact answer grounded in visible evidence.\n"
        "- NEVER copy literal card placeholders into final text (forbidden literals: token_1, token_2, stable-by-contact, unsupported, pre-event, post-event, preparing-for-X, recovering-from-X).\n"
        "- Avoid low-information yes/no forms and latent non-visual states.\n"
        "- Do not ask hidden-state intent or speculative timeline questions (e.g., about to/just/recently/trying to).\n"
        "- Use DISTINCT strategy codes across questions.\n"
        "\n"
        f"{diff_hint}\n"
        f"{dataset_hint}"
        f"{reasoning_domains_block}"
        f"{hard_task_cards}"
        f"{strategy_block}"
        f"Generate exactly {n} questions, HARDEST first.\n"
        "For each, follow this order strictly: task_card -> reasoning_domains -> reasoning_chain -> "
        "strategy_used -> visual_target -> two_answer_test -> text -> rationale.\n"
        "Question must end with '?' and have short verifiable answer.\n"
        "No XML tags inside question text.\n"
        "Final self-check before output:\n"
        "- If reasoning_domains has fewer than 2 domains, rewrite.\n"
        "- If no non-relational domain is present, rewrite.\n"
        "- If two_answer_test alternatives are not distinct, rewrite.\n"
        "- If two_answer_test contains vague alternatives (many/several/unclear/unknown), rewrite.\n"
        "- If question is not grounded in visual_target/reasoning_chain, rewrite.\n"
        "Output XML only:\n"
        "<questions>\n"
        f"{qa_template}\n"
        "</questions>"
    )
