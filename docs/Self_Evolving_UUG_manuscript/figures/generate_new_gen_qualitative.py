#!/usr/bin/env python3
"""
Generate temporary placeholder qualitative generation images using OpenAI
image generation API, then assemble them into a supplementary figure matching
the layout of supplementary_gen_figure_new.pdf.

Usage:
    export OPENAI_API_KEY="sk-..."
    python generate_new_gen_qualitative.py              # generate images + assemble figure
    python generate_new_gen_qualitative.py --assemble   # assemble figure from existing images only
    python generate_new_gen_qualitative.py --prompts 0 2 3  # generate only specific prompt indices

Output:
    figures/new_gen_qualitative_images/   — individual generated images
    figures/supplementary_gen_figure_v2.pdf
    figures/supplementary_gen_figure_v2.png

Notes:
    - These are TEMPORARY placeholders (generated via DALL-E / gpt-image-1).
      Replace with real model outputs once remote machine access is available.
    - Baseline images are generated with deliberately weakened/error-prone prompts
      to simulate typical compositional failures.
    - "Our" images use the full correct prompt.
    - Backbone style hints are added to approximate visual differences between
      BAGEL (warm photorealistic), BLIP3o (sharp, clean), VARGPT (slightly softer).
"""

import os
import sys
import json
import time
import base64
import argparse
from pathlib import Path
from io import BytesIO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT_DIR = Path(__file__).resolve().parent / "new_gen_qualitative_images"
FIG_DIR = Path(__file__).resolve().parent

# Three backbones — each has a style hint appended to prompts
BACKBONES = [
    {
        "name": "BAGEL",
        "label_baseline": "Bagel (Baseline)",
        "label_ours": "Bagel (Our)",
        "style": (
            "Photorealistic, warm color temperature, natural lighting, "
            "slightly soft focus, professional photography style"
        ),
    },
    {
        "name": "BLIP3o",
        "label_baseline": "BLIP-3o (Baseline)",
        "label_ours": "BLIP-3o (Our)",
        "style": (
            "Photorealistic, sharp and clean, high contrast, vivid colors, "
            "studio-quality digital photograph"
        ),
    },
    {
        "name": "VARGPT",
        "label_baseline": "VARGPT-1.1 (Baseline)",
        "label_ours": "VARGPT-1.1 (Our)",
        "style": (
            "Photorealistic but slightly painterly, softer edges, "
            "natural outdoor lighting, subtle film grain texture"
        ),
    },
]

# ---------------------------------------------------------------------------
# Prompt definitions
#
# Each entry has:
#   "display_prompt" — text shown as column header in figure
#   "ours"           — correct prompt (same for all backbones)
#   "baseline"       — dict of backbone_name → deliberately flawed prompt
#                      that produces typical compositional errors
# ---------------------------------------------------------------------------

PROMPTS = [
    {
        "display_prompt": (
            "Three cats sitting in a row\n"
            "on a bench — one orange,\n"
            "one black, one white"
        ),
        "ours": (
            "Three cats sitting in a row on a park bench. "
            "The left cat is orange tabby, the middle cat is solid black, "
            "the right cat is pure white. Daytime, outdoor park setting."
        ),
        "baseline": {
            # Counting error: wrong number; color-swap errors
            "BAGEL": (
                "Two cats sitting on a park bench, one is orange and one has "
                "mixed black and white fur. Daytime, outdoor park setting."
            ),
            # Color attribution error: colors muddled together
            "BLIP3o": (
                "Cats sitting on a park bench. The cats are calico colored "
                "with patches of orange, black, and white. Daytime, outdoor."
            ),
            # Count + position error: too many, not in a row
            "VARGPT": (
                "Four cats playing around a park bench, some orange, "
                "some dark colored. Daytime, outdoor park setting."
            ),
        },
    },
    {
        "display_prompt": (
            "A golden retriever wearing a\n"
            "green bandana, next to a gray\n"
            "cat wearing a red collar"
        ),
        "ours": (
            "A golden retriever dog wearing a bright green bandana around its "
            "neck, sitting next to a gray cat wearing a small red collar. "
            "Both animals are on a living room floor. Indoor setting."
        ),
        "baseline": {
            # Attribute swap: bandana and collar colors reversed
            "BAGEL": (
                "A golden retriever wearing a red bandana around its neck, "
                "next to a gray cat wearing a green collar. "
                "Both on a living room floor. Indoor setting."
            ),
            # Missing accessory: cat has no collar
            "BLIP3o": (
                "A golden retriever wearing a green bandana next to a gray "
                "cat on a living room floor. Indoor setting."
            ),
            # Wrong animal color: brown cat instead of gray
            "VARGPT": (
                "A golden retriever wearing a green scarf next to a brown "
                "tabby cat with a red collar. Living room floor, indoor."
            ),
        },
    },
    {
        "display_prompt": (
            "Exactly five red roses in a\n"
            "clear glass vase on a white\n"
            "marble table"
        ),
        "ours": (
            "Exactly five red roses arranged in a clear glass vase placed on "
            "a white marble table. Clean background, natural window light, "
            "each rose is clearly visible and countable."
        ),
        "baseline": {
            # Wrong count: 3 instead of 5
            "BAGEL": (
                "Three red roses in a glass vase on a white marble table. "
                "Clean background, natural window light."
            ),
            # Wrong count + wrong color: pink roses
            "BLIP3o": (
                "A bouquet of pink roses in a glass vase on a marble table. "
                "Clean background, natural window light."
            ),
            # Wrong count: too many (8+), overcrowded
            "VARGPT": (
                "Many red roses overflowing from a glass vase on a marble "
                "table. Clean background, natural window light."
            ),
        },
    },
    {
        "display_prompt": (
            "A small red ball on top of a\n"
            "large blue box, with a green\n"
            "plant behind the box"
        ),
        "ours": (
            "A small red ball sitting on top of a large blue box. Behind the "
            "box there is a green potted plant. Simple indoor setting with "
            "clean white background. The spatial arrangement is clear: ball "
            "on top of box, plant behind box."
        ),
        "baseline": {
            # Position error: ball beside box instead of on top
            "BAGEL": (
                "A small red ball next to a large blue box. A green potted "
                "plant is beside them. Simple indoor setting, white background."
            ),
            # Color swap: blue ball, red box
            "BLIP3o": (
                "A small blue ball on top of a large red box. A green "
                "potted plant is nearby. Indoor setting, white background."
            ),
            # Missing object: no plant
            "VARGPT": (
                "A red ball and a blue box on the floor. Simple indoor "
                "setting with clean white background."
            ),
        },
    },
]


# ---------------------------------------------------------------------------
# OpenAI image generation
# ---------------------------------------------------------------------------

def generate_image_openai(prompt: str, out_path: Path, model: str = "gpt-image-1",
                          size: str = "1024x1024", quality: str = "medium") -> bool:
    """Generate a single image via OpenAI API and save to out_path."""
    from openai import OpenAI

    client = OpenAI()  # uses OPENAI_API_KEY env var

    try:
        print(f"  Generating: {out_path.name}")
        print(f"    Prompt: {prompt[:90]}...")

        result = client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size=size,
            quality=quality,
        )

        # gpt-image-1 returns base64, dall-e-3 returns URL
        if hasattr(result.data[0], "b64_json") and result.data[0].b64_json:
            img_bytes = base64.b64decode(result.data[0].b64_json)
        elif hasattr(result.data[0], "url") and result.data[0].url:
            import requests
            resp = requests.get(result.data[0].url, timeout=60)
            resp.raise_for_status()
            img_bytes = resp.content
        else:
            print(f"    ERROR: No image data in response")
            return False

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(img_bytes)
        print(f"    Saved: {out_path} ({len(img_bytes)//1024}KB)")
        return True

    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def generate_all_images(prompt_indices=None, model="gpt-image-1",
                        size="1024x1024", quality="medium"):
    """Generate all images for the figure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    indices = prompt_indices or list(range(len(PROMPTS)))
    total = len(indices) * len(BACKBONES) * 2
    generated = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"Generating {total} images ({len(indices)} prompts x "
          f"{len(BACKBONES)} backbones x 2 conditions)")
    print(f"Output directory: {OUT_DIR}")
    print(f"Model: {model} | Size: {size} | Quality: {quality}")
    print(f"{'='*60}\n")

    for pi in indices:
        p = PROMPTS[pi]
        display = p["display_prompt"].replace("\n", " ").strip()
        print(f"\n--- Prompt {pi}: {display[:60]}... ---")

        for bi, bb in enumerate(BACKBONES):
            for condition in ["baseline", "ours"]:
                fname = f"p{pi}_{bb['name'].lower()}_{condition}.png"
                out_path = OUT_DIR / fname

                if out_path.exists():
                    print(f"  SKIP (exists): {fname}")
                    skipped += 1
                    continue

                # Build the full prompt with style hint
                if condition == "ours":
                    base_prompt = p["ours"]
                else:
                    base_prompt = p["baseline"][bb["name"]]

                full_prompt = f"{base_prompt} Style: {bb['style']}."

                ok = generate_image_openai(full_prompt, out_path, model=model,
                                           size=size, quality=quality)
                generated += 1

                if ok:
                    # Rate limiting — be gentle with the API
                    time.sleep(1.0)
                else:
                    print(f"    Will create placeholder for {fname}")
                    _create_placeholder(out_path, condition, bb["name"], pi)

    print(f"\n{'='*60}")
    print(f"Done: {generated} generated, {skipped} skipped (already existed)")
    print(f"{'='*60}\n")


def _create_placeholder(out_path: Path, condition: str, backbone: str, prompt_idx: int):
    """Create a colored placeholder image when API fails."""
    from PIL import Image, ImageDraw, ImageFont

    colors = {
        ("baseline", "BAGEL"):  "#FFE0E0",
        ("baseline", "BLIP3o"): "#FFE0E0",
        ("baseline", "VARGPT"): "#FFE0E0",
        ("ours", "BAGEL"):      "#E0FFE0",
        ("ours", "BLIP3o"):     "#E0FFE0",
        ("ours", "VARGPT"):     "#E0FFE0",
    }
    bg = colors.get((condition, backbone), "#F0F0F0")
    img = Image.new("RGB", (512, 512), bg)
    draw = ImageDraw.Draw(img)
    text = f"PLACEHOLDER\n{backbone}\n{condition}\nPrompt {prompt_idx}"
    draw.multiline_text((256, 256), text, fill="black", anchor="mm")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))


# ---------------------------------------------------------------------------
# Figure assembly — matches supplementary_gen_figure_new.pdf layout
# ---------------------------------------------------------------------------

def assemble_figure():
    """
    Assemble the grid figure from generated images.

    Layout (matching the existing supplementary_gen_figure_new.pdf):
        - Top row:        prompt text headers (4 columns)
        - Row pairs:      Baseline / Our for each backbone
        - Thick black lines separate backbone groups
        - Left side:      rotated row labels in monospace font
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.gridspec import GridSpec
    from PIL import Image

    n_prompts = len(PROMPTS)
    n_backbones = len(BACKBONES)
    n_rows = n_backbones * 2  # baseline + ours per backbone

    # -- Check that all images exist --
    missing = []
    for pi in range(n_prompts):
        for bb in BACKBONES:
            for cond in ["baseline", "ours"]:
                fname = f"p{pi}_{bb['name'].lower()}_{cond}.png"
                fpath = OUT_DIR / fname
                if not fpath.exists():
                    missing.append(fname)

    if missing:
        print(f"WARNING: {len(missing)} images missing. Creating placeholders...")
        for fname in missing:
            parts = fname.replace(".png", "").split("_")
            pi = int(parts[0][1:])
            bb_name = parts[1].upper()
            if bb_name == "BLIP3O":
                bb_name = "BLIP3o"
            elif bb_name == "VARGPT":
                bb_name = "VARGPT"
            cond = parts[2]
            _create_placeholder(OUT_DIR / fname, cond, bb_name, pi)

    # -- Figure dimensions --
    # Match the aspect ratio of the existing figure
    cell_size = 2.2  # inches per image cell
    label_width = 0.9  # left margin for row labels
    header_height = 0.7  # top margin for prompt text
    sep_height = 0.08  # separator line space
    fig_width = label_width + n_prompts * cell_size + 0.3
    fig_height = header_height + n_rows * cell_size + (n_backbones - 1) * sep_height + 0.3

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

    # Normalized coordinates helper
    def norm_x(x_in):
        return x_in / fig_width

    def norm_y(y_in):
        return 1.0 - y_in / fig_height

    # -- Draw prompt headers --
    for pi in range(n_prompts):
        cx = label_width + (pi + 0.5) * cell_size
        fig.text(
            norm_x(cx), norm_y(header_height * 0.42),
            PROMPTS[pi]["display_prompt"].replace("\n", "\n"),
            ha="center", va="center",
            fontsize=9, fontfamily="monospace", fontweight="bold",
            linespacing=1.2,
        )

    # -- Draw images and row labels --
    row_labels = []
    for bb in BACKBONES:
        row_labels.append(bb["label_baseline"])
        row_labels.append(bb["label_ours"])

    for ri in range(n_rows):
        bb_idx = ri // 2
        cond = "baseline" if ri % 2 == 0 else "ours"
        bb = BACKBONES[bb_idx]

        # Y position for this row
        y_top = header_height + ri * cell_size + bb_idx * sep_height

        # Row label (rotated, on the left)
        label_y = y_top + cell_size * 0.5
        fig.text(
            norm_x(label_width * 0.45), norm_y(label_y),
            row_labels[ri],
            ha="center", va="center",
            fontsize=10, fontfamily="monospace", fontweight="bold",
            rotation=90,
        )

        # Images for this row
        for pi in range(n_prompts):
            fname = f"p{pi}_{bb['name'].lower()}_{cond}.png"
            fpath = OUT_DIR / fname

            x_left = label_width + pi * cell_size
            # Add axes for this cell
            ax_rect = [
                norm_x(x_left + 0.05),
                norm_y(y_top + cell_size - 0.05),
                (cell_size - 0.10) / fig_width,
                (cell_size - 0.10) / fig_height,
            ]
            ax = fig.add_axes(ax_rect)

            try:
                img = mpimg.imread(str(fpath))
                ax.imshow(img, aspect="equal")
            except Exception:
                ax.text(0.5, 0.5, "MISSING", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="red")

            ax.set_xticks([])
            ax.set_yticks([])

            # Black border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("black")
                spine.set_linewidth(1.0)

    # -- Draw thick separator lines between backbone groups --
    for sep_idx in range(1, n_backbones):
        y_sep = header_height + sep_idx * 2 * cell_size + (sep_idx - 1) * sep_height + sep_height * 0.5
        fig.add_artist(plt.Line2D(
            [norm_x(label_width - 0.1), norm_x(label_width + n_prompts * cell_size + 0.1)],
            [norm_y(y_sep), norm_y(y_sep)],
            color="black", linewidth=3.0,
            transform=fig.transFigure, clip_on=False,
        ))

    # -- Save --
    for ext in ["pdf", "png"]:
        out_path = FIG_DIR / f"supplementary_gen_figure_v2.{ext}"
        fig.savefig(
            str(out_path),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
        print(f"Saved: {out_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate qualitative generation figure with OpenAI images"
    )
    parser.add_argument(
        "--assemble", action="store_true",
        help="Only assemble the figure from existing images (skip generation)"
    )
    parser.add_argument(
        "--prompts", nargs="*", type=int, default=None,
        help="Generate only specific prompt indices (e.g., --prompts 0 2 3)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-image-1",
        choices=["gpt-image-1", "dall-e-3"],
        help="OpenAI image generation model (default: gpt-image-1)"
    )
    parser.add_argument(
        "--size", type=str, default="1024x1024",
        help="Image size (default: 1024x1024)"
    )
    parser.add_argument(
        "--quality", type=str, default="medium",
        choices=["low", "medium", "high", "standard", "hd"],
        help="Image quality (default: medium)"
    )
    args = parser.parse_args()

    if not args.assemble:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("ERROR: Set OPENAI_API_KEY environment variable first.")
            print("  export OPENAI_API_KEY='sk-...'")
            sys.exit(1)

        generate_all_images(
            prompt_indices=args.prompts,
            model=args.model,
            size=args.size,
            quality=args.quality,
        )

    print("\nAssembling figure...")
    assemble_figure()
    print("\nDone!")


if __name__ == "__main__":
    main()
