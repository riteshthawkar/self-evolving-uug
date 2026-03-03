#!/usr/bin/env python3
"""
Compose a qualitative generation comparison figure.

Layout: N prompts, each with base model image (left) vs ours (right).
Prompt text shown above each pair.

Usage:
    python plot_qualitative.py \
        --config qualitative_config.json \
        --output ../figures/qualitative.pdf

Config JSON format:
[
    {
        "prompt": "A red bicycle leaning against a blue wall",
        "base_image": "/path/to/base_output.png",
        "ours_image": "/path/to/ours_output.png"
    },
    ...
]
"""

import argparse
import json
import pathlib
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_image(path):
    try:
        return mpimg.imread(path)
    except Exception:
        # Return a gray placeholder
        import numpy as np
        return np.ones((256, 256, 3), dtype=np.float32) * 0.85


def main():
    parser = argparse.ArgumentParser(description="Compose qualitative comparison figure")
    parser.add_argument("--config", type=str, required=True,
                        help="JSON config listing prompts and image paths")
    parser.add_argument("--output", type=str, default="../figures/qualitative.pdf")
    parser.add_argument("--max_prompts", type=int, default=4,
                        help="Maximum number of prompts to show")
    args = parser.parse_args()

    with open(args.config) as f:
        entries = json.load(f)

    entries = entries[:args.max_prompts]
    n = len(entries)

    fig, axes = plt.subplots(1, 2 * n, figsize=(3.2 * n, 3.6), constrained_layout=True)

    if n == 1:
        axes = [axes]  # normalize

    for i, entry in enumerate(entries):
        prompt = entry["prompt"]
        base_img = load_image(entry["base_image"])
        ours_img = load_image(entry["ours_image"])

        ax_base = axes[2 * i]
        ax_ours = axes[2 * i + 1]

        ax_base.imshow(base_img)
        ax_base.set_xticks([])
        ax_base.set_yticks([])
        ax_base.set_xlabel("Base", fontsize=9, fontweight="bold")

        ax_ours.imshow(ours_img)
        ax_ours.set_xticks([])
        ax_ours.set_yticks([])
        ax_ours.set_xlabel("Ours", fontsize=9, fontweight="bold")

        # Prompt title spanning both columns
        wrapped = "\n".join(textwrap.wrap(f'"{prompt}"', width=40))
        mid_x = (ax_base.get_position().x0 + ax_ours.get_position().x1) / 2
        fig.text(mid_x, ax_base.get_position().y1 + 0.02, wrapped,
                 ha="center", va="bottom", fontsize=8, fontstyle="italic",
                 transform=fig.transFigure)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), format=out.suffix.lstrip("."))
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
