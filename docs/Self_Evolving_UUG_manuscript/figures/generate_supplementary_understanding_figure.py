#!/usr/bin/env python3
"""
Generate the supplementary understanding qualitative figure in the same visual
style as the main-paper qualitative figure.

Edit the `UND_EXAMPLES` entries below to update:
  - image path
  - panel title
  - question
  - displayed answer
  - answer label (e.g. "Estimated", "Greedy", "SC (6/7)")

Outputs:
  - supplementary_understanding_figure.pdf
  - supplementary_understanding_figure.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
BASE = OUT_DIR / "qualitative_samples" / "understanding_new"


UND_EXAMPLES = [
    {
        "title": "(a) OCR section grounding",
        "image": BASE / "a_ocr_section__value_propositions.jpg",
        "question": "Which section is the\nhand writing in?",
        "answer_label": "Estimated",
        "answer": "Value Propositions",
    },
    {
        "title": "(b) Vehicle-scene relation",
        "image": BASE / "b_vehicle_relation__tricycle.jpg",
        "question": "What vehicle is carrying\nthe blue barrel?",
        "answer_label": "Estimated",
        "answer": "tricycle",
    },
    {
        "title": "(c) Local attribute binding",
        "image": BASE / "c_local_attribute__butterfly_cushion.jpg",
        "question": "What insect is printed on\nthe red cushion beside the cat?",
        "answer_label": "Estimated",
        "answer": "butterfly",
    },
    {
        "title": "(d) Text grounding via athlete ID",
        "image": BASE / "d_text_grounding__bib_911.jpg",
        "question": "What number is printed on\nthe front skier's bib?",
        "answer_label": "Estimated",
        "answer": "911",
    },
]


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 8,
    "axes.linewidth": 0.5,
})

COLOR_PASS = "#1b5e20"
COLOR_PASS_BG = "#e8f5e9"
COLOR_UND = "#1565c0"
COLOR_QA_BG = "#fafafa"
COLOR_TEXT = "#37474f"
COLOR_PANEL_HDR = "#263238"
COLOR_PANEL_BG = "#eceff1"


def add_image(fig, gs_slot, img_path, border_color, label=None):
    ax = fig.add_subplot(gs_slot)
    img = mpimg.imread(str(img_path))
    ax.imshow(img, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2.5)
    if label:
        ax.set_ylabel(
            label,
            fontsize=7,
            fontweight="bold",
            color=border_color,
            labelpad=4,
            rotation=90,
        )
    return ax


def main():
    fig = plt.figure(figsize=(7.1, 4.6))
    outer = gridspec.GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[0.08, 1.0, 0.62],
        hspace=0.10,
        wspace=0.08,
        left=0.06,
        right=0.98,
        top=0.98,
        bottom=0.03,
    )

    ax_hdr = fig.add_subplot(outer[0, :])
    ax_hdr.set_axis_off()
    bg_bar = FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        boxstyle="round,pad=0.01",
        facecolor=COLOR_PANEL_BG,
        edgecolor="none",
        transform=ax_hdr.transAxes,
        clip_on=False,
    )
    ax_hdr.add_patch(bg_bar)
    ax_hdr.text(
        0.5,
        0.5,
        "Understanding: Local Non-Salient Evidence",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=COLOR_PANEL_HDR,
        transform=ax_hdr.transAxes,
    )

    for col, ex in enumerate(UND_EXAMPLES):
        add_image(fig, outer[1, col], ex["image"], COLOR_UND, "Input Image" if col == 0 else None)
        ax_img = fig.axes[-1]
        ax_img.set_title(ex["title"], fontsize=8, fontweight="bold", pad=3)

        ax_box = fig.add_subplot(outer[2, col])
        ax_box.set_axis_off()
        ax_box.set_xlim(0, 1)
        ax_box.set_ylim(0, 1)

        bg = FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            boxstyle="round,pad=0.02",
            facecolor=COLOR_QA_BG,
            edgecolor="#bdbdbd",
            linewidth=0.7,
            transform=ax_box.transAxes,
            clip_on=False,
        )
        ax_box.add_patch(bg)

        ax_box.text(
            0.08,
            0.94,
            "Q:",
            fontsize=6,
            fontweight="bold",
            color=COLOR_TEXT,
            va="top",
            transform=ax_box.transAxes,
        )
        ax_box.text(
            0.17,
            0.94,
            ex["question"],
            fontsize=5.5,
            color=COLOR_TEXT,
            va="top",
            transform=ax_box.transAxes,
            linespacing=1.15,
            wrap=True,
            clip_on=True,
        )

        ax_box.plot(
            [0.06, 0.94],
            [0.46, 0.46],
            color="#bdbdbd",
            lw=0.4,
            transform=ax_box.transAxes,
            clip_on=False,
        )

        bw, bh = 0.28, 0.16
        badge = FancyBboxPatch(
            (0.04, 0.12),
            bw,
            bh,
            boxstyle="round,pad=0.008",
            facecolor=COLOR_PASS_BG,
            edgecolor=COLOR_PASS,
            linewidth=0.6,
            transform=ax_box.transAxes,
            clip_on=False,
        )
        ax_box.add_patch(badge)
        ax_box.text(
            0.04 + bw / 2,
            0.12 + bh / 2,
            ex["answer_label"],
            ha="center",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color=COLOR_PASS,
            transform=ax_box.transAxes,
        )
        ax_box.text(
            0.37,
            0.12 + bh / 2,
            ex["answer"],
            ha="left",
            va="center",
            fontsize=6,
            fontweight="bold",
            color=COLOR_PASS,
            fontstyle="italic",
            transform=ax_box.transAxes,
        )

    for out_path in [
        OUT_DIR / "supplementary_understanding_figure.pdf",
        OUT_DIR / "supplementary_understanding_figure.png",
    ]:
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight", pad_inches=0.04)
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
