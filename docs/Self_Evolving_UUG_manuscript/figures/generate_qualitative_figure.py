#!/usr/bin/env python3
"""
Generate a combined qualitative figure for the ECCV paper.

Layout (two panels, stacked):
  Panel A – Understanding: Self-Consistency Correction
      4 columns  |  image  +  SC correction box
  Panel B – Generation: QA Fidelity Assessment
      4 columns  |  source / generated images  +  QA verification box
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ──────────────────────────── paths ────────────────────────────
BASE_SRC = Path("/Users/ritesh.thawkar/Ritesh/self-evolving-uug/data/joint_pool_10k/images")
BASE_GEN = Path("/Users/ritesh.thawkar/Ritesh/self-evolving-uug/Bagel/scripts/runs/generated_images")
OUT_DIR  = Path(__file__).resolve().parent

# ──────────────── understanding examples ──────────────────────
UND_EXAMPLES = [
    {
        "title": "(a) Secondary-support grounding",
        "image": BASE_SRC / "vsr/000132.jpg",
        "question": "What is the suitcase lid\nresting on?",
        "greedy": "bed",
        "sc_answer": "table",
        "sc_votes": "6/7",
    },
    {
        "title": "(b) Local relation disambiguation",
        "image": BASE_SRC / "realworldqa/000405.jpg",
        "question": "What is immediately beside\nthe signpost?",
        "greedy": "tree trunk",
        "sc_answer": "rock",
        "sc_votes": "6/7",
    },
    {
        "title": "(c) Fine-grained action parsing",
        "image": BASE_SRC / "open_images/000063.jpg",
        "question": "What is the man at the\ncutting board doing?",
        "greedy": "preparing food",
        "sc_answer": "slicing meat",
        "sc_votes": "5/7",
    },
    {
        "title": "(d) Function hallucination correction",
        "image": BASE_SRC / "flickr30k/000961.jpg",
        "question": "Function of the tree part\nthe child is touching?",
        "greedy": "transport of nutrients",
        "sc_answer": "tree bark",
        "sc_votes": "5/7",
    },
]

# ────────────────── generation examples ───────────────────────
GEN_EXAMPLES = [
    {
        "title": "(e) Text fidelity + object count",
        "prompt": "Lufthansa airplane on tarmac,\nfour engines, blue tail",
        "src":  BASE_SRC / "nocaps/000008.jpg",
        "gen":  BASE_GEN / "step_001884_000008_cand0.png",
        "qa": [
            ("What airline?",      "lufthansa",  "hdasthan",  False),
            ("How many engines?",  "four",       "two",       False),
        ],
    },
    {
        "title": "(f) Count + interaction binding",
        "prompt": "Three people in restaurant,\nwoman pointing at menu",
        "src":  BASE_SRC / "flickr30k/001864.jpg",
        "gen":  BASE_GEN / "step_001865_001864_cand0.png",
        "qa": [
            ("How many people?",  "three",  "two",        False),
            ("Pointing at?",      "menu",   "chandelier", False),
            ("Chair color?",      "red",    "red",        True),
        ],
    },
    {
        "title": "(g) Attribute binding",
        "prompt": "White duck, red face;\nblack dog nearby",
        "src":  BASE_SRC / "flickr30k/000188.jpg",
        "gen":  BASE_GEN / "step_000189_000188_cand0.png",
        "qa": [
            ("Duck's face color?", "red",   "white", False),
            ("Dog color?",         "black", "black", True),
        ],
    },
    {
        "title": "(h) Multi-object counting",
        "prompt": "Three dogs playing with\na ball outdoors",
        "src":  BASE_SRC / "flickr30k/000008.jpg",
        "gen":  BASE_GEN / "step_000009_000008_cand0.png",
        "qa": [
            ("How many dogs?",  "three",           "five",            False),
            ("Activity?",       "playing w/ ball",  "playing w/ ball", True),
        ],
    },
]

# ──────────────────────────── style ────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size":       8,
    "axes.linewidth":  0.5,
})

COLOR_PASS      = "#1b5e20"
COLOR_FAIL      = "#b71c1c"
COLOR_PASS_BG   = "#e8f5e9"
COLOR_FAIL_BG   = "#ffebee"
COLOR_SRC       = "#1565c0"
COLOR_GEN       = "#e65100"
COLOR_UND       = "#1565c0"
COLOR_QA_BG     = "#fafafa"
COLOR_TEXT       = "#37474f"
COLOR_PANEL_HDR = "#263238"
COLOR_PANEL_BG  = "#eceff1"

N = 4  # columns

# ──────────────────────── helper ─────────────────────────────
def add_image(fig, gs_slot, img_path, border_color, label=None):
    ax = fig.add_subplot(gs_slot)
    img = mpimg.imread(str(img_path))
    ax.imshow(img, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2.5)
    if label:
        ax.set_ylabel(label, fontsize=7, fontweight="bold",
                       color=border_color, labelpad=4, rotation=90)
    return ax

# ──────────────────────── figure ─────────────────────────────
# Major rows:
#  0  Understanding panel header
#  1  Understanding images
#  2  Understanding SC boxes
#  3  Generation panel header
#  4  Generation title/prompt
#  5  Source images
#  6  Generated images
#  7  QA verification boxes
fig = plt.figure(figsize=(7.1, 9.0))

outer = gridspec.GridSpec(
    8, N,
    figure=fig,
    height_ratios=[0.06,        # 0: understanding panel header
                   1.0,         # 1: understanding images
                   0.62,        # 2: SC correction boxes
                   0.06,        # 3: generation panel header
                   0.13,        # 4: gen title/prompt
                   0.85,        # 5: source images
                   0.85,        # 6: generated images
                   0.55],       # 7: QA boxes
    hspace=0.10,
    wspace=0.08,
    left=0.06, right=0.98, top=0.98, bottom=0.01,
)

# ═══════════════════ PANEL A: UNDERSTANDING ═══════════════════

# ── Panel header (spans all columns) ──
ax_hdr_u = fig.add_subplot(outer[0, :])
ax_hdr_u.set_axis_off()
# Subtle background bar
bg_bar = FancyBboxPatch(
    (0.0, 0.0), 1.0, 1.0,
    boxstyle="round,pad=0.01",
    facecolor=COLOR_PANEL_BG, edgecolor="none",
    transform=ax_hdr_u.transAxes, clip_on=False,
)
ax_hdr_u.add_patch(bg_bar)
ax_hdr_u.text(0.5, 0.5, "Understanding: Self-Consistency Correction",
               ha="center", va="center", fontsize=9.5, fontweight="bold",
               color=COLOR_PANEL_HDR, transform=ax_hdr_u.transAxes)

for col, ex in enumerate(UND_EXAMPLES):
    # ── Image ──
    add_image(fig, outer[1, col], ex["image"], COLOR_UND,
              "Input Image" if col == 0 else None)

    # ── Title above image ──
    ax_img = fig.axes[-1]
    ax_img.set_title(ex["title"], fontsize=8, fontweight="bold", pad=3)

    # ── SC correction box ──
    ax_sc = fig.add_subplot(outer[2, col])
    ax_sc.set_axis_off()
    ax_sc.set_xlim(0, 1); ax_sc.set_ylim(0, 1)

    # background
    bg = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02",
        facecolor=COLOR_QA_BG, edgecolor="#bdbdbd", linewidth=0.7,
        transform=ax_sc.transAxes, clip_on=False,
    )
    ax_sc.add_patch(bg)

    # question (wrapped with bbox clipping off)
    ax_sc.text(0.08, 0.94, "Q:", fontsize=6, fontweight="bold",
               color=COLOR_TEXT, va="top", transform=ax_sc.transAxes)
    # Wrap question manually if needed
    q_text = ex["question"]
    ax_sc.text(0.17, 0.94, q_text, fontsize=5.5, color=COLOR_TEXT,
               va="top", transform=ax_sc.transAxes, linespacing=1.15,
               wrap=True, clip_on=True)

    # separator
    ax_sc.plot([0.06, 0.94], [0.52, 0.52], color="#bdbdbd", lw=0.4,
               transform=ax_sc.transAxes, clip_on=False)

    # Greedy answer (wrong) ─ FAIL badge
    bw, bh = 0.25, 0.15
    badge_g = FancyBboxPatch(
        (0.04, 0.30), bw, bh,
        boxstyle="round,pad=0.008",
        facecolor=COLOR_FAIL_BG, edgecolor=COLOR_FAIL, linewidth=0.6,
        transform=ax_sc.transAxes, clip_on=False,
    )
    ax_sc.add_patch(badge_g)
    ax_sc.text(0.04 + bw/2, 0.30 + bh/2, "Greedy",
               ha="center", va="center", fontsize=5.5, fontweight="bold",
               color=COLOR_FAIL, transform=ax_sc.transAxes)
    ax_sc.text(0.33, 0.30 + bh/2, ex["greedy"],
               ha="left", va="center", fontsize=6, fontweight="bold",
               color=COLOR_FAIL, fontstyle="italic",
               transform=ax_sc.transAxes)

    # SC answer (correct) ─ SC badge
    badge_s = FancyBboxPatch(
        (0.04, 0.06), bw, bh,
        boxstyle="round,pad=0.008",
        facecolor=COLOR_PASS_BG, edgecolor=COLOR_PASS, linewidth=0.6,
        transform=ax_sc.transAxes, clip_on=False,
    )
    ax_sc.add_patch(badge_s)
    ax_sc.text(0.04 + bw/2, 0.06 + bh/2, f"SC ({ex['sc_votes']})",
               ha="center", va="center", fontsize=5.5, fontweight="bold",
               color=COLOR_PASS, transform=ax_sc.transAxes)
    ax_sc.text(0.33, 0.06 + bh/2, ex["sc_answer"],
               ha="left", va="center", fontsize=6, fontweight="bold",
               color=COLOR_PASS, fontstyle="italic",
               transform=ax_sc.transAxes)


# ═══════════════════ PANEL B: GENERATION ═══════════════════

# ── Panel header ──
ax_hdr_g = fig.add_subplot(outer[3, :])
ax_hdr_g.set_axis_off()
bg_bar2 = FancyBboxPatch(
    (0.0, 0.0), 1.0, 1.0,
    boxstyle="round,pad=0.01",
    facecolor=COLOR_PANEL_BG, edgecolor="none",
    transform=ax_hdr_g.transAxes, clip_on=False,
)
ax_hdr_g.add_patch(bg_bar2)
ax_hdr_g.text(0.5, 0.5, "Generation: QA Fidelity Assessment",
               ha="center", va="center", fontsize=9.5, fontweight="bold",
               color=COLOR_PANEL_HDR, transform=ax_hdr_g.transAxes)

for col, ex in enumerate(GEN_EXAMPLES):
    # ── Title + prompt ──
    ax_t = fig.add_subplot(outer[4, col])
    ax_t.set_axis_off()
    ax_t.text(0.5, 0.95, ex["title"],
              ha="center", va="top", fontsize=8, fontweight="bold",
              transform=ax_t.transAxes)
    ax_t.text(0.5, 0.15, ex["prompt"],
              ha="center", va="top", fontsize=5.5,
              fontstyle="italic", color="#555555",
              transform=ax_t.transAxes, linespacing=1.15)

    # ── Source image ──
    add_image(fig, outer[5, col], ex["src"], COLOR_SRC,
              "Source" if col == 0 else None)

    # ── Generated image ──
    add_image(fig, outer[6, col], ex["gen"], COLOR_GEN,
              "Generated" if col == 0 else None)

    # ── QA verification box ──
    ax_qa = fig.add_subplot(outer[7, col])
    ax_qa.set_axis_off()
    ax_qa.set_xlim(0, 1); ax_qa.set_ylim(0, 1)

    bg = FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.02",
        facecolor=COLOR_QA_BG, edgecolor="#bdbdbd", linewidth=0.7,
        transform=ax_qa.transAxes, clip_on=False,
    )
    ax_qa.add_patch(bg)

    # header
    ax_qa.text(0.50, 0.95, "Solver QA Verification",
               ha="center", va="top", fontsize=6, fontweight="bold",
               color=COLOR_TEXT, transform=ax_qa.transAxes)
    ax_qa.plot([0.06, 0.94], [0.87, 0.87], color="#bdbdbd", lw=0.5,
               transform=ax_qa.transAxes, clip_on=False)

    n_qa = len(ex["qa"])
    usable = 0.82
    row_h  = usable / max(n_qa, 1)
    y_top  = 0.83

    for i, (question, expected, got, passed) in enumerate(ex["qa"]):
        y = y_top - i * row_h
        fg   = COLOR_PASS if passed else COLOR_FAIL
        bg_c = COLOR_PASS_BG if passed else COLOR_FAIL_BG
        tag  = "PASS" if passed else "FAIL"

        bw_qa, bh_qa = 0.16, 0.09
        badge = FancyBboxPatch(
            (0.04, y - bh_qa), bw_qa, bh_qa,
            boxstyle="round,pad=0.008",
            facecolor=bg_c, edgecolor=fg, linewidth=0.6,
            transform=ax_qa.transAxes, clip_on=False,
        )
        ax_qa.add_patch(badge)
        ax_qa.text(0.04 + bw_qa/2, y - bh_qa/2, tag,
                   ha="center", va="center", fontsize=5.5, fontweight="bold",
                   color=fg, transform=ax_qa.transAxes)

        ax_qa.text(0.23, y - 0.01, question,
                   ha="left", va="top", fontsize=5.5,
                   color=COLOR_TEXT, transform=ax_qa.transAxes)

        line2 = f"{expected}  ->  {got}"
        ax_qa.text(0.23, y - 0.01 - row_h * 0.42, line2,
                   ha="left", va="top", fontsize=5,
                   color=fg, fontstyle="italic", fontweight="bold",
                   transform=ax_qa.transAxes)


# ──────────────────────────── save ─────────────────────────────
out_files = [
    OUT_DIR / "qualitative_comparison.pdf",
    OUT_DIR / "qualitative_comparison.png",
    OUT_DIR / "qualitative_figure.pdf",
    OUT_DIR / "qualitative_figure.png",
]

for out_path in out_files:
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

for out_path in out_files:
    print(f"Saved: {out_path}")
