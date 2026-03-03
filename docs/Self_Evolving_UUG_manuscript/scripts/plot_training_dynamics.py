#!/usr/bin/env python3
"""
Plot training dynamics: understanding and generation metrics over training steps.

Expected layout: 2-row x 3-col grid.
  Top row:    understanding metrics per backbone (BLIP3o, BAGEL, VARGPT)
  Bottom row: generation metrics per backbone

Handles THREE log formats:
  1. understanding_trainer  — no "phase" key, ~75 fields
  2. generation_trainer     — "phase"="generation", best_metrics nested dict
  3. unified_trainer        — mixed phases; understanding has proposer_ste_difficulty,
     generation has flat best_reward (no best_metrics)

Usage:
    python plot_training_dynamics.py \
        --blip3o_log  /path/to/blip3o/iter_log.jsonl \
        --bagel_log   /path/to/bagel/iter_log.jsonl \
        --vargpt_log  /path/to/vargpt/iter_log.jsonl \
        --output      ../figures/training_dynamics.pdf
"""

import argparse
import json
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

COLORS = {
    "proposer_reward": "#2563EB",
    "solver_reward":   "#059669",
    "entropy":         "#D97706",
    "ste":             "#7C3AED",
    "gen_total":       "#DC2626",
    "cycle":           "#0891B2",
    "fidelity":        "#4F46E5",
}


def ema_smooth(values, alpha=0.05):
    """Exponential moving average smoothing."""
    if not values:
        return values
    smoothed = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        smoothed.append(s)
    return smoothed


def load_log(path):
    """Load iter_log.jsonl and split by phase.

    Handles all three trainer formats:
      - understanding_trainer: no "phase" key
      - generation_trainer: "phase"="generation", best_metrics dict
      - unified_trainer: mixed, generation has flat best_reward
    """
    understanding = defaultdict(list)
    generation = defaultdict(list)

    # Also try logs/rewards.jsonl for richer generation metrics
    log_path = pathlib.Path(path)
    rewards_log = log_path.parent / "logs" / "rewards.jsonl"
    gen_rewards_by_step = {}
    if rewards_log.exists():
        with open(rewards_log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                gen_rewards_by_step[row.get("step")] = row

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("skip_reason"):
                continue

            phase = row.get("phase", "understanding")

            if phase == "understanding":
                understanding["step"].append(row["step"])
                understanding["proposer_reward"].append(row.get("proposer_reward", 0.0))
                sr = row.get("solver_rewards_raw", [])
                understanding["solver_reward_mean"].append(
                    float(np.mean(sr)) if sr else 0.0
                )
                understanding["entropy_nats"].append(row.get("entropy_nats", 0.0))
                understanding["majority_fraction"].append(row.get("majority_fraction", 1.0))
                # STE difficulty (unified_trainer) or solver_update_scale (standalone)
                understanding["ste_difficulty"].append(
                    row.get("proposer_ste_difficulty", row.get("solver_update_scale", 0.0))
                )

            elif phase == "generation":
                generation["step"].append(row["step"])
                # Handle both formats
                bm = row.get("best_metrics", {})
                rr = gen_rewards_by_step.get(row["step"], {})
                rr_bm = rr.get("best_metrics", {})

                generation["total_reward"].append(
                    bm.get("total_reward")
                    or row.get("best_reward")
                    or rr_bm.get("total_reward", 0.0)
                )
                generation["cycle_score"].append(
                    bm.get("cycle_score", rr_bm.get("cycle_score", 0.0))
                )
                generation["spec_score"].append(
                    bm.get("spec_score", rr_bm.get("spec_score", 0.0))
                )

    return dict(understanding), dict(generation)


def plot_understanding(ax, data, title, show_ylabel=True):
    """Plot understanding metrics on a single axis."""
    if not data or not data.get("step"):
        ax.text(0.5, 0.5, "No understanding data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_title(title, fontweight="bold")
        return

    steps = data["step"]

    ax.plot(steps, ema_smooth(data["proposer_reward"]),
            color=COLORS["proposer_reward"], linewidth=1.2, label="Proposer reward")
    ax.plot(steps, ema_smooth(data["solver_reward_mean"]),
            color=COLORS["solver_reward"], linewidth=1.2, label="Solver reward (mean)")
    ax.plot(steps, ema_smooth(data["entropy_nats"]),
            color=COLORS["entropy"], linewidth=1.2, linestyle="--", label="SC entropy (nats)")

    # Plot STE if non-trivial
    ste = data.get("ste_difficulty", [])
    if ste and any(v > 0 for v in ste):
        ax.plot(steps, ema_smooth(ste),
                color=COLORS["ste"], linewidth=1.0, linestyle=":", label="STE difficulty")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Training step")
    if show_ylabel:
        ax.set_ylabel("Understanding metrics")
    ax.legend(loc="best", framealpha=0.8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))


def plot_generation(ax, data, title, show_ylabel=True):
    """Plot generation metrics on a single axis."""
    if not data or not data.get("step"):
        ax.text(0.5, 0.5, "No generation data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_title(title, fontweight="bold")
        return

    steps = data["step"]

    ax.plot(steps, ema_smooth(data["total_reward"]),
            color=COLORS["gen_total"], linewidth=1.5, label="Total reward")

    # Only plot sub-scores if available (generation_trainer format)
    cycles = data.get("cycle_score", [])
    specs = data.get("spec_score", [])
    if cycles and any(v > 0 for v in cycles):
        ax.plot(steps, ema_smooth(cycles),
                color=COLORS["cycle"], linewidth=1.2, label="Cycle consistency")
    if specs and any(v > 0 for v in specs):
        ax.plot(steps, ema_smooth(specs),
                color=COLORS["fidelity"], linewidth=1.2, linestyle="--", label="QA fidelity")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Training step")
    if show_ylabel:
        ax.set_ylabel("Generation metrics")
    ax.legend(loc="best", framealpha=0.8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))


def main():
    parser = argparse.ArgumentParser(description="Plot training dynamics across backbones")
    parser.add_argument("--blip3o_log", type=str, default=None, help="BLIP3o iter_log.jsonl")
    parser.add_argument("--bagel_log", type=str, default=None, help="BAGEL iter_log.jsonl")
    parser.add_argument("--vargpt_log", type=str, default=None, help="VARGPT iter_log.jsonl")
    parser.add_argument("--output", type=str, default="../figures/training_dynamics.pdf")
    args = parser.parse_args()

    backbones = [
        ("BLIP3o-8B", args.blip3o_log),
        ("BAGEL", args.bagel_log),
        ("VARGPT", args.vargpt_log),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), constrained_layout=True)

    for col, (name, logpath) in enumerate(backbones):
        if logpath and pathlib.Path(logpath).exists():
            u_data, g_data = load_log(logpath)
        else:
            u_data, g_data = {}, {}

        plot_understanding(axes[0, col], u_data, name, show_ylabel=(col == 0))
        plot_generation(axes[1, col], g_data, name, show_ylabel=(col == 0))

    fig.suptitle("Training Dynamics Across Three Backbones", fontsize=13, fontweight="bold", y=1.02)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), format=out.suffix.lstrip("."))
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
