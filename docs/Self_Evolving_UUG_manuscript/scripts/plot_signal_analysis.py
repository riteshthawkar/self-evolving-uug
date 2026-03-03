#!/usr/bin/env python3
"""
Plot signal analysis for the self-evolving framework.

Three panels:
  (a) STE distribution shift over training (early / mid / late)
  (b) STE vs self-consistency entropy scatter (complementarity)
  (c) Generation reward components over steps

Handles THREE log formats:
  1. understanding_trainer  — no "phase" key, ~75 fields
     STE proxy: solver_update_scale
  2. generation_trainer     — "phase"="generation", nested best_metrics dict
     Fields: best_metrics.{cycle_score, spec_score, diversity_score, total_reward}
  3. unified_trainer        — mixed phases
     Understanding: proposer_ste_difficulty, proposer_token_entropy_max/mean,
                    solver_update_scale
     Generation: flat best_reward (no best_metrics); use logs/rewards.jsonl fallback

Usage:
    python plot_signal_analysis.py \
        --understanding_log /path/to/understanding/iter_log.jsonl \
        --generation_log    /path/to/generation/iter_log.jsonl \
        --output            ../figures/signal_analysis.pdf
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

EARLY_COLOR  = "#3B82F6"  # blue
MID_COLOR    = "#F59E0B"  # amber
LATE_COLOR   = "#EF4444"  # red


def ema_smooth(values, alpha=0.05):
    smoothed = []
    s = values[0] if values else 0
    for v in values:
        s = alpha * v + (1 - alpha) * s
        smoothed.append(s)
    return smoothed


def load_understanding(path):
    """Load understanding phase data from iter_log.jsonl.

    Handles all three trainer formats:
      - understanding_trainer: no "phase" key, STE via solver_update_scale
      - unified_trainer: "phase"="understanding", STE via proposer_ste_difficulty
                         and proposer_token_entropy_max
      - generation_trainer: skipped (phase="generation" only)
    """
    data = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("skip_reason"):
                continue
            if row.get("phase", "understanding") != "understanding":
                continue

            data["step"].append(row["step"])
            data["entropy_nats"].append(row.get("entropy_nats", 0.0))
            data["majority_fraction"].append(row.get("majority_fraction", 1.0))
            data["solver_top1_prob"].append(row.get("solver_top1_prob", 1.0))
            data["proposer_reward"].append(row.get("proposer_reward", 0.0))

            sr = row.get("solver_rewards_raw", [])
            data["solver_reward_mean"].append(float(np.mean(sr)) if sr else 0.0)

            # STE difficulty: unified_trainer logs proposer_ste_difficulty
            # (quantile-normalized STE); understanding_trainer logs
            # solver_update_scale as a proxy.
            data["ste_difficulty"].append(
                row.get("proposer_ste_difficulty",
                         row.get("solver_update_scale", 0.0))
            )
            # Token-level entropy: unified_trainer logs proposer_token_entropy_max
            data["token_entropy_max"].append(
                row.get("proposer_token_entropy_max", 0.0)
            )
            data["token_entropy_mean"].append(
                row.get("proposer_token_entropy_mean", 0.0)
            )

    return dict(data)


def load_generation(path):
    """Load generation phase data from iter_log.jsonl.

    Handles all three trainer formats:
      - generation_trainer: nested best_metrics dict with sub-scores
      - unified_trainer: flat best_reward (no best_metrics); falls back
        to logs/rewards.jsonl for cycle_score / spec_score
      - understanding_trainer: no generation entries
    """
    data = defaultdict(list)

    # Try logs/rewards.jsonl for richer generation metrics (unified_trainer)
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
            if row.get("phase") != "generation":
                continue

            data["step"].append(row["step"])

            # Handle both formats:
            # 1. generation_trainer: best_metrics is a nested dict
            # 2. unified_trainer: flat best_reward, no best_metrics
            bm = row.get("best_metrics", {})
            # Also try rewards.jsonl for richer data
            rr = gen_rewards_by_step.get(row["step"], {})
            rr_bm = rr.get("best_metrics", {})

            data["total_reward"].append(
                bm.get("total_reward")
                or row.get("best_reward")
                or rr_bm.get("total_reward", 0.0)
            )
            data["cycle_score"].append(
                bm.get("cycle_score",
                        row.get("best_cycle_score",
                                 rr_bm.get("cycle_score", 0.0)))
            )
            data["spec_score"].append(
                bm.get("spec_score",
                        row.get("best_spec_score",
                                 rr_bm.get("spec_score", 0.0)))
            )
            data["diversity_score"].append(
                bm.get("diversity_score",
                        row.get("best_diversity_score",
                                 rr_bm.get("diversity_score", 0.0)))
            )

    return dict(data)


def panel_a_ste_distribution(ax, u_data):
    """(a) STE / entropy distribution shift: early vs mid vs late training.

    Uses proposer_ste_difficulty (unified_trainer) or solver_update_scale
    (understanding_trainer) when available; falls back to entropy_nats
    (self-consistency entropy) for visualization.
    """
    if not u_data or not u_data.get("step"):
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("(a) STE distribution over training", fontweight="bold")
        return

    # Prefer STE difficulty if non-trivial; otherwise fall back to entropy_nats
    ste_vals = np.array(u_data.get("ste_difficulty", []))
    if ste_vals.size > 0 and np.any(ste_vals > 0):
        values = ste_vals
        xlabel = "STE difficulty"
        title = "(a) STE distribution over training"
    else:
        values = np.array(u_data["entropy_nats"])
        xlabel = "Self-consistency entropy (nats)"
        title = "(a) Entropy distribution over training"

    n = len(values)
    third = max(n // 3, 1)

    early = values[:third]
    mid = values[third:2*third]
    late = values[2*third:]

    bins = np.linspace(0, max(values.max(), 0.01), 30)

    ax.hist(early, bins=bins, alpha=0.5, color=EARLY_COLOR,
            label=f"Early (steps 1\u2013{third})", density=True)
    ax.hist(mid, bins=bins, alpha=0.5, color=MID_COLOR,
            label=f"Mid (steps {third+1}\u2013{2*third})", density=True)
    ax.hist(late, bins=bins, alpha=0.5, color=LATE_COLOR,
            label=f"Late (steps {2*third+1}\u2013{n})", density=True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.8)


def panel_b_ste_vs_sc(ax, u_data):
    """(b) STE vs self-consistency scatter showing complementarity.

    X-axis: self-consistency entropy (entropy_nats)
    Y-axis: STE difficulty (proposer_ste_difficulty / solver_update_scale)
    Color: proposer reward
    """
    if not u_data or not u_data.get("step"):
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("(b) STE vs self-consistency", fontweight="bold")
        return

    entropy = np.array(u_data["entropy_nats"])
    ste = np.array(u_data.get("ste_difficulty", [0.0] * len(entropy)))
    rewards = np.array(u_data["proposer_reward"])

    # If STE is all zeros (understanding_trainer without STE), use
    # token_entropy_max as an alternative Y axis
    if not np.any(ste > 0):
        te_max = np.array(u_data.get("token_entropy_max", []))
        if te_max.size > 0 and np.any(te_max > 0):
            ste = te_max
            ylabel = "Token entropy (max)"
        else:
            ylabel = "STE difficulty"
    else:
        ylabel = "STE difficulty"

    sc = ax.scatter(entropy, ste, c=rewards, cmap="RdYlGn", s=12, alpha=0.6,
                    edgecolors="none", vmin=-1, vmax=1)
    plt.colorbar(sc, ax=ax, label="Proposer reward", pad=0.02, aspect=30)

    ax.set_xlabel("Self-consistency entropy (nats)")
    ax.set_ylabel(ylabel)
    ax.set_title("(b) STE vs self-consistency", fontweight="bold")

    # Annotate quadrants with median lines
    if np.any(ste > 0):
        ax.axhline(y=np.median(ste), color="gray", linewidth=0.5, linestyle=":")
    if np.any(entropy > 0):
        ax.axvline(x=np.median(entropy), color="gray", linewidth=0.5, linestyle=":")


def panel_c_gen_rewards(ax, g_data):
    """(c) Generation reward components over training."""
    if not g_data or not g_data.get("step"):
        ax.text(0.5, 0.5, "No generation data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("(c) Generation reward components", fontweight="bold")
        return

    steps = g_data["step"]
    ax.plot(steps, ema_smooth(g_data["total_reward"]),
            color="#DC2626", linewidth=1.5, linestyle="-", label="Total reward")

    # Only plot sub-scores if available (generation_trainer or rewards.jsonl)
    specs = g_data.get("spec_score", [])
    cycles = g_data.get("cycle_score", [])
    if specs and any(v > 0 for v in specs):
        ax.plot(steps, ema_smooth(specs),
                color="#4F46E5", linewidth=1.2, label="QA fidelity")
    if cycles and any(v > 0 for v in cycles):
        ax.plot(steps, ema_smooth(cycles),
                color="#0891B2", linewidth=1.2, label="Cycle consistency")

    # Diversity if non-trivial
    divs = g_data.get("diversity_score", [])
    if divs and any(v > 0 for v in divs):
        ax.plot(steps, ema_smooth(divs),
                color="#059669", linewidth=1.0, linestyle="--", label="Diversity")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Score")
    ax.set_title("(c) Generation reward components", fontweight="bold")
    ax.legend(loc="best", framealpha=0.8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))


def main():
    parser = argparse.ArgumentParser(description="Plot signal analysis")
    parser.add_argument("--understanding_log", type=str, required=True,
                        help="Path to understanding iter_log.jsonl (or unified log)")
    parser.add_argument("--generation_log", type=str, default=None,
                        help="Path to generation iter_log.jsonl (or same unified log)")
    parser.add_argument("--output", type=str, default="../figures/signal_analysis.pdf")
    args = parser.parse_args()

    u_data = load_understanding(args.understanding_log)
    gen_log = args.generation_log or args.understanding_log
    g_data = load_generation(gen_log)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)

    panel_a_ste_distribution(axes[0], u_data)
    panel_b_ste_vs_sc(axes[1], u_data)
    panel_c_gen_rewards(axes[2], g_data)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), format=out.suffix.lstrip("."))
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
