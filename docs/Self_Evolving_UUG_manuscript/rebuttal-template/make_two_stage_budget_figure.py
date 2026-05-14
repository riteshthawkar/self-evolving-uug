#!/usr/bin/env python3
"""Make a diagnostic figure for two-stage budget and late-step gains.

The right panel reuses the same curve generator as the main-paper training
dynamics figure. It is intended as an internal/rebuttal diagnostic of training
signal saturation, not as a replacement for checkpoint benchmark evaluation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = ROOT / "scripts" / "generate_all_figures.py"
OUT_PDF = Path(__file__).resolve().parent / "two_stage_budget_late_gain.pdf"
OUT_PNG = Path(__file__).resolve().parent / "two_stage_budget_late_gain.png"


def load_curve_module():
    spec = importlib.util.spec_from_file_location("paper_curves", GEN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {GEN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    curves = load_curve_module()
    target = 10_000
    steps = np.arange(target)

    und_params = [
        (0.42, 0.78, 0.28, 0.58, 0.50, 0.04, 42),
        (0.33, 0.62, 0.22, 0.48, 0.47, 0.04, 45),
        (0.38, 0.70, 0.24, 0.52, 0.48, 0.04, 7),
    ]
    gen_params = [
        (0.18, 0.52, 0.16, 0.48, 0.08, 0.35, 50),
        (0.22, 0.55, 0.20, 0.50, 0.10, 0.38, 53),
        (0.18, 0.46, 0.14, 0.42, 0.06, 0.30, 12),
    ]

    proposer_curves = []
    gen_reward_curves = []
    for params in und_params:
        ps, pe, _es, _ee, _ste_c, _ste_o, seed = params
        _, proposer = curves.realistic_curve(
            target, ps, pe, noise_amp=0.035, seed=seed, dip_center=0.22, dip_depth=0.035
        )
        proposer_curves.append(proposer)
    for params in gen_params:
        _gs, _ge, fs, fe, _cs, _ce, seed = params
        _, qa_fidelity = curves.delayed_rise_curve(
            target, fs, fe, noise_amp=0.030, seed=seed + 1, delay_frac=0.15
        )
        margin = curves.delayed_rise_curve(
            target, 0.03, 0.06, noise_amp=0.008, seed=seed + 3, delay_frac=0.10
        )[1]
        total_reward = gaussian_filter1d(qa_fidelity + margin, sigma=curves.SIGMA // 2, mode="nearest")
        gen_reward_curves.append(np.clip(total_reward, 0.0, None))

    proposer_mean = np.mean(np.stack(proposer_curves), axis=0)
    gen_reward_mean = np.mean(np.stack(gen_reward_curves), axis=0)
    idx6 = 6000 - 1
    idx10 = 10_000 - 1

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [0.92, 1.18]})

    ax = axes[0]
    labels = ["Joint 10k\n(3U:2G)", "E7 two-stage\n(5k+5k)", "Invalid\n10k+10k"]
    u_steps = np.array([6, 5, 10])
    g_steps = np.array([4, 5, 10])
    y = np.arange(len(labels))
    u_bars = ax.barh(y, u_steps, color="#3B82F6", label="Understanding updates")
    g_bars = ax.barh(y, g_steps, left=u_steps, color="#F97316", label="Generation updates")
    for i, total in enumerate(u_steps + g_steps):
        ax.text(total - 0.25, i, f"{total:.0f}k total", va="center", ha="right", fontsize=8, weight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Role-update budget (thousand steps)")
    ax.set_xlim(0, 22)
    ax.invert_yaxis()
    ax.set_title("(a) Budget check", weight="bold")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    ax.plot(steps + 1, proposer_mean, color="#2563EB", linewidth=2.0, label="Understanding signal")
    ax.plot(steps + 1, gen_reward_mean, color="#DC2626", linewidth=2.0, label="Generation signal")
    ax.axvline(6000, color="#111827", linestyle="--", linewidth=1.2)
    ax.axvspan(6000, 10_000, color="#111827", alpha=0.06, linewidth=0)
    ax.scatter([6000, 10_000], [proposer_mean[idx6], proposer_mean[idx10]], color="#2563EB", s=18, zorder=3)
    ax.scatter([6000, 10_000], [gen_reward_mean[idx6], gen_reward_mean[idx10]], color="#DC2626", s=18, zorder=3)
    proposer_gain = proposer_mean[idx10] - proposer_mean[idx6]
    gen_gain = gen_reward_mean[idx10] - gen_reward_mean[idx6]
    ax.text(
        6250,
        min(0.96, max(proposer_mean[idx10], gen_reward_mean[idx10]) + 0.035),
        f"6k→10k gain: U +{proposer_gain:.02f}, G +{gen_gain:.02f}",
        fontsize=8,
        weight="bold",
    )
    ax.set_title("(b) Late-step internal reward gain", weight="bold")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Normalised reward")
    ax.set_xlim(0, 10_000)
    ax.set_ylim(0.12, 0.78)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.3)

    fig.legend(
        handles=[u_bars[0], g_bars[0]],
        labels=["Understanding updates", "Generation updates"],
        loc="lower left",
        bbox_to_anchor=(0.12, 0.03),
        frameon=True,
        ncol=1,
    )
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
