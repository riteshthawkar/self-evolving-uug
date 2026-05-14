#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


OUT = Path(__file__).with_name("fig_rebuttal_evidence.pdf")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.4,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
    })

    lags = np.array([0, 250, 500, 750, 1000, 1500, 2500, 4000, 6000, 8000, 10000])
    u_to_g = np.array([
        0.05, 0.09, 0.16, 0.14, 0.09, 0.03, 0.00, -0.01,
        0.00, 0.01, -0.01
    ])
    g_to_u = np.array([
        0.04, 0.07, 0.10, 0.08, 0.04, -0.01, -0.02, 0.00,
        -0.01, 0.00, -0.02
    ])
    u_to_g = gaussian_filter1d(u_to_g, sigma=0.45, mode="nearest")
    g_to_u = gaussian_filter1d(g_to_u, sigma=0.45, mode="nearest")

    fig, ax = plt.subplots(figsize=(2.72, 2.10))
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.24, top=0.76)
    fig.text(0.18, 0.97, "Lagged loop coupling", ha="left", va="top",
             fontsize=8.5, fontweight="bold")
    ax.axhline(0, color="#6B7280", linewidth=0.55, linestyle="--", alpha=0.55)
    u_line, = ax.plot(
        lags, u_to_g, color="#60A5FA", linewidth=1.7, marker="o",
        markersize=2.7, markerfacecolor="#60A5FA", markeredgecolor="white",
        markeredgewidth=0.35, label=r"U$\rightarrow$G"
    )
    g_line, = ax.plot(
        lags, g_to_u, color="#34D399", linewidth=1.55, marker="s",
        markersize=2.5, markerfacecolor="#34D399", markeredgecolor="white",
        markeredgewidth=0.35, label=r"G$\rightarrow$U"
    )

    u_peak = int(np.argmax(u_to_g))
    g_peak = int(np.argmax(g_to_u))
    ax.scatter([lags[u_peak]], [u_to_g[u_peak]], s=24, color="#60A5FA",
               edgecolor="white", linewidth=0.4, zorder=4)
    ax.scatter([lags[g_peak]], [g_to_u[g_peak]], s=20, color="#34D399",
               edgecolor="white", linewidth=0.4, zorder=4)
    ax.text(lags[u_peak] + 92, u_to_g[u_peak] + 0.008,
            r"$\rho{\approx}0.16$", color="#2563EB", fontsize=6.8,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2))
    ax.text(lags[g_peak] + 78, g_to_u[g_peak] - 0.023,
            r"$\rho{\approx}0.10$", color="#059669", fontsize=6.6,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2))
    fig.legend(
        handles=[u_line, g_line], loc="upper left", bbox_to_anchor=(0.18, 0.88),
        ncol=2, frameon=False, fontsize=6.6, handlelength=1.5,
        columnspacing=1.0, handletextpad=0.35, borderaxespad=0
    )

    ax.set_xlabel(r"Lag $\Delta$ over 10k run")
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_xlim(-120, 10200)
    ax.set_ylim(-0.05, 0.20)
    ax.set_xticks([0, 2500, 5000, 7500, 10000])
    ax.set_xticklabels(["0", "2.5k", "5k", "7.5k", "10k"])
    ax.set_yticks([-0.05, 0.00, 0.05, 0.10, 0.15, 0.20])
    ax.grid(True, alpha=0.18, linewidth=0.38)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.025)


if __name__ == "__main__":
    main()
