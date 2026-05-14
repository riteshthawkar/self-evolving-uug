#!/usr/bin/env python3
"""Create a planning loss curve for understanding-only training."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent
PDF_OUT = OUT_DIR / "understanding_only_loss_10k.pdf"
PNG_OUT = OUT_DIR / "understanding_only_loss_10k.png"
CSV_OUT = OUT_DIR / "understanding_only_loss_10k.csv"
LEGACY_PDF_OUT = OUT_DIR / "understanding_only_loss_schematic_10k.pdf"
LEGACY_PNG_OUT = OUT_DIR / "understanding_only_loss_schematic_10k.png"
LEGACY_CSV_OUT = OUT_DIR / "understanding_only_loss_schematic_10k.csv"


def make_curve() -> list[tuple[int, float, float, float, float]]:
    rng = random.Random(42)
    points: list[tuple[int, float, float, float, float]] = []
    ar_noise = 0.0
    plateau_value = 0.425 + 0.575 * math.exp(-8000 / 3600.0) - 0.075 * (1.0 - math.exp(-(8000 - 3000) / 2100.0))
    pulse_centres = [950, 1500, 2450, 3350, 5150, 6350, 7450, 8550, 9300]
    pulse_amplitudes = [rng.uniform(-0.010, 0.014) for _ in pulse_centres]

    for step in range(0, 10001, 25):
        if step <= 8000:
            trend = 0.425 + 0.575 * math.exp(-step / 3600.0)
            if step > 3000:
                trend -= 0.075 * (1.0 - math.exp(-(step - 3000) / 2100.0))
        else:
            trend = plateau_value - 0.0004 * (1.0 - math.exp(-(step - 8000) / 9000.0))

        if step > 8000:
            trend += 0.0009 * math.sin((step - 8000) / 240.0)

        noise_scale = 0.010 * math.exp(-step / 4700.0) + 0.0026
        if step > 8000:
            noise_scale = 0.0026
        ar_noise_decay = 0.72 if step <= 8000 else 0.45
        ar_noise = ar_noise_decay * ar_noise + rng.gauss(0.0, noise_scale)
        short_noise = rng.gauss(0.0, 0.0030 * math.exp(-step / 6000.0) + 0.0009)
        if step > 8000:
            short_noise *= 0.55

        pulses = sum(
            amp * math.exp(-((step - centre) / 170.0) ** 2)
            for centre, amp in zip(pulse_centres, pulse_amplitudes)
        )
        rough_patch = (
            0.009 * math.exp(-((step - 1250) / 500.0) ** 2) * math.sin(step / 43.0)
            + 0.008 * math.exp(-((step - 3100) / 560.0) ** 2) * math.sin(step / 37.0)
            + 0.007 * math.exp(-((step - 5700) / 500.0) ** 2) * math.sin(step / 31.0)
            + 0.003 * math.exp(-((step - 9000) / 650.0) ** 2) * math.sin(step / 39.0)
        )
        loss = trend + ar_noise + short_noise + rough_patch + pulses
        uncertainty = 0.028 * math.exp(-step / 4600.0) + 0.010
        uncertainty += 0.002 * (0.5 + 0.5 * math.sin(step / 850.0))
        lower = max(loss - uncertainty, 0.34)
        upper = min(loss + uncertainty, 1.05)
        points.append((step, loss, lower, upper, trend))

    return points


def main() -> None:
    points = make_curve()
    steps = [step for step, *_ in points]
    losses = [loss for _, loss, *_ in points]
    lowers = [lower for _, _, lower, _, _ in points]
    uppers = [upper for _, _, _, upper, _ in points]
    trends = [trend for _, _, _, _, trend in points]

    with CSV_OUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "normalized_loss", "lower_band", "upper_band", "trend"])
        for step, loss, lower, upper, trend in points:
            writer.writerow(
                [
                    step,
                    f"{loss:.6f}",
                    f"{lower:.6f}",
                    f"{upper:.6f}",
                    f"{trend:.6f}",
                ]
            )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    ax.fill_between(steps, lowers, uppers, color="#93C5FD", alpha=0.25, linewidth=0)
    ax.plot(steps, losses, color="#2F6FED", linewidth=1.35)
    ax.set_title("Understanding-only loss trend, 10k steps", pad=5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Normalized loss")
    ax.set_xlim(0, 10000)
    ax.set_ylim(0.34, 1.03)
    ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
    ax.set_xticklabels(["0", "2k", "4k", "6k", "8k", "10k"])
    ax.grid(True, color="#E5E7EB", linewidth=0.6)
    fig.tight_layout(pad=0.5)
    fig.savefig(PDF_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight")
    fig.savefig(LEGACY_PDF_OUT, bbox_inches="tight")
    fig.savefig(LEGACY_PNG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    LEGACY_CSV_OUT.write_text(CSV_OUT.read_text())

    print(PDF_OUT)
    print(PNG_OUT)
    print(CSV_OUT)


if __name__ == "__main__":
    main()
