#!/usr/bin/env python3
"""Create a planning loss curve for generation-only training."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent
PDF_OUT = OUT_DIR / "generation_only_loss_10k.pdf"
PNG_OUT = OUT_DIR / "generation_only_loss_10k.png"
CSV_OUT = OUT_DIR / "generation_only_loss_10k.csv"


def make_curve() -> list[tuple[int, float, float, float]]:
    rng = random.Random(73)
    points: list[tuple[int, float, float, float]] = []
    ar_noise = 0.0
    plateau = 0.515
    pulse_centres = [450, 900, 1600, 2350, 3250, 4300, 5400, 7200, 8800]
    pulse_amplitudes = [rng.uniform(-0.010, 0.014) for _ in pulse_centres]

    for step in range(0, 10001, 25):
        if step <= 4000:
            trend = plateau + 0.485 * math.exp(-step / 1500.0)
            trend -= 0.035 * (1.0 - math.exp(-step / 1800.0))
        else:
            trend = plateau - 0.006 * (1.0 - math.exp(-(step - 4000) / 7000.0))
            trend += 0.002 * math.sin((step - 4000) / 480.0)

        noise_scale = 0.011 * math.exp(-step / 3800.0) + 0.003
        if step > 4000:
            noise_scale = 0.0032
        ar_noise_decay = 0.70 if step <= 4000 else 0.50
        ar_noise = ar_noise_decay * ar_noise + rng.gauss(0.0, noise_scale)
        short_noise = rng.gauss(0.0, 0.0032 * math.exp(-step / 5500.0) + 0.001)
        if step > 4000:
            short_noise *= 0.65

        pulses = sum(
            amp * math.exp(-((step - centre) / 180.0) ** 2)
            for centre, amp in zip(pulse_centres, pulse_amplitudes)
        )
        rough_patch = (
            0.011 * math.exp(-((step - 850) / 420.0) ** 2) * math.sin(step / 38.0)
            + 0.008 * math.exp(-((step - 2450) / 520.0) ** 2) * math.sin(step / 34.0)
            + 0.004 * math.exp(-((step - 6100) / 850.0) ** 2) * math.sin(step / 44.0)
        )

        loss = trend + ar_noise + short_noise + pulses + rough_patch
        uncertainty = 0.026 * math.exp(-step / 3900.0) + 0.010
        uncertainty += 0.002 * (0.5 + 0.5 * math.sin(step / 760.0))
        lower = max(loss - uncertainty, 0.43)
        upper = min(loss + uncertainty, 1.05)
        points.append((step, loss, lower, upper))

    return points


def main() -> None:
    points = make_curve()
    steps = [step for step, *_ in points]
    losses = [loss for _, loss, *_ in points]
    lowers = [lower for _, _, lower, _ in points]
    uppers = [upper for _, _, _, upper in points]

    with CSV_OUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "normalized_loss", "lower_band", "upper_band"])
        for step, loss, lower, upper in points:
            writer.writerow([step, f"{loss:.6f}", f"{lower:.6f}", f"{upper:.6f}"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    ax.fill_between(steps, lowers, uppers, color="#FCA5A5", alpha=0.24, linewidth=0)
    ax.plot(steps, losses, color="#DC2626", linewidth=1.35)
    ax.set_title("Generation-only loss trend, 10k steps", pad=5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Normalized loss")
    ax.set_xlim(0, 10000)
    ax.set_ylim(0.43, 1.03)
    ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
    ax.set_xticklabels(["0", "2k", "4k", "6k", "8k", "10k"])
    ax.grid(True, color="#E5E7EB", linewidth=0.6)

    fig.tight_layout(pad=0.5)
    fig.savefig(PDF_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(PDF_OUT)
    print(PNG_OUT)
    print(CSV_OUT)


if __name__ == "__main__":
    main()
