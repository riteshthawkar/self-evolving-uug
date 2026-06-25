#!/usr/bin/env python3
"""Plot actual two-stage loss curves from extracted rebuttal logs.

This script is intentionally data-only: it refuses to synthesize curves when no
loss rows are available. Use extract_two_stage_loss_logs.py first, or pass a CSV
created by that script.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle)]


def to_float(value: str) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def rolling_mean(xs: np.ndarray, ys: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(xs) == 0:
        return xs, ys
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    window = max(1, int(window))
    if len(ys) < window:
        return xs, ys
    kernel = np.ones(window, dtype=float) / float(window)
    return xs, np.convolve(ys, kernel, mode="same")


def collect_series(rows: Iterable[Dict[str, str]], loss_name: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    grouped: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row.get("loss_name") != loss_name:
            continue
        step_f = to_float(row.get("step", ""))
        loss_f = to_float(row.get("loss_value", ""))
        role = row.get("role", "").strip() or "unknown"
        did_step = str(row.get("did_step", "")).strip().lower()
        if step_f is None or loss_f is None:
            continue
        if did_step in {"false", "0"}:
            continue
        grouped[role].append((int(step_f), float(loss_f)))
    return {
        role: (np.array([p[0] for p in pairs], dtype=float), np.array([p[1] for p in pairs], dtype=float))
        for role, pairs in grouped.items()
        if pairs
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("docs/Self_Evolving_UUG_manuscript/rebuttal-template/two_stage_loss_logs.csv"),
        help="CSV produced by extract_two_stage_loss_logs.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/Self_Evolving_UUG_manuscript/rebuttal-template/two_stage_loss_curve.pdf"),
    )
    parser.add_argument("--png-output", type=Path, default=None)
    parser.add_argument("--mark-step", type=int, default=10000)
    parser.add_argument("--stage1-steps", type=int, default=10000)
    parser.add_argument("--loss-name", type=str, default="total_loss")
    parser.add_argument("--smooth-window", type=int, default=75)
    parser.add_argument("--title", type=str, default="Two-stage actual loss curve")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    series = collect_series(rows, args.loss_name)
    if not series:
        raise SystemExit(
            f"No actual '{args.loss_name}' rows found in {args.csv}. "
            "Run the two-stage experiment first, then run extract_two_stage_loss_logs.py."
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    colors = {
        "proposer": "#2563EB",
        "solver": "#7C3AED",
        "generator": "#DC2626",
        "dit": "#F97316",
        "unknown": "#111827",
    }
    for role in sorted(series):
        xs, ys = series[role]
        xs_s, ys_s = rolling_mean(xs, ys, args.smooth_window)
        ax.plot(xs_s, ys_s, linewidth=1.8, color=colors.get(role, "#111827"), label=f"{role} {args.loss_name}")

    max_step = max(max(xs) for xs, _ in series.values())
    if args.mark_step <= max_step:
        ax.axvline(args.mark_step, color="#111827", linestyle="--", linewidth=1.1)
        ax.axvspan(args.mark_step, max_step, color="#111827", alpha=0.06, linewidth=0)
        ax.text(
            args.mark_step + 80,
            ax.get_ylim()[1],
            f"{args.mark_step/1000:.0f}k mark",
            va="top",
            ha="left",
            fontsize=8,
            weight="bold",
        )
    else:
        ax.text(
            0.98,
            0.95,
            f"run ends at {max_step/1000:.2f}k\\n{args.mark_step/1000:.0f}k not reached",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.9},
        )
    ax.set_title(args.title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(args.loss_name.replace("_", " "))
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    if args.png_output is None:
        args.png_output = args.output.with_suffix(".png")
    fig.savefig(args.png_output, dpi=300)
    print(args.output)
    print(args.png_output)


if __name__ == "__main__":
    main()
