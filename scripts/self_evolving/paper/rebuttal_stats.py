#!/usr/bin/env python3
"""Summarize rebuttal-facing run statistics from self-evolving logs.

The script extracts quantities reviewers can audit directly from JSONL logs:
answer length distributions for the STE concern, phase mix, fatal errors, and
observed/projected wall-clock and GPU-hour cost. It accepts explicit log paths
or auto-discovers run directories under common output roots.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


LOG_NAMES = ("iter_log.jsonl", "metrics.jsonl")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    xs = sorted(float(v) for v in values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summarize_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None}
    xs = [float(v) for v in values]
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
    }


def answer_lengths_from_row(row: Dict[str, Any]) -> List[int]:
    lengths: List[int] = []
    answer_fields = (
        "solver_answers_raw",
        "solver_answers",
        "answers",
        "candidate_answers",
        "sc_answers",
    )
    scalar_fields = (
        "majority_answer_raw",
        "majority_answer",
        "answer",
        "solver_answer",
    )

    for key in answer_fields:
        value = row.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    lengths.append(len(item.split()))
                elif isinstance(item, dict):
                    text = item.get("answer") or item.get("text") or item.get("raw")
                    if isinstance(text, str) and text.strip():
                        lengths.append(len(text.split()))

    for key in scalar_fields:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            lengths.append(len(value.split()))

    nested = row.get("solver")
    if isinstance(nested, dict):
        for key in scalar_fields:
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                lengths.append(len(value.split()))
    return lengths


def find_log_file(path: Path) -> Optional[Path]:
    if path.is_file():
        return path
    for name in LOG_NAMES:
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def discover_logs(root: Path) -> List[Path]:
    search_roots = [root / "runs" / "final", root / "outputs"]
    logs: List[Path] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for name in LOG_NAMES:
            logs.extend(search_root.rglob(name))
    return sorted({p.resolve() for p in logs})


def summarize_log(path: Path, target_steps: int, gpus_per_run: int) -> Dict[str, Any]:
    phase_counts: Counter[str] = Counter()
    step_times: List[float] = []
    reward_values: List[float] = []
    answer_lengths: List[int] = []
    fatal_errors: List[str] = []
    rows = 0
    max_step = 0

    for row in iter_jsonl(path):
        rows += 1
        step = row.get("step")
        if isinstance(step, int):
            max_step = max(max_step, step)
        phase_counts[str(row.get("phase") or row.get("kind") or "unknown")] += 1
        step_time = row.get("step_time_sec")
        if isinstance(step_time, (int, float)) and step_time >= 0:
            step_times.append(float(step_time))
        for key in ("best_reward", "reward", "mean_reward", "generation_reward"):
            value = row.get(key)
            if isinstance(value, (int, float)):
                reward_values.append(float(value))
                break
        if row.get("kind") == "fatal_error" or row.get("error"):
            fatal_errors.append(str(row.get("error", ""))[:240])
        answer_lengths.extend(answer_lengths_from_row(row))

    observed_wall_hours = sum(step_times) / 3600.0 if step_times else None
    mean_step_time = statistics.fmean(step_times) if step_times else None
    projected_wall_hours = mean_step_time * target_steps / 3600.0 if mean_step_time else None
    projected_gpu_hours = projected_wall_hours * gpus_per_run if projected_wall_hours else None

    length_summary = summarize_values(answer_lengths)
    reward_summary = summarize_values(reward_values)
    step_time_summary = summarize_values(step_times)
    pct_le_5 = sum(1 for x in answer_lengths if x <= 5) / len(answer_lengths) if answer_lengths else None
    pct_le_10 = sum(1 for x in answer_lengths if x <= 10) / len(answer_lengths) if answer_lengths else None

    return {
        "path": str(path),
        "run_dir": str(path.parent),
        "rows": rows,
        "max_step": max_step,
        "target_steps": target_steps,
        "completion_fraction": (max_step / target_steps) if target_steps else None,
        "phase_counts": dict(sorted(phase_counts.items())),
        "step_time_sec": step_time_summary,
        "observed_wall_hours": observed_wall_hours,
        "projected_wall_hours": projected_wall_hours,
        "projected_gpu_hours": projected_gpu_hours,
        "answer_lengths": {
            **length_summary,
            "pct_le_5": pct_le_5,
            "pct_le_10": pct_le_10,
        },
        "generation_rewards": reward_summary,
        "fatal_errors": fatal_errors[:5],
    }


def aggregate(summaries: Sequence[Dict[str, Any]], target_steps: int, gpus_per_run: int) -> Dict[str, Any]:
    total_rows = sum(int(s["rows"]) for s in summaries)
    max_step = max((int(s["max_step"]) for s in summaries), default=0)
    projected_gpu_hours = [s["projected_gpu_hours"] for s in summaries if s.get("projected_gpu_hours")]
    projected_wall_hours = [s["projected_wall_hours"] for s in summaries if s.get("projected_wall_hours")]
    fatal_errors = [err for s in summaries for err in s.get("fatal_errors", [])]

    return {
        "num_logs": len(summaries),
        "total_rows": total_rows,
        "max_step_seen": max_step,
        "target_steps": target_steps,
        "gpus_per_run": gpus_per_run,
        "projected_wall_hours": summarize_values(projected_wall_hours),
        "projected_gpu_hours": summarize_values(projected_gpu_hours),
        "fatal_error_count": len(fatal_errors),
    }


def fmt_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def format_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    agg = report["aggregate"]
    lines.append("# Rebuttal Run Statistics")
    lines.append("")
    lines.append(
        f"Logs scanned: {agg['num_logs']} | rows: {agg['total_rows']} | "
        f"max step seen: {agg['max_step_seen']}/{agg['target_steps']}"
    )
    lines.append(
        "Projected compute per run from observed step times: "
        f"median {fmt_float(agg['projected_wall_hours']['median'], 1)} wall-h, "
        f"median {fmt_float(agg['projected_gpu_hours']['median'], 1)} GPU-h "
        f"({agg['gpus_per_run']} GPUs/run)."
    )
    if agg["fatal_error_count"]:
        lines.append(f"Fatal/error rows observed: {agg['fatal_error_count']}.")
    lines.append("")
    lines.append("| Run | Step | Answer n | Median words | P90 | <=5 words | <=10 words | Projected GPU-h | Status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for item in report["runs"]:
        ans = item["answer_lengths"]
        status = "failed" if item["fatal_errors"] else ("partial" if item["max_step"] < item["target_steps"] else "complete")
        lines.append(
            f"| `{Path(item['run_dir']).name}` "
            f"| {item['max_step']}/{item['target_steps']} "
            f"| {ans['n']} "
            f"| {fmt_float(ans['median'], 1)} "
            f"| {fmt_float(ans['p90'], 1)} "
            f"| {fmt_float(ans['pct_le_5'] * 100 if ans['pct_le_5'] is not None else None, 1)}% "
            f"| {fmt_float(ans['pct_le_10'] * 100 if ans['pct_le_10'] is not None else None, 1)}% "
            f"| {fmt_float(item['projected_gpu_hours'], 1)} "
            f"| {status} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", type=Path, help="JSONL files or run directories. Auto-discovers if omitted.")
    parser.add_argument("--target-steps", type=int, default=10000)
    parser.add_argument("--gpus-per-run", type=int, default=8)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--repo-root", type=Path, default=root)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = args.logs or discover_logs(args.repo_root)
    log_files: List[Path] = []
    for item in inputs:
        found = find_log_file(item.expanduser())
        if found is not None:
            log_files.append(found.resolve())
    log_files = sorted({p for p in log_files})

    summaries = [summarize_log(p, args.target_steps, args.gpus_per_run) for p in log_files]
    report = {
        "repo_root": str(args.repo_root.resolve()),
        "aggregate": aggregate(summaries, args.target_steps, args.gpus_per_run),
        "runs": summaries,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    markdown = format_markdown(report)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown + "\n", encoding="utf-8")

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
