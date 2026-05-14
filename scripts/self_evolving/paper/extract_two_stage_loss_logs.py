#!/usr/bin/env python3
"""Extract auditable loss logs for the two-stage rebuttal experiment.

The script does not synthesize values. It reads JSONL artifacts produced by
E7_two_stage.sh and writes a flat CSV containing only observed policy/loss
records. If the experiment has not been run yet, the output CSV contains only
the header and the metadata JSON reports zero extracted rows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


CSV_FIELDS = [
    "step",
    "stage",
    "phase",
    "source_log",
    "role",
    "loss_name",
    "loss_value",
    "ce_loss",
    "kl_loss",
    "advantage",
    "reward",
    "kl_coef_before",
    "kl_coef_after",
    "did_step",
    "skipped_reason",
]


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def scalar(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def text(value: Any) -> str:
    return "" if value is None else str(value)


def stage_for_step(step: Optional[int], stage1_steps: int) -> str:
    if step is None:
        return "unknown"
    return "stage1_understanding" if int(step) <= int(stage1_steps) else "stage2_generation"


def normalize_policy_record(
    row: Dict[str, Any],
    *,
    source_log: str,
    stage1_steps: int,
) -> List[Dict[str, Any]]:
    step_value = row.get("step")
    step = int(step_value) if isinstance(step_value, int) else None
    stats = row.get("stats") if isinstance(row.get("stats"), dict) else row
    role = text(row.get("role") or stats.get("role"))
    phase = text(row.get("phase") or row.get("source") or stats.get("phase"))
    skipped_reason = row.get("reason") or row.get("skipped_reason") or stats.get("skipped_reason")

    rows: List[Dict[str, Any]] = []
    for loss_name in ("total_loss", "grpo_loss", "dpo_loss", "ce_loss", "kl_loss"):
        loss_value = scalar(stats.get(loss_name))
        if loss_value is None:
            continue
        rows.append(
            {
                "step": "" if step is None else step,
                "stage": stage_for_step(step, stage1_steps),
                "phase": phase,
                "source_log": source_log,
                "role": role,
                "loss_name": loss_name,
                "loss_value": loss_value,
                "ce_loss": stats.get("ce_loss", ""),
                "kl_loss": stats.get("kl_loss", ""),
                "advantage": stats.get("advantage", stats.get("mean_advantage", "")),
                "reward": stats.get("reward", stats.get("mean_reward", "")),
                "kl_coef_before": stats.get("kl_coef_before", ""),
                "kl_coef_after": stats.get("kl_coef_after", ""),
                "did_step": stats.get("did_step", row.get("did_step", "")),
                "skipped_reason": text(skipped_reason),
            }
        )
    return rows


def normalize_iter_record(
    row: Dict[str, Any],
    *,
    source_log: str,
    stage1_steps: int,
) -> List[Dict[str, Any]]:
    step_value = row.get("step")
    step = int(step_value) if isinstance(step_value, int) else None
    phase = text(row.get("phase"))
    rows: List[Dict[str, Any]] = []

    nested_sources = [
        ("generator", row.get("generator_stats") or row.get("generator_update_stats")),
        ("proposer", row.get("proposer_stats")),
        ("dit", row.get("dit_stats")),
    ]
    for role, maybe_stats in nested_sources:
        if not isinstance(maybe_stats, dict):
            continue
        for loss_name in ("total_loss", "grpo_loss", "dpo_loss", "ce_loss", "kl_loss", "loss"):
            loss_value = scalar(maybe_stats.get(loss_name))
            if loss_value is None:
                continue
            rows.append(
                {
                    "step": "" if step is None else step,
                    "stage": stage_for_step(step, stage1_steps),
                    "phase": phase,
                    "source_log": source_log,
                    "role": role,
                    "loss_name": "dit_loss" if role == "dit" and loss_name == "loss" else loss_name,
                    "loss_value": loss_value,
                    "ce_loss": maybe_stats.get("ce_loss", ""),
                    "kl_loss": maybe_stats.get("kl_loss", ""),
                    "advantage": maybe_stats.get("advantage", maybe_stats.get("mean_advantage", "")),
                    "reward": maybe_stats.get("reward", maybe_stats.get("mean_reward", row.get("best_reward", ""))),
                    "kl_coef_before": maybe_stats.get("kl_coef_before", ""),
                    "kl_coef_after": maybe_stats.get("kl_coef_after", ""),
                    "did_step": maybe_stats.get("did_step", ""),
                    "skipped_reason": text(maybe_stats.get("skipped_reason") or row.get("generator_skipped_reason")),
                }
            )
    return rows


def normalize_metrics_record(
    row: Dict[str, Any],
    *,
    source_log: str,
    stage1_steps: int,
) -> List[Dict[str, Any]]:
    step_value = row.get("step")
    step = int(step_value) if isinstance(step_value, int) else None
    phase = text(row.get("phase") or row.get("kind"))
    metrics = {
        "solver/ce_loss_mean": ("solver", "ce_loss"),
        "solver/kl_loss_mean": ("solver", "kl_loss"),
        "proposer/ce_loss": ("proposer", "ce_loss"),
        "proposer/kl_loss": ("proposer", "kl_loss"),
        "generator/ce_loss": ("generator", "ce_loss"),
        "generator/kl_loss": ("generator", "kl_loss"),
        "generator/dpo_loss": ("generator", "dpo_loss"),
        "dit/loss": ("dit", "dit_loss"),
    }
    rows: List[Dict[str, Any]] = []
    for key, (role, loss_name) in metrics.items():
        loss_value = scalar(row.get(key))
        if loss_value is None:
            continue
        rows.append(
            {
                "step": "" if step is None else step,
                "stage": stage_for_step(step, stage1_steps),
                "phase": phase,
                "source_log": source_log,
                "role": role,
                "loss_name": loss_name,
                "loss_value": loss_value,
                "ce_loss": loss_value if loss_name == "ce_loss" else "",
                "kl_loss": loss_value if loss_name == "kl_loss" else "",
                "advantage": "",
                "reward": "",
                "kl_coef_before": "",
                "kl_coef_after": row.get(f"{role}/kl_coef", ""),
                "did_step": "",
                "skipped_reason": text(row.get("train/dit_skip_reason")),
            }
        )
    return rows


def candidate_logs(run_dir: Path) -> List[Path]:
    if run_dir.is_file():
        return [run_dir]
    names = [
        "logs/policy_updates.jsonl",
        "iter_log.jsonl",
        "metrics.jsonl",
        "logs/rewards.jsonl",
    ]
    logs = [run_dir / name for name in names if (run_dir / name).exists()]
    logs.extend(sorted(run_dir.rglob("policy_updates.jsonl")))
    logs.extend(sorted(run_dir.rglob("iter_log.jsonl")))
    logs.extend(sorted(run_dir.rglob("metrics.jsonl")))
    return sorted({path.resolve() for path in logs})


def extract_rows(run_dir: Path, stage1_steps: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for log_path in candidate_logs(run_dir):
        source_log = str(log_path)
        for record in iter_jsonl(log_path):
            name = log_path.name
            if name == "policy_updates.jsonl":
                rows.extend(
                    normalize_policy_record(record, source_log=source_log, stage1_steps=stage1_steps)
                )
            elif name == "metrics.jsonl":
                rows.extend(
                    normalize_metrics_record(record, source_log=source_log, stage1_steps=stage1_steps)
                )
            else:
                rows.extend(
                    normalize_iter_record(record, source_log=source_log, stage1_steps=stage1_steps)
                )
    rows.sort(key=lambda item: (int(item["step"]) if item["step"] != "" else -1, item["role"], item["loss_name"]))
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/blip3o/E7_two_stage"),
        help="Two-stage run directory or a JSONL log file.",
    )
    parser.add_argument(
        "--stage1-steps",
        type=int,
        default=5000,
        help="Step boundary between understanding-only and generation-only stages.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/Self_Evolving_UUG_manuscript/rebuttal-template/two_stage_loss_logs.csv"),
        help="CSV path to write.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("docs/Self_Evolving_UUG_manuscript/rebuttal-template/two_stage_loss_logs.meta.json"),
        help="Metadata JSON path to write.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    rows = extract_rows(run_dir, int(args.stage1_steps)) if run_dir.exists() else []
    count = write_csv(args.output, rows)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_dir": str(run_dir),
        "stage1_steps": int(args.stage1_steps),
        "output": str(args.output.resolve()),
        "rows_extracted": count,
        "logs_found": [str(path) for path in candidate_logs(run_dir)] if run_dir.exists() else [],
        "note": "Rows are extracted from observed logs only; no values are synthesized.",
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
