#!/usr/bin/env python3
"""Audit whether the paper/rebuttal experiment setup is runnable and complete.

This script intentionally checks readiness, not scientific correctness of the
reported numbers. It validates the protocol assets that reviewers asked about:
the 6k data pool, launch scripts, environment requirements, existing run
progress, failed runs, and the rebuttal evidence hooks.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
ARTIFACT_NAMES = {
    "config.json",
    "status.json",
    "summary.json",
    "ablation_summary.json",
    "iter_log.jsonl",
    "metrics.jsonl",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel_or_abs(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (root / p)


def count_images(path: Path) -> Tuple[int, Dict[str, int]]:
    if not path.exists():
        return 0, {}
    counts: Counter[str] = Counter()
    total = 0
    for f in path.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        total += 1
        try:
            source = f.relative_to(path).parts[0]
        except Exception:
            source = "."
        counts[source] += 1
    return total, dict(sorted(counts.items()))


def bounded_walk_dirs(root: Path, max_depth: int = 3) -> Iterable[Path]:
    if not root.exists():
        return
    base_depth = len(root.parts)
    skip = {"images", "generated_images", "generated", "cache", "__pycache__"}
    for current, dirs, files in os.walk(root):
        p = Path(current)
        depth = len(p.parts) - base_depth
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        if depth >= max_depth:
            dirs[:] = []
        if any(name in ARTIFACT_NAMES for name in files):
            yield p


def find_artifact_dirs(root: Path, experiment_id: str, extra_roots: Optional[Iterable[Path]] = None) -> List[Path]:
    candidates: List[Path] = []
    roots = [root]
    legacy = repo_root() / "runs" / "final" / experiment_id
    if legacy != root:
        roots.append(legacy)
    if extra_roots:
        roots.extend(extra_roots)
    for r in roots:
        if r.exists():
            candidates.extend(bounded_walk_dirs(r))
    # Prefer most recently modified directories and deduplicate.
    uniq = sorted({p.resolve() for p in candidates}, key=lambda p: p.stat().st_mtime, reverse=True)
    return uniq


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
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


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def summarize_run_dir(path: Path, default_total_steps: int, gpus_per_run: int) -> Dict[str, Any]:
    status = load_json(path / "status.json") or {}
    config = load_json(path / "config.json") or {}
    iter_log = path / "iter_log.jsonl"
    metrics_log = path / "metrics.jsonl"
    log_path = iter_log if iter_log.exists() else metrics_log

    phase_counts: Counter[str] = Counter()
    step_times: List[float] = []
    max_step = 0
    rows = 0
    fatal_errors: List[str] = []
    answer_lengths: List[int] = []

    for row in iter_jsonl(log_path):
        rows += 1
        step = row.get("step")
        if isinstance(step, int):
            max_step = max(max_step, step)
        phase = str(row.get("phase") or row.get("kind") or "unknown")
        phase_counts[phase] += 1
        t = row.get("step_time_sec")
        if isinstance(t, (int, float)) and t >= 0:
            step_times.append(float(t))
        if row.get("kind") == "fatal_error" or row.get("error"):
            fatal_errors.append(str(row.get("error", ""))[:240])
        answers = row.get("solver_answers_raw")
        if isinstance(answers, list):
            for ans in answers:
                if isinstance(ans, str) and ans.strip():
                    answer_lengths.append(len(ans.split()))

    progress = status.get("progress") if isinstance(status.get("progress"), dict) else {}
    if isinstance(progress.get("step"), int):
        max_step = max(max_step, int(progress["step"]))
    if isinstance(status.get("last_error"), str):
        fatal_errors.append(status["last_error"][:240])

    declared_total_steps = int(config.get("total_steps") or progress.get("steps_total") or default_total_steps)
    total_steps = max(declared_total_steps, int(default_total_steps))
    state = str(status.get("state") or "").lower()
    if state == "completed" or (total_steps and max_step >= total_steps):
        readiness = "complete"
    elif state == "failed" or fatal_errors:
        readiness = "failed"
    elif max_step > 0 or rows > 0:
        readiness = "partial"
    else:
        readiness = "not_started"

    mean_step = statistics.fmean(step_times) if step_times else None
    median_step = statistics.median(step_times) if step_times else None
    projected_wall_hours = None
    projected_gpu_hours = None
    if mean_step is not None and total_steps:
        projected_wall_hours = mean_step * total_steps / 3600.0
        projected_gpu_hours = projected_wall_hours * max(1, int(gpus_per_run))

    return {
        "path": str(path),
        "readiness": readiness,
        "state": state or None,
        "rows": rows,
        "max_step": max_step,
        "total_steps": total_steps,
        "declared_total_steps": declared_total_steps,
        "phase_counts": dict(sorted(phase_counts.items())),
        "mean_step_time_sec": mean_step,
        "median_step_time_sec": median_step,
        "projected_wall_hours": projected_wall_hours,
        "projected_gpu_hours": projected_gpu_hours,
        "answer_length_samples": len(answer_lengths),
        "answer_length_median": statistics.median(answer_lengths) if answer_lengths else None,
        "fatal_errors": fatal_errors[:3],
    }


def latest_run_summary(exp: Dict[str, Any], root: Path, default_total_steps: int, gpus_per_run: int) -> Dict[str, Any]:
    out_dir = rel_or_abs(root, str(exp.get("output_dir", "")))
    legacy_dirs = [rel_or_abs(root, str(p)) for p in exp.get("legacy_output_dirs", [])]
    dirs = find_artifact_dirs(out_dir, str(exp.get("id", "")), legacy_dirs)
    summaries = [summarize_run_dir(d, default_total_steps, gpus_per_run) for d in dirs]
    return {
        "output_dir": str(out_dir),
        "legacy_output_dirs": [str(p) for p in legacy_dirs],
        "artifact_dirs": [str(d) for d in dirs],
        "latest": summaries[0] if summaries else None,
        "all": summaries,
    }


def discover_data_candidates(root: Path) -> List[Dict[str, Any]]:
    data_root = root / "data"
    out: List[Dict[str, Any]] = []
    if not data_root.exists():
        return out
    for d in sorted(data_root.iterdir()):
        img_dir = d / "images"
        if not img_dir.is_dir():
            continue
        total, by_source = count_images(img_dir)
        out.append({"path": str(img_dir), "count": total, "sources": by_source})
    return out


def check_python_environment() -> Dict[str, Any]:
    python_bin = os.environ.get("PYTHON_BIN") or sys.executable or "python3"
    cmd = [
        python_bin,
        "-c",
        (
            "import json, torch; "
            "print(json.dumps({"
            "'torch_version': torch.__version__, "
            "'cuda_available': torch.cuda.is_available(), "
            "'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0"
            "}))"
        ),
    ]
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return {
            "python": python_bin,
            "torch_ready": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if proc.returncode != 0:
        err_text = (proc.stderr or proc.stdout).strip()
        err_lines = [line.strip() for line in err_text.splitlines() if line.strip()]
        err_summary = ""
        for line in err_lines:
            if any(key in line for key in ("ModuleNotFoundError", "ImportError", "OSError", "RuntimeError")):
                err_summary = line
                break
        if not err_summary and err_lines:
            err_summary = err_lines[-1]
        return {
            "python": python_bin,
            "torch_ready": False,
            "returncode": proc.returncode,
            "error": err_summary,
        }
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        payload = {}
    return {
        "python": python_bin,
        "torch_ready": True,
        **payload,
    }


def experiment_data_requirement(
    exp: Dict[str, Any],
    protocol: Dict[str, Any],
    root: Path,
    data_dir_override: Optional[str],
) -> Dict[str, Any]:
    data_dir_value = str(exp.get("data_dir") or data_dir_override or protocol["default_data_dir"])
    data_dir = rel_or_abs(root, data_dir_value)
    env = exp.get("env", {}) if isinstance(exp.get("env"), dict) else {}
    min_images = int(exp.get("minimum_data_images") or protocol.get("minimum_data_images", 0))
    if "TWO_STAGE_IMAGE_SAMPLES" in env:
        try:
            min_images = int(env["TWO_STAGE_IMAGE_SAMPLES"])
        except Exception:
            pass
    count, sources = count_images(data_dir)
    return {
        "path": str(data_dir),
        "count": count,
        "minimum_required": min_images,
        "ready": count >= min_images,
        "sources": sources,
    }


def audit(manifest: Dict[str, Any], root: Path, data_dir_override: Optional[str]) -> Dict[str, Any]:
    protocol = manifest["protocol"]
    data_dir = rel_or_abs(root, data_dir_override or protocol["default_data_dir"])
    image_count, by_source = count_images(data_dir)
    min_images = int(protocol.get("minimum_data_images", 0))
    expected_sources = set(protocol.get("expected_data_sources", []))
    present_sources = set(by_source)
    gpus_per_run = int(os.environ.get("GPUS_PER_RUN", protocol.get("default_gpus_per_run", 8)))
    total_steps = int(protocol.get("total_steps", 10000))

    experiments = []
    for exp in manifest.get("training_experiments", []):
        script_path = rel_or_abs(root, str(exp["script"]))
        exp_data = experiment_data_requirement(exp, protocol, root, data_dir_override)
        missing_env = []
        bad_env_paths = []
        for name in exp.get("required_env", []):
            value = os.environ.get(name, "")
            if not value:
                missing_env.append(name)
            elif name.endswith("_PATH") and not Path(value).expanduser().exists():
                bad_env_paths.append({"name": name, "value": value})
        run_info = latest_run_summary(exp, root, total_steps, gpus_per_run)
        latest = run_info.get("latest")
        status = latest["readiness"] if latest else "not_started"
        blockers = []
        if not script_path.exists():
            blockers.append(f"missing script: {exp['script']}")
        if not exp_data["ready"]:
            blockers.append(
                f"data has {exp_data['count']}/{exp_data['minimum_required']} images: {exp_data['path']}"
            )
        if missing_env:
            blockers.append(f"missing env: {', '.join(missing_env)}")
        if bad_env_paths:
            blockers.append("env path missing: " + ", ".join(f"{x['name']}={x['value']}" for x in bad_env_paths))
        if exp.get("required") and status in {"not_started", "failed"}:
            blockers.append(f"required experiment is {status}")
        experiments.append({
            "id": exp["id"],
            "backbone": exp.get("backbone"),
            "required": bool(exp.get("required", False)),
            "script": str(script_path),
            "script_exists": script_path.exists(),
            "missing_env": missing_env,
            "bad_env_paths": bad_env_paths,
            "status": status,
            "blockers": blockers,
            "data": exp_data,
            "run": run_info,
        })

    rebuttal = []
    for item in manifest.get("rebuttal_evidence", []):
        script_path = rel_or_abs(root, str(item["script"]))
        rebuttal.append({
            "id": item["id"],
            "description": item.get("description", ""),
            "script": str(script_path),
            "script_exists": script_path.exists(),
            "ready": script_path.exists(),
        })

    data_ready = image_count >= min_images and expected_sources.issubset(present_sources)
    return {
        "repo_root": str(root),
        "protocol": protocol,
        "data": {
            "path": str(data_dir),
            "ready": data_ready,
            "count": image_count,
            "minimum_required": min_images,
            "sources": by_source,
            "missing_expected_sources": sorted(expected_sources - present_sources),
            "candidates": discover_data_candidates(root),
        },
        "environment": check_python_environment(),
        "experiments": experiments,
        "rebuttal_evidence": rebuttal,
    }


def format_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    data = report["data"]
    lines.append("Experiment Readiness Audit")
    lines.append(f"repo: {report['repo_root']}")
    lines.append("")
    mark = "OK" if data["ready"] else "BLOCKED"
    lines.append(f"Data: {mark} {data['path']} ({data['count']}/{data['minimum_required']} images)")
    if data["sources"]:
        for source, count in data["sources"].items():
            lines.append(f"  - {source}: {count}")
    if data["missing_expected_sources"]:
        lines.append(f"  missing expected sources: {', '.join(data['missing_expected_sources'])}")
    if not data["ready"] and data["candidates"]:
        lines.append("  available image pools:")
        for cand in data["candidates"]:
            lines.append(f"  - {cand['path']}: {cand['count']} images")
    lines.append("")
    env = report.get("environment", {})
    env_mark = "OK" if env.get("torch_ready") else "BLOCKED"
    lines.append(f"Python/Torch: {env_mark} {env.get('python', 'python3')}")
    if env.get("torch_ready"):
        lines.append(
            "  torch="
            f"{env.get('torch_version', 'unknown')}, "
            f"cuda_available={env.get('cuda_available')}, "
            f"cuda_device_count={env.get('cuda_device_count')}"
        )
    elif env.get("error"):
        lines.append(f"  error: {env['error']}")
        lines.append("  set PYTHON_BIN to the training environment that has torch installed")
    lines.append("")
    lines.append("Training Experiments:")
    for exp in report["experiments"]:
        latest = exp["run"]["latest"] or {}
        progress = ""
        if latest:
            progress = f" step {latest.get('max_step', 0)}/{latest.get('total_steps', 0)}"
        req = "required" if exp["required"] else "optional"
        lines.append(f"- {exp['id']} [{req}] {exp['status']}{progress}")
        for blocker in exp["blockers"]:
            lines.append(f"  blocker: {blocker}")
        if latest.get("fatal_errors"):
            lines.append(f"  error: {latest['fatal_errors'][0]}")
        if latest.get("declared_total_steps") and latest.get("declared_total_steps") < latest.get("total_steps", 0):
            lines.append(
                f"  note: run was configured for {latest['declared_total_steps']} steps; "
                f"paper protocol requires {latest['total_steps']}"
            )
        if latest.get("projected_gpu_hours") is not None:
            lines.append(
                "  projected cost: "
                f"{latest['projected_wall_hours']:.1f} wall-h, "
                f"{latest['projected_gpu_hours']:.1f} GPU-h"
            )
    lines.append("")
    lines.append("Rebuttal Evidence Hooks:")
    for item in report["rebuttal_evidence"]:
        lines.append(f"- {item['id']}: {'OK' if item['ready'] else 'MISSING'}")
    return "\n".join(lines)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "scripts/self_evolving/paper/paper_experiments.json",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Return non-zero if required setup is blocked.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    report = audit(manifest, root, args.data_dir)
    payload = json.dumps(report, indent=2) if args.format == "json" else format_text(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

    if args.strict:
        blocked = not report["data"]["ready"] or any(
            exp["required"] and exp["blockers"] for exp in report["experiments"]
        ) or not report.get("environment", {}).get("torch_ready", False)
        return 1 if blocked else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
