#!/usr/bin/env python3
"""
Extract and summarize key statistics from iter_log.jsonl files.

Handles THREE log formats:
  1. understanding_trainer  — all understanding, ~75 fields, no "phase" key
  2. generation_trainer     — all generation, "phase"="generation", best_metrics nested dict
  3. unified_trainer        — mixed phases, "phase" = "understanding" | "generation"
     - Understanding: ~160 fields including proposer_ste_difficulty, proposer_token_entropy_max
     - Generation: flat format with best_reward (no best_metrics dict)

Also reads the logs/ subdirectory files:
  - logs/rewards.jsonl          (generation-phase detailed rewards with best_metrics)
  - logs/generation_candidates.jsonl (per-candidate scores)
  - logs/policy_updates.jsonl   (per-role update stats)

Usage:
    python extract_log_data.py \
        --log /path/to/runs/<run_name>/iter_log.jsonl \
        --output_csv /path/to/output.csv \
        --window 50
"""

import argparse
import csv
import json
import pathlib
from collections import defaultdict

import numpy as np


def _safe_mean(lst):
    return float(np.mean(lst)) if lst else 0.0


def _safe_max(lst):
    return float(np.max(lst)) if lst else 0.0


def extract(log_path, output_csv, window=50):
    log_path = pathlib.Path(log_path)
    understanding_rows = []
    generation_rows = []

    # Also try to read logs/rewards.jsonl for richer generation metrics
    rewards_log = log_path.parent / "logs" / "rewards.jsonl"
    gen_rewards_by_step = {}
    if rewards_log.exists():
        with open(rewards_log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                gen_rewards_by_step[row["step"]] = row

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            # Detect phase: unified_trainer sets "phase", standalone understanding_trainer does not
            phase = row.get("phase")
            if phase is None:
                # Standalone understanding_trainer (no phase key)
                phase = "understanding"
            if row.get("skip_reason"):
                # Skip entries (e.g. generated_pool_empty)
                continue

            if phase == "understanding":
                sr = row.get("solver_rewards_raw", [])
                understanding_rows.append({
                    "step": row["step"],
                    # Self-consistency entropy
                    "entropy_nats": row.get("entropy_nats", 0.0),
                    # Majority voting
                    "majority_fraction": row.get("majority_fraction", 1.0),
                    "majority_answer": row.get("majority_answer", ""),
                    # Proposer metrics
                    "proposer_reward": row.get("proposer_reward", 0.0),
                    "proposer_reward_raw": row.get("proposer_reward_raw", 0.0),
                    "proposer_reward_raw_gaussian": row.get("proposer_reward_raw_gaussian", 0.0),
                    "proposer_baseline": row.get("proposer_baseline", row.get("proposer_baseline_after", 0.0)),
                    "proposer_entropy_mu_used": row.get("proposer_entropy_mu_used", 0.0),
                    # STE (Solver Token Entropy) — unified_trainer only
                    "ste_difficulty": row.get("proposer_ste_difficulty", 0.0),
                    "token_entropy_max": row.get("proposer_token_entropy_max", 0.0),
                    "token_entropy_mean": row.get("proposer_token_entropy_mean", 0.0),
                    # Solver metrics
                    "solver_reward_mean": _safe_mean(sr),
                    "solver_reward_max": _safe_max(sr),
                    "solver_top1_prob": row.get("solver_top1_prob", 1.0),
                    "solver_top2_prob": row.get("solver_top2_prob", 0.0),
                    "solver_margin": row.get("solver_margin", 1.0),
                    "solver_baseline": row.get("solver_baseline", row.get("solver_baseline_after", 0.0)),
                    "solver_update_scale": row.get("solver_update_scale", 0.0),
                    "solver_update_applied": row.get("solver_update_applied", False),
                    "solver_informative_ratio": row.get("solver_informative_ratio", 0.0),
                    # Difficulty
                    "difficulty_bucket": row.get("difficulty_bucket_observed", ""),
                    "difficulty_target": row.get("difficulty_target_bucket", ""),
                    "easy_question": row.get("easy_question_detected", False),
                    # Curriculum
                    "curriculum_arm": row.get("curriculum_arm_selected_key", ""),
                    # Timing
                    "step_time_sec": row.get("step_time_sec", row.get("step_duration_sec", 0.0)),
                })

            elif phase == "generation":
                # Handle BOTH formats:
                # 1. generation_trainer: best_metrics is a nested dict
                # 2. unified_trainer: flat best_reward, no best_metrics
                bm = row.get("best_metrics", {})
                # Try rewards.jsonl for richer data
                rr = gen_rewards_by_step.get(row["step"], {})
                rr_bm = rr.get("best_metrics", {})

                generation_rows.append({
                    "step": row["step"],
                    "total_reward": (
                        bm.get("total_reward")
                        or row.get("best_reward")
                        or rr_bm.get("total_reward", 0.0)
                    ),
                    "cycle_score": bm.get("cycle_score", rr_bm.get("cycle_score", 0.0)),
                    "spec_score": bm.get("spec_score", rr_bm.get("spec_score", 0.0)),
                    "diversity_score": bm.get("diversity_score", rr_bm.get("diversity_score", 0.0)),
                    "contradiction_score": bm.get("contradiction_score", rr_bm.get("contradiction_score", 0.0)),
                    "spec_quality": row.get("spec_quality", 0.0),
                    "generator_baseline": row.get("generator_baseline", 0.0),
                    "generator_update_rule": row.get("generator_update_rule", ""),
                    "generator_update_mode": row.get("generator_update_mode", ""),
                    "step_time_sec": row.get("step_duration_sec", row.get("step_time_sec", 0.0)),
                })

    # Write CSVs
    if output_csv:
        out = pathlib.Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)

        if understanding_rows:
            u_path = out.with_name(out.stem + "_understanding.csv")
            with open(u_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=understanding_rows[0].keys())
                w.writeheader()
                w.writerows(understanding_rows)
            print(f"Understanding CSV: {u_path} ({len(understanding_rows)} rows)")

        if generation_rows:
            g_path = out.with_name(out.stem + "_generation.csv")
            with open(g_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=generation_rows[0].keys())
                w.writeheader()
                w.writerows(generation_rows)
            print(f"Generation CSV: {g_path} ({len(generation_rows)} rows)")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Log: {log_path}")
    print(f"{'='*60}")

    if understanding_rows:
        n = len(understanding_rows)
        entropies = [r["entropy_nats"] for r in understanding_rows]
        p_rewards = [r["proposer_reward"] for r in understanding_rows]
        s_rewards = [r["solver_reward_mean"] for r in understanding_rows]
        maj_fracs = [r["majority_fraction"] for r in understanding_rows]
        ste_diffs = [r["ste_difficulty"] for r in understanding_rows]
        easy_rate = np.mean([r["easy_question"] for r in understanding_rows])

        print(f"\nUnderstanding phase: {n} steps")
        print(f"  Steps: {understanding_rows[0]['step']} -> {understanding_rows[-1]['step']}")
        print(f"  Entropy (nats):     mean={np.mean(entropies):.4f}  std={np.std(entropies):.4f}")
        print(f"  Proposer reward:    mean={np.mean(p_rewards):.4f}  std={np.std(p_rewards):.4f}")
        print(f"  Solver reward:      mean={np.mean(s_rewards):.4f}  std={np.std(s_rewards):.4f}")
        print(f"  Majority fraction:  mean={np.mean(maj_fracs):.4f}")
        print(f"  STE difficulty:     mean={np.mean(ste_diffs):.4f}  std={np.std(ste_diffs):.4f}")
        print(f"  Easy question rate: {easy_rate:.2%}")

        # Difficulty bucket distribution
        buckets = [r["difficulty_bucket"] for r in understanding_rows if r["difficulty_bucket"]]
        if buckets:
            from collections import Counter
            bc = Counter(buckets)
            print(f"  Difficulty distribution: {dict(bc)}")

        # Windowed trend
        if n >= 2 * window:
            early = understanding_rows[:window]
            late = understanding_rows[-window:]
            print(f"\n  Trend (first {window} vs last {window} steps):")
            print(f"    Entropy:        {np.mean([r['entropy_nats'] for r in early]):.4f} -> {np.mean([r['entropy_nats'] for r in late]):.4f}")
            print(f"    Proposer:       {np.mean([r['proposer_reward'] for r in early]):.4f} -> {np.mean([r['proposer_reward'] for r in late]):.4f}")
            print(f"    Solver:         {np.mean([r['solver_reward_mean'] for r in early]):.4f} -> {np.mean([r['solver_reward_mean'] for r in late]):.4f}")
            print(f"    STE difficulty:  {np.mean([r['ste_difficulty'] for r in early]):.4f} -> {np.mean([r['ste_difficulty'] for r in late]):.4f}")
            print(f"    Easy rate:      {np.mean([r['easy_question'] for r in early]):.2%} -> {np.mean([r['easy_question'] for r in late]):.2%}")

    if generation_rows:
        totals = [r["total_reward"] for r in generation_rows]
        cycles = [r["cycle_score"] for r in generation_rows]
        specs = [r["spec_score"] for r in generation_rows]

        print(f"\nGeneration phase: {len(generation_rows)} steps")
        print(f"  Steps: {generation_rows[0]['step']} -> {generation_rows[-1]['step']}")
        print(f"  Total reward:     mean={np.mean(totals):.4f}  std={np.std(totals):.4f}")
        print(f"  Cycle score:      mean={np.mean(cycles):.4f}  std={np.std(cycles):.4f}")
        print(f"  QA fidelity:      mean={np.mean(specs):.4f}  std={np.std(specs):.4f}")

    if not understanding_rows and not generation_rows:
        print("  No valid entries found.")


def main():
    parser = argparse.ArgumentParser(description="Extract and summarize log data")
    parser.add_argument("--log", type=str, required=True, help="Path to iter_log.jsonl")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path stem")
    parser.add_argument("--window", type=int, default=50, help="Window for trend comparison")
    args = parser.parse_args()

    extract(args.log, args.output_csv, args.window)


if __name__ == "__main__":
    main()
