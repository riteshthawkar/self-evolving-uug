# BLIP3o Self-Evolving Experiments

Last updated: 2026-02-10

This document is the canonical experiment plan for the current BLIP3o-based self-evolving codebase.

## Scope

The plan covers:
- Understanding-only self-evolving (`U`)
- Generation-only self-evolving (`G`)
- Unified understanding + generation self-evolving (`X`)
- Method and reward ablations
- Minimal submission-ready experiment set

## Reproducibility and Fairness Rules

All comparisons must follow these rules:
- Match by **optimizer steps**, not just training calls.
- Log and compare effective updates with `did_step`.
- Keep decoding settings fixed across compared runs (CFG, inference steps, resolution).
- Keep data split and prompt/eval sets fixed.
- Report seed mean/std for multi-seed comparisons.
- Treat expected outcomes as hypotheses, not guaranteed results.

## Tier 1: Core Validation (Main Results)

| ID | Experiment | Purpose | Default launcher |
|---|---|---|---|
| E1.1 | Pretrained baseline (no self-evolving) | Starting point for all metrics | Eval-only (no training) |
| E1.2 | Understanding-only | Isolate U improvements | `BLIP3o/scripts/self_evolving/understanding/00_u00_main_method.sh` |
| E1.3 | Generation-only | Isolate G improvements | `BLIP3o/scripts/self_evolving/generation/00_g00_main_method.sh` |
| E1.4 | Unified main method (REINFORCE) | Joint training result | `BLIP3o/scripts/self_evolving/unified/00_x00_main_method.sh` |

Suggested seeds for final core reporting: `42, 123, 777`.

## Tier 2: Key Component Ablations

Run these from the unified setup (`X00`) by changing only listed flags.

| ID | Ablation | Config change |
|---|---|---|
| A2.1 | No cycle consistency | `--reward_cycle_weight 0.0` |
| A2.2 | No diversity reward | `--reward_diversity_weight 0.0` |
| A2.3 | No contradiction penalty | `--reward_contradiction_weight 0.0` |
| A2.4 | Spec-only reward | `--reward_cycle_weight 0.0 --reward_diversity_weight 0.0 --reward_contradiction_weight 0.0` |
| A2.5 | Jaccard cycle (old) | Switch cycle scoring to Jaccard in `_cycle_reward` |
| A2.6 | No proposer learning proxy | Use frozen-proposer style setup |
| A2.7 | No synthetic solver bridge | Disable synthetic solver updates |
| A2.8 | Single/shared adapter roles | Share role adapter instead of split roles |
| A2.9 | No KL regularization | `--kl_coef 0 --kl_min 0 --kl_max 0` |

## Tier 3: Design Choice Comparisons

| ID | Comparison | Launcher / change |
|---|---|---|
| C3.1 | REINFORCE vs DPO | `00_x00_main_method.sh` vs `01_x01_dpo_style.sh` |
| C3.2 | Strict token trace vs proxy fallback | `--strict_require_generation_tokens --generator_missing_trace_strategy skip` vs current proxy mode |
| C3.3 | U:G cycle ratio | Change `--understanding_steps_per_cycle` / `--generation_steps_per_cycle` |

## Tier 4: Analysis Figures and Audits

Must-have analysis:
- 4A: training dynamics (`spec`, `cycle`, `diversity`, `contradiction`, KL, reward, `did_step`).
- 4B: qualitative curriculum snapshots over training.
- 4E: self-judge bias audit (internal metrics vs external metrics correlation).

Optional analysis:
- 4C: per-candidate diversity score evolution.
- 4D: reward-component correlation heatmap.

## Minimal Submission-Ready Set (Must Run)

This is the minimum set recommended for a defensible submission.

1. E1.1 pretrained baseline eval (1 run).
2. E1.2 understanding-only main (`U00`) (1 seed).
3. E1.3 generation-only main (`G00`) (1 seed).
4. E1.4 unified main (`X00`) (3 seeds: 42/123/777).
5. A2.1 no-cycle ablation (1 seed).
6. A2.5 Jaccard-cycle ablation (3 seeds: 42/123/777).
7. A2.7 no synthetic solver bridge (1 seed).
8. A2.9 no-KL ablation (1 seed).
9. C3.1 DPO comparison (`X01`) (1 seed).
10. C3.2 strict-token-trace comparison (1 seed).
11. 4A + 4B + 4E analysis on all above checkpoints.

Total recommended minimal training runs: **14** (plus evaluation-only baseline).

## Evaluation Set (Minimal, Strong)

Use at least:
- One independent understanding benchmark (for U/X claims).
- One compositional generation benchmark (for G/X claims).
- One external or human-aligned generation quality check for self-judge bias audit.

Keep the same evaluation protocol across all compared runs.

## Notes

- `U00`, `G00`, and `X00/X01` scripts already exist under `BLIP3o/scripts/self_evolving/`.
- Some ablations above require either:
  - a dedicated copy script with modified CLI flags, or
  - running `BLIP3o/blip3o/train/train_self_evolving.py` directly with the same base flags.
- When reporting compute, include GPU-hours and effective optimizer steps (`did_step` totals).

