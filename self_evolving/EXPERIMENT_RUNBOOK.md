# Self-Evolving UUG Experiment Runbook

This file defines the experiment set to run from this single codebase, why each experiment matters, and exactly how to run it.

## Scope

Target experiment families:

1. Understanding-only self-evolving (EvoLMM-style)
2. Generation-only self-evolving
3. Unified understanding + generation self-evolving
4. RL baseline without self-evolving

## Current implementation status

| ID | Experiment | Status in current codebase | Primary launcher |
|---|---|---|---|
| U | Understanding-only self-evolving | Implemented | `self_evolving/scripts/understanding_experiments/*.sh` |
| G | Generation-only self-evolving | Implemented | `self_evolving/scripts/generation_experiments/00_g00_main_method.sh` |
| X | Unified self-evolving (U+G) | Implemented | `self_evolving/scripts/unified_experiments/00_x00_main_method.sh` |
| R | RL without self-evolving | Not yet implemented in unified runner (`--experiment rl_no_self_evolving` raises `NotImplementedError`) | Pending |

## Why these experiments are needed

### U: Understanding-only self-evolving
Why:
- Establishes the direct EvoLMM-style baseline in your unified codebase.
- Validates proposer-solver reward dynamics independent of generation.
- Provides ablation-ready logs for question quality and solver agreement.

What it answers:
- Does understanding improve with self-generated question-answer supervision?
- Which reward/schedule hyperparameters matter most?

### G: Generation-only self-evolving
Why:
- Isolates generator improvement from understanding-loop confounds.
- Tests whether proposer-generated specs + verifier feedback improve generation quality.

What it answers:
- Can generation improve from internal consistency/cycle/verifier rewards?
- Does spec quality gating stabilize updates?

### X: Unified co-evolution (U + G)
Why:
- This is the main research contribution: both capabilities evolve in one training process.
- Tests cross-capability transfer rather than isolated adaptation.

What it answers:
- Does joint training outperform separate U and G runs?
- Is alternating schedule (`U steps` vs `G steps`) stable and beneficial?

### R: RL without self-evolving baseline
Why:
- Required as a control to separate “RL gains” from “self-evolving gains”.
- Needed for fair claims in paper/report.

What it answers:
- How much improvement comes from RL alone when proposer-driven self-evolving loop is removed?

## Recommended execution order

1. U00 baseline and U-ablations (stability first)
2. G00 main generation run
3. X00 unified run
4. R00 RL-no-self-evolving baseline (after implementation)

## Environment setup (all experiments)

Set these before running scripts (adjust paths for your machine/container):

```bash
export REPO_ROOT="/workspace/self-evolving-uug"
cd "$REPO_ROOT"

export DATA_DIR="/workspace/self-evolving-uug/data/shared_uug_50k_balanced/images"
export WANDB_API_KEY="<your_wandb_key>"
export WANDB_MODE="online"             # online | offline | disabled
export WANDB_PROJECT="self-evolving-uug"
export WANDB_ENTITY="<team_or_user>"   # optional

export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NPROC_PER_NODE="8"
```

## Run commands

### U experiments (understanding)

Main baseline (3 seeds):

```bash
bash self_evolving/scripts/understanding_experiments/00_u00_main_method.sh
```

Main ablations:

```bash
bash self_evolving/scripts/understanding_experiments/01_u01_solver_samples.sh
bash self_evolving/scripts/understanding_experiments/02_u02_solver_gamma.sh
bash self_evolving/scripts/understanding_experiments/03_u03_proposer_update_freq.sh
bash self_evolving/scripts/understanding_experiments/04_u04_entropy_band.sh
bash self_evolving/scripts/understanding_experiments/05_u05_kl_sensitivity.sh
bash self_evolving/scripts/understanding_experiments/06_u06_lora_capacity.sh
bash self_evolving/scripts/understanding_experiments/07_u07_frozen_proposer_proxy.sh
```

Run full U matrix in one go:

```bash
bash self_evolving/scripts/understanding_experiments/90_run_all_understanding.sh
```

### G experiment (generation-only)

```bash
bash self_evolving/scripts/generation_experiments/00_g00_main_method.sh
```

### X experiment (unified)

```bash
bash self_evolving/scripts/unified_experiments/00_x00_main_method.sh
```

### R experiment (RL without self-evolving)

Current status:
- Not runnable through `self_evolving/run_experiment.py` yet.
- You should not report this baseline as complete until we add `run_rl_no_self_evolving(...)` and a dedicated script under `self_evolving/scripts/rl_experiments/`.

## Where outputs/logs are stored

### Understanding runs
- Root: `runs/understanding_experiments/...`
- Key files:
  - `iter_log.jsonl`
  - `logs/questions.jsonl`
  - `logs/solver_rollouts.jsonl`
  - `logs/rewards.jsonl`
  - `logs/policy_updates.jsonl`
  - `ablation_summary.json`

### Generation runs
- Root: `runs/generation_experiments/...`
- Key files:
  - `iter_log.jsonl`
  - `logs/proposer_prompts.jsonl`
  - `logs/generation_candidates.jsonl`
  - `logs/rewards.jsonl`
  - `logs/policy_updates.jsonl`
  - `ablation_summary.json`

### Unified runs
- Root: `runs/unified_experiments/...`
- Key files:
  - `iter_log.jsonl`
  - `logs/proposer_prompts.jsonl`
  - `logs/generation_candidates.jsonl`
  - `logs/rewards.jsonl`
  - `logs/policy_updates.jsonl`
  - `ablation_summary.json`

## Resume training if interrupted

All implemented experiment families support checkpoint resume.

Template:

```bash
python self_evolving/run_experiment.py \
  --experiment <understanding_self_evolving|generation_self_evolving|unified_self_evolving> \
  --data_dir "$DATA_DIR" \
  --resume_from <run_dir>/step_<NNNNN> \
  --start_step <NNNNN> \
  --total_steps <new_total_steps> \
  ...<same key config as original run>
```

Tip:
- On interruption, run directories contain `interruption.json` and `resume_hint.json` (implemented for generation/unified), plus step checkpoints.

## Minimal report checklist per family

For each U/G/X family, report:

1. Final metric means and std across seeds (where applicable)
2. Stability signals (`reward_mean`, KL coefficients, baseline drift)
3. Ablation summary from `ablation_summary.json`
4. Two qualitative examples from logs (best and failure case)

