# Unified Experiment Runner

Run experiments from one entrypoint:

```bash
python self_evolving/run_experiment.py \
  --experiment understanding_self_evolving \
  --data_dir /path/to/images/train \
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \
  --output_dir ./runs \
  --run_name u_only_seed42 \
  --total_steps 500 \
  --num_solver_samples 5 \
  --proposer_update_freq 5 \
  --use_lora
```

Resume example:

```bash
python self_evolving/run_experiment.py \
  --experiment understanding_self_evolving \
  --data_dir /path/to/images/train \
  --resume_from /path/to/runs/u_only_seed42/step_00500 \
  --start_step 500 \
  --total_steps 1000
```

## Modes

- `understanding_self_evolving`: Implemented in `self_evolving/experiments/understanding.py`
- `generation_self_evolving`: Implemented in `self_evolving/experiments/generation.py`
- `unified_self_evolving`: Implemented in `self_evolving/experiments/generation.py`
- `rl_no_self_evolving`: Reserved

## Reproducibility artifacts

Each run stores:

- `config.json` (full run config)
- `git_info.json` (commit/branch/dirty)
- `environment.json` (runtime package/device info)
- `iter_log.jsonl` (per-step records)
- `logs/rewards.jsonl` (step-level reward decomposition + baselines)
- `logs/policy_updates.jsonl` (per-update RL stats for role adapters)
- `ablation_summary.json` (run-level aggregated metrics for ablations)
- `step_XXXXX/` checkpoints (solver/proposer/generator adapters + trainer state)

Mode-specific logs:

- Understanding mode:
  - `logs/questions.jsonl`
  - `logs/solver_rollouts.jsonl`
- Generation/unified modes:
  - `logs/proposer_prompts.jsonl`
  - `logs/generation_candidates.jsonl`

## Understanding Experiment Matrix

The table below is for the **understanding-only self-evolving** study from this unified codebase.

| ID | Study goal | Factor(s) | Values | Seeds | Runs |
|---|---|---|---|---|---|
| `U00` | Main method robustness | Default config | Paper-aligned defaults | `42,123,777` | `3` |
| `U01` | Solver sample-count sensitivity | `num_solver_samples` | `3,5,7` | `42` | `3` |
| `U02` | Continuous solver reward softness | `solver_soft_gamma` | `0.5,0.7,1.0` | `42` | `3` |
| `U03` | Proposer update cadence | `proposer_update_freq` | `1,3,5,10` | `42` | `4` |
| `U04` | Entropy-band reward shaping | `prop_entropy_mu`,`prop_entropy_sigma` | `(0.70,0.25),(0.70,0.35),(0.90,0.25),(0.90,0.35),(1.10,0.25),(1.10,0.35)` | `42` | `6` |
| `U05` | KL control sensitivity | `kl_coef`,`kl_target` | `(2e-3,0.01),(1e-3,0.02),(5e-4,0.05)` | `42` | `3` |
| `U06` | LoRA capacity sensitivity | `lora_r`,`lora_alpha` | `(8,16),(16,32),(32,64)` | `42` | `3` |
| `U07` | Proposer-learning ablation (proxy) | proposer effectively frozen | `proposer_update_freq=total_steps+1` | `42` | `1` |

Total full matrix runs: `26`

## Understanding Launchers

All understanding launchers live in:

- `self_evolving/scripts/understanding_experiments/`

Each launcher is standalone and has no bash dependency on other scripts.
Scripts include only exports and direct run commands.

Per-experiment scripts:

- `00_u00_main_method.sh`
- `01_u01_solver_samples.sh`
- `02_u02_solver_gamma.sh`
- `03_u03_proposer_update_freq.sh`
- `04_u04_entropy_band.sh`
- `05_u05_kl_sensitivity.sh`
- `06_u06_lora_capacity.sh`
- `07_u07_frozen_proposer_proxy.sh`

Full matrix scripts:

- `90_run_all_understanding.sh`
- `run_understanding_all_standalone.sh`

Example (single experiment file):

```bash
export DATA_DIR="/path/to/images/train"
export OUTPUT_ROOT="./runs/understanding_experiments"
export WANDB_MODE="online"
export WANDB_PROJECT="self-evolving-uug-understanding"
bash self_evolving/scripts/understanding_experiments/04_u04_entropy_band.sh
```

Example (full matrix):

```bash
export DATA_DIR="/path/to/images/train"
export OUTPUT_ROOT="./runs/understanding_experiments"
bash self_evolving/scripts/understanding_experiments/90_run_all_understanding.sh
```

## Generation and Unified Launchers

Generation-only launcher:

- `self_evolving/scripts/generation_experiments/00_g00_main_method.sh`

Unified (understanding + generation) launcher:

- `self_evolving/scripts/unified_experiments/00_x00_main_method.sh`

Example:

```bash
export DATA_DIR="/path/to/images"
export OUTPUT_ROOT="./runs/generation_experiments"
bash self_evolving/scripts/generation_experiments/00_g00_main_method.sh
```

## Log and Naming Layout

Each experiment family writes to its own folder:

- `runs/understanding_experiments/U00_main_method/`
- `runs/understanding_experiments/U01_solver_samples/`
- ...

Inside each experiment family:

- One run directory per configuration via descriptive `--run_name` (e.g., `u04_mu0p90_sigma0p35_s42`).

## W&B Token and Env

W&B token is read from environment variable `WANDB_API_KEY` by the trainer.
Each experiment bash file exports these vars at the top (using current environment values):

- `WANDB_API_KEY`
- `WANDB_MODE`
- `WANDB_PROJECT`
- `WANDB_ENTITY`
- `WANDB_BASE_URL`
- `WANDB_LOG_IMAGES_EVERY`

Example:

```bash
export WANDB_API_KEY="<your_token>"
export WANDB_MODE="online"
export WANDB_PROJECT="self-evolving-uug-understanding"
export WANDB_ENTITY="<team_or_user>"
```
