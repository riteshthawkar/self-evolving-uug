# Understanding Experiment Launchers

This folder contains:

- A single self-contained runner with all understanding commands inline:
  - `run_understanding_all_standalone.sh`
- Separate family launchers (each one is now self-contained; no sourcing other scripts):
  - `00_u00_main_method.sh`
  - `01_u01_solver_samples.sh`
  - `02_u02_solver_gamma.sh`
  - `03_u03_proposer_update_freq.sh`
  - `04_u04_entropy_band.sh`
  - `05_u05_kl_sensitivity.sh`
  - `06_u06_lora_capacity.sh`
  - `07_u07_frozen_proposer_proxy.sh`
  - `90_run_all_understanding.sh`

## Recommended (dependency-free) usage

```bash
bash self_evolving/scripts/understanding_experiments/run_understanding_all_standalone.sh \
  --data_dir /path/to/images/train \
  --output_root ./runs/understanding_experiments \
  --suite full
```

This script does not source any other script and includes:

- all U00-U07 run commands
- Hugging Face/PyTorch cache exports
- W&B env handling and CLI wiring

## Per-experiment standalone usage

Each per-experiment script (`00` to `07`) is also fully standalone and includes:

- cache exports
- W&B env exports/flags
- argument parsing
- direct `run_experiment.py` commands

No `common.sh` dependency is required for these scripts.

## Common usage

```bash
bash self_evolving/scripts/understanding_experiments/00_u00_main_method.sh \
  --data_dir /path/to/images/train \
  --output_root ./runs/understanding_experiments
```

Common flags accepted by all scripts:

- `--data_dir` (required)
- `--model_name`
- `--output_root`
- `--total_steps`
- `--save_every`
- `--max_checkpoints`
- `--cuda_device`
- `--python_bin`
- `--wandb_mode`
- `--wandb_project`
- `--wandb_entity`
- `--wandb_log_images_every`
- `--dry_run`

Unknown flags are forwarded to `self_evolving/run_experiment.py`.

## W&B environment variables

Each bash file exports the following at the top (using existing env values if set):

- `WANDB_API_KEY` (token read by the Python trainer)
- `WANDB_MODE` (`online|offline|disabled`)
- `WANDB_PROJECT`
- `WANDB_ENTITY`
- `WANDB_BASE_URL`
- `WANDB_LOG_IMAGES_EVERY`

Typical usage:

```bash
export WANDB_API_KEY="<your_token>"
export WANDB_MODE="online"
export WANDB_PROJECT="self-evolving-uug-understanding"
export WANDB_ENTITY="<team_or_user>"
```

## Output naming

Each experiment family writes to its own folder under `--output_root`, and each run uses descriptive names like:

- `u01_nsamples_7_s42`
- `u04_mu0p90_sigma0p35_s42`
- `u07_frozen_proposer_proxy_s42`

Launcher logs are stored separately in:

- `<experiment_family>/launcher_logs/<UTC_TIMESTAMP>/<run_name>.log`
