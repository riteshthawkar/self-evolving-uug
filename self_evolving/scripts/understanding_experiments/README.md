# Understanding Experiment Launchers

All scripts here are pure standalone scripts:

- `00_u00_main_method.sh`
- `01_u01_solver_samples.sh`
- `02_u02_solver_gamma.sh`
- `03_u03_proposer_update_freq.sh`
- `04_u04_entropy_band.sh`
- `05_u05_kl_sensitivity.sh`
- `06_u06_lora_capacity.sh`
- `07_u07_frozen_proposer_proxy.sh`
- `90_run_all_understanding.sh`
- `run_understanding_all_standalone.sh`

No script sources any other bash file.  
Each script only has:

- `export` lines (cache, W&B, run config)
- direct `python self_evolving/run_experiment.py ...` commands

## W&B environment variables

Each script exports these (override before running if needed):

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

## Running

Set env vars, then run a script:

```bash
export DATA_DIR="/path/to/images/train"
export OUTPUT_ROOT="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/runs/understanding_experiments"
bash self_evolving/scripts/understanding_experiments/00_u00_main_method.sh
```

Run full matrix:

```bash
export DATA_DIR="/path/to/images/train"
bash self_evolving/scripts/understanding_experiments/90_run_all_understanding.sh
```
