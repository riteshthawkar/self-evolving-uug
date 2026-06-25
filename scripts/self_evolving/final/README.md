# Self-Evolving Training Scripts

This directory contains the fixed launchers used for BLIP3o self-evolving
training and ablations. The scripts are intended to be run from this directory
or through the paper manifest in `scripts/self_evolving/paper/`.

## Scripts

| ID | Script | Schedule | Trained components |
| --- | --- | --- | --- |
| E1 | `E1_main_joint.sh` | 3 understanding + 2 generation | Solver, proposer, generator adapter, DiT LoRA |
| E2 | `E2_understanding_only.sh` | understanding only | Solver, proposer |
| E3 | `E3_generation_only.sh` | generation only | Generator adapter, DiT LoRA |
| E4 | `E4_no_dit_rwr.sh` | joint, no DiT RWR | Solver, proposer, generator adapter |
| E5 | `E5_synthetic_loop.sh` | imageless joint loop | Solver, proposer, generator adapter, DiT LoRA |
| E6 | `E6_single_step.sh` | generation-centered unified step | Solver, proposer, generator adapter, DiT LoRA |
| E7 | `E7_two_stage.sh` | understanding stage then generation stage | Stage-specific adapters |

## Paper Manifest

The reproducible paper runs are defined in
`scripts/self_evolving/paper/paper_experiments.json` and launched through:

```bash
bash scripts/self_evolving/paper/run_experiment.sh blip3o_joint
bash scripts/self_evolving/paper/run_experiment.sh blip3o_two_stage
```

Useful checks:

```bash
python scripts/self_evolving/paper/check_experiment_readiness.py
python scripts/self_evolving/paper/validate_protocol.py
```

The default paper data path is `data/joint_pool_10k/images`, but the public
release does not include the private data-construction pipeline. Set
`DATA_DIR=/path/to/unlabeled/images` to use your local image pool, or use
`ALLOW_SMALL_DATA=1` for smoke tests.

## Slurm Usage

On a Slurm machine, first enter an allocated shell and then run the launcher:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=96G --time=3-00:00:00 --pty bash
source ~/.bashrc
conda activate /path/to/conda/env

CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
DATA_DIR=/path/to/images \
OUTPUT_DIR=$PWD/outputs/blip3o/E1_main_joint \
bash scripts/self_evolving/final/E1_main_joint.sh
```

For a short smoke test, set `TOTAL_STEPS=20`. Remove that override for full
training.

## Resume

Checkpoints are saved under the configured `OUTPUT_DIR` as `step_NNNNN`
directories. A checkpoint is complete when it contains `SAVE_OK`.

```bash
RESUME_FROM=/path/to/output_dir bash scripts/self_evolving/final/E1_main_joint.sh
RESUME_FROM=/path/to/output_dir/step_010000 bash scripts/self_evolving/final/E1_main_joint.sh
```

Resume restores trainable adapter weights, DiT LoRA weights, optimizer state,
KL coefficients, reward baselines, entropy/difficulty windows, and RNG state.
Replay buffers are rebuilt during training.

## Monitoring

Each run writes:

| Path | Contents |
| --- | --- |
| `status.json` | Latest progress, phase, checkpoint, and error state |
| `iter_log.jsonl` | Per-step structured records |
| `logs/training_watch.log` | Compact human-readable training trace |
| `logs/training_monitor.jsonl` | Structured monitor events |
| `logs/training_monitor.tsv` | Tabular monitor events |

The run registry is written under
`BLIP3o/blip3o/train/self_evolving/training_runs/<run_name>/` unless disabled
with `--disable_code_run_registry`.

## Evaluation

After training, pass a finished checkpoint to the evaluation scripts:

```bash
CHECKPOINT_DIR=/path/to/step_N bash BLIP3o/eval/understanding_eval_our.sh
CHECKPOINT_DIR=/path/to/step_N bash BLIP3o/eval/geneval/generation_our.sh
CHECKPOINT_DIR=/path/to/step_N bash BLIP3o/eval/dpg_bench/generate_dpg_our.sh
CHECKPOINT_DIR=/path/to/step_N bash BLIP3o/eval/wise/generate_wise_our.sh
```

WISE evaluation requires `OPENAI_API_KEY` in the environment. Do not store API
keys or Hugging Face tokens in repository files.
