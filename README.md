# Ask, Solve, Generate

Code for **Ask, Solve, Generate: Self-Evolving Unified Multimodal Understanding
and Generation via Self-Consistency Rewards**.

This repository contains the public training, inference, and evaluation code for
self-evolving unified multimodal understanding and generation. The release
focuses on the model-side implementation: the private data-construction pipeline,
paper source, local checkpoints, generated outputs, and cluster-specific logs are
not included.

| Resource | Status |
| --- | --- |
| Paper | Coming soon |
| Project page | Coming soon |
| Model checkpoints | Coming soon |

## Contents

- [Overview](#overview)
- [Release Scope](#release-scope)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Secrets](#secrets)
- [Data](#data)
- [Training](#training)
- [Inference](#inference)
- [Resume and Monitoring](#resume-and-monitoring)
- [Evaluation](#evaluation)
- [Protocol Checks](#protocol-checks)
- [Cluster Usage](#cluster-usage)
- [Citation](#citation)
- [License](#license)
- [Release Checklist](#release-checklist)

## Overview

The project studies whether unified multimodal models can improve both visual
understanding and image generation from unlabeled images. The training loop uses
three internal roles:

| Role | Purpose |
| --- | --- |
| Proposer | Generates visual questions from an unlabeled image |
| Solver | Answers and scores candidate questions through self-consistency |
| Generator | Produces images from question-answer-derived generation specs |

The implementation uses internal consistency signals instead of external answer
labels. For understanding, Solver Token Entropy (STE) provides a token-level
uncertainty signal for selecting useful updates. For generation, question-answer
fidelity and cycle-consistent captioning couple generated images back to the
understanding loop.

The main implementation is built around BLIP3o. BAGEL and VARGPT-v1.1
integrations are included to evaluate the same self-evolving recipe on different
unified model families.

## Release Scope

Included:

- BLIP3o self-evolving training and evaluation code.
- BAGEL training and evaluation integration.
- VARGPT-v1.1 training and evaluation integration.
- Paper-protocol launch manifests and reproducibility checks.
- Public release documentation and Apache-2.0 license.

Not included:

- Private data-construction or filtering pipelines.
- Training images, benchmark data mirrors, or generated images.
- Model checkpoints, local model downloads, caches, or run logs.
- Manuscript source, Overleaf folders, reviewer materials, or internal notes.
- API keys, Hugging Face tokens, W&B credentials, or machine-specific paths.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `BLIP3o/` | BLIP3o training, inference, and evaluation code |
| `Bagel/` | BAGEL baseline integration and self-evolving launchers |
| `vargpt_1_1/` | VARGPT-v1.1 baseline integration and self-evolving launchers |
| `scripts/self_evolving/final/` | Fixed BLIP3o experiment and ablation launchers |
| `scripts/self_evolving/paper/` | Manifest-based paper-protocol launch and audit utilities |
| `scripts/env/` | Shared environment bootstrap helpers |
| `.env.example` | Template for local secrets and cache paths |

Generated outputs are ignored by git. Keep datasets, model weights,
checkpoints, logs, and caches outside the tracked source tree or under ignored
paths such as `data/`, `models/`, `outputs/`, and `logs/`.

## Installation

Use separate environments for the three backends. The dependency stacks differ,
and mixing them in one environment can create version conflicts.

### BLIP3o

```bash
conda create -n uug-blip3o python=3.10 -y
conda activate uug-blip3o
pip install -r BLIP3o/requirements.safe.txt
```

If you need the full original stack instead of the safer pinned subset:

```bash
pip install -r BLIP3o/requirements.txt
```

### BAGEL

```bash
conda create -n uug-bagel python=3.10 -y
conda activate uug-bagel
pip install -r Bagel/requirements.txt
```

### VARGPT-v1.1

```bash
conda create -n uug-vargpt python=3.10 -y
conda activate uug-vargpt
pip install -r vargpt_1_1/VARGPT-family-training/requirements.txt
```

For VARGPT understanding evaluation, install the local evaluation package when
needed:

```bash
pip install -e vargpt_1_1/understand_eval
```

## Secrets

Do not commit secrets. Use environment variables or an untracked local `.env`
file. Start from the provided template:

```bash
cp .env.example .env
```

Common variables:

| Variable | Used for |
| --- | --- |
| `HF_TOKEN` | Downloading gated Hugging Face models or datasets |
| `OPENAI_API_KEY` | API-based WISE scoring |
| `WANDB_API_KEY` | Optional experiment logging |
| `HF_HOME` | Hugging Face cache location |

## Data

The public release expects a local directory of unlabeled images. The default
paper-protocol path is:

```text
data/joint_pool_10k/images/
```

A full paper-protocol run expects at least 10,000 images. You can also point the
launchers to any local image directory:

```bash
export DATA_DIR=/path/to/unlabeled/images
```

Supported image extensions are detected by the launch scripts, including
`.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp`.

## Training

The recommended entry point is the paper manifest launcher:

```bash
DATA_DIR=/path/to/unlabeled/images \
NPROC_PER_NODE=8 \
bash scripts/self_evolving/paper/run_experiment.sh blip3o_joint
```

Dry-run a command without launching training:

```bash
DRY_RUN=1 DATA_DIR=/path/to/unlabeled/images \
bash scripts/self_evolving/paper/run_experiment.sh blip3o_joint
```

Common experiment IDs:

| ID | Backend | Purpose |
| --- | --- | --- |
| `blip3o_joint` | BLIP3o | Main joint 3U:2G training run |
| `blip3o_understanding_only` | BLIP3o | Understanding-only ablation |
| `blip3o_generation_only` | BLIP3o | Generation-only ablation |
| `blip3o_no_dit_rwr` | BLIP3o | Joint run without DiT reward-weighted regression |
| `blip3o_synthetic_loop` | BLIP3o | Generated-only loop control |
| `blip3o_two_stage` | BLIP3o | Understanding stage followed by generation stage |
| `bagel_joint` | BAGEL | Main BAGEL joint run |
| `vargpt_joint` | VARGPT-v1.1 | Main VARGPT joint run |

The full manifest is in
`scripts/self_evolving/paper/paper_experiments.json`.

### Direct BLIP3o Launcher

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
DATA_DIR=/path/to/unlabeled/images \
OUTPUT_DIR=$PWD/outputs/blip3o/E1_main_joint \
bash scripts/self_evolving/final/E1_main_joint.sh
```

For a short smoke test:

```bash
TOTAL_STEPS=20 \
ALLOW_SMALL_DATA=1 \
DATA_DIR=/path/to/small/image/folder \
OUTPUT_DIR=$PWD/outputs/smoke/blip3o \
bash scripts/self_evolving/final/E1_main_joint.sh
```

### BAGEL Launcher

```bash
MODEL_PATH=/path/to/BAGEL-7B-MoT \
DATA_DIR=/path/to/unlabeled/images \
OUTPUT_DIR=$PWD/outputs/bagel/B1_unified_training \
NPROC_PER_NODE=8 \
bash Bagel/scripts/B1_unified_training.sh
```

`MODEL_PATH` must point to the local BAGEL model checkout or checkpoint
directory.

### VARGPT-v1.1 Launcher

```bash
cd vargpt_1_1/VARGPT-family-training

IMAGE_FOLDER=/path/to/unlabeled/images \
OUTPUT_DIR=/path/to/output/vargpt/joint \
bash examples/train_self_evolving/run_self_evolving.sh joint 8
```

Available VARGPT modes are `joint`, `u_only`, and `gen_only`.

## Inference

For BLIP3o interactive qualitative inspection:

```bash
python BLIP3o/gradio/app.py /path/to/BLIP3o-Model-8B
```

For a minimal BLIP3o text-to-image example:

```bash
python BLIP3o/inference.py /path/to/BLIP3o-Model-8B
```

The standalone BLIP3o example contains an inline prompt for quick inspection;
edit the prompt in the script or use the benchmark-generation wrappers for
batch sampling.

For VARGPT-v1.1 standalone examples:

```bash
python vargpt_1_1/inference_v1_1/understanding_vargpt_v1_1.py
python vargpt_1_1/inference_v1_1/generation_vargpt_v1_1.py
```

BAGEL inference utilities are exposed through `Bagel/inferencer.py` and are used
by the BAGEL training and evaluation launchers.

## Resume and Monitoring

BLIP3o checkpoints are saved under the configured `OUTPUT_DIR` as
`step_NNNNN` directories. A checkpoint is complete when it contains `SAVE_OK`.

```bash
RESUME_FROM=/path/to/output_dir \
bash scripts/self_evolving/final/E1_main_joint.sh

RESUME_FROM=/path/to/output_dir/step_010000 \
bash scripts/self_evolving/final/E1_main_joint.sh
```

Typical run files:

| Path | Contents |
| --- | --- |
| `status.json` | Latest progress, phase, checkpoint, and error state |
| `iter_log.jsonl` | Per-step structured records |
| `logs/training_watch.log` | Human-readable training trace |
| `logs/training_monitor.jsonl` | Structured monitor events |
| `logs/training_monitor.tsv` | Tabular monitor events |

## Evaluation

Use completed checkpoints as input. The exact benchmark data setup follows the
underlying benchmark and upstream evaluation tool requirements.

### BLIP3o Understanding

```bash
CHECKPOINT_DIR=/path/to/step_010000 \
NUM_GPUS=8 \
bash BLIP3o/eval/understanding_eval_our.sh
```

Override benchmark tasks when needed:

```bash
CHECKPOINT_DIR=/path/to/step_010000 \
TASKS="realworldqa,textvqa" \
bash BLIP3o/eval/understanding_eval_our.sh
```

### BLIP3o Generation

```bash
CHECKPOINT_DIR=/path/to/step_010000 \
bash BLIP3o/eval/geneval/generation_our.sh

CHECKPOINT_DIR=/path/to/step_010000 \
bash BLIP3o/eval/dpg_bench/generate_dpg_our.sh

CHECKPOINT_DIR=/path/to/step_010000 \
bash BLIP3o/eval/wise/generate_wise_our.sh
```

WISE scoring requires `OPENAI_API_KEY`.

### BAGEL Evaluation

Understanding benchmarks are launched through the BAGEL VLM evaluation
dispatcher:

```bash
bash Bagel/eval/vlm/evaluate.sh mmbench-dev-en
bash Bagel/eval/vlm/evaluate.sh mmmu-val
bash Bagel/eval/vlm/evaluate.sh mme
```

For generation benchmarks, follow the dependency and detector setup described
in `Bagel/EVAL.md`.

### VARGPT-v1.1 Evaluation

```bash
CHECKPOINT_DIR=/path/to/se_checkpoint_10000 \
NUM_GPUS=8 \
bash vargpt_1_1/understand_eval/understanding_eval_our.sh
```

Generation evaluation wrappers are under:

```text
vargpt_1_1/VARGPT-family-training/run_scripts/
```

For trained checkpoints, start with:

```bash
cd vargpt_1_1/VARGPT-family-training

CHECKPOINT_DIR=/path/to/se_checkpoint_10000 \
bash run_scripts/run_eval_vargpt_generation_our.sh
```

## Protocol Checks

Before launching full experiments, validate the static protocol and local run
readiness:

```bash
python scripts/self_evolving/paper/validate_protocol.py --format text

DATA_DIR=/path/to/unlabeled/images \
python scripts/self_evolving/paper/check_experiment_readiness.py --format text
```

Use `--strict` on the readiness checker when you want missing required assets to
return a non-zero exit code.

## Cluster Usage

On Slurm systems, enter an allocated shell before launching training or
evaluation:

```bash
srun --partition=gpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=96G \
  --time=3-00:00:00 \
  --pty bash
```

Then activate the appropriate conda environment and run the launcher.

## Citation

If this codebase is useful for your research, please cite:

```bibtex
@misc{thawkar2026asksolvegenerate,
  title = {Ask, Solve, Generate: Self-Evolving Unified Multimodal Understanding and Generation via Self-Consistency Rewards},
  author = {Thawkar, Ritesh and Venkatraman, Shravan and Thawakar, Omkar and Shaker, Abdelrahman M and Khan, Fahad Shahbaz and Cholakkal, Hisham and Khan, Salman and Anwer, Rao Muhammad},
  year = {2026},
  note = {Manuscript}
}
```

This entry intentionally omits arXiv, DOI, URL, and conference fields until the
public paper metadata is finalized.

## License

This repository is released under the Apache License 2.0. See `LICENSE`.

Third-party code and model integrations under `Bagel/`, `BLIP3o/`, and
`vargpt_1_1/` may retain their upstream licenses and usage terms. Check the
corresponding upstream projects and model cards before redistributing model
weights or derived checkpoints.

## Release Checklist

Before publishing a release branch, confirm that:

- `docs/`, private manuscripts, and reviewer materials are not tracked.
- `data_pipeline/` and private data-construction code are not tracked.
- Checkpoints, model weights, generated images, logs, and caches are not
  tracked.
- `.env`, API keys, Hugging Face tokens, and W&B credentials are not tracked.
- Paper links, project page links, model links, and BibTeX metadata are updated
  only after they are final.
