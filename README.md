<h1 align="center">
Ask, Solve, Generate: Self-Evolving Unified Multimodal Understanding and Generation via Self-Consistency Rewards
</h1>

<p align="center">
  <img alt="Paper" src="https://img.shields.io/badge/Paper-coming_soon-lightgrey.svg">
  <img alt="Project Page" src="https://img.shields.io/badge/Project_Page-coming_soon-lightgrey.svg">
  <img alt="Models" src="https://img.shields.io/badge/Models-coming_soon-fcd734.svg?logo=huggingface&logoColor=black">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-4caf50.svg"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white"></a>
</p>

<p align="center">
  Ritesh Thawkar · Shravan Venkatraman · Omkar Thawakar · Abdelrahman M Shaker ·
  Fahad Shahbaz Khan · Hisham Cholakkal · Salman Khan · Rao Muhammad Anwer
</p>

---

**Ask, Solve, Generate** studies whether unified multimodal models can improve
visual understanding and image generation from unlabeled images alone. The
training loop turns each image into self-generated questions, selects reliable
reference answers through self-consistency, and uses the same interaction signal
to update both understanding and generation components.

This repository contains the public training, inference, and evaluation code.
The release focuses on model-side implementation. Private data-construction
pipelines, manuscript source, local checkpoints, generated outputs, and
cluster-specific logs are intentionally not included.

## 📢 Announcement

- **[Release]** Public training, inference, and evaluation code is prepared for
  BLIP3o, BAGEL, and VARGPT-v1.1 backends.
- **[Coming soon]** Paper, project page, and released model checkpoints will be
  linked here after the public release metadata is finalized.

---

## 🔎 Overview

The self-evolving loop uses three internal roles:

| Role | Purpose |
| --- | --- |
| Proposer | Generates visual questions from an unlabeled image |
| Solver | Answers and scores candidate questions through self-consistency |
| Generator | Produces images from question-answer-derived generation specs |

The implementation uses internal consistency signals rather than external answer
labels. For understanding, Solver Token Entropy (STE) provides a token-level
uncertainty signal for selecting useful updates. For generation,
question-answer fidelity and cycle-consistent captioning connect generated
images back to the understanding loop.

The main implementation is built around BLIP3o. BAGEL and VARGPT-v1.1
integrations are included to evaluate the same self-evolving recipe on different
unified model families.

---

## ✨ Model Capabilities

| Capability | Input | Output | Description |
| --- | --- | --- | --- |
| Visual understanding | Image + question | Text answer | VQA, OCR-style reasoning, visual recognition, and multimodal reasoning |
| Image generation | Prompt or generated QA specification | Image | Text-to-image generation with self-supervised reward signals |
| Unified self-evolution | Unlabeled images | LoRA adapters | Joint improvement of proposer, solver, and generator roles |
| Multi-backbone training | BLIP3o, BAGEL, VARGPT-v1.1 | Backend-specific adapters | Same self-evolving recipe across different unified model families |

---

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture.png" alt="Ask, Solve, Generate architecture" width="100%">
</p>

The framework alternates understanding and generation steps. In the
understanding step, the Proposer creates image-grounded questions, the Solver
answers perturbed prompts, and self-consistency plus token-level entropy provide
training signals. In the generation step, question-answer-derived generation
specifications guide image synthesis, while the Solver evaluates QA fidelity and
cycle consistency.

---

## ✅ Supported Tasks

| Task | Backend support | Primary scripts |
| --- | --- | --- |
| BLIP3o self-evolving training | BLIP3o-8B | `scripts/E1_main_joint.sh` |
| BLIP3o ablations | BLIP3o-8B | `scripts/E2_understanding_only.sh` to `scripts/E7_two_stage.sh` |
| BAGEL self-evolving training | BAGEL | `Bagel/scripts/B1_unified_training.sh` |
| VARGPT-v1.1 self-evolving training | VARGPT-v1.1 | `vargpt_1_1/VARGPT-family-training/examples/train_self_evolving/run_self_evolving.sh` |
| Understanding evaluation | BLIP3o, BAGEL, VARGPT-v1.1 | `BLIP3o/eval/`, `Bagel/eval/vlm/`, `vargpt_1_1/understand_eval/` |
| Generation evaluation | BLIP3o, BAGEL, VARGPT-v1.1 | GenEval, DPG-Bench, and WISE wrappers under each backend |

---

## 🧩 Framework Components

The release combines:

- **Self-questioning** over unlabeled images through a proposer model.
- **Reference-answer selection** through solver self-consistency and entropy
  signals.
- **Generation supervision** from question-answer-derived image specifications.
- **Cycle consistency** that checks generated images against the original
  multimodal interaction.
- **Backend adapters** for BLIP3o, BAGEL, and VARGPT-v1.1.

---

## 📊 Main Results

Results are reported as base checkpoint -> self-evolved checkpoint under
matched evaluation settings. MME-P and MME-C use the raw MME perception and
cognition scores; other understanding metrics are percentages.

### 🧠 Visual Understanding

| Backbone | MMMU | MMBench | TextVQA | SEED | RWQA | MMVet | MME-P | MME-C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BLIP3o-8B | 50.6 -> 52.8 | 83.5 -> 86.1 | 83.1 -> 85.2 | 77.5 -> 79.4 | 69.0 -> 70.9 | 66.6 -> 68.7 | 1682.6 -> 1698.4 | 647.1 -> 660.3 |
| BAGEL | 55.3 -> 58.8 | 85.0 -> 87.1 | 86.0 -> 88.5 | 79.3 -> 81.8 | 71.2 -> 73.9 | 67.2 -> 69.5 | 1687.0 -> 1701.7 | 701.0 -> 715.9 |
| VARGPT-v1.1 | 48.6 -> 51.6 | 81.0 -> 83.7 | 82.0 -> 84.8 | 76.1 -> 79.2 | 67.5 -> 71.1 | 51.9 -> 54.0 | 1678.3 -> 1695.7 | 592.9 -> 606.4 |

### 🎨 Image Generation

GenEval scores are percentages.

| Backbone | Single Obj. | Two Obj. | Counting | Colors | Position | Color Attr. | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BLIP3o-8B | 100 -> 99 | 85 -> 93 | 63 -> 71 | 92 -> 94 | 90 -> 90 | 74 -> 75 | 84 -> 87 |
| BAGEL | 99 -> 99 | 94 -> 95 | 81 -> 87 | 88 -> 90 | 64 -> 67 | 63 -> 72 | 82 -> 85 |
| VARGPT-v1.1 | 96 -> 97 | 53 -> 59 | 48 -> 56 | 83 -> 85 | 13 -> 15 | 21 -> 24 | 53 -> 56 |

---

## 🗂️ Repository Layout

| Path | Purpose |
| --- | --- |
| `BLIP3o/` | BLIP3o training, inference, and evaluation code |
| `Bagel/` | BAGEL baseline integration and self-evolving launchers |
| `vargpt_1_1/` | VARGPT-v1.1 baseline integration and self-evolving launchers |
| `scripts/` | BLIP3o experiment launchers and release utilities |
| `.env.example` | Template for local secrets and cache paths |

Generated outputs are ignored by git. Keep datasets, model weights,
checkpoints, logs, and caches outside the tracked source tree or under ignored
paths such as `data/`, `models/`, `outputs/`, and `logs/`.

---

## ⚙️ Setup

Use separate environments for the three backends. The dependency stacks differ,
and mixing them in one environment can create version conflicts.

### 🔧 BLIP3o

Install a PyTorch build that matches your machine first, for example the
CUDA or ROCm wheel recommended by your cluster. Then install the BLIP3o
dependencies and local packages:

```bash
conda create -n uug-blip3o python=3.10 -y
conda activate uug-blip3o
# Install torch/torchvision/torchaudio for your CUDA or ROCm stack first.
pip install -r BLIP3o/requirements.safe.txt
pip install -e BLIP3o
pip install -e BLIP3o/eval/lmms-eval
```

`BLIP3o/requirements.safe.txt` intentionally avoids reinstalling PyTorch,
`xformers`, `flash-attn`, `deepspeed`, and `bitsandbytes`. If you want the
full upstream CUDA stack instead, use:

```bash
pip install -r BLIP3o/requirements.txt
pip install -e BLIP3o
pip install -e BLIP3o/eval/lmms-eval
```

### 🔧 BAGEL

```bash
conda create -n uug-bagel python=3.10 -y
conda activate uug-bagel
pip install -r Bagel/requirements.txt
```

Generation benchmark scoring may also require benchmark-specific detector
assets and setup from `Bagel/EVAL.md`.

### 🔧 VARGPT-v1.1

```bash
conda create -n uug-vargpt python=3.10 -y
conda activate uug-vargpt
cd vargpt_1_1/VARGPT-family-training
pip install -r requirements.txt
pip install -e .
cd ../..
```

For VARGPT understanding evaluation, install the local evaluation package when
needed:

```bash
pip install -e vargpt_1_1/understand_eval
```

The standalone upstream inference examples under `vargpt_1_1/inference_v1_1`
use `vargpt_1_1/requirements.txt`, which pins a PyTorch/flash-attn stack. Use
that file in a separate environment if it conflicts with the training setup.

---

## 🚀 Quick Start

After installing the BLIP3o environment, run a small launcher smoke test on any
local folder of images:

```bash
cp .env.example .env

TOTAL_STEPS=20 \
ALLOW_SMALL_DATA=1 \
DATA_DIR=/path/to/small/image/folder \
OUTPUT_DIR=$PWD/outputs/smoke/blip3o \
bash scripts/E1_main_joint.sh
```

For a full BLIP3o run, point `DATA_DIR` to the 10k unlabeled image pool and
remove the smoke-test overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
DATA_DIR=/path/to/unlabeled/images \
OUTPUT_DIR=$PWD/outputs/blip3o/E1_main_joint \
bash scripts/E1_main_joint.sh
```

---

## 🔐 Environment Variables

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

---

## 🤗 Model Zoo

Public model links will be added after release. Until then, the training and
evaluation scripts expect local checkpoint paths.

| Backend | Checkpoint status | Notes |
| --- | --- | --- |
| BLIP3o | Coming soon | Main implementation for self-evolving unified training |
| BAGEL | Coming soon | Baseline integration for the same training recipe |
| VARGPT-v1.1 | Coming soon | 7B+2B baseline integration |

---

## 🔮 Inference

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

---

## 🏋️ Training

BLIP3o training launchers are under `scripts/`. BAGEL and VARGPT-v1.1 use their
backend-native launchers, shown below.

Common BLIP3o launchers:

| Script | Purpose |
| --- | --- |
| `scripts/E1_main_joint.sh` | Main joint 3U:2G training run |
| `scripts/E2_understanding_only.sh` | Understanding-only ablation |
| `scripts/E3_generation_only.sh` | Generation-only ablation |
| `scripts/E4_no_dit_rwr.sh` | Joint run without DiT reward-weighted regression |
| `scripts/E5_synthetic_loop.sh` | Generated-only loop control |
| `scripts/E6_single_step.sh` | Generation-centered unified-step ablation |
| `scripts/E7_two_stage.sh` | Understanding stage followed by generation stage |

### ▶️ Direct BLIP3o Launcher

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
DATA_DIR=/path/to/unlabeled/images \
OUTPUT_DIR=$PWD/outputs/blip3o/E1_main_joint \
bash scripts/E1_main_joint.sh
```

For a short smoke test:

```bash
TOTAL_STEPS=20 \
ALLOW_SMALL_DATA=1 \
DATA_DIR=/path/to/small/image/folder \
OUTPUT_DIR=$PWD/outputs/smoke/blip3o \
bash scripts/E1_main_joint.sh
```

### ▶️ BAGEL Launcher

```bash
MODEL_PATH=/path/to/BAGEL-7B-MoT \
DATA_DIR=/path/to/unlabeled/images \
OUTPUT_DIR=$PWD/outputs/bagel/B1_unified_training \
NPROC_PER_NODE=8 \
bash Bagel/scripts/B1_unified_training.sh
```

`MODEL_PATH` must point to the local BAGEL model checkout or checkpoint
directory.

### ▶️ VARGPT-v1.1 Launcher

```bash
cd vargpt_1_1/VARGPT-family-training

IMAGE_FOLDER=/path/to/unlabeled/images \
OUTPUT_DIR=/path/to/output/vargpt/joint \
bash examples/train_self_evolving/run_self_evolving.sh joint 8
```

Available VARGPT modes are `joint`, `u_only`, and `gen_only`.

---

## ♻️ Resume and Monitoring

BLIP3o checkpoints are saved under the configured `OUTPUT_DIR` as
`step_NNNNN` directories. A checkpoint is complete when it contains `SAVE_OK`.

```bash
RESUME_FROM=/path/to/output_dir \
bash scripts/E1_main_joint.sh

RESUME_FROM=/path/to/output_dir/step_010000 \
bash scripts/E1_main_joint.sh
```

Typical run files:

| Path | Contents |
| --- | --- |
| `status.json` | Latest progress, phase, checkpoint, and error state |
| `iter_log.jsonl` | Per-step structured records |
| `logs/training_watch.log` | Human-readable training trace |
| `logs/training_monitor.jsonl` | Structured monitor events |
| `logs/training_monitor.tsv` | Tabular monitor events |

---

## 📏 Evaluation

Use completed checkpoints as input. The exact benchmark data setup follows the
underlying benchmark and upstream evaluation tool requirements.

### 🧠 BLIP3o Understanding

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

### 🎨 BLIP3o Generation

```bash
CHECKPOINT_DIR=/path/to/step_010000 \
bash BLIP3o/eval/geneval/generation_our.sh

CHECKPOINT_DIR=/path/to/step_010000 \
bash BLIP3o/eval/dpg_bench/generate_dpg_our.sh

CHECKPOINT_DIR=/path/to/step_010000 \
bash BLIP3o/eval/wise/generate_wise_our.sh
```

WISE scoring requires `OPENAI_API_KEY`.

### 📋 BAGEL Evaluation

Understanding benchmarks are launched through the BAGEL VLM evaluation
dispatcher:

```bash
bash Bagel/eval/vlm/evaluate.sh mmbench-dev-en
bash Bagel/eval/vlm/evaluate.sh mmmu-val
bash Bagel/eval/vlm/evaluate.sh mme
```

For generation benchmarks, follow the dependency and detector setup described
in `Bagel/EVAL.md`.

### 📋 VARGPT-v1.1 Evaluation

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

---

## 📚 Citation

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

---

## 🙏 Acknowledgements

This repository builds on BLIP3o, BAGEL, VARGPT-v1.1, lmms-eval, GenEval,
DPG-Bench, and WISE evaluation tooling. We thank the authors and maintainers of
these projects for releasing their code and models.

---

## 📄 License

This repository is released under the Apache License 2.0. See `LICENSE`.

Third-party code and model integrations under `Bagel/`, `BLIP3o/`, and
`vargpt_1_1/` may retain their upstream licenses and usage terms. Check the
corresponding upstream projects and model cards before redistributing model
weights or derived checkpoints.
