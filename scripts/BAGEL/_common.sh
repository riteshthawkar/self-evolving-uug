#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Common environment and defaults shared by BAGEL final-style experiments.
# Source this file from experiment launchers; do not run directly.
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Core BAGEL launch script (implementation entrypoint).
BAGEL_ENTRY="${BAGEL_ENTRY:-$REPO_ROOT/Bagel/scripts/B1_unified_training.sh}"

# Experiment defaults.
RUN_NAME="${RUN_NAME:-B1_unified_training}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/BAGEL-7B-MoT}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/joint_3k/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/runs/final_bagel}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"

TRAIN_STAGE="${TRAIN_STAGE:-strict}"               # warmup|strict
RUN_MODE="${RUN_MODE:-train}"                      # train|rollout
EXPERIMENT="${EXPERIMENT:-unified_self_evolving}" # unified/understanding/generation
STEPS="${STEPS:-1500}"

# Cache / runtime environment (same style as BLIP3o final scripts).
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache}"
CACHE_TMP_DIR="$CACHE_ROOT/tmp"
CACHE_TORCH_EXT_DIR="$CACHE_ROOT/torch_extensions"
CACHE_WANDB_DIR="$CACHE_ROOT/wandb"
CACHE_MIOPEN_DIR="$CACHE_ROOT/miopen"
CACHE_CUDA_DIR="$CACHE_ROOT/cuda"
mkdir -p \
  "$CACHE_ROOT" \
  "$CACHE_TMP_DIR" \
  "$CACHE_TORCH_EXT_DIR" \
  "$CACHE_WANDB_DIR" \
  "$CACHE_MIOPEN_DIR" \
  "$CACHE_CUDA_DIR" \
  "$CACHE_ROOT/assets"

export HF_HOME="$CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_ROOT"
export HF_ASSETS_CACHE="$CACHE_ROOT/assets"
export TRANSFORMERS_CACHE="$CACHE_ROOT"
export HF_DATASETS_CACHE="$CACHE_ROOT"
export HF_METRICS_CACHE="$CACHE_ROOT"
export TORCH_HOME="$CACHE_ROOT"
export TRITON_CACHE_DIR="$CACHE_ROOT"
export TORCH_EXTENSIONS_DIR="$CACHE_TORCH_EXT_DIR"
export XDG_CACHE_HOME="$CACHE_ROOT"
export TMPDIR="$CACHE_TMP_DIR"
export TMP="$CACHE_TMP_DIR"
export TEMP="$CACHE_TMP_DIR"
export WANDB_DIR="$CACHE_WANDB_DIR"
export WANDB_CACHE_DIR="$CACHE_WANDB_DIR"
export WANDB_CONFIG_DIR="$CACHE_WANDB_DIR"
export WANDB_DATA_DIR="$CACHE_WANDB_DIR"
export CUDA_CACHE_PATH="$CACHE_CUDA_DIR"
export MIOPEN_USER_DB_PATH="$CACHE_MIOPEN_DIR"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_MIOPEN_DIR"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

bagel_preflight() {
  if [[ ! -f "$BAGEL_ENTRY" ]]; then
    echo "[B1] ERROR: BAGEL entry script not found: $BAGEL_ENTRY" >&2
    return 1
  fi
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[B1] ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
    return 1
  fi
  if [[ ! -d "$DATA_DIR" ]]; then
    echo "[B1] ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
    return 1
  fi
  if ! find "$DATA_DIR" -type f \
      \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.bmp" \) \
      -print -quit | grep -q .; then
    echo "[B1] ERROR: DATA_DIR has no image files: $DATA_DIR" >&2
    return 1
  fi
  mkdir -p "$OUTPUT_DIR"
}

