#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# B1 — BAGEL Self-Evolving Understanding Rollout
# ══════════════════════════════════════════════════════════════════════════════
#
# Purpose:
#   Run the first BAGEL self-evolving experiment (understanding loop) using:
#   - proposer question generation from image
#   - multi-sample solver self-consistency
#   - greedy intuitive solver pass
#   - dual-track proposer reward logging
#
# Usage:
#   MODEL_PATH=/path/to/BAGEL-7B-MoT \
#   DATA_DIR=/path/to/images \
#   bash scripts/BAGEL/B1_understanding_rollout.sh
#
# Optional:
#   TRAIN_STAGE=warmup|strict
#   STEPS=500
#   DEVICE=cuda
#   OUTPUT_DIR=/custom/output
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/BAGEL-7B-MoT}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/joint_3k/images}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/BAGEL/B1_understanding_rollout}"

TRAIN_STAGE="${TRAIN_STAGE:-strict}"
STEPS="${STEPS:-500}"
DEVICE="${DEVICE:-cuda}"
MAX_LATENT_SIZE="${MAX_LATENT_SIZE:-64}"

# ── Stage-specific hyperparameters ──────────────────────────────────────────
if [[ "$TRAIN_STAGE" == "warmup" ]]; then
  STAGE_ARGS=(
    --disable_proposer_require_objective
    --disable_acceptance_require_non_easy
    --proposer_non_objective_penalty 0.0
    --rejected_question_penalty 0.25
    --proposer_entropy_mu 0.90
    --proposer_entropy_sigma 0.30
    --proposer_temperature 0.90
    --num_solver_samples 5
    --solver_temp_min 0.70
    --solver_temp_max 2.00
  )
elif [[ "$TRAIN_STAGE" == "strict" ]]; then
  STAGE_ARGS=(
    --proposer_require_objective
    --acceptance_require_non_easy
    --proposer_non_objective_penalty 0.20
    --rejected_question_penalty 0.35
    --proposer_entropy_mu 0.90
    --proposer_entropy_sigma 0.25
    --proposer_temperature 1.00
    --num_solver_samples 7
    --solver_temp_min 0.50
    --solver_temp_max 2.50
  )
else
  echo "[B1] ERROR: TRAIN_STAGE must be one of: warmup, strict (got: $TRAIN_STAGE)" >&2
  exit 1
fi

# ── Shared arguments ────────────────────────────────────────────────────────
SHARED_ARGS=(
  --max_new_tokens_proposer 256
  --max_new_tokens_solver 96
  --solver_unsolvable_maj_threshold 0.20
  --zero_entropy_eps 1e-6
  --seed 42
  --log_every 10
  --save_raw_generations
)

# ── Pre-flight checks ───────────────────────────────────────────────────────
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[B1] ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/llm_config.json" ]]; then
  echo "[B1] ERROR: Missing llm_config.json in MODEL_PATH: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/vit_config.json" ]]; then
  echo "[B1] ERROR: Missing vit_config.json in MODEL_PATH: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/ae.safetensors" ]]; then
  echo "[B1] ERROR: Missing ae.safetensors in MODEL_PATH: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/ema.safetensors" && ! -f "$MODEL_PATH/model.safetensors" ]]; then
  echo "[B1] ERROR: Missing ema.safetensors/model.safetensors in MODEL_PATH: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[B1] ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
  exit 1
fi

if ! find "$DATA_DIR" -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.bmp" \) \
  -print -quit | grep -q .; then
  echo "[B1] ERROR: DATA_DIR has no image files: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[B1] Starting BAGEL self-evolving rollout"
echo "[B1]   Stage:      $TRAIN_STAGE"
echo "[B1]   Model:      $MODEL_PATH"
echo "[B1]   Data:       $DATA_DIR"
echo "[B1]   Output:     $OUTPUT_DIR"
echo "[B1]   Steps:      $STEPS"
echo "[B1]   Device:     $DEVICE"

# ── Launch ──────────────────────────────────────────────────────────────────
cd "$REPO_ROOT/Bagel"
"$PYTHON_BIN" train/train_self_evolving.py \
  --model_path "$MODEL_PATH" \
  --device "$DEVICE" \
  --max_latent_size "$MAX_LATENT_SIZE" \
  --image_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  "${SHARED_ARGS[@]}" \
  "${STAGE_ARGS[@]}"

echo "[B1] Completed."

