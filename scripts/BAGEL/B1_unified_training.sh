#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# B1 — BAGEL Unified Self-Evolving Training (final-style experiment launcher)
# ══════════════════════════════════════════════════════════════════════════════
#
# This wrapper keeps a BLIP3o-final style interface while delegating execution
# to the maintained BAGEL pipeline script:
#   Bagel/scripts/B1_unified_training.sh
#
# Usage:
#   TRAIN_STAGE=warmup bash scripts/BAGEL/B1_unified_training.sh
#   RESUME_FROM=/path/to/checkpoints/step_000500.pt bash scripts/BAGEL/B1_unified_training.sh
#
# Common overrides:
#   MODEL_PATH=/path/to/BAGEL-7B-MoT
#   DATA_DIR=/path/to/images
#   OUTPUT_DIR=/path/to/run_dir
#   STEPS=1500
#   RUN_MODE=train|rollout
#   EXPERIMENT=unified_self_evolving|understanding_self_evolving|generation_self_evolving
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR_USER_SET=0
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR_USER_SET=1
fi
source "$SCRIPT_DIR/_common.sh"

# Core experiment defaults (aligned with BLIP3o-style final runs).
RUN_NAME="${RUN_NAME:-B1_unified_training_s42}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"

DEVICE="${DEVICE:-cuda}"
MULTI_GPU_SPLIT="${MULTI_GPU_SPLIT:-off}"
MODEL_DEVICE_INDEX="${MODEL_DEVICE_INDEX:-0}"
VAE_DEVICE_INDEX="${VAE_DEVICE_INDEX:-1}"
VAE_DEVICE="${VAE_DEVICE:-}"
MAX_LATENT_SIZE="${MAX_LATENT_SIZE:-64}"

ENABLE_LORA="${ENABLE_LORA:-1}"
ENABLE_SUDER="${ENABLE_SUDER:-1}"
POLICY_UPDATE_METHOD="${POLICY_UPDATE_METHOD:-grpo}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
RESUME_FROM="${RESUME_FROM:-}"

TRAIN_UNDERSTANDING_PROPOSER="${TRAIN_UNDERSTANDING_PROPOSER:-1}"
TRAIN_SOLVER="${TRAIN_SOLVER:-1}"
TRAIN_GENERATION_PROPOSER="${TRAIN_GENERATION_PROPOSER:-1}"

UNDERSTANDING_STEPS_PER_CYCLE="${UNDERSTANDING_STEPS_PER_CYCLE:-3}"
GENERATION_STEPS_PER_CYCLE="${GENERATION_STEPS_PER_CYCLE:-2}"
PROPOSER_GEN_ENTROPY_WEIGHT="${PROPOSER_GEN_ENTROPY_WEIGHT:-0.7}"
PROPOSER_GEN_BASELINE_MOMENTUM="${PROPOSER_GEN_BASELINE_MOMENTUM:-0.6}"

# Conservative ROCm defaults (can be overridden by env vars).
DISABLE_FLASH_ATTN="${DISABLE_FLASH_ATTN:-1}"
DISABLE_AUTOCAST="${DISABLE_AUTOCAST:-1}"
ROCM_SAFE_MODE="${ROCM_SAFE_MODE:-1}"
FORCE_MATH_SDPA="${FORCE_MATH_SDPA:-1}"
BAGEL_COMPILE_BLOCK_MASK="${BAGEL_COMPILE_BLOCK_MASK:-0}"

if [[ "$TRAIN_STAGE" != "warmup" && "$TRAIN_STAGE" != "strict" ]]; then
  echo "[B1] ERROR: TRAIN_STAGE must be one of: warmup, strict (got: $TRAIN_STAGE)" >&2
  exit 1
fi

if [[ "$TRAIN_STAGE" == "warmup" ]]; then
  RUN_NAME="${RUN_NAME}_warmup"
  # Warmup keeps slightly lower stress on generation path.
  GEN_IMAGE_SIZE="${GEN_IMAGE_SIZE:-512}"
  GEN_NUM_TIMESTEPS="${GEN_NUM_TIMESTEPS:-24}"
else
  RUN_NAME="${RUN_NAME}_strict"
  GEN_IMAGE_SIZE="${GEN_IMAGE_SIZE:-640}"
  GEN_NUM_TIMESTEPS="${GEN_NUM_TIMESTEPS:-50}"
fi

if [[ "$OUTPUT_DIR_USER_SET" != "1" ]]; then
  OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
fi
bagel_preflight

echo "[B1] Starting BAGEL unified final-style experiment"
echo "[B1]   Stage:      $TRAIN_STAGE"
echo "[B1]   Run mode:   $RUN_MODE"
echo "[B1]   Exp:        $EXPERIMENT"
echo "[B1]   Model:      $MODEL_PATH"
echo "[B1]   Data:       $DATA_DIR"
echo "[B1]   Output:     $OUTPUT_DIR"
echo "[B1]   Steps:      $STEPS"
if [[ -n "$RESUME_FROM" ]]; then
  echo "[B1]   Resume:     $RESUME_FROM"
fi

export PYTHON_BIN
export MODEL_PATH
export DATA_DIR
export OUTPUT_DIR
export TRAIN_STAGE
export RUN_MODE
export EXPERIMENT
export STEPS
export DEVICE
export VAE_DEVICE
export MULTI_GPU_SPLIT
export MODEL_DEVICE_INDEX
export VAE_DEVICE_INDEX
export MAX_LATENT_SIZE
export ENABLE_LORA
export ENABLE_SUDER
export POLICY_UPDATE_METHOD
export CHECKPOINT_EVERY
export RESUME_FROM
export TRAIN_UNDERSTANDING_PROPOSER
export TRAIN_SOLVER
export TRAIN_GENERATION_PROPOSER
export UNDERSTANDING_STEPS_PER_CYCLE
export GENERATION_STEPS_PER_CYCLE
export PROPOSER_GEN_ENTROPY_WEIGHT
export PROPOSER_GEN_BASELINE_MOMENTUM
export DISABLE_FLASH_ATTN
export DISABLE_AUTOCAST
export ROCM_SAFE_MODE
export FORCE_MATH_SDPA
export BAGEL_COMPILE_BLOCK_MASK
export GEN_IMAGE_SIZE
export GEN_NUM_TIMESTEPS

bash "$BAGEL_ENTRY"
