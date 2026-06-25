#!/usr/bin/env bash
set -euo pipefail

# E7 -- rebuttal control: sequential understanding then generation.
#
# Stage 1: understanding-only training for 10k steps (E2 protocol) over a fixed
# 10k-image subset.
# Stage 2: generation-only training for another 10k steps (E3 protocol) over
# the exact same fixed 10k-image subset while the Proposer/Solver are frozen
# for generation-phase rewards. Because the trainer resumes by global step,
# Stage 2 runs from step_010000 to step_020000 by default.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

DATA_DIR="${DATA_DIR:-${TWO_STAGE_DATA_DIR:-$REPO_ROOT/data/joint_pool_10k/images}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/blip3o/E7_two_stage}"
STAGE1_STEPS="${STAGE1_STEPS:-10000}"
STAGE2_STEPS="${STAGE2_STEPS:-10000}"
TOTAL_STEPS="${TOTAL_STEPS:-$((STAGE1_STEPS + STAGE2_STEPS))}"
TWO_STAGE_IMAGE_SAMPLES="${TWO_STAGE_IMAGE_SAMPLES:-10000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_STAGE="${TRAIN_STAGE:-strict}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
ALLOW_EXISTING_STAGE2="${ALLOW_EXISTING_STAGE2:-1}"
ALLOW_SMALL_DATA="${ALLOW_SMALL_DATA:-0}"

count_images() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    echo 0
    return
  fi
  find "$root" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' -o -iname '*.tiff' \) \
    | wc -l | tr -d ' '
}

stage_from_name() {
  local base
  base="$(basename "$1")"
  base="${base#step_}"
  base="${base#checkpoint-}"
  base="${base//[^0-9]/}"
  if [[ -z "$base" ]]; then
    echo 0
  else
    echo "$((10#$base))"
  fi
}

latest_checkpoint_at_least() {
  local root="$1"
  local min_step="$2"
  local best=""
  local best_step=0
  local candidate
  shopt -s nullglob
  for candidate in "$root"/step_* "$root"/checkpoints/step_* "$root"/checkpoint-*; do
    [[ -d "$candidate" ]] || continue
    if [[ ! -e "$candidate/SAVE_OK" && ! -e "$candidate/trainer_state.pt" && ! -d "$candidate/model" ]]; then
      continue
    fi
    local step
    step="$(stage_from_name "$candidate")"
    if [[ "$step" -ge "$min_step" && "$step" -ge "$best_step" ]]; then
      best="$candidate"
      best_step="$step"
    fi
  done
  shopt -u nullglob
  if [[ -n "$best" ]]; then
    printf '%s\n' "$best"
  fi
}

echo "[E7] Two-stage rebuttal control"
echo "[E7]   Data:       $DATA_DIR"
echo "[E7]   Output:     $OUTPUT_DIR"
echo "[E7]   Stage 1:    0 -> $STAGE1_STEPS understanding-only steps"
echo "[E7]   Stage 2:    $STAGE1_STEPS -> $TOTAL_STEPS generation-only steps ($STAGE2_STEPS additional steps)"
echo "[E7]   Image set:  same first $TWO_STAGE_IMAGE_SAMPLES sorted images, shuffled with seed 42 in both stages"
echo "[E7]   GPUs:       $NPROC_PER_NODE"

mkdir -p "$OUTPUT_DIR"

image_count="$(count_images "$DATA_DIR")"
if [[ "$TWO_STAGE_IMAGE_SAMPLES" -gt 0 && "$image_count" -lt "$TWO_STAGE_IMAGE_SAMPLES" ]]; then
  if [[ "$ALLOW_SMALL_DATA" == "1" ]]; then
    echo "[E7] WARNING: DATA_DIR has $image_count images, but TWO_STAGE_IMAGE_SAMPLES=$TWO_STAGE_IMAGE_SAMPLES." >&2
    echo "[E7]          Continuing because ALLOW_SMALL_DATA=1; this is not paper protocol." >&2
  else
    echo "[E7] ERROR: DATA_DIR has $image_count images, but TWO_STAGE_IMAGE_SAMPLES=$TWO_STAGE_IMAGE_SAMPLES." >&2
    echo "[E7] Provide the 10k rebuttal pool via DATA_DIR or TWO_STAGE_DATA_DIR." >&2
    echo "[E7] For smoke tests only, set ALLOW_SMALL_DATA=1." >&2
    exit 1
  fi
fi

existing_final="$(latest_checkpoint_at_least "$OUTPUT_DIR" "$TOTAL_STEPS" || true)"
if [[ -n "$existing_final" && "$ALLOW_EXISTING_STAGE2" == "1" ]]; then
  echo "[E7] Found final checkpoint at or beyond $TOTAL_STEPS, skipping: $existing_final"
  exit 0
fi

if [[ "$RUN_STAGE1" == "1" ]]; then
  stage1_ckpt="$(latest_checkpoint_at_least "$OUTPUT_DIR" "$STAGE1_STEPS" || true)"
  if [[ -n "$stage1_ckpt" ]]; then
    echo "[E7] Stage 1 checkpoint already exists: $stage1_ckpt"
  else
    echo "[E7] Running Stage 1 with E2_understanding_only.sh"
    DATA_DIR="$DATA_DIR" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    TOTAL_STEPS="$STAGE1_STEPS" \
    MAX_IMAGES="$TWO_STAGE_IMAGE_SAMPLES" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    TRAIN_STAGE="$TRAIN_STAGE" \
    GENERATION_STEPS_PER_CYCLE=0 \
    UNDERSTANDING_STEPS_PER_CYCLE=5 \
      bash "$SCRIPT_DIR/E2_understanding_only.sh"
  fi
else
  echo "[E7] RUN_STAGE1=0, expecting an existing checkpoint under $OUTPUT_DIR"
fi

stage1_ckpt="$(latest_checkpoint_at_least "$OUTPUT_DIR" "$STAGE1_STEPS" || true)"
if [[ -z "$stage1_ckpt" ]]; then
  echo "[E7] ERROR: no Stage 1 checkpoint >= $STAGE1_STEPS found under $OUTPUT_DIR" >&2
  exit 1
fi

if [[ "$RUN_STAGE2" == "1" ]]; then
  echo "[E7] Running Stage 2 with E3_generation_only.sh"
  echo "[E7]   Resume: $stage1_ckpt"
  DATA_DIR="$DATA_DIR" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  RESUME_FROM="$stage1_ckpt" \
  RESET_PROPOSER_BASELINE="${RESET_PROPOSER_BASELINE:-1}" \
  TOTAL_STEPS="$TOTAL_STEPS" \
  MAX_IMAGES="$TWO_STAGE_IMAGE_SAMPLES" \
  NPROC_PER_NODE="$NPROC_PER_NODE" \
  TRAIN_STAGE="$TRAIN_STAGE" \
  PROPOSER_GEN_REWARD_ENABLED=0 \
  GENERATION_STEPS_PER_CYCLE=5 \
  UNDERSTANDING_STEPS_PER_CYCLE=0 \
    bash "$SCRIPT_DIR/E3_generation_only.sh"
else
  echo "[E7] RUN_STAGE2=0, stopping after Stage 1 checkpoint: $stage1_ckpt"
fi
