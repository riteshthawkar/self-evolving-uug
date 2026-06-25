#!/usr/bin/env bash
set -euo pipefail

# E7 rebuttal experiment: true two-stage sequential training.
#
# This script is intentionally explicit for reproducibility and rebuttal use:
#
#   Stage 1: 10,000 understanding-only training steps
#            on the same fixed 10,000-image subset.
#
#   Stage 2: 10,000 additional generation-only training steps
#            resumed from the Stage-1 step_010000 checkpoint
#            on the same fixed 10,000-image subset.
#
# The trainer uses global step numbering, so Stage 2 runs from step_010001
# through step_020000. Therefore TOTAL_STEPS must be 20,000, not 10,000.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

# Rebuttal control data: both stages use the exact same deterministic subset.
# ImagePool sorts image paths, truncates to MAX_IMAGES, then shuffles with seed 42.
export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/joint_pool_10k/images}"
export TWO_STAGE_IMAGE_SAMPLES="${TWO_STAGE_IMAGE_SAMPLES:-10000}"

# True 10k + 10k schedule.
export STAGE1_STEPS=10000
export STAGE2_STEPS=10000
export TOTAL_STEPS=20000

# Keep outputs separate from earlier 5k+5k or pilot controls.
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/blip3o/E7_two_stage_10kU_10kG}"

echo "[E7-10kU-10kG] Launching true two-stage rebuttal experiment"
echo "[E7-10kU-10kG]   Data dir:       $DATA_DIR"
echo "[E7-10kU-10kG]   Image samples:  $TWO_STAGE_IMAGE_SAMPLES"
echo "[E7-10kU-10kG]   Stage 1 steps:  $STAGE1_STEPS understanding-only"
echo "[E7-10kU-10kG]   Stage 2 steps:  $STAGE2_STEPS generation-only"
echo "[E7-10kU-10kG]   Total steps:    $TOTAL_STEPS global steps"
echo "[E7-10kU-10kG]   Output dir:     $OUTPUT_DIR"

exec bash "$SCRIPT_DIR/E7_two_stage.sh"
