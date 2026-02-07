#!/usr/bin/env bash
set -euo pipefail

# Weights & Biases environment variables (set before running if needed).
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-self-evolving-uug-understanding}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_LOG_IMAGES_EVERY="${WANDB_LOG_IMAGES_EVERY:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'USAGE'
U01: Solver sample-count ablation.

Runs:
  - u01_nsamples_3_s42
  - u01_nsamples_5_s42
  - u01_nsamples_7_s42
USAGE
  common_usage
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if ! parse_common_args "$@"; then
  usage
  exit 1
fi

EXP_ID="U01_solver_samples"
echo "Running $EXP_ID"
print_shared_config

run_case "$EXP_ID" "u01_nsamples_3_s42" --seed 42 --num_solver_samples 3
run_case "$EXP_ID" "u01_nsamples_5_s42" --seed 42 --num_solver_samples 5
run_case "$EXP_ID" "u01_nsamples_7_s42" --seed 42 --num_solver_samples 7
