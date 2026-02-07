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
U02: Solver continuous-reward softness (gamma) ablation.

Runs:
  - u02_gamma_0p5_s42
  - u02_gamma_0p7_s42
  - u02_gamma_1p0_s42
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

EXP_ID="U02_solver_gamma"
echo "Running $EXP_ID"
print_shared_config

run_case "$EXP_ID" "u02_gamma_0p5_s42" --seed 42 --solver_soft_gamma 0.5
run_case "$EXP_ID" "u02_gamma_0p7_s42" --seed 42 --solver_soft_gamma 0.7
run_case "$EXP_ID" "u02_gamma_1p0_s42" --seed 42 --solver_soft_gamma 1.0
