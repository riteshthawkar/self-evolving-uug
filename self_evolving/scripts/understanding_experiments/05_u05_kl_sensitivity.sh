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
U05: KL-controller sensitivity ablation.

Runs:
  - u05_klcoef2e3_kltarget0p01_s42
  - u05_klcoef1e3_kltarget0p02_s42
  - u05_klcoef5e4_kltarget0p05_s42
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

EXP_ID="U05_kl_sensitivity"
echo "Running $EXP_ID"
print_shared_config

run_case "$EXP_ID" "u05_klcoef2e3_kltarget0p01_s42" --seed 42 --kl_coef 2e-3 --kl_target 0.01
run_case "$EXP_ID" "u05_klcoef1e3_kltarget0p02_s42" --seed 42 --kl_coef 1e-3 --kl_target 0.02
run_case "$EXP_ID" "u05_klcoef5e4_kltarget0p05_s42" --seed 42 --kl_coef 5e-4 --kl_target 0.05
