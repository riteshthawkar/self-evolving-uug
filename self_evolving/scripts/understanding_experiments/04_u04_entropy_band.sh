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
U04: Proposer entropy-band reward ablation (mu, sigma).

Runs:
  - u04_mu0p70_sigma0p25_s42
  - u04_mu0p70_sigma0p35_s42
  - u04_mu0p90_sigma0p25_s42
  - u04_mu0p90_sigma0p35_s42
  - u04_mu1p10_sigma0p25_s42
  - u04_mu1p10_sigma0p35_s42
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

EXP_ID="U04_entropy_band"
echo "Running $EXP_ID"
print_shared_config

run_case "$EXP_ID" "u04_mu0p70_sigma0p25_s42" --seed 42 --prop_entropy_mu 0.70 --prop_entropy_sigma 0.25
run_case "$EXP_ID" "u04_mu0p70_sigma0p35_s42" --seed 42 --prop_entropy_mu 0.70 --prop_entropy_sigma 0.35
run_case "$EXP_ID" "u04_mu0p90_sigma0p25_s42" --seed 42 --prop_entropy_mu 0.90 --prop_entropy_sigma 0.25
run_case "$EXP_ID" "u04_mu0p90_sigma0p35_s42" --seed 42 --prop_entropy_mu 0.90 --prop_entropy_sigma 0.35
run_case "$EXP_ID" "u04_mu1p10_sigma0p25_s42" --seed 42 --prop_entropy_mu 1.10 --prop_entropy_sigma 0.25
run_case "$EXP_ID" "u04_mu1p10_sigma0p35_s42" --seed 42 --prop_entropy_mu 1.10 --prop_entropy_sigma 0.35
