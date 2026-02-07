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
U03: Proposer update-frequency ablation.

Runs:
  - u03_propfreq_1_s42
  - u03_propfreq_3_s42
  - u03_propfreq_5_s42
  - u03_propfreq_10_s42
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

EXP_ID="U03_proposer_update_freq"
echo "Running $EXP_ID"
print_shared_config

run_case "$EXP_ID" "u03_propfreq_1_s42" --seed 42 --proposer_update_freq 1
run_case "$EXP_ID" "u03_propfreq_3_s42" --seed 42 --proposer_update_freq 3
run_case "$EXP_ID" "u03_propfreq_5_s42" --seed 42 --proposer_update_freq 5
run_case "$EXP_ID" "u03_propfreq_10_s42" --seed 42 --proposer_update_freq 10
