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
U07: Proposer-learning ablation (proxy frozen proposer).

Run:
  - u07_frozen_proposer_proxy_s42

Method:
  proposer_update_freq is set to total_steps + 1, so proposer adapter is never updated.
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

EXP_ID="U07_frozen_proposer_proxy"
FROZEN_FREQ=$((TOTAL_STEPS + 1))

echo "Running $EXP_ID"
print_shared_config

echo "Using proposer_update_freq=$FROZEN_FREQ (total_steps=$TOTAL_STEPS)"
run_case "$EXP_ID" "u07_frozen_proposer_proxy_s42" --seed 42 --proposer_update_freq "$FROZEN_FREQ"
