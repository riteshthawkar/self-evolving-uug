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
U06: LoRA capacity ablation.

Runs:
  - u06_lorar8_alpha16_s42
  - u06_lorar16_alpha32_s42
  - u06_lorar32_alpha64_s42
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

EXP_ID="U06_lora_capacity"
echo "Running $EXP_ID"
print_shared_config

run_case "$EXP_ID" "u06_lorar8_alpha16_s42" --seed 42 --lora_r 8 --lora_alpha 16
run_case "$EXP_ID" "u06_lorar16_alpha32_s42" --seed 42 --lora_r 16 --lora_alpha 32
run_case "$EXP_ID" "u06_lorar32_alpha64_s42" --seed 42 --lora_r 32 --lora_alpha 64
