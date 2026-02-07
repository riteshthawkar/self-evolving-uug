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

SUITE="full"
FORWARD_ARGS=()

usage() {
  cat <<'USAGE'
Run multiple understanding experiment scripts.

Usage:
  bash self_evolving/scripts/understanding_experiments/90_run_all_understanding.sh \
    --suite core|full \
    --data_dir /path/to/images/train [common args...]

Suites:
  core: 00-03
  full: 00-07
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$SUITE" != "core" && "$SUITE" != "full" ]]; then
  echo "Invalid --suite '$SUITE'. Must be core|full." >&2
  exit 1
fi

LAUNCH_TS="$(date -u +%Y%m%d_%H%M%S)"
export LAUNCH_TS

declare -a SCRIPTS
if [[ "$SUITE" == "core" ]]; then
  SCRIPTS=(
    00_u00_main_method.sh
    01_u01_solver_samples.sh
    02_u02_solver_gamma.sh
    03_u03_proposer_update_freq.sh
  )
else
  SCRIPTS=(
    00_u00_main_method.sh
    01_u01_solver_samples.sh
    02_u02_solver_gamma.sh
    03_u03_proposer_update_freq.sh
    04_u04_entropy_band.sh
    05_u05_kl_sensitivity.sh
    06_u06_lora_capacity.sh
    07_u07_frozen_proposer_proxy.sh
  )
fi

echo "Running suite=$SUITE with LAUNCH_TS=$LAUNCH_TS"
for script in "${SCRIPTS[@]}"; do
  echo "========================================"
  echo "Launching $script"
  bash "$SCRIPT_DIR/$script" "${FORWARD_ARGS[@]}"
done

echo "Suite complete."
