#!/usr/bin/env bash
set -euo pipefail

# Shared repo environment bootstrap.
BOOTSTRAP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_SEARCH_DIR="${BOOTSTRAP_DIR}"
while [[ "${BOOTSTRAP_SEARCH_DIR}" != "/" ]]; do
  if [[ -f "${BOOTSTRAP_SEARCH_DIR}/scripts/env/bootstrap_training_env.sh" ]]; then
    # shellcheck source=/dev/null
    source "${BOOTSTRAP_SEARCH_DIR}/scripts/env/bootstrap_training_env.sh"
    break
  fi
  BOOTSTRAP_SEARCH_DIR="$(dirname "${BOOTSTRAP_SEARCH_DIR}")"
done
unset BOOTSTRAP_DIR BOOTSTRAP_SEARCH_DIR

# ─── Understanding evaluation for base VARGPT model ─────────────────────────
#
#   bash understanding_eval.sh
#   BASE_MODEL=VARGPT-family/VARGPT-v1.1 TASKS="realworldqa,textvqa,gqa" bash understanding_eval.sh
#   NUM_GPUS=8 bash understanding_eval.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUDER_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TRAIN_ROOT="${SUDER_ROOT}/VARGPT-family-training"
EVAL_SCRIPT="${TRAIN_ROOT}/run_scripts/run_eval_vargpt_understanding_bench.sh"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "[ERROR] Eval script not found: ${EVAL_SCRIPT}" >&2
  exit 1
fi

# ─── Configuration (override via env) ───────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-VARGPT-family/VARGPT-v1.1}"
TASKS="${TASKS:-realworldqa,textvqa}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-39535}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_ROOT}/logs/understanding_eval}"
LOG_SAMPLES="${LOG_SAMPLES:-1}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-vargpt_base}"
MODEL="${MODEL:-vargpt_qwen2vl_v1_1}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
fi
export CUDA_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "============================================"
echo "Understanding Evaluation (Base VARGPT)"
echo "  Base model:      ${BASE_MODEL}"
echo "  Tasks:           ${TASKS}"
echo "  Output:          ${OUTPUT_DIR}"
echo "  Num GPUs:        ${NUM_GPUS}"
echo "============================================"

EVAL_SETS="pure" \
PURE_MODEL_PATH="${BASE_MODEL}" \
MODEL="${MODEL}" \
TASKS="${TASKS}" \
NUM_PROCESSES="${NUM_GPUS}" \
BATCH_SIZE="${BATCH_SIZE}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}" \
OUTPUT_ROOT="${OUTPUT_DIR}" \
LOG_SAMPLES="${LOG_SAMPLES}" \
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX}" \
bash "${EVAL_SCRIPT}"
