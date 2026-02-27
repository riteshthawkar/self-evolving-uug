#!/usr/bin/env bash
set -euo pipefail

# Multi-benchmark understanding evaluation for VARGPT (lmms_eval).
#
# Usage:
#   MODEL_PATH=/path/to/merged_or_hf_model \
#   TASKS=mmmu,mme,textvqa_val \
#   bash run_scripts/run_eval_vargpt_understanding_bench.sh
#
# Optional env:
#   MODEL                default: vargpt_qwen2vl_v1_1
#   NUM_PROCESSES        default: 8
#   MAIN_PROCESS_PORT    default: 39535
#   BATCH_SIZE           default: 1
#   OUTPUT_PATH          default: <train_root>/logs/understanding_eval/<timestamp>
#   LOG_SAMPLES          default: 1
#   LOG_SAMPLES_SUFFIX   default: vargpt_understanding
#   MODEL_ARGS_EXTRA     default: "" (comma-separated lmms_eval model args)
#   UNDERSTAND_EVAL_DIR  default: <repo>/suder_vargpt/understand_eval

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SUDER_ROOT="$(cd -- "${TRAIN_ROOT}/.." && pwd)"
UNDERSTAND_EVAL_DIR="${UNDERSTAND_EVAL_DIR:-${SUDER_ROOT}/understand_eval}"

MODEL="${MODEL:-vargpt_qwen2vl_v1_1}"
MODEL_PATH="${MODEL_PATH:-}"
TASKS="${TASKS:-mmmu}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-39535}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_SAMPLES="${LOG_SAMPLES:-1}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-vargpt_understanding}"
MODEL_ARGS_EXTRA="${MODEL_ARGS_EXTRA:-}"
OUTPUT_PATH="${OUTPUT_PATH:-${TRAIN_ROOT}/logs/understanding_eval/$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH is required." >&2
  echo "Example: MODEL_PATH=/path/to/model bash run_scripts/run_eval_vargpt_understanding_bench.sh" >&2
  exit 1
fi

if [[ ! -d "${UNDERSTAND_EVAL_DIR}" ]]; then
  echo "[ERROR] understand_eval dir not found: ${UNDERSTAND_EVAL_DIR}" >&2
  exit 1
fi

MODEL_ARGS="pretrained=${MODEL_PATH}"
if [[ -n "${MODEL_ARGS_EXTRA}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},${MODEL_ARGS_EXTRA}"
fi

mkdir -p "${OUTPUT_PATH}"
cd "${UNDERSTAND_EVAL_DIR}"

export PYTHONPATH="${UNDERSTAND_EVAL_DIR}:${PYTHONPATH:-}"

echo "=== VARGPT Understanding Evaluation ==="
echo "  model:      ${MODEL}"
echo "  model_path: ${MODEL_PATH}"
echo "  tasks:      ${TASKS}"
echo "  gpus:       ${NUM_PROCESSES}"
echo "  output:     ${OUTPUT_PATH}"

CMD=(
  python3 -m accelerate.commands.launch
  --num_processes "${NUM_PROCESSES}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  -m lmms_eval
  --model "${MODEL}"
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${OUTPUT_PATH}"
)

if [[ "${LOG_SAMPLES}" == "1" ]]; then
  CMD+=(--log_samples --log_samples_suffix "${LOG_SAMPLES_SUFFIX}")
fi

"${CMD[@]}"

echo "Done. Results: ${OUTPUT_PATH}"
