#!/usr/bin/env bash
set -euo pipefail

# Multi-benchmark understanding evaluation for VARGPT (lmms_eval).
#
# Usage:
#   PURE_MODEL_PATH=/path/to/pure_model \
#   TRAINED_MODEL_PATH=/path/to/trained_base_model \
#   TRAINED_LORA_PATH=/path/to/trained_lora_adapter \
#   TASKS=mmmu,mme,textvqa_val \
#   bash run_scripts/run_eval_vargpt_understanding_bench.sh
#
# Optional env:
#   MODEL                default: vargpt_qwen2vl_v1_1
#   EVAL_SETS            default: pure,trained_lora
#                        allowed tokens: pure,trained_lora
#   PURE_MODEL_PATH      base model path for pure evaluation set
#   PURE_MODEL_ARGS_EXTRA additional lmms model args for pure set
#   TRAINED_MODEL_PATH   base model path for trained set
#   TRAINED_LORA_PATH    optional LoRA adapter path for trained set
#   TRAINED_LORA_ADAPTER_NAME  adapter name to activate (default: solver)
#   TRAINED_MODEL_ARGS_EXTRA additional lmms model args for trained set
#   NUM_PROCESSES        default: 8
#   MAIN_PROCESS_PORT    default: 39535
#   BATCH_SIZE           default: 1
#   OUTPUT_ROOT          default: <train_root>/logs/understanding_eval/<timestamp>
#   LOG_SAMPLES          default: 1
#   LOG_SAMPLES_SUFFIX   default: vargpt_understanding
#   UNDERSTAND_EVAL_DIR  default: <repo>/suder_vargpt/understand_eval
#
# Legacy mode:
#   MODEL_PATH + MODEL_ARGS_EXTRA still works and runs only one set ("pure").

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SUDER_ROOT="$(cd -- "${TRAIN_ROOT}/.." && pwd)"
UNDERSTAND_EVAL_DIR="${UNDERSTAND_EVAL_DIR:-${SUDER_ROOT}/understand_eval}"

MODEL="${MODEL:-vargpt_qwen2vl_v1_1}"
TASKS="${TASKS:-mmmu}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-39535}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_SAMPLES="${LOG_SAMPLES:-1}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-vargpt_understanding}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_ROOT}/logs/understanding_eval/$(date +%Y%m%d_%H%M%S)}"

EVAL_SETS="${EVAL_SETS:-pure,trained_lora}"
PURE_MODEL_PATH="${PURE_MODEL_PATH:-}"
PURE_MODEL_ARGS_EXTRA="${PURE_MODEL_ARGS_EXTRA:-}"
TRAINED_MODEL_PATH="${TRAINED_MODEL_PATH:-}"
TRAINED_LORA_PATH="${TRAINED_LORA_PATH:-}"
TRAINED_LORA_ADAPTER_NAME="${TRAINED_LORA_ADAPTER_NAME:-solver}"
TRAINED_MODEL_ARGS_EXTRA="${TRAINED_MODEL_ARGS_EXTRA:-}"

MODEL_PATH="${MODEL_PATH:-}"
MODEL_ARGS_EXTRA="${MODEL_ARGS_EXTRA:-}"

if [[ ! -d "${UNDERSTAND_EVAL_DIR}" ]]; then
  echo "[ERROR] understand_eval dir not found: ${UNDERSTAND_EVAL_DIR}" >&2
  exit 1
fi

if [[ -n "${MODEL_PATH}" && -z "${PURE_MODEL_PATH}" && -z "${TRAINED_MODEL_PATH}" ]]; then
  EVAL_SETS="pure"
  PURE_MODEL_PATH="${MODEL_PATH}"
  PURE_MODEL_ARGS_EXTRA="${MODEL_ARGS_EXTRA}"
fi

build_model_args() {
  local base_path="$1"
  local extra_args="$2"
  local peft_path="${3:-}"
  local adapter_name="${4:-}"
  local model_args="pretrained=${base_path}"

  if [[ -n "${peft_path}" ]]; then
    model_args="${model_args},peft=${peft_path}"
    if [[ -n "${adapter_name}" ]]; then
      model_args="${model_args},peft_adapter_name=${adapter_name}"
    fi
  fi

  if [[ -n "${extra_args}" ]]; then
    model_args="${model_args},${extra_args}"
  fi
  echo "${model_args}"
}

run_eval_set() {
  local set_name="$1"
  local set_model_path="$2"
  local set_extra_args="$3"
  local set_peft_path="${4:-}"
  local set_adapter_name="${5:-}"

  if [[ -z "${set_model_path}" ]]; then
    echo "[ERROR] Missing model path for set '${set_name}'." >&2
    exit 1
  fi

  local set_output="${OUTPUT_ROOT}/${set_name}"
  local set_suffix="${LOG_SAMPLES_SUFFIX}_${set_name}"
  local set_model_args
  set_model_args="$(build_model_args "${set_model_path}" "${set_extra_args}" "${set_peft_path}" "${set_adapter_name}")"

  mkdir -p "${set_output}"

  echo "=== VARGPT Understanding Evaluation (${set_name}) ==="
  echo "  model:       ${MODEL}"
  echo "  model_path:  ${set_model_path}"
  if [[ -n "${set_peft_path}" ]]; then
    echo "  lora_path:   ${set_peft_path}"
    echo "  adapter:     ${set_adapter_name}"
  elif [[ "${set_name}" == "trained_lora" ]]; then
    echo "  note:        TRAINED_LORA_PATH is empty; evaluating trained set from base/merged model path only."
  fi
  echo "  tasks:       ${TASKS}"
  echo "  gpus:        ${NUM_PROCESSES}"
  echo "  output:      ${set_output}"

  local -a cmd=(
    python3 -m accelerate.commands.launch
    --num_processes "${NUM_PROCESSES}"
    --main_process_port "${MAIN_PROCESS_PORT}"
    -m lmms_eval
    --model "${MODEL}"
    --model_args "${set_model_args}"
    --tasks "${TASKS}"
    --batch_size "${BATCH_SIZE}"
    --output_path "${set_output}"
  )

  if [[ "${LOG_SAMPLES}" == "1" ]]; then
    cmd+=(--log_samples --log_samples_suffix "${set_suffix}")
  fi

  "${cmd[@]}"
}

mkdir -p "${OUTPUT_ROOT}"
cd "${UNDERSTAND_EVAL_DIR}"
export PYTHONPATH="${UNDERSTAND_EVAL_DIR}:${PYTHONPATH:-}"

IFS=',' read -r -a _eval_sets <<< "${EVAL_SETS}"
for raw_set in "${_eval_sets[@]}"; do
  set_name="$(echo "${raw_set}" | xargs)"
  [[ -z "${set_name}" ]] && continue

  case "${set_name}" in
    pure)
      run_eval_set "pure" "${PURE_MODEL_PATH}" "${PURE_MODEL_ARGS_EXTRA}"
      ;;
    trained_lora)
      run_eval_set "trained_lora" "${TRAINED_MODEL_PATH}" "${TRAINED_MODEL_ARGS_EXTRA}" "${TRAINED_LORA_PATH}" "${TRAINED_LORA_ADAPTER_NAME}"
      ;;
    *)
      echo "[ERROR] Unsupported EVAL_SETS token: '${set_name}'. Use pure,trained_lora." >&2
      exit 1
      ;;
  esac
done

echo "Done. Understanding eval sets saved under: ${OUTPUT_ROOT}"
