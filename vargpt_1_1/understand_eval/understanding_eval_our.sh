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

# ─── Understanding evaluation for self-evolving trained VARGPT (LoRA) ──────
#
# Usage:
#   CHECKPOINT_DIR=/path/to/se_checkpoint_2000 bash understanding_eval_our.sh
#   CHECKPOINT_DIR=/path/to/se_checkpoint_2000 ADAPTER=solver bash understanding_eval_our.sh
#   CHECKPOINT_DIR=/path/to/se_checkpoint_2000 NUM_GPUS=8 TASKS="realworldqa,textvqa,gqa" bash understanding_eval_our.sh
#
# CHECKPOINT_DIR can be:
#   - .../se_checkpoint_<n>
#   - .../se_checkpoint_<n>/model
#   - adapter folder directly

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
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Please set CHECKPOINT_DIR to your self-evolving checkpoint path}"
ADAPTER="${ADAPTER:-solver}"
TASKS="${TASKS:-realworldqa,textvqa}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-39535}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_ROOT}/logs/understanding_eval_our}"
LOG_SAMPLES="${LOG_SAMPLES:-1}"
MODEL="${MODEL:-vargpt_qwen2vl_v1_1}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
fi
export CUDA_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ ! -e "${CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] CHECKPOINT_DIR does not exist: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

ARTIFACT_DIR="${CHECKPOINT_DIR}"
if [[ -d "${CHECKPOINT_DIR}/model" ]]; then
  ARTIFACT_DIR="${CHECKPOINT_DIR}/model"
fi

TRAINED_MODEL_PATH="${TRAINED_MODEL_PATH:-}"
TRAINED_LORA_PATH="${TRAINED_LORA_PATH:-}"

if [[ -z "${TRAINED_MODEL_PATH}" && -z "${TRAINED_LORA_PATH}" ]]; then
  if [[ -f "${ARTIFACT_DIR}/config.json" ]]; then
    TRAINED_MODEL_PATH="${ARTIFACT_DIR}"
  elif [[ -f "${ARTIFACT_DIR}/${ADAPTER}/adapter_config.json" ]]; then
    TRAINED_LORA_PATH="${ARTIFACT_DIR}/${ADAPTER}"
  elif [[ -f "${ARTIFACT_DIR}/adapter_config.json" ]]; then
    TRAINED_LORA_PATH="${ARTIFACT_DIR}"
  elif [[ -f "${ARTIFACT_DIR}/default/adapter_config.json" ]]; then
    TRAINED_LORA_PATH="${ARTIFACT_DIR}/default"
  else
    echo "[ERROR] Could not find config.json or adapter_config.json under: ${ARTIFACT_DIR}" >&2
    exit 1
  fi
fi

CKPT_NAME="$(basename "${CHECKPOINT_DIR}")"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-vargpt_our_${ADAPTER}_${CKPT_NAME}}"

echo "============================================"
echo "Understanding Evaluation (Self-Evolving VARGPT)"
echo "  Base model:      ${BASE_MODEL}"
echo "  Checkpoint dir:  ${CHECKPOINT_DIR}"
echo "  Artifact dir:    ${ARTIFACT_DIR}"
if [[ -n "${TRAINED_MODEL_PATH}" ]]; then
  echo "  Model path:      ${TRAINED_MODEL_PATH}"
fi
if [[ -n "${TRAINED_LORA_PATH}" ]]; then
  echo "  LoRA path:       ${TRAINED_LORA_PATH}"
  echo "  Adapter:         ${ADAPTER}"
fi
echo "  Tasks:           ${TASKS}"
echo "  Output:          ${OUTPUT_DIR}"
echo "  Num GPUs:        ${NUM_GPUS}"
echo "  Log suffix:      ${LOG_SAMPLES_SUFFIX}"
echo "============================================"

EVAL_SETS="trained_lora" \
TRAINED_MODEL_PATH="${TRAINED_MODEL_PATH}" \
TRAINED_LORA_PATH="${TRAINED_LORA_PATH}" \
TRAINED_LORA_ADAPTER_NAME="${ADAPTER}" \
TRAINED_BASE_MODEL_PATH="${BASE_MODEL}" \
MODEL="${MODEL}" \
TASKS="${TASKS}" \
NUM_PROCESSES="${NUM_GPUS}" \
BATCH_SIZE="${BATCH_SIZE}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}" \
OUTPUT_ROOT="${OUTPUT_DIR}" \
LOG_SAMPLES="${LOG_SAMPLES}" \
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX}" \
bash "${EVAL_SCRIPT}"

