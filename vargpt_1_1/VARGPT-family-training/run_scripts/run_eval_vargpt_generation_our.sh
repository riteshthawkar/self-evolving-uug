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

# ─── Generation benchmark eval for self-evolving VARGPT (LoRA/merged) ──────
#
# Runs GenEval + WISE + DISE through the unified generation bench launcher.
# CHECKPOINT_DIR can be:
#   - .../se_checkpoint_<n>
#   - .../se_checkpoint_<n>/model
#   - adapter folder directly
#
# Examples:
#   CHECKPOINT_DIR=/path/to/se_checkpoint_2000 bash run_scripts/run_eval_vargpt_generation_our.sh
#   CHECKPOINT_DIR=/path/to/se_checkpoint_2000 ADAPTER=generator RUN_DISE=0 bash run_scripts/run_eval_vargpt_generation_our.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/run_eval_vargpt_generation_bench.sh"

if [[ ! -f "${BENCH_SCRIPT}" ]]; then
  echo "[ERROR] Bench script not found: ${BENCH_SCRIPT}" >&2
  exit 1
fi

BASE_MODEL="${BASE_MODEL:-VARGPT-family/VARGPT-v1.1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Please set CHECKPOINT_DIR to your self-evolving checkpoint path}"
ADAPTER="${ADAPTER:-generator}"
TRAINED_RUNTIME="${TRAINED_RUNTIME:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_ROOT}/logs/generation_eval_our}"
RUN_GENEVAL="${RUN_GENEVAL:-1}"
RUN_WISE="${RUN_WISE:-1}"
RUN_DISE="${RUN_DISE:-1}"

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
if [[ "${OUTPUT_ROOT}" == "${TRAIN_ROOT}/logs/generation_eval_our" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT}/${CKPT_NAME}"
fi

echo "============================================"
echo "Generation Evaluation (Self-Evolving VARGPT)"
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
echo "  Runtime:         ${TRAINED_RUNTIME}"
echo "  Run GenEval:     ${RUN_GENEVAL}"
echo "  Run WISE:        ${RUN_WISE}"
echo "  Run DISE:        ${RUN_DISE}"
echo "  Output root:     ${OUTPUT_ROOT}"
echo "============================================"

EVAL_SETS="trained_lora" \
TRAINED_RUNTIME="${TRAINED_RUNTIME}" \
TRAINED_MODEL_PATH="${TRAINED_MODEL_PATH}" \
TRAINED_LORA_PATH="${TRAINED_LORA_PATH}" \
TRAINED_LORA_ADAPTER_NAME="${ADAPTER}" \
TRAINED_BASE_MODEL_PATH="${BASE_MODEL}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RUN_GENEVAL="${RUN_GENEVAL}" \
RUN_WISE="${RUN_WISE}" \
RUN_DISE="${RUN_DISE}" \
bash "${BENCH_SCRIPT}"

