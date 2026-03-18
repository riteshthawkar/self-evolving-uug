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

# ─── Generation benchmark eval for original/base VARGPT ─────────────────────
#
# Runs GenEval + WISE + DISE through the unified generation bench launcher.
# Configure benchmark-specific env vars as needed (e.g., WISE_IMAGE_DIR,
# DISE_EVAL_CMD, DISE_EVAL_CMD_TEMPLATE).
#
# Examples:
#   bash run_scripts/run_eval_vargpt_generation_base.sh
#   BASE_MODEL=VARGPT-family/VARGPT-v1.1 RUN_DISE=0 bash run_scripts/run_eval_vargpt_generation_base.sh
#   PURE_VAR_MODEL_PATH=/path/to/model.pth PURE_RUNTIME=var bash run_scripts/run_eval_vargpt_generation_base.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/run_eval_vargpt_generation_bench.sh"

if [[ ! -f "${BENCH_SCRIPT}" ]]; then
  echo "[ERROR] Bench script not found: ${BENCH_SCRIPT}" >&2
  exit 1
fi

BASE_MODEL="${BASE_MODEL:-VARGPT-family/VARGPT-v1.1}"
PURE_RUNTIME="${PURE_RUNTIME:-auto}"
PURE_VAR_MODEL_PATH="${PURE_VAR_MODEL_PATH:-}"
PURE_MODEL_PATH="${PURE_MODEL_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_ROOT}/logs/generation_eval_base}"
RUN_GENEVAL="${RUN_GENEVAL:-1}"
RUN_WISE="${RUN_WISE:-1}"
RUN_DISE="${RUN_DISE:-1}"

if [[ -n "${PURE_LORA_PATH:-}" ]]; then
  echo "[ERROR] PURE_LORA_PATH is set in base eval wrapper. Use run_eval_vargpt_generation_our.sh for LoRA eval." >&2
  exit 1
fi

if [[ -z "${PURE_MODEL_PATH}" && -z "${PURE_VAR_MODEL_PATH}" ]]; then
  PURE_MODEL_PATH="${BASE_MODEL}"
fi

echo "============================================"
echo "Generation Evaluation (Base VARGPT)"
echo "  Base model:      ${PURE_MODEL_PATH:-<none>}"
echo "  VAR model path:  ${PURE_VAR_MODEL_PATH:-<none>}"
echo "  Runtime:         ${PURE_RUNTIME}"
echo "  Run GenEval:     ${RUN_GENEVAL}"
echo "  Run WISE:        ${RUN_WISE}"
echo "  Run DISE:        ${RUN_DISE}"
echo "  Output root:     ${OUTPUT_ROOT}"
echo "============================================"

EVAL_SETS="pure" \
PURE_RUNTIME="${PURE_RUNTIME}" \
PURE_MODEL_PATH="${PURE_MODEL_PATH}" \
PURE_VAR_MODEL_PATH="${PURE_VAR_MODEL_PATH}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RUN_GENEVAL="${RUN_GENEVAL}" \
RUN_WISE="${RUN_WISE}" \
RUN_DISE="${RUN_DISE}" \
bash "${BENCH_SCRIPT}"

