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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BLIP3O_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-BLIP3o/BLIP3o-Model-8B}"
TASKS="${TASKS:-realworldqa,textvqa}"
OUTPUT_DIR="${OUTPUT_DIR:-${BLIP3O_ROOT}/eval/logs}"
USE_FLASH_ATTENTION_2="${USE_FLASH_ATTENTION_2:-false}"

export PYTHONPATH="${BLIP3O_ROOT}:${BLIP3O_ROOT}/eval/lmms-eval:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"

python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model blip3o \
    --model_args "pretrained=${BASE_MODEL},use_flash_attention_2=${USE_FLASH_ATTENTION_2}" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix blip3o \
    --output_path "${OUTPUT_DIR}"
