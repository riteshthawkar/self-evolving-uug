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

# ─── Understanding evaluation for self-evolving trained models (LoRA) ───
#
# Usage:
#   CHECKPOINT_DIR=/path/to/step_00500 bash understanding_eval_our.sh
#   CHECKPOINT_DIR=/path/to/step_00500 NUM_GPUS=8 bash understanding_eval_our.sh
#   CHECKPOINT_DIR=/path/to/step_00500 ADAPTER=solver TASKS="realworldqa,textvqa" bash understanding_eval_our.sh

# ─── Multi-GPU configuration ───
NUM_GPUS="${NUM_GPUS:-8}"
# Expose all GPUs: build "0,1,2,...,N-1" string
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($NUM_GPUS - 1)))
fi
export CUDA_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BLIP3O_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${BLIP3O_ROOT}:${BLIP3O_ROOT}/eval/lmms-eval:${PYTHONPATH:-}"

# ─── Configuration (override via environment variables) ───
BASE_MODEL="${BASE_MODEL:-BLIP3o/BLIP3o-Model-8B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Please set CHECKPOINT_DIR to your training checkpoint path (e.g. /path/to/step_00500)}"
ADAPTER="${ADAPTER:-solver}"
USE_FLASH_ATTENTION_2="${USE_FLASH_ATTENTION_2:-false}"
TASKS="${TASKS:-mmmu_val,mmbench_en_dev,textvqa_val,seedbench,realworldqa,mmvet,mme}"
OUTPUT_DIR="${OUTPUT_DIR:-${BLIP3O_ROOT}/eval/logs}"

# Derive a suffix from the checkpoint path for log identification
CKPT_NAME="$(basename "$(dirname "$CHECKPOINT_DIR")")"_"$(basename "$CHECKPOINT_DIR")"
LOG_SUFFIX="blip3o_our_${ADAPTER}_${CKPT_NAME}"

echo "============================================"
echo "Understanding Evaluation (Self-Evolving)"
echo "  Base model:      ${BASE_MODEL}"
echo "  Checkpoint:      ${CHECKPOINT_DIR}"
echo "  Adapter:         ${ADAPTER}"
echo "  FlashAttn2:      ${USE_FLASH_ATTENTION_2}"
echo "  Tasks:           ${TASKS}"
echo "  Output:          ${OUTPUT_DIR}"
echo "  Num GPUs:        ${NUM_GPUS}"
echo "  Log suffix:      ${LOG_SUFFIX}"
echo "============================================"

python -m accelerate.commands.launch \
    --num_processes="${NUM_GPUS}" \
    -m lmms_eval \
    --model blip3o_our \
    --model_args "pretrained=${BASE_MODEL},checkpoint_dir=${CHECKPOINT_DIR},adapter=${ADAPTER},use_flash_attention_2=${USE_FLASH_ATTENTION_2}" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${LOG_SUFFIX}" \
    --output_path "${OUTPUT_DIR}"
