#!/usr/bin/env bash
set -euo pipefail

# ─── Understanding evaluation for self-evolving trained models (LoRA) ───
#
# Usage:
#   CHECKPOINT_DIR=/path/to/step_00500 bash understanding_eval_our.sh
#   CHECKPOINT_DIR=/path/to/step_00500 NUM_GPUS=8 bash understanding_eval_our.sh
#   CHECKPOINT_DIR=/path/to/step_00500 ADAPTER=solver TASKS="realworldqa,textvqa,gqa" bash understanding_eval_our.sh

export CACHE_ROOT="/workspace/self-evolving-uug/cache"
export HF_HOME="/workspace/self-evolving-uug/cache"
export HUGGINGFACE_HUB_CACHE="/workspace/self-evolving-uug/cache"
export TRANSFORMERS_CACHE="/workspace/self-evolving-uug/cache"
export HF_DATASETS_CACHE="/workspace/self-evolving-uug/cache"
export HF_METRICS_CACHE="/workspace/self-evolving-uug/cache"
export TORCH_HOME="/workspace/self-evolving-uug/cache"
export TRITON_CACHE_DIR="/workspace/self-evolving-uug/cache"
export XDG_CACHE_HOME="/workspace/self-evolving-uug/cache"
export TOKENIZERS_PARALLELISM="false"
export HF_TOKEN="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"
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
TASKS="${TASKS:-realworldqa,textvqa}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/self-evolving-uug/self-evolving-uug/BLIP3o/eval/logs}"

# Derive a suffix from the checkpoint path for log identification
CKPT_NAME="$(basename "$(dirname "$CHECKPOINT_DIR")")"_"$(basename "$CHECKPOINT_DIR")"
LOG_SUFFIX="blip3o_our_${ADAPTER}_${CKPT_NAME}"

echo "============================================"
echo "Understanding Evaluation (Self-Evolving)"
echo "  Base model:      ${BASE_MODEL}"
echo "  Checkpoint:      ${CHECKPOINT_DIR}"
echo "  Adapter:         ${ADAPTER}"
echo "  Tasks:           ${TASKS}"
echo "  Output:          ${OUTPUT_DIR}"
echo "  Num GPUs:        ${NUM_GPUS}"
echo "  Log suffix:      ${LOG_SUFFIX}"
echo "============================================"

python -m accelerate.commands.launch \
    --num_processes="${NUM_GPUS}" \
    -m lmms_eval \
    --model blip3o_our \
    --model_args "pretrained=${BASE_MODEL},checkpoint_dir=${CHECKPOINT_DIR},adapter=${ADAPTER}" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${LOG_SUFFIX}" \
    --output_path "${OUTPUT_DIR}"
