#!/usr/bin/env bash
set -euo pipefail

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
export CUDA_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BLIP3O_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${BLIP3O_ROOT}:${BLIP3O_ROOT}/eval/lmms-eval:${PYTHONPATH:-}"


python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model blip3o \
    --model_args "pretrained=BLIP3o/BLIP3o-Model-8B" \
    --tasks "realworldqa,textvqa" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix blip3o \
    --output_path "/workspace/self-evolving-uug/self-evolving-uug/BLIP3o/eval/logs"

