#!/bin/bash

# ─── GenEval generation for self-evolving trained models (LoRA + DiT) ───
#
# Usage:
#   bash generation_our.sh                                   # use defaults
#   CHECKPOINT_DIR=/path/to/step_00500 bash generation_our.sh
#   CHECKPOINT_DIR=/path/to/step_00500 N_CHUNKS=4 STEPS=50 bash generation_our.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BLIP3O_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${BLIP3O_ROOT}:${PYTHONPATH:-}"

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

# ─── Configuration (override via environment variables) ───
MODEL="${MODEL:-/workspace/self-evolving-uug/cache/models--BLIP3o--BLIP3o-Model-8B/snapshots/c2edfc20814d4624c8d73ca3de351ebc3fa86508}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Please set CHECKPOINT_DIR to your training checkpoint path (e.g. /path/to/step_00500)}"
ADAPTER="${ADAPTER:-generator}"
STEPS="${STEPS:-50}"
N_CHUNKS="${N_CHUNKS:-8}"
OUTDIR="${OUTDIR:-${CHECKPOINT_DIR}/geneval_qwen}"

echo "============================================"
echo "GenEval Generation (Self-Evolving)"
echo "  Base model:      ${MODEL}"
echo "  Checkpoint:      ${CHECKPOINT_DIR}"
echo "  Adapter:         ${ADAPTER}"
echo "  Diffusion steps: ${STEPS}"
echo "  N_CHUNKS:        ${N_CHUNKS}"
echo "  Output:          ${OUTDIR}"
echo "============================================"

# Launch processes in parallel for each GPU/chunk.
for i in $(seq 0 $(($N_CHUNKS - 1))); do
    echo "Launching process for GPU $i (chunk index $i of $N_CHUNKS)"
    CUDA_VISIBLE_DEVICES=$i python generate_our.py \
        --model "$MODEL" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --adapter "$ADAPTER" \
        --steps "$STEPS" \
        --outdir "$OUTDIR" \
        --index $i \
        --n_chunks $N_CHUNKS &
done

# Wait for all background processes to finish.
wait
echo "All generation processes finished."
echo "Output saved to: ${OUTDIR}"
