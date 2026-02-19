#!/bin/bash

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
export CUDA_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0


MODEL="/workspace/self-evolving-uug/cache/models--BLIP3o--BLIP3o-Model-8B/snapshots/c2edfc20814d4624c8d73ca3de351ebc3fa86508"



# Total number of GPUs/chunks.
N_CHUNKS=8

# Launch processes in parallel for each GPU/chunk.
for i in $(seq 0 $(($N_CHUNKS - 1))); do
    echo "Launching process for GPU $i (chunk index $i of $N_CHUNKS)"
    CUDA_VISIBLE_DEVICES=$i python generate.py --model "$MODEL" --index $i --n_chunks $N_CHUNKS &
done

# Wait for all background processes to finish.
wait
echo "All background processes finished."


