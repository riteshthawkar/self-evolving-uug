#!/bin/bash
# Generate DPG-Bench images using self-evolving trained model

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BLIP3O_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${BLIP3O_ROOT}:${PYTHONPATH:-}"

export CACHE_ROOT="/workspace/self-evolving-uug/cache"
export HF_HOME="/workspace/self-evolving-uug/cache"
export HUGGINGFACE_HUB_CACHE="/workspace/self-evolving-uug/cache"
export TRANSFORMERS_CACHE="/workspace/self-evolving-uug/cache"
export HF_DATASETS_CACHE="/workspace/self-evolving-uug/cache"
export TORCH_HOME="/workspace/self-evolving-uug/cache"
export TOKENIZERS_PARALLELISM="false"
export HF_TOKEN="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"

MODEL="${MODEL:-/workspace/self-evolving-uug/cache/models--BLIP3o--BLIP3o-Model-8B/snapshots/c2edfc20814d4624c8d73ca3de351ebc3fa86508}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Please set CHECKPOINT_DIR}"
ADAPTER="${ADAPTER:-generator}"
STEPS="${STEPS:-50}"
N_CHUNKS="${N_CHUNKS:-8}"
OUTDIR="${OUTDIR:-${CHECKPOINT_DIR}/dpg_bench_images}"

echo "=== DPG-Bench Generation (Self-Evolving) ==="
echo "  Model:      ${MODEL}"
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "  Adapter:    ${ADAPTER}"
echo "  Steps:      ${STEPS}"
echo "  Output:     ${OUTDIR}"

for i in $(seq 0 $(($N_CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python "${SCRIPT_DIR}/generate_dpg_our.py" \
        --model "$MODEL" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --adapter "$ADAPTER" \
        --steps "$STEPS" \
        --outdir "$OUTDIR" \
        --index $i --n_chunks $N_CHUNKS &
done
wait
echo "Done. Images saved to: ${OUTDIR}"
