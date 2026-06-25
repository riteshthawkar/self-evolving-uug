#!/bin/bash

# ─── GenEval generation for self-evolving trained models (LoRA + DiT) ───
#
# Usage:
#   bash generation_our.sh                                   # use defaults
#   CHECKPOINT_DIR=/path/to/step_00500 bash generation_our.sh
#   CHECKPOINT_DIR=/path/to/step_00500 N_CHUNKS=4 STEPS=50 bash generation_our.sh

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
BLIP3O_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${BLIP3O_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"

# ─── Configuration (override via environment variables) ───
MODEL="${MODEL:-BLIP3o/BLIP3o-Model-8B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Please set CHECKPOINT_DIR to your training checkpoint path (e.g. /path/to/step_00500)}"
ADAPTER="${ADAPTER:-generator}"
STEPS="${STEPS:-50}"
N_CHUNKS="${N_CHUNKS:-8}"
OUTDIR="${OUTDIR:-${CHECKPOINT_DIR}/geneval_qwen}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/geneval_prompt.jsonl}"

echo "============================================"
echo "GenEval Generation (Self-Evolving)"
echo "  Base model:      ${MODEL}"
echo "  Checkpoint:      ${CHECKPOINT_DIR}"
echo "  Adapter:         ${ADAPTER}"
echo "  Diffusion steps: ${STEPS}"
echo "  N_CHUNKS:        ${N_CHUNKS}"
echo "  Prompt file:     ${PROMPT_FILE}"
echo "  Output:          ${OUTDIR}"
echo "============================================"

# Launch processes in parallel for each GPU/chunk.
pids=()
for i in $(seq 0 $(($N_CHUNKS - 1))); do
    echo "Launching process for GPU $i (chunk index $i of $N_CHUNKS)"
    CUDA_VISIBLE_DEVICES=$i python "${SCRIPT_DIR}/generate_our.py" \
        --model "$MODEL" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --adapter "$ADAPTER" \
        --steps "$STEPS" \
        --prompt_file "$PROMPT_FILE" \
        --outdir "$OUTDIR" \
        --index $i \
        --n_chunks $N_CHUNKS &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [[ "$status" != "0" ]]; then
    echo "ERROR: one or more generation workers failed." >&2
    exit "$status"
fi
echo "All generation processes finished."
echo "Output saved to: ${OUTDIR}"
