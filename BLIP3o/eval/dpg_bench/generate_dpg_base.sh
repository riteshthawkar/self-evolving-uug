#!/bin/bash
# Generate DPG-Bench images using base BLIP3o model

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

MODEL="${MODEL:-BLIP3o/BLIP3o-Model-8B}"
N_CHUNKS="${N_CHUNKS:-8}"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/outputs/base_model}"

echo "=== DPG-Bench Generation (Base BLIP3o) ==="
echo "  Model:   ${MODEL}"
echo "  Output:  ${OUTDIR}"

pids=()
for i in $(seq 0 $(($N_CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python "${SCRIPT_DIR}/generate_dpg_base.py" \
        --model "$MODEL" \
        --outdir "$OUTDIR" \
        --index $i --n_chunks $N_CHUNKS &
done
wait
echo "Done. Images saved to: ${OUTDIR}"
