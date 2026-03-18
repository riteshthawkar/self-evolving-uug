#!/usr/bin/env bash
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

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0  MASTER_ADDR=127.0.0.1 MASTER_PORT=39547 llamafactory-cli train examples/train_vargpt_qwen2vl_1_1/vargpt_pretraining_7b_v1_1_stage3.yaml

