#!/usr/bin/env bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

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

# Set proxy and API key
export OPENAI_API_KEY="${OPENAI_API_KEY:-${openai_api_key:-}}"

export GPUS=1

DATASETS=("mme" "mmbench-dev-en" "mmvet" "mmmu-val" "mathvista-testmini" "mmvp")
# DATASETS=("mmmu-val_cot")

DATASETS_STR="${DATASETS[*]}"
export DATASETS_STR

bash scripts/eval/eval_vlm.sh \
    $output_path \
    --model-path $model_path