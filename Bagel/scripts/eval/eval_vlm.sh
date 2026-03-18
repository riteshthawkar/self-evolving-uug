#!/usr/bin/env bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Check if enough arguments are provided
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

if [ $# -lt 2 ]; then
    echo "Error: PREFIX_DIR and MODEL_PATH are required as the first and second arguments respectively."
    exit 1
fi

LOG_PATH=$1
if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
fi
shift 1
ARGS=("$@")
export MASTER_PORT=10042

FULL_MODEL_PATH="$PREFIX_DIR/$MODEL_PATH"

IFS=' ' read -r -a DATASETS <<< "$DATASETS_STR"

for DATASET in "${DATASETS[@]}"; do
    bash eval/vlm/evaluate.sh \
        "$DATASET" \
        --out-dir "$LOG_PATH/$DATASET" \
        "${ARGS[@]}"
done