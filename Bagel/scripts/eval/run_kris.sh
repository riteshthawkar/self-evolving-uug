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

export OPENAI_API_KEY="${OPENAI_API_KEY:-${openai_api_key:-}}"

GPUS=8


# generate images
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp_kris.py \
    --output_dir $output_path/bagel \
    --metadata_file ./eval/gen/kris/final_data.json \
    --max_latent_size 64 \
    --model-path $model_path \
    --think


# calculate score
python ./eval/gen/kris/metrics_common.py \
    --results_dir $output_path \
    --max_workers 8

python ./eval/gen/kris/metrics_knowledge.py \
    --results_dir $output_path \
    --max_workers 8

python ./eval/gen/kris/metrics_multi_element.py \
    --results_dir $output_path \
    --max_workers 8

python ./eval/gen/kris/metrics_temporal_prediction.py \
    --results_dir $output_path \
    --max_workers 8

python ./eval/gen/kris/metrics_view_change.py \
    --results_dir $output_path \
    --max_workers 8


# summarize score
python ./eval/gen/kris/summarize.py \
    --results_dir $output_path/bagel \