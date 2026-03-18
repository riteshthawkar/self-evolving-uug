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
    ./eval/gen/gen_images_mp_wise.py \
    --output_dir $output_path/images \
    --metadata-file ./eval/gen/wise/final_data.json \
    --resolution 1024 \
    --max-latent_size 64 \
    --model-path $model_path \
    --think


# calculate score
python3 eval/gen/wise/gpt_eval_mp.py \
        --json_path eval/gen/wise/data/cultural_common_sense.json \
        --image_dir $output_path/images \
        --output_dir $output_path

python3 eval/gen/wise/gpt_eval_mp.py \
        --json_path eval/gen/wise/data/spatio-temporal_reasoning.json \
        --image_dir $output_path/images \
        --output_dir $output_path

python3 eval/gen/wise/gpt_eval_mp.py \
        --json_path eval/gen/wise/data/natural_science.json \
        --image_dir $output_path/images \
        --output_dir $output_path

python3 eval/gen/wise/cal_score.py \
        --output_dir $output_path