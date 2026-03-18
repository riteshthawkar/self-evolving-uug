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
    ./eval/gen/gen_images_mp_imgedit.py \
    --output_dir $output_path/bagel \
    --metadata_file ./eval/gen/imgedit/Benchmark/singleturn/singleturn.json \
    --max_latent_size 64 \
    --model-path $model_path


# calculate score
python ./eval/gen/imgedit/basic_bench.py \
    --result_img_folder $output_path/bagel \
    --edit_json ./eval/gen/imgedit/Benchmark/singleturn/singleturn.json \
    --origin_img_root ./eval/gen/imgedit/Benchmark/singleturn \
    --num_processes 4 \
    --prompts_json ./eval/gen/imgedit/Benchmark/singleturn/judge_prompt.json


# summarize score
python ./eval/gen/imgedit/step1_get_avgscore.py \
    --result_json $output_path/bagel/result.json \
    --average_score_json $output_path/bagel/average_score.json

python ./eval/gen/imgedit/step2_typescore.py \
    --average_score_json  $output_path/bagel/average_score.json \
    --edit_json ./eval/gen/imgedit/Benchmark/singleturn/singleturn.json \
    --typescore_json $output_path/bagel/typescore.json