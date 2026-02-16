#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BLIP3O_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${BLIP3O_ROOT}:${BLIP3O_ROOT}/eval/lmms-eval:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-your/model/path/}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
TASKS="${TASKS:-mme}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/logs}"

mkdir -p "${OUTPUT_PATH}"

python -m accelerate.commands.launch \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model blip3o \
    --model_args "pretrained=${MODEL_PATH}" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix blip3o \
    --output_path "${OUTPUT_PATH}"

