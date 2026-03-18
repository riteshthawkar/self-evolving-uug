#!/bin/bash
# Script to run SUDER ablation on VARGPT
# This script launches the custom SUDER training stage implemented in the codebase.
#
# Requirements:
# - VARGPT environment setup
# - Pretrained model weights
# - Dataset (e.g., COCO or similar SFT dataset)

# Arguments:
# 1. MODEL_PATH: Path to the pretrained VARGPT-v1.1 model
# 2. DATA_PATH: Path to the dataset (or dataset name in dataset_info.json)
# 3. OUTPUT_DIR: Directory to save checkpoints and logs

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

MODEL_PATH=${1:-"path/to/vargpt_model"}
DATA_PATH=${2:-"coco_2014_caption"} # Example dataset name
OUTPUT_DIR=${3:-"output/suder_ablation_v1"}

echo "Starting SUDER Ablation Run..."
echo "Model: $MODEL_PATH"
echo "Dataset: $DATA_PATH"
echo "Output: $OUTPUT_DIR"

# Launch training
# Note: Adjust batch size and gradient accumulation based on GPU memory.
# SUDER involves generation during training, so it consumes more memory.
# We use BF16 and Gradient Checkpointing to save memory.

python src/train.py \
    --stage suder \
    --do_train \
    --model_name_or_path "$MODEL_PATH" \
    --dataset "$DATA_PATH" \
    --template vargpt_qwen2_vl \
    --finetuning_type full \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 50 \
    --learning_rate 1e-6 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16 \
    --vargpt_version qwen2vl-v1.1 \
    --pref_beta 0.1 \
    --ppo_epochs 1 \
    --report_to none

echo "SUDER Ablation Run Completed."
