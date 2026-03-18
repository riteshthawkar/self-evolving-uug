#!/bin/bash
# Evaluate generated images using DPG-Bench (mplug VQA)
#
# Usage:
#   bash evaluate_dpg.sh /path/to/generated_images 512
#   bash evaluate_dpg.sh /path/to/generated_images 512 4 8

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
ELLA_DIR="${SCRIPT_DIR}/ella_repo"

if [ ! -d "${ELLA_DIR}/dpg_bench" ]; then
    echo "ERROR: ELLA repo not found at ${ELLA_DIR}"
    echo "Please clone it first:"
    echo "  cd ${SCRIPT_DIR} && git clone https://github.com/TencentQQGYLab/ELLA.git ella_repo"
    exit 1
fi

IMAGE_PATH="${1:?Usage: bash evaluate_dpg.sh IMAGE_PATH RESOLUTION [PIC_NUM] [NUM_GPUS]}"
RESOLUTION="${2:-512}"
PIC_NUM="${3:-4}"
PROCESSES="${4:-8}"
PORT="${5:-29500}"

echo "=== DPG-Bench Evaluation ==="
echo "  Images:     ${IMAGE_PATH}"
echo "  Resolution: ${RESOLUTION}"
echo "  Pics/prompt:${PIC_NUM}"
echo "  GPUs:       ${PROCESSES}"

accelerate launch \
    --num_machines 1 \
    --num_processes $PROCESSES \
    --multi_gpu \
    --mixed_precision "fp16" \
    --main_process_port $PORT \
    "${ELLA_DIR}/dpg_bench/compute_dpg_bench.py" \
    --image_root_path "$IMAGE_PATH" \
    --resolution $RESOLUTION \
    --pic_num $PIC_NUM \
    --vqa_model "mplug"

echo "Evaluation complete. Check results in ${IMAGE_PATH}/"
