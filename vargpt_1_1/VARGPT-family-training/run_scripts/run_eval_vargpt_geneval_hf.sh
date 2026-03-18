#!/usr/bin/env bash
set -euo pipefail

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

# GenEval benchmark using VARGPT HF model (+ optional LoRA adapter).
#
# This script runs:
#   1) image generation via VARGPT model.generate()
#   2) GenEval scoring (evaluate_images.py)
#   3) summary report (summary_scores.py)
#
# Required env:
#   HF_PRETRAINED_PATH   base model path or HF id
#
# Optional env:
#   HF_PEFT_PATH         LoRA adapter path
#   HF_PEFT_ADAPTER_NAME adapter name to activate (default: default)
#   HF_DEVICE            default: cuda
#   HF_DTYPE             default: bfloat16 (bfloat16|float16|float32)
#   HF_MAX_NEW_TOKENS    default: 4096
#   HF_DO_SAMPLE         default: 1
#   HF_TEMPERATURE       default: 1.0
#   HF_TOP_P             default: 1.0
#   N_SAMPLES            default: 4
#   SEED                 default: 0
#   GENEVAL_METADATA_FILE default: <geneval_dir>/prompts/evaluation_metadata.jsonl
#   GENEVAL_OUT_DIR      default: <train_root>/outputs/geneval_hf/<timestamp>
#   GENEVAL_RESULTS_FILE default: <GENEVAL_OUT_DIR>/results.jsonl
#   GENEVAL_DETECTOR_PATH default: <geneval_dir>/model
#   GENEVAL_MODEL_CONFIG default: <geneval_dir>/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SUDER_ROOT="$(cd -- "${TRAIN_ROOT}/.." && pwd)"
VAR_MODEL_ROOT="${TRAIN_ROOT}/visionllm/vargpt_qwen_v1_1/var_model"
GENEVAL_DIR="${VAR_MODEL_ROOT}/evaluation/gen_eval"

HF_PRETRAINED_PATH="${HF_PRETRAINED_PATH:-}"
HF_PEFT_PATH="${HF_PEFT_PATH:-}"
HF_PEFT_ADAPTER_NAME="${HF_PEFT_ADAPTER_NAME:-default}"
HF_DEVICE="${HF_DEVICE:-cuda}"
HF_DTYPE="${HF_DTYPE:-bfloat16}"
HF_MAX_NEW_TOKENS="${HF_MAX_NEW_TOKENS:-4096}"
HF_DO_SAMPLE="${HF_DO_SAMPLE:-1}"
HF_TEMPERATURE="${HF_TEMPERATURE:-1.0}"
HF_TOP_P="${HF_TOP_P:-1.0}"
N_SAMPLES="${N_SAMPLES:-4}"
SEED="${SEED:-0}"

GENEVAL_METADATA_FILE="${GENEVAL_METADATA_FILE:-${GENEVAL_DIR}/prompts/evaluation_metadata.jsonl}"
GENEVAL_OUT_DIR="${GENEVAL_OUT_DIR:-${TRAIN_ROOT}/outputs/geneval_hf/$(date +%Y%m%d_%H%M%S)}"
GENEVAL_RESULTS_FILE="${GENEVAL_RESULTS_FILE:-${GENEVAL_OUT_DIR}/results.jsonl}"
GENEVAL_DETECTOR_PATH="${GENEVAL_DETECTOR_PATH:-${GENEVAL_DIR}/model}"
GENEVAL_MODEL_CONFIG="${GENEVAL_MODEL_CONFIG:-${GENEVAL_DIR}/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py}"

if [[ -z "${HF_PRETRAINED_PATH}" ]]; then
  echo "[ERROR] HF_PRETRAINED_PATH is required." >&2
  exit 1
fi
if [[ ! -d "${GENEVAL_DIR}" ]]; then
  echo "[ERROR] GenEval dir not found: ${GENEVAL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${GENEVAL_METADATA_FILE}" ]]; then
  echo "[ERROR] GenEval metadata file not found: ${GENEVAL_METADATA_FILE}" >&2
  exit 1
fi

mkdir -p "${GENEVAL_OUT_DIR}"
cd "${VAR_MODEL_ROOT}"

export PYTHONPATH="${TRAIN_ROOT}:${TRAIN_ROOT}/src:${SUDER_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"

echo "=== VARGPT GenEval (HF+LoRA) ==="
echo "  pretrained:     ${HF_PRETRAINED_PATH}"
if [[ -n "${HF_PEFT_PATH}" ]]; then
  echo "  peft:           ${HF_PEFT_PATH}"
  echo "  adapter:        ${HF_PEFT_ADAPTER_NAME}"
fi
echo "  metadata:       ${GENEVAL_METADATA_FILE}"
echo "  output:         ${GENEVAL_OUT_DIR}"

declare -a gen_cmd=(
  python "${SCRIPT_DIR}/geneval_generate_vargpt_hf.py"
  --train_root "${TRAIN_ROOT}"
  --pretrained "${HF_PRETRAINED_PATH}"
  --metadata_file "${GENEVAL_METADATA_FILE}"
  --outdir "${GENEVAL_OUT_DIR}"
  --n_samples "${N_SAMPLES}"
  --seed "${SEED}"
  --max_new_tokens "${HF_MAX_NEW_TOKENS}"
  --do_sample "${HF_DO_SAMPLE}"
  --temperature "${HF_TEMPERATURE}"
  --top_p "${HF_TOP_P}"
  --dtype "${HF_DTYPE}"
  --device "${HF_DEVICE}"
)
if [[ -n "${HF_PEFT_PATH}" ]]; then
  gen_cmd+=(--peft "${HF_PEFT_PATH}")
fi
if [[ -n "${HF_PEFT_ADAPTER_NAME}" ]]; then
  gen_cmd+=(--peft_adapter_name "${HF_PEFT_ADAPTER_NAME}")
fi
"${gen_cmd[@]}"

python evaluation/gen_eval/evaluate_images.py \
  "${GENEVAL_OUT_DIR}" \
  --outfile "${GENEVAL_RESULTS_FILE}" \
  --model-config "${GENEVAL_MODEL_CONFIG}" \
  --model-path "${GENEVAL_DETECTOR_PATH}"

python evaluation/gen_eval/summary_scores.py "${GENEVAL_RESULTS_FILE}"

echo "Done. GenEval results: ${GENEVAL_RESULTS_FILE}"
