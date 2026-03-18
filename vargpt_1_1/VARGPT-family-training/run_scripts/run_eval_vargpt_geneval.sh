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

# GenEval benchmark for VARGPT generation stack.
#
# This script runs:
#   1) image generation (infer4eval.py)
#   2) GenEval scoring (evaluate_images.py)
#   3) summary report (summary_scores.py)
#
# Required env:
#   VAR_MODEL_PATH       path to Infinity/VAR checkpoint (.pth)
#   VAE_PATH             path to VAE checkpoint
#   TEXT_ENCODER_CKPT    path/name for text encoder checkpoint
#
# Optional env:
#   PN                   default: 1M (choices: 0.06M, 0.25M, 1M)
#   CFG                  default: 3
#   TAU                  default: 1
#   N_SAMPLES            default: 4
#   MODEL_TYPE           default: infinity_2b
#   VAE_TYPE             default: 32
#   SEED                 default: 0
#   BF16                 default: 1
#   GENEVAL_METADATA_FILE default: <geneval_dir>/prompts/evaluation_metadata.jsonl
#   GENEVAL_OUT_DIR      default: <train_root>/outputs/geneval/<timestamp>
#   GENEVAL_RESULTS_FILE default: <GENEVAL_OUT_DIR>/results.jsonl
#   GENEVAL_DETECTOR_PATH default: <geneval_dir>/model
#   GENEVAL_MODEL_CONFIG default: <geneval_dir>/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VAR_MODEL_ROOT="${TRAIN_ROOT}/visionllm/vargpt_qwen_v1_1/var_model"
GENEVAL_DIR="${VAR_MODEL_ROOT}/evaluation/gen_eval"

VAR_MODEL_PATH="${VAR_MODEL_PATH:-}"
VAE_PATH="${VAE_PATH:-}"
TEXT_ENCODER_CKPT="${TEXT_ENCODER_CKPT:-}"

PN="${PN:-1M}"
CFG="${CFG:-3}"
TAU="${TAU:-1}"
N_SAMPLES="${N_SAMPLES:-4}"
MODEL_TYPE="${MODEL_TYPE:-infinity_2b}"
VAE_TYPE="${VAE_TYPE:-32}"
SEED="${SEED:-0}"
BF16="${BF16:-1}"

GENEVAL_METADATA_FILE="${GENEVAL_METADATA_FILE:-${GENEVAL_DIR}/prompts/evaluation_metadata.jsonl}"
GENEVAL_OUT_DIR="${GENEVAL_OUT_DIR:-${TRAIN_ROOT}/outputs/geneval/$(date +%Y%m%d_%H%M%S)}"
GENEVAL_RESULTS_FILE="${GENEVAL_RESULTS_FILE:-${GENEVAL_OUT_DIR}/results.jsonl}"
GENEVAL_DETECTOR_PATH="${GENEVAL_DETECTOR_PATH:-${GENEVAL_DIR}/model}"
GENEVAL_MODEL_CONFIG="${GENEVAL_MODEL_CONFIG:-${GENEVAL_DIR}/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py}"

if [[ -z "${VAR_MODEL_PATH}" || -z "${VAE_PATH}" || -z "${TEXT_ENCODER_CKPT}" ]]; then
  echo "[ERROR] VAR_MODEL_PATH, VAE_PATH and TEXT_ENCODER_CKPT are required." >&2
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

export PYTHONPATH="${TRAIN_ROOT}:${VAR_MODEL_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"

echo "=== VARGPT GenEval ==="
echo "  checkpoint:     ${VAR_MODEL_PATH}"
echo "  metadata:       ${GENEVAL_METADATA_FILE}"
echo "  output:         ${GENEVAL_OUT_DIR}"

python evaluation/gen_eval/infer4eval.py \
  --pn "${PN}" \
  --model_path "${VAR_MODEL_PATH}" \
  --vae_type "${VAE_TYPE}" \
  --vae_path "${VAE_PATH}" \
  --text_encoder_ckpt "${TEXT_ENCODER_CKPT}" \
  --cfg "${CFG}" \
  --tau "${TAU}" \
  --n_samples "${N_SAMPLES}" \
  --seed "${SEED}" \
  --bf16 "${BF16}" \
  --model_type "${MODEL_TYPE}" \
  --metadata_file "${GENEVAL_METADATA_FILE}" \
  --outdir "${GENEVAL_OUT_DIR}"

python evaluation/gen_eval/evaluate_images.py \
  "${GENEVAL_OUT_DIR}" \
  --outfile "${GENEVAL_RESULTS_FILE}" \
  --model-config "${GENEVAL_MODEL_CONFIG}" \
  --model-path "${GENEVAL_DETECTOR_PATH}"

python evaluation/gen_eval/summary_scores.py "${GENEVAL_RESULTS_FILE}"

echo "Done. GenEval results: ${GENEVAL_RESULTS_FILE}"
