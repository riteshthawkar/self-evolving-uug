#!/usr/bin/env bash
set -euo pipefail

# Build the exact unlabeled image pool described in the paper/supplement:
# 6,000 raw images sampled from COCO, SA-1B, TextVQA, GQA, and LAION-COCO,
# with all annotations discarded. This wrapper keeps the paper protocol
# separate from exploratory 3k/10k dataset builders.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/joint_6k}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-6000}"
SEED="${SEED:-42}"
DATASETS="${DATASETS:-coco,sa1b,textvqa,gqa,laion_coco}"
STREAMING_ARGS=()
FILTER_AFTER_DOWNLOAD="${FILTER_AFTER_DOWNLOAD:-0}"
STRICT_MIN_IMAGES="${STRICT_MIN_IMAGES:-6000}"

if [[ "${ALLOW_NON_STREAMING:-0}" == "1" ]]; then
  STREAMING_ARGS+=(--allow_non_streaming)
fi

echo "[DATA] Preparing paper image pool"
echo "[DATA]   output:        ${OUTPUT_DIR}"
echo "[DATA]   total_samples: ${TOTAL_SAMPLES}"
echo "[DATA]   datasets:      ${DATASETS}"
echo "[DATA]   seed:          ${SEED}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/download_joint_3k.py" \
  --output_dir "${OUTPUT_DIR}" \
  --total_samples "${TOTAL_SAMPLES}" \
  --datasets "${DATASETS}" \
  --seed "${SEED}" \
  "${STREAMING_ARGS[@]}"

if [[ "${FILTER_AFTER_DOWNLOAD}" == "1" ]]; then
  echo "[DATA] Applying image-quality filter in-place."
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/filter_dataset_images.py" \
    --image_root "${OUTPUT_DIR}/images" \
    --min_side "${FILTER_MIN_SIDE:-336}" \
    --min_regions "${FILTER_MIN_REGIONS:-3}" \
    --min_hue_clusters "${FILTER_MIN_HUE_CLUSTERS:-3}"
fi

IMAGE_COUNT="$(
  find "${OUTPUT_DIR}/images" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) \
    | wc -l | tr -d '[:space:]'
)"

if ! [[ "${IMAGE_COUNT}" =~ ^[0-9]+$ ]]; then
  IMAGE_COUNT=0
fi

echo "[DATA] Final image count: ${IMAGE_COUNT}"
if [[ "${IMAGE_COUNT}" -lt "${STRICT_MIN_IMAGES}" ]]; then
  echo "[DATA] ERROR: expected at least ${STRICT_MIN_IMAGES} images for the paper protocol." >&2
  echo "[DATA]        Got ${IMAGE_COUNT}. Re-run with working HF/network access or inspect ${OUTPUT_DIR}/download_summary.json." >&2
  exit 1
fi

echo "[DATA] Ready. Use:"
echo "[DATA]   DATA_DIR=${OUTPUT_DIR}/images bash scripts/self_evolving/final/E1_main_joint.sh"
