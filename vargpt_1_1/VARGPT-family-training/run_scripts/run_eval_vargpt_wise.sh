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

# WISE evaluation runner for generated images.
#
# This script evaluates an image directory on WISE (WiScore). It supports an
# optional image generation pre-step via WISE_GENERATE_CMD.
#
# Required env:
#   OPENAI_API_KEY       OpenAI API key for WISE GPT-based scoring.
#   WISE_IMAGE_DIR       directory containing generated images.
#
# Optional env:
#   WISE_REPO_DIR        default: <this_script_dir>/wise_repo
#   WISE_GPT_MODEL       default: gpt-4o-2024-05-13
#   WISE_MAX_WORKERS     default: 32
#   WISE_GENERATE_CMD    optional shell command run before scoring
#                        (must also set WISE_IMAGE_DIR as output path)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WISE_REPO_DIR="${WISE_REPO_DIR:-${SCRIPT_DIR}/wise_repo}"
WISE_IMAGE_DIR="${WISE_IMAGE_DIR:-}"
WISE_GPT_MODEL="${WISE_GPT_MODEL:-gpt-4o-2024-05-13}"
WISE_MAX_WORKERS="${WISE_MAX_WORKERS:-32}"
WISE_GENERATE_CMD="${WISE_GENERATE_CMD:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"

if [[ -n "${WISE_GENERATE_CMD}" ]]; then
  echo "[WISE] Running generation command..."
  eval "${WISE_GENERATE_CMD}"
fi

if [[ -z "${WISE_IMAGE_DIR}" ]]; then
  echo "[ERROR] WISE_IMAGE_DIR is required." >&2
  exit 1
fi

if [[ ! -d "${WISE_REPO_DIR}" ]]; then
  echo "[ERROR] WISE repo not found: ${WISE_REPO_DIR}" >&2
  echo "Clone it first, for example:" >&2
  echo "  git clone https://github.com/PKU-YuanGroup/WISE.git ${WISE_REPO_DIR}" >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is required for WISE evaluation." >&2
  exit 1
fi

echo "=== VARGPT WISE Evaluation ==="
echo "  wise_repo: ${WISE_REPO_DIR}"
echo "  image_dir: ${WISE_IMAGE_DIR}"
echo "  model:     ${WISE_GPT_MODEL}"
echo "  workers:   ${WISE_MAX_WORKERS}"

for CATEGORY in cultural_common_sense spatio-temporal_reasoning natural_science; do
  echo "--- Evaluating category: ${CATEGORY} ---"
  python "${WISE_REPO_DIR}/gpt_eval.py" \
    --json_path "${WISE_REPO_DIR}/data/${CATEGORY}.json" \
    --output_dir "${WISE_IMAGE_DIR}/Results/${CATEGORY}" \
    --image_dir "${WISE_IMAGE_DIR}" \
    --api_key "${OPENAI_API_KEY}" \
    --model "${WISE_GPT_MODEL}" \
    --max_workers "${WISE_MAX_WORKERS}"
done

echo "--- WISE aggregate scores ---"
for CATEGORY in cultural_common_sense spatio-temporal_reasoning natural_science; do
  RESULTS_FILE="${WISE_IMAGE_DIR}/Results/${CATEGORY}_scores_results.jsonl"
  if [[ -f "${RESULTS_FILE}" ]]; then
    python "${WISE_REPO_DIR}/Calculate.py" "${RESULTS_FILE}" --category all
  else
    echo "[WARN] Missing results file: ${RESULTS_FILE}"
  fi
done

echo "Done. WISE outputs: ${WISE_IMAGE_DIR}/Results"
