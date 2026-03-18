#!/bin/bash
# Evaluate WISE benchmark using GPT-4o (WiScore)
#
# Usage:
#   OPENAI_API_KEY=sk-... bash evaluate_wise.sh /path/to/generated_images
#
# Requires: WISE repo cloned at wise_repo/

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
WISE_DIR="${SCRIPT_DIR}/wise_repo"

if [ ! -d "${WISE_DIR}" ]; then
    echo "ERROR: WISE repo not found at ${WISE_DIR}"
    echo "Please clone it first:"
    echo "  cd ${SCRIPT_DIR} && git clone https://github.com/PKU-YuanGroup/WISE.git wise_repo"
    exit 1
fi

IMAGE_DIR="${1:?Usage: bash evaluate_wise.sh IMAGE_DIR [API_KEY] [MODEL]}"
API_KEY="${OPENAI_API_KEY:-${2:-}}"
GPT_MODEL="${3:-gpt-4o-2024-05-13}"
MAX_WORKERS="${MAX_WORKERS:-32}"

if [ -z "$API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set. Set it via environment or pass as second argument."
    exit 1
fi

echo "=== WISE Evaluation (WiScore) ==="
echo "  Images:  ${IMAGE_DIR}"
echo "  Model:   ${GPT_MODEL}"
echo "  Workers: ${MAX_WORKERS}"

# Evaluate all three categories
for CATEGORY in cultural_common_sense spatio-temporal_reasoning natural_science; do
    echo ""
    echo "--- Evaluating: ${CATEGORY} ---"
    python "${WISE_DIR}/gpt_eval.py" \
        --json_path "${WISE_DIR}/data/${CATEGORY}.json" \
        --output_dir "${IMAGE_DIR}/Results/${CATEGORY}" \
        --image_dir "${IMAGE_DIR}" \
        --api_key "${API_KEY}" \
        --model "${GPT_MODEL}" \
        --max_workers ${MAX_WORKERS}
done

echo ""
echo "=== Calculating Final Scores ==="
for CATEGORY in cultural_common_sense spatio-temporal_reasoning natural_science; do
    RESULTS_FILE="${IMAGE_DIR}/Results/${CATEGORY}_scores_results.jsonl"
    if [ -f "$RESULTS_FILE" ]; then
        echo ""
        echo "--- ${CATEGORY} ---"
        python "${WISE_DIR}/Calculate.py" "${RESULTS_FILE}" --category all
    else
        echo "WARNING: Results file not found: ${RESULTS_FILE}"
    fi
done

echo ""
echo "Evaluation complete. Results in: ${IMAGE_DIR}/Results/"
