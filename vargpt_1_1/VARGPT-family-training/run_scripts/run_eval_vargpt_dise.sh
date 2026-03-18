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

# DISE evaluation hook for VARGPT generated images.
#
# NOTE:
#   DISE evaluator is not bundled in this repository. This script executes a
#   user-provided DISE command in DISE_WORKDIR.
#
# Required env:
#   DISE_EVAL_CMD        shell command that runs DISE evaluation
#
# Optional env:
#   DISE_WORKDIR         default: <train_root>

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DISE_WORKDIR="${DISE_WORKDIR:-${TRAIN_ROOT}}"
DISE_EVAL_CMD="${DISE_EVAL_CMD:-}"

if [[ -z "${DISE_EVAL_CMD}" ]]; then
  echo "[ERROR] DISE_EVAL_CMD is required." >&2
  echo "Example:" >&2
  echo "  DISE_WORKDIR=/path/to/dise_repo DISE_EVAL_CMD='python eval.py --images /path/to/images' \\" >&2
  echo "  bash run_scripts/run_eval_vargpt_dise.sh" >&2
  exit 1
fi

if [[ ! -d "${DISE_WORKDIR}" ]]; then
  echo "[ERROR] DISE_WORKDIR not found: ${DISE_WORKDIR}" >&2
  exit 1
fi

echo "=== VARGPT DISE Evaluation Hook ==="
echo "  workdir: ${DISE_WORKDIR}"
echo "  cmd:     ${DISE_EVAL_CMD}"

(
  cd "${DISE_WORKDIR}"
  eval "${DISE_EVAL_CMD}"
)

echo "Done. DISE command completed."
