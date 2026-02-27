#!/usr/bin/env bash
set -euo pipefail

# Unified generation benchmark launcher for VARGPT.
# Runs GenEval, WISE, and DISE (only these three).
#
# Toggle each benchmark:
#   RUN_GENEVAL=1 RUN_WISE=1 RUN_DISE=1 bash run_scripts/run_eval_vargpt_generation_bench.sh
#
# Defaults:
#   RUN_GENEVAL=1
#   RUN_WISE=1
#   RUN_DISE=1
#
# Bench-specific env vars are consumed by:
#   - run_eval_vargpt_geneval.sh
#   - run_eval_vargpt_wise.sh
#   - run_eval_vargpt_dise.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

RUN_GENEVAL="${RUN_GENEVAL:-1}"
RUN_WISE="${RUN_WISE:-1}"
RUN_DISE="${RUN_DISE:-1}"

echo "=== VARGPT Generation Benchmark Launcher ==="
echo "  GenEval: ${RUN_GENEVAL}"
echo "  WISE:    ${RUN_WISE}"
echo "  DISE:    ${RUN_DISE}"

if [[ "${RUN_GENEVAL}" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_eval_vargpt_geneval.sh"
fi

if [[ "${RUN_WISE}" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_eval_vargpt_wise.sh"
fi

if [[ "${RUN_DISE}" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_eval_vargpt_dise.sh"
fi

echo "Done. Selected generation benchmarks completed."
