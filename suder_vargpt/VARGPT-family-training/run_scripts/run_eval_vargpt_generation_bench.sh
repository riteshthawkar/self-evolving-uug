#!/usr/bin/env bash
set -euo pipefail

# Unified generation benchmark launcher for VARGPT.
# Runs GenEval, WISE, and DISE (only these three).
#
# Usage:
#   PURE_VAR_MODEL_PATH=/path/to/pure_var_ckpt.pth \
#   TRAINED_VAR_MODEL_PATH=/path/to/trained_var_ckpt.pth \
#   bash run_scripts/run_eval_vargpt_generation_bench.sh
#
# Defaults:
#   RUN_GENEVAL=1
#   RUN_WISE=1
#   RUN_DISE=1
#   EVAL_SETS=pure,trained_lora
#
# Bench-specific env vars are consumed by:
#   - run_eval_vargpt_geneval.sh
#   - run_eval_vargpt_wise.sh
#   - run_eval_vargpt_dise.sh
#
# Optional env for set orchestration:
#   EVAL_SETS                 comma list: pure,trained_lora
#   PURE_VAR_MODEL_PATH       checkpoint for pure set
#   TRAINED_VAR_MODEL_PATH    checkpoint for trained set
#   OUTPUT_ROOT               default: <train_root>/outputs/generation_eval_sets/<timestamp>
#   WISE_IMAGE_DIR            global override for WISE input dir
#   PURE_WISE_IMAGE_DIR       set-specific WISE input dir override
#   TRAINED_WISE_IMAGE_DIR    set-specific WISE input dir override
#   DISE_EVAL_CMD             global DISE command override
#   DISE_EVAL_CMD_TEMPLATE    template with __IMAGE_DIR__, __SET_NAME__, __SET_DIR__
#
# Legacy mode:
#   If VAR_MODEL_PATH is set and set-specific model paths are empty, script
#   runs a single set ("pure") for backward compatibility.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RUN_GENEVAL="${RUN_GENEVAL:-1}"
RUN_WISE="${RUN_WISE:-1}"
RUN_DISE="${RUN_DISE:-1}"
EVAL_SETS="${EVAL_SETS:-pure,trained_lora}"

PURE_VAR_MODEL_PATH="${PURE_VAR_MODEL_PATH:-}"
TRAINED_VAR_MODEL_PATH="${TRAINED_VAR_MODEL_PATH:-}"
VAR_MODEL_PATH="${VAR_MODEL_PATH:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_ROOT}/outputs/generation_eval_sets/$(date +%Y%m%d_%H%M%S)}"

WISE_IMAGE_DIR="${WISE_IMAGE_DIR:-}"
PURE_WISE_IMAGE_DIR="${PURE_WISE_IMAGE_DIR:-}"
TRAINED_WISE_IMAGE_DIR="${TRAINED_WISE_IMAGE_DIR:-}"

DISE_EVAL_CMD="${DISE_EVAL_CMD:-}"
DISE_EVAL_CMD_TEMPLATE="${DISE_EVAL_CMD_TEMPLATE:-}"

if [[ -n "${VAR_MODEL_PATH}" && -z "${PURE_VAR_MODEL_PATH}" && -z "${TRAINED_VAR_MODEL_PATH}" ]]; then
  EVAL_SETS="pure"
  PURE_VAR_MODEL_PATH="${VAR_MODEL_PATH}"
fi

resolve_var_model_path() {
  local set_name="$1"
  case "${set_name}" in
    pure) echo "${PURE_VAR_MODEL_PATH}" ;;
    trained_lora) echo "${TRAINED_VAR_MODEL_PATH}" ;;
    *) echo "" ;;
  esac
}

resolve_wise_image_dir() {
  local set_name="$1"
  local default_dir="$2"
  case "${set_name}" in
    pure)
      if [[ -n "${PURE_WISE_IMAGE_DIR}" ]]; then
        echo "${PURE_WISE_IMAGE_DIR}"
      elif [[ -n "${WISE_IMAGE_DIR}" ]]; then
        echo "${WISE_IMAGE_DIR}"
      else
        echo "${default_dir}"
      fi
      ;;
    trained_lora)
      if [[ -n "${TRAINED_WISE_IMAGE_DIR}" ]]; then
        echo "${TRAINED_WISE_IMAGE_DIR}"
      elif [[ -n "${WISE_IMAGE_DIR}" ]]; then
        echo "${WISE_IMAGE_DIR}"
      else
        echo "${default_dir}"
      fi
      ;;
    *)
      echo "${default_dir}"
      ;;
  esac
}

resolve_dise_cmd() {
  local set_name="$1"
  local image_dir="$2"
  local set_dir="$3"
  if [[ -n "${DISE_EVAL_CMD}" ]]; then
    echo "${DISE_EVAL_CMD}"
    return
  fi
  if [[ -n "${DISE_EVAL_CMD_TEMPLATE}" ]]; then
    local cmd="${DISE_EVAL_CMD_TEMPLATE}"
    cmd="${cmd//__IMAGE_DIR__/${image_dir}}"
    cmd="${cmd//__SET_NAME__/${set_name}}"
    cmd="${cmd//__SET_DIR__/${set_dir}}"
    echo "${cmd}"
    return
  fi
  echo ""
}

echo "=== VARGPT Generation Benchmark Launcher ==="
echo "  Sets:    ${EVAL_SETS}"
echo "  GenEval: ${RUN_GENEVAL}"
echo "  WISE:    ${RUN_WISE}"
echo "  DISE:    ${RUN_DISE}"
echo "  Output:  ${OUTPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a _eval_sets <<< "${EVAL_SETS}"
for raw_set in "${_eval_sets[@]}"; do
  set_name="$(echo "${raw_set}" | xargs)"
  [[ -z "${set_name}" ]] && continue

  case "${set_name}" in
    pure|trained_lora) ;;
    *)
      echo "[ERROR] Unsupported EVAL_SETS token: '${set_name}'. Use pure,trained_lora." >&2
      exit 1
      ;;
  esac

  set_var_model_path="$(resolve_var_model_path "${set_name}")"
  if [[ -z "${set_var_model_path}" ]]; then
    echo "[ERROR] Missing VAR checkpoint for set '${set_name}'." >&2
    echo "Set PURE_VAR_MODEL_PATH / TRAINED_VAR_MODEL_PATH as needed." >&2
    exit 1
  fi

  set_dir="${OUTPUT_ROOT}/${set_name}"
  geneval_dir="${set_dir}/geneval"
  wise_image_dir="$(resolve_wise_image_dir "${set_name}" "${geneval_dir}")"
  mkdir -p "${set_dir}"

  echo "--- Running set: ${set_name} ---"
  echo "  checkpoint: ${set_var_model_path}"
  echo "  set_dir:    ${set_dir}"

  if [[ "${RUN_GENEVAL}" == "1" ]]; then
    VAR_MODEL_PATH="${set_var_model_path}" \
    GENEVAL_OUT_DIR="${geneval_dir}" \
    bash "${SCRIPT_DIR}/run_eval_vargpt_geneval.sh"
  fi

  if [[ "${RUN_WISE}" == "1" ]]; then
    if [[ -z "${wise_image_dir}" ]]; then
      echo "[ERROR] WISE_IMAGE_DIR is empty for set '${set_name}'." >&2
      exit 1
    fi
    WISE_IMAGE_DIR="${wise_image_dir}" \
    bash "${SCRIPT_DIR}/run_eval_vargpt_wise.sh"
  fi

  if [[ "${RUN_DISE}" == "1" ]]; then
    set_dise_cmd="$(resolve_dise_cmd "${set_name}" "${wise_image_dir}" "${set_dir}")"
    if [[ -z "${set_dise_cmd}" ]]; then
      echo "[ERROR] DISE requires DISE_EVAL_CMD or DISE_EVAL_CMD_TEMPLATE." >&2
      exit 1
    fi
    DISE_EVAL_CMD="${set_dise_cmd}" \
    bash "${SCRIPT_DIR}/run_eval_vargpt_dise.sh"
  fi
done

echo "Done. Generation eval sets saved under: ${OUTPUT_ROOT}"
