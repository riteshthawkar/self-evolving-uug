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

# Unified generation benchmark launcher for VARGPT (GenEval + WISE + DISE).
#
# Supports two runtime modes per set:
#   1) var mode (legacy): Infinity/VAR checkpoint (.pth)
#   2) hf mode (self-evolving): base model + optional LoRA adapter
#
# Primary set selection:
#   EVAL_SETS=pure,trained_lora  (default)
#
# For var mode inputs:
#   PURE_VAR_MODEL_PATH
#   TRAINED_VAR_MODEL_PATH
#
# For hf mode inputs:
#   PURE_MODEL_PATH / TRAINED_MODEL_PATH         (full model dir or adapter dir)
#   PURE_LORA_PATH / TRAINED_LORA_PATH           (optional explicit adapter path)
#   PURE_LORA_ADAPTER_NAME / TRAINED_LORA_ADAPTER_NAME (default: default)
#   PURE_BASE_MODEL_PATH / TRAINED_BASE_MODEL_PATH      (if adapter path lacks/has wrong base)
#   PURE_SE_RUN_DIR / TRAINED_SE_RUN_DIR         (auto-resolve latest se_checkpoint_*/model)
#   PURE_SE_STEP / TRAINED_SE_STEP               (optional explicit step)
#
# Runtime override:
#   PURE_RUNTIME=auto|var|hf
#   TRAINED_RUNTIME=auto|var|hf
#
# Legacy compatibility:
#   VAR_MODEL_PATH alone => pure set in var mode.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RUN_GENEVAL="${RUN_GENEVAL:-1}"
RUN_WISE="${RUN_WISE:-1}"
RUN_DISE="${RUN_DISE:-1}"
EVAL_SETS="${EVAL_SETS:-pure,trained_lora}"

PURE_RUNTIME="${PURE_RUNTIME:-auto}"
TRAINED_RUNTIME="${TRAINED_RUNTIME:-auto}"

PURE_VAR_MODEL_PATH="${PURE_VAR_MODEL_PATH:-}"
TRAINED_VAR_MODEL_PATH="${TRAINED_VAR_MODEL_PATH:-}"
VAR_MODEL_PATH="${VAR_MODEL_PATH:-}"

PURE_MODEL_PATH="${PURE_MODEL_PATH:-}"
PURE_LORA_PATH="${PURE_LORA_PATH:-}"
PURE_LORA_ADAPTER_NAME="${PURE_LORA_ADAPTER_NAME:-default}"
PURE_BASE_MODEL_PATH="${PURE_BASE_MODEL_PATH:-}"
PURE_SE_RUN_DIR="${PURE_SE_RUN_DIR:-}"
PURE_SE_STEP="${PURE_SE_STEP:-}"

TRAINED_MODEL_PATH="${TRAINED_MODEL_PATH:-}"
TRAINED_LORA_PATH="${TRAINED_LORA_PATH:-}"
TRAINED_LORA_ADAPTER_NAME="${TRAINED_LORA_ADAPTER_NAME:-default}"
TRAINED_BASE_MODEL_PATH="${TRAINED_BASE_MODEL_PATH:-}"
TRAINED_SE_RUN_DIR="${TRAINED_SE_RUN_DIR:-}"
TRAINED_SE_STEP="${TRAINED_SE_STEP:-}"

HF_DEVICE="${HF_DEVICE:-cuda}"
HF_DTYPE="${HF_DTYPE:-bfloat16}"
HF_MAX_NEW_TOKENS="${HF_MAX_NEW_TOKENS:-4096}"
HF_DO_SAMPLE="${HF_DO_SAMPLE:-1}"
HF_TEMPERATURE="${HF_TEMPERATURE:-1.0}"
HF_TOP_P="${HF_TOP_P:-1.0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_ROOT}/outputs/generation_eval_sets/$(date +%Y%m%d_%H%M%S)}"

WISE_IMAGE_DIR="${WISE_IMAGE_DIR:-}"
PURE_WISE_IMAGE_DIR="${PURE_WISE_IMAGE_DIR:-}"
TRAINED_WISE_IMAGE_DIR="${TRAINED_WISE_IMAGE_DIR:-}"

DISE_EVAL_CMD="${DISE_EVAL_CMD:-}"
DISE_EVAL_CMD_TEMPLATE="${DISE_EVAL_CMD_TEMPLATE:-}"

if [[ -n "${VAR_MODEL_PATH}" && -z "${PURE_VAR_MODEL_PATH}" && -z "${TRAINED_VAR_MODEL_PATH}" ]]; then
  EVAL_SETS="pure"
  PURE_VAR_MODEL_PATH="${VAR_MODEL_PATH}"
  PURE_RUNTIME="var"
fi

_latest_se_checkpoint_dir() {
  local run_dir="$1"
  if [[ ! -d "${run_dir}" ]]; then
    return 1
  fi
  local latest
  latest="$(
    find "${run_dir}" -maxdepth 1 -type d -name 'se_checkpoint_*' -print 2>/dev/null \
      | awk -F'_' '{print $NF "|" $0}' \
      | awk -F'|' '$1 ~ /^[0-9]+$/' \
      | sort -t'|' -k1,1n \
      | tail -n 1 \
      | cut -d'|' -f2-
  )"
  [[ -n "${latest}" ]] && echo "${latest}"
}

_resolve_se_artifact_dir() {
  local run_dir="$1"
  local step="$2"
  local resolved=""
  if [[ -z "${run_dir}" ]]; then
    return 0
  fi
  if [[ ! -d "${run_dir}" ]]; then
    echo "[ERROR] SE run dir not found: ${run_dir}" >&2
    exit 1
  fi
  if [[ -n "${step}" ]]; then
    local explicit="${run_dir}/se_checkpoint_${step}"
    if [[ ! -d "${explicit}" ]]; then
      echo "[ERROR] Requested checkpoint not found: ${explicit}" >&2
      exit 1
    fi
    resolved="${explicit}"
  else
    resolved="$(_latest_se_checkpoint_dir "${run_dir}")"
    if [[ -z "${resolved}" ]]; then
      resolved="${run_dir}"
    fi
  fi
  if [[ -d "${resolved}/model" ]]; then
    echo "${resolved}/model"
  else
    echo "${resolved}"
  fi
}

_json_base_model_from_adapter() {
  local adapter_dir="$1"
  python3 - <<'PY' "${adapter_dir}"
import json, os, sys
adapter_dir = sys.argv[1]
cfg_path = os.path.join(adapter_dir, "adapter_config.json")
if not os.path.isfile(cfg_path):
    print("")
    raise SystemExit(0)
try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(cfg.get("base_model_name_or_path", "") or "")
except Exception:
    print("")
PY
}

_resolve_model_and_adapter() {
  local path_hint="$1"
  local adapter_name="$2"
  local base_override="${3:-}"

  local resolved_model=""
  local resolved_adapter=""

  if [[ -z "${path_hint}" ]]; then
    echo ";;"
    return 0
  fi
  if [[ ! -e "${path_hint}" ]]; then
    echo "[ERROR] Path does not exist: ${path_hint}" >&2
    exit 1
  fi

  if [[ -f "${path_hint}" ]]; then
    # likely legacy VAR .pth path
    echo "${path_hint};"
    return 0
  fi

  if [[ -f "${path_hint}/config.json" ]]; then
    resolved_model="${path_hint}"
    resolved_adapter=""
    echo "${resolved_model};${resolved_adapter}"
    return 0
  fi

  if [[ -f "${path_hint}/${adapter_name}/adapter_config.json" ]]; then
    resolved_adapter="${path_hint}/${adapter_name}"
  elif [[ -f "${path_hint}/adapter_config.json" ]]; then
    resolved_adapter="${path_hint}"
  elif [[ -f "${path_hint}/default/adapter_config.json" ]]; then
    resolved_adapter="${path_hint}/default"
  else
    echo "[ERROR] Could not find config.json or adapter_config.json under: ${path_hint}" >&2
    exit 1
  fi

  local inferred_base="${base_override}"
  if [[ -z "${inferred_base}" ]]; then
    inferred_base="$(_json_base_model_from_adapter "${resolved_adapter}")"
  fi
  if [[ -z "${inferred_base}" ]]; then
    echo "[ERROR] Could not infer base model for adapter: ${resolved_adapter}" >&2
    echo "Set *_BASE_MODEL_PATH for this eval set." >&2
    exit 1
  fi
  resolved_model="${inferred_base}"
  echo "${resolved_model};${resolved_adapter}"
}

resolve_set_runtime() {
  local forced="$1"
  local set_var="$2"
  local set_model="$3"
  local set_lora="$4"

  if [[ "${forced}" == "var" || "${forced}" == "hf" ]]; then
    echo "${forced}"
    return 0
  fi
  if [[ -n "${set_var}" ]]; then
    echo "var"
    return 0
  fi
  if [[ -n "${set_model}" && -f "${set_model}" ]]; then
    echo "var"
    return 0
  fi
  if [[ -n "${set_model}" || -n "${set_lora}" ]]; then
    echo "hf"
    return 0
  fi
  echo ""
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

  set_runtime=""
  set_var_model_path=""
  set_model_path=""
  set_lora_path=""
  set_lora_adapter_name=""
  set_base_override=""
  set_se_run_dir=""
  set_se_step=""

  case "${set_name}" in
    pure)
      set_runtime="${PURE_RUNTIME}"
      set_var_model_path="${PURE_VAR_MODEL_PATH}"
      set_model_path="${PURE_MODEL_PATH}"
      set_lora_path="${PURE_LORA_PATH}"
      set_lora_adapter_name="${PURE_LORA_ADAPTER_NAME}"
      set_base_override="${PURE_BASE_MODEL_PATH}"
      set_se_run_dir="${PURE_SE_RUN_DIR}"
      set_se_step="${PURE_SE_STEP}"
      ;;
    trained_lora)
      set_runtime="${TRAINED_RUNTIME}"
      set_var_model_path="${TRAINED_VAR_MODEL_PATH}"
      set_model_path="${TRAINED_MODEL_PATH}"
      set_lora_path="${TRAINED_LORA_PATH}"
      set_lora_adapter_name="${TRAINED_LORA_ADAPTER_NAME}"
      set_base_override="${TRAINED_BASE_MODEL_PATH}"
      set_se_run_dir="${TRAINED_SE_RUN_DIR}"
      set_se_step="${TRAINED_SE_STEP}"
      ;;
  esac

  if [[ -z "${set_model_path}" && -z "${set_lora_path}" && -z "${set_var_model_path}" && -n "${set_se_run_dir}" ]]; then
    set_model_path="$(_resolve_se_artifact_dir "${set_se_run_dir}" "${set_se_step}")"
  fi

  set_runtime="$(resolve_set_runtime "${set_runtime}" "${set_var_model_path}" "${set_model_path}" "${set_lora_path}")"
  if [[ -z "${set_runtime}" ]]; then
    echo "[ERROR] Could not determine runtime for set '${set_name}'." >&2
    echo "Provide *_VAR_MODEL_PATH (var mode) OR *_MODEL_PATH/*_LORA_PATH (hf mode)." >&2
    exit 1
  fi

  set_dir="${OUTPUT_ROOT}/${set_name}"
  geneval_dir="${set_dir}/geneval"
  wise_image_dir="$(resolve_wise_image_dir "${set_name}" "${geneval_dir}")"
  mkdir -p "${set_dir}"

  echo "--- Running set: ${set_name} ---"
  echo "  runtime:    ${set_runtime}"
  echo "  set_dir:    ${set_dir}"

  if [[ "${RUN_GENEVAL}" == "1" ]]; then
    if [[ "${set_runtime}" == "var" ]]; then
      if [[ -z "${set_var_model_path}" && -n "${set_model_path}" && -f "${set_model_path}" ]]; then
        set_var_model_path="${set_model_path}"
      fi
      if [[ -z "${set_var_model_path}" ]]; then
        echo "[ERROR] Missing VAR checkpoint for set '${set_name}'." >&2
        exit 1
      fi
      echo "  var_ckpt:   ${set_var_model_path}"
      VAR_MODEL_PATH="${set_var_model_path}" \
      GENEVAL_OUT_DIR="${geneval_dir}" \
      bash "${SCRIPT_DIR}/run_eval_vargpt_geneval.sh"
    else
      hf_pair=";;"
      if [[ -n "${set_model_path}" ]]; then
        hf_pair="$(_resolve_model_and_adapter "${set_model_path}" "${set_lora_adapter_name}" "${set_base_override}")"
      elif [[ -n "${set_lora_path}" ]]; then
        hf_pair="$(_resolve_model_and_adapter "${set_lora_path}" "${set_lora_adapter_name}" "${set_base_override}")"
      fi
      hf_pretrained="${hf_pair%%;*}"
      hf_discovered_adapter="${hf_pair#*;}"
      if [[ -z "${set_lora_path}" ]]; then
        set_lora_path="${hf_discovered_adapter}"
      fi
      if [[ -z "${hf_pretrained}" ]]; then
        echo "[ERROR] Missing HF pretrained model path for set '${set_name}'." >&2
        exit 1
      fi
      echo "  hf_model:   ${hf_pretrained}"
      if [[ -n "${set_lora_path}" ]]; then
        echo "  hf_adapter: ${set_lora_path} (adapter=${set_lora_adapter_name})"
      fi
      HF_PRETRAINED_PATH="${hf_pretrained}" \
      HF_PEFT_PATH="${set_lora_path}" \
      HF_PEFT_ADAPTER_NAME="${set_lora_adapter_name}" \
      HF_DEVICE="${HF_DEVICE}" \
      HF_DTYPE="${HF_DTYPE}" \
      HF_MAX_NEW_TOKENS="${HF_MAX_NEW_TOKENS}" \
      HF_DO_SAMPLE="${HF_DO_SAMPLE}" \
      HF_TEMPERATURE="${HF_TEMPERATURE}" \
      HF_TOP_P="${HF_TOP_P}" \
      GENEVAL_OUT_DIR="${geneval_dir}" \
      bash "${SCRIPT_DIR}/run_eval_vargpt_geneval_hf.sh"
    fi
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
