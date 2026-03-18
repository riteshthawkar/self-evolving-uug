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

# Multi-benchmark understanding evaluation for VARGPT (lmms_eval).
#
# Usage:
#   PURE_MODEL_PATH=/path/to/pure_model \
#   TRAINED_MODEL_PATH=/path/to/trained_base_model \
#   TRAINED_LORA_PATH=/path/to/trained_lora_adapter \
#   TASKS=mmmu,mme,textvqa_val \
#   bash run_scripts/run_eval_vargpt_understanding_bench.sh
#
# Optional env:
#   MODEL                default: vargpt_qwen2vl_v1_1
#   EVAL_SETS            default: pure,trained_lora
#                        allowed tokens: pure,trained_lora
#   PURE_MODEL_PATH      base model path for pure evaluation set
#   PURE_BASE_MODEL_PATH explicit base model if PURE_MODEL_PATH is adapter-only
#   PURE_SE_RUN_DIR      self-evolving run output dir for pure set (optional)
#   PURE_SE_STEP         specific se checkpoint step for pure set (optional)
#   PURE_MODEL_ARGS_EXTRA additional lmms model args for pure set
#   TRAINED_MODEL_PATH   base model path for trained set
#   TRAINED_LORA_PATH    optional LoRA adapter path for trained set
#   TRAINED_LORA_ADAPTER_NAME  adapter name to activate (default: solver)
#   TRAINED_SE_RUN_DIR   self-evolving run output dir for trained set
#   TRAINED_SE_STEP      specific se checkpoint step for trained set
#   TRAINED_BASE_MODEL_PATH explicit base model if adapter config is missing/incomplete
#   TRAINED_MODEL_ARGS_EXTRA additional lmms model args for trained set
#   NUM_PROCESSES        default: 8
#   MAIN_PROCESS_PORT    default: 39535
#   BATCH_SIZE           default: 1
#   OUTPUT_ROOT          default: <train_root>/logs/understanding_eval/<timestamp>
#   LOG_SAMPLES          default: 1
#   LOG_SAMPLES_SUFFIX   default: vargpt_understanding
#   UNDERSTAND_EVAL_DIR  default: <repo>/suder_vargpt/understand_eval
#
# Legacy mode:
#   MODEL_PATH + MODEL_ARGS_EXTRA still works and runs only one set ("pure").

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SUDER_ROOT="$(cd -- "${TRAIN_ROOT}/.." && pwd)"
UNDERSTAND_EVAL_DIR="${UNDERSTAND_EVAL_DIR:-${SUDER_ROOT}/understand_eval}"

MODEL="${MODEL:-vargpt_qwen2vl_v1_1}"
TASKS="${TASKS:-mmmu}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-39535}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_SAMPLES="${LOG_SAMPLES:-1}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-vargpt_understanding}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_ROOT}/logs/understanding_eval/$(date +%Y%m%d_%H%M%S)}"

EVAL_SETS="${EVAL_SETS:-pure,trained_lora}"
PURE_MODEL_PATH="${PURE_MODEL_PATH:-}"
PURE_BASE_MODEL_PATH="${PURE_BASE_MODEL_PATH:-}"
PURE_SE_RUN_DIR="${PURE_SE_RUN_DIR:-}"
PURE_SE_STEP="${PURE_SE_STEP:-}"
PURE_MODEL_ARGS_EXTRA="${PURE_MODEL_ARGS_EXTRA:-}"
TRAINED_MODEL_PATH="${TRAINED_MODEL_PATH:-}"
TRAINED_LORA_PATH="${TRAINED_LORA_PATH:-}"
TRAINED_LORA_ADAPTER_NAME="${TRAINED_LORA_ADAPTER_NAME:-solver}"
TRAINED_SE_RUN_DIR="${TRAINED_SE_RUN_DIR:-}"
TRAINED_SE_STEP="${TRAINED_SE_STEP:-}"
TRAINED_BASE_MODEL_PATH="${TRAINED_BASE_MODEL_PATH:-}"
TRAINED_MODEL_ARGS_EXTRA="${TRAINED_MODEL_ARGS_EXTRA:-}"

MODEL_PATH="${MODEL_PATH:-}"
MODEL_ARGS_EXTRA="${MODEL_ARGS_EXTRA:-}"

if [[ ! -d "${UNDERSTAND_EVAL_DIR}" ]]; then
  echo "[ERROR] understand_eval dir not found: ${UNDERSTAND_EVAL_DIR}" >&2
  exit 1
fi

if [[ -n "${MODEL_PATH}" && -z "${PURE_MODEL_PATH}" && -z "${TRAINED_MODEL_PATH}" ]]; then
  EVAL_SETS="pure"
  PURE_MODEL_PATH="${MODEL_PATH}"
  PURE_MODEL_ARGS_EXTRA="${MODEL_ARGS_EXTRA}"
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
  if [[ ! -d "${path_hint}" ]]; then
    echo "[ERROR] Path does not exist: ${path_hint}" >&2
    exit 1
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
    echo "[ERROR] Could not find a full model (config.json) or adapter_config.json under: ${path_hint}" >&2
    exit 1
  fi

  local inferred_base="${base_override}"
  if [[ -z "${inferred_base}" ]]; then
    inferred_base="$(_json_base_model_from_adapter "${resolved_adapter}")"
  fi
  if [[ -z "${inferred_base}" ]]; then
    echo "[ERROR] Could not infer base model for adapter: ${resolved_adapter}" >&2
    echo "Set TRAINED_BASE_MODEL_PATH (or PURE_MODEL_PATH for pure adapter eval)." >&2
    exit 1
  fi
  resolved_model="${inferred_base}"
  echo "${resolved_model};${resolved_adapter}"
}

if [[ -z "${PURE_MODEL_PATH}" && -n "${PURE_SE_RUN_DIR}" ]]; then
  PURE_MODEL_PATH="$(_resolve_se_artifact_dir "${PURE_SE_RUN_DIR}" "${PURE_SE_STEP}")"
fi

if [[ -z "${TRAINED_MODEL_PATH}" && -z "${TRAINED_LORA_PATH}" && -n "${TRAINED_SE_RUN_DIR}" ]]; then
  TRAINED_MODEL_PATH="$(_resolve_se_artifact_dir "${TRAINED_SE_RUN_DIR}" "${TRAINED_SE_STEP}")"
fi

build_model_args() {
  local base_path="$1"
  local extra_args="$2"
  local peft_path="${3:-}"
  local adapter_name="${4:-}"
  local model_args="pretrained=${base_path}"

  if [[ -n "${peft_path}" ]]; then
    model_args="${model_args},peft=${peft_path}"
    if [[ -n "${adapter_name}" ]]; then
      model_args="${model_args},peft_adapter_name=${adapter_name}"
    fi
  fi

  if [[ -n "${extra_args}" ]]; then
    model_args="${model_args},${extra_args}"
  fi
  echo "${model_args}"
}

run_eval_set() {
  local set_name="$1"
  local set_model_path="$2"
  local set_extra_args="$3"
  local set_peft_path="${4:-}"
  local set_adapter_name="${5:-}"
  local set_base_override="${6:-}"

  if [[ -n "${set_model_path}" ]]; then
    local resolved_pair
    resolved_pair="$(_resolve_model_and_adapter "${set_model_path}" "${set_adapter_name}" "${set_base_override}")"
    set_model_path="${resolved_pair%%;*}"
    local discovered_adapter="${resolved_pair#*;}"
    if [[ -z "${set_peft_path}" ]]; then
      set_peft_path="${discovered_adapter}"
    fi
  elif [[ -n "${set_peft_path}" ]]; then
    local resolved_pair
    resolved_pair="$(_resolve_model_and_adapter "${set_peft_path}" "${set_adapter_name}" "${set_base_override}")"
    set_model_path="${resolved_pair%%;*}"
  fi

  if [[ -z "${set_model_path}" ]]; then
    echo "[ERROR] Missing model path for set '${set_name}'." >&2
    exit 1
  fi

  local set_output="${OUTPUT_ROOT}/${set_name}"
  local set_suffix="${LOG_SAMPLES_SUFFIX}_${set_name}"
  local set_model_args
  set_model_args="$(build_model_args "${set_model_path}" "${set_extra_args}" "${set_peft_path}" "${set_adapter_name}")"

  mkdir -p "${set_output}"

  echo "=== VARGPT Understanding Evaluation (${set_name}) ==="
  echo "  model:       ${MODEL}"
  echo "  model_path:  ${set_model_path}"
  if [[ -n "${set_peft_path}" ]]; then
    echo "  lora_path:   ${set_peft_path}"
    echo "  adapter:     ${set_adapter_name}"
  elif [[ "${set_name}" == "trained_lora" ]]; then
    echo "  note:        TRAINED_LORA_PATH is empty; evaluating trained set from base/merged model path only."
  fi
  echo "  tasks:       ${TASKS}"
  echo "  gpus:        ${NUM_PROCESSES}"
  echo "  output:      ${set_output}"

  local -a cmd=(
    python3 -m accelerate.commands.launch
    --num_processes "${NUM_PROCESSES}"
    --main_process_port "${MAIN_PROCESS_PORT}"
    -m lmms_eval
    --model "${MODEL}"
    --model_args "${set_model_args}"
    --tasks "${TASKS}"
    --batch_size "${BATCH_SIZE}"
    --output_path "${set_output}"
  )

  if [[ "${LOG_SAMPLES}" == "1" ]]; then
    cmd+=(--log_samples --log_samples_suffix "${set_suffix}")
  fi

  "${cmd[@]}"
}

mkdir -p "${OUTPUT_ROOT}"
cd "${UNDERSTAND_EVAL_DIR}"
export PYTHONPATH="${UNDERSTAND_EVAL_DIR}:${PYTHONPATH:-}"

# Ensure legacy import alias exists:
#   lmms_eval.models.visionllm.vargpt -> lmms_eval.models.visionllm.vargpt_llava
# Several VARGPT eval modules import through `...visionllm.vargpt...`.
VISIONLLM_DIR="${UNDERSTAND_EVAL_DIR}/lmms_eval/models/visionllm"
if [[ -d "${VISIONLLM_DIR}/vargpt_llava" && ! -e "${VISIONLLM_DIR}/vargpt" ]]; then
  ln -sfn vargpt_llava "${VISIONLLM_DIR}/vargpt"
fi
if [[ ! -e "${VISIONLLM_DIR}/vargpt" ]]; then
  echo "[ERROR] Missing visionllm alias path: ${VISIONLLM_DIR}/vargpt" >&2
  echo "[ERROR] Expected ${VISIONLLM_DIR}/vargpt_llava to exist." >&2
  exit 1
fi

# Also support legacy top-level imports:
#   visionllm.vargpt...
TOP_VISIONLLM_LINK="${UNDERSTAND_EVAL_DIR}/visionllm"
if [[ -d "${VISIONLLM_DIR}" && ! -e "${TOP_VISIONLLM_LINK}" ]]; then
  ln -sfn "${VISIONLLM_DIR}" "${TOP_VISIONLLM_LINK}"
fi
if [[ ! -e "${TOP_VISIONLLM_LINK}" ]]; then
  echo "[ERROR] Missing top-level visionllm package alias: ${TOP_VISIONLLM_LINK}" >&2
  exit 1
fi

IFS=',' read -r -a _eval_sets <<< "${EVAL_SETS}"
for raw_set in "${_eval_sets[@]}"; do
  set_name="$(echo "${raw_set}" | xargs)"
  [[ -z "${set_name}" ]] && continue

  case "${set_name}" in
    pure)
      run_eval_set "pure" "${PURE_MODEL_PATH}" "${PURE_MODEL_ARGS_EXTRA}" "" "" "${PURE_BASE_MODEL_PATH}"
      ;;
    trained_lora)
      if [[ -n "${TRAINED_LORA_PATH}" ]]; then
        run_eval_set "trained_lora" "${TRAINED_MODEL_PATH}" "${TRAINED_MODEL_ARGS_EXTRA}" "${TRAINED_LORA_PATH}" "${TRAINED_LORA_ADAPTER_NAME}" "${TRAINED_BASE_MODEL_PATH}"
      else
        run_eval_set "trained_lora" "${TRAINED_MODEL_PATH}" "${TRAINED_MODEL_ARGS_EXTRA}" "" "${TRAINED_LORA_ADAPTER_NAME}" "${TRAINED_BASE_MODEL_PATH}"
      fi
      ;;
    *)
      echo "[ERROR] Unsupported EVAL_SETS token: '${set_name}'. Use pure,trained_lora." >&2
      exit 1
      ;;
  esac
done

echo "Done. Understanding eval sets saved under: ${OUTPUT_ROOT}"
