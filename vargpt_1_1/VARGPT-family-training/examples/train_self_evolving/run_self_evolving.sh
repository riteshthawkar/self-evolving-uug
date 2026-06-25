#!/bin/bash
# =============================================================================
# VARGPT Self-Evolving Training Launch Script
#
# Experiments:
#   E1 (Joint 3U+2G):       vargpt_se_joint.yaml          (main experiment)
#   E2 (Gen-only 0U+5G):    vargpt_se_gen_only.yaml       (ablation)
#   E3 (U-only 5U+0G):      vargpt_se_u_only.yaml         (ablation)
#
# Usage:
#   bash examples/train_self_evolving/run_self_evolving.sh [joint|gen_only|u_only] [NUM_GPUS]
#
# Environment variables:
#   RESUME_FROM   - checkpoint path to resume from (optional)
#   MASTER_PORT   - DDP master port (default: 39600)
#   WANDB_PROJECT - W&B project name (default: vargpt-self-evolving)
# =============================================================================
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

HF_TOKEN_FILE="${HF_TOKEN_FILE:-${ORIGINAL_HOME:-$HOME}/.cache/huggingface/token}"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(< "$HF_TOKEN_FILE")"
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
detect_num_gpus() {
    # 1) explicit arg/environment
    if [ -n "${NUM_GPUS:-}" ]; then
        echo "${NUM_GPUS}"
        return 0
    fi

    # 2) derive from visible-device envs if present
    local visible="${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-}}"
    if [ -n "${visible}" ]; then
        local cleaned="${visible// /}"
        if [ -n "${cleaned}" ]; then
            # Count comma-separated entries.
            awk -F',' '{print NF}' <<< "${cleaned}"
            return 0
        fi
    fi

    # 3) derive from torch runtime if available
    local tcount
    tcount="$(
        python - <<'PY' 2>/dev/null || true
import torch
try:
    print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
    )"
    tcount="$(echo "${tcount}" | tr -d '[:space:]')"
    if [[ "${tcount}" =~ ^[0-9]+$ ]] && [ "${tcount}" -gt 0 ]; then
        echo "${tcount}"
        return 0
    fi

    # 4) derive from nvidia-smi if available
    if command -v nvidia-smi >/dev/null 2>&1; then
        local n
        n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')"
        if [[ "${n}" =~ ^[0-9]+$ ]] && [ "${n}" -gt 0 ]; then
            echo "${n}"
            return 0
        fi
    fi

    # 5) derive from rocm-smi if available
    if command -v rocm-smi >/dev/null 2>&1; then
        local r
        r="$(
            rocm-smi 2>/dev/null \
              | awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+0x/ {c++} END {print c+0}'
        )"
        r="$(echo "${r}" | tr -d '[:space:]')"
        if [[ "${r}" =~ ^[0-9]+$ ]] && [ "${r}" -gt 0 ]; then
            echo "${r}"
            return 0
        fi
    fi

    # 6) safe fallback
    echo 1
}

resolve_launcher() {
    LAUNCHER=()
    if python -c "import llamafactory.cli" >/dev/null 2>&1; then
        LAUNCHER=("python" "-m" "llamafactory.cli")
        return 0
    fi
    if command -v llamafactory-cli >/dev/null 2>&1; then
        LAUNCHER=("llamafactory-cli")
        return 0
    fi
    return 1
}

check_version_stack() {
    local py_exec
    py_exec="$(python - <<'PY' 2>/dev/null || true
import sys
print(sys.executable)
PY
    )"
    py_exec="$(echo "${py_exec}" | tr -d '[:space:]')"
    if [ -z "${py_exec}" ]; then
        py_exec="python"
    fi

    local check_out
    check_out="$(
        python - <<'PY' 2>&1 || true
import sys

requirements = [
    "transformers>=4.41.2,<=4.46.1",
    "datasets>=2.16.0,<=3.1.0",
    "accelerate>=0.34.0,<=1.0.1",
    "peft>=0.11.1,<=0.12.0",
    "trl>=0.8.6,<=0.9.6",
]

try:
    from transformers.utils.versions import require_version
except Exception as exc:
    print(f"FAILED: cannot import transformers version utilities: {exc}")
    sys.exit(2)

errors = []
for req in requirements:
    try:
        require_version(req, "")
    except Exception as exc:
        errors.append((req, str(exc).splitlines()[0]))

if errors:
    print("FAILED")
    for req, msg in errors:
        print(f"  - {req} :: {msg}")
    sys.exit(3)

print("OK")
PY
    )"

    if ! grep -q "^OK$" <<< "${check_out}"; then
        echo "[ERROR] Incompatible Python package stack for VARGPT/LlamaFactory." >&2
        echo "[ERROR] Active python: ${py_exec}" >&2
        echo "${check_out}" | sed 's/^/[ERROR] /' >&2
        echo "[ERROR] Fix in this exact environment with:" >&2
        echo "[ERROR]   ${py_exec} -m pip install -U \\" >&2
        echo "[ERROR]     \"transformers>=4.41.2,<=4.46.1\" \\" >&2
        echo "[ERROR]     \"datasets>=2.16.0,<=3.1.0\" \\" >&2
        echo "[ERROR]     \"accelerate>=0.34.0,<=1.0.1\" \\" >&2
        echo "[ERROR]     \"peft>=0.11.1,<=0.12.0\" \\" >&2
        echo "[ERROR]     \"trl>=0.8.6,<=0.9.6\" \\" >&2
        echo "[ERROR]     \"tokenizers>=0.19.0,<0.20.4\" \\" >&2
        echo "[ERROR]     \"deepspeed==0.15.4\"" >&2
        echo "[ERROR] Do not reinstall torch in ROCm env." >&2
        exit 1
    fi
}

# ── Parse arguments ──────────────────────────────────────────────────────────
EXPERIMENT=${1:-joint}
NUM_GPUS="${2:-}"

if [ -z "${NUM_GPUS}" ]; then
    NUM_GPUS="$(detect_num_gpus)"
fi
NUM_GPUS="$(echo "${NUM_GPUS}" | tr -d '[:space:]')"

if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [ "${NUM_GPUS}" -lt 1 ]; then
    echo "[ERROR] NUM_GPUS must be a positive integer, got: '${NUM_GPUS}'" >&2
    exit 1
fi

case "$EXPERIMENT" in
    joint)
        CONFIG="examples/train_self_evolving/vargpt_se_joint.yaml"
        ;;
    gen_only)
        CONFIG="examples/train_self_evolving/vargpt_se_gen_only.yaml"
        ;;
    u_only)
        CONFIG="examples/train_self_evolving/vargpt_se_u_only.yaml"
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Usage: $0 [joint|gen_only|u_only] [NUM_GPUS]"
        exit 1
        ;;
esac

# ── Environment setup ────────────────────────────────────────────────────────
export MASTER_PORT=${MASTER_PORT:-39600}
export WANDB_PROJECT=${WANDB_PROJECT:-vargpt-self-evolving}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
find_repo_root() {
    local d="$1"
    while [ -n "${d}" ] && [ "${d}" != "/" ]; do
        if [ -d "${d}/src/llamafactory" ] && [ -f "${d}/examples/train_self_evolving/vargpt_se_joint.yaml" ]; then
            echo "${d}"
            return 0
        fi
        d="$(dirname "${d}")"
    done
    return 1
}
REPO_ROOT="$(find_repo_root "${SCRIPT_DIR}" || true)"
if [ -z "${REPO_ROOT}" ]; then
    # Fallback to historical layout assumption.
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
CONFIG="${REPO_ROOT}/${CONFIG}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
IMAGE_FOLDER="${IMAGE_FOLDER:-}"

# Ensure visionllm packages are importable and legacy alias exists.
if [ -d "${REPO_ROOT}/visionllm" ]; then
    find "${REPO_ROOT}/visionllm" -type d ! -name '__pycache__' \
        -exec sh -c 'test -f "$1/__init__.py" || touch "$1/__init__.py"' _ {} \;
    if [ ! -e "${REPO_ROOT}/visionllm/vargpt" ] && [ -d "${REPO_ROOT}/visionllm/vargpt_llava" ]; then
        ln -sfn vargpt_llava "${REPO_ROOT}/visionllm/vargpt"
    fi
fi

# Keep CUDA/HIP visibility aligned for mixed launcher stacks.
if [ -z "${HIP_VISIBLE_DEVICES:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
fi

ROCM_RUNTIME="$(
    python - <<'PY' 2>/dev/null || true
import torch
try:
    print(1 if getattr(torch.version, "hip", None) else 0)
except Exception:
    print(0)
PY
)"
ROCM_RUNTIME="$(echo "${ROCM_RUNTIME}" | tr -d '[:space:]')"
if ! [[ "${ROCM_RUNTIME}" =~ ^[0-9]+$ ]]; then
    ROCM_RUNTIME=0
fi

FLASH_ATTN_MODE="${FLASH_ATTN_MODE:-}"
if [ -z "${FLASH_ATTN_MODE}" ]; then
    if [ "${ROCM_RUNTIME}" = "1" ]; then
        FLASH_ATTN_MODE="sdpa"
    else
        FLASH_ATTN_MODE="disabled"
    fi
fi

if ! resolve_launcher; then
    echo "[ERROR] Could not find LlamaFactory launcher." >&2
    echo "Install with: pip install -e ${REPO_ROOT} --no-deps" >&2
    echo "NOTE: this repo pins torch==2.1.0 in requirements; avoid reinstalling torch on ROCm." >&2
    echo "Or ensure 'python -m llamafactory.cli' imports in current environment." >&2
    exit 1
fi
check_version_stack

LLAMAFACTORY_MODULE_DIR="$(
    python - <<'PY' 2>/dev/null || true
import os
import llamafactory
print(os.path.abspath(os.path.dirname(llamafactory.__file__)))
PY
)"
LLAMAFACTORY_MODULE_DIR="$(echo "${LLAMAFACTORY_MODULE_DIR}" | tr -d '[:space:]')"
EXPECTED_MODULE_DIR="${REPO_ROOT}/src/llamafactory"
if [ ! -d "${EXPECTED_MODULE_DIR}" ]; then
    echo "[ERROR] Expected llamafactory source dir does not exist: ${EXPECTED_MODULE_DIR}" >&2
    echo "[ERROR] This usually means you are running the wrong launcher copy/path." >&2
    echo "[ERROR] Run this launcher from the VARGPT-family-training checkout, for example:" >&2
    echo "[ERROR]   bash examples/train_self_evolving/run_self_evolving.sh joint 8" >&2
    exit 1
fi
if [ -z "${LLAMAFACTORY_MODULE_DIR}" ]; then
    echo "[ERROR] Could not resolve llamafactory module path." >&2
    exit 1
fi
if [[ "${LLAMAFACTORY_MODULE_DIR}" == *"/site-packages/llamafactory" ]]; then
    echo "[ERROR] Wrong llamafactory package resolved." >&2
    echo "[ERROR] Current:  ${LLAMAFACTORY_MODULE_DIR}" >&2
    echo "[ERROR] Expected: ${EXPECTED_MODULE_DIR}" >&2
    echo "[ERROR] Fix in this env: pip uninstall -y llamafactory && pip install -e ${REPO_ROOT} --no-deps" >&2
    exit 1
fi
if [ "${LLAMAFACTORY_MODULE_DIR}" != "${EXPECTED_MODULE_DIR}" ]; then
    echo "[WARN] llamafactory path differs from script repo root." >&2
    echo "[WARN] Current:  ${LLAMAFACTORY_MODULE_DIR}" >&2
    echo "[WARN] Expected: ${EXPECTED_MODULE_DIR}" >&2
    echo "[WARN] Continuing because module is not from site-packages." >&2
fi
LLAMAFACTORY_LAUNCHER_PY="${LLAMAFACTORY_MODULE_DIR}/launcher.py"
if [ ! -f "${LLAMAFACTORY_LAUNCHER_PY}" ]; then
    echo "[ERROR] Could not resolve launcher.py at: ${LLAMAFACTORY_LAUNCHER_PY}" >&2
    exit 1
fi

# Fail fast when torch cannot see an accelerator.
TORCH_ACCEL_COUNT="$(
    python - <<'PY' 2>/dev/null || true
import torch
try:
    print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
)"
TORCH_ACCEL_COUNT="$(echo "${TORCH_ACCEL_COUNT}" | tr -d '[:space:]')"
if ! [[ "${TORCH_ACCEL_COUNT}" =~ ^[0-9]+$ ]]; then
    TORCH_ACCEL_COUNT=0
fi
if [ "${TORCH_ACCEL_COUNT}" -lt 1 ]; then
    echo "[ERROR] PyTorch cannot see any GPU accelerator in this environment." >&2
    echo "[ERROR] rocm-smi can list devices, but torch/deepspeed is falling back to CPU." >&2
    echo "[ERROR] Check ROCm-enabled torch install and container device passthrough (/dev/kfd, /dev/dri)." >&2
    exit 1
fi
if [ "${NUM_GPUS}" -gt "${TORCH_ACCEL_COUNT}" ]; then
    echo "[WARN] Requested NUM_GPUS=${NUM_GPUS}, but torch sees ${TORCH_ACCEL_COUNT}. Capping to ${TORCH_ACCEL_COUNT}."
    NUM_GPUS="${TORCH_ACCEL_COUNT}"
fi

# Require explicit image-folder mode for this launcher.
if [ -z "${IMAGE_FOLDER}" ]; then
    echo "[ERROR] IMAGE_FOLDER is not set." >&2
    echo "[ERROR] This launcher expects image-folder mode for self-evolving training." >&2
    echo "[ERROR] Example: IMAGE_FOLDER=/path/to/unlabeled/images \\" >&2
    echo "[ERROR]          bash examples/train_self_evolving/run_self_evolving.sh joint 8" >&2
    exit 1
fi
if [ ! -d "${IMAGE_FOLDER}" ]; then
    echo "[ERROR] IMAGE_FOLDER does not exist: ${IMAGE_FOLDER}" >&2
    exit 1
fi
IMAGE_COUNT="$(
    find "${IMAGE_FOLDER}" -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' -o -iname '*.tiff' \) \
      | wc -l | tr -d '[:space:]'
)"
if ! [[ "${IMAGE_COUNT}" =~ ^[0-9]+$ ]]; then
    IMAGE_COUNT=0
fi
if [ "${IMAGE_COUNT}" -lt 1 ]; then
    echo "[ERROR] IMAGE_FOLDER has no supported images: ${IMAGE_FOLDER}" >&2
    echo "[ERROR] Supported extensions: jpg, jpeg, png, webp, bmp, tiff" >&2
    exit 1
fi

# Optional: HuggingFace mirror (uncomment if needed)
# export HF_ENDPOINT=https://hf-mirror.com

echo "=============================================="
echo "VARGPT Self-Evolving Training"
echo "=============================================="
echo "  Experiment : $EXPERIMENT"
echo "  Config     : $CONFIG"
echo "  GPUs       : $NUM_GPUS"
echo "  Master Port: $MASTER_PORT"
echo "  W&B Project: $WANDB_PROJECT"
echo "  Launcher   : ${LAUNCHER[*]}"
echo "  Dataset dir: ${DATASET_DIR:-data}"
echo "  Image folder: ${IMAGE_FOLDER}"
echo "  Image count : ${IMAGE_COUNT}"
echo "=============================================="

# ── Build temporary YAML with overrides ──────────────────────────────────────
DATASET_DIR="${DATASET_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${REPO_ROOT}/outputs/vargpt/${EXPERIMENT}"
fi
if [ -n "${RESUME_FROM:-}" ]; then
    OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"
else
    OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"
fi
TMP_CONFIG="$(mktemp "/tmp/vargpt_se_${EXPERIMENT}_XXXX.yaml")"
RUN_CONFIG="${TMP_CONFIG}"
cp "${CONFIG}" "${RUN_CONFIG}"
if [ "${KEEP_TMP_CONFIG:-0}" = "1" ]; then
    trap ':' EXIT
else
    trap '[[ -n "${TMP_CONFIG:-}" && -f "${TMP_CONFIG}" ]] && rm -f "${TMP_CONFIG}"' EXIT
fi

yaml_quote() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

yaml_delete_key() {
    local key="$1"
    local tmp_file="${RUN_CONFIG}.tmp"
    awk -v k="${key}" '
        $0 ~ "^[[:space:]]*" k "[[:space:]]*:" {next}
        {print}
    ' "${RUN_CONFIG}" > "${tmp_file}"
    mv "${tmp_file}" "${RUN_CONFIG}"
}

SUPPORTED_SE_KEYS="$(
    python - <<'PY' 2>/dev/null || true
try:
    from llamafactory.hparams.finetuning_args import FinetuningArguments
    fields = getattr(FinetuningArguments, "__dataclass_fields__", {})
    for name in sorted(fields.keys()):
        if name.startswith("se_"):
            print(name)
except Exception:
    pass
PY
)"

se_key_supported() {
    local key="$1"
    grep -qx "${key}" <<< "${SUPPORTED_SE_KEYS}"
}

append_se_override() {
    local key="$1"
    local value="$2"
    if se_key_supported "${key}"; then
        echo "${key}: ${value}"
    else
        echo "[WARN] Skipping unsupported self-evolving key in this codebase: ${key}" >&2
    fi
}

# Remove keys we always override to avoid duplicate-key ambiguity.
yaml_delete_key "dataset_dir"
yaml_delete_key "resume_from_checkpoint"
yaml_delete_key "overwrite_output_dir"
yaml_delete_key "output_dir"
yaml_delete_key "se_image_folder"
yaml_delete_key "se_proposer_num_candidates"
yaml_delete_key "se_use_ref_answer_scoring"
yaml_delete_key "flash_attn"
yaml_delete_key "se_solver_use_forced_choice_from_proposer"
yaml_delete_key "se_solver_skip_update_on_easy"
yaml_delete_key "se_save_every"
yaml_delete_key "se_fail_on_step_error"
yaml_delete_key "se_max_consecutive_step_errors"
yaml_delete_key "se_max_total_step_errors"
yaml_delete_key "se_generation_failfast_enabled"
yaml_delete_key "se_generation_failfast_consecutive_skips"
yaml_delete_key "se_generation_failfast_min_success_rate"
yaml_delete_key "se_generat_trainion_failfast_min_success_rate"

{
    echo ""
    echo "# --- auto overrides from run_self_evolving.sh ---"
    echo "dataset_dir: \"$(yaml_quote "${DATASET_DIR}")\""
    echo "flash_attn: ${FLASH_ATTN_MODE}"
    append_se_override "se_total_steps" "${SE_TOTAL_STEPS:-10000}"
    append_se_override "se_save_every" "${SE_SAVE_EVERY:-200}"
    echo "save_steps: ${SAVE_STEPS:-200}"

    if [ -n "${RESUME_FROM:-}" ]; then
        echo "resume_from_checkpoint: \"$(yaml_quote "${RESUME_FROM}")\""
    fi
    if [ -n "${OUTPUT_DIR}" ]; then
        echo "output_dir: \"$(yaml_quote "${OUTPUT_DIR}")\""
    fi
    echo "overwrite_output_dir: ${OVERWRITE_OUTPUT_DIR}"

    append_se_override "se_image_folder" "\"$(yaml_quote "${IMAGE_FOLDER}")\""

    append_se_override "se_proposer_num_candidates" "${SE_PROPOSER_NUM_CANDIDATES:-3}"
    append_se_override "se_use_ref_answer_scoring" "${SE_USE_REF_ANSWER_SCORING:-false}"
    append_se_override "se_num_solver_samples" "${SE_NUM_SOLVER_SAMPLES:-7}"
    append_se_override "se_proposer_spot_check_samples" "${SE_PROPOSER_SPOT_CHECK_SAMPLES:-3}"
    append_se_override "se_proposer_question_quality_min_score" "${SE_PROPOSER_QUESTION_QUALITY_MIN_SCORE:-0.60}"
    append_se_override "se_proposer_question_structural_min_score" "${SE_PROPOSER_QUESTION_STRUCTURAL_MIN_SCORE:-0.50}"
    append_se_override "se_proposer_question_model_judge_enabled" "${SE_PROPOSER_QUESTION_MODEL_JUDGE_ENABLED:-true}"
    append_se_override "se_proposer_question_model_judge_weight" "${SE_PROPOSER_QUESTION_MODEL_JUDGE_WEIGHT:-0.15}"
    append_se_override "se_solver_temp_min" "${SE_SOLVER_TEMP_MIN:-0.5}"
    append_se_override "se_solver_temp_max" "${SE_SOLVER_TEMP_MAX:-2.5}"
    append_se_override "se_solver_top_p_min" "${SE_SOLVER_TOP_P_MIN:-0.3}"
    append_se_override "se_solver_top_p_max" "${SE_SOLVER_TOP_P_MAX:-1.0}"
    append_se_override "se_solver_use_forced_choice_from_proposer" "${SE_SOLVER_USE_FORCED_CHOICE_FROM_PROPOSER:-true}"
    append_se_override "se_solver_skip_update_on_easy" "${SE_SOLVER_SKIP_UPDATE_ON_EASY:-true}"
    append_se_override "se_easy_update_majority_frac_threshold" "${SE_EASY_UPDATE_MAJORITY_FRAC_THRESHOLD:-0.85}"
    append_se_override "se_difficulty_sampler_enabled" "${SE_DIFFICULTY_SAMPLER_ENABLED:-true}"
    append_se_override "se_difficulty_target_easy" "${SE_DIFFICULTY_TARGET_EASY:-0.0}"
    append_se_override "se_difficulty_target_medium" "${SE_DIFFICULTY_TARGET_MEDIUM:-0.7}"
    append_se_override "se_difficulty_target_hard" "${SE_DIFFICULTY_TARGET_HARD:-0.3}"
    append_se_override "se_proposer_warm_start_enabled" "${SE_PROPOSER_WARM_START_ENABLED:-true}"
    append_se_override "se_proposer_warm_start_max_steps" "${SE_PROPOSER_WARM_START_MAX_STEPS:-30}"
    append_se_override "se_hardness_debt_enabled" "${SE_HARDNESS_DEBT_ENABLED:-true}"
    append_se_override "se_hardness_debt_inc_easy" "${SE_HARDNESS_DEBT_INC_EASY:-1.5}"
    append_se_override "se_hardness_debt_dec_non_easy" "${SE_HARDNESS_DEBT_DEC_NON_EASY:-1.0}"
    append_se_override "se_hardness_debt_hard_recovery_threshold" "${SE_HARDNESS_DEBT_HARD_RECOVERY_THRESHOLD:-3.0}"
    append_se_override "se_all_easy_explore_trigger" "${SE_ALL_EASY_EXPLORE_TRIGGER:-2}"
    append_se_override "se_all_easy_explore_steps" "${SE_ALL_EASY_EXPLORE_STEPS:-16}"
    append_se_override "se_all_easy_explore_num_candidates" "${SE_ALL_EASY_EXPLORE_NUM_CANDIDATES:-6}"
    append_se_override "se_fail_on_step_error" "${SE_FAIL_ON_STEP_ERROR:-true}"
    append_se_override "se_max_consecutive_step_errors" "${SE_MAX_CONSECUTIVE_STEP_ERRORS:-0}"
    append_se_override "se_max_total_step_errors" "${SE_MAX_TOTAL_STEP_ERRORS:-0}"
    append_se_override "se_generation_failfast_enabled" "${SE_GENERATION_FAILFAST_ENABLED:-true}"
    append_se_override "se_generation_failfast_consecutive_skips" "${SE_GENERATION_FAILFAST_CONSECUTIVE_SKIPS:-5}"
    append_se_override "se_generation_failfast_min_success_rate" "${SE_GENERATION_FAILFAST_MIN_SUCCESS_RATE:-0.10}"
    append_se_override "se_proposer_early_failfast_enabled" "${SE_PROPOSER_EARLY_FAILFAST_ENABLED:-true}"
    append_se_override "se_proposer_early_failfast_stop" "${SE_PROPOSER_EARLY_FAILFAST_STOP:-false}"
    append_se_override "se_proposer_early_failfast_recover" "${SE_PROPOSER_EARLY_FAILFAST_RECOVER:-true}"
} >> "${RUN_CONFIG}"

# Sanity-check effective YAML before launch.
effective_value() {
    local key="$1"
    awk -F':' -v k="$key" '
        $0 ~ "^[[:space:]]*" k "[[:space:]]*:" {
            v=$0
            sub(/^[^:]*:[[:space:]]*/, "", v)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
            gsub(/^"|"$/, "", v)
        }
        END { print v }
    ' "${RUN_CONFIG}"
}

EFFECTIVE_STAGE="$(effective_value stage)"
EFFECTIVE_DO_TRAIN="$(echo "$(effective_value do_train)" | tr '[:upper:]' '[:lower:]')"
EFFECTIVE_TOTAL_STEPS="$(effective_value se_total_steps)"
EFFECTIVE_IMAGE_FOLDER="$(effective_value se_image_folder)"
EFFECTIVE_OVERWRITE_OUTPUT_DIR="$(echo "$(effective_value overwrite_output_dir)" | tr '[:upper:]' '[:lower:]')"
EFFECTIVE_OUTPUT_DIR="$(effective_value output_dir)"

if [ "${EFFECTIVE_STAGE}" != "self_evolving" ]; then
    echo "[ERROR] Effective config stage is '${EFFECTIVE_STAGE}', expected 'self_evolving'." >&2
    exit 1
fi
if [ "${EFFECTIVE_DO_TRAIN}" != "true" ]; then
    echo "[ERROR] Effective config do_train is '${EFFECTIVE_DO_TRAIN}', expected 'true'." >&2
    exit 1
fi
if ! [[ "${EFFECTIVE_TOTAL_STEPS}" =~ ^[0-9]+$ ]] || [ "${EFFECTIVE_TOTAL_STEPS}" -lt 1 ]; then
    echo "[ERROR] Effective se_total_steps is invalid: '${EFFECTIVE_TOTAL_STEPS}'." >&2
    exit 1
fi
if [ -z "${EFFECTIVE_IMAGE_FOLDER}" ]; then
    echo "[ERROR] Effective se_image_folder is empty in run config: ${RUN_CONFIG}" >&2
    exit 1
fi
if [ -z "${EFFECTIVE_OUTPUT_DIR}" ]; then
    echo "[ERROR] Effective output_dir is empty in run config: ${RUN_CONFIG}" >&2
    exit 1
fi

echo "  Effective stage       : ${EFFECTIVE_STAGE}"
echo "  Effective do_train    : ${EFFECTIVE_DO_TRAIN}"
echo "  Effective total_steps : ${EFFECTIVE_TOTAL_STEPS}"
echo "  Effective se_image_folder: ${EFFECTIVE_IMAGE_FOLDER}"
echo "  Effective output_dir  : ${EFFECTIVE_OUTPUT_DIR}"
echo "  Effective overwrite_output_dir: ${EFFECTIVE_OVERWRITE_OUTPUT_DIR}"
echo "  Effective flash_attn  : ${FLASH_ATTN_MODE}"
echo "  Run config : ${RUN_CONFIG}"

if [ -z "${RESUME_FROM:-}" ] && [ "${EFFECTIVE_OVERWRITE_OUTPUT_DIR}" != "true" ]; then
    if [ -f "${EFFECTIVE_OUTPUT_DIR}/summary.json" ] || [ -f "${EFFECTIVE_OUTPUT_DIR}/status.json" ] || [ -d "${EFFECTIVE_OUTPUT_DIR}/checkpoints" ]; then
        echo "[ERROR] OUTPUT_DIR already contains an existing self-evolving run:" >&2
        echo "[ERROR]   ${EFFECTIVE_OUTPUT_DIR}" >&2
        echo "[ERROR] Set RESUME_FROM to continue, choose a new OUTPUT_DIR, or set OVERWRITE_OUTPUT_DIR=true." >&2
        exit 1
    fi
fi

mkdir -p "${EFFECTIVE_OUTPUT_DIR}"
TORCHRUN_LOG_DIR="${EFFECTIVE_OUTPUT_DIR}/torchrun_logs"
mkdir -p "${TORCHRUN_LOG_DIR}"

# ── Launch ───────────────────────────────────────────────────────────────────
if [ "$NUM_GPUS" -gt 1 ]; then
    if ! command -v torchrun >/dev/null 2>&1; then
        echo "[ERROR] torchrun not found in PATH." >&2
        exit 1
    fi
    echo "  Launching with torchrun (DDP, $NUM_GPUS GPUs)..."
    torchrun \
      --nnodes 1 \
      --node_rank 0 \
      --nproc_per_node "${NUM_GPUS}" \
      --master_addr 127.0.0.1 \
      --master_port "${MASTER_PORT}" \
      --log-dir "${TORCHRUN_LOG_DIR}" \
      --tee 3 \
      "${LLAMAFACTORY_LAUNCHER_PY}" \
      "${RUN_CONFIG}"
else
    echo "  Launching single-GPU..."
    python "${LLAMAFACTORY_LAUNCHER_PY}" "${RUN_CONFIG}"
fi

# Post-launch sanity: successful training should materialize output_dir.
if [ ! -d "${EFFECTIVE_OUTPUT_DIR}" ]; then
    echo "[ERROR] Training launcher returned success, but output_dir was not created:" >&2
    echo "[ERROR]   ${EFFECTIVE_OUTPUT_DIR}" >&2
    echo "[ERROR] Likely distributed launch did not spawn workers (check NPROC_PER_NODE/torchrun environment)." >&2
    exit 1
fi

echo ""
echo "=============================================="
echo "Training complete: $EXPERIMENT"
echo "=============================================="
