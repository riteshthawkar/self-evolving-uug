#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# VARGPT Self-Evolving: Joint (3U+2G) Combined Experiment
# ══════════════════════════════════════════════════════════════════════════════

# ── Project root ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
find_repo_root() {
    local d="$1"
    while [[ -n "${d}" && "${d}" != "/" ]]; do
        if [[ -d "${d}/src/llamafactory" && -f "${d}/examples/train_self_evolving/vargpt_se_joint.yaml" ]]; then
            echo "${d}"
            return 0
        fi
        d="$(dirname "${d}")"
    done
    return 1
}
REPO_ROOT="$(find_repo_root "${SCRIPT_DIR}" || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$REPO_ROOT"

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG="examples/train_self_evolving/vargpt_se_joint.yaml"
CONFIG="${REPO_ROOT}/${CONFIG}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-39600}"

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

# ══════════════════════════════════════════════════════════════════════════════
# DATA: Just set IMAGE_FOLDER to your folder of images.
#       Subfolders are scanned recursively. No JSON needed.
#       Supports: .jpg .jpeg .png .webp .bmp .tiff
#
#       Example folder structure (any nesting works):
#         /path/to/my/images/
#           ├── cats/img1.jpg
#           ├── dogs/img2.png
#           └── landscapes/sunset.webp
# ══════════════════════════════════════════════════════════════════════════════
IMAGE_FOLDER="${IMAGE_FOLDER:-/path/to/your/images}"

# ── Environment setup ────────────────────────────────────────────────────────
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
export MASTER_PORT
export TOKENIZERS_PARALLELISM="false"
export WANDB_MODE="${WANDB_MODE:-disabled}"
if [[ -z "${HIP_VISIBLE_DEVICES:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
fi

# ── Auto-create missing __init__.py in visionllm + symlink vargpt ───────────
find "$REPO_ROOT/visionllm" -type d ! -name '__pycache__' \
    -exec sh -c 'test -f "$1/__init__.py" || touch "$1/__init__.py"' _ {} \;
if [[ ! -e "$REPO_ROOT/visionllm/vargpt" ]]; then
    ln -sfn vargpt_llava "$REPO_ROOT/visionllm/vargpt"
fi

# VARGPT image size constraints
export SE_MAX_IMAGE_SIDE=1024
export SE_MIN_IMAGE_SIDE=56
export SE_IMAGE_SIZE_MULTIPLE=28

# CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

# NCCL / DDP settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# ── Self-evolving launch overrides (can be changed via env) ─────────────────
SE_NUM_SOLVER_SAMPLES="${SE_NUM_SOLVER_SAMPLES:-7}"
SE_PROPOSER_SPOT_CHECK_SAMPLES="${SE_PROPOSER_SPOT_CHECK_SAMPLES:-3}"
SE_SOLVER_TEMP_MIN="${SE_SOLVER_TEMP_MIN:-0.5}"
SE_SOLVER_TEMP_MAX="${SE_SOLVER_TEMP_MAX:-2.5}"
SE_SOLVER_TOP_P_MIN="${SE_SOLVER_TOP_P_MIN:-0.3}"
SE_SOLVER_TOP_P_MAX="${SE_SOLVER_TOP_P_MAX:-1.0}"
SE_SOLVER_SKIP_UPDATE_ON_EASY="${SE_SOLVER_SKIP_UPDATE_ON_EASY:-false}"
SE_SOLVER_USE_FORCED_CHOICE_FROM_PROPOSER="${SE_SOLVER_USE_FORCED_CHOICE_FROM_PROPOSER:-true}"
SE_EASY_UPDATE_MAJORITY_FRAC_THRESHOLD="${SE_EASY_UPDATE_MAJORITY_FRAC_THRESHOLD:-1.0}"
SE_DIFFICULTY_SAMPLER_ENABLED="${SE_DIFFICULTY_SAMPLER_ENABLED:-true}"
SE_DIFFICULTY_TARGET_EASY="${SE_DIFFICULTY_TARGET_EASY:-0.0}"
SE_DIFFICULTY_TARGET_MEDIUM="${SE_DIFFICULTY_TARGET_MEDIUM:-0.7}"
SE_DIFFICULTY_TARGET_HARD="${SE_DIFFICULTY_TARGET_HARD:-0.3}"
SE_PROPOSER_WARM_START_ENABLED="${SE_PROPOSER_WARM_START_ENABLED:-true}"
SE_PROPOSER_WARM_START_MAX_STEPS="${SE_PROPOSER_WARM_START_MAX_STEPS:-30}"
SE_HARDNESS_DEBT_ENABLED="${SE_HARDNESS_DEBT_ENABLED:-true}"
SE_HARDNESS_DEBT_INC_EASY="${SE_HARDNESS_DEBT_INC_EASY:-1.5}"
SE_HARDNESS_DEBT_DEC_NON_EASY="${SE_HARDNESS_DEBT_DEC_NON_EASY:-1.0}"
SE_HARDNESS_DEBT_HARD_RECOVERY_THRESHOLD="${SE_HARDNESS_DEBT_HARD_RECOVERY_THRESHOLD:-3.0}"
SE_ALL_EASY_EXPLORE_TRIGGER="${SE_ALL_EASY_EXPLORE_TRIGGER:-2}"
SE_ALL_EASY_EXPLORE_STEPS="${SE_ALL_EASY_EXPLORE_STEPS:-16}"
SE_ALL_EASY_EXPLORE_NUM_CANDIDATES="${SE_ALL_EASY_EXPLORE_NUM_CANDIDATES:-6}"
SE_PROPOSER_EARLY_FAILFAST_ENABLED="${SE_PROPOSER_EARLY_FAILFAST_ENABLED:-true}"
SE_PROPOSER_EARLY_FAILFAST_STOP="${SE_PROPOSER_EARLY_FAILFAST_STOP:-false}"
SE_PROPOSER_EARLY_FAILFAST_RECOVER="${SE_PROPOSER_EARLY_FAILFAST_RECOVER:-true}"
DATASET_DIR="${DATASET_DIR:-data_temp}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config not found: $CONFIG" >&2
    exit 1
fi

if ! resolve_launcher; then
    echo "[ERROR] Could not find LlamaFactory launcher." >&2
    echo "[ERROR] Run: pip install -e . --no-deps" >&2
    echo "[ERROR] NOTE: this repo pins torch==2.1.0 in requirements; avoid reinstalling torch on ROCm." >&2
    echo "[ERROR] Or ensure 'python -m llamafactory.cli' imports in current env." >&2
    exit 1
fi

LLAMAFACTORY_MODULE_DIR="$(
python - <<'PY' 2>/dev/null || true
import os
import llamafactory
print(os.path.abspath(os.path.dirname(llamafactory.__file__)))
PY
)"
LLAMAFACTORY_MODULE_DIR="$(echo "${LLAMAFACTORY_MODULE_DIR}" | tr -d '[:space:]')"
EXPECTED_MODULE_DIR="${REPO_ROOT}/src/llamafactory"
if [[ ! -d "${EXPECTED_MODULE_DIR}" ]]; then
    echo "[ERROR] Expected llamafactory source dir does not exist: ${EXPECTED_MODULE_DIR}" >&2
    echo "[ERROR] This usually means you are running the wrong launcher copy/path." >&2
    echo "[ERROR] Use: /workspace/self-evolving-uug/self-evolving-uug/vargpt_1_1/VARGPT-family-training/run_scripts/run_vargpt_se_joint.sh" >&2
    exit 1
fi
if [[ -z "${LLAMAFACTORY_MODULE_DIR}" ]]; then
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
if [[ "${LLAMAFACTORY_MODULE_DIR}" != "${EXPECTED_MODULE_DIR}" ]]; then
    echo "[WARN] llamafactory path differs from script repo root." >&2
    echo "[WARN] Current:  ${LLAMAFACTORY_MODULE_DIR}" >&2
    echo "[WARN] Expected: ${EXPECTED_MODULE_DIR}" >&2
    echo "[WARN] Continuing because module is not from site-packages." >&2
fi
LLAMAFACTORY_LAUNCHER_PY="${LLAMAFACTORY_MODULE_DIR}/launcher.py"
if [[ ! -f "${LLAMAFACTORY_LAUNCHER_PY}" ]]; then
    echo "[ERROR] Could not resolve launcher.py at: ${LLAMAFACTORY_LAUNCHER_PY}" >&2
    exit 1
fi

if [[ ! -d "$IMAGE_FOLDER" ]]; then
    echo "[ERROR] IMAGE_FOLDER not found: $IMAGE_FOLDER" >&2
    echo "[ERROR] Set IMAGE_FOLDER to your image directory in this script." >&2
    exit 1
fi

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
if [[ "${TORCH_ACCEL_COUNT}" -lt 1 ]]; then
    echo "[ERROR] PyTorch cannot see any GPU accelerator in this environment." >&2
    echo "[ERROR] rocm-smi may list devices, but torch/deepspeed is currently on CPU." >&2
    exit 1
fi
if [[ "${NPROC_PER_NODE}" -gt "${TORCH_ACCEL_COUNT}" ]]; then
    echo "[WARN] Requested NPROC_PER_NODE=${NPROC_PER_NODE}, but torch sees ${TORCH_ACCEL_COUNT}. Capping."
    NPROC_PER_NODE="${TORCH_ACCEL_COUNT}"
fi

# ── Print experiment info ────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  VARGPT Self-Evolving: Joint (3U+2G) Combined Experiment"
echo "═══════════════════════════════════════════════════════════"
echo "  Config       : $CONFIG"
echo "  GPUs         : $NPROC_PER_NODE"
echo "  Image folder : $IMAGE_FOLDER"
echo "  Solver K     : $SE_NUM_SOLVER_SAMPLES"
echo "  Spot-check K : $SE_PROPOSER_SPOT_CHECK_SAMPLES"
echo "  Temp range   : [$SE_SOLVER_TEMP_MIN, $SE_SOLVER_TEMP_MAX]"
echo "  Dataset dir  : $DATASET_DIR"
echo "  Launcher     : ${LAUNCHER[*]}"
echo "  Torch accel  : $TORCH_ACCEL_COUNT"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Build temporary YAML with overrides ──────────────────────────────────────
TMP_CONFIG="$(mktemp "/tmp/vargpt_se_joint_XXXX.yaml")"
RUN_CONFIG="${TMP_CONFIG}"
cp "${CONFIG}" "${RUN_CONFIG}"
if [[ "${KEEP_TMP_CONFIG:-0}" == "1" ]]; then
    trap ':' EXIT
else
    trap '[[ -n "${TMP_CONFIG:-}" && -f "${TMP_CONFIG}" ]] && rm -f "${TMP_CONFIG}"' EXIT
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
    OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"
else
    OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-true}"
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

# Remove keys we always override to avoid duplicate-key ambiguity.
yaml_delete_key "dataset_dir"
yaml_delete_key "resume_from_checkpoint"
yaml_delete_key "overwrite_output_dir"
yaml_delete_key "output_dir"
yaml_delete_key "se_image_folder"
yaml_delete_key "se_solver_use_forced_choice_from_proposer"
yaml_delete_key "se_solver_skip_update_on_easy"

{
    echo ""
    echo "# --- auto overrides from run_vargpt_se_joint.sh ---"
    if [[ -n "${RESUME_FROM:-}" ]]; then
        echo "resume_from_checkpoint: \"$(yaml_quote "${RESUME_FROM}")\""
    fi
    if [[ -n "${OUTPUT_DIR}" ]]; then
        echo "output_dir: \"$(yaml_quote "${OUTPUT_DIR}")\""
    fi
    echo "overwrite_output_dir: ${OVERWRITE_OUTPUT_DIR}"
    echo "dataset_dir: \"$(yaml_quote "${DATASET_DIR}")\""
    echo "se_image_folder: \"$(yaml_quote "${IMAGE_FOLDER}")\""
    echo "se_num_solver_samples: ${SE_NUM_SOLVER_SAMPLES}"
    echo "se_proposer_spot_check_samples: ${SE_PROPOSER_SPOT_CHECK_SAMPLES}"
    echo "se_solver_temp_min: ${SE_SOLVER_TEMP_MIN}"
    echo "se_solver_temp_max: ${SE_SOLVER_TEMP_MAX}"
    echo "se_solver_top_p_min: ${SE_SOLVER_TOP_P_MIN}"
    echo "se_solver_top_p_max: ${SE_SOLVER_TOP_P_MAX}"
    echo "se_solver_use_forced_choice_from_proposer: ${SE_SOLVER_USE_FORCED_CHOICE_FROM_PROPOSER}"
    echo "se_solver_skip_update_on_easy: ${SE_SOLVER_SKIP_UPDATE_ON_EASY}"
    echo "se_easy_update_majority_frac_threshold: ${SE_EASY_UPDATE_MAJORITY_FRAC_THRESHOLD}"
    echo "se_difficulty_sampler_enabled: ${SE_DIFFICULTY_SAMPLER_ENABLED}"
    echo "se_difficulty_target_easy: ${SE_DIFFICULTY_TARGET_EASY}"
    echo "se_difficulty_target_medium: ${SE_DIFFICULTY_TARGET_MEDIUM}"
    echo "se_difficulty_target_hard: ${SE_DIFFICULTY_TARGET_HARD}"
    echo "se_proposer_warm_start_enabled: ${SE_PROPOSER_WARM_START_ENABLED}"
    echo "se_proposer_warm_start_max_steps: ${SE_PROPOSER_WARM_START_MAX_STEPS}"
    echo "se_hardness_debt_enabled: ${SE_HARDNESS_DEBT_ENABLED}"
    echo "se_hardness_debt_inc_easy: ${SE_HARDNESS_DEBT_INC_EASY}"
    echo "se_hardness_debt_dec_non_easy: ${SE_HARDNESS_DEBT_DEC_NON_EASY}"
    echo "se_hardness_debt_hard_recovery_threshold: ${SE_HARDNESS_DEBT_HARD_RECOVERY_THRESHOLD}"
    echo "se_all_easy_explore_trigger: ${SE_ALL_EASY_EXPLORE_TRIGGER}"
    echo "se_all_easy_explore_steps: ${SE_ALL_EASY_EXPLORE_STEPS}"
    echo "se_all_easy_explore_num_candidates: ${SE_ALL_EASY_EXPLORE_NUM_CANDIDATES}"
    echo "se_proposer_early_failfast_enabled: ${SE_PROPOSER_EARLY_FAILFAST_ENABLED}"
    echo "se_proposer_early_failfast_stop: ${SE_PROPOSER_EARLY_FAILFAST_STOP}"
    echo "se_proposer_early_failfast_recover: ${SE_PROPOSER_EARLY_FAILFAST_RECOVER}"
} >> "${RUN_CONFIG}"

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
if [[ "${EFFECTIVE_STAGE}" != "self_evolving" ]]; then
    echo "[ERROR] Effective config stage is '${EFFECTIVE_STAGE}', expected 'self_evolving'." >&2
    exit 1
fi
if [[ "${EFFECTIVE_DO_TRAIN}" != "true" ]]; then
    echo "[ERROR] Effective config do_train is '${EFFECTIVE_DO_TRAIN}', expected 'true'." >&2
    exit 1
fi
if ! [[ "${EFFECTIVE_TOTAL_STEPS}" =~ ^[0-9]+$ ]] || [[ "${EFFECTIVE_TOTAL_STEPS}" -lt 1 ]]; then
    echo "[ERROR] Effective se_total_steps is invalid: '${EFFECTIVE_TOTAL_STEPS}'." >&2
    exit 1
fi
if [[ -z "${EFFECTIVE_IMAGE_FOLDER}" ]]; then
    echo "[ERROR] Effective se_image_folder is empty in run config: ${RUN_CONFIG}" >&2
    exit 1
fi
if [[ -z "${EFFECTIVE_OUTPUT_DIR}" ]]; then
    echo "[ERROR] Effective output_dir is empty in run config: ${RUN_CONFIG}" >&2
    exit 1
fi
echo "  Effective stage       : ${EFFECTIVE_STAGE}"
echo "  Effective do_train    : ${EFFECTIVE_DO_TRAIN}"
echo "  Effective total_steps : ${EFFECTIVE_TOTAL_STEPS}"
echo "  Effective se_image_folder: ${EFFECTIVE_IMAGE_FOLDER}"
echo "  Effective output_dir  : ${EFFECTIVE_OUTPUT_DIR}"
echo "  Effective overwrite_output_dir: ${EFFECTIVE_OVERWRITE_OUTPUT_DIR}"
echo "  Run config   : ${RUN_CONFIG}"

FORCE_TORCHRUN=1 \
torchrun \
  --nnodes 1 \
  --node_rank 0 \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_addr 127.0.0.1 \
  --master_port "${MASTER_PORT}" \
  "${LLAMAFACTORY_LAUNCHER_PY}" \
  "${RUN_CONFIG}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Training complete."
echo "═══════════════════════════════════════════════════════════"
