#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# VARGPT Self-Evolving: Joint (3U+2G) Combined Experiment
# ══════════════════════════════════════════════════════════════════════════════

# ── Project root ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG="examples/train_self_evolving/vargpt_se_joint.yaml"
NPROC_PER_NODE=8
MASTER_PORT=39600

resolve_launcher() {
    if command -v llamafactory-cli >/dev/null 2>&1; then
        echo "llamafactory-cli"
        return 0
    fi
    if python -c "import llamafactory.cli" >/dev/null 2>&1; then
        echo "python -m llamafactory.cli"
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
SE_SOLVER_SKIP_UPDATE_ON_EASY="${SE_SOLVER_SKIP_UPDATE_ON_EASY:-true}"
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

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config not found: $CONFIG" >&2
    exit 1
fi

if ! LAUNCHER_CMD="$(resolve_launcher)"; then
    echo "[ERROR] Could not find LlamaFactory launcher." >&2
    echo "[ERROR] Run: pip install -e ." >&2
    echo "[ERROR] Or ensure 'python -m llamafactory.cli' imports in current env." >&2
    exit 1
fi

if [[ ! -d "$IMAGE_FOLDER" ]]; then
    echo "[ERROR] IMAGE_FOLDER not found: $IMAGE_FOLDER" >&2
    echo "[ERROR] Set IMAGE_FOLDER to your image directory in this script." >&2
    exit 1
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
echo "  Launcher     : $LAUNCHER_CMD"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Launch training ──────────────────────────────────────────────────────────
FORCE_TORCHRUN=1 \
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE="$NPROC_PER_NODE" \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT="$MASTER_PORT" \
    bash -lc "$LAUNCHER_CMD train \"$CONFIG\" \
        --se_image_folder "$IMAGE_FOLDER" \
        --dataset_dir "$DATASET_DIR" \
        --se_num_solver_samples "$SE_NUM_SOLVER_SAMPLES" \
        --se_proposer_spot_check_samples "$SE_PROPOSER_SPOT_CHECK_SAMPLES" \
        --se_solver_temp_min "$SE_SOLVER_TEMP_MIN" \
        --se_solver_temp_max "$SE_SOLVER_TEMP_MAX" \
        --se_solver_top_p_min "$SE_SOLVER_TOP_P_MIN" \
        --se_solver_top_p_max "$SE_SOLVER_TOP_P_MAX" \
        --se_solver_skip_update_on_easy "$SE_SOLVER_SKIP_UPDATE_ON_EASY" \
        --se_easy_update_majority_frac_threshold "$SE_EASY_UPDATE_MAJORITY_FRAC_THRESHOLD" \
        --se_difficulty_sampler_enabled "$SE_DIFFICULTY_SAMPLER_ENABLED" \
        --se_difficulty_target_easy "$SE_DIFFICULTY_TARGET_EASY" \
        --se_difficulty_target_medium "$SE_DIFFICULTY_TARGET_MEDIUM" \
        --se_difficulty_target_hard "$SE_DIFFICULTY_TARGET_HARD" \
        --se_proposer_warm_start_enabled "$SE_PROPOSER_WARM_START_ENABLED" \
        --se_proposer_warm_start_max_steps "$SE_PROPOSER_WARM_START_MAX_STEPS" \
        --se_hardness_debt_enabled "$SE_HARDNESS_DEBT_ENABLED" \
        --se_hardness_debt_inc_easy "$SE_HARDNESS_DEBT_INC_EASY" \
        --se_hardness_debt_dec_non_easy "$SE_HARDNESS_DEBT_DEC_NON_EASY" \
        --se_hardness_debt_hard_recovery_threshold "$SE_HARDNESS_DEBT_HARD_RECOVERY_THRESHOLD" \
        --se_all_easy_explore_trigger "$SE_ALL_EASY_EXPLORE_TRIGGER" \
        --se_all_easy_explore_steps "$SE_ALL_EASY_EXPLORE_STEPS" \
        --se_all_easy_explore_num_candidates "$SE_ALL_EASY_EXPLORE_NUM_CANDIDATES" \
        --se_proposer_early_failfast_enabled "$SE_PROPOSER_EARLY_FAILFAST_ENABLED" \
        --se_proposer_early_failfast_stop "$SE_PROPOSER_EARLY_FAILFAST_STOP" \
        --se_proposer_early_failfast_recover "$SE_PROPOSER_EARLY_FAILFAST_RECOVER" \
        ${SE_EXTRA_ARGS:-}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Training complete."
echo "═══════════════════════════════════════════════════════════"
