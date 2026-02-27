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

# ── Parse arguments ──────────────────────────────────────────────────────────
EXPERIMENT=${1:-joint}
NUM_GPUS=${2:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}

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
echo "=============================================="

# ── Build command ────────────────────────────────────────────────────────────
CMD_ARGS=""

# Resume from checkpoint
if [ -n "${RESUME_FROM:-}" ]; then
    echo "  Resuming from: $RESUME_FROM"
    CMD_ARGS="$CMD_ARGS --resume_from_checkpoint $RESUME_FROM"
fi

# Optional direct image-folder mode
if [ -n "${IMAGE_FOLDER:-}" ]; then
    echo "  Image folder: $IMAGE_FOLDER"
    CMD_ARGS="$CMD_ARGS --se_image_folder $IMAGE_FOLDER"
fi

# Self-evolving controller overrides (env-overridable)
CMD_ARGS="$CMD_ARGS --se_num_solver_samples ${SE_NUM_SOLVER_SAMPLES:-7}"
CMD_ARGS="$CMD_ARGS --se_proposer_spot_check_samples ${SE_PROPOSER_SPOT_CHECK_SAMPLES:-3}"
CMD_ARGS="$CMD_ARGS --se_solver_temp_min ${SE_SOLVER_TEMP_MIN:-0.5}"
CMD_ARGS="$CMD_ARGS --se_solver_temp_max ${SE_SOLVER_TEMP_MAX:-2.5}"
CMD_ARGS="$CMD_ARGS --se_solver_top_p_min ${SE_SOLVER_TOP_P_MIN:-0.3}"
CMD_ARGS="$CMD_ARGS --se_solver_top_p_max ${SE_SOLVER_TOP_P_MAX:-1.0}"
CMD_ARGS="$CMD_ARGS --se_solver_skip_update_on_easy ${SE_SOLVER_SKIP_UPDATE_ON_EASY:-true}"
CMD_ARGS="$CMD_ARGS --se_easy_update_majority_frac_threshold ${SE_EASY_UPDATE_MAJORITY_FRAC_THRESHOLD:-1.0}"
CMD_ARGS="$CMD_ARGS --se_difficulty_sampler_enabled ${SE_DIFFICULTY_SAMPLER_ENABLED:-true}"
CMD_ARGS="$CMD_ARGS --se_difficulty_target_easy ${SE_DIFFICULTY_TARGET_EASY:-0.0}"
CMD_ARGS="$CMD_ARGS --se_difficulty_target_medium ${SE_DIFFICULTY_TARGET_MEDIUM:-0.7}"
CMD_ARGS="$CMD_ARGS --se_difficulty_target_hard ${SE_DIFFICULTY_TARGET_HARD:-0.3}"
CMD_ARGS="$CMD_ARGS --se_proposer_warm_start_enabled ${SE_PROPOSER_WARM_START_ENABLED:-true}"
CMD_ARGS="$CMD_ARGS --se_proposer_warm_start_max_steps ${SE_PROPOSER_WARM_START_MAX_STEPS:-30}"
CMD_ARGS="$CMD_ARGS --se_hardness_debt_enabled ${SE_HARDNESS_DEBT_ENABLED:-true}"
CMD_ARGS="$CMD_ARGS --se_hardness_debt_inc_easy ${SE_HARDNESS_DEBT_INC_EASY:-1.5}"
CMD_ARGS="$CMD_ARGS --se_hardness_debt_dec_non_easy ${SE_HARDNESS_DEBT_DEC_NON_EASY:-1.0}"
CMD_ARGS="$CMD_ARGS --se_hardness_debt_hard_recovery_threshold ${SE_HARDNESS_DEBT_HARD_RECOVERY_THRESHOLD:-3.0}"
CMD_ARGS="$CMD_ARGS --se_all_easy_explore_trigger ${SE_ALL_EASY_EXPLORE_TRIGGER:-2}"
CMD_ARGS="$CMD_ARGS --se_all_easy_explore_steps ${SE_ALL_EASY_EXPLORE_STEPS:-16}"
CMD_ARGS="$CMD_ARGS --se_all_easy_explore_num_candidates ${SE_ALL_EASY_EXPLORE_NUM_CANDIDATES:-6}"
CMD_ARGS="$CMD_ARGS --se_proposer_early_failfast_enabled ${SE_PROPOSER_EARLY_FAILFAST_ENABLED:-true}"
CMD_ARGS="$CMD_ARGS --se_proposer_early_failfast_stop ${SE_PROPOSER_EARLY_FAILFAST_STOP:-false}"
CMD_ARGS="$CMD_ARGS --se_proposer_early_failfast_recover ${SE_PROPOSER_EARLY_FAILFAST_RECOVER:-true}"

# Optional free-form extra args
if [ -n "${SE_EXTRA_ARGS:-}" ]; then
    CMD_ARGS="$CMD_ARGS ${SE_EXTRA_ARGS}"
fi

# ── Launch ───────────────────────────────────────────────────────────────────
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "  Launching with torchrun (DDP, $NUM_GPUS GPUs)..."
    FORCE_TORCHRUN=1 \
    NNODES=1 \
    NODE_RANK=0 \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=$MASTER_PORT \
        llamafactory-cli train "$CONFIG" $CMD_ARGS
else
    echo "  Launching single-GPU..."
    llamafactory-cli train "$CONFIG" $CMD_ARGS
fi

echo ""
echo "=============================================="
echo "Training complete: $EXPERIMENT"
echo "=============================================="
