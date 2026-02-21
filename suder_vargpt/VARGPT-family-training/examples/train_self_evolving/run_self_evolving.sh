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
