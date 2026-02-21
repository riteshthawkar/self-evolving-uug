#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# VARGPT Self-Evolving: Joint (3U+2G) Combined Experiment
# ══════════════════════════════════════════════════════════════════════════════
#
# This is the PRIMARY experiment for the VARGPT self-evolving framework.
# All three roles are trained jointly:
#   • Proposer LoRA  — learns to propose useful questions / specs
#   • Solver LoRA    — improves visual understanding via GRPO
#   • Generator LoRA — improves image generation via discrete-token GRPO
#     (+ vargpt_gen + image_gen_projector updated during G-steps)
#
# Cycle: [U U U G G] repeated → 3 understanding + 2 generation steps
#
# What this experiment proves:
#   ✓ Self-evolving framework works on autoregressive discrete-token UUGs
#   ✓ GRPO on discrete VAR tokens (vocab=64) improves generation quality
#   ✓ Joint U+G training provides mutual benefit (same as BLIP3o E1)
#   ✓ Framework is model-agnostic (works on both diffusion and AR models)
#
# Usage:
#   bash run_scripts/run_vargpt_se_joint.sh
#   NPROC_PER_NODE=4 bash run_scripts/run_vargpt_se_joint.sh
#   RESUME_FROM=saves/vargpt_v1_1_se_joint_3U2G/se_checkpoint_500 bash run_scripts/run_vargpt_se_joint.sh
#
# Environment variables:
#   NPROC_PER_NODE  - Number of GPUs (default: auto-detect)
#   RESUME_FROM     - Checkpoint path to resume from (optional)
#   MASTER_PORT     - DDP master port (default: 39600)
#   WANDB_MODE      - W&B mode: disabled/online/offline (default: disabled)
#   SE_TOTAL_STEPS  - Override total training steps (default: from yaml)
# ══════════════════════════════════════════════════════════════════════════════

# ── Project root (script assumes it's run from VARGPT-family-training/) ──────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG="examples/train_self_evolving/vargpt_se_joint.yaml"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
MASTER_PORT="${MASTER_PORT:-39600}"
RESUME_FROM="${RESUME_FROM:-}"
WANDB_MODE="${WANDB_MODE:-disabled}"

# ── Environment setup ────────────────────────────────────────────────────────
export MASTER_PORT
export TOKENIZERS_PARALLELISM="false"
export WANDB_MODE

# VARGPT image size constraints for self-evolving
export SE_MAX_IMAGE_SIDE="${SE_MAX_IMAGE_SIDE:-1024}"
export SE_MIN_IMAGE_SIDE="${SE_MIN_IMAGE_SIDE:-56}"
export SE_IMAGE_SIZE_MULTIPLE="${SE_IMAGE_SIZE_MULTIPLE:-28}"

# CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

# NCCL / DDP settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Uncomment if using HuggingFace mirror
# export HF_ENDPOINT=https://hf-mirror.com

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config file not found: $CONFIG" >&2
    echo "[ERROR] Run this script from the VARGPT-family-training/ directory." >&2
    exit 1
fi

if ! command -v llamafactory-cli &>/dev/null; then
    echo "[ERROR] llamafactory-cli not found. Install LLaMA-Factory first." >&2
    echo "[ERROR] pip install -e . (from VARGPT-family-training/)" >&2
    exit 1
fi

# ── Build extra CLI args ─────────────────────────────────────────────────────
EXTRA_ARGS=""

# Resume from checkpoint
if [[ -n "$RESUME_FROM" ]]; then
    if [[ ! -d "$RESUME_FROM" ]]; then
        echo "[WARN] Resume path does not exist: $RESUME_FROM"
        echo "[WARN] Proceeding anyway (LLaMA-Factory may handle this)."
    fi
    EXTRA_ARGS="$EXTRA_ARGS --resume_from_checkpoint $RESUME_FROM"
fi

# Optional: override total steps from environment
if [[ -n "${SE_TOTAL_STEPS:-}" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --se_total_steps $SE_TOTAL_STEPS"
fi

# ── Print experiment info ────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  VARGPT Self-Evolving: Joint (3U+2G) Combined Experiment"
echo "═══════════════════════════════════════════════════════════"
echo "  Repo root   : $REPO_ROOT"
echo "  Config      : $CONFIG"
echo "  GPUs        : $NPROC_PER_NODE"
echo "  Master port : $MASTER_PORT"
echo "  W&B mode    : $WANDB_MODE"
if [[ -n "$RESUME_FROM" ]]; then
echo "  Resume from : $RESUME_FROM"
fi
if [[ -n "${SE_TOTAL_STEPS:-}" ]]; then
echo "  Total steps : $SE_TOTAL_STEPS (override)"
fi
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Launch training ──────────────────────────────────────────────────────────
if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
    echo "[INFO] Launching with torchrun (DDP, $NPROC_PER_NODE GPUs)..."
    echo ""
    FORCE_TORCHRUN=1 \
    NNODES=1 \
    NODE_RANK=0 \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="$MASTER_PORT" \
        llamafactory-cli train "$CONFIG" $EXTRA_ARGS
else
    echo "[INFO] Launching single-GPU training..."
    echo ""
    llamafactory-cli train "$CONFIG" $EXTRA_ARGS
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Training complete."
echo "  Output dir: saves/vargpt_v1_1_se_joint_3U2G"
echo "═══════════════════════════════════════════════════════════"
