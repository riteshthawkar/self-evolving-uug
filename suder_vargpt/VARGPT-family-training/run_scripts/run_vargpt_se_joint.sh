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

# ── Data paths (edit these) ──────────────────────────────────────────────────
DATASET="train_vargpt_v1_1_demo"
DATASET_DIR="data"
IMAGE_DIR="data"

# ── Environment setup ────────────────────────────────────────────────────────
export MASTER_PORT
export TOKENIZERS_PARALLELISM="false"
export WANDB_MODE="${WANDB_MODE:-disabled}"

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

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config not found: $CONFIG" >&2
    exit 1
fi

if ! command -v llamafactory-cli &>/dev/null; then
    echo "[ERROR] llamafactory-cli not found. Run: pip install -e ." >&2
    exit 1
fi

# ── Print experiment info ────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  VARGPT Self-Evolving: Joint (3U+2G) Combined Experiment"
echo "═══════════════════════════════════════════════════════════"
echo "  Config      : $CONFIG"
echo "  GPUs        : $NPROC_PER_NODE"
echo "  Dataset     : $DATASET"
echo "  Dataset dir : $DATASET_DIR"
echo "  Image dir   : $IMAGE_DIR"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Launch training ──────────────────────────────────────────────────────────
FORCE_TORCHRUN=1 \
NNODES=1 \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT="$MASTER_PORT" \
    llamafactory-cli train "$CONFIG" \
        --dataset "$DATASET" \
        --dataset_dir "$DATASET_DIR" \
        --image_dir "$IMAGE_DIR"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Training complete."
echo "═══════════════════════════════════════════════════════════"
