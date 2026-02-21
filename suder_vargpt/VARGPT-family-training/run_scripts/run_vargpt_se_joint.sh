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
IMAGE_FOLDER="/path/to/your/images"

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

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config not found: $CONFIG" >&2
    exit 1
fi

if ! command -v llamafactory-cli &>/dev/null; then
    echo "[ERROR] llamafactory-cli not found. Run: pip install -e ." >&2
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
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Launch training ──────────────────────────────────────────────────────────
FORCE_TORCHRUN=1 \
NNODES=1 \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT="$MASTER_PORT" \
    llamafactory-cli train "$CONFIG" \
        --se_image_folder "$IMAGE_FOLDER"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Training complete."
echo "═══════════════════════════════════════════════════════════"
