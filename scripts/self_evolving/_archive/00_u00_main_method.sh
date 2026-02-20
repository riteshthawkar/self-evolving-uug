#!/usr/bin/env bash

# Experiment U00: Understanding-only main run (hardcoded configuration).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/BLIP3o"

# Cache locations
export CACHE_ROOT="$REPO_ROOT/.cache"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export TORCH_HOME="$CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TOKENIZERS_PARALLELISM="false"
export AMDGPU_ASIC_ID_TABLE_PATH="/usr/share/libdrm/amdgpu.ids"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Checkpointing (safer for DDP + LoRA)
export SE_USE_GRADIENT_CHECKPOINTING=1
export SE_GRADIENT_CHECKPOINT_USE_REENTRANT=0

mkdir -p \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$HF_METRICS_CACHE" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$XDG_CACHE_HOME"

# Disable host + ROCm GPU core dumps
ulimit -Sc 0
ulimit -Hc 0

# Weights & Biases
export WANDB_MODE="online"
export WANDB_PROJECT="self-evolve-uug"

# Hardcoded experiment settings
DATA_DIR="$REPO_ROOT/data/benchmark_10k/images"
MODEL_NAME="BLIP3o/BLIP3o-Model-8B"
OUTPUT_DIR="$REPO_ROOT/runs/understanding_experiments/U00_main_method"
RUN_NAME="u00_main_default_s42_und_only_hardcoded"

TOTAL_STEPS=10000
SAVE_EVERY=500
MAX_CHECKPOINTS=20

CUDA_DEVICE=0
PYTHON_BIN="python3"
NPROC_PER_NODE=8
MASTER_PORT=29500
ATTN_IMPLEMENTATION="sdpa"
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Training knobs
LR=1e-6
WEIGHT_DECAY=0.01
GRAD_CLIP=1.0
GRAD_ACCUM_STEPS=4
PROPOSER_UPDATE_FREQ=1

TEMP=1.0
TOP_P=1.0
MAX_NEW_TOKENS_SOLVER=128
MAX_NEW_TOKENS_PROPOSER=128
NUM_SOLVER_SAMPLES=7

SOLVER_SOFT_GAMMA=0.7
SOLVER_TEMP_MIN=0.7
SOLVER_TEMP_MAX=1.3
SC_ENTROPY_MIN=0.15
SC_ENTROPY_MAX=1.20
SC_MARGIN_MAX=0.90
SC_NEGATIVE_WEIGHT=0.25
LEN_PENALTY_WEIGHT=0.10
LEN_PENALTY_TARGET_WORDS=6
PROP_ENTROPY_MU=0.90
PROP_ENTROPY_SIGMA=0.35
PROP_ENTROPY_MU_MIN=0.35
PROP_ENTROPY_MU_MAX=1.50

KL_COEF=0.01
KL_TARGET=0.005
KL_ADAPT_RATE=0.10
KL_MIN=1e-6
KL_MAX=1e2
BASELINE_MOMENTUM=0.9

CLEAR_CACHE_EVERY=25
SEED=42

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" -m torch.distributed.run --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  "$REPO_ROOT/BLIP3o/blip3o/train/train_self_evolving.py" \
  --experiment understanding_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  --dtype bfloat16 \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --device_map single \
  --cuda_device "$CUDA_DEVICE" \
  --total_steps "$TOTAL_STEPS" \
  --save_every "$SAVE_EVERY" \
  --log_every 1 \
  --max_checkpoints "$MAX_CHECKPOINTS" \
  --deterministic \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,mm_projector \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --grad_clip "$GRAD_CLIP" \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" \
  --proposer_update_freq "$PROPOSER_UPDATE_FREQ" \
  --temp "$TEMP" \
  --top_p "$TOP_P" \
  --max_new_tokens_solver "$MAX_NEW_TOKENS_SOLVER" \
  --max_new_tokens_proposer "$MAX_NEW_TOKENS_PROPOSER" \
  --num_solver_samples "$NUM_SOLVER_SAMPLES" \
  --solver_soft_gamma "$SOLVER_SOFT_GAMMA" \
  --solver_use_temperature_mix \
  --solver_temp_min "$SOLVER_TEMP_MIN" \
  --solver_temp_max "$SOLVER_TEMP_MAX" \
  --sc_entropy_min "$SC_ENTROPY_MIN" \
  --sc_entropy_max "$SC_ENTROPY_MAX" \
  --sc_margin_max "$SC_MARGIN_MAX" \
  --sc_negative_weight "$SC_NEGATIVE_WEIGHT" \
  --allow_solver_update_when_uninformative \
  --len_penalty_weight "$LEN_PENALTY_WEIGHT" \
  --len_penalty_target_words "$LEN_PENALTY_TARGET_WORDS" \
  --prop_entropy_mu "$PROP_ENTROPY_MU" \
  --prop_entropy_sigma "$PROP_ENTROPY_SIGMA" \
  --fixed_prop_entropy_target \
  --prop_entropy_mu_min "$PROP_ENTROPY_MU_MIN" \
  --prop_entropy_mu_max "$PROP_ENTROPY_MU_MAX" \
  --kl_coef "$KL_COEF" \
  --kl_target "$KL_TARGET" \
  --kl_adapt_rate "$KL_ADAPT_RATE" \
  --kl_min "$KL_MIN" \
  --kl_max "$KL_MAX" \
  --baseline_momentum "$BASELINE_MOMENTUM" \
  --clear_cache_every "$CLEAR_CACHE_EVERY" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_log_images_every 0 \
  --wandb_run_name "$RUN_NAME" \
  --seed "$SEED"
