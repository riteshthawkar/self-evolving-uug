#!/usr/bin/env bash

# Experiment U06: LoRA Capacity Sensitivity
# Understanding-only self-evolving with different LoRA ranks and alphas.
# Changing values: (lora_r, lora_alpha) in {(8,16), (16,32), (32,64)}; seed fixed at 42

export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/BLIP3o:${PYTHONPATH:-}"

# Cache locations
export CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_METRICS_CACHE="${HF_METRICS_CACHE:-$HF_HOME/metrics}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_METRICS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

# Weights & Biases
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_PROJECT="${WANDB_PROJECT:-self-evolving-uug-understanding}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_LOG_IMAGES_EVERY="${WANDB_LOG_IMAGES_EVERY:-0}"

# Run defaults (override via environment)
export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/shared_uug_50k_balanced/images}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/runs/understanding_experiments}"
export TOTAL_STEPS="${TOTAL_STEPS:-6000}"
export SAVE_EVERY="${SAVE_EVERY:-200}"
export MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-3}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

mkdir -p "$OUTPUT_ROOT"

# Run: u06_lorar8_alpha16_s42
# Changes LoRA capacity to r=8, alpha=16.
torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" "$REPO_ROOT/BLIP3o/blip3o/train/train_self_evolving.py" \
  --experiment understanding_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_ROOT/U06_lora_capacity" \
  --run_name "u06_lorar8_alpha16_s42" \
  --dtype bfloat16 \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --device_map auto \
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
  --lr 1e-6 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --grad_accum_steps 4 \
  --proposer_update_freq 5 \
  --temp 1.0 \
  --top_p 1.0 \
  --max_new_tokens_solver 128 \
  --max_new_tokens_proposer 128 \
  --num_solver_samples 5 \
  --solver_soft_gamma 0.7 \
  --len_penalty_weight 0.10 \
  --len_penalty_target_words 6 \
  --prop_entropy_mu 0.90 \
  --prop_entropy_sigma 0.35 \
  --kl_coef 0.01 \
  --kl_target 0.02 \
  --kl_adapt_rate 0.10 \
  --kl_min 1e-8 \
  --kl_max 1e2 \
  --baseline_momentum 0.9 \
  --clear_cache_every 25 \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_log_images_every "$WANDB_LOG_IMAGES_EVERY" \
  --wandb_run_name "u06_lorar8_alpha16_s42" \
  --seed 42 \
  --lora_r 8 \
  --lora_alpha 16

# Run: u06_lorar16_alpha32_s42
# Changes LoRA capacity to r=16, alpha=32.
torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" "$REPO_ROOT/BLIP3o/blip3o/train/train_self_evolving.py" \
  --experiment understanding_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_ROOT/U06_lora_capacity" \
  --run_name "u06_lorar16_alpha32_s42" \
  --dtype bfloat16 \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --device_map auto \
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
  --lr 1e-6 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --grad_accum_steps 4 \
  --proposer_update_freq 5 \
  --temp 1.0 \
  --top_p 1.0 \
  --max_new_tokens_solver 128 \
  --max_new_tokens_proposer 128 \
  --num_solver_samples 5 \
  --solver_soft_gamma 0.7 \
  --len_penalty_weight 0.10 \
  --len_penalty_target_words 6 \
  --prop_entropy_mu 0.90 \
  --prop_entropy_sigma 0.35 \
  --kl_coef 0.01 \
  --kl_target 0.02 \
  --kl_adapt_rate 0.10 \
  --kl_min 1e-8 \
  --kl_max 1e2 \
  --baseline_momentum 0.9 \
  --clear_cache_every 25 \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_log_images_every "$WANDB_LOG_IMAGES_EVERY" \
  --wandb_run_name "u06_lorar16_alpha32_s42" \
  --seed 42 \
  --lora_r 16 \
  --lora_alpha 32

# Run: u06_lorar32_alpha64_s42
# Changes LoRA capacity to r=32, alpha=64.
torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" "$REPO_ROOT/BLIP3o/blip3o/train/train_self_evolving.py" \
  --experiment understanding_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_ROOT/U06_lora_capacity" \
  --run_name "u06_lorar32_alpha64_s42" \
  --dtype bfloat16 \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --device_map auto \
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
  --lr 1e-6 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --grad_accum_steps 4 \
  --proposer_update_freq 5 \
  --temp 1.0 \
  --top_p 1.0 \
  --max_new_tokens_solver 128 \
  --max_new_tokens_proposer 128 \
  --num_solver_samples 5 \
  --solver_soft_gamma 0.7 \
  --len_penalty_weight 0.10 \
  --len_penalty_target_words 6 \
  --prop_entropy_mu 0.90 \
  --prop_entropy_sigma 0.35 \
  --kl_coef 0.01 \
  --kl_target 0.02 \
  --kl_adapt_rate 0.10 \
  --kl_min 1e-8 \
  --kl_max 1e2 \
  --baseline_momentum 0.9 \
  --clear_cache_every 25 \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_log_images_every "$WANDB_LOG_IMAGES_EVERY" \
  --wandb_run_name "u06_lorar32_alpha64_s42" \
  --seed 42 \
  --lora_r 32 \
  --lora_alpha 64

