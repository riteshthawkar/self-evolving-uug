#!/usr/bin/env bash

# Experiment U00: Understanding-only main run (hardcoded, no runtime input knobs).

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

# Weights & Biases
export WANDB_MODE="online"
export WANDB_PROJECT="self-evolve-uug"

# Device/runtime
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

mkdir -p \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$HF_METRICS_CACHE" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$REPO_ROOT/runs/understanding_experiments/U00_main_method"

# Disable host + ROCm GPU core dumps
ulimit -Sc 0
ulimit -Hc 0

torchrun --standalone \
  --nproc_per_node 8 \
  --master_port 29500 \
  "$REPO_ROOT/BLIP3o/blip3o/train/train_self_evolving.py" \
  --experiment understanding_self_evolving \
  --data_dir "$REPO_ROOT/data/benchmark_10k/images" \
  --data_split all \
  --model_name BLIP3o/BLIP3o-Model-8B \
  --output_dir "$REPO_ROOT/runs/understanding_experiments/U00_main_method" \
  --run_name u00_main_default_s42_und_only_fixed_hardcoded \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --device_map single \
  --cuda_device 0 \
  --total_steps 10000 \
  --save_every 500 \
  --log_every 1 \
  --max_checkpoints 5 \
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
  --proposer_update_freq 1 \
  --temp 1.0 \
  --top_p 1.0 \
  --max_new_tokens_solver 64 \
  --max_new_tokens_proposer 96 \
  --num_solver_samples 5 \
  --solver_soft_gamma 0.7 \
  --solver_use_temperature_mix \
  --solver_temp_min 0.7 \
  --solver_temp_max 1.5 \
  --solver_top_p_min 0.5 \
  --solver_top_p_max 1.0 \
  --sc_entropy_min 0.30 \
  --sc_entropy_max 1.50 \
  --sc_margin_max 0.70 \
  --sc_informative_ratio_min 0.50 \
  --sc_negative_weight 0.25 \
  --easy_solver_penalty_scale 1.0 \
  --skip_solver_update_when_uninformative \
  --disable_solver_always_update_with_informative_scaling \
  --solver_update_min_scale 0.20 \
  --len_penalty_weight 0.10 \
  --len_penalty_target_words 6 \
  --prop_entropy_mu 0.90 \
  --prop_entropy_sigma 0.35 \
  --adaptive_prop_entropy_target \
  --prop_entropy_ema_momentum 0.95 \
  --prop_entropy_mu_min 0.35 \
  --prop_entropy_mu_max 1.50 \
  --zero_entropy_reward_cap 0.02 \
  --easy_question_penalty 0.15 \
  --kl_coef 0.01 \
  --kl_target 0.01 \
  --kl_adapt_rate 0.05 \
  --kl_min 1e-5 \
  --kl_max 1e2 \
  --baseline_momentum 0.9 \
  --clear_cache_every 25 \
  --wandb_mode online \
  --wandb_project self-evolve-uug \
  --wandb_log_images_every 0 \
  --wandb_run_name u00_main_default_s42_und_only_fixed_hardcoded \
  --seed 42
