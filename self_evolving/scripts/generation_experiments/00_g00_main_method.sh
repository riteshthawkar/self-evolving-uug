#!/usr/bin/env bash

# Experiment G00: Generation-Only Self-Evolving Main Run
# Updates generator + proposer adapters; solver remains frozen as verifier.

export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO_ROOT"

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
export AMDGPU_ASIC_ID_TABLE_PATH="${AMDGPU_ASIC_ID_TABLE_PATH:-/usr/share/libdrm/amdgpu.ids}"

# Weights & Biases
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_PROJECT="${WANDB_PROJECT:-self-evolving-uug-generation}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_LOG_IMAGES_EVERY="${WANDB_LOG_IMAGES_EVERY:-0}"

# Run defaults (override via environment)
export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/shared_uug_50k_balanced/images}"
export MODEL_NAME="${MODEL_NAME:-BLIP3o/BLIP3o-Model-8B}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/runs/generation_experiments}"
export TOTAL_STEPS="${TOTAL_STEPS:-6000}"
export SAVE_EVERY="${SAVE_EVERY:-200}"
export MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-3}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29510}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# Use local original BLIP3o classes from bundled checkout.
export BLIP3O_REPO="${BLIP3O_REPO:-$REPO_ROOT/BLIP3o}"
export BLIP3O_USE_LOCAL_CLASSES="${BLIP3O_USE_LOCAL_CLASSES:-1}"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" self_evolving/run_experiment.py \
  --experiment generation_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_ROOT/G00_main_method" \
  --run_name "g00_main_default_s42" \
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
  --proposer_update_freq 5 \
  --generator_update_freq 1 \
  --generator_update_rule reinforce \
  --temp 1.0 \
  --top_p 1.0 \
  --max_new_tokens_solver 128 \
  --max_new_tokens_proposer 256 \
  --max_new_tokens_caption 96 \
  --max_new_tokens_generator 768 \
  --num_solver_samples 5 \
  --num_solver_samples_spec 3 \
  --num_generations 4 \
  --generation_num_inference_steps 30 \
  --generation_guidance_scale 2.0 \
  --allow_missing_generation_tokens \
  --generator_missing_trace_strategy proxy \
  --verification_use_reference_solver \
  --reward_spec_weight 0.65 \
  --reward_cycle_weight 0.20 \
  --reward_diversity_weight 0.10 \
  --reward_contradiction_weight 0.20 \
  --min_spec_quality_for_update 0.35 \
  --min_spec_qa_pairs 2 \
  --max_expected_words 8 \
  --max_question_words 24 \
  --solver_soft_gamma 0.7 \
  --len_penalty_weight 0.10 \
  --len_penalty_target_words 6 \
  --prop_entropy_mu 0.90 \
  --prop_entropy_sigma 0.35 \
  --kl_coef 1e-3 \
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
  --wandb_run_name "g00_main_default_s42" \
  --seed 42
