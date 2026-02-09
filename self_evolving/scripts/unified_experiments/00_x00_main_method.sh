#!/usr/bin/env bash

# Experiment X00: Unified Self-Evolving Main Run
# Uses original BLIP3o (`BLIP3o/BLIP3o-Model-8B`) with alternating
# understanding and generation updates in a single loop.

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

# Weights & Biases
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_PROJECT="${WANDB_PROJECT:-self-evolving-uug-unified}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_LOG_IMAGES_EVERY="${WANDB_LOG_IMAGES_EVERY:-0}"

# Run defaults (override via environment)
export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/shared_uug_50k_balanced/images}"
export MODEL_NAME="${MODEL_NAME:-BLIP3o/BLIP3o-Model-8B}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/runs/unified_experiments}"
export TOTAL_STEPS="${TOTAL_STEPS:-10000}"
export SAVE_EVERY="${SAVE_EVERY:-200}"
export MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-3}"
export SAVE_GENERATED_IMAGES_EVERY="${SAVE_GENERATED_IMAGES_EVERY:-500}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
export NUM_SOLVER_SAMPLES="${NUM_SOLVER_SAMPLES:-3}"
export NUM_SOLVER_SAMPLES_SPEC="${NUM_SOLVER_SAMPLES_SPEC:-2}"
export GENERATION_NUM_INFERENCE_STEPS="${GENERATION_NUM_INFERENCE_STEPS:-20}"
export SOLVER_UPDATE_FREQ="${SOLVER_UPDATE_FREQ:-2}"
export SYNTHETIC_SOLVER_UPDATE_FREQ="${SYNTHETIC_SOLVER_UPDATE_FREQ:-2}"
export MAX_NEW_TOKENS_SOLVER="${MAX_NEW_TOKENS_SOLVER:-96}"
export MAX_NEW_TOKENS_PROPOSER="${MAX_NEW_TOKENS_PROPOSER:-192}"
export MAX_NEW_TOKENS_CAPTION="${MAX_NEW_TOKENS_CAPTION:-64}"
export MAX_NEW_TOKENS_GENERATOR="${MAX_NEW_TOKENS_GENERATOR:-512}"
export CLEAR_CACHE_EVERY="${CLEAR_CACHE_EVERY:-10}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29520}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# Use local original BLIP3o classes from official main branch (patched for diffusers compatibility).
export BLIP3O_REPO="${BLIP3O_REPO:-$REPO_ROOT/BLIP3o-official}"
export BLIP3O_USE_LOCAL_CLASSES="${BLIP3O_USE_LOCAL_CLASSES:-1}"
# Decoder fallback source for BLIP3o-Model-8B latent->image decoding.
export BLIP3O_DIFFUSION_REPO="${BLIP3O_DIFFUSION_REPO:-BLIP3o/BLIP3o-Model}"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" self_evolving/run_experiment.py \
  --experiment unified_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_ROOT/X00_main_method" \
  --run_name "x00_main_default_s42" \
  --dtype bfloat16 \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --device_map single \
  --cuda_device "$CUDA_DEVICE" \
  --total_steps "$TOTAL_STEPS" \
  --save_every "$SAVE_EVERY" \
  --log_every 1 \
  --max_checkpoints "$MAX_CHECKPOINTS" \
  --save_generated_images_every "$SAVE_GENERATED_IMAGES_EVERY" \
  --deterministic \
  --require_decoder_for_blip3o \
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
  --enable_solver_updates \
  --solver_update_freq "$SOLVER_UPDATE_FREQ" \
  --temp 1.0 \
  --top_p 1.0 \
  --max_new_tokens_solver "$MAX_NEW_TOKENS_SOLVER" \
  --max_new_tokens_proposer "$MAX_NEW_TOKENS_PROPOSER" \
  --max_new_tokens_caption "$MAX_NEW_TOKENS_CAPTION" \
  --max_new_tokens_generator "$MAX_NEW_TOKENS_GENERATOR" \
  --num_solver_samples "$NUM_SOLVER_SAMPLES" \
  --num_solver_samples_spec "$NUM_SOLVER_SAMPLES_SPEC" \
  --num_generations "$NUM_GENERATIONS" \
  --generation_num_inference_steps "$GENERATION_NUM_INFERENCE_STEPS" \
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
  --understanding_steps_per_cycle 3 \
  --generation_steps_per_cycle 2 \
  --synthetic_solver_update_freq "$SYNTHETIC_SOLVER_UPDATE_FREQ" \
  --kl_coef 1e-3 \
  --kl_target 0.02 \
  --kl_adapt_rate 0.10 \
  --kl_min 1e-8 \
  --kl_max 1e2 \
  --baseline_momentum 0.9 \
  --clear_cache_every "$CLEAR_CACHE_EVERY" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_log_images_every "$WANDB_LOG_IMAGES_EVERY" \
  --wandb_run_name "x00_main_default_s42" \
  --seed 42
