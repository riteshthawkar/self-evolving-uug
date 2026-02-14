#!/usr/bin/env bash
set -euo pipefail

# Experiment X05
# Real unlabeled images + folder-based generated mixing + DiT denoising updates.

REPO_ROOT="/Users/ritesh.thawkar/Ritesh/self-evolving-uug"
PYTHON_BIN="python3"
DATA_DIR="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/data/shared_uug_50k_balanced/images"
OUTPUT_DIR="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/runs/unified_experiments/X05_folder_mix_with_dit_update"
RUN_NAME="x05_folder_mix_with_dit_update_s42_fixed"
GENERATED_MIX_DIR="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/runs/unified_experiments/generated_mix_pool_x05"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$GENERATED_MIX_DIR"
find "$GENERATED_MIX_DIR" -maxdepth 1 -type f \( -name "*.json" -o -name "*.png" \) -delete

export PYTHONPATH="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/BLIP3o"
export HF_HOME="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/cache"
export HUGGINGFACE_HUB_CACHE="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/cache"
export HF_DATASETS_CACHE="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/cache"
export HF_METRICS_CACHE="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/cache"
export TORCH_HOME="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/cache"
export TRITON_CACHE_DIR="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/cache"
export XDG_CACHE_HOME="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/cache"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_DISTRIBUTED_DEBUG="OFF"
export NCCL_DEBUG="WARN"
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[X05] ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
  exit 1
fi
if ! find "$DATA_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) -print -quit | grep -q .; then
  echo "[X05] ERROR: DATA_DIR has no image files: $DATA_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node 8 \
  --master_port 29525 \
  "/Users/ritesh.thawkar/Ritesh/self-evolving-uug/BLIP3o/blip3o/train/train_self_evolving.py" \
  --experiment unified_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --model_name BLIP3o/BLIP3o-Model-8B \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --device_map single \
  --cuda_device 0 \
  --total_steps 10000 \
  --save_every 500 \
  --log_every 1 \
  --max_checkpoints 5 \
  --save_generated_images_every 500 \
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
  --grad_accum_steps 4 \
  --proposer_update_freq 1 \
  --generator_update_freq 1 \
  --generator_update_rule grpo \
  --enable_solver_updates \
  --solver_update_freq 2 \
  --temp 1.0 \
  --top_p 1.0 \
  --max_new_tokens_solver 96 \
  --max_new_tokens_proposer 192 \
  --max_new_tokens_caption 64 \
  --max_new_tokens_generator 512 \
  --num_solver_samples 7 \
  --num_solver_samples_spec 2 \
  --num_generations 3 \
  --generation_num_inference_steps 20 \
  --generation_guidance_scale 2.0 \
  --allow_missing_generation_tokens \
  --generator_missing_trace_strategy proxy \
  --generator_proxy_max_ratio 1.0 \
  --acceptance_require_target_bucket \
  --disable_difficulty_sampler \
  --proposer_hardening_max_retries 3 \
  --proposer_force_hardening_max_retries 2 \
  --solver_skip_update_on_easy \
  --reward_spec_weight 0.65 \
  --reward_cycle_weight 0.20 \
  --reward_diversity_weight 0.10 \
  --reward_contradiction_weight 0.20 \
  --min_spec_quality_for_update 0.35 \
  --min_spec_qa_pairs 2 \
  --max_expected_words 8 \
  --max_question_words 24 \
  --solver_soft_gamma 0.7 \
  --solver_use_temperature_mix \
  --solver_temp_min 0.7 \
  --solver_temp_max 1.3 \
  --sc_entropy_min 0.15 \
  --sc_entropy_max 1.20 \
  --sc_margin_max 0.90 \
  --sc_negative_weight 0.25 \
  --skip_solver_update_when_uninformative \
  --adaptive_prop_entropy_target \
  --prop_entropy_ema_momentum 0.90 \
  --prop_entropy_mu_min 0.40 \
  --prop_entropy_mu_max 1.50 \
  --len_penalty_weight 0.10 \
  --len_penalty_target_words 6 \
  --prop_entropy_mu 0.90 \
  --prop_entropy_sigma 0.35 \
  --understanding_steps_per_cycle 3 \
  --generation_steps_per_cycle 2 \
  --synthetic_solver_update_freq 2 \
  --solver_hardness_min_entropy 0.20 \
  --kl_coef 0.01 \
  --kl_target 0.02 \
  --kl_adapt_rate 0.10 \
  --kl_min 1e-8 \
  --kl_max 1e2 \
  --baseline_momentum 0.9 \
  --clear_cache_every 10 \
  --use_ref_answer_scoring \
  --replay_buffer_size 1000 \
  --replay_min_reward 0.50 \
  --replay_max_staleness 500 \
  --gen_mix_source_mode folder \
  --generated_mix_dir "$GENERATED_MIX_DIR" \
  --generated_mix_min_reward 0.50 \
  --generated_mix_max_files 5000 \
  --generated_mix_refresh_every 10 \
  --gen_mix_ratio_start 0.02 \
  --gen_mix_ratio_max 0.25 \
  --gen_mix_ratio_warmup_steps 1000 \
  --dit_update_enabled \
  --dit_update_freq 1 \
  --dit_lr 5e-7 \
  --dit_weight_decay 0.01 \
  --dit_grad_clip 1.0 \
  --dit_grad_accum_steps 1 \
  --dit_conditioning_dropout 0.10 \
  --dit_loss_weight 1.0 \
  --dit_prompt_suffix_token_id 151665 \
  --wandb_mode disabled \
  --wandb_project self-evolving-uug-unified \
  --wandb_run_name "$RUN_NAME" \
  --seed 42
