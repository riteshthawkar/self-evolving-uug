#!/usr/bin/env bash
set -euo pipefail

# Experiment X06
# Real unlabeled images + folder-based generated mixing.
# Generation candidate ranking uses internal CLIP-style reward from BLIP3o's
# own frozen embedding path (no external reward model checkpoint).
# Generation-side learning uses DiT updates only (no proxy text-policy updates).

REPO_ROOT="/workspace/self-evolving-uug/self-evolving-uug"
PYTHON_BIN="python3"
DATA_DIR="/workspace/self-evolving-uug/data/benchmark_10k/images"
OUTPUT_DIR="/workspace/self-evolving-uug/self-evolving-uug/runs/unified_experiments/X06_folder_generated_mix_self_clip_reward"
RUN_NAME="x06_folder_generated_mix_self_clip_reward_s42_fixed"
GENERATED_MIX_DIR="/workspace/self-evolving-uug/self-evolving-uug/runs/unified_experiments/generated_mix_pool_x06"
TRAIN_STAGE="${TRAIN_STAGE:-warmup}"   # warmup | strict
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

if [[ "$TRAIN_STAGE" == "warmup" ]]; then
  RUN_NAME="${RUN_NAME}_warmup"
  STAGE_ARGS=(
    --disable_acceptance_require_non_easy
    --difficulty_target_easy 0.30
    --difficulty_target_medium 0.50
    --difficulty_target_hard 0.20
    --rejected_question_penalty 0.10
    --fixed_prop_entropy_target
    --prop_entropy_mu 0.90
    --solver_temp_min 0.70
    --solver_temp_max 1.60
    --solver_top_p_min 0.35
    --solver_top_p_max 1.00
  )
elif [[ "$TRAIN_STAGE" == "strict" ]]; then
  RUN_NAME="${RUN_NAME}_strict"
  STAGE_ARGS=(
    --acceptance_require_non_easy
    --difficulty_target_easy 0.10
    --difficulty_target_medium 0.70
    --difficulty_target_hard 0.20
    --rejected_question_penalty 0.35
    --fixed_prop_entropy_target
    --prop_entropy_ema_momentum 0.90
    --prop_entropy_mu_min 0.65
    --prop_entropy_mu_max 1.50
    --solver_temp_min 0.70
    --solver_temp_max 1.30
    --solver_top_p_min 0.50
    --solver_top_p_max 1.00
  )
else
  echo "[X06] ERROR: TRAIN_STAGE must be one of: warmup, strict (got: $TRAIN_STAGE)" >&2
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$GENERATED_MIX_DIR"
# Prevent cross-run contamination from older generated pools.
find "$GENERATED_MIX_DIR" -maxdepth 1 -type f \( -name "*.json" -o -name "*.png" \) -delete

CACHE_ROOT="/workspace/self-evolving-uug/self-evolving-uug/cache"
CACHE_TMP_DIR="$CACHE_ROOT/tmp"
CACHE_TORCH_EXT_DIR="$CACHE_ROOT/torch_extensions"
CACHE_WANDB_DIR="$CACHE_ROOT/wandb"
CACHE_MIOPEN_DIR="$CACHE_ROOT/miopen"
CACHE_CUDA_DIR="$CACHE_ROOT/cuda"
mkdir -p "$CACHE_ROOT" "$CACHE_TMP_DIR" "$CACHE_TORCH_EXT_DIR" "$CACHE_WANDB_DIR" "$CACHE_MIOPEN_DIR" "$CACHE_CUDA_DIR" "$CACHE_ROOT/assets"

export PYTHONPATH="/workspace/self-evolving-uug/self-evolving-uug/BLIP3o"
export HF_HOME="$CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_ROOT"
export HF_ASSETS_CACHE="$CACHE_ROOT/assets"
export TRANSFORMERS_CACHE="$CACHE_ROOT"
export HF_DATASETS_CACHE="$CACHE_ROOT"
export HF_METRICS_CACHE="$CACHE_ROOT"
export TORCH_HOME="$CACHE_ROOT"
export TRITON_CACHE_DIR="$CACHE_ROOT"
export TORCH_EXTENSIONS_DIR="$CACHE_TORCH_EXT_DIR"
export XDG_CACHE_HOME="$CACHE_ROOT"
export TMPDIR="$CACHE_TMP_DIR"
export TMP="$CACHE_TMP_DIR"
export TEMP="$CACHE_TMP_DIR"
export WANDB_DIR="$CACHE_WANDB_DIR"
export WANDB_CACHE_DIR="$CACHE_WANDB_DIR"
export WANDB_CONFIG_DIR="$CACHE_WANDB_DIR"
export WANDB_DATA_DIR="$CACHE_WANDB_DIR"
export CUDA_CACHE_PATH="$CACHE_CUDA_DIR"
export MIOPEN_USER_DB_PATH="$CACHE_MIOPEN_DIR"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_MIOPEN_DIR"
export TOKENIZERS_PARALLELISM="false"
export SE_MAX_IMAGE_SIDE="${SE_MAX_IMAGE_SIDE:-896}"
export SE_MIN_IMAGE_SIDE="${SE_MIN_IMAGE_SIDE:-56}"
export SE_IMAGE_SIZE_MULTIPLE="${SE_IMAGE_SIZE_MULTIPLE:-28}"
GENERATION_IMAGE_SIDE="${GENERATION_IMAGE_SIDE:-896}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_DISTRIBUTED_DEBUG="OFF"
export NCCL_DEBUG="WARN"
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[X06] ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
  exit 1
fi
if ! find "$DATA_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) -print -quit | grep -q .; then
  echo "[X06] ERROR: DATA_DIR has no image files: $DATA_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port 29526 \
  "/workspace/self-evolving-uug/self-evolving-uug/BLIP3o/blip3o/train/train_self_evolving.py" \
  --experiment unified_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --model_name BLIP3o/BLIP3o-Model-8B \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  --dtype bfloat16 \
  --attn_implementation "$ATTN_IMPL" \
  --device_map single \
  --cuda_device 0 \
  --total_steps 10000 \
  --save_every 2000 \
  --log_every 1 \
  --max_checkpoints 5 \
  --save_generated_images_every 2000 \
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
  --generator_update_freq 0 \
  --generator_update_rule grpo \
  --enable_solver_updates \
  --solver_update_freq 1 \
  --temp 1.0 \
  --top_p 1.0 \
  --max_new_tokens_solver 96 \
  --max_new_tokens_proposer 160 \
  --max_new_tokens_caption 64 \
  --max_new_tokens_generator 512 \
  --num_solver_samples 5 \
  --num_solver_samples_spec 2 \
  --num_generations 3 \
  --generation_num_inference_steps 20 \
  --generation_guidance_scale 2.0 \
  --generation_height "$GENERATION_IMAGE_SIDE" \
  --generation_width "$GENERATION_IMAGE_SIDE" \
  --difficulty_sampler_enabled \
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
  --sc_entropy_min 0.15 \
  --sc_entropy_max 1.20 \
  --sc_margin_max 0.90 \
  --entropy_iqr_min_threshold 0.10 \
  --sc_negative_weight 0.25 \
  --skip_solver_update_when_uninformative \
  --len_penalty_weight 0.10 \
  --len_penalty_target_words 6 \
  --prop_entropy_sigma 0.25 \
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
  --no_ref_answer_scoring \
  --use_self_clip_reward_scoring \
  --unicorn_reconstruction_buffer_size 128 \
  --disable_unicorn_reconstruction_generator \
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
  "${STAGE_ARGS[@]}" \
  --seed 42
