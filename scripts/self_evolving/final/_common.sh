#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Common environment and variables shared by all final experiments.
# Source this file; do not execute it directly.
# ══════════════════════════════════════════════════════════════════════════════

REPO_ROOT="/workspace/self-evolving-uug/self-evolving-uug"
PYTHON_BIN="python3"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
GENERATION_IMAGE_SIDE="${GENERATION_IMAGE_SIDE:-896}"

# ── Default data ─────────────────────────────────────────────────────────────
# For JOINT experiments (E1, E3, E4, E5, E6): use the natural-image pool.
#   DiT/generator cannot produce charts → chart images waste generation steps.
#   Download with: python scripts/download_joint_3k.py
#
# For UNDERSTANDING-ONLY (E2): the chart-heavy 50k pool is fine (no G steps).
#   Override: DATA_DIR=/workspace/.../shared_uug_50k_balanced/images bash E2_...
#
DATA_DIR="${DATA_DIR:-/workspace/self-evolving-uug/data/joint_3k/images}"

# ── Cache / environment ──────────────────────────────────────────────────────
CACHE_ROOT="/workspace/self-evolving-uug/self-evolving-uug/cache"
CACHE_TMP_DIR="$CACHE_ROOT/tmp"
CACHE_TORCH_EXT_DIR="$CACHE_ROOT/torch_extensions"
CACHE_WANDB_DIR="$CACHE_ROOT/wandb"
CACHE_MIOPEN_DIR="$CACHE_ROOT/miopen"
CACHE_CUDA_DIR="$CACHE_ROOT/cuda"
mkdir -p \
  "$CACHE_ROOT" \
  "$CACHE_TMP_DIR" \
  "$CACHE_TORCH_EXT_DIR" \
  "$CACHE_WANDB_DIR" \
  "$CACHE_MIOPEN_DIR" \
  "$CACHE_CUDA_DIR" \
  "$CACHE_ROOT/assets"

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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_DISTRIBUTED_DEBUG="OFF"
export NCCL_DEBUG="WARN"
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

TRAIN_ENTRY="/workspace/self-evolving-uug/self-evolving-uug/BLIP3o/blip3o/train/train_self_evolving.py"

# ── Resume helper ────────────────────────────────────────────────────────────
RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "[COMMON] Resuming from checkpoint: $RESUME_FROM"
  RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM")
  if [[ "${RESET_PROPOSER_BASELINE:-0}" == "1" ]]; then
    echo "[COMMON] Resetting proposer baseline on resume."
    RESUME_ARGS+=(--reset_proposer_baseline)
  fi
fi

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [[ ! -d "$DATA_DIR" ]]; then
  echo "[COMMON] ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
  echo "[COMMON] If 50k data is not downloaded, set DATA_DIR to your local image dir." >&2
  exit 1
fi

# ── Shared training arguments (model, LoRA, optimiser, sampling) ─────────────
# These are identical across all experiments to ensure fair comparison.
SHARED_ARGS=(
  --model_name BLIP3o/BLIP3o-Model-8B
  --dtype bfloat16
  --attn_implementation "$ATTN_IMPL"
  --device_map single
  --cuda_device 0

  # Training schedule
  --total_steps 1500
  --save_every 50
  --log_every 1
  --max_checkpoints 30
  --save_generated_images_every 50
  --deterministic

  # Model / LoRA
  --require_decoder_for_blip3o
  --use_lora
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,mm_projector

  # Optimiser
  --lr 1e-6
  --weight_decay 0.01
  --grad_clip 1.0
  --grad_accum_steps 4

  # Sampling
  --temp 1.0
  --top_p 1.0
  --max_new_tokens_solver 96
  --max_new_tokens_proposer 384
  --max_new_tokens_caption 64
  --max_new_tokens_generator 512
  --num_solver_samples 5
  --num_solver_samples_spec 2
  --num_generations 3
  --proposer_num_candidates 3
  --proposer_spot_check_samples 2

  # Image generation
  --generation_num_inference_steps 50
  --generation_guidance_scale 2.0
  --generation_height "$GENERATION_IMAGE_SIDE"
  --generation_width  "$GENERATION_IMAGE_SIDE"

  # Reward weights
  --reward_spec_weight 0.65
  --reward_cycle_weight 0.20
  --reward_diversity_weight 0.10
  --reward_contradiction_weight 0.20

  # Spec quality gates
  --min_spec_quality_for_update 0.35
  --min_spec_qa_pairs 2
  --max_expected_words 8
  --max_question_words 24

  # Solver details
  --solver_soft_gamma 0.7
  --solver_use_temperature_mix
  --sc_entropy_min 0.15
  --sc_entropy_max 1.20
  --sc_margin_max 0.90
  --entropy_iqr_min_threshold 0.10
  --sc_negative_weight 0.25
  --skip_solver_update_when_uninformative
  --len_penalty_weight 0.10
  --len_penalty_target_words 6
  --solver_hardness_min_entropy 0.20
  --easy_update_majority_frac_threshold 0.80
  --disable_entropy_iqr_filter

  # Proposer entropy target
  --prop_entropy_sigma 0.25

  # KL regularisation
  --kl_coef 0.01
  --kl_target 0.02
  --kl_adapt_rate 0.10
  --kl_min 1e-8
  --kl_max 1e2

  # Proposer optimization
  --proposer_update_rule grpo
  --proposer_grpo_gen_group_size 3

  # Baselines
  --baseline_momentum 0.6

  # Misc
  --clear_cache_every 10
  --use_ref_answer_scoring

  # Unicorn reconstruction (disabled)
  --disable_unicorn_reconstruction_sft
  --disable_unicorn_reconstruction_generator

  # Replay buffer (disabled)
  --replay_buffer_size 1
  --replay_min_reward 1.10
  --replay_max_staleness 1
  --gen_mix_source_mode buffer
  --gen_mix_ratio_start 0.0
  --gen_mix_ratio_max 0.0
  --gen_mix_ratio_warmup_steps 1

  # Logging
  --wandb_mode disabled
  --wandb_project self-evolving-uug-final

  --seed 42
)

# ── Warmup stage args (shared default) ───────────────────────────────────────
WARMUP_STAGE_ARGS=(
  --acceptance_require_non_easy
  --disable_proposer_require_objective
  --proposer_non_objective_penalty 0.0
  --difficulty_target_easy   0.30
  --difficulty_target_medium 0.50
  --difficulty_target_hard   0.20
  --rejected_question_penalty 0.25
  --zero_entropy_reward_cap 0.20
  --difficulty_sampler_min_samples 8
  --fixed_prop_entropy_target
  --prop_entropy_mu 0.90
  --solver_temp_min  0.70
  --solver_temp_max  1.60
  --solver_top_p_min 0.35
  --solver_top_p_max 1.00
)
