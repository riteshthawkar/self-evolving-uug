#!/usr/bin/env bash
set -euo pipefail

# Shared repo environment bootstrap.
BOOTSTRAP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_SEARCH_DIR="${BOOTSTRAP_DIR}"
while [[ "${BOOTSTRAP_SEARCH_DIR}" != "/" ]]; do
  if [[ -f "${BOOTSTRAP_SEARCH_DIR}/scripts/env/bootstrap_training_env.sh" ]]; then
    # shellcheck source=/dev/null
    source "${BOOTSTRAP_SEARCH_DIR}/scripts/env/bootstrap_training_env.sh"
    break
  fi
  BOOTSTRAP_SEARCH_DIR="$(dirname "${BOOTSTRAP_SEARCH_DIR}")"
done
unset BOOTSTRAP_DIR BOOTSTRAP_SEARCH_DIR

# ══════════════════════════════════════════════════════════════════════════════
# E4 — Ablation: Full Joint Training WITHOUT DiT RWR
# ══════════════════════════════════════════════════════════════════════════════
#
# Same as E1 (full joint training) but with DiT updates completely disabled:
#   • No DiT denoising MSE loss
#   • No reward-weighted regression
#   • No joint LLM→DiT gradient flow
#
# What still gets trained:
#   • Solver LoRA — understanding via GRPO (same as E1)
#   • Generator text-conditioning LoRA only when valid generator traces exist
#   • Proposer LoRA — visual-understanding curriculum (same as E1)
#
# What this experiment proves:
#   ✓ Diffusion Generator LoRA/RWR is essential for generation quality
#   ✓ Compare E4 vs E1: GenEval gap = DiT contribution
#   ✓ Understanding should be similar to E1 (DiT doesn't affect understanding)
#   ✓ This ablation is UNIQUE to our work — no competitor trains DiT jointly
#
# Why this matters for the paper:
#   SUDER/UniCorn/CoRL all use AR-based generation where the same LLM produces
#   discrete image tokens. They CAN'T train a separate DiT because there is none.
#   Our framework handles the ADDITIONAL challenge of training a continuous-latent
#   denoiser (DiT) jointly with the LLM conditioning encoder.
#
# Usage:
#   TRAIN_STAGE=warmup bash E4_no_dit_rwr.sh
#   RESUME_FROM=/path/to/step_N TRAIN_STAGE=warmup bash E4_no_dit_rwr.sh
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

HF_TOKEN_FILE="${HF_TOKEN_FILE:-${ORIGINAL_HOME:-$HOME}/.cache/huggingface/token}"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(< "$HF_TOKEN_FILE")"
fi
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/joint_pool_10k/images}"
MIN_DATA_IMAGES="${MIN_DATA_IMAGES:-10000}"
ALLOW_SMALL_DATA="${ALLOW_SMALL_DATA:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/blip3o/E4_no_dit_rwr}"
RUN_NAME="E4_no_dit_rwr_s42"
TRAIN_STAGE="${TRAIN_STAGE:-strict}"
RESUME_FROM="${RESUME_FROM:-}"
RESET_PROPOSER_BASELINE="${RESET_PROPOSER_BASELINE:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29526}"
ATTN_IMPL="${ATTN_IMPL:-auto}"
GENERATION_IMAGE_SIDE="${GENERATION_IMAGE_SIDE:-896}"
TRAIN_ENTRY="${TRAIN_ENTRY:-$REPO_ROOT/BLIP3o/blip3o/train/train_self_evolving.py}"
TOTAL_STEPS="${TOTAL_STEPS:-10000}"
LOG_EVERY="${LOG_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-50}"
SAVE_GENERATED_IMAGES_EVERY="${SAVE_GENERATED_IMAGES_EVERY:-50}"

# ── Stage-specific hyperparameters ──────────────────────────────────────────
if [[ "$TRAIN_STAGE" == "warmup" ]]; then
  RUN_NAME="${RUN_NAME}_warmup"
  STAGE_ARGS=(
    --acceptance_require_non_easy
    --proposer_require_objective
    --proposer_non_objective_penalty 0.20
    --proposer_certificate_strict_struct
    --proposer_certificate_min_score 0.60
    --proposer_easy_reward_floor -0.65
    --proposer_all_easy_rank_spread 0.20
    --all_easy_explore_trigger 2
    --all_easy_explore_steps 16
    --all_easy_explore_num_candidates 6
    --all_easy_explore_temp_boost 1.20
    --all_easy_explore_top_p_boost 0.20
    --all_easy_explore_penalty_boost 0.70
    --easy_constraint_target_rate 0.10
    --easy_constraint_lr 0.35
    --easy_constraint_penalty_scale 0.90
    --easy_constraint_selection_scale 0.75
    --proposer_early_step1 12
    --proposer_early_step2 24
    --proposer_early_candidate_non_easy_min 0.08
    --proposer_early_selected_non_easy_min 0.10
    --proposer_early_all_easy_rate_max 0.93
    --proposer_early_reward_clipped_rate_max 0.85
    --proposer_early_solver_updates_min 1
    --proposer_early_collapse_streak_max 3
    --difficulty_target_easy   0.0
    --difficulty_target_medium 0.60
    --difficulty_target_hard   0.40
    --rejected_question_penalty 0.25
    --zero_entropy_reward_cap 0.45
    --difficulty_sampler_min_samples 8
    --fixed_prop_entropy_target
    --prop_entropy_mu 0.90
    --solver_temp_min  0.70
    --solver_temp_max  2.00
    --solver_top_p_min 0.35
    --solver_top_p_max 1.00
  )
elif [[ "$TRAIN_STAGE" == "strict" ]]; then
  RUN_NAME="${RUN_NAME}_strict"
  STAGE_ARGS=(
    --acceptance_require_non_easy
    --proposer_require_objective
    --proposer_non_objective_penalty 0.20
    --proposer_certificate_strict_struct
    --proposer_certificate_min_score 0.60
    --proposer_easy_reward_floor -0.65
    --proposer_all_easy_rank_spread 0.20
    --all_easy_explore_trigger 2
    --all_easy_explore_steps 16
    --all_easy_explore_num_candidates 6
    --all_easy_explore_temp_boost 1.20
    --all_easy_explore_top_p_boost 0.20
    --all_easy_explore_penalty_boost 0.70
    --easy_constraint_target_rate 0.10
    --easy_constraint_lr 0.35
    --easy_constraint_penalty_scale 0.90
    --easy_constraint_selection_scale 0.75
    --proposer_early_step1 12
    --proposer_early_step2 24
    --proposer_early_candidate_non_easy_min 0.08
    --proposer_early_selected_non_easy_min 0.10
    --proposer_early_all_easy_rate_max 0.93
    --proposer_early_reward_clipped_rate_max 0.85
    --proposer_early_solver_updates_min 1
    --proposer_early_collapse_streak_max 3
    --difficulty_target_easy   0.0
    --difficulty_target_medium 0.70
    --difficulty_target_hard   0.30
    --rejected_question_penalty 0.35
    --zero_entropy_reward_cap 0.45
    --difficulty_sampler_min_samples 8
    --fixed_prop_entropy_target
    --prop_entropy_ema_momentum 0.90
    --prop_entropy_mu_min 0.65
    --prop_entropy_mu_max 1.50
    --solver_temp_min  0.50
    --solver_temp_max  2.50
    --solver_top_p_min 0.30
    --solver_top_p_max 1.00
  )
else
  echo "[E4] ERROR: TRAIN_STAGE must be one of: warmup, strict (got: $TRAIN_STAGE)" >&2
  exit 1
fi

# ── Resume from checkpoint (optional) ───────────────────────────────────────
RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "[E4] Resuming from checkpoint: $RESUME_FROM"
  RESUME_ARGS=(--resume_from "$RESUME_FROM")
  if [[ "${RESET_PROPOSER_BASELINE:-0}" == "1" ]]; then
    echo "[E4] Resetting proposer baseline on resume."
    RESUME_ARGS+=(--reset_proposer_baseline)
  fi
fi

# ── Directory / cache setup ──────────────────────────────────────────────────
cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache}"
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

# ── Environment ──────────────────────────────────────────────────────────────
export PYTHONPATH="$REPO_ROOT/BLIP3o"
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
if [[ -z "${HIP_VISIBLE_DEVICES:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
fi
if [[ -z "${HIP_VISIBLE_DEVICES:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
fi

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [[ ! -d "$DATA_DIR" ]]; then
  echo "[E4] ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
  exit 1
fi
IMAGE_COUNT="$(find "$DATA_DIR" -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.bmp" -o -iname "*.tiff" \) \
  | wc -l | tr -d '[:space:]')"
if [[ "$IMAGE_COUNT" -lt "$MIN_DATA_IMAGES" && "$ALLOW_SMALL_DATA" != "1" ]]; then
  echo "[E4] ERROR: DATA_DIR has $IMAGE_COUNT images; paper protocol requires at least $MIN_DATA_IMAGES." >&2
  echo "[E4] Set DATA_DIR to a local directory of unlabeled training images." >&2
  echo "[E4] For smoke tests only, set ALLOW_SMALL_DATA=1." >&2
  exit 1
fi

echo "[E4] Starting experiment E4 (No DiT RWR Ablation)"
echo "[E4]   Stage:       $TRAIN_STAGE"
echo "[E4]   Run name:    $RUN_NAME"
echo "[E4]   Output dir:  $OUTPUT_DIR"
echo "[E4]   Data dir:    $DATA_DIR"
echo "[E4]   GPUs:        $NPROC_PER_NODE"
echo "[E4]   Attn impl:   $ATTN_IMPL"
echo "[E4]   NOTE: DiT training DISABLED (LLM LoRA only)"
if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "[E4]   Resume from: $RESUME_FROM"
fi

# ── Launch ───────────────────────────────────────────────────────────────────
"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  "$TRAIN_ENTRY" \
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
  \
  `# ── Training schedule ──────────────────────────────────────────────────` \
  --total_steps "$TOTAL_STEPS" \
  --save_every "$SAVE_EVERY" \
  --log_every "$LOG_EVERY" \
  --max_checkpoints "${MAX_CHECKPOINTS:-10000}" \
  --save_generated_images_every "$SAVE_GENERATED_IMAGES_EVERY" \
  --deterministic \
  \
  `# ── Model / LoRA ───────────────────────────────────────────────────────` \
  --require_decoder_for_blip3o \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  \
  `# ── Optimiser (understanding-side GRPO) ────────────────────────────────` \
  --lr 1e-6 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --grad_accum_steps 1 \
  \
  `# ── Role update frequencies (all active, same as E1) ───────────────────` \
  --proposer_update_freq 1 \
  --generator_update_freq 1 \
  --enable_solver_updates \
  --solver_update_freq 1 \
  \
  `# ── Generator token policy path; BLIP3o has no image-token traces ──────` \
  --generator_update_rule grpo \
  --generator_missing_trace_strategy skip \
  --grpo_clip_ratio 0.2 \
  --grpo_min_group_std 1e-6 \
  \
  `# ── Sampling ────────────────────────────────────────────────────────────` \
  --temp 1.3 \
  --top_p 1.0 \
  --max_new_tokens_solver 96 \
  --max_new_tokens_proposer 384 \
  --max_new_tokens_caption 64 \
  --max_new_tokens_generator 512 \
  --num_solver_samples 7 \
  --num_solver_samples_spec 2 \
  --num_generations 3 \
  --proposer_num_candidates 3 \
  --proposer_spot_check_samples 3 \
  --grpo_extra_sc_samples 3 \
  \
  `# ── Image generation (BLIP3o diffusion) ─────────────────────────────────` \
  --generation_num_inference_steps 50 \
  --generation_guidance_scale 2.0 \
  --generation_height "$GENERATION_IMAGE_SIDE" \
  --generation_width  "$GENERATION_IMAGE_SIDE" \
  \
  `# ── Difficulty curriculum ───────────────────────────────────────────────` \
  --difficulty_sampler_enabled \
  \
  `# ── Reward weights ──────────────────────────────────────────────────────` \
  --reward_spec_weight 0.65 \
  --reward_cycle_weight 0.20 \
  --reward_diversity_weight 0.10 \
  --reward_contradiction_weight 0.20 \
  \
  `# ── Spec quality gates ──────────────────────────────────────────────────` \
  --min_spec_quality_for_update 0.35 \
  --min_spec_qa_pairs 2 \
  --max_expected_words 8 \
  --max_question_words 24 \
  \
  `# ── Solver details ──────────────────────────────────────────────────────` \
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
  --solver_hardness_min_entropy 0.20 \
  --easy_update_majority_frac_threshold 1.00 \
  --entropy_iqr_filter_enabled \
  \
  `# ── Proposer entropy target ─────────────────────────────────────────────` \
  --prop_entropy_sigma 0.25 \
  \
  `# ── Cycle scheduling (same as E1: 3U + 2G) ────────────────────────────` \
  --understanding_steps_per_cycle 3 \
  --generation_steps_per_cycle 2 \
  --synthetic_solver_update_freq 0 \
  \
  `# ── KL regularisation ───────────────────────────────────────────────────` \
  --kl_coef 0.01 \
  --kl_target 0.02 \
  --kl_adapt_rate 0.10 \
  --kl_min 0.001 \
  --kl_max 1e2 \
  \
  `# ── Proposer optimization ──────────────────────────────────────────────` \
  --proposer_update_rule grpo \
  --proposer_grpo_gen_group_size 3 \
  --proposer_grpo_unverified_extra_margin 0.02 \
  \
  `# ── Baselines ──────────────────────────────────────────────────────────` \
  --baseline_momentum 0.6 \
  \
  `# ── Misc ───────────────────────────────────────────────────────────────` \
  --clear_cache_every 10 \
  --no_ref_answer_scoring \
  \
  `# ── Unicorn reconstruction (disabled) ──────────────────────────────────` \
  --disable_unicorn_reconstruction_sft \
  --disable_unicorn_reconstruction_generator \
  \
  `# ── Replay buffer (disabled) ───────────────────────────────────────────` \
  --replay_buffer_size 1 \
  --replay_min_reward 1.10 \
  --replay_max_staleness 1 \
  --gen_mix_source_mode buffer \
  --gen_mix_ratio_start 0.0 \
  --gen_mix_ratio_max 0.0 \
  --gen_mix_ratio_warmup_steps 1 \
  \
  `# ── DiT DISABLED — no denoising loss, no RWR, no joint conditioning ───` \
  --dit_update_freq 0 \
  --dit_lr 5e-7 \
  --dit_weight_decay 0.01 \
  --dit_grad_clip 1.0 \
  --dit_grad_accum_steps 1 \
  --dit_conditioning_dropout 0.10 \
  --dit_loss_weight 1.0 \
  --dit_prompt_suffix_token_id 151665 \
  \
  `# ── Main-method coupling only: no G-step proposer/solver updates ───────` \
  \
  `# ── Logging / W&B ─────────────────────────────────────────────────────` \
  --wandb_mode disabled \
  --wandb_project self-evolving-uug-final \
  --wandb_run_name "$RUN_NAME" \
  \
  `# ── Stage-specific args (difficulty curriculum) ────────────────────────` \
  "${STAGE_ARGS[@]}" \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  --seed 42
