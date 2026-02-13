#!/usr/bin/env bash

# Experiment X00: Unified Self-Evolving Main Run
# Uses original BLIP3o (`BLIP3o/BLIP3o-Model-8B`) with alternating
# understanding and generation updates in a single loop.
#
# Model target: This script defaults to BLIP3o-Model-8B (original).
# The TRAINING_PIPELINE.md references BLIP3o-NEXT/EvoLMM; to run
# that variant, override MODEL_NAME and ensure the decoder pipeline
# is compatible (see model_api.py loader).
#
# Portability: all paths default to $REPO_ROOT-relative locations.
# Override CACHE_ROOT, DATA_DIR, MODEL_NAME via environment for your setup.

export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/BLIP3o:${PYTHONPATH:-}"

# Cache locations (override CACHE_ROOT to point at your local cache dir)
export CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT}"
export HF_METRICS_CACHE="${HF_METRICS_CACHE:-$CACHE_ROOT}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT}"
export TOKENIZERS_PARALLELISM="false"
export AMDGPU_ASIC_ID_TABLE_PATH="${AMDGPU_ASIC_ID_TABLE_PATH:-/usr/share/libdrm/amdgpu.ids}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

# Weights & Biases
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_PROJECT="${WANDB_PROJECT:-self-evolve-uug}"

# Run defaults (override via environment)
# DATA_DIR must be set or defaults to $REPO_ROOT/data  (override for your setup)
export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
export INCLUDE_SUBFOLDERS="${INCLUDE_SUBFOLDERS:-}"
# NOTE: This experiment targets the original BLIP3o-Model-8B.  For BLIP3o-NEXT
# experiments, override MODEL_NAME accordingly and verify decoder compatibility.
export MODEL_NAME="${MODEL_NAME:-BLIP3o/BLIP3o-Model-8B}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/runs/unified_experiments}"
export TOTAL_STEPS="${TOTAL_STEPS:-10000}"
export SAVE_EVERY="${SAVE_EVERY:-500}"
export MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-5}"
export SAVE_GENERATED_IMAGES_EVERY="${SAVE_GENERATED_IMAGES_EVERY:-500}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-3}"
export NUM_SOLVER_SAMPLES="${NUM_SOLVER_SAMPLES:-7}"
export NUM_SOLVER_SAMPLES_SPEC="${NUM_SOLVER_SAMPLES_SPEC:-2}"
export PROPOSER_UPDATE_FREQ="${PROPOSER_UPDATE_FREQ:-1}"
export SOLVER_USE_TEMPERATURE_MIX="${SOLVER_USE_TEMPERATURE_MIX:-1}"
export SOLVER_TEMP_MIN="${SOLVER_TEMP_MIN:-0.7}"
export SOLVER_TEMP_MAX="${SOLVER_TEMP_MAX:-1.3}"
export SC_ENTROPY_MIN="${SC_ENTROPY_MIN:-0.15}"
export SC_ENTROPY_MAX="${SC_ENTROPY_MAX:-1.20}"
export SC_MARGIN_MAX="${SC_MARGIN_MAX:-0.90}"
export SC_NEGATIVE_WEIGHT="${SC_NEGATIVE_WEIGHT:-0.25}"
export SKIP_SOLVER_UPDATE_WHEN_UNINFORMATIVE="${SKIP_SOLVER_UPDATE_WHEN_UNINFORMATIVE:-1}"
export GENERATION_NUM_INFERENCE_STEPS="${GENERATION_NUM_INFERENCE_STEPS:-20}"
export SOLVER_UPDATE_FREQ="${SOLVER_UPDATE_FREQ:-2}"
export SYNTHETIC_SOLVER_UPDATE_FREQ="${SYNTHETIC_SOLVER_UPDATE_FREQ:-2}"
export SYNTHETIC_SOLVER_HARD_ONLY="${SYNTHETIC_SOLVER_HARD_ONLY:-1}"
export SOLVER_HARDNESS_MIN_ENTROPY="${SOLVER_HARDNESS_MIN_ENTROPY:-0.20}"
export MAX_NEW_TOKENS_SOLVER="${MAX_NEW_TOKENS_SOLVER:-96}"
export MAX_NEW_TOKENS_PROPOSER="${MAX_NEW_TOKENS_PROPOSER:-192}"
export MAX_NEW_TOKENS_CAPTION="${MAX_NEW_TOKENS_CAPTION:-64}"
export MAX_NEW_TOKENS_GENERATOR="${MAX_NEW_TOKENS_GENERATOR:-512}"
export GENERATOR_PROXY_MAX_RATIO="${GENERATOR_PROXY_MAX_RATIO:-1.00}"
export CLEAR_CACHE_EVERY="${CLEAR_CACHE_EVERY:-10}"
# Self-evolving feedback loop: generated images mix into understanding training.
# The mix ratio ramps linearly from START → MAX over WARMUP_STEPS.
# Replay buffer naturally fills once generation starts producing good images.
export USE_REF_ANSWER_SCORING="${USE_REF_ANSWER_SCORING:-0}"
export REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-1000}"
export REPLAY_MIN_REWARD="${REPLAY_MIN_REWARD:-0.50}"
export REPLAY_MAX_STALENESS="${REPLAY_MAX_STALENESS:-500}"
export GEN_MIX_RATIO_START="${GEN_MIX_RATIO_START:-0.02}"
export GEN_MIX_RATIO_MAX="${GEN_MIX_RATIO_MAX:-0.25}"
export GEN_MIX_RATIO_WARMUP_STEPS="${GEN_MIX_RATIO_WARMUP_STEPS:-1000}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29520}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# Use local BLIP3o classes from the current repository checkout.
export BLIP3O_REPO="${BLIP3O_REPO:-$REPO_ROOT/BLIP3o}"
export BLIP3O_USE_LOCAL_CLASSES="${BLIP3O_USE_LOCAL_CLASSES:-1}"
# Decoder fallback source for BLIP3o-Model-8B latent->image decoding.
export BLIP3O_DIFFUSION_REPO="${BLIP3O_DIFFUSION_REPO:-BLIP3o/BLIP3o-Model}"
# Checkpoint setting: non-reentrant is safer with DDP + multi-adapter LoRA.
export SE_USE_GRADIENT_CHECKPOINTING="${SE_USE_GRADIENT_CHECKPOINTING:-1}"
export SE_GRADIENT_CHECKPOINT_USE_REENTRANT="0"

# Disable host + ROCm GPU core dumps (prevents core* and gpucore.*)
ulimit -Sc 0
ulimit -Hc 0


export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

SC_TEMP_FLAGS=()
if [[ "$SOLVER_USE_TEMPERATURE_MIX" == "1" ]]; then
  SC_TEMP_FLAGS+=(--solver_use_temperature_mix)
else
  SC_TEMP_FLAGS+=(--disable_solver_temperature_mix)
fi

SC_UPDATE_FLAGS=()
if [[ "$SKIP_SOLVER_UPDATE_WHEN_UNINFORMATIVE" == "1" ]]; then
  SC_UPDATE_FLAGS+=(--skip_solver_update_when_uninformative)
else
  SC_UPDATE_FLAGS+=(--allow_solver_update_when_uninformative)
fi

SYNTH_HARD_FLAGS=()
if [[ "$SYNTHETIC_SOLVER_HARD_ONLY" == "1" ]]; then
  SYNTH_HARD_FLAGS+=(--synthetic_solver_hard_only)
fi

SUBFOLDER_FLAGS=()
if [[ -n "$INCLUDE_SUBFOLDERS" ]]; then
  SUBFOLDER_FLAGS+=(--include_subfolders "$INCLUDE_SUBFOLDERS")
fi

REF_SCORING_FLAGS=()
if [[ "$USE_REF_ANSWER_SCORING" == "1" ]]; then
  REF_SCORING_FLAGS+=(--use_ref_answer_scoring)
fi

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" "$REPO_ROOT/BLIP3o/blip3o/train/train_self_evolving.py" \
  --experiment unified_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  "${SUBFOLDER_FLAGS[@]}" \
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
  --grad_accum_steps 4 \
  --proposer_update_freq "$PROPOSER_UPDATE_FREQ" \
  --generator_update_freq 1 \
  --generator_update_rule grpo \
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
  --generator_proxy_max_ratio "$GENERATOR_PROXY_MAX_RATIO" \
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
  "${SC_TEMP_FLAGS[@]}" \
  --solver_temp_min "$SOLVER_TEMP_MIN" \
  --solver_temp_max "$SOLVER_TEMP_MAX" \
  --sc_entropy_min "$SC_ENTROPY_MIN" \
  --sc_entropy_max "$SC_ENTROPY_MAX" \
  --sc_margin_max "$SC_MARGIN_MAX" \
  --sc_negative_weight "$SC_NEGATIVE_WEIGHT" \
  "${SC_UPDATE_FLAGS[@]}" \
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
  --synthetic_solver_update_freq "$SYNTHETIC_SOLVER_UPDATE_FREQ" \
  "${SYNTH_HARD_FLAGS[@]}" \
  --solver_hardness_min_entropy "$SOLVER_HARDNESS_MIN_ENTROPY" \
  --kl_coef 0.01 \
  --kl_target 0.02 \
  --kl_adapt_rate 0.10 \
  --kl_min 1e-8 \
  --kl_max 1e2 \
  --baseline_momentum 0.9 \
  --clear_cache_every "$CLEAR_CACHE_EVERY" \
  "${REF_SCORING_FLAGS[@]}" \
  --replay_buffer_size "$REPLAY_BUFFER_SIZE" \
  --replay_min_reward "$REPLAY_MIN_REWARD" \
  --replay_max_staleness "$REPLAY_MAX_STALENESS" \
  --gen_mix_ratio_start "$GEN_MIX_RATIO_START" \
  --gen_mix_ratio_max "$GEN_MIX_RATIO_MAX" \
  --gen_mix_ratio_warmup_steps "$GEN_MIX_RATIO_WARMUP_STEPS" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "x00_main_default_s42" \
  --seed 42
