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
# E1 — Main Experiment: Full Joint Training (Understanding + Generation + DiT)
# ══════════════════════════════════════════════════════════════════════════════
#
# This is the PRIMARY result for the paper. All components are trained jointly:
#   • Solver LoRA   — improves visual understanding via GRPO
#   • Generator LoRA — improves text-to-image conditioning via denoising gradients
#   • DiT LoRA       — improves image generation via RWR (reward-weighted MSE)
#   • Proposer LoRA  — learns the visual-understanding curriculum
#
# Key differences from X09 (the easy-data pilot):
#   • Uses the paper 10k natural-image pool, not chart-heavy 50k
#   • Trains for 10k steps (was 650 in pilot)
#
# What this experiment proves:
#   ✓ Our framework improves BOTH understanding AND generation
#   ✓ On a diffusion-based UUG model (BLIP3o) — first in the literature
#   ✓ Without any external supervision (fully self-evolving)
#
# Compare against:
#   E2 — understanding-only  (shows joint training preserves understanding)
#   E3 — generation-only     (shows joint training preserves generation)
#   E4 — no DiT RWR          (isolates DiT contribution)
#   E5 — fully imageless     (shows understanding improves without real images)
#
# Usage:
#   TRAIN_STAGE=warmup bash E1_main_joint.sh
#   RESUME_FROM=/path/to/step_N TRAIN_STAGE=warmup bash E1_main_joint.sh
#   USE_REF_ANSWER_SCORING=1 bash E1_main_joint.sh  # legacy Solver-derived reference-answer scoring
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
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/BLIP3o/run_outputs}"
RUN_NAME="${RUN_NAME:-E1_main_joint_s42_rewardfix}"
TRAIN_STAGE="${TRAIN_STAGE:-strict}"
RESUME_FROM="${RESUME_FROM:-}"
RESET_PROPOSER_BASELINE="${RESET_PROPOSER_BASELINE:-0}"
MASTER_PORT="${MASTER_PORT:-29523}"
ATTN_IMPL="${ATTN_IMPL:-auto}"
GENERATION_IMAGE_SIDE="${GENERATION_IMAGE_SIDE:-896}"
TRAIN_ENTRY="${TRAIN_ENTRY:-$REPO_ROOT/BLIP3o/blip3o/train/train_self_evolving.py}"
TOTAL_STEPS="${TOTAL_STEPS:-10000}"
LOG_EVERY="${LOG_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-50}"
SAVE_GENERATED_IMAGES_EVERY="${SAVE_GENERATED_IMAGES_EVERY:-50}"
UNDERSTANDING_STEPS_PER_CYCLE="${UNDERSTANDING_STEPS_PER_CYCLE:-3}"
GENERATION_STEPS_PER_CYCLE="${GENERATION_STEPS_PER_CYCLE:-2}"
GENERATOR_UPDATE_FREQ="${GENERATOR_UPDATE_FREQ:-1}"
DIT_UPDATE_ENABLED="${DIT_UPDATE_ENABLED:-1}"
DIT_UPDATE_FREQ="${DIT_UPDATE_FREQ:-1}"
DIT_LORA="${DIT_LORA:-1}"
DIT_LORA_R="${DIT_LORA_R:-16}"
DIT_LORA_ALPHA="${DIT_LORA_ALPHA:-32}"
DIT_LORA_DROPOUT="${DIT_LORA_DROPOUT:-0.0}"
DIT_LORA_TARGETS="${DIT_LORA_TARGETS:-attn2.to_q,attn2.to_k,attn2.to_v,attn2.to_out.0,caption_projection.linear_1,caption_projection.linear_2}"
PROPOSER_GEN_REWARD_ENABLED="${PROPOSER_GEN_REWARD_ENABLED:-0}"
GEN_STEP_SOLVER_UPDATE_ENABLED="${GEN_STEP_SOLVER_UPDATE_ENABLED:-0}"
LR="${LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
SOLVER_MERGER_LORA="${SOLVER_MERGER_LORA:-1}"
SOLVER_MERGER_LORA_R="${SOLVER_MERGER_LORA_R:-4}"
SOLVER_MERGER_LORA_ALPHA="${SOLVER_MERGER_LORA_ALPHA:-8}"
SOLVER_MERGER_LORA_LR="${SOLVER_MERGER_LORA_LR:-2e-7}"
SOLVER_MERGER_LORA_TARGETS="${SOLVER_MERGER_LORA_TARGETS:-visual.merger.mlp.0,visual.merger.mlp.2}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
BNB_4BIT_QUANT_TYPE="${BNB_4BIT_QUANT_TYPE:-nf4}"
BNB_4BIT_USE_DOUBLE_QUANT="${BNB_4BIT_USE_DOUBLE_QUANT:-1}"
BNB_4BIT_COMPUTE_DTYPE="${BNB_4BIT_COMPUTE_DTYPE:-bfloat16}"
NUM_SOLVER_SAMPLES="${NUM_SOLVER_SAMPLES:-7}"
NUM_SOLVER_SAMPLES_SPEC="${NUM_SOLVER_SAMPLES_SPEC:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-3}"
PROPOSER_NUM_CANDIDATES="${PROPOSER_NUM_CANDIDATES:-3}"
PROPOSER_SPOT_CHECK_SAMPLES="${PROPOSER_SPOT_CHECK_SAMPLES:-3}"
GRPO_EXTRA_SC_SAMPLES="${GRPO_EXTRA_SC_SAMPLES:-3}"
GENERATION_NUM_INFERENCE_STEPS="${GENERATION_NUM_INFERENCE_STEPS:-50}"
GENERATION_GUIDANCE_SCALE="${GENERATION_GUIDANCE_SCALE:-2.0}"
USE_REF_ANSWER_SCORING="${USE_REF_ANSWER_SCORING:-0}"
REWARD_SPEC_WEIGHT="${REWARD_SPEC_WEIGHT:-0.65}"
REWARD_CYCLE_WEIGHT="${REWARD_CYCLE_WEIGHT:-0.20}"
REWARD_DIVERSITY_WEIGHT="${REWARD_DIVERSITY_WEIGHT:-0.10}"
REWARD_CONTRADICTION_WEIGHT="${REWARD_CONTRADICTION_WEIGHT:-0.20}"
MIN_SPEC_QUALITY_FOR_UPDATE="${MIN_SPEC_QUALITY_FOR_UPDATE:-0.35}"
MIN_SPEC_QA_PAIRS="${MIN_SPEC_QA_PAIRS:-2}"
KL_COEF="${KL_COEF:-0.01}"
KL_TARGET="${KL_TARGET:-0.02}"
KL_ADAPT_RATE="${KL_ADAPT_RATE:-0.10}"
KL_MIN="${KL_MIN:-0.001}"
KL_MAX="${KL_MAX:-1e2}"
SOLVER_TOKEN_ENTROPY_ENABLED="${SOLVER_TOKEN_ENTROPY_ENABLED:-1}"
SOLVER_TOKEN_ENTROPY_TOKENS="${SOLVER_TOKEN_ENTROPY_TOKENS:-5}"
SOLVER_TOKEN_ENTROPY_WINDOW_SIZE="${SOLVER_TOKEN_ENTROPY_WINDOW_SIZE:-128}"
SOLVER_TOKEN_ENTROPY_SIGMOID_ALPHA="${SOLVER_TOKEN_ENTROPY_SIGMOID_ALPHA:-1.5}"
SOLVER_TOKEN_ENTROPY_SIGMOID_BETA="${SOLVER_TOKEN_ENTROPY_SIGMOID_BETA:-2.0}"
SOLVER_TOKEN_ENTROPY_AGGREGATION="${SOLVER_TOKEN_ENTROPY_AGGREGATION:-max}"
PROPOSER_STE_PRIMARY_WEIGHT="${PROPOSER_STE_PRIMARY_WEIGHT:-0.70}"
PROPOSER_SAMPLE_ENTROPY_WEIGHT="${PROPOSER_SAMPLE_ENTROPY_WEIGHT:-0.30}"
PROPOSER_STE_REWARD_WEIGHT="${PROPOSER_STE_REWARD_WEIGHT:-0.30}"
SOLVER_PPS_ENABLED="${SOLVER_PPS_ENABLED:-1}"
DIT_REWARD_LOSS_WEIGHT="${DIT_REWARD_LOSS_WEIGHT:-0.5}"
REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-1}"
REPLAY_MIN_REWARD="${REPLAY_MIN_REWARD:-1.10}"
REPLAY_MAX_STALENESS="${REPLAY_MAX_STALENESS:-1}"
GEN_MIX_SOURCE_MODE="${GEN_MIX_SOURCE_MODE:-folder}"
GEN_MIX_RATIO_START="${GEN_MIX_RATIO_START:-0.0}"
GEN_MIX_RATIO_MAX="${GEN_MIX_RATIO_MAX:-0.0}"
GEN_MIX_RATIO_WARMUP_STEPS="${GEN_MIX_RATIO_WARMUP_STEPS:-1}"

DIT_ARGS=()
if [[ "$DIT_UPDATE_ENABLED" == "1" ]]; then
  DIT_ARGS+=(
    --dit_update_enabled
    --require_dit_update
    --dit_lora_r "$DIT_LORA_R"
    --dit_lora_alpha "$DIT_LORA_ALPHA"
    --dit_lora_dropout "$DIT_LORA_DROPOUT"
    --dit_lora_targets "$DIT_LORA_TARGETS"
  )
  if [[ "$DIT_LORA" != "1" ]]; then
    DIT_ARGS+=(--disable_dit_lora)
  fi
fi

PROPOSER_GEN_REWARD_ARGS=()
if [[ "$PROPOSER_GEN_REWARD_ENABLED" == "1" ]]; then
  PROPOSER_GEN_REWARD_ARGS+=(
    --proposer_gen_reward_enabled
    --proposer_gen_entropy_weight 0.7
    --proposer_gen_baseline_momentum 0.6
  )
fi

GEN_STEP_SOLVER_ARGS=()
if [[ "$GEN_STEP_SOLVER_UPDATE_ENABLED" == "1" ]]; then
  GEN_STEP_SOLVER_ARGS+=(--gen_step_solver_update_enabled)
fi

REF_ANSWER_SCORING_ARGS=()
case "$USE_REF_ANSWER_SCORING" in
  1|true|TRUE|yes|YES)
    REF_ANSWER_SCORING_ARGS+=(--use_ref_answer_scoring)
    REWARD_MODE="ref_answer"
    ;;
  0|false|FALSE|no|NO)
    REF_ANSWER_SCORING_ARGS+=(--no_ref_answer_scoring)
    REWARD_MODE="multi_component"
    ;;
  *)
    echo "[E1] ERROR: USE_REF_ANSWER_SCORING must be 0/1 or true/false (got: $USE_REF_ANSWER_SCORING)" >&2
    exit 1
    ;;
esac

STE_ARGS=(
  --solver_token_entropy_tokens "$SOLVER_TOKEN_ENTROPY_TOKENS"
  --solver_token_entropy_window_size "$SOLVER_TOKEN_ENTROPY_WINDOW_SIZE"
  --solver_token_entropy_sigmoid_alpha "$SOLVER_TOKEN_ENTROPY_SIGMOID_ALPHA"
  --solver_token_entropy_sigmoid_beta "$SOLVER_TOKEN_ENTROPY_SIGMOID_BETA"
  --solver_token_entropy_aggregation "$SOLVER_TOKEN_ENTROPY_AGGREGATION"
  --proposer_ste_primary_weight "$PROPOSER_STE_PRIMARY_WEIGHT"
  --proposer_sample_entropy_weight "$PROPOSER_SAMPLE_ENTROPY_WEIGHT"
  --proposer_ste_reward_weight "$PROPOSER_STE_REWARD_WEIGHT"
)
if [[ "$SOLVER_TOKEN_ENTROPY_ENABLED" != "1" ]]; then
  STE_ARGS+=(--disable_solver_token_entropy)
fi

PPS_ARGS=()
if [[ "$SOLVER_PPS_ENABLED" != "1" ]]; then
  PPS_ARGS+=(--disable_solver_pps)
fi

LORA_ARGS=()
if [[ "$USE_LORA" == "1" ]]; then
  LORA_ARGS+=(
    --use_lora
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --lora_targets "$LORA_TARGETS"
  )
  if [[ "$SOLVER_MERGER_LORA" == "1" ]]; then
    LORA_ARGS+=(
      --solver_merger_lora
      --solver_merger_lora_r "$SOLVER_MERGER_LORA_R"
      --solver_merger_lora_alpha "$SOLVER_MERGER_LORA_ALPHA"
      --solver_merger_lora_lr "$SOLVER_MERGER_LORA_LR"
      --solver_merger_lora_targets "$SOLVER_MERGER_LORA_TARGETS"
    )
  fi
else
  LORA_ARGS+=(--no_lora)
fi

QLORA_ARGS=()
if [[ "$LOAD_IN_4BIT" == "1" ]]; then
  if [[ "$USE_LORA" != "1" ]]; then
    echo "[E1] ERROR: LOAD_IN_4BIT=1 is QLoRA mode and requires USE_LORA=1" >&2
    exit 1
  fi
  QLORA_ARGS+=(
    --load_in_4bit
    --bnb_4bit_quant_type "$BNB_4BIT_QUANT_TYPE"
    --bnb_4bit_compute_dtype "$BNB_4BIT_COMPUTE_DTYPE"
  )
  if [[ "$BNB_4BIT_USE_DOUBLE_QUANT" != "1" ]]; then
    QLORA_ARGS+=(--disable_bnb_4bit_use_double_quant)
  fi
fi

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
    --proposer_easy_reward_cap 0.20
    --proposer_easy_gotcha_reward_cap 0.45
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
    --proposer_easy_reward_cap 0.20
    --proposer_easy_gotcha_reward_cap 0.45
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
  echo "[E1] ERROR: TRAIN_STAGE must be one of: warmup, strict (got: $TRAIN_STAGE)" >&2
  exit 1
fi

case "$SOLVER_TOKEN_ENTROPY_AGGREGATION" in
  max|mean) ;;
  *)
    echo "[E1] ERROR: SOLVER_TOKEN_ENTROPY_AGGREGATION must be max or mean (got: $SOLVER_TOKEN_ENTROPY_AGGREGATION)" >&2
    exit 1
    ;;
esac

# ── Resume from checkpoint (optional) ───────────────────────────────────────
RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "[E1] Resuming from checkpoint: $RESUME_FROM"
  RESUME_ARGS=(--resume_from "$RESUME_FROM")
  if [[ "${RESET_PROPOSER_BASELINE:-0}" == "1" ]]; then
    echo "[E1] Resetting proposer baseline on resume."
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

count_csv_devices() {
  local value="$1"
  if [[ -z "$value" || "$value" == "NoDevFiles" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"$value"
}

detect_physical_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')"
    if [[ "$n" =~ ^[0-9]+$ && "$n" -gt 0 ]]; then
      echo "$n"
      return
    fi
  fi
  "$PYTHON_BIN" - <<'PY' 2>/dev/null || echo 0
try:
    import torch
    print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
}

make_device_list() {
  local n="$1"
  if [[ "$n" -le 0 ]]; then
    echo ""
    return
  fi
  local out="0"
  local i
  for ((i=1; i<n; i++)); do
    out="${out},${i}"
  done
  echo "$out"
}

if [[ -z "${HIP_VISIBLE_DEVICES:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
fi
if [[ -z "${HIP_VISIBLE_DEVICES:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  DETECTED_GPU_COUNT="$(detect_physical_gpus)"
  if [[ ! "$DETECTED_GPU_COUNT" =~ ^[0-9]+$ || "$DETECTED_GPU_COUNT" -le 0 ]]; then
    DETECTED_GPU_COUNT=1
  fi
  export HIP_VISIBLE_DEVICES="$(make_device_list "$DETECTED_GPU_COUNT")"
  export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
fi
VISIBLE_GPU_COUNT="$(count_csv_devices "${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-}}")"
if [[ "$VISIBLE_GPU_COUNT" -le 0 ]]; then
  VISIBLE_GPU_COUNT=1
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-$VISIBLE_GPU_COUNT}"
if [[ "$NPROC_PER_NODE" -gt "$VISIBLE_GPU_COUNT" ]]; then
  echo "[E1] ERROR: NPROC_PER_NODE=$NPROC_PER_NODE but only $VISIBLE_GPU_COUNT GPU(s) are visible." >&2
  echo "[E1] For a single H200 run, use: CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 ..." >&2
  exit 1
fi

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [[ ! -d "$DATA_DIR" ]]; then
  echo "[E1] ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
  exit 1
fi
IMAGE_COUNT="$(find "$DATA_DIR" -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.bmp" -o -iname "*.tiff" \) \
  | wc -l | tr -d '[:space:]')"
if [[ "$IMAGE_COUNT" -lt "$MIN_DATA_IMAGES" && "$ALLOW_SMALL_DATA" != "1" ]]; then
  echo "[E1] ERROR: DATA_DIR has $IMAGE_COUNT images; paper protocol requires at least $MIN_DATA_IMAGES." >&2
  echo "[E1] Set DATA_DIR to a local directory of unlabeled training images." >&2
  echo "[E1] For smoke tests only, set ALLOW_SMALL_DATA=1." >&2
  exit 1
fi

echo "[E1] Starting experiment E1 (Full Joint Training — Main Result)"
echo "[E1]   Stage:       $TRAIN_STAGE"
echo "[E1]   Run name:    $RUN_NAME"
echo "[E1]   Output dir:  $OUTPUT_DIR"
echo "[E1]   Data dir:    $DATA_DIR"
echo "[E1]   GPUs:        $NPROC_PER_NODE"
echo "[E1]   Port:        $MASTER_PORT"
echo "[E1]   Attn impl:   $ATTN_IMPL"
echo "[E1]   Total steps: $TOTAL_STEPS"
echo "[E1]   Logging:     log_every=$LOG_EVERY save_every=$SAVE_EVERY save_images_every=$SAVE_GENERATED_IMAGES_EVERY"
echo "[E1]   Cycle:       U=$UNDERSTANDING_STEPS_PER_CYCLE, G=$GENERATION_STEPS_PER_CYCLE"
echo "[E1]   Params:      use_lora=$USE_LORA r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT qlora4=$LOAD_IN_4BIT"
echo "[E1]   Samples:     PPS/solver=$NUM_SOLVER_SAMPLES, candidates K=$PROPOSER_NUM_CANDIDATES, generations L=$NUM_GENERATIONS"
echo "[E1]   STE/PPS:     enabled=$SOLVER_TOKEN_ENTROPY_ENABLED aggregation=$SOLVER_TOKEN_ENTROPY_AGGREGATION window=$SOLVER_TOKEN_ENTROPY_WINDOW_SIZE pps=$SOLVER_PPS_ENABLED"
echo "[E1]   Rewards:     qa=$REWARD_SPEC_WEIGHT cycle=$REWARD_CYCLE_WEIGHT diversity=$REWARD_DIVERSITY_WEIGHT contradiction=$REWARD_CONTRADICTION_WEIGHT"
echo "[E1]   Reward mode: $REWARD_MODE"
echo "[E1]   Gen freq:    $GENERATOR_UPDATE_FREQ, DiT enabled: $DIT_UPDATE_ENABLED (freq=$DIT_UPDATE_FREQ)"
echo "[E1]   Gen->U aux:  proposer_reward=$PROPOSER_GEN_REWARD_ENABLED, solver_update=$GEN_STEP_SOLVER_UPDATE_ENABLED"
if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "[E1]   Resume from: $RESUME_FROM"
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
  "${LORA_ARGS[@]}" \
  ${QLORA_ARGS[@]+"${QLORA_ARGS[@]}"} \
  \
  `# ── Optimiser (understanding-side GRPO) ────────────────────────────────` \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --grad_clip "$GRAD_CLIP" \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" \
  \
  `# ── Role update frequencies ─────────────────────────────────────────────` \
  --proposer_update_freq 1 \
  --generator_update_freq "$GENERATOR_UPDATE_FREQ" \
  --enable_solver_updates \
  --solver_update_freq 1 \
  \
  `# ── Generator token policy path; BLIP3o routes to DiT denoising ────────` \
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
  --num_solver_samples "$NUM_SOLVER_SAMPLES" \
  --num_solver_samples_spec "$NUM_SOLVER_SAMPLES_SPEC" \
  --num_generations "$NUM_GENERATIONS" \
  --proposer_num_candidates "$PROPOSER_NUM_CANDIDATES" \
  --proposer_spot_check_samples "$PROPOSER_SPOT_CHECK_SAMPLES" \
  --grpo_extra_sc_samples "$GRPO_EXTRA_SC_SAMPLES" \
  \
  `# ── Image generation (BLIP3o diffusion) ─────────────────────────────────` \
  --generation_num_inference_steps "$GENERATION_NUM_INFERENCE_STEPS" \
  --generation_guidance_scale "$GENERATION_GUIDANCE_SCALE" \
  --generation_height "$GENERATION_IMAGE_SIDE" \
  --generation_width  "$GENERATION_IMAGE_SIDE" \
  \
  `# ── Difficulty curriculum ───────────────────────────────────────────────` \
  --difficulty_sampler_enabled \
  \
  `# ── Reward weights ──────────────────────────────────────────────────────` \
  --reward_spec_weight "$REWARD_SPEC_WEIGHT" \
  --reward_cycle_weight "$REWARD_CYCLE_WEIGHT" \
  --reward_diversity_weight "$REWARD_DIVERSITY_WEIGHT" \
  --reward_contradiction_weight "$REWARD_CONTRADICTION_WEIGHT" \
  \
  `# ── Spec quality gates ──────────────────────────────────────────────────` \
  --min_spec_quality_for_update "$MIN_SPEC_QUALITY_FOR_UPDATE" \
  --min_spec_qa_pairs "$MIN_SPEC_QA_PAIRS" \
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
  --disable_solver_always_update_with_informative_scaling \
  --solver_skip_update_on_easy \
  --disable_solver_update_on_low_info_easy \
  --len_penalty_weight 0.10 \
  --len_penalty_target_words 6 \
  --solver_hardness_min_entropy 0.20 \
  --easy_update_majority_frac_threshold 0.85 \
  --entropy_iqr_filter_enabled \
  \
  `# ── Proposer entropy target ─────────────────────────────────────────────` \
  --prop_entropy_sigma 0.25 \
  \
  `# ── Cycle scheduling ────────────────────────────────────────────────────` \
  --understanding_steps_per_cycle "$UNDERSTANDING_STEPS_PER_CYCLE" \
  --generation_steps_per_cycle "$GENERATION_STEPS_PER_CYCLE" \
  --synthetic_solver_update_freq 0 \
  \
  `# ── KL regularisation ───────────────────────────────────────────────────` \
  --kl_coef "$KL_COEF" \
  --kl_target "$KL_TARGET" \
  --kl_adapt_rate "$KL_ADAPT_RATE" \
  --kl_min "$KL_MIN" \
  --kl_max "$KL_MAX" \
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
  "${REF_ANSWER_SCORING_ARGS[@]}" \
  \
  `# ── Unicorn reconstruction (disabled) ──────────────────────────────────` \
  --disable_unicorn_reconstruction_sft \
  --disable_unicorn_reconstruction_generator \
  \
  `# ── Replay buffer (disabled) ───────────────────────────────────────────` \
  --replay_buffer_size "$REPLAY_BUFFER_SIZE" \
  --replay_min_reward "$REPLAY_MIN_REWARD" \
  --replay_max_staleness "$REPLAY_MAX_STALENESS" \
  --gen_mix_source_mode "$GEN_MIX_SOURCE_MODE" \
  --gen_mix_ratio_start "$GEN_MIX_RATIO_START" \
  --gen_mix_ratio_max "$GEN_MIX_RATIO_MAX" \
  --gen_mix_ratio_warmup_steps "$GEN_MIX_RATIO_WARMUP_STEPS" \
  \
  `# ── DiT SFT + Joint Conditioning + RWR ─────────────────────────────────` \
  ${DIT_ARGS[@]+"${DIT_ARGS[@]}"} \
  --dit_update_freq "$DIT_UPDATE_FREQ" \
  --dit_lr 5e-7 \
  --dit_weight_decay 0.01 \
  --dit_grad_clip 1.0 \
  --dit_grad_accum_steps 1 \
  --dit_conditioning_dropout 0.10 \
  --dit_loss_weight 1.0 \
  --dit_prompt_suffix_token_id 151665 \
  --dit_joint_conditioning_train \
  --dit_joint_conditioning_lr 5e-7 \
  --dit_reward_loss_weight "$DIT_REWARD_LOSS_WEIGHT" \
  \
  `# ── Optional generation-phase Proposer/Solver ablations (off by default)` \
  ${PROPOSER_GEN_REWARD_ARGS[@]+"${PROPOSER_GEN_REWARD_ARGS[@]}"} \
  ${GEN_STEP_SOLVER_ARGS[@]+"${GEN_STEP_SOLVER_ARGS[@]}"} \
  "${STE_ARGS[@]}" \
  ${PPS_ARGS[@]+"${PPS_ARGS[@]}"} \
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
