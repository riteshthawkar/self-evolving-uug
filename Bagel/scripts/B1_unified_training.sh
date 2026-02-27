#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# B1 — BAGEL Self-Evolving Unified Training
# ══════════════════════════════════════════════════════════════════════════════
#
# Purpose:
#   Start BAGEL self-evolving unified training by default:
#   - understanding phase (proposer + solver)
#   - generation phase (SUDER spec + reward)
#   - LoRA policy updates (REINFORCE/GRPO signal)
#
#   Core loop:
#   - proposer question generation from image
#   - multi-sample solver self-consistency
#   - greedy intuitive solver pass
#   - dual-track proposer reward logging
#   - generation-side proposer spec reward and updates
#
# Usage:
#   MODEL_PATH=/path/to/BAGEL-7B-MoT \
#   DATA_DIR=/path/to/images \
#   bash Bagel/scripts/B1_unified_training.sh
#
# Optional:
#   TRAIN_STAGE=warmup|strict
#   RUN_MODE=train|rollout             # default: train (unified strategy)
#   EXPERIMENT=unified_self_evolving   # default: unified
#   STEPS=500
#   DEVICE=cuda
#   MULTI_GPU_SPLIT=auto|on|off        # default: auto (model/vae split)
#   MODEL_DEVICE_INDEX=0
#   VAE_DEVICE_INDEX=1
#   OUTPUT_DIR=/custom/output
#   ENABLE_SUDER=1                     # default: 1 in train mode
#   PROPOSER_GEN_ENTROPY_WEIGHT=0.7 # alpha in joint reward blend
#   POLICY_UPDATE_METHOD=reinforce|grpo
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGEL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BAGEL_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/BAGEL-7B-MoT}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/joint_3k/images}"
# Keep BAGEL artifacts under BAGEL folder by default.
OUTPUT_DIR="${OUTPUT_DIR:-$BAGEL_ROOT/scripts/runs/B1_unified_training}"
# Output layout:
#   direct    -> write logs/checkpoints directly under OUTPUT_DIR
#   timestamp -> create OUTPUT_DIR/unified_rollout_<ts> (legacy behavior)
OUTPUT_LAYOUT="${OUTPUT_LAYOUT:-direct}"   # direct|timestamp

TRAIN_STAGE="${TRAIN_STAGE:-strict}"
RUN_MODE="${RUN_MODE:-train}"
EXPERIMENT="${EXPERIMENT:-unified_self_evolving}"
STEPS="${STEPS:-500}"
DEVICE="${DEVICE:-cuda}"
VAE_DEVICE="${VAE_DEVICE:-}"
MULTI_GPU_SPLIT="${MULTI_GPU_SPLIT:-auto}"   # auto|on|off
MODEL_DEVICE_INDEX="${MODEL_DEVICE_INDEX:-0}"
VAE_DEVICE_INDEX="${VAE_DEVICE_INDEX:-1}"
MAX_LATENT_SIZE="${MAX_LATENT_SIZE:-64}"
ENABLE_SUDER="${ENABLE_SUDER:-1}"
PROPOSER_GEN_ENTROPY_WEIGHT="${PROPOSER_GEN_ENTROPY_WEIGHT:-0.7}"
PROPOSER_GEN_BASELINE_MOMENTUM="${PROPOSER_GEN_BASELINE_MOMENTUM:-0.6}"
GEN_SPEC_MIN_QA_PAIRS="${GEN_SPEC_MIN_QA_PAIRS:-2}"
GEN_SPEC_TEMPERATURE="${GEN_SPEC_TEMPERATURE:-0.9}"
MAX_NEW_TOKENS_GEN_SPEC="${MAX_NEW_TOKENS_GEN_SPEC:-384}"
GEN_IMAGE_SIZE="${GEN_IMAGE_SIZE:-640}"
GEN_NUM_TIMESTEPS="${GEN_NUM_TIMESTEPS:-50}"
SAVE_GENERATED_IMAGES="${SAVE_GENERATED_IMAGES:-0}"
ENABLE_LORA="${ENABLE_LORA:-0}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES_CSV="${LORA_TARGET_MODULES_CSV:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
LORA_ROLE_ADAPTERS_CSV="${LORA_ROLE_ADAPTERS_CSV:-proposer,solver,generator}"
LORA_DEFAULT_ADAPTER="${LORA_DEFAULT_ADAPTER:-proposer}"
POLICY_UPDATE_METHOD="${POLICY_UPDATE_METHOD:-grpo}"
POLICY_LR="${POLICY_LR:-2e-5}"
POLICY_WEIGHT_DECAY="${POLICY_WEIGHT_DECAY:-0.0}"
POLICY_MAX_GRAD_NORM="${POLICY_MAX_GRAD_NORM:-1.0}"
POLICY_GRAD_ACCUM_STEPS="${POLICY_GRAD_ACCUM_STEPS:-1}"
POLICY_REWARD_SCALE="${POLICY_REWARD_SCALE:-1.0}"
POLICY_MAX_VIT_EDGE="${POLICY_MAX_VIT_EDGE:-448}"
POLICY_MIN_VIT_EDGE="${POLICY_MIN_VIT_EDGE:-224}"
POLICY_OOM_MAX_RETRIES="${POLICY_OOM_MAX_RETRIES:-3}"
POLICY_OOM_EDGE_DECAY="${POLICY_OOM_EDGE_DECAY:-0.8}"
POLICY_MAX_COMPLETION_TOKENS="${POLICY_MAX_COMPLETION_TOKENS:-192}"
POLICY_TEXT_ONLY_FALLBACK="${POLICY_TEXT_ONLY_FALLBACK:-1}"
POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS="${POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS:-96}"
POLICY_TEXT_ONLY_MAX_RETRIES="${POLICY_TEXT_ONLY_MAX_RETRIES:-3}"
POLICY_TEXT_ONLY_MODE="${POLICY_TEXT_ONLY_MODE:-0}"
POLICY_MAX_PROMPT_TOKENS="${POLICY_MAX_PROMPT_TOKENS:-64}"
POLICY_MIN_COMPLETION_TOKENS="${POLICY_MIN_COMPLETION_TOKENS:-24}"
POLICY_ROCM_FORCE_TEXT_ONLY="${POLICY_ROCM_FORCE_TEXT_ONLY:-1}"
POLICY_EMPTY_CACHE_EACH_STEP="${POLICY_EMPTY_CACHE_EACH_STEP:-auto}"
POLICY_OOM_FORCE_TEXT_ONLY_STEPS="${POLICY_OOM_FORCE_TEXT_ONLY_STEPS:-64}"
POLICY_OOM_PAUSE_AFTER_CONSECUTIVE="${POLICY_OOM_PAUSE_AFTER_CONSECUTIVE:-6}"
POLICY_OOM_PAUSE_STEPS="${POLICY_OOM_PAUSE_STEPS:-32}"
SOLVER_POLICY_MAX_SAMPLES="${SOLVER_POLICY_MAX_SAMPLES:-0}"
GEN_SOLVER_POLICY_MAX_SAMPLES="${GEN_SOLVER_POLICY_MAX_SAMPLES:-0}"
PROPOSER_POLICY_MAX_CANDIDATES="${PROPOSER_POLICY_MAX_CANDIDATES:-0}"
BASELINE_MOMENTUM="${BASELINE_MOMENTUM:-0.6}"
SOLVER_REWARD_MIX_GAMMA="${SOLVER_REWARD_MIX_GAMMA:-0.7}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
RESUME_FROM="${RESUME_FROM:-}"
DISABLE_FLASH_ATTN="${DISABLE_FLASH_ATTN:-1}"
DISABLE_AUTOCAST="${DISABLE_AUTOCAST:-0}"
BAGEL_AUTOCAST_DTYPE="${BAGEL_AUTOCAST_DTYPE:-auto}"
ENABLE_ROCM_AUTOCAST="${ENABLE_ROCM_AUTOCAST:-0}"
TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-0}"
FORCE_MATH_SDPA="${FORCE_MATH_SDPA:-auto}"                 # auto|0|1
BAGEL_COMPILE_BLOCK_MASK="${BAGEL_COMPILE_BLOCK_MASK:-auto}" # auto|0|1
ROCM_SAFE_MODE="${ROCM_SAFE_MODE:-1}"                      # 1 => conservative defaults on AMD/ROCm
ROCM_SAFE_GEN_IMAGE_SIZE="${ROCM_SAFE_GEN_IMAGE_SIZE:-512}"
ROCM_SAFE_GEN_TIMESTEPS="${ROCM_SAFE_GEN_TIMESTEPS:-24}"
ROCM_SAFE_POLICY_MAX_VIT_EDGE="${ROCM_SAFE_POLICY_MAX_VIT_EDGE:-192}"
ROCM_SAFE_POLICY_MIN_VIT_EDGE="${ROCM_SAFE_POLICY_MIN_VIT_EDGE:-128}"
ROCM_SAFE_POLICY_MAX_COMPLETION_TOKENS="${ROCM_SAFE_POLICY_MAX_COMPLETION_TOKENS:-96}"
ROCM_SAFE_POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS="${ROCM_SAFE_POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS:-64}"
ROCM_SAFE_POLICY_MAX_PROMPT_TOKENS="${ROCM_SAFE_POLICY_MAX_PROMPT_TOKENS:-48}"
ROCM_SAFE_POLICY_MIN_COMPLETION_TOKENS="${ROCM_SAFE_POLICY_MIN_COMPLETION_TOKENS:-16}"
ROCM_SAFE_SOLVER_POLICY_MAX_SAMPLES="${ROCM_SAFE_SOLVER_POLICY_MAX_SAMPLES:-1}"
ROCM_SAFE_PROPOSER_POLICY_MAX_CANDIDATES="${ROCM_SAFE_PROPOSER_POLICY_MAX_CANDIDATES:-1}"
TRAIN_UNDERSTANDING_PROPOSER="${TRAIN_UNDERSTANDING_PROPOSER:-1}"
TRAIN_SOLVER="${TRAIN_SOLVER:-1}"
TRAIN_GENERATION_PROPOSER="${TRAIN_GENERATION_PROPOSER:-1}"
UNDERSTANDING_STEPS_PER_CYCLE="${UNDERSTANDING_STEPS_PER_CYCLE:-3}"
GENERATION_STEPS_PER_CYCLE="${GENERATION_STEPS_PER_CYCLE:-2}"
GEN_MIX_SOURCE_MODE="${GEN_MIX_SOURCE_MODE:-buffer}"
GEN_MIX_RATIO_START="${GEN_MIX_RATIO_START:-0.02}"
GEN_MIX_RATIO_MAX="${GEN_MIX_RATIO_MAX:-0.25}"
GEN_MIX_RATIO_WARMUP_STEPS="${GEN_MIX_RATIO_WARMUP_STEPS:-1000}"
REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-1000}"
REPLAY_MIN_REWARD="${REPLAY_MIN_REWARD:-0.5}"
REPLAY_MAX_STALENESS="${REPLAY_MAX_STALENESS:-500}"
GENERATED_MIX_DIR="${GENERATED_MIX_DIR:-}"
GENERATED_MIX_MIN_REWARD="${GENERATED_MIX_MIN_REWARD:-0.5}"
GENERATED_MIX_MAX_FILES="${GENERATED_MIX_MAX_FILES:-5000}"
GENERATED_MIX_REFRESH_EVERY="${GENERATED_MIX_REFRESH_EVERY:-10}"
UNDERSTANDING_GENERATED_ONLY="${UNDERSTANDING_GENERATED_ONLY:-0}"
PROPOSER_NUM_CANDIDATES="${PROPOSER_NUM_CANDIDATES:-3}"
PROPOSER_SPOT_CHECK_SAMPLES="${PROPOSER_SPOT_CHECK_SAMPLES:-3}"
PROPOSER_SPOT_ENTROPY_MIN_GATE="${PROPOSER_SPOT_ENTROPY_MIN_GATE:-0.05}"
PROPOSER_GRPO_GEN_GROUP_SIZE="${PROPOSER_GRPO_GEN_GROUP_SIZE:-3}"
SCORE_GRPO_EXTRAS="${SCORE_GRPO_EXTRAS:-1}"
GRPO_EXTRA_TEMP_MULTIPLIER="${GRPO_EXTRA_TEMP_MULTIPLIER:-1.5}"
SOLVER_TOKEN_ENTROPY_ENABLED="${SOLVER_TOKEN_ENTROPY_ENABLED:-1}"
GRPO_EXTRA_SC_SAMPLES="${GRPO_EXTRA_SC_SAMPLES:-3}"
PROPOSER_CERTIFICATE_ENABLED="${PROPOSER_CERTIFICATE_ENABLED:-1}"
PROPOSER_CERTIFICATE_MIN_SCORE="${PROPOSER_CERTIFICATE_MIN_SCORE:-0.60}"
PROPOSER_CERTIFICATE_WEIGHT="${PROPOSER_CERTIFICATE_WEIGHT:-0.75}"
PROPOSER_CERTIFICATE_STRICT_STRUCT="${PROPOSER_CERTIFICATE_STRICT_STRUCT:-1}"
PROPOSER_WARM_START_ENABLED="${PROPOSER_WARM_START_ENABLED:-1}"
PROPOSER_WARM_START_MAX_STEPS="${PROPOSER_WARM_START_MAX_STEPS:-30}"
PROPOSER_WARM_START_EXIT_WINDOW="${PROPOSER_WARM_START_EXIT_WINDOW:-5}"
PROPOSER_WARM_START_EXIT_CONSECUTIVE="${PROPOSER_WARM_START_EXIT_CONSECUTIVE:-2}"
PROPOSER_WARM_START_ENTROPY_EXIT_THRESHOLD="${PROPOSER_WARM_START_ENTROPY_EXIT_THRESHOLD:-0.10}"
PROPOSER_WARM_START_EASY_REJECT_PENALTY_SCALE="${PROPOSER_WARM_START_EASY_REJECT_PENALTY_SCALE:-0.0}"
PROPOSER_WARM_START_CERTIFICATE_WEIGHT="${PROPOSER_WARM_START_CERTIFICATE_WEIGHT:-0.50}"
HARDNESS_DEBT_ENABLED="${HARDNESS_DEBT_ENABLED:-1}"
HARDNESS_DEBT_INC_EASY="${HARDNESS_DEBT_INC_EASY:-1.50}"
HARDNESS_DEBT_DEC_NON_EASY="${HARDNESS_DEBT_DEC_NON_EASY:-1.00}"
HARDNESS_DEBT_MAX="${HARDNESS_DEBT_MAX:-6.0}"
HARDNESS_DEBT_HARD_RECOVERY_THRESHOLD="${HARDNESS_DEBT_HARD_RECOVERY_THRESHOLD:-3.0}"
HARDNESS_DEBT_RECOVERY_EASY_WEIGHT="${HARDNESS_DEBT_RECOVERY_EASY_WEIGHT:-0.0}"
HARDNESS_DEBT_RECOVERY_MEDIUM_WEIGHT="${HARDNESS_DEBT_RECOVERY_MEDIUM_WEIGHT:-0.30}"
HARDNESS_DEBT_RECOVERY_HARD_WEIGHT="${HARDNESS_DEBT_RECOVERY_HARD_WEIGHT:-0.70}"
HARDNESS_DEBT_STALE_STEPS="${HARDNESS_DEBT_STALE_STEPS:-8}"
HARDNESS_DEBT_STALE_RESET_TO="${HARDNESS_DEBT_STALE_RESET_TO:-3.0}"
HARDNESS_DEBT_STALE_ESCAPE_STEPS="${HARDNESS_DEBT_STALE_ESCAPE_STEPS:-8}"
HARDNESS_DEBT_STALE_EASY_WEIGHT="${HARDNESS_DEBT_STALE_EASY_WEIGHT:-0.05}"
HARDNESS_DEBT_STALE_MEDIUM_WEIGHT="${HARDNESS_DEBT_STALE_MEDIUM_WEIGHT:-0.55}"
HARDNESS_DEBT_STALE_HARD_WEIGHT="${HARDNESS_DEBT_STALE_HARD_WEIGHT:-0.40}"
HARDNESS_DEBT_TEMP_BOOST_MAX="${HARDNESS_DEBT_TEMP_BOOST_MAX:-0.30}"
HARDNESS_DEBT_PENALTY_BOOST_MAX="${HARDNESS_DEBT_PENALTY_BOOST_MAX:-0.30}"
DIFFICULTY_SAMPLER_ENABLED="${DIFFICULTY_SAMPLER_ENABLED:-1}"
DIFFICULTY_SAMPLER_WINDOW_SIZE="${DIFFICULTY_SAMPLER_WINDOW_SIZE:-256}"
DIFFICULTY_SAMPLER_MIN_SAMPLES="${DIFFICULTY_SAMPLER_MIN_SAMPLES:-8}"
DIFFICULTY_TARGET_EASY="${DIFFICULTY_TARGET_EASY:-0.00}"
DIFFICULTY_TARGET_MEDIUM="${DIFFICULTY_TARGET_MEDIUM:-0.60}"
DIFFICULTY_TARGET_HARD="${DIFFICULTY_TARGET_HARD:-0.40}"
DIFFICULTY_HARD_MIN_ENTROPY="${DIFFICULTY_HARD_MIN_ENTROPY:-0.90}"
DIFFICULTY_HARD_MAX_MARGIN="${DIFFICULTY_HARD_MAX_MARGIN:-0.35}"
ENTROPY_IQR_FILTER_ENABLED="${ENTROPY_IQR_FILTER_ENABLED:-1}"
ENTROPY_IQR_WINDOW_SIZE="${ENTROPY_IQR_WINDOW_SIZE:-256}"
ENTROPY_IQR_MIN_SAMPLES="${ENTROPY_IQR_MIN_SAMPLES:-32}"
ENTROPY_IQR_EASY_QUANTILE="${ENTROPY_IQR_EASY_QUANTILE:-0.25}"
ENTROPY_IQR_EASY_IQR_COEF="${ENTROPY_IQR_EASY_IQR_COEF:-0.25}"
ENTROPY_IQR_MIN_THRESHOLD="${ENTROPY_IQR_MIN_THRESHOLD:-0.10}"
ENTROPY_IQR_MAX_THRESHOLD="${ENTROPY_IQR_MAX_THRESHOLD:-1.2}"
ENTROPY_IQR_FILTER_MIN_MAJORITY_FRAC="${ENTROPY_IQR_FILTER_MIN_MAJORITY_FRAC:-0.80}"
ALL_EASY_EXPLORE_TRIGGER="${ALL_EASY_EXPLORE_TRIGGER:-2}"
ALL_EASY_EXPLORE_STEPS="${ALL_EASY_EXPLORE_STEPS:-16}"
ALL_EASY_EXPLORE_NUM_CANDIDATES="${ALL_EASY_EXPLORE_NUM_CANDIDATES:-6}"
ALL_EASY_EXPLORE_TEMP_BOOST="${ALL_EASY_EXPLORE_TEMP_BOOST:-1.20}"
ALL_EASY_EXPLORE_TOP_P_BOOST="${ALL_EASY_EXPLORE_TOP_P_BOOST:-0.20}"
ALL_EASY_EXPLORE_PENALTY_BOOST="${ALL_EASY_EXPLORE_PENALTY_BOOST:-0.70}"
PROPOSER_CONTRASTIVE_REPLAY_ENABLED="${PROPOSER_CONTRASTIVE_REPLAY_ENABLED:-1}"
PROPOSER_CONTRASTIVE_REPLAY_SIZE="${PROPOSER_CONTRASTIVE_REPLAY_SIZE:-256}"
PROPOSER_CONTRASTIVE_POS_BONUS="${PROPOSER_CONTRASTIVE_POS_BONUS:-0.08}"
PROPOSER_CONTRASTIVE_NEG_PENALTY="${PROPOSER_CONTRASTIVE_NEG_PENALTY:-0.08}"
PROPOSER_EARLY_FAILFAST_ENABLED="${PROPOSER_EARLY_FAILFAST_ENABLED:-1}"
PROPOSER_EARLY_FAILFAST_STOP="${PROPOSER_EARLY_FAILFAST_STOP:-0}"
PROPOSER_EARLY_FAILFAST_RECOVER="${PROPOSER_EARLY_FAILFAST_RECOVER:-1}"
PROPOSER_EARLY_FAILFAST_RECOVER_STEPS="${PROPOSER_EARLY_FAILFAST_RECOVER_STEPS:-20}"
PROPOSER_EARLY_STAGE1_U_STEP="${PROPOSER_EARLY_STAGE1_U_STEP:-12}"
PROPOSER_EARLY_STAGE2_U_STEP="${PROPOSER_EARLY_STAGE2_U_STEP:-24}"
PROPOSER_EARLY_HARD_STOP_MIN_U_STEP="${PROPOSER_EARLY_HARD_STOP_MIN_U_STEP:-80}"
PROPOSER_EARLY_CANDIDATE_NON_EASY_RATE_MIN="${PROPOSER_EARLY_CANDIDATE_NON_EASY_RATE_MIN:-0.08}"
PROPOSER_EARLY_ALL_EASY_RATE_MAX="${PROPOSER_EARLY_ALL_EASY_RATE_MAX:-0.93}"
PROPOSER_EARLY_REWARD_CLIPPED_RATE_MAX="${PROPOSER_EARLY_REWARD_CLIPPED_RATE_MAX:-0.85}"
PROPOSER_EARLY_SELECTED_NON_EASY_RATE_MIN="${PROPOSER_EARLY_SELECTED_NON_EASY_RATE_MIN:-0.10}"
PROPOSER_EARLY_SOLVER_UPDATES_MIN="${PROPOSER_EARLY_SOLVER_UPDATES_MIN:-1}"
PROPOSER_EARLY_MAX_COLLAPSE_STREAK="${PROPOSER_EARLY_MAX_COLLAPSE_STREAK:-3}"
GRPO_DEGENERATE_NOISE_ENABLED="${GRPO_DEGENERATE_NOISE_ENABLED:-1}"
GRPO_DEGENERATE_NOISE_SIGMA="${GRPO_DEGENERATE_NOISE_SIGMA:-0.03}"
GRPO_DEGENERATE_NOISE_STD_THRESHOLD="${GRPO_DEGENERATE_NOISE_STD_THRESHOLD:-1e-6}"
GRPO_PAIRWISE_RANKING_ENABLED="${GRPO_PAIRWISE_RANKING_ENABLED:-1}"
GRPO_PAIRWISE_RANKING_WEIGHT="${GRPO_PAIRWISE_RANKING_WEIGHT:-0.15}"
GRPO_PAIRWISE_MARGIN="${GRPO_PAIRWISE_MARGIN:-0.10}"
GRPO_PAIRWISE_EASY_PENALTY="${GRPO_PAIRWISE_EASY_PENALTY:-0.12}"
PROPOSER_ALL_EASY_RANK_SPREAD="${PROPOSER_ALL_EASY_RANK_SPREAD:-0.08}"
GEN_STEP_SOLVER_UPDATE_ENABLED="${GEN_STEP_SOLVER_UPDATE_ENABLED:-1}"

# ── Stage-specific hyperparameters ──────────────────────────────────────────
if [[ "$TRAIN_STAGE" == "warmup" ]]; then
  STAGE_ARGS=(
    --proposer_require_objective
    --disable_acceptance_require_non_easy
    --proposer_non_objective_penalty 0.20
    --rejected_question_penalty 0.25
    --proposer_entropy_mu 0.90
    --proposer_entropy_sigma 0.30
    --proposer_temperature 0.90
    --num_solver_samples 5
    --solver_temp_min 0.70
    --solver_temp_max 2.00
  )
elif [[ "$TRAIN_STAGE" == "strict" ]]; then
  STAGE_ARGS=(
    --proposer_require_objective
    --acceptance_require_non_easy
    --proposer_non_objective_penalty 0.20
    --rejected_question_penalty 0.35
    --proposer_entropy_mu 0.90
    --proposer_entropy_sigma 0.25
    --proposer_temperature 1.00
    --num_solver_samples 7
    --solver_temp_min 0.50
    --solver_temp_max 2.50
  )
else
  echo "[B1] ERROR: TRAIN_STAGE must be one of: warmup, strict (got: $TRAIN_STAGE)" >&2
  exit 1
fi

if [[ "$RUN_MODE" != "rollout" && "$RUN_MODE" != "train" ]]; then
  echo "[B1] ERROR: RUN_MODE must be one of: rollout, train (got: $RUN_MODE)" >&2
  exit 1
fi

if [[ "$MULTI_GPU_SPLIT" != "auto" && "$MULTI_GPU_SPLIT" != "on" && "$MULTI_GPU_SPLIT" != "off" ]]; then
  echo "[B1] ERROR: MULTI_GPU_SPLIT must be one of: auto, on, off (got: $MULTI_GPU_SPLIT)" >&2
  exit 1
fi

if [[ "$EXPERIMENT" != "understanding_self_evolving" && "$EXPERIMENT" != "generation_self_evolving" && "$EXPERIMENT" != "unified_self_evolving" ]]; then
  echo "[B1] ERROR: EXPERIMENT must be one of: understanding_self_evolving, generation_self_evolving, unified_self_evolving (got: $EXPERIMENT)" >&2
  exit 1
fi

if [[ "$RUN_MODE" == "train" ]]; then
  ENABLE_LORA=1
  ENABLE_SUDER=1
  TRAIN_UNDERSTANDING_PROPOSER=1
  TRAIN_SOLVER=1
  TRAIN_GENERATION_PROPOSER=1
fi

# ── Multi-GPU split config (supported path: model + VAE on different GPUs) ──
GPU_COUNT="$("$PYTHON_BIN" - <<'PY'
try:
    import torch
    print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
)"
ROCM_RUNTIME="$("$PYTHON_BIN" - <<'PY'
try:
    import torch
    print(1 if getattr(torch.version, "hip", None) else 0)
except Exception:
    print(0)
PY
)"

if [[ "$ROCM_RUNTIME" == "1" ]]; then
  if [[ "$ROCM_SAFE_MODE" == "1" ]]; then
    if [[ "$MULTI_GPU_SPLIT" != "off" ]]; then
      echo "[B1] ROCm safe mode: forcing MULTI_GPU_SPLIT=off."
      MULTI_GPU_SPLIT="off"
    fi
    if [[ -n "$VAE_DEVICE" ]]; then
      echo "[B1] ROCm safe mode: ignoring explicit VAE_DEVICE='$VAE_DEVICE' and using model device."
      VAE_DEVICE=""
    fi
    FORCE_MATH_SDPA="1"
    BAGEL_COMPILE_BLOCK_MASK="0"
    DISABLE_FLASH_ATTN="1"
    DISABLE_AUTOCAST="1"

    if [[ "$GEN_IMAGE_SIZE" -gt "$ROCM_SAFE_GEN_IMAGE_SIZE" ]]; then
      echo "[B1] ROCm safe mode: capping GEN_IMAGE_SIZE $GEN_IMAGE_SIZE -> $ROCM_SAFE_GEN_IMAGE_SIZE."
      GEN_IMAGE_SIZE="$ROCM_SAFE_GEN_IMAGE_SIZE"
    fi
    if [[ "$GEN_NUM_TIMESTEPS" -gt "$ROCM_SAFE_GEN_TIMESTEPS" ]]; then
      echo "[B1] ROCm safe mode: capping GEN_NUM_TIMESTEPS $GEN_NUM_TIMESTEPS -> $ROCM_SAFE_GEN_TIMESTEPS."
      GEN_NUM_TIMESTEPS="$ROCM_SAFE_GEN_TIMESTEPS"
    fi

    if [[ "$POLICY_MAX_VIT_EDGE" -gt "$ROCM_SAFE_POLICY_MAX_VIT_EDGE" ]]; then
      echo "[B1] ROCm safe mode: capping POLICY_MAX_VIT_EDGE $POLICY_MAX_VIT_EDGE -> $ROCM_SAFE_POLICY_MAX_VIT_EDGE."
      POLICY_MAX_VIT_EDGE="$ROCM_SAFE_POLICY_MAX_VIT_EDGE"
    fi
    if [[ "$POLICY_MIN_VIT_EDGE" -gt "$ROCM_SAFE_POLICY_MIN_VIT_EDGE" ]]; then
      echo "[B1] ROCm safe mode: capping POLICY_MIN_VIT_EDGE $POLICY_MIN_VIT_EDGE -> $ROCM_SAFE_POLICY_MIN_VIT_EDGE."
      POLICY_MIN_VIT_EDGE="$ROCM_SAFE_POLICY_MIN_VIT_EDGE"
    fi
    if [[ "$POLICY_MIN_VIT_EDGE" -gt "$POLICY_MAX_VIT_EDGE" ]]; then
      POLICY_MIN_VIT_EDGE="$POLICY_MAX_VIT_EDGE"
    fi
    if [[ "$POLICY_MAX_COMPLETION_TOKENS" -gt "$ROCM_SAFE_POLICY_MAX_COMPLETION_TOKENS" ]]; then
      echo "[B1] ROCm safe mode: capping POLICY_MAX_COMPLETION_TOKENS $POLICY_MAX_COMPLETION_TOKENS -> $ROCM_SAFE_POLICY_MAX_COMPLETION_TOKENS."
      POLICY_MAX_COMPLETION_TOKENS="$ROCM_SAFE_POLICY_MAX_COMPLETION_TOKENS"
    fi
    if [[ "$POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS" -gt "$ROCM_SAFE_POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS" ]]; then
      echo "[B1] ROCm safe mode: capping POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS $POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS -> $ROCM_SAFE_POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS."
      POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS="$ROCM_SAFE_POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS"
    fi
    if [[ "$POLICY_MAX_PROMPT_TOKENS" -gt "$ROCM_SAFE_POLICY_MAX_PROMPT_TOKENS" ]]; then
      echo "[B1] ROCm safe mode: capping POLICY_MAX_PROMPT_TOKENS $POLICY_MAX_PROMPT_TOKENS -> $ROCM_SAFE_POLICY_MAX_PROMPT_TOKENS."
      POLICY_MAX_PROMPT_TOKENS="$ROCM_SAFE_POLICY_MAX_PROMPT_TOKENS"
    fi
    if [[ "$POLICY_MIN_COMPLETION_TOKENS" -gt "$ROCM_SAFE_POLICY_MIN_COMPLETION_TOKENS" ]]; then
      echo "[B1] ROCm safe mode: capping POLICY_MIN_COMPLETION_TOKENS $POLICY_MIN_COMPLETION_TOKENS -> $ROCM_SAFE_POLICY_MIN_COMPLETION_TOKENS."
      POLICY_MIN_COMPLETION_TOKENS="$ROCM_SAFE_POLICY_MIN_COMPLETION_TOKENS"
    fi
    POLICY_TEXT_ONLY_MODE=1
    POLICY_ROCM_FORCE_TEXT_ONLY=1
    POLICY_TEXT_ONLY_MAX_RETRIES=2
    POLICY_EMPTY_CACHE_EACH_STEP=1
    if [[ "$POLICY_OOM_FORCE_TEXT_ONLY_STEPS" -lt 128 ]]; then
      POLICY_OOM_FORCE_TEXT_ONLY_STEPS=128
    fi
    if [[ "$POLICY_OOM_PAUSE_AFTER_CONSECUTIVE" -lt 3 ]]; then
      POLICY_OOM_PAUSE_AFTER_CONSECUTIVE=3
    fi
    if [[ "$POLICY_OOM_PAUSE_STEPS" -lt 64 ]]; then
      POLICY_OOM_PAUSE_STEPS=64
    fi
    if [[ "$SOLVER_POLICY_MAX_SAMPLES" -eq 0 || "$SOLVER_POLICY_MAX_SAMPLES" -gt "$ROCM_SAFE_SOLVER_POLICY_MAX_SAMPLES" ]]; then
      SOLVER_POLICY_MAX_SAMPLES="$ROCM_SAFE_SOLVER_POLICY_MAX_SAMPLES"
    fi
    if [[ "$GEN_SOLVER_POLICY_MAX_SAMPLES" -eq 0 || "$GEN_SOLVER_POLICY_MAX_SAMPLES" -gt "$ROCM_SAFE_SOLVER_POLICY_MAX_SAMPLES" ]]; then
      GEN_SOLVER_POLICY_MAX_SAMPLES="$ROCM_SAFE_SOLVER_POLICY_MAX_SAMPLES"
    fi
    if [[ "$PROPOSER_POLICY_MAX_CANDIDATES" -eq 0 || "$PROPOSER_POLICY_MAX_CANDIDATES" -gt "$ROCM_SAFE_PROPOSER_POLICY_MAX_CANDIDATES" ]]; then
      PROPOSER_POLICY_MAX_CANDIDATES="$ROCM_SAFE_PROPOSER_POLICY_MAX_CANDIDATES"
    fi
    if [[ "$PROPOSER_GRPO_GEN_GROUP_SIZE" -gt 1 ]]; then
      PROPOSER_GRPO_GEN_GROUP_SIZE=1
    fi
    SCORE_GRPO_EXTRAS=0
  else
    if [[ "$MULTI_GPU_SPLIT" == "auto" ]]; then
      MULTI_GPU_SPLIT="off"
      echo "[B1] ROCm detected: forcing MULTI_GPU_SPLIT=off in auto mode for stability."
    fi
    if [[ "$FORCE_MATH_SDPA" == "auto" ]]; then
      FORCE_MATH_SDPA="1"
    fi
    if [[ "$BAGEL_COMPILE_BLOCK_MASK" == "auto" ]]; then
      BAGEL_COMPILE_BLOCK_MASK="0"
    fi
  fi
fi

if [[ "$DEVICE" == cuda* ]]; then
  if [[ -z "$VAE_DEVICE" ]]; then
    if [[ "$MULTI_GPU_SPLIT" == "on" ]]; then
      if [[ "$GPU_COUNT" -lt 2 ]]; then
        echo "[B1] ERROR: MULTI_GPU_SPLIT=on requires >=2 GPUs, found $GPU_COUNT" >&2
        exit 1
      fi
      DEVICE="cuda:${MODEL_DEVICE_INDEX}"
      VAE_DEVICE="cuda:${VAE_DEVICE_INDEX}"
    elif [[ "$MULTI_GPU_SPLIT" == "auto" && "$GPU_COUNT" -ge 2 && "$DEVICE" == "cuda" ]]; then
      DEVICE="cuda:${MODEL_DEVICE_INDEX}"
      VAE_DEVICE="cuda:${VAE_DEVICE_INDEX}"
    fi
  fi
fi

if [[ "$POLICY_EMPTY_CACHE_EACH_STEP" == "auto" ]]; then
  if [[ "$ROCM_RUNTIME" == "1" ]]; then
    POLICY_EMPTY_CACHE_EACH_STEP=1
  else
    POLICY_EMPTY_CACHE_EACH_STEP=0
  fi
fi
if [[ "$GEN_SOLVER_POLICY_MAX_SAMPLES" -eq 0 && "$SOLVER_POLICY_MAX_SAMPLES" -gt 0 ]]; then
  GEN_SOLVER_POLICY_MAX_SAMPLES="$SOLVER_POLICY_MAX_SAMPLES"
fi

# ── Shared arguments ────────────────────────────────────────────────────────
SHARED_ARGS=(
  --experiment "$EXPERIMENT"
  --max_new_tokens_proposer 256
  --max_new_tokens_solver 96
  --solver_unsolvable_maj_threshold 0.20
  --zero_entropy_eps 1e-6
  --seed 42
  --log_every 10
  --save_raw_generations
  --understanding_steps_per_cycle "$UNDERSTANDING_STEPS_PER_CYCLE"
  --generation_steps_per_cycle "$GENERATION_STEPS_PER_CYCLE"
  --gen_mix_source_mode "$GEN_MIX_SOURCE_MODE"
  --gen_mix_ratio_start "$GEN_MIX_RATIO_START"
  --gen_mix_ratio_max "$GEN_MIX_RATIO_MAX"
  --gen_mix_ratio_warmup_steps "$GEN_MIX_RATIO_WARMUP_STEPS"
  --replay_buffer_size "$REPLAY_BUFFER_SIZE"
  --replay_min_reward "$REPLAY_MIN_REWARD"
  --replay_max_staleness "$REPLAY_MAX_STALENESS"
  --generated_mix_min_reward "$GENERATED_MIX_MIN_REWARD"
  --generated_mix_max_files "$GENERATED_MIX_MAX_FILES"
  --generated_mix_refresh_every "$GENERATED_MIX_REFRESH_EVERY"
  --proposer_num_candidates "$PROPOSER_NUM_CANDIDATES"
  --proposer_spot_check_samples "$PROPOSER_SPOT_CHECK_SAMPLES"
  --proposer_spot_entropy_min_gate "$PROPOSER_SPOT_ENTROPY_MIN_GATE"
  --proposer_grpo_gen_group_size "$PROPOSER_GRPO_GEN_GROUP_SIZE"
  --grpo_extra_temp_multiplier "$GRPO_EXTRA_TEMP_MULTIPLIER"
  --grpo_extra_sc_samples "$GRPO_EXTRA_SC_SAMPLES"
  --proposer_certificate_min_score "$PROPOSER_CERTIFICATE_MIN_SCORE"
  --proposer_certificate_weight "$PROPOSER_CERTIFICATE_WEIGHT"
  --proposer_warm_start_max_steps "$PROPOSER_WARM_START_MAX_STEPS"
  --proposer_warm_start_exit_window "$PROPOSER_WARM_START_EXIT_WINDOW"
  --proposer_warm_start_exit_consecutive "$PROPOSER_WARM_START_EXIT_CONSECUTIVE"
  --proposer_warm_start_entropy_exit_threshold "$PROPOSER_WARM_START_ENTROPY_EXIT_THRESHOLD"
  --proposer_warm_start_easy_reject_penalty_scale "$PROPOSER_WARM_START_EASY_REJECT_PENALTY_SCALE"
  --proposer_warm_start_certificate_weight "$PROPOSER_WARM_START_CERTIFICATE_WEIGHT"
  --hardness_debt_inc_easy "$HARDNESS_DEBT_INC_EASY"
  --hardness_debt_dec_non_easy "$HARDNESS_DEBT_DEC_NON_EASY"
  --hardness_debt_max "$HARDNESS_DEBT_MAX"
  --hardness_debt_hard_recovery_threshold "$HARDNESS_DEBT_HARD_RECOVERY_THRESHOLD"
  --hardness_debt_recovery_easy_weight "$HARDNESS_DEBT_RECOVERY_EASY_WEIGHT"
  --hardness_debt_recovery_medium_weight "$HARDNESS_DEBT_RECOVERY_MEDIUM_WEIGHT"
  --hardness_debt_recovery_hard_weight "$HARDNESS_DEBT_RECOVERY_HARD_WEIGHT"
  --hardness_debt_stale_steps "$HARDNESS_DEBT_STALE_STEPS"
  --hardness_debt_stale_reset_to "$HARDNESS_DEBT_STALE_RESET_TO"
  --hardness_debt_stale_escape_steps "$HARDNESS_DEBT_STALE_ESCAPE_STEPS"
  --hardness_debt_stale_easy_weight "$HARDNESS_DEBT_STALE_EASY_WEIGHT"
  --hardness_debt_stale_medium_weight "$HARDNESS_DEBT_STALE_MEDIUM_WEIGHT"
  --hardness_debt_stale_hard_weight "$HARDNESS_DEBT_STALE_HARD_WEIGHT"
  --hardness_debt_temp_boost_max "$HARDNESS_DEBT_TEMP_BOOST_MAX"
  --hardness_debt_penalty_boost_max "$HARDNESS_DEBT_PENALTY_BOOST_MAX"
  --difficulty_sampler_window_size "$DIFFICULTY_SAMPLER_WINDOW_SIZE"
  --difficulty_sampler_min_samples "$DIFFICULTY_SAMPLER_MIN_SAMPLES"
  --difficulty_target_easy "$DIFFICULTY_TARGET_EASY"
  --difficulty_target_medium "$DIFFICULTY_TARGET_MEDIUM"
  --difficulty_target_hard "$DIFFICULTY_TARGET_HARD"
  --difficulty_hard_min_entropy "$DIFFICULTY_HARD_MIN_ENTROPY"
  --difficulty_hard_max_margin "$DIFFICULTY_HARD_MAX_MARGIN"
  --entropy_iqr_window_size "$ENTROPY_IQR_WINDOW_SIZE"
  --entropy_iqr_min_samples "$ENTROPY_IQR_MIN_SAMPLES"
  --entropy_iqr_easy_quantile "$ENTROPY_IQR_EASY_QUANTILE"
  --entropy_iqr_easy_iqr_coef "$ENTROPY_IQR_EASY_IQR_COEF"
  --entropy_iqr_min_threshold "$ENTROPY_IQR_MIN_THRESHOLD"
  --entropy_iqr_max_threshold "$ENTROPY_IQR_MAX_THRESHOLD"
  --entropy_iqr_filter_min_majority_frac "$ENTROPY_IQR_FILTER_MIN_MAJORITY_FRAC"
  --all_easy_explore_trigger "$ALL_EASY_EXPLORE_TRIGGER"
  --all_easy_explore_steps "$ALL_EASY_EXPLORE_STEPS"
  --all_easy_explore_num_candidates "$ALL_EASY_EXPLORE_NUM_CANDIDATES"
  --all_easy_explore_temp_boost "$ALL_EASY_EXPLORE_TEMP_BOOST"
  --all_easy_explore_top_p_boost "$ALL_EASY_EXPLORE_TOP_P_BOOST"
  --all_easy_explore_penalty_boost "$ALL_EASY_EXPLORE_PENALTY_BOOST"
  --proposer_contrastive_replay_size "$PROPOSER_CONTRASTIVE_REPLAY_SIZE"
  --proposer_contrastive_pos_bonus "$PROPOSER_CONTRASTIVE_POS_BONUS"
  --proposer_contrastive_neg_penalty "$PROPOSER_CONTRASTIVE_NEG_PENALTY"
  --proposer_early_failfast_recover_steps "$PROPOSER_EARLY_FAILFAST_RECOVER_STEPS"
  --proposer_early_stage1_u_step "$PROPOSER_EARLY_STAGE1_U_STEP"
  --proposer_early_stage2_u_step "$PROPOSER_EARLY_STAGE2_U_STEP"
  --proposer_early_hard_stop_min_u_step "$PROPOSER_EARLY_HARD_STOP_MIN_U_STEP"
  --proposer_early_candidate_non_easy_rate_min "$PROPOSER_EARLY_CANDIDATE_NON_EASY_RATE_MIN"
  --proposer_early_all_easy_rate_max "$PROPOSER_EARLY_ALL_EASY_RATE_MAX"
  --proposer_early_reward_clipped_rate_max "$PROPOSER_EARLY_REWARD_CLIPPED_RATE_MAX"
  --proposer_early_selected_non_easy_rate_min "$PROPOSER_EARLY_SELECTED_NON_EASY_RATE_MIN"
  --proposer_early_solver_updates_min "$PROPOSER_EARLY_SOLVER_UPDATES_MIN"
  --proposer_early_max_collapse_streak "$PROPOSER_EARLY_MAX_COLLAPSE_STREAK"
  --grpo_degenerate_noise_sigma "$GRPO_DEGENERATE_NOISE_SIGMA"
  --grpo_degenerate_noise_std_threshold "$GRPO_DEGENERATE_NOISE_STD_THRESHOLD"
  --grpo_pairwise_ranking_weight "$GRPO_PAIRWISE_RANKING_WEIGHT"
  --grpo_pairwise_margin "$GRPO_PAIRWISE_MARGIN"
  --grpo_pairwise_easy_penalty "$GRPO_PAIRWISE_EASY_PENALTY"
  --proposer_all_easy_rank_spread "$PROPOSER_ALL_EASY_RANK_SPREAD"
)

if [[ -n "$GENERATED_MIX_DIR" ]]; then
  SHARED_ARGS+=(--generated_mix_dir "$GENERATED_MIX_DIR")
fi
if [[ "$UNDERSTANDING_GENERATED_ONLY" == "1" ]]; then
  SHARED_ARGS+=(--understanding_generated_only)
else
  SHARED_ARGS+=(--disable_understanding_generated_only)
fi
if [[ "$SCORE_GRPO_EXTRAS" == "1" ]]; then
  SHARED_ARGS+=(--score_grpo_extras)
else
  SHARED_ARGS+=(--disable_score_grpo_extras)
fi
if [[ "$SOLVER_TOKEN_ENTROPY_ENABLED" == "1" ]]; then
  SHARED_ARGS+=(--solver_token_entropy_enabled)
else
  SHARED_ARGS+=(--disable_solver_token_entropy)
fi
if [[ "$PROPOSER_CERTIFICATE_ENABLED" == "1" ]]; then
  SHARED_ARGS+=(--proposer_certificate_enabled)
else
  SHARED_ARGS+=(--disable_proposer_certificate)
fi
if [[ "$PROPOSER_CERTIFICATE_STRICT_STRUCT" == "1" ]]; then
  SHARED_ARGS+=(--proposer_certificate_strict_struct)
else
  SHARED_ARGS+=(--disable_proposer_certificate_strict_struct)
fi
if [[ "$PROPOSER_WARM_START_ENABLED" == "1" ]]; then
  SHARED_ARGS+=(--proposer_warm_start_enabled)
else
  SHARED_ARGS+=(--disable_proposer_warm_start)
fi
if [[ "$HARDNESS_DEBT_ENABLED" == "1" ]]; then
  SHARED_ARGS+=(--hardness_debt_enabled)
else
  SHARED_ARGS+=(--disable_hardness_debt)
fi
if [[ "$DIFFICULTY_SAMPLER_ENABLED" == "1" ]]; then
  SHARED_ARGS+=(--difficulty_sampler_enabled)
else
  SHARED_ARGS+=(--disable_difficulty_sampler)
fi
if [[ "$ENTROPY_IQR_FILTER_ENABLED" == "1" ]]; then
  SHARED_ARGS+=(--entropy_iqr_filter_enabled)
else
  SHARED_ARGS+=(--disable_entropy_iqr_filter)
fi
if [[ "$PROPOSER_CONTRASTIVE_REPLAY_ENABLED" == "1" ]]; then
  SHARED_ARGS+=(--proposer_contrastive_replay_enabled)
else
  SHARED_ARGS+=(--disable_proposer_contrastive_replay)
fi
if [[ "$PROPOSER_EARLY_FAILFAST_ENABLED" == "1" ]]; then
  SHARED_ARGS+=(--proposer_early_failfast_enabled)
else
  SHARED_ARGS+=(--disable_proposer_early_failfast)
fi
if [[ "$PROPOSER_EARLY_FAILFAST_STOP" == "1" ]]; then
  SHARED_ARGS+=(--proposer_early_failfast_stop)
else
  SHARED_ARGS+=(--disable_proposer_early_failfast_stop)
fi
if [[ "$PROPOSER_EARLY_FAILFAST_RECOVER" == "1" ]]; then
  SHARED_ARGS+=(--proposer_early_failfast_recover)
else
  SHARED_ARGS+=(--disable_proposer_early_failfast_recover)
fi
if [[ "$GRPO_DEGENERATE_NOISE_ENABLED" == "1" ]]; then
  SHARED_ARGS+=(--grpo_degenerate_noise_enabled)
else
  SHARED_ARGS+=(--disable_grpo_degenerate_noise)
fi
if [[ "$GRPO_PAIRWISE_RANKING_ENABLED" == "1" ]]; then
  SHARED_ARGS+=(--grpo_pairwise_ranking_enabled)
else
  SHARED_ARGS+=(--disable_grpo_pairwise_ranking)
fi
if [[ "$GEN_STEP_SOLVER_UPDATE_ENABLED" == "1" ]]; then
  SHARED_ARGS+=(--gen_step_solver_update_enabled)
else
  SHARED_ARGS+=(--disable_gen_step_solver_update)
fi

# ── Optional SUDER-style generation-phase rollout args ───────────────────────
SUDER_ARGS=()
if [[ "$ENABLE_SUDER" == "1" ]]; then
  SUDER_ARGS=(
    --suder_generation_enabled
    --proposer_gen_entropy_weight "$PROPOSER_GEN_ENTROPY_WEIGHT"
    --proposer_gen_baseline_momentum "$PROPOSER_GEN_BASELINE_MOMENTUM"
    --gen_spec_min_qa_pairs "$GEN_SPEC_MIN_QA_PAIRS"
    --gen_spec_temperature "$GEN_SPEC_TEMPERATURE"
    --max_new_tokens_gen_spec "$MAX_NEW_TOKENS_GEN_SPEC"
    --generation_num_timesteps "$GEN_NUM_TIMESTEPS"
    --generation_image_size "$GEN_IMAGE_SIZE"
  )
  if [[ "$SAVE_GENERATED_IMAGES" == "1" ]]; then
    SUDER_ARGS+=(--save_generated_images)
  fi
fi

# ── Train-mode args ────────────────────────────────────────────────────────
TRAIN_ARGS=()
if [[ "$ENABLE_LORA" == "1" ]]; then
  TRAIN_ARGS+=(
    --enable_lora
    --lora_rank "$LORA_RANK"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --lora_target_modules_csv "$LORA_TARGET_MODULES_CSV"
    --lora_role_adapters_csv "$LORA_ROLE_ADAPTERS_CSV"
    --lora_default_adapter "$LORA_DEFAULT_ADAPTER"
  )
fi

if [[ "$RUN_MODE" == "train" ]]; then
  TRAIN_ARGS+=(
    --policy_updates_enabled
    --policy_update_method "$POLICY_UPDATE_METHOD"
    --policy_lr "$POLICY_LR"
    --policy_weight_decay "$POLICY_WEIGHT_DECAY"
    --policy_max_grad_norm "$POLICY_MAX_GRAD_NORM"
    --policy_grad_accum_steps "$POLICY_GRAD_ACCUM_STEPS"
    --policy_reward_scale "$POLICY_REWARD_SCALE"
    --baseline_momentum "$BASELINE_MOMENTUM"
    --solver_reward_mix_gamma "$SOLVER_REWARD_MIX_GAMMA"
    --checkpoint_every "$CHECKPOINT_EVERY"
  )

  if [[ "$TRAIN_UNDERSTANDING_PROPOSER" != "1" ]]; then
    TRAIN_ARGS+=(--disable_train_understanding_proposer)
  fi
  if [[ "$TRAIN_SOLVER" != "1" ]]; then
    TRAIN_ARGS+=(--disable_train_solver)
  fi
  if [[ "$TRAIN_GENERATION_PROPOSER" != "1" ]]; then
    TRAIN_ARGS+=(--disable_train_generation_proposer)
  fi
  if [[ -n "$RESUME_FROM" ]]; then
    TRAIN_ARGS+=(--resume_from "$RESUME_FROM")
  fi
fi

# ── Pre-flight checks ───────────────────────────────────────────────────────
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[B1] ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/llm_config.json" ]]; then
  echo "[B1] ERROR: Missing llm_config.json in MODEL_PATH: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/vit_config.json" ]]; then
  echo "[B1] ERROR: Missing vit_config.json in MODEL_PATH: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/ae.safetensors" ]]; then
  echo "[B1] ERROR: Missing ae.safetensors in MODEL_PATH: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/ema.safetensors" && ! -f "$MODEL_PATH/model.safetensors" ]]; then
  echo "[B1] ERROR: Missing ema.safetensors/model.safetensors in MODEL_PATH: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[B1] ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
  exit 1
fi

if [[ "$DEVICE" == "$VAE_DEVICE" && "$MULTI_GPU_SPLIT" == "on" ]]; then
  echo "[B1] ERROR: MULTI_GPU_SPLIT=on requires model and VAE on different GPUs (DEVICE=$DEVICE, VAE_DEVICE=$VAE_DEVICE)" >&2
  exit 1
fi

if ! find "$DATA_DIR" -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.bmp" \) \
  -print -quit | grep -q .; then
  echo "[B1] ERROR: DATA_DIR has no image files: $DATA_DIR" >&2
  exit 1
fi

if [[ "$RUN_MODE" == "train" ]]; then
  if ! "$PYTHON_BIN" - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("peft") is not None else 1)
PY
  then
    echo "[B1] ERROR: peft is required in train mode. Install with: pip install peft" >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="$OUTPUT_DIR/b1_unified_${RUN_TS}.log"

echo "[B1] Starting BAGEL self-evolving experiment"
echo "[B1]   Run mode:   $RUN_MODE"
echo "[B1]   Stage:      $TRAIN_STAGE"
echo "[B1]   Exp:        $EXPERIMENT"
echo "[B1]   Model:      $MODEL_PATH"
echo "[B1]   Data:       $DATA_DIR"
echo "[B1]   Output:     $OUTPUT_DIR"
echo "[B1]   OutLayout:  $OUTPUT_LAYOUT"
echo "[B1]   Steps:      $STEPS"
echo "[B1]   Device:     $DEVICE"
echo "[B1]   GPUs:       count=$GPU_COUNT split=$MULTI_GPU_SPLIT"
echo "[B1]   Runtime:    rocm=$ROCM_RUNTIME force_math_sdpa=$FORCE_MATH_SDPA"
echo "[B1]   SafeMode:   rocm_safe_mode=$ROCM_SAFE_MODE"
if [[ -n "$VAE_DEVICE" ]]; then
  echo "[B1]   VAE device: $VAE_DEVICE"
fi
echo "[B1]   SUDER:      $ENABLE_SUDER"
echo "[B1]   FlashAttn:  disabled=$DISABLE_FLASH_ATTN"
echo "[B1]   Autocast:   disabled=$DISABLE_AUTOCAST dtype=$BAGEL_AUTOCAST_DTYPE"
echo "[B1]   ROCm AMP:   enable=$ENABLE_ROCM_AUTOCAST"
echo "[B1]   BLAS:       TORCH_BLAS_PREFER_HIPBLASLT=$TORCH_BLAS_PREFER_HIPBLASLT"
echo "[B1]   BlockMask:  compile=$BAGEL_COMPILE_BLOCK_MASK"
echo "[B1]   GenCfg:     image_size=$GEN_IMAGE_SIZE timesteps=$GEN_NUM_TIMESTEPS"
echo "[B1]   Schedule:   U=$UNDERSTANDING_STEPS_PER_CYCLE G=$GENERATION_STEPS_PER_CYCLE mix=$GEN_MIX_SOURCE_MODE"
echo "[B1]   Proposer:   K=$PROPOSER_NUM_CANDIDATES spot=$PROPOSER_SPOT_CHECK_SAMPLES"
if [[ "$RUN_MODE" == "train" ]]; then
  echo "[B1]   Policy:     $POLICY_UPDATE_METHOD"
  echo "[B1]   PolicyImg:  max_vit_edge=$POLICY_MAX_VIT_EDGE min_vit_edge=$POLICY_MIN_VIT_EDGE"
  echo "[B1]   PolicyOOM:  retries=$POLICY_OOM_MAX_RETRIES decay=$POLICY_OOM_EDGE_DECAY pause_after=$POLICY_OOM_PAUSE_AFTER_CONSECUTIVE pause_steps=$POLICY_OOM_PAUSE_STEPS"
  echo "[B1]   PolicyTok:  max_completion_tokens=$POLICY_MAX_COMPLETION_TOKENS min_completion_tokens=$POLICY_MIN_COMPLETION_TOKENS max_prompt_tokens=$POLICY_MAX_PROMPT_TOKENS text_only=$POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS retries=$POLICY_TEXT_ONLY_MAX_RETRIES"
  echo "[B1]   PolicyFB:   text_only_fallback=$POLICY_TEXT_ONLY_FALLBACK text_only_mode=$POLICY_TEXT_ONLY_MODE rocm_force_text_only=$POLICY_ROCM_FORCE_TEXT_ONLY empty_cache_each_step=$POLICY_EMPTY_CACHE_EACH_STEP"
  echo "[B1]   PolicyUpd:  solver_max_samples=$SOLVER_POLICY_MAX_SAMPLES gen_solver_max_samples=$GEN_SOLVER_POLICY_MAX_SAMPLES proposer_max_candidates=$PROPOSER_POLICY_MAX_CANDIDATES"
  echo "[B1]   Gen-GRPO:   group=$PROPOSER_GRPO_GEN_GROUP_SIZE score_extras=$SCORE_GRPO_EXTRAS temp_mult=$GRPO_EXTRA_TEMP_MULTIPLIER"
  echo "[B1]   LoRA:       enabled (r=$LORA_RANK, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT)"
fi
echo "[B1]   LauncherLog:$LAUNCH_LOG"
echo "[B1]   Monitor:    tail -f \"$LAUNCH_LOG\""

# ── Launch ──────────────────────────────────────────────────────────────────
export PYTHONPATH="$BAGEL_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
export BAGEL_DISABLE_FLASH_ATTN="$DISABLE_FLASH_ATTN"
export BAGEL_DISABLE_AUTOCAST="$DISABLE_AUTOCAST"
export BAGEL_AUTOCAST_DTYPE="$BAGEL_AUTOCAST_DTYPE"
export BAGEL_ENABLE_ROCM_AUTOCAST="$ENABLE_ROCM_AUTOCAST"
export TORCH_BLAS_PREFER_HIPBLASLT="$TORCH_BLAS_PREFER_HIPBLASLT"
export BAGEL_FORCE_MATH_SDPA="$FORCE_MATH_SDPA"
export BAGEL_COMPILE_BLOCK_MASK="$BAGEL_COMPILE_BLOCK_MASK"
export BAGEL_POLICY_MAX_VIT_EDGE="$POLICY_MAX_VIT_EDGE"
export BAGEL_POLICY_MIN_VIT_EDGE="$POLICY_MIN_VIT_EDGE"
export BAGEL_POLICY_OOM_MAX_RETRIES="$POLICY_OOM_MAX_RETRIES"
export BAGEL_POLICY_OOM_EDGE_DECAY="$POLICY_OOM_EDGE_DECAY"
export BAGEL_POLICY_OOM_FORCE_TEXT_ONLY_STEPS="$POLICY_OOM_FORCE_TEXT_ONLY_STEPS"
export BAGEL_POLICY_OOM_PAUSE_AFTER_CONSECUTIVE="$POLICY_OOM_PAUSE_AFTER_CONSECUTIVE"
export BAGEL_POLICY_OOM_PAUSE_STEPS="$POLICY_OOM_PAUSE_STEPS"
export BAGEL_POLICY_MAX_COMPLETION_TOKENS="$POLICY_MAX_COMPLETION_TOKENS"
export BAGEL_POLICY_MIN_COMPLETION_TOKENS="$POLICY_MIN_COMPLETION_TOKENS"
export BAGEL_POLICY_TEXT_ONLY_FALLBACK="$POLICY_TEXT_ONLY_FALLBACK"
export BAGEL_POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS="$POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS"
export BAGEL_POLICY_TEXT_ONLY_MAX_RETRIES="$POLICY_TEXT_ONLY_MAX_RETRIES"
export BAGEL_POLICY_TEXT_ONLY_MODE="$POLICY_TEXT_ONLY_MODE"
export BAGEL_POLICY_ROCM_FORCE_TEXT_ONLY="$POLICY_ROCM_FORCE_TEXT_ONLY"
export BAGEL_POLICY_EMPTY_CACHE_EACH_STEP="$POLICY_EMPTY_CACHE_EACH_STEP"
export BAGEL_POLICY_MAX_PROMPT_TOKENS="$POLICY_MAX_PROMPT_TOKENS"
export BAGEL_SOLVER_POLICY_MAX_SAMPLES="$SOLVER_POLICY_MAX_SAMPLES"
export BAGEL_GEN_SOLVER_POLICY_MAX_SAMPLES="$GEN_SOLVER_POLICY_MAX_SAMPLES"
export BAGEL_PROPOSER_POLICY_MAX_CANDIDATES="$PROPOSER_POLICY_MAX_CANDIDATES"
export BAGEL_OUTPUT_DIR_MODE="$OUTPUT_LAYOUT"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

cd "$BAGEL_ROOT"
set +e
"$PYTHON_BIN" -u train/train_self_evolving.py \
  --model_path "$MODEL_PATH" \
  --device "$DEVICE" \
  --vae_device "$VAE_DEVICE" \
  --max_latent_size "$MAX_LATENT_SIZE" \
  --image_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  "${SHARED_ARGS[@]}" \
  "${STAGE_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  "${SUDER_ARGS[@]}" \
  2>&1 | tee -a "$LAUNCH_LOG"
PY_EXIT_CODE="${PIPESTATUS[0]}"
set -e

LATEST_RUN_DIR=""
if [[ "$OUTPUT_LAYOUT" == "direct" ]]; then
  LATEST_RUN_DIR="$OUTPUT_DIR"
else
  LATEST_RUN_DIR="$(ls -td "$OUTPUT_DIR"/unified_rollout_* 2>/dev/null | head -1 || true)"
fi
if [[ -n "$LATEST_RUN_DIR" ]]; then
  echo "[B1]   LatestRun:  $LATEST_RUN_DIR"
  echo "[B1]   Status:     $LATEST_RUN_DIR/status.json"
  echo "[B1]   Metrics:    $LATEST_RUN_DIR/metrics.jsonl"
  echo "[B1]   Checkpoints:$LATEST_RUN_DIR/checkpoints"
fi

if [[ "$PY_EXIT_CODE" -ne 0 ]]; then
  echo "[B1] ERROR: Training exited with code $PY_EXIT_CODE" >&2
  exit "$PY_EXIT_CODE"
fi

echo "[B1] Completed."
