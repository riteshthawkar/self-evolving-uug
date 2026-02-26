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
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/BAGEL/B1_unified_training}"

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
BASELINE_MOMENTUM="${BASELINE_MOMENTUM:-0.9}"
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
echo "[B1]   Steps:      $STEPS"
echo "[B1]   Device:     $DEVICE"
echo "[B1]   GPUs:       count=$GPU_COUNT split=$MULTI_GPU_SPLIT"
echo "[B1]   Runtime:    rocm=$ROCM_RUNTIME force_math_sdpa=$FORCE_MATH_SDPA"
if [[ -n "$VAE_DEVICE" ]]; then
  echo "[B1]   VAE device: $VAE_DEVICE"
fi
echo "[B1]   SUDER:      $ENABLE_SUDER"
echo "[B1]   FlashAttn:  disabled=$DISABLE_FLASH_ATTN"
echo "[B1]   Autocast:   disabled=$DISABLE_AUTOCAST dtype=$BAGEL_AUTOCAST_DTYPE"
echo "[B1]   ROCm AMP:   enable=$ENABLE_ROCM_AUTOCAST"
echo "[B1]   BLAS:       TORCH_BLAS_PREFER_HIPBLASLT=$TORCH_BLAS_PREFER_HIPBLASLT"
echo "[B1]   BlockMask:  compile=$BAGEL_COMPILE_BLOCK_MASK"
echo "[B1]   Schedule:   U=$UNDERSTANDING_STEPS_PER_CYCLE G=$GENERATION_STEPS_PER_CYCLE mix=$GEN_MIX_SOURCE_MODE"
echo "[B1]   Proposer:   K=$PROPOSER_NUM_CANDIDATES spot=$PROPOSER_SPOT_CHECK_SAMPLES"
if [[ "$RUN_MODE" == "train" ]]; then
  echo "[B1]   Policy:     $POLICY_UPDATE_METHOD"
  echo "[B1]   PolicyImg:  max_vit_edge=$POLICY_MAX_VIT_EDGE min_vit_edge=$POLICY_MIN_VIT_EDGE"
  echo "[B1]   PolicyOOM:  retries=$POLICY_OOM_MAX_RETRIES decay=$POLICY_OOM_EDGE_DECAY"
  echo "[B1]   PolicyTok:  max_completion_tokens=$POLICY_MAX_COMPLETION_TOKENS"
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
export BAGEL_POLICY_MAX_COMPLETION_TOKENS="$POLICY_MAX_COMPLETION_TOKENS"

cd "$BAGEL_ROOT"
set +e
"$PYTHON_BIN" train/train_self_evolving.py \
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

LATEST_RUN_DIR="$(ls -td "$OUTPUT_DIR"/unified_rollout_* 2>/dev/null | head -1 || true)"
if [[ -n "$LATEST_RUN_DIR" ]]; then
  echo "[B1]   LatestRun:  $LATEST_RUN_DIR"
  echo "[B1]   Status:     $LATEST_RUN_DIR/status.json"
  echo "[B1]   Metrics:    $LATEST_RUN_DIR/metrics.jsonl"
fi

if [[ "$PY_EXIT_CODE" -ne 0 ]]; then
  echo "[B1] ERROR: Training exited with code $PY_EXIT_CODE" >&2
  exit "$PY_EXIT_CODE"
fi

echo "[B1] Completed."
