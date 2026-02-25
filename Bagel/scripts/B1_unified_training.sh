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
#   STEPS=500
#   DEVICE=cuda
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
STEPS="${STEPS:-500}"
DEVICE="${DEVICE:-cuda}"
MAX_LATENT_SIZE="${MAX_LATENT_SIZE:-64}"
ENABLE_SUDER="${ENABLE_SUDER:-1}"
PROPOSER_GEN_ENTROPY_WEIGHT="${PROPOSER_GEN_ENTROPY_WEIGHT:-0.7}"
PROPOSER_GEN_BASELINE_MOMENTUM="${PROPOSER_GEN_BASELINE_MOMENTUM:-0.6}"
GEN_SPEC_MIN_QA_PAIRS="${GEN_SPEC_MIN_QA_PAIRS:-2}"
GEN_SPEC_TEMPERATURE="${GEN_SPEC_TEMPERATURE:-0.9}"
MAX_NEW_TOKENS_GEN_SPEC="${MAX_NEW_TOKENS_GEN_SPEC:-384}"
GEN_IMAGE_SIZE="${GEN_IMAGE_SIZE:-1024}"
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
BASELINE_MOMENTUM="${BASELINE_MOMENTUM:-0.9}"
SOLVER_REWARD_MIX_GAMMA="${SOLVER_REWARD_MIX_GAMMA:-0.7}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
RESUME_FROM="${RESUME_FROM:-}"
DISABLE_FLASH_ATTN="${DISABLE_FLASH_ATTN:-1}"
DISABLE_AUTOCAST="${DISABLE_AUTOCAST:-0}"
BAGEL_AUTOCAST_DTYPE="${BAGEL_AUTOCAST_DTYPE:-auto}"
ENABLE_ROCM_AUTOCAST="${ENABLE_ROCM_AUTOCAST:-0}"
TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-0}"
TRAIN_UNDERSTANDING_PROPOSER="${TRAIN_UNDERSTANDING_PROPOSER:-1}"
TRAIN_SOLVER="${TRAIN_SOLVER:-1}"
TRAIN_GENERATION_PROPOSER="${TRAIN_GENERATION_PROPOSER:-1}"

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

if [[ "$RUN_MODE" == "train" ]]; then
  ENABLE_LORA=1
  ENABLE_SUDER=1
  TRAIN_UNDERSTANDING_PROPOSER=1
  TRAIN_SOLVER=1
  TRAIN_GENERATION_PROPOSER=1
fi

# ── Shared arguments ────────────────────────────────────────────────────────
SHARED_ARGS=(
  --max_new_tokens_proposer 256
  --max_new_tokens_solver 96
  --solver_unsolvable_maj_threshold 0.20
  --zero_entropy_eps 1e-6
  --seed 42
  --log_every 10
  --save_raw_generations
)

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

echo "[B1] Starting BAGEL self-evolving experiment"
echo "[B1]   Run mode:   $RUN_MODE"
echo "[B1]   Stage:      $TRAIN_STAGE"
echo "[B1]   Model:      $MODEL_PATH"
echo "[B1]   Data:       $DATA_DIR"
echo "[B1]   Output:     $OUTPUT_DIR"
echo "[B1]   Steps:      $STEPS"
echo "[B1]   Device:     $DEVICE"
echo "[B1]   SUDER:      $ENABLE_SUDER"
echo "[B1]   FlashAttn:  disabled=$DISABLE_FLASH_ATTN"
echo "[B1]   Autocast:   disabled=$DISABLE_AUTOCAST dtype=$BAGEL_AUTOCAST_DTYPE"
echo "[B1]   ROCm AMP:   enable=$ENABLE_ROCM_AUTOCAST"
echo "[B1]   BLAS:       TORCH_BLAS_PREFER_HIPBLASLT=$TORCH_BLAS_PREFER_HIPBLASLT"
if [[ "$RUN_MODE" == "train" ]]; then
  echo "[B1]   Policy:     $POLICY_UPDATE_METHOD"
  echo "[B1]   LoRA:       enabled (r=$LORA_RANK, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT)"
fi

# ── Launch ──────────────────────────────────────────────────────────────────
export PYTHONPATH="$BAGEL_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
export BAGEL_DISABLE_FLASH_ATTN="$DISABLE_FLASH_ATTN"
export BAGEL_DISABLE_AUTOCAST="$DISABLE_AUTOCAST"
export BAGEL_AUTOCAST_DTYPE="$BAGEL_AUTOCAST_DTYPE"
export BAGEL_ENABLE_ROCM_AUTOCAST="$ENABLE_ROCM_AUTOCAST"
export TORCH_BLAS_PREFER_HIPBLASLT="$TORCH_BLAS_PREFER_HIPBLASLT"

cd "$BAGEL_ROOT"
"$PYTHON_BIN" train/train_self_evolving.py \
  --model_path "$MODEL_PATH" \
  --device "$DEVICE" \
  --max_latent_size "$MAX_LATENT_SIZE" \
  --image_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  "${SHARED_ARGS[@]}" \
  "${STAGE_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  "${SUDER_ARGS[@]}"

echo "[B1] Completed."
