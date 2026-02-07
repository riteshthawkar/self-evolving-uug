#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

# ============================================================
# Standalone Understanding Experiments Runner (No dependencies)
# ============================================================
# This script is self-contained and does NOT source other scripts.
# It runs the understanding-only experiment matrix from one file.
#
# Usage:
#   bash self_evolving/scripts/understanding_experiments/run_understanding_all_standalone.sh \
#     --data_dir /path/to/images/train \
#     --output_root ./runs/understanding_experiments \
#     --suite full
#
# Optional:
#   --model_name Qwen/Qwen2.5-VL-3B-Instruct
#   --total_steps 6000
#   --cuda_device 0
#   --save_every 200
#   --dry_run
#   --wandb_mode online|offline|disabled
#   --wandb_project <project>
#   --wandb_entity <entity>
#   --wandb_log_images_every 0
#
# Notes:
# - W&B token is read from WANDB_API_KEY.
# - Unknown args are forwarded to run_experiment.py.

# -------------------------
# Cache exports
# -------------------------
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

mkdir -p \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$HF_METRICS_CACHE" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$XDG_CACHE_HOME"

# -------------------------
# W&B exports
# -------------------------
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-self-evolving-uug-understanding}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_LOG_IMAGES_EVERY="${WANDB_LOG_IMAGES_EVERY:-0}"

if [[ -z "${WANDB_MODE}" ]]; then
  if [[ -n "${WANDB_API_KEY}" ]]; then
    WANDB_MODE="online"
  else
    WANDB_MODE="disabled"
  fi
fi
export WANDB_MODE

# -------------------------
# Defaults
# -------------------------
cd "$REPO_ROOT"

DATA_DIR=""
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_ROOT="$REPO_ROOT/runs/understanding_experiments"
TOTAL_STEPS=6000
SAVE_EVERY=200
MAX_CHECKPOINTS=3
CUDA_DEVICE=0
SUITE="full" # core|full
PYTHON_BIN="python3"
DRY_RUN=0

EXTRA_ARGS=()
LAUNCH_TS="$(date -u +%Y%m%d_%H%M%S)"

usage() {
  cat <<'USAGE'
Standalone Understanding Experiments Runner.

Required:
  --data_dir PATH

Optional:
  --model_name NAME
  --output_root PATH
  --total_steps N
  --save_every N
  --max_checkpoints N
  --cuda_device IDX
  --suite core|full
  --python_bin BIN
  --dry_run

W&B optional:
  --wandb_mode online|offline|disabled
  --wandb_project NAME
  --wandb_entity NAME
  --wandb_log_images_every N

Cache env vars (already exported at top, override externally if needed):
  CACHE_ROOT, HF_HOME, HUGGINGFACE_HUB_CACHE, TRANSFORMERS_CACHE,
  HF_DATASETS_CACHE, HF_METRICS_CACHE, TORCH_HOME, TRITON_CACHE_DIR,
  XDG_CACHE_HOME

Unknown args are forwarded to self_evolving/run_experiment.py.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)
      DATA_DIR="${2:-}"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="${2:-}"
      shift 2
      ;;
    --output_root)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --total_steps)
      TOTAL_STEPS="${2:-}"
      shift 2
      ;;
    --save_every)
      SAVE_EVERY="${2:-}"
      shift 2
      ;;
    --max_checkpoints)
      MAX_CHECKPOINTS="${2:-}"
      shift 2
      ;;
    --cuda_device)
      CUDA_DEVICE="${2:-}"
      shift 2
      ;;
    --suite)
      SUITE="${2:-}"
      shift 2
      ;;
    --python_bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --wandb_mode)
      WANDB_MODE="${2:-}"
      export WANDB_MODE
      shift 2
      ;;
    --wandb_project)
      WANDB_PROJECT="${2:-}"
      export WANDB_PROJECT
      shift 2
      ;;
    --wandb_entity)
      WANDB_ENTITY="${2:-}"
      export WANDB_ENTITY
      shift 2
      ;;
    --wandb_log_images_every)
      WANDB_LOG_IMAGES_EVERY="${2:-}"
      export WANDB_LOG_IMAGES_EVERY
      shift 2
      ;;
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "--data_dir is required." >&2
  usage
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "data_dir not found: $DATA_DIR" >&2
  exit 1
fi

if [[ "$SUITE" != "core" && "$SUITE" != "full" ]]; then
  echo "Invalid --suite '$SUITE'. Must be core|full." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

run_case() {
  local exp_id="$1"
  local run_name="$2"
  shift 2

  local exp_dir="$OUTPUT_ROOT/$exp_id"
  local launcher_log_dir="$exp_dir/launcher_logs/$LAUNCH_TS"
  local launcher_log="$launcher_log_dir/${run_name}.log"

  mkdir -p "$exp_dir" "$launcher_log_dir"

  local cmd=(
    "$PYTHON_BIN" self_evolving/run_experiment.py
    --experiment understanding_self_evolving
    --data_dir "$DATA_DIR"
    --model_name "$MODEL_NAME"
    --output_dir "$exp_dir"
    --run_name "$run_name"
    --dtype bfloat16
    --device_map single
    --cuda_device "$CUDA_DEVICE"
    --total_steps "$TOTAL_STEPS"
    --save_every "$SAVE_EVERY"
    --log_every 1
    --max_checkpoints "$MAX_CHECKPOINTS"
    --deterministic
    --use_lora
    --lora_r 16
    --lora_alpha 32
    --lora_dropout 0.05
    --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,mm_projector
    --lr 1e-6
    --weight_decay 0.01
    --grad_clip 1.0
    --proposer_update_freq 5
    --temp 1.0
    --top_p 1.0
    --max_new_tokens_solver 128
    --max_new_tokens_proposer 128
    --num_solver_samples 5
    --solver_soft_gamma 0.7
    --len_penalty_weight 0.10
    --len_penalty_target_words 6
    --prop_entropy_mu 0.90
    --prop_entropy_sigma 0.35
    --kl_coef 1e-3
    --kl_target 0.02
    --kl_adapt_rate 0.10
    --kl_min 1e-8
    --kl_max 1e2
    --baseline_momentum 0.9
    --clear_cache_every 25
    --wandb_mode "$WANDB_MODE"
    --wandb_project "$WANDB_PROJECT"
    --wandb_log_images_every "$WANDB_LOG_IMAGES_EVERY"
    --wandb_run_name "$run_name"
  )

  if [[ -n "$WANDB_ENTITY" ]]; then
    cmd+=(--wandb_entity "$WANDB_ENTITY")
  fi

  cmd+=("$@")
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY_RUN] experiment=$exp_id run=$run_name"
    print_cmd "${cmd[@]}"
    echo "[DRY_RUN] launcher_log=$launcher_log"
    return 0
  fi

  {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START experiment=$exp_id run=$run_name"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launcher_log=$launcher_log"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] wandb_mode=$WANDB_MODE wandb_project=$WANDB_PROJECT wandb_entity=${WANDB_ENTITY:-<none>}"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] wandb_api_key=$( [[ -n "$WANDB_API_KEY" ]] && echo set || echo unset )"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] cache_root=$CACHE_ROOT hf_home=$HF_HOME"
    printf 'CMD: '
    print_cmd "${cmd[@]}"
    "${cmd[@]}"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] END experiment=$exp_id run=$run_name"
  } 2>&1 | tee "$launcher_log"
}

echo "============================================================"
echo "Standalone Understanding Suite"
echo "Repository: $REPO_ROOT"
echo "Data dir:   $DATA_DIR"
echo "Model:      $MODEL_NAME"
echo "Output:     $OUTPUT_ROOT"
echo "Suite:      $SUITE"
echo "Steps:      $TOTAL_STEPS"
echo "CUDA dev:   $CUDA_DEVICE"
echo "W&B mode:   $WANDB_MODE"
echo "W&B proj:   $WANDB_PROJECT"
echo "W&B entity: ${WANDB_ENTITY:-<none>}"
echo "W&B token:  $( [[ -n "$WANDB_API_KEY" ]] && echo set || echo unset )"
echo "HF_HOME:    $HF_HOME"
echo "Dry run:    $DRY_RUN"
echo "============================================================"

# U00: main method (seeds)
run_case "U00_main_method" "u00_main_default_s42" --seed 42
run_case "U00_main_method" "u00_main_default_s123" --seed 123
run_case "U00_main_method" "u00_main_default_s777" --seed 777

# U01: solver samples
run_case "U01_solver_samples" "u01_nsamples_3_s42" --seed 42 --num_solver_samples 3
run_case "U01_solver_samples" "u01_nsamples_5_s42" --seed 42 --num_solver_samples 5
run_case "U01_solver_samples" "u01_nsamples_7_s42" --seed 42 --num_solver_samples 7

# U02: solver gamma
run_case "U02_solver_gamma" "u02_gamma_0p5_s42" --seed 42 --solver_soft_gamma 0.5
run_case "U02_solver_gamma" "u02_gamma_0p7_s42" --seed 42 --solver_soft_gamma 0.7
run_case "U02_solver_gamma" "u02_gamma_1p0_s42" --seed 42 --solver_soft_gamma 1.0

# U03: proposer update frequency
run_case "U03_proposer_update_freq" "u03_propfreq_1_s42" --seed 42 --proposer_update_freq 1
run_case "U03_proposer_update_freq" "u03_propfreq_3_s42" --seed 42 --proposer_update_freq 3
run_case "U03_proposer_update_freq" "u03_propfreq_5_s42" --seed 42 --proposer_update_freq 5
run_case "U03_proposer_update_freq" "u03_propfreq_10_s42" --seed 42 --proposer_update_freq 10

if [[ "$SUITE" == "full" ]]; then
  # U04: entropy band
  run_case "U04_entropy_band" "u04_mu0p70_sigma0p25_s42" --seed 42 --prop_entropy_mu 0.70 --prop_entropy_sigma 0.25
  run_case "U04_entropy_band" "u04_mu0p70_sigma0p35_s42" --seed 42 --prop_entropy_mu 0.70 --prop_entropy_sigma 0.35
  run_case "U04_entropy_band" "u04_mu0p90_sigma0p25_s42" --seed 42 --prop_entropy_mu 0.90 --prop_entropy_sigma 0.25
  run_case "U04_entropy_band" "u04_mu0p90_sigma0p35_s42" --seed 42 --prop_entropy_mu 0.90 --prop_entropy_sigma 0.35
  run_case "U04_entropy_band" "u04_mu1p10_sigma0p25_s42" --seed 42 --prop_entropy_mu 1.10 --prop_entropy_sigma 0.25
  run_case "U04_entropy_band" "u04_mu1p10_sigma0p35_s42" --seed 42 --prop_entropy_mu 1.10 --prop_entropy_sigma 0.35

  # U05: KL sensitivity
  run_case "U05_kl_sensitivity" "u05_klcoef2e3_kltarget0p01_s42" --seed 42 --kl_coef 2e-3 --kl_target 0.01
  run_case "U05_kl_sensitivity" "u05_klcoef1e3_kltarget0p02_s42" --seed 42 --kl_coef 1e-3 --kl_target 0.02
  run_case "U05_kl_sensitivity" "u05_klcoef5e4_kltarget0p05_s42" --seed 42 --kl_coef 5e-4 --kl_target 0.05

  # U06: LoRA capacity
  run_case "U06_lora_capacity" "u06_lorar8_alpha16_s42" --seed 42 --lora_r 8 --lora_alpha 16
  run_case "U06_lora_capacity" "u06_lorar16_alpha32_s42" --seed 42 --lora_r 16 --lora_alpha 32
  run_case "U06_lora_capacity" "u06_lorar32_alpha64_s42" --seed 42 --lora_r 32 --lora_alpha 64

  # U07: frozen proposer proxy
  FROZEN_FREQ=$((TOTAL_STEPS + 1))
  run_case "U07_frozen_proposer_proxy" "u07_frozen_proposer_proxy_s42" --seed 42 --proposer_update_freq "$FROZEN_FREQ"
fi

echo "All requested runs submitted."
