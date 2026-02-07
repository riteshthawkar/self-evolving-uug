#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# Cache exports
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

mkdir -p   "$HF_HOME"   "$HUGGINGFACE_HUB_CACHE"   "$TRANSFORMERS_CACHE"   "$HF_DATASETS_CACHE"   "$HF_METRICS_CACHE"   "$TORCH_HOME"   "$TRITON_CACHE_DIR"   "$XDG_CACHE_HOME"

# W&B environment variables
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-self-evolving-uug-understanding}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_LOG_IMAGES_EVERY="${WANDB_LOG_IMAGES_EVERY:-0}"

if [[ -z "$WANDB_MODE" ]]; then
  if [[ -n "$WANDB_API_KEY" ]]; then
    WANDB_MODE="online"
  else
    WANDB_MODE="disabled"
  fi
fi

DATA_DIR="${DATA_DIR:-}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/runs/understanding_experiments}"
TOTAL_STEPS="${TOTAL_STEPS:-6000}"
SAVE_EVERY="${SAVE_EVERY:-200}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-3}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"

PASSTHROUGH_ARGS=()
LAUNCH_TS="${LAUNCH_TS:-$(date -u +%Y%m%d_%H%M%S)}"

common_usage() {
  cat <<'USAGE'
Common options:
  --data_dir PATH          Required. Image root for training.
  --model_name NAME        Model id (default: Qwen/Qwen2.5-VL-3B-Instruct).
  --output_root PATH       Root folder for experiment outputs.
  --total_steps N          Training steps per run (default: 6000).
  --save_every N           Checkpoint interval (default: 200).
  --max_checkpoints N      Number of checkpoints retained per run (default: 3).
  --cuda_device IDX        CUDA index for single-device mode (default: 0).
  --python_bin BIN         Python executable (default: python3).
  --wandb_mode MODE        online|offline|disabled. Default: online if WANDB_API_KEY is set, else disabled.
  --wandb_project NAME     W&B project name.
  --wandb_entity NAME      W&B entity/team.
  --wandb_log_images_every N  Log sample image every N steps (default: 0).
  --dry_run                Print generated commands without executing.
  --help                   Show script-specific usage.

Any unknown arguments are forwarded to run_experiment.py.
USAGE
}

usage() {
  cat <<'USAGE'
U01: Solver sample-count ablation.

Runs:
  - u01_nsamples_3_s42
  - u01_nsamples_5_s42
  - u01_nsamples_7_s42
USAGE
  common_usage
}

parse_common_args() {
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
      --python_bin)
        PYTHON_BIN="${2:-}"
        shift 2
        ;;
      --wandb_mode)
        WANDB_MODE="${2:-}"
        shift 2
        ;;
      --wandb_project)
        WANDB_PROJECT="${2:-}"
        shift 2
        ;;
      --wandb_entity)
        WANDB_ENTITY="${2:-}"
        shift 2
        ;;
      --wandb_log_images_every)
        WANDB_LOG_IMAGES_EVERY="${2:-}"
        shift 2
        ;;
      --dry_run)
        DRY_RUN=1
        shift
        ;;
      --)
        shift
        while [[ $# -gt 0 ]]; do
          PASSTHROUGH_ARGS+=("$1")
          shift
        done
        ;;
      --help|-h)
        return 2
        ;;
      *)
        PASSTHROUGH_ARGS+=("$1")
        shift
        ;;
    esac
  done

  if [[ -z "$DATA_DIR" ]]; then
    echo "--data_dir is required." >&2
    return 1
  fi
  if [[ ! -d "$DATA_DIR" ]]; then
    echo "data_dir not found: $DATA_DIR" >&2
    return 1
  fi

  mkdir -p "$OUTPUT_ROOT"
}

print_shared_config() {
  echo "Repository: $REPO_ROOT"
  echo "Data dir:   $DATA_DIR"
  echo "Model:      $MODEL_NAME"
  echo "Output:     $OUTPUT_ROOT"
  echo "Steps:      $TOTAL_STEPS"
  echo "CUDA dev:   $CUDA_DEVICE"
  echo "W&B mode:   $WANDB_MODE"
  echo "W&B proj:   $WANDB_PROJECT"
  echo "W&B entity: ${WANDB_ENTITY:-<none>}"
  echo "W&B token:  $( [[ -n "$WANDB_API_KEY" ]] && echo set || echo unset )"
  echo "HF_HOME:    $HF_HOME"
  echo "Dry run:    $DRY_RUN"
}

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
  if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    cmd+=("${PASSTHROUGH_ARGS[@]}")
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

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if ! parse_common_args "$@"; then
  usage
  exit 1
fi

EXP_ID="U01_solver_samples"


echo "Running $EXP_ID"
print_shared_config

run_case "$EXP_ID" "u01_nsamples_3_s42" --seed 42 --num_solver_samples 3
run_case "$EXP_ID" "u01_nsamples_5_s42" --seed 42 --num_solver_samples 5
run_case "$EXP_ID" "u01_nsamples_7_s42" --seed 42 --num_solver_samples 7
