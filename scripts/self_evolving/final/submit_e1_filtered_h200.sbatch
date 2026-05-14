#!/usr/bin/env bash
#SBATCH --job-name=blip3o-e1-filtered
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# If your cluster requires these, uncomment and edit:
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --constraint=h200

# Submit from the repository root after creating the Slurm output directory:
#   mkdir -p logs
#   sbatch scripts/self_evolving/final/submit_e1_filtered_h200.sbatch
# For a larger host-memory allocation, override at submission time, e.g.:
#   sbatch --mem=64G --cpus-per-task=8 scripts/self_evolving/final/submit_e1_filtered_h200.sbatch
# Slurm opens logs/%x_%j.out before this script starts, so the logs directory
# must exist before sbatch is called.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
ORIGINAL_HOME="${HOME:-}"

cd "$REPO_ROOT"

to_abs_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$REPO_ROOT/$path"
  fi
}

nearest_existing_parent() {
  local path="$1"
  while [[ ! -d "$path" && "$path" != "/" ]]; do
    path="$(dirname "$path")"
  done
  printf '%s\n' "$path"
}

is_writable_path_candidate() {
  local path="$1"
  local parent
  parent="$(nearest_existing_parent "$path")"
  [[ -n "$parent" && -d "$parent" && -w "$parent" ]]
}

# Optional module setup. Example:
#   sbatch --export=ALL,MODULES_TO_LOAD="cuda/12.4 gcc/11" scripts/self_evolving/final/submit_e1_filtered_h200.sbatch
if [[ -n "${MODULES_TO_LOAD:-}" ]] && type module >/dev/null 2>&1; then
  module purge || true
  for module_name in ${MODULES_TO_LOAD}; do
    module load "$module_name"
  done
fi

# Conda activation. Override CONDA_ENV if your environment has another name.
# Keep this enabled for batch jobs unless you are submitting from an already
# activated environment and exporting that PATH intentionally.
CONDA_ENV="${CONDA_ENV:-blip3o}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"
if [[ "$SKIP_CONDA_ACTIVATE" == "1" ]]; then
  echo "[slurm] SKIP_CONDA_ACTIVATE=1: using existing PATH/python environment."
else
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE:-}" ]]; then
    eval "$("$CONDA_EXE" shell.bash hook)"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [[ -f "$ORIGINAL_HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$ORIGINAL_HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$ORIGINAL_HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$ORIGINAL_HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "[slurm] ERROR: conda was not found. Set CONDA_EXE or load your conda module." >&2
    echo "[slurm] If your submitted environment already has the right python, use SKIP_CONDA_ACTIVATE=1." >&2
    exit 1
  fi
  conda activate "$CONDA_ENV"
fi

# Main experiment defaults. Override any of these via sbatch --export.
FILTERED_DATA_DEFAULT="$REPO_ROOT/data/high_utility_pool_10k_h200_qwen3vl30b_a3b_bf16/images"
if [[ ! -d "$FILTERED_DATA_DEFAULT" ]]; then
  ALT_FILTERED_DATA="$REPO_ROOT/data/high_utility_pool_10k_h200_qwen3vl30b_a3b/images"
  if [[ -d "$ALT_FILTERED_DATA" ]]; then
    FILTERED_DATA_DEFAULT="$ALT_FILTERED_DATA"
  fi
fi

SMOKE="${SMOKE:-0}"
if [[ "$SMOKE" == "1" ]]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-E1_filtered10k_smoke_h200}"
  TOTAL_STEPS="${TOTAL_STEPS:-100}"
else
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-E1_filtered10k_main_h200}"
  TOTAL_STEPS="${TOTAL_STEPS:-10000}"
fi

DEFAULT_OUTPUT_DIR="$REPO_ROOT/outputs/blip3o/$EXPERIMENT_NAME"
OUTPUT_DIR_SOURCE="default"

# Slurm exports the submitting shell by default.  Generic variables such as
# OUTPUT_DIR and CACHE_ROOT are commonly set by cluster/container wrappers
# (for example /var/outputs), so accept them only if the destination is
# writable.  Use E1_OUTPUT_DIR/E1_CACHE_ROOT for intentional overrides.
if [[ -n "${E1_OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$(to_abs_path "$E1_OUTPUT_DIR")"
  OUTPUT_DIR_SOURCE="E1_OUTPUT_DIR"
  if ! is_writable_path_candidate "$OUTPUT_DIR"; then
    echo "[slurm] ERROR: E1_OUTPUT_DIR is not writable: $OUTPUT_DIR" >&2
    exit 1
  fi
elif [[ -n "${OUTPUT_DIR:-}" ]]; then
  REQUESTED_OUTPUT_DIR="$(to_abs_path "$OUTPUT_DIR")"
  if is_writable_path_candidate "$REQUESTED_OUTPUT_DIR"; then
    OUTPUT_DIR="$REQUESTED_OUTPUT_DIR"
    OUTPUT_DIR_SOURCE="OUTPUT_DIR"
  else
    echo "[slurm] WARNING: ignoring inherited OUTPUT_DIR=$REQUESTED_OUTPUT_DIR because its nearest existing parent is not writable." >&2
    echo "[slurm] WARNING: use E1_OUTPUT_DIR=/writable/path for an explicit output override." >&2
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    OUTPUT_DIR_SOURCE="default_ignored_inherited_OUTPUT_DIR"
  fi
else
  OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
fi

DATA_DIR="${DATA_DIR:-$FILTERED_DATA_DEFAULT}"
DATA_DIR="$(to_abs_path "$DATA_DIR")"

DEFAULT_CACHE_ROOT="$OUTPUT_DIR/cache_runtime"
CACHE_ROOT_SOURCE="default"
if [[ -n "${E1_CACHE_ROOT:-}" ]]; then
  CACHE_ROOT="$(to_abs_path "$E1_CACHE_ROOT")"
  CACHE_ROOT_SOURCE="E1_CACHE_ROOT"
  if ! is_writable_path_candidate "$CACHE_ROOT"; then
    echo "[slurm] ERROR: E1_CACHE_ROOT is not writable: $CACHE_ROOT" >&2
    exit 1
  fi
elif [[ -n "${CACHE_ROOT:-}" ]]; then
  REQUESTED_CACHE_ROOT="$(to_abs_path "$CACHE_ROOT")"
  if is_writable_path_candidate "$REQUESTED_CACHE_ROOT"; then
    CACHE_ROOT="$REQUESTED_CACHE_ROOT"
    CACHE_ROOT_SOURCE="CACHE_ROOT"
  else
    echo "[slurm] WARNING: ignoring inherited CACHE_ROOT=$REQUESTED_CACHE_ROOT because its nearest existing parent is not writable." >&2
    echo "[slurm] WARNING: use E1_CACHE_ROOT=/writable/path for an explicit cache override." >&2
    CACHE_ROOT="$DEFAULT_CACHE_ROOT"
    CACHE_ROOT_SOURCE="default_ignored_inherited_CACHE_ROOT"
  fi
else
  CACHE_ROOT="$DEFAULT_CACHE_ROOT"
fi

mkdir -p \
  "$OUTPUT_DIR/slurm" \
  "$CACHE_ROOT/home" \
  "$CACHE_ROOT/tmp" \
  "$CACHE_ROOT/xdg_cache" \
  "$CACHE_ROOT/xdg_config" \
  "$CACHE_ROOT/xdg_data" \
  "$CACHE_ROOT/huggingface" \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/pip" \
  "$CACHE_ROOT/uv" \
  "$CACHE_ROOT/matplotlib" \
  "$CACHE_ROOT/python/pycache" \
  "$CACHE_ROOT/python/userbase" \
  "$CACHE_ROOT/wandb"

SLURM_TEE_LOG="$OUTPUT_DIR/slurm/${SLURM_JOB_NAME:-blip3o-e1-filtered}-${SLURM_JOB_ID:-manual}.log"
exec > >(tee -a "$SLURM_TEE_LOG") 2>&1

echo "[slurm] job_id=${SLURM_JOB_ID:-manual}"
echo "[slurm] node_list=${SLURM_JOB_NODELIST:-local}"
echo "[slurm] repo=$REPO_ROOT"
echo "[slurm] conda_env=${CONDA_DEFAULT_ENV:-skipped_or_preexisting}"
echo "[slurm] data_dir=$DATA_DIR"
echo "[slurm] output_dir=$OUTPUT_DIR"
echo "[slurm] output_dir_source=$OUTPUT_DIR_SOURCE"
echo "[slurm] cache_root=$CACHE_ROOT"
echo "[slurm] cache_root_source=$CACHE_ROOT_SOURCE"
echo "[slurm] smoke=$SMOKE total_steps=$TOTAL_STEPS"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[slurm] ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
  echo "[slurm] Set DATA_DIR to the filtered pool images directory." >&2
  exit 1
fi

IMAGE_COUNT="$(find "$DATA_DIR" -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.bmp" -o -iname "*.tiff" \) \
  | wc -l | tr -d '[:space:]')"
echo "[slurm] image_count=$IMAGE_COUNT"

# Keep runtime/cache artifacts out of $HOME.  If Hugging Face auth only exists
# in the original home folder, read the token once and pass it through env.
if [[ -z "${HF_TOKEN:-}" && -n "$ORIGINAL_HOME" && -f "$ORIGINAL_HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(< "$ORIGINAL_HOME/.cache/huggingface/token")"
fi
export HOME="$CACHE_ROOT/home"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg_cache"
export XDG_CONFIG_HOME="$CACHE_ROOT/xdg_config"
export XDG_DATA_HOME="$CACHE_ROOT/xdg_data"
export TMPDIR="$CACHE_ROOT/tmp"
export TMP="$CACHE_ROOT/tmp"
export TEMP="$CACHE_ROOT/tmp"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/python/pycache"
export PYTHONUSERBASE="$CACHE_ROOT/python/userbase"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"

# Single-H200 defaults. Slurm normally maps the allocated GPU to visible id 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export REPO_ROOT
export DATA_DIR
export OUTPUT_DIR
export CACHE_ROOT
export TRAIN_STAGE="${TRAIN_STAGE:-strict}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
export TOTAL_STEPS
export SAVE_EVERY="${SAVE_EVERY:-50}"
export SAVE_GENERATED_IMAGES_EVERY="${SAVE_GENERATED_IMAGES_EVERY:-50}"
export LOG_EVERY="${LOG_EVERY:-1}"
export MIN_DATA_IMAGES="${MIN_DATA_IMAGES:-6000}"
export ALLOW_SMALL_DATA="${ALLOW_SMALL_DATA:-0}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-12345} % 40000)))}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

echo "[slurm] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[slurm] nproc_per_node=$NPROC_PER_NODE"
echo "[slurm] master_port=$MASTER_PORT"
echo "[slurm] attn_impl=$ATTN_IMPL"
echo "[slurm] wandb_mode=$WANDB_MODE"

if [[ "${USE_SRUN:-0}" == "1" ]]; then
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    bash "$REPO_ROOT/scripts/self_evolving/final/E1_main_joint.sh"
else
  bash "$REPO_ROOT/scripts/self_evolving/final/E1_main_joint.sh"
fi
