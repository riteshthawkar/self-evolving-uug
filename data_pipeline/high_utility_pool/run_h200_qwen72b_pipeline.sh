#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Edit these if your paths/model choice differ.
MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
SOURCE_DIR="${SOURCE_DIR:-data/high_utility_pool_10k/images}"
OUTPUT_DIR="${OUTPUT_DIR:-data/high_utility_pool_10k_h200_qwen3vl30b_a3b_bf16}"
VLM_BACKEND="${VLM_BACKEND:-openai_compatible}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
HEALTH_URL="${BASE_URL%/v1}/health"

to_abs_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$REPO_ROOT" "$path"
  fi
}

# Set VLM_MAX_IMAGES=512 and DRY_RUN=1 for a quick audit.
VLM_MAX_IMAGES="${VLM_MAX_IMAGES:-10000}"
DRY_RUN="${DRY_RUN:-0}"

# H200 serving defaults. The data builder sends VLM calls sequentially, so 4
# concurrent sequences and 4k context are enough and reduce startup/KV memory.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-2400}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
DISABLE_ASYNC_SCHEDULING="${DISABLE_ASYNC_SCHEDULING:-1}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":{\"count\":1,\"width\":768,\"height\":768}}}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-1}"
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"
DTYPE="${DTYPE:-bfloat16}"
STARTUP_STALL_TIMEOUT_S="${STARTUP_STALL_TIMEOUT_S:-900}"
HEALTH_POLL_SECONDS="${HEALTH_POLL_SECONDS:-10}"
# Some vLLM wheels do not ship a compatible DeepGEMM build. This is mainly
# needed for FP8 checkpoints, but keeping it off is harmless for BF16 models.
VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

# Set START_SERVER=0 if you already started vLLM manually.
START_SERVER="${START_SERVER:-1}"
if [[ "$VLM_BACKEND" != "openai_compatible" ]]; then
  START_SERVER=0
fi

SOURCE_DIR="$(to_abs_path "$SOURCE_DIR")"
OUTPUT_DIR="$(to_abs_path "$OUTPUT_DIR")"
EXP_CACHE_ROOT="${EXP_CACHE_ROOT:-${OUTPUT_DIR}/cache_runtime}"
EXP_CACHE_ROOT="$(to_abs_path "$EXP_CACHE_ROOT")"
SERVER_LOG="${SERVER_LOG:-${OUTPUT_DIR}.vllm.log}"
SERVER_LOG="$(to_abs_path "$SERVER_LOG")"
RUN_USER="${USER:-$(id -un 2>/dev/null || id -u)}"
# Unix IPC socket paths have a hard ~107 character limit. vLLM appends a UUID
# to this directory, so the RPC path must stay short even when caches live in a
# longer scratch/output directory.
VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-/tmp/vllm_rpc_${RUN_USER}_${PORT}}"

mkdir -p "$(dirname "$OUTPUT_DIR")"

# Fail before starting the expensive VLM server if the image pool is missing.
if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "[h200] ERROR: SOURCE_DIR does not exist: $SOURCE_DIR" >&2
  echo "[h200] Set SOURCE_DIR to the image folder on this machine, for example:" >&2
  echo "[h200]   SOURCE_DIR=/absolute/path/to/images bash $0" >&2
  exit 2
fi
SOURCE_COUNT="$(find "$SOURCE_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' \) | wc -l | tr -d ' ')"
if [[ "$SOURCE_COUNT" == "0" ]]; then
  echo "[h200] ERROR: SOURCE_DIR contains no supported images: $SOURCE_DIR" >&2
  exit 2
fi

mkdir -p "$EXP_CACHE_ROOT"/{tmp,xdg_cache,xdg_config,xdg_data,huggingface,torch,triton,nvidia,pip,wandb,vllm,matplotlib,python,home}

# vLLM versions differ in which cache root they respect for IPC paths. Setting
# HOME is the robust option for keeping ~/.cache/vllm/rpc off the real home dir.
export REAL_HOME="${HOME:-}"
export HOME="$EXP_CACHE_ROOT/home"

export XDG_CACHE_HOME="$EXP_CACHE_ROOT/xdg_cache"
export XDG_CONFIG_HOME="$EXP_CACHE_ROOT/xdg_config"
export XDG_DATA_HOME="$EXP_CACHE_ROOT/xdg_data"
export TMPDIR="$EXP_CACHE_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PYTHONPYCACHEPREFIX="$EXP_CACHE_ROOT/python/pycache"
export PYTHONUSERBASE="$EXP_CACHE_ROOT/python/userbase"

export HF_HOME="$EXP_CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HUGGINGFACE_ASSETS_CACHE="$HF_ASSETS_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DATASETS_CACHE="$HF_DATASETS_CACHE"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME/sentence_transformers"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1

export TORCH_HOME="$EXP_CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$EXP_CACHE_ROOT/torch/extensions"
export TORCHINDUCTOR_CACHE_DIR="$EXP_CACHE_ROOT/torch/inductor"
export PYTORCH_KERNEL_CACHE_PATH="$EXP_CACHE_ROOT/torch/kernels"
export TRITON_CACHE_DIR="$EXP_CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$EXP_CACHE_ROOT/nvidia/ComputeCache"

export VLLM_CACHE_ROOT="$EXP_CACHE_ROOT/vllm/cache"
export VLLM_CONFIG_ROOT="$EXP_CACHE_ROOT/vllm/config"
export VLLM_ASSETS_CACHE="$EXP_CACHE_ROOT/vllm/assets"
export VLLM_XLA_CACHE_PATH="$EXP_CACHE_ROOT/vllm/xla_cache"
export VLLM_RPC_BASE_PATH
export VLLM_ENGINE_READY_TIMEOUT_S
export VLLM_USE_DEEP_GEMM
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1

export PIP_CACHE_DIR="$EXP_CACHE_ROOT/pip"
export UV_CACHE_DIR="$EXP_CACHE_ROOT/uv"
export MPLCONFIGDIR="$EXP_CACHE_ROOT/matplotlib"
export WANDB_DIR="$EXP_CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$EXP_CACHE_ROOT/wandb/cache"
export WANDB_CONFIG_DIR="$EXP_CACHE_ROOT/wandb/config"
export WANDB_DATA_DIR="$EXP_CACHE_ROOT/wandb/data"

mkdir -p \
  "$HOME/.cache/vllm/rpc" \
  "$XDG_CACHE_HOME/vllm/rpc" \
  "$VLLM_CACHE_ROOT" \
  "$VLLM_CONFIG_ROOT" \
  "$VLLM_ASSETS_CACHE" \
  "$VLLM_XLA_CACHE_PATH" \
  "$VLLM_RPC_BASE_PATH"

if [[ "$START_SERVER" == "1" ]]; then
  echo "[h200] starting vLLM server: $MODEL"
  echo "[h200] source images: $SOURCE_DIR ($SOURCE_COUNT files)"
  echo "[h200] VLM backend: $VLM_BACKEND"
  echo "[h200] cache root: $EXP_CACHE_ROOT"
  echo "[h200] HOME redirected to: $HOME"
  echo "[h200] VLLM_RPC_BASE_PATH: $VLLM_RPC_BASE_PATH"
  echo "[h200] VLLM_USE_DEEP_GEMM: $VLLM_USE_DEEP_GEMM"
  echo "[h200] ENFORCE_EAGER: $ENFORCE_EAGER"
  echo "[h200] SKIP_MM_PROFILING: $SKIP_MM_PROFILING"
  echo "[h200] DISABLE_ASYNC_SCHEDULING: $DISABLE_ASYNC_SCHEDULING"
  echo "[h200] LIMIT_MM_PER_PROMPT: $LIMIT_MM_PER_PROMPT"
  echo "[h200] MM_PROCESSOR_CACHE_GB: $MM_PROCESSOR_CACHE_GB"
  echo "[h200] server log: $SERVER_LOG"
  VLLM_ARGS=()
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    VLLM_ARGS+=(--enforce-eager)
  fi
  if [[ "$SKIP_MM_PROFILING" == "1" ]]; then
    VLLM_ARGS+=(--skip-mm-profiling)
  fi
  if [[ "$DISABLE_ASYNC_SCHEDULING" == "1" ]]; then
    VLLM_ARGS+=(--no-async-scheduling)
  fi
  if [[ -n "$LIMIT_MM_PER_PROMPT" ]]; then
    VLLM_ARGS+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
  fi
  if [[ -n "$MM_PROCESSOR_CACHE_GB" ]]; then
    VLLM_ARGS+=(--mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB")
  fi
  if [[ -n "$GENERATION_CONFIG" ]]; then
    VLLM_ARGS+=(--generation-config "$GENERATION_CONFIG")
  fi
  if [[ -n "$DTYPE" ]]; then
    VLLM_ARGS+=(--dtype "$DTYPE")
  fi
  VLLM_IMAGE_FETCH_TIMEOUT=60 \
  vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    "${VLLM_ARGS[@]}" \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID="$!"
  trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

  echo "[h200] waiting for server at $HEALTH_URL"
  LAST_LOG_SUM=""
  LAST_LOG_CHANGE_TS="$(date +%s)"
  for i in $(seq 1 240); do
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
      echo "[h200] vLLM server healthy after $((i * HEALTH_POLL_SECONDS))s"
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[h200] ERROR: vLLM server exited during startup. Last log lines:" >&2
      tail -n 160 "$SERVER_LOG" >&2 || true
      exit 1
    fi
    if [[ -f "$SERVER_LOG" ]]; then
      CURRENT_LOG_SUM="$(tail -n 20 "$SERVER_LOG" | cksum 2>/dev/null | awk '{print $1}' || true)"
      if [[ -n "$CURRENT_LOG_SUM" && "$CURRENT_LOG_SUM" != "$LAST_LOG_SUM" ]]; then
        LAST_LOG_SUM="$CURRENT_LOG_SUM"
        LAST_LOG_CHANGE_TS="$(date +%s)"
      fi
      NOW_TS="$(date +%s)"
      if (( NOW_TS - LAST_LOG_CHANGE_TS > STARTUP_STALL_TIMEOUT_S )); then
        echo "[h200] ERROR: vLLM startup log has not advanced for $STARTUP_STALL_TIMEOUT_S seconds." >&2
        echo "[h200] This usually means the server is stuck in multimodal profiling/warmup, not checkpoint loading." >&2
        echo "[h200] Last log lines:" >&2
        tail -n 160 "$SERVER_LOG" >&2 || true
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
      fi
    fi
    if (( i % 3 == 0 )); then
      echo "[h200] waiting $((i * HEALTH_POLL_SECONDS))s for vLLM startup; latest server log:"
      tail -n 8 "$SERVER_LOG" || true
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
      fi
    fi
    sleep "$HEALTH_POLL_SECONDS"
  done
fi

if [[ "$VLM_BACKEND" == "openai_compatible" ]] && ! curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
  echo "[h200] ERROR: vLLM server is not healthy after waiting. Last log lines:" >&2
  tail -n 160 "$SERVER_LOG" >&2 || true
  exit 1
fi

EXTRA_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
  EXTRA_ARGS+=(--dry_run)
fi

echo "[h200] running VLM data audit"
python3 data_pipeline/high_utility_pool/build_high_utility_pool.py \
  --local_source "$SOURCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --target_count 10000 \
  --no_openimages \
  --min_heuristic_score 0.1 \
  --vlm_backend "$VLM_BACKEND" \
  --vlm_model "$MODEL" \
  --vlm_dtype "$DTYPE" \
  --vlm_base_url "$BASE_URL" \
  --vlm_timeout 240 \
  --vlm_max_images "$VLM_MAX_IMAGES" \
  --vlm_candidate_top_k 10000 \
  --vlm_selection_strategy stratified \
  --download_workers 8 \
  --vlm_sleep_sec 0 \
  "${EXTRA_ARGS[@]}"

echo "[h200] done"
echo "[h200] output: $OUTPUT_DIR"
echo "[h200] audit report: $OUTPUT_DIR/audit_report.json"
echo "[h200] VLM scores: $OUTPUT_DIR/scores/vlm_scores.jsonl"
