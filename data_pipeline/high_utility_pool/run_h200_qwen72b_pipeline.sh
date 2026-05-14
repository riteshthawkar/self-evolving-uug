#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Edit these if your paths/model choice differ.
MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"
SOURCE_DIR="${SOURCE_DIR:-data/high_utility_pool_10k/images}"
OUTPUT_DIR="${OUTPUT_DIR:-data/high_utility_pool_10k_h200_qwen3vl30b_a3b}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
HEALTH_URL="${BASE_URL%/v1}/health"

# Set VLM_MAX_IMAGES=512 and DRY_RUN=1 for a quick audit.
VLM_MAX_IMAGES="${VLM_MAX_IMAGES:-10000}"
DRY_RUN="${DRY_RUN:-0}"

# H200 serving defaults. The data builder sends VLM calls sequentially, so 4
# concurrent sequences and 4k context are enough and reduce startup/KV memory.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-2400}"

# Set START_SERVER=0 if you already started vLLM manually.
START_SERVER="${START_SERVER:-1}"
SERVER_LOG="${SERVER_LOG:-${OUTPUT_DIR}.vllm.log}"

mkdir -p "$(dirname "$OUTPUT_DIR")"

# Keep caches off $HOME. Override EXP_CACHE_ROOT to a large scratch path.
EXP_CACHE_ROOT="${EXP_CACHE_ROOT:-${OUTPUT_DIR}/cache_runtime}"
mkdir -p "$EXP_CACHE_ROOT"/{tmp,xdg_cache,xdg_config,xdg_data,huggingface,torch,triton,nvidia,pip,wandb,vllm,matplotlib,python}

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
export VLLM_RPC_BASE_PATH="$EXP_CACHE_ROOT/vllm/rpc"
export VLLM_ENGINE_READY_TIMEOUT_S
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1

export PIP_CACHE_DIR="$EXP_CACHE_ROOT/pip"
export UV_CACHE_DIR="$EXP_CACHE_ROOT/uv"
export MPLCONFIGDIR="$EXP_CACHE_ROOT/matplotlib"
export WANDB_DIR="$EXP_CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$EXP_CACHE_ROOT/wandb/cache"
export WANDB_CONFIG_DIR="$EXP_CACHE_ROOT/wandb/config"
export WANDB_DATA_DIR="$EXP_CACHE_ROOT/wandb/data"

if [[ "$START_SERVER" == "1" ]]; then
  echo "[h200] starting vLLM server: $MODEL"
  echo "[h200] cache root: $EXP_CACHE_ROOT"
  echo "[h200] server log: $SERVER_LOG"
  VLLM_IMAGE_FETCH_TIMEOUT=60 \
  vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID="$!"
  trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

  echo "[h200] waiting for server at $HEALTH_URL"
  for _ in $(seq 1 240); do
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[h200] ERROR: vLLM server exited during startup. Last log lines:" >&2
      tail -n 160 "$SERVER_LOG" >&2 || true
      exit 1
    fi
    sleep 10
  done
fi

if ! curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
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
  --vlm_backend openai_compatible \
  --vlm_model "$MODEL" \
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
