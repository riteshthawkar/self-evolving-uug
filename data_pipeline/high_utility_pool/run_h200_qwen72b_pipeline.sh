#!/usr/bin/env bash
set -euo pipefail

# Edit these if your paths/model choice differ.
MODEL="${MODEL:-Qwen/Qwen2.5-VL-72B-Instruct-AWQ}"
SOURCE_DIR="${SOURCE_DIR:-data/high_utility_pool_10k/images}"
OUTPUT_DIR="${OUTPUT_DIR:-data/high_utility_pool_10k_h200_qwen72b}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Set VLM_MAX_IMAGES=512 and DRY_RUN=1 for a quick audit.
VLM_MAX_IMAGES="${VLM_MAX_IMAGES:-10000}"
DRY_RUN="${DRY_RUN:-0}"

# H200 serving defaults. Increase MAX_NUM_SEQS if you confirm stable memory.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

# Set START_SERVER=0 if you already started vLLM manually.
START_SERVER="${START_SERVER:-1}"
SERVER_LOG="${SERVER_LOG:-${OUTPUT_DIR}.vllm.log}"

mkdir -p "$(dirname "$OUTPUT_DIR")"

if [[ "$START_SERVER" == "1" ]]; then
  echo "[h200] starting vLLM server: $MODEL"
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

  echo "[h200] waiting for server at $BASE_URL/health"
  for _ in $(seq 1 120); do
    if curl -fsS "${BASE_URL%/v1}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 10
  done
fi

if ! curl -fsS "${BASE_URL%/v1}/health" >/dev/null 2>&1; then
  echo "[h200] ERROR: vLLM server is not healthy. Check $SERVER_LOG" >&2
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
