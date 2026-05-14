#!/usr/bin/env bash
set -euo pipefail

MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct-AWQ}"
HOST="${VLM_HOST:-0.0.0.0}"
PORT="${VLM_PORT:-8000}"
TP_SIZE="${VLM_TENSOR_PARALLEL_SIZE:-1}"
GPU_UTIL="${VLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${VLM_MAX_NUM_SEQS:-8}"
SERVED_MODEL_NAME="${VLM_SERVED_MODEL_NAME:-$MODEL}"

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  "$@"
