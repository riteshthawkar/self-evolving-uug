#!/usr/bin/env bash
set -euo pipefail

MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct-AWQ}"
HOST="${VLM_HOST:-0.0.0.0}"
PORT="${VLM_PORT:-8000}"
TP_SIZE="${VLM_TENSOR_PARALLEL_SIZE:-1}"
GPU_UTIL="${VLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${VLM_MAX_NUM_SEQS:-1}"
SERVED_MODEL_NAME="${VLM_SERVED_MODEL_NAME:-$MODEL}"
DTYPE="${VLM_DTYPE:-bfloat16}"
GENERATION_CONFIG="${VLM_GENERATION_CONFIG:-vllm}"
LIMIT_MM_PER_PROMPT="${VLM_LIMIT_MM_PER_PROMPT:-{\"image\":{\"count\":1,\"width\":768,\"height\":768}}}"
MM_PROCESSOR_CACHE_GB="${VLM_MM_PROCESSOR_CACHE_GB:-1}"

EXTRA_ARGS=()
if [[ "${VLM_ENFORCE_EAGER:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ "${VLM_SKIP_MM_PROFILING:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-mm-profiling)
fi
if [[ "${VLM_DISABLE_ASYNC_SCHEDULING:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--no-async-scheduling)
fi
if [[ -n "$LIMIT_MM_PER_PROMPT" ]]; then
  EXTRA_ARGS+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
fi
if [[ -n "$MM_PROCESSOR_CACHE_GB" ]]; then
  EXTRA_ARGS+=(--mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB")
fi
if [[ -n "$GENERATION_CONFIG" ]]; then
  EXTRA_ARGS+=(--generation-config "$GENERATION_CONFIG")
fi
if [[ -n "$DTYPE" ]]; then
  EXTRA_ARGS+=(--dtype "$DTYPE")
fi

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  "${EXTRA_ARGS[@]}" \
  "$@"
