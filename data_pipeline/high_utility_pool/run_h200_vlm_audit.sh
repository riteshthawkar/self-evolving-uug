#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
VLM_BACKEND="${VLM_BACKEND:-transformers_vlm}"
BASE_URL="${VLM_BASE_URL:-http://127.0.0.1:8000/v1}"
SOURCE_DIR="${SOURCE_DIR:-data/high_utility_pool_10k/images}"
OUTPUT_DIR="${OUTPUT_DIR:-data/high_utility_pool_10k_h200_vlm_audit}"
TARGET_COUNT="${TARGET_COUNT:-10000}"
VLM_MAX_IMAGES="${VLM_MAX_IMAGES:-10000}"
VLM_CANDIDATE_TOP_K="${VLM_CANDIDATE_TOP_K:-10000}"
VLM_TIMEOUT="${VLM_TIMEOUT:-240}"
MIN_HEURISTIC_SCORE="${MIN_HEURISTIC_SCORE:-0.1}"
VLM_DTYPE="${VLM_DTYPE:-auto}"
VLM_DEVICE_MAP="${VLM_DEVICE_MAP:-auto}"
VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-}"

exec "$PYTHON_BIN" data_pipeline/high_utility_pool/build_high_utility_pool.py \
  --local_source "$SOURCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --target_count "$TARGET_COUNT" \
  --no_openimages \
  --min_heuristic_score "$MIN_HEURISTIC_SCORE" \
  --vlm_backend "$VLM_BACKEND" \
  --vlm_model "$MODEL" \
  --vlm_dtype "$VLM_DTYPE" \
  --vlm_device_map "$VLM_DEVICE_MAP" \
  --vlm_attn_implementation "$VLM_ATTN_IMPLEMENTATION" \
  --vlm_base_url "$BASE_URL" \
  --vlm_timeout "$VLM_TIMEOUT" \
  --vlm_max_images "$VLM_MAX_IMAGES" \
  --vlm_candidate_top_k "$VLM_CANDIDATE_TOP_K" \
  --vlm_selection_strategy stratified \
  --download_workers 8 \
  --vlm_sleep_sec 0 \
  "$@"
