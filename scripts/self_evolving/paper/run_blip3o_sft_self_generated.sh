#!/usr/bin/env bash
set -euo pipefail

# Full-parameter SFT baseline for Table 6: train BLIP3o on self-generated
# prompt/image pairs packaged as webdataset shards.

BOOTSTRAP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_SEARCH_DIR="${BOOTSTRAP_DIR}"
while [[ "${BOOTSTRAP_SEARCH_DIR}" != "/" ]]; do
  if [[ -f "${BOOTSTRAP_SEARCH_DIR}/scripts/env/bootstrap_training_env.sh" ]]; then
    # shellcheck source=/dev/null
    source "${BOOTSTRAP_SEARCH_DIR}/scripts/env/bootstrap_training_env.sh"
    break
  fi
  BOOTSTRAP_SEARCH_DIR="$(dirname "${BOOTSTRAP_SEARCH_DIR}")"
done
unset BOOTSTRAP_DIR BOOTSTRAP_SEARCH_DIR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/blip3o/param_strategy/sft_self_generated}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-BLIP3o/BLIP3o-Model-8B}"
GENERATED_MIX_DIR="${GENERATED_MIX_DIR:-$REPO_ROOT/outputs/blip3o/E5_synthetic_loop/generated_mix_pool}"
SFT_WEBDATASET_DIR="${SFT_WEBDATASET_DIR:-$OUTPUT_DIR/webdataset}"
MAX_SFT_SAMPLES="${MAX_SFT_SAMPLES:-0}"
MIN_SFT_REWARD="${MIN_SFT_REWARD:-}"
SFT_IMAGE_SIZE="${SFT_IMAGE_SIZE:-896}"
SFT_SHARD_SIZE="${SFT_SHARD_SIZE:-1000}"
NPROC_DATASET="${NPROC_DATASET:-128}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$REPO_ROOT/BLIP3o/deepspeed_scripts/zero1.json}"
BITS="${BITS:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
RUN_NAME="${RUN_NAME:-blip3o_sft_self_generated_s42}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$REPO_ROOT/BLIP3o"
export HF_HOME="${HF_HOME:-$REPO_ROOT/cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false

if ! compgen -G "$SFT_WEBDATASET_DIR/*.tar" >/dev/null; then
  if [[ ! -d "$GENERATED_MIX_DIR" ]]; then
    echo "[SFT] ERROR: no SFT shards in $SFT_WEBDATASET_DIR and GENERATED_MIX_DIR does not exist: $GENERATED_MIX_DIR" >&2
    echo "[SFT] Run a generation/self-generated experiment first, or set SFT_WEBDATASET_DIR to existing .tar shards." >&2
    exit 1
  fi
  PREPARE_ARGS=(
    --generated-dir "$GENERATED_MIX_DIR"
    --output-dir "$SFT_WEBDATASET_DIR"
    --shard-size "$SFT_SHARD_SIZE"
    --image-size "$SFT_IMAGE_SIZE"
    --overwrite
  )
  if [[ "$MAX_SFT_SAMPLES" != "0" ]]; then
    PREPARE_ARGS+=(--max-samples "$MAX_SFT_SAMPLES")
  fi
  if [[ -n "$MIN_SFT_REWARD" ]]; then
    PREPARE_ARGS+=(--min-reward "$MIN_SFT_REWARD")
  fi
  "$PYTHON_BIN" "$REPO_ROOT/scripts/self_evolving/paper/prepare_self_generated_sft_webdataset.py" "${PREPARE_ARGS[@]}"
fi

echo "[SFT] Starting BLIP3o self-generated-data SFT baseline"
echo "[SFT]   Webdataset: $SFT_WEBDATASET_DIR"
echo "[SFT]   Output:     $OUTPUT_DIR"
echo "[SFT]   Model:      $MODEL_NAME_OR_PATH"
echo "[SFT]   GPUs:       $NPROC_PER_NODE"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  "$REPO_ROOT/BLIP3o/blip3o/train/train_mem.py" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --version qwen \
  --freeze_backbone False \
  --data_type "mix" \
  --image_folder "$SFT_WEBDATASET_DIR" \
  --gen_vision_tower eva-clip-E-14-plus \
  --gen_projector_type mlp2x_gelu \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --bits "$BITS" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 1 \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --warmup_ratio 0.003 \
  --lr_scheduler_type cosine \
  --model_max_length "$MODEL_MAX_LENGTH" \
  --logging_steps 1 \
  --tf32 True \
  --gradient_checkpointing True \
  --dataloader_num_workers "$NPROC_DATASET" \
  --lazy_preprocess True \
  --gen_pooling early_pool2d_4 \
  --n_query 64 \
  --n_und_query 0 \
  --report_to none \
  --run_name "$RUN_NAME"
