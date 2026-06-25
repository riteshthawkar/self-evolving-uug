#!/usr/bin/env bash
set -euo pipefail

# Single-entry launcher for paper/rebuttal training experiments.
#
# Usage:
#   bash scripts/self_evolving/paper/run_experiment.sh blip3o_joint
#   DRY_RUN=1 bash scripts/self_evolving/paper/run_experiment.sh bagel_joint
#   DATA_DIR=/path/to/unlabeled/images NPROC_PER_NODE=8 \
#     bash scripts/self_evolving/paper/run_experiment.sh blip3o_two_stage

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

EXPERIMENT_ID="${1:-}"
if [[ -z "$EXPERIMENT_ID" ]]; then
  echo "Usage: bash scripts/self_evolving/paper/run_experiment.sh EXPERIMENT_ID" >&2
  echo "Try: blip3o_joint, blip3o_two_stage, bagel_joint, vargpt_joint" >&2
  exit 2
fi

if [[ -n "${DATA_DIR+x}" ]]; then
  USER_SUPPLIED_DATA_DIR="$DATA_DIR"
else
  USER_SUPPLIED_DATA_DIR=""
fi
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/joint_pool_10k/images}"
TWO_STAGE_DATA_DIR="${TWO_STAGE_DATA_DIR:-${USER_SUPPLIED_DATA_DIR:-$REPO_ROOT/data/joint_pool_10k/images}}"
TWO_STAGE_IMAGE_SAMPLES="${TWO_STAGE_IMAGE_SAMPLES:-10000}"
TWO_STAGE_UNDERSTANDING_STEPS="${TWO_STAGE_UNDERSTANDING_STEPS:-10000}"
TWO_STAGE_GENERATION_STEPS="${TWO_STAGE_GENERATION_STEPS:-10000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_GPUS="${NUM_GPUS:-$NPROC_PER_NODE}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_MISSING_DATA="${ALLOW_MISSING_DATA:-0}"
TRAIN_STAGE="${TRAIN_STAGE:-strict}"

count_images() {
  if [[ ! -d "$1" ]]; then
    echo 0
    return
  fi
  find "$1" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' \) \
    | wc -l | tr -d ' '
}

run_cmd() {
  printf '[run_experiment]'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

preflight_data() {
  if [[ "$ALLOW_MISSING_DATA" == "1" || "$DRY_RUN" == "1" ]]; then
    return
  fi
  if [[ ! -d "$DATA_DIR" ]]; then
    echo "[run_experiment] ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
    echo "[run_experiment] Set DATA_DIR to a local directory of unlabeled training images." >&2
    exit 1
  fi
  local n
  n="$(count_images "$DATA_DIR")"
  if [[ "$n" -lt 10000 ]]; then
    echo "[run_experiment] ERROR: DATA_DIR has $n images, expected at least 10000: $DATA_DIR" >&2
    exit 1
  fi
}

preflight_model_path() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[run_experiment] WARNING: $name is unset for $EXPERIMENT_ID; dry-run will show an empty value." >&2
      return
    fi
    echo "[run_experiment] ERROR: $name is required for $EXPERIMENT_ID." >&2
    exit 1
  fi
  if [[ "$DRY_RUN" != "1" && ! -e "$value" ]]; then
    echo "[run_experiment] ERROR: $name does not exist: $value" >&2
    exit 1
  fi
}

launch_blip3o() {
  local script="$1"
  local out="$2"
  shift 2
  preflight_data
  run_cmd env \
    DATA_DIR="$DATA_DIR" \
    OUTPUT_DIR="$out" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    TRAIN_STAGE="$TRAIN_STAGE" \
    "$@" \
    bash "$script"
}

launch_blip3o_variant() {
  local out_suffix="$1"
  shift
  launch_blip3o \
    "$REPO_ROOT/scripts/self_evolving/final/E1_main_joint.sh" \
    "$OUTPUT_ROOT/blip3o/$out_suffix" \
    "$@"
}

launch_bagel() {
  local out="$1"
  shift
  local model_path="${MODEL_PATH:-}"
  preflight_data
  preflight_model_path MODEL_PATH "$model_path"
  run_cmd env \
    DATA_DIR="$DATA_DIR" \
    OUTPUT_DIR="$out" \
    MODEL_PATH="$model_path" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    RESUME_FROM= \
    LORA_CHECKPOINT_PATH= \
    NUM_GPUS="$NUM_GPUS" \
    TRAIN_STAGE="$TRAIN_STAGE" \
    "$@" \
    bash "$REPO_ROOT/Bagel/scripts/B1_unified_training.sh"
}

launch_vargpt() {
  local mode="$1"
  local out="$2"
  preflight_data
  run_cmd env \
    IMAGE_FOLDER="$DATA_DIR" \
    OUTPUT_DIR="$out" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    NUM_GPUS="$NUM_GPUS" \
    RESUME_FROM= \
    bash "$REPO_ROOT/vargpt_1_1/VARGPT-family-training/examples/train_self_evolving/run_self_evolving.sh" \
    "$mode" "$NUM_GPUS"
}

case "$EXPERIMENT_ID" in
  blip3o_joint)
    launch_blip3o "$REPO_ROOT/scripts/self_evolving/final/E1_main_joint.sh" "$OUTPUT_ROOT/blip3o/E1_main_joint"
    ;;
  blip3o_understanding_only)
    launch_blip3o "$REPO_ROOT/scripts/self_evolving/final/E2_understanding_only.sh" "$OUTPUT_ROOT/blip3o/E2_understanding_only"
    ;;
  blip3o_generation_only)
    launch_blip3o "$REPO_ROOT/scripts/self_evolving/final/E3_generation_only.sh" "$OUTPUT_ROOT/blip3o/E3_generation_only"
    ;;
  blip3o_no_dit_rwr)
    launch_blip3o "$REPO_ROOT/scripts/self_evolving/final/E4_no_dit_rwr.sh" "$OUTPUT_ROOT/blip3o/E4_no_dit_rwr"
    ;;
  blip3o_synthetic_loop)
    launch_blip3o "$REPO_ROOT/scripts/self_evolving/final/E5_synthetic_loop.sh" "$OUTPUT_ROOT/blip3o/E5_synthetic_loop"
    ;;
  blip3o_single_step)
    launch_blip3o "$REPO_ROOT/scripts/self_evolving/final/E6_single_step.sh" "$OUTPUT_ROOT/blip3o/E6_single_step"
    ;;
  blip3o_two_stage)
    DATA_DIR="$TWO_STAGE_DATA_DIR" launch_blip3o \
      "$REPO_ROOT/scripts/self_evolving/final/E7_two_stage.sh" \
      "$OUTPUT_ROOT/blip3o/E7_two_stage" \
      TWO_STAGE_IMAGE_SAMPLES="$TWO_STAGE_IMAGE_SAMPLES" \
      STAGE1_STEPS="$TWO_STAGE_UNDERSTANDING_STEPS" \
      STAGE2_STEPS="$TWO_STAGE_GENERATION_STEPS"
    ;;
  blip3o_ablate_no_ste)
    launch_blip3o_variant "ablations/no_ste" SOLVER_TOKEN_ENTROPY_ENABLED=0
    ;;
  blip3o_ablate_no_self_consistency)
    launch_blip3o_variant "ablations/no_self_consistency" PROPOSER_SAMPLE_ENTROPY_WEIGHT=0.0
    ;;
  blip3o_ablate_no_prompt_perturbation)
    launch_blip3o_variant "ablations/no_prompt_perturbation" SOLVER_PPS_ENABLED=0
    ;;
  blip3o_ablate_no_qa_fidelity)
    launch_blip3o_variant "ablations/no_qa_fidelity" REWARD_SPEC_WEIGHT=0.0
    ;;
  blip3o_ablate_no_cycle_consistency)
    launch_blip3o_variant "ablations/no_cycle_consistency" REWARD_CYCLE_WEIGHT=0.0
    ;;
  blip3o_ste_mean)
    launch_blip3o_variant "rebuttal/ste_mean" SOLVER_TOKEN_ENTROPY_AGGREGATION=mean
    ;;
  blip3o_strategy_lora)
    launch_blip3o_variant "param_strategy/lora" USE_LORA=1 LOAD_IN_4BIT=0
    ;;
  blip3o_strategy_qlora)
    launch_blip3o_variant "param_strategy/qlora_4bit" USE_LORA=1 LOAD_IN_4BIT=1
    ;;
  blip3o_strategy_full_finetune)
    launch_blip3o_variant "param_strategy/full_finetune" USE_LORA=0 LOAD_IN_4BIT=0
    ;;
  blip3o_strategy_sft_self_generated)
    run_cmd env \
      DATA_DIR="$DATA_DIR" \
      OUTPUT_DIR="$OUTPUT_ROOT/blip3o/param_strategy/sft_self_generated" \
      NPROC_PER_NODE="$NPROC_PER_NODE" \
      bash "$REPO_ROOT/scripts/self_evolving/paper/run_blip3o_sft_self_generated.sh"
    ;;
  blip3o_lora_r4)
    launch_blip3o_variant "sweeps/lora_r4" LORA_R=4
    ;;
  blip3o_lora_r8)
    launch_blip3o_variant "sweeps/lora_r8" LORA_R=8
    ;;
  blip3o_lora_r16)
    launch_blip3o_variant "sweeps/lora_r16" LORA_R=16
    ;;
  blip3o_lora_r32)
    launch_blip3o_variant "sweeps/lora_r32" LORA_R=32
    ;;
  blip3o_lora_r64)
    launch_blip3o_variant "sweeps/lora_r64" LORA_R=64
    ;;
  blip3o_pps_n3)
    launch_blip3o_variant "sweeps/pps_n3" NUM_SOLVER_SAMPLES=3
    ;;
  blip3o_pps_n5)
    launch_blip3o_variant "sweeps/pps_n5" NUM_SOLVER_SAMPLES=5
    ;;
  blip3o_pps_n7)
    launch_blip3o_variant "sweeps/pps_n7" NUM_SOLVER_SAMPLES=7
    ;;
  blip3o_pps_n9)
    launch_blip3o_variant "sweeps/pps_n9" NUM_SOLVER_SAMPLES=9
    ;;
  blip3o_pps_n11)
    launch_blip3o_variant "sweeps/pps_n11" NUM_SOLVER_SAMPLES=11
    ;;
  blip3o_proposer_k1)
    launch_blip3o_variant "sweeps/proposer_k1" PROPOSER_NUM_CANDIDATES=1
    ;;
  blip3o_proposer_k3)
    launch_blip3o_variant "sweeps/proposer_k3" PROPOSER_NUM_CANDIDATES=3
    ;;
  blip3o_proposer_k5)
    launch_blip3o_variant "sweeps/proposer_k5" PROPOSER_NUM_CANDIDATES=5
    ;;
  blip3o_ste_w64)
    launch_blip3o_variant "sweeps/ste_w64" SOLVER_TOKEN_ENTROPY_WINDOW_SIZE=64
    ;;
  blip3o_ste_w128)
    launch_blip3o_variant "sweeps/ste_w128" SOLVER_TOKEN_ENTROPY_WINDOW_SIZE=128
    ;;
  blip3o_ste_w256)
    launch_blip3o_variant "sweeps/ste_w256" SOLVER_TOKEN_ENTROPY_WINDOW_SIZE=256
    ;;
  blip3o_lr_5e7)
    launch_blip3o_variant "sweeps/lr_5e-7" LR=5e-7
    ;;
  blip3o_lr_1e6)
    launch_blip3o_variant "sweeps/lr_1e-6" LR=1e-6
    ;;
  blip3o_lr_2e6)
    launch_blip3o_variant "sweeps/lr_2e-6" LR=2e-6
    ;;
  blip3o_wd_0)
    launch_blip3o_variant "sweeps/wd_0" WEIGHT_DECAY=0.0
    ;;
  blip3o_wd_0p01)
    launch_blip3o_variant "sweeps/wd_0p01" WEIGHT_DECAY=0.01
    ;;
  blip3o_wd_0p05)
    launch_blip3o_variant "sweeps/wd_0p05" WEIGHT_DECAY=0.05
    ;;
  blip3o_dropout_0)
    launch_blip3o_variant "sweeps/dropout_0" LORA_DROPOUT=0.0
    ;;
  blip3o_dropout_0p05)
    launch_blip3o_variant "sweeps/dropout_0p05" LORA_DROPOUT=0.05
    ;;
  blip3o_dropout_0p10)
    launch_blip3o_variant "sweeps/dropout_0p10" LORA_DROPOUT=0.10
    ;;
  blip3o_kl_0p005)
    launch_blip3o_variant "sweeps/kl_0p005" KL_COEF=0.005
    ;;
  blip3o_kl_0p01)
    launch_blip3o_variant "sweeps/kl_0p01" KL_COEF=0.01
    ;;
  blip3o_kl_0p020)
    launch_blip3o_variant "sweeps/kl_0p020" KL_COEF=0.020
    ;;
  blip3o_gen_l1)
    launch_blip3o_variant "sweeps/gen_l1" NUM_GENERATIONS=1
    ;;
  blip3o_gen_l3)
    launch_blip3o_variant "sweeps/gen_l3" NUM_GENERATIONS=3
    ;;
  blip3o_gen_l5)
    launch_blip3o_variant "sweeps/gen_l5" NUM_GENERATIONS=5
    ;;
  blip3o_min_qa1)
    launch_blip3o_variant "sweeps/min_qa1" MIN_SPEC_QA_PAIRS=1
    ;;
  blip3o_min_qa2)
    launch_blip3o_variant "sweeps/min_qa2" MIN_SPEC_QA_PAIRS=2
    ;;
  blip3o_min_qa3)
    launch_blip3o_variant "sweeps/min_qa3" MIN_SPEC_QA_PAIRS=3
    ;;
  blip3o_reward_qa_0p50)
    launch_blip3o_variant "sweeps/reward_qa_0p50" REWARD_SPEC_WEIGHT=0.50
    ;;
  blip3o_reward_qa_0p65)
    launch_blip3o_variant "sweeps/reward_qa_0p65" REWARD_SPEC_WEIGHT=0.65
    ;;
  blip3o_reward_qa_0p80)
    launch_blip3o_variant "sweeps/reward_qa_0p80" REWARD_SPEC_WEIGHT=0.80
    ;;
  blip3o_reward_cycle_0p10)
    launch_blip3o_variant "sweeps/reward_cycle_0p10" REWARD_CYCLE_WEIGHT=0.10
    ;;
  blip3o_reward_cycle_0p20)
    launch_blip3o_variant "sweeps/reward_cycle_0p20" REWARD_CYCLE_WEIGHT=0.20
    ;;
  blip3o_reward_cycle_0p30)
    launch_blip3o_variant "sweeps/reward_cycle_0p30" REWARD_CYCLE_WEIGHT=0.30
    ;;
  blip3o_min_spec_quality_0p25)
    launch_blip3o_variant "sweeps/min_spec_quality_0p25" MIN_SPEC_QUALITY_FOR_UPDATE=0.25
    ;;
  blip3o_min_spec_quality_0p35)
    launch_blip3o_variant "sweeps/min_spec_quality_0p35" MIN_SPEC_QUALITY_FOR_UPDATE=0.35
    ;;
  blip3o_min_spec_quality_0p45)
    launch_blip3o_variant "sweeps/min_spec_quality_0p45" MIN_SPEC_QUALITY_FOR_UPDATE=0.45
    ;;
  bagel_joint)
    launch_bagel "$OUTPUT_ROOT/bagel/B1_unified_training"
    ;;
  bagel_understanding_only)
    launch_bagel "$OUTPUT_ROOT/bagel/B2_understanding_only" \
      UNDERSTANDING_STEPS_PER_CYCLE=5 GENERATION_STEPS_PER_CYCLE=0 \
      TRAIN_GENERATOR=0 TRAIN_GENERATION_PROPOSER=0 DIT_UPDATE_ENABLED=0
    ;;
  bagel_generation_only)
    launch_bagel "$OUTPUT_ROOT/bagel/B3_generation_only" \
      UNDERSTANDING_STEPS_PER_CYCLE=0 GENERATION_STEPS_PER_CYCLE=5 \
      TRAIN_SOLVER=0 TRAIN_UNDERSTANDING_PROPOSER=0
    ;;
  vargpt_joint)
    launch_vargpt joint "$OUTPUT_ROOT/vargpt/joint"
    ;;
  vargpt_understanding_only)
    launch_vargpt u_only "$OUTPUT_ROOT/vargpt/u_only"
    ;;
  vargpt_generation_only)
    launch_vargpt gen_only "$OUTPUT_ROOT/vargpt/gen_only"
    ;;
  *)
    echo "[run_experiment] ERROR: unknown experiment id: $EXPERIMENT_ID" >&2
    echo "[run_experiment] See scripts/self_evolving/paper/paper_experiments.json for valid ids." >&2
    exit 2
    ;;
esac
