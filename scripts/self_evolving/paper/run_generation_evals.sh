#!/usr/bin/env bash
set -euo pipefail

# Paper/rebuttal generation benchmark launcher.
#
# Examples:
#   CHECKPOINT_DIR=/path/to/blip3o/step_10000 \
#     bash scripts/self_evolving/paper/run_generation_evals.sh blip3o
#
#   MODEL_PATH=/path/to/BAGEL-7B-MoT \
#   OUTPUT_ROOT=outputs/evals/bagel_base \
#     BENCHMARKS=geneval,wise bash scripts/self_evolving/paper/run_generation_evals.sh bagel
#
#   CHECKPOINT_DIR=/path/to/vargpt/se_checkpoint_10000 \
#     bash scripts/self_evolving/paper/run_generation_evals.sh vargpt

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

BACKBONE="${1:-${BACKBONE:-blip3o}}"
BACKBONE="$(printf '%s' "$BACKBONE" | tr '[:upper:]' '[:lower:]')"
BENCHMARKS="${BENCHMARKS:-geneval,dpg,wise}"
MODE="${MODE:-ours}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/paper_evals/${BACKBONE}/${MODE}}"
N_CHUNKS="${N_CHUNKS:-8}"
STEPS="${STEPS:-50}"
DPG_RESOLUTION="${DPG_RESOLUTION:-896}"
DPG_PIC_NUM="${DPG_PIC_NUM:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_SCORING="${RUN_SCORING:-1}"

IFS=',' read -r -a BENCH_ARRAY <<< "$BENCHMARKS"

has_benchmark() {
  local needle="$1"
  local item
  for item in "${BENCH_ARRAY[@]}"; do
    item="${item//[[:space:]]/}"
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

run_cmd() {
  printf '[run_generation_evals]'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

require_checkpoint_for_ours() {
  if [[ "$MODE" == "ours" && -z "$CHECKPOINT_DIR" ]]; then
    echo "[run_generation_evals] ERROR: CHECKPOINT_DIR is required when MODE=ours." >&2
    exit 1
  fi
}

run_blip3o() {
  require_checkpoint_for_ours
  mkdir -p "$OUTPUT_ROOT"

  if has_benchmark geneval; then
    local out="$OUTPUT_ROOT/geneval"
    if [[ "$MODE" == "base" ]]; then
      echo "[run_generation_evals] BLIP3o GenEval base -> $out"
      run_cmd env OUTDIR="$out" N_CHUNKS="$N_CHUNKS" STEPS="$STEPS" \
        bash "$REPO_ROOT/BLIP3o/eval/geneval/generation.sh"
    else
      echo "[run_generation_evals] BLIP3o GenEval ours -> $out"
      run_cmd env CHECKPOINT_DIR="$CHECKPOINT_DIR" OUTDIR="$out" N_CHUNKS="$N_CHUNKS" STEPS="$STEPS" \
        bash "$REPO_ROOT/BLIP3o/eval/geneval/generation_our.sh"
    fi
    echo "[run_generation_evals] GenEval generation complete. Score with a GenEval evaluator compatible with BLIP3o outputs."
  fi

  if has_benchmark dpg; then
    local out="$OUTPUT_ROOT/dpg_bench_images"
    if [[ "$MODE" == "base" ]]; then
      echo "[run_generation_evals] BLIP3o DPG-Bench base generation -> $out"
      run_cmd env OUTDIR="$out" N_CHUNKS="$N_CHUNKS" STEPS="$STEPS" \
        bash "$REPO_ROOT/BLIP3o/eval/dpg_bench/generate_dpg_base.sh"
    else
      echo "[run_generation_evals] BLIP3o DPG-Bench ours generation -> $out"
      run_cmd env CHECKPOINT_DIR="$CHECKPOINT_DIR" OUTDIR="$out" N_CHUNKS="$N_CHUNKS" STEPS="$STEPS" \
        bash "$REPO_ROOT/BLIP3o/eval/dpg_bench/generate_dpg_our.sh"
    fi
    if [[ "$RUN_SCORING" == "1" ]]; then
      run_cmd bash "$REPO_ROOT/BLIP3o/eval/dpg_bench/evaluate_dpg.sh" \
        "$out" "$DPG_RESOLUTION" "$DPG_PIC_NUM" "$N_CHUNKS"
    fi
  fi

  if has_benchmark wise; then
    local out="$OUTPUT_ROOT/wise_images"
    if [[ "$MODE" == "base" ]]; then
      echo "[run_generation_evals] BLIP3o WISE base generation -> $out"
      run_cmd env OUTDIR="$out" N_CHUNKS="$N_CHUNKS" STEPS="$STEPS" \
        bash "$REPO_ROOT/BLIP3o/eval/wise/generate_wise_base.sh"
    else
      echo "[run_generation_evals] BLIP3o WISE ours generation -> $out"
      run_cmd env CHECKPOINT_DIR="$CHECKPOINT_DIR" OUTDIR="$out" N_CHUNKS="$N_CHUNKS" STEPS="$STEPS" \
        bash "$REPO_ROOT/BLIP3o/eval/wise/generate_wise_our.sh"
    fi
    if [[ "$RUN_SCORING" == "1" ]]; then
      run_cmd bash "$REPO_ROOT/BLIP3o/eval/wise/evaluate_wise.sh" "$out"
    fi
  fi
}

run_bagel() {
  local model_path_value="${MODEL_PATH:-${model_path:-$CHECKPOINT_DIR}}"
  if [[ -z "$model_path_value" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      model_path_value="/path/to/BAGEL-7B-MoT"
    else
      echo "[run_generation_evals] ERROR: MODEL_PATH or CHECKPOINT_DIR is required for BAGEL." >&2
      exit 1
    fi
  fi
  mkdir -p "$OUTPUT_ROOT"
  pushd "$REPO_ROOT/Bagel" >/dev/null
  if has_benchmark geneval; then
    echo "[run_generation_evals] BAGEL GenEval -> $OUTPUT_ROOT/geneval"
    run_cmd env model_path="$model_path_value" output_path="$OUTPUT_ROOT/geneval" \
      bash "$REPO_ROOT/Bagel/scripts/eval/run_geneval.sh"
  fi
  if has_benchmark wise; then
    echo "[run_generation_evals] BAGEL WISE -> $OUTPUT_ROOT/wise"
    run_cmd env model_path="$model_path_value" output_path="$OUTPUT_ROOT/wise" \
      bash "$REPO_ROOT/Bagel/scripts/eval/run_wise.sh"
  fi
  if has_benchmark dpg; then
    echo "[run_generation_evals] WARNING: BAGEL DPG-Bench launcher is not present in this checkout; use BLIP3o DPG tooling or add a BAGEL-compatible DPG generator before claiming BAGEL DPG results." >&2
  fi
  popd >/dev/null
}

run_vargpt() {
  require_checkpoint_for_ours
  mkdir -p "$OUTPUT_ROOT"
  local script="$REPO_ROOT/vargpt_1_1/VARGPT-family-training/run_scripts/run_eval_vargpt_generation_our.sh"
  if [[ "$MODE" == "base" ]]; then
    script="$REPO_ROOT/vargpt_1_1/VARGPT-family-training/run_scripts/run_eval_vargpt_generation_base.sh"
  fi
  local run_geneval=0
  local run_wise=0
  has_benchmark geneval && run_geneval=1
  has_benchmark wise && run_wise=1
  if has_benchmark dpg; then
    echo "[run_generation_evals] WARNING: VARGPT generation bench wrapper supports GenEval/WISE/DISE in this checkout, not DPG-Bench." >&2
  fi
  echo "[run_generation_evals] VARGPT generation evals -> $OUTPUT_ROOT"
  run_cmd env CHECKPOINT_DIR="$CHECKPOINT_DIR" OUTPUT_ROOT="$OUTPUT_ROOT" RUN_GENEVAL="$run_geneval" RUN_WISE="$run_wise" \
    bash "$script"
}

case "$BACKBONE" in
  blip3o|blip3o-8b)
    run_blip3o
    ;;
  bagel|bagel-7b)
    run_bagel
    ;;
  vargpt|vargpt-v1.1|vargpt-v1_1)
    run_vargpt
    ;;
  *)
    echo "[run_generation_evals] ERROR: unknown backbone '$BACKBONE'." >&2
    exit 1
    ;;
esac

echo "[run_generation_evals] Done. Outputs under: $OUTPUT_ROOT"
