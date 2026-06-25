# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

usage() {
  cat <<'EOF'
Usage: bash Bagel/eval/vlm/evaluate.sh DATASET [evaluation args...]

DATASET options:
  mme, mmvet, mmbench-dev-en, mmbench-dev-cn, mmbench-test-en,
  mmbench-test-cn, mmmu-dev, mmmu-val, mmmu-val_cot, mmmu-test,
  mathvista-testmini, mathvista-test, pope, pope_cot,
  vqa-gqa-testdev, mmvp
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

DATASET="$1"
shift
ARGS=("$@")

: "${GPUS:=1}"
: "${CHECKPOINT:=}"
: "${ARNOLD_WORKER_NUM:=1}"
: "${ARNOLD_ID:=0}"
: "${ARNOLD_WORKER_0_HOST:=127.0.0.1}"
: "${MASTER_PORT:=29500}"

echo "CHECKPOINT: ${CHECKPOINT}"

# Parse options
for arg in "${ARGS[@]}"; do
  case "${arg}" in
    --auto)
      GPUS=1
      ;;
  esac
done
echo "GPUS: ${GPUS}"

run_torch() {
  torchrun \
    --nnodes="${ARNOLD_WORKER_NUM}" \
    --node_rank="${ARNOLD_ID}" \
    --master_addr="${ARNOLD_WORKER_0_HOST}" \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    "$@"
}

case "${DATASET}" in
  mme)
    python -m eval.vlm.eval.mme.eval "${ARGS[@]}"
    ;;
  mmvet)
    python -m eval.vlm.eval.mmvet.evaluate_mmvet --datasets mmvet "${ARGS[@]}"
    ;;
  mmbench-dev-en)
    run_torch -m eval.vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_dev_20230712 "${ARGS[@]}"
    ;;
  mmbench-dev-cn)
    run_torch -m eval.vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_dev_cn_20231003 "${ARGS[@]}"
    ;;
  mmbench-test-en)
    run_torch -m eval.vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_test_en_20231003 "${ARGS[@]}"
    ;;
  mmbench-test-cn)
    run_torch -m eval.vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_test_cn_20231003 "${ARGS[@]}"
    ;;
  mmmu-dev)
    run_torch -m eval.vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_dev "${ARGS[@]}"
    ;;
  mmmu-val)
    run_torch -m eval.vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_validation "${ARGS[@]}"
    ;;
  mmmu-val_cot)
    run_torch -m eval.vlm.eval.mmmu.evaluate_mmmu_cot --datasets MMMU_validation_cot "${ARGS[@]}"
    ;;
  mmmu-test)
    run_torch -m eval.vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_test "${ARGS[@]}"
    ;;
  mathvista-testmini)
    run_torch -m eval.vlm.eval.mathvista.evaluate_mathvista --datasets MathVista_testmini "${ARGS[@]}"
    ;;
  mathvista-test)
    run_torch -m eval.vlm.eval.mathvista.evaluate_mathvista --datasets MathVista_test "${ARGS[@]}"
    ;;
  pope)
    run_torch -m eval.vlm.eval.pope.evaluate_pope --datasets pope "${ARGS[@]}"
    ;;
  pope_cot)
    run_torch -m eval.vlm.eval.pope.evaluate_pope --datasets pope_cot --cot "${ARGS[@]}"
    ;;
  vqa-gqa-testdev)
    run_torch -m eval.vlm.eval.vqa.evaluate_vqa --datasets gqa_testdev_llava "${ARGS[@]}"
    ;;
  mmvp)
    run_torch -m eval.vlm.eval.mmvp.evaluate_mmvp --datasets MMVP "${ARGS[@]}"
    ;;
  *)
    echo "ERROR: unknown DATASET '${DATASET}'" >&2
    usage >&2
    exit 2
    ;;
esac
