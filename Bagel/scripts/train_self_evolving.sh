#!/usr/bin/env bash
# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Shared repo environment bootstrap.
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

: "${MODEL_PATH:?Set MODEL_PATH to BAGEL model directory (contains ema.safetensors)}"
: "${IMAGE_DIR:?Set IMAGE_DIR to an image folder for understanding rollouts}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for rollout logs}"

STEPS="${STEPS:-10000}"
DEVICE="${DEVICE:-cuda}"
MAX_LATENT_SIZE="${MAX_LATENT_SIZE:-64}"

python3 train/train_self_evolving.py \
  --model_path "${MODEL_PATH}" \
  --device "${DEVICE}" \
  --max_latent_size "${MAX_LATENT_SIZE}" \
  --image_dir "${IMAGE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --steps "${STEPS}" \
  --log_every 10 \
  --max_new_tokens_proposer 256 \
  --max_new_tokens_solver 96 \
  --proposer_temperature 0.9 \
  --num_solver_samples 7 \
  --solver_temp_min 0.5 \
  --solver_temp_max 2.0 \
  --proposer_entropy_mu 0.9 \
  --proposer_entropy_sigma 0.25 \
  --solver_unsolvable_maj_threshold 0.20 \
  --zero_entropy_eps 1e-6 \
  --proposer_non_objective_penalty 0.20 \
  --rejected_question_penalty 0.35 \
  --proposer_require_objective \
  --acceptance_require_non_easy
