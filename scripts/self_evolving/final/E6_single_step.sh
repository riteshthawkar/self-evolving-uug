#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# E6 — Ablation: Single-Step Joint Training (1U + 1G per cycle)
# ══════════════════════════════════════════════════════════════════════════════
#
# Same as E1 (full joint training) but with a 1:1 cycle ratio instead of 3:2.
# Every step alternates between understanding and generation:
#
#   Step 1: Understanding (proposer → solver → GRPO)
#   Step 2: Generation   (proposer → generator → DiT RWR)
#   Step 3: Understanding ...
#   Step 4: Generation   ...
#
# In E1 (and X09), the cycle is 3U + 2G:
#   Steps 1-3: Understanding
#   Steps 4-5: Generation
#   Steps 6-8: Understanding
#   Steps 9-10: Generation ...
#
# What this experiment tests:
#   • Does the U:G ratio matter?
#   • Is 3:2 better because understanding needs more steps (harder to learn)?
#   • Or is 1:1 better because it provides more frequent cross-task feedback?
#
# Key insight:
#   With 3:2, the model does 3 understanding steps in a row before switching
#   to generation. The understanding LoRA changes accumulate before the
#   generator sees the updated model. With 1:1, the feedback loop is tighter:
#   each understanding improvement is immediately followed by a generation
#   step that benefits from the updated conditioning.
#
# What this experiment proves:
#   ✓ Compare E6 vs E1: optimal cycle ratio for joint training
#   ✓ If E6 ≈ E1: ratio doesn't matter much → simpler configuration
#   ✓ If E6 > E1: tighter coupling is better → argues for co-evolution
#   ✓ If E6 < E1: understanding benefits from consecutive batching
#
# Usage:
#   TRAIN_STAGE=warmup bash E6_single_step.sh
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

EXP_NAME="E6_single_step"
OUTPUT_DIR="/workspace/self-evolving-uug/self-evolving-uug/runs/final/${EXP_NAME}"
RUN_NAME="${EXP_NAME}_s42"
TRAIN_STAGE="${TRAIN_STAGE:-warmup}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  E6 — Single-Step Joint (1U + 1G per cycle)                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Stage:       ${TRAIN_STAGE}"
echo "║  Data:        ${DATA_DIR}"
echo "║  Output:      ${OUTPUT_DIR}"
echo "║  GPUs:        ${NPROC_PER_NODE}"
echo "║  Cycle: 1 U-step + 1 G-step (alternating every step)       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port 29528 \
  "$TRAIN_ENTRY" \
  --experiment unified_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  \
  `# ── Role update frequencies (same as E1) ────────────────────────────` \
  --proposer_update_freq 1 \
  --generator_update_freq 1 \
  --enable_solver_updates \
  --solver_update_freq 1 \
  \
  `# ── Generator GRPO ──────────────────────────────────────────────────` \
  --generator_update_rule grpo \
  --generator_missing_trace_strategy skip \
  --grpo_clip_ratio 0.2 \
  --grpo_min_group_std 1e-4 \
  \
  `# ── Difficulty curriculum ───────────────────────────────────────────` \
  --difficulty_sampler_enabled \
  --solver_skip_update_on_easy \
  \
  `# ── Cycle: 1 U-step + 1 G-step (tight alternation) ────────────────` \
  --understanding_steps_per_cycle 1 \
  --generation_steps_per_cycle 1 \
  --synthetic_solver_update_freq 0 \
  \
  `# ── DiT SFT + Joint Conditioning + RWR (same as E1) ───────────────` \
  --dit_update_enabled \
  --dit_update_freq 1 \
  --dit_lr 5e-7 \
  --dit_weight_decay 0.01 \
  --dit_grad_clip 1.0 \
  --dit_grad_accum_steps 1 \
  --dit_conditioning_dropout 0.10 \
  --dit_loss_weight 1.0 \
  --dit_prompt_suffix_token_id 151665 \
  --dit_joint_conditioning_train \
  --dit_joint_conditioning_lr 5e-7 \
  --dit_reward_loss_weight 0.5 \
  \
  `# ── Proposer dual reward (same as E1) ──────────────────────────────` \
  --proposer_gen_reward_enabled \
  --proposer_gen_entropy_weight 0.7 \
  --proposer_gen_baseline_momentum 0.6 \
  --gen_step_solver_update_enabled \
  \
  `# ── Shared args ────────────────────────────────────────────────────` \
  "${SHARED_ARGS[@]}" \
  --wandb_run_name "$RUN_NAME" \
  "${WARMUP_STAGE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
