#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# E3 — Ablation: Generation-Only Training
# ══════════════════════════════════════════════════════════════════════════════
#
# Trains ONLY the generation pathway (generator LoRA + DiT).
# Understanding phase is completely disabled:
#   • No solver LoRA updates
#   • No understanding-phase proposer reward
#   • Solver runs only as a frozen verifier (for spec/cycle rewards)
#
# What gets trained:
#   • Generator LoRA — text-to-latent conditioning via GRPO
#   • DiT weights — denoising via RWR (reward-weighted MSE)
#   • Generator LoRA also gets gradients from DiT joint conditioning
#   • Proposer LoRA — but only from generation reward (no entropy reward)
#
# What this experiment proves:
#   ✓ Generation improves when trained in isolation
#   ✓ Compare E3 vs E1 to see if joint training helps generation
#   ✓ Compare E3 understanding metrics to show generation-only may hurt understanding
#     (This is what UniCorn's CPR mechanism was designed to prevent)
#
# Usage:
#   TRAIN_STAGE=warmup bash E3_generation_only.sh
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

EXP_NAME="E3_generation_only"
OUTPUT_DIR="/workspace/self-evolving-uug/self-evolving-uug/runs/final/${EXP_NAME}"
RUN_NAME="${EXP_NAME}_s42"
TRAIN_STAGE="${TRAIN_STAGE:-warmup}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  E3 — Generation-Only Ablation                             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Stage:       ${TRAIN_STAGE}"
echo "║  Data:        ${DATA_DIR}"
echo "║  Output:      ${OUTPUT_DIR}"
echo "║  GPUs:        ${NPROC_PER_NODE}"
echo "║  NOTE: Understanding training DISABLED                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port 29525 \
  "$TRAIN_ENTRY" \
  --experiment unified_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  \
  `# ── Role update frequencies ─────────────────────────────────────────` \
  `# Generator active; Solver DISABLED (runs as frozen verifier only)    ` \
  --proposer_update_freq 1 \
  --generator_update_freq 1 \
  --solver_update_freq 0 \
  \
  `# ── Generator GRPO ──────────────────────────────────────────────────` \
  --generator_update_rule grpo \
  --generator_missing_trace_strategy skip \
  --grpo_clip_ratio 0.2 \
  --grpo_min_group_std 1e-4 \
  \
  `# ── Difficulty curriculum (disabled — no understanding phase) ───────` \
  --difficulty_sampler_enabled \
  --solver_skip_update_on_easy \
  \
  `# ── Cycle scheduling: ALL generation, no understanding ─────────────` \
  --understanding_steps_per_cycle 0 \
  --generation_steps_per_cycle 5 \
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
  `# ── Proposer generation reward ONLY (no understanding reward) ──────` \
  --proposer_gen_reward_enabled \
  --proposer_gen_entropy_weight 0.7 \
  --proposer_gen_baseline_momentum 0.6 \
  \
  `# ── Shared args ────────────────────────────────────────────────────` \
  "${SHARED_ARGS[@]}" \
  --wandb_run_name "$RUN_NAME" \
  "${WARMUP_STAGE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
