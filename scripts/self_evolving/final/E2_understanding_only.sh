#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# E2 — Ablation: Understanding-Only Training
# ══════════════════════════════════════════════════════════════════════════════
#
# Trains ONLY the understanding pathway (solver + proposer).
# Generation phase is completely disabled:
#   • No generator LoRA updates
#   • No DiT RWR updates
#   • No generation-phase proposer reward
#
# Only the proposer-solver loop runs:
#   • Proposer generates questions about images
#   • Solver answers with multiple samples (self-consistency)
#   • Solver LoRA is updated via GRPO on non-easy questions
#   • Proposer is updated via GRPO to produce harder questions
#
# What this experiment proves:
#   ✓ Understanding improves when trained in isolation
#   ✓ Compare E2 vs E1 to see if joint training helps or hurts understanding
#   ✓ Compare E2 vs E3 to show understanding-only doesn't help generation
#
# Usage:
#   TRAIN_STAGE=warmup bash E2_understanding_only.sh
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

EXP_NAME="E2_understanding_only"
OUTPUT_DIR="/workspace/self-evolving-uug/self-evolving-uug/runs/final/${EXP_NAME}"
RUN_NAME="${EXP_NAME}_s42"
TRAIN_STAGE="${TRAIN_STAGE:-warmup}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  E2 — Understanding-Only Ablation                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Stage:       ${TRAIN_STAGE}"
echo "║  Data:        ${DATA_DIR}"
echo "║  Output:      ${OUTPUT_DIR}"
echo "║  GPUs:        ${NPROC_PER_NODE}"
echo "║  NOTE: Generation training DISABLED                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port 29524 \
  "$TRAIN_ENTRY" \
  --experiment unified_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  \
  `# ── Role update frequencies ─────────────────────────────────────────` \
  `# Proposer + Solver active; Generator DISABLED                        ` \
  --proposer_update_freq 1 \
  --generator_update_freq 0 \
  --enable_solver_updates \
  --solver_update_freq 1 \
  \
  `# ── Generator GRPO (still needed for arg parsing but freq=0) ───────` \
  --generator_update_rule grpo \
  --generator_missing_trace_strategy skip \
  --grpo_clip_ratio 0.2 \
  --grpo_min_group_std 1e-4 \
  \
  `# ── Difficulty curriculum ───────────────────────────────────────────` \
  --difficulty_sampler_enabled \
  --solver_skip_update_on_easy \
  \
  `# ── Cycle scheduling: ALL understanding, no generation ─────────────` \
  --understanding_steps_per_cycle 5 \
  --generation_steps_per_cycle 0 \
  --synthetic_solver_update_freq 0 \
  \
  `# ── DiT DISABLED ───────────────────────────────────────────────────` \
  --dit_update_freq 0 \
  --dit_lr 5e-7 \
  --dit_weight_decay 0.01 \
  --dit_grad_clip 1.0 \
  --dit_grad_accum_steps 1 \
  --dit_conditioning_dropout 0.10 \
  --dit_loss_weight 1.0 \
  --dit_prompt_suffix_token_id 151665 \
  \
  `# ── NO proposer generation reward (no G-steps) ────────────────────` \
  \
  `# ── Shared args ────────────────────────────────────────────────────` \
  "${SHARED_ARGS[@]}" \
  --wandb_run_name "$RUN_NAME" \
  "${WARMUP_STAGE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
