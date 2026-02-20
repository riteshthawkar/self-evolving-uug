#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# E4 — Ablation: Full Joint Training WITHOUT DiT RWR
# ══════════════════════════════════════════════════════════════════════════════
#
# Same as E1 (full joint training) but with DiT updates completely disabled:
#   • No DiT denoising MSE loss
#   • No reward-weighted regression
#   • No joint LLM→DiT gradient flow
#
# What still gets trained:
#   • Solver LoRA — understanding via GRPO (same as E1)
#   • Generator LoRA — via GRPO on token traces only (no denoising gradient)
#   • Proposer LoRA — dual reward from both tasks (same as E1)
#
# What this experiment proves:
#   ✓ DiT fine-tuning (RWR) is essential for generation quality
#   ✓ Compare E4 vs E1: GenEval gap = DiT contribution
#   ✓ Understanding should be similar to E1 (DiT doesn't affect understanding)
#   ✓ This ablation is UNIQUE to our work — no competitor trains DiT jointly
#
# Why this matters for the paper:
#   SUDER/UniCorn/CoRL all use AR-based generation where the same LLM produces
#   discrete image tokens. They CAN'T train a separate DiT because there is none.
#   Our framework handles the ADDITIONAL challenge of training a continuous-latent
#   denoiser (DiT) jointly with the LLM conditioning encoder.
#
# Usage:
#   TRAIN_STAGE=warmup bash E4_no_dit_rwr.sh
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

EXP_NAME="E4_no_dit_rwr"
OUTPUT_DIR="/workspace/self-evolving-uug/self-evolving-uug/runs/final/${EXP_NAME}"
RUN_NAME="${EXP_NAME}_s42"
TRAIN_STAGE="${TRAIN_STAGE:-warmup}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  E4 — No DiT RWR Ablation                                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Stage:       ${TRAIN_STAGE}"
echo "║  Data:        ${DATA_DIR}"
echo "║  Output:      ${OUTPUT_DIR}"
echo "║  GPUs:        ${NPROC_PER_NODE}"
echo "║  NOTE: DiT training DISABLED (LLM LoRA only)               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port 29526 \
  "$TRAIN_ENTRY" \
  --experiment unified_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  \
  `# ── Role update frequencies (all active, same as E1) ───────────────` \
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
  `# ── Cycle scheduling (same as E1) ──────────────────────────────────` \
  --understanding_steps_per_cycle 3 \
  --generation_steps_per_cycle 2 \
  --synthetic_solver_update_freq 0 \
  \
  `# ── DiT DISABLED — no denoising loss, no RWR, no joint conditioning ` \
  --dit_update_freq 0 \
  --dit_lr 5e-7 \
  --dit_weight_decay 0.01 \
  --dit_grad_clip 1.0 \
  --dit_grad_accum_steps 1 \
  --dit_conditioning_dropout 0.10 \
  --dit_loss_weight 1.0 \
  --dit_prompt_suffix_token_id 151665 \
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
