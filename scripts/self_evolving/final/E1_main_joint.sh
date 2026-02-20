#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# E1 — Main Experiment: Full Joint Training (Understanding + Generation + DiT)
# ══════════════════════════════════════════════════════════════════════════════
#
# This is the PRIMARY result for the paper. All components are trained jointly:
#   • Solver LoRA   — improves visual understanding via GRPO
#   • Generator LoRA — improves text-to-image conditioning via GRPO + denoising
#   • DiT weights   — improves image generation via RWR (reward-weighted MSE)
#   • Proposer LoRA  — learns curriculum via dual reward from both tasks
#
# Key differences from X09 (the easy-data pilot):
#   • Uses 50k_balanced data (includes AI2D, PlotQA, infographic_vqa — harder)
#   • Trains for 1500 steps (was 650 in pilot)
#
# What this experiment proves:
#   ✓ Our framework improves BOTH understanding AND generation
#   ✓ On a diffusion-based UUG model (BLIP3o) — first in the literature
#   ✓ Without any external supervision (fully self-evolving)
#
# Compare against:
#   E2 — understanding-only  (shows joint training preserves understanding)
#   E3 — generation-only     (shows joint training preserves generation)
#   E4 — no DiT RWR          (isolates DiT contribution)
#   E5 — X09 easy data       (shows data quality matters)
#
# Usage:
#   TRAIN_STAGE=warmup bash E1_main_joint.sh
#   RESUME_FROM=/path/to/step_N TRAIN_STAGE=warmup bash E1_main_joint.sh
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

EXP_NAME="E1_main_joint"
OUTPUT_DIR="/workspace/self-evolving-uug/self-evolving-uug/runs/final/${EXP_NAME}"
RUN_NAME="${EXP_NAME}_s42"
TRAIN_STAGE="${TRAIN_STAGE:-warmup}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  E1 — Full Joint Training (Main Experiment)                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Stage:       ${TRAIN_STAGE}"
echo "║  Data:        ${DATA_DIR}"
echo "║  Output:      ${OUTPUT_DIR}"
echo "║  GPUs:        ${NPROC_PER_NODE}"
echo "╚══════════════════════════════════════════════════════════════╝"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port 29523 \
  "$TRAIN_ENTRY" \
  --experiment unified_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  \
  `# ── Role update frequencies ─────────────────────────────────────────` \
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
  `# ── Cycle scheduling (3 U-steps : 2 G-steps) ──────────────────────` \
  --understanding_steps_per_cycle 3 \
  --generation_steps_per_cycle 2 \
  --synthetic_solver_update_freq 0 \
  \
  `# ── DiT SFT + Joint Conditioning + RWR ─────────────────────────────` \
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
  `# ── Proposer dual reward (understanding + generation) ──────────────` \
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
