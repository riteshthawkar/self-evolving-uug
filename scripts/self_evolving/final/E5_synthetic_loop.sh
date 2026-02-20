#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# E5 — Fully Imageless Self-Evolving Loop (ZERO External Images)
# ══════════════════════════════════════════════════════════════════════════════
#
# The most radical experiment: NO external images are used at ANY point.
# The proposer generates specs from TEXT TOPICS/THEMES alone — it never sees
# a real image. This creates a FULLY synthetic self-evolving loop:
#
#   1. Topic sampled from diverse theme list (no image)
#   2. Proposer (text-only) → imagines a scene → writes generation prompt + QA
#      The proposer creates QA pairs based on EXPECTATIONS of what the image
#      should contain (consistency, alignment, object location, grounding, etc.)
#   3. Generator creates candidate images from the prompt
#   4. Best generated image is selected by solver verification
#   5. Solver answers QA questions about the GENERATED image
#   6. Solver gets GRPO reward → updates solver LoRA
#   7. DiT gets RWR reward → updates denoiser
#   8. Proposer gets dual reward → learns to write more verifiable prompts
#
# The self-evolving loop:
#   • Better proposer → richer prompts with verifiable details → better images
#   • Better generator → more faithful images → solver learns real visual grounding
#   • Better solver → more accurate verification → better reward signal → better generator
#   • All components improve using ONLY self-generated content
#
# Implementation:
#   • imageless_proposer_mode = True (KEY: proposer uses text-only generation)
#   • understanding_steps_per_cycle = 0  (NO understanding-only steps)
#   • generation_steps_per_cycle = 5     (ALL steps are generation)
#   • gen_step_solver_update_enabled = True (solver trains on generated images)
#   • use_ref_answer_scoring = False (auto-disabled: no real image for ref answers)
#
# Why this matters for the paper (KEY DIFFERENTIATOR):
#   This is UNIQUE to our framework. No competitor can do this:
#     - SUDER uses real images for BOTH understanding and generation
#     - UniCorn uses real images as seeds for generation
#     - CoRL uses labeled external data with ground-truth answers
#
#   E5 proves that our framework can improve BOTH generation AND understanding
#   WITHOUT any external image supervision — the model teaches itself entirely
#   from its own text reasoning and image generation.
#
#   The proposer IMAGINES scenes from topics, the generator CREATES them, and
#   the solver LEARNS to understand them — a fully autonomous learning loop.
#
# What this experiment proves:
#   ✓ Understanding can improve from purely self-generated images
#   ✓ Generation improves without any real image reference
#   ✓ The proposer can learn to write good specs from text alone
#   ✓ Compare E5 vs E1: quantifies the value of real images
#   ✓ Compare E5 vs E3: solver training on generated images adds value
#   ✓ If E5 improves → TRUE self-evolution, strongest novelty claim
#
# Usage:
#   TRAIN_STAGE=warmup bash E5_synthetic_loop.sh
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

EXP_NAME="E5_synthetic_loop"
OUTPUT_DIR="/workspace/self-evolving-uug/self-evolving-uug/runs/final/${EXP_NAME}"
RUN_NAME="${EXP_NAME}_s42"
TRAIN_STAGE="${TRAIN_STAGE:-warmup}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  E5 — Fully Imageless Self-Evolving Loop                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Stage:       ${TRAIN_STAGE}"
echo "║  Data:        ${DATA_DIR}  (used for pool init only)"
echo "║  Output:      ${OUTPUT_DIR}"
echo "║  GPUs:        ${NPROC_PER_NODE}"
echo "║  NOTE: ZERO external images — proposer generates from TOPICS║"
echo "║  Cycle: 0 U-steps + 5 G-steps (fully imageless generation) ║"
echo "╚══════════════════════════════════════════════════════════════╝"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port 29527 \
  "$TRAIN_ENTRY" \
  --experiment unified_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  \
  `# ── Role update frequencies (ALL active) ────────────────────────────` \
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
  `# ── IMAGELESS PROPOSER MODE (KEY: no external images at all) ───────` \
  --imageless_proposer_mode \
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
  `# ── Proposer dual reward + solver trains on generated images ───────` \
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
