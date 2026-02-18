#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# Experiment X09 — SUDER Dual-Reward + Joint LLM+DiT Training + RWR
# ══════════════════════════════════════════════════════════════════════════════
#
# Builds on X02 (unlabeled-only, no gen→understanding mixing) with three new
# co-training mechanisms inspired by the SUDER paper:
#
#   1. Proposer Dual Reward  (--proposer_gen_reward_enabled)
#      ─────────────────────────────────────────────────────
#      Proposer now gets reward from BOTH training tasks:
#        • Understanding phase: entropy-based Gaussian reward (existing).
#        • Generation phase:   best candidate total_reward (NEW).
#      A separate EMA baseline (proposer_gen_baseline, momentum=0.6) is kept
#      for the generation signal to prevent cross-scale contamination with
#      the understanding baseline.
#      Effect: proposer learns to write specs that lead to high-quality images,
#      closing the spec→image feedback loop that was missing in X02.
#
#   2. Joint LLM+DiT Conditioning Training  (--dit_joint_conditioning_train)
#      ────────────────────────────────────────────────────────────────────────
#      In X02 the LLM runs under torch.no_grad() during DiT updates → the
#      generator LoRA params (q/k/v/o/gate/up/down lora_A+B) get zero
#      gradients from the denoising loss; only the DiT denoiser is trained.
#      With this flag:
#        • torch.no_grad() is replaced by contextlib.nullcontext() in the
#          LLM conditioning forward pass.
#        • Generator LoRA params are added to the DiT optimizer as a second
#          param group at dit_joint_conditioning_lr=5e-7 (same as dit_lr here,
#          lower LR can be tuned independently).
#        • Gradients flow: MSE_loss → DiT → z_latents → generator LoRA.
#      Effect: trains BOTH "how to denoise" (DiT) AND "how to encode text for
#      diffusion conditioning" (generator LoRA) from the same denoising loss.
#
#   3. Reward-Weighted Regression  (--dit_reward_loss_weight 0.5)
#      ──────────────────────────────────────────────────────────
#      DiT denoising loss is scaled per-step by (1 + w × reward):
#        loss_rwr = mse_loss × max(0,  1 + 0.5 × best_reward)
#      With reward ∈ [0, 1]:  scale ∈ [1.0, 1.5].
#      High-reward generations (best_reward→1) are reinforced 1.5× more
#      strongly than low-reward ones (best_reward→0, scale→1.0).
#      This is the continuous-action analogue of GRPO for diffusion models.
#
# Comparison target:
#   X02 (02_x02_unlabeled_only_no_gen_to_understanding.sh) — same data,
#   same understanding setup, same DiT SFT, but WITHOUT the three features
#   above.  Metrics diff X09-X02 isolates the contribution of SUDER+RWR.
#
# Bug fixes applied at step ~500 (v2):
#   [BUG1] Baseline-clamp advantage collapse:
#     The proposer update used min(baseline, reward) when reward < 0, which
#     caused advantage = reward - clamped_baseline = 0.0 at equilibrium.
#     After ~50 steps the baseline tracked the penalty floor and the proposer
#     received zero gradient forever. Fixed: use raw baseline (no clamping).
#     Standard REINFORCE advantage = reward - baseline is correct as-is.
#
#   [BUG2] Proposer trained only on final question tokens (8-12 tokens):
#     completion was set to str(question or proposer_out), so only the final
#     question text received gradient — not the 300-token rationale that
#     actually determines difficulty. Fixed: str(proposer_out or question)
#     so the full XML output (rationale + question) is the training target.
#
#   [BUG3] IQR filter locked at IQR=0 blocking all solver updates:
#     After 256 steps of entropy=0, the filter history was all-zeros →
#     IQR=0 → threshold=0 → blocked everything. The easy-case gate already
#     handles this. Fixed: --entropy_iqr_filter_enabled false.
#
#   Resume: set RESUME_FROM + RESET_PROPOSER_BASELINE=1 to apply the reset.
#
# Architecture routing (unchanged from X02):
#   generator_update_freq=1  →  auto-routes to correct implementation:
#     • Discrete-token AR  (Janus, VARGPT) → generator GRPO on token traces.
#     • Continuous latent  (BLIP3o, BAGEL) → DiT denoising MSE fallback.
# ══════════════════════════════════════════════════════════════════════════════

REPO_ROOT="/workspace/self-evolving-uug/self-evolving-uug"
PYTHON_BIN="python3"
DATA_DIR="/workspace/self-evolving-uug/data/benchmark_10k/images"
OUTPUT_DIR="/workspace/self-evolving-uug/self-evolving-uug/runs/unified_experiments/X09_suder_joint_rwr"
RUN_NAME="x09_suder_joint_rwr_s42"
TRAIN_STAGE="${TRAIN_STAGE:-warmup}"   # warmup | strict
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

# ── Stage-specific hyperparameters ──────────────────────────────────────────
# warmup: relaxed difficulty distribution, wider temperature/top-p ranges.
#         Use this for the first run to let all modules learn the task.
# strict: tighter difficulty distribution, higher rejected-question penalty.
#         Use after warmup checkpoint to push performance on hard samples.
#
# Fixes applied vs original X02/X09 warmup defaults:
#   --rejected_question_penalty 0.25  (was 0.10) — stronger signal to push
#     proposer away from easy/non-objective questions immediately, not after
#     hundreds of near-zero-advantage steps.
#   --zero_entropy_reward_cap 0.20    (was 0.10) — harder negative for trivially
#     unanimous questions so the proposer actually learns to avoid them.
#   --difficulty_sampler_min_samples 8 (was 32) — curriculum activates after
#     just 8 difficulty observations (~40 total steps) instead of ~160 steps.
if [[ "$TRAIN_STAGE" == "warmup" ]]; then
  RUN_NAME="${RUN_NAME}_warmup"
  STAGE_ARGS=(
    --acceptance_require_non_easy
    --disable_proposer_require_objective
    --proposer_non_objective_penalty 0.0
    --difficulty_target_easy   0.30
    --difficulty_target_medium 0.50
    --difficulty_target_hard   0.20
    --rejected_question_penalty 0.25
    --zero_entropy_reward_cap 0.20
    --difficulty_sampler_min_samples 8
    --fixed_prop_entropy_target
    --prop_entropy_mu 0.90
    --solver_temp_min  0.70
    --solver_temp_max  1.60
    --solver_top_p_min 0.35
    --solver_top_p_max 1.00
  )
elif [[ "$TRAIN_STAGE" == "strict" ]]; then
  RUN_NAME="${RUN_NAME}_strict"
  STAGE_ARGS=(
    --acceptance_require_non_easy
    --disable_proposer_require_objective
    --proposer_non_objective_penalty 0.0
    --difficulty_target_easy   0.10
    --difficulty_target_medium 0.70
    --difficulty_target_hard   0.20
    --rejected_question_penalty 0.35
    --zero_entropy_reward_cap 0.20
    --difficulty_sampler_min_samples 8
    --fixed_prop_entropy_target
    --prop_entropy_ema_momentum 0.90
    --prop_entropy_mu_min 0.65
    --prop_entropy_mu_max 1.50
    --solver_temp_min  0.70
    --solver_temp_max  1.30
    --solver_top_p_min 0.50
    --solver_top_p_max 1.00
  )
else
  echo "[X09] ERROR: TRAIN_STAGE must be one of: warmup, strict (got: $TRAIN_STAGE)" >&2
  exit 1
fi

# ── Resume from checkpoint (optional) ───────────────────────────────────────
# To resume X09 from an existing checkpoint (e.g. after a restart or bug fix):
#   export RESUME_FROM="/workspace/.../X09_.../step_N"
#   TRAIN_STAGE=warmup bash 09_x09_suder_joint_rwr.sh
#
# RESET_PROPOSER_BASELINE: set to "1" when resuming after the baseline-clamp
# bug fix (steps 1-500 had advantage≈0). Resets proposer_baseline and
# proposer_gen_baseline to 0.0 and clears entropy/difficulty history so the
# IQR filter re-warms from scratch. Remove (set to "") after first checkpoint
# post-resume, otherwise the baseline resets every restart.
RESET_PROPOSER_BASELINE="${RESET_PROPOSER_BASELINE:-1}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "[X09] Resuming from checkpoint: $RESUME_FROM"
  RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM")
  if [[ "${RESET_PROPOSER_BASELINE}" == "1" ]]; then
    echo "[X09] reset_proposer_baseline=1: baseline and entropy windows will be reset on resume."
    RESUME_ARGS+=(--reset_proposer_baseline)
  fi
fi

# ── Directory / cache setup ──────────────────────────────────────────────────
cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

CACHE_ROOT="/workspace/self-evolving-uug/self-evolving-uug/cache"
CACHE_TMP_DIR="$CACHE_ROOT/tmp"
CACHE_TORCH_EXT_DIR="$CACHE_ROOT/torch_extensions"
CACHE_WANDB_DIR="$CACHE_ROOT/wandb"
CACHE_MIOPEN_DIR="$CACHE_ROOT/miopen"
CACHE_CUDA_DIR="$CACHE_ROOT/cuda"
mkdir -p \
  "$CACHE_ROOT" \
  "$CACHE_TMP_DIR" \
  "$CACHE_TORCH_EXT_DIR" \
  "$CACHE_WANDB_DIR" \
  "$CACHE_MIOPEN_DIR" \
  "$CACHE_CUDA_DIR" \
  "$CACHE_ROOT/assets"

# ── Environment ──────────────────────────────────────────────────────────────
export PYTHONPATH="/workspace/self-evolving-uug/self-evolving-uug/BLIP3o"
export HF_HOME="$CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_ROOT"
export HF_ASSETS_CACHE="$CACHE_ROOT/assets"
export TRANSFORMERS_CACHE="$CACHE_ROOT"
export HF_DATASETS_CACHE="$CACHE_ROOT"
export HF_METRICS_CACHE="$CACHE_ROOT"
export TORCH_HOME="$CACHE_ROOT"
export TRITON_CACHE_DIR="$CACHE_ROOT"
export TORCH_EXTENSIONS_DIR="$CACHE_TORCH_EXT_DIR"
export XDG_CACHE_HOME="$CACHE_ROOT"
export TMPDIR="$CACHE_TMP_DIR"
export TMP="$CACHE_TMP_DIR"
export TEMP="$CACHE_TMP_DIR"
export WANDB_DIR="$CACHE_WANDB_DIR"
export WANDB_CACHE_DIR="$CACHE_WANDB_DIR"
export WANDB_CONFIG_DIR="$CACHE_WANDB_DIR"
export WANDB_DATA_DIR="$CACHE_WANDB_DIR"
export CUDA_CACHE_PATH="$CACHE_CUDA_DIR"
export MIOPEN_USER_DB_PATH="$CACHE_MIOPEN_DIR"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_MIOPEN_DIR"
export TOKENIZERS_PARALLELISM="false"
export SE_MAX_IMAGE_SIDE="${SE_MAX_IMAGE_SIDE:-896}"
export SE_MIN_IMAGE_SIDE="${SE_MIN_IMAGE_SIDE:-56}"
export SE_IMAGE_SIZE_MULTIPLE="${SE_IMAGE_SIZE_MULTIPLE:-28}"
GENERATION_IMAGE_SIDE="${GENERATION_IMAGE_SIDE:-896}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_DISTRIBUTED_DEBUG="OFF"
export NCCL_DEBUG="WARN"
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [[ ! -d "$DATA_DIR" ]]; then
  echo "[X09] ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
  exit 1
fi
if ! find "$DATA_DIR" -type f \
    \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) \
    -print -quit | grep -q .; then
  echo "[X09] ERROR: DATA_DIR has no image files: $DATA_DIR" >&2
  exit 1
fi

echo "[X09] Starting experiment X09 (SUDER dual-reward + joint LLM+DiT + RWR)"
echo "[X09]   Stage:       $TRAIN_STAGE"
echo "[X09]   Run name:    $RUN_NAME"
echo "[X09]   Output dir:  $OUTPUT_DIR"
echo "[X09]   Data dir:    $DATA_DIR"
echo "[X09]   GPUs:        $NPROC_PER_NODE"
echo "[X09]   Attn impl:   $ATTN_IMPL"
if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "[X09]   Resume from: $RESUME_FROM"
fi

# ── Launch ───────────────────────────────────────────────────────────────────
"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port 29523 \
  "/workspace/self-evolving-uug/self-evolving-uug/BLIP3o/blip3o/train/train_self_evolving.py" \
  --experiment unified_self_evolving \
  --data_dir "$DATA_DIR" \
  --data_split all \
  --model_name BLIP3o/BLIP3o-Model-8B \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  --dtype bfloat16 \
  --attn_implementation "$ATTN_IMPL" \
  --device_map single \
  --cuda_device 0 \
  \
  `# ── Training schedule ──────────────────────────────────────────────────` \
  --total_steps 10000 \
  --save_every 2000 \
  --log_every 1 \
  --max_checkpoints 5 \
  --save_generated_images_every 2000 \
  --deterministic \
  \
  `# ── Model / LoRA ───────────────────────────────────────────────────────` \
  --require_decoder_for_blip3o \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,mm_projector \
  \
  `# ── Optimiser (understanding-side GRPO) ────────────────────────────────` \
  --lr 1e-6 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --grad_accum_steps 4 \
  \
  `# ── Role update frequencies ─────────────────────────────────────────────` \
  --proposer_update_freq 1 \
  --generator_update_freq 1 \
  --enable_solver_updates \
  --solver_update_freq 1 \
  \
  `# ── Generator GRPO (Paradigm A — discrete token AR models) ─────────────` \
  --generator_update_rule grpo \
  --generator_missing_trace_strategy skip \
  --grpo_clip_ratio 0.2 \
  --grpo_min_group_std 1e-4 \
  \
  `# ── Sampling ────────────────────────────────────────────────────────────` \
  --temp 1.0 \
  --top_p 1.0 \
  --max_new_tokens_solver 96 \
  --max_new_tokens_proposer 512 \
  --max_new_tokens_caption 64 \
  --max_new_tokens_generator 512 \
  --num_solver_samples 5 \
  --num_solver_samples_spec 2 \
  --num_generations 3 \
  --proposer_num_candidates 3 \
  --proposer_spot_check_samples 2 \
  \
  `# ── Image generation (Paradigm B — BLIP3o diffusion) ───────────────────` \
  --generation_num_inference_steps 20 \
  --generation_guidance_scale 2.0 \
  --generation_height "$GENERATION_IMAGE_SIDE" \
  --generation_width  "$GENERATION_IMAGE_SIDE" \
  \
  `# ── Difficulty curriculum ───────────────────────────────────────────────` \
  --difficulty_sampler_enabled \
  --solver_skip_update_on_easy \
  \
  `# ── Reward weights ──────────────────────────────────────────────────────` \
  --reward_spec_weight 0.65 \
  --reward_cycle_weight 0.20 \
  --reward_diversity_weight 0.10 \
  --reward_contradiction_weight 0.20 \
  \
  `# ── Spec quality gates ──────────────────────────────────────────────────` \
  --min_spec_quality_for_update 0.35 \
  --min_spec_qa_pairs 2 \
  --max_expected_words 8 \
  --max_question_words 24 \
  \
  `# ── Solver details ──────────────────────────────────────────────────────` \
  --solver_soft_gamma 0.7 \
  --solver_use_temperature_mix \
  --sc_entropy_min 0.15 \
  --sc_entropy_max 1.20 \
  --sc_margin_max 0.90 \
  --entropy_iqr_min_threshold 0.10 \
  --sc_negative_weight 0.25 \
  --skip_solver_update_when_uninformative \
  --len_penalty_weight 0.10 \
  --len_penalty_target_words 6 \
  --solver_hardness_min_entropy 0.20 \
  `# Lower the majority-fraction gate for blocking easy solver updates:        ` \
  `# was 0.95 (block only if 5/5 agree). At 0.80 (block only if ≥4/5 agree)  ` \
  `# the solver still trains on questions where at least 1 sample disagrees,  ` \
  `# giving real gradient signal rather than purely easy-penalty updates.      ` \
  --easy_update_majority_frac_threshold 0.80 \
  `# IQR filter disabled: at step 500 the 256-step history was all entropy=0` \
  `# so IQR=0, threshold=0 → filter blocked ALL updates redundantly. The    ` \
  `# easy-case gate (solver_skip_update_on_easy) already handles this.       ` \
  --disable_entropy_iqr_filter \
  \
  `# ── Proposer entropy target ─────────────────────────────────────────────` \
  --prop_entropy_sigma 0.25 \
  \
  `# ── Cycle scheduling ────────────────────────────────────────────────────` \
  --understanding_steps_per_cycle 3 \
  --generation_steps_per_cycle 2 \
  --synthetic_solver_update_freq 0 \
  \
  `# ── KL regularisation ───────────────────────────────────────────────────` \
  --kl_coef 0.01 \
  --kl_target 0.02 \
  --kl_adapt_rate 0.10 \
  --kl_min 1e-8 \
  --kl_max 1e2 \
  \
  `# ── Proposer optimization algorithm ────────────────────────────────────` \
  `#   grpo: group-normalized advantages, lower variance, no EMA baseline.  ` \
  `#         Uses proposer_num_candidates=3 as the group in U-steps.         ` \
  `#   reinforce: single-sample REINFORCE with EMA baseline (legacy).        ` \
  --proposer_update_rule grpo \
  --proposer_grpo_gen_group_size 3 \
  \
  `# ── Baselines ───────────────────────────────────────────────────────────` \
  --baseline_momentum 0.6 \
  \
  `# ── Misc ────────────────────────────────────────────────────────────────` \
  --clear_cache_every 10 \
  --use_ref_answer_scoring \
  \
  `# ── Unicorn reconstruction (disabled — same as X02) ────────────────────` \
  --disable_unicorn_reconstruction_sft \
  --disable_unicorn_reconstruction_generator \
  \
  `# ── Replay buffer (disabled — same as X02) ─────────────────────────────` \
  --replay_buffer_size 1 \
  --replay_min_reward 1.10 \
  --replay_max_staleness 1 \
  --gen_mix_source_mode buffer \
  --gen_mix_ratio_start 0.0 \
  --gen_mix_ratio_max 0.0 \
  --gen_mix_ratio_warmup_steps 1 \
  \
  `# ── DiT SFT updater (Paradigm B — BLIP3o) ──────────────────────────────` \
  --dit_update_enabled \
  --dit_update_freq 1 \
  --dit_lr 5e-7 \
  --dit_weight_decay 0.01 \
  --dit_grad_clip 1.0 \
  --dit_grad_accum_steps 1 \
  --dit_conditioning_dropout 0.10 \
  --dit_loss_weight 1.0 \
  --dit_prompt_suffix_token_id 151665 \
  \
  `# ════════════════════════════════════════════════════════════════════════` \
  `# NEW in X09 — SUDER dual-reward + joint LLM+DiT training + RWR          ` \
  `# ════════════════════════════════════════════════════════════════════════` \
  \
  `# [1] Proposer joint reward: α*gaussian(entropy_on_generated) + (1-α)*quality` \
  `#   α=0.7 → mostly entropy-driven (same objective as understanding phase),  ` \
  `#   keeping a 0.3 quality floor so images stay faithful to the spec.        ` \
  `#   gen_step_solver_update_enabled: also trains solver on generated images   ` \
  `#   during every G-step (joint step) — solver rollouts already computed for  ` \
  `#   scoring, so no extra inference cost.                                     ` \
  --proposer_gen_reward_enabled \
  --proposer_gen_entropy_weight 0.7 \
  --proposer_gen_baseline_momentum 0.6 \
  --gen_step_solver_update_enabled \
  \
  `# [2] Joint LLM+DiT: remove no_grad from conditioning encoder → LoRA grads` \
  --dit_joint_conditioning_train \
  --dit_joint_conditioning_lr 5e-7 \
  \
  `# [3] RWR: scale DiT loss by (1 + 0.5 * best_reward) → reward shaping    ` \
  --dit_reward_loss_weight 0.5 \
  \
  `# ── Logging / W&B ───────────────────────────────────────────────────────` \
  --wandb_mode disabled \
  --wandb_project self-evolving-uug-unified \
  --wandb_run_name "$RUN_NAME" \
  \
  `# ── Stage-specific args (difficulty curriculum) ─────────────────────────` \
  "${STAGE_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --seed 42
