"""
Configuration dataclasses for the VARGPT self-evolving training pipeline.

Ported from the BLIP3o implementation with VARGPT-specific adaptations:
  - Removed DiT/diffusion-specific parameters
  - Added VAR (autoregressive discrete token) generation parameters
  - Removed model/device params (handled by LLaMA-Factory)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SelfEvolvingConfig:
    """Unified config for the self-evolving proposer-solver-generator training loop.

    Ported from BLIP3o's UnifiedSelfEvolvingConfig, adapted for VARGPT v1.1:
      - Image generation uses autoregressive discrete tokens (Infinity VAR)
        instead of continuous latent diffusion (DiT).
      - Training infrastructure is LLaMA-Factory (HF Trainer) based.
      - Multi-adapter LoRA managed by adapter_manager.py, not custom code.
    """

    # ── Experiment identity ──────────────────────────────────────────────
    experiment_name: str = "vargpt_self_evolving"
    run_name: Optional[str] = None

    # ── U/G Cycle ────────────────────────────────────────────────────────
    understanding_steps_per_cycle: int = 3
    generation_steps_per_cycle: int = 2
    total_steps: int = 10000

    # ── Optimization ─────────────────────────────────────────────────────
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    proposer_update_freq: int = 1
    generator_update_freq: int = 1
    enable_solver_updates: bool = True
    solver_update_freq: int = 1
    synthetic_solver_update_freq: int = 1
    synthetic_solver_hard_only: bool = False
    solver_hardness_min_entropy: float = 0.2

    # ── Decoding ─────────────────────────────────────────────────────────
    temp: float = 1.0
    top_p: float = 1.0
    max_new_tokens_solver: int = 128
    max_new_tokens_proposer: int = 256
    max_new_tokens_caption: int = 96
    num_solver_samples: int = 5
    num_solver_samples_spec: int = 3

    # ── VARGPT Image Generation (replaces DiT/diffusion params) ─────────
    var_cfg_scale: float = 3.0           # classifier-free guidance scale
    var_tau: float = 0.5                 # temperature for VAR discrete token sampling
    var_top_k: int = 900                 # top-k for VAR sampling
    var_top_p: float = 0.97              # top-p for VAR sampling
    var_sampling_per_bits: int = 1       # sampling resolution for BSQ
    num_generations: int = 3             # K candidate images per G-step
    image_gen_grpo_group_size: int = 4   # candidates for generation GRPO
    image_gen_loss_lambda: float = 3.0   # weight of image gen loss (matches get_gen_loss_1_1)
    gen_reward_mode: str = "clip"        # "clip", "nll", or "embedding"
    gen_projector_trainable_roles: Tuple[str, ...] = ("generator",)

    # ── Generator Update Rule ────────────────────────────────────────────
    generator_update_rule: str = "grpo"   # grpo|reinforce
    grpo_clip_ratio: float = 0.2          # PPO-style importance ratio clipping
    grpo_min_group_std: float = 1e-6      # skip GRPO update if reward std below

    # ── Imageless Proposer Mode ──────────────────────────────────────────
    imageless_proposer_mode: bool = False

    # ── Proposer Configuration ───────────────────────────────────────────
    proposer_update_rule: str = "grpo"             # reinforce|grpo
    proposer_grpo_gen_group_size: int = 3
    proposer_num_candidates: int = 5               # K candidate questions per proposer call
    proposer_spot_check_samples: int = 3           # solver samples for spot-checking
    proposer_certificate_enabled: bool = True
    proposer_certificate_min_score: float = 0.55
    proposer_certificate_weight: float = 0.75
    proposer_certificate_strict_struct: bool = True
    proposer_question_quality_min_score: float = 0.60
    proposer_question_structural_min_score: float = 0.50
    proposer_question_model_judge_enabled: bool = True
    proposer_question_model_judge_weight: float = 0.15
    proposer_gen_reward_enabled: bool = True
    proposer_gen_entropy_weight: float = 0.7
    proposer_gen_baseline_momentum: float = 0.6
    gen_step_solver_update_enabled: bool = False
    reset_proposer_baseline: bool = False

    # ── Reward Shaping ───────────────────────────────────────────────────
    solver_soft_gamma: float = 0.7
    solver_use_temperature_mix: bool = True
    solver_use_forced_choice_from_proposer: bool = True
    solver_temp_min: float = 0.4
    solver_temp_max: float = 2.6
    solver_top_p_min: float = 0.35
    solver_top_p_max: float = 1.0
    sc_entropy_min: float = 0.15
    sc_entropy_max: float = 1.2
    sc_margin_max: float = 0.90
    sc_informative_ratio_min: float = 0.25
    sc_negative_weight: float = 0.25
    easy_solver_penalty_scale: float = 1.0
    skip_solver_update_when_uninformative: bool = True
    solver_always_update_with_informative_scaling: bool = False
    solver_update_min_scale: float = 0.20
    len_penalty_weight: float = 0.10
    len_penalty_target_words: int = 6
    prop_entropy_mu: float = 0.90
    prop_entropy_sigma: float = 0.35
    adaptive_prop_entropy_target: bool = True
    prop_entropy_ema_momentum: float = 0.90
    prop_entropy_mu_min: float = 0.40
    prop_entropy_mu_max: float = 1.5
    zero_entropy_reward_cap: float = 0.10
    proposer_unsolvable_reward_cap: float = 0.10
    solver_unsolvable_maj_threshold: float = 0.20
    proposer_non_objective_penalty: float = 0.20
    proposer_require_objective: bool = True
    proposer_slot_compiler_enabled: bool = True
    proposer_slot_compiler_strict: bool = True
    proposer_reasoning_min_domains: int = 2
    proposer_reasoning_require_non_relation: bool = True
    proposer_reasoning_min_chain_words: int = 8
    proposer_spot_entropy_min_gate: float = 0.05
    solver_token_entropy_enabled: bool = True
    solver_token_entropy_tokens: int = 5
    solver_token_entropy_window_size: int = 128
    solver_token_entropy_sigmoid_alpha: float = 1.5
    solver_token_entropy_sigmoid_beta: float = 2.0
    solver_skip_update_on_easy: bool = True
    easy_update_majority_frac_threshold: float = 0.85
    acceptance_require_non_easy: bool = True
    rejected_question_penalty: float = 0.35

    # ── Entropy IQR Filter ───────────────────────────────────────────────
    entropy_iqr_filter_enabled: bool = True
    entropy_iqr_window_size: int = 256
    entropy_iqr_min_samples: int = 32
    entropy_iqr_easy_quantile: float = 0.25
    entropy_iqr_easy_iqr_coef: float = 0.25
    entropy_iqr_min_threshold: float = 0.02
    entropy_iqr_max_threshold: float = 1.2
    entropy_iqr_filter_min_majority_frac: float = 0.80

    # ── Difficulty Curriculum ────────────────────────────────────────────
    difficulty_sampler_enabled: bool = True
    difficulty_sampler_window_size: int = 256
    difficulty_sampler_min_samples: int = 32
    difficulty_target_easy: float = 0.10
    difficulty_target_medium: float = 0.50
    difficulty_target_hard: float = 0.40
    difficulty_hard_min_entropy: float = 0.90
    difficulty_hard_max_margin: float = 0.35

    # ── Proposer Warm-Start (entropy-free bootstrap) ─────────────────────
    proposer_warm_start_enabled: bool = True
    proposer_warm_start_max_steps: int = 30
    proposer_warm_start_exit_window: int = 5
    proposer_warm_start_exit_consecutive: int = 2
    proposer_warm_start_entropy_exit_threshold: float = 0.10
    proposer_warm_start_easy_reject_penalty_scale: float = 0.0
    proposer_warm_start_certificate_weight: float = 0.50

    # ── Hardness Debt Controller ──────────────────────────────────────────
    hardness_debt_enabled: bool = True
    hardness_debt_inc_easy: float = 1.50
    hardness_debt_dec_non_easy: float = 1.00
    hardness_debt_max: float = 6.0
    hardness_debt_hard_recovery_threshold: float = 3.0
    hardness_debt_recovery_easy_weight: float = 0.0
    hardness_debt_recovery_medium_weight: float = 0.30
    hardness_debt_recovery_hard_weight: float = 0.70
    hardness_debt_stale_steps: int = 8
    hardness_debt_stale_reset_to: float = 3.0
    hardness_debt_stale_escape_steps: int = 8
    hardness_debt_stale_easy_weight: float = 0.05
    hardness_debt_stale_medium_weight: float = 0.55
    hardness_debt_stale_hard_weight: float = 0.40
    hardness_debt_temp_boost_max: float = 0.30
    hardness_debt_penalty_boost_max: float = 0.30

    # ── All-Easy Exploration ──────────────────────────────────────────────
    all_easy_explore_trigger: int = 2
    all_easy_explore_steps: int = 10
    all_easy_explore_num_candidates: int = 6
    all_easy_explore_temp_boost: float = 1.20
    all_easy_explore_top_p_boost: float = 0.15
    all_easy_explore_penalty_boost: float = 0.50

    # ── Contrastive replay + GRPO ranking controls ──────────────────────
    proposer_contrastive_replay_enabled: bool = True
    proposer_contrastive_replay_size: int = 256
    proposer_contrastive_pos_bonus: float = 0.08
    proposer_contrastive_neg_penalty: float = 0.08
    proposer_easy_reward_floor: float = -0.35
    proposer_all_easy_rank_spread: float = 0.08
    grpo_pairwise_ranking_enabled: bool = True
    grpo_pairwise_ranking_weight: float = 0.15
    grpo_pairwise_margin: float = 0.10
    grpo_pairwise_easy_penalty: float = 0.12

    # ── Early Fail-Fast / Recovery ───────────────────────────────────────
    proposer_early_failfast_enabled: bool = True
    proposer_early_failfast_stop: bool = False
    proposer_early_failfast_recover: bool = True
    proposer_early_failfast_recover_steps: int = 20
    proposer_early_stage1_u_step: int = 12
    proposer_early_stage2_u_step: int = 24
    proposer_early_hard_stop_min_u_step: int = 80
    proposer_early_candidate_non_easy_rate_min: float = 0.08
    proposer_early_all_easy_rate_max: float = 0.93
    proposer_early_reward_clipped_rate_max: float = 0.85
    proposer_early_selected_non_easy_rate_min: float = 0.10
    proposer_early_solver_updates_min: int = 1
    proposer_early_max_collapse_streak: int = 3

    # ── Runtime Safety / Health Gates ────────────────────────────────────
    fail_on_step_error: bool = True
    max_consecutive_step_errors: int = 0
    max_total_step_errors: int = 0
    generation_failfast_enabled: bool = True
    generation_failfast_consecutive_skips: int = 5
    generation_failfast_window: int = 20
    generation_failfast_min_success_rate: float = 0.10

    # ── Generation Reward Weights ────────────────────────────────────────
    reward_spec_weight: float = 0.65
    reward_cycle_weight: float = 0.20
    reward_diversity_weight: float = 0.10
    reward_contradiction_weight: float = 0.20
    min_spec_quality_for_update: float = 0.35
    min_spec_qa_pairs: int = 2
    max_expected_words: int = 8
    max_question_words: int = 24

    # ── Reference-answer Scoring ─────────────────────────────────────────
    use_ref_answer_scoring: bool = False
    verification_use_reference_solver: bool = False
    use_self_clip_reward_scoring: bool = False

    # ── Spec Quality ─────────────────────────────────────────────────────
    unicorn_generation_enabled: bool = True
    unicorn_target_difficulty: str = "medium"
    unicorn_spec_rejection_enabled: bool = True
    unicorn_spec_max_retries: int = 2
    unicorn_spec_min_quality: float = 0.55
    unicorn_spec_min_alignment: float = 0.55

    # ── KL Control ───────────────────────────────────────────────────────
    kl_coef: float = 0.01
    kl_target: float = 0.02
    kl_adapt_rate: float = 0.10
    kl_min: float = 0.001       # FIXED: was 1e-8, causing KL coef to collapse to ~0
    kl_max: float = 1e2

    # ── Baselines ────────────────────────────────────────────────────────
    baseline_momentum: float = 0.6

    # ── Replay Buffer ────────────────────────────────────────────────────
    replay_buffer_size: int = 1
    replay_min_reward: float = 1.10
    replay_max_staleness: int = 1

    # ── Gen-Mix Ratio (understanding step: mix generated images) ─────────
    gen_mix_ratio_start: float = 0.0
    gen_mix_ratio_max: float = 0.0
    gen_mix_ratio_warmup_steps: int = 1

    # ── Reward EMA ───────────────────────────────────────────────────────
    reward_ema_momentum: float = 0.95

    # ── Logging ──────────────────────────────────────────────────────────
    seed: int = 42
    log_every: int = 1
    save_every: int = 500
    max_checkpoints: int = 5
    clear_cache_every: int = 25
    save_generated_images_every: int = 0

    # ── W&B ──────────────────────────────────────────────────────────────
    wandb_mode: str = "disabled"
    wandb_project: str = "self-evolving-uug-vargpt"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_images_every: int = 0

    # ── Image Folder Mode ─────────────────────────────────────────────────
    image_folder: Optional[str] = None  # path to folder of images (bypasses dataset JSON)

    # ── Resume ───────────────────────────────────────────────────────────
    resume_from: Optional[str] = None
    start_step: int = 0

    @classmethod
    def from_finetuning_args(cls, finetuning_args) -> "SelfEvolvingConfig":
        """Create config from LLaMA-Factory's FinetuningArguments.

        Reads ``se_*`` prefixed fields from finetuning_args and maps them
        to SelfEvolvingConfig fields.
        """
        kwargs = {}
        for f in cls.__dataclass_fields__:
            se_key = f"se_{f}"
            if hasattr(finetuning_args, se_key):
                kwargs[f] = getattr(finetuning_args, se_key)
            elif hasattr(finetuning_args, f):
                kwargs[f] = getattr(finetuning_args, f)
        return cls(**kwargs)
