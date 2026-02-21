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
    grad_accum_steps: int = 4
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
    num_generations: int = 4             # K candidate images per G-step
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
    proposer_num_candidates: int = 3               # K candidate questions per proposer call
    proposer_spot_check_samples: int = 2           # solver samples for spot-checking
    proposer_gen_reward_enabled: bool = True
    proposer_gen_entropy_weight: float = 0.7
    proposer_gen_baseline_momentum: float = 0.6
    gen_step_solver_update_enabled: bool = False
    reset_proposer_baseline: bool = False

    # ── Reward Shaping ───────────────────────────────────────────────────
    solver_soft_gamma: float = 0.7
    solver_use_temperature_mix: bool = True
    solver_temp_min: float = 0.7
    solver_temp_max: float = 1.3
    solver_top_p_min: float = 0.5
    solver_top_p_max: float = 1.0
    sc_entropy_min: float = 0.15
    sc_entropy_max: float = 1.2
    sc_margin_max: float = 0.90
    sc_informative_ratio_min: float = 0.25
    sc_negative_weight: float = 0.25
    easy_solver_penalty_scale: float = 1.0
    skip_solver_update_when_uninformative: bool = True
    solver_always_update_with_informative_scaling: bool = True
    solver_update_min_scale: float = 0.20
    len_penalty_weight: float = 0.10
    len_penalty_target_words: int = 6
    prop_entropy_mu: float = 0.90
    prop_entropy_sigma: float = 0.35
    adaptive_prop_entropy_target: bool = False
    prop_entropy_ema_momentum: float = 0.90
    prop_entropy_mu_min: float = 0.40
    prop_entropy_mu_max: float = 1.5
    zero_entropy_reward_cap: float = 0.10
    proposer_unsolvable_reward_cap: float = 0.10
    solver_unsolvable_maj_threshold: float = 0.20
    proposer_non_objective_penalty: float = 0.20
    proposer_require_objective: bool = True
    solver_skip_update_on_easy: bool = True
    easy_update_majority_frac_threshold: float = 0.95
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
    difficulty_sampler_enabled: bool = False
    difficulty_sampler_window_size: int = 256
    difficulty_sampler_min_samples: int = 32
    difficulty_target_easy: float = 0.20
    difficulty_target_medium: float = 0.60
    difficulty_target_hard: float = 0.20
    difficulty_hard_min_entropy: float = 0.90
    difficulty_hard_max_margin: float = 0.35

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
    use_ref_answer_scoring: bool = True
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
    kl_min: float = 1e-8
    kl_max: float = 1e2

    # ── Baselines ────────────────────────────────────────────────────────
    baseline_momentum: float = 0.6

    # ── Replay Buffer ────────────────────────────────────────────────────
    replay_buffer_size: int = 1000
    replay_min_reward: float = 0.5
    replay_max_staleness: int = 500

    # ── Gen-Mix Ratio (understanding step: mix generated images) ─────────
    gen_mix_ratio_start: float = 0.02
    gen_mix_ratio_max: float = 0.25
    gen_mix_ratio_warmup_steps: int = 1000

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
