"""
Configuration dataclasses for the self-evolving training pipeline.
Ported from self_evolving/experiments/understanding.py and generation.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from .utils import DEFAULT_LORA_TARGETS


@dataclass
class UnderstandingSelfEvolvingConfig:
    """Config for understanding-only self-evolving training."""

    # Experiment identity
    experiment_name: str = "understanding_self_evolving"
    run_name: Optional[str] = None
    output_dir: str = "./runs"

    # Data
    data_dir: str = ""
    data_split: str = "all"  # train|val|test|all
    include_subfolders: Optional[Tuple[str, ...]] = None
    max_images: Optional[int] = None

    # Model
    model_name: str = "BLIP3o/BLIP3o-Model-8B"
    dtype: str = "bfloat16"
    cuda_device: int = 0
    device_map: str = "single"  # single|auto|cpu
    attn_implementation: str = "auto"  # auto|sdpa|eager|flash_attention_2|none

    # Optimization
    total_steps: int = 10000
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 4
    proposer_update_freq: int = 5

    # Decoding
    temp: float = 1.0
    top_p: float = 1.0
    max_new_tokens_solver: int = 128
    max_new_tokens_proposer: int = 128
    num_solver_samples: int = 5

    # Reward shaping
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
    adaptive_prop_entropy_target: bool = True
    prop_entropy_ema_momentum: float = 0.90
    prop_entropy_mu_min: float = 0.40
    prop_entropy_mu_max: float = 1.5
    zero_entropy_reward_cap: float = 0.10  # cap proposer reward when entropy=0 (unanimous)
    easy_question_penalty: float = 0.15  # subtract from proposer reward for trivially easy questions
    proposer_non_objective_penalty: float = 0.20  # subtract from proposer reward when question is subjective/open-ended
    proposer_require_objective: bool = True
    proposer_hardening_on_easy: bool = True
    proposer_hardening_max_retries: int = 2
    solver_skip_update_on_easy: bool = True
    easy_update_majority_frac_threshold: float = 0.95
    entropy_iqr_filter_enabled: bool = True
    entropy_iqr_window_size: int = 256
    entropy_iqr_min_samples: int = 32
    entropy_iqr_easy_quantile: float = 0.25
    entropy_iqr_easy_iqr_coef: float = 0.25
    entropy_iqr_min_threshold: float = 0.02
    entropy_iqr_max_threshold: float = 1.2
    entropy_iqr_filter_min_majority_frac: float = 0.80
    difficulty_sampler_enabled: bool = True
    difficulty_sampler_window_size: int = 256
    difficulty_sampler_min_samples: int = 32
    difficulty_sampler_max_retries: int = 1
    difficulty_target_easy: float = 0.20
    difficulty_target_medium: float = 0.60
    difficulty_target_hard: float = 0.20
    difficulty_hard_min_entropy: float = 0.90
    difficulty_hard_max_margin: float = 0.35

    # KL control
    kl_coef: float = 0.01
    kl_target: float = 0.02
    kl_adapt_rate: float = 0.10
    kl_min: float = 1e-8
    kl_max: float = 1e2

    # Baselines
    baseline_momentum: float = 0.9

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = DEFAULT_LORA_TARGETS

    # Repro + logging
    seed: int = 42
    deterministic: bool = True
    log_every: int = 1
    save_every: int = 500
    max_checkpoints: int = 5
    clear_cache_every: int = 25

    # W&B
    wandb_mode: str = "disabled"  # online|offline|disabled
    wandb_project: str = "self-evolving-uug"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_images_every: int = 0

    # Resume
    resume_from: Optional[str] = None
    start_step: int = 0


@dataclass
class GenerationSelfEvolvingConfig:
    """Config for generation-only self-evolving training."""

    experiment_name: str = "generation_self_evolving"
    run_name: Optional[str] = None
    output_dir: str = "./runs"

    # Data
    data_dir: str = ""
    data_split: str = "all"
    include_subfolders: Optional[Tuple[str, ...]] = None
    max_images: Optional[int] = None

    # Model
    model_name: str = "BLIP3o/BLIP3o-Model-8B"
    dtype: str = "bfloat16"
    cuda_device: int = 0
    device_map: str = "single"
    attn_implementation: str = "auto"

    # Optimization
    total_steps: int = 10000
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 4
    proposer_update_freq: int = 5
    generator_update_freq: int = 1
    enable_solver_updates: bool = False
    solver_update_freq: int = 0

    # Decoding
    temp: float = 1.0
    top_p: float = 1.0
    max_new_tokens_solver: int = 128
    max_new_tokens_proposer: int = 256
    max_new_tokens_caption: int = 96
    max_new_tokens_generator: int = 768
    num_solver_samples: int = 5
    num_solver_samples_spec: int = 3
    num_generations: int = 4

    # Generation backend
    generation_num_inference_steps: int = 30
    generation_guidance_scale: float = 2.0
    generation_height: int = 1024
    generation_width: int = 1024
    require_decoder_for_blip3o: bool = True
    allow_latent_visualization_fallback: bool = False
    strict_require_generation_tokens: bool = True
    generator_missing_trace_strategy: str = "proxy"  # proxy|skip|error
    verification_use_reference_solver: bool = True
    generator_update_rule: str = "reinforce"  # reinforce|dpo|grpo
    dpo_beta: float = 0.1
    dpo_label_smoothing: float = 0.0
    dpo_min_reward_gap: float = 0.0
    dpo_min_spec_gap: float = 0.0
    dpo_min_confidence_gap: float = 0.0
    dpo_max_contradiction: float = 1.0
    dpo_pair_selection: str = "best_worst"  # best_worst|best_hard_negative
    generator_proxy_max_ratio: float = 1.0
    grpo_clip_ratio: float = 0.2              # PPO-style importance ratio clipping for GRPO
    grpo_min_group_std: float = 1e-6          # skip GRPO update if reward std below this

    # Reward shaping
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
    adaptive_prop_entropy_target: bool = True
    prop_entropy_ema_momentum: float = 0.90
    prop_entropy_mu_min: float = 0.40
    prop_entropy_mu_max: float = 1.5
    zero_entropy_reward_cap: float = 0.10
    easy_question_penalty: float = 0.15
    proposer_non_objective_penalty: float = 0.20
    proposer_require_objective: bool = True
    proposer_hardening_on_easy: bool = True
    proposer_hardening_max_retries: int = 2
    solver_skip_update_on_easy: bool = True
    easy_update_majority_frac_threshold: float = 0.95
    entropy_iqr_filter_enabled: bool = True
    entropy_iqr_window_size: int = 256
    entropy_iqr_min_samples: int = 32
    entropy_iqr_easy_quantile: float = 0.25
    entropy_iqr_easy_iqr_coef: float = 0.25
    entropy_iqr_min_threshold: float = 0.02
    entropy_iqr_max_threshold: float = 1.2
    entropy_iqr_filter_min_majority_frac: float = 0.80
    difficulty_sampler_enabled: bool = True
    difficulty_sampler_window_size: int = 256
    difficulty_sampler_min_samples: int = 32
    difficulty_sampler_max_retries: int = 1
    difficulty_target_easy: float = 0.20
    difficulty_target_medium: float = 0.60
    difficulty_target_hard: float = 0.20
    difficulty_hard_min_entropy: float = 0.90
    difficulty_hard_max_margin: float = 0.35
    reward_spec_weight: float = 0.65
    reward_cycle_weight: float = 0.20
    reward_diversity_weight: float = 0.10
    reward_contradiction_weight: float = 0.20
    min_spec_quality_for_update: float = 0.35
    min_spec_qa_pairs: int = 2
    max_expected_words: int = 8
    max_question_words: int = 24

    # KL control
    kl_coef: float = 0.01
    kl_target: float = 0.02
    kl_adapt_rate: float = 0.10
    kl_min: float = 1e-8
    kl_max: float = 1e2

    # Baselines
    baseline_momentum: float = 0.9

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = DEFAULT_LORA_TARGETS

    # Repro + logging
    seed: int = 42
    deterministic: bool = True
    log_every: int = 1
    save_every: int = 500
    max_checkpoints: int = 5
    clear_cache_every: int = 25
    save_generated_images_every: int = 0

    # W&B
    wandb_mode: str = "disabled"
    wandb_project: str = "self-evolving-uug"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_images_every: int = 0

    # Resume
    resume_from: Optional[str] = None
    start_step: int = 0


@dataclass
class UnifiedSelfEvolvingConfig(GenerationSelfEvolvingConfig):
    """Config for unified (alternating understanding + generation) training."""

    experiment_name: str = "unified_self_evolving"
    understanding_steps_per_cycle: int = 3
    generation_steps_per_cycle: int = 2
    synthetic_solver_update_freq: int = 1
    synthetic_solver_hard_only: bool = False
    solver_hardness_min_entropy: float = 0.2

    # ---- Self-evolving feedback loop ---- #
    # Reference-answer log-prob scoring for generation.
    # When False, uses existing multi-component scoring (spec+cycle+diversity).
    use_ref_answer_scoring: bool = False

    # Replay buffer: stores best generated images for mixing into understanding training.
    replay_buffer_size: int = 1000
    replay_min_reward: float = 0.5
    replay_max_staleness: int = 500

    # Generated-image mixing ratio for understanding step.
    # Linearly ramps from start → max over warmup_steps.
    gen_mix_ratio_start: float = 0.02     # initial ratio (conservative start)
    gen_mix_ratio_max: float = 0.25       # cap (real data always >= 75%)
    gen_mix_ratio_warmup_steps: int = 1000  # longer warmup for stability

    # Generator reward EMA tracking (for monitoring / logging)
    reward_ema_momentum: float = 0.95
