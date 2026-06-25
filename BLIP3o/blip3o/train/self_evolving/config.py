"""
Configuration dataclasses for the self-evolving training pipeline.
Ported from self_evolving/experiments/understanding.py and generation.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from .utils import (
    DEFAULT_DIT_LORA_TARGETS,
    DEFAULT_SOLVER_MERGER_LORA_TARGETS,
    DEFAULT_TEXT_LORA_TARGETS,
)


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
    grad_accum_steps: int = 1
    proposer_update_freq: int = 1  # update proposer every understanding step (was 5 — too sparse)

    # Decoding
    temp: float = 1.0
    top_p: float = 1.0
    max_new_tokens_solver: int = 128
    max_new_tokens_proposer: int = 128
    num_solver_samples: int = 7

    # Reward shaping
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
    solver_update_on_low_info_easy: bool = False
    solver_low_info_easy_penalty_scale: float = 2.5
    skip_solver_update_when_uninformative: bool = True
    solver_always_update_with_informative_scaling: bool = False
    solver_update_min_scale: float = 0.20
    len_penalty_weight: float = 0.10
    len_penalty_target_words: int = 6
    prop_entropy_mu: float = 0.90
    prop_entropy_sigma: float = 0.35
    adaptive_prop_entropy_target: bool = True  # re-enabled: EMA no longer incorporates zero-entropy observations
    prop_entropy_ema_momentum: float = 0.90
    prop_entropy_mu_min: float = 0.40
    prop_entropy_mu_max: float = 1.5
    zero_entropy_reward_cap: float = 0.10  # hard negative magnitude when entropy=0 (trivially easy)
    proposer_easy_reward_cap: float = 0.20  # max proposer reward for collapsed/easy questions
    proposer_easy_gotcha_reward_cap: float = 0.50  # max reward when dual-track answers disagree
    proposer_unsolvable_reward_cap: float = 0.10  # hard negative magnitude when question is unsolvable
    solver_unsolvable_maj_threshold: float = 0.20  # majority fraction at or below this → question treated as unsolvable
    proposer_non_objective_penalty: float = 0.20  # subtract from proposer reward when question is subjective/open-ended
    proposer_low_info_majority_penalty: float = 0.50
    proposer_slot_compiler_enabled: bool = True
    proposer_slot_compiler_strict: bool = True
    proposer_trivial_archetype_penalty: float = 0.25
    proposer_answer_family_repeat_penalty: float = 0.25
    proposer_answer_family_repeat_target: float = 0.25
    proposer_candidate_noncanonical_penalty: float = 0.12
    proposer_candidate_low_info_penalty: float = 0.10
    solver_noncanonical_answer_penalty: float = 0.10
    solver_low_info_answer_penalty: float = 0.08
    curriculum_arm_enabled: bool = True
    curriculum_arm_prompt_enabled: bool = True
    curriculum_arm_ema_momentum: float = 0.90
    curriculum_arm_progress_weight: float = 0.20
    curriculum_arm_underuse_weight: float = 0.12
    curriculum_arm_easy_penalty_weight: float = 0.15
    curriculum_arm_solver_gain_weight: float = 0.10
    curriculum_arm_prompt_temp: float = 0.60
    curriculum_arm_candidate_bonus: float = 0.08
    curriculum_arm_reward_scale: float = 0.10
    replay_priority_enabled: bool = True
    replay_priority_hardness_weight: float = 0.50
    replay_priority_update_weight: float = 0.30
    replay_priority_novelty_weight: float = 0.20
    replay_anchor_inject_k: int = 2
    replay_anchor_inject_easy_streak: int = 2
    proposer_require_objective: bool = True
    # Multi-candidate generation (single proposer call, pick hardest via spot-check)
    proposer_num_candidates: int = 5      # K candidate questions generated in one proposer call
    proposer_spot_check_samples: int = 3  # 3 samples give ternary entropy for candidate selection
    proposer_spot_entropy_min_gate: float = 0.05
    # Reasoning-first proposer schema validation.
    proposer_reasoning_min_domains: int = 2
    proposer_reasoning_require_non_relation: bool = True
    proposer_reasoning_min_chain_words: int = 8
    proposer_certificate_enabled: bool = True
    proposer_certificate_min_score: float = 0.55
    proposer_certificate_weight: float = 0.75
    proposer_certificate_strict_struct: bool = True
    # Zero-extra-compute proposer warm start: when entropy signal is dead at
    # cold start, use structure/certificate shaping to bootstrap proposer.
    proposer_warm_start_enabled: bool = True
    proposer_warm_start_max_steps: int = 30
    proposer_warm_start_exit_window: int = 5
    proposer_warm_start_exit_consecutive: int = 2
    proposer_warm_start_entropy_exit_threshold: float = 0.10
    proposer_warm_start_easy_reject_penalty_scale: float = 0.0
    proposer_warm_start_certificate_weight: float = 0.50
    # Logit-Margin Difficulty Signal (LMDS): continuous difficulty measurement
    # from the solver's internal confidence on its greedy (V-Zero) answer.
    # Solver Token Entropy (STE): full softmax entropy of logit distribution
    # at each answer token.  Unlike logit margin (top1-top2 gap), STE captures
    # genuine multi-way uncertainty.  Forced-choice "A or B?" gives H ≈ ln(2)
    # ≈ 0.69; genuinely hard open-ended questions give H >> 1.0.  This makes
    # STE naturally resistant to forced-choice gaming.
    solver_token_entropy_enabled: bool = True
    solver_token_entropy_tokens: int = 5  # first K answer tokens to analyze
    solver_token_entropy_window_size: int = 128  # rolling window for quantile normalization
    solver_token_entropy_sigmoid_alpha: float = 1.5  # sigmoid steepness (cold-window fallback)
    solver_token_entropy_sigmoid_beta: float = 2.0  # sigmoid midpoint (in nats; ln(2)≈0.69, tuned for entropy scale)
    solver_token_entropy_aggregation: str = "max"  # rebuttal ablation: max or mean over first K answer tokens
    proposer_ste_primary_weight: float = 0.70  # warm-start primary STE weight
    proposer_sample_entropy_weight: float = 0.30  # complementary self-consistency/sample-entropy weight
    proposer_ste_reward_weight: float = 0.30  # weight after warm-start (complementary to sample entropy)
    proposer_ste_warm_start_weight: float = 0.70  # weight during warm-start (primary signal)
    # Prompt-Perturbed Sampling (PPS): use 7 different prompt framings instead
    # of repeating the same prompt.  Entropy now measures ROBUSTNESS of
    # understanding rather than stochastic variation.
    solver_pps_enabled: bool = True
    # Hardness debt controller: fast steering away from prolonged easy collapse.
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
    proposer_easy_reward_floor: float = -0.35
    proposer_all_easy_rank_spread: float = 0.08
    # Contrastive self-consistency replay (no extra model calls).
    proposer_contrastive_replay_enabled: bool = True
    proposer_contrastive_replay_size: int = 256
    proposer_contrastive_pos_bonus: float = 0.08
    proposer_contrastive_neg_penalty: float = 0.08
    # Early fail-fast health checks to stop collapsed runs quickly.
    proposer_early_failfast_enabled: bool = True
    proposer_early_failfast_stop: bool = False
    proposer_early_failfast_recover: bool = True
    proposer_early_failfast_recover_steps: int = 20
    proposer_early_hard_stop_min_u_step: int = 80
    proposer_early_step1: int = 12
    proposer_early_step2: int = 24
    proposer_early_candidate_non_easy_min: float = 0.08
    proposer_early_all_easy_rate_max: float = 0.93
    proposer_early_reward_clipped_rate_max: float = 0.85
    proposer_early_selected_non_easy_min: float = 0.10
    proposer_early_solver_updates_min: int = 1
    proposer_early_collapse_streak_max: int = 3
    proposer_health_window_size: int = 256
    solver_skip_update_on_easy: bool = True
    easy_update_majority_frac_threshold: float = 0.95
    acceptance_require_non_easy: bool = True
    rejected_question_penalty: float = 0.35
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
    difficulty_target_easy: float = 0.10
    difficulty_target_medium: float = 0.50
    difficulty_target_hard: float = 0.40
    difficulty_hard_min_entropy: float = 0.90
    difficulty_hard_max_margin: float = 0.35

    # Proposer update algorithm
    # "reinforce" → single-sample REINFORCE with EMA baseline (original).
    # "grpo"      → group-relative policy optimization: reuse all K proposer candidates
    #               as the GRPO group, normalizing advantages across the group.
    #               No extra inference cost — candidates are already generated for selection.
    proposer_update_rule: str = "grpo"
    # Gen-phase proposer: how many specs to sample for the GRPO group.
    # Understanding phase always uses proposer_num_candidates.
    proposer_grpo_gen_group_size: int = 3
    # Score extra GRPO candidates with a configurable solver spot-check instead of
    # assigning them a neutral reward of 0.0.  This gives real differential
    # signal between candidates, preventing the GRPO loss from collapsing to
    # zero at EMA equilibrium.  Costs ~4 extra solver forward passes per step.
    score_grpo_extras: bool = True
    # Temperature multiplier for extra GRPO candidate generation.  Higher values
    # increase diversity, making it more likely that at least one extra produces
    # a non-easy question with a different reward → real gradient signal.
    grpo_extra_temp_multiplier: float = 2.0
    # Number of solver spot-check samples for extra GRPO candidates.
    # 3 samples give ternary entropy outcomes (0, 0.637, 1.099) vs binary
    # with 2 samples (0 or 0.693), enabling differential reward signal.
    grpo_extra_sc_samples: int = 3
    # Text-only proposer shaping (no extra model calls):
    # rewards harder-looking templates and penalizes repeated templates.
    proposer_question_history_size: int = 256
    proposer_text_bonus_max: float = 0.20
    proposer_text_penalty_max: float = 0.35
    proposer_repeat_penalty_unit: float = 0.08
    proposer_repeat_penalty_max: float = 0.35
    proposer_text_step_dup_penalty: float = 0.15
    proposer_template_cooldown_steps: int = 24
    proposer_template_cooldown_penalty: float = 0.35
    proposer_strategy_window_size: int = 256
    proposer_strategy_target_share: float = 0.12
    proposer_strategy_overuse_penalty: float = 0.20
    proposer_strategy_underuse_bonus: float = 0.08
    proposer_anchor_replay_size: int = 256
    proposer_anchor_strategy_bonus: float = 0.03
    proposer_anchor_template_bonus: float = 0.02
    proposer_anchor_min_reward: float = 0.35
    proposer_reward_mode: str = "hybrid"  # gaussian|band|hybrid
    proposer_band_reward_weight: float = 0.90  # used when proposer_reward_mode=hybrid
    easy_constraint_enabled: bool = True
    easy_constraint_target_rate: float = 0.18
    easy_constraint_lr: float = 0.20
    easy_constraint_lambda_max: float = 3.0
    easy_constraint_penalty_scale: float = 0.60
    easy_constraint_selection_scale: float = 0.50
    easy_rate_ema_momentum: float = 0.90
    adaptive_exploration_enabled: bool = True
    exploration_easy_rate_threshold: float = 0.45
    exploration_temp_boost_max: float = 0.90
    exploration_top_p_boost_max: float = 0.15
    exploration_penalty_boost_max: float = 2.00
    collapse_detector_enabled: bool = True
    collapse_easy_rate_threshold: float = 0.65
    collapse_std_threshold: float = 0.10
    collapse_streak_trigger: int = 3
    collapse_lambda_boost: float = 0.25
    collapse_cooldown_penalty_boost: float = 0.30
    collapse_std_window_size: int = 32
    all_easy_explore_trigger: int = 2
    all_easy_explore_steps: int = 10
    all_easy_explore_num_candidates: int = 6
    all_easy_explore_temp_boost: float = 1.20
    all_easy_explore_top_p_boost: float = 0.15
    all_easy_explore_penalty_boost: float = 0.50
    # Optional exploration noise when GRPO rewards collapse to identical values.
    grpo_degenerate_noise_enabled: bool = True
    grpo_degenerate_noise_sigma: float = 0.03
    grpo_degenerate_noise_std_threshold: float = 1e-6
    grpo_pairwise_ranking_enabled: bool = True
    grpo_pairwise_ranking_weight: float = 0.15
    grpo_pairwise_margin: float = 0.10
    grpo_pairwise_easy_penalty: float = 0.12
    # Generation-phase proposer GRPO: unverified extras are assigned
    # (chosen_reward - margin), clamped to [-1, 1], so they cannot outrank
    # the verified chosen candidate.
    proposer_grpo_unverified_extra_margin: float = 0.02
    # Confidence-based proposer reward: instead of binary -cap for entropy=0,
    # use the solver's mean logprob as a continuous penalty scaler.
    # KL control
    kl_coef: float = 0.01
    kl_target: float = 0.02
    kl_adapt_rate: float = 0.10
    kl_min: float = 0.001
    kl_max: float = 1e2

    # Baselines
    baseline_momentum: float = 0.6  # was 0.9: lower so advantage doesn't collapse when rewards are uniformly negative

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = DEFAULT_TEXT_LORA_TARGETS
    solver_merger_lora_enabled: bool = True
    solver_merger_lora_r: int = 4
    solver_merger_lora_alpha: int = 8
    solver_merger_lora_lr: float = 2e-7
    solver_merger_lora_target_modules: Tuple[str, ...] = DEFAULT_SOLVER_MERGER_LORA_TARGETS
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Repro + logging
    seed: int = 42
    deterministic: bool = True
    log_every: int = 1
    save_every: int = 50
    max_checkpoints: int = 5
    clear_cache_every: int = 25
    code_run_registry_enabled: bool = True
    code_run_registry_dir: Optional[str] = None

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
    grad_accum_steps: int = 1
    proposer_update_freq: int = 1
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
    num_solver_samples: int = 7
    num_solver_samples_spec: int = 3
    num_generations: int = 3

    # Generation backend
    generation_num_inference_steps: int = 50
    generation_guidance_scale: float = 2.0
    generation_height: int = 896
    generation_width: int = 896
    require_decoder_for_blip3o: bool = True
    allow_latent_visualization_fallback: bool = False
    strict_require_generation_tokens: bool = True
    generator_missing_trace_strategy: str = "skip"  # proxy|skip|error
    verification_use_reference_solver: bool = False  # use trained solver LoRA for mutual supervision
    use_self_clip_reward_scoring: bool = False  # CLIP-style reward from model's own frozen embeddings
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
    use_diverse_prompts: bool = False         # Exp 2: generate N diverse prompts per image instead of 1
    enable_frozen_judge: bool = False         # Exp 3: use EMA-updated frozen solver as judge
    judge_ema_decay: float = 0.99
    judge_gpu_id: Optional[int] = None        # offload judge to separate GPU if needed
    unicorn_generation_enabled: bool = True
    unicorn_target_difficulty: str = "medium"  # easy|medium|hard
    unicorn_spec_rejection_enabled: bool = True
    unicorn_spec_max_retries: int = 2
    unicorn_spec_min_quality: float = 0.55
    unicorn_spec_min_alignment: float = 0.55
    unicorn_reconstruction_sft_enabled: bool = True
    unicorn_reconstruction_buffer_size: int = 512
    unicorn_reconstruction_step_freq: int = 1
    unicorn_reconstruction_updates_per_step: int = 2
    unicorn_reconstruction_min_quality: float = 0.55
    unicorn_reconstruction_enable_proposer: bool = True
    unicorn_reconstruction_enable_generator: bool = True
    dit_update_enabled: bool = False
    require_dit_update: bool = False
    dit_update_freq: int = 1
    dit_lr: float = 5e-7
    dit_weight_decay: float = 0.01
    dit_grad_clip: float = 1.0
    dit_grad_accum_steps: int = 1
    dit_conditioning_dropout: float = 0.10
    dit_loss_weight: float = 1.0
    dit_prompt_suffix_token_id: int = 151665
    # Research default: train diffusion denoiser adapters instead of full DiT weights.
    # Full-DiT unfreeze remains available as an ablation via dit_lora_enabled=False.
    dit_lora_enabled: bool = True
    dit_lora_r: int = 16
    dit_lora_alpha: int = 32
    dit_lora_dropout: float = 0.0
    dit_lora_target_modules: Tuple[str, ...] = DEFAULT_DIT_LORA_TARGETS
    # Joint LLM+DiT training: remove torch.no_grad() from _prepare_conditioning so
    # denoising loss gradients flow back through z_latents into the generator LoRA.
    # This trains the text conditioning encoder jointly with the DiT denoiser.
    dit_joint_conditioning_train: bool = False
    # LR for generator LoRA params when trained jointly with DiT (usually same as dit_lr).
    dit_joint_conditioning_lr: float = 5e-7
    # Reward-weighted denoising: scale denoising loss by image-quality reward from solver.
    # loss = reward_weight * MSE(noise_pred, target). Implements RWR (continuous GRPO).
    # When 0.0, pure SFT denoising (current default). Requires dit_joint_conditioning_train.
    dit_reward_loss_weight: float = 0.0
    # Proposer reward in generation phase (SUDER dual-reward).
    # When True: proposer LoRA is updated during generation steps using a joint reward
    # signal that combines entropy (hardness of QA on the generated image) and image
    # quality.  The two signals are blended with proposer_gen_entropy_weight.
    proposer_gen_reward_enabled: bool = False
    # Blend coefficient α for the joint generation proposer reward:
    #   reward = α * gaussian_reward(entropy) + (1-α) * image_quality
    # α=1.0 → pure entropy (same objective as understanding phase, recommended)
    # α=0.0 → pure image quality (original behaviour)
    # α=0.7 → default: mostly entropy-driven, with quality as a regulariser
    proposer_gen_entropy_weight: float = 0.7
    # Separate EMA momentum for the proposer generation-phase baseline.
    # Kept separate from the understanding-phase proposer_baseline to avoid contamination.
    proposer_gen_baseline_momentum: float = 0.6
    # When True: also update the solver model on generated images during generation
    # steps (joint understanding). The solver already runs on generated images for
    # scoring — this reuses those rollouts to train the solver too, effectively
    # making generation steps also function as understanding steps on synthetic data.
    gen_step_solver_update_enabled: bool = False
    # One-shot flag: reset proposer_baseline (and proposer_gen_baseline) to 0.0 on
    # resume. Also clears the entropy/difficulty history windows so the IQR filter
    # re-warms from scratch. Use this when resuming after a bug fix that caused the
    # baseline to lock (e.g. the baseline-clamp equilibrium bug). Has no effect on
    # fresh runs. Remove this flag after the first checkpoint is saved post-resume.
    reset_proposer_baseline: bool = False

    # ---- Imageless proposer mode (E5: fully synthetic self-evolving loop) ---- #
    # When True, the proposer generates specs from text topics/themes instead of
    # seeing a source image.  This enables a FULLY synthetic self-evolving loop
    # where no external images are provided at any point:
    #   topic → proposer (text-only) → prompt + QA spec → generator → image
    #   → solver answers QA on generated image → rewards → all components update.
    # The model teaches itself using only its own generations.
    imageless_proposer_mode: bool = False

    # Reward shaping
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
    solver_update_on_low_info_easy: bool = False
    solver_low_info_easy_penalty_scale: float = 2.5
    skip_solver_update_when_uninformative: bool = True
    solver_always_update_with_informative_scaling: bool = False
    solver_update_min_scale: float = 0.20
    len_penalty_weight: float = 0.10
    len_penalty_target_words: int = 6
    prop_entropy_mu: float = 0.90
    prop_entropy_sigma: float = 0.35
    adaptive_prop_entropy_target: bool = True  # re-enabled: EMA no longer incorporates zero-entropy observations
    prop_entropy_ema_momentum: float = 0.90
    prop_entropy_mu_min: float = 0.40
    prop_entropy_mu_max: float = 1.5
    zero_entropy_reward_cap: float = 0.10
    proposer_easy_reward_cap: float = 0.20
    proposer_easy_gotcha_reward_cap: float = 0.50
    proposer_unsolvable_reward_cap: float = 0.10
    solver_unsolvable_maj_threshold: float = 0.20
    proposer_non_objective_penalty: float = 0.20
    proposer_low_info_majority_penalty: float = 0.50
    proposer_slot_compiler_enabled: bool = True
    proposer_slot_compiler_strict: bool = True
    proposer_trivial_archetype_penalty: float = 0.25
    proposer_answer_family_repeat_penalty: float = 0.25
    proposer_answer_family_repeat_target: float = 0.25
    proposer_candidate_noncanonical_penalty: float = 0.12
    proposer_candidate_low_info_penalty: float = 0.10
    solver_noncanonical_answer_penalty: float = 0.10
    solver_low_info_answer_penalty: float = 0.08
    curriculum_arm_enabled: bool = True
    curriculum_arm_prompt_enabled: bool = True
    curriculum_arm_ema_momentum: float = 0.90
    curriculum_arm_progress_weight: float = 0.20
    curriculum_arm_underuse_weight: float = 0.12
    curriculum_arm_easy_penalty_weight: float = 0.15
    curriculum_arm_solver_gain_weight: float = 0.10
    curriculum_arm_prompt_temp: float = 0.60
    curriculum_arm_candidate_bonus: float = 0.08
    curriculum_arm_reward_scale: float = 0.10
    replay_priority_enabled: bool = True
    replay_priority_hardness_weight: float = 0.50
    replay_priority_update_weight: float = 0.30
    replay_priority_novelty_weight: float = 0.20
    replay_anchor_inject_k: int = 2
    replay_anchor_inject_easy_streak: int = 2
    proposer_require_objective: bool = True
    # Multi-candidate generation (single proposer call, pick hardest via spot-check)
    proposer_num_candidates: int = 5      # K candidate questions generated in one proposer call
    proposer_spot_check_samples: int = 3  # 3 samples give ternary entropy for candidate selection
    proposer_spot_entropy_min_gate: float = 0.05
    # Reasoning-first proposer schema validation.
    proposer_reasoning_min_domains: int = 2
    proposer_reasoning_require_non_relation: bool = True
    proposer_reasoning_min_chain_words: int = 8
    proposer_certificate_enabled: bool = True
    proposer_certificate_min_score: float = 0.55
    proposer_certificate_weight: float = 0.75
    proposer_certificate_strict_struct: bool = True
    # Zero-extra-compute proposer warm start: when entropy signal is dead at
    # cold start, use structure/certificate shaping to bootstrap proposer.
    proposer_warm_start_enabled: bool = True
    proposer_warm_start_max_steps: int = 30
    proposer_warm_start_exit_window: int = 5
    proposer_warm_start_exit_consecutive: int = 2
    proposer_warm_start_entropy_exit_threshold: float = 0.10
    proposer_warm_start_easy_reject_penalty_scale: float = 0.0
    proposer_warm_start_certificate_weight: float = 0.50
    # Logit-Margin Difficulty Signal (LMDS): continuous difficulty measurement
    # from the solver's internal confidence on its greedy (V-Zero) answer.
    # Solver Token Entropy (STE): full softmax entropy of logit distribution
    # at each answer token.  Unlike logit margin (top1-top2 gap), STE captures
    # genuine multi-way uncertainty.  Forced-choice "A or B?" gives H ≈ ln(2)
    # ≈ 0.69; genuinely hard open-ended questions give H >> 1.0.  This makes
    # STE naturally resistant to forced-choice gaming.
    solver_token_entropy_enabled: bool = True
    solver_token_entropy_tokens: int = 5  # first K answer tokens to analyze
    solver_token_entropy_window_size: int = 128  # rolling window for quantile normalization
    solver_token_entropy_sigmoid_alpha: float = 1.5  # sigmoid steepness (cold-window fallback)
    solver_token_entropy_sigmoid_beta: float = 2.0  # sigmoid midpoint (in nats; ln(2)≈0.69, tuned for entropy scale)
    solver_token_entropy_aggregation: str = "max"  # rebuttal ablation: max or mean over first K answer tokens
    proposer_ste_primary_weight: float = 0.70  # warm-start primary STE weight
    proposer_sample_entropy_weight: float = 0.30  # complementary self-consistency/sample-entropy weight
    proposer_ste_reward_weight: float = 0.30  # weight after warm-start (complementary to sample entropy)
    proposer_ste_warm_start_weight: float = 0.70  # weight during warm-start (primary signal)
    # Prompt-Perturbed Sampling (PPS): use 7 different prompt framings instead
    # of repeating the same prompt.  Entropy now measures ROBUSTNESS of
    # understanding rather than stochastic variation.
    solver_pps_enabled: bool = True
    # Hardness debt controller: fast steering away from prolonged easy collapse.
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
    proposer_easy_reward_floor: float = -0.35
    proposer_all_easy_rank_spread: float = 0.08
    # Contrastive self-consistency replay (no extra model calls).
    proposer_contrastive_replay_enabled: bool = True
    proposer_contrastive_replay_size: int = 256
    proposer_contrastive_pos_bonus: float = 0.08
    proposer_contrastive_neg_penalty: float = 0.08
    # Early fail-fast health checks to stop collapsed runs quickly.
    proposer_early_failfast_enabled: bool = True
    proposer_early_failfast_stop: bool = False
    proposer_early_failfast_recover: bool = True
    proposer_early_failfast_recover_steps: int = 20
    proposer_early_hard_stop_min_u_step: int = 80
    proposer_early_step1: int = 12
    proposer_early_step2: int = 24
    proposer_early_candidate_non_easy_min: float = 0.08
    proposer_early_all_easy_rate_max: float = 0.93
    proposer_early_reward_clipped_rate_max: float = 0.85
    proposer_early_selected_non_easy_min: float = 0.10
    proposer_early_solver_updates_min: int = 1
    proposer_early_collapse_streak_max: int = 3
    proposer_health_window_size: int = 256
    solver_skip_update_on_easy: bool = True
    easy_update_majority_frac_threshold: float = 0.95
    acceptance_require_non_easy: bool = True
    rejected_question_penalty: float = 0.35
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
    difficulty_target_easy: float = 0.10
    difficulty_target_medium: float = 0.50
    difficulty_target_hard: float = 0.40
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
    kl_min: float = 0.001
    kl_max: float = 1e2

    # Proposer update algorithm (mirrors UnderstandingSelfEvolvingConfig — must be kept in sync)
    # "reinforce" → single-sample REINFORCE with EMA baseline (original).
    # "grpo"      → group-relative policy optimization on K proposer candidates.
    proposer_update_rule: str = "grpo"
    proposer_grpo_gen_group_size: int = 3
    # Score extra GRPO candidates with a configurable solver spot-check.
    score_grpo_extras: bool = True
    # Temperature multiplier for extra GRPO candidate generation.
    grpo_extra_temp_multiplier: float = 2.0
    # Number of solver spot-check samples for extra GRPO candidates.
    grpo_extra_sc_samples: int = 3
    # Text-only proposer shaping (no extra model calls):
    # rewards harder-looking templates and penalizes repeated templates.
    proposer_question_history_size: int = 256
    proposer_text_bonus_max: float = 0.20
    proposer_text_penalty_max: float = 0.35
    proposer_repeat_penalty_unit: float = 0.08
    proposer_repeat_penalty_max: float = 0.35
    proposer_text_step_dup_penalty: float = 0.15
    proposer_template_cooldown_steps: int = 24
    proposer_template_cooldown_penalty: float = 0.35
    proposer_strategy_window_size: int = 256
    proposer_strategy_target_share: float = 0.12
    proposer_strategy_overuse_penalty: float = 0.20
    proposer_strategy_underuse_bonus: float = 0.08
    proposer_anchor_replay_size: int = 256
    proposer_anchor_strategy_bonus: float = 0.03
    proposer_anchor_template_bonus: float = 0.02
    proposer_anchor_min_reward: float = 0.35
    proposer_reward_mode: str = "hybrid"  # gaussian|band|hybrid
    proposer_band_reward_weight: float = 0.90  # used when proposer_reward_mode=hybrid
    easy_constraint_enabled: bool = True
    easy_constraint_target_rate: float = 0.18
    easy_constraint_lr: float = 0.20
    easy_constraint_lambda_max: float = 3.0
    easy_constraint_penalty_scale: float = 0.60
    easy_constraint_selection_scale: float = 0.50
    easy_rate_ema_momentum: float = 0.90
    adaptive_exploration_enabled: bool = True
    exploration_easy_rate_threshold: float = 0.45
    exploration_temp_boost_max: float = 0.90
    exploration_top_p_boost_max: float = 0.15
    exploration_penalty_boost_max: float = 2.00
    collapse_detector_enabled: bool = True
    collapse_easy_rate_threshold: float = 0.65
    collapse_std_threshold: float = 0.10
    collapse_streak_trigger: int = 3
    collapse_lambda_boost: float = 0.25
    collapse_cooldown_penalty_boost: float = 0.30
    collapse_std_window_size: int = 32
    all_easy_explore_trigger: int = 2
    all_easy_explore_steps: int = 10
    all_easy_explore_num_candidates: int = 6
    all_easy_explore_temp_boost: float = 1.20
    all_easy_explore_top_p_boost: float = 0.15
    all_easy_explore_penalty_boost: float = 0.50
    # Optional exploration noise when GRPO rewards collapse to identical values.
    grpo_degenerate_noise_enabled: bool = True
    grpo_degenerate_noise_sigma: float = 0.03
    grpo_degenerate_noise_std_threshold: float = 1e-6
    grpo_pairwise_ranking_enabled: bool = True
    grpo_pairwise_ranking_weight: float = 0.15
    grpo_pairwise_margin: float = 0.10
    grpo_pairwise_easy_penalty: float = 0.12
    # Generation-phase proposer GRPO: unverified extras are assigned
    # (chosen_reward - margin), clamped to [-1, 1], so they cannot outrank
    # the verified chosen candidate.
    proposer_grpo_unverified_extra_margin: float = 0.02
    # Baselines
    baseline_momentum: float = 0.6  # was 0.9: lower so advantage doesn't collapse when rewards are uniformly negative

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = DEFAULT_TEXT_LORA_TARGETS
    solver_merger_lora_enabled: bool = True
    solver_merger_lora_r: int = 4
    solver_merger_lora_alpha: int = 8
    solver_merger_lora_lr: float = 2e-7
    solver_merger_lora_target_modules: Tuple[str, ...] = DEFAULT_SOLVER_MERGER_LORA_TARGETS
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Repro + logging
    seed: int = 42
    deterministic: bool = True
    log_every: int = 1
    save_every: int = 50
    max_checkpoints: int = 5
    clear_cache_every: int = 25
    save_generated_images_every: int = 0
    code_run_registry_enabled: bool = True
    code_run_registry_dir: Optional[str] = None

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
    cycle_starts_with_generation: bool = False
    bootstrap_generated_pool_steps: int = 0
    synthetic_solver_update_freq: int = 0
    synthetic_solver_hard_only: bool = False
    solver_hardness_min_entropy: float = 0.2

    # ---- Self-evolving feedback loop ---- #
    # Solver-derived reference-answer log-prob scoring for generation (MODE B).
    # The Solver answers Qs on the real image, producing internal reference
    # answers used for logP(ref_answer | candidate, Q). These are not dataset
    # labels, human annotations, or external supervision.
    # Continuous reward, no hallucination, mutual supervision.
    # When False, falls back to multi-component scoring (spec+cycle+diversity).
    use_ref_answer_scoring: bool = False

    # Replay buffer: stores best generated images for mixing into understanding training.
    replay_buffer_size: int = 1000
    replay_min_reward: float = 0.5
    replay_max_staleness: int = 500

    # Generated-data source for understanding mixing.
    # - "buffer": in-memory replay buffer (default)
    # - "folder": filesystem-backed generated pool with random sampling
    gen_mix_source_mode: str = "buffer"  # buffer|folder
    generated_mix_dir: Optional[str] = None
    generated_mix_min_reward: float = 0.5
    generated_mix_max_files: int = 5000
    generated_mix_refresh_every: int = 10
    understanding_generated_only: bool = False
    strict_imageless_mode: bool = False

    # Generated-image mixing ratio for understanding step.
    # Linearly ramps from start → max over warmup_steps.
    gen_mix_ratio_start: float = 0.02     # initial ratio (conservative start)
    gen_mix_ratio_max: float = 0.25       # cap (real data always >= 75%)
    gen_mix_ratio_warmup_steps: int = 1000  # longer warmup for stability

    # Generator reward EMA tracking (for monitoring / logging)
    reward_ema_momentum: float = 0.95
