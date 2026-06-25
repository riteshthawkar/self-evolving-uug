# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ModelLoadConfig:
    model_path: str
    device: str = "cuda"
    vae_device: str = ""
    lora_checkpoint_path: str = ""
    max_latent_size: int = 64
    vit_max_num_patch_per_side: int = 70
    latent_patch_size: int = 2
    connector_act: str = "gelu_pytorch_tanh"
    # Inference transforms used by InterleaveInferencer.
    vae_max_image_size: int = 1024
    vae_min_image_size: int = 512
    vae_stride: int = 16
    vit_max_image_size: int = 980
    vit_min_image_size: int = 224
    vit_stride: int = 14

    # Optional LoRA runtime setup (applied on the BAGEL language model only).
    enable_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules_csv: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_role_adapters_csv: str = "proposer,solver,generator"
    lora_default_adapter: str = "proposer"

    def lora_target_modules(self) -> List[str]:
        vals = [v.strip() for v in str(self.lora_target_modules_csv or "").split(",")]
        vals = [v for v in vals if v]
        if vals:
            return vals
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def lora_role_adapters(self) -> List[str]:
        vals = [v.strip() for v in str(self.lora_role_adapters_csv or "").split(",")]
        vals = [v for v in vals if v]
        return vals if vals else ["proposer", "solver", "generator"]


@dataclass
class RolloutConfig:
    image_dir: str
    output_dir: str
    experiment_name: str = "understanding_self_evolving"
    steps: int = 500
    seed: int = 42
    log_every: int = 10

    # Generation controls
    max_new_tokens_proposer: int = 256
    max_new_tokens_solver: int = 96
    proposer_temperature: float = 0.9
    num_solver_samples: int = 7
    solver_temp_min: float = 0.5
    solver_temp_max: float = 2.0

    # Self-consistency / reward shaping
    proposer_entropy_mu: float = 0.9
    proposer_entropy_sigma: float = 0.25
    solver_unsolvable_maj_threshold: float = 0.20
    zero_entropy_eps: float = 1e-6
    zero_entropy_reward_cap: float = 0.45
    proposer_non_objective_penalty: float = 0.20
    proposer_require_objective: bool = True
    acceptance_require_non_easy: bool = True
    rejected_question_penalty: float = 0.35

    # Persist raw generations for auditability and reproducibility.
    save_raw_generations: bool = True

    # Generation phase (BLIP-style candidate scoring on generated images).
    suder_generation_enabled: bool = False
    max_new_tokens_gen_spec: int = 384
    gen_spec_temperature: float = 0.9
    gen_spec_min_qa_pairs: int = 2
    proposer_gen_entropy_weight: float = 0.7
    proposer_gen_baseline_momentum: float = 0.6
    generation_num_candidates: int = 3

    # Generation inference controls.
    generation_cfg_text_scale: float = 4.0
    generation_cfg_img_scale: float = 1.5
    generation_num_timesteps: int = 50
    generation_timestep_shift: float = 3.0
    generation_image_size: int = 1024
    save_generated_images: bool = False

    # BLIP-style generation reward weights / gates.
    reward_spec_weight: float = 0.65
    reward_cycle_weight: float = 0.20
    reward_diversity_weight: float = 0.10
    reward_contradiction_weight: float = 0.20
    min_spec_quality_for_update: float = 0.35
    min_spec_qa_pairs: int = 2
    max_expected_words: int = 8
    max_question_words: int = 24

    # Policy update (phase-2 training) knobs.
    policy_updates_enabled: bool = False
    policy_update_method: str = "grpo"  # reinforce|grpo
    policy_use_bf16: bool = True
    policy_lr: float = 1e-6
    policy_weight_decay: float = 0.01
    policy_max_grad_norm: float = 1.0
    policy_grad_accum_steps: int = 1
    policy_reward_scale: float = 1.0
    baseline_momentum: float = 0.6
    grpo_eps: float = 1e-6
    kl_coef: float = 0.01
    kl_target: float = 0.02
    kl_adapt_rate: float = 0.10
    kl_min: float = 0.001
    kl_max: float = 1e2
    solver_reward_mix_gamma: float = 0.7
    solver_skip_easy_updates: bool = True
    solver_easy_update_majority_threshold: float = 0.85
    train_understanding_proposer: bool = True
    train_solver: bool = True
    train_generation_proposer: bool = True
    train_generator: bool = True
    checkpoint_every: int = 100
    resume_from: str = ""
    save_lora_only: bool = True

    # Unified scheduler controls (BLIP3o-style alternating phases).
    understanding_steps_per_cycle: int = 3
    generation_steps_per_cycle: int = 2

    # Generation -> understanding feedback loop.
    replay_buffer_size: int = 1
    replay_min_reward: float = 1.10
    replay_max_staleness: int = 1
    gen_mix_source_mode: str = "buffer"  # buffer|folder
    generated_mix_dir: str = ""
    generated_mix_min_reward: float = 0.5
    generated_mix_max_files: int = 5000
    generated_mix_refresh_every: int = 10
    understanding_generated_only: bool = False
    gen_mix_ratio_start: float = 0.0
    gen_mix_ratio_max: float = 0.0
    gen_mix_ratio_warmup_steps: int = 1
    reward_ema_momentum: float = 0.95

    # Proposer framework parity knobs (progressively wired in trainer updates).
    proposer_num_candidates: int = 5
    proposer_spot_check_samples: int = 3
    proposer_spot_entropy_min_gate: float = 0.05
    proposer_grpo_gen_group_size: int = 3
    score_grpo_extras: bool = True
    grpo_extra_temp_multiplier: float = 1.5
    grpo_extra_sc_samples: int = 3
    understanding_skip_no_acceptable: bool = True
    understanding_require_acceptable_for_update: bool = True
    understanding_update_require_disagreement: bool = True
    proposer_reject_unsolvable: bool = True
    solver_skip_unsolvable_updates: bool = True
    solver_token_entropy_enabled: bool = True
    solver_token_entropy_tokens: int = 5
    solver_token_entropy_window_size: int = 128
    solver_token_entropy_sigmoid_alpha: float = 1.5
    solver_token_entropy_sigmoid_beta: float = 2.0
    ste_spot_easy_quantile: float = 0.30
    proposer_ste_primary_weight: float = 0.70
    proposer_sample_entropy_weight: float = 0.30

    # Proposer certification and warm-start.
    proposer_certificate_enabled: bool = True
    proposer_certificate_min_score: float = 0.55
    proposer_certificate_weight: float = 0.75
    proposer_certificate_strict_struct: bool = True
    proposer_warm_start_enabled: bool = True
    proposer_warm_start_max_steps: int = 30
    proposer_warm_start_exit_window: int = 5
    proposer_warm_start_exit_consecutive: int = 2
    proposer_warm_start_entropy_exit_threshold: float = 0.10
    proposer_warm_start_easy_reject_penalty_scale: float = 0.0
    proposer_warm_start_certificate_weight: float = 0.50

    # Hardness debt controller.
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

    # Difficulty sampler and entropy-IQR adaptive easy thresholding.
    difficulty_sampler_enabled: bool = True
    difficulty_sampler_window_size: int = 256
    difficulty_sampler_min_samples: int = 32
    difficulty_target_easy: float = 0.10
    difficulty_target_medium: float = 0.50
    difficulty_target_hard: float = 0.40
    difficulty_hard_min_entropy: float = 0.90
    difficulty_hard_max_margin: float = 0.35
    entropy_iqr_filter_enabled: bool = True
    entropy_iqr_window_size: int = 256
    entropy_iqr_min_samples: int = 32
    entropy_iqr_easy_quantile: float = 0.25
    entropy_iqr_easy_iqr_coef: float = 0.25
    entropy_iqr_min_threshold: float = 0.02
    entropy_iqr_max_threshold: float = 1.2
    entropy_iqr_filter_min_majority_frac: float = 0.80

    # All-easy recovery exploration.
    all_easy_explore_trigger: int = 2
    all_easy_explore_steps: int = 10
    all_easy_explore_num_candidates: int = 6
    all_easy_explore_temp_boost: float = 1.20
    all_easy_explore_top_p_boost: float = 0.15
    all_easy_explore_penalty_boost: float = 0.50

    # Contrastive replay shaping for proposer.
    proposer_contrastive_replay_enabled: bool = True
    proposer_contrastive_replay_size: int = 256
    proposer_contrastive_pos_bonus: float = 0.08
    proposer_contrastive_neg_penalty: float = 0.08

    # Early failfast / recover (health checks for collapse).
    proposer_early_failfast_enabled: bool = True
    proposer_early_failfast_stop: bool = False
    proposer_early_failfast_recover: bool = True
    proposer_early_failfast_recover_steps: int = 20
    proposer_early_stage1_u_step: int = 12
    proposer_early_stage2_u_step: int = 20
    proposer_early_hard_stop_min_u_step: int = 80
    proposer_early_candidate_non_easy_rate_min: float = 0.08
    proposer_early_all_easy_rate_max: float = 0.93
    proposer_early_reward_clipped_rate_max: float = 0.85
    proposer_early_selected_non_easy_rate_min: float = 0.10
    proposer_early_solver_updates_min: int = 1
    proposer_early_max_collapse_streak: int = 3

    # GRPO stabilizers.
    grpo_degenerate_noise_enabled: bool = True
    grpo_degenerate_noise_sigma: float = 0.03
    grpo_degenerate_noise_std_threshold: float = 1e-6
    grpo_pairwise_ranking_enabled: bool = True
    grpo_pairwise_ranking_weight: float = 0.15
    grpo_pairwise_margin: float = 0.10
    grpo_pairwise_easy_penalty: float = 0.12
    proposer_all_easy_rank_spread: float = 0.08

    # Generation-side joint understanding parity.
    gen_step_solver_update_enabled: bool = False

    # Optional distributed self-evolving runtime.
    dist_enabled: bool = False
    dist_backend: str = "nccl"
    dist_world_size: int = 1
    dist_rank: int = 0
    dist_local_rank: int = 0
    dist_main_process: bool = True
    dist_data_shard: bool = True

    def solver_temperatures(self) -> List[float]:
        n = max(1, int(self.num_solver_samples))
        if n == 1:
            return [float(self.solver_temp_min)]
        tmin = float(self.solver_temp_min)
        tmax = float(self.solver_temp_max)
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        step = (tmax - tmin) / float(n - 1)
        return [tmin + step * float(i) for i in range(n)]

    def normalized_update_method(self) -> str:
        method = str(self.policy_update_method or "reinforce").strip().lower()
        if method not in {"reinforce", "grpo"}:
            return "reinforce"
        return method

    def normalized_experiment_name(self) -> str:
        exp = str(self.experiment_name or "understanding_self_evolving").strip().lower()
        if exp not in {"understanding_self_evolving", "generation_self_evolving", "unified_self_evolving"}:
            return "understanding_self_evolving"
        return exp

    def normalized_gen_mix_source_mode(self) -> str:
        mode = str(self.gen_mix_source_mode or "buffer").strip().lower()
        if mode not in {"buffer", "folder"}:
            return "buffer"
        return mode

    def cycle_length(self) -> int:
        u = max(0, int(self.understanding_steps_per_cycle))
        g = max(0, int(self.generation_steps_per_cycle))
        return max(1, u + g)

    def current_gen_mix_ratio(self, step: int, start_step: int) -> float:
        start = max(0.0, min(1.0, float(self.gen_mix_ratio_start)))
        mx = max(0.0, min(1.0, float(self.gen_mix_ratio_max)))
        if mx <= 0.0:
            return 0.0
        warmup = max(1, int(self.gen_mix_ratio_warmup_steps))
        elapsed = max(0, int(step) - int(start_step))
        frac = min(1.0, float(elapsed) / float(warmup))
        return float(start + frac * (mx - start))

    def distributed_active(self) -> bool:
        return bool(self.dist_enabled) and int(self.dist_world_size) > 1
