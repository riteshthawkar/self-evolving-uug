"""
Unified entry point for self-evolving training experiments.

Replaces the legacy external self-evolving runner with native BLIP3o model loading.
Supports: understanding_self_evolving, generation_self_evolving, unified_self_evolving.

Usage:
    python -m blip3o.train.train_self_evolving --experiment understanding_self_evolving --data_dir /path/to/images
    torchrun --nproc_per_node=4 -m blip3o.train.train_self_evolving --experiment unified_self_evolving --data_dir /path

"""

import argparse
import os
from typing import Optional, Tuple


EXPERIMENT_CHOICES = (
    "understanding_self_evolving",
    "generation_self_evolving",
    "unified_self_evolving",
)

DEFAULT_TEXT_LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
DEFAULT_SOLVER_MERGER_LORA_TARGETS = "visual.merger.mlp.0,visual.merger.mlp.2"
DEFAULT_DIT_LORA_TARGETS = (
    "attn2.to_q,attn2.to_k,attn2.to_v,attn2.to_out.0,"
    "caption_projection.linear_1,caption_projection.linear_2"
)


def _parse_subfolders(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    if not value:
        return None
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    return names or None


def _parse_csv_tuple(value: Optional[str]) -> Tuple[str, ...]:
    if not value:
        return tuple()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BLIP3o self-evolving experiment runner")

    # Core
    p.add_argument("--experiment", type=str, required=True, choices=EXPERIMENT_CHOICES)
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--data_split", type=str, default="all", choices=["train", "val", "test", "all"])
    p.add_argument("--output_dir", type=str, default="./runs")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--model_name", type=str, default="BLIP3o/BLIP3o-Model-8B")
    p.add_argument("--include_subfolders", type=str, default=None)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--non_deterministic", dest="deterministic", action="store_false")

    # Device
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "eager", "flash_attention_2", "none"],
    )
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--device_map", type=str, default="single", choices=["single", "auto", "cpu"])

    # Train loop
    p.add_argument("--total_steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--proposer_update_freq", type=int, default=1)
    p.add_argument("--generator_update_freq", type=int, default=1)
    p.add_argument("--enable_solver_updates", action="store_true", default=False)
    p.add_argument("--solver_update_freq", type=int, default=0)

    # Decoding
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_new_tokens_solver", type=int, default=128)
    p.add_argument("--max_new_tokens_proposer", type=int, default=128)
    p.add_argument("--max_new_tokens_caption", type=int, default=96)
    p.add_argument("--max_new_tokens_generator", type=int, default=768)
    p.add_argument("--num_solver_samples", type=int, default=7)
    p.add_argument("--num_solver_samples_spec", type=int, default=3)
    p.add_argument("--num_generations", type=int, default=3)

    # Reward shaping
    p.add_argument("--solver_soft_gamma", type=float, default=0.7)
    p.add_argument("--solver_use_temperature_mix", action="store_true", default=True)
    p.add_argument("--disable_solver_temperature_mix", dest="solver_use_temperature_mix", action="store_false")
    p.add_argument("--solver_use_forced_choice_from_proposer", action="store_true", default=True)
    p.add_argument(
        "--disable_solver_use_forced_choice_from_proposer",
        dest="solver_use_forced_choice_from_proposer",
        action="store_false",
    )
    p.add_argument("--solver_temp_min", type=float, default=0.4)
    p.add_argument("--solver_temp_max", type=float, default=2.6)
    p.add_argument("--solver_top_p_min", type=float, default=0.35)
    p.add_argument("--solver_top_p_max", type=float, default=1.0)
    p.add_argument("--sc_entropy_min", type=float, default=0.15)
    p.add_argument("--sc_entropy_max", type=float, default=1.2)
    p.add_argument("--sc_margin_max", type=float, default=0.90)
    p.add_argument("--sc_informative_ratio_min", type=float, default=0.25)
    p.add_argument("--sc_negative_weight", type=float, default=0.25)
    p.add_argument("--easy_solver_penalty_scale", type=float, default=1.0)
    p.add_argument("--solver_low_info_easy_penalty_scale", type=float, default=2.5)
    p.add_argument("--solver_update_on_low_info_easy", action="store_true", default=False)
    p.add_argument(
        "--disable_solver_update_on_low_info_easy",
        dest="solver_update_on_low_info_easy",
        action="store_false",
    )
    p.add_argument("--skip_solver_update_when_uninformative", action="store_true", default=True)
    p.add_argument(
        "--allow_solver_update_when_uninformative",
        dest="skip_solver_update_when_uninformative",
        action="store_false",
    )
    p.add_argument("--solver_always_update_with_informative_scaling", action="store_true", default=False)
    p.add_argument(
        "--disable_solver_always_update_with_informative_scaling",
        dest="solver_always_update_with_informative_scaling",
        action="store_false",
    )
    p.add_argument("--solver_update_min_scale", type=float, default=0.20)
    p.add_argument("--len_penalty_weight", type=float, default=0.10)
    p.add_argument("--len_penalty_target_words", type=int, default=6)
    p.add_argument("--prop_entropy_mu", type=float, default=0.90)
    p.add_argument("--prop_entropy_sigma", type=float, default=0.35)
    p.add_argument("--adaptive_prop_entropy_target", action="store_true", default=True)
    p.add_argument("--fixed_prop_entropy_target", dest="adaptive_prop_entropy_target", action="store_false")
    p.add_argument("--prop_entropy_ema_momentum", type=float, default=0.95)
    p.add_argument("--prop_entropy_mu_min", type=float, default=0.05)
    p.add_argument("--prop_entropy_mu_max", type=float, default=1.5)
    p.add_argument("--zero_entropy_reward_cap", type=float, default=0.10)
    p.add_argument("--proposer_easy_reward_cap", type=float, default=0.20)
    p.add_argument("--proposer_easy_gotcha_reward_cap", type=float, default=0.50)
    p.add_argument("--proposer_non_objective_penalty", type=float, default=0.20)
    p.add_argument("--proposer_low_info_majority_penalty", type=float, default=0.50)
    p.add_argument("--proposer_slot_compiler_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_proposer_slot_compiler",
        dest="proposer_slot_compiler_enabled",
        action="store_false",
    )
    p.add_argument("--proposer_slot_compiler_strict", action="store_true", default=True)
    p.add_argument(
        "--disable_proposer_slot_compiler_strict",
        dest="proposer_slot_compiler_strict",
        action="store_false",
    )
    p.add_argument("--proposer_trivial_archetype_penalty", type=float, default=0.25)
    p.add_argument("--proposer_answer_family_repeat_penalty", type=float, default=0.25)
    p.add_argument("--proposer_answer_family_repeat_target", type=float, default=0.25)
    p.add_argument("--proposer_candidate_noncanonical_penalty", type=float, default=0.12)
    p.add_argument("--proposer_candidate_low_info_penalty", type=float, default=0.10)
    p.add_argument("--solver_noncanonical_answer_penalty", type=float, default=0.10)
    p.add_argument("--solver_low_info_answer_penalty", type=float, default=0.08)
    p.add_argument("--curriculum_arm_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_curriculum_arm",
        dest="curriculum_arm_enabled",
        action="store_false",
    )
    p.add_argument("--curriculum_arm_prompt_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_curriculum_arm_prompt",
        dest="curriculum_arm_prompt_enabled",
        action="store_false",
    )
    p.add_argument("--curriculum_arm_ema_momentum", type=float, default=0.90)
    p.add_argument("--curriculum_arm_progress_weight", type=float, default=0.20)
    p.add_argument("--curriculum_arm_underuse_weight", type=float, default=0.12)
    p.add_argument("--curriculum_arm_easy_penalty_weight", type=float, default=0.15)
    p.add_argument("--curriculum_arm_solver_gain_weight", type=float, default=0.10)
    p.add_argument("--curriculum_arm_prompt_temp", type=float, default=0.60)
    p.add_argument("--curriculum_arm_candidate_bonus", type=float, default=0.08)
    p.add_argument("--curriculum_arm_reward_scale", type=float, default=0.10)
    p.add_argument("--replay_priority_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_replay_priority",
        dest="replay_priority_enabled",
        action="store_false",
    )
    p.add_argument("--replay_priority_hardness_weight", type=float, default=0.50)
    p.add_argument("--replay_priority_update_weight", type=float, default=0.30)
    p.add_argument("--replay_priority_novelty_weight", type=float, default=0.20)
    p.add_argument("--replay_anchor_inject_k", type=int, default=2)
    p.add_argument("--replay_anchor_inject_easy_streak", type=int, default=2)
    p.add_argument("--proposer_require_objective", action="store_true", default=True)
    p.add_argument(
        "--disable_proposer_require_objective",
        dest="proposer_require_objective",
        action="store_false",
    )
    # Single-shot multi-question generation (replaces retry loop)
    p.add_argument("--proposer_num_candidates", type=int, default=3)
    p.add_argument("--proposer_spot_check_samples", type=int, default=3)
    p.add_argument("--proposer_spot_entropy_min_gate", type=float, default=0.05)
    p.add_argument("--proposer_certificate_min_score", type=float, default=0.55)
    p.add_argument(
        "--proposer_certificate_strict_struct",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--disable_proposer_certificate_strict_struct",
        dest="proposer_certificate_strict_struct",
        action="store_false",
    )
    p.add_argument("--proposer_easy_reward_floor", type=float, default=-0.35)
    p.add_argument("--proposer_all_easy_rank_spread", type=float, default=0.08)
    p.add_argument("--solver_token_entropy_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_solver_token_entropy",
        dest="solver_token_entropy_enabled",
        action="store_false",
    )
    p.add_argument("--solver_token_entropy_tokens", type=int, default=5)
    p.add_argument("--solver_token_entropy_window_size", type=int, default=128)
    p.add_argument("--solver_token_entropy_sigmoid_alpha", type=float, default=1.5)
    p.add_argument("--solver_token_entropy_sigmoid_beta", type=float, default=2.0)
    p.add_argument(
        "--solver_token_entropy_aggregation",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="Aggregate first-K answer-token entropies for STE difficulty.",
    )
    p.add_argument("--proposer_ste_primary_weight", type=float, default=0.70)
    p.add_argument("--proposer_sample_entropy_weight", type=float, default=0.30)
    p.add_argument("--proposer_ste_reward_weight", type=float, default=0.30)
    p.add_argument("--solver_pps_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_solver_pps",
        dest="solver_pps_enabled",
        action="store_false",
    )
    p.add_argument("--solver_skip_update_on_easy", action="store_true", default=True)
    p.add_argument(
        "--allow_solver_update_on_easy",
        dest="solver_skip_update_on_easy",
        action="store_false",
    )
    p.add_argument("--easy_update_majority_frac_threshold", type=float, default=0.95)
    p.add_argument("--acceptance_require_non_easy", action="store_true", default=True)
    p.add_argument(
        "--disable_acceptance_require_non_easy",
        dest="acceptance_require_non_easy",
        action="store_false",
    )
    p.add_argument("--rejected_question_penalty", type=float, default=0.35)
    p.add_argument("--entropy_iqr_filter_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_entropy_iqr_filter",
        dest="entropy_iqr_filter_enabled",
        action="store_false",
    )
    p.add_argument("--entropy_iqr_window_size", type=int, default=256)
    p.add_argument("--entropy_iqr_min_samples", type=int, default=32)
    p.add_argument("--entropy_iqr_easy_quantile", type=float, default=0.25)
    p.add_argument("--entropy_iqr_easy_iqr_coef", type=float, default=0.25)
    p.add_argument("--entropy_iqr_min_threshold", type=float, default=0.02)
    p.add_argument("--entropy_iqr_max_threshold", type=float, default=1.2)
    p.add_argument("--entropy_iqr_filter_min_majority_frac", type=float, default=0.80)
    p.add_argument("--difficulty_sampler_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_difficulty_sampler",
        dest="difficulty_sampler_enabled",
        action="store_false",
    )
    p.add_argument("--difficulty_sampler_window_size", type=int, default=256)
    p.add_argument("--difficulty_sampler_min_samples", type=int, default=32)
    p.add_argument("--difficulty_target_easy", type=float, default=0.10)
    p.add_argument("--difficulty_target_medium", type=float, default=0.50)
    p.add_argument("--difficulty_target_hard", type=float, default=0.40)
    p.add_argument("--difficulty_hard_min_entropy", type=float, default=0.90)
    p.add_argument("--difficulty_hard_max_margin", type=float, default=0.35)
    p.add_argument("--easy_constraint_target_rate", type=float, default=0.18)
    p.add_argument("--easy_constraint_lr", type=float, default=0.20)
    p.add_argument("--easy_constraint_penalty_scale", type=float, default=0.60)
    p.add_argument("--easy_constraint_selection_scale", type=float, default=0.50)
    p.add_argument("--all_easy_explore_trigger", type=int, default=2)
    p.add_argument("--all_easy_explore_steps", type=int, default=10)
    p.add_argument("--all_easy_explore_num_candidates", type=int, default=6)
    p.add_argument("--all_easy_explore_temp_boost", type=float, default=1.20)
    p.add_argument("--all_easy_explore_top_p_boost", type=float, default=0.15)
    p.add_argument("--all_easy_explore_penalty_boost", type=float, default=0.50)
    p.add_argument("--proposer_early_step1", type=int, default=12)
    p.add_argument("--proposer_early_step2", type=int, default=24)
    p.add_argument("--proposer_early_candidate_non_easy_min", type=float, default=0.08)
    p.add_argument("--proposer_early_all_easy_rate_max", type=float, default=0.93)
    p.add_argument("--proposer_early_reward_clipped_rate_max", type=float, default=0.85)
    p.add_argument("--proposer_early_selected_non_easy_min", type=float, default=0.10)
    p.add_argument("--proposer_early_solver_updates_min", type=int, default=1)
    p.add_argument("--proposer_early_collapse_streak_max", type=int, default=3)
    p.add_argument("--proposer_early_failfast_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_proposer_early_failfast",
        dest="proposer_early_failfast_enabled",
        action="store_false",
    )
    p.add_argument("--proposer_early_failfast_stop", action="store_true", default=False)
    p.add_argument(
        "--disable_proposer_early_failfast_stop",
        dest="proposer_early_failfast_stop",
        action="store_false",
    )
    p.add_argument("--proposer_early_failfast_recover", action="store_true", default=True)
    p.add_argument(
        "--disable_proposer_early_failfast_recover",
        dest="proposer_early_failfast_recover",
        action="store_false",
    )
    p.add_argument("--proposer_early_failfast_recover_steps", type=int, default=20)
    p.add_argument("--proposer_early_hard_stop_min_u_step", type=int, default=80)
    p.add_argument("--reward_spec_weight", type=float, default=0.65)
    p.add_argument("--reward_cycle_weight", type=float, default=0.20)
    p.add_argument("--reward_diversity_weight", type=float, default=0.10)
    p.add_argument("--reward_contradiction_weight", type=float, default=0.20)

    # KL
    p.add_argument("--kl_coef", type=float, default=0.01)
    p.add_argument("--kl_target", type=float, default=0.02)
    p.add_argument("--kl_adapt_rate", type=float, default=0.10)
    p.add_argument("--kl_min", type=float, default=0.001)
    p.add_argument("--kl_max", type=float, default=1e2)
    p.add_argument("--baseline_momentum", type=float, default=0.6)

    # LoRA
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--no_lora", dest="use_lora", action="store_false")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_targets",
        type=str,
        default=DEFAULT_TEXT_LORA_TARGETS,
    )
    p.add_argument("--solver_merger_lora", dest="solver_merger_lora_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_solver_merger_lora",
        dest="solver_merger_lora_enabled",
        action="store_false",
    )
    p.add_argument("--solver_merger_lora_r", type=int, default=4)
    p.add_argument("--solver_merger_lora_alpha", type=int, default=8)
    p.add_argument("--solver_merger_lora_lr", type=float, default=2e-7)
    p.add_argument(
        "--solver_merger_lora_targets",
        type=str,
        default=DEFAULT_SOLVER_MERGER_LORA_TARGETS,
        help="Comma-separated visual-language merger/projection modules trained only by the solver adapter.",
    )
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    p.add_argument("--bnb_4bit_use_double_quant", action="store_true", default=True)
    p.add_argument(
        "--disable_bnb_4bit_use_double_quant",
        dest="bnb_4bit_use_double_quant",
        action="store_false",
    )
    p.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
    )

    # Logging + checkpoints
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--max_checkpoints", type=int, default=5)
    p.add_argument("--clear_cache_every", type=int, default=25)
    p.add_argument("--save_generated_images_every", type=int, default=0)
    p.add_argument(
        "--code_run_registry_dir",
        type=str,
        default=os.environ.get("SELF_EVOLVING_CODE_RUN_REGISTRY", None),
        help=(
            "Directory for lightweight run metadata mirrored beside training code. "
            "Defaults to blip3o/train/self_evolving/training_runs."
        ),
    )
    p.add_argument("--code_run_registry", dest="code_run_registry_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_code_run_registry",
        dest="code_run_registry_enabled",
        action="store_false",
    )
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--start_step", type=int, default=0)

    # Generation backend
    p.add_argument("--generation_num_inference_steps", type=int, default=50)
    p.add_argument("--generation_guidance_scale", type=float, default=2.0)
    p.add_argument("--generation_height", type=int, default=896)
    p.add_argument("--generation_width", type=int, default=896)
    p.add_argument("--require_decoder_for_blip3o", action="store_true", default=True)
    p.add_argument("--allow_missing_decoder_for_blip3o", dest="require_decoder_for_blip3o", action="store_false")
    p.add_argument("--allow_latent_visualization_fallback", action="store_true", default=False)
    p.add_argument("--strict_require_generation_tokens", dest="strict_require_generation_tokens", action="store_true")
    p.add_argument("--allow_missing_generation_tokens", dest="strict_require_generation_tokens", action="store_false")
    p.set_defaults(strict_require_generation_tokens=True)
    p.add_argument("--generator_missing_trace_strategy", type=str, default="skip", choices=["proxy", "skip", "error"])
    # Solver always uses trained LoRA for mutual supervision (understanding ↔ generation).
    # Legacy flag kept for backward compat but defaults to False (trained solver).
    p.add_argument("--verification_use_reference_solver", action="store_true", default=False)
    p.add_argument(
        "--use_self_clip_reward_scoring",
        action="store_true",
        default=False,
        help="Use CLIP-style reward from model's own frozen embeddings for generated candidate ranking.",
    )
    p.add_argument(
        "--disable_self_clip_reward_scoring",
        dest="use_self_clip_reward_scoring",
        action="store_false",
    )
    p.add_argument("--generator_update_rule", type=str, default="reinforce", choices=["reinforce", "dpo", "grpo"])
    p.add_argument("--dpo_beta", type=float, default=0.1)
    p.add_argument("--dpo_label_smoothing", type=float, default=0.0)
    p.add_argument("--dpo_min_reward_gap", type=float, default=0.0)
    p.add_argument("--dpo_min_spec_gap", type=float, default=0.0)
    p.add_argument("--dpo_min_confidence_gap", type=float, default=0.0)
    p.add_argument("--dpo_max_contradiction", type=float, default=1.0)
    p.add_argument("--dpo_pair_selection", type=str, default="best_worst", choices=["best_worst", "best_hard_negative"])
    p.add_argument("--generator_proxy_max_ratio", type=float, default=1.0)
    p.add_argument("--grpo_clip_ratio", type=float, default=0.2)
    p.add_argument("--grpo_min_group_std", type=float, default=1e-6)
    p.add_argument("--unicorn_generation_enabled", action="store_true", default=True)
    p.add_argument("--disable_unicorn_generation", dest="unicorn_generation_enabled", action="store_false")
    p.add_argument("--unicorn_target_difficulty", type=str, default="medium", choices=["easy", "medium", "hard"])
    p.add_argument("--unicorn_spec_rejection_enabled", action="store_true", default=True)
    p.add_argument("--disable_unicorn_spec_rejection", dest="unicorn_spec_rejection_enabled", action="store_false")
    p.add_argument("--unicorn_spec_max_retries", type=int, default=2)
    p.add_argument("--unicorn_spec_min_quality", type=float, default=0.55)
    p.add_argument("--unicorn_spec_min_alignment", type=float, default=0.55)
    p.add_argument("--unicorn_reconstruction_sft_enabled", action="store_true", default=True)
    p.add_argument("--disable_unicorn_reconstruction_sft", dest="unicorn_reconstruction_sft_enabled", action="store_false")
    p.add_argument("--unicorn_reconstruction_buffer_size", type=int, default=512)
    p.add_argument("--unicorn_reconstruction_step_freq", type=int, default=1)
    p.add_argument("--unicorn_reconstruction_updates_per_step", type=int, default=2)
    p.add_argument("--unicorn_reconstruction_min_quality", type=float, default=0.55)
    p.add_argument("--unicorn_reconstruction_enable_proposer", action="store_true", default=True)
    p.add_argument(
        "--disable_unicorn_reconstruction_proposer",
        dest="unicorn_reconstruction_enable_proposer",
        action="store_false",
    )
    p.add_argument("--unicorn_reconstruction_enable_generator", action="store_true", default=True)
    p.add_argument(
        "--disable_unicorn_reconstruction_generator",
        dest="unicorn_reconstruction_enable_generator",
        action="store_false",
    )
    p.add_argument("--dit_update_enabled", action="store_true", default=False)
    p.add_argument("--disable_dit_update", dest="dit_update_enabled", action="store_false")
    p.add_argument(
        "--require_dit_update",
        action="store_true",
        default=False,
        help="Fail fast if --dit_update_enabled cannot create an active DiT updater.",
    )
    p.add_argument("--dit_update_freq", type=int, default=1)
    p.add_argument("--dit_lr", type=float, default=5e-7)
    p.add_argument("--dit_weight_decay", type=float, default=0.01)
    p.add_argument("--dit_grad_clip", type=float, default=1.0)
    p.add_argument("--dit_grad_accum_steps", type=int, default=1)
    p.add_argument("--dit_conditioning_dropout", type=float, default=0.10)
    p.add_argument("--dit_loss_weight", type=float, default=1.0)
    p.add_argument("--dit_prompt_suffix_token_id", type=int, default=151665)
    p.add_argument("--dit_lora", dest="dit_lora_enabled", action="store_true", default=True,
                   help="Train LoRA adapters inside the BLIP3o DiT instead of full DiT weights.")
    p.add_argument("--disable_dit_lora", dest="dit_lora_enabled", action="store_false",
                   help="Ablation: unfreeze full DiT weights for the DiT updater.")
    p.add_argument("--dit_lora_r", type=int, default=16)
    p.add_argument("--dit_lora_alpha", type=int, default=32)
    p.add_argument("--dit_lora_dropout", type=float, default=0.0)
    p.add_argument("--dit_lora_targets", type=str, default=DEFAULT_DIT_LORA_TARGETS)
    # Joint LLM+DiT training (Change 2+3)
    p.add_argument("--dit_joint_conditioning_train", action="store_true", default=False,
                   help="Remove no_grad from LLM conditioning; train generator LoRA jointly with DiT.")
    p.add_argument("--disable_dit_joint_conditioning_train",
                   dest="dit_joint_conditioning_train", action="store_false")
    p.add_argument("--dit_joint_conditioning_lr", type=float, default=5e-7,
                   help="LR for generator LoRA when trained jointly with DiT.")
    p.add_argument("--dit_reward_loss_weight", type=float, default=0.0,
                   help="Scale denoising loss by image-quality reward (RWR). 0=pure SFT.")
    # Proposer dual reward in generation phase (Change 1 / SUDER)
    p.add_argument("--proposer_gen_reward_enabled", action="store_true", default=False,
                   help="Update proposer LoRA during generation steps using joint entropy+quality reward.")
    p.add_argument("--disable_proposer_gen_reward",
                   dest="proposer_gen_reward_enabled", action="store_false")
    p.add_argument("--proposer_gen_entropy_weight", type=float, default=0.7,
                   help="α in: reward = α*gaussian_reward(entropy) + (1-α)*image_quality. "
                        "1.0 = pure entropy (same objective as understanding), 0.0 = pure quality.")
    p.add_argument("--proposer_gen_baseline_momentum", type=float, default=0.6,
                   help="EMA momentum for the generation-phase proposer baseline (separate from understanding).")
    p.add_argument("--gen_step_solver_update_enabled", action="store_true", default=False,
                   help="Also train the solver on generated images during every generation step "
                        "(joint step). Solver rollouts are already computed for scoring — this "
                        "reuses them without extra inference cost.")
    p.add_argument("--reset_proposer_baseline", action="store_true", default=False,
                   help="Reset proposer_baseline and proposer_gen_baseline to 0.0 on resume. "
                        "Also clears entropy/difficulty history windows. Use once after bug fixes "
                        "that caused the baseline to lock. Remove flag after first checkpoint post-resume.")
    # E5: Imageless proposer mode
    p.add_argument("--imageless_proposer_mode", action="store_true", default=False,
                   help="E5: Proposer generates specs from text topics/themes, not from images. "
                        "Enables a fully synthetic self-evolving loop with ZERO external images.")
    p.add_argument("--proposer_update_rule", type=str, default="grpo",
                   choices=["grpo", "reinforce"],
                   help="Proposer optimization algorithm. 'grpo' (default): group-normalized "
                        "advantages, lower variance, no EMA baseline required. "
                        "'reinforce': single-sample with EMA baseline (original).")
    p.add_argument("--proposer_grpo_gen_group_size", type=int, default=3,
                   help="Number of specs to sample for the GRPO group in generation-phase "
                        "proposer updates. Understanding phase always uses proposer_num_candidates.")
    p.add_argument(
        "--grpo_extra_sc_samples",
        type=int,
        default=3,
        help="Number of solver spot-check samples for extra proposer GRPO candidates.",
    )
    p.add_argument(
        "--proposer_grpo_unverified_extra_margin",
        type=float,
        default=0.02,
        help="Generation-phase proposer GRPO: margin subtracted from chosen reward when "
             "assigning proxy rewards to unverified extra specs.",
    )

    # Unified scheduler
    p.add_argument("--understanding_steps_per_cycle", type=int, default=3)
    p.add_argument("--generation_steps_per_cycle", type=int, default=2)
    p.add_argument(
        "--cycle_starts_with_generation",
        action="store_true",
        default=False,
        help="When set, each U/G cycle starts with generation steps before understanding steps.",
    )
    p.add_argument(
        "--bootstrap_generated_pool_steps",
        type=int,
        default=0,
        help="Number of initial generation-only steps before the regular U/G cycle begins.",
    )
    p.add_argument("--synthetic_solver_update_freq", type=int, default=0)
    p.add_argument("--synthetic_solver_hard_only", action="store_true", default=False)
    p.add_argument("--solver_hardness_min_entropy", type=float, default=0.2)
    p.add_argument("--min_spec_quality_for_update", type=float, default=0.35)
    p.add_argument("--min_spec_qa_pairs", type=int, default=2)
    p.add_argument("--max_expected_words", type=int, default=8)
    p.add_argument("--max_question_words", type=int, default=24)

    # Self-evolving feedback loop
    p.add_argument("--use_ref_answer_scoring", action="store_true", default=True,
                    help="Use Solver-derived reference-answer log-prob scoring (default, MODE B)")
    p.add_argument("--no_ref_answer_scoring", dest="use_ref_answer_scoring", action="store_false",
                    help="Fall back to multi-component scoring (MODE A)")
    p.add_argument("--replay_buffer_size", type=int, default=1000)
    p.add_argument("--replay_min_reward", type=float, default=0.5)
    p.add_argument("--replay_max_staleness", type=int, default=500)
    p.add_argument(
        "--gen_mix_source_mode",
        type=str,
        default="buffer",
        choices=["buffer", "folder"],
        help="Source for generation->understanding mix: in-memory replay buffer or folder-backed pool.",
    )
    p.add_argument(
        "--generated_mix_dir",
        type=str,
        default=None,
        help="Directory used when --gen_mix_source_mode=folder (defaults to run_dir/generated_mix_pool).",
    )
    p.add_argument("--generated_mix_min_reward", type=float, default=0.5)
    p.add_argument("--generated_mix_max_files", type=int, default=5000)
    p.add_argument("--generated_mix_refresh_every", type=int, default=10)
    p.add_argument(
        "--understanding_generated_only",
        action="store_true",
        default=False,
        help="Use only generated images for understanding phase (skips U step when generated pool is empty).",
    )
    p.add_argument(
        "--disable_understanding_generated_only",
        dest="understanding_generated_only",
        action="store_false",
    )
    p.add_argument(
        "--strict_imageless_mode",
        action="store_true",
        default=False,
        help=(
            "Enforce imageless E5 constraints: imageless proposer + generated-only understanding + "
            "Solver-derived reference-answer scoring disabled."
        ),
    )
    p.add_argument("--gen_mix_ratio_start", type=float, default=0.02)
    p.add_argument("--gen_mix_ratio_max", type=float, default=0.25)
    p.add_argument("--gen_mix_ratio_warmup_steps", type=int, default=1000)
    p.add_argument("--reward_ema_momentum", type=float, default=0.95)

    # W&B
    p.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "disabled"), choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "self-evolving-uug"))
    p.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", None))
    p.add_argument("--wandb_run_name", type=str, default=os.environ.get("WANDB_RUN_NAME", None))
    p.add_argument("--wandb_log_images_every", type=int, default=0)

    # Experiments
    p.add_argument("--use_diverse_prompts", action="store_true", default=False)
    p.add_argument("--enable_frozen_judge", action="store_true", default=False)
    p.add_argument("--judge_ema_decay", type=float, default=0.995)
    p.add_argument("--judge_gpu_id", type=int, default=None)

    return p


def _build_understanding_config(args):
    from blip3o.train.self_evolving.config import UnderstandingSelfEvolvingConfig

    lora_targets = _parse_csv_tuple(args.lora_targets)
    solver_merger_lora_targets = _parse_csv_tuple(args.solver_merger_lora_targets)
    return UnderstandingSelfEvolvingConfig(
        run_name=args.run_name,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        data_split=args.data_split,
        include_subfolders=_parse_subfolders(args.include_subfolders),
        max_images=args.max_images,
        model_name=args.model_name,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        cuda_device=args.cuda_device,
        device_map=args.device_map,
        total_steps=args.total_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        proposer_update_freq=args.proposer_update_freq,
        temp=args.temp,
        top_p=args.top_p,
        max_new_tokens_solver=args.max_new_tokens_solver,
        max_new_tokens_proposer=args.max_new_tokens_proposer,
        num_solver_samples=args.num_solver_samples,
        solver_soft_gamma=args.solver_soft_gamma,
        solver_use_temperature_mix=args.solver_use_temperature_mix,
        solver_use_forced_choice_from_proposer=args.solver_use_forced_choice_from_proposer,
        solver_temp_min=args.solver_temp_min,
        solver_temp_max=args.solver_temp_max,
        solver_top_p_min=args.solver_top_p_min,
        solver_top_p_max=args.solver_top_p_max,
        sc_entropy_min=args.sc_entropy_min,
        sc_entropy_max=args.sc_entropy_max,
        sc_margin_max=args.sc_margin_max,
        sc_informative_ratio_min=args.sc_informative_ratio_min,
        sc_negative_weight=args.sc_negative_weight,
        easy_solver_penalty_scale=args.easy_solver_penalty_scale,
        solver_update_on_low_info_easy=args.solver_update_on_low_info_easy,
        solver_low_info_easy_penalty_scale=args.solver_low_info_easy_penalty_scale,
        skip_solver_update_when_uninformative=args.skip_solver_update_when_uninformative,
        solver_always_update_with_informative_scaling=args.solver_always_update_with_informative_scaling,
        solver_update_min_scale=args.solver_update_min_scale,
        len_penalty_weight=args.len_penalty_weight,
        len_penalty_target_words=args.len_penalty_target_words,
        prop_entropy_mu=args.prop_entropy_mu,
        prop_entropy_sigma=args.prop_entropy_sigma,
        adaptive_prop_entropy_target=args.adaptive_prop_entropy_target,
        prop_entropy_ema_momentum=args.prop_entropy_ema_momentum,
        prop_entropy_mu_min=args.prop_entropy_mu_min,
        prop_entropy_mu_max=args.prop_entropy_mu_max,
        zero_entropy_reward_cap=args.zero_entropy_reward_cap,
        proposer_easy_reward_cap=args.proposer_easy_reward_cap,
        proposer_easy_gotcha_reward_cap=args.proposer_easy_gotcha_reward_cap,
        proposer_non_objective_penalty=args.proposer_non_objective_penalty,
        proposer_low_info_majority_penalty=args.proposer_low_info_majority_penalty,
        proposer_slot_compiler_enabled=args.proposer_slot_compiler_enabled,
        proposer_slot_compiler_strict=args.proposer_slot_compiler_strict,
        proposer_trivial_archetype_penalty=args.proposer_trivial_archetype_penalty,
        proposer_answer_family_repeat_penalty=args.proposer_answer_family_repeat_penalty,
        proposer_answer_family_repeat_target=args.proposer_answer_family_repeat_target,
        proposer_candidate_noncanonical_penalty=args.proposer_candidate_noncanonical_penalty,
        proposer_candidate_low_info_penalty=args.proposer_candidate_low_info_penalty,
        solver_noncanonical_answer_penalty=args.solver_noncanonical_answer_penalty,
        solver_low_info_answer_penalty=args.solver_low_info_answer_penalty,
        curriculum_arm_enabled=args.curriculum_arm_enabled,
        curriculum_arm_prompt_enabled=args.curriculum_arm_prompt_enabled,
        curriculum_arm_ema_momentum=args.curriculum_arm_ema_momentum,
        curriculum_arm_progress_weight=args.curriculum_arm_progress_weight,
        curriculum_arm_underuse_weight=args.curriculum_arm_underuse_weight,
        curriculum_arm_easy_penalty_weight=args.curriculum_arm_easy_penalty_weight,
        curriculum_arm_solver_gain_weight=args.curriculum_arm_solver_gain_weight,
        curriculum_arm_prompt_temp=args.curriculum_arm_prompt_temp,
        curriculum_arm_candidate_bonus=args.curriculum_arm_candidate_bonus,
        curriculum_arm_reward_scale=args.curriculum_arm_reward_scale,
        replay_priority_enabled=args.replay_priority_enabled,
        replay_priority_hardness_weight=args.replay_priority_hardness_weight,
        replay_priority_update_weight=args.replay_priority_update_weight,
        replay_priority_novelty_weight=args.replay_priority_novelty_weight,
        replay_anchor_inject_k=args.replay_anchor_inject_k,
        replay_anchor_inject_easy_streak=args.replay_anchor_inject_easy_streak,
        proposer_require_objective=args.proposer_require_objective,
        proposer_num_candidates=args.proposer_num_candidates,
        proposer_spot_check_samples=args.proposer_spot_check_samples,
        proposer_spot_entropy_min_gate=args.proposer_spot_entropy_min_gate,
        proposer_certificate_min_score=args.proposer_certificate_min_score,
        proposer_certificate_strict_struct=args.proposer_certificate_strict_struct,
        proposer_easy_reward_floor=args.proposer_easy_reward_floor,
        proposer_all_easy_rank_spread=args.proposer_all_easy_rank_spread,
        solver_token_entropy_enabled=args.solver_token_entropy_enabled,
        solver_token_entropy_tokens=args.solver_token_entropy_tokens,
        solver_token_entropy_window_size=args.solver_token_entropy_window_size,
        solver_token_entropy_sigmoid_alpha=args.solver_token_entropy_sigmoid_alpha,
        solver_token_entropy_sigmoid_beta=args.solver_token_entropy_sigmoid_beta,
        solver_token_entropy_aggregation=args.solver_token_entropy_aggregation,
        proposer_ste_primary_weight=args.proposer_ste_primary_weight,
        proposer_sample_entropy_weight=args.proposer_sample_entropy_weight,
        proposer_ste_reward_weight=args.proposer_ste_reward_weight,
        solver_pps_enabled=args.solver_pps_enabled,
        grpo_extra_sc_samples=args.grpo_extra_sc_samples,
        solver_skip_update_on_easy=args.solver_skip_update_on_easy,
        easy_update_majority_frac_threshold=args.easy_update_majority_frac_threshold,
        acceptance_require_non_easy=args.acceptance_require_non_easy,
        rejected_question_penalty=args.rejected_question_penalty,
        entropy_iqr_filter_enabled=args.entropy_iqr_filter_enabled,
        entropy_iqr_window_size=args.entropy_iqr_window_size,
        entropy_iqr_min_samples=args.entropy_iqr_min_samples,
        entropy_iqr_easy_quantile=args.entropy_iqr_easy_quantile,
        entropy_iqr_easy_iqr_coef=args.entropy_iqr_easy_iqr_coef,
        entropy_iqr_min_threshold=args.entropy_iqr_min_threshold,
        entropy_iqr_max_threshold=args.entropy_iqr_max_threshold,
        entropy_iqr_filter_min_majority_frac=args.entropy_iqr_filter_min_majority_frac,
        difficulty_sampler_enabled=args.difficulty_sampler_enabled,
        difficulty_sampler_window_size=args.difficulty_sampler_window_size,
        difficulty_sampler_min_samples=args.difficulty_sampler_min_samples,
        difficulty_target_easy=args.difficulty_target_easy,
        difficulty_target_medium=args.difficulty_target_medium,
        difficulty_target_hard=args.difficulty_target_hard,
        difficulty_hard_min_entropy=args.difficulty_hard_min_entropy,
        difficulty_hard_max_margin=args.difficulty_hard_max_margin,
        easy_constraint_target_rate=args.easy_constraint_target_rate,
        easy_constraint_lr=args.easy_constraint_lr,
        easy_constraint_penalty_scale=args.easy_constraint_penalty_scale,
        easy_constraint_selection_scale=args.easy_constraint_selection_scale,
        all_easy_explore_trigger=args.all_easy_explore_trigger,
        all_easy_explore_steps=args.all_easy_explore_steps,
        all_easy_explore_num_candidates=args.all_easy_explore_num_candidates,
        all_easy_explore_temp_boost=args.all_easy_explore_temp_boost,
        all_easy_explore_top_p_boost=args.all_easy_explore_top_p_boost,
        all_easy_explore_penalty_boost=args.all_easy_explore_penalty_boost,
        proposer_early_step1=args.proposer_early_step1,
        proposer_early_step2=args.proposer_early_step2,
        proposer_early_candidate_non_easy_min=args.proposer_early_candidate_non_easy_min,
        proposer_early_all_easy_rate_max=args.proposer_early_all_easy_rate_max,
        proposer_early_reward_clipped_rate_max=args.proposer_early_reward_clipped_rate_max,
        proposer_early_selected_non_easy_min=args.proposer_early_selected_non_easy_min,
        proposer_early_solver_updates_min=args.proposer_early_solver_updates_min,
        proposer_early_collapse_streak_max=args.proposer_early_collapse_streak_max,
        proposer_early_failfast_enabled=args.proposer_early_failfast_enabled,
        proposer_early_failfast_stop=args.proposer_early_failfast_stop,
        proposer_early_failfast_recover=args.proposer_early_failfast_recover,
        proposer_early_failfast_recover_steps=args.proposer_early_failfast_recover_steps,
        proposer_early_hard_stop_min_u_step=args.proposer_early_hard_stop_min_u_step,
        kl_coef=args.kl_coef,
        kl_target=args.kl_target,
        kl_adapt_rate=args.kl_adapt_rate,
        kl_min=args.kl_min,
        kl_max=args.kl_max,
        baseline_momentum=args.baseline_momentum,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
        solver_merger_lora_enabled=args.solver_merger_lora_enabled,
        solver_merger_lora_r=args.solver_merger_lora_r,
        solver_merger_lora_alpha=args.solver_merger_lora_alpha,
        solver_merger_lora_lr=args.solver_merger_lora_lr,
        solver_merger_lora_target_modules=solver_merger_lora_targets,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        seed=args.seed,
        deterministic=args.deterministic,
        log_every=args.log_every,
        save_every=args.save_every,
        max_checkpoints=args.max_checkpoints,
        clear_cache_every=args.clear_cache_every,
        code_run_registry_enabled=args.code_run_registry_enabled,
        code_run_registry_dir=args.code_run_registry_dir,
        resume_from=args.resume_from,
        start_step=args.start_step,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_log_images_every=args.wandb_log_images_every,
    )


def _build_generation_config(args):
    from blip3o.train.self_evolving.config import GenerationSelfEvolvingConfig

    lora_targets = _parse_csv_tuple(args.lora_targets)
    solver_merger_lora_targets = _parse_csv_tuple(args.solver_merger_lora_targets)
    dit_lora_targets = _parse_csv_tuple(args.dit_lora_targets)
    return GenerationSelfEvolvingConfig(
        run_name=args.run_name,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        data_split=args.data_split,
        include_subfolders=_parse_subfolders(args.include_subfolders),
        max_images=args.max_images,
        model_name=args.model_name,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        cuda_device=args.cuda_device,
        device_map=args.device_map,
        total_steps=args.total_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        proposer_update_freq=args.proposer_update_freq,
        generator_update_freq=args.generator_update_freq,
        enable_solver_updates=args.enable_solver_updates,
        solver_update_freq=args.solver_update_freq,
        temp=args.temp,
        top_p=args.top_p,
        max_new_tokens_solver=args.max_new_tokens_solver,
        max_new_tokens_proposer=args.max_new_tokens_proposer,
        max_new_tokens_caption=args.max_new_tokens_caption,
        max_new_tokens_generator=args.max_new_tokens_generator,
        num_solver_samples=args.num_solver_samples,
        num_solver_samples_spec=args.num_solver_samples_spec,
        num_generations=args.num_generations,
        generation_num_inference_steps=args.generation_num_inference_steps,
        generation_guidance_scale=args.generation_guidance_scale,
        generation_height=args.generation_height,
        generation_width=args.generation_width,
        require_decoder_for_blip3o=args.require_decoder_for_blip3o,
        allow_latent_visualization_fallback=args.allow_latent_visualization_fallback,
        strict_require_generation_tokens=args.strict_require_generation_tokens,
        generator_missing_trace_strategy=args.generator_missing_trace_strategy,
        verification_use_reference_solver=args.verification_use_reference_solver,
        use_self_clip_reward_scoring=args.use_self_clip_reward_scoring,
        generator_update_rule=args.generator_update_rule,
        dpo_beta=args.dpo_beta,
        dpo_label_smoothing=args.dpo_label_smoothing,
        dpo_min_reward_gap=args.dpo_min_reward_gap,
        dpo_min_spec_gap=args.dpo_min_spec_gap,
        dpo_min_confidence_gap=args.dpo_min_confidence_gap,
        dpo_max_contradiction=args.dpo_max_contradiction,
        dpo_pair_selection=args.dpo_pair_selection,
        generator_proxy_max_ratio=args.generator_proxy_max_ratio,
        grpo_clip_ratio=args.grpo_clip_ratio,
        grpo_min_group_std=args.grpo_min_group_std,
        unicorn_generation_enabled=args.unicorn_generation_enabled,
        unicorn_target_difficulty=args.unicorn_target_difficulty,
        unicorn_spec_rejection_enabled=args.unicorn_spec_rejection_enabled,
        unicorn_spec_max_retries=args.unicorn_spec_max_retries,
        unicorn_spec_min_quality=args.unicorn_spec_min_quality,
        unicorn_spec_min_alignment=args.unicorn_spec_min_alignment,
        unicorn_reconstruction_sft_enabled=args.unicorn_reconstruction_sft_enabled,
        unicorn_reconstruction_buffer_size=args.unicorn_reconstruction_buffer_size,
        unicorn_reconstruction_step_freq=args.unicorn_reconstruction_step_freq,
        unicorn_reconstruction_updates_per_step=args.unicorn_reconstruction_updates_per_step,
        unicorn_reconstruction_min_quality=args.unicorn_reconstruction_min_quality,
        unicorn_reconstruction_enable_proposer=args.unicorn_reconstruction_enable_proposer,
        unicorn_reconstruction_enable_generator=args.unicorn_reconstruction_enable_generator,
        dit_update_enabled=args.dit_update_enabled,
        require_dit_update=args.require_dit_update,
        dit_update_freq=args.dit_update_freq,
        dit_lr=args.dit_lr,
        dit_weight_decay=args.dit_weight_decay,
        dit_grad_clip=args.dit_grad_clip,
        dit_grad_accum_steps=args.dit_grad_accum_steps,
        dit_conditioning_dropout=args.dit_conditioning_dropout,
        dit_loss_weight=args.dit_loss_weight,
        dit_prompt_suffix_token_id=args.dit_prompt_suffix_token_id,
        dit_lora_enabled=args.dit_lora_enabled,
        dit_lora_r=args.dit_lora_r,
        dit_lora_alpha=args.dit_lora_alpha,
        dit_lora_dropout=args.dit_lora_dropout,
        dit_lora_target_modules=dit_lora_targets,
        dit_joint_conditioning_train=args.dit_joint_conditioning_train,
        dit_joint_conditioning_lr=args.dit_joint_conditioning_lr,
        dit_reward_loss_weight=args.dit_reward_loss_weight,
        proposer_gen_reward_enabled=args.proposer_gen_reward_enabled,
        proposer_gen_entropy_weight=args.proposer_gen_entropy_weight,
        proposer_gen_baseline_momentum=args.proposer_gen_baseline_momentum,
        gen_step_solver_update_enabled=args.gen_step_solver_update_enabled,
        reset_proposer_baseline=args.reset_proposer_baseline,
        proposer_update_rule=args.proposer_update_rule,
        proposer_grpo_gen_group_size=args.proposer_grpo_gen_group_size,
        grpo_extra_sc_samples=args.grpo_extra_sc_samples,
        proposer_grpo_unverified_extra_margin=args.proposer_grpo_unverified_extra_margin,
        solver_soft_gamma=args.solver_soft_gamma,
        solver_use_temperature_mix=args.solver_use_temperature_mix,
        solver_use_forced_choice_from_proposer=args.solver_use_forced_choice_from_proposer,
        solver_temp_min=args.solver_temp_min,
        solver_temp_max=args.solver_temp_max,
        solver_top_p_min=args.solver_top_p_min,
        solver_top_p_max=args.solver_top_p_max,
        sc_entropy_min=args.sc_entropy_min,
        sc_entropy_max=args.sc_entropy_max,
        sc_margin_max=args.sc_margin_max,
        sc_informative_ratio_min=args.sc_informative_ratio_min,
        sc_negative_weight=args.sc_negative_weight,
        easy_solver_penalty_scale=args.easy_solver_penalty_scale,
        solver_update_on_low_info_easy=args.solver_update_on_low_info_easy,
        solver_low_info_easy_penalty_scale=args.solver_low_info_easy_penalty_scale,
        skip_solver_update_when_uninformative=args.skip_solver_update_when_uninformative,
        solver_always_update_with_informative_scaling=args.solver_always_update_with_informative_scaling,
        solver_update_min_scale=args.solver_update_min_scale,
        len_penalty_weight=args.len_penalty_weight,
        len_penalty_target_words=args.len_penalty_target_words,
        prop_entropy_mu=args.prop_entropy_mu,
        prop_entropy_sigma=args.prop_entropy_sigma,
        adaptive_prop_entropy_target=args.adaptive_prop_entropy_target,
        prop_entropy_ema_momentum=args.prop_entropy_ema_momentum,
        prop_entropy_mu_min=args.prop_entropy_mu_min,
        prop_entropy_mu_max=args.prop_entropy_mu_max,
        zero_entropy_reward_cap=args.zero_entropy_reward_cap,
        proposer_easy_reward_cap=args.proposer_easy_reward_cap,
        proposer_easy_gotcha_reward_cap=args.proposer_easy_gotcha_reward_cap,
        proposer_non_objective_penalty=args.proposer_non_objective_penalty,
        proposer_low_info_majority_penalty=args.proposer_low_info_majority_penalty,
        proposer_slot_compiler_enabled=args.proposer_slot_compiler_enabled,
        proposer_slot_compiler_strict=args.proposer_slot_compiler_strict,
        proposer_trivial_archetype_penalty=args.proposer_trivial_archetype_penalty,
        proposer_answer_family_repeat_penalty=args.proposer_answer_family_repeat_penalty,
        proposer_answer_family_repeat_target=args.proposer_answer_family_repeat_target,
        proposer_candidate_noncanonical_penalty=args.proposer_candidate_noncanonical_penalty,
        proposer_candidate_low_info_penalty=args.proposer_candidate_low_info_penalty,
        solver_noncanonical_answer_penalty=args.solver_noncanonical_answer_penalty,
        solver_low_info_answer_penalty=args.solver_low_info_answer_penalty,
        curriculum_arm_enabled=args.curriculum_arm_enabled,
        curriculum_arm_prompt_enabled=args.curriculum_arm_prompt_enabled,
        curriculum_arm_ema_momentum=args.curriculum_arm_ema_momentum,
        curriculum_arm_progress_weight=args.curriculum_arm_progress_weight,
        curriculum_arm_underuse_weight=args.curriculum_arm_underuse_weight,
        curriculum_arm_easy_penalty_weight=args.curriculum_arm_easy_penalty_weight,
        curriculum_arm_solver_gain_weight=args.curriculum_arm_solver_gain_weight,
        curriculum_arm_prompt_temp=args.curriculum_arm_prompt_temp,
        curriculum_arm_candidate_bonus=args.curriculum_arm_candidate_bonus,
        curriculum_arm_reward_scale=args.curriculum_arm_reward_scale,
        replay_priority_enabled=args.replay_priority_enabled,
        replay_priority_hardness_weight=args.replay_priority_hardness_weight,
        replay_priority_update_weight=args.replay_priority_update_weight,
        replay_priority_novelty_weight=args.replay_priority_novelty_weight,
        replay_anchor_inject_k=args.replay_anchor_inject_k,
        replay_anchor_inject_easy_streak=args.replay_anchor_inject_easy_streak,
        proposer_require_objective=args.proposer_require_objective,
        proposer_num_candidates=args.proposer_num_candidates,
        proposer_spot_check_samples=args.proposer_spot_check_samples,
        proposer_spot_entropy_min_gate=args.proposer_spot_entropy_min_gate,
        proposer_certificate_min_score=args.proposer_certificate_min_score,
        proposer_certificate_strict_struct=args.proposer_certificate_strict_struct,
        proposer_easy_reward_floor=args.proposer_easy_reward_floor,
        proposer_all_easy_rank_spread=args.proposer_all_easy_rank_spread,
        solver_token_entropy_enabled=args.solver_token_entropy_enabled,
        solver_token_entropy_tokens=args.solver_token_entropy_tokens,
        solver_token_entropy_window_size=args.solver_token_entropy_window_size,
        solver_token_entropy_sigmoid_alpha=args.solver_token_entropy_sigmoid_alpha,
        solver_token_entropy_sigmoid_beta=args.solver_token_entropy_sigmoid_beta,
        solver_token_entropy_aggregation=args.solver_token_entropy_aggregation,
        proposer_ste_primary_weight=args.proposer_ste_primary_weight,
        proposer_sample_entropy_weight=args.proposer_sample_entropy_weight,
        proposer_ste_reward_weight=args.proposer_ste_reward_weight,
        solver_pps_enabled=args.solver_pps_enabled,
        solver_skip_update_on_easy=args.solver_skip_update_on_easy,
        easy_update_majority_frac_threshold=args.easy_update_majority_frac_threshold,
        acceptance_require_non_easy=args.acceptance_require_non_easy,
        rejected_question_penalty=args.rejected_question_penalty,
        entropy_iqr_filter_enabled=args.entropy_iqr_filter_enabled,
        entropy_iqr_window_size=args.entropy_iqr_window_size,
        entropy_iqr_min_samples=args.entropy_iqr_min_samples,
        entropy_iqr_easy_quantile=args.entropy_iqr_easy_quantile,
        entropy_iqr_easy_iqr_coef=args.entropy_iqr_easy_iqr_coef,
        entropy_iqr_min_threshold=args.entropy_iqr_min_threshold,
        entropy_iqr_max_threshold=args.entropy_iqr_max_threshold,
        entropy_iqr_filter_min_majority_frac=args.entropy_iqr_filter_min_majority_frac,
        difficulty_sampler_enabled=args.difficulty_sampler_enabled,
        difficulty_sampler_window_size=args.difficulty_sampler_window_size,
        difficulty_sampler_min_samples=args.difficulty_sampler_min_samples,
        difficulty_target_easy=args.difficulty_target_easy,
        difficulty_target_medium=args.difficulty_target_medium,
        difficulty_target_hard=args.difficulty_target_hard,
        difficulty_hard_min_entropy=args.difficulty_hard_min_entropy,
        difficulty_hard_max_margin=args.difficulty_hard_max_margin,
        easy_constraint_target_rate=args.easy_constraint_target_rate,
        easy_constraint_lr=args.easy_constraint_lr,
        easy_constraint_penalty_scale=args.easy_constraint_penalty_scale,
        easy_constraint_selection_scale=args.easy_constraint_selection_scale,
        all_easy_explore_trigger=args.all_easy_explore_trigger,
        all_easy_explore_steps=args.all_easy_explore_steps,
        all_easy_explore_num_candidates=args.all_easy_explore_num_candidates,
        all_easy_explore_temp_boost=args.all_easy_explore_temp_boost,
        all_easy_explore_top_p_boost=args.all_easy_explore_top_p_boost,
        all_easy_explore_penalty_boost=args.all_easy_explore_penalty_boost,
        proposer_early_step1=args.proposer_early_step1,
        proposer_early_step2=args.proposer_early_step2,
        proposer_early_candidate_non_easy_min=args.proposer_early_candidate_non_easy_min,
        proposer_early_all_easy_rate_max=args.proposer_early_all_easy_rate_max,
        proposer_early_reward_clipped_rate_max=args.proposer_early_reward_clipped_rate_max,
        proposer_early_selected_non_easy_min=args.proposer_early_selected_non_easy_min,
        proposer_early_solver_updates_min=args.proposer_early_solver_updates_min,
        proposer_early_collapse_streak_max=args.proposer_early_collapse_streak_max,
        proposer_early_failfast_enabled=args.proposer_early_failfast_enabled,
        proposer_early_failfast_stop=args.proposer_early_failfast_stop,
        proposer_early_failfast_recover=args.proposer_early_failfast_recover,
        proposer_early_failfast_recover_steps=args.proposer_early_failfast_recover_steps,
        proposer_early_hard_stop_min_u_step=args.proposer_early_hard_stop_min_u_step,
        reward_spec_weight=args.reward_spec_weight,
        reward_cycle_weight=args.reward_cycle_weight,
        reward_diversity_weight=args.reward_diversity_weight,
        reward_contradiction_weight=args.reward_contradiction_weight,
        min_spec_quality_for_update=args.min_spec_quality_for_update,
        min_spec_qa_pairs=args.min_spec_qa_pairs,
        max_expected_words=args.max_expected_words,
        max_question_words=args.max_question_words,
        kl_coef=args.kl_coef,
        kl_target=args.kl_target,
        kl_adapt_rate=args.kl_adapt_rate,
        kl_min=args.kl_min,
        kl_max=args.kl_max,
        baseline_momentum=args.baseline_momentum,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
        solver_merger_lora_enabled=args.solver_merger_lora_enabled,
        solver_merger_lora_r=args.solver_merger_lora_r,
        solver_merger_lora_alpha=args.solver_merger_lora_alpha,
        solver_merger_lora_lr=args.solver_merger_lora_lr,
        solver_merger_lora_target_modules=solver_merger_lora_targets,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        seed=args.seed,
        deterministic=args.deterministic,
        log_every=args.log_every,
        save_every=args.save_every,
        max_checkpoints=args.max_checkpoints,
        clear_cache_every=args.clear_cache_every,
        save_generated_images_every=args.save_generated_images_every,
        code_run_registry_enabled=args.code_run_registry_enabled,
        code_run_registry_dir=args.code_run_registry_dir,
        resume_from=args.resume_from,
        start_step=args.start_step,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_log_images_every=args.wandb_log_images_every,
        use_diverse_prompts=args.use_diverse_prompts,
        enable_frozen_judge=args.enable_frozen_judge,
        judge_ema_decay=args.judge_ema_decay,
        judge_gpu_id=args.judge_gpu_id,
        imageless_proposer_mode=args.imageless_proposer_mode,
    )


def _build_unified_config(args):
    from blip3o.train.self_evolving.config import UnifiedSelfEvolvingConfig

    lora_targets = _parse_csv_tuple(args.lora_targets)
    solver_merger_lora_targets = _parse_csv_tuple(args.solver_merger_lora_targets)
    dit_lora_targets = _parse_csv_tuple(args.dit_lora_targets)
    strict_imageless_mode = bool(args.strict_imageless_mode)
    imageless_proposer_mode = bool(args.imageless_proposer_mode) or strict_imageless_mode
    understanding_generated_only = bool(args.understanding_generated_only) or strict_imageless_mode
    cycle_starts_with_generation = bool(args.cycle_starts_with_generation)
    if strict_imageless_mode and not cycle_starts_with_generation:
        print(
            "[Config] strict_imageless_mode enabled: forcing cycle_starts_with_generation=True "
            "to warm generated pool before understanding steps."
        )
        cycle_starts_with_generation = True
    use_ref_answer_scoring = bool(args.use_ref_answer_scoring)
    if imageless_proposer_mode and use_ref_answer_scoring:
        print(
            "[Config] imageless proposer mode is enabled; disabling Solver-derived reference-answer scoring "
            "because it requires a real reference image."
        )
        use_ref_answer_scoring = False
    bootstrap_generated_pool_steps = max(0, int(args.bootstrap_generated_pool_steps))
    if bootstrap_generated_pool_steps > 0 and int(args.generation_steps_per_cycle) <= 0:
        print(
            "[Config] bootstrap_generated_pool_steps > 0 but generation_steps_per_cycle <= 0; "
            "forcing bootstrap_generated_pool_steps=0."
        )
        bootstrap_generated_pool_steps = 0
    if strict_imageless_mode and bootstrap_generated_pool_steps <= 0:
        print(
            "[Config] strict_imageless_mode is enabled without bootstrap steps. "
            "Early understanding steps may skip until generated pool warms up."
        )

    solver_freq = (
        max(1, args.solver_update_freq) if args.solver_update_freq > 0
        else args.synthetic_solver_update_freq
    )
    return UnifiedSelfEvolvingConfig(
        run_name=args.run_name,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        data_split=args.data_split,
        include_subfolders=_parse_subfolders(args.include_subfolders),
        max_images=args.max_images,
        model_name=args.model_name,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        cuda_device=args.cuda_device,
        device_map=args.device_map,
        total_steps=args.total_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        proposer_update_freq=args.proposer_update_freq,
        generator_update_freq=args.generator_update_freq,
        enable_solver_updates=args.enable_solver_updates,
        solver_update_freq=solver_freq,
        temp=args.temp,
        top_p=args.top_p,
        max_new_tokens_solver=args.max_new_tokens_solver,
        max_new_tokens_proposer=args.max_new_tokens_proposer,
        max_new_tokens_caption=args.max_new_tokens_caption,
        max_new_tokens_generator=args.max_new_tokens_generator,
        num_solver_samples=args.num_solver_samples,
        num_solver_samples_spec=args.num_solver_samples_spec,
        num_generations=args.num_generations,
        generation_num_inference_steps=args.generation_num_inference_steps,
        generation_guidance_scale=args.generation_guidance_scale,
        generation_height=args.generation_height,
        generation_width=args.generation_width,
        require_decoder_for_blip3o=args.require_decoder_for_blip3o,
        allow_latent_visualization_fallback=args.allow_latent_visualization_fallback,
        strict_require_generation_tokens=args.strict_require_generation_tokens,
        generator_missing_trace_strategy=args.generator_missing_trace_strategy,
        verification_use_reference_solver=args.verification_use_reference_solver,
        use_self_clip_reward_scoring=args.use_self_clip_reward_scoring,
        generator_update_rule=args.generator_update_rule,
        dpo_beta=args.dpo_beta,
        dpo_label_smoothing=args.dpo_label_smoothing,
        dpo_min_reward_gap=args.dpo_min_reward_gap,
        dpo_min_spec_gap=args.dpo_min_spec_gap,
        dpo_min_confidence_gap=args.dpo_min_confidence_gap,
        dpo_max_contradiction=args.dpo_max_contradiction,
        dpo_pair_selection=args.dpo_pair_selection,
        generator_proxy_max_ratio=args.generator_proxy_max_ratio,
        grpo_clip_ratio=args.grpo_clip_ratio,
        grpo_min_group_std=args.grpo_min_group_std,
        unicorn_generation_enabled=args.unicorn_generation_enabled,
        unicorn_target_difficulty=args.unicorn_target_difficulty,
        unicorn_spec_rejection_enabled=args.unicorn_spec_rejection_enabled,
        unicorn_spec_max_retries=args.unicorn_spec_max_retries,
        unicorn_spec_min_quality=args.unicorn_spec_min_quality,
        unicorn_spec_min_alignment=args.unicorn_spec_min_alignment,
        unicorn_reconstruction_sft_enabled=args.unicorn_reconstruction_sft_enabled,
        unicorn_reconstruction_buffer_size=args.unicorn_reconstruction_buffer_size,
        unicorn_reconstruction_step_freq=args.unicorn_reconstruction_step_freq,
        unicorn_reconstruction_updates_per_step=args.unicorn_reconstruction_updates_per_step,
        unicorn_reconstruction_min_quality=args.unicorn_reconstruction_min_quality,
        unicorn_reconstruction_enable_proposer=args.unicorn_reconstruction_enable_proposer,
        unicorn_reconstruction_enable_generator=args.unicorn_reconstruction_enable_generator,
        dit_update_enabled=args.dit_update_enabled,
        require_dit_update=args.require_dit_update,
        dit_update_freq=args.dit_update_freq,
        dit_lr=args.dit_lr,
        dit_weight_decay=args.dit_weight_decay,
        dit_grad_clip=args.dit_grad_clip,
        dit_grad_accum_steps=args.dit_grad_accum_steps,
        dit_conditioning_dropout=args.dit_conditioning_dropout,
        dit_loss_weight=args.dit_loss_weight,
        dit_prompt_suffix_token_id=args.dit_prompt_suffix_token_id,
        dit_lora_enabled=args.dit_lora_enabled,
        dit_lora_r=args.dit_lora_r,
        dit_lora_alpha=args.dit_lora_alpha,
        dit_lora_dropout=args.dit_lora_dropout,
        dit_lora_target_modules=dit_lora_targets,
        dit_joint_conditioning_train=args.dit_joint_conditioning_train,
        dit_joint_conditioning_lr=args.dit_joint_conditioning_lr,
        dit_reward_loss_weight=args.dit_reward_loss_weight,
        proposer_gen_reward_enabled=args.proposer_gen_reward_enabled,
        proposer_gen_entropy_weight=args.proposer_gen_entropy_weight,
        proposer_gen_baseline_momentum=args.proposer_gen_baseline_momentum,
        gen_step_solver_update_enabled=args.gen_step_solver_update_enabled,
        reset_proposer_baseline=args.reset_proposer_baseline,
        proposer_update_rule=args.proposer_update_rule,
        proposer_grpo_gen_group_size=args.proposer_grpo_gen_group_size,
        grpo_extra_sc_samples=args.grpo_extra_sc_samples,
        proposer_grpo_unverified_extra_margin=args.proposer_grpo_unverified_extra_margin,
        solver_soft_gamma=args.solver_soft_gamma,
        solver_use_temperature_mix=args.solver_use_temperature_mix,
        solver_use_forced_choice_from_proposer=args.solver_use_forced_choice_from_proposer,
        solver_temp_min=args.solver_temp_min,
        solver_temp_max=args.solver_temp_max,
        solver_top_p_min=args.solver_top_p_min,
        solver_top_p_max=args.solver_top_p_max,
        sc_entropy_min=args.sc_entropy_min,
        sc_entropy_max=args.sc_entropy_max,
        sc_margin_max=args.sc_margin_max,
        sc_informative_ratio_min=args.sc_informative_ratio_min,
        sc_negative_weight=args.sc_negative_weight,
        easy_solver_penalty_scale=args.easy_solver_penalty_scale,
        solver_update_on_low_info_easy=args.solver_update_on_low_info_easy,
        solver_low_info_easy_penalty_scale=args.solver_low_info_easy_penalty_scale,
        skip_solver_update_when_uninformative=args.skip_solver_update_when_uninformative,
        solver_always_update_with_informative_scaling=args.solver_always_update_with_informative_scaling,
        solver_update_min_scale=args.solver_update_min_scale,
        len_penalty_weight=args.len_penalty_weight,
        len_penalty_target_words=args.len_penalty_target_words,
        prop_entropy_mu=args.prop_entropy_mu,
        prop_entropy_sigma=args.prop_entropy_sigma,
        adaptive_prop_entropy_target=args.adaptive_prop_entropy_target,
        prop_entropy_ema_momentum=args.prop_entropy_ema_momentum,
        prop_entropy_mu_min=args.prop_entropy_mu_min,
        prop_entropy_mu_max=args.prop_entropy_mu_max,
        zero_entropy_reward_cap=args.zero_entropy_reward_cap,
        proposer_easy_reward_cap=args.proposer_easy_reward_cap,
        proposer_easy_gotcha_reward_cap=args.proposer_easy_gotcha_reward_cap,
        proposer_non_objective_penalty=args.proposer_non_objective_penalty,
        proposer_low_info_majority_penalty=args.proposer_low_info_majority_penalty,
        proposer_slot_compiler_enabled=args.proposer_slot_compiler_enabled,
        proposer_slot_compiler_strict=args.proposer_slot_compiler_strict,
        proposer_trivial_archetype_penalty=args.proposer_trivial_archetype_penalty,
        proposer_answer_family_repeat_penalty=args.proposer_answer_family_repeat_penalty,
        proposer_answer_family_repeat_target=args.proposer_answer_family_repeat_target,
        proposer_candidate_noncanonical_penalty=args.proposer_candidate_noncanonical_penalty,
        proposer_candidate_low_info_penalty=args.proposer_candidate_low_info_penalty,
        solver_noncanonical_answer_penalty=args.solver_noncanonical_answer_penalty,
        solver_low_info_answer_penalty=args.solver_low_info_answer_penalty,
        curriculum_arm_enabled=args.curriculum_arm_enabled,
        curriculum_arm_prompt_enabled=args.curriculum_arm_prompt_enabled,
        curriculum_arm_ema_momentum=args.curriculum_arm_ema_momentum,
        curriculum_arm_progress_weight=args.curriculum_arm_progress_weight,
        curriculum_arm_underuse_weight=args.curriculum_arm_underuse_weight,
        curriculum_arm_easy_penalty_weight=args.curriculum_arm_easy_penalty_weight,
        curriculum_arm_solver_gain_weight=args.curriculum_arm_solver_gain_weight,
        curriculum_arm_prompt_temp=args.curriculum_arm_prompt_temp,
        curriculum_arm_candidate_bonus=args.curriculum_arm_candidate_bonus,
        curriculum_arm_reward_scale=args.curriculum_arm_reward_scale,
        replay_priority_enabled=args.replay_priority_enabled,
        replay_priority_hardness_weight=args.replay_priority_hardness_weight,
        replay_priority_update_weight=args.replay_priority_update_weight,
        replay_priority_novelty_weight=args.replay_priority_novelty_weight,
        replay_anchor_inject_k=args.replay_anchor_inject_k,
        replay_anchor_inject_easy_streak=args.replay_anchor_inject_easy_streak,
        proposer_require_objective=args.proposer_require_objective,
        proposer_num_candidates=args.proposer_num_candidates,
        proposer_spot_check_samples=args.proposer_spot_check_samples,
        proposer_spot_entropy_min_gate=args.proposer_spot_entropy_min_gate,
        proposer_certificate_min_score=args.proposer_certificate_min_score,
        proposer_certificate_strict_struct=args.proposer_certificate_strict_struct,
        proposer_easy_reward_floor=args.proposer_easy_reward_floor,
        proposer_all_easy_rank_spread=args.proposer_all_easy_rank_spread,
        solver_token_entropy_enabled=args.solver_token_entropy_enabled,
        solver_token_entropy_tokens=args.solver_token_entropy_tokens,
        solver_token_entropy_window_size=args.solver_token_entropy_window_size,
        solver_token_entropy_sigmoid_alpha=args.solver_token_entropy_sigmoid_alpha,
        solver_token_entropy_sigmoid_beta=args.solver_token_entropy_sigmoid_beta,
        solver_token_entropy_aggregation=args.solver_token_entropy_aggregation,
        proposer_ste_primary_weight=args.proposer_ste_primary_weight,
        proposer_sample_entropy_weight=args.proposer_sample_entropy_weight,
        proposer_ste_reward_weight=args.proposer_ste_reward_weight,
        solver_pps_enabled=args.solver_pps_enabled,
        solver_skip_update_on_easy=args.solver_skip_update_on_easy,
        easy_update_majority_frac_threshold=args.easy_update_majority_frac_threshold,
        acceptance_require_non_easy=args.acceptance_require_non_easy,
        rejected_question_penalty=args.rejected_question_penalty,
        entropy_iqr_filter_enabled=args.entropy_iqr_filter_enabled,
        entropy_iqr_window_size=args.entropy_iqr_window_size,
        entropy_iqr_min_samples=args.entropy_iqr_min_samples,
        entropy_iqr_easy_quantile=args.entropy_iqr_easy_quantile,
        entropy_iqr_easy_iqr_coef=args.entropy_iqr_easy_iqr_coef,
        entropy_iqr_min_threshold=args.entropy_iqr_min_threshold,
        entropy_iqr_max_threshold=args.entropy_iqr_max_threshold,
        entropy_iqr_filter_min_majority_frac=args.entropy_iqr_filter_min_majority_frac,
        difficulty_sampler_enabled=args.difficulty_sampler_enabled,
        difficulty_sampler_window_size=args.difficulty_sampler_window_size,
        difficulty_sampler_min_samples=args.difficulty_sampler_min_samples,
        difficulty_target_easy=args.difficulty_target_easy,
        difficulty_target_medium=args.difficulty_target_medium,
        difficulty_target_hard=args.difficulty_target_hard,
        difficulty_hard_min_entropy=args.difficulty_hard_min_entropy,
        difficulty_hard_max_margin=args.difficulty_hard_max_margin,
        easy_constraint_target_rate=args.easy_constraint_target_rate,
        easy_constraint_lr=args.easy_constraint_lr,
        easy_constraint_penalty_scale=args.easy_constraint_penalty_scale,
        easy_constraint_selection_scale=args.easy_constraint_selection_scale,
        all_easy_explore_trigger=args.all_easy_explore_trigger,
        all_easy_explore_steps=args.all_easy_explore_steps,
        all_easy_explore_num_candidates=args.all_easy_explore_num_candidates,
        all_easy_explore_temp_boost=args.all_easy_explore_temp_boost,
        all_easy_explore_top_p_boost=args.all_easy_explore_top_p_boost,
        all_easy_explore_penalty_boost=args.all_easy_explore_penalty_boost,
        proposer_early_step1=args.proposer_early_step1,
        proposer_early_step2=args.proposer_early_step2,
        proposer_early_candidate_non_easy_min=args.proposer_early_candidate_non_easy_min,
        proposer_early_all_easy_rate_max=args.proposer_early_all_easy_rate_max,
        proposer_early_reward_clipped_rate_max=args.proposer_early_reward_clipped_rate_max,
        proposer_early_selected_non_easy_min=args.proposer_early_selected_non_easy_min,
        proposer_early_solver_updates_min=args.proposer_early_solver_updates_min,
        proposer_early_collapse_streak_max=args.proposer_early_collapse_streak_max,
        proposer_early_failfast_enabled=args.proposer_early_failfast_enabled,
        proposer_early_failfast_stop=args.proposer_early_failfast_stop,
        proposer_early_failfast_recover=args.proposer_early_failfast_recover,
        proposer_early_failfast_recover_steps=args.proposer_early_failfast_recover_steps,
        proposer_early_hard_stop_min_u_step=args.proposer_early_hard_stop_min_u_step,
        reward_spec_weight=args.reward_spec_weight,
        reward_cycle_weight=args.reward_cycle_weight,
        reward_diversity_weight=args.reward_diversity_weight,
        reward_contradiction_weight=args.reward_contradiction_weight,
        min_spec_quality_for_update=args.min_spec_quality_for_update,
        min_spec_qa_pairs=args.min_spec_qa_pairs,
        max_expected_words=args.max_expected_words,
        max_question_words=args.max_question_words,
        kl_coef=args.kl_coef,
        kl_target=args.kl_target,
        kl_adapt_rate=args.kl_adapt_rate,
        kl_min=args.kl_min,
        kl_max=args.kl_max,
        baseline_momentum=args.baseline_momentum,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
        solver_merger_lora_enabled=args.solver_merger_lora_enabled,
        solver_merger_lora_r=args.solver_merger_lora_r,
        solver_merger_lora_alpha=args.solver_merger_lora_alpha,
        solver_merger_lora_lr=args.solver_merger_lora_lr,
        solver_merger_lora_target_modules=solver_merger_lora_targets,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        seed=args.seed,
        deterministic=args.deterministic,
        log_every=args.log_every,
        save_every=args.save_every,
        max_checkpoints=args.max_checkpoints,
        clear_cache_every=args.clear_cache_every,
        save_generated_images_every=args.save_generated_images_every,
        code_run_registry_enabled=args.code_run_registry_enabled,
        code_run_registry_dir=args.code_run_registry_dir,
        resume_from=args.resume_from,
        start_step=args.start_step,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_log_images_every=args.wandb_log_images_every,
        understanding_steps_per_cycle=args.understanding_steps_per_cycle,
        generation_steps_per_cycle=args.generation_steps_per_cycle,
        cycle_starts_with_generation=cycle_starts_with_generation,
        bootstrap_generated_pool_steps=bootstrap_generated_pool_steps,
        synthetic_solver_update_freq=args.synthetic_solver_update_freq,
        synthetic_solver_hard_only=args.synthetic_solver_hard_only,
        solver_hardness_min_entropy=args.solver_hardness_min_entropy,
        # Self-evolving feedback loop
        use_ref_answer_scoring=use_ref_answer_scoring,
        replay_buffer_size=args.replay_buffer_size,
        replay_min_reward=args.replay_min_reward,
        replay_max_staleness=args.replay_max_staleness,
        gen_mix_source_mode=args.gen_mix_source_mode,
        generated_mix_dir=args.generated_mix_dir,
        generated_mix_min_reward=args.generated_mix_min_reward,
        generated_mix_max_files=args.generated_mix_max_files,
        generated_mix_refresh_every=args.generated_mix_refresh_every,
        understanding_generated_only=understanding_generated_only,
        strict_imageless_mode=strict_imageless_mode,
        gen_mix_ratio_start=args.gen_mix_ratio_start,
        gen_mix_ratio_max=args.gen_mix_ratio_max,
        gen_mix_ratio_warmup_steps=args.gen_mix_ratio_warmup_steps,
        reward_ema_momentum=args.reward_ema_momentum,
        use_diverse_prompts=args.use_diverse_prompts,
        enable_frozen_judge=args.enable_frozen_judge,
        judge_ema_decay=args.judge_ema_decay,
        judge_gpu_id=args.judge_gpu_id,
        imageless_proposer_mode=imageless_proposer_mode,
    )


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.load_in_4bit and not args.use_lora:
        parser.error("--load_in_4bit is a QLoRA mode and requires --use_lora")

    if args.experiment == "understanding_self_evolving":
        cfg = _build_understanding_config(args)
        from blip3o.train.self_evolving.understanding_trainer import UnderstandingSelfEvolvingTrainer
        trainer = UnderstandingSelfEvolvingTrainer(cfg)
        trainer.train()

    elif args.experiment == "generation_self_evolving":
        cfg = _build_generation_config(args)
        from blip3o.train.self_evolving.generation_trainer import GenerationSelfEvolvingTrainer
        trainer = GenerationSelfEvolvingTrainer(cfg)
        trainer.train()

    elif args.experiment == "unified_self_evolving":
        cfg = _build_unified_config(args)
        from blip3o.train.self_evolving.unified_trainer import UnifiedSelfEvolvingTrainer
        trainer = UnifiedSelfEvolvingTrainer(cfg)
        trainer.train()

    else:
        raise NotImplementedError(
            f"Experiment mode '{args.experiment}' is not implemented."
        )


if __name__ == "__main__":
    main()
