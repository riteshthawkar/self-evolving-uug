# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import traceback
from datetime import timedelta

import numpy as np
import torch

# Ensure BAGEL imports resolve regardless of launch cwd/module style.
_THIS_FILE = os.path.abspath(__file__)
_BAGEL_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
_REPO_ROOT = os.path.dirname(_BAGEL_ROOT)
for _path in (_BAGEL_ROOT, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from train.self_evolving.config import ModelLoadConfig, RolloutConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BAGEL self-evolving framework (rollout + optional policy updates)."
    )
    p.add_argument(
        "--experiment",
        type=str,
        default="understanding_self_evolving",
        choices=["understanding_self_evolving", "generation_self_evolving", "unified_self_evolving"],
        help="Training mode: understanding only, generation only, or unified alternating U/G.",
    )

    # Model/runtime
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--vae_device", type=str, default="")
    p.add_argument("--distributed", action="store_true", default=False)
    p.add_argument("--disable_distributed", dest="distributed", action="store_false")
    p.add_argument("--dist_backend", type=str, default="nccl")
    p.add_argument("--dist_timeout_minutes", type=int, default=120)
    p.add_argument("--dist_data_shard", action="store_true", default=True)
    p.add_argument("--disable_dist_data_shard", dest="dist_data_shard", action="store_false")
    p.add_argument(
        "--lora_checkpoint_path",
        type=str,
        default="",
        help=(
            "Optional LoRA checkpoint to load before rollout/training. "
            "Accepts: step_XXXXXX.pt, checkpoint directory, or step_XXXXXX_lora folder."
        ),
    )
    p.add_argument("--max_latent_size", type=int, default=64)
    p.add_argument("--enable_lora", action="store_true", default=False)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules_csv",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument(
        "--lora_role_adapters_csv",
        type=str,
        default="proposer,solver,generator",
    )
    p.add_argument("--lora_default_adapter", type=str, default="proposer")

    # Data/outputs
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--understanding_steps_per_cycle", type=int, default=3)
    p.add_argument("--generation_steps_per_cycle", type=int, default=2)
    p.add_argument("--replay_buffer_size", type=int, default=1)
    p.add_argument("--replay_min_reward", type=float, default=1.10)
    p.add_argument("--replay_max_staleness", type=int, default=1)
    p.add_argument(
        "--gen_mix_source_mode",
        type=str,
        default="buffer",
        choices=["buffer", "folder"],
        help="Source for generation->understanding mixing in unified mode.",
    )
    p.add_argument("--generated_mix_dir", type=str, default="")
    p.add_argument("--generated_mix_min_reward", type=float, default=0.5)
    p.add_argument("--generated_mix_max_files", type=int, default=5000)
    p.add_argument("--generated_mix_refresh_every", type=int, default=10)
    p.add_argument("--understanding_generated_only", action="store_true", default=False)
    p.add_argument(
        "--disable_understanding_generated_only",
        dest="understanding_generated_only",
        action="store_false",
    )
    p.add_argument("--gen_mix_ratio_start", type=float, default=0.0)
    p.add_argument("--gen_mix_ratio_max", type=float, default=0.0)
    p.add_argument("--gen_mix_ratio_warmup_steps", type=int, default=1)
    p.add_argument("--reward_ema_momentum", type=float, default=0.95)

    # Generation/reward knobs
    p.add_argument("--max_new_tokens_proposer", type=int, default=256)
    p.add_argument("--max_new_tokens_solver", type=int, default=96)
    p.add_argument("--proposer_temperature", type=float, default=0.9)
    p.add_argument("--num_solver_samples", type=int, default=7)
    p.add_argument("--solver_temp_min", type=float, default=0.5)
    p.add_argument("--solver_temp_max", type=float, default=2.0)
    p.add_argument("--proposer_entropy_mu", type=float, default=0.9)
    p.add_argument("--proposer_entropy_sigma", type=float, default=0.25)
    p.add_argument("--solver_unsolvable_maj_threshold", type=float, default=0.20)
    p.add_argument("--zero_entropy_eps", type=float, default=1e-6)
    p.add_argument("--proposer_non_objective_penalty", type=float, default=0.20)
    p.add_argument("--rejected_question_penalty", type=float, default=0.35)
    p.add_argument("--proposer_require_objective", action="store_true", default=True)
    p.add_argument(
        "--disable_proposer_require_objective",
        dest="proposer_require_objective",
        action="store_false",
    )
    p.add_argument("--acceptance_require_non_easy", action="store_true", default=True)
    p.add_argument(
        "--disable_acceptance_require_non_easy",
        dest="acceptance_require_non_easy",
        action="store_false",
    )
    p.add_argument("--save_raw_generations", action="store_true", default=True)
    p.add_argument(
        "--disable_save_raw_generations",
        dest="save_raw_generations",
        action="store_false",
    )

    # Generation phase reward / scoring controls.
    p.add_argument("--suder_generation_enabled", action="store_true", default=False)
    p.add_argument(
        "--disable_suder_generation",
        dest="suder_generation_enabled",
        action="store_false",
    )
    p.add_argument("--max_new_tokens_gen_spec", type=int, default=384)
    p.add_argument("--gen_spec_temperature", type=float, default=0.9)
    p.add_argument("--gen_spec_min_qa_pairs", type=int, default=2)
    p.add_argument("--proposer_gen_entropy_weight", type=float, default=0.7)
    p.add_argument("--proposer_gen_baseline_momentum", type=float, default=0.6)
    p.add_argument("--generation_num_candidates", type=int, default=3)
    p.add_argument("--generation_cfg_text_scale", type=float, default=4.0)
    p.add_argument("--generation_cfg_img_scale", type=float, default=1.5)
    p.add_argument("--generation_num_timesteps", type=int, default=50)
    p.add_argument("--generation_timestep_shift", type=float, default=3.0)
    p.add_argument("--generation_image_size", type=int, default=1024)
    p.add_argument("--reward_spec_weight", type=float, default=0.65)
    p.add_argument("--reward_cycle_weight", type=float, default=0.20)
    p.add_argument("--reward_diversity_weight", type=float, default=0.10)
    p.add_argument("--reward_contradiction_weight", type=float, default=0.20)
    p.add_argument("--min_spec_quality_for_update", type=float, default=0.35)
    p.add_argument("--min_spec_qa_pairs", type=int, default=2)
    p.add_argument("--max_expected_words", type=int, default=8)
    p.add_argument("--max_question_words", type=int, default=24)
    p.add_argument("--save_generated_images", action="store_true", default=False)
    p.add_argument(
        "--disable_save_generated_images",
        dest="save_generated_images",
        action="store_false",
    )

    # Policy update knobs (phase-2).
    p.add_argument("--policy_updates_enabled", action="store_true", default=False)
    p.add_argument(
        "--disable_policy_updates",
        dest="policy_updates_enabled",
        action="store_false",
    )
    p.add_argument("--policy_update_method", type=str, default="grpo", choices=["reinforce", "grpo"])
    p.add_argument("--policy_use_bf16", action="store_true", default=True)
    p.add_argument(
        "--disable_policy_use_bf16",
        dest="policy_use_bf16",
        action="store_false",
    )
    p.add_argument("--policy_lr", type=float, default=1e-6)
    p.add_argument("--policy_weight_decay", type=float, default=0.01)
    p.add_argument("--policy_max_grad_norm", type=float, default=1.0)
    p.add_argument("--policy_grad_accum_steps", type=int, default=1)
    p.add_argument("--policy_reward_scale", type=float, default=1.0)
    p.add_argument("--baseline_momentum", type=float, default=0.6)
    p.add_argument("--grpo_eps", type=float, default=1e-6)
    p.add_argument("--kl_coef", type=float, default=0.01)
    p.add_argument("--kl_target", type=float, default=0.02)
    p.add_argument("--kl_adapt_rate", type=float, default=0.10)
    p.add_argument("--kl_min", type=float, default=0.001)
    p.add_argument("--kl_max", type=float, default=1e2)
    p.add_argument("--solver_reward_mix_gamma", type=float, default=0.7)
    p.add_argument("--solver_skip_easy_updates", action="store_true", default=True)
    p.add_argument(
        "--disable_solver_skip_easy_updates",
        dest="solver_skip_easy_updates",
        action="store_false",
    )
    p.add_argument("--solver_easy_update_majority_threshold", type=float, default=0.85)
    p.add_argument("--proposer_num_candidates", type=int, default=5)
    p.add_argument("--proposer_spot_check_samples", type=int, default=3)
    p.add_argument("--proposer_spot_entropy_min_gate", type=float, default=0.05)
    p.add_argument("--proposer_grpo_gen_group_size", type=int, default=3)
    p.add_argument("--grpo_extra_sc_samples", type=int, default=3)
    p.add_argument("--understanding_skip_no_acceptable", action="store_true", default=True)
    p.add_argument(
        "--disable_understanding_skip_no_acceptable",
        dest="understanding_skip_no_acceptable",
        action="store_false",
    )
    p.add_argument("--understanding_require_acceptable_for_update", action="store_true", default=True)
    p.add_argument(
        "--disable_understanding_require_acceptable_for_update",
        dest="understanding_require_acceptable_for_update",
        action="store_false",
    )
    p.add_argument("--understanding_update_require_disagreement", action="store_true", default=True)
    p.add_argument(
        "--disable_understanding_update_require_disagreement",
        dest="understanding_update_require_disagreement",
        action="store_false",
    )
    p.add_argument("--proposer_reject_unsolvable", action="store_true", default=True)
    p.add_argument(
        "--disable_proposer_reject_unsolvable",
        dest="proposer_reject_unsolvable",
        action="store_false",
    )
    p.add_argument("--solver_skip_unsolvable_updates", action="store_true", default=True)
    p.add_argument(
        "--disable_solver_skip_unsolvable_updates",
        dest="solver_skip_unsolvable_updates",
        action="store_false",
    )
    p.add_argument("--score_grpo_extras", action="store_true", default=True)
    p.add_argument(
        "--disable_score_grpo_extras",
        dest="score_grpo_extras",
        action="store_false",
    )
    p.add_argument("--grpo_extra_temp_multiplier", type=float, default=1.5)
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
    p.add_argument("--ste_spot_easy_quantile", type=float, default=0.30)
    p.add_argument("--proposer_ste_primary_weight", type=float, default=0.70)
    p.add_argument("--proposer_sample_entropy_weight", type=float, default=0.30)
    p.add_argument("--proposer_certificate_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_proposer_certificate",
        dest="proposer_certificate_enabled",
        action="store_false",
    )
    p.add_argument("--proposer_certificate_min_score", type=float, default=0.55)
    p.add_argument("--proposer_certificate_weight", type=float, default=0.75)
    p.add_argument("--proposer_certificate_strict_struct", action="store_true", default=True)
    p.add_argument(
        "--disable_proposer_certificate_strict_struct",
        dest="proposer_certificate_strict_struct",
        action="store_false",
    )
    p.add_argument("--proposer_warm_start_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_proposer_warm_start",
        dest="proposer_warm_start_enabled",
        action="store_false",
    )
    p.add_argument("--proposer_warm_start_max_steps", type=int, default=30)
    p.add_argument("--proposer_warm_start_exit_window", type=int, default=5)
    p.add_argument("--proposer_warm_start_exit_consecutive", type=int, default=2)
    p.add_argument("--proposer_warm_start_entropy_exit_threshold", type=float, default=0.10)
    p.add_argument("--proposer_warm_start_easy_reject_penalty_scale", type=float, default=0.0)
    p.add_argument("--proposer_warm_start_certificate_weight", type=float, default=0.50)
    p.add_argument("--hardness_debt_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_hardness_debt",
        dest="hardness_debt_enabled",
        action="store_false",
    )
    p.add_argument("--hardness_debt_inc_easy", type=float, default=1.50)
    p.add_argument("--hardness_debt_dec_non_easy", type=float, default=1.00)
    p.add_argument("--hardness_debt_max", type=float, default=6.0)
    p.add_argument("--hardness_debt_hard_recovery_threshold", type=float, default=3.0)
    p.add_argument("--hardness_debt_recovery_easy_weight", type=float, default=0.0)
    p.add_argument("--hardness_debt_recovery_medium_weight", type=float, default=0.30)
    p.add_argument("--hardness_debt_recovery_hard_weight", type=float, default=0.70)
    p.add_argument("--hardness_debt_stale_steps", type=int, default=8)
    p.add_argument("--hardness_debt_stale_reset_to", type=float, default=3.0)
    p.add_argument("--hardness_debt_stale_escape_steps", type=int, default=8)
    p.add_argument("--hardness_debt_stale_easy_weight", type=float, default=0.05)
    p.add_argument("--hardness_debt_stale_medium_weight", type=float, default=0.55)
    p.add_argument("--hardness_debt_stale_hard_weight", type=float, default=0.40)
    p.add_argument("--hardness_debt_temp_boost_max", type=float, default=0.30)
    p.add_argument("--hardness_debt_penalty_boost_max", type=float, default=0.30)
    p.add_argument("--difficulty_sampler_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_difficulty_sampler",
        dest="difficulty_sampler_enabled",
        action="store_false",
    )
    p.add_argument("--difficulty_sampler_window_size", type=int, default=256)
    p.add_argument("--difficulty_sampler_min_samples", type=int, default=32)
    p.add_argument("--difficulty_target_easy", type=float, default=0.10)
    p.add_argument("--difficulty_target_medium", type=float, default=0.70)
    p.add_argument("--difficulty_target_hard", type=float, default=0.30)
    p.add_argument("--difficulty_hard_min_entropy", type=float, default=0.90)
    p.add_argument("--difficulty_hard_max_margin", type=float, default=0.35)
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
    p.add_argument("--all_easy_explore_trigger", type=int, default=2)
    p.add_argument("--all_easy_explore_steps", type=int, default=10)
    p.add_argument("--all_easy_explore_num_candidates", type=int, default=6)
    p.add_argument("--all_easy_explore_temp_boost", type=float, default=1.20)
    p.add_argument("--all_easy_explore_top_p_boost", type=float, default=0.15)
    p.add_argument("--all_easy_explore_penalty_boost", type=float, default=0.50)
    p.add_argument("--proposer_contrastive_replay_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_proposer_contrastive_replay",
        dest="proposer_contrastive_replay_enabled",
        action="store_false",
    )
    p.add_argument("--proposer_contrastive_replay_size", type=int, default=256)
    p.add_argument("--proposer_contrastive_pos_bonus", type=float, default=0.08)
    p.add_argument("--proposer_contrastive_neg_penalty", type=float, default=0.08)
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
    p.add_argument("--proposer_early_stage1_u_step", type=int, default=12)
    p.add_argument("--proposer_early_stage2_u_step", type=int, default=20)
    p.add_argument("--proposer_early_hard_stop_min_u_step", type=int, default=80)
    p.add_argument("--proposer_early_candidate_non_easy_rate_min", type=float, default=0.08)
    p.add_argument("--proposer_early_all_easy_rate_max", type=float, default=0.93)
    p.add_argument("--proposer_early_reward_clipped_rate_max", type=float, default=0.85)
    p.add_argument("--proposer_early_selected_non_easy_rate_min", type=float, default=0.10)
    p.add_argument("--proposer_early_solver_updates_min", type=int, default=1)
    p.add_argument("--proposer_early_max_collapse_streak", type=int, default=3)
    p.add_argument("--grpo_degenerate_noise_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_grpo_degenerate_noise",
        dest="grpo_degenerate_noise_enabled",
        action="store_false",
    )
    p.add_argument("--grpo_degenerate_noise_sigma", type=float, default=0.03)
    p.add_argument("--grpo_degenerate_noise_std_threshold", type=float, default=1e-6)
    p.add_argument("--grpo_pairwise_ranking_enabled", action="store_true", default=True)
    p.add_argument(
        "--disable_grpo_pairwise_ranking",
        dest="grpo_pairwise_ranking_enabled",
        action="store_false",
    )
    p.add_argument("--grpo_pairwise_ranking_weight", type=float, default=0.15)
    p.add_argument("--grpo_pairwise_margin", type=float, default=0.10)
    p.add_argument("--grpo_pairwise_easy_penalty", type=float, default=0.12)
    p.add_argument("--proposer_all_easy_rank_spread", type=float, default=0.20)
    p.add_argument("--gen_step_solver_update_enabled", action="store_true", default=False)
    p.add_argument(
        "--disable_gen_step_solver_update",
        dest="gen_step_solver_update_enabled",
        action="store_false",
    )
    p.add_argument("--train_understanding_proposer", action="store_true", default=True)
    p.add_argument(
        "--disable_train_understanding_proposer",
        dest="train_understanding_proposer",
        action="store_false",
    )
    p.add_argument("--train_solver", action="store_true", default=True)
    p.add_argument(
        "--disable_train_solver",
        dest="train_solver",
        action="store_false",
    )
    p.add_argument("--train_generation_proposer", action="store_true", default=True)
    p.add_argument(
        "--disable_train_generation_proposer",
        dest="train_generation_proposer",
        action="store_false",
    )
    p.add_argument("--train_generator", action="store_true", default=True)
    p.add_argument(
        "--disable_train_generator",
        dest="train_generator",
        action="store_false",
    )
    p.add_argument("--checkpoint_every", type=int, default=100)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--save_lora_only", action="store_true", default=True)
    p.add_argument(
        "--disable_save_lora_only",
        dest="save_lora_only",
        action="store_false",
    )
    return p


def _set_seed(seed: int, rank_offset: int = 0) -> None:
    seed = int(seed) + int(rank_offset) * 100003
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _init_distributed_runtime(args) -> dict:
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1") or "1"))
    rank = max(0, int(os.environ.get("RANK", "0") or "0"))
    local_rank = max(0, int(os.environ.get("LOCAL_RANK", str(rank)) or str(rank)))
    distributed_active = bool(args.distributed) or world_size > 1
    backend = str(args.dist_backend or "nccl").strip().lower()

    if distributed_active:
        if not (torch.distributed.is_available()):
            raise RuntimeError("Distributed mode requested but torch.distributed is unavailable.")
        use_cuda_device = bool(torch.cuda.is_available() and str(args.device).startswith("cuda"))
        if (not use_cuda_device) and backend == "nccl":
            backend = "gloo"
        if use_cuda_device:
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
            vae_raw = str(args.vae_device or "").strip()
            # In distributed mode, pin model+VAE to local rank by default to avoid
            # accidental cross-rank contention on a single VAE GPU.
            if (not vae_raw) or vae_raw.startswith("cuda"):
                args.vae_device = str(args.device)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend,
                timeout=timedelta(minutes=max(1, int(args.dist_timeout_minutes))),
            )
        world_size = max(1, int(torch.distributed.get_world_size()))
        rank = max(0, int(torch.distributed.get_rank()))
        local_rank = max(0, int(os.environ.get("LOCAL_RANK", str(local_rank)) or str(local_rank)))

    return {
        "enabled": bool(distributed_active),
        "backend": backend,
        "world_size": int(world_size),
        "rank": int(rank),
        "local_rank": int(local_rank),
        "main_process": bool(rank == 0),
    }


def _destroy_distributed_runtime() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    args = _build_parser().parse_args()
    dist_ctx = _init_distributed_runtime(args)
    try:
        if bool(args.policy_updates_enabled) and not bool(args.enable_lora):
            raise ValueError("--policy_updates_enabled requires --enable_lora for trainable role adapters.")
        _set_seed(int(args.seed), rank_offset=int(dist_ctx["rank"]))
        # Delay heavy imports (model + cv2 dependencies) until actual execution.
        from train.self_evolving.model_loader import load_bagel_runtime
        from train.self_evolving.trainer import SelfEvolvingUnderstandingTrainer

        model_cfg = ModelLoadConfig(
            model_path=args.model_path,
            device=args.device,
            vae_device=str(args.vae_device or ""),
            lora_checkpoint_path=str(args.lora_checkpoint_path or ""),
            max_latent_size=int(args.max_latent_size),
            enable_lora=bool(args.enable_lora),
            lora_rank=int(args.lora_rank),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            lora_target_modules_csv=str(args.lora_target_modules_csv),
            lora_role_adapters_csv=str(args.lora_role_adapters_csv),
            lora_default_adapter=str(args.lora_default_adapter),
        )
        rollout_cfg = RolloutConfig(
        experiment_name=str(args.experiment),
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        steps=int(args.steps),
        seed=int(args.seed),
        log_every=int(args.log_every),
        max_new_tokens_proposer=int(args.max_new_tokens_proposer),
        max_new_tokens_solver=int(args.max_new_tokens_solver),
        proposer_temperature=float(args.proposer_temperature),
        num_solver_samples=int(args.num_solver_samples),
        solver_temp_min=float(args.solver_temp_min),
        solver_temp_max=float(args.solver_temp_max),
        proposer_entropy_mu=float(args.proposer_entropy_mu),
        proposer_entropy_sigma=float(args.proposer_entropy_sigma),
        solver_unsolvable_maj_threshold=float(args.solver_unsolvable_maj_threshold),
        zero_entropy_eps=float(args.zero_entropy_eps),
        proposer_non_objective_penalty=float(args.proposer_non_objective_penalty),
        rejected_question_penalty=float(args.rejected_question_penalty),
        proposer_require_objective=bool(args.proposer_require_objective),
        acceptance_require_non_easy=bool(args.acceptance_require_non_easy),
        save_raw_generations=bool(args.save_raw_generations),
        suder_generation_enabled=bool(args.suder_generation_enabled),
        max_new_tokens_gen_spec=int(args.max_new_tokens_gen_spec),
        gen_spec_temperature=float(args.gen_spec_temperature),
        gen_spec_min_qa_pairs=int(args.gen_spec_min_qa_pairs),
        proposer_gen_entropy_weight=float(args.proposer_gen_entropy_weight),
        proposer_gen_baseline_momentum=float(args.proposer_gen_baseline_momentum),
        generation_num_candidates=max(1, int(args.generation_num_candidates)),
        generation_cfg_text_scale=float(args.generation_cfg_text_scale),
        generation_cfg_img_scale=float(args.generation_cfg_img_scale),
        generation_num_timesteps=int(args.generation_num_timesteps),
        generation_timestep_shift=float(args.generation_timestep_shift),
        generation_image_size=int(args.generation_image_size),
        reward_spec_weight=float(args.reward_spec_weight),
        reward_cycle_weight=float(args.reward_cycle_weight),
        reward_diversity_weight=float(args.reward_diversity_weight),
        reward_contradiction_weight=float(args.reward_contradiction_weight),
        min_spec_quality_for_update=float(args.min_spec_quality_for_update),
        min_spec_qa_pairs=max(1, int(args.min_spec_qa_pairs)),
        max_expected_words=max(1, int(args.max_expected_words)),
        max_question_words=max(1, int(args.max_question_words)),
        save_generated_images=bool(args.save_generated_images),
        policy_updates_enabled=bool(args.policy_updates_enabled),
        policy_update_method=str(args.policy_update_method),
        policy_use_bf16=bool(args.policy_use_bf16),
        policy_lr=float(args.policy_lr),
        policy_weight_decay=float(args.policy_weight_decay),
        policy_max_grad_norm=float(args.policy_max_grad_norm),
        policy_grad_accum_steps=int(args.policy_grad_accum_steps),
        policy_reward_scale=float(args.policy_reward_scale),
        baseline_momentum=float(args.baseline_momentum),
        grpo_eps=float(args.grpo_eps),
        kl_coef=float(args.kl_coef),
        kl_target=float(args.kl_target),
        kl_adapt_rate=float(args.kl_adapt_rate),
        kl_min=float(args.kl_min),
        kl_max=float(args.kl_max),
        solver_reward_mix_gamma=float(args.solver_reward_mix_gamma),
        solver_skip_easy_updates=bool(args.solver_skip_easy_updates),
        solver_easy_update_majority_threshold=float(args.solver_easy_update_majority_threshold),
        proposer_num_candidates=int(args.proposer_num_candidates),
        proposer_spot_check_samples=int(args.proposer_spot_check_samples),
        proposer_spot_entropy_min_gate=float(args.proposer_spot_entropy_min_gate),
        proposer_grpo_gen_group_size=max(1, int(args.proposer_grpo_gen_group_size)),
        grpo_extra_sc_samples=max(1, int(args.grpo_extra_sc_samples)),
        understanding_skip_no_acceptable=bool(args.understanding_skip_no_acceptable),
        understanding_require_acceptable_for_update=bool(args.understanding_require_acceptable_for_update),
        understanding_update_require_disagreement=bool(args.understanding_update_require_disagreement),
        proposer_reject_unsolvable=bool(args.proposer_reject_unsolvable),
        solver_skip_unsolvable_updates=bool(args.solver_skip_unsolvable_updates),
        score_grpo_extras=bool(args.score_grpo_extras),
        grpo_extra_temp_multiplier=float(args.grpo_extra_temp_multiplier),
        solver_token_entropy_enabled=bool(args.solver_token_entropy_enabled),
        solver_token_entropy_tokens=int(args.solver_token_entropy_tokens),
        solver_token_entropy_window_size=int(args.solver_token_entropy_window_size),
        solver_token_entropy_sigmoid_alpha=float(args.solver_token_entropy_sigmoid_alpha),
        solver_token_entropy_sigmoid_beta=float(args.solver_token_entropy_sigmoid_beta),
        ste_spot_easy_quantile=float(args.ste_spot_easy_quantile),
        proposer_ste_primary_weight=float(args.proposer_ste_primary_weight),
        proposer_sample_entropy_weight=float(args.proposer_sample_entropy_weight),
        proposer_certificate_enabled=bool(args.proposer_certificate_enabled),
        proposer_certificate_min_score=float(args.proposer_certificate_min_score),
        proposer_certificate_weight=float(args.proposer_certificate_weight),
        proposer_certificate_strict_struct=bool(args.proposer_certificate_strict_struct),
        proposer_warm_start_enabled=bool(args.proposer_warm_start_enabled),
        proposer_warm_start_max_steps=max(1, int(args.proposer_warm_start_max_steps)),
        proposer_warm_start_exit_window=max(1, int(args.proposer_warm_start_exit_window)),
        proposer_warm_start_exit_consecutive=max(1, int(args.proposer_warm_start_exit_consecutive)),
        proposer_warm_start_entropy_exit_threshold=float(args.proposer_warm_start_entropy_exit_threshold),
        proposer_warm_start_easy_reject_penalty_scale=float(args.proposer_warm_start_easy_reject_penalty_scale),
        proposer_warm_start_certificate_weight=float(args.proposer_warm_start_certificate_weight),
        hardness_debt_enabled=bool(args.hardness_debt_enabled),
        hardness_debt_inc_easy=float(args.hardness_debt_inc_easy),
        hardness_debt_dec_non_easy=float(args.hardness_debt_dec_non_easy),
        hardness_debt_max=float(args.hardness_debt_max),
        hardness_debt_hard_recovery_threshold=float(args.hardness_debt_hard_recovery_threshold),
        hardness_debt_recovery_easy_weight=float(args.hardness_debt_recovery_easy_weight),
        hardness_debt_recovery_medium_weight=float(args.hardness_debt_recovery_medium_weight),
        hardness_debt_recovery_hard_weight=float(args.hardness_debt_recovery_hard_weight),
        hardness_debt_stale_steps=max(1, int(args.hardness_debt_stale_steps)),
        hardness_debt_stale_reset_to=float(args.hardness_debt_stale_reset_to),
        hardness_debt_stale_escape_steps=max(1, int(args.hardness_debt_stale_escape_steps)),
        hardness_debt_stale_easy_weight=float(args.hardness_debt_stale_easy_weight),
        hardness_debt_stale_medium_weight=float(args.hardness_debt_stale_medium_weight),
        hardness_debt_stale_hard_weight=float(args.hardness_debt_stale_hard_weight),
        hardness_debt_temp_boost_max=float(args.hardness_debt_temp_boost_max),
        hardness_debt_penalty_boost_max=float(args.hardness_debt_penalty_boost_max),
        difficulty_sampler_enabled=bool(args.difficulty_sampler_enabled),
        difficulty_sampler_window_size=max(8, int(args.difficulty_sampler_window_size)),
        difficulty_sampler_min_samples=max(1, int(args.difficulty_sampler_min_samples)),
        difficulty_target_easy=float(args.difficulty_target_easy),
        difficulty_target_medium=float(args.difficulty_target_medium),
        difficulty_target_hard=float(args.difficulty_target_hard),
        difficulty_hard_min_entropy=float(args.difficulty_hard_min_entropy),
        difficulty_hard_max_margin=float(args.difficulty_hard_max_margin),
        entropy_iqr_filter_enabled=bool(args.entropy_iqr_filter_enabled),
        entropy_iqr_window_size=max(8, int(args.entropy_iqr_window_size)),
        entropy_iqr_min_samples=max(1, int(args.entropy_iqr_min_samples)),
        entropy_iqr_easy_quantile=float(args.entropy_iqr_easy_quantile),
        entropy_iqr_easy_iqr_coef=float(args.entropy_iqr_easy_iqr_coef),
        entropy_iqr_min_threshold=float(args.entropy_iqr_min_threshold),
        entropy_iqr_max_threshold=float(args.entropy_iqr_max_threshold),
        entropy_iqr_filter_min_majority_frac=float(args.entropy_iqr_filter_min_majority_frac),
        all_easy_explore_trigger=max(1, int(args.all_easy_explore_trigger)),
        all_easy_explore_steps=max(1, int(args.all_easy_explore_steps)),
        all_easy_explore_num_candidates=max(1, int(args.all_easy_explore_num_candidates)),
        all_easy_explore_temp_boost=float(args.all_easy_explore_temp_boost),
        all_easy_explore_top_p_boost=float(args.all_easy_explore_top_p_boost),
        all_easy_explore_penalty_boost=float(args.all_easy_explore_penalty_boost),
        proposer_contrastive_replay_enabled=bool(args.proposer_contrastive_replay_enabled),
        proposer_contrastive_replay_size=max(8, int(args.proposer_contrastive_replay_size)),
        proposer_contrastive_pos_bonus=float(args.proposer_contrastive_pos_bonus),
        proposer_contrastive_neg_penalty=float(args.proposer_contrastive_neg_penalty),
        proposer_early_failfast_enabled=bool(args.proposer_early_failfast_enabled),
        proposer_early_failfast_stop=bool(args.proposer_early_failfast_stop),
        proposer_early_failfast_recover=bool(args.proposer_early_failfast_recover),
        proposer_early_failfast_recover_steps=max(1, int(args.proposer_early_failfast_recover_steps)),
        proposer_early_stage1_u_step=max(1, int(args.proposer_early_stage1_u_step)),
        proposer_early_stage2_u_step=max(1, int(args.proposer_early_stage2_u_step)),
        proposer_early_hard_stop_min_u_step=max(1, int(args.proposer_early_hard_stop_min_u_step)),
        proposer_early_candidate_non_easy_rate_min=float(args.proposer_early_candidate_non_easy_rate_min),
        proposer_early_all_easy_rate_max=float(args.proposer_early_all_easy_rate_max),
        proposer_early_reward_clipped_rate_max=float(args.proposer_early_reward_clipped_rate_max),
        proposer_early_selected_non_easy_rate_min=float(args.proposer_early_selected_non_easy_rate_min),
        proposer_early_solver_updates_min=max(0, int(args.proposer_early_solver_updates_min)),
        proposer_early_max_collapse_streak=max(0, int(args.proposer_early_max_collapse_streak)),
        grpo_degenerate_noise_enabled=bool(args.grpo_degenerate_noise_enabled),
        grpo_degenerate_noise_sigma=float(args.grpo_degenerate_noise_sigma),
        grpo_degenerate_noise_std_threshold=float(args.grpo_degenerate_noise_std_threshold),
        grpo_pairwise_ranking_enabled=bool(args.grpo_pairwise_ranking_enabled),
        grpo_pairwise_ranking_weight=float(args.grpo_pairwise_ranking_weight),
        grpo_pairwise_margin=float(args.grpo_pairwise_margin),
        grpo_pairwise_easy_penalty=float(args.grpo_pairwise_easy_penalty),
        proposer_all_easy_rank_spread=float(args.proposer_all_easy_rank_spread),
        gen_step_solver_update_enabled=bool(args.gen_step_solver_update_enabled),
        train_understanding_proposer=bool(args.train_understanding_proposer),
        train_solver=bool(args.train_solver),
        train_generation_proposer=bool(args.train_generation_proposer),
        train_generator=bool(args.train_generator),
        checkpoint_every=int(args.checkpoint_every),
        resume_from=str(args.resume_from),
        save_lora_only=bool(args.save_lora_only),
        understanding_steps_per_cycle=max(0, int(args.understanding_steps_per_cycle)),
        generation_steps_per_cycle=max(0, int(args.generation_steps_per_cycle)),
        replay_buffer_size=max(1, int(args.replay_buffer_size)),
        replay_min_reward=float(args.replay_min_reward),
        replay_max_staleness=max(0, int(args.replay_max_staleness)),
        gen_mix_source_mode=str(args.gen_mix_source_mode),
        generated_mix_dir=str(args.generated_mix_dir or ""),
        generated_mix_min_reward=float(args.generated_mix_min_reward),
        generated_mix_max_files=max(1, int(args.generated_mix_max_files)),
        generated_mix_refresh_every=max(1, int(args.generated_mix_refresh_every)),
        understanding_generated_only=bool(args.understanding_generated_only),
        gen_mix_ratio_start=float(args.gen_mix_ratio_start),
        gen_mix_ratio_max=float(args.gen_mix_ratio_max),
        gen_mix_ratio_warmup_steps=max(1, int(args.gen_mix_ratio_warmup_steps)),
        reward_ema_momentum=float(args.reward_ema_momentum),
        dist_enabled=bool(dist_ctx["enabled"]),
        dist_backend=str(dist_ctx["backend"]),
        dist_world_size=int(dist_ctx["world_size"]),
        dist_rank=int(dist_ctx["rank"]),
        dist_local_rank=int(dist_ctx["local_rank"]),
        dist_main_process=bool(dist_ctx["main_process"]),
        dist_data_shard=bool(args.dist_data_shard),
        )

        os.makedirs(rollout_cfg.output_dir, exist_ok=True)
        runtime = load_bagel_runtime(model_cfg)
        if bool(dist_ctx["enabled"]):
            print(
                "[self_evolving] distributed runtime: "
                f"backend={dist_ctx['backend']} world_size={dist_ctx['world_size']} "
                f"rank={dist_ctx['rank']} local_rank={dist_ctx['local_rank']} "
                f"device={args.device} vae_device={str(args.vae_device or args.device)}"
            )
        exp = rollout_cfg.normalized_experiment_name()
        if exp == "understanding_self_evolving":
            trainer = SelfEvolvingUnderstandingTrainer(runtime=runtime, cfg=rollout_cfg)
        else:
            from train.self_evolving.unified_trainer import UnifiedSelfEvolvingTrainer
            # generation-only convenience mode.
            if exp == "generation_self_evolving":
                rollout_cfg.understanding_steps_per_cycle = 0
                rollout_cfg.generation_steps_per_cycle = max(1, int(rollout_cfg.generation_steps_per_cycle))
            trainer = UnifiedSelfEvolvingTrainer(runtime=runtime, cfg=rollout_cfg)
        run_started_at = float(time.time())
        try:
            summary = trainer.run()
        except Exception as exc:
            # Best-effort fatal status emission for external monitoring tools.
            if hasattr(trainer, "_write_status") and hasattr(trainer, "_progress_core"):
                error_text = f"{type(exc).__name__}: {exc}"
                step_hint = int(getattr(trainer, "start_step", 1)) - 1
                try:
                    progress = trainer._progress_core(  # type: ignore[attr-defined]
                        step=step_hint,
                        phase="failed",
                        run_started_at=run_started_at,
                    )
                except Exception:
                    progress = {
                        "step": int(step_hint),
                        "phase": "failed",
                        "timestamp_unix": float(time.time()),
                    }
                metrics = {
                    "fatal_error": error_text,
                    "fatal_error_type": type(exc).__name__,
                }
                try:
                    if hasattr(trainer, "_append_metrics"):
                        trainer._append_metrics(  # type: ignore[attr-defined]
                            {
                                "kind": "fatal_error",
                                "error": error_text,
                                "traceback": traceback.format_exc(),
                                **progress,
                                **metrics,
                            }
                        )
                except Exception:
                    pass
                try:
                    trainer._write_status(  # type: ignore[attr-defined]
                        state="failed",
                        progress=progress,
                        metrics=metrics,
                        last_error=error_text,
                    )
                except Exception:
                    pass
            raise
        if bool(dist_ctx["main_process"]):
            print("[self_evolving] run summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
    finally:
        _destroy_distributed_runtime()


if __name__ == "__main__":
    main()
