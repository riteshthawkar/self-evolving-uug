# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import random
import sys

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

    # Model/runtime
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--vae_device", type=str, default="")
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
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)

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

    # SUDER-style generation phase (proposer joint reward logging).
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
    p.add_argument("--generation_cfg_text_scale", type=float, default=4.0)
    p.add_argument("--generation_cfg_img_scale", type=float, default=1.5)
    p.add_argument("--generation_num_timesteps", type=int, default=50)
    p.add_argument("--generation_timestep_shift", type=float, default=3.0)
    p.add_argument("--generation_image_size", type=int, default=1024)
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
    p.add_argument("--policy_update_method", type=str, default="reinforce", choices=["reinforce", "grpo"])
    p.add_argument("--policy_use_bf16", action="store_true", default=True)
    p.add_argument(
        "--disable_policy_use_bf16",
        dest="policy_use_bf16",
        action="store_false",
    )
    p.add_argument("--policy_lr", type=float, default=2e-5)
    p.add_argument("--policy_weight_decay", type=float, default=0.0)
    p.add_argument("--policy_max_grad_norm", type=float, default=1.0)
    p.add_argument("--policy_grad_accum_steps", type=int, default=1)
    p.add_argument("--policy_reward_scale", type=float, default=1.0)
    p.add_argument("--baseline_momentum", type=float, default=0.9)
    p.add_argument("--grpo_eps", type=float, default=1e-6)
    p.add_argument("--solver_reward_mix_gamma", type=float, default=0.7)
    p.add_argument("--solver_skip_easy_updates", action="store_true", default=True)
    p.add_argument(
        "--disable_solver_skip_easy_updates",
        dest="solver_skip_easy_updates",
        action="store_false",
    )
    p.add_argument("--solver_easy_update_majority_threshold", type=float, default=0.98)
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
    p.add_argument("--checkpoint_every", type=int, default=100)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--save_lora_only", action="store_true", default=True)
    p.add_argument(
        "--disable_save_lora_only",
        dest="save_lora_only",
        action="store_false",
    )
    return p


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    args = _build_parser().parse_args()
    if bool(args.policy_updates_enabled) and not bool(args.enable_lora):
        raise ValueError("--policy_updates_enabled requires --enable_lora for trainable role adapters.")
    _set_seed(int(args.seed))
    # Delay heavy imports (model + cv2 dependencies) until actual execution.
    from train.self_evolving.model_loader import load_bagel_runtime
    from train.self_evolving.trainer import SelfEvolvingUnderstandingTrainer

    model_cfg = ModelLoadConfig(
        model_path=args.model_path,
        device=args.device,
        vae_device=str(args.vae_device or ""),
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
        generation_cfg_text_scale=float(args.generation_cfg_text_scale),
        generation_cfg_img_scale=float(args.generation_cfg_img_scale),
        generation_num_timesteps=int(args.generation_num_timesteps),
        generation_timestep_shift=float(args.generation_timestep_shift),
        generation_image_size=int(args.generation_image_size),
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
        solver_reward_mix_gamma=float(args.solver_reward_mix_gamma),
        solver_skip_easy_updates=bool(args.solver_skip_easy_updates),
        solver_easy_update_majority_threshold=float(args.solver_easy_update_majority_threshold),
        train_understanding_proposer=bool(args.train_understanding_proposer),
        train_solver=bool(args.train_solver),
        train_generation_proposer=bool(args.train_generation_proposer),
        checkpoint_every=int(args.checkpoint_every),
        resume_from=str(args.resume_from),
        save_lora_only=bool(args.save_lora_only),
    )

    os.makedirs(rollout_cfg.output_dir, exist_ok=True)
    runtime = load_bagel_runtime(model_cfg)
    trainer = SelfEvolvingUnderstandingTrainer(runtime=runtime, cfg=rollout_cfg)
    summary = trainer.run()
    print("[self_evolving] run summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
