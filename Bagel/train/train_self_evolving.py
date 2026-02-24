# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

# Allow direct execution via `python train/train_self_evolving.py`.
if __package__ is None or __package__ == "":
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from train.self_evolving.config import ModelLoadConfig, RolloutConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BAGEL self-evolving framework (phase-1 understanding rollout)."
    )

    # Model/runtime
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_latent_size", type=int, default=64)

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
    _set_seed(int(args.seed))
    # Delay heavy imports (model + cv2 dependencies) until actual execution.
    from train.self_evolving.model_loader import load_bagel_runtime
    from train.self_evolving.trainer import SelfEvolvingUnderstandingTrainer

    model_cfg = ModelLoadConfig(
        model_path=args.model_path,
        device=args.device,
        max_latent_size=int(args.max_latent_size),
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
