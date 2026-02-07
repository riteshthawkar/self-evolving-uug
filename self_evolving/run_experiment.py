"""
Unified experiment entrypoint.

This keeps all experiment modes under a single reproducible CLI surface.
Currently implemented:
- understanding_self_evolving

Planned modes:
- generation_self_evolving
- unified_self_evolving
- rl_no_self_evolving
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


EXPERIMENT_CHOICES = (
    "understanding_self_evolving",
    "generation_self_evolving",
    "unified_self_evolving",
    "rl_no_self_evolving",
)


def _parse_subfolders(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    if not value:
        return None
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    return names or None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified self-evolving experiment runner")

    # Core
    p.add_argument("--experiment", type=str, required=True, choices=EXPERIMENT_CHOICES)
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--data_split", type=str, default="train", choices=["train", "val", "test", "all"])
    p.add_argument("--output_dir", type=str, default="./runs")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--include_subfolders", type=str, default=None)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--non_deterministic", dest="deterministic", action="store_false")

    # Device
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--device_map", type=str, default="single", choices=["single", "auto", "cpu"])

    # Train loop
    p.add_argument("--total_steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--proposer_update_freq", type=int, default=5)

    # Decoding
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_new_tokens_solver", type=int, default=128)
    p.add_argument("--max_new_tokens_proposer", type=int, default=128)
    p.add_argument("--num_solver_samples", type=int, default=5)

    # Reward shaping
    p.add_argument("--solver_soft_gamma", type=float, default=0.7)
    p.add_argument("--len_penalty_weight", type=float, default=0.10)
    p.add_argument("--len_penalty_target_words", type=int, default=6)
    p.add_argument("--prop_entropy_mu", type=float, default=0.90)
    p.add_argument("--prop_entropy_sigma", type=float, default=0.35)

    # KL
    p.add_argument("--kl_coef", type=float, default=1e-3)
    p.add_argument("--kl_target", type=float, default=0.02)
    p.add_argument("--kl_adapt_rate", type=float, default=0.10)
    p.add_argument("--kl_min", type=float, default=1e-8)
    p.add_argument("--kl_max", type=float, default=1e2)
    p.add_argument("--baseline_momentum", type=float, default=0.9)

    # LoRA
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--no_lora", dest="use_lora", action="store_false")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,mm_projector",
    )

    # Logging + checkpoints
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--max_checkpoints", type=int, default=3)
    p.add_argument("--clear_cache_every", type=int, default=25)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--start_step", type=int, default=0)

    # W&B
    p.add_argument(
        "--wandb_mode",
        type=str,
        default=os.environ.get("WANDB_MODE", "disabled"),
        choices=["online", "offline", "disabled"],
    )
    p.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "self-evolving-uug"))
    p.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", None))
    p.add_argument("--wandb_run_name", type=str, default=os.environ.get("WANDB_RUN_NAME", None))
    p.add_argument("--wandb_log_images_every", type=int, default=0)

    return p


def run_understanding_self_evolving(args: argparse.Namespace):
    from self_evolving.experiments import (
        UnderstandingSelfEvolvingConfig,
        UnderstandingSelfEvolvingTrainer,
    )

    lora_targets = tuple(x.strip() for x in args.lora_targets.split(",") if x.strip())
    cfg = UnderstandingSelfEvolvingConfig(
        run_name=args.run_name,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        data_split=args.data_split,
        include_subfolders=_parse_subfolders(args.include_subfolders),
        max_images=args.max_images,
        model_name=args.model_name,
        dtype=args.dtype,
        cuda_device=args.cuda_device,
        device_map=args.device_map,
        total_steps=args.total_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        proposer_update_freq=args.proposer_update_freq,
        temp=args.temp,
        top_p=args.top_p,
        max_new_tokens_solver=args.max_new_tokens_solver,
        max_new_tokens_proposer=args.max_new_tokens_proposer,
        num_solver_samples=args.num_solver_samples,
        solver_soft_gamma=args.solver_soft_gamma,
        len_penalty_weight=args.len_penalty_weight,
        len_penalty_target_words=args.len_penalty_target_words,
        prop_entropy_mu=args.prop_entropy_mu,
        prop_entropy_sigma=args.prop_entropy_sigma,
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
        seed=args.seed,
        deterministic=args.deterministic,
        log_every=args.log_every,
        save_every=args.save_every,
        max_checkpoints=args.max_checkpoints,
        clear_cache_every=args.clear_cache_every,
        resume_from=args.resume_from,
        start_step=args.start_step,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_log_images_every=args.wandb_log_images_every,
    )
    trainer = UnderstandingSelfEvolvingTrainer(cfg)
    trainer.train()


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.experiment == "understanding_self_evolving":
        run_understanding_self_evolving(args)
        return

    raise NotImplementedError(
        f"Experiment mode '{args.experiment}' is not implemented yet in the unified runner. "
        "Use 'understanding_self_evolving' for now."
    )


if __name__ == "__main__":
    main()
