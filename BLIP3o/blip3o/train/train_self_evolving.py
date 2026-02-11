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


def _parse_subfolders(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    if not value:
        return None
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    return names or None


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
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--proposer_update_freq", type=int, default=5)
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
    p.add_argument("--num_solver_samples", type=int, default=5)
    p.add_argument("--num_solver_samples_spec", type=int, default=3)
    p.add_argument("--num_generations", type=int, default=4)

    # Reward shaping
    p.add_argument("--solver_soft_gamma", type=float, default=0.7)
    p.add_argument("--solver_use_temperature_mix", action="store_true", default=True)
    p.add_argument("--disable_solver_temperature_mix", dest="solver_use_temperature_mix", action="store_false")
    p.add_argument("--solver_temp_min", type=float, default=0.7)
    p.add_argument("--solver_temp_max", type=float, default=1.3)
    p.add_argument("--sc_entropy_min", type=float, default=0.15)
    p.add_argument("--sc_entropy_max", type=float, default=1.2)
    p.add_argument("--sc_margin_max", type=float, default=0.90)
    p.add_argument("--sc_negative_weight", type=float, default=0.25)
    p.add_argument("--skip_solver_update_when_uninformative", action="store_true", default=True)
    p.add_argument(
        "--allow_solver_update_when_uninformative",
        dest="skip_solver_update_when_uninformative",
        action="store_false",
    )
    p.add_argument("--len_penalty_weight", type=float, default=0.10)
    p.add_argument("--len_penalty_target_words", type=int, default=6)
    p.add_argument("--prop_entropy_mu", type=float, default=0.90)
    p.add_argument("--prop_entropy_sigma", type=float, default=0.35)
    p.add_argument("--adaptive_prop_entropy_target", action="store_true", default=True)
    p.add_argument("--fixed_prop_entropy_target", dest="adaptive_prop_entropy_target", action="store_false")
    p.add_argument("--prop_entropy_ema_momentum", type=float, default=0.95)
    p.add_argument("--prop_entropy_mu_min", type=float, default=0.05)
    p.add_argument("--prop_entropy_mu_max", type=float, default=1.5)
    p.add_argument("--reward_spec_weight", type=float, default=0.65)
    p.add_argument("--reward_cycle_weight", type=float, default=0.20)
    p.add_argument("--reward_diversity_weight", type=float, default=0.10)
    p.add_argument("--reward_contradiction_weight", type=float, default=0.20)

    # KL
    p.add_argument("--kl_coef", type=float, default=0.01)
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
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--max_checkpoints", type=int, default=5)
    p.add_argument("--clear_cache_every", type=int, default=25)
    p.add_argument("--save_generated_images_every", type=int, default=0)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--start_step", type=int, default=0)

    # Generation backend
    p.add_argument("--generation_num_inference_steps", type=int, default=30)
    p.add_argument("--generation_guidance_scale", type=float, default=2.0)
    p.add_argument("--generation_height", type=int, default=1024)
    p.add_argument("--generation_width", type=int, default=1024)
    p.add_argument("--require_decoder_for_blip3o", action="store_true", default=True)
    p.add_argument("--allow_missing_decoder_for_blip3o", dest="require_decoder_for_blip3o", action="store_false")
    p.add_argument("--allow_latent_visualization_fallback", action="store_true", default=False)
    p.add_argument("--strict_require_generation_tokens", dest="strict_require_generation_tokens", action="store_true")
    p.add_argument("--allow_missing_generation_tokens", dest="strict_require_generation_tokens", action="store_false")
    p.set_defaults(strict_require_generation_tokens=True)
    p.add_argument("--generator_missing_trace_strategy", type=str, default="proxy", choices=["proxy", "skip", "error"])
    p.add_argument("--verification_use_reference_solver", action="store_true", default=True)
    p.add_argument("--verification_use_trainable_solver", dest="verification_use_reference_solver", action="store_false")
    p.add_argument("--generator_update_rule", type=str, default="reinforce", choices=["reinforce", "dpo"])
    p.add_argument("--dpo_beta", type=float, default=0.1)
    p.add_argument("--dpo_label_smoothing", type=float, default=0.0)
    p.add_argument("--dpo_min_reward_gap", type=float, default=0.0)
    p.add_argument("--dpo_min_spec_gap", type=float, default=0.0)
    p.add_argument("--dpo_min_confidence_gap", type=float, default=0.0)
    p.add_argument("--dpo_max_contradiction", type=float, default=1.0)
    p.add_argument("--dpo_pair_selection", type=str, default="best_worst", choices=["best_worst", "best_hard_negative"])
    p.add_argument("--generator_proxy_max_ratio", type=float, default=1.0)

    # Unified scheduler
    p.add_argument("--understanding_steps_per_cycle", type=int, default=3)
    p.add_argument("--generation_steps_per_cycle", type=int, default=2)
    p.add_argument("--synthetic_solver_update_freq", type=int, default=1)
    p.add_argument("--synthetic_solver_hard_only", action="store_true", default=False)
    p.add_argument("--solver_hardness_min_entropy", type=float, default=0.2)
    p.add_argument("--min_spec_quality_for_update", type=float, default=0.35)
    p.add_argument("--min_spec_qa_pairs", type=int, default=2)
    p.add_argument("--max_expected_words", type=int, default=8)
    p.add_argument("--max_question_words", type=int, default=24)

    # W&B
    p.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "disabled"), choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "self-evolving-uug"))
    p.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", None))
    p.add_argument("--wandb_run_name", type=str, default=os.environ.get("WANDB_RUN_NAME", None))
    p.add_argument("--wandb_log_images_every", type=int, default=0)

    return p


def _build_understanding_config(args):
    from blip3o.train.self_evolving.config import UnderstandingSelfEvolvingConfig

    lora_targets = tuple(x.strip() for x in args.lora_targets.split(",") if x.strip())
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
        solver_temp_min=args.solver_temp_min,
        solver_temp_max=args.solver_temp_max,
        sc_entropy_min=args.sc_entropy_min,
        sc_entropy_max=args.sc_entropy_max,
        sc_margin_max=args.sc_margin_max,
        sc_negative_weight=args.sc_negative_weight,
        skip_solver_update_when_uninformative=args.skip_solver_update_when_uninformative,
        len_penalty_weight=args.len_penalty_weight,
        len_penalty_target_words=args.len_penalty_target_words,
        prop_entropy_mu=args.prop_entropy_mu,
        prop_entropy_sigma=args.prop_entropy_sigma,
        adaptive_prop_entropy_target=args.adaptive_prop_entropy_target,
        prop_entropy_ema_momentum=args.prop_entropy_ema_momentum,
        prop_entropy_mu_min=args.prop_entropy_mu_min,
        prop_entropy_mu_max=args.prop_entropy_mu_max,
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


def _build_generation_config(args):
    from blip3o.train.self_evolving.config import GenerationSelfEvolvingConfig

    lora_targets = tuple(x.strip() for x in args.lora_targets.split(",") if x.strip())
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
        generator_update_rule=args.generator_update_rule,
        dpo_beta=args.dpo_beta,
        dpo_label_smoothing=args.dpo_label_smoothing,
        dpo_min_reward_gap=args.dpo_min_reward_gap,
        dpo_min_spec_gap=args.dpo_min_spec_gap,
        dpo_min_confidence_gap=args.dpo_min_confidence_gap,
        dpo_max_contradiction=args.dpo_max_contradiction,
        dpo_pair_selection=args.dpo_pair_selection,
        generator_proxy_max_ratio=args.generator_proxy_max_ratio,
        solver_soft_gamma=args.solver_soft_gamma,
        solver_use_temperature_mix=args.solver_use_temperature_mix,
        solver_temp_min=args.solver_temp_min,
        solver_temp_max=args.solver_temp_max,
        sc_entropy_min=args.sc_entropy_min,
        sc_entropy_max=args.sc_entropy_max,
        sc_margin_max=args.sc_margin_max,
        sc_negative_weight=args.sc_negative_weight,
        skip_solver_update_when_uninformative=args.skip_solver_update_when_uninformative,
        len_penalty_weight=args.len_penalty_weight,
        len_penalty_target_words=args.len_penalty_target_words,
        prop_entropy_mu=args.prop_entropy_mu,
        prop_entropy_sigma=args.prop_entropy_sigma,
        adaptive_prop_entropy_target=args.adaptive_prop_entropy_target,
        prop_entropy_ema_momentum=args.prop_entropy_ema_momentum,
        prop_entropy_mu_min=args.prop_entropy_mu_min,
        prop_entropy_mu_max=args.prop_entropy_mu_max,
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
        seed=args.seed,
        deterministic=args.deterministic,
        log_every=args.log_every,
        save_every=args.save_every,
        max_checkpoints=args.max_checkpoints,
        clear_cache_every=args.clear_cache_every,
        save_generated_images_every=args.save_generated_images_every,
        resume_from=args.resume_from,
        start_step=args.start_step,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_log_images_every=args.wandb_log_images_every,
    )


def _build_unified_config(args):
    from blip3o.train.self_evolving.config import UnifiedSelfEvolvingConfig

    lora_targets = tuple(x.strip() for x in args.lora_targets.split(",") if x.strip())
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
        generator_update_rule=args.generator_update_rule,
        dpo_beta=args.dpo_beta,
        dpo_label_smoothing=args.dpo_label_smoothing,
        dpo_min_reward_gap=args.dpo_min_reward_gap,
        dpo_min_spec_gap=args.dpo_min_spec_gap,
        dpo_min_confidence_gap=args.dpo_min_confidence_gap,
        dpo_max_contradiction=args.dpo_max_contradiction,
        dpo_pair_selection=args.dpo_pair_selection,
        generator_proxy_max_ratio=args.generator_proxy_max_ratio,
        solver_soft_gamma=args.solver_soft_gamma,
        solver_use_temperature_mix=args.solver_use_temperature_mix,
        solver_temp_min=args.solver_temp_min,
        solver_temp_max=args.solver_temp_max,
        sc_entropy_min=args.sc_entropy_min,
        sc_entropy_max=args.sc_entropy_max,
        sc_margin_max=args.sc_margin_max,
        sc_negative_weight=args.sc_negative_weight,
        skip_solver_update_when_uninformative=args.skip_solver_update_when_uninformative,
        len_penalty_weight=args.len_penalty_weight,
        len_penalty_target_words=args.len_penalty_target_words,
        prop_entropy_mu=args.prop_entropy_mu,
        prop_entropy_sigma=args.prop_entropy_sigma,
        adaptive_prop_entropy_target=args.adaptive_prop_entropy_target,
        prop_entropy_ema_momentum=args.prop_entropy_ema_momentum,
        prop_entropy_mu_min=args.prop_entropy_mu_min,
        prop_entropy_mu_max=args.prop_entropy_mu_max,
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
        seed=args.seed,
        deterministic=args.deterministic,
        log_every=args.log_every,
        save_every=args.save_every,
        max_checkpoints=args.max_checkpoints,
        clear_cache_every=args.clear_cache_every,
        save_generated_images_every=args.save_generated_images_every,
        resume_from=args.resume_from,
        start_step=args.start_step,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_log_images_every=args.wandb_log_images_every,
        understanding_steps_per_cycle=args.understanding_steps_per_cycle,
        generation_steps_per_cycle=args.generation_steps_per_cycle,
        synthetic_solver_update_freq=args.synthetic_solver_update_freq,
        synthetic_solver_hard_only=args.synthetic_solver_hard_only,
        solver_hardness_min_entropy=args.solver_hardness_min_entropy,
    )


def main():
    parser = _build_parser()
    args = parser.parse_args()

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
