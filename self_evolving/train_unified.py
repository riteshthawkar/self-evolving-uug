"""
Unified training script for self-evolving BLIP3o-NEXT.
Implements alternating understanding + generation training.
"""

import os
import sys
import argparse
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_evolving.data.image_pool import ImagePool, ImagePoolConfig, BatchDataLoader
from self_evolving.roles.proposer import ProposerRole, VerificationSpec
from self_evolving.roles.solver import SolverRole
from self_evolving.roles.generator import GeneratorRole
from self_evolving.roles.editor import EditorRole, EditReward
from self_evolving.judge import FrozenJudge, CrossPlayEvaluator
from self_evolving.grpo_trainer import InternalRewardConfig, InternalRewardFunction
from self_evolving.rl_controller import RLController, RLControllerConfig, EvoLMMReward


@dataclass
class UnifiedTrainingConfig:
    """Configuration for unified self-evolving training."""
    # Data
    data_dir: str = ""
    max_images: Optional[int] = None
    batch_size: int = 4
    
    # Model
    model_name: str = "BLIP3o/BLIP3o-NEXT-4B"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    
    # Training schedule
    num_epochs: int = 1
    understanding_ratio: int = 3  # Steps per cycle
    generation_ratio: int = 1     # Steps per cycle
    editing_ratio: int = 1        # Steps per cycle (0 to disable)
    
    # Learning rates
    understanding_lr: float = 1e-4
    generation_lr: float = 1e-5
    
    # KL regularization
    kl_coefficient: float = 0.01
    kl_adaptive: bool = True
    kl_target: float = 1.0
    
    # Understanding (Proposer/Solver)
    n_questions: int = 3
    n_answer_samples: int = 5
    proposer_temperature: float = 0.7
    solver_temperature: float = 0.5
    entropy_target: float = 1.0
    
    # Generation
    n_generations: int = 4       # Images per prompt
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    image_size: int = 512
    
    # Internal rewards
    cycle_weight: float = 0.3
    verification_weight: float = 0.4
    diversity_weight: float = 0.2
    use_frozen_judge: bool = True
    judge_update_every: int = 100
    
    # Editing
    edit_weight: float = 0.5
    preserve_weight: float = 0.5
    n_edit_samples: int = 4
    
    # Cross-play
    use_cross_play: bool = True
    snapshot_every: int = 500
    
    # Output
    output_dir: str = "./outputs_unified"
    save_every: int = 500
    log_every: int = 10
    eval_every: int = 100
    
    # Debug
    debug: bool = False
    cuda_device: int = 0


def setup_model(config: UnifiedTrainingConfig):
    """Load BLIP3o-NEXT model with LoRA adapters."""
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    
    print(f"Loading model: {config.model_name}")
    
    # Set device
    device = torch.device(f"cuda:{config.cuda_device}" if torch.cuda.is_available() else "cpu")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": config.cuda_device},
        trust_remote_code=True,
    )
    
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, processor, device


def understanding_step(
    proposer: ProposerRole,
    solver: SolverRole,
    image,
    config: UnifiedTrainingConfig,
) -> dict:
    """
    Run one understanding step (proposer + solver).
    
    Returns:
        Dict with rewards and metrics
    """
    # Proposer generates questions
    questions_list = proposer.propose_questions(
        image=image,
        n_questions=config.n_questions,
        temperature=config.proposer_temperature,
    )
    questions = questions_list[0] if questions_list else []
    
    if not questions:
        return {'proposer_reward': 0.0, 'solver_reward': 0.0, 'entropy': 0.0}
    
    # Solver answers questions
    solver_rewards = []
    entropies = []
    
    for question in questions:
        answers, agreement = solver.solve(
            image=image,
            question=question,
            n_samples=config.n_answer_samples,
            temperature=config.solver_temperature,
        )
        entropy = solver.compute_entropy(answers)
        solver_rewards.append(agreement)
        entropies.append(entropy)
    
    # Compute rewards
    mean_solver_reward = sum(solver_rewards) / len(solver_rewards)
    mean_entropy = sum(entropies) / len(entropies)
    
    proposer_reward = proposer.compute_entropy_reward(
        questions=questions,
        solver_entropy=mean_entropy,
        target_entropy=config.entropy_target,
    )
    
    return {
        'proposer_reward': proposer_reward,
        'solver_reward': mean_solver_reward,
        'entropy': mean_entropy,
        'n_questions': len(questions),
    }


def generation_step(
    proposer: ProposerRole,
    generator: GeneratorRole,
    internal_reward: InternalRewardFunction,
    image,
    config: UnifiedTrainingConfig,
) -> dict:
    """
    Run one generation step.
    
    1. Proposer generates creative prompt + spec from image
    2. Generator creates images
    3. Internal reward evaluates images
    
    Returns:
        Dict with rewards and metrics
    """
    # Proposer generates prompt and spec
    results = proposer.propose_creative_prompt(
        image=image,
        temperature=config.proposer_temperature,
    )
    
    if not results:
        return {'generation_reward': 0.0, 'prompt': ''}
    
    prompt, spec = results[0]
    
    if not prompt:
        return {'generation_reward': 0.0, 'prompt': ''}
    
    # Generator creates images
    generated_images = generator.generate(
        prompt=prompt,
        n_samples=config.n_generations,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        height=config.image_size,
        width=config.image_size,
    )
    
    if not generated_images:
        return {'generation_reward': 0.0, 'prompt': prompt}
    
    # Compute internal rewards
    rewards = []
    for gen_image in generated_images:
        reward = internal_reward._compute_single_reward(prompt, gen_image, spec)
        rewards.append(reward)
    
    # Diversity bonus
    if len(generated_images) > 1:
        diversity = internal_reward.compute_batch_diversity(generated_images)
    else:
        diversity = 0.5
    
    mean_reward = sum(rewards) / len(rewards)
    best_reward = max(rewards)
    
    return {
        'generation_reward': mean_reward,
        'best_reward': best_reward,
        'diversity': diversity,
        'prompt': prompt,
        'n_generated': len(generated_images),
    }


def editing_step(
    editor: EditorRole,
    image,
    config: UnifiedTrainingConfig,
) -> dict:
    """
    Run one editing step (anchor-based self-play).
    
    1. Proposer generates edit instruction from anchor
    2. Generator produces edited image
    3. Solver verifies edit success + preservation
    
    Returns:
        Dict with rewards and metrics
    """
    result = editor.edit_step(
        anchor_image=image,
        n_edit_samples=config.n_edit_samples,
        edit_weight=config.edit_weight,
        preserve_weight=config.preserve_weight,
        temperature=config.proposer_temperature,
    )
    
    if result is None:
        return {
            'edit_success': 0.0,
            'preservation': 0.0,
            'edit_reward': 0.0,
            'instruction': '',
        }
    
    return {
        'edit_success': result.edit_success_score,
        'preservation': result.preservation_score,
        'edit_reward': result.combined_reward,
        'instruction': result.instruction.instruction if result.instruction else '',
    }

def train_unified(config: UnifiedTrainingConfig):
    """
    Main unified training loop.
    
    Alternates between understanding and generation steps.
    """
    # Setup
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load model
    model, processor, device = setup_model(config)
    
    # Create roles
    proposer = ProposerRole(model, processor, device=str(device))
    solver = SolverRole(model, processor, device=str(device))
    generator = GeneratorRole(model, processor, device=str(device), freeze_diffusion=True)
    
    # Create editor role (uses solver and generator internally)
    editor = EditorRole(
        model=model,
        processor=processor,
        solver=solver,
        generator=generator,
        device=str(device),
    )
    
    # Create internal reward function
    internal_reward_config = InternalRewardConfig(
        cycle_weight=config.cycle_weight,
        verification_weight=config.verification_weight,
        diversity_weight=config.diversity_weight,
        use_frozen_judge=config.use_frozen_judge,
    )
    internal_reward = InternalRewardFunction(
        model=model,
        processor=processor,
        config=internal_reward_config,
    )
    
    # Load data
    print(f"Loading images from: {config.data_dir}")
    pool_config = ImagePoolConfig(
        data_dir=config.data_dir,
        max_images=config.max_images,
    )
    image_pool = ImagePool(pool_config)
    dataloader = BatchDataLoader(image_pool, batch_size=config.batch_size)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.understanding_lr,
    )
    
    # RL Controller (reference policy, adaptive β, EMA baselines)
    rl_config = RLControllerConfig(
        kl_coeff=config.kl_coefficient,
        kl_target=config.kl_target,
        kl_adaptive=config.kl_adaptive,
        proposer_update_every=5,  # EvoLMM schedule
    )
    rl_controller = RLController(model, rl_config)
    
    # EvoLMM-style reward computer
    # EvoLMM-style reward computer
    evlomm_reward = EvoLMMReward(
        entropy_target=config.entropy_target,
        entropy_sigma=0.5,
    )
    
    # Cross-play evaluator for anti-reward-hacking
    cross_play = None
    if config.use_cross_play:
        cross_play = CrossPlayEvaluator(
            snapshots_dir=os.path.join(config.output_dir, "snapshots"),
            max_snapshots=5,
            divergence_threshold=0.3,
        )
    
    # Training metrics
    total_steps = 0
    # 3-phase cycle: understanding -> generation -> editing
    cycle_period = config.understanding_ratio + config.generation_ratio + config.editing_ratio
    
    metrics = {
        'proposer_reward': [],
        'solver_reward': [],
        'generation_reward': [],
        'diversity': [],
        'edit_success': [],
        'preservation': [],
        'edit_reward': [],
    }
    
    print(f"\n=== Starting Unified 3-Phase Training ===")
    print(f"  Images: {len(image_pool)}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Schedule per cycle: {config.understanding_ratio} U + {config.generation_ratio} G + {config.editing_ratio} E")
    print(f"  Cross-play: {config.use_cross_play}")
    print(f"  Epochs: {config.num_epochs}")
    
    for epoch in range(config.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{config.num_epochs} ---")
        
        for batch_idx, (images, metas) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            
            for img, meta in zip(images, metas):
                # Determine phase (3-phase scheduling)
                step_in_cycle = total_steps % cycle_period
                u_end = config.understanding_ratio
                g_end = u_end + config.generation_ratio
                
                if step_in_cycle < u_end:
                    # Phase U: Understanding step
                    phase = 'U'
                    result = understanding_step(proposer, solver, img, config)
                    metrics['proposer_reward'].append(result['proposer_reward'])
                    metrics['solver_reward'].append(result['solver_reward'])
                elif step_in_cycle < g_end:
                    # Phase G: Generation step
                    phase = 'G'
                    result = generation_step(
                        proposer, generator, internal_reward, img, config
                    )
                    metrics['generation_reward'].append(result['generation_reward'])
                    if 'diversity' in result:
                        metrics['diversity'].append(result['diversity'])
                else:
                    # Phase E: Editing step
                    phase = 'E'
                    result = editing_step(editor, img, config)
                    metrics['edit_success'].append(result['edit_success'])
                    metrics['preservation'].append(result['preservation'])
                    metrics['edit_reward'].append(result['edit_reward'])
                
                total_steps += 1
                
                # Logging
                if total_steps % config.log_every == 0:
                    log_str = f"\n  Step {total_steps} [{phase}]: "
                    if metrics['proposer_reward']:
                        log_str += f"P_R={sum(metrics['proposer_reward'][-10:])/min(10,len(metrics['proposer_reward'])):.3f} "
                    if metrics['solver_reward']:
                        log_str += f"S_R={sum(metrics['solver_reward'][-10:])/min(10,len(metrics['solver_reward'])):.3f} "
                    if metrics['generation_reward']:
                        log_str += f"G_R={sum(metrics['generation_reward'][-10:])/min(10,len(metrics['generation_reward'])):.3f} "
                    if metrics['edit_reward']:
                        log_str += f"E_R={sum(metrics['edit_reward'][-10:])/min(10,len(metrics['edit_reward'])):.3f} "
                    if metrics['diversity']:
                        log_str += f"Div={sum(metrics['diversity'][-10:])/min(10,len(metrics['diversity'])):.3f}"
                    print(log_str)
                
                # Update frozen judge
                if config.use_frozen_judge and total_steps % config.judge_update_every == 0:
                    internal_reward.update_frozen_judge()
                    print(f"  [Step {total_steps}] Updated frozen judge")
                
                # Cross-play snapshot
                if cross_play and total_steps % config.snapshot_every == 0:
                    cross_play.add_snapshot(solver, total_steps)
                    print(f"  [Step {total_steps}] Saved cross-play snapshot")
                
                # Save checkpoint
                if config.save_every > 0 and total_steps % config.save_every == 0:
                    save_path = os.path.join(config.output_dir, f"checkpoint_{total_steps}")
                    model.save_pretrained(save_path)
                    print(f"  Saved checkpoint to: {save_path}")
                
                if config.debug and total_steps >= 20:
                    print("\n[DEBUG MODE] Stopping early after 20 steps")
                    break
            
            if config.debug and total_steps >= 20:
                break
        
        # Epoch summary
        print(f"\n=== Epoch {epoch + 1} Complete ===")
        for key, values in metrics.items():
            if values:
                print(f"  Avg {key}: {sum(values)/len(values):.4f}")
    
    # Final save
    final_path = os.path.join(config.output_dir, "final")
    model.save_pretrained(final_path)
    print(f"\n=== Training Complete ===")
    print(f"Final model saved to: {final_path}")
    
    # Log reward stats
    reward_stats = internal_reward.get_stats()
    print(f"Internal reward stats: {reward_stats}")


def main():
    parser = argparse.ArgumentParser(description="Unified self-evolving training")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with training images")
    parser.add_argument("--model_name", type=str, default="BLIP3o/BLIP3o-NEXT-4B")
    parser.add_argument("--output_dir", type=str, default="./outputs_unified")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--understanding_ratio", type=int, default=3)
    parser.add_argument("--generation_ratio", type=int, default=1)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    config = UnifiedTrainingConfig(
        data_dir=args.data_dir,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_images=args.max_images,
        understanding_ratio=args.understanding_ratio,
        generation_ratio=args.generation_ratio,
        cuda_device=args.cuda_device,
        debug=args.debug,
    )
    
    train_unified(config)


if __name__ == "__main__":
    main()
