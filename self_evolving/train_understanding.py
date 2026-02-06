"""
Training script for understanding-only self-evolution.
Validates EvoLMM loop on BLIP3o-NEXT backbone.
Phase 1 validation: Proposer + Solver with self-consistency reward.
"""

import os
import sys
import argparse
import torch
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_evolving.data.image_pool import ImagePool, ImagePoolConfig, BatchDataLoader
from self_evolving.roles.proposer import ProposerRole
from self_evolving.roles.solver import SolverRole


@dataclass
class TrainingConfig:
    """Configuration for understanding-only training."""
    # Data
    data_dir: str = ""
    max_images: Optional[int] = None
    batch_size: int = 4
    
    # Model
    model_name: str = "BLIP3o/BLIP3o-NEXT-4B"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    
    # Training
    num_epochs: int = 1
    learning_rate: float = 1e-4
    kl_coefficient: float = 0.01
    kl_adaptive: bool = True
    kl_target: float = 1.0
    
    # Proposer
    n_questions: int = 3
    proposer_temperature: float = 0.7
    
    # Solver
    n_answer_samples: int = 5
    solver_temperature: float = 0.5
    
    # Proposer reward
    entropy_target: float = 1.0
    entropy_sigma: float = 0.5
    
    # Output
    output_dir: str = "./outputs"
    save_every: int = 100
    log_every: int = 10
    
    # Debug
    debug: bool = False


def setup_model(config: TrainingConfig):
    """Load BLIP3o-NEXT model with LoRA adapters."""
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    
    print(f"Loading model: {config.model_name}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if config.use_lora:
        # Setup LoRA for proposer and solver
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
    
    return model, processor


def train_understanding_loop(config: TrainingConfig):
    """
    Main training loop for understanding-only self-evolution.
    
    Implements EvoLMM's propose/solve loop:
    1. Proposer generates questions about image
    2. Solver answers questions multiple times
    3. Solver reward: agreement-based self-consistency
    4. Proposer reward: entropy band (moderately difficult questions)
    5. Policy update with KL regularization
    """
    # Setup
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model, processor = setup_model(config)
    
    # Create reference model for KL
    reference_model = None
    if config.kl_coefficient > 0:
        # Clone initial model state for KL reference
        import copy
        reference_model = copy.deepcopy(model)
        for param in reference_model.parameters():
            param.requires_grad = False
    
    # Create roles
    proposer = ProposerRole(model, processor, lora_adapter_name="default", device=device)
    solver = SolverRole(model, processor, lora_adapter_name="default", device=device)
    
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
        lr=config.learning_rate,
    )
    
    # Training metrics
    total_steps = 0
    running_proposer_reward = 0.0
    running_solver_reward = 0.0
    kl_coeff = config.kl_coefficient
    
    print(f"\nStarting training for {config.num_epochs} epochs...")
    print(f"  Images: {len(image_pool)}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Questions per image: {config.n_questions}")
    print(f"  Answer samples: {config.n_answer_samples}")
    
    for epoch in range(config.num_epochs):
        epoch_proposer_rewards = []
        epoch_solver_rewards = []
        
        for batch_idx, (images, metas) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            
            for img, meta in zip(images, metas):
                # === Proposer: Generate questions ===
                questions_list = proposer.propose_questions(
                    image=img,
                    n_questions=config.n_questions,
                    n_samples=1,
                    temperature=config.proposer_temperature,
                )
                questions = questions_list[0] if questions_list else []
                
                if not questions:
                    continue
                
                # === Solver: Answer questions ===
                solver_rewards = []
                solver_entropies = []
                
                for question in questions:
                    answers, agreement_reward = solver.solve(
                        image=img,
                        question=question,
                        n_samples=config.n_answer_samples,
                        temperature=config.solver_temperature,
                    )
                    
                    entropy = solver.compute_entropy(answers)
                    solver_rewards.append(agreement_reward)
                    solver_entropies.append(entropy)
                
                # === Compute rewards ===
                # Solver reward: mean agreement across questions
                mean_solver_reward = sum(solver_rewards) / len(solver_rewards) if solver_rewards else 0.0
                
                # Proposer reward: entropy band (prefer moderate difficulty)
                mean_entropy = sum(solver_entropies) / len(solver_entropies) if solver_entropies else 0.0
                proposer_reward = proposer.compute_entropy_reward(
                    questions=questions,
                    solver_entropy=mean_entropy,
                    target_entropy=config.entropy_target,
                    sigma=config.entropy_sigma,
                )
                
                # === Policy update (simplified REINFORCE) ===
                # In practice, you'd accumulate gradients over the batch
                # and apply proper policy gradient updates
                
                # For now, just log rewards
                epoch_proposer_rewards.append(proposer_reward)
                epoch_solver_rewards.append(mean_solver_reward)
                
                total_steps += 1
                
                # Logging
                if total_steps % config.log_every == 0:
                    avg_p = sum(epoch_proposer_rewards[-config.log_every:]) / config.log_every
                    avg_s = sum(epoch_solver_rewards[-config.log_every:]) / config.log_every
                    print(f"\n  Step {total_steps}: Proposer R={avg_p:.3f}, Solver R={avg_s:.3f}, "
                          f"Entropy={mean_entropy:.3f}")
                
                # Save checkpoint
                if config.save_every > 0 and total_steps % config.save_every == 0:
                    save_path = os.path.join(config.output_dir, f"checkpoint_{total_steps}")
                    model.save_pretrained(save_path)
                    print(f"\n  Saved checkpoint to: {save_path}")
                
                if config.debug and total_steps >= 10:
                    break
            
            if config.debug and total_steps >= 10:
                break
        
        # Epoch summary
        epoch_avg_p = sum(epoch_proposer_rewards) / len(epoch_proposer_rewards) if epoch_proposer_rewards else 0
        epoch_avg_s = sum(epoch_solver_rewards) / len(epoch_solver_rewards) if epoch_solver_rewards else 0
        print(f"\nEpoch {epoch+1} complete:")
        print(f"  Avg Proposer Reward: {epoch_avg_p:.4f}")
        print(f"  Avg Solver Reward: {epoch_avg_s:.4f}")
    
    # Final save
    final_path = os.path.join(config.output_dir, "final")
    model.save_pretrained(final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Self-evolving understanding training")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with training images")
    parser.add_argument("--model_name", type=str, default="BLIP3o/BLIP3o-NEXT-4B")
    parser.add_argument("--output_dir", type=str, default="./outputs_understanding")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        data_dir=args.data_dir,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_images=args.max_images,
        debug=args.debug,
    )
    
    train_understanding_loop(config)


if __name__ == "__main__":
    main()
