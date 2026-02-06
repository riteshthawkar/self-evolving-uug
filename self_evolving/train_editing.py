"""
Training script for image editing self-evolution.
Phase 5: Anchor-based edit → verify → reward loop.

This is the "MVP" approach from pipeline Appendix B:
1. Sample raw image x (anchor)
2. Proposer: "change attribute A"
3. Generator: produce edited image I'
4. Solver: verify "A changed" + "rest unchanged"
5. GRPO/REINFORCE on generator with combined reward
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
from self_evolving.roles.editor import EditorRole, EditInstruction, EditResult, EditReward
from self_evolving.rl_controller import RLController, RLControllerConfig


@dataclass
class EditTrainingConfig:
    """Configuration for edit self-evolution training."""
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
    learning_rate: float = 1e-5
    
    # KL regularization
    kl_coefficient: float = 0.01
    kl_target: float = 1.0
    kl_adaptive: bool = True
    
    # Editing
    n_edit_samples: int = 4  # Images per edit instruction
    edit_temperature: float = 0.7
    n_verify_samples: int = 5  # Solver samples per verification
    
    # Rewards
    edit_weight: float = 0.5
    preserve_weight: float = 0.5
    use_perceptual: bool = False
    
    # Generation
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    image_size: int = 512
    
    # Output
    output_dir: str = "./outputs_edit"
    save_every: int = 500
    log_every: int = 10
    
    # Debug
    debug: bool = False
    cuda_device: int = 0


def setup_model(config: EditTrainingConfig):
    """Load BLIP3o-NEXT model with LoRA adapters."""
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    
    print(f"Loading model: {config.model_name}")
    
    device = torch.device(f"cuda:{config.cuda_device}" if torch.cuda.is_available() else "cpu")
    
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
    
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


def train_editing(config: EditTrainingConfig):
    """
    Main training loop for edit self-evolution.
    
    This implements the MVP approach from pipeline Appendix B:
    - Simple anchor → edit → verify loop
    - Focus on training generator with editing supervision
    """
    # Setup
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load model
    model, processor, device = setup_model(config)
    
    # Create editor role (includes solver and generator)
    editor = EditorRole(
        model=model,
        processor=processor,
        device=str(device),
        freeze_diffusion=True,
    )
    
    # Create edit reward function
    edit_reward = EditReward(
        edit_weight=config.edit_weight,
        preserve_weight=config.preserve_weight,
        use_perceptual=config.use_perceptual,
    )
    
    # RL Controller
    rl_config = RLControllerConfig(
        kl_coeff=config.kl_coefficient,
        kl_target=config.kl_target,
        kl_adaptive=config.kl_adaptive,
    )
    rl_controller = RLController(model, rl_config)
    
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
    
    # Metrics
    total_steps = 0
    metrics = {
        'edit_success': [],
        'preservation': [],
        'combined_reward': [],
    }
    
    print(f"\n=== Starting Edit Self-Evolution Training ===")
    print(f"  Images: {len(image_pool)}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Edit samples: {config.n_edit_samples}")
    print(f"  Epochs: {config.num_epochs}")
    
    for epoch in range(config.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{config.num_epochs} ---")
        
        for batch_idx, (images, metas) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            
            for img, meta in zip(images, metas):
                # Run edit step
                result = editor.edit_step(
                    anchor_image=img,
                    n_edit_samples=config.n_edit_samples,
                    edit_weight=config.edit_weight,
                    preserve_weight=config.preserve_weight,
                    temperature=config.edit_temperature,
                )
                
                if result is None:
                    continue
                
                # Record metrics
                metrics['edit_success'].append(result.edit_success_score)
                metrics['preservation'].append(result.preservation_score)
                metrics['combined_reward'].append(result.combined_reward)
                
                total_steps += 1
                
                # Logging
                if total_steps % config.log_every == 0:
                    recent_edit = metrics['edit_success'][-10:]
                    recent_pres = metrics['preservation'][-10:]
                    recent_comb = metrics['combined_reward'][-10:]
                    
                    print(f"\n  Step {total_steps}: "
                          f"Edit={sum(recent_edit)/len(recent_edit):.3f} "
                          f"Preserve={sum(recent_pres)/len(recent_pres):.3f} "
                          f"Combined={sum(recent_comb)/len(recent_comb):.3f}")
                    
                    if result.instruction:
                        print(f"    Last edit: {result.instruction.instruction}")
                
                # Save checkpoint
                if config.save_every > 0 and total_steps % config.save_every == 0:
                    save_path = os.path.join(config.output_dir, f"checkpoint_{total_steps}")
                    model.save_pretrained(save_path)
                    
                    # Save example edit
                    example_path = os.path.join(config.output_dir, f"edit_example_{total_steps}.png")
                    result.edited_image.save(example_path)
                    
                    print(f"  Saved checkpoint to: {save_path}")
                
                if config.debug and total_steps >= 10:
                    print("\n[DEBUG MODE] Stopping early")
                    break
            
            if config.debug and total_steps >= 10:
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


def main():
    parser = argparse.ArgumentParser(description="Edit self-evolution training")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with anchor images")
    parser.add_argument("--model_name", type=str, default="BLIP3o/BLIP3o-NEXT-4B")
    parser.add_argument("--output_dir", type=str, default="./outputs_edit")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--n_edit_samples", type=int, default=4)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    config = EditTrainingConfig(
        data_dir=args.data_dir,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_images=args.max_images,
        n_edit_samples=args.n_edit_samples,
        cuda_device=args.cuda_device,
        debug=args.debug,
    )
    
    train_editing(config)


if __name__ == "__main__":
    main()
