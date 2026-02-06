"""
SelfEvolvingGRPOTrainer: Extended GRPO trainer with internal reward functions.
Replaces external rewards (PaddleOCR, GenEval) with self-consistency based internal rewards.
"""

import os
import sys
import copy
import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any, Callable
from dataclasses import dataclass, field
from PIL import Image

# Add BLIP3o path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'BLIP3o'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'BLIP3o', 'trl'))

from self_evolving.rewards.cycle_consistency import CycleConsistencyReward
from self_evolving.rewards.verification import VerificationReward
from self_evolving.rewards.diversity import DiversityReward
from self_evolving.judge import FrozenJudge
from self_evolving.roles.solver import SolverRole


@dataclass
class InternalRewardConfig:
    """Configuration for internal reward functions."""
    # Reward weights
    cycle_weight: float = 0.3
    verification_weight: float = 0.4
    diversity_weight: float = 0.2
    anti_collusion_weight: float = 0.1
    
    # Cycle consistency
    cycle_n_samples: int = 5
    
    # Verification
    verify_n_samples: int = 5
    use_frozen_judge: bool = True
    judge_ema_decay: float = 0.99
    
    # Diversity
    min_diversity: float = 0.1
    target_diversity: float = 0.5
    
    # Anti-collusion
    use_cross_play: bool = True
    cross_play_aggregation: str = "min"  # min, median, mean


class InternalRewardFunction:
    """
    Self-consistency based internal reward function for GRPO.
    
    Replaces external evaluators (PaddleOCR, GenEval) with:
    1. Cycle consistency: prompt → image → caption → compare
    2. Verification: answer spec questions about generated image
    3. Diversity: prevent mode collapse
    4. Anti-collusion: frozen judge, cross-play
    """
    
    def __init__(
        self,
        model,
        processor,
        config: Optional[InternalRewardConfig] = None,
        vision_encoder=None,
        vision_processor=None,
    ):
        """
        Initialize internal reward function.
        
        Args:
            model: BLIP3o-NEXT model for captioning and verification
            processor: Processor for model
            config: Reward configuration
            vision_encoder: Optional frozen vision encoder for diversity
            vision_processor: Optional vision processor
        """
        self.model = model
        self.processor = processor
        self.config = config or InternalRewardConfig()
        
        # Initialize reward components
        self.cycle_reward = CycleConsistencyReward(
            model=model,
            processor=processor,
            n_samples=self.config.cycle_n_samples,
        )
        
        # Solver for verification
        self.solver = SolverRole(
            model=model,
            processor=processor,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        self.verification_reward = VerificationReward(
            solver=self.solver,
            n_samples=self.config.verify_n_samples,
            use_frozen_judge=self.config.use_frozen_judge,
            ema_decay=self.config.judge_ema_decay,
        )
        
        # Diversity reward (optional, needs vision encoder)
        if vision_encoder is not None:
            self.diversity_reward = DiversityReward(
                vision_encoder=vision_encoder,
                processor=vision_processor,
                min_diversity=self.config.min_diversity,
                target_diversity=self.config.target_diversity,
            )
        else:
            self.diversity_reward = None
        
        # Track metrics for logging
        self.reward_stats = {
            'cycle': [],
            'verification': [],
            'diversity': [],
            'combined': [],
        }

    def __call__(
        self,
        prompts: List[str],
        images: List[Image.Image],
        specs: Optional[List[Any]] = None,
    ) -> torch.Tensor:
        """
        Compute internal rewards for generated images.
        
        Args:
            prompts: List of generation prompts
            images: List of generated images
            specs: Optional verification specs
            
        Returns:
            Tensor of rewards [batch_size]
        """
        device = next(self.model.parameters()).device
        batch_size = len(prompts)
        rewards = torch.zeros(batch_size, device=device)
        
        for i, (prompt, image) in enumerate(zip(prompts, images)):
            reward = self._compute_single_reward(prompt, image, specs[i] if specs else None)
            rewards[i] = reward
        
        return rewards

    def _compute_single_reward(
        self,
        prompt: str,
        image: Image.Image,
        spec: Optional[Any] = None,
    ) -> float:
        """Compute reward for a single prompt-image pair."""
        rewards = {}
        
        # 1. Cycle consistency
        try:
            cycle_r = self.cycle_reward(prompt, image)
            rewards['cycle'] = cycle_r
            self.reward_stats['cycle'].append(cycle_r)
        except Exception as e:
            print(f"Cycle reward error: {e}")
            rewards['cycle'] = 0.0
        
        # 2. Verification (if spec provided)
        if spec is not None:
            try:
                verify_r, _ = self.verification_reward(image, spec)
                rewards['verification'] = verify_r
                self.reward_stats['verification'].append(verify_r)
            except Exception as e:
                print(f"Verification reward error: {e}")
                rewards['verification'] = 0.0
        else:
            # Generate basic verification questions from prompt
            rewards['verification'] = self._verify_from_prompt(prompt, image)
        
        # 3. Diversity (computed at batch level, so skip here)
        rewards['diversity'] = 0.5  # Placeholder, computed at batch level
        
        # Combine rewards
        combined = (
            self.config.cycle_weight * rewards['cycle'] +
            self.config.verification_weight * rewards['verification'] +
            self.config.diversity_weight * rewards['diversity']
        )
        
        self.reward_stats['combined'].append(combined)
        return combined

    def _verify_from_prompt(self, prompt: str, image: Image.Image) -> float:
        """Generate verification questions from prompt and answer them."""
        # Extract key attributes from prompt
        questions = self._generate_questions_from_prompt(prompt)
        
        if not questions:
            return 0.5  # Neutral reward if no questions
        
        # Answer each question
        scores = []
        for question in questions:
            answers, agreement = self.solver.solve(image, question, n_samples=3)
            scores.append(agreement)
        
        return sum(scores) / len(scores) if scores else 0.5

    def _generate_questions_from_prompt(self, prompt: str) -> List[str]:
        """Extract verification questions from a prompt."""
        questions = []
        
        # Basic heuristics to extract testable attributes
        prompt_lower = prompt.lower()
        
        # Color mentions
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange', 'pink']
        for color in colors:
            if color in prompt_lower:
                questions.append(f"Is there something {color} in this image?")
        
        # Number mentions
        import re
        numbers = re.findall(r'\b(one|two|three|four|five|\d+)\b', prompt_lower)
        if numbers:
            questions.append(f"How many main objects are in this image?")
        
        # Object types (simple extraction)
        common_objects = ['car', 'dog', 'cat', 'person', 'tree', 'house', 'sky', 'water', 'bird']
        for obj in common_objects:
            if obj in prompt_lower:
                questions.append(f"Is there a {obj} in this image?")
        
        return questions[:3]  # Limit to 3 questions

    def compute_batch_diversity(self, images: List[Image.Image]) -> float:
        """Compute diversity across a batch of images."""
        if self.diversity_reward is None or len(images) < 2:
            return 0.5
        
        try:
            diversity = self.diversity_reward(images)
            self.reward_stats['diversity'].append(diversity)
            return diversity
        except Exception as e:
            print(f"Diversity reward error: {e}")
            return 0.5

    def update_frozen_judge(self):
        """Update the frozen judge with current solver weights."""
        self.verification_reward.update_frozen_judge()

    def get_stats(self) -> Dict[str, float]:
        """Get average reward statistics."""
        stats = {}
        for key, values in self.reward_stats.items():
            if values:
                stats[f"reward/{key}_mean"] = sum(values) / len(values)
                stats[f"reward/{key}_std"] = np.std(values).item() if len(values) > 1 else 0.0
        return stats

    def reset_stats(self):
        """Reset reward statistics."""
        for key in self.reward_stats:
            self.reward_stats[key] = []


class SelfEvolvingGRPOTrainer:
    """
    Extended GRPO trainer with internal reward functions.
    
    This class wraps the existing GRPOTrainer and replaces the external
    reward computation with internal self-consistency based rewards.
    
    Key changes from standard GRPO:
    1. Replace `_calculate_rewards` with internal reward function
    2. Add frozen judge for anti-reward-hacking
    3. Support generation + understanding alternating training
    4. Add proposer curriculum (entropy band)
    """
    
    def __init__(
        self,
        model,
        processor,
        train_dataset=None,
        eval_dataset=None,
        internal_reward_config: Optional[InternalRewardConfig] = None,
        grpo_config=None,
        peft_config=None,
        vision_encoder=None,
        vision_processor=None,
    ):
        """
        Initialize SelfEvolvingGRPOTrainer.
        
        Args:
            model: BLIP3o-NEXT model
            processor: Processor for model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            internal_reward_config: Config for internal rewards
            grpo_config: GRPO training config
            peft_config: PEFT config for LoRA
            vision_encoder: Optional vision encoder for diversity
            vision_processor: Optional vision processor
        """
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.grpo_config = grpo_config
        self.peft_config = peft_config
        
        # Initialize internal reward function
        self.internal_reward = InternalRewardFunction(
            model=model,
            processor=processor,
            config=internal_reward_config or InternalRewardConfig(),
            vision_encoder=vision_encoder,
            vision_processor=vision_processor,
        )
        
        # Will be initialized when training starts
        self.grpo_trainer = None
        self.step_count = 0

    def _create_reward_wrapper(self):
        """
        Create a reward function wrapper compatible with GRPOTrainer.
        
        GRPOTrainer expects: Callable[[list, list], list[float]]
        """
        def reward_wrapper(prompts: list, completions: list) -> list[float]:
            """Wrapper that converts GRPOTrainer format to our internal format."""
            # For image generation, 'completions' are actually images
            # We need to handle this appropriately
            rewards = []
            
            for prompt, completion in zip(prompts, completions):
                # If completion is an image, use it directly
                if isinstance(completion, Image.Image):
                    reward = self.internal_reward._compute_single_reward(prompt, completion)
                else:
                    # Text completion - use a simpler reward
                    reward = 0.5  # Placeholder for text completions
                
                rewards.append(reward)
            
            return rewards
        
        return reward_wrapper

    def _patched_calculate_rewards(self, original_method):
        """
        Create a patched version of _calculate_rewards that uses internal rewards.
        """
        def patched_method(inputs, prompts, generated_images):
            """Use internal rewards instead of external OCR."""
            device = self.model.device if hasattr(self.model, 'device') else 'cuda'
            
            # Compute internal rewards
            rewards = []
            for prompt, image in zip(prompts, generated_images):
                reward = self.internal_reward._compute_single_reward(prompt, image)
                rewards.append(reward)
            
            # Compute batch diversity and adjust rewards
            if len(generated_images) > 1:
                diversity = self.internal_reward.compute_batch_diversity(generated_images)
                # Add diversity bonus
                for i in range(len(rewards)):
                    rewards[i] = rewards[i] + self.internal_reward.config.diversity_weight * diversity
            
            # Convert to tensor format expected by GRPOTrainer
            rewards_per_func = torch.tensor(rewards, device=device).unsqueeze(-1)
            
            # Gather across processes if distributed
            try:
                from accelerate.utils import gather
                rewards_per_func = gather(rewards_per_func)
            except:
                pass
            
            return rewards_per_func
        
        return patched_method

    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Run training with internal rewards.
        
        This method:
        1. Initializes the GRPOTrainer with patched reward function
        2. Runs training
        3. Periodically updates the frozen judge
        """
        # Import GRPO components
        try:
            from trl.trainer.grpo_trainer import GRPOTrainer
            from trl.trainer.grpo_config import GRPOConfig
        except ImportError:
            raise ImportError(
                "Could not import GRPOTrainer. Make sure BLIP3o is installed."
            )
        
        # Create GRPO config if not provided
        if self.grpo_config is None:
            self.grpo_config = GRPOConfig(
                output_dir="./outputs_grpo",
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=1e-5,
                num_generations=4,  # Sample 4 images per prompt
                beta=0.01,  # KL coefficient
            )
        
        # Create reward wrapper
        reward_func = self._create_reward_wrapper()
        
        # Initialize trainer
        self.grpo_trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=[reward_func],
            args=self.grpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.processor,
            peft_config=self.peft_config,
        )
        
        # Patch the _calculate_rewards method
        original_calculate_rewards = self.grpo_trainer._calculate_rewards
        self.grpo_trainer._calculate_rewards = self._patched_calculate_rewards(original_calculate_rewards)
        
        # Add callback for frozen judge updates
        class FrozenJudgeCallback:
            def __init__(self, internal_reward, update_every=100):
                self.internal_reward = internal_reward
                self.update_every = update_every
                self.step = 0
            
            def on_step_end(self, args, state, control, **kwargs):
                self.step += 1
                if self.step % self.update_every == 0:
                    self.internal_reward.update_frozen_judge()
                    print(f"[Step {self.step}] Updated frozen judge")
        
        callback = FrozenJudgeCallback(self.internal_reward)
        self.grpo_trainer.add_callback(callback)
        
        # Run training
        print("Starting GRPO training with internal rewards...")
        result = self.grpo_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Log final stats
        stats = self.internal_reward.get_stats()
        print(f"Training complete. Final reward stats: {stats}")
        
        return result

    def save_model(self, output_dir: str):
        """Save the trained model."""
        if self.grpo_trainer is not None:
            self.grpo_trainer.save_model(output_dir)
        else:
            self.model.save_pretrained(output_dir)
        
        print(f"Model saved to: {output_dir}")

    def evaluate(self):
        """Run evaluation with internal rewards."""
        if self.grpo_trainer is not None:
            return self.grpo_trainer.evaluate()
        else:
            raise RuntimeError("Trainer not initialized. Call train() first.")


# Convenience function for creating the trainer
def create_self_evolving_trainer(
    model_name_or_path: str,
    train_data_path: Optional[str] = None,
    output_dir: str = "./outputs_grpo",
    internal_reward_config: Optional[InternalRewardConfig] = None,
    **kwargs
):
    """
    Create a SelfEvolvingGRPOTrainer with sensible defaults.
    
    Args:
        model_name_or_path: BLIP3o-NEXT model name or path
        train_data_path: Path to training data
        output_dir: Output directory
        internal_reward_config: Internal reward configuration
        **kwargs: Additional arguments
        
    Returns:
        SelfEvolvingGRPOTrainer instance
    """
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    # Load model and processor
    print(f"Loading model: {model_name_or_path}")
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Create trainer
    trainer = SelfEvolvingGRPOTrainer(
        model=model,
        processor=processor,
        internal_reward_config=internal_reward_config,
        **kwargs
    )
    
    return trainer
