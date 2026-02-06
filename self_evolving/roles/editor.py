"""
EditorRole: Image editing self-evolution for BLIP3o-NEXT.
Implements anchor-based edit → verify loop from pipeline Appendix B.

Key insight: Editing provides built-in constraints (anchor image)
that make self-supervision cleaner than open-ended generation.

Loop:
1. Anchor image x
2. Proposer: "change attribute A" (edit instruction)
3. Generator: produces edited image I'
4. Solver: verifies "A changed" + "rest unchanged"
5. Reward: edit_success + preservation_score
"""

import torch
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from PIL import Image
import copy


@dataclass
class EditInstruction:
    """Structured edit instruction with verification plan."""
    # The edit to perform
    instruction: str  # e.g., "change the bird to red"
    
    # Attribute being changed
    target_attribute: str  # e.g., "bird color"
    old_value: Optional[str] = None  # e.g., "blue"
    new_value: Optional[str] = None  # e.g., "red"
    
    # Verification questions
    edit_success_questions: List[str] = field(default_factory=list)
    # e.g., ["Is the bird red?", "What color is the bird?"]
    
    preservation_questions: List[str] = field(default_factory=list)
    # e.g., ["Is the background unchanged?", "Are there still trees?"]
    
    # Expected answers (optional, for stricter verification)
    expected_answers: Dict[str, str] = field(default_factory=dict)


@dataclass
class EditResult:
    """Result of an edit operation."""
    anchor_image: Image.Image
    edited_image: Image.Image
    instruction: EditInstruction
    
    # Rewards
    edit_success_score: float = 0.0
    preservation_score: float = 0.0
    combined_reward: float = 0.0
    
    # Verification details
    edit_answers: Dict[str, str] = field(default_factory=dict)
    preservation_answers: Dict[str, str] = field(default_factory=dict)


class EditorRole:
    """
    Image editing role for self-evolving training.
    
    Uses BLIP3o-NEXT's editing capability where:
    - Reference image tokens are concatenated with text tokens
    - VAE latents can be used for reconstruction consistency
    """
    
    def __init__(
        self,
        model,
        processor,
        solver=None,  # SolverRole for verification
        generator=None,  # GeneratorRole for image generation
        lora_adapter_name: str = "editor",
        device: str = "cuda",
        freeze_diffusion: bool = True,
    ):
        """
        Initialize EditorRole.
        
        Args:
            model: BLIP3o-NEXT model
            processor: Processor for model
            solver: SolverRole for verification (optional, will create if None)
            generator: GeneratorRole for generation (optional, will create if None)
            lora_adapter_name: Name of LoRA adapter for this role
            device: Device to use
            freeze_diffusion: Whether to freeze diffusion decoder
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.lora_adapter_name = lora_adapter_name
        self.freeze_diffusion = freeze_diffusion
        
        # Use provided solver or create new one
        if solver is not None:
            self.solver = solver
        else:
            from self_evolving.roles.solver import SolverRole
            self.solver = SolverRole(model, processor, device=device)
        
        # Use provided generator or create new one
        if generator is not None:
            self.generator = generator
        else:
            from self_evolving.roles.generator import GeneratorRole
            self.generator = GeneratorRole(
                model, processor, device=device, freeze_diffusion=freeze_diffusion
            )
        
        # Edit instruction templates
        self.edit_templates = [
            "Change the {attr} to {value}",
            "Make the {attr} {value}",
            "Transform the {attr} into {value}",
            "Replace the {attr} with {value}",
        ]
        
        # Common editable attributes
        self.editable_attributes = {
            'color': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'black', 'white'],
            'time': ['day', 'night', 'sunset', 'sunrise', 'dusk', 'dawn'],
            'weather': ['sunny', 'rainy', 'snowy', 'cloudy', 'foggy', 'stormy'],
            'season': ['spring', 'summer', 'autumn', 'winter'],
            'style': ['realistic', 'cartoon', 'painting', 'sketch', 'watercolor', 'oil painting'],
            'mood': ['happy', 'sad', 'peaceful', 'dramatic', 'mysterious', 'vibrant'],
        }
        
        # System prompt for edit instruction generation
        self.edit_proposer_template = """<|im_start|>system
You are an image editing assistant. Given an image, propose a specific, verifiable edit.
Focus on edits that:
1. Change a single clear attribute
2. Leave the rest of the image unchanged
3. Can be verified by asking simple questions
<|im_end|>
<|im_start|>user
<image>
Propose an edit for this image. Output in this exact format:
EDIT: [edit instruction]
ATTRIBUTE: [what attribute changes]
OLD_VALUE: [current value]
NEW_VALUE: [target value]
VERIFY_EDIT: [question to verify edit succeeded]
VERIFY_PRESERVE: [question to verify something was preserved]
<|im_end|>
<|im_start|>assistant
"""

    def propose_edit(
        self,
        anchor_image: Image.Image,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> Optional[EditInstruction]:
        """
        Generate an edit instruction for an anchor image.
        
        Args:
            anchor_image: Image to edit
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            EditInstruction or None if generation fails
        """
        # Prepare input
        inputs = self.processor(
            text=self.edit_proposer_template,
            images=anchor_image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response
        return self._parse_edit_response(response)

    def _parse_edit_response(self, response: str) -> Optional[EditInstruction]:
        """Parse the model's edit proposal response."""
        try:
            # Extract fields
            lines = response.split('\n')
            fields = {}
            
            for line in lines:
                for key in ['EDIT:', 'ATTRIBUTE:', 'OLD_VALUE:', 'NEW_VALUE:', 
                           'VERIFY_EDIT:', 'VERIFY_PRESERVE:']:
                    if line.strip().startswith(key):
                        fields[key.rstrip(':')] = line.split(key, 1)[1].strip()
            
            if 'EDIT' not in fields:
                return None
            
            return EditInstruction(
                instruction=fields.get('EDIT', ''),
                target_attribute=fields.get('ATTRIBUTE', 'unknown'),
                old_value=fields.get('OLD_VALUE'),
                new_value=fields.get('NEW_VALUE'),
                edit_success_questions=[fields['VERIFY_EDIT']] if 'VERIFY_EDIT' in fields else [],
                preservation_questions=[fields['VERIFY_PRESERVE']] if 'VERIFY_PRESERVE' in fields else [],
            )
        except Exception as e:
            print(f"Failed to parse edit response: {e}")
            return None

    def apply_edit(
        self,
        anchor_image: Image.Image,
        instruction: EditInstruction,
        n_samples: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        use_anchor_latents: bool = True,
    ) -> List[Image.Image]:
        """
        Apply an edit to an anchor image.
        
        Args:
            anchor_image: Image to edit
            instruction: Edit instruction
            n_samples: Number of edited images to generate
            num_inference_steps: Diffusion steps
            guidance_scale: Classifier-free guidance scale
            use_anchor_latents: Whether to use anchor VAE latents for consistency
            
        Returns:
            List of edited images
        """
        # Build edit prompt
        edit_prompt = f"Edit the image: {instruction.instruction}"
        
        # Generate edited images
        # In full implementation, this would use BLIP3o-NEXT's editing mode
        # with reference image tokens concatenated
        edited_images = self.generator.generate(
            prompt=edit_prompt,
            reference_image=anchor_image,  # Pass anchor for editing mode
            n_samples=n_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        return edited_images

    def verify_edit(
        self,
        anchor_image: Image.Image,
        edited_image: Image.Image,
        instruction: EditInstruction,
        n_samples: int = 5,
    ) -> Tuple[float, float, Dict]:
        """
        Verify an edit using solver.
        
        Checks:
        1. Edit success: Did the specified change happen?
        2. Preservation: Is everything else unchanged?
        
        Args:
            anchor_image: Original image
            edited_image: Edited image
            instruction: Edit instruction with verification questions
            n_samples: Samples per verification question
            
        Returns:
            Tuple of (edit_success_score, preservation_score, details)
        """
        details = {
            'edit_answers': {},
            'preservation_answers': {},
        }
        
        # 1. Verify edit success on edited image
        edit_scores = []
        for question in instruction.edit_success_questions:
            answers, agreement = self.solver.solve(
                image=edited_image,
                question=question,
                n_samples=n_samples,
            )
            edit_scores.append(agreement)
            details['edit_answers'][question] = {
                'answers': answers,
                'agreement': agreement,
            }
        
        edit_success_score = sum(edit_scores) / len(edit_scores) if edit_scores else 0.5
        
        # 2. Verify preservation on edited image
        # (Things that should NOT have changed)
        preservation_scores = []
        for question in instruction.preservation_questions:
            # Ask same question on both images
            anchor_answers, anchor_agree = self.solver.solve(
                image=anchor_image,
                question=question,
                n_samples=n_samples,
            )
            edited_answers, edited_agree = self.solver.solve(
                image=edited_image,
                question=question,
                n_samples=n_samples,
            )
            
            # Preservation score: answers should match between anchor and edited
            anchor_mode = max(set(anchor_answers), key=anchor_answers.count)
            edited_mode = max(set(edited_answers), key=edited_answers.count)
            
            preserved = 1.0 if anchor_mode.lower().strip() == edited_mode.lower().strip() else 0.0
            preservation_scores.append(preserved)
            
            details['preservation_answers'][question] = {
                'anchor_answer': anchor_mode,
                'edited_answer': edited_mode,
                'preserved': preserved,
            }
        
        preservation_score = sum(preservation_scores) / len(preservation_scores) if preservation_scores else 0.5
        
        return edit_success_score, preservation_score, details

    def edit_step(
        self,
        anchor_image: Image.Image,
        n_edit_samples: int = 4,
        edit_weight: float = 0.5,
        preserve_weight: float = 0.5,
        temperature: float = 0.7,
    ) -> Optional[EditResult]:
        """
        Complete edit self-play step.
        
        1. Propose edit
        2. Apply edit (generate G samples)
        3. Verify each sample
        4. Return best result
        
        Args:
            anchor_image: Image to edit
            n_edit_samples: Number of edit samples to generate
            edit_weight: Weight for edit success in combined reward
            preserve_weight: Weight for preservation in combined reward
            temperature: Temperature for edit proposal
            
        Returns:
            EditResult with best edited image and scores
        """
        # 1. Propose edit
        instruction = self.propose_edit(anchor_image, temperature=temperature)
        
        if instruction is None:
            print("Failed to generate edit instruction")
            return None
        
        # Add default verification if none provided
        if not instruction.edit_success_questions:
            instruction.edit_success_questions = [
                f"Was the {instruction.target_attribute} changed?"
            ]
        if not instruction.preservation_questions:
            instruction.preservation_questions = [
                "Is the overall composition similar to the original?",
                "Are the main objects in similar positions?"
            ]
        
        # 2. Apply edit
        edited_images = self.apply_edit(
            anchor_image=anchor_image,
            instruction=instruction,
            n_samples=n_edit_samples,
        )
        
        if not edited_images:
            print("Failed to generate edited images")
            return None
        
        # 3. Verify and score each sample
        best_result = None
        best_score = -1.0
        
        for edited_image in edited_images:
            edit_score, preserve_score, details = self.verify_edit(
                anchor_image=anchor_image,
                edited_image=edited_image,
                instruction=instruction,
            )
            
            combined = edit_weight * edit_score + preserve_weight * preserve_score
            
            if combined > best_score:
                best_score = combined
                best_result = EditResult(
                    anchor_image=anchor_image,
                    edited_image=edited_image,
                    instruction=instruction,
                    edit_success_score=edit_score,
                    preservation_score=preserve_score,
                    combined_reward=combined,
                    edit_answers=details['edit_answers'],
                    preservation_answers=details['preservation_answers'],
                )
        
        return best_result


class EditReward:
    """
    Reward function for image editing self-evolution.
    
    Combines:
    - Edit success (did the change happen?)
    - Identity preservation (is everything else the same?)
    - Optional: Perceptual similarity for fine-grained preservation
    """
    
    def __init__(
        self,
        edit_weight: float = 0.5,
        preserve_weight: float = 0.5,
        perceptual_weight: float = 0.0,
        use_perceptual: bool = False,
        perceptual_model=None,
    ):
        """
        Initialize edit reward.
        
        Args:
            edit_weight: Weight for edit success
            preserve_weight: Weight for preservation
            perceptual_weight: Weight for perceptual similarity
            use_perceptual: Whether to use perceptual loss
            perceptual_model: Optional perceptual model for similarity
        """
        self.edit_weight = edit_weight
        self.preserve_weight = preserve_weight
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual
        self.perceptual_model = perceptual_model

    def __call__(
        self,
        edit_success: float,
        preservation: float,
        anchor_image: Optional[Image.Image] = None,
        edited_image: Optional[Image.Image] = None,
    ) -> float:
        """
        Compute edit reward.
        
        Args:
            edit_success: Edit success score [0, 1]
            preservation: Preservation score [0, 1]
            anchor_image: Optional anchor for perceptual similarity
            edited_image: Optional edited for perceptual similarity
            
        Returns:
            Combined reward
        """
        reward = (
            self.edit_weight * edit_success +
            self.preserve_weight * preservation
        )
        
        # Optional: Add perceptual similarity
        if self.use_perceptual and anchor_image and edited_image:
            perceptual_sim = self._compute_perceptual_similarity(
                anchor_image, edited_image
            )
            reward += self.perceptual_weight * perceptual_sim
        
        return reward

    def _compute_perceptual_similarity(
        self,
        anchor: Image.Image,
        edited: Image.Image,
    ) -> float:
        """Compute perceptual similarity between images."""
        if self.perceptual_model is None:
            # Fallback: Simple pixel-level similarity
            import numpy as np
            
            anchor_np = np.array(anchor.resize((256, 256))).astype(float)
            edited_np = np.array(edited.resize((256, 256))).astype(float)
            
            # Normalized L2 distance
            diff = np.sqrt(np.mean((anchor_np - edited_np) ** 2))
            max_diff = 255.0 * np.sqrt(3)  # Max possible diff
            similarity = 1.0 - (diff / max_diff)
            
            return similarity
        else:
            # Use perceptual model
            # This would use LPIPS or similar
            pass
        
        return 0.5
