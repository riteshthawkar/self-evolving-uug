"""
GeneratorRole: Generates images via BLIP3o-NEXT's discrete token policy.
Wraps the 729-token AR generation path with frozen SANA diffusion.
"""

import torch
from typing import List, Optional
from PIL import Image


class GeneratorRole:
    """
    Generator produces images via BLIP3o-NEXT's discrete token policy.
    
    Key points:
    - AR model generates 729 discrete SigLIP-2 tokens
    - SANA diffusion refines tokens to pixel output
    - For RL, diffusion is frozen; only AR policy is updated
    
    This class wraps BLIP3o-NEXT's generation path for the self-evolving loop.
    """
    
    GENERATION_TEMPLATE = """<|im_start|>system
You are a helpful assistant that generates images based on descriptions.
<|im_end|>
<|im_start|>user
Please generate image based on the following caption: {prompt}
<|im_end|>
<|im_start|>assistant
<image>"""

    def __init__(
        self,
        model,
        processor,
        diffusion_model=None,  # Can be passed separately if needed
        lora_adapter_name: str = "generator",
        device: str = "cuda",
        freeze_diffusion: bool = True,
    ):
        """
        Initialize GeneratorRole.
        
        Args:
            model: BLIP3o-NEXT model
            processor: Processor for tokenization
            diffusion_model: SANA diffusion model (optional, may be part of model)
            lora_adapter_name: Name of the LoRA adapter for generator
            device: Device to run on
            freeze_diffusion: Whether to freeze diffusion during training
        """
        self.model = model
        self.processor = processor
        self.diffusion_model = diffusion_model
        self.lora_adapter_name = lora_adapter_name
        self.device = device
        self.freeze_diffusion = freeze_diffusion
        
        # Freeze diffusion if specified
        if freeze_diffusion and hasattr(model, 'get_diffusion_model'):
            diff = model.get_diffusion_model()
            if diff is not None:
                for param in diff.parameters():
                    param.requires_grad = False
                print("[GeneratorRole] Diffusion model frozen")

    def generate(
        self,
        prompt: str,
        n_samples: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 1024,
        width: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> List[Image.Image]:
        """
        Generate images from a text prompt.
        
        Uses BLIP3o-NEXT's text-to-image generation path:
        1. Tokenize prompt
        2. AR model generates 729 discrete image tokens
        3. SANA diffusion renders tokens to pixels
        
        Args:
            prompt: Text prompt for generation
            n_samples: Number of images to generate
            num_inference_steps: Diffusion steps
            guidance_scale: Classifier-free guidance scale
            height: Output height
            width: Output width
            temperature: AR sampling temperature
            top_p: AR nucleus sampling threshold
            
        Returns:
            List of generated PIL Images
        """
        # Switch to generator adapter if using PEFT
        if hasattr(self.model, 'set_adapter'):
            try:
                self.model.set_adapter(self.lora_adapter_name)
            except ValueError:
                pass
        
        images = []
        for _ in range(n_samples):
            image = self._generate_single(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                temperature=temperature,
                top_p=top_p,
            )
            images.append(image)
        
        return images

    def _generate_single(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 1024,
        width: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> Image.Image:
        """Generate a single image."""
        
        # Use BLIP3o-NEXT's native generation if available
        if hasattr(self.model, 'generate_image'):
            return self.model.generate_image(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            )
        
        # Otherwise, use the standard generation path
        formatted_prompt = self.GENERATION_TEMPLATE.format(prompt=prompt)
        
        inputs = self.processor(
            text=formatted_prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            # Generate image tokens (729 discrete tokens)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=729,  # 27x27 = 729 tokens for 384x384/14 patches
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
            
            # Decode through diffusion (this is model-specific)
            if hasattr(self.model, 'decode_image_tokens'):
                image = self.model.decode_image_tokens(
                    outputs,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                )
            else:
                # Fallback: use the model's full generation pipeline
                raise NotImplementedError(
                    "Model doesn't have decode_image_tokens method. "
                    "Use BLIP3o-NEXT's native generate_image method."
                )
        
        return image

    def generate_trajectories(
        self,
        prompt: str,
        n_trajectories: int = 4,
        return_tokens: bool = True,
    ) -> dict:
        """
        Generate multiple image token trajectories for GRPO.
        
        This is the key method for RL training:
        - Sample G trajectories (each = 729 discrete tokens)
        - Return both tokens and rendered images
        
        Args:
            prompt: Text prompt
            n_trajectories: Number of trajectories to sample
            return_tokens: Whether to return raw tokens
            
        Returns:
            Dict with 'images', 'tokens', and 'log_probs'
        """
        # Switch to generator adapter
        if hasattr(self.model, 'set_adapter'):
            try:
                self.model.set_adapter(self.lora_adapter_name)
            except ValueError:
                pass
        
        formatted_prompt = self.GENERATION_TEMPLATE.format(prompt=prompt)
        
        inputs = self.processor(
            text=formatted_prompt,
            return_tensors="pt"
        ).to(self.device)
        
        all_tokens = []
        all_images = []
        all_log_probs = []
        
        for _ in range(n_trajectories):
            with torch.no_grad():
                # Generate with output_scores for log_probs
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=729,
                    temperature=1.0,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                
                tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]
                all_tokens.append(tokens)
                
                # Compute log probs from scores
                if outputs.scores:
                    log_probs = self._compute_log_probs(outputs.scores, tokens)
                    all_log_probs.append(log_probs)
                
                # Render image
                if hasattr(self.model, 'decode_image_tokens'):
                    image = self.model.decode_image_tokens(tokens)
                    all_images.append(image)
        
        result = {
            'images': all_images,
            'tokens': torch.stack(all_tokens) if all_tokens else None,
            'log_probs': torch.stack(all_log_probs) if all_log_probs else None,
        }
        
        return result

    def _compute_log_probs(
        self,
        scores: tuple,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities from generation scores."""
        log_probs = []
        for i, score in enumerate(scores):
            if i < tokens.shape[1]:
                token_id = tokens[0, i]
                log_prob = torch.log_softmax(score, dim=-1)[0, token_id]
                log_probs.append(log_prob)
        return torch.stack(log_probs) if log_probs else torch.tensor([])

    def compute_policy_gradient_loss(
        self,
        tokens: torch.Tensor,
        rewards: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor] = None,
        beta: float = 0.01,
    ) -> torch.Tensor:
        """
        Compute policy gradient loss for RL training.
        
        Args:
            tokens: Generated token sequences [batch, seq_len]
            rewards: Rewards for each sequence [batch]
            ref_log_probs: Reference policy log probs for KL penalty
            beta: KL penalty coefficient
            
        Returns:
            Policy gradient loss
        """
        # This is a placeholder - actual implementation would use
        # the GRPO trainer's loss computation
        raise NotImplementedError(
            "Use GRPOTrainer for policy optimization. "
            "This method is for reference only."
        )
