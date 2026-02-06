"""
ProposerRole: Generates questions/prompts for images.
Adapted from EvoLMM's Proposer concept for BLIP3o-NEXT.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from PIL import Image


@dataclass
class VerificationSpec:
    """
    Structured specification for verifying generated images.
    Contains questions and expected answers for the Solver to verify.
    """
    questions: List[str]
    expected_answers: Dict[str, str] = field(default_factory=dict)
    attributes: List[str] = field(default_factory=list)  # e.g., ["red", "car", "3 people"]
    
    def to_dict(self) -> dict:
        return {
            "questions": self.questions,
            "expected_answers": self.expected_answers,
            "attributes": self.attributes,
        }


class ProposerRole:
    """
    Proposer generates questions/prompts for images using LoRA adapter.
    
    For understanding: generates questions about the image
    For generation: generates creative prompts/edit instructions + verification specs
    
    Adapted from EvoLMM's proposer for BLIP3o-NEXT.
    """
    
    # Prompt templates for different proposer modes
    QUESTION_TEMPLATE = """<|im_start|>system
You are an expert at generating insightful questions about images. Generate questions that test understanding of visual content.
<|im_end|>
<|im_start|>user
<image>
Generate {n_questions} diverse questions about this image. Questions should cover:
1. Object identification
2. Spatial relationships
3. Colors and attributes
4. Actions or states
5. Counting

Output format: One question per line.
<|im_end|>
<|im_start|>assistant
"""

    CREATIVE_PROMPT_TEMPLATE = """<|im_start|>system
You are an expert at generating creative image descriptions and edit instructions.
<|im_end|>
<|im_start|>user
<image>
Based on this image, generate a creative variation prompt. The prompt should describe a modified version of this scene.
Examples:
- "Same scene but at night with a full moon"
- "Replace the dog with a cat, keep everything else"
- "Add snow falling and make it winter"

Also generate 3 verification questions that can check if the generated image matches the prompt.

Output format:
PROMPT: [your creative prompt]
Q1: [verification question 1]
A1: [expected answer 1]
Q2: [verification question 2]
A2: [expected answer 2]
Q3: [verification question 3]
A3: [expected answer 3]
<|im_end|>
<|im_start|>assistant
"""

    def __init__(
        self,
        model,
        processor,
        lora_adapter_name: str = "proposer",
        device: str = "cuda",
    ):
        """
        Initialize ProposerRole.
        
        Args:
            model: BLIP3o-NEXT model (with LoRA adapters)
            processor: Processor for tokenization and image processing
            lora_adapter_name: Name of the LoRA adapter for proposer
            device: Device to run on
        """
        self.model = model
        self.processor = processor
        self.lora_adapter_name = lora_adapter_name
        self.device = device
        
    def propose_questions(
        self,
        image: Image.Image,
        n_questions: int = 5,
        n_samples: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ) -> List[List[str]]:
        """
        Generate questions about an image (for understanding loop).
        
        Args:
            image: Input image
            n_questions: Number of questions to generate
            n_samples: Number of sample generations
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of question lists (one per sample)
        """
        # Switch to proposer adapter if using PEFT
        if hasattr(self.model, 'set_adapter'):
            try:
                self.model.set_adapter(self.lora_adapter_name)
            except ValueError:
                pass
        
        # Detect model type from model class name
        model_class = self.model.__class__.__name__.lower()
        
        # Use appropriate input format based on model type
        if 'qwen' in model_class and hasattr(self.processor, 'apply_chat_template'):
            # Qwen2-VL style: use chat template with image
            question_prompt = f"""Generate {n_questions} diverse questions about this image. Questions should cover:
1. Object identification
2. Spatial relationships
3. Colors and attributes
4. Actions or states
5. Counting

Output format: One question per line, ending with ?"""
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question_prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        elif 'blip2' in model_class:
            # BLIP-2 style
            inputs = self.processor(
                images=image,
                text=f"Generate {n_questions} questions about this image:",
                return_tensors="pt",
            ).to(self.device)
        else:
            # Default: use our template
            prompt = self.QUESTION_TEMPLATE.format(n_questions=n_questions)
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
        
        # Generate
        all_questions = []
        for _ in range(n_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 0.7,
                    do_sample=True,
                )
            
            # Decode - handle different output formats
            if isinstance(inputs, dict) and 'input_ids' in inputs:
                input_len = inputs['input_ids'].shape[-1]
                output_text = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
            else:
                output_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            questions = self._parse_questions(output_text)
            all_questions.append(questions)
        
        return all_questions

    def propose_creative_prompt(
        self,
        image: Image.Image,
        n_samples: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
    ) -> List[Tuple[str, VerificationSpec]]:
        """
        Generate creative prompts and verification specs (for generation loop).
        
        Args:
            image: Input image (for grounding)
            n_samples: Number of sample generations
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of (prompt, spec) tuples
        """
        # Switch to proposer adapter if using PEFT
        if hasattr(self.model, 'set_adapter'):
            try:
                self.model.set_adapter(self.lora_adapter_name)
            except ValueError:
                pass
        
        prompt = self.CREATIVE_PROMPT_TEMPLATE
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        results = []
        for _ in range(n_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                )
            
            # Decode and parse
            text = self.processor.decode(outputs[0], skip_special_tokens=True)
            creative_prompt, spec = self._parse_creative_output(text)
            results.append((creative_prompt, spec))
        
        return results

    def _parse_questions(self, text: str) -> List[str]:
        """Parse generated text into list of questions."""
        lines = text.strip().split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            # Remove numbering if present
            if line and line[0].isdigit():
                line = line.lstrip('0123456789.)-: ')
            if line and line.endswith('?'):
                questions.append(line)
        return questions

    def _parse_creative_output(self, text: str) -> Tuple[str, VerificationSpec]:
        """Parse creative prompt output into prompt and verification spec."""
        lines = text.strip().split('\n')
        
        creative_prompt = ""
        questions = []
        expected = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("PROMPT:"):
                creative_prompt = line[7:].strip()
            elif line.startswith("Q") and ":" in line:
                q_num = line.split(":")[0]
                question = line.split(":", 1)[1].strip()
                questions.append(question)
            elif line.startswith("A") and ":" in line:
                a_num = line.split(":")[0]
                q_idx = int(a_num[1:]) - 1
                if q_idx < len(questions):
                    expected[questions[q_idx]] = line.split(":", 1)[1].strip()
        
        spec = VerificationSpec(
            questions=questions,
            expected_answers=expected,
        )
        
        return creative_prompt, spec

    def compute_entropy_reward(
        self,
        questions: List[str],
        solver_entropy: float,
        target_entropy: float = 1.0,
        sigma: float = 0.5,
    ) -> float:
        """
        Compute proposer reward based on question difficulty (entropy band).
        
        Prefers "moderately difficult" questions that are neither too easy
        (low entropy = high agreement) nor too hard (high entropy = random).
        
        Args:
            questions: Generated questions
            solver_entropy: Solver's answer entropy for these questions
            target_entropy: Target entropy (sweet spot)
            sigma: Bandwidth of Gaussian reward
            
        Returns:
            Proposer reward
        """
        # Gaussian reward centered at target entropy
        # r = exp(-0.5 * ((h - target) / sigma)^2)
        import math
        reward = math.exp(-0.5 * ((solver_entropy - target_entropy) / sigma) ** 2)
        return reward
