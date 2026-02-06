"""
SolverRole: Answers questions about images, provides self-consistency reward.
Adapted from EvoLMM's Solver concept for BLIP3o-NEXT.
"""

import torch
import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from PIL import Image


class SolverRole:
    """
    Solver answers questions about images with multiple samples.
    Computes self-consistency reward based on answer agreement.
    
    Adapted from EvoLMM's solver for BLIP3o-NEXT.
    """
    
    ANSWER_TEMPLATE = """<|im_start|>system
You are a helpful assistant that answers questions about images accurately and concisely.
<|im_end|>
<|im_start|>user
<image>
{question}

Answer briefly and directly.
<|im_end|>
<|im_start|>assistant
"""

    def __init__(
        self,
        model,
        processor,
        lora_adapter_name: str = "solver",
        device: str = "cuda",
    ):
        """
        Initialize SolverRole.
        
        Args:
            model: BLIP3o-NEXT model (with LoRA adapters)
            processor: Processor for tokenization and image processing
            lora_adapter_name: Name of the LoRA adapter for solver
            device: Device to run on
        """
        self.model = model
        self.processor = processor
        self.lora_adapter_name = lora_adapter_name
        self.device = device
        
    def solve(
        self,
        image: Image.Image,
        question: str,
        n_samples: int = 5,
        temperature: float = 0.7,
        max_new_tokens: int = 64,
    ) -> Tuple[List[str], float]:
        """
        Answer a question about an image with multiple samples.
        
        Args:
            image: Input image
            question: Question to answer
            n_samples: Number of answer samples
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens per answer
            
        Returns:
            Tuple of (list of answers, agreement reward)
        """
        # Switch to solver adapter if using PEFT and adapter exists
        if hasattr(self.model, 'set_adapter'):
            try:
                self.model.set_adapter(self.lora_adapter_name)
            except ValueError:
                pass  # Adapter not loaded yet, use base model
        
        # Detect model type from model class name
        model_class = self.model.__class__.__name__.lower()
        
        # Use appropriate input format based on model type
        if 'qwen' in model_class and hasattr(self.processor, 'apply_chat_template'):
            # Qwen2-VL style: use chat template with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"{question}\n\nAnswer briefly and directly."},
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
                text=question,
                return_tensors="pt",
            ).to(self.device)
        else:
            # Default: use our template
            prompt = self.ANSWER_TEMPLATE.format(question=question)
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
        
        # Generate multiple samples
        answers = []
        for _ in range(n_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 0.7,
                    do_sample=True,
                )
            
            # Decode - handle different output formats
            if hasattr(outputs, 'shape') and len(outputs.shape) == 2:
                # Skip input tokens if present
                input_len = inputs.get('input_ids', inputs.get('input_features', torch.tensor([]))).shape[-1] if hasattr(inputs, 'get') else 0
                if isinstance(inputs, dict) and 'input_ids' in inputs:
                    input_len = inputs['input_ids'].shape[-1]
                    output_text = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
                else:
                    output_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            else:
                output_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer
            answer = self._extract_answer(output_text)
            answers.append(answer)
        
        # Compute agreement reward
        reward = self.compute_agreement_reward(answers)
        
        return answers, reward

    def solve_batch(
        self,
        image: Image.Image,
        questions: List[str],
        n_samples: int = 5,
        temperature: float = 0.7,
        max_new_tokens: int = 64,
    ) -> List[Tuple[List[str], float]]:
        """
        Answer multiple questions about an image.
        
        Args:
            image: Input image
            questions: List of questions
            n_samples: Number of answer samples per question
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens per answer
            
        Returns:
            List of (answers, reward) tuples for each question
        """
        results = []
        for question in questions:
            answers, reward = self.solve(
                image, question, n_samples, temperature, max_new_tokens
            )
            results.append((answers, reward))
        return results

    def _extract_answer(self, text: str) -> str:
        """Extract the answer from generated text."""
        # Remove common prefixes
        answer = text.strip()
        
        # If the full prompt is in the output, extract just the answer
        if "<|im_start|>assistant" in answer:
            answer = answer.split("<|im_start|>assistant")[-1]
        
        return answer.strip()

    def compute_agreement_reward(
        self,
        answers: List[str],
        normalize: bool = True,
    ) -> float:
        """
        Compute agreement-based reward from multiple answer samples.
        
        Uses EvoLMM-style continuous reward based on answer distribution.
        High agreement = high reward (confident answer)
        Low agreement = low reward (uncertain/ambiguous)
        
        Args:
            answers: List of answer strings
            normalize: Whether to normalize answers before comparison
            
        Returns:
            Agreement reward in [0, 1]
        """
        if not answers:
            return 0.0
        
        # Normalize answers for comparison
        if normalize:
            normalized = [self._normalize_answer(a) for a in answers]
        else:
            normalized = answers
        
        # Count answer frequencies
        counter = Counter(normalized)
        
        # Agreement = fraction of samples matching the mode
        mode_count = counter.most_common(1)[0][1]
        agreement = mode_count / len(answers)
        
        return agreement

    def compute_entropy(self, answers: List[str]) -> float:
        """
        Compute entropy of answer distribution.
        
        Low entropy = high agreement (easy question or clear answer)
        High entropy = low agreement (hard question or ambiguous)
        
        Args:
            answers: List of answer strings
            
        Returns:
            Entropy value
        """
        if not answers:
            return 0.0
        
        normalized = [self._normalize_answer(a) for a in answers]
        counter = Counter(normalized)
        
        total = len(answers)
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)
        
        return entropy

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Lowercase, strip, remove punctuation
        import re
        answer = answer.lower().strip()
        answer = re.sub(r'[^\w\s]', '', answer)
        answer = ' '.join(answer.split())  # Normalize whitespace
        return answer

    def verify_with_spec(
        self,
        image: Image.Image,
        spec: "VerificationSpec",  # From proposer.py
        n_samples: int = 5,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Verify an image against a VerificationSpec.
        
        Args:
            image: Image to verify
            spec: Specification with questions and expected answers
            n_samples: Samples per question
            
        Returns:
            Tuple of (overall score, per-question scores)
        """
        per_question_scores = {}
        
        for question in spec.questions:
            answers, agreement = self.solve(image, question, n_samples)
            
            # If we have an expected answer, check for match
            if question in spec.expected_answers:
                expected = self._normalize_answer(spec.expected_answers[question])
                mode_answer = Counter([self._normalize_answer(a) for a in answers]).most_common(1)[0][0]
                
                # Simple match score (can be made more sophisticated)
                if expected in mode_answer or mode_answer in expected:
                    score = agreement  # High score if matches expected
                else:
                    score = 0.0  # Low score if doesn't match
            else:
                # No expected answer, use agreement as score
                score = agreement
            
            per_question_scores[question] = score
        
        # Overall score = mean of per-question scores
        overall = sum(per_question_scores.values()) / len(per_question_scores) if per_question_scores else 0.0
        
        return overall, per_question_scores
