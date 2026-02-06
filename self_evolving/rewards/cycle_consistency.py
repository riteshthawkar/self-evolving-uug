"""
Cycle Consistency Reward for self-evolving generation training.
p → I → c₁..cₙ → sim(p, aggregate(cᵢ))
"""

import torch
from typing import List, Optional, Callable
from PIL import Image


def cycle_consistency_reward(
    prompt: str,
    image: Image.Image,
    captioner: Callable[[Image.Image], str],
    n_samples: int = 5,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    aggregation: str = "mean",
) -> float:
    """
    Compute cycle-consistency reward for generated images.
    
    Flow: prompt → generate image → caption image → compare to prompt
    
    Args:
        prompt: Original generation prompt
        image: Generated image to evaluate
        captioner: Function that captions an image
        n_samples: Number of caption samples
        similarity_fn: Function to compute text similarity (default: token overlap)
        aggregation: How to aggregate caption similarities ("mean", "max", "min")
        
    Returns:
        Cycle-consistency reward in [0, 1]
    """
    # Generate multiple captions
    captions = []
    for _ in range(n_samples):
        caption = captioner(image)
        captions.append(caption)
    
    # Use default similarity if not provided
    if similarity_fn is None:
        similarity_fn = _token_overlap_similarity
    
    # Compute similarity between prompt and each caption
    similarities = [similarity_fn(prompt, cap) for cap in captions]
    
    # Aggregate
    if aggregation == "mean":
        return sum(similarities) / len(similarities)
    elif aggregation == "max":
        return max(similarities)
    elif aggregation == "min":
        return min(similarities)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def _token_overlap_similarity(text1: str, text2: str) -> float:
    """
    Simple token overlap similarity.
    Jaccard similarity between word sets.
    """
    import re
    
    # Tokenize
    def tokenize(text):
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return set(tokens)
    
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    
    return len(intersection) / len(union)


def embedding_similarity(
    text1: str,
    text2: str,
    encoder,
    processor,
) -> float:
    """
    Compute cosine similarity between text embeddings.
    
    Args:
        text1: First text
        text2: Second text
        encoder: Text encoder model
        processor: Text processor/tokenizer
        
    Returns:
        Cosine similarity in [-1, 1]
    """
    import torch.nn.functional as F
    
    # Encode texts
    inputs1 = processor(text1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = processor(text2, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        emb1 = encoder(**inputs1).last_hidden_state.mean(dim=1)
        emb2 = encoder(**inputs2).last_hidden_state.mean(dim=1)
    
    # Cosine similarity
    similarity = F.cosine_similarity(emb1, emb2, dim=-1).item()
    
    return similarity


class CycleConsistencyReward:
    """
    Stateful cycle-consistency reward with built-in captioner.
    """
    
    def __init__(
        self,
        model,
        processor,
        n_samples: int = 5,
        use_embeddings: bool = False,
        temperature: float = 0.7,
    ):
        """
        Initialize cycle-consistency reward.
        
        Args:
            model: VLM model for captioning
            processor: Processor for model
            n_samples: Number of caption samples
            use_embeddings: Whether to use embedding similarity
            temperature: Captioning temperature
        """
        self.model = model
        self.processor = processor
        self.n_samples = n_samples
        self.use_embeddings = use_embeddings
        self.temperature = temperature
        
        # Caption prompt template
        self.caption_template = """<|im_start|>system
You are a helpful assistant that describes images.
<|im_end|>
<|im_start|>user
<image>
Describe this image in detail.
<|im_end|>
<|im_start|>assistant
"""
    
    def __call__(
        self,
        prompt: str,
        image: Image.Image,
    ) -> float:
        """Compute reward for a prompt-image pair."""
        return cycle_consistency_reward(
            prompt=prompt,
            image=image,
            captioner=self._caption,
            n_samples=self.n_samples,
            similarity_fn=self._similarity if self.use_embeddings else None,
        )
    
    def _caption(self, image: Image.Image) -> str:
        """Generate a caption for an image."""
        
        # Check if model/processor supports chat templates (e.g., Qwen2.5-VL)
        if hasattr(self.processor, 'apply_chat_template'):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this image in detail."},
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
            ).to(self.model.device)
        else:
            # Fallback to manual template for other models
            inputs = self.processor(
                text=self.caption_template,
                images=image,
                return_tensors="pt"
            ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=self.temperature,
                do_sample=True,
            )
        
        text = self.processor.decode(outputs[0], skip_special_tokens=True)
        # Extract just the caption if needed (model dependent)
        if "<|im_start|>assistant" in text:
            text = text.split("<|im_start|>assistant")[-1]
        return text.strip()
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Compute similarity using model's text encoder."""
        # Use token overlap as fallback
        return _token_overlap_similarity(text1, text2)
