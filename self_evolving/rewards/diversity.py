"""
Diversity Reward for anti-mode-collapse.
Penalizes low variance across generated samples.
"""

import torch
from typing import List, Optional
from PIL import Image


def diversity_reward(
    images: List[Image.Image],
    vision_encoder,
    processor,
    normalize: bool = True,
) -> float:
    """
    Compute diversity reward across generated images.
    
    Uses variance of vision embeddings in frozen encoder space.
    Low variance = mode collapse = low reward.
    
    Args:
        images: List of generated images
        vision_encoder: Frozen vision encoder (SigLIP2/CLIP)
        processor: Image processor
        normalize: Whether to normalize features
        
    Returns:
        Diversity reward (higher = more diverse)
    """
    if len(images) < 2:
        return 1.0  # Can't compute diversity with < 2 images
    
    # Extract features
    features = []
    for img in images:
        feat = _extract_features(img, vision_encoder, processor)
        features.append(feat)
    
    features = torch.stack(features)  # [N, D]
    
    if normalize:
        features = torch.nn.functional.normalize(features, dim=-1)
    
    # Compute variance across samples
    variance = features.var(dim=0).mean().item()
    
    # Also compute pairwise distances
    pairwise_distances = _pairwise_cosine_distances(features)
    mean_distance = pairwise_distances.mean().item()
    
    # Combine variance and distance
    diversity = (variance + mean_distance) / 2
    
    return diversity


def _extract_features(
    image: Image.Image,
    vision_encoder,
    processor,
) -> torch.Tensor:
    """Extract vision features from an image."""
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        # Get last hidden state or pooled output
        outputs = vision_encoder(**inputs)
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)
    
    return features.squeeze(0)


def _pairwise_cosine_distances(features: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine distances."""
    # Normalize
    normalized = torch.nn.functional.normalize(features, dim=-1)
    
    # Cosine similarity matrix
    similarity = torch.mm(normalized, normalized.t())
    
    # Convert to distance (1 - similarity)
    distance = 1 - similarity
    
    # Get upper triangular (excluding diagonal)
    mask = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    pairwise = distance[mask]
    
    return pairwise


class DiversityReward:
    """
    Stateful diversity reward with vision encoder.
    """
    
    def __init__(
        self,
        vision_encoder,
        processor,
        min_diversity: float = 0.1,
        target_diversity: float = 0.5,
    ):
        """
        Initialize diversity reward.
        
        Args:
            vision_encoder: Frozen vision encoder
            processor: Image processor
            min_diversity: Minimum acceptable diversity
            target_diversity: Target diversity level
        """
        self.vision_encoder = vision_encoder
        self.processor = processor
        self.min_diversity = min_diversity
        self.target_diversity = target_diversity
        
        # Freeze encoder
        for param in vision_encoder.parameters():
            param.requires_grad = False
    
    def __call__(self, images: List[Image.Image]) -> float:
        """Compute diversity reward for a batch of images."""
        raw_diversity = diversity_reward(
            images=images,
            vision_encoder=self.vision_encoder,
            processor=self.processor,
        )
        
        # Scale relative to target
        if raw_diversity < self.min_diversity:
            return 0.0
        elif raw_diversity >= self.target_diversity:
            return 1.0
        else:
            return (raw_diversity - self.min_diversity) / (self.target_diversity - self.min_diversity)


def entropy_diversity(
    captions: List[str],
    tokenizer=None,
) -> float:
    """
    Compute diversity based on caption entropy.
    
    Args:
        captions: List of generated captions
        tokenizer: Optional tokenizer for more granular analysis
        
    Returns:
        Entropy-based diversity score
    """
    import math
    from collections import Counter
    
    if len(captions) < 2:
        return 1.0
    
    # Simple word-level diversity
    all_words = []
    for cap in captions:
        words = cap.lower().split()
        all_words.extend(words)
    
    # Word frequency distribution
    counter = Counter(all_words)
    total = len(all_words)
    
    # Entropy
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)
    
    # Normalize by max entropy (log of vocab size)
    max_entropy = math.log(len(counter)) if counter else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy
