"""
Reward functions for the self-evolving training pipeline.
Consolidates cycle-consistency, verification, diversity, and contradiction rewards.
Ported from self_evolving/rewards/.
"""

import math
import re
import torch
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple
from PIL import Image

from .utils import _is_bare_tokenizer, _ensure_blip3o_mm_utils, _build_chat_text


# ===================== Cycle Consistency ===================================

def _token_overlap_similarity(text1: str, text2: str) -> float:
    """Simple Jaccard token-overlap similarity."""

    def tokenize(text):
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
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
    model,
    processor,
) -> float:
    """Compute cosine similarity between text embeddings using the LLM backbone.

    Works with decoder-only models (like BLIP3o/Qwen) by mean-pooling the
    last hidden state.  The model is used in no-grad / eval mode so this
    is a pure inference call.
    """
    import torch.nn.functional as F

    def _embed(text: str) -> torch.Tensor:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1]  # [1, seq_len, dim]
        mask = inputs.get("attention_mask")
        if mask is not None:
            mask = mask.unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            emb = hidden.mean(dim=1)
        return F.normalize(emb.squeeze(0), dim=-1)

    with torch.no_grad():
        emb1 = _embed(text1)
        emb2 = _embed(text2)

    return float(torch.dot(emb1, emb2).item())


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
    Flow: prompt -> generate image -> caption image -> compare to prompt
    """
    captions = []
    for _ in range(n_samples):
        caption = captioner(image)
        captions.append(caption)

    if similarity_fn is None:
        similarity_fn = _token_overlap_similarity

    similarities = [similarity_fn(prompt, cap) for cap in captions]

    if aggregation == "mean":
        return sum(similarities) / len(similarities)
    elif aggregation == "max":
        return max(similarities)
    elif aggregation == "min":
        return min(similarities)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


class CycleConsistencyReward:
    """Stateful cycle-consistency reward with built-in captioner."""

    def __init__(
        self,
        model,
        processor,
        n_samples: int = 5,
        use_embeddings: bool = False,
        temperature: float = 0.7,
    ):
        self.model = model
        self.processor = processor
        self.n_samples = n_samples
        self.use_embeddings = use_embeddings
        self.temperature = temperature

        self.caption_template = (
            "<|im_start|>system\n"
            "You are a helpful assistant that describes images.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "<image>\n"
            "Describe this image in detail.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def __call__(self, prompt: str, image: Image.Image) -> float:
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
        from .utils import _prepare_mm_inputs

        prompt_text = "Describe this image in detail."
        chat_text = _build_chat_text(self.processor, image, prompt_text)
        inputs = _prepare_mm_inputs(
            self.processor,
            next(self.model.parameters()).device,
            image,
            chat_text,
            model=self.model,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=self.temperature,
                do_sample=True,
            )

        # Decode only the generated tokens (skip the prompt portion)
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        completion_ids = outputs[0, input_len:]
        text = self.processor.decode(completion_ids, skip_special_tokens=True)
        if "<|im_start|>assistant" in text:
            text = text.split("<|im_start|>assistant")[-1]
        return text.strip()

    def _similarity(self, text1: str, text2: str) -> float:
        """Compute similarity using the model's text encoder."""
        try:
            return embedding_similarity(text1, text2, self.model, self.processor)
        except Exception:
            return _token_overlap_similarity(text1, text2)


# ===================== Diversity ===========================================

def diversity_reward(
    images: List[Image.Image],
    vision_encoder=None,
    processor=None,
    normalize: bool = True,
) -> float:
    """
    Compute diversity reward across generated images.
    Uses variance of vision embeddings or pixel-level fallback.
    """
    if len(images) < 2:
        return 1.0

    if vision_encoder is not None and processor is not None:
        try:
            features = []
            for img in images:
                feat = _extract_features(img, vision_encoder, processor)
                features.append(feat)

            features = torch.stack(features)  # [N, D]
            if normalize:
                features = torch.nn.functional.normalize(features, dim=-1)

            variance = features.var(dim=0).mean().item()
            pairwise_distances = _pairwise_cosine_distances(features)
            mean_distance = pairwise_distances.mean().item()
            return (variance + mean_distance) / 2
        except TypeError:
            pass  # Fall through to pixel-level fallback

    # Pixel-level fallback: two-component metric
    return _image_diversity_score_pixel(images)


def _extract_features(image: Image.Image, vision_encoder, processor) -> torch.Tensor:
    """Extract vision features from an image."""
    if _is_bare_tokenizer(processor):
        # For bare tokenizers (BLIP3o), we can't use processor(images=...).
        # Fall back to pixel-level diversity (caller handles this).
        raise TypeError("Cannot extract vision features with a bare tokenizer processor.")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = vision_encoder(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)

    return features.squeeze(0)


def _pairwise_cosine_distances(features: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine distances."""
    normalized = torch.nn.functional.normalize(features, dim=-1)
    similarity = torch.mm(normalized, normalized.t())
    distance = 1 - similarity
    mask = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    pairwise = distance[mask]
    return pairwise


def _image_diversity_score_pixel(images: List[Image.Image]) -> float:
    """Two-component pixel-level diversity: grayscale cosine + color histogram."""
    import numpy as np

    if len(images) < 2:
        return 1.0

    sz = (64, 64)
    gray_vecs = []
    hists = []

    for img in images:
        arr = np.array(img.resize(sz).convert("L"), dtype=np.float32).ravel()
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        gray_vecs.append(arr)

        rgb = np.array(img.resize(sz).convert("RGB"), dtype=np.float32)
        hist = np.concatenate(
            [
                np.histogram(rgb[:, :, c], bins=32, range=(0, 256), density=True)[0]
                for c in range(3)
            ]
        )
        hists.append(hist)

    # Grayscale cosine distance
    cos_dists = []
    for i in range(len(gray_vecs)):
        for j in range(i + 1, len(gray_vecs)):
            sim = float(np.dot(gray_vecs[i], gray_vecs[j]))
            cos_dists.append(1.0 - max(-1.0, min(1.0, sim)))
    gray_div = float(np.mean(cos_dists)) if cos_dists else 0.0

    # Color histogram Bhattacharyya distance
    bhatt_dists = []
    for i in range(len(hists)):
        for j in range(i + 1, len(hists)):
            bc = float(np.sum(np.sqrt(np.maximum(hists[i] * hists[j], 0.0))))
            bhatt_dists.append(1.0 - max(0.0, min(1.0, bc)))
    color_div = float(np.mean(bhatt_dists)) if bhatt_dists else 0.0

    return max(0.0, min(1.0, 0.5 * gray_div + 0.5 * color_div))


class DiversityReward:
    """Stateful diversity reward with vision encoder."""

    def __init__(
        self,
        vision_encoder=None,
        processor=None,
        min_diversity: float = 0.1,
        target_diversity: float = 0.5,
    ):
        self.vision_encoder = vision_encoder
        self.processor = processor
        self.min_diversity = min_diversity
        self.target_diversity = target_diversity

        if vision_encoder is not None:
            for param in vision_encoder.parameters():
                param.requires_grad = False

    def __call__(self, images: List[Image.Image]) -> float:
        raw_diversity = diversity_reward(
            images=images,
            vision_encoder=self.vision_encoder,
            processor=self.processor,
        )

        if raw_diversity < self.min_diversity:
            return 0.0
        elif raw_diversity >= self.target_diversity:
            return 1.0
        else:
            return (raw_diversity - self.min_diversity) / (
                self.target_diversity - self.min_diversity
            )


# ===================== Verification ========================================

def verification_reward(
    image: Image.Image,
    spec,
    solver,
    n_samples: int = 5,
    match_threshold: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """Compute verification reward based on spec compliance."""
    return solver.verify_with_spec(image, spec, n_samples)


def paired_contrast_check(
    image: Image.Image,
    positive_question: str,
    negative_question: str,
    solver,
    n_samples: int = 5,
) -> float:
    """Anti-collusion check using paired contrast questions."""
    pos_answers, pos_agreement = solver.solve(image, positive_question, n_samples)
    neg_answers, neg_agreement = solver.solve(image, negative_question, n_samples)

    def is_affirmative(answer: str) -> bool:
        answer = answer.lower().strip()
        return answer.startswith("yes") or answer in ["true", "correct", "1"]

    pos_mode = max(set(pos_answers), key=pos_answers.count)
    neg_mode = max(set(neg_answers), key=neg_answers.count)

    pos_is_yes = is_affirmative(pos_mode)
    neg_is_yes = is_affirmative(neg_mode)

    if pos_is_yes == neg_is_yes:
        return (pos_agreement + neg_agreement) / 2

    return 0.0


def entropy_diversity(
    captions: List[str],
    tokenizer=None,
) -> float:
    """Compute diversity based on caption entropy."""
    if len(captions) < 2:
        return 1.0

    all_words = []
    for cap in captions:
        words = cap.lower().split()
        all_words.extend(words)

    counter = Counter(all_words)
    total = len(all_words)

    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)

    max_entropy = math.log(len(counter)) if counter else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return normalized_entropy
