"""
Reward / scoring pipeline for the VARGPT self-evolving framework.

Ported from BLIP3o's rewards.py with adaptations:
  - Uses VARGPT's own NLL as a proxy reward signal (following SUDER pattern)
  - Adds CLIP-based text-image alignment scoring
  - Embedding similarity adapted for Qwen2-VL processor
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from .utils import (
    _build_chat_text,
    _decode_tokens,
    _prepare_mm_inputs,
    _unwrap_model,
    normalize_answer,
    use_adapter,
)
from .generation_helpers import _soft_match

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLIP-based reward (text-image alignment)
# ---------------------------------------------------------------------------

_clip_model = None
_clip_processor = None


def _load_clip():
    """Lazily load CLIP model for text-image similarity scoring."""
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor
    try:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if torch.cuda.is_available():
            _clip_model = _clip_model.cuda().eval()
        else:
            _clip_model = _clip_model.eval()
        logger.info("[Rewards] Loaded CLIP model for scoring")
    except Exception as e:
        logger.warning(f"[Rewards] Failed to load CLIP: {e}")
        _clip_model = None
        _clip_processor = None
    return _clip_model, _clip_processor


def clip_similarity(image: Image.Image, text: str) -> float:
    """Compute CLIP cosine similarity between image and text.

    Returns a float in [-1, 1], typically [0, 0.4] for realistic pairs.
    """
    model, processor = _load_clip()
    if model is None or processor is None:
        return 0.0

    try:
        inputs = processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Normalize embeddings
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            similarity = (image_embeds * text_embeds).sum(dim=-1)
            return float(similarity.item())
    except Exception as e:
        logger.warning(f"[Rewards] CLIP similarity failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# VARGPT NLL-based reward (model's own generation loss)
# ---------------------------------------------------------------------------


def vargpt_nll_reward(
    model: torch.nn.Module,
    tokenizer,
    image: Image.Image,
    text: str,
    device: torch.device,
) -> float:
    """Compute reward as negative NLL from VARGPT's image generation loss.

    This follows the SUDER pattern: reward = -loss(image_gen | text).
    Lower generation loss → higher reward → image matches text well.

    Parameters
    ----------
    model : torch.nn.Module
        VARGPT model.
    tokenizer : tokenizer
        Tokenizer for text input.
    image : Image.Image
        Generated image to score.
    text : str
        Prompt text.
    device : torch.device
        Target device.

    Returns
    -------
    float
        Negative NLL (higher is better).
    """
    base_model = _unwrap_model(model)
    was_training = base_model.training

    try:
        base_model.eval()

        # Prepare image tensor for pixel_gen_values
        try:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        except Exception:
            return 0.0

        # Tokenize prompt
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = input_ids.clone()

        with torch.no_grad():
            # Disable all adapters for unbiased scoring
            if hasattr(base_model, "disable_adapter"):
                with base_model.disable_adapter():
                    outputs = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        pixel_gen_values=[img_tensor],
                        dpo_training=True,
                        use_cache=False,
                    )
            else:
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_gen_values=[img_tensor],
                    dpo_training=True,
                    use_cache=False,
                )

        if outputs.loss is not None:
            nll = float(outputs.loss.item())
            if math.isfinite(nll):
                return -nll  # negative NLL → higher is better
        return 0.0

    except Exception as e:
        logger.warning(f"[Rewards] NLL reward failed: {e}")
        return 0.0
    finally:
        if was_training:
            base_model.train(True)


# ---------------------------------------------------------------------------
# Embedding similarity reward (model's own representations)
# ---------------------------------------------------------------------------


def embedding_similarity(
    model: torch.nn.Module,
    processor,
    image: Image.Image,
    text: str,
    device: torch.device,
) -> float:
    """Compute embedding-based quality score for a generated image.

    Compares the model's hidden-state representations from two passes:
      - Text-only: encodes just the text prompt
      - Multimodal: encodes the text prompt + generated image
    The cosine similarity between these embeddings indicates how well the
    image aligns with the text from the model's perspective.

    Parameters
    ----------
    model : torch.nn.Module
        VARGPT model.
    processor : processor
        Processor for multimodal inputs.
    image : Image.Image
        Image to score.
    text : str
        Text to compare against.
    device : torch.device
        Target device.

    Returns
    -------
    float
        Cosine similarity in [-1, 1].
    """
    from .utils import _build_text_only_chat, _prepare_text_only_inputs

    base_model = _unwrap_model(model)
    was_training = base_model.training

    try:
        base_model.eval()

        # ── 1. Text-only embedding ──────────────────────────────────────
        text_chat = _build_text_only_chat(processor, text)
        text_inputs = _prepare_text_only_inputs(processor, device, text_chat)
        text_inputs["use_cache"] = False
        text_inputs["output_hidden_states"] = True

        # ── 2. Multimodal embedding ─────────────────────────────────────
        mm_chat = _build_chat_text(processor, image, text)
        mm_inputs = _prepare_mm_inputs(processor, device, image, mm_chat, model=model)
        mm_inputs["use_cache"] = False
        mm_inputs["output_hidden_states"] = True

        with torch.no_grad():
            if hasattr(base_model, "disable_adapter"):
                with base_model.disable_adapter():
                    text_out = base_model(**text_inputs)
                    mm_out = base_model(**mm_inputs)
            else:
                text_out = base_model(**text_inputs)
                mm_out = base_model(**mm_inputs)

        if text_out.hidden_states is not None and mm_out.hidden_states is not None:
            # Mean-pool last hidden states
            text_emb = text_out.hidden_states[-1].mean(dim=1)  # (1, H)
            mm_emb = mm_out.hidden_states[-1].mean(dim=1)      # (1, H)

            # Cosine similarity
            text_emb = F.normalize(text_emb, dim=-1)
            mm_emb = F.normalize(mm_emb, dim=-1)
            sim = (text_emb * mm_emb).sum(dim=-1)
            return float(sim.item())

        return 0.0

    except Exception as e:
        logger.warning(f"[Rewards] Embedding similarity failed: {e}")
        return 0.0
    finally:
        if was_training:
            base_model.train(True)


# ---------------------------------------------------------------------------
# Solver-based verification reward
# ---------------------------------------------------------------------------


def solver_verification_reward(
    model: torch.nn.Module,
    processor,
    image: Image.Image,
    questions: List[str],
    expected_answers: List[str],
    device: torch.device,
    num_samples: int = 3,
    max_new_tokens: int = 64,
    solver_adapter_name: str = "solver",
) -> Tuple[float, Dict[str, float]]:
    """Score a generated image by having the solver answer verification QA.

    For each QA pair, generate solver answers and compare with expected.
    The reward combines:
      - Accuracy: soft_match(solver_answer, expected_answer)
      - Consistency: entropy of solver answers (lower = more consistent)

    Returns
    -------
    reward : float
        Combined verification reward in [0, 1].
    details : dict
        Per-QA statistics.
    """
    if not questions or not expected_answers:
        return 0.0, {"no_qa_pairs": True}

    base_model = _unwrap_model(model)
    was_training = base_model.training

    try:
        base_model.eval()

        qa_scores = []
        for q, exp_a in zip(questions, expected_answers):
            # Build solver prompt
            from .prompts import build_solver_prompt
            solver_prompt = build_solver_prompt(q)

            # Build chat text with image
            chat_text = _build_chat_text(processor, image, solver_prompt)

            # Prepare inputs
            mm_inputs = _prepare_mm_inputs(
                processor, device, image, chat_text, model=model
            )

            # Generate solver answers
            answers = []
            with torch.no_grad():
                with use_adapter(base_model, solver_adapter_name):
                    for _ in range(num_samples):
                        try:
                            gen_ids = base_model.generate(
                                **mm_inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=1.0,
                                top_p=0.9,
                            )
                            # Decode only new tokens
                            input_len = mm_inputs["input_ids"].shape[1]
                            new_ids = gen_ids[0, input_len:]
                            answer_text = _decode_tokens(
                                getattr(processor, "tokenizer", processor),
                                new_ids,
                            )
                            from .utils import _parse_answer
                            answer = _parse_answer(answer_text)
                            answers.append(normalize_answer(answer))
                        except Exception:
                            answers.append("")

            # Score this QA pair
            if not answers or all(a == "" for a in answers):
                qa_scores.append(0.0)
                continue

            # Accuracy: best match among answers
            match_scores = [_soft_match(a, exp_a) for a in answers if a]
            accuracy = max(match_scores) if match_scores else 0.0

            # Consistency: how much answers agree
            unique_answers = set(a for a in answers if a)
            consistency = 1.0 / max(len(unique_answers), 1)

            qa_score = 0.7 * accuracy + 0.3 * consistency
            qa_scores.append(qa_score)

        reward = sum(qa_scores) / max(len(qa_scores), 1)
        details = {
            "num_qa": len(qa_scores),
            "mean_qa_score": reward,
            "qa_scores": qa_scores,
        }
        return reward, details

    except Exception as e:
        logger.warning(f"[Rewards] Solver verification failed: {e}")
        return 0.0, {"error": str(e)}
    finally:
        if was_training:
            base_model.train(True)


# ---------------------------------------------------------------------------
# Combined reward scoring
# ---------------------------------------------------------------------------


def score_generated_image(
    model: torch.nn.Module,
    processor,
    image: Image.Image,
    prompt: str,
    questions: List[str],
    expected_answers: List[str],
    device: torch.device,
    config,
) -> Tuple[float, Dict[str, float]]:
    """Compute a combined reward for a generated image.

    Combines multiple reward signals based on config.gen_reward_mode:
      - "clip": CLIP text-image similarity
      - "nll": Model's own generation NLL
      - "embedding": Model's embedding similarity
      - "combined": All signals combined

    Also includes solver verification if QA pairs are available.

    Returns
    -------
    total_reward : float
        Combined reward in [0, 1].
    details : dict
        Component-level scores.
    """
    details = {}
    components = []

    mode = getattr(config, "gen_reward_mode", "clip")

    # CLIP similarity
    if mode in ("clip", "combined"):
        clip_score = clip_similarity(image, prompt)
        # Normalize CLIP score to [0, 1] (typical range is [0.15, 0.35])
        clip_reward = min(1.0, max(0.0, (clip_score - 0.1) / 0.25))
        details["clip_score"] = clip_score
        details["clip_reward"] = clip_reward
        components.append(("clip", clip_reward, 0.4))

    # NLL reward
    if mode in ("nll", "combined"):
        tokenizer = getattr(processor, "tokenizer", processor)
        nll_score = vargpt_nll_reward(model, tokenizer, image, prompt, device)
        # Normalize: typical NLL range is [-10, -2], map to [0, 1]
        nll_reward = min(1.0, max(0.0, (nll_score + 10.0) / 8.0))
        details["nll_score"] = nll_score
        details["nll_reward"] = nll_reward
        components.append(("nll", nll_reward, 0.3))

    # Solver verification
    if questions and expected_answers:
        solver_reward, solver_details = solver_verification_reward(
            model, processor, image, questions, expected_answers,
            device, num_samples=getattr(config, "num_solver_samples_spec", 3),
        )
        details["solver_reward"] = solver_reward
        details.update({f"solver_{k}": v for k, v in solver_details.items()
                       if isinstance(v, (int, float))})
        components.append(("solver", solver_reward, 0.5))

    # Compute weighted total
    if not components:
        return 0.5, details  # neutral reward if no scoring available

    total_weight = sum(w for _, _, w in components)
    total_reward = sum(score * w for _, score, w in components) / total_weight

    details["total_reward"] = total_reward
    details["reward_mode"] = mode
    return total_reward, details
