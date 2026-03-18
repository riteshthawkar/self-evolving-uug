"""
Reward / scoring pipeline for the VARGPT self-evolving framework.

Ported from BLIP3o's rewards.py with adaptations:
  - Uses VARGPT's own NLL as a proxy reward signal (following SUDER pattern)
  - Adds CLIP-based text-image alignment scoring
  - Embedding similarity adapted for Qwen2-VL processor
"""

import contextlib
import logging
import math
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .utils import (
    _build_chat_text,
    _build_text_only_chat,
    _decode_tokens,
    _prepare_mm_inputs,
    _prepare_text_only_inputs,
    _unwrap_model,
    majority_vote,
    normalize_answer,
    shannon_entropy_nats,
    use_adapter,
)
from .generation_helpers import (
    GEN_CYCLE_CAPTION_PROMPT,
    GenerationQAPair,
    _jaccard_similarity,
    _per_candidate_diversity_scores,
    _soft_match,
    _yes_no_polarity,
)

logger = logging.getLogger(__name__)


def _infer_required_gen_pad_tokens(base_model: torch.nn.Module, pixel_gen_values) -> Optional[int]:
    try:
        coerce = getattr(base_model, "_coerce_pixel_gen_values", None)
        if callable(coerce):
            pv = coerce(pixel_gen_values)
        else:
            if torch.is_tensor(pixel_gen_values):
                pv = pixel_gen_values
            elif isinstance(pixel_gen_values, Sequence):
                flat = []
                for x in pixel_gen_values:
                    if torch.is_tensor(x):
                        flat.append(x)
                    elif isinstance(x, Sequence):
                        flat.extend([y for y in x if torch.is_tensor(y)])
                if not flat:
                    return None
                pv = torch.cat([t if t.ndim == 4 else t.unsqueeze(0) for t in flat], dim=0)
            else:
                return None

        if pv.ndim == 4:
            bsz = int(pv.shape[0])
            t_frames = 1
            h, w = int(pv.shape[-2]), int(pv.shape[-1])
        elif pv.ndim == 5:
            bsz = int(pv.shape[0])
            t_frames = int(pv.shape[2])
            h, w = int(pv.shape[-2]), int(pv.shape[-1])
        else:
            return None

        from visionllm.vargpt_qwen_v1_1.var_model.infinity.utils.dynamic_resolution import dynamic_resolution_h_w

        hw_ratio = float(h) / float(max(1, w))
        ratio_keys = list(dynamic_resolution_h_w.keys())
        if not ratio_keys:
            return None
        ratio_key = min(ratio_keys, key=lambda k: abs(float(k) - hw_ratio))

        vargpt_args = getattr(base_model, "vargpt_gen_args", None)
        pn = getattr(vargpt_args, "pn", "0.25M")
        if pn not in dynamic_resolution_h_w[ratio_key]:
            return None
        scale_schedule = dynamic_resolution_h_w[ratio_key][pn]["scales"]
        scale_schedule = [
            (min(int(pt), int(t_frames // 4 + 1)), int(ph), int(pw))
            for (pt, ph, pw) in scale_schedule
        ][:100]
        if not scale_schedule:
            return None

        training_seq_len = sum(int(pt * ph * pw) for (pt, ph, pw) in scale_schedule)
        first_scale_len = int(scale_schedule[0][0] * scale_schedule[0][1] * scale_schedule[0][2])
        x_blc_wo_prefix_len = max(0, training_seq_len - first_scale_len)
        l_end = int(x_blc_wo_prefix_len + 1)
        pad_mult = int(getattr(getattr(base_model, "vargpt_gen", None), "pad_to_multiplier", 1) or 1)
        padded_len = ((l_end + pad_mult - 1) // pad_mult) * pad_mult
        return int(bsz * padded_len)
    except Exception:
        return None


def _append_explicit_image_gen_tokens(
    base_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor],
    required_pad_tokens: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if required_pad_tokens is None or required_pad_tokens <= 0:
        return input_ids, attention_mask, labels

    special_tokens = getattr(getattr(base_model, "config", None), "special_tokens", None)
    if not isinstance(special_tokens, dict):
        return input_ids, attention_mask, labels

    start_id = special_tokens.get("image_gen_start_token_id", None)
    pad_id = special_tokens.get("image_gen_pad_token_id", None)
    end_id = special_tokens.get("image_gen_end_token_id", None)
    if pad_id is None:
        return input_ids, attention_mask, labels

    bsz = int(input_ids.shape[0])
    if bsz != 1:
        return input_ids, attention_mask, labels

    cur_pad = int((input_ids == int(pad_id)).sum().item())
    add_pad = max(0, int(required_pad_tokens - cur_pad))
    if add_pad == 0:
        return input_ids, attention_mask, labels

    append_ids = []
    has_start = (start_id is not None) and int((input_ids == int(start_id)).sum().item()) > 0
    has_end = (end_id is not None) and int((input_ids == int(end_id)).sum().item()) > 0
    if (start_id is not None) and (not has_start):
        append_ids.append(int(start_id))
    append_ids.extend([int(pad_id)] * add_pad)
    if (end_id is not None) and (not has_end):
        append_ids.append(int(end_id))
    if not append_ids:
        return input_ids, attention_mask, labels

    append_tensor = torch.tensor(append_ids, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0)
    input_ids = torch.cat([input_ids, append_tensor], dim=1)

    if attention_mask is not None:
        append_mask = torch.ones((1, append_tensor.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, append_mask], dim=1)

    if labels is not None:
        append_labels = torch.full((1, append_tensor.shape[1]), fill_value=-100, dtype=labels.dtype, device=labels.device)
        labels = torch.cat([labels, append_labels], dim=1)

    return input_ids, attention_mask, labels


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
        required_pad = _infer_required_gen_pad_tokens(base_model, [img_tensor])
        input_ids, attention_mask, labels = _append_explicit_image_gen_tokens(
            base_model=base_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            required_pad_tokens=required_pad,
        )

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


def _adapter_disabled(base_model: torch.nn.Module):
    if hasattr(base_model, "disable_adapter"):
        return base_model.disable_adapter()
    return contextlib.nullcontext()


def _text_embedding(
    model: torch.nn.Module,
    processor,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    base_model = _unwrap_model(model)
    was_training = base_model.training
    try:
        base_model.eval()
        chat_text = _build_text_only_chat(processor, text)
        text_inputs = _prepare_text_only_inputs(processor, device, chat_text)
        text_inputs["use_cache"] = False
        text_inputs["output_hidden_states"] = True

        with torch.no_grad():
            with _adapter_disabled(base_model):
                outputs = base_model(**text_inputs)

        hidden = outputs.hidden_states[-1]
        attention_mask = text_inputs.get("attention_mask")
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            embedding = hidden.mean(dim=1)
        return F.normalize(embedding.squeeze(0), dim=-1)
    finally:
        if was_training:
            base_model.train(True)


def text_embedding_similarity(
    model: torch.nn.Module,
    processor,
    text_a: str,
    text_b: str,
    device: torch.device,
) -> float:
    try:
        emb_a = _text_embedding(model, processor, text_a, device)
        emb_b = _text_embedding(model, processor, text_b, device)
        return float(torch.dot(emb_a, emb_b).item())
    except Exception:
        return _jaccard_similarity(text_a, text_b)


def generate_cycle_caption(
    model: torch.nn.Module,
    processor,
    image: Image.Image,
    device: torch.device,
    max_new_tokens: int = 96,
    solver_adapter_name: str = "solver",
) -> str:
    base_model = _unwrap_model(model)
    was_training = base_model.training
    try:
        base_model.eval()
        chat_text = _build_chat_text(processor, image, GEN_CYCLE_CAPTION_PROMPT)
        mm_inputs = _prepare_mm_inputs(processor, device, image, chat_text, model=model)
        with torch.no_grad():
            with use_adapter(base_model, solver_adapter_name):
                gen_ids = base_model.generate(
                    **mm_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.4,
                    top_p=1.0,
                )
        input_len = mm_inputs["input_ids"].shape[1]
        new_ids = gen_ids[0, input_len:]
        tokenizer = getattr(processor, "tokenizer", processor)
        caption = _decode_tokens(tokenizer, new_ids)
        return " ".join(str(caption or "").split())
    except Exception as e:
        logger.warning(f"[Rewards] Caption generation failed: {e}")
        return ""
    finally:
        if was_training:
            base_model.train(True)


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
    """Score a generated image by having the solver answer verification QA."""
    if not questions or not expected_answers:
        return 0.0, {"no_qa_pairs": True}

    base_model = _unwrap_model(model)
    was_training = base_model.training

    try:
        base_model.eval()

        qa_scores = []
        qa_logs = []
        for q, exp_a in zip(questions, expected_answers):
            from .prompts import build_solver_prompt
            solver_prompt = build_solver_prompt(q)
            chat_text = _build_chat_text(processor, image, solver_prompt)
            mm_inputs = _prepare_mm_inputs(
                processor, device, image, chat_text, model=model
            )

            answers = []
            raw_answers = []
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
                            input_len = mm_inputs["input_ids"].shape[1]
                            new_ids = gen_ids[0, input_len:]
                            answer_text = _decode_tokens(
                                getattr(processor, "tokenizer", processor),
                                new_ids,
                            )
                            from .utils import _parse_answer
                            answer = _parse_answer(answer_text)
                            raw_answers.append(answer_text)
                            answers.append(normalize_answer(answer))
                        except Exception:
                            raw_answers.append("")
                            answers.append("")

            if not answers or all(a == "" for a in answers):
                qa_scores.append(0.0)
                qa_logs.append(
                    {
                        "question": q,
                        "expected": normalize_answer(exp_a),
                        "status": "skipped",
                        "skip_reason": "empty_solver_answers",
                        "solver_answers_norm": answers,
                        "solver_outputs_raw": raw_answers,
                    }
                )
                continue

            entropy_nats, majority_fraction, majority_answer = _entropy_majority(answers)
            expected = normalize_answer(exp_a)
            match_score = _soft_match(majority_answer, expected)
            contradiction = 1.0 if (
                _yes_no_polarity(expected) != 0
                and _yes_no_polarity(majority_answer) != 0
                and _yes_no_polarity(expected) != _yes_no_polarity(majority_answer)
            ) else 0.0
            qa_score = 0.7 * float(match_score) + 0.3 * float(majority_fraction)
            qa_scores.append(float(qa_score))
            qa_logs.append(
                {
                    "question": q,
                    "expected": expected,
                    "status": "ok",
                    "majority_answer": majority_answer,
                    "majority_fraction": float(majority_fraction),
                    "entropy_nats": float(entropy_nats),
                    "match_score": float(match_score),
                    "combined_score": float(qa_score),
                    "contradiction": float(contradiction),
                    "solver_answers_norm": answers,
                    "solver_outputs_raw": raw_answers,
                }
            )

        reward = sum(qa_scores) / max(len(qa_scores), 1)
        details = {
            "num_qa": len(qa_scores),
            "mean_qa_score": reward,
            "qa_scores": qa_scores,
            "qa_logs": qa_logs,
            "contradiction_score": _mean(
                [float(log.get("contradiction", 0.0)) for log in qa_logs if str(log.get("status", "")) == "ok"]
            ),
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


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _entropy_majority(answers: List[str]) -> Tuple[float, float, str]:
    majority_answer, majority_count = majority_vote(answers)
    n = max(1, len([answer for answer in answers if str(answer or "").strip()]))
    majority_fraction = float(majority_count) / float(n)
    hist = {}
    for answer in answers:
        if not str(answer or "").strip():
            continue
        hist[answer] = hist.get(answer, 0) + 1
    probs = [float(count) / float(max(1, n)) for count in hist.values()] if hist else [1.0]
    entropy_nats = shannon_entropy_nats(probs)
    return float(entropy_nats), float(majority_fraction), str(majority_answer)


def cycle_reward(
    model: torch.nn.Module,
    processor,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    config,
    solver_adapter_name: str = "solver",
) -> Tuple[float, str]:
    caption = generate_cycle_caption(
        model=model,
        processor=processor,
        image=image,
        device=device,
        max_new_tokens=int(getattr(config, "max_new_tokens_caption", 96)),
        solver_adapter_name=solver_adapter_name,
    )
    caption_similarity = 0.0
    if caption:
        caption_similarity = text_embedding_similarity(
            model=model,
            processor=processor,
            text_a=prompt,
            text_b=caption,
            device=device,
        )
    image_text_similarity = embedding_similarity(
        model=model,
        processor=processor,
        image=image,
        text=prompt,
        device=device,
    )
    score = 0.5 * float(caption_similarity) + 0.5 * float(image_text_similarity)
    return _clamp01(score), caption


def score_generated_candidates(
    model: torch.nn.Module,
    processor,
    images: List[Image.Image],
    prompt: str,
    questions: List[str],
    expected_answers: List[str],
    device: torch.device,
    config,
    spec_quality: float = 1.0,
    solver_adapter_name: str = "solver",
) -> List[Dict[str, float]]:
    if not images:
        return []

    diversity_scores = _per_candidate_diversity_scores(images)
    positive_weight_sum = (
        float(getattr(config, "reward_spec_weight", 0.65))
        + float(getattr(config, "reward_cycle_weight", 0.20))
        + float(getattr(config, "reward_diversity_weight", 0.10))
    )
    if positive_weight_sum <= 0.0:
        positive_weight_sum = 1.0
    weight_spec = float(getattr(config, "reward_spec_weight", 0.65)) / positive_weight_sum
    weight_cycle = float(getattr(config, "reward_cycle_weight", 0.20)) / positive_weight_sum
    weight_diversity = float(getattr(config, "reward_diversity_weight", 0.10)) / positive_weight_sum

    scored_candidates: List[Dict[str, float]] = []
    for idx, image in enumerate(images):
        spec_score, spec_details = solver_verification_reward(
            model=model,
            processor=processor,
            image=image,
            questions=questions,
            expected_answers=expected_answers,
            device=device,
            num_samples=int(getattr(config, "num_solver_samples_spec", 3)),
            max_new_tokens=int(getattr(config, "max_new_tokens_solver", 64)),
            solver_adapter_name=solver_adapter_name,
        )
        contradiction_score = float(spec_details.get("contradiction_score", 0.0))
        qa_logs = list(spec_details.get("qa_logs", []))
        cycle_score, cycle_caption = cycle_reward(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            device=device,
            config=config,
            solver_adapter_name=solver_adapter_name,
        )
        diversity_score = float(diversity_scores[idx]) if idx < len(diversity_scores) else 0.0
        base_reward = (
            weight_spec * float(spec_score)
            + weight_cycle * float(cycle_score)
            + weight_diversity * float(diversity_score)
            - float(getattr(config, "reward_contradiction_weight", 0.20)) * contradiction_score
        )
        base_reward = _clamp01(base_reward)
        total_reward = _clamp01(float(spec_quality) * float(base_reward))
        qa_confidence = _mean(
            [float(log.get("majority_fraction", 0.0)) for log in qa_logs if str(log.get("status", "")) == "ok"]
        )
        mean_entropy_nats = _mean(
            [float(log.get("entropy_nats", 0.0)) for log in qa_logs if str(log.get("status", "")) == "ok"]
        )
        scored_candidates.append(
            {
                "candidate_idx": int(idx),
                "spec_score": float(spec_score),
                "contradiction_score": float(contradiction_score),
                "cycle_score": float(cycle_score),
                "cycle_caption": cycle_caption,
                "diversity_score": float(diversity_score),
                "base_reward": float(base_reward),
                "spec_quality": float(spec_quality),
                "total_reward": float(total_reward),
                "qa_confidence": float(qa_confidence),
                "mean_entropy_nats": float(mean_entropy_nats),
                "qa_logs": qa_logs,
            }
        )
    return scored_candidates


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
    scored = score_generated_candidates(
        model=model,
        processor=processor,
        images=[image],
        prompt=prompt,
        questions=questions,
        expected_answers=expected_answers,
        device=device,
        config=config,
        spec_quality=1.0,
    )
    if not scored:
        return 0.0, {}
    details = dict(scored[0])
    reward = float(details.get("total_reward", 0.0))
    return reward, details
