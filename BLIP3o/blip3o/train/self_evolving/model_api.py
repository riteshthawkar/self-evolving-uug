"""Model API discovery, introspection, and loading utilities."""

import inspect
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from .generation_helpers import _signature_param_names
from .utils import _unwrap_model


def _iter_wrapper_objects(root_obj) -> List[Tuple[str, object]]:
    """Return model wrapper chain candidates where custom generation APIs may live."""
    results: List[Tuple[str, object]] = []
    queue: List[Tuple[str, object]] = [("model", root_obj)]
    seen = set()
    attrs = ("module", "base_model", "model", "language_model", "backbone")

    while queue:
        path, obj = queue.pop(0)
        if obj is None:
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        results.append((path, obj))
        for attr in attrs:
            child = getattr(obj, attr, None)
            if child is not None and child is not obj:
                queue.append((f"{path}.{attr}", child))
    return results


def _supports_kwarg_anywhere(model, kwarg_name: str) -> bool:
    try:
        wrappers = _iter_wrapper_objects(model)
    except Exception:
        wrappers = [("model", model)]
    for _, obj in wrappers:
        for fn_name in ("generate", "prepare_inputs_for_generation", "forward"):
            names, has_var_kw = _signature_param_names(obj, fn_name)
            if has_var_kw or kwarg_name in names:
                return True
    return False


def _adapt_mm_generate_inputs(model, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize multimodal kwargs across BLIP3o/Qwen wrappers."""
    out = dict(inputs)
    supports_images = _supports_kwarg_anywhere(model, "images")
    supports_pixel_values = _supports_kwarg_anywhere(model, "pixel_values")
    supports_grid_thw = _supports_kwarg_anywhere(model, "grid_thw")
    supports_image_grid_thw = _supports_kwarg_anywhere(model, "image_grid_thw")

    if "images" in out and (not supports_images) and supports_pixel_values and "pixel_values" not in out:
        out["pixel_values"] = out.pop("images")
    if "pixel_values" in out and (not supports_pixel_values) and supports_images and "images" not in out:
        out["images"] = out.pop("pixel_values")

    if "image_grid_thw" in out and (not supports_image_grid_thw) and supports_grid_thw and "grid_thw" not in out:
        out["grid_thw"] = out.pop("image_grid_thw")
    if "grid_thw" in out and (not supports_grid_thw) and supports_image_grid_thw and "image_grid_thw" not in out:
        out["image_grid_thw"] = out.pop("grid_thw")
    return out


def _parse_unused_model_kwargs_from_error(exc: Exception) -> List[str]:
    msg = str(exc)
    m = re.search(r"model_kwargs` are not used by the model: \[(.*?)\]", msg)
    if not m:
        return []
    raw = m.group(1).strip()
    if not raw:
        return []
    parts = [p.strip().strip("'").strip('"') for p in raw.split(",")]
    return [p for p in parts if p]


def _collect_image_token_ids(model) -> List[int]:
    ids: List[int] = []

    def _collect_from_cfg(cfg):
        if cfg is None:
            return
        for name in ("image_token_id", "image_token_index", "vision_token_id"):
            value = getattr(cfg, name, None)
            if isinstance(value, int):
                ids.append(int(value))
            elif isinstance(value, (list, tuple)):
                for v in value:
                    if isinstance(v, int):
                        ids.append(int(v))

    model_ref = _unwrap_model(model)
    cfg = getattr(model_ref, "config", None)
    _collect_from_cfg(cfg)
    _collect_from_cfg(getattr(cfg, "text_config", None) if cfg is not None else None)
    return sorted(set(ids))


def _count_image_tokens_in_inputs(input_ids: torch.Tensor, image_token_ids: List[int]) -> int:
    if input_ids is None or not torch.is_tensor(input_ids) or not image_token_ids:
        return 0
    total = 0
    for tok_id in image_token_ids:
        total += int((input_ids == int(tok_id)).sum().item())
    return int(total)


def _find_generation_callable(root_obj) -> Tuple[Optional[str], Optional[object], Optional[str], List[str]]:
    """Find `generate_images`/`generate_image` across nested wrapper objects."""
    inspected: List[str] = []
    for path, obj in _iter_wrapper_objects(root_obj):
        inspected.append(f"{path}:{type(obj).__name__}")
        for name in ("generate_images", "generate_image"):
            fn = getattr(obj, name, None)
            if callable(fn):
                return name, obj, path, inspected
    return None, None, None, inspected


def _find_callable_object(root_obj, callable_name: str) -> Tuple[Optional[object], Optional[str]]:
    try:
        wrappers = _iter_wrapper_objects(root_obj)
    except Exception:
        wrappers = [("model", root_obj)]
    for path, obj in wrappers:
        fn = getattr(obj, callable_name, None)
        if callable(fn):
            return obj, path
    return None, None


def _extract_tokenizer_from_processor(processor):
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        return tok
    mm_proc = getattr(processor, "multimodal_processor", None)
    if mm_proc is not None:
        tok = getattr(mm_proc, "tokenizer", None)
        if tok is not None:
            return tok
    # If the processor IS a tokenizer (BLIP3o case), return it directly.
    if hasattr(processor, "encode") and hasattr(processor, "decode"):
        return processor
    return None


def _build_compat_config(model_name: str, expected_model_type: Optional[str]):
    """Build a compatibility config for explicit BLIP3o class loading."""
    try:
        from transformers import AutoConfig
    except Exception:
        return None

    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None

    if expected_model_type and getattr(cfg, "model_type", None) != expected_model_type:
        try:
            setattr(cfg, "model_type", str(expected_model_type))
        except Exception:
            pass

    # Some checkpoints keep hidden size nested under text_config.
    text_cfg = getattr(cfg, "text_config", None)
    if not hasattr(cfg, "hidden_size"):
        hidden_size = None
        if text_cfg is not None:
            hidden_size = getattr(text_cfg, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(text_cfg, "d_model", None)
        if hidden_size is not None:
            try:
                setattr(cfg, "hidden_size", int(hidden_size))
            except Exception:
                pass
    return cfg


def _load_from_explicit_class(
    cls,
    model_name: str,
    *,
    torch_dtype: torch.dtype,
    device_map,
    attn_implementation: Optional[str],
):
    errors: List[str] = []
    base_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }

    attempts: List[Dict[str, object]] = []
    if attn_implementation is not None:
        attempts.append({**base_kwargs, "attn_implementation": attn_implementation})
    attempts.append(base_kwargs)

    expected_model_type = getattr(getattr(cls, "config_class", None), "model_type", None)
    compat_cfg = _build_compat_config(model_name, expected_model_type)

    for kwargs in attempts:
        if compat_cfg is not None:
            try:
                return cls.from_pretrained(model_name, config=compat_cfg, **kwargs)
            except Exception as exc:
                errors.append(f"{cls.__name__}(config+{kwargs}): {repr(exc)}")
        try:
            return cls.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            errors.append(f"{cls.__name__}({kwargs}): {repr(exc)}")

    raise RuntimeError(" | ".join(errors))


def _resolve_model_type_hint(model_name: str) -> str:
    """Best-effort model type probe to detect BLIP3o checkpoints from local paths."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return str(getattr(cfg, "model_type", "") or "").strip().lower()
    except Exception:
        return ""


def _load_blip3o_model(
    model_name: str,
    *,
    torch_dtype: torch.dtype,
    device_map,
    attn_implementation: Optional[str] = None,
):
    """
    Load model using native BLIP3o inference class when possible.

    **Design choice — InferenceLM over CausalLM:**

    We intentionally load ``blip3oQwenForInferenceLM`` instead of
    ``blip3oQwenForCausalLM``.  The CausalLM wrapper overrides
    ``generate(...)`` with custom input preparation via
    ``prepare_inputs_labels_for_understanding()`` before delegating to
    ``super().generate()``.  Note: diffusion decoding is NOT inside
    ``generate()`` — it lives in a separate ``generate_image()`` method.

    The reason we avoid CausalLM is that its ``forward()`` override
    injects multimodal fusion logic (vision-token interleaving) that
    can conflict with our training loop, where we need fine-grained
    control over:
      1. Policy-gradient and DPO loss computation (raw forward + logits)
      2. LoRA adapter switching mid-forward (``use_adapter`` context)
      3. Separate diffusion-decoding via ``_blip3o_diffusion_pipe``

    InferenceLM provides a thinner wrapper that delegates to standard
    HuggingFace ``forward()`` / ``generate()`` with minimal overrides.

    Training still works correctly because only LoRA adapters (or the full
    model in non-LoRA mode) receive gradient updates — the model class
    itself doesn't need to be the "training" variant.

    For non-BLIP3o models, we fall back to AutoModelForCausalLM.
    """
    model_name_l = (model_name or "").lower()
    model_type_hint = _resolve_model_type_hint(model_name)
    is_blip3o_target = ("blip3o" in model_name_l) or model_type_hint.startswith("blip3o")

    explicit_errors: List[str] = []

    # Official BLIP3o inference loader target.
    try:
        from blip3o.model.language_model.blip3o_qwen_inference import blip3oQwenForInferenceLM
    except Exception as exc:
        blip3oQwenForInferenceLM = None
        explicit_errors.append(f"Import blip3oQwenForInferenceLM failed: {repr(exc)}")

    if blip3oQwenForInferenceLM is not None:
        try:
            return _load_from_explicit_class(
                blip3oQwenForInferenceLM,
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
        except Exception as exc:
            explicit_errors.append(f"blip3oQwenForInferenceLM: {exc}")

    if is_blip3o_target:
        details = " | ".join(explicit_errors) if explicit_errors else "no explicit errors captured"
        raise RuntimeError(
            f"Failed to load BLIP3o inference model '{model_name}'. Details: {details}"
        )

    from transformers import AutoModelForCausalLM

    errors: Dict[str, str] = {}
    auto_classes = [AutoModelForCausalLM]

    try:
        from transformers import AutoModelForVision2Seq
        auto_classes.insert(0, AutoModelForVision2Seq)
    except ImportError:
        pass
    try:
        from transformers import AutoModelForImageTextToText
        auto_classes.insert(0, AutoModelForImageTextToText)
    except ImportError:
        pass

    attempts: List[Dict[str, object]] = []
    base_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if attn_implementation is not None:
        attempts.append({**base_kwargs, "attn_implementation": attn_implementation})
    attempts.append(base_kwargs)

    for cls in auto_classes:
        for kwargs in attempts:
            try:
                return cls.from_pretrained(model_name, **kwargs)
            except Exception as exc:
                errors[f"{cls.__name__}({kwargs})"] = repr(exc)

    details = "; ".join(f"{n}: {e}" for n, e in errors.items())
    if explicit_errors:
        details = f"{details}; explicit: {' | '.join(explicit_errors)}"
    raise RuntimeError(f"Failed to load model '{model_name}': {details}")
