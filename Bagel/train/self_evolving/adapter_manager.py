# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from typing import Dict, Iterable, List, Optional, Tuple

import torch


ROLE_PROPOSER = "proposer"
ROLE_SOLVER = "solver"
ROLE_GENERATOR = "generator"
_ALL_ROLES = (ROLE_PROPOSER, ROLE_SOLVER, ROLE_GENERATOR)


def _normalize_name(name: str) -> str:
    return str(name or "").strip().lower()


def _unique_non_empty(vals: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for val in vals:
        norm = _normalize_name(val)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _active_adapter_name(model) -> Optional[str]:
    active = getattr(model, "active_adapter", None)
    if isinstance(active, (list, tuple)):
        return str(active[0]) if active else None
    if active is None:
        return None
    return str(active)


def setup_lora_role_adapters(
    language_model,
    *,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    role_adapter_names: List[str],
    default_adapter: str,
) -> Tuple[torch.nn.Module, Dict[str, str], List[str]]:
    """Apply PEFT LoRA to BAGEL language model and create role adapters."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PEFT is required for LoRA self-evolving training. "
            "Install with `pip install peft` in the BAGEL environment."
        ) from exc

    role_adapter_names = _unique_non_empty(role_adapter_names)
    default_adapter = _normalize_name(default_adapter)
    if not role_adapter_names:
        role_adapter_names = [ROLE_PROPOSER, ROLE_SOLVER, ROLE_GENERATOR]
    if not default_adapter:
        default_adapter = ROLE_PROPOSER

    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=max(1, int(lora_rank)),
        lora_alpha=max(1, int(lora_alpha)),
        lora_dropout=max(0.0, float(lora_dropout)),
        target_modules=list(target_modules),
        inference_mode=False,
    )

    if hasattr(language_model, "peft_config"):
        peft_model = language_model
    else:
        peft_model = get_peft_model(language_model, lcfg)

    if hasattr(peft_model, "add_adapter"):
        existing = set(getattr(peft_model, "peft_config", {}).keys())
        for adapter_name in role_adapter_names:
            if adapter_name in existing or adapter_name == "default":
                continue
            peft_model.add_adapter(adapter_name, lcfg)

    available = list(getattr(peft_model, "peft_config", {}).keys())
    if not available:
        available = ["default"]
    fallback = default_adapter if default_adapter in available else available[0]

    role_to_adapter: Dict[str, str] = {}
    for role in _ALL_ROLES:
        if role in available:
            role_to_adapter[role] = role
        elif role == ROLE_GENERATOR and "default" in available:
            role_to_adapter[role] = "default"
        else:
            role_to_adapter[role] = fallback

    if hasattr(peft_model, "set_adapter"):
        try:
            peft_model.set_adapter(role_to_adapter.get(default_adapter, fallback))
        except Exception:
            peft_model.set_adapter(fallback)

    return peft_model, role_to_adapter, available


@contextlib.contextmanager
def use_adapter(model, adapter_name: Optional[str]):
    """Temporarily switch active adapter if PEFT adapters are available."""
    if not hasattr(model, "set_adapter"):
        yield
        return

    if adapter_name is None:
        disable_adapter = getattr(model, "disable_adapter", None)
        if callable(disable_adapter):
            with disable_adapter():
                yield
            return
        yield
        return

    if not adapter_name:
        yield
        return

    prev = _active_adapter_name(model)
    try:
        model.set_adapter(adapter_name)
    except Exception:
        yield
        return

    try:
        yield
    finally:
        if prev:
            try:
                model.set_adapter(prev)
            except Exception:
                pass


def collect_adapter_parameters(model, adapter_name: Optional[str]) -> List[torch.nn.Parameter]:
    """Collect trainable LoRA parameters for a specific adapter."""
    if not adapter_name or not hasattr(model, "named_parameters"):
        return [p for p in model.parameters() if p.requires_grad]

    adapter_name = str(adapter_name)
    params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad and "lora_" not in name:
            continue
        is_named_adapter = (f".{adapter_name}." in name) or (f"lora_{adapter_name}" in name)
        is_default = adapter_name == "default" and ("lora_" in name and ".default." in name)
        if is_named_adapter or is_default:
            params.append(param)

    seen = set()
    unique_params: List[torch.nn.Parameter] = []
    for p in params:
        ptr = p.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        unique_params.append(p)
    return unique_params
