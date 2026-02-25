# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Optional, Union

import torch


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_DTYPE_ALIASES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def is_rocm_runtime() -> bool:
    return bool(getattr(torch.version, "hip", None))


def _normalize_device_type(device: Union[str, torch.device]) -> str:
    if isinstance(device, torch.device):
        return str(device.type)
    return str(device)


def _read_env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    val = str(raw).strip().lower()
    if val in _TRUE_VALUES:
        return True
    if val in _FALSE_VALUES:
        return False
    return bool(default)


def resolve_lowp_dtype(device: Union[str, torch.device]) -> Optional[torch.dtype]:
    device_type = _normalize_device_type(device)
    if device_type != "cuda":
        return None

    raw = str(os.environ.get("BAGEL_AUTOCAST_DTYPE", "auto")).strip().lower()
    if raw and raw != "auto":
        if raw in {"off", "none", "disable", "disabled"}:
            return None
        mapped = _DTYPE_ALIASES.get(raw)
        if mapped is not None:
            # fp32 implies no mixed precision autocast.
            if mapped == torch.float32:
                return None
            return mapped

    if is_rocm_runtime():
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def lowp_dtype_for_tensor(tensor: torch.Tensor) -> torch.dtype:
    resolved = resolve_lowp_dtype(tensor.device)
    if resolved is None:
        return tensor.dtype
    return resolved


def cast_to_lowp(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
        return tensor
    target = lowp_dtype_for_tensor(tensor)
    if tensor.dtype == target:
        return tensor
    return tensor.to(target)


def autocast_context(device: Union[str, torch.device], *, enabled: bool = True):
    if not bool(enabled):
        return nullcontext()
    if _read_env_flag("BAGEL_DISABLE_AUTOCAST", default=False):
        return nullcontext()

    device_type = _normalize_device_type(device)
    dtype = resolve_lowp_dtype(device_type)
    if device_type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device_type, enabled=True, dtype=dtype)
