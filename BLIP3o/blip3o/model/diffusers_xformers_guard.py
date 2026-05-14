"""Compatibility guard for optional xFormers acceleration in diffusers.

Some cluster images contain an xformers package that is importable by name but
ABI-incompatible with the active PyTorch/CUDA/Triton stack. Diffusers checks for
package presence and then imports ``xformers.ops`` inside attention modules; a
broken wheel can therefore crash BLIP3o at import time before training starts.

This guard verifies that ``xformers.ops`` actually initializes. If it does not,
we mark xFormers unavailable in diffusers so the standard PyTorch attention
path is used consistently.
"""

from __future__ import annotations

import logging
from typing import Optional

_CHECKED = False
_IMPORT_ERROR: Optional[BaseException] = None


def _mark_diffusers_xformers_unavailable(exc: BaseException) -> None:
    def _xformers_unavailable() -> bool:
        return False

    try:
        from diffusers.utils import import_utils as diffusers_import_utils

        diffusers_import_utils._xformers_available = False
        diffusers_import_utils.is_xformers_available = _xformers_unavailable
    except Exception as patch_exc:
        logging.warning(
            "Failed to patch diffusers import_utils after xFormers import "
            "failure (%s: %s): %s: %s",
            type(exc).__name__,
            exc,
            type(patch_exc).__name__,
            patch_exc,
        )

    try:
        import diffusers.utils as diffusers_utils

        diffusers_utils.is_xformers_available = _xformers_unavailable
    except Exception as patch_exc:
        logging.warning(
            "Failed to patch diffusers.utils after xFormers import failure "
            "(%s: %s): %s: %s",
            type(exc).__name__,
            exc,
            type(patch_exc).__name__,
            patch_exc,
        )


def disable_broken_xformers_for_diffusers() -> None:
    """Disable diffusers xFormers integration if ``xformers.ops`` is broken."""
    global _CHECKED, _IMPORT_ERROR
    if _CHECKED:
        return
    _CHECKED = True

    try:
        import xformers.ops  # noqa: F401
        return
    except Exception as exc:
        _IMPORT_ERROR = exc
        logging.warning(
            "xFormers ops are unavailable for diffusers; using PyTorch "
            "attention backends. This is expected when xformers was built for "
            "a different PyTorch/CUDA/Triton stack. Import failure: %s: %s",
            type(exc).__name__,
            exc,
        )
        _mark_diffusers_xformers_unavailable(exc)


def get_xformers_import_error() -> Optional[BaseException]:
    return _IMPORT_ERROR
