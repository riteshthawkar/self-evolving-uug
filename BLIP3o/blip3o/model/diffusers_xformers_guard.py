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
_TRANSFORMERS_PATCHED = False


_TRANSFORMERS_UTILS_CONSTANTS = {
    "CONFIG_NAME": "config.json",
    "WEIGHTS_NAME": "pytorch_model.bin",
    "WEIGHTS_INDEX_NAME": "pytorch_model.bin.index.json",
    "TF2_WEIGHTS_NAME": "tf_model.h5",
    "TF2_WEIGHTS_INDEX_NAME": "tf_model.h5.index.json",
    "TF_WEIGHTS_NAME": "model.ckpt",
    "FLAX_WEIGHTS_NAME": "flax_model.msgpack",
    "FLAX_WEIGHTS_INDEX_NAME": "flax_model.msgpack.index.json",
    "SAFE_WEIGHTS_NAME": "model.safetensors",
    "SAFE_WEIGHTS_INDEX_NAME": "model.safetensors.index.json",
}


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


def patch_transformers_utils_for_diffusers() -> None:
    """Restore removed transformers.utils constants expected by diffusers.

    Some newer Transformers builds no longer re-export legacy checkpoint-name
    constants from ``transformers.utils``. Older Diffusers releases still import
    them there when loading pipelines. Setting missing constants keeps the local
    import contract intact without pinning the whole environment.
    """
    global _TRANSFORMERS_PATCHED
    if _TRANSFORMERS_PATCHED:
        return
    _TRANSFORMERS_PATCHED = True

    try:
        import transformers.utils as transformers_utils
    except Exception as exc:
        logging.warning(
            "Failed to inspect transformers.utils for diffusers compatibility: "
            "%s: %s",
            type(exc).__name__,
            exc,
        )
        return

    patched = []
    for name, value in _TRANSFORMERS_UTILS_CONSTANTS.items():
        if not hasattr(transformers_utils, name):
            setattr(transformers_utils, name, value)
            patched.append(name)

    if patched:
        logging.warning(
            "Patched transformers.utils constants required by diffusers: %s",
            ", ".join(sorted(patched)),
        )


def apply_diffusers_import_guards() -> None:
    """Apply BLIP3o compatibility guards before importing diffusers modules."""
    patch_transformers_utils_for_diffusers()
    disable_broken_xformers_for_diffusers()


def get_xformers_import_error() -> Optional[BaseException]:
    return _IMPORT_ERROR
