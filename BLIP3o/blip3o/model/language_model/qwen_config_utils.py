"""Configuration compatibility helpers for BLIP3o Qwen wrappers."""

from __future__ import annotations

from typing import Any, Optional


def _get_cfg_value(cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _set_cfg_value(cfg: Any, name: str, value: Any) -> None:
    if cfg is None or value is None:
        return
    if isinstance(cfg, dict):
        cfg[name] = value
    else:
        setattr(cfg, name, value)


def _first_int_from_config(cfg: Any, names: tuple[str, ...]) -> Optional[int]:
    for name in names:
        value = _get_cfg_value(cfg, name)
        if isinstance(value, int):
            return int(value)
        if value is not None:
            try:
                return int(value)
            except Exception:
                continue
    return None


def ensure_qwen_vl_config_compat(config: Any) -> Any:
    """Mirror nested text-config fields expected by BLIP3o's Qwen wrappers.

    Recent Transformers Qwen2.5-VL configs keep language-model fields under
    ``text_config``. The original BLIP3o wrappers expect ``hidden_size`` and
    ``vocab_size`` on the root config. Mirroring those fields preserves the
    checkpoint architecture while making the wrapper robust across Transformers
    releases.
    """
    text_cfg = _get_cfg_value(config, "text_config")

    hidden_size = _first_int_from_config(config, ("hidden_size", "d_model", "embed_dim"))
    if hidden_size is None:
        hidden_size = _first_int_from_config(text_cfg, ("hidden_size", "d_model", "embed_dim"))
    if hidden_size is not None:
        _set_cfg_value(config, "hidden_size", hidden_size)

    vocab_size = _first_int_from_config(config, ("vocab_size", "vocabulary_size"))
    if vocab_size is None:
        vocab_size = _first_int_from_config(text_cfg, ("vocab_size", "vocabulary_size"))
    if vocab_size is not None:
        _set_cfg_value(config, "vocab_size", vocab_size)

    return config
