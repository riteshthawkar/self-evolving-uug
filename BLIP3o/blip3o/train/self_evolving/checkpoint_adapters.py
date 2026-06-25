"""Utilities for packaging and loading PEFT adapter checkpoints.

Self-evolving BLIP-3o can contain nested PEFT modules: the outer VLM adapters
for solver/proposer/generator and an inner DiT LoRA adapter. PEFT's generic
``save_pretrained(selected_adapters=...)`` can serialize nested adapter tensors
into the outer role adapter directory. These helpers keep role checkpoints
limited to tensors described by their own adapter_config.json.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except Exception:  # pragma: no cover - safetensors is expected with PEFT
    safe_load_file = None
    safe_save_file = None


LogFn = Optional[Callable[[str], None]]


def find_adapter_config_dir(adapter_root: str | Path) -> Path:
    """Return the directory containing adapter_config.json.

    PEFT sometimes writes selected adapters into a nested directory named after
    the adapter, e.g. solver/default/adapter_config.json.
    """

    root = Path(adapter_root)
    if (root / "adapter_config.json").is_file():
        return root
    for child in sorted(root.iterdir()) if root.is_dir() else ():
        if child.is_dir() and (child / "adapter_config.json").is_file():
            return child
    raise FileNotFoundError(f"adapter_config.json not found under {root}")


def _load_config(adapter_dir: Path) -> Dict:
    with (adapter_dir / "adapter_config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalise_targets(target_modules) -> Tuple[str, ...]:
    if target_modules is None:
        return tuple()
    if isinstance(target_modules, str):
        return (target_modules,)
    return tuple(str(target) for target in target_modules if str(target))


def _matches_target(key: str, targets: Iterable[str]) -> bool:
    return any(f".{target}." in key or key.endswith(f".{target}") for target in targets)


def _split_state_dict_by_config(state_dict: Dict[str, torch.Tensor], adapter_dir: Path):
    targets = _normalise_targets(_load_config(adapter_dir).get("target_modules"))
    if not targets:
        return dict(state_dict), {}, targets
    kept = {key: value for key, value in state_dict.items() if _matches_target(key, targets)}
    removed = {key: value for key, value in state_dict.items() if key not in kept}
    return kept, removed, targets


def _write_report(adapter_dir: Path, report: Dict) -> None:
    with (adapter_dir / "adapter_packaging_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)


def sanitize_peft_adapter_dir(
    adapter_root: str | Path,
    *,
    in_place: bool,
    backup: bool = False,
    log: LogFn = None,
) -> Path:
    """Filter adapter weights to match adapter_config target_modules.

    When ``in_place`` is false, a temporary sanitized copy is returned and the
    original checkpoint is left untouched. When true, the checkpoint is rewritten
    only if extra tensors are found; optional backup preserves the original file.
    """

    adapter_dir = find_adapter_config_dir(adapter_root)
    weight_path = adapter_dir / "adapter_model.safetensors"
    weight_kind = "safetensors"
    if not weight_path.is_file():
        weight_path = adapter_dir / "adapter_model.bin"
        weight_kind = "bin"
    if not weight_path.is_file():
        return adapter_dir

    if weight_kind == "safetensors":
        if safe_load_file is None or safe_save_file is None:
            raise ImportError("safetensors is required to sanitize adapter_model.safetensors")
        state_dict = safe_load_file(str(weight_path), device="cpu")
    else:
        state_dict = torch.load(weight_path, map_location="cpu")

    kept, removed, targets = _split_state_dict_by_config(state_dict, adapter_dir)
    report = {
        "adapter_dir": str(adapter_dir),
        "weight_file": weight_path.name,
        "target_modules": list(targets),
        "total_tensors": len(state_dict),
        "kept_tensors": len(kept),
        "removed_tensors": len(removed),
        "removed_examples": sorted(removed)[:20],
    }

    if not removed:
        return adapter_dir
    if not kept:
        raise RuntimeError(
            f"Refusing to sanitize {adapter_dir}: all {len(state_dict)} tensors would be removed."
        )

    destination = adapter_dir
    destination_weight = weight_path
    if not in_place:
        destination = Path(tempfile.mkdtemp(prefix="blip3o_peft_adapter_"))
        shutil.copytree(adapter_dir, destination, dirs_exist_ok=True)
        destination_weight = destination / weight_path.name
        report["source_adapter_dir"] = str(adapter_dir)
        report["adapter_dir"] = str(destination)
    elif backup:
        backup_path = weight_path.with_suffix(weight_path.suffix + ".mixed.bak")
        if not backup_path.exists():
            shutil.copy2(weight_path, backup_path)
        report["backup_file"] = str(backup_path)

    if weight_kind == "safetensors":
        safe_save_file(kept, str(destination_weight))
    else:
        torch.save(kept, destination_weight)
    _write_report(destination, report)

    if log is not None:
        log(
            f"Sanitized PEFT adapter {adapter_dir}: removed {len(removed)} extra tensors, "
            f"kept {len(kept)}."
        )
    return destination


def prepare_peft_adapter_dir_for_loading(adapter_root: str | Path, log: LogFn = None) -> Path:
    """Return a load-safe adapter directory without mutating the source checkpoint."""

    return sanitize_peft_adapter_dir(adapter_root, in_place=False, backup=False, log=log)
