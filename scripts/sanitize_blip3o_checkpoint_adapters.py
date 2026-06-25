#!/usr/bin/env python3
"""Sanitize BLIP-3o PEFT adapter folders in a self-evolving checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ADAPTERS = REPO_ROOT / "BLIP3o" / "blip3o" / "train" / "self_evolving" / "checkpoint_adapters.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=["solver", "proposer", "generator", "dit_lora"],
        help="Adapter subdirectories to check.",
    )
    parser.add_argument("--in-place", action="store_true", help="Rewrite adapter weights in the checkpoint.")
    parser.add_argument("--backup", action="store_true", help="Keep .mixed.bak copies before rewriting.")
    return parser.parse_args()


def _load_sanitize_peft_adapter_dir():
    spec = importlib.util.spec_from_file_location("blip3o_checkpoint_adapters", CHECKPOINT_ADAPTERS)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load checkpoint adapter helpers from {CHECKPOINT_ADAPTERS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.sanitize_peft_adapter_dir


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    sanitize_peft_adapter_dir = _load_sanitize_peft_adapter_dir()
    for adapter in args.adapters:
        adapter_root = checkpoint_dir / adapter
        if not adapter_root.exists():
            print(f"[skip] {adapter}: not present")
            continue
        sanitized = sanitize_peft_adapter_dir(
            adapter_root,
            in_place=bool(args.in_place),
            backup=bool(args.backup),
            log=print,
        )
        action = "rewritten" if args.in_place else "checked"
        print(f"[ok] {adapter}: {action} via {sanitized}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
