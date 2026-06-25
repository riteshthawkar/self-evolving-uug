#!/usr/bin/env python3
"""Sanitize BLIP-3o PEFT adapter folders in a self-evolving checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BLIP3O_ROOT = REPO_ROOT / "BLIP3o"
if str(BLIP3O_ROOT) not in sys.path:
    sys.path.insert(0, str(BLIP3O_ROOT))

from blip3o.train.self_evolving.checkpoint_adapters import sanitize_peft_adapter_dir  # noqa: E402


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


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

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
