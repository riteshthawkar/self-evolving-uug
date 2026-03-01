#!/usr/bin/env python3
# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


def _unique_non_empty(vals: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in vals:
        s = str(v or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _default_role_mapping(role_adapters_csv: str, default_adapter: str) -> Dict[str, str]:
    available = _unique_non_empty(role_adapters_csv.split(","))
    if not available:
        available = ["proposer", "solver", "generator"]
    role_to_adapter: Dict[str, str] = {}
    fallback = str(default_adapter or "").strip() or available[0]
    for role in ("proposer", "solver", "generator"):
        if role in available:
            role_to_adapter[role] = role
        elif role == "generator" and "default" in available:
            role_to_adapter[role] = "default"
        else:
            role_to_adapter[role] = fallback
    return role_to_adapter


def _matches_adapter_key(key: str, adapter_name: str) -> bool:
    name = str(adapter_name or "").strip()
    if not name:
        return False
    k = str(key)
    return (f".{name}." in k) or (f"lora_{name}" in k) or (
        name == "default" and ("lora_" in k and ".default." in k)
    )


def _resolve_checkpoint_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        ckpts = sorted(p.glob("step_*.pt"))
        if ckpts:
            return ckpts[-1]
    raise FileNotFoundError(f"Checkpoint not found: {p}")


def _infer_output_dir(checkpoint_file: Path) -> Path:
    return checkpoint_file.with_name(f"{checkpoint_file.stem}_lora")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export proposer/solver/generator role-wise LoRA files from a BAGEL step checkpoint."
        )
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to step_XXXXXX.pt or a checkpoints directory.",
    )
    p.add_argument(
        "--output_dir",
        default="",
        help="Optional output directory. Defaults to sibling step_XXXXXX_lora.",
    )
    p.add_argument(
        "--lora_role_adapters_csv",
        default="proposer,solver,generator",
        help="Comma-separated adapter names available in checkpoint.",
    )
    p.add_argument(
        "--lora_default_adapter",
        default="proposer",
        help="Default adapter fallback if role-specific one is not present.",
    )
    p.add_argument(
        "--role_to_adapter_json",
        default="",
        help='Optional explicit mapping JSON, e.g. \'{"proposer":"proposer","solver":"solver","generator":"generator"}\'.',
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_dir if it already exists.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    ckpt = _resolve_checkpoint_path(args.checkpoint)
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt}")

    model_state = payload.get("model_state")
    if not isinstance(model_state, dict) or not model_state:
        raise RuntimeError(f"Checkpoint has no model_state: {ckpt}")

    if str(args.role_to_adapter_json or "").strip():
        role_to_adapter = json.loads(str(args.role_to_adapter_json))
        if not isinstance(role_to_adapter, dict):
            raise RuntimeError("--role_to_adapter_json must parse to a JSON object.")
        role_to_adapter = {
            str(k).strip(): str(v).strip()
            for k, v in role_to_adapter.items()
            if str(k).strip() and str(v).strip()
        }
    else:
        role_to_adapter = _default_role_mapping(
            role_adapters_csv=str(args.lora_role_adapters_csv or ""),
            default_adapter=str(args.lora_default_adapter or ""),
        )

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir or "").strip()
        else _infer_output_dir(ckpt)
    )
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output dir already exists: {out_dir}. Use --overwrite to replace."
            )
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    files_meta: Dict[str, Dict[str, object]] = {}
    for role, adapter in sorted(role_to_adapter.items()):
        adapter = str(adapter)
        role_state = {
            k: v
            for k, v in model_state.items()
            if _matches_adapter_key(str(k), adapter)
        }
        role_file = out_dir / f"role_{role}.pt"
        torch.save(
            {
                "role": str(role),
                "adapter_name": str(adapter),
                "state_dict": role_state,
            },
            role_file,
        )
        files_meta[str(role)] = {
            "file": role_file.name,
            "adapter_name": str(adapter),
            "tensor_count": int(len(role_state)),
        }

    manifest = {
        "source_checkpoint": str(ckpt),
        "step": int(payload.get("step", -1)),
        "policy_update_method": str(payload.get("policy_update_method", "")),
        "role_to_adapter": role_to_adapter,
        "files": files_meta,
    }
    with (out_dir / "adapter_roles.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[export_lora] source: {ckpt}")
    print(f"[export_lora] output: {out_dir}")
    for role, meta in sorted(files_meta.items()):
        print(
            f"[export_lora] role={role} adapter={meta['adapter_name']} tensors={meta['tensor_count']} file={meta['file']}"
        )


if __name__ == "__main__":
    main()
