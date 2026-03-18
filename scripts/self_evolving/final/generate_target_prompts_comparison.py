#!/usr/bin/env python3
"""Generate selected qualitative prompts for BLIP3o, BAGEL, and VARGPT.

This wrapper runs both base and updated model generation for each backbone.
It writes prompt metadata files automatically and dispatches each backbone's
existing generation script with consistent prompt subsets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Set


@dataclass(frozen=True)
class PromptSpec:
    item_id: int
    text: str
    backbones: Set[str]


PROMPTS: Sequence[PromptSpec] = (
    PromptSpec(4, "A closed red umbrella leaning against an open blue umbrella", {"blip3o"}),
    PromptSpec(11, "Three red roses and two white roses in a vase", {"blip3o"}),
    PromptSpec(13, "A small cactus to the left of a tall sunflower in a garden", {"bagel"}),
    PromptSpec(19, "A blue rose in a red vase on a yellow tablecloth", {"bagel"}),
    PromptSpec(6, "A ripe yellow banana and an unripe green banana on a plate", {"vargpt"}),
    PromptSpec(8, "Four identical white coffee mugs in a row on a counter", {"vargpt"}),
    PromptSpec(
        22,
        "A person wearing a red shirt and blue pants holding a yellow umbrella",
        {"blip3o", "bagel", "vargpt"},
    ),
    PromptSpec(
        26,
        "A large orange cat sitting on top of a small purple box",
        {"blip3o", "bagel", "vargpt"},
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _fmt_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run(cmd: Sequence[str], cwd: Path | None = None, env: dict | None = None, dry_run: bool = False) -> None:
    print(f"[RUN] {_fmt_cmd(cmd)}")
    if cwd is not None:
        print(f"      cwd={cwd}")
    if dry_run:
        return
    subprocess.run(list(cmd), check=True, cwd=str(cwd) if cwd else None, env=env)


def _select_prompts(backbone: str) -> List[PromptSpec]:
    return [p for p in PROMPTS if backbone in p.backbones]


def _write_blip_csv(path: Path, prompts: Iterable[PromptSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "text"])
        writer.writeheader()
        for p in prompts:
            writer.writerow({"item_id": str(p.item_id), "text": p.text})


def _write_jsonl(path: Path, prompts: Iterable[PromptSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in prompts:
            payload = {"item_id": p.item_id, "prompt": p.text}
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _validate_paths(args: argparse.Namespace, selected: Set[str]) -> None:
    if "blip3o" in selected:
        if not args.blip_base_model:
            raise ValueError("--blip_base_model is required when running BLIP3o.")
        if not args.blip_updated_checkpoint:
            raise ValueError("--blip_updated_checkpoint is required when running BLIP3o.")
    if "bagel" in selected:
        if not args.bagel_base_model:
            raise ValueError("--bagel_base_model is required when running BAGEL.")
        if not args.bagel_updated_model:
            raise ValueError("--bagel_updated_model is required when running BAGEL.")
        if args.bagel_num_images % args.bagel_batch_size != 0:
            raise ValueError("--bagel_num_images must be divisible by --bagel_batch_size.")
    if "vargpt" in selected:
        if not args.vargpt_base_pretrained:
            raise ValueError("--vargpt_base_pretrained is required when running VARGPT.")
        if not args.vargpt_updated_pretrained and not args.vargpt_updated_peft:
            raise ValueError(
                "Provide at least one of --vargpt_updated_pretrained or --vargpt_updated_peft for updated VARGPT run."
            )


def _run_blip(args: argparse.Namespace, repo_root: Path, output_root: Path, prompts_dir: Path) -> None:
    blip_prompts = _select_prompts("blip3o")
    csv_file = prompts_dir / "blip3o_selected_prompts.csv"
    _write_blip_csv(csv_file, blip_prompts)

    base_out = output_root / "blip3o" / "base"
    updated_out = output_root / "blip3o" / "updated"
    base_out.mkdir(parents=True, exist_ok=True)
    updated_out.mkdir(parents=True, exist_ok=True)

    blip_root = repo_root / "BLIP3o"
    base_script = blip_root / "eval" / "dpg_bench" / "generate_dpg_base.py"
    our_script = blip_root / "eval" / "dpg_bench" / "generate_dpg_our.py"

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(blip_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    env["TOKENIZERS_PARALLELISM"] = "false"

    cmd_base = [
        args.python_bin,
        str(base_script),
        "--model",
        args.blip_base_model,
        "--csv_path",
        str(csv_file),
        "--outdir",
        str(base_out),
        "--n_samples",
        str(args.blip_n_samples),
        "--steps",
        str(args.blip_steps),
        "--scale",
        str(args.blip_scale),
        "--index",
        "0",
        "--n_chunks",
        "1",
    ]
    _run(cmd_base, cwd=repo_root, env=env, dry_run=args.dry_run)

    cmd_updated = [
        args.python_bin,
        str(our_script),
        "--model",
        args.blip_base_model,
        "--checkpoint_dir",
        args.blip_updated_checkpoint,
        "--adapter",
        args.blip_adapter,
        "--csv_path",
        str(csv_file),
        "--outdir",
        str(updated_out),
        "--n_samples",
        str(args.blip_n_samples),
        "--steps",
        str(args.blip_steps),
        "--scale",
        str(args.blip_scale),
        "--index",
        "0",
        "--n_chunks",
        "1",
    ]
    _run(cmd_updated, cwd=repo_root, env=env, dry_run=args.dry_run)


def _run_bagel(args: argparse.Namespace, repo_root: Path, output_root: Path, prompts_dir: Path) -> None:
    bagel_prompts = _select_prompts("bagel")
    metadata_file = prompts_dir / "bagel_selected_prompts.jsonl"
    _write_jsonl(metadata_file, bagel_prompts)

    base_out = output_root / "bagel" / "base"
    updated_out = output_root / "bagel" / "updated"
    base_out.mkdir(parents=True, exist_ok=True)
    updated_out.mkdir(parents=True, exist_ok=True)

    bagel_root = repo_root / "Bagel"

    common = [
        args.python_bin,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(args.bagel_nproc),
    ]

    cmd_base = [
        *common,
        "--master_port",
        str(args.bagel_master_port),
        "eval/gen/gen_images_mp.py",
        "--output_dir",
        str(base_out),
        "--metadata_file",
        str(metadata_file),
        "--num_images",
        str(args.bagel_num_images),
        "--batch_size",
        str(args.bagel_batch_size),
        "--cfg_scale",
        str(args.bagel_cfg_scale),
        "--resolution",
        str(args.bagel_resolution),
        "--max_latent_size",
        str(args.bagel_max_latent_size),
        "--model-path",
        args.bagel_base_model,
    ]
    _run(cmd_base, cwd=bagel_root, dry_run=args.dry_run)

    cmd_updated = [
        *common,
        "--master_port",
        str(args.bagel_master_port + 1),
        "eval/gen/gen_images_mp.py",
        "--output_dir",
        str(updated_out),
        "--metadata_file",
        str(metadata_file),
        "--num_images",
        str(args.bagel_num_images),
        "--batch_size",
        str(args.bagel_batch_size),
        "--cfg_scale",
        str(args.bagel_cfg_scale),
        "--resolution",
        str(args.bagel_resolution),
        "--max_latent_size",
        str(args.bagel_max_latent_size),
        "--model-path",
        args.bagel_updated_model,
    ]
    _run(cmd_updated, cwd=bagel_root, dry_run=args.dry_run)


def _run_vargpt(args: argparse.Namespace, repo_root: Path, output_root: Path, prompts_dir: Path) -> None:
    vargpt_prompts = _select_prompts("vargpt")
    metadata_file = prompts_dir / "vargpt_selected_prompts.jsonl"
    _write_jsonl(metadata_file, vargpt_prompts)

    base_out = output_root / "vargpt" / "base"
    updated_out = output_root / "vargpt" / "updated"
    base_out.mkdir(parents=True, exist_ok=True)
    updated_out.mkdir(parents=True, exist_ok=True)

    train_root = Path(args.vargpt_train_root).resolve()
    gen_script = repo_root / "vargpt_1_1" / "VARGPT-family-training" / "run_scripts" / "geneval_generate_vargpt_hf.py"

    common = [
        args.python_bin,
        str(gen_script),
        "--train_root",
        str(train_root),
        "--metadata_file",
        str(metadata_file),
        "--n_samples",
        str(args.vargpt_n_samples),
        "--seed",
        str(args.vargpt_seed),
        "--max_new_tokens",
        str(args.vargpt_max_new_tokens),
        "--do_sample",
        str(args.vargpt_do_sample),
        "--temperature",
        str(args.vargpt_temperature),
        "--top_p",
        str(args.vargpt_top_p),
        "--dtype",
        args.vargpt_dtype,
        "--device",
        args.vargpt_device,
    ]

    cmd_base = [
        *common,
        "--pretrained",
        args.vargpt_base_pretrained,
        "--outdir",
        str(base_out),
    ]
    if args.vargpt_base_peft:
        cmd_base.extend(["--peft", args.vargpt_base_peft])
        cmd_base.extend(["--peft_adapter_name", args.vargpt_peft_adapter_name])
    _run(cmd_base, cwd=repo_root, dry_run=args.dry_run)

    updated_pretrained = args.vargpt_updated_pretrained or args.vargpt_base_pretrained
    cmd_updated = [
        *common,
        "--pretrained",
        updated_pretrained,
        "--outdir",
        str(updated_out),
    ]
    if args.vargpt_updated_peft:
        cmd_updated.extend(["--peft", args.vargpt_updated_peft])
        cmd_updated.extend(["--peft_adapter_name", args.vargpt_peft_adapter_name])
    _run(cmd_updated, cwd=repo_root, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    default_out = repo_root / "runs" / "qualitative_prompt_comparison" / datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Generate target prompts for base vs updated models.")
    parser.add_argument("--python_bin", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--output_root", default=str(default_out), help="Root output directory.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["blip3o", "bagel", "vargpt"],
        default=["blip3o", "bagel", "vargpt"],
        help="Run only selected backbones.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")

    parser.add_argument("--blip_base_model", default="", help="BLIP3o base model path/HF id.")
    parser.add_argument("--blip_updated_checkpoint", default="", help="BLIP3o updated checkpoint directory.")
    parser.add_argument("--blip_adapter", default="generator", help="BLIP3o adapter name in checkpoint.")
    parser.add_argument("--blip_n_samples", type=int, default=4, help="Samples per prompt for BLIP3o.")
    parser.add_argument("--blip_steps", type=int, default=50, help="Diffusion steps for BLIP3o.")
    parser.add_argument("--blip_scale", type=float, default=3.0, help="Guidance scale for BLIP3o.")

    parser.add_argument("--bagel_base_model", default="", help="BAGEL base model directory.")
    parser.add_argument("--bagel_updated_model", default="", help="BAGEL updated model directory.")
    parser.add_argument("--bagel_nproc", type=int, default=1, help="GPUs for BAGEL torchrun.")
    parser.add_argument("--bagel_num_images", type=int, default=4, help="Images per prompt for BAGEL.")
    parser.add_argument("--bagel_batch_size", type=int, default=1, help="Batch size for BAGEL generation.")
    parser.add_argument("--bagel_cfg_scale", type=float, default=4.0, help="CFG scale for BAGEL.")
    parser.add_argument("--bagel_resolution", type=int, default=1024, help="Generation resolution for BAGEL.")
    parser.add_argument("--bagel_max_latent_size", type=int, default=64, help="Max latent size for BAGEL.")
    parser.add_argument("--bagel_master_port", type=int, default=29561, help="Master port for BAGEL torchrun.")

    parser.add_argument(
        "--vargpt_train_root",
        default=str(repo_root / "vargpt_1_1" / "VARGPT-family-training"),
        help="Path to VARGPT-family-training root.",
    )
    parser.add_argument("--vargpt_base_pretrained", default="", help="VARGPT base model path/HF id.")
    parser.add_argument("--vargpt_base_peft", default="", help="Optional base PEFT adapter path.")
    parser.add_argument(
        "--vargpt_updated_pretrained",
        default="",
        help="Optional updated pretrained path (if fully merged checkpoint).",
    )
    parser.add_argument("--vargpt_updated_peft", default="", help="Optional updated PEFT adapter path.")
    parser.add_argument("--vargpt_peft_adapter_name", default="default", help="Adapter name for PEFT.")
    parser.add_argument("--vargpt_n_samples", type=int, default=4, help="Samples per prompt for VARGPT.")
    parser.add_argument("--vargpt_seed", type=int, default=0, help="Base random seed for VARGPT.")
    parser.add_argument("--vargpt_max_new_tokens", type=int, default=4096, help="max_new_tokens for VARGPT.")
    parser.add_argument("--vargpt_do_sample", type=int, default=1, choices=[0, 1], help="Use sampling for VARGPT.")
    parser.add_argument("--vargpt_temperature", type=float, default=1.0, help="Temperature for VARGPT.")
    parser.add_argument("--vargpt_top_p", type=float, default=1.0, help="Top-p for VARGPT.")
    parser.add_argument("--vargpt_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--vargpt_device", default="cuda", help="VARGPT device string.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    output_root = Path(args.output_root).resolve()
    prompts_dir = output_root / "prompt_files"
    selected = set(args.only)

    _validate_paths(args, selected)

    output_root.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    print("=== Target Prompt Generation ===")
    print(f"repo_root:   {repo_root}")
    print(f"output_root: {output_root}")
    print(f"backbones:   {', '.join(sorted(selected))}")

    if "blip3o" in selected:
        _run_blip(args, repo_root, output_root, prompts_dir)
    if "bagel" in selected:
        _run_bagel(args, repo_root, output_root, prompts_dir)
    if "vargpt" in selected:
        _run_vargpt(args, repo_root, output_root, prompts_dir)

    print("Done.")


if __name__ == "__main__":
    main()
