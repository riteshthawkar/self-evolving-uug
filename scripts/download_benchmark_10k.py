#!/usr/bin/env python3
"""
Download benchmark-aligned image datasets and save a fixed sample count per dataset.

Default behavior:
- Downloads 10,000 images per selected dataset
- Uses streaming-only loading (no full dataset download)
- Saves to: <output_dir>/images/<dataset_alias>/
- Writes per-dataset manifest JSONL and a summary JSON
- Fails fast if a dataset cannot reach the requested sample count

Designed for self-evolving training where only raw images are required.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from PIL import Image

try:
    from datasets import Dataset, IterableDataset, load_dataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires `datasets`. Install with: pip install datasets pillow"
    ) from exc

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


@dataclass(frozen=True)
class DatasetSpec:
    alias: str
    dataset_ids: Tuple[str, ...]
    split: str = "train"
    config_name: Optional[str] = None
    prefer_streaming: bool = True


PRESET_SPECS: Dict[str, DatasetSpec] = {
    # Benchmark/domain aligned and >=10k in typical train splits.
    "chartqa": DatasetSpec(
        alias="chartqa",
        dataset_ids=("ahmed-masry/ChartQA",),
        split="train",
        prefer_streaming=True,
    ),
    "docvqa": DatasetSpec(
        alias="docvqa",
        dataset_ids=("lmms-lab/DocVQA",),
        split="train",
        prefer_streaming=True,
    ),
    "sa1b": DatasetSpec(
        alias="sa1b",
        dataset_ids=("facebook/segment-anything-1-billion",),
        split="train",
        prefer_streaming=True,
    ),
    "laion_coco": DatasetSpec(
        alias="laion_coco",
        dataset_ids=("laion/laion-coco",),
        split="train",
        prefer_streaming=True,
    ),
    # Optional OCR-heavy domain. Some mirrors vary; tries IDs in order.
    "textvqa": DatasetSpec(
        alias="textvqa",
        dataset_ids=(
            "lmms-lab/TextVQA",
            "textvqa",
        ),
        split="train",
        prefer_streaming=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download fixed-count benchmark-aligned image subsets."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/benchmark_10k"),
        help="Output root directory (default: data/benchmark_10k).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="chartqa,docvqa,sa1b,laion_coco",
        help=(
            "Comma-separated preset aliases. "
            f"Available: {', '.join(sorted(PRESET_SPECS.keys()))}"
        ),
    )
    parser.add_argument(
        "--samples_per_dataset",
        type=int,
        default=10_000,
        help="Target sample count per dataset (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic subset selection.",
    )
    parser.add_argument(
        "--streaming_only",
        action="store_true",
        default=True,
        help=(
            "Use HuggingFace streaming only (default). "
            "Prevents full-dataset materialization on local disk."
        ),
    )
    parser.add_argument(
        "--allow_non_streaming",
        dest="streaming_only",
        action="store_false",
        help="Allow non-streaming fallback (may download full dataset).",
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=20_000,
        help="Streaming shuffle buffer size (default: 20000).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail if any dataset cannot reach the target sample count.",
    )
    parser.add_argument(
        "--allow_underfilled",
        dest="strict",
        action="store_false",
        help="Allow datasets with fewer saved samples than requested.",
    )
    parser.add_argument(
        "--max_scan_multiplier",
        type=int,
        default=12,
        help=(
            "Max records to scan per dataset = samples_per_dataset * multiplier "
            "(default: 12)."
        ),
    )
    return parser.parse_args()


def load_with_fallback(spec: DatasetSpec, *, streaming_only: bool):
    errors: List[str] = []
    streaming_options = (True,) if streaming_only else (spec.prefer_streaming, not spec.prefer_streaming)
    for dataset_id in spec.dataset_ids:
        for streaming in streaming_options:
            try:
                kwargs = {
                    "path": dataset_id,
                    "split": spec.split,
                    "streaming": streaming,
                }
                if spec.config_name:
                    kwargs["name"] = spec.config_name
                ds = load_dataset(**kwargs)
                if streaming_only and not isinstance(ds, IterableDataset):
                    raise RuntimeError(
                        "Expected IterableDataset in streaming-only mode "
                        f"for dataset_id={dataset_id}, got {type(ds).__name__}."
                    )
                return ds, dataset_id, streaming
            except Exception as exc:  # pragma: no cover
                errors.append(
                    f"id={dataset_id} split={spec.split} streaming={streaming}: {repr(exc)}"
                )
    joined = "\n  - ".join(errors)
    raise RuntimeError(
        f"Failed to load dataset preset '{spec.alias}'. Attempts:\n  - {joined}"
    )


def try_decode_image(value) -> Optional[Image.Image]:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        try:
            return Image.open(io.BytesIO(value)).convert("RGB")
        except Exception:
            return None
    if isinstance(value, str):
        path = Path(value)
        if path.is_file():
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                return None
        return None
    if isinstance(value, dict):
        if "bytes" in value:
            return try_decode_image(value.get("bytes"))
        if "path" in value:
            return try_decode_image(value.get("path"))
        if "array" in value and np is not None:
            arr = value.get("array")
            try:
                return Image.fromarray(np.asarray(arr)).convert("RGB")
            except Exception:
                return None
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            img = try_decode_image(item)
            if img is not None:
                return img
    if np is not None:
        try:
            if isinstance(value, np.ndarray):
                return Image.fromarray(value).convert("RGB")
        except Exception:
            return None
    return None


def extract_image(record: Dict) -> Tuple[Optional[Image.Image], Optional[str]]:
    preferred_keys = (
        "image",
        "img",
        "photo",
        "jpg",
        "png",
    )
    for key in preferred_keys:
        if key in record:
            img = try_decode_image(record[key])
            if img is not None:
                return img, key
    for key, value in record.items():
        img = try_decode_image(value)
        if img is not None:
            return img, key
    return None, None


def iter_records(
    ds,
    *,
    seed: int,
    max_records: int,
    shuffle_buffer: int,
    streaming_only: bool,
) -> Iterator[Dict]:
    if isinstance(ds, IterableDataset):
        buffer_size = max(1_000, min(max_records, int(shuffle_buffer)))
        shuffled = ds.shuffle(seed=seed, buffer_size=buffer_size)
        for idx, row in enumerate(shuffled):
            if idx >= max_records:
                break
            yield row
        return
    if isinstance(ds, Dataset):
        if streaming_only:
            raise RuntimeError(
                "Received non-streaming Dataset while --streaming_only is enabled. "
                "This would risk full local download."
            )
        total = len(ds)
        indices = list(range(total))
        rng = random.Random(seed)
        rng.shuffle(indices)
        for idx in indices[: max_records]:
            yield ds[int(idx)]
        return
    # Unknown dataset type; best effort sequential iteration.
    for idx, row in enumerate(ds):
        if idx >= max_records:
            break
        yield row


def save_subset_for_spec(
    spec: DatasetSpec,
    *,
    output_root: Path,
    samples_target: int,
    seed: int,
    strict: bool,
    max_scan_multiplier: int,
    shuffle_buffer: int,
    streaming_only: bool,
) -> Dict[str, object]:
    ds, dataset_id, streaming = load_with_fallback(spec, streaming_only=streaming_only)
    images_dir = output_root / "images" / spec.alias
    manifests_dir = output_root / "manifests"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / f"{spec.alias}.jsonl"

    saved = 0
    scanned = 0
    image_key = None
    max_scan = max(samples_target * max_scan_multiplier, samples_target)

    with manifest_path.open("w", encoding="utf-8") as mf:
        for record in iter_records(
            ds,
            seed=seed,
            max_records=max_scan,
            shuffle_buffer=shuffle_buffer,
            streaming_only=streaming_only,
        ):
            scanned += 1
            img, key = extract_image(record)
            if img is None:
                continue
            if image_key is None:
                image_key = key

            out_name = f"{saved:06d}.jpg"
            out_path = images_dir / out_name
            img.save(out_path, format="JPEG", quality=95, optimize=True)

            mf.write(
                json.dumps(
                    {
                        "alias": spec.alias,
                        "dataset_id": dataset_id,
                        "split": spec.split,
                        "streaming": streaming,
                        "image_key": key,
                        "saved_index": saved,
                        "image_relpath": str(Path("images") / spec.alias / out_name),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            saved += 1
            if saved >= samples_target:
                break

    if strict and saved < samples_target:
        raise RuntimeError(
            f"Dataset '{spec.alias}' saved {saved}/{samples_target}. "
            f"Scanned {scanned} rows from '{dataset_id}'."
        )

    return {
        "alias": spec.alias,
        "dataset_id": dataset_id,
        "split": spec.split,
        "streaming": streaming,
        "image_key_detected": image_key,
        "saved": saved,
        "target": samples_target,
        "scanned": scanned,
        "output_dir": str(images_dir),
        "manifest": str(manifest_path),
    }


def main() -> int:
    args = parse_args()

    aliases = [part.strip() for part in args.datasets.split(",") if part.strip()]
    if not aliases:
        raise ValueError("No datasets selected.")

    unknown = [a for a in aliases if a not in PRESET_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown dataset alias(es): {unknown}. "
            f"Available: {sorted(PRESET_SPECS.keys())}"
        )

    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[download] Output root: {out_root}")
    print(f"[download] Target per dataset: {args.samples_per_dataset}")
    print(f"[download] Datasets: {', '.join(aliases)}")
    print(
        "[download] Mode: "
        f"streaming_only={args.streaming_only} "
        f"shuffle_buffer={args.shuffle_buffer} "
        f"max_scan_multiplier={args.max_scan_multiplier}"
    )

    summary: List[Dict[str, object]] = []
    failures: List[str] = []

    for i, alias in enumerate(aliases):
        spec = PRESET_SPECS[alias]
        run_seed = args.seed + i
        print(f"\n[download] -> {alias} (seed={run_seed})")
        try:
            result = save_subset_for_spec(
                spec,
                output_root=out_root,
                samples_target=args.samples_per_dataset,
                seed=run_seed,
                strict=args.strict,
                max_scan_multiplier=args.max_scan_multiplier,
                shuffle_buffer=args.shuffle_buffer,
                streaming_only=args.streaming_only,
            )
            summary.append(result)
            print(
                f"[download]    saved={result['saved']}/{result['target']} "
                f"dataset_id={result['dataset_id']} split={result['split']}"
            )
        except Exception as exc:
            failures.append(f"{alias}: {exc}")
            print(f"[download]    FAILED: {exc}", file=sys.stderr)
            if args.strict:
                break

    summary_path = out_root / "download_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_root": str(out_root),
                "samples_per_dataset": args.samples_per_dataset,
                "selected_aliases": aliases,
                "strict": args.strict,
                "streaming_only": args.streaming_only,
                "shuffle_buffer": args.shuffle_buffer,
                "max_scan_multiplier": args.max_scan_multiplier,
                "results": summary,
                "failures": failures,
            },
            f,
            indent=2,
        )

    print(f"\n[download] Summary: {summary_path}")
    if failures:
        print("[download] Failures detected:")
        for item in failures:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
