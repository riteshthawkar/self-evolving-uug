#!/usr/bin/env python3
"""
Build a shared 50k image pool for cross-experiment comparability.

- Downloads from a fixed set of public HF datasets aligned with EvoLMM-style
  reasoning images and BLIP-style broad chart/plot sources.
- Saves images locally in a single folder tree.
- Deduplicates by SHA1 image content.
- Emits reproducible manifests and train/val/test splits.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from datasets import load_dataset
from PIL import Image


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_id: str
    split: str
    target_count: int
    preferred_image_keys: Tuple[str, ...]
    streaming: bool = True


DEFAULT_SOURCES: Tuple[SourceSpec, ...] = (
    # EvoLMM-aligned structured reasoning sources
    SourceSpec(
        name="chartqa",
        dataset_id="ahmed-masry/ChartQA",
        split="train",
        target_count=10_000,
        preferred_image_keys=("image",),
        streaming=True,
    ),
    SourceSpec(
        name="ai2d",
        dataset_id="lmms-lab/ai2d",
        split="test",
        target_count=3_000,
        preferred_image_keys=("image",),
        streaming=True,
    ),
    SourceSpec(
        name="mathvista",
        dataset_id="AI4Math/MathVista",
        split="test",
        target_count=2_500,
        preferred_image_keys=("decoded_image", "image"),
        streaming=True,
    ),
    SourceSpec(
        name="mathvision",
        dataset_id="MathLLMs/MathVision",
        split="test",
        target_count=2_000,
        preferred_image_keys=("decoded_image", "image"),
        streaming=True,
    ),
    SourceSpec(
        name="geometry3k",
        dataset_id="hiyouga/geometry3k",
        split="train",
        target_count=1_500,
        preferred_image_keys=("images", "image"),
        streaming=True,
    ),
    SourceSpec(
        name="infographic_vqa",
        dataset_id="nimapourjafar/mm_infographic_vqa",
        split="train",
        target_count=1_000,
        preferred_image_keys=("images", "image"),
        streaming=True,
    ),
    SourceSpec(
        name="chartx",
        dataset_id="geoskyr/ChartX",
        split="validation",
        target_count=2_000,
        preferred_image_keys=("img", "image", "images"),
        streaming=True,
    ),
    # Larger source to reach 50k while keeping chart/plot domain
    SourceSpec(
        name="plotqa",
        dataset_id="achang/plot_qa",
        split="train",
        target_count=28_000,
        preferred_image_keys=("image",),
        streaming=True,
    ),
)


def _stable_sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _iter_candidate_keys(example: Dict[str, Any], preferred: Iterable[str]) -> Iterable[str]:
    seen = set()
    for key in preferred:
        if key in example and key not in seen:
            seen.add(key)
            yield key
    for key in example.keys():
        lk = key.lower()
        if ("image" in lk or lk in {"img", "images", "decoded_image"}) and key not in seen:
            seen.add(key)
            yield key


def _open_image_from_value(value: Any, timeout: int = 30) -> Optional[Image.Image]:
    # Handle list-like fields (take first image payload)
    if isinstance(value, list):
        if not value:
            return None
        return _open_image_from_value(value[0], timeout=timeout)

    # PIL image (datasets Image feature often decodes into PIL)
    if isinstance(value, Image.Image):
        return value

    # HF image dict style {"bytes": ..., "path": ...}
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            try:
                return Image.open(io.BytesIO(value["bytes"]))
            except Exception:
                return None
        path = value.get("path")
        if isinstance(path, str):
            return _open_image_from_value(path, timeout=timeout)

    # String path or URL
    if isinstance(value, str):
        if value.startswith("http://") or value.startswith("https://"):
            try:
                r = requests.get(value, timeout=timeout)
                r.raise_for_status()
                return Image.open(io.BytesIO(r.content))
            except Exception:
                return None
        p = Path(value)
        if p.exists() and p.is_file():
            try:
                return Image.open(p)
            except Exception:
                return None

    # Bytes blob
    if isinstance(value, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(bytes(value)))
        except Exception:
            return None

    return None


def _extract_image(example: Dict[str, Any], preferred_keys: Tuple[str, ...]) -> Optional[Image.Image]:
    for key in _iter_candidate_keys(example, preferred_keys):
        img = _open_image_from_value(example.get(key))
        if img is not None:
            return img
    return None


def _save_jpeg(img: Image.Image, out_path: Path, quality: int = 95) -> Tuple[str, int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    data = buf.getvalue()
    sha = _stable_sha1(data)
    with out_path.open("wb") as f:
        f.write(data)
    return sha, rgb.width, rgb.height


def _source_counts(sources: Tuple[SourceSpec, ...]) -> Dict[str, int]:
    return {s.name: s.target_count for s in sources}


def build_dataset(
    output_dir: Path,
    sources: Tuple[SourceSpec, ...],
    target_total: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    quality: int,
) -> Dict[str, Any]:
    images_root = output_dir / "images"
    manifests_dir = output_dir / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    seen_sha: set[str] = set()

    print("Planned source quotas:")
    for s in sources:
        mode = "streaming" if s.streaming else "materialized"
        print(f"- {s.name:16s} {s.target_count:6d}  ({s.dataset_id} [{s.split}] | {mode})")
    print(f"Target total: {target_total}")

    source_stats: Dict[str, Dict[str, int]] = {
        s.name: {
            "requested": s.target_count,
            "saved": 0,
            "duplicates": 0,
            "failed_image": 0,
            "iterated": 0,
        }
        for s in sources
    }

    global_idx = 0
    for spec in sources:
        print(f"\n[Source] {spec.name} :: loading {spec.dataset_id} [{spec.split}] ...")
        ds = load_dataset(spec.dataset_id, split=spec.split, streaming=spec.streaming)
        ds_size_text = "streaming"
        if not spec.streaming:
            try:
                ds_size_text = str(len(ds))
            except Exception:
                ds_size_text = "unknown"
        print(f"[Source] {spec.name} :: loaded {ds_size_text} rows")

        for row_idx, ex in enumerate(ds):
            st = source_stats[spec.name]
            st["iterated"] += 1

            if st["saved"] >= spec.target_count:
                break
            if len(all_records) >= target_total:
                break

            img = _extract_image(ex, spec.preferred_image_keys)
            if img is None:
                st["failed_image"] += 1
                continue

            out_rel = Path(spec.name) / f"{spec.name}_{st['saved']:06d}.jpg"
            out_abs = images_root / out_rel

            try:
                sha, width, height = _save_jpeg(img, out_abs, quality=quality)
            except Exception:
                st["failed_image"] += 1
                continue

            if sha in seen_sha:
                st["duplicates"] += 1
                try:
                    out_abs.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            seen_sha.add(sha)
            st["saved"] += 1

            rec = {
                "global_index": global_idx,
                "source": spec.name,
                "dataset_id": spec.dataset_id,
                "dataset_split": spec.split,
                "source_row_index": row_idx,
                "image_relpath": str(out_rel.as_posix()),
                "image_abspath": str(out_abs.resolve()),
                "sha1": sha,
                "width": width,
                "height": height,
            }
            all_records.append(rec)
            global_idx += 1

            if st["saved"] % 500 == 0:
                print(f"[Source] {spec.name} saved={st['saved']} / {spec.target_count}")

        print(
            f"[Source] {spec.name} done: saved={st['saved']} requested={st['requested']} "
            f"iterated={st['iterated']} duplicates={st['duplicates']} failed_image={st['failed_image']}"
        )

    # If still below target, top up using plotqa as fallback (same domain)
    if len(all_records) < target_total:
        deficit = target_total - len(all_records)
        print(f"\n[Top-up] Need {deficit} more images. Topping up from plotqa ...")
        ds = load_dataset("achang/plot_qa", split="train", streaming=True)
        st_name = "plotqa_topup"
        source_stats[st_name] = {
            "requested": deficit,
            "saved": 0,
            "duplicates": 0,
            "failed_image": 0,
            "iterated": 0,
        }
        st = source_stats[st_name]

        for row_idx, ex in enumerate(ds):
            if len(all_records) >= target_total:
                break
            st["iterated"] += 1

            img = _extract_image(ex, ("image",))
            if img is None:
                st["failed_image"] += 1
                continue

            out_rel = Path("plotqa_topup") / f"plotqa_topup_{st['saved']:06d}.jpg"
            out_abs = images_root / out_rel

            try:
                sha, width, height = _save_jpeg(img, out_abs, quality=quality)
            except Exception:
                st["failed_image"] += 1
                continue

            if sha in seen_sha:
                st["duplicates"] += 1
                try:
                    out_abs.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            seen_sha.add(sha)
            st["saved"] += 1
            rec = {
                "global_index": global_idx,
                "source": "plotqa_topup",
                "dataset_id": "achang/plot_qa",
                "dataset_split": "train",
                "source_row_index": row_idx,
                "image_relpath": str(out_rel.as_posix()),
                "image_abspath": str(out_abs.resolve()),
                "sha1": sha,
                "width": width,
                "height": height,
            }
            all_records.append(rec)
            global_idx += 1

            if st["saved"] % 500 == 0:
                print(f"[Top-up] saved={st['saved']} / {deficit}")

        print(
            f"[Top-up] done: saved={st['saved']} requested={st['requested']} "
            f"iterated={st['iterated']} duplicates={st['duplicates']} failed_image={st['failed_image']}"
        )

    if len(all_records) == 0:
        raise RuntimeError("No images were collected. Check dataset access/network.")

    # Global split (reproducible)
    rnd = random.Random(seed)
    rnd.shuffle(all_records)

    n = len(all_records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_recs = all_records[:n_train]
    val_recs = all_records[n_train : n_train + n_val]
    test_recs = all_records[n_train + n_val :]

    def dump_jsonl(path: Path, records: List[Dict[str, Any]]):
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump_jsonl(manifests_dir / "all.jsonl", all_records)
    dump_jsonl(manifests_dir / "train.jsonl", train_recs)
    dump_jsonl(manifests_dir / "val.jsonl", val_recs)
    dump_jsonl(manifests_dir / "test.jsonl", test_recs)

    with (manifests_dir / "source_counts.json").open("w", encoding="utf-8") as f:
        json.dump(source_stats, f, indent=2)

    summary = {
        "target_total": target_total,
        "actual_total": n,
        "split_counts": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
        },
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": 1.0 - train_ratio - val_ratio,
        },
        "seed": seed,
        "image_root": str(images_root.resolve()),
        "manifests_dir": str(manifests_dir.resolve()),
        "sources": [
            {
                "name": s.name,
                "dataset_id": s.dataset_id,
                "split": s.split,
                "target_count": s.target_count,
                "streaming": s.streaming,
            }
            for s in sources
        ],
    }

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nBuild complete")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build shared 50k multimodal image pool")
    ap.add_argument(
        "--output_dir",
        type=str,
        default="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/data/shared_uug_50k",
    )
    ap.add_argument("--target_total", type=int, default=50_000)
    ap.add_argument("--train_ratio", type=float, default=0.90)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument(
        "--quick_test",
        action="store_true",
        help="Small dry-run for validation (25 images/source, 200 total)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1)")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0,1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    sources = DEFAULT_SOURCES
    target_total = args.target_total

    if args.quick_test:
        sources = tuple(
            SourceSpec(
                name=s.name,
                dataset_id=s.dataset_id,
                split=s.split,
                target_count=min(25, s.target_count),
                preferred_image_keys=s.preferred_image_keys,
                streaming=s.streaming,
            )
            for s in DEFAULT_SOURCES
        )
        target_total = 200

    out = Path(args.output_dir)
    build_dataset(
        output_dir=out,
        sources=sources,
        target_total=target_total,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        quality=args.jpeg_quality,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
