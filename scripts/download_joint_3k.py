#!/usr/bin/env python3
"""
Download a 3k image pool optimized for JOINT self-evolving training (E1/E3/E4/E5/E6).

Design rationale:
  Joint training (U+G) requires images that are BOTH:
    1. Rich enough for understanding (counting, spatial, attributes, text-in-scene)
    2. Generatable by the DiT (photorealistic / artistic — NOT charts/diagrams)

  This script builds a balanced natural-image pool aligned with both:
    Understanding benchmarks: MME-P, RealWorldQA, TextVQA, SEED
    Generation benchmarks: GenEval, DPG-Bench, WISE

  Charts/diagrams are deliberately EXCLUDED because diffusion models cannot
  generate them, which would poison generation-step reward signals.
  For pure understanding experiments (E2), use the existing chart-heavy 50k pool.

Domain composition (default):
  Natural scenes:     ~1500  (50%)  — COCO / SA-1B
  Text-in-wild:       ~450   (15%)  — TextVQA (signs, labels, menus)
  Compositional:      ~600   (20%)  — GQA (attribute binding, relations)
  Artistic/diverse:   ~450   (15%)  — LAION-COCO (diverse styles)

Usage:
  # Default 3k pool (recommended for E1)
  python download_joint_3k.py

  # Custom size
  python download_joint_3k.py --total_samples 5000

  # Custom output directory
  python download_joint_3k.py --output_dir data/joint_3k_v2

  # Include a small chart slice for mixed experiments
  python download_joint_3k.py --include_charts --chart_ratio 0.10

After download, update DATA_DIR in _common.sh or pass at runtime:
  DATA_DIR=/path/to/joint_3k/images TRAIN_STAGE=warmup bash E1_main_joint.sh
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from PIL import Image

try:
    from datasets import Dataset, IterableDataset, get_dataset_split_names, load_dataset
except ImportError as exc:
    raise RuntimeError(
        "Requires `datasets`: pip install datasets pillow"
    ) from exc

try:
    import numpy as np
except ImportError:
    np = None

try:
    import requests
except ImportError:
    requests = None


# ══════════════════════════════════════════════════════════════════════════════
# Dataset specifications
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DatasetSpec:
    alias: str
    dataset_ids: Tuple[str, ...]
    domain: str
    split: str = "train"
    fallback_splits: Tuple[str, ...] = ()
    config_name: Optional[str] = None
    prefer_streaming: bool = True


# Generation-compatible datasets (natural / photorealistic / artistic)
NATURAL_SPECS: Dict[str, DatasetSpec] = {
    "coco": DatasetSpec(
        alias="coco",
        dataset_ids=(
            "astro21/coco-caption-train-split-10k",
            "HuggingFaceM4/COCO",
        ),
        domain="natural",
        split="train",
    ),
    "sa1b": DatasetSpec(
        alias="sa1b",
        dataset_ids=(
            "facebook/segment-anything-1-billion",
            "HuggingFaceM4/COCO",
        ),
        domain="natural",
        split="train",
    ),
    "textvqa": DatasetSpec(
        alias="textvqa",
        dataset_ids=(
            "lmms-lab/TextVQA",
            "lmms-lab/textvqa",
            "textvqa",
        ),
        domain="text_in_wild",
        split="train",
    ),
    "gqa": DatasetSpec(
        alias="gqa",
        dataset_ids=(
            "lmms-lab/GQA",
            "Graphcore/gqa",
            "echarlaix/gqa",
        ),
        domain="compositional",
        split="train",
    ),
    "laion_coco": DatasetSpec(
        alias="laion_coco",
        dataset_ids=("laion/laion-coco",),
        domain="artistic",
        split="train",
    ),
}

# Chart/diagram datasets (understanding-only — breaks generation)
CHART_SPECS: Dict[str, DatasetSpec] = {
    "chartqa": DatasetSpec(
        alias="chartqa",
        dataset_ids=("ahmed-masry/ChartQA",),
        domain="chart",
        split="train",
    ),
    "ai2d": DatasetSpec(
        alias="ai2d",
        dataset_ids=("lmms-lab/ai2d",),
        domain="chart",
        split="train",
        fallback_splits=("test",),
    ),
}

# Default domain weights for natural-only pool
DEFAULT_NATURAL_WEIGHTS = {
    "natural": 0.50,       # COCO + SA-1B
    "text_in_wild": 0.15,  # TextVQA (signs, labels, storefronts)
    "compositional": 0.20, # GQA (attribute binding, spatial relations)
    "artistic": 0.15,      # LAION-COCO (diverse styles)
}


# ══════════════════════════════════════════════════════════════════════════════
# Image extraction helpers (same as download_benchmark_10k.py)
# ══════════════════════════════════════════════════════════════════════════════

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
        if value.startswith(("http://", "https://")):
            try:
                if requests is not None:
                    resp = requests.get(value, timeout=15)
                    if resp.status_code == 200:
                        return Image.open(io.BytesIO(resp.content)).convert("RGB")
                else:
                    from urllib.request import urlopen
                    with urlopen(value, timeout=15) as resp:
                        return Image.open(io.BytesIO(resp.read())).convert("RGB")
            except Exception:
                pass
        return None
    if isinstance(value, dict):
        for k in ("bytes", "path", "array"):
            if k in value:
                return try_decode_image(value[k])
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            img = try_decode_image(item)
            if img is not None:
                return img
    if np is not None and isinstance(value, np.ndarray):
        try:
            return Image.fromarray(value).convert("RGB")
        except Exception:
            pass
    return None


def extract_image(record: Dict) -> Tuple[Optional[Image.Image], Optional[str]]:
    preferred = ("image", "img", "photo", "jpg", "jpeg", "png", "webp",
                 "image_url", "url", "uri", "path", "image_path", "source_url")
    for key, value in record.items():
        if str(key).lower() in preferred:
            img = try_decode_image(value)
            if img is not None:
                return img, str(key)
    for key, value in record.items():
        if any(tok in str(key).lower() for tok in ("image", "img", "photo", "url", "path")):
            img = try_decode_image(value)
            if img is not None:
                return img, str(key)
    for key, value in record.items():
        if isinstance(value, str):
            continue
        img = try_decode_image(value)
        if img is not None:
            return img, str(key)
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ══════════════════════════════════════════════════════════════════════════════

def load_with_fallback(spec: DatasetSpec, *, streaming_only: bool = True):
    errors = []
    for dataset_id in spec.dataset_ids:
        splits = [spec.split] + list(spec.fallback_splits)
        try:
            discovered = get_dataset_split_names(dataset_id, config_name=spec.config_name)
            for s in discovered:
                if s not in splits:
                    splits.append(s)
        except Exception:
            pass
        for split_name in splits:
            for streaming in (True, False) if not streaming_only else (True,):
                try:
                    kwargs = {"path": dataset_id, "split": split_name, "streaming": streaming}
                    if spec.config_name:
                        kwargs["name"] = spec.config_name
                    ds = load_dataset(**kwargs)
                    return ds, dataset_id, split_name
                except Exception as exc:
                    errors.append(f"{dataset_id}/{split_name}: {exc}")
    raise RuntimeError(f"Failed to load '{spec.alias}':\n" + "\n".join(f"  - {e}" for e in errors))


def save_images_from_spec(
    spec: DatasetSpec,
    *,
    output_root: Path,
    target: int,
    seed: int,
    streaming_only: bool = True,
    max_scan_multiplier: int = 12,
    shuffle_buffer: int = 20_000,
) -> Dict:
    if target <= 0:
        return {"alias": spec.alias, "saved": 0, "target": 0}

    ds, dataset_id, used_split = load_with_fallback(spec, streaming_only=streaming_only)

    images_dir = output_root / "images" / spec.alias
    images_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    scanned = 0
    max_scan = target * max_scan_multiplier

    manifest_records = []

    if isinstance(ds, IterableDataset):
        buf = max(1000, min(target * 2, shuffle_buffer))
        ds_iter = iter(ds.shuffle(seed=seed, buffer_size=buf))
    elif isinstance(ds, Dataset):
        indices = list(range(len(ds)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        ds_iter = (ds[i] for i in indices[:max_scan])
    else:
        ds_iter = iter(ds)

    for record in ds_iter:
        if saved >= target:
            break
        scanned += 1
        if scanned > max_scan:
            break

        img, key = extract_image(record)
        if img is None:
            continue

        # Basic quality filter: skip tiny images
        w, h = img.size
        if w < 128 or h < 128:
            continue

        out_name = f"{saved:06d}.jpg"
        out_path = images_dir / out_name
        img.save(out_path, format="JPEG", quality=95, optimize=True)

        manifest_records.append({
            "alias": spec.alias,
            "domain": spec.domain,
            "dataset_id": dataset_id,
            "split": used_split,
            "saved_index": saved,
            "image_relpath": f"images/{spec.alias}/{out_name}",
            "width": w,
            "height": h,
        })
        saved += 1

        if saved % 100 == 0:
            print(f"    [{spec.alias}] {saved}/{target} saved ({scanned} scanned)")

    # Write per-dataset manifest
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{spec.alias}.jsonl"
    with open(manifest_path, "w") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec) + "\n")

    print(f"  [{spec.alias}] Done: {saved}/{target} (scanned {scanned})")
    return {
        "alias": spec.alias,
        "domain": spec.domain,
        "dataset_id": dataset_id,
        "saved": saved,
        "target": target,
        "scanned": scanned,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Allocation
# ══════════════════════════════════════════════════════════════════════════════

def allocate_by_domain(
    specs: Dict[str, DatasetSpec],
    total: int,
    domain_weights: Dict[str, float],
) -> Dict[str, int]:
    """Allocate samples to datasets based on domain weights."""
    # Group specs by domain
    by_domain: Dict[str, List[str]] = {}
    for alias, spec in specs.items():
        by_domain.setdefault(spec.domain, []).append(alias)

    # Only consider domains that have specs
    active_weights = {d: w for d, w in domain_weights.items() if d in by_domain}
    if not active_weights:
        # Equal split if no matching weights
        per = total // len(specs)
        return {alias: per for alias in specs}

    # Normalize weights
    wsum = sum(active_weights.values())
    norm = {d: w / wsum for d, w in active_weights.items()}

    # Allocate per domain, then split within domain
    targets: Dict[str, int] = {}
    allocated = 0
    domains = sorted(norm.keys())
    for i, domain in enumerate(domains):
        aliases = sorted(by_domain[domain])
        if i == len(domains) - 1:
            domain_total = total - allocated  # last domain gets remainder
        else:
            domain_total = round(total * norm[domain])
        per_alias = domain_total // len(aliases)
        remainder = domain_total - per_alias * len(aliases)
        for j, alias in enumerate(aliases):
            targets[alias] = per_alias + (1 if j < remainder else 0)
        allocated += domain_total

    return targets


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Download generation-compatible image pool for joint self-evolving training."
    )
    p.add_argument("--output_dir", type=Path, default=Path("data/joint_3k"),
                   help="Output directory (default: data/joint_3k)")
    p.add_argument("--total_samples", type=int, default=3000,
                   help="Total images to download (default: 3000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--streaming_only", action="store_true", default=True)
    p.add_argument("--allow_non_streaming", dest="streaming_only", action="store_false")
    p.add_argument("--max_scan_multiplier", type=int, default=12)
    p.add_argument("--shuffle_buffer", type=int, default=20_000)

    # Optional chart inclusion for mixed experiments
    p.add_argument("--include_charts", action="store_true", default=False,
                   help="Include a small chart/diagram slice (for understanding diversity)")
    p.add_argument("--chart_ratio", type=float, default=0.10,
                   help="Fraction of total to allocate to charts (default: 0.10 = 10%%)")

    # Dataset selection
    p.add_argument("--datasets", type=str, default="coco,textvqa,gqa,laion_coco",
                   help="Comma-separated natural dataset aliases")
    p.add_argument("--chart_datasets", type=str, default="chartqa,ai2d",
                   help="Comma-separated chart dataset aliases (used with --include_charts)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    total = args.total_samples
    natural_aliases = [a.strip() for a in args.datasets.split(",") if a.strip()]

    print("=" * 70)
    print(f"  Joint Pool Builder — {total} images")
    print(f"  Output: {out_root}")
    print(f"  Natural datasets: {', '.join(natural_aliases)}")

    # Build spec dict for natural datasets
    natural_specs = {}
    for alias in natural_aliases:
        if alias not in NATURAL_SPECS:
            print(f"  WARNING: Unknown natural dataset '{alias}', skipping")
            continue
        natural_specs[alias] = NATURAL_SPECS[alias]

    chart_specs = {}
    chart_total = 0
    if args.include_charts:
        chart_total = round(total * args.chart_ratio)
        chart_aliases = [a.strip() for a in args.chart_datasets.split(",") if a.strip()]
        for alias in chart_aliases:
            if alias not in CHART_SPECS:
                print(f"  WARNING: Unknown chart dataset '{alias}', skipping")
                continue
            chart_specs[alias] = CHART_SPECS[alias]
        print(f"  Chart datasets: {', '.join(chart_specs.keys())} ({chart_total} images, {args.chart_ratio:.0%})")

    natural_total = total - chart_total
    print(f"  Natural: {natural_total} | Charts: {chart_total}")
    print("=" * 70)

    # Allocate
    natural_targets = allocate_by_domain(natural_specs, natural_total, DEFAULT_NATURAL_WEIGHTS)
    chart_targets = {}
    if chart_specs and chart_total > 0:
        per_chart = chart_total // len(chart_specs)
        rem = chart_total - per_chart * len(chart_specs)
        for i, alias in enumerate(sorted(chart_specs.keys())):
            chart_targets[alias] = per_chart + (1 if i < rem else 0)

    print("\nAllocation:")
    for alias, t in sorted(natural_targets.items()):
        print(f"  {alias:15s} ({natural_specs[alias].domain:15s}): {t:5d}")
    for alias, t in sorted(chart_targets.items()):
        print(f"  {alias:15s} ({chart_specs[alias].domain:15s}): {t:5d}")
    print()

    # Download
    all_results = []
    all_specs = {**natural_specs, **chart_specs}
    all_targets = {**natural_targets, **chart_targets}

    for i, (alias, target) in enumerate(sorted(all_targets.items())):
        if target <= 0:
            continue
        spec = all_specs[alias]
        print(f"\n[{i+1}/{len(all_targets)}] Downloading {alias} ({target} images)...")
        try:
            result = save_images_from_spec(
                spec,
                output_root=out_root,
                target=target,
                seed=args.seed + i,
                streaming_only=args.streaming_only,
                max_scan_multiplier=args.max_scan_multiplier,
                shuffle_buffer=args.shuffle_buffer,
            )
            all_results.append(result)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            all_results.append({"alias": alias, "error": str(exc)})

    # Write combined manifest
    manifests_dir = out_root / "manifests"
    manifests_dir.mkdir(exist_ok=True)
    all_manifest = manifests_dir / "all.jsonl"
    global_idx = 0
    with open(all_manifest, "w") as f:
        for alias in sorted(all_targets.keys()):
            per_manifest = manifests_dir / f"{alias}.jsonl"
            if not per_manifest.exists():
                continue
            with open(per_manifest) as mf:
                for line in mf:
                    rec = json.loads(line.strip())
                    rec["global_index"] = global_idx
                    rec["source"] = alias
                    f.write(json.dumps(rec) + "\n")
                    global_idx += 1

    # Write download summary
    total_saved = sum(r.get("saved", 0) for r in all_results if "error" not in r)
    summary = {
        "total_requested": total,
        "total_saved": total_saved,
        "natural_total": natural_total,
        "chart_total": chart_total,
        "include_charts": args.include_charts,
        "chart_ratio": args.chart_ratio if args.include_charts else 0.0,
        "datasets": all_results,
        "domain_weights": DEFAULT_NATURAL_WEIGHTS,
    }
    summary_path = out_root / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  Download complete: {total_saved}/{total} images")
    print(f"  Summary: {summary_path}")
    print(f"  Manifest: {all_manifest}")
    print(f"\n  To use with experiments:")
    print(f"    DATA_DIR={out_root}/images TRAIN_STAGE=warmup bash E1_main_joint.sh")
    print("=" * 70)

    return 0 if total_saved >= total * 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
