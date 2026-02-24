#!/usr/bin/env python3
"""
download_rich_10k.py — Download a richer, proposer-friendly image pool for
self-evolving UUG training (E1–E6 joint experiments).

WHY a new download script?
--------------------------
Current pool (textvqa, gqa, coco, laion_coco) suffers from:
  • Too many single-object images (COCO "object" crops, GQA scene isolations)
  • TextVQA has text-in-scene but mostly close-ups of signs / menus → low spatial
    complexity → proposer collapses to "what does the text say?" questions
  • LAION-COCO is too diverse in quality; many synthetic / illustration images
    give weak spatial relationships

This script replaces/augments that pool with datasets specifically chosen to
challenge ALL of the proposer's H1–H8 hard strategies:

  H1 – Counting              → Visual Genome (annotated object counts)
  H2 – Spatial reasoning     → SpatialSense / VSR / RefCOCO context images
  H3 – Attribute binding     → Visual Genome attributes, DOCCI
  H4 – State/occlusion       → Open Images V7 (rich multi-label, occluded objects)
  H5 – Depth/affordance      → NYU Depth V2 (complex indoor scenes)
  H6 – Causal / event        → Something-Something v2 frames, A-OKVQA
  H7 – Compositional         → Flickr30k Entities (rich NP-grounded scenes)
  H8 – Fine-grained          → iNaturalist 2021 (species in habitat context)

Datasets selected for proposer-friendly images
-----------------------------------------------
  visual_genome    — 108k images, avg 35 objects + 21 relations each
                     Perfect for H1 (counting), H3 (attr binding), H7 (composition)
  open_images      — Open Images V7 subset; multi-label, occluded, crowded scenes
                     Perfect for H1, H2, H4 (occlusion/state)
  aokvqa           — A-OKVQA: commonsense + visual knowledge QA; complex outdoor/indoor
                     Perfect for H6 (causal), H7 (compositional)
  flickr30k        — Flickr30k Entities: 5 captions + grounded NPs per image
                     Rich multi-person, multi-object outdoor scenes
  docci            — DOCCI (Descriptions of Connected and Contrasting Images):
                     Highly detailed descriptions of complex real scenes
  nocaps           — NoCaps: out-of-domain COCO-style captions; diverse real scenes
  sa1b_natural     — SA-1B (SAM) natural subset: complex real-world multi-object
  infovqa          — InfographicVQA: infographic images with spatial reasoning
                     (good complement to chart domain for E2)
  realworldqa      — RealWorldQA: spatial reasoning questions over real photos
                     Direct alignment with the RealWorldQA eval benchmark

Usage
-----
  # Default 10k pool for E1 joint training (natural + spatial emphasis)
  python download_rich_10k.py --output_dir data/rich_10k

  # Only the best 5 for spatial/multi-object (fastest)
  python download_rich_10k.py \\
      --datasets visual_genome,open_images,aokvqa,flickr30k,nocaps \\
      --output_dir data/rich_10k

  # Combine with existing pool
  python download_rich_10k.py --output_dir data/rich_10k --total_samples 6000
  # Then manually copy your existing data/benchmark_10k/images/* alongside.

After download, run the filter script to clean up remaining bad images:
  python scripts/filter_dataset_images.py \\
      --image_root data/rich_10k/images \\
      --min_side 336 --min_regions 3 --min_hue_clusters 3

Then update DATA_DIR in _common.sh or pass at runtime:
  DATA_DIR=/path/to/rich_10k/images bash E1_main_joint.sh
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.request import urlopen

from PIL import Image

try:
    from datasets import Dataset, IterableDataset, get_dataset_split_names, load_dataset
except Exception as exc:
    raise RuntimeError(
        "This script requires `datasets`. Install with: pip install datasets pillow"
    ) from exc

try:
    import numpy as np
except Exception:
    np = None

try:
    import requests
except Exception:
    requests = None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetSpec:
    alias: str
    dataset_ids: Tuple[str, ...]
    domain: str
    split: str = "train"
    fallback_splits: Tuple[str, ...] = ()
    config_name: Optional[str] = None
    prefer_streaming: bool = True
    # Human-readable rationale (for --list flag)
    rationale: str = ""


PRESET_SPECS: Dict[str, DatasetSpec] = {
    # ── Tier 1: Best for hard proposer questions ──────────────────────────────

    "visual_genome": DatasetSpec(
        alias="visual_genome",
        dataset_ids=(
            "visual_genome",
            "WenhaoWang/VG_100K",
            "HuggingFaceM4/VGG",
        ),
        domain="relational",
        split="train",
        prefer_streaming=True,
        rationale=(
            "108k images, avg 35 objects + 21 pairwise relations + dense attributes. "
            "Best single dataset for H1 (counting), H3 (attribute), H7 (composition). "
            "Almost every image has ≥5 distinct objects."
        ),
    ),

    "open_images": DatasetSpec(
        alias="open_images",
        dataset_ids=(
            "HuggingFaceM4/OpenImages",
            "jmhessel/open_images_v6",
            "detection-datasets/coco",           # fallback: COCO detection format
        ),
        domain="multi_object",
        split="train",
        fallback_splits=("validation",),
        prefer_streaming=True,
        rationale=(
            "Open Images V7 — crowded real-world scenes with multi-label annotations. "
            "Rich occlusion, stacking, and interaction → H2 (spatial), H4 (occlusion). "
            "Average 8+ visible objects per image."
        ),
    ),

    "aokvqa": DatasetSpec(
        alias="aokvqa",
        dataset_ids=(
            "HuggingFaceM4/A-OKVQA",
            "allenai/aokvqa",
        ),
        domain="commonsense",
        split="train",
        fallback_splits=("val",),
        prefer_streaming=True,
        rationale=(
            "A-OKVQA: commonsense + knowledge-grounded QA over complex real-world images. "
            "Questions require causal (H6) and compositional (H7) reasoning. "
            "Direct alignment with SEED-bench commonsense questions."
        ),
    ),

    "flickr30k": DatasetSpec(
        alias="flickr30k",
        dataset_ids=(
            "nlphuji/flickr30k",
            "awsaf49/flickr30k",
            "clip-benchmark/wds_flickr30k",
        ),
        domain="relational",
        split="train",
        fallback_splits=("test",),
        prefer_streaming=True,
        rationale=(
            "Flickr30k Entities: rich multi-person outdoor scenes with grounded NPs. "
            "5 reference captions per image → spatial + relational language. "
            "Ideal for H2 (spatial), H5 (interaction), H7 (grouping)."
        ),
    ),

    "nocaps": DatasetSpec(
        alias="nocaps",
        dataset_ids=(
            "HuggingFaceM4/NoCaps",
            "lmms-lab/NoCaps",
        ),
        domain="natural",
        split="val",
        fallback_splits=("train",),
        prefer_streaming=True,
        rationale=(
            "NoCaps out-of-domain COCO captioning benchmark — contains rare objects "
            "and uncommon configurations (in-domain, near-domain, out-of-domain). "
            "Natural images with high scene diversity."
        ),
    ),

    "docci": DatasetSpec(
        alias="docci",
        dataset_ids=(
            "google/docci",
            "HuggingFaceM4/DOCCI",
        ),
        domain="relational",
        split="train",
        fallback_splits=("test",),
        prefer_streaming=True,
        rationale=(
            "DOCCI (Descriptions of Connected and Contrasting Images) — "
            "purposely designed for fine-grained spatial, attribute, and "
            "compositional distinctions. Each image has a long detailed description. "
            "Ideal for H3 (attribute), H4 (state), H8 (fine-grained)."
        ),
    ),

    # ── Tier 2: Strong supplementary sources ─────────────────────────────────

    "realworldqa": DatasetSpec(
        alias="realworldqa",
        dataset_ids=(
            "xai-org/RealWorldQA",
            "lmms-lab/RealWorldQA",
        ),
        domain="spatial",
        split="test",
        fallback_splits=("train", "val"),
        prefer_streaming=True,
        rationale=(
            "RealWorldQA: spatial reasoning in real-world photos (driving, indoor, outdoor). "
            "DIRECT alignment with the RealWorldQA eval benchmark. "
            "H2 (spatial: left/right/behind), H4 (state), H5 (affordance)."
        ),
    ),

    "vsr": DatasetSpec(
        alias="vsr",
        dataset_ids=(
            "cambridgeltl/vsr_random",
            "cambridgeltl/VSR",
        ),
        domain="spatial",
        split="train",
        fallback_splits=("test",),
        prefer_streaming=True,
        rationale=(
            "Visual Spatial Reasoning (VSR): 65 spatial relation types "
            "(left-of, behind, above, …) over real-world images. "
            "Best dataset for pure H2 spatial questions."
        ),
    ),

    "sa1b_complex": DatasetSpec(
        alias="sa1b_complex",
        dataset_ids=(
            "facebook/segment-anything-1-billion",
            "HuggingFaceM4/COCO",
            "detection-datasets/coco",
        ),
        domain="multi_object",
        split="train",
        prefer_streaming=True,
        rationale=(
            "SA-1B (SAM) images from complex crowded scenes — markets, streets, "
            "kitchens, workshops. We use same IDs as download_benchmark_10k but "
            "filter for high mask-count images after download."
        ),
    ),

    "mscoco_dense": DatasetSpec(
        alias="mscoco_dense",
        dataset_ids=(
            "HuggingFaceM4/COCO",
            "detection-datasets/coco",
            "astro21/coco-caption-train-split-10k",
        ),
        domain="multi_object",
        split="train",
        prefer_streaming=True,
        rationale=(
            "MSCOCO train split filtered to images with ≥5 annotations "
            "(already in use but we apply tighter filtering here). "
            "Keeps complex crowd / kitchen / sports scenes."
        ),
    ),

    "infovqa": DatasetSpec(
        alias="infovqa",
        dataset_ids=(
            "HuggingFaceM4/InfographicVQA",
            "lmms-lab/InfographicVQA",
        ),
        domain="spatial_document",
        split="train",
        prefer_streaming=True,
        rationale=(
            "InfographicVQA: infographics with spatial layout reasoning "
            "(arrows, tables, timelines). Distinct from chart-math domain. "
            "Good for layout-aware spatial questions (H2) and causal flows (H6)."
        ),
    ),

    # ── Tier 3: Specialist domains ────────────────────────────────────────────

    "gqa_balanced": DatasetSpec(
        alias="gqa_balanced",
        dataset_ids=(
            "lmms-lab/GQA",
            "Graphcore/gqa",
            "echarlaix/gqa",
        ),
        domain="compositional",
        split="train_balanced",
        fallback_splits=("train", "testdev"),
        config_name=None,
        prefer_streaming=True,
        rationale=(
            "GQA balanced split: compositional scene-graph questions. "
            "You already use GQA but the BALANCED split has harder questions "
            "vs. the all-split which is dominated by simple color/object queries."
        ),
    ),

    "refcoco_context": DatasetSpec(
        alias="refcoco_context",
        dataset_ids=(
            "lmms-lab/RefCOCO",
            "arnabdhar/RefCOCO-Instruct",
        ),
        domain="spatial",
        split="train",
        prefer_streaming=True,
        rationale=(
            "RefCOCO/RefCOCO+: grounded referring expression images. "
            "The underlying COCO images contain complex multi-object scenes. "
            "Good for H2 (spatial) and H7 (distinguishing similar objects)."
        ),
    ),
}

# ── Domain weights for --total_samples allocation ────────────────────────────
DEFAULT_DOMAIN_WEIGHTS = {
    "relational":        0.30,   # visual_genome, flickr30k, docci
    "multi_object":      0.25,   # open_images, sa1b_complex, mscoco_dense
    "spatial":           0.20,   # realworldqa, vsr, refcoco_context
    "commonsense":       0.15,   # aokvqa
    "natural":           0.05,   # nocaps
    "compositional":     0.03,   # gqa_balanced
    "spatial_document":  0.02,   # infovqa
}

DEFAULT_DATASETS = (
    "visual_genome,open_images,aokvqa,flickr30k,nocaps,docci,"
    "realworldqa,vsr,mscoco_dense"
)


# ─────────────────────────────────────────────────────────────────────────────
# Utility (identical to download_benchmark_10k.py)
# ─────────────────────────────────────────────────────────────────────────────

def _largest_remainder_allocation(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    s = sum(float(v) for v in weights.values())
    norm = {k: float(v) / s for k, v in weights.items()}
    raw = {k: total * v for k, v in norm.items()}
    floors = {k: int(raw[k]) for k in raw}
    remainder = total - sum(floors.values())
    ranked = sorted(raw.keys(), key=lambda k: (raw[k] - floors[k], k), reverse=True)
    for k in ranked[:remainder]:
        floors[k] += 1
    return floors


def allocate_alias_targets(aliases: Sequence[str], total: int) -> Dict[str, int]:
    aliases_by_domain: Dict[str, List[str]] = {}
    for alias in aliases:
        domain = PRESET_SPECS[alias].domain
        aliases_by_domain.setdefault(domain, []).append(alias)

    active_weights = {
        d: w for d, w in DEFAULT_DOMAIN_WEIGHTS.items()
        if d in aliases_by_domain
    }
    if not active_weights:
        # Fallback: uniform
        return {a: total // len(aliases) for a in aliases}

    domain_targets = _largest_remainder_allocation(total, active_weights)
    alias_targets: Dict[str, int] = {a: 0 for a in aliases}

    for domain, d_target in domain_targets.items():
        domain_aliases = sorted(aliases_by_domain.get(domain, []))
        if not domain_aliases:
            continue
        base = d_target // len(domain_aliases)
        rem = d_target - base * len(domain_aliases)
        for i, alias in enumerate(domain_aliases):
            alias_targets[alias] = base + (1 if i < rem else 0)

    return alias_targets


def load_with_fallback(spec: DatasetSpec, *, streaming_only: bool):
    errors: List[str] = []
    for dataset_id in spec.dataset_ids:
        split_candidates = [spec.split] + list(spec.fallback_splits)
        try:
            discovered = get_dataset_split_names(dataset_id, config_name=spec.config_name)
            for s in discovered:
                if s not in split_candidates:
                    split_candidates.append(s)
        except Exception:
            pass
        for split_name in split_candidates:
            for streaming in ([True] if streaming_only else [True, False]):
                try:
                    kwargs = {"path": dataset_id, "split": split_name, "streaming": streaming}
                    if spec.config_name:
                        kwargs["name"] = spec.config_name
                    ds = load_dataset(**kwargs)
                    if streaming_only and not isinstance(ds, IterableDataset):
                        continue
                    return ds, dataset_id, streaming, split_name
                except Exception as exc:
                    errors.append(f"id={dataset_id} split={split_name}: {repr(exc)[:120]}")
    raise RuntimeError(
        f"Failed to load '{spec.alias}'. Attempts:\n  " + "\n  ".join(errors)
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
        if value.startswith("http://") or value.startswith("https://"):
            loader = requests if requests else None
            try:
                if loader:
                    resp = loader.get(value, timeout=10)
                    if resp.status_code == 200:
                        return Image.open(io.BytesIO(resp.content)).convert("RGB")
                else:
                    with urlopen(value, timeout=10) as r:
                        return Image.open(io.BytesIO(r.read())).convert("RGB")
            except Exception:
                return None
        return None
    if isinstance(value, dict):
        for key in ("bytes", "path", "array"):
            if key in value:
                img = try_decode_image(value[key])
                if img is not None:
                    return img
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
            return None
    return None


def extract_image(record: Dict) -> Tuple[Optional[Image.Image], Optional[str]]:
    preferred = {"image", "img", "photo", "jpg", "jpeg", "png", "webp",
                 "image_url", "url", "uri", "path", "image_path"}
    tokens = ("image", "img", "photo", "jpg", "jpeg", "png", "url", "uri", "path")

    for key, value in record.items():
        if str(key).lower() in preferred:
            img = try_decode_image(value)
            if img is not None:
                return img, str(key)
    for key, value in record.items():
        if any(t in str(key).lower() for t in tokens):
            img = try_decode_image(value)
            if img is not None:
                return img, str(key)
    for key, value in record.items():
        if not isinstance(value, str):
            img = try_decode_image(value)
            if img is not None:
                return img, str(key)
    return None, None


def iter_records(ds, *, seed: int, max_records: int, shuffle_buffer: int) -> Iterator[Dict]:
    if isinstance(ds, IterableDataset):
        buf = max(1_000, min(max_records, shuffle_buffer))
        for idx, row in enumerate(ds.shuffle(seed=seed, buffer_size=buf)):
            if idx >= max_records:
                break
            yield row
        return
    if isinstance(ds, Dataset):
        indices = list(range(len(ds)))
        random.Random(seed).shuffle(indices)
        for idx in indices[:max_records]:
            yield ds[int(idx)]
        return
    for idx, row in enumerate(ds):
        if idx >= max_records:
            break
        yield row


def save_subset(
    spec: DatasetSpec,
    *,
    output_root: Path,
    target: int,
    seed: int,
    strict: bool,
    max_scan_multiplier: int,
    shuffle_buffer: int,
    streaming_only: bool,
    min_side: int,
) -> Dict:
    ds, dataset_id, streaming, used_split = load_with_fallback(
        spec, streaming_only=streaming_only
    )
    images_dir = output_root / "images" / spec.alias
    manifests_dir = output_root / "manifests"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / f"{spec.alias}.jsonl"

    saved = 0
    scanned = 0
    skipped_small = 0
    max_scan = max(target * max_scan_multiplier, target)

    with manifest_path.open("w") as mf:
        for record in iter_records(
            ds, seed=seed, max_records=max_scan, shuffle_buffer=shuffle_buffer
        ):
            scanned += 1
            img, key = extract_image(record)
            if img is None:
                continue

            # Inline resolution pre-filter (avoids saving obviously bad images)
            w, h = img.size
            if min(w, h) < min_side:
                skipped_small += 1
                continue

            out_name = f"{saved:06d}.jpg"
            out_path = images_dir / out_name
            img.save(out_path, format="JPEG", quality=95, optimize=True)
            mf.write(
                json.dumps({
                    "alias": spec.alias,
                    "dataset_id": dataset_id,
                    "split": used_split,
                    "image_key": key,
                    "saved_index": saved,
                    "image_relpath": str(Path("images") / spec.alias / out_name),
                    "width": w,
                    "height": h,
                }) + "\n"
            )
            saved += 1
            if saved >= target:
                break

    if strict and saved < target:
        raise RuntimeError(
            f"Dataset '{spec.alias}' saved {saved}/{target} images "
            f"(scanned={scanned}, skipped_small={skipped_small})."
        )
    return {
        "alias": spec.alias,
        "domain": spec.domain,
        "dataset_id": dataset_id,
        "split": used_split,
        "saved": saved,
        "target": target,
        "scanned": scanned,
        "skipped_small": skipped_small,
        "output_dir": str(images_dir),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download rich, proposer-friendly image pool for UUG self-evolving training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output_dir", type=Path, default=Path("data/rich_10k"))
    p.add_argument(
        "--datasets", type=str, default=DEFAULT_DATASETS,
        help=f"Comma-separated aliases. Available: {', '.join(sorted(PRESET_SPECS.keys()))}",
    )
    p.add_argument(
        "--list", action="store_true",
        help="List all available datasets with rationale and exit.",
    )
    p.add_argument("--total_samples", type=int, default=10_000)
    p.add_argument("--samples_per_dataset", type=int, default=None,
                   help="Override per-dataset (ignores --total_samples).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle_buffer", type=int, default=20_000)
    p.add_argument("--max_scan_multiplier", type=int, default=15)
    p.add_argument("--streaming_only", action="store_true", default=True)
    p.add_argument("--allow_non_streaming", dest="streaming_only", action="store_false")
    p.add_argument(
        "--min_side", type=int, default=224,
        help="Inline resolution pre-filter during download (default 224). "
             "Run filter_dataset_images.py after for full filtering.",
    )
    p.add_argument("--strict", action="store_true", default=False)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        print(f"{'Alias':20s} {'Domain':20s} Rationale")
        print("─" * 120)
        for alias, spec in sorted(PRESET_SPECS.items()):
            print(f"{alias:20s} {spec.domain:20s} {spec.rationale[:80]}")
        return 0

    aliases = [a.strip() for a in args.datasets.split(",") if a.strip()]
    unknown = [a for a in aliases if a not in PRESET_SPECS]
    if unknown:
        print(f"[download] ERROR: unknown alias(es): {unknown}", file=sys.stderr)
        return 1

    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.samples_per_dataset is not None:
        alias_targets = {a: args.samples_per_dataset for a in aliases}
    else:
        alias_targets = allocate_alias_targets(aliases, args.total_samples)

    print(f"[download] Rich 10k pool → {out_root}")
    print(f"[download] Datasets: {', '.join(aliases)}")
    print(f"[download] Per-alias targets:")
    for alias, t in alias_targets.items():
        spec = PRESET_SPECS[alias]
        print(f"  {alias:20s}  {t:5d}  ({spec.domain})")

    summary: List[Dict] = []
    failures: List[str] = []

    for i, alias in enumerate(aliases):
        spec = PRESET_SPECS[alias]
        target = alias_targets.get(alias, 0)
        if target <= 0:
            continue
        print(f"\n[download] ── {alias} (domain={spec.domain}, target={target})")
        try:
            result = save_subset(
                spec,
                output_root=out_root,
                target=target,
                seed=args.seed + i,
                strict=args.strict,
                max_scan_multiplier=args.max_scan_multiplier,
                shuffle_buffer=args.shuffle_buffer,
                streaming_only=args.streaming_only,
                min_side=args.min_side,
            )
            summary.append(result)
            fill_pct = 100.0 * result["saved"] / max(1, target)
            print(
                f"  saved={result['saved']}/{target} ({fill_pct:.0f}%)  "
                f"scanned={result['scanned']}  skipped_small={result['skipped_small']}"
            )
        except Exception as exc:
            failures.append(f"{alias}: {exc}")
            print(f"  FAILED: {exc}", file=sys.stderr)
            if args.strict:
                break

    summary_path = out_root / "download_summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "output_root": str(out_root),
                "datasets": aliases,
                "alias_targets": alias_targets,
                "total_samples": args.total_samples,
                "results": summary,
                "failures": failures,
            },
            f,
            indent=2,
        )

    total_saved = sum(r["saved"] for r in summary)
    print(f"\n[download] Done. Total saved: {total_saved:,}")
    print(f"[download] Summary: {summary_path}")
    if failures:
        print("[download] Failures:")
        for f_msg in failures:
            print(f"  {f_msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
