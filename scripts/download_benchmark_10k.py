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
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.request import urlopen

from PIL import Image

try:
    from datasets import IterableDataset, load_dataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires `datasets`. Install with: pip install datasets pillow"
    ) from exc

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


@dataclass(frozen=True)
class DatasetSpec:
    alias: str
    dataset_ids: Tuple[str, ...]
    domain: str
    split: str = "train"
    fallback_splits: Tuple[str, ...] = ()
    config_name: Optional[str] = None
    prefer_streaming: bool = True


PRESET_SPECS: Dict[str, DatasetSpec] = {
    # Benchmark/domain aligned and >=10k in typical train splits.
    "chartqa": DatasetSpec(
        alias="chartqa",
        dataset_ids=("ahmed-masry/ChartQA",),
        domain="chart_math",
        split="train",
        prefer_streaming=True,
    ),
    "docvqa": DatasetSpec(
        alias="docvqa",
        dataset_ids=("lmms-lab/DocVQA",),
        domain="ocr",
        config_name="DocVQA",
        split="validation",
        fallback_splits=("test",),
        prefer_streaming=True,
    ),
    "sa1b": DatasetSpec(
        alias="sa1b",
        dataset_ids=(
            # Preferred id (may be unavailable in some environments)
            "facebook/segment-anything-1-billion",
            # Fallback natural-image datasets
            "HuggingFaceM4/COCO",
            "astro21/coco-caption-train-split-10k",
        ),
        domain="natural",
        split="train",
        prefer_streaming=True,
    ),
    "laion_coco": DatasetSpec(
        alias="laion_coco",
        dataset_ids=("laion/laion-coco",),
        domain="natural",
        split="train",
        prefer_streaming=True,
    ),
    # Optional OCR-heavy domain. Some mirrors vary; tries IDs in order.
    "textvqa": DatasetSpec(
        alias="textvqa",
        dataset_ids=(
            "lmms-lab/TextVQA",
            "lmms-lab/textvqa",
            "textvqa",
        ),
        domain="ocr",
        split="train",
        prefer_streaming=True,
    ),
    # Optional chart/math domain expansion.
    "mathvista": DatasetSpec(
        alias="mathvista",
        dataset_ids=(
            "AI4Math/MathVista",
            "lmms-lab/MathVista",
        ),
        domain="chart_math",
        split="train",
        prefer_streaming=True,
    ),
    # Optional compositional reasoning visuals.
    "gqa": DatasetSpec(
        alias="gqa",
        dataset_ids=(
            "lmms-lab/GQA",
            "Graphcore/gqa",
            "echarlaix/gqa",
        ),
        domain="compositional",
        split="train",
        prefer_streaming=True,
    ),
    "coco10k": DatasetSpec(
        alias="coco10k",
        dataset_ids=("astro21/coco-caption-train-split-10k",),
        domain="natural",
        split="train",
        prefer_streaming=True,
    ),
}

DEFAULT_DOMAIN_DISTRIBUTION = "natural:0.40,ocr:0.25,chart_math:0.20,compositional:0.15"


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
        default="sa1b,laion_coco,docvqa,textvqa,chartqa,mathvista,gqa",
        help=(
            "Comma-separated preset aliases. "
            f"Available: {', '.join(sorted(PRESET_SPECS.keys()))}"
        ),
    )
    parser.add_argument(
        "--samples_per_dataset",
        type=int,
        default=10_000,
        help=(
            "Target sample count per dataset (default: 10000). "
            "Ignored if --total_samples is provided."
        ),
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=None,
        help=(
            "Combined total samples across selected datasets. "
            "When set, per-dataset targets are allocated by --domain_distribution."
        ),
    )
    parser.add_argument(
        "--domain_distribution",
        type=str,
        default=DEFAULT_DOMAIN_DISTRIBUTION,
        help=(
            "Domain weights for --total_samples. Format: "
            "'natural:0.40,ocr:0.25,chart_math:0.20,compositional:0.15'."
        ),
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


def _largest_remainder_allocation(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights:
        raise ValueError("weights must be non-empty")
    s = sum(float(v) for v in weights.values())
    if s <= 0:
        raise ValueError("weights must sum to a positive value")
    norm = {k: float(v) / s for k, v in weights.items()}
    raw = {k: total * v for k, v in norm.items()}
    floors = {k: int(raw[k]) for k in raw}
    remainder = total - sum(floors.values())
    ranked = sorted(raw.keys(), key=lambda k: (raw[k] - floors[k], k), reverse=True)
    for k in ranked[:remainder]:
        floors[k] += 1
    return floors


def _parse_domain_distribution(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"Invalid domain_distribution token '{part}'. Expected '<domain>:<weight>'."
            )
        domain, weight = part.split(":", 1)
        domain = domain.strip()
        weight = float(weight.strip())
        if weight < 0:
            raise ValueError(f"Weight for domain '{domain}' must be non-negative.")
        out[domain] = weight
    if not out:
        raise ValueError("domain_distribution parsed empty.")
    return out


def allocate_alias_targets(
    aliases: Sequence[str],
    total_samples: int,
    domain_distribution: Dict[str, float],
) -> Dict[str, int]:
    if total_samples <= 0:
        raise ValueError("total_samples must be > 0")

    aliases_by_domain: Dict[str, List[str]] = {}
    for alias in aliases:
        domain = PRESET_SPECS[alias].domain
        aliases_by_domain.setdefault(domain, []).append(alias)

    present_domains = set(aliases_by_domain.keys())
    active_weights = {
        domain: weight
        for domain, weight in domain_distribution.items()
        if domain in present_domains
    }
    if not active_weights:
        raise ValueError(
            "None of the selected datasets map to domains in --domain_distribution."
        )

    domain_targets = _largest_remainder_allocation(total_samples, active_weights)
    alias_targets: Dict[str, int] = {a: 0 for a in aliases}

    for domain, d_target in domain_targets.items():
        domain_aliases = sorted(aliases_by_domain.get(domain, []))
        if not domain_aliases:
            continue
        # Evenly split domain quota across datasets in this domain.
        base = d_target // len(domain_aliases)
        rem = d_target - base * len(domain_aliases)
        for i, alias in enumerate(domain_aliases):
            alias_targets[alias] = base + (1 if i < rem else 0)

    # Safety: keep exact sum.
    assigned = sum(alias_targets.values())
    if assigned != total_samples:
        raise RuntimeError(
            f"Internal allocation mismatch: assigned={assigned}, total={total_samples}"
        )
    return alias_targets


def load_with_fallback(spec: DatasetSpec) -> Tuple[IterableDataset, str, str]:
    """Always loads as a streaming IterableDataset — never downloads the full dataset.

    Tries each (dataset_id, split) pair in declaration order.
    Does NOT call get_dataset_split_names() to avoid triggering metadata downloads.
    """
    errors: List[str] = []
    split_candidates: List[str] = [spec.split] + [
        s for s in spec.fallback_splits if s != spec.split
    ]
    for dataset_id in spec.dataset_ids:
        for split_name in split_candidates:
            try:
                kwargs: Dict = {
                    "path": dataset_id,
                    "split": split_name,
                    "streaming": True,   # ALWAYS streaming — never materialise on disk
                    "trust_remote_code": False,
                }
                if spec.config_name:
                    kwargs["name"] = spec.config_name
                ds = load_dataset(**kwargs)
                if not isinstance(ds, IterableDataset):
                    # Refuse non-streaming result to avoid accidental full download.
                    errors.append(
                        f"id={dataset_id} split={split_name}: "
                        "load_dataset returned non-streaming Dataset despite "
                        "streaming=True; skipping to next candidate."
                    )
                    continue
                return ds, dataset_id, split_name
            except Exception as exc:
                errors.append(
                    f"id={dataset_id} split={split_name}: {repr(exc)[:160]}"
                )
    raise RuntimeError(
        f"Could not stream dataset preset '{spec.alias}' from any source.\n"
        + "\n".join(f"  • {e}" for e in errors)
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
            # Support URL-backed datasets without requiring the requests package.
            if requests is not None:
                try:
                    resp = requests.get(value, timeout=10)
                    if resp.status_code != 200:
                        return None
                    return Image.open(io.BytesIO(resp.content)).convert("RGB")
                except Exception:
                    return None
            try:
                with urlopen(value, timeout=10) as resp:
                    content = resp.read()
                return Image.open(io.BytesIO(content)).convert("RGB")
            except Exception:
                return None
        # Do not probe arbitrary strings as local paths; this is slow and causes
        # path-length errors on caption-like fields.
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
    preferred_exact = {
        "image",
        "img",
        "photo",
        "jpg",
        "jpeg",
        "png",
        "webp",
        "image_url",
        "url",
        "uri",
        "path",
        "image_path",
        "source_url",
    }
    preferred_tokens = ("image", "img", "photo", "jpg", "jpeg", "png", "webp", "url", "uri", "path")

    # Pass 1: exact well-known key names (case-insensitive).
    for key, value in record.items():
        key_s = str(key).lower()
        if key_s in preferred_exact:
            img = try_decode_image(value)
            if img is not None:
                return img, str(key)

    # Pass 2: heuristic key names that look image-related.
    for key, value in record.items():
        key_s = str(key).lower()
        if any(token in key_s for token in preferred_tokens):
            img = try_decode_image(value)
            if img is not None:
                return img, str(key)

    # Pass 3: only non-string structured payloads; skip free-form text fields.
    for key, value in record.items():
        if isinstance(value, str):
            continue
        img = try_decode_image(value)
        if img is not None:
            return img, str(key)
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
                        "split": used_split,
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
        "domain": spec.domain,
        "dataset_id": dataset_id,
        "split": used_split,
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

    if args.total_samples is not None:
        distribution = _parse_domain_distribution(args.domain_distribution)
        alias_targets = allocate_alias_targets(
            aliases=aliases,
            total_samples=int(args.total_samples),
            domain_distribution=distribution,
        )
        print(f"[download] Combined target: {args.total_samples}")
        print(f"[download] Domain distribution: {args.domain_distribution}")
        print(
            "[download] Per-dataset targets: "
            + ", ".join(f"{k}:{alias_targets[k]}" for k in aliases)
        )
    else:
        alias_targets = {alias: int(args.samples_per_dataset) for alias in aliases}

    print(f"[download] Output root: {out_root}")
    if args.total_samples is None:
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
        target = alias_targets[alias]
        if target <= 0:
            continue
        run_seed = args.seed + i
        print(f"\n[download] -> {alias} (domain={spec.domain}, target={target}, seed={run_seed})")
        try:
            result = save_subset_for_spec(
                spec,
                output_root=out_root,
                samples_target=target,
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
                "total_samples": args.total_samples,
                "selected_aliases": aliases,
                "alias_targets": alias_targets,
                "domain_distribution": args.domain_distribution,
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
    if args.total_samples is not None:
        total_saved = sum(int(r.get("saved", 0)) for r in summary)
        if args.strict and total_saved != int(args.total_samples):
            print(
                f"[download] ERROR: strict total mismatch saved={total_saved} "
                f"expected={args.total_samples}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
