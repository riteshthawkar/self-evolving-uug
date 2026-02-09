#!/usr/bin/env python3
"""Create a split-aware structured dataset layout from an existing shared_uug dataset.

Output layout:
  <output_dir>/
    train/<dataset_name>/*.jpg
    val/<dataset_name>/*.jpg
    test/<dataset_name>/*.jpg
    manifests/{all,train,val,test}.jsonl
    structure_summary.json

It links images by default (hardlink), falling back to copy if linking fails.
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ALLOWED_SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def normalize_source(row: dict) -> str:
    val = row.get("dataset_name") or row.get("rebalance_source") or row.get("source")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return "unknown"


def resolve_src_path(row: dict, input_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []

    image_abspath = row.get("image_abspath")
    if isinstance(image_abspath, str) and image_abspath.strip():
        candidates.append(Path(image_abspath))

    for key in ("structured_relpath", "image_relpath", "path", "filepath"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        rel = Path(value)
        candidates.append(input_dir / rel)
        candidates.append(input_dir / "images" / rel)

    seen = set()
    for cand in candidates:
        abs_path = cand.expanduser().resolve()
        if abs_path in seen:
            continue
        seen.add(abs_path)
        if abs_path.exists() and abs_path.is_file():
            return abs_path
    return None


def process_split(
    split: str,
    rows: List[dict],
    *,
    input_dir: Path,
    output_dir: Path,
    link_mode: str,
    counters: Dict[str, int],
) -> tuple[List[dict], int]:
    records_out: List[dict] = []
    skipped_missing = 0

    for row in rows:
        source = normalize_source(row)
        src_path = resolve_src_path(row, input_dir=input_dir)
        if src_path is None:
            skipped_missing += 1
            continue

        idx_key = f"{split}:{source}"
        idx = counters[idx_key]
        counters[idx_key] += 1

        dst_rel = Path(split) / source / f"{source}_{idx:06d}.jpg"
        dst_abs = output_dir / dst_rel
        link_or_copy(src_path, dst_abs, mode=link_mode)

        out_row = dict(row)
        out_row["source"] = source
        out_row["dataset_name"] = source
        out_row["split"] = split
        out_row["structured_relpath"] = dst_rel.as_posix()
        out_row["image_relpath"] = dst_rel.as_posix()
        out_row["image_abspath"] = str(dst_abs.resolve())
        out_row["structured_index"] = idx
        records_out.append(out_row)

    return records_out, skipped_missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Restructure shared_uug dataset into split/dataset folders.")
    ap.add_argument(
        "--input_dir",
        type=str,
        default="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/data/shared_uug_50k_balanced",
        help="Input dataset dir containing manifests/{train,val,test}.jsonl and images.",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/data/shared_uug_50k_balanced_structured",
        help="Output dir for structured dataset.",
    )
    ap.add_argument(
        "--link_mode",
        type=str,
        default="hardlink",
        choices=("hardlink", "copy"),
        help="Use hardlinks (default) or full file copies.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output_dir if it exists.",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset directory not found: {input_dir}")

    manifests_in = input_dir / "manifests"
    split_rows: Dict[str, List[dict]] = {}
    for split in ALLOWED_SPLITS:
        path = manifests_in / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing required manifest: {path}")
        split_rows[split] = read_jsonl(path)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counters: Dict[str, int] = defaultdict(int)
    out_by_split: Dict[str, List[dict]] = {}
    skipped_by_split: Dict[str, int] = {}

    for split in ALLOWED_SPLITS:
        out_rows, skipped = process_split(
            split,
            split_rows[split],
            input_dir=input_dir,
            output_dir=output_dir,
            link_mode=args.link_mode,
            counters=counters,
        )
        out_by_split[split] = out_rows
        skipped_by_split[split] = skipped

    all_rows: List[dict] = []
    for split in ALLOWED_SPLITS:
        all_rows.extend(out_by_split[split])

    manifests_out = output_dir / "manifests"
    write_jsonl(manifests_out / "all.jsonl", all_rows)
    for split in ALLOWED_SPLITS:
        write_jsonl(manifests_out / f"{split}.jsonl", out_by_split[split])

    split_counts = {split: len(out_by_split[split]) for split in ALLOWED_SPLITS}
    source_counts: Dict[str, Dict[str, int]] = {}
    for split, rows in out_by_split.items():
        c: Dict[str, int] = defaultdict(int)
        for row in rows:
            c[normalize_source(row)] += 1
        source_counts[split] = dict(sorted(c.items()))

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "layout": "<split>/<dataset_name>/*.jpg",
        "link_mode": args.link_mode,
        "counts": {
            "all": len(all_rows),
            **split_counts,
        },
        "skipped_missing_images": skipped_by_split,
        "source_counts": source_counts,
    }

    with (output_dir / "structure_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
