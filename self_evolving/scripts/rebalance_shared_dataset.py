#!/usr/bin/env python3
"""Rebalance an existing shared_uug dataset into equal per-source counts.

Creates a new dataset directory with:
- images/<source>/*.jpg
- manifests/{all,train,val,test}.jsonl
- rebalance_summary.json

For low-resource sources, it oversamples with replacement and creates hardlinks
(to avoid disk duplication). If hardlink fails, it falls back to copy.
"""

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebalance shared_uug dataset to equal source counts")
    ap.add_argument(
        "--input_dir",
        type=str,
        default="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/data/shared_uug_50k",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="/Users/ritesh.thawkar/Ritesh/self-evolving-uug/data/shared_uug_50k_balanced",
    )
    ap.add_argument("--target_total", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.90)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    in_manifest = input_dir / "manifests" / "all.jsonl"
    if not in_manifest.exists():
        raise FileNotFoundError(f"Missing input manifest: {in_manifest}")

    rows = read_jsonl(in_manifest)
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        source = r.get("source", "unknown")
        if source == "plotqa_topup":
            source = "plotqa"
        rr = dict(r)
        rr["source"] = source
        by_source[source].append(rr)

    sources = sorted(by_source.keys())
    n_sources = len(sources)
    if n_sources == 0:
        raise RuntimeError("No sources found in manifest")

    base = args.target_total // n_sources
    rem = args.target_total % n_sources
    target_by_source = {s: base + (1 if i < rem else 0) for i, s in enumerate(sources)}

    rnd = random.Random(args.seed)
    selected: List[dict] = []
    source_stats: Dict[str, dict] = {}

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    print("Rebalancing sources:")
    for s in sources:
        pool = by_source[s]
        available = len(pool)
        target = target_by_source[s]
        print(f"- {s}: available={available}, target={target}")

        if available >= target:
            chosen = rnd.sample(pool, target)
            oversampled = 0
        else:
            chosen = list(pool)
            oversampled = target - available
            for _ in range(oversampled):
                chosen.append(rnd.choice(pool))

        source_stats[s] = {
            "available": available,
            "target": target,
            "oversampled": oversampled,
        }

        for idx, rec in enumerate(chosen):
            src_abs = Path(rec["image_abspath"])
            if not src_abs.exists():
                # fallback using input_dir/images + relpath
                src_abs = input_dir / "images" / rec["image_relpath"]
            if not src_abs.exists():
                continue

            dst_rel = Path(s) / f"{s}_{idx:06d}.jpg"
            dst_abs = output_dir / "images" / dst_rel
            make_link_or_copy(src_abs, dst_abs)

            new_rec = dict(rec)
            new_rec["rebalance_source"] = s
            new_rec["image_relpath"] = dst_rel.as_posix()
            new_rec["image_abspath"] = str(dst_abs)
            new_rec["rebalance_idx"] = idx
            selected.append(new_rec)

    # if any missing due broken paths, top-up from existing selected
    while len(selected) < args.target_total and selected:
        rec = rnd.choice(selected)
        s = rec["rebalance_source"]
        idx = len([x for x in selected if x.get("rebalance_source") == s])
        src_abs = Path(rec["image_abspath"])
        dst_rel = Path(s) / f"{s}_{idx:06d}.jpg"
        dst_abs = output_dir / "images" / dst_rel
        make_link_or_copy(src_abs, dst_abs)
        nrec = dict(rec)
        nrec["image_relpath"] = dst_rel.as_posix()
        nrec["image_abspath"] = str(dst_abs)
        nrec["rebalance_idx"] = idx
        selected.append(nrec)

    # trim if over target
    if len(selected) > args.target_total:
        selected = rnd.sample(selected, args.target_total)

    rnd.shuffle(selected)

    n = len(selected)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train = selected[:n_train]
    val = selected[n_train:n_train + n_val]
    test = selected[n_train + n_val:]

    manifests = output_dir / "manifests"
    write_jsonl(manifests / "all.jsonl", selected)
    write_jsonl(manifests / "train.jsonl", train)
    write_jsonl(manifests / "val.jsonl", val)
    write_jsonl(manifests / "test.jsonl", test)

    # split source counts
    def count_by_source(rows: List[dict]) -> Dict[str, int]:
        out = defaultdict(int)
        for r in rows:
            out[r.get("rebalance_source", r.get("source", "unknown"))] += 1
        return dict(sorted(out.items()))

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target_total": args.target_total,
        "actual_total": n,
        "split_counts": {"train": n_train, "val": n_val, "test": n_test},
        "source_targets": target_by_source,
        "source_stats": source_stats,
        "all_source_counts": count_by_source(selected),
        "train_source_counts": count_by_source(train),
        "val_source_counts": count_by_source(val),
        "test_source_counts": count_by_source(test),
        "seed": args.seed,
    }

    with (output_dir / "rebalance_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nRebalance complete")
    print(json.dumps({
        "actual_total": n,
        "split_counts": summary["split_counts"],
        "all_source_counts": summary["all_source_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
