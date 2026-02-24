#!/usr/bin/env python3
"""
filter_dataset_images.py — Remove small / single-object images from a downloaded
image pool (produced by download_benchmark_10k.py or download_joint_3k.py).

Philosophy
----------
The proposer's H1-H8 hard strategies (counting, spatial, occlusion, depth,
affordance, causal, compositional) all REQUIRE:
  • Multiple distinct objects or regions
  • Sufficient resolution to distinguish fine-grained attributes
  • Enough scene complexity to generate non-trivial distractors

Images that fail these criteria cause the proposer to fall back to trivial
"what color is the object?" questions → entropy ≈ 0 → no reward gradient.

Filtering pipeline (each image must pass ALL gates):
  1. Resolution gate   : min(W, H) >= MIN_SIDE   (default: 336 px)
                         max(W, H) >= MAX_SIDE   (default: 448 px)
  2. Aspect-ratio gate : W/H and H/W both <= MAX_ASPECT  (default: 3.0)
                         rejects thin panoramic strips / banners
  3. Pixel-variance gate: stdev of grayscale pixel values >= MIN_STD
                         (default: 18.0)  — rejects blank / near-solid images
  4. Edge-complexity gate: Laplacian variance >= MIN_LAPLACIAN  (default: 40.0)
                         — rejects blurry images with no structure
  5. Multi-region gate  : number of spatially distinct edge-dense blocks
                         >= MIN_REGIONS  (default: 3)
                         Divides image into NxN grid (default 4x4), counts blocks
                         with above-threshold edge energy → proxy for ≥3 objects
  6. Color-diversity gate: number of dominant hues (k-means in HSV space)
                         >= MIN_HUE_CLUSTERS  (default: 3)
                         rejects monochrome single-object images
  7. (Optional) JPEG size gate: compressed file size >= MIN_JPEG_KB (default: 15 KB)
                         tiny files are almost always solid-color or text icons

Outputs
-------
  • Keeps accepted images in-place (or copies to --output_dir if provided)
  • Writes a JSON report: filter_report.json in the output directory
  • Prints per-alias and overall statistics

Usage
-----
  # Dry-run: just print statistics, do not move/delete anything
  python filter_dataset_images.py --image_root data/benchmark_10k/images --dry_run

  # In-place filter (deletes rejected images)
  python filter_dataset_images.py --image_root data/benchmark_10k/images

  # Save accepted images to a NEW directory (keeps originals untouched)
  python filter_dataset_images.py \\
      --image_root data/benchmark_10k/images \\
      --output_dir data/benchmark_10k_filtered/images

  # Strict mode: tighten gates for best proposer quality
  python filter_dataset_images.py \\
      --image_root data/benchmark_10k/images \\
      --min_side 448 --min_laplacian 80 --min_regions 4 --min_hue_clusters 4

  # Loose mode: keep most images, only remove truly bad ones
  python filter_dataset_images.py \\
      --image_root data/benchmark_10k/images \\
      --min_side 224 --min_std 10 --min_laplacian 20 --min_regions 2

Requirements
------------
  pip install pillow numpy scipy
  (scipy is used only for label connectivity; falls back to simple block-count)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

# ── Optional scipy for connected-component counting ───────────────────────────
try:
    from scipy.ndimage import label as scipy_label
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Core filter logic
# ─────────────────────────────────────────────────────────────────────────────

def _load_image(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, Exception):
        return None


def _grayscale_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of discrete Laplacian — measures edge sharpness / detail."""
    kernel = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]], dtype=np.float32)
    # Manual 2-D convolution (no scipy/cv2 dependency)
    from numpy.lib.stride_tricks import sliding_window_view
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    windows = sliding_window_view(gray, (3, 3))          # (H-2, W-2, 3, 3)
    conv = (windows * kernel).sum(axis=(-2, -1))         # (H-2, W-2)
    return float(np.var(conv))


def _count_edge_regions(gray: np.ndarray, grid_n: int = 4,
                         edge_thresh: float = 20.0) -> int:
    """
    Divide into grid_n × grid_n non-overlapping blocks.
    Count blocks whose mean |Laplacian| > edge_thresh.
    Proxy for "number of distinct structured regions".
    """
    h, w = gray.shape
    bh, bw = max(1, h // grid_n), max(1, w // grid_n)
    count = 0
    for r in range(grid_n):
        for c in range(grid_n):
            block = gray[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            if block.size == 0:
                continue
            lap = np.abs(np.diff(block, axis=0)).mean() + \
                  np.abs(np.diff(block, axis=1)).mean()
            if lap > edge_thresh:
                count += 1
    return count


def _count_hue_clusters(img: Image.Image, n_colors: int = 8,
                          min_hue_clusters: int = 3) -> int:
    """
    Quantise to n_colors palette colors, convert to HSV, count distinct hue
    buckets (10° wide) that appear.  Returns number of occupied hue buckets.
    Fast: uses PIL quantize (median-cut), no sklearn needed.
    """
    small = img.resize((64, 64), Image.BILINEAR)
    quantized = small.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    palette_rgb = quantized.getpalette()          # flat list [R,G,B, R,G,B ...]
    if palette_rgb is None:
        return 0
    palette_rgb_arr = np.array(palette_rgb, dtype=np.uint8).reshape(-1, 3)[:n_colors]

    # Count how many palette colors appear in the image
    hist = quantized.histogram()[:n_colors]
    total_pixels = small.size[0] * small.size[1]
    min_px = max(1, int(total_pixels * 0.01))          # ≥1% of pixels

    hue_buckets = set()
    for idx, count in enumerate(hist):
        if count < min_px:
            continue
        r, g, b = palette_rgb_arr[idx].astype(np.float32) / 255.0
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        delta = max_c - min_c
        if delta < 0.10:                               # near-achromatic → no hue
            # Still count as a hue bucket (white/black/grey counts as 1 bucket)
            hue_buckets.add(-1)
            continue
        if max_c == r:
            hue = 60.0 * (((g - b) / delta) % 6)
        elif max_c == g:
            hue = 60.0 * (((b - r) / delta) + 2)
        else:
            hue = 60.0 * (((r - g) / delta) + 4)
        bucket = int(hue // 10) % 36                  # 10° buckets
        hue_buckets.add(bucket)

    return len(hue_buckets)


def evaluate_image(
    path: Path,
    *,
    min_side: int,
    max_side: int,
    max_aspect: float,
    min_std: float,
    min_laplacian: float,
    min_regions: int,
    min_hue_clusters: int,
    min_jpeg_kb: float,
    grid_n: int,
) -> Tuple[bool, str, Dict]:
    """
    Returns (accepted: bool, reject_reason: str, stats: dict).
    reject_reason is '' when accepted.
    """
    stats: Dict = {"path": str(path)}

    # ── 0. File-size gate (cheap, do first) ──────────────────────────────────
    try:
        file_kb = path.stat().st_size / 1024.0
    except OSError:
        return False, "file_not_found", stats
    stats["file_kb"] = round(file_kb, 1)
    if file_kb < min_jpeg_kb:
        return False, f"file_too_small({file_kb:.1f}KB<{min_jpeg_kb}KB)", stats

    # ── 1. Load ───────────────────────────────────────────────────────────────
    img = _load_image(path)
    if img is None:
        return False, "load_failed", stats

    w, h = img.size
    stats["width"] = w
    stats["height"] = h

    # ── 2. Resolution gate ───────────────────────────────────────────────────
    short_side = min(w, h)
    long_side  = max(w, h)
    stats["short_side"] = short_side
    if short_side < min_side:
        return False, f"short_side({short_side}<{min_side})", stats
    if long_side < max_side:
        return False, f"long_side({long_side}<{max_side})", stats

    # ── 3. Aspect-ratio gate ─────────────────────────────────────────────────
    aspect = long_side / max(short_side, 1)
    stats["aspect"] = round(aspect, 2)
    if aspect > max_aspect:
        return False, f"aspect({aspect:.2f}>{max_aspect})", stats

    # ── 4. Pixel-variance gate ───────────────────────────────────────────────
    gray = _grayscale_array(img)
    std = float(np.std(gray))
    stats["pixel_std"] = round(std, 2)
    if std < min_std:
        return False, f"pixel_std({std:.2f}<{min_std})", stats

    # ── 5. Edge-complexity (Laplacian variance) gate ─────────────────────────
    lap_var = _laplacian_variance(gray)
    stats["laplacian_var"] = round(lap_var, 2)
    if lap_var < min_laplacian:
        return False, f"laplacian({lap_var:.2f}<{min_laplacian})", stats

    # ── 6. Multi-region gate ─────────────────────────────────────────────────
    n_regions = _count_edge_regions(gray, grid_n=grid_n)
    stats["edge_regions"] = n_regions
    if n_regions < min_regions:
        return False, f"edge_regions({n_regions}<{min_regions})", stats

    # ── 7. Color-diversity gate ───────────────────────────────────────────────
    n_hues = _count_hue_clusters(img, min_hue_clusters=min_hue_clusters)
    stats["hue_clusters"] = n_hues
    if n_hues < min_hue_clusters:
        return False, f"hue_clusters({n_hues}<{min_hue_clusters})", stats

    return True, "", stats


# ─────────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────────

def _collect_image_paths(image_root: Path) -> List[Path]:
    """Recursively collect .jpg / .jpeg / .png files."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = []
    for f in sorted(image_root.rglob("*")):
        if f.suffix.lower() in exts and f.is_file():
            paths.append(f)
    return paths


def run_filter(args: argparse.Namespace) -> int:
    image_root = Path(args.image_root).expanduser().resolve()
    if not image_root.exists():
        print(f"[filter] ERROR: image_root does not exist: {image_root}", file=sys.stderr)
        return 1

    output_dir: Optional[Path] = None
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    report_path = (output_dir or image_root.parent) / "filter_report.json"

    print(f"[filter] Scanning images in: {image_root}")
    all_paths = _collect_image_paths(image_root)
    print(f"[filter] Found {len(all_paths):,} image files")
    if not all_paths:
        print("[filter] Nothing to do.")
        return 0

    # ── Filter parameters ─────────────────────────────────────────────────────
    params = dict(
        min_side=args.min_side,
        max_side=args.max_side,
        max_aspect=args.max_aspect,
        min_std=args.min_std,
        min_laplacian=args.min_laplacian,
        min_regions=args.min_regions,
        min_hue_clusters=args.min_hue_clusters,
        min_jpeg_kb=args.min_jpeg_kb,
        grid_n=args.grid_n,
    )
    print("[filter] Filter thresholds:")
    for k, v in params.items():
        print(f"  {k:30s} = {v}")
    if args.dry_run:
        print("[filter] DRY-RUN mode: no files will be modified.")

    # ── Per-alias counters ────────────────────────────────────────────────────
    alias_stats: Dict[str, Dict] = {}   # alias -> {accepted, rejected, reasons}
    accepted_paths: List[Path] = []
    rejected_paths: List[Path] = []
    reject_reasons: Dict[str, int] = {}
    all_stats: List[Dict] = []

    total = len(all_paths)
    for i, path in enumerate(all_paths):
        if (i + 1) % 500 == 0 or (i + 1) == total:
            pct = 100.0 * (i + 1) / total
            print(f"  [{i+1:6d}/{total}] {pct:5.1f}%  accepted={len(accepted_paths)}  "
                  f"rejected={len(rejected_paths)}", end="\r")

        # Alias is the subdirectory name directly under image_root
        try:
            rel = path.relative_to(image_root)
            alias = rel.parts[0] if len(rel.parts) > 1 else "_root"
        except ValueError:
            alias = "_root"

        if alias not in alias_stats:
            alias_stats[alias] = {"accepted": 0, "rejected": 0, "total": 0}
        alias_stats[alias]["total"] += 1

        ok, reason, stats = evaluate_image(path, **params)
        stats["alias"] = alias
        stats["accepted"] = ok
        stats["reason"] = reason
        all_stats.append(stats)

        if ok:
            accepted_paths.append(path)
            alias_stats[alias]["accepted"] += 1
            if output_dir and not args.dry_run:
                dest = output_dir / path.relative_to(image_root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
        else:
            rejected_paths.append(path)
            alias_stats[alias]["rejected"] += 1
            reject_reasons[reason.split("(")[0]] = reject_reasons.get(reason.split("(")[0], 0) + 1
            if not output_dir and not args.dry_run:
                path.unlink(missing_ok=True)

    print()  # newline after \r

    # ── Summary ───────────────────────────────────────────────────────────────
    n_acc = len(accepted_paths)
    n_rej = len(rejected_paths)
    acc_pct = 100.0 * n_acc / max(1, total)

    print(f"\n[filter] ══════════════ RESULTS ══════════════")
    print(f"  Total scanned : {total:,}")
    print(f"  Accepted      : {n_acc:,}  ({acc_pct:.1f}%)")
    print(f"  Rejected      : {n_rej:,}  ({100-acc_pct:.1f}%)")

    print(f"\n[filter] Rejection breakdown:")
    for reason, cnt in sorted(reject_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:40s} {cnt:6,}  ({100*cnt/max(1,total):.1f}%)")

    print(f"\n[filter] Per-alias breakdown:")
    for alias in sorted(alias_stats):
        s = alias_stats[alias]
        pct = 100.0 * s["accepted"] / max(1, s["total"])
        print(f"  {alias:20s}  total={s['total']:5,}  accepted={s['accepted']:5,}  "
              f"rejected={s['rejected']:5,}  ({pct:.1f}%)")

    if args.dry_run:
        print(f"\n[filter] DRY-RUN: {n_rej:,} images WOULD be removed.")
    elif output_dir:
        print(f"\n[filter] Accepted images copied to: {output_dir}")
    else:
        print(f"\n[filter] Removed {n_rej:,} rejected images in-place.")

    # ── Write report ──────────────────────────────────────────────────────────
    report = {
        "image_root": str(image_root),
        "output_dir": str(output_dir) if output_dir else None,
        "dry_run": args.dry_run,
        "filter_params": params,
        "summary": {
            "total": total,
            "accepted": n_acc,
            "rejected": n_rej,
            "acceptance_rate_pct": round(acc_pct, 2),
        },
        "rejection_breakdown": reject_reasons,
        "per_alias": alias_stats,
        "per_image": all_stats if args.save_per_image_report else [],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[filter] Report written to: {report_path}")

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Filter small / single-object images from a self-evolving training pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument(
        "--image_root", type=str, required=True,
        help="Root directory of images (e.g. data/benchmark_10k/images). "
             "Sub-directories per alias are expected (alias/000000.jpg …).",
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="If provided, copy ACCEPTED images here (original files untouched). "
             "If omitted, rejected files are DELETED in-place.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Report statistics only; do not move or delete anything.",
    )
    p.add_argument(
        "--save_per_image_report", action="store_true",
        help="Include per-image stats in the JSON report (large file).",
    )

    # Resolution thresholds
    res_grp = p.add_argument_group("Resolution thresholds")
    res_grp.add_argument(
        "--min_side", type=int, default=336,
        help="Minimum of (width, height) in pixels. "
             "336 = ViT-L/14 336px input; raise to 448 for stricter filtering.",
    )
    res_grp.add_argument(
        "--max_side", type=int, default=448,
        help="Minimum of max(width, height).  Images smaller than this in BOTH "
             "dimensions lack enough spatial extent for hard questions.",
    )
    res_grp.add_argument(
        "--max_aspect", type=float, default=3.0,
        help="Max aspect ratio (long/short). Banners, strips, icon rows are ≥3.",
    )

    # Content thresholds
    content_grp = p.add_argument_group("Content complexity thresholds")
    content_grp.add_argument(
        "--min_std", type=float, default=18.0,
        help="Minimum grayscale pixel std-dev. <10 = near-blank / solid-color.",
    )
    content_grp.add_argument(
        "--min_laplacian", type=float, default=40.0,
        help="Minimum Laplacian variance (edge sharpness). "
             "<20 = blurry/featureless. 40–80 keeps most natural photos.",
    )
    content_grp.add_argument(
        "--min_regions", type=int, default=3,
        help="Minimum number of grid blocks with detectable edge energy. "
             "proxy for ≥3 distinct spatial regions / objects.",
    )
    content_grp.add_argument(
        "--grid_n", type=int, default=4,
        help="Grid size for region counting (grid_n × grid_n blocks).",
    )
    content_grp.add_argument(
        "--min_hue_clusters", type=int, default=3,
        help="Minimum distinct color hue clusters. "
             "<3 = monochrome / single-material image.",
    )
    content_grp.add_argument(
        "--min_jpeg_kb", type=float, default=15.0,
        help="Minimum JPEG file size in KB. Tiny files are near-blank or icons.",
    )

    return p


def main() -> int:
    args = _build_parser().parse_args()
    return run_filter(args)


if __name__ == "__main__":
    raise SystemExit(main())
