#!/usr/bin/env python3
"""
Build a high-utility unlabeled image pool for self-evolving BLIP3o training.

The pipeline is deliberately low-space:
  * reuse local image pools with hardlinks by default;
  * download only selected Open Images thumbnails, not full datasets;
  * keep resumable manifests and per-image scores;
  * make VLM judging optional and bounded by an explicit image limit.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import requests
from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

OPENIMAGES_METADATA_URL = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-with-rotation.csv"
)
OPENIMAGES_BOXES_URL = (
    "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
)
OPENIMAGES_RELATIONSHIPS_URL = (
    "https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-vrd.csv"
)

DEFAULT_DOMAIN_WEIGHTS = {
    "relational": 0.24,
    "spatial": 0.20,
    "ocr": 0.16,
    "chart_document": 0.10,
    "openimages_dense": 0.20,
    "natural": 0.10,
}

SOURCE_DOMAIN_HINTS = {
    "textvqa": "ocr",
    "docvqa": "chart_document",
    "chartqa": "chart_document",
    "infographic": "chart_document",
    "mathvista": "chart_document",
    "realworldqa": "spatial",
    "vsr": "spatial",
    "gqa": "relational",
    "visual_genome": "relational",
    "flickr30k": "relational",
    "nocaps": "natural",
    "coco": "natural",
    "open_images": "openimages_dense",
    "openimages": "openimages_dense",
}


@dataclass
class Candidate:
    candidate_id: str
    source: str
    domain: str
    path: Optional[str] = None
    url: Optional[str] = None
    source_url: Optional[str] = None
    license: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotation_score: float = 0.0


@dataclass
class ScoredCandidate:
    candidate: Candidate
    accepted: bool
    reject_reason: str
    heuristic_score: float
    final_score: float
    width: int = 0
    height: int = 0
    short_side: int = 0
    aspect: float = 0.0
    file_kb: float = 0.0
    pixel_std: float = 0.0
    laplacian_var: float = 0.0
    edge_regions: int = 0
    hue_clusters: int = 0
    ocr_proxy: float = 0.0
    density_proxy: float = 0.0
    predicted_disagreement: float = 0.0
    dhash: str = ""
    sha256: str = ""
    vlm_score: Optional[Dict[str, Any]] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a 10k high-utility unlabeled image pool for BLIP3o self-evolution."
    )
    p.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    p.add_argument("--output_dir", type=Path, default=Path("data/high_utility_pool_10k"))
    p.add_argument(
        "--local_source",
        action="append",
        default=[],
        help="Local image root to reuse. Can be passed multiple times.",
    )
    p.add_argument("--target_count", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_short_side", type=int, default=336)
    p.add_argument("--max_aspect", type=float, default=3.0)
    p.add_argument("--min_file_kb", type=float, default=12.0)
    p.add_argument("--min_heuristic_score", type=float, default=0.46)
    p.add_argument("--duplicate_hamming_threshold", type=int, default=4)
    p.add_argument("--link_mode", choices=["hardlink", "copy"], default="hardlink")
    p.add_argument("--normalize_long_side", type=int, default=896)
    p.add_argument("--jpeg_quality", type=int, default=92)
    p.add_argument("--max_output_gb", type=float, default=8.0)
    p.add_argument("--domain_weights", type=str, default="")
    p.add_argument("--write_rejected_images", action="store_true", default=False)

    p.add_argument("--no_openimages", action="store_true", default=False)
    p.add_argument("--openimages_target", type=int, default=7000)
    p.add_argument("--openimages_min_boxes", type=int, default=4)
    p.add_argument("--openimages_cache_dir", type=Path, default=Path("data_pipeline/high_utility_pool/cache/openimages"))
    p.add_argument("--download_timeout", type=float, default=20.0)
    p.add_argument("--download_retries", type=int, default=2)
    p.add_argument(
        "--download_workers",
        type=int,
        default=8,
        help="Parallel workers for URL download and image scoring.",
    )

    p.add_argument(
        "--vlm_backend",
        choices=["none", "openai", "openai_compatible", "gemini", "mlx_vlm", "transformers_vlm"],
        default="none",
        help="Optional expensive VLM judge. Use after heuristic prefiltering.",
    )
    p.add_argument("--vlm_model", type=str, default="")
    p.add_argument(
        "--vlm_dtype",
        type=str,
        default=os.environ.get("VLM_DTYPE", "auto"),
        help="Local Transformers VLM dtype: auto, bfloat16, float16, or float32.",
    )
    p.add_argument(
        "--vlm_device_map",
        type=str,
        default=os.environ.get("VLM_DEVICE_MAP", "auto"),
        help="Local Transformers VLM device_map, usually auto on a large GPU node.",
    )
    p.add_argument(
        "--vlm_attn_implementation",
        type=str,
        default=os.environ.get("VLM_ATTN_IMPLEMENTATION", ""),
        help="Optional local Transformers attention implementation, e.g. flash_attention_2 or sdpa.",
    )
    p.add_argument(
        "--vlm_trust_remote_code",
        action="store_true",
        default=os.environ.get("VLM_TRUST_REMOTE_CODE", "0") == "1",
        help="Pass trust_remote_code=True when loading a local Transformers VLM.",
    )
    p.add_argument(
        "--vlm_batch_size",
        type=int,
        default=int(os.environ.get("VLM_BATCH_SIZE", "1")),
        help="Batch size for local Transformers VLM judging. Does not change the judge prompt or scoring rule.",
    )
    p.add_argument(
        "--vlm_max_new_tokens",
        type=int,
        default=int(os.environ.get("VLM_MAX_NEW_TOKENS", "384")),
        help="Maximum generation tokens for VLM JSON judgments.",
    )
    p.add_argument(
        "--vlm_base_url",
        type=str,
        default="",
        help="OpenAI-compatible server base URL, e.g. http://127.0.0.1:8000/v1.",
    )
    p.add_argument(
        "--vlm_api_key_env",
        type=str,
        default="OPENAI_COMPATIBLE_API_KEY",
        help="Environment variable containing the OpenAI-compatible API key.",
    )
    p.add_argument("--vlm_timeout", type=float, default=180.0)
    p.add_argument("--vlm_max_images", type=int, default=0)
    p.add_argument("--vlm_candidate_top_k", type=int, default=3000)
    p.add_argument(
        "--vlm_selection_strategy",
        choices=["stratified", "top"],
        default="stratified",
        help="Which accepted candidates the VLM audits before final ranking.",
    )
    p.add_argument("--vlm_sleep_sec", type=float, default=0.2)
    p.add_argument("--vlm_cache_path", type=Path, default=None)

    p.add_argument("--smoke_test", action="store_true", help="Alias for a tiny, local-only run.")
    p.add_argument("--dry_run", action="store_true", help="Score and report, but do not create final image pool.")
    return _merge_config(p.parse_args())


def _merge_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        if args.smoke_test:
            args.target_count = min(args.target_count, 64)
            args.no_openimages = True
        return args
    explicit_cli_keys = {
        token[2:].split("=", 1)[0].replace("-", "_")
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    with args.config.expanduser().open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key, value in cfg.items():
        if not hasattr(args, key):
            continue
        if key in explicit_cli_keys:
            continue
        current = getattr(args, key)
        if isinstance(current, Path):
            value = Path(value)
        setattr(args, key, value)
    if args.smoke_test:
        args.target_count = min(args.target_count, 64)
        args.no_openimages = True
    return args


def _jsonl_append(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def _json_dump(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[jsonl] warning: skipping malformed line {line_no} in {path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )


def _parse_domain_weights(text: str) -> Dict[str, float]:
    if not text:
        return dict(DEFAULT_DOMAIN_WEIGHTS)
    out: Dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid domain weight token: {part!r}")
        name, value = part.split(":", 1)
        out[name.strip()] = float(value)
    if not out:
        return dict(DEFAULT_DOMAIN_WEIGHTS)
    return out


def _infer_domain(path_or_alias: str) -> str:
    text = path_or_alias.lower()
    if text in DEFAULT_DOMAIN_WEIGHTS:
        return text
    for key, domain in SOURCE_DOMAIN_HINTS.items():
        if key in text:
            return domain
    return "natural"


def _safe_id(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out[:160] or hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def iter_local_candidates(roots: Sequence[str]) -> Iterator[Candidate]:
    for root_text in roots:
        root = Path(root_text).expanduser().resolve()
        if not root.exists():
            print(f"[local] WARNING: missing source {root}", file=sys.stderr)
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = path.relative_to(root)
            source_alias = rel.parts[0] if len(rel.parts) > 1 else root.name
            source = f"local:{source_alias}"
            domain = _infer_domain(source_alias)
            cid = _safe_id(f"{source_alias}_{rel.as_posix()}")
            yield Candidate(
                candidate_id=cid,
                source=source,
                domain=domain,
                path=str(path),
                metadata={"relative_path": rel.as_posix(), "root": str(root)},
            )


def _download_file(url: str, path: Path, *, timeout: float, retries: int):
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Optional[Exception] = None
    for attempt in range(max(1, retries + 1)):
        try:
            with requests.get(url, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".tmp")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, path)
                return
        except Exception as exc:
            last_exc = exc
            time.sleep(min(5.0, 0.5 * (attempt + 1)))
    raise RuntimeError(f"failed to download {url}: {last_exc}")


def _ensure_openimages_cache(cache_dir: Path, *, timeout: float, retries: int):
    files = {
        "metadata": (OPENIMAGES_METADATA_URL, cache_dir / "validation-images-with-rotation.csv"),
        "boxes": (OPENIMAGES_BOXES_URL, cache_dir / "validation-annotations-bbox.csv"),
        "relationships": (OPENIMAGES_RELATIONSHIPS_URL, cache_dir / "validation-annotations-vrd.csv"),
    }
    for name, (url, path) in files.items():
        print(f"[openimages] cache {name}: {path}")
        _download_file(url, path, timeout=timeout, retries=retries)


def _build_openimages_annotation_scores(cache_dir: Path) -> Dict[str, Dict[str, Any]]:
    scores: Dict[str, Dict[str, Any]] = {}
    boxes_path = cache_dir / "validation-annotations-bbox.csv"
    with boxes_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("ImageID") or row.get("ImageId")
            if not image_id:
                continue
            rec = scores.setdefault(
                image_id,
                {
                    "box_count": 0,
                    "labels": set(),
                    "occluded": 0,
                    "truncated": 0,
                    "group_of": 0,
                    "inside": 0,
                },
            )
            rec["box_count"] += 1
            if row.get("LabelName"):
                rec["labels"].add(row["LabelName"])
            rec["occluded"] += 1 if row.get("IsOccluded") == "1" else 0
            rec["truncated"] += 1 if row.get("IsTruncated") == "1" else 0
            rec["group_of"] += 1 if row.get("IsGroupOf") == "1" else 0
            rec["inside"] += 1 if row.get("IsInside") == "1" else 0

    relationships_path = cache_dir / "validation-annotations-vrd.csv"
    if relationships_path.exists():
        with relationships_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get("ImageID") or row.get("ImageId")
                if not image_id:
                    continue
                rec = scores.setdefault(
                    image_id,
                    {
                        "box_count": 0,
                        "labels": set(),
                        "occluded": 0,
                        "truncated": 0,
                        "group_of": 0,
                        "inside": 0,
                    },
                )
                rec["relationship_count"] = int(rec.get("relationship_count", 0)) + 1

    for rec in scores.values():
        labels = rec.get("labels", set())
        rec["label_count"] = len(labels)
        rec["labels"] = sorted(labels)[:32]
        box_count = int(rec.get("box_count", 0))
        rel_count = int(rec.get("relationship_count", 0))
        label_count = int(rec.get("label_count", 0))
        occluded = int(rec.get("occluded", 0))
        group_of = int(rec.get("group_of", 0))
        rec["annotation_score"] = (
            min(box_count / 12.0, 1.0) * 0.40
            + min(label_count / 8.0, 1.0) * 0.20
            + min(rel_count / 4.0, 1.0) * 0.25
            + min((occluded + group_of) / 4.0, 1.0) * 0.15
        )
    return scores


def iter_openimages_candidates(
    *,
    cache_dir: Path,
    target: int,
    min_boxes: int,
    timeout: float,
    retries: int,
) -> Iterator[Candidate]:
    _ensure_openimages_cache(cache_dir, timeout=timeout, retries=retries)
    annotation = _build_openimages_annotation_scores(cache_dir)
    metadata_path = cache_dir / "validation-images-with-rotation.csv"
    rows: List[Tuple[float, Dict[str, str], Dict[str, Any]]] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("ImageID") or row.get("ImageId")
            if not image_id:
                continue
            ann = annotation.get(image_id)
            if not ann:
                continue
            if int(ann.get("box_count", 0)) < min_boxes:
                continue
            url = row.get("Thumbnail300KURL") or row.get("OriginalURL")
            if not url:
                continue
            rows.append((float(ann.get("annotation_score", 0.0)), row, ann))
    rows.sort(key=lambda item: item[0], reverse=True)
    for score, row, ann in rows[: max(0, target)]:
        image_id = row.get("ImageID") or row.get("ImageId") or ""
        yield Candidate(
            candidate_id=_safe_id(f"openimages_v7_validation_{image_id}"),
            source="openimages_v7_validation",
            domain="openimages_dense",
            url=row.get("Thumbnail300KURL") or row.get("OriginalURL"),
            source_url=row.get("OriginalLandingURL") or row.get("OriginalURL"),
            license=row.get("License"),
            annotation_score=score,
            metadata={
                "image_id": image_id,
                "author": row.get("Author"),
                "title": row.get("Title"),
                "rotation": row.get("Rotation"),
                "openimages": ann,
            },
        )


def _open_image(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img.load()
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _load_candidate_image(
    candidate: Candidate,
    *,
    cache_dir: Path,
    timeout: float,
    retries: int,
) -> Tuple[Optional[Image.Image], Optional[Path], str]:
    if candidate.path:
        path = Path(candidate.path)
        return _open_image(path), path, ""
    if not candidate.url:
        return None, None, "missing_path_or_url"
    cache_path = cache_dir / "downloaded" / f"{candidate.candidate_id}.jpg"
    if not cache_path.exists():
        try:
            _download_file(candidate.url, cache_path, timeout=timeout, retries=retries)
        except Exception as exc:
            return None, None, f"download_failed:{type(exc).__name__}"
    img = _open_image(cache_path)
    if img is None:
        return None, cache_path, "image_load_failed"
    return img, cache_path, ""


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1] * -4.0
    lap = center + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(np.var(lap))


def _edge_regions(gray: np.ndarray, grid_n: int = 4) -> Tuple[int, float]:
    h, w = gray.shape
    bh, bw = max(1, h // grid_n), max(1, w // grid_n)
    energies: List[float] = []
    for r in range(grid_n):
        for c in range(grid_n):
            block = gray[r * bh : min(h, (r + 1) * bh), c * bw : min(w, (c + 1) * bw)]
            if block.size < 4:
                continue
            dy = np.abs(np.diff(block, axis=0)).mean()
            dx = np.abs(np.diff(block, axis=1)).mean()
            energies.append(float(dx + dy))
    if not energies:
        return 0, 0.0
    threshold = max(18.0, float(np.median(energies)) * 0.8)
    count = sum(1 for value in energies if value >= threshold)
    return int(count), float(sum(energies) / len(energies))


def _hue_clusters(img: Image.Image) -> int:
    small = img.resize((64, 64), Image.BILINEAR)
    q = small.quantize(colors=10, method=Image.Quantize.MEDIANCUT)
    palette = q.getpalette()
    if palette is None:
        return 0
    colors = np.array(palette, dtype=np.uint8).reshape(-1, 3)[:10]
    hist = q.histogram()[:10]
    buckets = set()
    for idx, count in enumerate(hist):
        if count < 40:
            continue
        r, g, b = [float(x) / 255.0 for x in colors[idx]]
        max_c, min_c = max(r, g, b), min(r, g, b)
        delta = max_c - min_c
        if delta < 0.08:
            buckets.add(-1)
            continue
        if max_c == r:
            hue = 60.0 * (((g - b) / delta) % 6)
        elif max_c == g:
            hue = 60.0 * (((b - r) / delta) + 2)
        else:
            hue = 60.0 * (((r - g) / delta) + 4)
        buckets.add(int(hue // 20))
    return len(buckets)


def _dhash(img: Image.Image) -> str:
    small = img.convert("L").resize((9, 8), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.int16)
    bits = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def _hamming_hex(a: str, b: str) -> int:
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return 64


def _image_sha256(path: Optional[Path], img: Image.Image) -> str:
    if path is not None and path.exists():
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _ocr_proxy(gray: np.ndarray) -> float:
    """Cheap text/layout proxy: horizontal/vertical edge richness in small regions."""
    if gray.shape[0] < 12 or gray.shape[1] < 12:
        return 0.0
    dy = np.abs(np.diff(gray, axis=0))
    dx = np.abs(np.diff(gray, axis=1))
    edge = (dy[:-1, :] > 18).mean() + (dx[:, :-1] > 18).mean()
    # Text-rich and document/chart images often have many sharp aligned edges.
    return float(min(edge * 4.0, 1.0))


def score_image(
    candidate: Candidate,
    img: Image.Image,
    path: Optional[Path],
    *,
    min_short_side: int,
    max_aspect: float,
    min_file_kb: float,
) -> ScoredCandidate:
    w, h = img.size
    short, long = min(w, h), max(w, h)
    aspect = float(long / max(1, short))
    file_kb = float(path.stat().st_size / 1024.0) if path is not None and path.exists() else 0.0
    gray = np.asarray(img.convert("L").resize((256, 256), Image.BILINEAR), dtype=np.float32)
    pixel_std = float(np.std(gray))
    lap = _laplacian_variance(gray)
    edge_count, edge_energy = _edge_regions(gray)
    hue_count = _hue_clusters(img)
    ocr = _ocr_proxy(gray)
    dh = _dhash(img)
    sha = _image_sha256(path, img)

    reject = ""
    if short < min_short_side:
        reject = f"short_side<{min_short_side}"
    elif aspect > max_aspect:
        reject = f"aspect>{max_aspect}"
    elif file_kb and file_kb < min_file_kb:
        reject = f"file_kb<{min_file_kb}"
    elif pixel_std < 12.0:
        reject = "low_pixel_variance"
    elif lap < 8.0:
        reject = "low_sharpness"

    clarity = 0.55 * min(lap / 90.0, 1.0) + 0.45 * min(pixel_std / 55.0, 1.0)
    resolution = min(short / 768.0, 1.0)
    density = min(edge_count / 12.0, 1.0)
    color = min(hue_count / 5.0, 1.0)
    annotation = float(candidate.annotation_score)
    domain_bonus = 0.10 if candidate.domain in {"ocr", "chart_document", "spatial", "relational"} else 0.0
    predicted_disagreement = min(
        1.0,
        0.45 * density + 0.20 * min(ocr, 1.0) + 0.20 * annotation + 0.15 * color,
    )
    heuristic = (
        0.22 * clarity
        + 0.14 * resolution
        + 0.23 * density
        + 0.12 * color
        + 0.12 * min(ocr, 1.0)
        + 0.12 * annotation
        + 0.05 * predicted_disagreement
        + domain_bonus
    )
    heuristic = float(max(0.0, min(1.0, heuristic)))

    return ScoredCandidate(
        candidate=candidate,
        accepted=not bool(reject),
        reject_reason=reject,
        heuristic_score=heuristic,
        final_score=heuristic,
        width=w,
        height=h,
        short_side=short,
        aspect=aspect,
        file_kb=file_kb,
        pixel_std=pixel_std,
        laplacian_var=lap,
        edge_regions=edge_count,
        hue_clusters=hue_count,
        ocr_proxy=ocr,
        density_proxy=float(edge_energy),
        predicted_disagreement=predicted_disagreement,
        dhash=dh,
        sha256=sha,
    )


def score_candidate_worker(
    candidate: Candidate,
    *,
    cache_dir: Path,
    timeout: float,
    retries: int,
    min_short_side: int,
    max_aspect: float,
    min_file_kb: float,
    min_heuristic_score: float,
) -> ScoredCandidate:
    img, local_path, load_reason = _load_candidate_image(
        candidate,
        cache_dir=cache_dir,
        timeout=timeout,
        retries=retries,
    )
    if img is None:
        return ScoredCandidate(
            candidate=candidate,
            accepted=False,
            reject_reason=load_reason or "load_failed",
            heuristic_score=0.0,
            final_score=0.0,
        )
    item = score_image(
        candidate,
        img,
        local_path,
        min_short_side=min_short_side,
        max_aspect=max_aspect,
        min_file_kb=min_file_kb,
    )
    if item.heuristic_score < min_heuristic_score:
        item.accepted = False
        item.reject_reason = item.reject_reason or "heuristic_score_below_threshold"
    return item


VLM_PROMPT = """You are filtering images for a self-evolving vision-language training loop.
We need images that help a proposer generate hard, objective, visually grounded VQA questions.
Score the image as JSON only with these keys:
clarity, object_density, relation_density, ocr_richness, counting_affordance,
spatial_affordance, attribute_affordance, ambiguity_risk, unsafe_or_private_risk,
final_keep_score, possible_question_types, short_reason.
Scores are 1 to 5 except risks, where 1 is low risk and 5 is high risk.
Keep possible_question_types to at most 4 short labels, not full questions.
Keep short_reason under 16 words and do not include text outside the JSON.
Prefer images with many objects, relations, readable text/layout, spatial structure,
attribute/state differences, and low ambiguity. Penalize blurry, single-object,
private, unsafe, watermark-heavy, or unanswerable images."""


def _image_data_url(img: Image.Image, max_side: int = 768, quality: int = 85) -> str:
    work = img.copy()
    work.thumbnail((max_side, max_side), Image.BILINEAR)
    buf = io.BytesIO()
    work.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_jsonish(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def judge_openai(img: Image.Image, *, model: str) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model or "gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": VLM_PROMPT},
                    {"type": "input_image", "image_url": _image_data_url(img)},
                ],
            }
        ],
    )
    return _parse_jsonish(response.output_text)


def judge_openai_compatible(
    img: Image.Image,
    *,
    model: str,
    base_url: str,
    api_key_env: str,
    timeout: float,
) -> Dict[str, Any]:
    if not model:
        raise ValueError("--vlm_model is required for openai_compatible backend")
    resolved_base_url = base_url or os.environ.get("OPENAI_COMPATIBLE_BASE_URL") or "http://127.0.0.1:8000/v1"
    api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    from openai import OpenAI

    client = OpenAI(base_url=resolved_base_url, api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_PROMPT},
                    {"type": "image_url", "image_url": {"url": _image_data_url(img)}},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=512,
    )
    content = response.choices[0].message.content or ""
    return _parse_jsonish(content)


def judge_gemini(img: Image.Image, *, model: str) -> Dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise RuntimeError("Gemini backend requires `google-genai`.") from exc

    client = genai.Client()
    work = img.copy()
    work.thumbnail((768, 768), Image.BILINEAR)
    buf = io.BytesIO()
    work.save(buf, format="JPEG", quality=85, optimize=True)
    response = client.models.generate_content(
        model=model or "gemini-2.0-flash",
        contents=[
            VLM_PROMPT,
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
        ],
    )
    return _parse_jsonish(response.text or "")


_MLX_VLM_STATE: Dict[str, Any] = {}
_TRANSFORMERS_VLM_STATE: Dict[str, Any] = {}


def judge_mlx_vlm(image_path: Path, *, model: str) -> Dict[str, Any]:
    """Judge one image with a local MLX VLM loaded once per process."""
    model_id = model or "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    state_key = f"mlx::{model_id}"
    state = _MLX_VLM_STATE.get(state_key)
    if state is None:
        # Avoid the Xet path on low-space machines; standard HTTP cache is more
        # transparent and resumed cleanly in local testing.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        local_model, processor = load(model_id)
        config = load_config(model_id)
        state = {
            "model": local_model,
            "processor": processor,
            "config": config,
            "generate": generate,
            "apply_chat_template": apply_chat_template,
        }
        _MLX_VLM_STATE[state_key] = state

    prompt = state["apply_chat_template"](
        state["processor"],
        state["config"],
        VLM_PROMPT,
        num_images=1,
    )
    result = state["generate"](
        state["model"],
        state["processor"],
        prompt=prompt,
        image=str(image_path),
        max_tokens=384,
        temperature=0.0,
        verbose=False,
    )
    payload = _parse_jsonish(str(getattr(result, "text", "") or ""))
    payload["_local_vlm_backend"] = "mlx_vlm"
    payload["_local_vlm_model"] = model_id
    payload["_generation_tokens"] = int(getattr(result, "generation_tokens", 0) or 0)
    payload["_peak_memory_gb"] = float(getattr(result, "peak_memory", 0.0) or 0.0)
    return payload


def _resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    name = (dtype_name or "auto").strip().lower()
    if name in {"", "auto"}:
        return "auto"
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    attr = aliases.get(name)
    if attr is None or not hasattr(torch_module, attr):
        raise ValueError(f"Unsupported --vlm_dtype '{dtype_name}'")
    return getattr(torch_module, attr)


def _move_batch_to_device(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    for key, value in list(batch.items()):
        if hasattr(value, "to"):
            batch[key] = value.to(device)
    return batch


def _transformers_vlm_messages(image_path: Path) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                # Qwen processor versions differ here: some docs show
                # file:// URIs, but several HF processor builds accept only a
                # plain local path, HTTP(S) URL, or base64 string.
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": VLM_PROMPT},
            ],
        }
    ]


def _apply_transformers_vlm_template(processor: Any, messages: Sequence[List[Dict[str, Any]]]) -> Any:
    """Apply Qwen chat template across Transformers processor API variants."""
    common_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    attempts = [
        {"processor_kwargs": {"padding": True}},
        {"padding": True},
        {"tokenizer_kwargs": {"padding": True}},
    ]
    last_exc: Optional[Exception] = None
    for extra_kwargs in attempts:
        try:
            return processor.apply_chat_template(
                messages,
                **common_kwargs,
                **extra_kwargs,
            )
        except TypeError as exc:
            last_exc = exc
        except ValueError as exc:
            if "processor_kwargs" not in str(exc) and "padding" not in str(exc):
                raise
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("failed to apply Transformers VLM chat template")


def _load_transformers_vlm_state(
    *,
    model_id: str,
    dtype: str,
    device_map: str,
    attn_implementation: str,
    trust_remote_code: bool,
) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch_dtype = _resolve_torch_dtype(torch, dtype)
    load_kwargs: Dict[str, Any] = {
        "device_map": device_map or "auto",
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    def load_model(kwargs: Dict[str, Any]) -> Any:
        try:
            return AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=torch_dtype,
                **kwargs,
            )
        except TypeError:
            return AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                **kwargs,
            )

    try:
        local_model = load_model(load_kwargs)
    except Exception as exc:
        if not attn_implementation:
            raise
        fallback_kwargs = dict(load_kwargs)
        fallback_kwargs.pop("attn_implementation", None)
        print(
            f"[vlm] warning: failed to load with attn_implementation={attn_implementation}: "
            f"{type(exc).__name__}: {exc}. Retrying with model default attention.",
            file=sys.stderr,
            flush=True,
        )
        local_model = load_model(fallback_kwargs)
        attn_implementation = ""

    local_model.eval()
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    first_param = next(local_model.parameters(), None)
    input_device = first_param.device if first_param is not None else torch.device("cuda:0")
    return {
        "model": local_model,
        "processor": processor,
        "torch": torch,
        "input_device": input_device,
        "attn_implementation": attn_implementation,
    }


def _get_transformers_vlm_state(
    *,
    model: str,
    dtype: str,
    device_map: str,
    attn_implementation: str,
    trust_remote_code: bool,
) -> Dict[str, Any]:
    model_id = model or "Qwen/Qwen3-VL-30B-A3B-Instruct"
    state_key = "::".join(
        [
            "transformers",
            model_id,
            dtype or "auto",
            device_map or "auto",
            attn_implementation or "",
            str(bool(trust_remote_code)),
        ]
    )
    state = _TRANSFORMERS_VLM_STATE.get(state_key)
    if state is None:
        state = _load_transformers_vlm_state(
            model_id=model_id,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        state["model_id"] = model_id
        state["dtype"] = dtype or "auto"
        state["device_map"] = device_map or "auto"
        _TRANSFORMERS_VLM_STATE[state_key] = state
    return state


def judge_transformers_vlm_batch(
    image_paths: Sequence[Path],
    *,
    model: str,
    dtype: str,
    device_map: str,
    attn_implementation: str,
    trust_remote_code: bool,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    """Judge independent images as a batch without changing the judge prompt."""
    if not image_paths:
        return []
    state = _get_transformers_vlm_state(
        model=model,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )
    processor = state["processor"]
    local_model = state["model"]
    messages = [_transformers_vlm_messages(path) for path in image_paths]
    inputs = _apply_transformers_vlm_template(processor, messages)
    inputs = _move_batch_to_device(inputs, state["input_device"])
    torch = state["torch"]
    with torch.inference_mode():
        generated_ids = local_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    prompt_len = int(inputs["input_ids"].shape[-1])
    generated_ids_trimmed = generated_ids[:, prompt_len:]
    texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    payloads: List[Dict[str, Any]] = []
    for text in texts:
        payload = _parse_jsonish(text)
        payload["_local_vlm_backend"] = "transformers_vlm"
        payload["_local_vlm_model"] = state["model_id"]
        payload["_local_vlm_dtype"] = state["dtype"]
        payload["_local_vlm_device_map"] = state["device_map"]
        if state.get("attn_implementation"):
            payload["_local_vlm_attn_implementation"] = state["attn_implementation"]
        payload["_local_vlm_batch_size"] = len(image_paths)
        payloads.append(payload)
    return payloads


def judge_transformers_vlm(
    image_path: Path,
    *,
    model: str,
    dtype: str,
    device_map: str,
    attn_implementation: str,
    trust_remote_code: bool,
    max_new_tokens: int = 384,
) -> Dict[str, Any]:
    """Judge one image with a Hugging Face Transformers VLM loaded once."""
    return judge_transformers_vlm_batch(
        [image_path],
        model=model,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        max_new_tokens=max_new_tokens,
    )[0]


def _vlm_keep_score(vlm: Dict[str, Any]) -> Optional[float]:
    keep = _as_float(vlm.get("final_keep_score"))
    if keep is None:
        return None
    ambiguity = _as_float(vlm.get("ambiguity_risk"))
    unsafe = _as_float(vlm.get("unsafe_or_private_risk"))
    # Missing risk fields mean the judge did not follow the schema, so treat
    # them conservatively instead of rewarding the candidate as low risk.
    if ambiguity is None:
        ambiguity = 4.0
    if unsafe is None:
        unsafe = 3.0
    risk_penalty = max(0.0, (ambiguity - 3.5) * 0.08) + max(0.0, (unsafe - 2.0) * 0.20)
    return max(0.0, min(1.0, keep / 5.0 - risk_penalty))


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


VLM_NUMERIC_KEYS = {
    "clarity",
    "object_density",
    "relation_density",
    "ocr_richness",
    "counting_affordance",
    "spatial_affordance",
    "attribute_affordance",
    "ambiguity_risk",
    "unsafe_or_private_risk",
    "final_keep_score",
}


def _normalize_vlm_score(vlm: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(vlm)
    missing: List[str] = []
    for key in sorted(VLM_NUMERIC_KEYS):
        value = _as_float(payload.get(key))
        if value is None:
            missing.append(key)
            continue
        payload[key] = int(max(1, min(5, round(value))))

    question_types = payload.get("possible_question_types")
    if isinstance(question_types, str):
        payload["possible_question_types"] = [
            part.strip() for part in re.split(r"[,|;]", question_types) if part.strip()
        ][:4]
    elif isinstance(question_types, list):
        payload["possible_question_types"] = [str(part).strip() for part in question_types if str(part).strip()][:4]
    else:
        payload["possible_question_types"] = []
        missing.append("possible_question_types")

    reason = payload.get("short_reason")
    if reason is None:
        payload["short_reason"] = ""
        missing.append("short_reason")
    else:
        payload["short_reason"] = str(reason).strip()

    if missing:
        payload["_schema_missing_keys"] = sorted(set(missing))
    return payload


def _quantile_pick(items: Sequence[ScoredCandidate], count: int) -> List[ScoredCandidate]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[0]]
    positions = {
        int(round(i * (len(items) - 1) / float(count - 1)))
        for i in range(count)
    }
    return [items[idx] for idx in sorted(positions)]


def _select_vlm_candidates(
    scored: Sequence[ScoredCandidate],
    *,
    max_images: int,
    candidate_top_k: int,
    selection_strategy: str,
) -> List[ScoredCandidate]:
    ranked = sorted(
        [s for s in scored if s.accepted],
        key=lambda s: s.heuristic_score,
        reverse=True,
    )
    pool = ranked[: max(candidate_top_k, max_images)]
    if max_images <= 0 or not pool:
        return []
    if selection_strategy == "top":
        return pool[:max_images]

    by_domain: Dict[str, List[ScoredCandidate]] = {}
    for item in pool:
        by_domain.setdefault(item.candidate.domain, []).append(item)
    domains = sorted(by_domain, key=lambda d: (-len(by_domain[d]), d))
    per_domain = max(1, max_images // max(1, len(domains)))
    remainder = max_images % max(1, len(domains))
    domain_picks: List[List[ScoredCandidate]] = []
    selected: List[ScoredCandidate] = []
    seen: Set[str] = set()

    for idx, domain in enumerate(domains):
        quota = per_domain + (1 if idx < remainder else 0)
        domain_picks.append(_quantile_pick(by_domain[domain], quota))

    for round_idx in range(max((len(picks) for picks in domain_picks), default=0)):
        for picks in domain_picks:
            if round_idx >= len(picks):
                continue
            item = picks[round_idx]
            cid = item.candidate.candidate_id
            if cid in seen:
                continue
            selected.append(item)
            seen.add(cid)
            if len(selected) >= max_images:
                return selected

    for item in pool:
        cid = item.candidate.candidate_id
        if cid in seen:
            continue
        selected.append(item)
        seen.add(cid)
        if len(selected) >= max_images:
            break
    return selected


def _store_vlm_result(
    item: ScoredCandidate,
    *,
    backend: str,
    model: str,
    vlm: Dict[str, Any],
    cache_path: Path,
) -> None:
    vlm = _normalize_vlm_score(vlm)
    item.vlm_score = vlm
    vlm01 = _vlm_keep_score(vlm)
    if vlm01 is not None:
        item.final_score = 0.60 * item.heuristic_score + 0.40 * vlm01
    row = {
        "candidate_id": item.candidate.candidate_id,
        "source": item.candidate.source,
        "domain": item.candidate.domain,
        "heuristic_score": item.heuristic_score,
        "final_score": item.final_score,
        "vlm_backend": backend,
        "vlm_model": model,
        "vlm_score": vlm,
    }
    _jsonl_append(cache_path, row)


def _run_transformers_vlm_judge_batched(
    ranked: Sequence[ScoredCandidate],
    *,
    cache: Dict[str, Dict[str, Any]],
    model: str,
    dtype: str,
    device_map: str,
    attn_implementation: str,
    trust_remote_code: bool,
    max_new_tokens: int,
    max_images: int,
    batch_size: int,
    cache_path: Path,
    cache_dir: Path,
    timeout: float,
    retries: int,
    sleep_sec: float,
) -> None:
    judged = 0
    pending_items: List[ScoredCandidate] = []
    pending_paths: List[Path] = []

    def flush_pending() -> None:
        nonlocal judged, pending_items, pending_paths
        if not pending_items:
            return
        try:
            vlm_payloads = judge_transformers_vlm_batch(
                pending_paths,
                model=model,
                dtype=dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
                trust_remote_code=trust_remote_code,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            print(
                f"[vlm] batch of {len(pending_items)} failed: {type(exc).__name__}: {exc}; "
                "falling back to single-image generation for this batch",
                file=sys.stderr,
                flush=True,
            )
            vlm_payloads = []
            for path in pending_paths:
                try:
                    vlm_payloads.append(
                        judge_transformers_vlm(
                            path,
                            model=model,
                            dtype=dtype,
                            device_map=device_map,
                            attn_implementation=attn_implementation,
                            trust_remote_code=trust_remote_code,
                            max_new_tokens=max_new_tokens,
                        )
                    )
                except Exception as single_exc:
                    vlm_payloads.append(
                        {
                            "_failed": True,
                            "_failure_type": type(single_exc).__name__,
                            "_failure_message": str(single_exc),
                        }
                    )

        for item, vlm in zip(pending_items, vlm_payloads):
            if vlm.get("_failed"):
                print(
                    f"[vlm] failed {item.candidate.candidate_id}: "
                    f"{vlm.get('_failure_type')}: {vlm.get('_failure_message')}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            _store_vlm_result(
                item,
                backend="transformers_vlm",
                model=model,
                vlm=vlm,
                cache_path=cache_path,
            )
            judged += 1
            if judged % 10 == 0 or judged == max_images:
                print(f"[vlm] judged={judged}/{max_images}", flush=True)
            time.sleep(max(0.0, sleep_sec))
        pending_items = []
        pending_paths = []

    for item in ranked:
        if judged >= max_images:
            break
        cid = item.candidate.candidate_id
        if cid in cache:
            item.vlm_score = _normalize_vlm_score(cache[cid].get("vlm_score") or {})
            vlm01 = _vlm_keep_score(item.vlm_score or {})
            if vlm01 is not None:
                item.final_score = 0.60 * item.heuristic_score + 0.40 * vlm01
            judged += 1
            continue
        img, local_path, reason = _load_candidate_image(
            item.candidate,
            cache_dir=cache_dir,
            timeout=timeout,
            retries=retries,
        )
        if img is None or local_path is None:
            print(f"[vlm] skip {cid}: {reason}", file=sys.stderr)
            continue
        pending_items.append(item)
        pending_paths.append(local_path)
        if len(pending_items) >= batch_size:
            flush_pending()
    flush_pending()


def run_vlm_judge(
    scored: List[ScoredCandidate],
    *,
    backend: str,
    model: str,
    dtype: str,
    device_map: str,
    attn_implementation: str,
    trust_remote_code: bool,
    batch_size: int,
    max_new_tokens: int,
    base_url: str,
    api_key_env: str,
    max_images: int,
    candidate_top_k: int,
    selection_strategy: str,
    cache_path: Path,
    cache_dir: Path,
    timeout: float,
    api_timeout: float,
    retries: int,
    sleep_sec: float,
):
    if backend == "none" or max_images <= 0:
        return
    if backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("[vlm] OPENAI_API_KEY is not set; skipping OpenAI judge.", file=sys.stderr)
        return
    if backend == "gemini" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("[vlm] GEMINI_API_KEY/GOOGLE_API_KEY is not set; skipping Gemini judge.", file=sys.stderr)
        return

    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists():
        for row in _read_jsonl(cache_path):
            cid = str(row.get("candidate_id", ""))
            if cid:
                cache[cid] = row
        print(f"[vlm] loaded {len(cache)} cached VLM judgments from {cache_path}", flush=True)

    selection_limit = min(
        max(candidate_top_k, max_images),
        max_images + max(5, max_images // 5),
    )
    ranked = _select_vlm_candidates(
        scored,
        max_images=selection_limit,
        candidate_top_k=candidate_top_k,
        selection_strategy=selection_strategy,
    )
    print(
        f"[vlm] backend={backend} model={model or 'default'} "
        f"strategy={selection_strategy} candidates={len(ranked)} cache={cache_path} "
        f"batch_size={max(1, batch_size)} max_new_tokens={max_new_tokens}",
        flush=True,
    )
    if backend == "transformers_vlm" and max(1, batch_size) > 1:
        _run_transformers_vlm_judge_batched(
            ranked,
            cache=cache,
            model=model,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            max_new_tokens=max_new_tokens,
            max_images=max_images,
            batch_size=max(1, batch_size),
            cache_path=cache_path,
            cache_dir=cache_dir,
            timeout=timeout,
            retries=retries,
            sleep_sec=sleep_sec,
        )
        return

    judged = 0
    for item in ranked:
        if judged >= max_images:
            break
        cid = item.candidate.candidate_id
        if cid in cache:
            item.vlm_score = _normalize_vlm_score(cache[cid].get("vlm_score") or {})
            vlm01 = _vlm_keep_score(item.vlm_score or {})
            if vlm01 is not None:
                item.final_score = 0.60 * item.heuristic_score + 0.40 * vlm01
            judged += 1
            continue
        img, local_path, reason = _load_candidate_image(
            item.candidate,
            cache_dir=cache_dir,
            timeout=timeout,
            retries=retries,
        )
        if img is None:
            print(f"[vlm] skip {cid}: {reason}", file=sys.stderr)
            continue
        try:
            if backend == "openai":
                vlm = judge_openai(img, model=model)
            elif backend == "openai_compatible":
                vlm = judge_openai_compatible(
                    img,
                    model=model,
                    base_url=base_url,
                    api_key_env=api_key_env,
                    timeout=api_timeout,
                )
            elif backend == "gemini":
                vlm = judge_gemini(img, model=model)
            elif backend == "mlx_vlm":
                if local_path is None:
                    raise RuntimeError("mlx_vlm backend requires a local image path")
                vlm = judge_mlx_vlm(local_path, model=model)
            elif backend == "transformers_vlm":
                if local_path is None:
                    raise RuntimeError("transformers_vlm backend requires a local image path")
                vlm = judge_transformers_vlm(
                    local_path,
                    model=model,
                    dtype=dtype,
                    device_map=device_map,
                    attn_implementation=attn_implementation,
                    trust_remote_code=trust_remote_code,
                    max_new_tokens=max_new_tokens,
                )
            else:
                raise ValueError(f"unsupported VLM backend {backend}")
            _store_vlm_result(
                item,
                backend=backend,
                model=model,
                vlm=vlm,
                cache_path=cache_path,
            )
            judged += 1
            if judged % 10 == 0 or judged == max_images:
                print(f"[vlm] judged={judged}/{max_images}", flush=True)
            time.sleep(max(0.0, sleep_sec))
        except Exception as exc:
            print(f"[vlm] failed {cid}: {type(exc).__name__}: {exc}", file=sys.stderr)


def _near_duplicate(dhash: str, selected_hashes: Sequence[str], threshold: int) -> bool:
    if not dhash:
        return False
    return any(_hamming_hex(dhash, other) <= threshold for other in selected_hashes)


def _quota_counts(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    s = sum(max(0.0, float(v)) for v in weights.values())
    if s <= 0:
        return {}
    raw = {k: total * max(0.0, float(v)) / s for k, v in weights.items()}
    floor = {k: int(math.floor(v)) for k, v in raw.items()}
    remaining = total - sum(floor.values())
    for k in sorted(raw, key=lambda x: raw[x] - floor[x], reverse=True)[:remaining]:
        floor[k] += 1
    return floor


def select_candidates(
    scored: List[ScoredCandidate],
    *,
    target_count: int,
    min_score: float,
    domain_weights: Dict[str, float],
    duplicate_threshold: int,
) -> Tuple[List[ScoredCandidate], List[ScoredCandidate]]:
    accepted = [
        item for item in scored
        if item.accepted and item.final_score >= min_score
    ]
    accepted.sort(key=lambda s: s.final_score, reverse=True)
    quotas = _quota_counts(target_count, domain_weights)
    selected: List[ScoredCandidate] = []
    selected_hashes: List[str] = []
    selected_by_domain: Dict[str, int] = {}
    rejected_duplicates: List[ScoredCandidate] = []

    def try_add(item: ScoredCandidate, enforce_quota: bool) -> bool:
        if len(selected) >= target_count:
            return False
        domain = item.candidate.domain
        if enforce_quota and quotas and selected_by_domain.get(domain, 0) >= quotas.get(domain, 0):
            return False
        if _near_duplicate(item.dhash, selected_hashes, duplicate_threshold):
            item.accepted = False
            item.reject_reason = "near_duplicate"
            rejected_duplicates.append(item)
            return False
        selected.append(item)
        selected_hashes.append(item.dhash)
        selected_by_domain[domain] = selected_by_domain.get(domain, 0) + 1
        return True

    for item in accepted:
        try_add(item, enforce_quota=True)
    if len(selected) < target_count:
        for item in accepted:
            if item in selected:
                continue
            try_add(item, enforce_quota=False)
            if len(selected) >= target_count:
                break
    return selected, rejected_duplicates


def _normalized_image_bytes(img: Image.Image, *, long_side: int, quality: int) -> bytes:
    work = img.copy()
    if long_side > 0:
        work.thumbnail((long_side, long_side), Image.BILINEAR)
    buf = io.BytesIO()
    work.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _materialize_image(
    item: ScoredCandidate,
    out_path: Path,
    *,
    link_mode: str,
    normalize_long_side: int,
    jpeg_quality: int,
    cache_dir: Path,
    timeout: float,
    retries: int,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = Path(item.candidate.path) if item.candidate.path else None
    if src is not None and src.exists() and link_mode == "hardlink" and src.suffix.lower() in {".jpg", ".jpeg"}:
        try:
            if out_path.exists():
                return
            os.link(src, out_path)
            return
        except OSError:
            pass
    img, _, reason = _load_candidate_image(
        item.candidate,
        cache_dir=cache_dir,
        timeout=timeout,
        retries=retries,
    )
    if img is None:
        raise RuntimeError(f"failed to reload selected image {item.candidate.candidate_id}: {reason}")
    data = _normalized_image_bytes(img, long_side=normalize_long_side, quality=jpeg_quality)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
    os.replace(tmp, out_path)


def _folder_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _score_to_json(item: ScoredCandidate) -> Dict[str, Any]:
    data = asdict(item)
    return data


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))
    output_dir = args.output_dir.expanduser().resolve()
    logs_dir = output_dir / "scores"
    cache_dir = output_dir / "cache"
    openimages_cache = args.openimages_cache_dir.expanduser().resolve()
    vlm_cache_path = args.vlm_cache_path or (logs_dir / "vlm_scores.jsonl")
    domain_weights = _parse_domain_weights(args.domain_weights)

    if not args.local_source:
        default_local = Path("data/joint_pool_10k/images")
        if default_local.exists():
            args.local_source = [str(default_local)]

    print(f"[pool] output_dir={output_dir}")
    print(f"[pool] target_count={args.target_count} max_output_gb={args.max_output_gb}")
    print(f"[pool] local_sources={args.local_source}")

    candidates: List[Candidate] = list(iter_local_candidates(args.local_source))
    print(f"[pool] local candidates={len(candidates)}")

    if not args.no_openimages and args.openimages_target > 0:
        try:
            oi = list(
                iter_openimages_candidates(
                    cache_dir=openimages_cache,
                    target=int(args.openimages_target),
                    min_boxes=int(args.openimages_min_boxes),
                    timeout=float(args.download_timeout),
                    retries=int(args.download_retries),
                )
            )
            print(f"[pool] openimages candidates={len(oi)}")
            candidates.extend(oi)
        except Exception as exc:
            print(f"[openimages] WARNING: source disabled after failure: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not candidates:
        raise RuntimeError("No candidates found. Pass --local_source or enable Open Images.")

    random.shuffle(candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    scored_path = logs_dir / "heuristic_scores.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    if scored_path.exists():
        scored_path.unlink()
    if rejected_path.exists():
        rejected_path.unlink()

    scored: List[ScoredCandidate] = []
    workers = max(1, int(getattr(args, "download_workers", 1)))
    print(f"[score] workers={workers}")

    def handle_item(idx: int, item: ScoredCandidate):
        scored.append(item)
        _jsonl_append(scored_path, _score_to_json(item))
        if not item.accepted:
            _jsonl_append(rejected_path, _score_to_json(item))
        if idx % 500 == 0 or idx == len(candidates):
            kept = sum(1 for x in scored if x.accepted)
            print(f"[score] {idx}/{len(candidates)} scored; accepted_prefilter={kept}", flush=True)

    if workers <= 1:
        for idx, candidate in enumerate(candidates, start=1):
            item = score_candidate_worker(
                candidate,
                cache_dir=cache_dir,
                timeout=float(args.download_timeout),
                retries=int(args.download_retries),
                min_short_side=int(args.min_short_side),
                max_aspect=float(args.max_aspect),
                min_file_kb=float(args.min_file_kb),
                min_heuristic_score=float(args.min_heuristic_score),
            )
            handle_item(idx, item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    score_candidate_worker,
                    candidate,
                    cache_dir=cache_dir,
                    timeout=float(args.download_timeout),
                    retries=int(args.download_retries),
                    min_short_side=int(args.min_short_side),
                    max_aspect=float(args.max_aspect),
                    min_file_kb=float(args.min_file_kb),
                    min_heuristic_score=float(args.min_heuristic_score),
                )
                for candidate in candidates
            ]
            for idx, future in enumerate(as_completed(futures), start=1):
                handle_item(idx, future.result())

    run_vlm_judge(
        scored,
        backend=str(args.vlm_backend),
        model=str(args.vlm_model or ""),
        dtype=str(args.vlm_dtype or "auto"),
        device_map=str(args.vlm_device_map or "auto"),
        attn_implementation=str(args.vlm_attn_implementation or ""),
        trust_remote_code=bool(args.vlm_trust_remote_code),
        batch_size=max(1, int(args.vlm_batch_size)),
        max_new_tokens=max(32, int(args.vlm_max_new_tokens)),
        base_url=str(args.vlm_base_url or ""),
        api_key_env=str(args.vlm_api_key_env or "OPENAI_COMPATIBLE_API_KEY"),
        max_images=int(args.vlm_max_images),
        candidate_top_k=int(args.vlm_candidate_top_k),
        selection_strategy=str(args.vlm_selection_strategy),
        cache_path=vlm_cache_path,
        cache_dir=cache_dir,
        timeout=float(args.download_timeout),
        api_timeout=float(args.vlm_timeout),
        retries=int(args.download_retries),
        sleep_sec=float(args.vlm_sleep_sec),
    )

    selected, duplicate_rejects = select_candidates(
        scored,
        target_count=int(args.target_count),
        min_score=float(args.min_heuristic_score),
        domain_weights=domain_weights,
        duplicate_threshold=int(args.duplicate_hamming_threshold),
    )
    for item in duplicate_rejects:
        _jsonl_append(rejected_path, _score_to_json(item))

    images_root = output_dir / "images"
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    if not args.dry_run:
        if images_root.exists():
            shutil.rmtree(images_root)
        materialized_bytes = 0
        max_output_bytes = int(float(args.max_output_gb) * (1024 ** 3))
        for rank, item in enumerate(selected, start=1):
            if materialized_bytes > max_output_bytes:
                current_gb = materialized_bytes / (1024 ** 3)
                print(f"[pool] disk budget reached at {current_gb:.2f} GiB; stopping materialization.")
                selected = selected[: rank - 1]
                break
            domain = item.candidate.domain
            out_name = f"{rank:05d}_{item.candidate.candidate_id}.jpg"
            out_path = images_root / domain / out_name
            _materialize_image(
                item,
                out_path,
                link_mode=str(args.link_mode),
                normalize_long_side=int(args.normalize_long_side),
                jpeg_quality=int(args.jpeg_quality),
                cache_dir=cache_dir,
                timeout=float(args.download_timeout),
                retries=int(args.download_retries),
            )
            try:
                materialized_bytes += out_path.stat().st_size
            except OSError:
                pass
            row = _score_to_json(item)
            row["rank"] = rank
            row["image_path"] = str(out_path)
            row["relative_image_path"] = out_path.relative_to(output_dir).as_posix()
            _jsonl_append(manifest_path, row)
    else:
        for rank, item in enumerate(selected, start=1):
            row = _score_to_json(item)
            row["rank"] = rank
            _jsonl_append(manifest_path, row)

    counts_by_domain: Dict[str, int] = {}
    counts_by_source: Dict[str, int] = {}
    for item in selected:
        counts_by_domain[item.candidate.domain] = counts_by_domain.get(item.candidate.domain, 0) + 1
        counts_by_source[item.candidate.source] = counts_by_source.get(item.candidate.source, 0) + 1

    scores = [item.final_score for item in selected]
    vlm_scored = [item for item in scored if item.vlm_score]
    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_count": int(args.target_count),
        "selected_count": len(selected),
        "candidate_count": len(candidates),
        "accepted_prefilter_count": sum(1 for item in scored if item.accepted),
        "output_dir": str(output_dir),
        "images_root": str(images_root),
        "manifest_path": str(manifest_path),
        "heuristic_scores_path": str(scored_path),
        "rejected_path": str(rejected_path),
        "vlm_scores_path": str(vlm_cache_path),
        "dry_run": bool(args.dry_run),
        "link_mode": str(args.link_mode),
        "max_output_gb": float(args.max_output_gb),
        "actual_images_gb": _folder_size_bytes(images_root) / (1024 ** 3),
        "domain_counts": counts_by_domain,
        "source_counts": counts_by_source,
        "score_min": min(scores) if scores else None,
        "score_mean": float(sum(scores) / len(scores)) if scores else None,
        "score_max": max(scores) if scores else None,
        "domain_weights": domain_weights,
        "vlm": {
            "backend": str(args.vlm_backend),
            "model": str(args.vlm_model or ""),
            "base_url": str(args.vlm_base_url or ""),
            "max_images": int(args.vlm_max_images),
            "candidate_top_k": int(args.vlm_candidate_top_k),
            "selection_strategy": str(args.vlm_selection_strategy),
            "timeout": float(args.vlm_timeout),
            "batch_size": max(1, int(args.vlm_batch_size)),
            "max_new_tokens": max(32, int(args.vlm_max_new_tokens)),
            "attn_implementation": str(args.vlm_attn_implementation or ""),
            "scored_count": len(vlm_scored),
            "schema_missing_count": sum(
                1 for item in vlm_scored
                if item.vlm_score and item.vlm_score.get("_schema_missing_keys")
            ),
        },
        "filters": {
            "min_short_side": int(args.min_short_side),
            "max_aspect": float(args.max_aspect),
            "min_file_kb": float(args.min_file_kb),
            "min_heuristic_score": float(args.min_heuristic_score),
            "duplicate_hamming_threshold": int(args.duplicate_hamming_threshold),
        },
    }
    _json_dump(output_dir / "audit_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if len(selected) < int(args.target_count):
        print(
            f"[pool] WARNING: selected {len(selected)} images, below target {args.target_count}. "
            "Increase --openimages_target, lower --min_heuristic_score, or add local sources.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
