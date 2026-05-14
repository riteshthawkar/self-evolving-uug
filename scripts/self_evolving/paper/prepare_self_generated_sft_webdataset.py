#!/usr/bin/env python3
"""Package self-generated image/prompt pairs as BLIP3o SFT webdataset shards."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_metadata(generated_dir: Path) -> Iterable[dict[str, Any]]:
    for meta_path in sorted(generated_dir.rglob("*.json")):
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        payload["_meta_path"] = str(meta_path)
        yield payload


def _resolve_image_path(payload: dict[str, Any], meta_path: Path) -> Path | None:
    raw = str(payload.get("image_path") or payload.get("path") or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (meta_path.parent / p).resolve()
        return p
    stem = meta_path.with_suffix("")
    for suffix in IMAGE_EXTENSIONS:
        candidate = stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _jpeg_bytes(path: Path, image_size: int | None) -> bytes:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if image_size:
            img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95, optimize=True)
        return buf.getvalue()


def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    tf.addfile(info, io.BytesIO(data))


def build_shards(args: argparse.Namespace) -> dict[str, Any]:
    generated_dir = Path(args.generated_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for old in output_dir.glob(f"{args.shard_prefix}-*.tar"):
            old.unlink()

    records: list[tuple[Path, str, dict[str, Any]]] = []
    skipped = 0
    for payload in _iter_metadata(generated_dir):
        meta_path = Path(str(payload["_meta_path"]))
        reward = payload.get("reward", payload.get("total_reward"))
        if args.min_reward is not None and isinstance(reward, (int, float)) and float(reward) < args.min_reward:
            skipped += 1
            continue
        prompt = " ".join(str(payload.get("prompt") or payload.get("caption") or "").split())
        if not prompt:
            skipped += 1
            continue
        image_path = _resolve_image_path(payload, meta_path)
        if image_path is None or not image_path.exists():
            skipped += 1
            continue
        records.append((image_path, prompt, payload))
        if args.max_samples and len(records) >= args.max_samples:
            break

    if not records:
        raise SystemExit(f"No self-generated image/prompt pairs found in {generated_dir}")

    shard_size = max(1, int(args.shard_size))
    shards: list[str] = []
    written = 0
    for shard_idx, offset in enumerate(range(0, len(records), shard_size)):
        shard_records = records[offset: offset + shard_size]
        shard_path = output_dir / f"{args.shard_prefix}-{shard_idx:06d}.tar"
        with tarfile.open(shard_path, "w") as tf:
            for local_idx, (image_path, prompt, payload) in enumerate(shard_records):
                idx = offset + local_idx
                key = f"{idx:08d}"
                jpg = _jpeg_bytes(image_path, args.image_size)
                _add_bytes(tf, f"{key}.jpg", jpg)
                _add_bytes(tf, f"{key}.txt", (prompt + "\n").encode("utf-8"))
                meta = {
                    "source_image_path": str(image_path),
                    "source_meta_path": payload.get("_meta_path"),
                    "prompt": prompt,
                    "reward": payload.get("reward", payload.get("total_reward")),
                    "raw_reward": payload.get("raw_reward"),
                    "step_generated": payload.get("step_generated"),
                    "questions": payload.get("questions", []),
                    "reference_answers": payload.get("reference_answers", []),
                }
                _add_bytes(tf, f"{key}.json", json.dumps(meta, sort_keys=True).encode("utf-8"))
                written += 1
        shards.append(str(shard_path))

    manifest = {
        "generated_dir": str(generated_dir),
        "output_dir": str(output_dir),
        "num_records": written,
        "num_skipped": skipped,
        "shard_size": shard_size,
        "shards": shards,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated-dir", required=True, help="Directory containing self-generated image JSON sidecars.")
    ap.add_argument("--output-dir", required=True, help="Directory where webdataset .tar shards will be written.")
    ap.add_argument("--shard-prefix", default="self_generated_sft")
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--min-reward", type=float, default=None)
    ap.add_argument("--image-size", type=int, default=896, help="Optional max side before JPEG packaging; 0 disables resizing.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.image_size <= 0:
        args.image_size = None
    manifest = build_shards(args)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
