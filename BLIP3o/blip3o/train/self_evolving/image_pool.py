"""
ImagePool: Data loader for self-evolving training.
Adapted from EvoLMM/src/train.py:198-270
Scans folders for images without requiring labels (unsupervised).
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from PIL import Image


_KNOWN_SPLITS = {"train", "val", "test"}


@dataclass
class ImagePoolConfig:
    """Configuration for ImagePool data loader."""
    data_dir: str
    include_subfolders: Optional[List[str]] = None
    split: Optional[str] = None  # train|val|test|None(all)
    prefer_manifest: bool = False
    manifest_name: Optional[str] = None
    seed: int = 42
    max_images: Optional[int] = None  # Limit for debugging


class ImagePool:
    """
    Unsupervised image pool for self-evolving training.

    Scans directories for images without requiring any labels.
    This is the core data source for EvoLMM-style self-evolution.
    """

    DEFAULT_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

    def __init__(self, config: ImagePoolConfig):
        self.config = config
        self.paths: List[str] = []
        self.metas: List[Dict[str, str]] = []

        root = os.path.abspath(config.data_dir)
        if not os.path.isdir(root):
            raise RuntimeError(f"[ImagePool] data_dir not found: {root}")
        self._root = root

        split = (config.split or "").strip().lower()
        if split == "all":
            split = ""
        if split and split not in _KNOWN_SPLITS:
            raise ValueError(
                f"[ImagePool] Unknown split '{config.split}'. "
                f"Expected one of: {_KNOWN_SPLITS} or None."
            )

        loaded_from_manifest = False
        if config.prefer_manifest:
            loaded_from_manifest = self._load_from_manifest(split=split or None)

        if not loaded_from_manifest:
            self._scan_directory(split=split or None)

        if not self.paths:
            raise RuntimeError(f"[ImagePool] No images found under: {root}")

        if not loaded_from_manifest:
            paired = sorted(zip(self.paths, self.metas), key=lambda x: x[0])
            self.paths = [p for p, _ in paired]
            self.metas = [m for _, m in paired]

        if config.max_images and len(self.paths) > config.max_images:
            self.paths = self.paths[: config.max_images]
            self.metas = self.metas[: config.max_images]

        if loaded_from_manifest:
            print(
                f"[ImagePool] Loaded {len(self.paths)} images from manifest under: "
                f"{root} (split={split or 'all'})"
            )
        else:
            print(
                f"[ImagePool] Found {len(self.paths)} images under: "
                f"{root} (split={split or 'all'})"
            )

        self.indices = list(range(len(self.paths)))
        rnd = random.Random(config.seed)
        rnd.shuffle(self.indices)
        self._current_idx = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_img_file(self, filename: str) -> bool:
        fnl = filename.lower()
        return fnl.endswith(self.DEFAULT_EXTS) and not os.path.basename(fnl).startswith(".")

    def _pick_dirs(
        self, base_dir: str, include_names: Optional[List[str]]
    ) -> List[Tuple[str, str]]:
        chosen: List[Tuple[str, str]] = []
        if include_names:
            for name in include_names:
                sub = os.path.join(base_dir, name)
                if os.path.isdir(sub):
                    chosen.append((name, sub))
                else:
                    print(f"[ImagePool] WARNING: requested subfolder not found: {name}")
        else:
            for name in sorted(os.listdir(base_dir)):
                sub = os.path.join(base_dir, name)
                if os.path.isdir(sub) and not name.startswith("."):
                    chosen.append((name, sub))
        return chosen

    def _build_meta_from_path(
        self,
        path: str,
        *,
        split_hint: Optional[str] = None,
        dataset_hint: Optional[str] = None,
    ) -> Dict[str, str]:
        rel = os.path.relpath(path, self._root)
        parts = rel.split(os.sep)

        split = "train"
        dataset = "folder"
        subfolder = ""

        if parts and parts[0] in _KNOWN_SPLITS:
            split = parts[0]
            if len(parts) > 1:
                dataset = parts[1]
                subfolder = parts[1]
        elif parts and parts[0] == "images":
            if len(parts) > 1:
                dataset = parts[1]
                subfolder = parts[1]
        elif parts:
            dataset = parts[0]
            subfolder = parts[0]

        if split_hint in _KNOWN_SPLITS:
            split = split_hint
        if dataset_hint:
            dataset = dataset_hint
            subfolder = dataset_hint

        return {
            "path": path,
            "dataset": dataset,
            "split": split,
            "subfolder": subfolder,
            "filename": os.path.basename(path),
        }

    def _scan_directory(self, split: Optional[str]):
        root = self._root
        include_names = (
            list(self.config.include_subfolders) if self.config.include_subfolders else None
        )

        # Preferred modern layout: <root>/<split>/<dataset>/image.jpg
        if split and os.path.isdir(os.path.join(root, split)):
            split_root = os.path.join(root, split)
            chosen = self._pick_dirs(split_root, include_names)
            if not chosen:
                chosen = [("", split_root)]
            for dataset_name, dataset_path in chosen:
                for r, _dirs, files in os.walk(dataset_path):
                    for fn in files:
                        if self._is_img_file(fn):
                            full = os.path.join(r, fn)
                            meta = self._build_meta_from_path(
                                full,
                                split_hint=split,
                                dataset_hint=dataset_name or None,
                            )
                            self.paths.append(full)
                            self.metas.append(meta)
            return

        if split and not os.path.isdir(os.path.join(root, split)):
            print(
                f"[ImagePool] NOTE: split directory '{split}' not found under {root}. "
                "Falling back to legacy directory scan."
            )

        # Legacy layout: <root>/images/<dataset>/image.jpg
        scan_root = (
            os.path.join(root, "images")
            if os.path.isdir(os.path.join(root, "images"))
            else root
        )
        chosen = self._pick_dirs(scan_root, include_names)
        if not chosen:
            print(
                f"[ImagePool] NOTE: No subfolders selected/found under {scan_root}; "
                "falling back to scanning images directly under that directory."
            )
            chosen = [("", scan_root)]

        for dataset_name, dataset_path in chosen:
            for r, _dirs, files in os.walk(dataset_path):
                for fn in files:
                    if self._is_img_file(fn):
                        full = os.path.join(r, fn)
                        meta = self._build_meta_from_path(
                            full,
                            split_hint=split,
                            dataset_hint=dataset_name or None,
                        )
                        self.paths.append(full)
                        self.metas.append(meta)

    def _manifest_candidates(self, split: Optional[str]) -> List[str]:
        if self.config.manifest_name:
            return [self.config.manifest_name]
        if split in _KNOWN_SPLITS:
            return [f"{split}.jsonl"]
        return ["all.jsonl"]

    def _resolve_manifest_image_path(self, record: Dict) -> Optional[str]:
        root = self._root
        candidates: List[str] = []
        image_abspath = record.get("image_abspath")
        if image_abspath:
            candidates.append(str(image_abspath))

        for key in ("structured_relpath", "image_relpath", "path", "filepath"):
            rel = record.get(key)
            if not rel:
                continue
            rel = str(rel)
            candidates.append(os.path.join(root, rel))
            candidates.append(os.path.join(root, "images", rel))

        seen = set()
        for candidate in candidates:
            candidate_abs = os.path.abspath(candidate)
            if candidate_abs in seen:
                continue
            seen.add(candidate_abs)
            if os.path.isfile(candidate_abs):
                return candidate_abs
        return None

    def _infer_dataset_from_record(self, record: Dict, parts: List[str]) -> str:
        for key in ("dataset_name", "rebalance_source", "source", "dataset"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if parts:
            if parts[0] in _KNOWN_SPLITS and len(parts) > 1:
                return parts[1]
            if parts[0] == "images" and len(parts) > 1:
                return parts[1]
            return parts[0]
        return "folder"

    def _infer_split_from_record(
        self, record: Dict, parts: List[str], split_hint: Optional[str]
    ) -> str:
        for key in ("split", "data_split"):
            value = record.get(key)
            if isinstance(value, str):
                value_norm = value.strip().lower()
                if value_norm in _KNOWN_SPLITS:
                    return value_norm
        if parts and parts[0] in _KNOWN_SPLITS:
            return parts[0]
        if split_hint in _KNOWN_SPLITS:
            return split_hint
        return "train"

    def _load_from_manifest(self, split: Optional[str]) -> bool:
        manifests_dir = os.path.join(self._root, "manifests")
        if not os.path.isdir(manifests_dir):
            return False

        manifest_path = None
        for name in self._manifest_candidates(split):
            candidate = os.path.join(manifests_dir, name)
            if os.path.isfile(candidate):
                manifest_path = candidate
                break
        if manifest_path is None:
            return False

        include_names = set(self.config.include_subfolders or [])
        loaded = 0
        skipped_missing = 0
        skipped_filter = 0
        skipped_split = 0
        parse_errors = 0

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    parse_errors += 1
                    continue

                image_path = self._resolve_manifest_image_path(record)
                if image_path is None:
                    skipped_missing += 1
                    continue

                rel = os.path.relpath(image_path, self._root)
                parts = rel.split(os.sep)
                record_split = self._infer_split_from_record(
                    record, parts, split_hint=split
                )
                if split in _KNOWN_SPLITS and record_split != split:
                    skipped_split += 1
                    continue

                dataset = self._infer_dataset_from_record(record, parts)
                if include_names and dataset not in include_names:
                    skipped_filter += 1
                    continue

                meta = {
                    "path": image_path,
                    "dataset": dataset,
                    "split": record_split,
                    "subfolder": dataset,
                    "filename": os.path.basename(image_path),
                }
                if isinstance(record.get("source"), str):
                    meta["source"] = record["source"]
                if isinstance(record.get("dataset_id"), str):
                    meta["dataset_id"] = record["dataset_id"]
                if isinstance(record.get("dataset_split"), str):
                    meta["dataset_split"] = record["dataset_split"]

                self.paths.append(image_path)
                self.metas.append(meta)
                loaded += 1

        if loaded == 0:
            return False

        print(
            f"[ImagePool] Manifest load: {os.path.relpath(manifest_path, self._root)} | "
            f"loaded={loaded} skipped_missing={skipped_missing} "
            f"skipped_filter={skipped_filter} skipped_split={skipped_split} "
            f"parse_errors={parse_errors}"
        )
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self) -> Tuple[Image.Image, dict]:
        if self._current_idx >= len(self.indices):
            raise StopIteration

        idx = self.indices[self._current_idx]
        self._current_idx += 1
        return self.get_image(idx)

    def get_image(self, idx: int) -> Tuple[Image.Image, dict]:
        """Get image and metadata by index."""
        path = self.paths[idx]
        meta = (
            dict(self.metas[idx])
            if idx < len(self.metas)
            else self._build_meta_from_path(path)
        )
        meta.setdefault("path", path)
        meta.setdefault("filename", os.path.basename(path))

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[ImagePool] Error loading {path}: {e}")
            raise RuntimeError(f"Failed to load image: {path}")

        return img, meta

    def get_batch(
        self, batch_size: int, start_idx: Optional[int] = None
    ) -> List[Tuple[Image.Image, dict]]:
        """Get a batch of images."""
        if start_idx is None:
            start_idx = self._current_idx

        batch = []
        for i in range(batch_size):
            idx = (start_idx + i) % len(self.indices)
            shuffled_idx = self.indices[idx]
            batch.append(self.get_image(shuffled_idx))

        self._current_idx = (start_idx + batch_size) % len(self.indices)
        return batch

    def sample_random(self, n: int = 1) -> List[Tuple[Image.Image, dict]]:
        """Sample n random images."""
        indices = random.choices(self.indices, k=n)
        return [self.get_image(idx) for idx in indices]


class BatchDataLoader:
    """
    DataLoader-like interface for ImagePool.
    Provides batched iteration with optional preprocessing.
    """

    def __init__(
        self,
        pool: ImagePool,
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.pool = pool
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.pool)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.pool)))
        if self.shuffle:
            random.shuffle(indices)

        batch_images = []
        batch_metas = []

        for idx in indices:
            img, meta = self.pool.get_image(idx)
            batch_images.append(img)
            batch_metas.append(meta)

            if len(batch_images) == self.batch_size:
                yield batch_images, batch_metas
                batch_images = []
                batch_metas = []

        # Handle last incomplete batch
        if batch_images and not self.drop_last:
            yield batch_images, batch_metas
