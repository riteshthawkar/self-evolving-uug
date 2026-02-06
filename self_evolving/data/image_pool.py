"""
ImagePool: Data loader for self-evolving training.
Adapted from EvoLMM/src/train.py:198-270
Scans folders for images without requiring labels (unsupervised).
"""

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from PIL import Image
import torch


@dataclass
class ImagePoolConfig:
    """Configuration for ImagePool data loader."""
    data_dir: str
    include_subfolders: Optional[List[str]] = None
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

        root = os.path.abspath(config.data_dir)
        if not os.path.isdir(root):
            raise RuntimeError(f"[ImagePool] data_dir not found: {root}")

        # Determine which FIRST-LEVEL subfolders to scan
        if config.include_subfolders:
            # Exact-name filter
            chosen = []
            for name in config.include_subfolders:
                sub = os.path.join(root, name)
                if os.path.isdir(sub):
                    chosen.append((name, sub))
                else:
                    print(f"[ImagePool] WARNING: requested subfolder not found: {name}")
        else:
            # All first-level subfolders (skip hidden)
            chosen = []
            for name in sorted(os.listdir(root)):
                sub = os.path.join(root, name)
                if os.path.isdir(sub) and not name.startswith("."):
                    chosen.append((name, sub))

        # Fallback: if no subfolders matched, still look directly under root
        if not chosen:
            print(f"[ImagePool] NOTE: No subfolders selected/found under {root}; "
                  f"falling back to scanning images directly under root.")
            chosen = [("", root)]

        # Walk each chosen subfolder recursively and collect images
        def _is_img(fn: str) -> bool:
            fnl = fn.lower()
            return fnl.endswith(self.DEFAULT_EXTS) and not os.path.basename(fnl).startswith(".")

        for sub_name, sub_path in chosen:
            for r, _dirs, files in os.walk(sub_path):
                for fn in files:
                    if _is_img(fn):
                        full = os.path.join(r, fn)
                        self.paths.append(full)

        if not self.paths:
            raise RuntimeError(f"[ImagePool] No images found under: {root} (subfolders={[n for n, _ in chosen]})")

        self.paths.sort()
        
        # Apply max_images limit if specified
        if config.max_images and len(self.paths) > config.max_images:
            self.paths = self.paths[:config.max_images]
        
        print(f"[ImagePool] Found {len(self.paths)} images under: {root} "
              f"(subfolders={[n for n, _ in chosen]})")

        # Deterministic permutation using cfg.seed
        self.indices = list(range(len(self.paths)))
        rnd = random.Random(config.seed)
        rnd.shuffle(self.indices)

        # Keep root to compute subfolder/relative path in meta
        self._root = root
        self._current_idx = 0

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
        
        # Build metadata
        rel = os.path.relpath(path, self._root)
        parts = rel.split(os.sep)
        subfolder = parts[0] if len(parts) > 1 else ""
        
        meta = {
            "path": path,
            "dataset": "folder",
            "split": "train",
            "subfolder": subfolder,
            "filename": os.path.basename(path),
        }
        
        # Load image
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[ImagePool] Error loading {path}: {e}")
            # Return a placeholder or skip
            raise RuntimeError(f"Failed to load image: {path}")
        
        return img, meta

    def get_batch(self, batch_size: int, start_idx: Optional[int] = None) -> List[Tuple[Image.Image, dict]]:
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
