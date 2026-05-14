"""Small image helpers used by BLIP3o model code."""

from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image


def numpy_to_pil(images: np.ndarray) -> List[Image.Image]:
    """Convert a batch of NHWC images in [0, 1] to PIL images.

    This mirrors the lightweight behavior BLIP3o previously imported from
    ``diffusers.pipelines.pipeline_utils`` without importing the full diffusers
    pipeline stack at model import time.
    """
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None, ...]
    if images.ndim != 4:
        raise ValueError(f"Expected image array with 3 or 4 dimensions, got {images.ndim}.")

    if images.dtype != np.uint8:
        images = (images * 255).round().clip(0, 255).astype("uint8")

    if images.shape[-1] == 1:
        return [Image.fromarray(image.squeeze(-1), mode="L") for image in images]
    return [Image.fromarray(image) for image in images]
