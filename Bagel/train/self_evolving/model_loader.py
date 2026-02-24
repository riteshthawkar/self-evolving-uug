# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import torch
from safetensors.torch import load_file
from torchvision import transforms as tv_transforms
from torchvision.transforms import functional as tv_functional, InterpolationMode

from data.data_utils import add_special_tokens
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer

from .config import ModelLoadConfig

try:
    from data.transforms import ImageTransform as _BagelImageTransform
except ModuleNotFoundError as exc:
    if exc.name != "cv2":
        raise
    _BagelImageTransform = None


class _FallbackImageTransform:
    """Fallback transform when optional cv2 dependency is unavailable."""

    def __init__(
        self,
        max_image_size: int,
        min_image_size: int,
        image_stride: int,
        max_pixels: int = 14 * 14 * 9 * 1024,
    ) -> None:
        self.max_image_size = int(max_image_size)
        self.min_image_size = int(min_image_size)
        self.stride = int(image_stride)
        self.max_pixels = int(max_pixels)
        self.to_tensor = tv_transforms.ToTensor()
        self.normalize = tv_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

    def _make_divisible(self, value: float) -> int:
        v = int(round(value / float(self.stride)) * self.stride)
        return max(self.stride, v)

    def resize_transform(self, image, img_num: int = 1):
        width, height = image.size
        scale = min(self.max_image_size / max(width, height), 1.0)
        scale = max(scale, self.min_image_size / min(width, height))
        new_w = self._make_divisible(width * scale)
        new_h = self._make_divisible(height * scale)

        # Enforce per-image pixel budget similar to BAGEL transform policy.
        max_pixels = max(1, self.max_pixels // max(1, int(img_num)))
        if new_w * new_h > max_pixels:
            pixel_scale = (float(max_pixels) / float(new_w * new_h)) ** 0.5
            new_w = self._make_divisible(new_w * pixel_scale)
            new_h = self._make_divisible(new_h * pixel_scale)

        return tv_functional.resize(
            image,
            [new_h, new_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __call__(self, image, img_num: int = 1):
        image = self.resize_transform(image, img_num=img_num)
        tensor = self.to_tensor(image)
        return self.normalize(tensor)


def _build_image_transform(*, max_image_size: int, min_image_size: int, image_stride: int):
    if _BagelImageTransform is not None:
        return _BagelImageTransform(
            max_image_size=max_image_size,
            min_image_size=min_image_size,
            image_stride=image_stride,
        )
    print("[model_loader] cv2 not available; using fallback image transform implementation.")
    return _FallbackImageTransform(
        max_image_size=max_image_size,
        min_image_size=min_image_size,
        image_stride=image_stride,
    )


@dataclass
class BagelRuntime:
    model: Bagel
    vae_model: torch.nn.Module
    tokenizer: Qwen2Tokenizer
    new_token_ids: Dict[str, int]
    vae_transform: object
    vit_transform: object
    inferencer: InterleaveInferencer
    device: torch.device


def _resolve_weights_path(model_path: str) -> str:
    candidates = [
        os.path.join(model_path, "ema.safetensors"),
        os.path.join(model_path, "model.safetensors"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Could not find model weights. Expected one of: "
        f"{', '.join(candidates)}"
    )


def load_bagel_runtime(cfg: ModelLoadConfig) -> BagelRuntime:
    device = torch.device(cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    llm_config = Qwen2Config.from_json_file(os.path.join(cfg.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(cfg.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(cfg.model_path, "ae.safetensors"))

    model_cfg = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=cfg.vit_max_num_patch_per_side,
        connector_act=cfg.connector_act,
        latent_patch_size=cfg.latent_patch_size,
        max_latent_size=cfg.max_latent_size,
    )

    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, model_cfg)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(cfg.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    weights_path = _resolve_weights_path(cfg.model_path)
    state_dict = load_file(weights_path, device="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    del state_dict
    missing = len(getattr(msg, "missing_keys", []) or [])
    unexpected = len(getattr(msg, "unexpected_keys", []) or [])
    print(f"[model_loader] loaded weights from {weights_path} (missing={missing}, unexpected={unexpected})")

    model = model.to(device).eval()
    vae_model = vae_model.to(device).eval()

    vae_transform = _build_image_transform(
        max_image_size=cfg.vae_max_image_size,
        min_image_size=cfg.vae_min_image_size,
        image_stride=cfg.vae_stride,
    )
    vit_transform = _build_image_transform(
        max_image_size=cfg.vit_max_image_size,
        min_image_size=cfg.vit_min_image_size,
        image_stride=cfg.vit_stride,
    )

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    return BagelRuntime(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        inferencer=inferencer,
        device=device,
    )
