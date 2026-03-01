# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file
from torchvision import transforms as tv_transforms
from torchvision.transforms import functional as tv_functional, InterpolationMode

# Ensure BAGEL imports resolve regardless of launch cwd/module style.
_BAGEL_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _BAGEL_ROOT.parent
for _path in (str(_BAGEL_ROOT), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

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

from .adapter_manager import setup_lora_role_adapters
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
    vae_device: torch.device
    lora_enabled: bool = False
    role_to_adapter: Dict[str, str] = field(default_factory=dict)
    lora_adapters: Dict[str, bool] = field(default_factory=dict)

    def adapter_for_role(self, role: str) -> str:
        key = str(role or "").strip().lower()
        if key in self.role_to_adapter:
            return str(self.role_to_adapter[key])
        if self.role_to_adapter:
            return str(next(iter(self.role_to_adapter.values())))
        return ""


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


def _remap_adapter_state_dict(state_dict: Dict[str, torch.Tensor], src_adapter: str, dst_adapter: str) -> Dict[str, torch.Tensor]:
    src = str(src_adapter or "").strip()
    dst = str(dst_adapter or "").strip()
    if not src or not dst or src == dst:
        return dict(state_dict)

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key_new = str(key)
        key_new = key_new.replace(f".{src}.", f".{dst}.")
        key_new = key_new.replace(f"lora_{src}", f"lora_{dst}")
        remapped[key_new] = value
    return remapped


def _is_tensor_state_dict(obj: Any) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    for value in obj.values():
        if not torch.is_tensor(value):
            return False
    return True


def _load_role_lora_folder(
    language_model: torch.nn.Module,
    role_to_adapter: Dict[str, str],
    root: Path,
) -> Dict[str, int]:
    manifest_path = root / "adapter_roles.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    manifest_role_to_adapter = manifest.get("role_to_adapter", {})
    files_meta = manifest.get("files", {})

    roles_loaded = 0
    tensors_loaded = 0
    for role, dst_adapter in role_to_adapter.items():
        role_name = str(role)
        file_name = f"role_{role_name}.pt"
        if isinstance(files_meta, dict) and isinstance(files_meta.get(role_name), dict):
            file_name = str(files_meta[role_name].get("file") or file_name)
        role_file = root / file_name
        if not role_file.is_file():
            continue

        payload = torch.load(role_file, map_location="cpu")
        state_dict: Optional[Dict[str, torch.Tensor]] = None
        src_adapter = str(dst_adapter)
        if isinstance(payload, dict) and _is_tensor_state_dict(payload.get("state_dict", {})):
            state_dict = dict(payload["state_dict"])
            src_adapter = str(payload.get("adapter_name") or src_adapter)
        elif _is_tensor_state_dict(payload):
            state_dict = dict(payload)
            src_adapter = str(manifest_role_to_adapter.get(role_name) or src_adapter)
        if state_dict is None or not state_dict:
            continue

        if src_adapter != str(dst_adapter):
            state_dict = _remap_adapter_state_dict(
                state_dict=state_dict,
                src_adapter=src_adapter,
                dst_adapter=str(dst_adapter),
            )
        msg = language_model.load_state_dict(state_dict, strict=False)
        missing = len(getattr(msg, "missing_keys", []) or [])
        unexpected = len(getattr(msg, "unexpected_keys", []) or [])
        print(
            f"[model_loader] loaded role adapter role={role_name} file={role_file.name} "
            f"(src_adapter={src_adapter}, dst_adapter={dst_adapter}, tensors={len(state_dict)}, "
            f"missing={missing}, unexpected={unexpected})"
        )
        roles_loaded += 1
        tensors_loaded += int(len(state_dict))

    return {"roles_loaded": roles_loaded, "tensors_loaded": tensors_loaded}


def load_role_lora_checkpoint(
    language_model: torch.nn.Module,
    *,
    checkpoint_path: str,
    role_to_adapter: Dict[str, str],
) -> Dict[str, Any]:
    """Load LoRA checkpoints for proposer/solver/generator adapters.

    Supported inputs:
    - `step_XXXXXX.pt` checkpoint payload containing `model_state`
    - checkpoint root directory containing `step_*.pt`
    - role-export directory `step_XXXXXX_lora` with `role_<role>.pt` files
    """
    root = Path(str(checkpoint_path or "")).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"LoRA checkpoint path not found: {root}")

    if root.is_file():
        payload = torch.load(root, map_location="cpu")
        model_state = payload.get("model_state") if isinstance(payload, dict) else None
        if _is_tensor_state_dict(model_state):
            msg = language_model.load_state_dict(model_state, strict=False)
            missing = len(getattr(msg, "missing_keys", []) or [])
            unexpected = len(getattr(msg, "unexpected_keys", []) or [])
            return {
                "source": str(root),
                "mode": "step_pt_model_state",
                "roles_loaded": int(len(role_to_adapter)),
                "tensors_loaded": int(len(model_state)),
                "missing": int(missing),
                "unexpected": int(unexpected),
            }
        if _is_tensor_state_dict(payload):
            msg = language_model.load_state_dict(payload, strict=False)
            missing = len(getattr(msg, "missing_keys", []) or [])
            unexpected = len(getattr(msg, "unexpected_keys", []) or [])
            return {
                "source": str(root),
                "mode": "raw_state_dict",
                "roles_loaded": int(len(role_to_adapter)),
                "tensors_loaded": int(len(payload)),
                "missing": int(missing),
                "unexpected": int(unexpected),
            }
        raise RuntimeError(f"Unsupported LoRA checkpoint file format: {root}")

    role_stats = _load_role_lora_folder(language_model, role_to_adapter, root)
    if int(role_stats.get("roles_loaded", 0)) > 0:
        return {
            "source": str(root),
            "mode": "role_adapter_folder",
            **role_stats,
        }

    # If path is a checkpoint directory, load latest step checkpoint.
    step_ckpts = sorted(root.glob("step_*.pt"))
    if step_ckpts:
        return load_role_lora_checkpoint(
            language_model,
            checkpoint_path=str(step_ckpts[-1]),
            role_to_adapter=role_to_adapter,
        )

    raise RuntimeError(
        f"No role adapter files or step checkpoints found under: {root}"
    )


def load_bagel_runtime(cfg: ModelLoadConfig) -> BagelRuntime:
    device = torch.device(cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    vae_device = torch.device(cfg.vae_device if str(cfg.vae_device or "").strip() else str(device))

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

    lora_enabled = bool(cfg.enable_lora)
    role_to_adapter: Dict[str, str] = {}
    lora_adapters: Dict[str, bool] = {}
    if lora_enabled:
        model.language_model, role_to_adapter, available = setup_lora_role_adapters(
            model.language_model,
            lora_rank=int(cfg.lora_rank),
            lora_alpha=int(cfg.lora_alpha),
            lora_dropout=float(cfg.lora_dropout),
            target_modules=cfg.lora_target_modules(),
            role_adapter_names=cfg.lora_role_adapters(),
            default_adapter=str(cfg.lora_default_adapter),
        )
        lora_adapters = {str(name): True for name in available}
        print(
            "[model_loader] enabled LoRA role adapters "
            f"(active_default={cfg.lora_default_adapter}, adapters={sorted(list(lora_adapters.keys()))})"
        )
        lora_ckpt_path = str(getattr(cfg, "lora_checkpoint_path", "") or "").strip()
        if lora_ckpt_path:
            stats = load_role_lora_checkpoint(
                model.language_model,
                checkpoint_path=lora_ckpt_path,
                role_to_adapter=role_to_adapter,
            )
            print(
                "[model_loader] loaded LoRA checkpoint "
                f"(source={stats.get('source')}, mode={stats.get('mode')}, "
                f"roles_loaded={stats.get('roles_loaded')}, tensors_loaded={stats.get('tensors_loaded')})"
            )

    model = model.to(device).eval()
    vae_model = vae_model.to(vae_device).eval()
    print(f"[model_loader] runtime devices: model={device}, vae={vae_device}")

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
        vae_device=vae_device,
        lora_enabled=lora_enabled,
        role_to_adapter=role_to_adapter,
        lora_adapters=lora_adapters,
    )
