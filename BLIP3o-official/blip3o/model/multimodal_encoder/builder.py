import os
from .clip_encoder import CLIPVisionTower
from .imagebind import ImageBindWrapper
from .open_clip_encoder import OpenCLIPVisionTower
from .siglip_encoder import SigLipVisionTower
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2

from .eva_clip.eva_clip_encoder import EvaClipVisionTower
from .dev_eva_clip.eva_vit import EvaViTWrapper

from blip3o.model.nextdit_crossattn import NextDiTCrossAttnConfig, NextDiTCrossAttn
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


def _ceil_to_multiple(value, multiple):
    value_i = int(value)
    multiple_i = int(multiple)
    return multiple_i * ((value_i + multiple_i - 1) // multiple_i)


def _infer_lumina_ffn_multiplier(hidden_size=1792, multiple_of=256):
    """
    BLIP3o checkpoints were trained with a Lumina FFN hidden width of 4864
    for the default (dim=1792, inner_dim=4*dim) setting.

    Some diffusers builds changed LuminaFeedForward internals and construct
    width 7168 by default, which then causes checkpoint load mismatches:
      checkpoint: [4864, 1792]
      current   : [7168, 1792]

    Detect the runtime behavior and return a corrective multiplier only when
    needed. If the runtime already builds the expected width, return None.
    """
    try:
        from diffusers.models.attention import LuminaFeedForward
    except Exception:
        return None

    try:
        ff = LuminaFeedForward(
            dim=int(hidden_size),
            inner_dim=4 * int(hidden_size),
            multiple_of=int(multiple_of),
            ffn_dim_multiplier=None,
        )
    except Exception:
        return None

    observed = None
    try:
        for module in ff.modules():
            in_features = getattr(module, "in_features", None)
            out_features = getattr(module, "out_features", None)
            if isinstance(in_features, int) and isinstance(out_features, int) and in_features == int(hidden_size):
                observed = int(out_features)
                break
    except Exception:
        return None

    if observed is None or observed <= 0:
        return None

    expected = _ceil_to_multiple((2 * (4 * int(hidden_size))) / 3, int(multiple_of))
    if observed == expected:
        return None

    if observed < expected:
        # Unexpected layout; don't guess.
        return None

    ratio = float(expected) / float(observed)
    if ratio <= 0.0 or ratio >= 1.0:
        return None
    return ratio


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if "siglip" in vision_tower:
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    if "eva" in vision_tower:
        return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')




def build_gen_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'gen_vision_tower')
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if "siglip" in vision_tower:
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    if "eva" in vision_tower:
        return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')



def build_dit(vision_tower_cfg, **kwargs):
    if not hasattr(vision_tower_cfg, "hidden_size"):
        model_name_or_path = getattr(vision_tower_cfg, "model_name_or_path", "")
        if "3B" in model_name_or_path:
            vision_tower_cfg.hidden_size = 2048
        elif "7B" in model_name_or_path:
            vision_tower_cfg.hidden_size = 3584

    ffn_dim_multiplier = getattr(vision_tower_cfg, "dit_ffn_dim_multiplier", None)
    if ffn_dim_multiplier is None:
        ffn_dim_multiplier = getattr(vision_tower_cfg, "ffn_dim_multiplier", None)
    if ffn_dim_multiplier is None:
        inferred = _infer_lumina_ffn_multiplier(hidden_size=1792, multiple_of=256)
        if inferred is not None:
            ffn_dim_multiplier = inferred
            print(
                f"[BLIP3o] Adjusting Lumina FFN multiplier to {ffn_dim_multiplier:.6f} "
                "for diffusers/checkpoint compatibility."
            )

    dit_cfg = NextDiTCrossAttnConfig(
        latent_embedding_size=vision_tower_cfg.hidden_size,
        ffn_dim_multiplier=ffn_dim_multiplier,
    )
    dit = NextDiTCrossAttn(dit_cfg)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("Alpha-VLLM/Lumina-Next-SFT-diffusers", subfolder="scheduler")
    return dit, noise_scheduler

