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
        if "3B" in vision_tower_cfg.model_name_or_path:
            vision_tower_cfg.hidden_size = 2048
        elif "7B" in vision_tower_cfg.model_name_or_path:
            vision_tower_cfg.hidden_size = 3584

    dit = NextDiTCrossAttn(NextDiTCrossAttnConfig(latent_embedding_size=vision_tower_cfg.hidden_size))
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("Alpha-VLLM/Lumina-Next-SFT-diffusers", subfolder="scheduler")
    return dit, noise_scheduler


