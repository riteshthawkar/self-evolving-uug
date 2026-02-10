"""BLIP3o diffusion decoder pipeline support."""

import json
import math
import os
import pathlib
import shutil
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .model_api import _find_callable_object, _iter_wrapper_objects
from .generation_helpers import _ensure_pil_image


def _decode_blip3o_generate_image_output(api_obj, image_out):
    decode_fn = getattr(api_obj, "decode_latents", None)
    if not callable(decode_fn):
        decode_owner, _ = _find_callable_object(api_obj, "decode_latents")
        if decode_owner is not None:
            api_obj = decode_owner
            decode_fn = getattr(api_obj, "decode_latents", None)

    model_obj_fn = getattr(api_obj, "get_model", None)
    if not callable(model_obj_fn):
        model_owner, _ = _find_callable_object(api_obj, "get_model")
        if model_owner is not None:
            model_obj_fn = getattr(model_owner, "get_model", None)
    model_obj = None
    if callable(model_obj_fn):
        try:
            model_obj = model_obj_fn()
        except Exception:
            model_obj = None
    vae_obj = None
    if model_obj is not None:
        vae_obj = getattr(model_obj, "vae", None)
        if vae_obj is None:
            vae_obj = getattr(model_obj, "sana_vae", None)
    if not callable(decode_fn) and vae_obj is None:
        return None

    latents = image_out
    if torch.is_tensor(latents) and latents.dtype in (torch.bfloat16, torch.float16):
        latents = latents.to(torch.float32)

    decode_device = None
    decode_dtype = None
    if vae_obj is not None:
        try:
            p = next(vae_obj.parameters())
            decode_device = p.device
            decode_dtype = p.dtype
        except Exception:
            pass

    candidates: List[torch.Tensor] = []
    if torch.is_tensor(latents):
        candidates.append(latents)
        try:
            if latents.ndim == 3:
                bsz = int(latents.shape[0])
                n = int(latents.shape[1])
                c = int(latents.shape[2])

                s = int(round(math.sqrt(float(n))))
                if s * s == n:
                    candidates.append(latents.permute(0, 2, 1).contiguous().view(bsz, c, s, s))

                s2 = int(round(math.sqrt(float(c))))
                if s2 * s2 == c:
                    candidates.append(latents.contiguous().view(bsz, n, s2, s2))

                model_obj_getter = getattr(api_obj, "get_model", None)
                if callable(model_obj_getter):
                    m = model_obj_getter()
                    dit = getattr(m, "dit", None)
                    cfg = getattr(dit, "config", None) if dit is not None else None
                    in_channels = int(getattr(cfg, "in_channels", 0) or 0)
                    input_size = int(getattr(cfg, "input_size", 0) or 0)
                    target = in_channels * input_size * input_size
                    if bsz > 0 and target > 0 and int(latents[0].numel()) == target:
                        candidates.append(
                            latents.permute(0, 2, 1).contiguous().view(
                                bsz, in_channels, input_size, input_size
                            )
                        )

                total = n * c
                for side in (64, 56, 48, 40, 32, 24, 16, 8):
                    area = side * side
                    if total % area != 0:
                        continue
                    channels = total // area
                    if channels <= 0 or channels > 128:
                        continue
                    candidates.append(
                        latents.permute(0, 2, 1).contiguous().view(
                            bsz, channels, side, side
                        )
                    )
        except Exception:
            pass

    for cand in candidates:
        cand_for_decode = cand
        if torch.is_tensor(cand_for_decode):
            try:
                if decode_device is not None:
                    tgt_dtype = decode_dtype if decode_dtype is not None else cand_for_decode.dtype
                    if tgt_dtype in (torch.float16, torch.bfloat16):
                        tgt_dtype = torch.float32
                    cand_for_decode = cand_for_decode.to(device=decode_device, dtype=tgt_dtype)
            except Exception:
                cand_for_decode = cand

        try:
            if callable(decode_fn):
                decoded = decode_fn(cand_for_decode, return_tensor=False)
            elif vae_obj is not None:
                lat_for_vae = cand_for_decode
                cfg = getattr(vae_obj, "config", None)
                scaling = getattr(cfg, "scaling_factor", None) if cfg is not None else None
                shift = getattr(cfg, "shift_factor", None) if cfg is not None else None
                if scaling is not None:
                    lat_for_vae = lat_for_vae / float(scaling)
                if shift is not None:
                    lat_for_vae = lat_for_vae + float(shift)
                sample = vae_obj.decode(lat_for_vae)
                decoded = sample.sample if hasattr(sample, "sample") else sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
                decoded = [Image.fromarray(np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)) for x in decoded]
            else:
                continue
        except Exception:
            continue
        try:
            if isinstance(decoded, list) and decoded:
                return _ensure_pil_image(decoded[0])
            return _ensure_pil_image(decoded)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# BLIP3o model-name helpers
# ---------------------------------------------------------------------------


def _is_original_blip3o_model_name(model_name: str) -> bool:
    name = (model_name or "").strip().lower()
    return "blip3o-model" in name and "next" not in name


def _is_blip3o_next_model_name(model_name: str) -> bool:
    name = (model_name or "").strip().lower()
    return "blip3o-next" in name


def _looks_like_unregistered_blip3o_arch_error(text: str) -> bool:
    t = (text or "").lower()
    return ("blip3o_qwen" in t) and ("does not recognize this architecture" in t or "unrecognized" in t)


def _is_next_style_blip3o_class(cls) -> bool:
    """
    Detect BLIP3o-NEXT style classes (Qwen3-based) that are incompatible with
    original BLIP3o-Model checkpoints (Qwen2.5-VL based).
    """
    try:
        for base in cls.mro():
            name = getattr(base, "__name__", "").lower()
            if "qwen3" in name:
                return True
    except Exception:
        pass
    module_name = str(getattr(cls, "__module__", "")).lower()
    return "qwen3" in module_name


def _resolve_multimodal_encoder_for_pipeline(model):
    """Select the object that exposes ``generate_image`` for original BLIP3o pipelines."""
    try:
        for _, obj in _iter_wrapper_objects(model):
            if callable(getattr(obj, "generate_image", None)):
                return obj
    except Exception:
        pass
    return model


# ---------------------------------------------------------------------------
# Diffusion-pipeline helpers (original BLIP3o decoder)
# ---------------------------------------------------------------------------


def _link_or_copy_file(src: pathlib.Path, dst: pathlib.Path):
    if dst.exists():
        return
    try:
        os.symlink(str(src), str(dst))
        return
    except Exception:
        pass
    try:
        os.link(str(src), str(dst))
        return
    except Exception:
        pass
    shutil.copy2(str(src), str(dst))


def _ensure_diffusers_component_weight_aliases(component_dir: pathlib.Path):
    """Create ``diffusion_pytorch_model.*`` aliases when only generic names exist."""
    if not component_dir.is_dir():
        return
    target_sf = component_dir / "diffusion_pytorch_model.safetensors"
    target_bin = component_dir / "diffusion_pytorch_model.bin"
    if target_sf.exists() or target_bin.exists():
        return

    sf_candidates: List[pathlib.Path] = []
    bin_candidates: List[pathlib.Path] = []

    preferred_sf = component_dir / "pytorch_model.safetensors"
    preferred_bin = component_dir / "pytorch_model.bin"
    if preferred_sf.is_file():
        sf_candidates.append(preferred_sf)
    if preferred_bin.is_file():
        bin_candidates.append(preferred_bin)

    for p in sorted(component_dir.glob("*.safetensors")):
        if p.is_file():
            sf_candidates.append(p)
    for p in sorted(component_dir.glob("*.bin")):
        if p.is_file():
            bin_candidates.append(p)

    sf_candidates = list(dict.fromkeys(sf_candidates))
    bin_candidates = list(dict.fromkeys(bin_candidates))

    if sf_candidates:
        try:
            _link_or_copy_file(sf_candidates[0], target_sf)
            return
        except Exception:
            pass
    if bin_candidates:
        try:
            _link_or_copy_file(bin_candidates[0], target_bin)
        except Exception:
            pass


def _build_original_blip3o_diffusion_pipeline(
    model_name: str,
    *,
    multimodal_encoder,
    processor,
    torch_dtype: torch.dtype,
    device: torch.device,
):
    """Build original BLIP3o diffusion decoder pipeline from HF model repo."""
    try:
        from diffusers import DiffusionPipeline
    except Exception as exc:
        raise RuntimeError(
            "diffusers is required for original BLIP3o diffusion-decoder backend."
        ) from exc

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        # BLIP3o processors may nest tokenizer under multimodal_processor
        mm_proc = getattr(processor, "multimodal_processor", None)
        if mm_proc is not None:
            tokenizer = getattr(mm_proc, "tokenizer", None)
    if tokenizer is None and hasattr(processor, "tokenizer_image_token"):
        tokenizer = processor
    if tokenizer is None:
        # Last resort: try AutoTokenizer from the same model repo
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_name), trust_remote_code=True,
            )
        except Exception:
            pass
    if tokenizer is None:
        raise RuntimeError("Processor does not expose tokenizer required by BLIP3o diffusion pipeline.")

    attempts = ("pipeline_llava_gen", "pipeline_ar_gen")
    repo_candidates: List[str] = []
    primary_repo = str(model_name or "").strip()
    if primary_repo:
        repo_candidates.append(primary_repo)

    lowered = primary_repo.lower()
    if "blip3o-model-8b" in lowered:
        repo_candidates.append("BLIP3o/BLIP3o-Model")

    extra_decoder_repo = os.environ.get("BLIP3O_DIFFUSION_REPO", "").strip()
    if extra_decoder_repo:
        repo_candidates.append(extra_decoder_repo)

    repo_candidates = list(dict.fromkeys(repo_candidates))

    errors: List[str] = []
    pipe = None
    for repo_id in repo_candidates:
        for custom_pipeline in attempts:
            try:
                try:
                    pipe = DiffusionPipeline.from_pretrained(
                        repo_id,
                        custom_pipeline=custom_pipeline,
                        subfolder="diffusion-decoder",
                        tokenizer=tokenizer,
                        multimodal_encoder=multimodal_encoder,
                        safety_checker=None,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="bf16",
                    )
                except TypeError:
                    pipe = DiffusionPipeline.from_pretrained(
                        repo_id,
                        custom_pipeline=custom_pipeline,
                        subfolder="diffusion-decoder",
                        tokenizer=tokenizer,
                        multimodal_encoder=multimodal_encoder,
                        safety_checker=None,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="bf16",
                    )
                break
            except Exception as exc:
                errors.append(f"{repo_id}:{custom_pipeline}: {repr(exc)}")
        if pipe is not None:
            break

    # Fallback: resolve local ``diffusion-decoder`` path via snapshot_download.
    if pipe is None:
        try:
            from huggingface_hub import snapshot_download

            for repo_id in repo_candidates:
                local_repo = pathlib.Path(
                    snapshot_download(
                        repo_id=repo_id,
                        allow_patterns=[
                            "diffusion-decoder/**",
                            "pipeline_llava_gen.py",
                            "pipeline_ar_gen.py",
                        ],
                    )
                )
                diff_dir = local_repo / "diffusion-decoder"
                if diff_dir.is_dir():
                    for component in sorted(diff_dir.iterdir()):
                        if component.is_dir():
                            _ensure_diffusers_component_weight_aliases(component)

                    model_index_path = diff_dir / "model_index.json"
                    if model_index_path.is_file():
                        try:
                            with open(model_index_path, "r") as f:
                                mi = json.load(f)
                            mm_enc_cfg = mi.get("multimodal_encoder")
                            if isinstance(mm_enc_cfg, list) and "transformers_modules" in mm_enc_cfg[0]:
                                mi["multimodal_encoder"] = ["transformers", "PreTrainedModel"]
                                with open(model_index_path, "w") as f:
                                    json.dump(mi, f, indent=2)
                        except Exception:
                            pass

                    for custom_pipeline in attempts:
                        cp_arg: object = custom_pipeline
                        root_cp = local_repo / f"{custom_pipeline}.py"
                        sub_cp = diff_dir / f"{custom_pipeline}.py"
                        if root_cp.is_file():
                            cp_arg = str(root_cp)
                        elif sub_cp.is_file():
                            cp_arg = str(sub_cp)
                        try:
                            try:
                                pipe = DiffusionPipeline.from_pretrained(
                                    str(diff_dir),
                                    custom_pipeline=cp_arg,
                                    tokenizer=tokenizer,
                                    multimodal_encoder=multimodal_encoder,
                                    safety_checker=None,
                                    trust_remote_code=True,
                                    torch_dtype=torch_dtype,
                                    use_safetensors=True,
                                    variant="bf16",
                                )
                                break
                            except TypeError:
                                pipe = DiffusionPipeline.from_pretrained(
                                    str(diff_dir),
                                    custom_pipeline=cp_arg,
                                    tokenizer=tokenizer,
                                    multimodal_encoder=multimodal_encoder,
                                    safety_checker=None,
                                    torch_dtype=torch_dtype,
                                    use_safetensors=True,
                                    variant="bf16",
                                )
                        except Exception as exc:
                            errors.append(f"local_snapshot:{repo_id}:{custom_pipeline}: {repr(exc)}")
                    if pipe is not None:
                        break
                else:
                    errors.append(
                        f"local_snapshot:{repo_id}: missing diffusion-decoder dir under {local_repo}"
                    )
        except Exception as exc:
            errors.append(f"local_snapshot: {repr(exc)}")

    if pipe is None:
        detail = " | ".join(errors)
        raise RuntimeError(f"Failed to build BLIP3o diffusion pipeline. {detail}")

    try:
        pipe = pipe.to(device)
    except Exception:
        pass
    return pipe
