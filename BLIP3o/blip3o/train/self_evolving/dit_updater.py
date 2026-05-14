"""Diffusion Generator updater for generation/unified self-evolving training.

This updater applies a denoising objective to BLIP3o's diffusion transformer
(DiT) LoRA adapters using real source images and generation prompts. When
reward weighting is enabled, the denoising loss is scaled by the Solver-derived
generation reward, giving a diffusion-compatible Generator update without
unfreezing the base denoiser weights.
"""

import contextlib
import gc
import math
import traceback as _traceback
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from .utils import _clip_grad_norm_multi_device, _unwrap_model

try:
    from blip3o.mm_utils import process_images as _process_images_fn
except Exception:
    _process_images_fn = None


class DiTUpdater:
    """Direct denoising updater for the model's DiT module."""

    def __init__(
        self,
        model: torch.nn.Module,
        processor,
        config,
    ):
        self.model = model
        self.processor = processor
        self.config = config
        self.step_id = 0
        self.grad_accum_steps = max(1, int(getattr(config, "dit_grad_accum_steps", 1)))
        self.grad_clip = float(getattr(config, "dit_grad_clip", 1.0))
        self.cond_dropout = float(getattr(config, "dit_conditioning_dropout", 0.10))
        self.loss_weight = float(getattr(config, "dit_loss_weight", 1.0))
        self.prompt_suffix_token_id = int(getattr(config, "dit_prompt_suffix_token_id", 151665))

        self._accum_count = 0
        self._has_real_grad_in_window = False

        model_ref = _unwrap_model(model)
        core_model_getter = getattr(model_ref, "get_model", None)
        if not callable(core_model_getter):
            raise RuntimeError("DiT updater requires model.get_model().")
        self.model_ref = model_ref
        self.core_model = core_model_getter()
        self.dit = getattr(self.core_model, "dit", None)
        self.dit_base = self._unwrap_dit_module(self.dit)
        self.noise_scheduler = getattr(self.core_model, "noise_scheduler", None)
        self.gen_vision_tower = getattr(self.core_model, "get_gen_vision_tower", lambda: None)()
        if self.dit is None:
            raise RuntimeError("DiT updater requires core model to expose `dit`.")
        if self.noise_scheduler is None:
            raise RuntimeError("DiT updater requires core model to expose `noise_scheduler`.")
        if self.gen_vision_tower is None:
            raise RuntimeError("DiT updater requires a generation vision tower (`get_gen_vision_tower`).")

        # Eagerly load the gen vision tower NOW — before any distributed collectives.
        # Eva-CLIP towers created with delay_load=True skip weight loading until
        # load_model() is called.  If we let this happen lazily inside _prepare_latents
        # (during the first generation step), different ranks may call load_model() at
        # different times.  Some ranks will then enter DDP buffer-broadcast collectives
        # while others are still loading weights, causing a NCCL timeout after 10 min.
        # Loading eagerly in __init__ (before dist collectives start) is safe because
        # all ranks construct DiTUpdater at the same point in trainer __init__.
        self._ensure_gen_vision_tower_loaded()
        # After load_model() the tower weights land on CPU. Move them to the same
        # device as the DiT parameters so the forward pass doesn't cause device drift.
        try:
            _dit_device = next(iter(self.dit.parameters())).device
            self.gen_vision_tower.to(_dit_device)
        except Exception:
            pass
        # Barrier: make sure ALL ranks have finished loading before any rank proceeds
        # to construct optimisers / enter the training loop (which may trigger collectives).
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        self.dit_lora_enabled = bool(getattr(config, "dit_lora_enabled", True))
        if self.dit_lora_enabled:
            named_dit_params = list(self.dit.named_parameters())
            for name, param in named_dit_params:
                param.requires_grad_("lora_" in name)
            self.params = [
                param
                for name, param in named_dit_params
                if "lora_" in name and param.requires_grad
            ]
            if not self.params:
                raise RuntimeError(
                    "dit_lora_enabled=True but no trainable DiT LoRA parameters were found. "
                    "Check --dit_lora_targets against the loaded BLIP3o DiT module names."
                )
        else:
            for p in self.dit.parameters():
                p.requires_grad_(True)
            self.params = [p for p in self.dit.parameters() if p.requires_grad]
        if not self.params:
            raise RuntimeError("No trainable DiT parameters found.")
        self.trainable_param_count = int(sum(p.numel() for p in self.params))
        self.trainable_param_tensors = int(len(self.params))

        lr = float(getattr(config, "dit_lr", getattr(config, "lr", 1e-6)))
        weight_decay = float(getattr(config, "dit_weight_decay", getattr(config, "weight_decay", 0.01)))

        # ── Joint LLM+DiT conditioning training (Change 2+3) ────────────────────
        # When dit_joint_conditioning_train=True, we also train the generator LoRA
        # (the LLM's conditioning encoder) jointly with the DiT using the same
        # denoising loss. Gradients flow: MSE → DiT → z_latents → generator LoRA.
        # This trains BOTH "how to denoise given conditioning" (DiT) AND
        # "how to encode text into conditioning" (generator LoRA) simultaneously.
        self.joint_conditioning_train = bool(getattr(config, "dit_joint_conditioning_train", False))
        self.reward_loss_weight = float(getattr(config, "dit_reward_loss_weight", 0.0))
        self.generator_lora_params: list = []

        param_groups = [{"params": self.params, "lr": lr, "weight_decay": weight_decay}]

        if self.joint_conditioning_train:
            # Collect generator LoRA parameters from the LLM backbone.
            # These are the q/k/v/o/gate/up/down LoRA A+B matrices for "generator" adapter.
            joint_lr = float(getattr(config, "dit_joint_conditioning_lr", lr))
            gen_lora_params = []
            try:
                from peft import PeftModel
                model_unwrapped = _unwrap_model(model)
                # Try PEFT adapter parameter access first
                def _is_generator_lora(name: str) -> bool:
                    if ".dit." in name or name.startswith("dit."):
                        return False
                    return (
                        ("lora_A" in name or "lora_B" in name)
                        and (".generator." in name or "generator." in name)
                    )

                if hasattr(model_unwrapped, "peft_config") and "generator" in getattr(model_unwrapped, "peft_config", {}):
                    for name, p in model_unwrapped.named_parameters():
                        if _is_generator_lora(name):
                            p.requires_grad_(True)
                            gen_lora_params.append(p)
                elif hasattr(model_unwrapped, "base_model"):
                    for name, p in model_unwrapped.named_parameters():
                        if _is_generator_lora(name):
                            p.requires_grad_(True)
                            gen_lora_params.append(p)
            except Exception:
                pass

            if gen_lora_params:
                self.generator_lora_params = gen_lora_params
                param_groups.append({
                    "params": gen_lora_params,
                    "lr": joint_lr,
                    "weight_decay": weight_decay,
                })
                print(
                    f"[DiTUpdater] Joint conditioning training enabled: "
                    f"{len(gen_lora_params)} generator LoRA params added to optimizer "
                    f"(lr={joint_lr:.2e})."
                )
            else:
                print(
                    "[DiTUpdater] WARNING: dit_joint_conditioning_train=True but no "
                    "generator LoRA params found. Falling back to DiT-only training."
                )

        self.opt = torch.optim.AdamW(param_groups)

        self.distributed = bool(dist.is_available() and dist.is_initialized())
        self.world_size = int(dist.get_world_size()) if self.distributed else 1
        if not self.distributed or dist.get_rank() == 0:
            mode = "lora" if self.dit_lora_enabled else "full"
            print(
                f"[DiTUpdater] DiT update mode={mode}; "
                f"trainable_tensors={self.trainable_param_tensors}; "
                f"trainable_params={self.trainable_param_count}; "
                f"lr={lr:.2e}; weight_decay={weight_decay:.2e}"
            )

    @staticmethod
    def _unwrap_dit_module(dit: Optional[torch.nn.Module]) -> Optional[torch.nn.Module]:
        if dit is None:
            return None
        getter = getattr(dit, "get_base_model", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                pass
        base_model = getattr(dit, "base_model", None)
        inner = getattr(base_model, "model", None) if base_model is not None else None
        return inner if inner is not None else dit

    def state_dict(self) -> Dict:
        return {
            "optimizer": self.opt.state_dict(),
            "step_id": int(self.step_id),
            "accum_count": int(self._accum_count),
            "has_real_grad_in_window": bool(self._has_real_grad_in_window),
            "dit_lora_enabled": bool(self.dit_lora_enabled),
            "trainable_param_count": int(self.trainable_param_count),
        }

    def load_state_dict(self, state: Dict):
        if not isinstance(state, dict):
            return
        if "optimizer" in state and isinstance(state.get("optimizer"), dict):
            try:
                self.opt.load_state_dict(state["optimizer"])
            except Exception as exc:
                if not self.distributed or dist.get_rank() == 0:
                    print(f"[DiTUpdater] WARNING: failed to restore optimizer state: {exc}")
        if "step_id" in state:
            self.step_id = int(state["step_id"])
        if "accum_count" in state:
            self._accum_count = int(state["accum_count"])
        if "has_real_grad_in_window" in state:
            self._has_real_grad_in_window = bool(state["has_real_grad_in_window"])

    def _dist_all_bool(self, value: bool) -> bool:
        if not self.distributed:
            return bool(value)
        device = self.params[0].device
        tensor = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return bool(int(tensor.item()) == 1)

    def _dist_any_bool(self, value: bool) -> bool:
        if not self.distributed:
            return bool(value)
        device = self.params[0].device
        tensor = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return bool(int(tensor.item()) == 1)

    def _average_gradients(self):
        if not self.distributed:
            return
        # Average gradients for DiT params AND generator LoRA params (if joint training).
        all_trainable = list(self.params) + list(self.generator_lora_params)
        for param in all_trainable:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= float(self.world_size)

    def _tokenizer(self):
        tok = getattr(self.processor, "tokenizer", None)
        return tok if tok is not None else self.processor

    def _find_image_processor(self):
        """Return the image_processor from the gen_vision_tower, searching inner wrappers if needed.

        Eva-CLIP towers only set self.image_processor inside load_model(), which is
        skipped when delay_load=True (common in distributed training).  In that case
        we construct the processor on-the-fly from the tower's always-available config.
        """
        # Direct attribute on the tower (set when load_model() was called).
        ip = getattr(self.gen_vision_tower, "image_processor", None)
        if ip is not None:
            return ip

        # EvaClipVisionTower: self.config is always set in __init__ (even with
        # delay_load=True), but self.image_processor is only set inside load_model().
        # Reconstruct the processor directly from the config dict.
        tower_cls_name = type(self.gen_vision_tower).__name__
        if "Eva" in tower_cls_name or "EVA" in tower_cls_name or "eva" in tower_cls_name.lower():
            tower_cfg = getattr(self.gen_vision_tower, "config", None)
            if tower_cfg is None:
                tower_cfg = getattr(self.gen_vision_tower, "cfg_only", None)
            if isinstance(tower_cfg, dict):
                try:
                    image_size = int(tower_cfg["vision_cfg"]["image_size"])
                except (KeyError, TypeError, ValueError):
                    image_size = 224
            else:
                image_size = int(getattr(self.gen_vision_tower, "image_size", 224))
            try:
                from blip3o.model.multimodal_encoder.eva_clip.eva_clip_processors import (
                    EvaClipImageTrainProcessor,
                )
                return EvaClipImageTrainProcessor(image_size)
            except ImportError:
                pass  # fall through to remaining search paths

        # Towers that wrap an inner model under .vision_tower (e.g. SigLipVisionTower)
        inner = getattr(self.gen_vision_tower, "vision_tower", None)
        if inner is not None:
            ip = getattr(inner, "image_processor", None)
            if ip is not None:
                return ip

        # Final fallback: check core_model for a gen-side processor
        ip = getattr(self.core_model, "gen_image_processor", None)
        if ip is not None:
            return ip
        return None

    @staticmethod
    def _call_image_processor(image_processor, image: Image.Image) -> torch.Tensor:
        """Call the image processor regardless of whether it uses .preprocess() or __call__.

        Handles three processor families:
        1. EvaClipImageTrainProcessor / SigLipImageProcessor — expose .preprocess()
           but may return pixel_values as a **list of numpy arrays** rather than a tensor.
        2. Standard HuggingFace BaseImageProcessor — callable with images=[...], returns
           a BatchFeature with pixel_values already stacked into a tensor.
        3. Raw torchvision-style callable — returns a tensor directly via __call__.
        """
        if hasattr(image_processor, "preprocess"):
            out = image_processor.preprocess(image, return_tensors="pt")
        else:
            out = image_processor(images=[image], return_tensors="pt")

        if isinstance(out, dict):
            pixel_values = out.get("pixel_values")
        else:
            pixel_values = getattr(out, "pixel_values", None)

        if pixel_values is None:
            raise RuntimeError("Image processor did not return pixel_values.")

        # EvaClipImageTrainProcessor.preprocess() accumulates transformed images
        # as a list of numpy arrays and wraps them in BatchFeature — the tensor_type="pt"
        # conversion only works if numpy is available AND the data is already float32.
        # Guard against receiving a list/ndarray here and convert explicitly.
        if not torch.is_tensor(pixel_values):
            import numpy as np
            if isinstance(pixel_values, (list, tuple)):
                arrays = []
                for pv in pixel_values:
                    if torch.is_tensor(pv):
                        arrays.append(pv)
                    elif isinstance(pv, np.ndarray):
                        arrays.append(torch.from_numpy(np.ascontiguousarray(pv)))
                    else:
                        arrays.append(torch.tensor(pv, dtype=torch.float32))
                pixel_values = torch.stack(arrays, dim=0)
            elif isinstance(pixel_values, np.ndarray):
                pixel_values = torch.from_numpy(np.ascontiguousarray(pixel_values))
            else:
                pixel_values = torch.tensor(pixel_values, dtype=torch.float32)

        return pixel_values

    def _prepare_image_tensor(self, image: Image.Image, device: torch.device) -> torch.Tensor:
        image_processor = self._find_image_processor()
        if image_processor is None:
            raise RuntimeError(
                "Generation vision tower does not expose image_processor. "
                f"Tower type: {type(self.gen_vision_tower).__name__}, "
                f"attrs: {[a for a in dir(self.gen_vision_tower) if not a.startswith('_')][:20]}"
            )
        model_cfg = getattr(self.core_model, "config", None)

        # Use the shared process_images helper when available, but only when the
        # image_processor is a standard HF processor that accepts __call__ with
        # keyword args (images=..., return_tensors=...).  Custom processors such
        # as SigLipImageProcessor only expose .preprocess(), and process_images
        # internally calls image_processor(images, return_tensors='pt') which
        # would fail for those.  We detect HF-style processors by the absence of
        # a dedicated .preprocess() method (HF processors inherit it from
        # BaseImageProcessor, but also override __call__ — the distinguishing
        # sign of a pure-custom processor is that it has .preprocess but no
        # from_pretrained classmethod).
        _is_custom_processor = (
            hasattr(image_processor, "preprocess")
            and not hasattr(image_processor, "from_pretrained")
        )
        use_process_images_fn = (
            _process_images_fn is not None
            and model_cfg is not None
            and not _is_custom_processor
        )
        if use_process_images_fn:
            pixel_values = _process_images_fn([image], image_processor, model_cfg)
        else:
            pixel_values = self._call_image_processor(image_processor, image)

        if not torch.is_tensor(pixel_values):
            raise RuntimeError(
                f"Image preprocessing did not return a tensor (got {type(pixel_values)})."
            )

        tower_dtype = None
        tower_device = device
        try:
            p0 = next(self.gen_vision_tower.parameters())
            tower_dtype = p0.dtype
            tower_device = p0.device
        except Exception:
            tower_dtype = torch.float32
            tower_device = device

        return pixel_values.to(device=tower_device, dtype=tower_dtype)

    def _reshape_latents(
        self,
        latents: torch.Tensor,
        *,
        in_channels: int,
        input_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        x = latents
        if x.ndim == 4:
            if x.shape[1] == in_channels:
                pass
            elif x.shape[-1] == in_channels:
                x = x.permute(0, 3, 1, 2).contiguous()
            else:
                raise RuntimeError(
                    f"Unexpected 4D latent shape from gen tower: {tuple(x.shape)} "
                    f"(expected channel={in_channels})."
                )
        elif x.ndim == 3:
            if x.shape[-1] == in_channels:
                n_tokens = int(x.shape[1])
                side = int(round(math.sqrt(float(n_tokens))))
                if side * side != n_tokens:
                    raise RuntimeError(
                        f"Cannot reshape token latents {tuple(x.shape)} to square spatial map."
                    )
                x = x.permute(0, 2, 1).contiguous().view(x.shape[0], in_channels, side, side)
            elif x.shape[1] == in_channels:
                n_tokens = int(x.shape[2])
                side = int(round(math.sqrt(float(n_tokens))))
                if side * side != n_tokens:
                    raise RuntimeError(
                        f"Cannot reshape channel-first token latents {tuple(x.shape)} to square spatial map."
                    )
                x = x.view(x.shape[0], in_channels, side, side)
            else:
                raise RuntimeError(
                    f"Unexpected 3D latent shape from gen tower: {tuple(x.shape)} "
                    f"(expected one axis to equal in_channels={in_channels})."
                )
        else:
            raise RuntimeError(f"Unsupported latent rank: ndim={x.ndim}, shape={tuple(x.shape)}")

        if x.shape[-2:] != (input_size, input_size):
            x = F.interpolate(
                x.float(),
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=x.dtype)
        return x.to(device=device, dtype=dtype)

    def _ensure_gen_vision_tower_loaded(self):
        """Lazily call load_model() on the gen vision tower if it was created with delay_load=True.

        EvaClipVisionTower (and similar towers) skip their model construction when
        delay_load=True, storing only the config dict.  The DiT updater needs the
        tower to be fully loaded to run inference.  We detect an unloaded tower by
        checking the is_loaded flag or by confirming vision_tower is absent.
        """
        is_loaded = getattr(self.gen_vision_tower, "is_loaded", None)
        if is_loaded is True:
            return  # Already loaded; nothing to do.

        has_inner = hasattr(self.gen_vision_tower, "vision_tower")
        if has_inner:
            return  # Inner tower exists — treat as loaded even if flag is missing.

        load_fn = getattr(self.gen_vision_tower, "load_model", None)
        if callable(load_fn):
            try:
                load_fn()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to lazy-load gen vision tower "
                    f"({type(self.gen_vision_tower).__name__}): {e}"
                ) from e

    def _prepare_latents(self, image: Image.Image) -> Tuple[torch.Tensor, torch.device, torch.dtype]:
        model_device = self.params[0].device
        model_dtype = self.params[0].dtype

        # Ensure the gen vision tower is fully initialised before running inference.
        # Eva-CLIP towers created with delay_load=True need load_model() called first.
        self._ensure_gen_vision_tower_loaded()

        gen_images = self._prepare_image_tensor(image=image, device=model_device)

        with torch.no_grad():
            outputs = self.gen_vision_tower(gen_images)
            if isinstance(outputs, torch.Tensor):
                latents = outputs
            else:
                latents = getattr(outputs, "last_hidden_state", None)
                if latents is None:
                     # Fallback for models that output only pooled embeddings or image_embeds
                     latents = getattr(outputs, "image_embeds", None)
                if latents is None and isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                    latents = outputs[0]
                if latents is None:
                    raise RuntimeError(f"Vision tower output type {type(outputs)} has no recognized latents.")

            # Apply the same pooling that the main training forward-pass uses.
            # For BLIP3o the gen_vision_tower outputs (B, 729, 1152) token features.
            # pool_img converts this to a 4-D spatial tensor (B, 1152, S, S) via
            # average-pooling with the configured stride.  The DiT was trained on
            # these spatial features, so we must pass them through the same path.
            get_pooling = getattr(self.model_ref, "get_gen_pooling", None)
            pool_img = getattr(self.model_ref, "pool_img", None)
            pooling = str(get_pooling() if callable(get_pooling) else "")
            if "early" in pooling and callable(pool_img) and latents.ndim == 3:
                latents = pool_img(latents)

        dit_cfg = getattr(self.dit_base, "config", None)
        # The DiT config's in_channels defaults to 1792 (Lumina pre-training), but
        # BLIP3o checkpoints are trained with SigLip visual features (hidden_size=1152).
        # After pool_img the latent is (B, 1152, S, S), so derive in_channels from the
        # actual latent rather than blindly trusting the config default.
        if latents.ndim == 4:
            # 4-D spatial tensor from pool_img: (B, C, H, W)
            in_channels = int(latents.shape[1])
            input_size = int(latents.shape[-1])   # use actual spatial size; reshape will interpolate if needed
        elif latents.ndim == 3:
            # Still a token tensor (B, N, C) — attempt to resolve from DiT config or latent dims.
            in_channels = int(getattr(dit_cfg, "in_channels", 1792))
            input_size = int(getattr(dit_cfg, "input_size", 8))
            # If neither token axis matches in_channels, try down_projector as last resort.
            if latents.shape[-1] != in_channels and latents.shape[1] != in_channels:
                down_projector = getattr(self.core_model, "down_projector", None)
                if down_projector is not None:
                    proj_dtype = latents.dtype
                    try:
                        proj_dtype = next(down_projector.parameters()).dtype
                    except Exception:
                        pass
                    with torch.no_grad():
                        latents = down_projector(latents.to(proj_dtype))
                    # Re-derive in_channels after projection in case it changed shape.
                    if latents.ndim == 4:
                        in_channels = int(latents.shape[1])
                        input_size = int(latents.shape[-1])
        else:
            in_channels = int(getattr(dit_cfg, "in_channels", 1792))
            input_size = int(getattr(dit_cfg, "input_size", 8))

        # Read the target spatial resolution from the DiT config for interpolation purposes.
        # Only override input_size if the config specifies a different value AND the latent
        # was not already in 4-D form (where the spatial size is the ground truth).
        dit_input_size = int(getattr(dit_cfg, "input_size", input_size))
        print(
            f"[DiTUpdater] latent shape after pool: {tuple(latents.shape)}, "
            f"in_channels={in_channels}, dit_input_size={dit_input_size}"
        )
        latents = self._reshape_latents(
            latents=latents,
            in_channels=in_channels,
            input_size=dit_input_size,
            device=model_device,
            dtype=model_dtype,
        )
        return latents, model_device, model_dtype

    def _prepare_conditioning(
        self,
        prompt: str,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tokenizer = self._tokenizer()
        if tokenizer is None or not hasattr(tokenizer, "__call__"):
            raise RuntimeError("DiT updater requires a tokenizer-like processor.")

        enc = tokenizer([str(prompt)], padding="longest", return_tensors="pt")
        if not isinstance(enc, dict):
            enc = dict(enc)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)

        if self.prompt_suffix_token_id >= 0:
            suffix = torch.full(
                (input_ids.shape[0], 1),
                fill_value=self.prompt_suffix_token_id,
                dtype=input_ids.dtype,
                device=device,
            )
            input_ids = torch.cat([input_ids, suffix], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)],
                dim=1,
            )

        embed_tokens = getattr(self.core_model, "embed_tokens", None)
        latent_queries = getattr(self.core_model, "latent_queries", None)
        if embed_tokens is None or latent_queries is None:
            raise RuntimeError("Core model does not expose embed_tokens/latent_queries for DiT conditioning.")

        # When joint_conditioning_train=True, we allow gradients to flow through
        # the LLM forward pass so that the generator LoRA params receive gradient
        # signal from the DiT denoising loss.  Otherwise freeze the conditioning
        # encoder with no_grad for efficiency (original behaviour).
        _cond_ctx = contextlib.nullcontext() if self.joint_conditioning_train else torch.no_grad()
        with _cond_ctx:
            text_embeds = embed_tokens(input_ids)
            if text_embeds.shape[0] != batch_size:
                text_embeds = text_embeds.expand(batch_size, -1, -1).contiguous()
            queries = latent_queries.to(device=device, dtype=text_embeds.dtype).repeat(batch_size, 1, 1)
            model_inputs = torch.cat([text_embeds, queries], dim=1)
            model_mask = torch.cat(
                [
                    attention_mask.expand(batch_size, -1),
                    torch.ones((batch_size, queries.shape[1]), dtype=attention_mask.dtype, device=device),
                ],
                dim=1,
            )
            outputs = self.core_model(
                inputs_embeds=model_inputs,
                attention_mask=model_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is not None and len(hidden_states) > 0:
                    hidden = hidden_states[-1]
                elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                    hidden = outputs[0]
                else:
                    raise RuntimeError("Core model forward did not return hidden states for DiT conditioning.")
            z_latents = hidden[:, -queries.shape[1] :, :].to(device=device, dtype=dtype)

        expected_dim = int(getattr(getattr(self.dit_base, "config", None), "latent_embedding_size", z_latents.shape[-1]))
        if z_latents.shape[-1] != expected_dim:
            down_projector = getattr(self.core_model, "down_projector", None)
            if down_projector is not None:
                proj_dtype = z_latents.dtype
                try:
                    proj_dtype = next(down_projector.parameters()).dtype
                except Exception:
                    pass
                # Allow gradients through the down_projector too when joint training.
                _proj_ctx = contextlib.nullcontext() if self.joint_conditioning_train else torch.no_grad()
                with _proj_ctx:
                    z_latents = down_projector(z_latents.to(proj_dtype)).to(device=device, dtype=dtype)
        if z_latents.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Conditioning width mismatch for DiT: got {z_latents.shape[-1]}, expected {expected_dim}."
            )
        return z_latents

    def _ensure_scheduler_timesteps(self):
        """Ensure the noise scheduler has timesteps set.

        FlowMatchEulerDiscreteScheduler initializes .timesteps to None and
        requires an explicit set_timesteps() call.  If the training loop has
        not already done this, we call it here with a default of 1000 steps
        (matching num_train_timesteps) so that we can sample uniformly across
        all noise levels during the SFT denoising objective.
        """
        timesteps = getattr(self.noise_scheduler, "timesteps", None)
        if timesteps is not None and torch.is_tensor(timesteps) and timesteps.numel() > 0:
            return  # Already initialized; nothing to do.
        total = int(getattr(getattr(self.noise_scheduler, "config", None), "num_train_timesteps", 1000))
        set_ts = getattr(self.noise_scheduler, "set_timesteps", None)
        if callable(set_ts):
            try:
                import numpy as np
                # Use a linear sigma schedule matching the training convention.
                sigmas = np.linspace(1.0, 1.0 / total, total)
                set_ts(total, sigmas=sigmas)
            except TypeError:
                # Some scheduler versions do not accept sigmas kwarg.
                try:
                    set_ts(total)
                except Exception:
                    pass
            except Exception:
                pass

    def _sample_training_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        self._ensure_scheduler_timesteps()
        total = int(getattr(getattr(self.noise_scheduler, "config", None), "num_train_timesteps", 1000))
        timesteps = getattr(self.noise_scheduler, "timesteps", None)
        if timesteps is None or not torch.is_tensor(timesteps) or timesteps.numel() <= 0:
            raise RuntimeError("Noise scheduler does not expose valid timesteps.")
        timesteps = timesteps.to(device=device)

        u = torch.rand(size=(batch_size,), device="cpu")
        indices = (u * float(total)).long()
        indices = indices.clamp(min=0, max=int(timesteps.numel()) - 1).to(device=device)
        return timesteps.index_select(0, indices)

    def _get_sigmas(self, timesteps: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        get_sigmas_fn = getattr(self.model_ref, "get_sigmas", None)
        if callable(get_sigmas_fn):
            return get_sigmas_fn(
                timesteps,
                latents.device,
                n_dim=latents.ndim,
                dtype=latents.dtype,
            )

        sigmas = getattr(self.noise_scheduler, "sigmas", None)
        schedule_timesteps = getattr(self.noise_scheduler, "timesteps", None)
        if sigmas is None or schedule_timesteps is None:
            raise RuntimeError("Scheduler does not expose sigmas/timesteps.")
        sigmas = sigmas.to(device=latents.device, dtype=latents.dtype)
        schedule_timesteps = schedule_timesteps.to(device=latents.device)
        step_indices = []
        for t in timesteps:
            idx = (schedule_timesteps == t).nonzero(as_tuple=False)
            if idx.numel() <= 0:
                raise RuntimeError(f"Unable to map timestep {int(t.item())} to scheduler sigma.")
            step_indices.append(int(idx[0].item()))
        sigma = sigmas[step_indices].flatten()
        while sigma.ndim < latents.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _condition_with_dropout(self, z_latents: torch.Tensor) -> torch.Tensor:
        mask_drop_fn = getattr(self.model_ref, "mask_drop", None)
        if callable(mask_drop_fn):
            return mask_drop_fn(z_latents, drop_prob=self.cond_dropout)
        if self.cond_dropout <= 0.0:
            return z_latents
        keep = torch.bernoulli(
            torch.full((z_latents.shape[0],), 1.0 - self.cond_dropout, device=z_latents.device, dtype=z_latents.dtype)
        )
        while keep.ndim < z_latents.ndim:
            keep = keep.unsqueeze(-1)
        return z_latents * keep

    def _build_zero_anchor_loss(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        anchor = torch.zeros((), device=device, dtype=dtype)
        # Include both DiT params AND generator LoRA params (when joint training) so
        # that the gradient graph is consistent across all parameter groups even on
        # skipped steps.  Without this, distributed all_reduce on LoRA grads would
        # find None/zero on some ranks and valid grads on others → NCCL divergence.
        all_trainable = list(self.params) + list(self.generator_lora_params)
        for p in all_trainable:
            anchor = anchor + (p.sum() * 0.0)
        return anchor

    def step(
        self,
        *,
        image: Image.Image,
        prompt: str,
        device: torch.device,
        reward: Optional[float] = None,
    ) -> Dict[str, object]:
        torch.set_grad_enabled(True)
        self.step_id += 1
        self.dit.train(True)
        # When jointly training the LLM conditioning encoder, set the core model to
        # train mode so that dropout / layer-norm behave correctly during forward.
        # In the default (frozen) path the core_model forward runs under no_grad
        # and its train/eval state does not matter for correctness.
        if self.joint_conditioning_train:
            self.core_model.train(True)

        # Restore requires_grad for DiT params AND generator LoRA params.
        # External code (e.g. FSDP, checkpoint loading) may have disabled grads.
        for p in list(self.params) + list(self.generator_lora_params):
            if not p.requires_grad:
                p.requires_grad_(True)

        if self._accum_count == 0:
            self.opt.zero_grad(set_to_none=True)
            self._has_real_grad_in_window = False

        local_ready = True
        local_skip_reason: Optional[str] = None
        latents: Optional[torch.Tensor] = None
        z_latents: Optional[torch.Tensor] = None
        model_device = self.params[0].device
        model_dtype = self.params[0].dtype

        try:
            latents, model_device, model_dtype = self._prepare_latents(image=image)
            z_latents = self._prepare_conditioning(
                prompt=prompt,
                batch_size=int(latents.shape[0]),
                device=model_device,
                dtype=model_dtype,
            )
        except Exception as exc:
            local_ready = False
            _exc_msg = str(exc)
            if not _exc_msg:
                _exc_msg = repr(exc)
            _exc_short = _exc_msg.replace("\n", " ")[:200]
            local_skip_reason = f"prepare_failed:{type(exc).__name__}:{_exc_short}"
            print(
                f"[DiTUpdater] prepare failed at step {self.step_id}: {type(exc).__name__}: {_exc_msg}\n"
                # + _traceback.format_exc() # Reduce noise in standard logs
            )

        ready_all = self._dist_all_bool(local_ready)
        if not ready_all:
            skipped_reason = local_skip_reason if local_skip_reason else "distributed_peer_prepare_failed"
            loss = self._build_zero_anchor_loss(device=model_device, dtype=model_dtype)
            has_real_grad = False
            valid_latent_tokens = 0.0
        else:
            assert latents is not None and z_latents is not None
            bsz = int(latents.shape[0])
            timesteps = self._sample_training_timestep(batch_size=bsz, device=model_device)
            sigmas = self._get_sigmas(timesteps=timesteps, latents=latents)
            noise = torch.randn_like(latents, device=model_device)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            z_latents = self._condition_with_dropout(z_latents)

            # Lumina DiT uses torch.utils.checkpoint when gradient checkpointing
            # is enabled. Checkpointed blocks require at least one input tensor
            # with requires_grad=True; otherwise the output is detached and loss
            # has no grad_fn. Enable grads on noisy latents in that case.
            dit_inner = self.dit_base if self.dit_base is not None else getattr(self.dit, "model", self.dit)
            gc_active = bool(getattr(dit_inner, "gradient_checkpointing", False))
            if gc_active and not noisy_latents.requires_grad:
                noisy_latents = noisy_latents.detach().requires_grad_(True)

            noise_pred = self.dit(
                x=noisy_latents,
                timestep=timesteps,
                z_latents=z_latents,
            )
            target = noise - latents
            mse = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
            loss = mse.to(dtype=model_dtype) * self.loss_weight

            # ── Reward-Weighted Regression (RWR) ────────────────────────────────
            # Scale denoising loss by (1 + w * reward) so that high-reward
            # generations are reinforced more strongly.  This implements the
            # continuous-action analogue of GRPO for the DiT+generator pathway.
            # reward_loss_weight=0 recovers the plain denoising objective.
            if self.reward_loss_weight > 0.0 and reward is not None:
                reward_scale = 1.0 + self.reward_loss_weight * float(reward)
                reward_scale = max(0.0, reward_scale)  # never invert the loss
                loss = loss * reward_scale

            valid_latent_tokens = float(latents.shape[-1] * latents.shape[-2])

            local_finite = bool(torch.isfinite(loss.detach()).all().item())
            finite_all = self._dist_all_bool(local_finite)
            if not finite_all:
                skipped_reason = "non_finite_dit_loss"
                loss = self._build_zero_anchor_loss(device=model_device, dtype=model_dtype)
                has_real_grad = False
            else:
                has_graph = bool(loss.requires_grad)
                graph_all = self._dist_all_bool(has_graph)
                if not graph_all:
                    skipped_reason = "dit_loss_no_grad_graph"
                    loss = self._build_zero_anchor_loss(device=model_device, dtype=model_dtype)
                    has_real_grad = False
                else:
                    skipped_reason = None
                    has_real_grad = True

        scaled_loss = loss / float(self.grad_accum_steps)
        scaled_loss.backward()

        has_real_grad = self._dist_any_bool(has_real_grad)
        self._accum_count += 1
        if has_real_grad:
            self._has_real_grad_in_window = True

        did_step = False
        if self._accum_count >= self.grad_accum_steps:
            if self._has_real_grad_in_window:
                self._average_gradients()
                # Clip gradients across all jointly-trained params
                # (DiT params + generator LoRA params if joint_conditioning_train=True).
                _all_trainable = list(self.params) + list(self.generator_lora_params)
                if _clip_grad_norm_multi_device(_all_trainable, self.grad_clip):
                    self.opt.step()
                    did_step = True
                else:
                    skipped_reason = skipped_reason or "non_finite_gradient"
            self.opt.zero_grad(set_to_none=True)
            self._accum_count = 0
            self._has_real_grad_in_window = False

        self.dit.train(False)
        # Restore core_model to eval mode after joint-training forward pass.
        # The LLM is normally kept in eval mode (frozen inference) everywhere else
        # in the training loop.  Leaving it in train mode would affect dropout and
        # batch-norm behavior in subsequent understanding-phase forward passes.
        if self.joint_conditioning_train:
            self.core_model.train(False)
        if (
            torch.cuda.is_available()
            and int(getattr(self.config, "clear_cache_every", 0)) > 0
            and (self.step_id % int(getattr(self.config, "clear_cache_every", 0)) == 0)
        ):
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            gc.collect()

        return {
            "loss": float(loss.detach().item()) if torch.isfinite(loss.detach()).all() else 0.0,
            "did_step": bool(did_step),
            "skipped_reason": skipped_reason,
            "valid_latent_tokens": float(valid_latent_tokens),
            "reward": float(reward) if reward is not None else None,
            "joint_conditioning": bool(self.joint_conditioning_train),
            "lora_enabled": bool(self.dit_lora_enabled),
            "trainable_params": float(self.trainable_param_count),
            "objective": (
                "diffusion_reward_weighted_denoising"
                if self.reward_loss_weight > 0.0 and reward is not None
                else "diffusion_denoising"
            ),
        }
