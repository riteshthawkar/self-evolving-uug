"""DiT denoising updater for generation/unified self-evolving training.

This updater applies an SFT-style denoising objective directly on BLIP3o's
diffusion transformer (DiT) using real source images and generation prompts.
"""

import gc
import math
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
        self.noise_scheduler = getattr(self.core_model, "noise_scheduler", None)
        self.gen_vision_tower = getattr(self.core_model, "get_gen_vision_tower", lambda: None)()
        if self.dit is None:
            raise RuntimeError("DiT updater requires core model to expose `dit`.")
        if self.noise_scheduler is None:
            raise RuntimeError("DiT updater requires core model to expose `noise_scheduler`.")
        if self.gen_vision_tower is None:
            raise RuntimeError("DiT updater requires a generation vision tower (`get_gen_vision_tower`).")

        for p in self.dit.parameters():
            p.requires_grad_(True)
        self.params = [p for p in self.dit.parameters() if p.requires_grad]
        if not self.params:
            raise RuntimeError("No trainable DiT parameters found.")

        lr = float(getattr(config, "dit_lr", getattr(config, "lr", 1e-6)))
        weight_decay = float(getattr(config, "dit_weight_decay", getattr(config, "weight_decay", 0.01)))
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=weight_decay)

        self.distributed = bool(dist.is_available() and dist.is_initialized())
        self.world_size = int(dist.get_world_size()) if self.distributed else 1

    def state_dict(self) -> Dict:
        return {
            "optimizer": self.opt.state_dict(),
            "step_id": int(self.step_id),
            "accum_count": int(self._accum_count),
            "has_real_grad_in_window": bool(self._has_real_grad_in_window),
        }

    def load_state_dict(self, state: Dict):
        if not isinstance(state, dict):
            return
        if "optimizer" in state and isinstance(state.get("optimizer"), dict):
            self.opt.load_state_dict(state["optimizer"])
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
        for param in self.params:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= float(self.world_size)

    def _tokenizer(self):
        tok = getattr(self.processor, "tokenizer", None)
        return tok if tok is not None else self.processor

    def _prepare_image_tensor(self, image: Image.Image, device: torch.device) -> torch.Tensor:
        image_processor = getattr(self.gen_vision_tower, "image_processor", None)
        if image_processor is None:
            raise RuntimeError("Generation vision tower does not expose image_processor.")
        model_cfg = getattr(self.core_model, "config", None)

        if _process_images_fn is not None and model_cfg is not None:
            pixel_values = _process_images_fn([image], image_processor, model_cfg)
        else:
            proc_out = image_processor(images=[image], return_tensors="pt")
            if isinstance(proc_out, dict):
                pixel_values = proc_out.get("pixel_values")
            else:
                pixel_values = getattr(proc_out, "pixel_values", None)
            if pixel_values is None:
                raise RuntimeError("Failed to preprocess image for generation vision tower.")

        if not torch.is_tensor(pixel_values):
            raise RuntimeError("Image preprocessing did not return a tensor.")

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

    def _prepare_latents(self, image: Image.Image) -> Tuple[torch.Tensor, torch.device, torch.dtype]:
        model_device = self.params[0].device
        model_dtype = self.params[0].dtype
        gen_images = self._prepare_image_tensor(image=image, device=model_device)

        with torch.no_grad():
            latents = self.gen_vision_tower(gen_images)
            get_pooling = getattr(self.model_ref, "get_gen_pooling", None)
            pool_img = getattr(self.model_ref, "pool_img", None)
            pooling = str(get_pooling() if callable(get_pooling) else "")
            if "early" in pooling and callable(pool_img) and latents.ndim == 3:
                latents = pool_img(latents)

        dit_cfg = getattr(self.dit, "config", None)
        in_channels = int(getattr(dit_cfg, "in_channels", 1792))
        input_size = int(getattr(dit_cfg, "input_size", 8))
        if (
            latents.ndim == 3
            and latents.shape[-1] != in_channels
            and latents.shape[1] != in_channels
        ):
            down_projector = getattr(self.core_model, "down_projector", None)
            if down_projector is not None:
                proj_dtype = latents.dtype
                try:
                    proj_dtype = next(down_projector.parameters()).dtype
                except Exception:
                    pass
                with torch.no_grad():
                    latents = down_projector(latents.to(proj_dtype))
        latents = self._reshape_latents(
            latents=latents,
            in_channels=in_channels,
            input_size=input_size,
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

        with torch.no_grad():
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

        expected_dim = int(getattr(getattr(self.dit, "config", None), "latent_embedding_size", z_latents.shape[-1]))
        if z_latents.shape[-1] != expected_dim:
            down_projector = getattr(self.core_model, "down_projector", None)
            if down_projector is not None:
                proj_dtype = z_latents.dtype
                try:
                    proj_dtype = next(down_projector.parameters()).dtype
                except Exception:
                    pass
                with torch.no_grad():
                    z_latents = down_projector(z_latents.to(proj_dtype)).to(device=device, dtype=dtype)
        if z_latents.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Conditioning width mismatch for DiT: got {z_latents.shape[-1]}, expected {expected_dim}."
            )
        return z_latents

    def _sample_training_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
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
        for p in self.params:
            anchor = anchor + (p.sum() * 0.0)
        return anchor

    def step(
        self,
        *,
        image: Image.Image,
        prompt: str,
        device: torch.device,
    ) -> Dict[str, object]:
        self.step_id += 1
        self.dit.train(True)

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
            local_skip_reason = f"prepare_failed:{type(exc).__name__}"

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

            noise_pred = self.dit(
                x=noisy_latents,
                timestep=timesteps,
                z_latents=z_latents,
            )
            target = noise - latents
            mse = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
            loss = mse.to(dtype=model_dtype) * self.loss_weight
            valid_latent_tokens = float(latents.shape[-1] * latents.shape[-2])

            local_finite = bool(torch.isfinite(loss.detach()).all().item())
            finite_all = self._dist_all_bool(local_finite)
            if not finite_all:
                skipped_reason = "non_finite_dit_loss"
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
                _clip_grad_norm_multi_device(self.params, self.grad_clip)
                self.opt.step()
                did_step = True
            self.opt.zero_grad(set_to_none=True)
            self._accum_count = 0
            self._has_real_grad_in_window = False

        self.dit.train(False)
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
        }
