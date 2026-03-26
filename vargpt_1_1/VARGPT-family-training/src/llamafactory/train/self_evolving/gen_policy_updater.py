"""
VARImageGenPolicyUpdater: GRPO on discrete image tokens for VARGPT v1.1.

This is the novel component for the VARGPT self-evolving framework.
Unlike BLIP3o (which uses diffusion with continuous latents), VARGPT
generates images as discrete tokens via the Infinity VAR model.

The key insight: since image generation is autoregressive over discrete
tokens (vocab=64, BSQ codebook), we can compute per-token log-probabilities
and apply GRPO exactly as for text generation.

Flow:
  1. Generate K candidate images (inference, no grad)
  2. Score each candidate (CLIP similarity, solver verification, etc.)
  3. For GRPO update (with grad):
     - Forward each candidate through the model with pixel_gen_values
     - Extract gen_logits (B, L, V=64) and image_gen_labels (B, L)
     - Compute per-token log-probs
     - Apply group-normalized advantages with importance ratio clipping
"""

import contextlib
import gc
import logging
import math
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from .utils import (
    _clip_grad_norm_multi_device,
    _unwrap_model,
    use_adapter,
)
from .adapter_manager import (
    ROLE_GENERATOR,
    use_role,
    use_base_model,
    collect_role_params,
)

logger = logging.getLogger(__name__)


class VARImageGenPolicyUpdater:
    """GRPO updater for VARGPT image generation using discrete VAR tokens.

    Implements Group Relative Policy Optimization (GRPO) where the policy
    generates discrete image tokens autoregressively:
      - Generate K candidate images
      - Score them with external reward signals
      - Compute group-normalized advantages
      - Update the generator LoRA + vargpt_gen + projector via policy gradient

    Parameters
    ----------
    model : torch.nn.Module
        VARGPT model (possibly DDP-wrapped).
    tokenizer : tokenizer
        Tokenizer for text input preparation.
    config : SelfEvolvingConfig
        Training configuration.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.kl_coef = config.kl_coef
        self.step_id = 0
        self.grad_accum_steps = max(1, getattr(config, "grad_accum_steps", 1))
        self._accum_count = 0
        self._has_real_grad_in_window = False

        # Collect trainable params: LoRA + vargpt_gen + image_gen_projector
        self.params = collect_role_params(
            model, ROLE_GENERATOR, include_generator_modules=True
        )
        if not self.params:
            raise RuntimeError(
                "No trainable parameters found for generator role. "
                "Ensure the model has the 'default' adapter and vargpt_gen modules."
            )
        self.opt = torch.optim.AdamW(
            self.params, lr=config.lr, weight_decay=config.weight_decay
        )

        logger.info(
            f"[VARImageGenPolicyUpdater] Initialized with {len(self.params)} params"
        )

    def state_dict(self) -> Dict:
        return {
            "optimizer": self.opt.state_dict(),
            "kl_coef": float(self.kl_coef),
            "step_id": int(self.step_id),
        }

    def load_state_dict(self, state: Dict):
        if not isinstance(state, dict):
            return
        if "optimizer" in state:
            self.opt.load_state_dict(state["optimizer"])
        if "kl_coef" in state:
            self.kl_coef = float(state["kl_coef"])
        if "step_id" in state:
            self.step_id = int(state["step_id"])

    def _adapt_beta(self, kl_val: float):
        """Adaptive KL coefficient."""
        target = max(self.config.kl_target, 1e-8)
        delta = (kl_val - target) / target
        beta = self.kl_coef * math.exp(self.config.kl_adapt_rate * delta)
        beta = max(self.config.kl_min, min(self.config.kl_max, beta))
        self.kl_coef = float(beta)

    def _infer_required_gen_pad_tokens(
        self,
        base_model: torch.nn.Module,
        pixel_gen_values,
    ) -> Optional[int]:
        """Infer required number of <|image_gen_pad|> tokens for current sample.

        Mirrors VARGPT v1.1 sequence construction:
          x_BLC = [sos] + word_embed(x_BLC_wo_prefix)
          need_to_pad = ceil(len(x_BLC)/pad_to_multiplier)*pad_to_multiplier - len(x_BLC)
          n_image_gen_features = B * (len(x_BLC) + need_to_pad)
        """
        try:
            coerce = getattr(base_model, "_coerce_pixel_gen_values", None)
            if callable(coerce):
                pv = coerce(pixel_gen_values)
            else:
                if torch.is_tensor(pixel_gen_values):
                    pv = pixel_gen_values
                elif isinstance(pixel_gen_values, Sequence):
                    flat = []
                    for x in pixel_gen_values:
                        if torch.is_tensor(x):
                            flat.append(x)
                        elif isinstance(x, Sequence):
                            flat.extend([y for y in x if torch.is_tensor(y)])
                    if not flat:
                        return None
                    pv = torch.cat(
                        [t if t.ndim == 4 else t.unsqueeze(0) for t in flat],
                        dim=0,
                    )
                else:
                    return None

            if pv.ndim == 4:
                bsz = int(pv.shape[0])
                t_frames = 1
                h, w = int(pv.shape[-2]), int(pv.shape[-1])
            elif pv.ndim == 5:
                bsz = int(pv.shape[0])
                t_frames = int(pv.shape[2])
                h, w = int(pv.shape[-2]), int(pv.shape[-1])
            else:
                return None

            from visionllm.vargpt_qwen_v1_1.var_model.infinity.utils.dynamic_resolution import dynamic_resolution_h_w

            hw_ratio = float(h) / float(max(1, w))
            ratio_keys = list(dynamic_resolution_h_w.keys())
            if not ratio_keys:
                return None
            ratio_key = min(ratio_keys, key=lambda k: abs(float(k) - hw_ratio))

            vargpt_args = getattr(base_model, "vargpt_gen_args", None)
            pn = getattr(vargpt_args, "pn", "0.25M")
            if pn not in dynamic_resolution_h_w[ratio_key]:
                return None
            scale_schedule = dynamic_resolution_h_w[ratio_key][pn]["scales"]
            scale_schedule = [
                (min(int(pt), int(t_frames // 4 + 1)), int(ph), int(pw))
                for (pt, ph, pw) in scale_schedule
            ]
            if not scale_schedule:
                return None

            # Matches get_vae_gt_xin_v1_1(training_scales=100)
            scale_schedule = scale_schedule[:100]
            training_seq_len = sum(int(pt * ph * pw) for (pt, ph, pw) in scale_schedule)
            first_scale_len = int(scale_schedule[0][0] * scale_schedule[0][1] * scale_schedule[0][2])
            x_blc_wo_prefix_len = max(0, training_seq_len - first_scale_len)
            l_end = int(x_blc_wo_prefix_len + 1)  # +sos

            pad_mult = int(getattr(getattr(base_model, "vargpt_gen", None), "pad_to_multiplier", 1) or 1)
            padded_len = ((l_end + pad_mult - 1) // pad_mult) * pad_mult
            return int(bsz * padded_len)
        except Exception:
            return None

    def _append_explicit_image_gen_tokens(
        self,
        base_model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        required_pad_tokens: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Append explicit image-gen token segment to input sequence.

        Segment format per sample: <|image_gen_start|> + N*<|image_gen_pad|> + <|image_gen_end|>
        """
        if required_pad_tokens is None or required_pad_tokens <= 0:
            return input_ids, attention_mask, labels

        special_tokens = getattr(getattr(base_model, "config", None), "special_tokens", None)
        if not isinstance(special_tokens, dict):
            return input_ids, attention_mask, labels

        start_id = special_tokens.get("image_gen_start_token_id", None)
        pad_id = special_tokens.get("image_gen_pad_token_id", None)
        end_id = special_tokens.get("image_gen_end_token_id", None)
        if pad_id is None:
            return input_ids, attention_mask, labels

        bsz = int(input_ids.shape[0])
        if bsz != 1:
            # Current self-evolving G-step uses batch=1.
            # Keep conservative behavior for unexpected batching.
            return input_ids, attention_mask, labels

        cur_pad = int((input_ids == int(pad_id)).sum().item())
        add_pad = max(0, int(required_pad_tokens - cur_pad))
        if add_pad == 0:
            return input_ids, attention_mask, labels

        append_ids = []
        has_start = (start_id is not None) and int((input_ids == int(start_id)).sum().item()) > 0
        has_end = (end_id is not None) and int((input_ids == int(end_id)).sum().item()) > 0
        if (start_id is not None) and (not has_start):
            append_ids.append(int(start_id))
        append_ids.extend([int(pad_id)] * add_pad)
        if (end_id is not None) and (not has_end):
            append_ids.append(int(end_id))

        if not append_ids:
            return input_ids, attention_mask, labels

        append_tensor = torch.tensor(
            append_ids, dtype=input_ids.dtype, device=input_ids.device
        ).unsqueeze(0)
        input_ids = torch.cat([input_ids, append_tensor], dim=1)

        if attention_mask is not None:
            append_mask = torch.ones(
                (1, append_tensor.shape[1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask, append_mask], dim=1)

        if labels is not None:
            append_labels = torch.full(
                (1, append_tensor.shape[1]),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([labels, append_labels], dim=1)

        return input_ids, attention_mask, labels

    def _compute_gen_log_probs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_gen_values: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        use_grad: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        """Forward pass to get per-sample log-probabilities of image generation.

        The VARGPT model's forward() with pixel_gen_values triggers the v1.1
        image generation path (line 2456 of modeling_vargpt_qwen2_vl.py):
          1. LLM backbone produces hidden states
          2. Hidden states at image_gen_pad positions → projector → VAR decoder
          3. VAR decoder returns gen_logits (B, L, V=64)
          4. Labels from VAE encoding of the target image

        Returns
        -------
        seq_log_probs : Tensor (B,)
            Per-sample sequence log-probabilities.
        gen_logits : Tensor (B, L, V) or None
            Raw logits (only when use_grad=True).
        stats : dict
            Statistics for logging.
        """
        base_model = _unwrap_model(model)

        required_pad = self._infer_required_gen_pad_tokens(base_model, pixel_gen_values)
        input_ids, attention_mask, labels = self._append_explicit_image_gen_tokens(
            base_model=base_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            required_pad_tokens=required_pad,
        )

        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_gen_values": pixel_gen_values,
            "dpo_training": True,  # Enables gen loss path even in eval mode
            "use_cache": False,
        }
        if labels is not None:
            forward_kwargs["labels"] = labels

        ctx = torch.enable_grad() if use_grad else torch.no_grad()
        with ctx:
            outputs = model(**forward_kwargs)

        gen_logits = getattr(outputs, "gen_logits", None)
        gen_labels = getattr(outputs, "image_gen_labels", None)

        if gen_logits is None or gen_labels is None:
            # No image generation happened (e.g., no pixel_gen_values matched)
            dummy = torch.zeros(1, device=input_ids.device)
            return dummy, None, {"gen_log_prob": 0.0, "gen_tokens": 0}

        # Handle bit_label mode (V*2 logits for binary classification)
        vargpt_gen_args = getattr(base_model, "vargpt_gen_args", None)
        use_bit_label = (
            vargpt_gen_args is not None
            and getattr(vargpt_gen_args, "use_bit_label", False)
        )

        B = gen_logits.shape[0]

        if use_bit_label:
            # gen_logits shape: (B, L, V*2) → (B, L, V, 2)
            # gen_labels shape: (B, L) with multi-label binary encoding
            tmp_bs, tmp_seq_len, tmp_channel = gen_logits.shape
            reshaped = gen_logits.reshape(tmp_bs, tmp_seq_len, -1, 2).float()
            # Per-bit log-probs: (B, L, V, 2) → log_softmax → gather
            log_probs = F.log_softmax(reshaped, dim=-1)
            # Sum across bits for each position
            # gen_labels for bit mode is (B, L, V) or similar
            # Simplified: use the loss from the model directly
            if outputs.loss is not None:
                seq_log_probs = -outputs.loss.unsqueeze(0).expand(B)
            else:
                seq_log_probs = torch.zeros(B, device=gen_logits.device)
        else:
            # gen_logits shape: (B, L, V=64) — standard categorical
            V = gen_logits.shape[-1]
            gen_logits_flat = gen_logits.float()
            log_probs = F.log_softmax(gen_logits_flat, dim=-1)  # (B, L, V)

            # Gather log-probs at ground truth positions
            gen_labels_safe = gen_labels.clamp(0, V - 1)  # safety clamp
            token_log_probs = log_probs.gather(
                -1, gen_labels_safe.unsqueeze(-1)
            ).squeeze(-1)  # (B, L)

            # Apply scale-aware weighting if available
            scale_schedule = getattr(base_model, "_last_scale_schedule", None)
            reweight = (
                vargpt_gen_args is not None
                and getattr(vargpt_gen_args, "reweight_loss_by_scale", False)
                and scale_schedule is not None
            )
            if reweight:
                try:
                    training_scales = getattr(base_model, "_last_training_scales", None)
                    lw = []
                    last_scale_area = np.sqrt(np.array(scale_schedule[-1]).prod())
                    max_scales = training_scales or len(scale_schedule)
                    for (pt, ph, pw) in scale_schedule[:max_scales]:
                        this_scale_area = np.sqrt(pt * ph * pw)
                        weight = last_scale_area / this_scale_area
                        lw.extend([weight for _ in range(pt * ph * pw)])
                    lw_tensor = torch.tensor(
                        lw, device=token_log_probs.device,
                        dtype=token_log_probs.dtype,
                    )[None, :]  # (1, L)
                    # Truncate/pad to match actual sequence length
                    L = token_log_probs.shape[1]
                    if lw_tensor.shape[1] > L:
                        lw_tensor = lw_tensor[:, :L]
                    elif lw_tensor.shape[1] < L:
                        pad = torch.ones(
                            1, L - lw_tensor.shape[1],
                            device=lw_tensor.device, dtype=lw_tensor.dtype,
                        )
                        lw_tensor = torch.cat([lw_tensor, pad], dim=1)
                    lw_tensor = lw_tensor / lw_tensor.sum(dim=-1, keepdim=True)
                    seq_log_probs = (token_log_probs * lw_tensor).sum(dim=-1)  # (B,)
                except Exception:
                    seq_log_probs = token_log_probs.mean(dim=-1)
            else:
                seq_log_probs = token_log_probs.mean(dim=-1)  # (B,)

        stats = {
            "gen_log_prob": float(seq_log_probs.mean().item()),
            "gen_tokens": int(gen_logits.shape[1]),
        }

        return seq_log_probs, gen_logits if use_grad else None, stats

    def step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_gen_values_list: List[List[torch.Tensor]],
        rewards: List[float],
        device: torch.device,
        ddp_no_sync: bool = False,
    ) -> Dict[str, float]:
        """Execute one GRPO update step on K candidate images.

        Parameters
        ----------
        input_ids : Tensor (1, S)
            Text input ids (prompt for image generation).
        attention_mask : Tensor (1, S)
            Attention mask.
        labels : Tensor (1, S)
            Labels for text loss (prompt tokens masked with -100).
        pixel_gen_values_list : List[List[Tensor]]
            K candidate images, each as [img_tensor] for pixel_gen_values.
        rewards : List[float]
            K scalar rewards, one per candidate.
        device : torch.device
            Target device.
        ddp_no_sync : bool
            If True, use no_sync for DDP compatibility.

        Returns
        -------
        Dict with training metrics.
        """
        self.step_id += 1
        K = len(rewards)

        if K == 0:
            return {"gen_grpo_loss": 0.0, "gen_grpo_skipped": True}

        # ── Group-normalize rewards (GRPO advantages) ────────────────────
        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        r_mean = r_tensor.mean()
        r_std = r_tensor.std()

        if r_std < self.config.grpo_min_group_std:
            logger.info(
                f"[VARImageGenPolicyUpdater] Skipping GRPO: reward std={r_std:.4f} "
                f"< threshold={self.config.grpo_min_group_std}"
            )
            return {
                "gen_grpo_loss": 0.0,
                "gen_grpo_skipped": True,
                "gen_reward_mean": float(r_mean),
                "gen_reward_std": float(r_std),
            }

        advantages = ((r_tensor - r_mean) / (r_std + 1e-8)).tolist()

        # ── Compute reference log-probs (no grad, base model) ────────────
        ref_log_probs = []
        with torch.no_grad():
            base_model = _unwrap_model(self.model)
            was_training = bool(getattr(base_model, "training", False))
            try:
                base_model.eval()
                with use_adapter(base_model, None):
                    for k in range(K):
                        lp, _, _ = self._compute_gen_log_probs(
                            base_model,
                            input_ids.to(device),
                            attention_mask.to(device),
                            pixel_gen_values_list[k],
                            labels=labels.to(device),
                            use_grad=False,
                        )
                        ref_log_probs.append(lp.detach())
            finally:
                if was_training:
                    base_model.train(True)

        # ── Compute policy log-probs (with grad) and GRPO loss ───────────
        self.model.train(True)
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_kl = 0.0
        total_pg = 0.0

        if self._accum_count == 0:
            self.opt.zero_grad(set_to_none=True)
            self._has_real_grad_in_window = False

        model_ref = _unwrap_model(self.model)
        if hasattr(model_ref, "set_adapter"):
            model_ref.set_adapter(ROLE_GENERATOR)

        for k in range(K):
            adv = advantages[k]

            pi_lp, _, _ = self._compute_gen_log_probs(
                self.model,
                input_ids.to(device),
                attention_mask.to(device),
                pixel_gen_values_list[k],
                labels=labels.to(device),
                use_grad=True,
            )

            ref_lp = ref_log_probs[k].to(device)

            # Importance ratio (log space for stability)
            log_ratio = pi_lp - ref_lp
            ratio = log_ratio.exp().clamp(max=10.0)

            # Clipped surrogate loss (PPO-style)
            eps = self.config.grpo_clip_ratio
            surr1 = -adv * ratio
            surr2 = -adv * ratio.clamp(1.0 - eps, 1.0 + eps)
            pg_loss = torch.max(surr1, surr2).mean()

            # KL penalty
            kl = log_ratio.clamp(min=0.0).mean()

            candidate_loss = pg_loss + self.kl_coef * kl
            total_loss = total_loss + candidate_loss / K

            total_kl += float(kl.detach().item())
            total_pg += float(pg_loss.detach().item())

        # ── Backward ────────────────────────────────────────────────────
        scaled_loss = total_loss / self.grad_accum_steps

        _no_sync_ctx = (
            self.model.no_sync()
            if (ddp_no_sync and hasattr(self.model, "no_sync"))
            else contextlib.nullcontext()
        )
        try:
            with _no_sync_ctx:
                scaled_loss.backward()
        except Exception as e:
            logger.warning(f"[VARImageGenPolicyUpdater] Backward failed: {e}")
            if ddp_no_sync and dist.is_available() and dist.is_initialized():
                raise RuntimeError(
                    f"[VARImageGenPolicyUpdater] Backward failed under DDP at step {self.step_id}: {e}"
                ) from e
            return {
                "gen_grpo_loss": 0.0,
                "gen_grpo_skipped": True,
                "gen_grpo_error": str(e),
            }

        self._accum_count += 1
        self._has_real_grad_in_window = True

        did_step = False
        if self._accum_count >= self.grad_accum_steps:
            # Manual gradient sync for DDP no_sync
            if ddp_no_sync and dist.is_available() and dist.is_initialized():
                for p in self.params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= float(dist.get_world_size())

            if self._has_real_grad_in_window:
                _clip_grad_norm_multi_device(self.params, self.config.grad_clip)
                self.opt.step()
                did_step = True
            else:
                self.opt.zero_grad(set_to_none=True)
            self._accum_count = 0
            self._has_real_grad_in_window = False

        # Adapt KL coefficient
        avg_kl = total_kl / max(K, 1)
        self._adapt_beta(avg_kl)

        # ── Cleanup ─────────────────────────────────────────────────────
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "gen_grpo_loss": float(total_loss.detach().item()),
            "gen_grpo_pg_loss": total_pg / max(K, 1),
            "gen_grpo_kl": avg_kl,
            "gen_grpo_kl_coef": float(self.kl_coef),
            "gen_grpo_did_step": did_step,
            "gen_reward_mean": float(r_mean),
            "gen_reward_std": float(r_std),
            "gen_grpo_skipped": False,
            "gen_grpo_K": K,
        }
