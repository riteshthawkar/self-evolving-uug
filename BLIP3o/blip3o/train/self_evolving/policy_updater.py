"""
RolePolicyUpdater: KL-regularized REINFORCE updater for a role adapter.
Ported from self_evolving/experiments/understanding.py.

This is the core RL update logic — model-agnostic, only needs:
  model.forward() and use_adapter() context manager.
"""

import gc
import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from .utils import (
    _build_chat_text,
    _clip_grad_norm_multi_device,
    _collect_trainable_params,
    _prepare_mm_inputs,
    use_adapter,
)


class RolePolicyUpdater:
    """
    KL-regularized REINFORCE updater for a role adapter.

    Computes:
        loss = advantage * CE_loss + beta * KL_loss

    with adaptive beta based on KL target.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor,
        config,
        adapter_name: Optional[str],
        reference_model: Optional[torch.nn.Module] = None,
    ):
        self.model = model
        self.processor = processor
        self.config = config
        self.adapter_name = adapter_name
        self.reference_model = reference_model
        self.kl_coef = config.kl_coef
        self.step_id = 0
        self.grad_accum_steps = max(1, getattr(config, "grad_accum_steps", 1))
        self._accum_count = 0

        params = list(_collect_trainable_params(model, adapter_name))
        if not params:
            raise RuntimeError(
                f"No trainable parameters found for adapter={adapter_name!r}"
            )
        self.params = params
        self.opt = torch.optim.AdamW(
            params, lr=config.lr, weight_decay=config.weight_decay
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
        target = max(self.config.kl_target, 1e-8)
        delta = (kl_val - target) / target
        beta = self.kl_coef * math.exp(self.config.kl_adapt_rate * delta)
        beta = max(self.config.kl_min, min(self.config.kl_max, beta))
        self.kl_coef = float(beta)

    def step(
        self,
        image: Image.Image,
        prompt: str,
        completion: str,
        reward: float,
        baseline: float,
        device: torch.device,
    ) -> Dict[str, float]:
        self.step_id += 1

        chat_prompt = _build_chat_text(self.processor, image, prompt)
        chat_full = chat_prompt + completion

        inputs_prompt = _prepare_mm_inputs(
            self.processor, device, image, chat_prompt, model=self.model
        )
        inputs_full = _prepare_mm_inputs(
            self.processor, device, image, chat_full, model=self.model
        )

        input_ids = inputs_full["input_ids"]
        labels = input_ids.clone()
        prompt_len = inputs_prompt["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        valid_mask = labels[:, 1:] != -100

        # For BLIP3o, forward() doesn't accept 'images'/'image_sizes' —
        # those are only for generate().  Filter them out for forward calls.
        _forward_keys_to_drop = ("images", "image_sizes")
        forward_full = {k: v for k, v in inputs_full.items()
                        if k not in _forward_keys_to_drop}

        if torch.cuda.is_available() and getattr(self.config, "clear_cache_every", 0) <= 1:
            torch.cuda.empty_cache()
            gc.collect()

        self.model.train(True)
        policy_inputs = dict(forward_full)
        policy_inputs["labels"] = labels
        # Avoid allocating KV cache during training forwards
        policy_inputs["use_cache"] = False
        def _run_policy_forward():
            with use_adapter(self.model, self.adapter_name):
                return self.model(**policy_inputs)

        try:
            out_pi = _run_policy_forward()
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                out_pi = _run_policy_forward()
            else:
                raise
        ce_loss = out_pi.loss
        ref_inputs = dict(forward_full)
        ref_inputs["use_cache"] = False
        if self.reference_model is not None:
            def _run_ref_forward_ref_model():
                with torch.no_grad():
                    return self.reference_model(**ref_inputs)
            try:
                out_ref = _run_ref_forward_ref_model()
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    out_ref = _run_ref_forward_ref_model()
                else:
                    raise
        else:
            def _run_ref_forward_base_adapter():
                with torch.no_grad():
                    with use_adapter(self.model, None):
                        return self.model(**ref_inputs)
            try:
                out_ref = _run_ref_forward_base_adapter()
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    out_ref = _run_ref_forward_base_adapter()
                else:
                    raise
        if valid_mask.any():
            vocab = out_pi.logits.shape[-1]
            pi_shift = out_pi.logits[:, :-1, :].reshape(-1, vocab)
            ref_shift = out_ref.logits[:, :-1, :].reshape(-1, vocab)
            valid_pos = valid_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)

            # Compute KL only on completion tokens in small chunks to cap peak memory.
            kl_sum = torch.zeros((), device=ce_loss.device, dtype=torch.float32)
            chunk_size = 32
            for chunk in valid_pos.split(chunk_size):
                pi_chunk = pi_shift.index_select(0, chunk)
                ref_chunk = ref_shift.index_select(0, chunk)
                logp_pi_chunk = F.log_softmax(pi_chunk, dim=-1)
                logp_ref_chunk = F.log_softmax(ref_chunk, dim=-1)
                kl_chunk = (logp_pi_chunk.exp() * (logp_pi_chunk - logp_ref_chunk)).sum(dim=-1)
                kl_sum = kl_sum + kl_chunk.float().sum()

            kl_loss = kl_sum / valid_pos.numel()
            kl_loss = kl_loss.to(dtype=ce_loss.dtype)
        else:
            kl_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)

        advantage = float(reward - baseline)
        beta_before = float(self.kl_coef)
        total_loss = advantage * ce_loss + beta_before * kl_loss

        # Gradient accumulation: scale loss and accumulate
        scaled_loss = total_loss / self.grad_accum_steps
        if self._accum_count == 0:
            self.opt.zero_grad(set_to_none=True)
        scaled_loss.backward()
        self._accum_count += 1

        did_step = False
        if self._accum_count >= self.grad_accum_steps:
            _clip_grad_norm_multi_device(self.params, self.config.grad_clip)
            self.opt.step()
            self._accum_count = 0
            did_step = True
        self.model.train(False)

        kl_val = float(kl_loss.item())
        self._adapt_beta(kl_val)

        ce_loss_val = float(ce_loss.item())
        total_loss_val = float(total_loss.item())

        try:
            del inputs_prompt, inputs_full, input_ids, labels, policy_inputs
            del out_pi, out_ref, valid_mask, total_loss, ce_loss
            if "pi_shift" in locals():
                del pi_shift
            if "ref_shift" in locals():
                del ref_shift
            if "valid_pos" in locals():
                del valid_pos
        except Exception:
            pass

        gc.collect()

        if (
            torch.cuda.is_available()
            and self.config.clear_cache_every > 0
            and self.step_id % self.config.clear_cache_every == 0
        ):
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            gc.collect()

        return {
            "ce_loss": ce_loss_val,
            "kl_loss": kl_val,
            "advantage": advantage,
            "kl_coef_before": beta_before,
            "kl_coef_after": float(self.kl_coef),
            "total_loss": total_loss_val,
            "did_step": did_step,
        }
