"""
RolePolicyUpdater: KL-regularized REINFORCE/GRPO updater for a role adapter.

Ported from BLIP3o's policy_updater.py for VARGPT v1.1.
Core RL update logic — model-agnostic, only needs:
  model.forward() and use_adapter() context manager.

Adapted for:
  - Qwen2-VL / VARGPT input format (pixel_values, not images)
  - LLaMA-Factory tokenizer/processor conventions
  - DDP no_sync pattern for multi-backward steps
"""

import contextlib
import gc
import math
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from .utils import (
    _build_chat_text,
    _build_text_only_chat,
    _clip_grad_norm_multi_device,
    _prepare_mm_inputs,
    _prepare_text_only_inputs,
    _unwrap_model,
    use_adapter,
)
from .adapter_manager import use_role, collect_role_params


def _aligned_prompt_prefix_len(
    prompt_ids: torch.Tensor,
    full_ids: torch.Tensor,
    completion_text: str,
    processor=None,
) -> int:
    """Return a robust prompt-prefix length for loss masking.

    In multimodal chat templates (e.g. Qwen2-VL), the prompt-only and
    prompt+completion tokenizations can have different lengths even for
    the same prompt, because vision token expansion may vary. We use
    multiple strategies to find the right boundary:

    1. If prompt_len < full_len: use prompt_len directly (ideal case).
    2. Otherwise: estimate completion token count from text and subtract
       from full_len (handles multimodal token length mismatches).
    3. Fallback to LCP (longest-common-prefix) from the END of the
       sequences, which is more reliable for finding where the
       completion starts.
    """
    prompt_len = int(prompt_ids.shape[1])
    full_len = int(full_ids.shape[1])
    if full_len <= 0:
        return 0

    completion_text = str(completion_text or "").strip()

    # Case 1: prompt shorter than full — the simple/ideal case
    if prompt_len < full_len:
        return prompt_len

    # Case 2: prompt >= full (common with multimodal tokenizers)
    # Estimate completion tokens and subtract from full_len
    if completion_text:
        est_comp = _estimate_completion_token_count(processor, completion_text)
        if est_comp > 0:
            # Add a small margin (2 tokens) for special tokens like </s>
            boundary = max(0, full_len - est_comp - 2)
            # Sanity: don't mask more than 80% of the sequence
            boundary = max(boundary, full_len // 5)
            return boundary

    # Case 3: no completion text → mask everything as prompt
    if not completion_text:
        return min(prompt_len, full_len)

    # Case 4: fallback — match from the END of the sequences backwards
    # to find where they start diverging (i.e., where completion begins)
    p = prompt_ids[0]
    f = full_ids[0]
    common_suffix = 0
    max_check = min(prompt_len, full_len)
    while common_suffix < max_check:
        pi = prompt_len - 1 - common_suffix
        fi = full_len - 1 - common_suffix
        if int(p[pi].item()) == int(f[fi].item()):
            common_suffix += 1
        else:
            break
    # The completion starts where the suffix matching stops
    boundary = max(0, full_len - common_suffix - 1)
    # Sanity: ensure at least 1 completion token
    boundary = min(boundary, full_len - 1)
    return boundary


def _estimate_completion_token_count(processor, completion_text: str) -> int:
    """Best-effort completion token count for mask fallback."""
    text = str(completion_text or "").strip()
    if not text:
        return 0
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        tok = processor
    if tok is not None and hasattr(tok, "encode"):
        try:
            ids = tok.encode(text, add_special_tokens=False)
            count = int(len(ids))
            if count > 0:
                return count
        except TypeError:
            try:
                ids = tok.encode(text)
                count = int(len(ids))
                if count > 0:
                    return count
            except Exception:
                pass
        except Exception:
            pass
    # Heuristic fallback: ~1.3 tokens per word for English text.
    word_count = len(text.split())
    return max(1, int(word_count * 1.3))


class RolePolicyUpdater:
    """KL-regularized REINFORCE updater for a role adapter (proposer/solver).

    Computes:
        loss = advantage * CE_loss + beta * KL_loss

    CE_loss is negative log-likelihood. A positive advantage should reduce CE
    and increase completion probability; a negative advantage should increase CE.

    with adaptive beta based on KL target.

    Parameters
    ----------
    model : torch.nn.Module
        The VARGPT model (possibly DDP-wrapped).
    processor : processor
        Tokenizer / processor for text + image preprocessing.
    config : SelfEvolvingConfig
        Training configuration.
    adapter_name : str
        Name of the LoRA adapter ("proposer", "solver", or "default").
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
        self._has_real_grad_in_window = False

        # Collect trainable params for this role
        is_generator = adapter_name in ("default", "generator")
        self.params = collect_role_params(
            model, adapter_name or "default",
            include_generator_modules=is_generator,
        )
        if not self.params:
            raise RuntimeError(
                f"No trainable parameters found for adapter={adapter_name!r}"
            )
        self.opt = torch.optim.AdamW(
            self.params, lr=config.lr, weight_decay=config.weight_decay
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
        """Adaptive KL coefficient based on target KL divergence."""
        target = max(self.config.kl_target, 1e-8)
        delta = (kl_val - target) / target
        beta = self.kl_coef * math.exp(self.config.kl_adapt_rate * delta)
        beta = max(self.config.kl_min, min(self.config.kl_max, beta))
        self.kl_coef = float(beta)

    def step(
        self,
        image,  # Image.Image or None (for imageless proposer mode)
        prompt: str,
        completion: str,
        reward: float,
        baseline: float,
        device: torch.device,
        ddp_no_sync: bool = False,
    ) -> Dict[str, float]:
        """Execute one REINFORCE update step.

        Parameters
        ----------
        image : Image.Image or None
            Input image (None for imageless proposer).
        prompt : str
            The prompt text.
        completion : str
            The model's completion text.
        reward : float
            Scalar reward for this (prompt, completion) pair.
        baseline : float
            EMA baseline for advantage computation.
        device : torch.device
            Target device.
        ddp_no_sync : bool
            If True, wrap backward in model.no_sync() and manually allreduce.

        Returns
        -------
        Dict with training metrics.
        """
        self.step_id += 1

        # ── Prepare inputs ──────────────────────────────────────────────
        if image is not None:
            chat_prompt = _build_chat_text(self.processor, image, prompt)
            chat_full = chat_prompt + completion

            inputs_prompt = _prepare_mm_inputs(
                self.processor, device, image, chat_prompt, model=self.model
            )
            inputs_full = _prepare_mm_inputs(
                self.processor, device, image, chat_full, model=self.model
            )
        else:
            chat_prompt = _build_text_only_chat(self.processor, prompt)
            chat_full = chat_prompt + completion

            inputs_prompt = _prepare_text_only_inputs(
                self.processor, device, chat_prompt,
            )
            inputs_full = _prepare_text_only_inputs(
                self.processor, device, chat_full,
            )

        input_ids = inputs_full["input_ids"]
        labels = input_ids.clone()
        prompt_len = _aligned_prompt_prefix_len(
            inputs_prompt["input_ids"],
            input_ids,
            completion_text=completion,
            processor=self.processor,
        )
        labels[:, :prompt_len] = -100
        valid_mask = labels[:, 1:] != -100
        valid_token_count = int(valid_mask.sum().item())
        forced_tail_tokens = 0

        if valid_token_count <= 0 and str(completion or "").strip():
            full_len = int(input_ids.shape[1])
            prompt_raw_len = int(inputs_prompt["input_ids"].shape[1])
            tail_from_len_delta = max(0, full_len - min(prompt_raw_len, full_len))
            tail_from_completion = _estimate_completion_token_count(self.processor, completion)
            forced_tail_tokens = max(tail_from_len_delta, tail_from_completion)
            forced_tail_tokens = max(0, min(forced_tail_tokens, max(0, full_len - 1)))
            if forced_tail_tokens > 0:
                labels[:, : full_len - forced_tail_tokens] = -100
                valid_mask = labels[:, 1:] != -100
                valid_token_count = int(valid_mask.sum().item())

        # Do NOT return early on skip — all DDP ranks must call forward/backward.
        skip_backward = valid_token_count <= 0

        # For VARGPT forward, filter out generation-specific keys
        _forward_keys_to_drop = (
            "pixel_gen_values", "inference_image_gen", "dpo_training",
        )
        forward_full = {k: v for k, v in inputs_full.items()
                        if k not in _forward_keys_to_drop}

        if torch.cuda.is_available() and getattr(self.config, "clear_cache_every", 0) <= 1:
            torch.cuda.empty_cache()
            gc.collect()

        # ── Reference forward (no grad) ─────────────────────────────────
        ref_inputs = dict(forward_full)
        ref_inputs["use_cache"] = False

        if self.reference_model is not None:
            with torch.no_grad():
                out_ref = self.reference_model(**ref_inputs)
        else:
            ref_model = _unwrap_model(self.model)
            was_training = bool(getattr(ref_model, "training", False))
            try:
                ref_model.eval()
                with torch.no_grad():
                    with use_adapter(ref_model, None):
                        out_ref = ref_model(**ref_inputs)
            finally:
                if was_training:
                    ref_model.train(True)

        # ── Policy forward (with grad) ──────────────────────────────────
        self.model.train(True)
        policy_inputs = dict(forward_full)
        policy_inputs["labels"] = labels
        policy_inputs["use_cache"] = False

        with use_adapter(self.model, self.adapter_name):
            out_pi = self.model(**policy_inputs)

        ce_loss = out_pi.loss

        if not bool(torch.isfinite(ce_loss.detach()).all().item()):
            skip_backward = True

        # ── KL divergence on completion tokens ──────────────────────────
        if not skip_backward and valid_mask.any():
            vocab = out_pi.logits.shape[-1]
            pi_shift = out_pi.logits[:, :-1, :].reshape(-1, vocab)
            ref_shift = out_ref.logits[:, :-1, :].reshape(-1, vocab)
            valid_pos = valid_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)

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

        # ── Compute advantage and total loss ────────────────────────────
        advantage = float(reward - baseline)
        beta_before = float(self.kl_coef)

        if skip_backward:
            total_loss = (out_pi.logits.sum() * 0.0)
            if not torch.isfinite(total_loss):
                total_loss = torch.zeros(
                    (), device=out_pi.logits.device,
                    dtype=out_pi.logits.dtype, requires_grad=True,
                )
            skipped_reason = (
                "no_valid_completion_tokens" if valid_token_count <= 0
                else "non_finite_ce_loss"
            )
        else:
            total_loss = advantage * ce_loss + beta_before * kl_loss
            skipped_reason = None
            if not bool(torch.isfinite(total_loss.detach()).all().item()):
                total_loss = out_pi.logits.sum() * 0.0
                skipped_reason = "non_finite_total_loss"

        has_real_grad = skipped_reason is None
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor(
                [1 if has_real_grad else 0],
                dtype=torch.int32,
                device=total_loss.device,
            )
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            has_real_grad = bool(int(t.item()) == 1)

        # ── Backward with gradient accumulation ─────────────────────────
        scaled_loss = total_loss / self.grad_accum_steps
        if self._accum_count == 0:
            self.opt.zero_grad(set_to_none=True)
            self._has_real_grad_in_window = False

        # Restore adapter for backward
        model_ref = _unwrap_model(self.model)
        restore_adapter = None
        if self.adapter_name is not None and hasattr(model_ref, "set_adapter"):
            restore_adapter = getattr(model_ref, "active_adapter", None)
            if isinstance(restore_adapter, (list, tuple)):
                restore_adapter = restore_adapter[0] if restore_adapter else None
            try:
                model_ref.set_adapter(self.adapter_name)
            except Exception:
                restore_adapter = None

        # DDP no_sync for multi-backward scenarios
        _no_sync_ctx = (
            self.model.no_sync()
            if (ddp_no_sync and hasattr(self.model, "no_sync"))
            else contextlib.nullcontext()
        )
        try:
            with _no_sync_ctx:
                scaled_loss.backward()
        finally:
            if restore_adapter is not None:
                try:
                    model_ref.set_adapter(restore_adapter)
                except Exception:
                    pass

        self._accum_count += 1
        if has_real_grad:
            self._has_real_grad_in_window = True

        did_step = False
        if self._accum_count >= self.grad_accum_steps:
            # Manual gradient sync when DDP allreduce was skipped
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

        self.model.train(False)

        # ── Metrics ─────────────────────────────────────────────────────
        if skipped_reason:
            return {
                "ce_loss": float(ce_loss.detach().item()) if torch.isfinite(ce_loss.detach()).all() else float("nan"),
                "kl_loss": float(kl_loss.detach().item()),
                "advantage": advantage,
                "kl_coef_before": beta_before,
                "kl_coef_after": float(self.kl_coef),
                "total_loss": 0.0,
                "did_step": bool(did_step),
                "skipped_reason": skipped_reason,
                "valid_token_count": float(valid_token_count),
                "forced_tail_tokens": float(forced_tail_tokens),
            }

        kl_val = float(kl_loss.item())
        self._adapt_beta(kl_val)

        ce_loss_val = float(ce_loss.item())
        total_loss_val = float(total_loss.item())

        try:
            del inputs_prompt, inputs_full, input_ids, labels, policy_inputs
            del out_pi, out_ref, valid_mask, total_loss, ce_loss
        except Exception:
            pass

        gc.collect()
        if (
            torch.cuda.is_available()
            and self.config.clear_cache_every > 0
            and self.step_id % self.config.clear_cache_every == 0
        ):
            torch.cuda.empty_cache()
            gc.collect()

        return {
            "ce_loss": ce_loss_val,
            "kl_loss": kl_val,
            "advantage": advantage,
            "kl_coef_before": beta_before,
            "kl_coef_after": float(self.kl_coef),
            "total_loss": total_loss_val,
            "did_step": did_step,
            "valid_token_count": float(valid_token_count),
            "forced_tail_tokens": float(forced_tail_tokens),
        }
