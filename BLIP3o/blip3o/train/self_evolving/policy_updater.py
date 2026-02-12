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
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from .utils import (
    _build_chat_text,
    _clip_grad_norm_multi_device,
    _collect_trainable_params,
    _prepare_mm_inputs,
    _unwrap_model,
    use_adapter,
)


def _aligned_prompt_prefix_len(
    prompt_ids: torch.Tensor,
    full_ids: torch.Tensor,
    completion_text: str,
) -> int:
    """Return a robust prompt-prefix length for loss masking.

    In multimodal chat templates, ``prompt`` and ``prompt+completion`` can be
    serialized with slightly different control-token layouts. In that case,
    using plain ``len(prompt_ids)`` can incorrectly mask all completion tokens.
    We recover a stable boundary by falling back to longest-common-prefix when
    completion text is non-empty.
    """
    prompt_len = int(prompt_ids.shape[1])
    full_len = int(full_ids.shape[1])
    if full_len <= 0:
        return 0
    if prompt_len < full_len:
        return prompt_len
    if not str(completion_text or "").strip():
        return min(prompt_len, full_len)

    max_len = min(prompt_len, full_len)
    lcp = 0
    p = prompt_ids[0]
    f = full_ids[0]
    while lcp < max_len and int(p[lcp].item()) == int(f[lcp].item()):
        lcp += 1
    return min(lcp, full_len)


def _estimate_completion_token_count(processor, completion_text: str) -> int:
    """Best-effort completion token count used for mask fallback."""
    text = str(completion_text or "").strip()
    if not text:
        return 0
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        tok = processor
    if tok is None or not hasattr(tok, "encode"):
        return 0
    try:
        ids = tok.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tok.encode(text)
    except Exception:
        return 0
    try:
        return int(len(ids))
    except Exception:
        return 0


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
        self._has_real_grad_in_window = False

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
        prompt_len = _aligned_prompt_prefix_len(
            inputs_prompt["input_ids"],
            input_ids,
            completion_text=completion,
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
        # NOTE: Do NOT return early when valid_token_count <= 0.
        # In DDP, all ranks must participate in the forward pass (which triggers
        # collective buffer sync). If some ranks return early while others
        # proceed to self.model(**inputs), the NCCL broadcast will hang and
        # eventually time out.  Instead, we run the forward pass on all ranks
        # and gate the backward/optimizer step on valid_token_count > 0.
        skip_backward = valid_token_count <= 0

        # For BLIP3o, forward() doesn't accept 'images'/'image_sizes' —
        # those are only for generate().  Filter them out for forward calls.
        _forward_keys_to_drop = ("images", "image_sizes")
        forward_full = {k: v for k, v in inputs_full.items()
                        if k not in _forward_keys_to_drop}

        if torch.cuda.is_available() and getattr(self.config, "clear_cache_every", 0) <= 1:
            torch.cuda.empty_cache()
            gc.collect()

        ref_inputs = dict(forward_full)
        ref_inputs["use_cache"] = False
        # IMPORTANT: for self-reference KL (reference_model is None), run the
        # reference pass BEFORE the trainable policy forward. This avoids
        # mutating module runtime state between checkpointed forward and
        # backward recompute.
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
            ref_model = _unwrap_model(self.model)
            def _run_ref_forward_base_adapter():
                was_training = bool(getattr(ref_model, "training", False))
                try:
                    # Keep reference KL pass outside DDP/checkpoint autograd path.
                    ref_model.eval()
                    with torch.no_grad():
                        with use_adapter(ref_model, None):
                            return ref_model(**ref_inputs)
                finally:
                    if was_training:
                        ref_model.train(True)
            try:
                out_ref = _run_ref_forward_base_adapter()
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    out_ref = _run_ref_forward_base_adapter()
                else:
                    raise
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

        # Check for non-finite ce_loss — mark for skip but do NOT return early
        # to keep DDP in sync across ranks.
        if not bool(torch.isfinite(ce_loss.detach()).all().item()):
            skip_backward = True

        if not skip_backward and valid_mask.any():
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

        if skip_backward:
            # No valid completion tokens or non-finite loss on this rank.
            # We use a zero loss for backward so DDP AllReduce still fires
            # (contributing zero gradients), keeping all ranks in sync.
            # Backward a zero loss that is still connected to the model's
            # computation graph so DDP AllReduce fires for all parameters.
            # Use sum() so every parameter contributes; multiply by 0 so
            # the actual gradient is zero.  nan_to_num guards against the
            # (rare) case where logits contain NaN.
            total_loss = (out_pi.logits.sum() * 0.0)
            if not torch.isfinite(total_loss):
                total_loss = torch.zeros((), device=out_pi.logits.device,
                                         dtype=out_pi.logits.dtype, requires_grad=True)
            skipped_reason = "no_valid_completion_tokens" if valid_token_count <= 0 else "non_finite_ce_loss"
        else:
            total_loss = advantage * ce_loss + beta_before * kl_loss
            skipped_reason = None
            if not bool(torch.isfinite(total_loss.detach()).all().item()):
                # Non-finite total_loss: backward zero instead.
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

        # Gradient accumulation: scale loss and accumulate.
        # All ranks MUST call backward() on the DDP model to participate
        # in the AllReduce gradient sync.  Ranks with no valid tokens
        # backward a zero loss, contributing zero gradients.
        scaled_loss = total_loss / self.grad_accum_steps
        if self._accum_count == 0:
            self.opt.zero_grad(set_to_none=True)
            self._has_real_grad_in_window = False
        restore_adapter = None
        model_ref = _unwrap_model(self.model)
        if self.adapter_name is not None and hasattr(model_ref, "set_adapter"):
            restore_adapter = getattr(model_ref, "active_adapter", None)
            try:
                model_ref.set_adapter(self.adapter_name)
            except Exception:
                restore_adapter = None
        try:
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
            if self._has_real_grad_in_window:
                _clip_grad_norm_multi_device(self.params, self.config.grad_clip)
                self.opt.step()
                did_step = True
            else:
                # All microbatches in this accumulation window were effectively skipped.
                # Do not step AdamW to avoid decoupled-weight-decay drift.
                self.opt.zero_grad(set_to_none=True)
            self._accum_count = 0
            self._has_real_grad_in_window = False
        self.model.train(False)

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
            "valid_token_count": float(valid_token_count),
            "forced_tail_tokens": float(forced_tail_tokens),
        }
