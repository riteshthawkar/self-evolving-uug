"""Text-based policy updaters for generation training (REINFORCE, DPO, GRPO)."""

import gc
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from .generation_helpers import _prepare_text_inputs
from .policy_updater import _reinforce_loss_from_ce
from .utils import (_build_chat_text, _clip_grad_norm_multi_device, _collect_trainable_params, _prepare_mm_inputs, _unwrap_model, use_adapter)


def _aligned_prompt_prefix_len(
    prompt_ids: torch.Tensor,
    full_ids: torch.Tensor,
    completion_text: str,
    processor=None,
) -> int:
    """Return a robust prompt-prefix length for completion masking.

    In multimodal chat templates, ``prompt`` and ``prompt+completion`` can be
    serialized with different image-token counts (e.g. Qwen2-VL dynamic
    resolution).  This function uses multiple strategies to find the correct
    prompt/completion boundary.
    """
    prompt_len = int(prompt_ids.shape[1])
    full_len = int(full_ids.shape[1])
    if full_len <= 0:
        return 0

    # Case 1: normal — prompt is shorter than full.
    if prompt_len < full_len:
        return prompt_len

    # Case 2: empty completion — mask everything.
    completion_str = str(completion_text or "").strip()
    if not completion_str:
        return min(prompt_len, full_len)

    # ── prompt_len >= full_len but completion is non-empty ──

    # Strategy A: estimate completion tokens and subtract from full_len.
    est_comp_tokens = _estimate_completion_token_count(processor, completion_text)
    if est_comp_tokens > 0:
        boundary = max(0, full_len - est_comp_tokens)
        boundary = max(1, boundary)
        return boundary

    # Strategy B: reverse suffix matching.
    if prompt_len > 0 and full_len > 0:
        p = prompt_ids[0]
        f = full_ids[0]
        suffix_match = 0
        max_suffix = min(prompt_len, full_len)
        while suffix_match < max_suffix:
            pi = prompt_len - 1 - suffix_match
            fi = full_len - 1 - suffix_match
            if int(p[pi].item()) == int(f[fi].item()):
                suffix_match += 1
            else:
                break
        if suffix_match > 0:
            lcp = 0
            cmp_len = min(prompt_len, full_len - suffix_match) if suffix_match < full_len else 0
            while lcp < cmp_len and int(p[lcp].item()) == int(f[lcp].item()):
                lcp += 1
            return max(1, min(lcp, full_len - 1))

    # Final fallback: leave at least the last 10% of tokens unmasked.
    return max(1, full_len - max(1, full_len // 10))


def _estimate_completion_token_count(processor, completion_text: str) -> int:
    """Best-effort completion token count used for mask fallback."""
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


def _text_supervised_step(
    updater,
    *,
    prompt: str,
    completion: str,
    device: torch.device,
    image: Optional[Image.Image] = None,
    completion_token_ids: Optional[List[int]] = None,
) -> Dict[str, float]:
    """DDP-safe CE-only supervised step for text completion training.

    Reuses each updater's optimizer/accumulation state so SFT and RL-like
    updates can share the same adapter parameters without optimizer conflicts.
    """
    if not completion or not str(completion).strip():
        raise ValueError("Supervised update requires non-empty completion text.")

    updater.step_id += 1
    updater.model.train(True)

    if image is None:
        text_prompt = prompt
        use_token_ids = bool(completion_token_ids)
        if use_token_ids:
            prompt_inputs = _prepare_text_inputs(updater.processor, device, text_prompt)
            prompt_ids = prompt_inputs["input_ids"]
            if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
                raise RuntimeError("Expected single-example prompt batch for token-trace supervised update.")
            comp_ids = torch.tensor(completion_token_ids, dtype=torch.long, device=prompt_ids.device).view(1, -1)
            full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
            full_mask = torch.ones_like(full_ids, dtype=torch.long)
            prompt_mask = prompt_inputs.get("attention_mask")
            if prompt_mask is None:
                prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long)
            inputs_prompt = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
            inputs_full = {"input_ids": full_ids, "attention_mask": full_mask}
        else:
            text_full = prompt + completion
            inputs_prompt = _prepare_text_inputs(updater.processor, device, text_prompt)
            inputs_full = _prepare_text_inputs(updater.processor, device, text_full)
    else:
        chat_prompt = _build_chat_text(updater.processor, image, prompt)
        chat_full = chat_prompt + completion
        inputs_prompt = _prepare_mm_inputs(updater.processor, device, image, chat_prompt, model=updater.model)
        inputs_full = _prepare_mm_inputs(updater.processor, device, image, chat_full, model=updater.model)

    input_ids = inputs_full["input_ids"]
    labels = input_ids.clone()
    prompt_len = _aligned_prompt_prefix_len(
        inputs_prompt["input_ids"],
        input_ids,
        completion_text=completion,
        processor=updater.processor,
    )
    labels[:, :prompt_len] = -100
    valid_mask = labels[:, 1:] != -100
    valid_token_count = int(valid_mask.sum().item())
    forced_tail_tokens = 0
    if valid_token_count <= 0 and str(completion or "").strip():
        full_len = int(input_ids.shape[1])
        prompt_raw_len = int(inputs_prompt["input_ids"].shape[1])
        tail_from_len_delta = max(0, full_len - min(prompt_raw_len, full_len))
        tail_from_completion = _estimate_completion_token_count(updater.processor, completion)
        forced_tail_tokens = max(tail_from_len_delta, tail_from_completion)
        # Ensure at least 1 token is unmasked for non-empty completions.
        forced_tail_tokens = max(1, min(forced_tail_tokens, max(1, full_len - 1)))
        if forced_tail_tokens > 0:
            labels[:, : full_len - forced_tail_tokens] = -100
            valid_mask = labels[:, 1:] != -100
            valid_token_count = int(valid_mask.sum().item())

    forward_inputs = {k: v for k, v in inputs_full.items() if k not in ("images", "image_sizes")}
    # Avoid undefined CE over an all-ignored label tensor.  Empty/invalid
    # completions are handled as a finite zero-gradient no-op below.
    if valid_token_count > 0:
        forward_inputs["labels"] = labels
    forward_inputs["use_cache"] = False
    with use_adapter(updater.model, updater.adapter_name):
        out = updater.model(**forward_inputs)
    if valid_token_count > 0 and out.loss is not None:
        ce_loss = out.loss
    else:
        ce_loss = torch.zeros((), device=out.logits.device, dtype=out.logits.dtype)

    skip_backward = False
    skipped_reason: Optional[str] = None
    if valid_token_count <= 0:
        skip_backward = True
        skipped_reason = "no_valid_completion_tokens"
    elif not bool(torch.isfinite(ce_loss.detach()).all().item()):
        skip_backward = True
        skipped_reason = "non_finite_ce_loss"

    if skip_backward:
        total_loss = torch.nan_to_num(
            out.logits,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).sum() * 0.0
        if not torch.isfinite(total_loss):
            total_loss = torch.zeros((), device=out.logits.device, dtype=out.logits.dtype, requires_grad=True)
    else:
        total_loss = ce_loss

    has_real_grad = skipped_reason is None
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor(
            [1 if has_real_grad else 0],
            dtype=torch.int32,
            device=total_loss.device,
        )
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        has_real_grad = bool(int(t.item()) == 1)

    scaled_loss = total_loss / updater.grad_accum_steps
    if updater._accum_count == 0:
        updater.opt.zero_grad(set_to_none=True)
        updater._has_real_grad_in_window = False
    restore_adapter = None
    model_ref = _unwrap_model(updater.model)
    if updater.adapter_name is not None and hasattr(model_ref, "set_adapter"):
        restore_adapter = getattr(model_ref, "active_adapter", None)
        try:
            model_ref.set_adapter(updater.adapter_name)
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

    updater._accum_count += 1
    if has_real_grad:
        updater._has_real_grad_in_window = True

    did_step = False
    if updater._accum_count >= updater.grad_accum_steps:
        if updater._has_real_grad_in_window:
            if _clip_grad_norm_multi_device(updater.params, updater.config.grad_clip):
                updater.opt.step()
                did_step = True
            else:
                skipped_reason = skipped_reason or "non_finite_gradient"
                updater.opt.zero_grad(set_to_none=True)
        else:
            updater.opt.zero_grad(set_to_none=True)
        updater._accum_count = 0
        updater._has_real_grad_in_window = False
    updater.model.train(False)

    if (
        torch.cuda.is_available()
        and updater.config.clear_cache_every > 0
        and updater.step_id % updater.config.clear_cache_every == 0
    ):
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()

    return {
        "ce_loss": float(ce_loss.detach().item()) if torch.isfinite(ce_loss.detach()).all() else float("nan"),
        "total_loss": float(total_loss.detach().item()) if torch.isfinite(total_loss.detach()).all() else 0.0,
        "did_step": bool(did_step),
        "skipped_reason": skipped_reason,
        "valid_token_count": float(valid_token_count),
        "forced_tail_tokens": float(forced_tail_tokens),
    }


class TextPolicyUpdater:
    """KL-regularized REINFORCE updater for text-only trajectories (generator role)."""

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
            raise RuntimeError(f"No trainable parameters found for adapter={adapter_name!r}")
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

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
        if not math.isfinite(float(kl_val)):
            return
        rate = float(getattr(self.config, "kl_adapt_rate", 0.0))
        if not math.isfinite(rate):
            return
        current = float(self.kl_coef)
        if not math.isfinite(current) or current <= 0.0:
            current = max(float(self.config.kl_min), 1e-8)
        delta = max(-50.0, min(50.0, (float(kl_val) - target) / target))
        beta = current * math.exp(max(-50.0, min(50.0, rate * delta)))
        beta = max(self.config.kl_min, min(self.config.kl_max, beta))
        self.kl_coef = float(beta)

    def step(
        self,
        prompt: str,
        completion: str,
        reward: float,
        baseline: float,
        device: torch.device,
        image: Optional[Image.Image] = None,
        completion_token_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        if not completion or not str(completion).strip():
            raise ValueError("Generator update requires non-empty token completion trace.")
        self.step_id += 1

        if image is None:
            text_prompt = prompt
            use_token_ids = bool(completion_token_ids)
            if use_token_ids:
                prompt_inputs = _prepare_text_inputs(self.processor, device, text_prompt)
                prompt_ids = prompt_inputs["input_ids"]
                if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
                    raise RuntimeError("Expected single-example prompt batch for token-trace generator update.")
                comp_ids = torch.tensor(completion_token_ids, dtype=torch.long, device=prompt_ids.device).view(1, -1)
                full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
                full_mask = torch.ones_like(full_ids, dtype=torch.long)
                prompt_mask = prompt_inputs.get("attention_mask")
                if prompt_mask is None:
                    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long)
                inputs_prompt = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
                inputs_full = {"input_ids": full_ids, "attention_mask": full_mask}
            else:
                text_full = prompt + completion
                inputs_prompt = _prepare_text_inputs(self.processor, device, text_prompt)
                inputs_full = _prepare_text_inputs(self.processor, device, text_full)
        else:
            chat_prompt = _build_chat_text(self.processor, image, prompt)
            chat_full = chat_prompt + completion
            inputs_prompt = _prepare_mm_inputs(self.processor, device, image, chat_prompt, model=self.model)
            inputs_full = _prepare_mm_inputs(self.processor, device, image, chat_full, model=self.model)

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
            # Ensure at least 1 token is unmasked for non-empty completions.
            forced_tail_tokens = max(1, min(forced_tail_tokens, max(1, full_len - 1)))
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

        # For BLIP3o forward(), we must NOT pass 'images' to model(**inputs)
        # because the CausalLM forward() doesn't accept it the same way
        # generate() does.  Extract images for generate() calls only.
        forward_full = {k: v for k, v in inputs_full.items()
                        if k not in ("images", "image_sizes")}
        forward_prompt = {k: v for k, v in inputs_prompt.items()
                          if k not in ("images", "image_sizes")}

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
        # Only compute CE when there is a real completion-token target.
        # All-ignored labels make CE undefined and can surface as NaN.
        if valid_token_count > 0:
            policy_inputs["labels"] = labels
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
        if valid_token_count > 0 and out_pi.loss is not None:
            ce_loss = out_pi.loss
        else:
            ce_loss = torch.zeros((), device=out_pi.logits.device, dtype=out_pi.logits.dtype)

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
            total_loss = torch.nan_to_num(
                out_pi.logits,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).sum() * 0.0
            if not torch.isfinite(total_loss):
                total_loss = torch.zeros((), device=out_pi.logits.device,
                                         dtype=out_pi.logits.dtype, requires_grad=True)
            skipped_reason = "no_valid_completion_tokens" if valid_token_count <= 0 else "non_finite_ce_loss"
        else:
            # CE is -logprob, so minimizing advantage * CE maximizes
            # advantage * logprob.
            total_loss = _reinforce_loss_from_ce(
                ce_loss,
                kl_loss,
                advantage,
                beta_before,
            )
            skipped_reason = None
            if not bool(torch.isfinite(total_loss.detach()).all().item()):
                # Non-finite total_loss: backward zero instead.
                total_loss = torch.nan_to_num(
                    out_pi.logits,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).sum() * 0.0
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
                if _clip_grad_norm_multi_device(self.params, self.config.grad_clip):
                    self.opt.step()
                    did_step = True
                else:
                    skipped_reason = skipped_reason or "non_finite_gradient"
                    self.opt.zero_grad(set_to_none=True)
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

        try:
            del inputs_prompt, inputs_full, input_ids, labels, policy_inputs
            del out_pi, out_ref, valid_mask
            if "pi_shift" in locals():
                del pi_shift
            if "ref_shift" in locals():
                del ref_shift
            if "valid_pos" in locals():
                del valid_pos
        except Exception:
            pass

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
            "ce_loss": float(ce_loss.item()),
            "kl_loss": kl_val,
            "advantage": advantage,
            "kl_coef_before": beta_before,
            "kl_coef_after": float(self.kl_coef),
            "total_loss": float(total_loss.item()),
            "did_step": did_step,
            "valid_token_count": float(valid_token_count),
            "forced_tail_tokens": float(forced_tail_tokens),
        }

    def sft_step(
        self,
        *,
        prompt: str,
        completion: str,
        device: torch.device,
        image: Optional[Image.Image] = None,
        completion_token_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        return _text_supervised_step(
            self,
            prompt=prompt,
            completion=completion,
            device=device,
            image=image,
            completion_token_ids=completion_token_ids,
        )


# ---------------------------------------------------------------------------
# TextPreferenceDPOUpdater (DPO for generator role)
# ---------------------------------------------------------------------------


class TextPreferenceDPOUpdater:
    """Pairwise DPO updater for generator role."""

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
        self.step_id = 0
        self.kl_coef = 0.0
        self.grad_accum_steps = max(1, getattr(config, "grad_accum_steps", 1))
        self._accum_count = 0
        self._has_real_grad_in_window = False

        self.dpo_beta = float(max(1e-6, getattr(config, "dpo_beta", 0.1)))
        self.dpo_label_smoothing = float(getattr(config, "dpo_label_smoothing", 0.0))
        if not (0.0 <= self.dpo_label_smoothing < 0.5):
            raise ValueError("dpo_label_smoothing must satisfy 0.0 <= value < 0.5")

        params = list(_collect_trainable_params(model, adapter_name))
        if not params:
            raise RuntimeError(f"No trainable parameters found for adapter={adapter_name!r}")
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    def state_dict(self) -> Dict:
        return {
            "optimizer": self.opt.state_dict(),
            "step_id": int(self.step_id),
            "dpo_beta": float(self.dpo_beta),
            "dpo_label_smoothing": float(self.dpo_label_smoothing),
            "kl_coef": float(self.kl_coef),
        }

    def load_state_dict(self, state: Dict):
        if not isinstance(state, dict):
            return
        if "optimizer" in state:
            self.opt.load_state_dict(state["optimizer"])
        if "step_id" in state:
            self.step_id = int(state["step_id"])
        if "dpo_beta" in state:
            self.dpo_beta = float(state["dpo_beta"])
        if "dpo_label_smoothing" in state:
            self.dpo_label_smoothing = float(state["dpo_label_smoothing"])
        if "kl_coef" in state:
            self.kl_coef = float(state["kl_coef"])

    def _build_inputs(
        self,
        *,
        prompt: str,
        completion: str,
        device: torch.device,
        image: Optional[Image.Image] = None,
        completion_token_ids: Optional[List[int]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not completion or not str(completion).strip():
            raise ValueError("DPO update requires non-empty completion text.")

        if image is None:
            text_prompt = prompt
            use_token_ids = bool(completion_token_ids)
            if use_token_ids:
                prompt_inputs = _prepare_text_inputs(self.processor, device, text_prompt)
                prompt_ids = prompt_inputs["input_ids"]
                if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
                    raise RuntimeError("Expected single-example prompt batch for DPO token-trace update.")
                comp_ids = torch.tensor(completion_token_ids, dtype=torch.long, device=prompt_ids.device).view(1, -1)
                full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
                full_mask = torch.ones_like(full_ids, dtype=torch.long)
                prompt_mask = prompt_inputs.get("attention_mask")
                if prompt_mask is None:
                    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long)
                inputs_prompt = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
                inputs_full = {"input_ids": full_ids, "attention_mask": full_mask}
            else:
                text_full = prompt + completion
                inputs_prompt = _prepare_text_inputs(self.processor, device, text_prompt)
                inputs_full = _prepare_text_inputs(self.processor, device, text_full)
        else:
            chat_prompt = _build_chat_text(self.processor, image, prompt)
            chat_full = chat_prompt + completion
            inputs_prompt = _prepare_mm_inputs(self.processor, device, image, chat_prompt, model=self.model)
            inputs_full = _prepare_mm_inputs(self.processor, device, image, chat_full, model=self.model)
        return inputs_prompt, inputs_full

    def _sequence_logp_from_logits(
        self,
        *,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        prompt_len: int,
    ) -> Tuple[torch.Tensor, int]:
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != -100

        logp = F.log_softmax(logits, dim=-1)
        shift_logp = logp[:, :-1, :]
        gathered = shift_logp.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)

        valid_count = int(valid_mask.sum().item())
        if valid_count <= 0:
            seq_logp = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        else:
            seq_logp = gathered[valid_mask].mean()
        return seq_logp, valid_count

    def _forward_seq_logp(
        self,
        *,
        model: torch.nn.Module,
        adapter_name: Optional[str],
        inputs_prompt: Dict[str, torch.Tensor],
        inputs_full: Dict[str, torch.Tensor],
        completion_text: str,
        no_grad: bool,
    ) -> Tuple[torch.Tensor, int]:
        run_model = _unwrap_model(model) if no_grad else model
        context = torch.no_grad() if no_grad else torch.enable_grad()
        was_training = bool(getattr(run_model, "training", False))
        try:
            if no_grad:
                # Disable checkpoint wrappers and DDP reducer hooks for reference pass.
                run_model.eval()
            with context:
                # Filter out generate()-only keys that forward() doesn't accept
                forward_inputs = {
                    k: v for k, v in inputs_full.items()
                    if k not in ("images", "image_sizes")
                }
                forward_inputs["use_cache"] = False
                with use_adapter(run_model, adapter_name):
                    out = run_model(**forward_inputs)
                prompt_len = _aligned_prompt_prefix_len(
                    inputs_prompt["input_ids"],
                    forward_inputs["input_ids"],
                    completion_text=completion_text,
                    processor=self.processor,
                )
                seq_logp, token_count = self._sequence_logp_from_logits(
                    logits=out.logits,
                    input_ids=forward_inputs["input_ids"],
                    prompt_len=prompt_len,
                )
        finally:
            if no_grad and was_training:
                run_model.train(True)
        return seq_logp, token_count

    def step(
        self,
        *,
        prompt: str,
        chosen_completion: str,
        rejected_completion: str,
        device: torch.device,
        chosen_image: Optional[Image.Image] = None,
        rejected_image: Optional[Image.Image] = None,
        chosen_completion_token_ids: Optional[List[int]] = None,
        rejected_completion_token_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        self.step_id += 1
        self.model.train(True)

        chosen_prompt_inputs, chosen_full_inputs = self._build_inputs(
            prompt=prompt,
            completion=chosen_completion,
            device=device,
            image=chosen_image,
            completion_token_ids=chosen_completion_token_ids,
        )
        rejected_prompt_inputs, rejected_full_inputs = self._build_inputs(
            prompt=prompt,
            completion=rejected_completion,
            device=device,
            image=rejected_image,
            completion_token_ids=rejected_completion_token_ids,
        )

        if self.reference_model is not None:
            ref_model = self.reference_model
            ref_adapter_name = None
        else:
            ref_model = _unwrap_model(self.model)
            ref_adapter_name = None

        # IMPORTANT: when using self-reference (no frozen reference model),
        # compute reference log-probs before trainable forwards to avoid
        # checkpoint-recompute metadata mismatches.
        ref_logp_chosen, _ = self._forward_seq_logp(
            model=ref_model,
            adapter_name=ref_adapter_name,
            inputs_prompt=chosen_prompt_inputs,
            inputs_full=chosen_full_inputs,
            completion_text=chosen_completion,
            no_grad=True,
        )
        ref_logp_rejected, _ = self._forward_seq_logp(
            model=ref_model,
            adapter_name=ref_adapter_name,
            inputs_prompt=rejected_prompt_inputs,
            inputs_full=rejected_full_inputs,
            completion_text=rejected_completion,
            no_grad=True,
        )

        pi_logp_chosen, chosen_token_count = self._forward_seq_logp(
            model=self.model,
            adapter_name=self.adapter_name,
            inputs_prompt=chosen_prompt_inputs,
            inputs_full=chosen_full_inputs,
            completion_text=chosen_completion,
            no_grad=False,
        )
        pi_logp_rejected, rejected_token_count = self._forward_seq_logp(
            model=self.model,
            adapter_name=self.adapter_name,
            inputs_prompt=rejected_prompt_inputs,
            inputs_full=rejected_full_inputs,
            completion_text=rejected_completion,
            no_grad=False,
        )

        pi_gap = pi_logp_chosen - pi_logp_rejected
        ref_gap = ref_logp_chosen - ref_logp_rejected
        preference_margin = pi_gap - ref_gap

        scaled_margin = self.dpo_beta * preference_margin
        pos_term = -F.logsigmoid(scaled_margin)
        if self.dpo_label_smoothing > 0.0:
            neg_term = -F.logsigmoid(-scaled_margin)
            dpo_loss = (1.0 - self.dpo_label_smoothing) * pos_term + self.dpo_label_smoothing * neg_term
        else:
            dpo_loss = pos_term
        skipped_reason: Optional[str] = None
        skip_backward = False
        if chosen_token_count <= 0 or rejected_token_count <= 0:
            skipped_reason = "no_valid_completion_tokens"
            skip_backward = True
        elif not bool(torch.isfinite(pi_logp_chosen.detach()).all().item()):
            skipped_reason = "non_finite_pi_logp_chosen"
            skip_backward = True
        elif not bool(torch.isfinite(pi_logp_rejected.detach()).all().item()):
            skipped_reason = "non_finite_pi_logp_rejected"
            skip_backward = True
        elif not bool(torch.isfinite(ref_logp_chosen.detach()).all().item()):
            skipped_reason = "non_finite_ref_logp_chosen"
            skip_backward = True
        elif not bool(torch.isfinite(ref_logp_rejected.detach()).all().item()):
            skipped_reason = "non_finite_ref_logp_rejected"
            skip_backward = True
        elif not bool(torch.isfinite(dpo_loss.detach()).all().item()):
            skipped_reason = "non_finite_dpo_loss"
            skip_backward = True

        if skip_backward:
            # Keep DDP collectives in sync: backward a zero-connected objective.
            dpo_loss_for_backward = (
                torch.nan_to_num(pi_logp_chosen, nan=0.0, posinf=0.0, neginf=0.0) * 0.0
                + torch.nan_to_num(pi_logp_rejected, nan=0.0, posinf=0.0, neginf=0.0) * 0.0
            )
        else:
            dpo_loss_for_backward = dpo_loss
        has_real_grad = skipped_reason is None
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor(
                [1 if has_real_grad else 0],
                dtype=torch.int32,
                device=dpo_loss_for_backward.device,
            )
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            has_real_grad = bool(int(t.item()) == 1)

        # Gradient accumulation: scale loss and accumulate
        scaled_loss = dpo_loss_for_backward / self.grad_accum_steps
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
                if _clip_grad_norm_multi_device(self.params, self.config.grad_clip):
                    self.opt.step()
                    did_step = True
                else:
                    skipped_reason = skipped_reason or "non_finite_gradient"
                    self.opt.zero_grad(set_to_none=True)
            else:
                # All microbatches in this accumulation window were effectively skipped.
                # Do not step AdamW to avoid decoupled-weight-decay drift.
                self.opt.zero_grad(set_to_none=True)
            self._accum_count = 0
            self._has_real_grad_in_window = False
        self.model.train(False)

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

        reported_dpo_loss = float(dpo_loss.detach().item()) if torch.isfinite(dpo_loss.detach()).all() else float("nan")
        return {
            "dpo_loss": reported_dpo_loss,
            "dpo_beta": float(self.dpo_beta),
            "label_smoothing": float(self.dpo_label_smoothing),
            "pi_logp_chosen": float(pi_logp_chosen.detach().item()),
            "pi_logp_rejected": float(pi_logp_rejected.detach().item()),
            "ref_logp_chosen": float(ref_logp_chosen.detach().item()),
            "ref_logp_rejected": float(ref_logp_rejected.detach().item()),
            "pi_gap": float(pi_gap.detach().item()),
            "ref_gap": float(ref_gap.detach().item()),
            "preference_margin": float(preference_margin.detach().item()),
            "chosen_token_count": float(chosen_token_count),
            "rejected_token_count": float(rejected_token_count),
            "kl_coef_before": 0.0,
            "kl_coef_after": 0.0,
            "did_step": bool(did_step),
            "skipped_reason": skipped_reason,
        }

    def sft_step(
        self,
        *,
        prompt: str,
        completion: str,
        device: torch.device,
        image: Optional[Image.Image] = None,
        completion_token_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        return _text_supervised_step(
            self,
            prompt=prompt,
            completion=completion,
            device=device,
            image=image,
            completion_token_ids=completion_token_ids,
        )


# ---------------------------------------------------------------------------
# TextGRPOUpdater (Group Relative Policy Optimization for generator role)
# ---------------------------------------------------------------------------


class TextGRPOUpdater:
    """GRPO updater for generator role.

    Takes a group of (completion, reward) pairs from N candidates,
    computes group-normalised advantages, and applies weighted policy
    gradient on *all* completions in one optimiser step.

    Unlike DPO (which uses only best-vs-worst), GRPO uses every candidate,
    giving a stronger, lower-variance training signal.
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
        self.clip_ratio = float(getattr(config, "grpo_clip_ratio", 0.2))
        self.min_group_std = float(getattr(config, "grpo_min_group_std", 1e-6))

        params = list(_collect_trainable_params(model, adapter_name))
        if not params:
            raise RuntimeError(f"No trainable parameters found for adapter={adapter_name!r}")
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    # ---- state management (same pattern as other updaters) ----

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
        if not math.isfinite(float(kl_val)):
            return
        rate = float(getattr(self.config, "kl_adapt_rate", 0.0))
        if not math.isfinite(rate):
            return
        current = float(self.kl_coef)
        if not math.isfinite(current) or current <= 0.0:
            current = max(float(self.config.kl_min), 1e-8)
        delta = max(-50.0, min(50.0, (float(kl_val) - target) / target))
        beta = current * math.exp(max(-50.0, min(50.0, rate * delta)))
        beta = max(self.config.kl_min, min(self.config.kl_max, beta))
        self.kl_coef = float(beta)

    # ---- per-completion forward pass ----

    def _seq_logp(
        self,
        prompt_inputs: Dict[str, torch.Tensor],
        full_inputs: Dict[str, torch.Tensor],
        completion_text: str,
        *,
        use_ref: bool,
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Forward pass for one completion, returning (seq_logp, valid_count, logits).

        If *use_ref* is True, runs under no_grad on the reference / base adapter.
        Otherwise runs the trainable policy adapter with grad enabled.
        """
        forward_inputs = {
            k: v for k, v in full_inputs.items()
            if k not in ("images", "image_sizes")
        }
        forward_inputs["use_cache"] = False

        if use_ref:
            if self.reference_model is not None:
                run_model = self.reference_model
                ctx_adapter = None
            else:
                run_model = _unwrap_model(self.model)
                ctx_adapter = None
            was_training = bool(getattr(run_model, "training", False))
            try:
                run_model.eval()
                with torch.no_grad():
                    with use_adapter(run_model, ctx_adapter):
                        out = run_model(**forward_inputs)
            finally:
                if was_training:
                    run_model.train(True)
        else:
            with use_adapter(self.model, self.adapter_name):
                out = self.model(**forward_inputs)

        prompt_len = _aligned_prompt_prefix_len(
            prompt_inputs["input_ids"],
            forward_inputs["input_ids"],
            completion_text=completion_text,
            processor=self.processor,
        )
        input_ids = forward_inputs["input_ids"]
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != -100

        logp = F.log_softmax(out.logits[:, :-1, :], dim=-1)
        gathered = logp.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)

        valid_count = int(valid_mask.sum().item())
        if valid_count > 0:
            seq_logp = gathered[valid_mask].mean()
        else:
            seq_logp = torch.tensor(0.0, device=out.logits.device, dtype=out.logits.dtype)

        return seq_logp, valid_count, out.logits

    # ---- main step ----

    def step(
        self,
        *,
        prompt: str,
        completions: List[str],
        rewards: List[float],
        device: torch.device,
        images: Optional[List[Optional[Image.Image]]] = None,
        completion_token_ids: Optional[List[Optional[List[int]]]] = None,
        ddp_no_sync: bool = False,
        baseline_shifted: bool = False,
    ) -> Dict[str, float]:
        """Run one GRPO update using a group of (completion, reward) pairs.

        Args:
            baseline_shifted: If True, rewards have already been shifted by an
                EMA baseline (e.g., the proposer path in unified_trainer.py).
                In this case we use rewards directly as advantages (scaled by
                max|r|) to avoid the loss-cancellation deadlock that occurs
                with standard group normalization when importance ratios ≈ 1.0.
                If False (default), use standard GRPO group normalization.
        """
        self.step_id += 1
        n = len(completions)
        assert n == len(rewards), "completions and rewards must have same length"

        # --- 1. Compute advantages ---
        r_tensor = torch.tensor(rewards, dtype=torch.float64)
        r_mean = float(r_tensor.mean().item())
        r_std = float(r_tensor.std(correction=0).item())

        if baseline_shifted:
            # ── Baseline-shifted path (proposer) ─────────────────────────
            #
            # Rewards have ALREADY been shifted by the cross-step EMA
            # baseline in the caller (e.g., unified_trainer.py):
            #   rewards_shifted = [r - ema_baseline for r in raw_rewards]
            #
            # Standard GRPO re-centers on the group mean, producing
            # advantages that sum to exactly zero.  When importance ratios
            # ≈ 1.0 (early training), total loss = sum(-adv * 1.0) = 0.0
            # — a deadlock where the proposer gets zero gradient.
            #
            # FIX: Use rewards directly as advantages.  They already encode
            # the absolute signal via EMA baseline.  Scale by max(|r|) to
            # keep them in [-1, 1] for gradient stability.
            #
            # Example with rewards [-0.067, 0.033, 0.033]:
            #   advantages = [-1.0, 0.49, 0.49], sum = -0.02 ≠ 0
            #   loss ≈ 0.02  (non-zero gradient!)
            r_abs_max = max(abs(float(r)) for r in rewards) if rewards else 1.0
            skip_low_std = r_abs_max < self.min_group_std
            if skip_low_std:
                # NGRPO (arXiv:2509.18851): inject virtual max-reward sample
                # to create non-zero negative advantages for degenerate groups.
                # Without this, all-zero rewards → zero gradient → deadlock.
                _r_virtual = 1.0
                _r_aug = [float(r) for r in rewards] + [_r_virtual]
                _mu_aug = sum(_r_aug) / len(_r_aug)
                _var_aug = sum((x - _mu_aug) ** 2 for x in _r_aug) / len(_r_aug)
                _std_aug = _var_aug ** 0.5
                if _std_aug > 1e-8:
                    advantages = [(float(r) - _mu_aug) / (_std_aug + 1e-8) for r in rewards]
                    # NGRPO produced valid non-zero advantages → allow backward pass.
                    # Without this, the later `elif skip_low_std:` gate would force
                    # skip_backward=True, zeroing the gradient we just computed.
                    skip_low_std = False
                else:
                    advantages = [0.0] * n
            else:
                advantages = [float(r) / (r_abs_max + 1e-8) for r in rewards]
        else:
            # ── Standard GRPO group normalization (generator) ────────────
            # Center on group mean and normalize by std.  This works well
            # when rewards span a range (some good, some bad candidates).
            skip_low_std = r_std < self.min_group_std
            if skip_low_std:
                # NGRPO virtual max-reward for degenerate standard groups
                _r_virtual = 1.0
                _r_aug = [float(r) for r in rewards] + [_r_virtual]
                _mu_aug = sum(_r_aug) / len(_r_aug)
                _var_aug = sum((x - _mu_aug) ** 2 for x in _r_aug) / len(_r_aug)
                _std_aug = _var_aug ** 0.5
                if _std_aug > 1e-8:
                    advantages = [(float(r) - _mu_aug) / (_std_aug + 1e-8) for r in rewards]
                    # NGRPO produced valid non-zero advantages → allow backward pass.
                    skip_low_std = False
                else:
                    advantages = [0.0] * n
            else:
                advantages = [(r - r_mean) / (r_std + 1e-8) for r in rewards]

        # --- 2. Build inputs for each completion ---
        self.model.train(True)
        if torch.cuda.is_available() and getattr(self.config, "clear_cache_every", 0) <= 1:
            torch.cuda.empty_cache()
            gc.collect()

        beta_before = float(self.kl_coef)
        total_grpo_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_kl_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        valid_count_total = 0
        valid_completions = 0
        last_logits = None  # keep one reference for DDP zero-loss fallback

        # Process each completion sequentially to avoid OOM.
        for i in range(n):
            comp = completions[i]
            if not comp or not str(comp).strip():
                continue

            adv_i = advantages[i]
            img_i = images[i] if images is not None and i < len(images) else None
            tid_i = (completion_token_ids[i]
                     if completion_token_ids is not None and i < len(completion_token_ids)
                     else None)

            # Build prompt and full inputs
            if img_i is not None:
                chat_prompt = _build_chat_text(self.processor, img_i, prompt)
                chat_full = chat_prompt + comp
                inputs_prompt = _prepare_mm_inputs(
                    self.processor, device, img_i, chat_prompt, model=self.model
                )
                inputs_full = _prepare_mm_inputs(
                    self.processor, device, img_i, chat_full, model=self.model
                )
            elif tid_i is not None:
                inputs_prompt = _prepare_text_inputs(self.processor, device, prompt)
                prompt_ids = inputs_prompt["input_ids"]
                comp_ids = torch.tensor(tid_i, dtype=torch.long, device=device).view(1, -1)
                full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
                full_mask = torch.ones_like(full_ids, dtype=torch.long)
                inputs_full = {"input_ids": full_ids, "attention_mask": full_mask}
            else:
                inputs_prompt = _prepare_text_inputs(self.processor, device, prompt)
                inputs_full = _prepare_text_inputs(self.processor, device, prompt + comp)

            # Reference log-prob (no grad)
            ref_logp, ref_vc, _ = self._seq_logp(
                inputs_prompt, inputs_full, comp, use_ref=True,
            )

            # Policy log-prob (with grad)
            self.model.train(True)
            pi_logp, pi_vc, logits_i = self._seq_logp(
                inputs_prompt, inputs_full, comp, use_ref=False,
            )
            last_logits = logits_i

            if pi_vc <= 0:
                continue

            valid_completions += 1
            valid_count_total += pi_vc

            # Simplified per-sequence KL: difference in mean log-probs.
            kl_i = (pi_logp - ref_logp).clamp(min=0.0)

            # Clipped GRPO policy gradient loss.
            # Importance ratio = pi(completion) / ref(completion).
            log_ratio_i = pi_logp - ref_logp.detach()
            ratio_i = log_ratio_i.exp().clamp(max=10.0)  # safety cap
            surr1 = -adv_i * ratio_i
            surr2 = -adv_i * ratio_i.clamp(
                1.0 - self.clip_ratio, 1.0 + self.clip_ratio,
            )
            # Pessimistic bound: take the worse (larger) loss.
            pg_loss_i = torch.max(surr1, surr2)

            total_grpo_loss = total_grpo_loss + pg_loss_i
            total_kl_loss = total_kl_loss + kl_i

        # --- 3. Combine and backward ---
        if valid_completions > 0:
            avg_grpo_loss = total_grpo_loss / valid_completions
            avg_kl_loss = total_kl_loss / valid_completions
            total_loss = avg_grpo_loss + beta_before * avg_kl_loss
            if not bool(torch.isfinite(total_loss.detach()).all().item()):
                skip_backward = True
                skipped_reason = "non_finite_loss"
            elif skip_low_std:
                # Advantages were zeroed, so total_grpo_loss ≈ 0; mark as skipped
                # but still run backward (zero-gradient) for DDP sync.
                skip_backward = True
                skipped_reason = "group_std_too_low"
            else:
                skip_backward = False
                skipped_reason = None
        else:
            skip_backward = True
            skipped_reason = "group_std_too_low" if skip_low_std else "no_valid_completions"
            total_loss = None

        if skip_backward:
            # DDP sync: backward zero gradient
            if last_logits is not None:
                zero_loss = torch.nan_to_num(
                    last_logits,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).sum() * 0.0
            else:
                # No forward was run at all; run a dummy forward for DDP sync
                dummy_inputs = _prepare_text_inputs(self.processor, device, prompt)
                forward_inputs = {
                    k: v for k, v in dummy_inputs.items()
                    if k not in ("images", "image_sizes")
                }
                forward_inputs["use_cache"] = False
                with use_adapter(self.model, self.adapter_name):
                    dummy_out = self.model(**forward_inputs)
                zero_loss = torch.nan_to_num(
                    dummy_out.logits,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).sum() * 0.0
            total_loss = zero_loss

        has_real_grad = skipped_reason is None
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor(
                [1 if has_real_grad else 0],
                dtype=torch.int32, device=device,
            )
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            has_real_grad = bool(int(t.item()) == 1)

        # Gradient accumulation
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
        # When ddp_no_sync=True, skip DDP allreduce during backward to avoid
        # "gradient undefined but allreduced" errors from multiple backward passes
        # in the same training step.  Gradients are manually synced before opt.step.
        import contextlib as _contextlib
        _no_sync_ctx = (
            self.model.no_sync()
            if (ddp_no_sync and hasattr(self.model, "no_sync"))
            else _contextlib.nullcontext()
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
            # Manual gradient sync when DDP allreduce was skipped.
            if ddp_no_sync and dist.is_available() and dist.is_initialized():
                for p in self.params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= float(dist.get_world_size())
            if self._has_real_grad_in_window:
                if _clip_grad_norm_multi_device(self.params, self.config.grad_clip):
                    self.opt.step()
                    did_step = True
                else:
                    skipped_reason = skipped_reason or "non_finite_gradient"
                    self.opt.zero_grad(set_to_none=True)
            else:
                self.opt.zero_grad(set_to_none=True)
            self._accum_count = 0
            self._has_real_grad_in_window = False

        self.model.train(False)

        # Adapt KL coefficient
        kl_val = float(total_kl_loss.detach().item()) if torch.isfinite(total_kl_loss.detach()).all() else 0.0
        if skipped_reason is None:
            self._adapt_beta(kl_val)

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
            "grpo_loss": float(avg_grpo_loss.detach().item()) if valid_completions > 0 and torch.is_tensor(avg_grpo_loss) else 0.0,
            "group_size": n,
            "mean_reward": r_mean,
            "std_reward": r_std,
            "mean_advantage": float(sum(advantages) / len(advantages)),
            "max_advantage": float(max(advantages)),
            "min_advantage": float(min(advantages)),
            "kl_loss": kl_val,
            "kl_coef_before": beta_before,
            "kl_coef_after": float(self.kl_coef),
            "did_step": did_step,
            "skipped_reason": skipped_reason,
            "valid_completions": valid_completions,
        }

    def sft_step(
        self,
        *,
        prompt: str,
        completion: str,
        device: torch.device,
        image: Optional[Image.Image] = None,
        completion_token_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        return _text_supervised_step(
            self,
            prompt=prompt,
            completion=completion,
            device=device,
            image=image,
            completion_token_ids=completion_token_ids,
        )
