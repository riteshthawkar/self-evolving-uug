"""Text-based policy updaters for generation training (REINFORCE and DPO)."""

import gc
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from .generation_helpers import _prepare_text_inputs
from .utils import (_build_chat_text, _clip_grad_norm_multi_device, _collect_trainable_params, _prepare_mm_inputs, _unwrap_model, use_adapter)


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
        delta = (kl_val - target) / target
        beta = self.kl_coef * math.exp(self.config.kl_adapt_rate * delta)
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
        prompt_len = inputs_prompt["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        valid_mask = labels[:, 1:] != -100

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
        ce_loss = out_pi.loss
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

        did_step = False
        if self._accum_count >= self.grad_accum_steps:
            _clip_grad_norm_multi_device(self.params, self.config.grad_clip)
            self.opt.step()
            self._accum_count = 0
            did_step = True
        self.model.train(False)

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
        }


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
                prompt_len = int(inputs_prompt["input_ids"].shape[1])
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
            no_grad=True,
        )
        ref_logp_rejected, _ = self._forward_seq_logp(
            model=ref_model,
            adapter_name=ref_adapter_name,
            inputs_prompt=rejected_prompt_inputs,
            inputs_full=rejected_full_inputs,
            no_grad=True,
        )

        pi_logp_chosen, chosen_token_count = self._forward_seq_logp(
            model=self.model,
            adapter_name=self.adapter_name,
            inputs_prompt=chosen_prompt_inputs,
            inputs_full=chosen_full_inputs,
            no_grad=False,
        )
        pi_logp_rejected, rejected_token_count = self._forward_seq_logp(
            model=self.model,
            adapter_name=self.adapter_name,
            inputs_prompt=rejected_prompt_inputs,
            inputs_full=rejected_full_inputs,
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

        # Gradient accumulation: scale loss and accumulate
        scaled_loss = dpo_loss / self.grad_accum_steps
        if self._accum_count == 0:
            self.opt.zero_grad(set_to_none=True)
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

        did_step = False
        if self._accum_count >= self.grad_accum_steps:
            _clip_grad_norm_multi_device(self.params, self.config.grad_clip)
            self.opt.step()
            self._accum_count = 0
            did_step = True
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

        return {
            "dpo_loss": float(dpo_loss.item()),
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
            "did_step": did_step,
        }
