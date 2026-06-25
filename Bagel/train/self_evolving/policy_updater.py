# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from modeling.bagel.runtime_precision import autocast_context

from .adapter_manager import collect_adapter_parameters, use_adapter
from .config import RolloutConfig
from .model_loader import BagelRuntime


def _build_single_sample_attention_mask(
    *,
    split_lens: List[int],
    attn_modes: List[str],
    device: torch.device,
) -> torch.Tensor:
    """Build a dense attention mask for one sample to avoid flex-attn OOM in policy updates."""
    if len(split_lens) != len(attn_modes):
        raise ValueError(
            f"split_lens and attn_modes must have same length, got {len(split_lens)} vs {len(attn_modes)}."
        )

    sample_len = int(sum(int(v) for v in split_lens))
    allow = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for seg_len_raw, attn_mode in zip(split_lens, attn_modes):
        seg_len = int(seg_len_raw)
        if seg_len <= 0:
            continue
        seg_slice = slice(csum, csum + seg_len)
        mode = str(attn_mode)
        if mode == "causal":
            allow[seg_slice, seg_slice] = torch.ones((seg_len, seg_len), dtype=torch.bool, device=device).tril()
            if csum > 0:
                allow[seg_slice, :csum] = True
        elif mode in {"full", "noise"}:
            allow[seg_slice, seg_slice] = True
            if csum > 0:
                allow[seg_slice, :csum] = True
        else:
            raise ValueError(f"Unsupported attn_mode={mode!r}.")
        csum += seg_len

    # For noise segments, block attending to other noise segments.
    csum = 0
    for seg_len_raw, attn_mode in zip(split_lens, attn_modes):
        seg_len = int(seg_len_raw)
        if seg_len > 0 and str(attn_mode) == "noise":
            allow[:, csum : csum + seg_len] = False
            allow[csum : csum + seg_len, csum : csum + seg_len] = True
        csum += seg_len

    mask = torch.full((sample_len, sample_len), float("-inf"), dtype=torch.float32, device=device)
    mask = mask.masked_fill(allow, 0.0)
    return mask


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out: Dict = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _compute_advantage(
    *,
    reward: float,
    baseline: float,
    method: str,
    group_rewards: Optional[List[float]],
    eps: float,
) -> float:
    if str(method) == "grpo" and group_rewards and len(group_rewards) > 1:
        vals = [float(v) for v in group_rewards]
        mean = float(sum(vals) / float(len(vals)))
        var = float(sum((v - mean) ** 2 for v in vals) / float(len(vals)))
        std = math.sqrt(max(0.0, var))
        if std > float(eps):
            return float((reward - mean) / std)
        return float(reward - mean)
    return float(reward - baseline)


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _is_oom_like_error(msg: str) -> bool:
    text = str(msg or "").lower()
    return ("out of memory" in text) or ("cuda out of memory" in text) or ("hip out of memory" in text)


def _is_runtime_retryable_error(msg: str) -> bool:
    text = str(msg or "").lower()
    if _is_oom_like_error(text):
        return True
    # ROCm/CUDA kernel selection + precision instability that often recovers with
    # lower edge / text-only fallback / autocast-off retries.
    retry_markers = (
        "hipblas_status_invalid_value",
        "hipblasltmatmulalgogetheuristic",
        "cublas_status_not_supported",
        "cublas_status_alloc_failed",
        "cudnn_status_not_supported",
        "mat1 and mat2 must have the same dtype",
        "expected scalar type",
        "runtime precision failure",
    )
    return any(marker in text for marker in retry_markers)


def _build_retry_caps(initial_cap: int, *, min_cap: int) -> List[int]:
    caps: List[int] = []
    curr = max(int(min_cap), int(initial_cap))
    while True:
        if not caps or caps[-1] != curr:
            caps.append(curr)
        if curr <= int(min_cap):
            break
        next_cap = max(int(min_cap), int(round(float(curr) * 0.75)))
        if next_cap >= curr:
            next_cap = max(int(min_cap), curr - 8)
        if next_cap == curr:
            break
        curr = int(next_cap)
    return caps


def _build_understanding_train_batch(
    runtime: BagelRuntime,
    *,
    image: Image.Image,
    prompt: str,
    completion: str,
    policy_max_edge: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    include_image: bool = True,
) -> Optional[Dict]:
    model = runtime.model
    tokenizer = runtime.tokenizer
    new_token_ids = runtime.new_token_ids

    curr_lens = [0]
    curr_rope = [0]
    image_inputs: Optional[Dict] = None
    image_split_len = 0
    if bool(include_image):
        if policy_max_edge is None:
            policy_max_edge = int(os.environ.get("BAGEL_POLICY_MAX_VIT_EDGE", "448") or "448")
        if policy_max_edge > 0:
            w, h = image.size
            max_edge = max(int(w), int(h))
            if max_edge > policy_max_edge:
                scale = float(policy_max_edge) / float(max_edge)
                new_w = max(1, int(round(float(w) * scale)))
                new_h = max(1, int(round(float(h) * scale)))
                image = image.resize((new_w, new_h), resample=Image.BICUBIC)

        image_inputs, curr_lens, curr_rope = model.prepare_vit_images(
            curr_kvlens=curr_lens,
            curr_rope=curr_rope,
            images=[image],
            transforms=runtime.vit_transform,
            new_token_ids=new_token_ids,
        )
        image_split_len = int(image_inputs["packed_seqlens"][0].item())

    prompt_text = str(prompt or "")
    max_prompt_tokens = max(
        8,
        int(os.environ.get("BAGEL_POLICY_MAX_PROMPT_TOKENS", "64") or "64"),
    )
    try:
        prompt_ids = tokenizer.encode(prompt_text)
        if len(prompt_ids) > max_prompt_tokens:
            prompt_text = tokenizer.decode(prompt_ids[:max_prompt_tokens])
    except Exception:
        # Fallback to character clipping if tokenizer decode path fails.
        max_prompt_chars = max(32, int(os.environ.get("BAGEL_POLICY_MAX_PROMPT_CHARS", "256") or "256"))
        if len(prompt_text) > max_prompt_chars:
            prompt_text = prompt_text[:max_prompt_chars]

    prompt_inputs, curr_lens, curr_rope = model.prepare_prompts(
        curr_kvlens=curr_lens,
        curr_rope=curr_rope,
        prompts=[prompt_text],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )

    completion_ids = tokenizer.encode(str(completion or ""))
    if max_completion_tokens is None:
        max_completion_tokens = int(os.environ.get("BAGEL_POLICY_MAX_COMPLETION_TOKENS", "192") or "192")
    max_completion_tokens = max(8, int(max_completion_tokens))
    if len(completion_ids) > max_completion_tokens:
        completion_ids = completion_ids[:max_completion_tokens]
    if not completion_ids:
        return None

    bos = int(new_token_ids["bos_token_id"])
    eos = int(new_token_ids["eos_token_id"])

    shifted_completion_ids = [bos] + completion_ids
    completion_input_ids = shifted_completion_ids + [eos]
    completion_labels = completion_ids + [eos]

    completion_start_idx = int(curr_lens[0])
    completion_start_pos = int(curr_rope[0])
    completion_len = len(completion_input_ids)
    completion_loss_len = len(shifted_completion_ids)
    tensor_device = prompt_inputs["packed_text_ids"].device

    completion_input_ids_t = torch.tensor(completion_input_ids, dtype=torch.long, device=tensor_device)
    completion_indexes_t = torch.arange(
        completion_start_idx,
        completion_start_idx + completion_len,
        dtype=torch.long,
        device=tensor_device,
    )
    completion_pos_t = torch.arange(
        completion_start_pos,
        completion_start_pos + completion_len,
        dtype=torch.long,
        device=tensor_device,
    )
    ce_loss_indexes = torch.arange(
        completion_start_idx,
        completion_start_idx + completion_loss_len,
        dtype=torch.long,
        device=tensor_device,
    )

    prompt_split_len = int(prompt_inputs["text_token_lens"][0].item())

    if bool(include_image) and image_inputs is not None:
        packed_text_ids = torch.cat(
            [
                image_inputs["packed_text_ids"],
                prompt_inputs["packed_text_ids"],
                completion_input_ids_t,
            ],
            dim=0,
        )
        packed_text_indexes = torch.cat(
            [
                image_inputs["packed_text_indexes"],
                prompt_inputs["packed_text_indexes"],
                completion_indexes_t,
            ],
            dim=0,
        )
        packed_position_ids = torch.cat(
            [
                image_inputs["packed_position_ids"],
                prompt_inputs["packed_text_position_ids"],
                completion_pos_t,
            ],
            dim=0,
        )
        split_lens = [int(image_split_len), int(prompt_split_len), int(completion_len)]
        attn_modes = ["full", "causal", "causal"]
    else:
        packed_text_ids = torch.cat(
            [
                prompt_inputs["packed_text_ids"],
                completion_input_ids_t,
            ],
            dim=0,
        )
        packed_text_indexes = torch.cat(
            [
                prompt_inputs["packed_text_indexes"],
                completion_indexes_t,
            ],
            dim=0,
        )
        packed_position_ids = torch.cat(
            [
                prompt_inputs["packed_text_position_ids"],
                completion_pos_t,
            ],
            dim=0,
        )
        split_lens = [int(prompt_split_len), int(completion_len)]
        attn_modes = ["causal", "causal"]

    sequence_length = int(completion_start_idx + completion_len)
    nested_attention_mask = _build_single_sample_attention_mask(
        split_lens=split_lens,
        attn_modes=attn_modes,
        device=tensor_device,
    )

    out = {
        "sequence_length": sequence_length,
        "sample_lens": [sequence_length],
        "nested_attention_masks": [nested_attention_mask],
        "split_lens": split_lens,
        "attn_modes": attn_modes,
        "packed_text_ids": packed_text_ids,
        "packed_text_indexes": packed_text_indexes,
        "packed_position_ids": packed_position_ids,
        "ce_loss_indexes": ce_loss_indexes,
        "packed_label_ids": torch.tensor(completion_labels, dtype=torch.long, device=tensor_device),
    }
    if bool(include_image) and image_inputs is not None:
        out["packed_vit_tokens"] = image_inputs["packed_vit_tokens"]
        out["packed_vit_token_indexes"] = image_inputs["packed_vit_token_indexes"]
        out["packed_vit_position_ids"] = image_inputs["packed_vit_position_ids"]
        out["vit_token_seqlens"] = image_inputs["vit_token_seqlens"]
    return out


def _module_device_dtype(module) -> Tuple[torch.device, torch.dtype]:
    for p in module.parameters():
        return p.device, p.dtype
    for b in module.buffers():
        return b.device, b.dtype
    return torch.device("cpu"), torch.float32


def _build_generation_train_batch(
    runtime: BagelRuntime,
    *,
    image: Image.Image,
    prompt: str,
    policy_max_edge: Optional[int] = None,
    max_prompt_tokens: Optional[int] = None,
) -> Optional[Dict]:
    """Build one-sample visual-generation batch for reward-weighted MSE training."""
    model = runtime.model
    tokenizer = runtime.tokenizer
    new_token_ids = runtime.new_token_ids

    curr_lens = [0]
    curr_rope = [0]

    prompt_text = str(prompt or "")
    if max_prompt_tokens is None:
        max_prompt_tokens = int(
            os.environ.get(
                "BAGEL_GENERATOR_POLICY_MAX_PROMPT_TOKENS",
                os.environ.get("BAGEL_POLICY_MAX_PROMPT_TOKENS", "96"),
            )
            or "96"
        )
    max_prompt_tokens = max(8, int(max_prompt_tokens))
    try:
        prompt_ids = tokenizer.encode(prompt_text)
        if len(prompt_ids) > max_prompt_tokens:
            prompt_text = tokenizer.decode(prompt_ids[:max_prompt_tokens])
    except Exception:
        max_prompt_chars = max(32, int(os.environ.get("BAGEL_GENERATOR_POLICY_MAX_PROMPT_CHARS", "384") or "384"))
        if len(prompt_text) > max_prompt_chars:
            prompt_text = prompt_text[:max_prompt_chars]

    prompt_inputs, curr_lens, curr_rope = model.prepare_prompts(
        curr_kvlens=curr_lens,
        curr_rope=curr_rope,
        prompts=[prompt_text],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    prompt_split_len = int(prompt_inputs["text_token_lens"][0].item())

    if policy_max_edge is None:
        policy_max_edge = int(os.environ.get("BAGEL_GENERATOR_POLICY_MAX_VAE_EDGE", "512") or "512")
    if int(policy_max_edge) > 0:
        w, h = image.size
        max_edge = max(int(w), int(h))
        if max_edge > int(policy_max_edge):
            scale = float(policy_max_edge) / float(max_edge)
            new_w = max(1, int(round(float(w) * scale)))
            new_h = max(1, int(round(float(h) * scale)))
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)

    timestep = float(torch.randn((), device="cpu").item())
    vae_inputs, _, _ = model.prepare_vae_images(
        curr_kvlens=curr_lens,
        curr_rope=curr_rope,
        images=[image],
        transforms=runtime.vae_transform,
        new_token_ids=new_token_ids,
        timestep=timestep,
    )

    num_latent_tokens = int(vae_inputs["packed_vae_token_indexes"].numel())
    if num_latent_tokens <= 0:
        return None

    image_split_len = int(vae_inputs["packed_seqlens"][0].item())
    image_offset = int(prompt_split_len)

    vae_device, vae_dtype = _module_device_dtype(runtime.vae_model)
    with torch.no_grad():
        padded_images = vae_inputs["padded_images"]
        if padded_images.device != vae_device or padded_images.dtype != vae_dtype:
            padded_images = padded_images.to(device=vae_device, dtype=vae_dtype)
        padded_latent = runtime.vae_model.encode(padded_images)
    if padded_latent.device != runtime.device:
        padded_latent = padded_latent.to(runtime.device)

    vae_text_indexes_global = vae_inputs["packed_text_indexes"] + int(image_offset)
    packed_vae_token_indexes = vae_inputs["packed_vae_token_indexes"] + int(image_offset)

    packed_text_ids = torch.cat(
        [
            prompt_inputs["packed_text_ids"],
            vae_inputs["packed_text_ids"],
        ],
        dim=0,
    )
    packed_text_indexes = torch.cat(
        [
            prompt_inputs["packed_text_indexes"],
            vae_text_indexes_global,
        ],
        dim=0,
    )
    packed_position_ids = torch.cat(
        [
            prompt_inputs["packed_text_position_ids"],
            vae_inputs["packed_position_ids"],
        ],
        dim=0,
    )

    sequence_length = int(prompt_split_len + image_split_len)
    split_lens = [int(prompt_split_len), int(image_split_len)]
    attn_modes = ["causal", "noise"]
    mask_device = packed_text_ids.device
    nested_attention_mask = _build_single_sample_attention_mask(
        split_lens=split_lens,
        attn_modes=attn_modes,
        device=mask_device,
    )

    timestep_value = float(vae_inputs["packed_timesteps"][0].item())
    packed_timesteps = torch.full(
        (num_latent_tokens,),
        fill_value=timestep_value,
        dtype=torch.float32,
        device=mask_device,
    )
    latent_token_dim = int(model.latent_patch_size) * int(model.latent_patch_size) * int(model.latent_channel)
    packed_noise = torch.randn(
        (num_latent_tokens, latent_token_dim),
        dtype=padded_latent.dtype,
        device=runtime.device,
    )
    mse_loss_indexes = packed_vae_token_indexes.clone()

    return {
        "sequence_length": sequence_length,
        "sample_lens": [sequence_length],
        "nested_attention_masks": [nested_attention_mask],
        "split_lens": split_lens,
        "attn_modes": attn_modes,
        "packed_text_ids": packed_text_ids,
        "packed_text_indexes": packed_text_indexes,
        "packed_position_ids": packed_position_ids,
        "padded_latent": padded_latent,
        "patchified_vae_latent_shapes": vae_inputs["patchified_vae_latent_shapes"],
        "packed_latent_position_ids": vae_inputs["packed_vae_position_ids"],
        "packed_vae_token_indexes": packed_vae_token_indexes,
        "packed_timesteps": packed_timesteps,
        "packed_noise": packed_noise,
        "mse_loss_indexes": mse_loss_indexes,
    }


@dataclass
class PolicyStepResult:
    skipped: bool
    reason: str
    reward: float
    baseline: float
    advantage: float
    loss: float
    ce_loss: float
    grad_norm: float
    optimizer_step_applied: bool
    token_count: int

    def to_dict(self) -> Dict:
        return {
            "skipped": bool(self.skipped),
            "reason": str(self.reason),
            "reward": float(self.reward),
            "baseline": float(self.baseline),
            "advantage": float(self.advantage),
            "loss": float(self.loss),
            "ce_loss": float(self.ce_loss),
            "grad_norm": float(self.grad_norm),
            "optimizer_step_applied": bool(self.optimizer_step_applied),
            "token_count": int(self.token_count),
        }


class BagelRolePolicyUpdater:
    """Reward-weighted policy updater for one LoRA role adapter."""

    def __init__(
        self,
        *,
        runtime: BagelRuntime,
        cfg: RolloutConfig,
        role: str,
        adapter_name: str,
    ) -> None:
        self.runtime = runtime
        self.cfg = cfg
        self.role = str(role)
        self.adapter_name = str(adapter_name or "")
        self.update_method = cfg.normalized_update_method()
        self.grad_accum_steps = max(1, int(cfg.policy_grad_accum_steps))
        self.kl_coef = float(cfg.kl_coef)
        self.step_id = 0
        self._accum_count = 0
        self._has_grad_window = False

        params = collect_adapter_parameters(runtime.model.language_model, self.adapter_name)
        if not params:
            raise RuntimeError(
                f"No trainable parameters found for role={self.role} adapter={self.adapter_name}."
            )
        self.params = params
        self.optimizer = torch.optim.AdamW(
            self.params,
            lr=float(cfg.policy_lr),
            weight_decay=float(cfg.policy_weight_decay),
        )
        self._rocm_runtime = bool(getattr(torch.version, "hip", None))
        self._rocm_force_text_only = _env_flag("BAGEL_POLICY_ROCM_FORCE_TEXT_ONLY", "1")
        self._empty_cache_each_step = _env_flag(
            "BAGEL_POLICY_EMPTY_CACHE_EACH_STEP",
            "1" if self._rocm_runtime else "0",
        )
        self._oom_force_text_only_steps = max(
            1,
            int(os.environ.get("BAGEL_POLICY_OOM_FORCE_TEXT_ONLY_STEPS", "64") or "64"),
        )
        self._oom_pause_after = max(
            1,
            int(os.environ.get("BAGEL_POLICY_OOM_PAUSE_AFTER_CONSECUTIVE", "6") or "6"),
        )
        self._oom_pause_steps = max(
            1,
            int(os.environ.get("BAGEL_POLICY_OOM_PAUSE_STEPS", "32") or "32"),
        )
        self._consecutive_oom = 0
        self._consecutive_runtime_fail = 0
        self._force_text_only_until_step = 0
        self._pause_until_step = 0

    def state_dict(self) -> Dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "kl_coef": float(self.kl_coef),
            "step_id": int(self.step_id),
            "accum_count": int(self._accum_count),
            "has_grad_window": bool(self._has_grad_window),
            "consecutive_oom": int(self._consecutive_oom),
            "consecutive_runtime_fail": int(self._consecutive_runtime_fail),
            "force_text_only_until_step": int(self._force_text_only_until_step),
            "pause_until_step": int(self._pause_until_step),
        }

    def load_state_dict(self, state: Dict) -> None:
        if not isinstance(state, dict):
            return
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "kl_coef" in state:
            self.kl_coef = float(state["kl_coef"])
        self.step_id = int(state.get("step_id", self.step_id))
        self._accum_count = int(state.get("accum_count", self._accum_count))
        self._has_grad_window = bool(state.get("has_grad_window", self._has_grad_window))
        self._consecutive_oom = int(state.get("consecutive_oom", self._consecutive_oom))
        self._consecutive_runtime_fail = int(
            state.get("consecutive_runtime_fail", self._consecutive_runtime_fail)
        )
        self._force_text_only_until_step = int(
            state.get("force_text_only_until_step", self._force_text_only_until_step)
        )
        self._pause_until_step = int(state.get("pause_until_step", self._pause_until_step))

    def _reset_grad_window(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self._accum_count = 0
        self._has_grad_window = False

    def _adapt_beta(self, kl_val: float) -> None:
        target = max(float(self.cfg.kl_target), 1e-8)
        delta = (float(kl_val) - target) / target
        beta = float(self.kl_coef) * math.exp(float(self.cfg.kl_adapt_rate) * delta)
        beta = max(float(self.cfg.kl_min), min(float(self.cfg.kl_max), beta))
        self.kl_coef = float(beta)

    def finalize(self) -> bool:
        """Flush pending accumulated gradients; returns whether optimizer stepped."""
        if self._accum_count <= 0:
            return False
        stepped = False
        if self._has_grad_window:
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=float(self.cfg.policy_max_grad_norm))
            self.optimizer.step()
            stepped = True
        self._reset_grad_window()
        return stepped

    def step(
        self,
        *,
        image: Image.Image,
        prompt: str,
        completion: str,
        reward: float,
        baseline: float,
        group_rewards: Optional[List[float]] = None,
    ) -> Dict:
        self.step_id += 1

        completion_text = str(completion or "").strip()
        if not completion_text:
            return PolicyStepResult(
                skipped=True,
                reason="empty_completion",
                reward=float(reward),
                baseline=float(baseline),
                advantage=0.0,
                loss=0.0,
                ce_loss=0.0,
                grad_norm=0.0,
                optimizer_step_applied=False,
                token_count=0,
            ).to_dict()

        scaled_reward = float(reward) * float(self.cfg.policy_reward_scale)
        scaled_baseline = float(baseline) * float(self.cfg.policy_reward_scale)
        advantage = _compute_advantage(
            reward=scaled_reward,
            baseline=scaled_baseline,
            method=self.update_method,
            group_rewards=group_rewards,
            eps=float(self.cfg.grpo_eps),
        )

        if self.step_id <= self._pause_until_step:
            return PolicyStepResult(
                skipped=True,
                reason="runtime_pause_cooldown",
                reward=scaled_reward,
                baseline=scaled_baseline,
                advantage=advantage,
                loss=0.0,
                ce_loss=0.0,
                grad_norm=0.0,
                optimizer_step_applied=False,
                token_count=0,
            ).to_dict()

        if self._empty_cache_each_step and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        base_policy_edge = max(64, int(os.environ.get("BAGEL_POLICY_MAX_VIT_EDGE", "448") or "448"))
        min_policy_edge = max(64, int(os.environ.get("BAGEL_POLICY_MIN_VIT_EDGE", "224") or "224"))
        if min_policy_edge > base_policy_edge:
            min_policy_edge = base_policy_edge
        oom_max_retries = max(1, int(os.environ.get("BAGEL_POLICY_OOM_MAX_RETRIES", "3") or "3"))
        edge_decay = float(os.environ.get("BAGEL_POLICY_OOM_EDGE_DECAY", "0.8") or "0.8")
        if edge_decay <= 0.1 or edge_decay >= 1.0:
            edge_decay = 0.8
        max_completion_tokens = max(
            8,
            int(os.environ.get("BAGEL_POLICY_MAX_COMPLETION_TOKENS", "192") or "192"),
        )
        text_only_fallback = str(
            os.environ.get("BAGEL_POLICY_TEXT_ONLY_FALLBACK", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        text_only_max_completion_tokens = max(
            8,
            int(
                os.environ.get(
                    "BAGEL_POLICY_TEXT_ONLY_MAX_COMPLETION_TOKENS",
                    str(min(max_completion_tokens, 96)),
                )
                or str(min(max_completion_tokens, 96))
            ),
        )
        text_only_max_retries = max(
            1,
            int(os.environ.get("BAGEL_POLICY_TEXT_ONLY_MAX_RETRIES", "3") or "3"),
        )
        min_completion_floor = max(
            8,
            int(os.environ.get("BAGEL_POLICY_MIN_COMPLETION_TOKENS", "24") or "24"),
        )
        text_only_mode_requested = _env_flag("BAGEL_POLICY_TEXT_ONLY_MODE", "0")
        text_only_mode_active = bool(
            text_only_mode_requested
            or (self._rocm_runtime and self._rocm_force_text_only)
            or (self.step_id <= self._force_text_only_until_step)
        )
        text_only_caps = _build_retry_caps(
            text_only_max_completion_tokens,
            min_cap=min_completion_floor,
        )[:text_only_max_retries]

        if text_only_mode_active:
            attempt_specs: List[Dict[str, object]] = [
                {"include_image": False, "policy_edge": 0, "completion_cap": int(cap)}
                for cap in text_only_caps
            ]
        else:
            attempt_specs = [
                {
                    "include_image": True,
                    "policy_edge": int(base_policy_edge),
                    "completion_cap": int(max_completion_tokens),
                }
            ]
            while len(attempt_specs) < oom_max_retries:
                prev_edge = int(attempt_specs[-1].get("policy_edge", base_policy_edge))
                next_edge = max(min_policy_edge, int(round(float(prev_edge) * edge_decay)))
                if next_edge >= prev_edge:
                    next_edge = max(min_policy_edge, int(prev_edge) - 32)
                if next_edge == prev_edge:
                    break
                attempt_specs.append(
                    {
                        "include_image": True,
                        "policy_edge": int(next_edge),
                        "completion_cap": int(max_completion_tokens),
                    }
                )
            if text_only_fallback:
                for cap in text_only_caps:
                    attempt_specs.append(
                        {"include_image": False, "policy_edge": 0, "completion_cap": int(cap)}
                    )

        model = self.runtime.model
        was_training = bool(model.training)
        prev_visual_gen = bool(model.config.visual_gen)
        prev_visual_und = bool(model.config.visual_und)
        model.config.visual_gen = False
        model.train(True)

        autocast_enabled = bool(self.cfg.policy_use_bf16)
        token_count = 0
        oom_attempts_this_step = 0
        runtime_retry_attempts = 0

        try:
            for attempt_idx, attempt in enumerate(attempt_specs, start=1):
                include_image = bool(attempt.get("include_image", True))
                policy_edge = int(attempt.get("policy_edge", 0))
                completion_cap = int(
                    attempt.get(
                        "completion_cap",
                        max_completion_tokens if include_image else text_only_max_completion_tokens,
                    )
                )
                batch = _build_understanding_train_batch(
                    self.runtime,
                    image=image,
                    prompt=prompt,
                    completion=completion_text,
                    policy_max_edge=(int(policy_edge) if include_image else None),
                    max_completion_tokens=int(completion_cap),
                    include_image=bool(include_image),
                )
                if batch is None:
                    return PolicyStepResult(
                        skipped=True,
                        reason="empty_completion_ids",
                        reward=scaled_reward,
                        baseline=scaled_baseline,
                        advantage=0.0,
                        loss=0.0,
                        ce_loss=0.0,
                        grad_norm=0.0,
                        optimizer_step_applied=False,
                        token_count=0,
                    ).to_dict()

                batch = _to_device(batch, self.runtime.device)
                token_count = int(batch["packed_label_ids"].numel())
                grad_norm = 0.0
                opt_step = False
                ce_value = 0.0
                loss_value = 0.0

                try:
                    ref_logits = None
                    model.config.visual_und = bool(include_image)
                    model.train(False)
                    with torch.no_grad():
                        with use_adapter(self.runtime.model.language_model, None):
                            with autocast_context(self.runtime.device, enabled=autocast_enabled):
                                ref_outputs = model(
                                    **batch,
                                    return_ce_logits=True,
                                )
                                ref_logits = ref_outputs.get("ce_logits", None)

                    model.train(True)
                    with use_adapter(self.runtime.model.language_model, self.adapter_name):
                        model.config.visual_und = bool(include_image)
                        with autocast_context(self.runtime.device, enabled=autocast_enabled):
                            outputs = model(
                                **batch,
                                return_ce_logits=True,
                            )
                            ce_loss = outputs.get("ce", None)
                            if ce_loss is None or int(ce_loss.numel()) <= 0:
                                return PolicyStepResult(
                                    skipped=True,
                                    reason="empty_ce_loss",
                                    reward=scaled_reward,
                                    baseline=scaled_baseline,
                                    advantage=advantage,
                                    loss=0.0,
                                    ce_loss=0.0,
                                    grad_norm=0.0,
                                    optimizer_step_applied=False,
                                    token_count=token_count,
                                ).to_dict()
                            ce_mean = ce_loss.mean()
                            pi_logits = outputs.get("ce_logits", None)
                            if (
                                ref_logits is not None
                                and pi_logits is not None
                                and int(ref_logits.numel()) > 0
                                and int(pi_logits.numel()) > 0
                            ):
                                logp_pi = F.log_softmax(pi_logits, dim=-1)
                                logp_ref = F.log_softmax(ref_logits, dim=-1)
                                kl_loss = (logp_pi.exp() * (logp_pi - logp_ref)).sum(dim=-1).mean()
                                kl_loss = kl_loss.to(dtype=ce_mean.dtype)
                            else:
                                kl_loss = torch.zeros((), device=ce_mean.device, dtype=ce_mean.dtype)
                            beta_before = float(self.kl_coef)
                            # ce_mean is negative log-likelihood. A positive
                            # advantage should lower CE and increase the
                            # completion probability; a negative advantage
                            # should do the opposite.
                            loss = float(advantage) * ce_mean + beta_before * kl_loss

                        if not bool(torch.isfinite(loss.detach()).all().item()):
                            kl_value = float(kl_loss.detach().item()) if "kl_loss" in locals() else 0.0
                            return PolicyStepResult(
                                skipped=True,
                                reason="non_finite_loss",
                                reward=scaled_reward,
                                baseline=scaled_baseline,
                                advantage=advantage,
                                loss=float(loss.detach().item()),
                                ce_loss=float(ce_mean.detach().item()),
                                grad_norm=0.0,
                                optimizer_step_applied=False,
                                token_count=token_count,
                            ).to_dict() | {
                                "kl_loss": float(kl_value),
                                "kl_coef_before": float(beta_before),
                                "kl_coef_after": float(self.kl_coef),
                            }

                        ce_value = float(ce_mean.detach().item())
                        kl_value = float(kl_loss.detach().item())
                        loss_value = float(loss.detach().item())
                        scaled_loss = loss / float(self.grad_accum_steps)
                        scaled_loss.backward()
                        self._accum_count += 1
                        self._has_grad_window = True

                        if self._accum_count >= self.grad_accum_steps:
                            grad_norm = float(
                                torch.nn.utils.clip_grad_norm_(
                                    self.params,
                                    max_norm=float(self.cfg.policy_max_grad_norm),
                                ).item()
                            )
                            self.optimizer.step()
                            opt_step = True
                            self._reset_grad_window()

                    self._adapt_beta(kl_value)
                    self._consecutive_oom = 0
                    self._consecutive_runtime_fail = 0
                    return PolicyStepResult(
                        skipped=False,
                        reason="ok",
                        reward=scaled_reward,
                        baseline=scaled_baseline,
                        advantage=advantage,
                        loss=loss_value,
                        ce_loss=ce_value,
                        grad_norm=float(grad_norm),
                        optimizer_step_applied=bool(opt_step),
                        token_count=token_count,
                    ).to_dict() | {
                        "kl_loss": float(kl_value),
                        "kl_coef_before": float(beta_before),
                        "kl_coef_after": float(self.kl_coef),
                    }
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    oom_like = _is_oom_like_error(msg)
                    retryable = _is_runtime_retryable_error(msg)
                    if not retryable:
                        raise
                    if oom_like:
                        oom_attempts_this_step += 1
                    else:
                        runtime_retry_attempts += 1
                        if autocast_enabled:
                            autocast_enabled = False
                            print(
                                f"[policy_updater][role={self.role}] runtime precision/kernel failure; "
                                "disabling autocast for retry path."
                            )
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                    self._reset_grad_window()
                    if attempt_idx < len(attempt_specs):
                        next_attempt = attempt_specs[attempt_idx]
                        next_include_image = bool(next_attempt.get("include_image", True))
                        next_cap = int(
                            next_attempt.get(
                                "completion_cap",
                                max_completion_tokens
                                if next_include_image
                                else text_only_max_completion_tokens,
                            )
                        )
                        if next_include_image:
                            next_edge = int(next_attempt.get("policy_edge", 0))
                            print(
                                f"[policy_updater][role={self.role}] "
                                f"{'OOM' if oom_like else 'runtime'} failure at max_vit_edge={policy_edge}; "
                                f"retrying with max_vit_edge={next_edge}, completion_cap={next_cap}."
                            )
                        else:
                            print(
                                f"[policy_updater][role={self.role}] "
                                f"{'OOM' if oom_like else 'runtime'} failure at max_vit_edge={policy_edge}; "
                                f"retrying with text-only policy fallback, completion_cap={next_cap}."
                            )
                        continue
                    reason = "runtime_retry_exhausted"
                    if oom_like:
                        self._consecutive_oom += 1
                        reason = "cuda_oom"
                    else:
                        self._consecutive_runtime_fail += 1
                    if include_image:
                        self._force_text_only_until_step = max(
                            int(self._force_text_only_until_step),
                            int(self.step_id + self._oom_force_text_only_steps),
                        )
                        if oom_like:
                            reason = "cuda_oom_force_text_only"
                        else:
                            reason = "runtime_force_text_only"
                        print(
                            f"[policy_updater][role={self.role}] forcing text-only updates for next "
                            f"{self._oom_force_text_only_steps} steps."
                        )
                    if (
                        (oom_like and self._consecutive_oom >= self._oom_pause_after)
                        or ((not oom_like) and self._consecutive_runtime_fail >= self._oom_pause_after)
                    ):
                        self._pause_until_step = int(self.step_id + self._oom_pause_steps)
                        self._consecutive_oom = 0
                        self._consecutive_runtime_fail = 0
                        reason = "cuda_oom_pause" if oom_like else "runtime_pause"
                        print(
                            f"[policy_updater][role={self.role}] pausing policy updates for "
                            f"{self._oom_pause_steps} steps after repeated "
                            f"{'OOM' if oom_like else 'runtime failures'}."
                        )
                    return PolicyStepResult(
                        skipped=True,
                        reason=reason,
                        reward=scaled_reward,
                        baseline=scaled_baseline,
                        advantage=advantage,
                        loss=loss_value,
                        ce_loss=ce_value,
                        grad_norm=0.0,
                        optimizer_step_applied=False,
                        token_count=token_count,
                    ).to_dict() | {
                        "kl_loss": 0.0,
                        "kl_coef_before": float(self.kl_coef),
                        "kl_coef_after": float(self.kl_coef),
                    }
        finally:
            model.config.visual_gen = prev_visual_gen
            model.config.visual_und = prev_visual_und
            model.train(was_training)

        if oom_attempts_this_step <= 0:
            self._consecutive_oom = 0

        return PolicyStepResult(
            skipped=True,
            reason="unknown_retry_exit",
            reward=scaled_reward,
            baseline=scaled_baseline,
            advantage=advantage,
            loss=0.0,
            ce_loss=0.0,
            grad_norm=0.0,
            optimizer_step_applied=False,
            token_count=token_count,
        ).to_dict() | {
            "kl_loss": 0.0,
            "kl_coef_before": float(self.kl_coef),
            "kl_coef_after": float(self.kl_coef),
        }


class BagelGeneratorPolicyUpdater(BagelRolePolicyUpdater):
    """Reward-weighted generator updater on BAGEL visual-generation (MSE) path."""

    def step(
        self,
        *,
        image: Image.Image,
        prompt: str,
        reward: float,
        baseline: float,
        group_rewards: Optional[List[float]] = None,
    ) -> Dict:
        self.step_id += 1

        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return PolicyStepResult(
                skipped=True,
                reason="empty_prompt",
                reward=float(reward),
                baseline=float(baseline),
                advantage=0.0,
                loss=0.0,
                ce_loss=0.0,
                grad_norm=0.0,
                optimizer_step_applied=False,
                token_count=0,
            ).to_dict()

        scaled_reward = float(reward) * float(self.cfg.policy_reward_scale)
        scaled_baseline = float(baseline) * float(self.cfg.policy_reward_scale)
        advantage = _compute_advantage(
            reward=scaled_reward,
            baseline=scaled_baseline,
            method=self.update_method,
            group_rewards=group_rewards,
            eps=float(self.cfg.grpo_eps),
        )

        if self.step_id <= self._pause_until_step:
            return PolicyStepResult(
                skipped=True,
                reason="runtime_pause_cooldown",
                reward=scaled_reward,
                baseline=scaled_baseline,
                advantage=advantage,
                loss=0.0,
                ce_loss=0.0,
                grad_norm=0.0,
                optimizer_step_applied=False,
                token_count=0,
            ).to_dict()

        if self._empty_cache_each_step and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        base_policy_edge = max(
            96,
            int(os.environ.get("BAGEL_GENERATOR_POLICY_MAX_VAE_EDGE", "512") or "512"),
        )
        min_policy_edge = max(
            64,
            int(os.environ.get("BAGEL_GENERATOR_POLICY_MIN_VAE_EDGE", "160") or "160"),
        )
        if min_policy_edge > base_policy_edge:
            min_policy_edge = base_policy_edge
        oom_max_retries = max(
            1,
            int(os.environ.get("BAGEL_GENERATOR_POLICY_OOM_MAX_RETRIES", "3") or "3"),
        )
        edge_decay = float(os.environ.get("BAGEL_GENERATOR_POLICY_OOM_EDGE_DECAY", "0.8") or "0.8")
        if edge_decay <= 0.1 or edge_decay >= 1.0:
            edge_decay = 0.8
        max_prompt_tokens = max(
            8,
            int(
                os.environ.get(
                    "BAGEL_GENERATOR_POLICY_MAX_PROMPT_TOKENS",
                    os.environ.get("BAGEL_POLICY_MAX_PROMPT_TOKENS", "96"),
                )
                or "96"
            ),
        )

        attempt_specs = [int(base_policy_edge)]
        while len(attempt_specs) < oom_max_retries:
            prev_edge = int(attempt_specs[-1])
            next_edge = max(min_policy_edge, int(round(float(prev_edge) * edge_decay)))
            if next_edge >= prev_edge:
                next_edge = max(min_policy_edge, int(prev_edge) - 32)
            if next_edge == prev_edge:
                break
            attempt_specs.append(int(next_edge))

        model = self.runtime.model
        was_training = bool(model.training)
        prev_visual_gen = bool(model.config.visual_gen)
        prev_visual_und = bool(model.config.visual_und)
        model.config.visual_gen = True
        model.config.visual_und = False
        model.train(True)

        autocast_enabled = bool(self.cfg.policy_use_bf16)
        token_count = 0
        mse_value = 0.0
        loss_value = 0.0

        try:
            for attempt_idx, policy_edge in enumerate(attempt_specs, start=1):
                batch = _build_generation_train_batch(
                    self.runtime,
                    image=image,
                    prompt=prompt_text,
                    policy_max_edge=int(policy_edge),
                    max_prompt_tokens=int(max_prompt_tokens),
                )
                if batch is None:
                    out = PolicyStepResult(
                        skipped=True,
                        reason="empty_generation_batch",
                        reward=scaled_reward,
                        baseline=scaled_baseline,
                        advantage=advantage,
                        loss=0.0,
                        ce_loss=0.0,
                        grad_norm=0.0,
                        optimizer_step_applied=False,
                        token_count=0,
                    ).to_dict()
                    out["mse_loss"] = 0.0
                    out["objective"] = "mse"
                    return out

                batch = _to_device(batch, self.runtime.device)
                token_count = int(batch["packed_vae_token_indexes"].numel())
                grad_norm = 0.0
                opt_step = False

                try:
                    ref_preds = None
                    model.train(False)
                    with torch.no_grad():
                        with use_adapter(self.runtime.model.language_model, None):
                            with autocast_context(self.runtime.device, enabled=autocast_enabled):
                                ref_outputs = model(
                                    **batch,
                                    return_mse_preds=True,
                                )
                                ref_preds = ref_outputs.get("mse_preds", None)

                    model.train(True)
                    with use_adapter(self.runtime.model.language_model, self.adapter_name):
                        with autocast_context(self.runtime.device, enabled=autocast_enabled):
                            outputs = model(
                                **batch,
                                return_mse_preds=True,
                            )
                            mse = outputs.get("mse", None)
                            if mse is None or int(mse.numel()) <= 0:
                                out = PolicyStepResult(
                                    skipped=True,
                                    reason="empty_mse_loss",
                                    reward=scaled_reward,
                                    baseline=scaled_baseline,
                                    advantage=advantage,
                                    loss=0.0,
                                    ce_loss=0.0,
                                    grad_norm=0.0,
                                    optimizer_step_applied=False,
                                    token_count=token_count,
                                ).to_dict()
                                out["mse_loss"] = 0.0
                                out["kl_loss"] = 0.0
                                out["kl_coef_before"] = float(self.kl_coef)
                                out["kl_coef_after"] = float(self.kl_coef)
                                out["objective"] = "mse"
                                return out
                            mse_mean = mse.mean()
                            pi_preds = outputs.get("mse_preds", None)
                            if (
                                ref_preds is not None
                                and pi_preds is not None
                                and int(ref_preds.numel()) > 0
                                and int(pi_preds.numel()) > 0
                            ):
                                kl_loss = F.mse_loss(pi_preds, ref_preds, reduction="mean")
                                kl_loss = kl_loss.to(dtype=mse_mean.dtype)
                            else:
                                kl_loss = torch.zeros((), device=mse_mean.device, dtype=mse_mean.dtype)
                            beta_before = float(self.kl_coef)
                            loss = mse_mean * float(advantage) + beta_before * kl_loss

                        if not bool(torch.isfinite(loss.detach()).all().item()):
                            out = PolicyStepResult(
                                skipped=True,
                                reason="non_finite_loss",
                                reward=scaled_reward,
                                baseline=scaled_baseline,
                                advantage=advantage,
                                loss=float(loss.detach().item()),
                                ce_loss=float(mse_mean.detach().item()),
                                grad_norm=0.0,
                                optimizer_step_applied=False,
                                token_count=token_count,
                            ).to_dict()
                            out["mse_loss"] = float(mse_mean.detach().item())
                            out["kl_loss"] = float(kl_loss.detach().item())
                            out["kl_coef_before"] = float(beta_before)
                            out["kl_coef_after"] = float(self.kl_coef)
                            out["objective"] = "mse"
                            return out

                        mse_value = float(mse_mean.detach().item())
                        kl_value = float(kl_loss.detach().item())
                        loss_value = float(loss.detach().item())
                        scaled_loss = loss / float(self.grad_accum_steps)
                        scaled_loss.backward()
                        self._accum_count += 1
                        self._has_grad_window = True

                        if self._accum_count >= self.grad_accum_steps:
                            grad_norm = float(
                                torch.nn.utils.clip_grad_norm_(
                                    self.params,
                                    max_norm=float(self.cfg.policy_max_grad_norm),
                                ).item()
                            )
                            self.optimizer.step()
                            opt_step = True
                            self._reset_grad_window()

                    self._adapt_beta(kl_value)
                    self._consecutive_oom = 0
                    self._consecutive_runtime_fail = 0
                    out = PolicyStepResult(
                        skipped=False,
                        reason="ok",
                        reward=scaled_reward,
                        baseline=scaled_baseline,
                        advantage=advantage,
                        loss=loss_value,
                        ce_loss=mse_value,
                        grad_norm=float(grad_norm),
                        optimizer_step_applied=bool(opt_step),
                        token_count=token_count,
                    ).to_dict()
                    out["mse_loss"] = float(mse_value)
                    out["kl_loss"] = float(kl_value)
                    out["kl_coef_before"] = float(beta_before)
                    out["kl_coef_after"] = float(self.kl_coef)
                    out["objective"] = "mse"
                    return out
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    oom_like = _is_oom_like_error(msg)
                    retryable = _is_runtime_retryable_error(msg)
                    if not retryable:
                        raise
                    if (not oom_like) and autocast_enabled:
                        autocast_enabled = False
                        print(
                            f"[policy_updater][role={self.role}] runtime precision/kernel failure; "
                            "disabling autocast for retry path."
                        )
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                    self._reset_grad_window()
                    if attempt_idx < len(attempt_specs):
                        next_edge = int(attempt_specs[attempt_idx])
                        print(
                            f"[policy_updater][role={self.role}] "
                            f"{'OOM' if oom_like else 'runtime'} failure at max_vae_edge={policy_edge}; "
                            f"retrying with max_vae_edge={next_edge}."
                        )
                        continue
                    reason = "runtime_retry_exhausted"
                    if oom_like:
                        self._consecutive_oom += 1
                        reason = "cuda_oom"
                    else:
                        self._consecutive_runtime_fail += 1
                    if (
                        (oom_like and self._consecutive_oom >= self._oom_pause_after)
                        or ((not oom_like) and self._consecutive_runtime_fail >= self._oom_pause_after)
                    ):
                        self._pause_until_step = int(self.step_id + self._oom_pause_steps)
                        self._consecutive_oom = 0
                        self._consecutive_runtime_fail = 0
                        reason = "cuda_oom_pause" if oom_like else "runtime_pause"
                        print(
                            f"[policy_updater][role={self.role}] pausing policy updates for "
                            f"{self._oom_pause_steps} steps after repeated "
                            f"{'OOM' if oom_like else 'runtime failures'}."
                        )
                    out = PolicyStepResult(
                        skipped=True,
                        reason=reason,
                        reward=scaled_reward,
                        baseline=scaled_baseline,
                        advantage=advantage,
                        loss=loss_value,
                        ce_loss=mse_value,
                        grad_norm=0.0,
                        optimizer_step_applied=False,
                        token_count=token_count,
                    ).to_dict()
                    out["mse_loss"] = float(mse_value)
                    out["kl_loss"] = 0.0
                    out["kl_coef_before"] = float(self.kl_coef)
                    out["kl_coef_after"] = float(self.kl_coef)
                    out["objective"] = "mse"
                    return out
        finally:
            model.config.visual_gen = prev_visual_gen
            model.config.visual_und = prev_visual_und
            model.train(was_training)

        out = PolicyStepResult(
            skipped=True,
            reason="unknown_retry_exit",
            reward=scaled_reward,
            baseline=scaled_baseline,
            advantage=advantage,
            loss=0.0,
            ce_loss=0.0,
            grad_norm=0.0,
            optimizer_step_applied=False,
            token_count=token_count,
        ).to_dict()
        out["mse_loss"] = 0.0
        out["kl_loss"] = 0.0
        out["kl_coef_before"] = float(self.kl_coef)
        out["kl_coef_after"] = float(self.kl_coef)
        out["objective"] = "mse"
        return out
