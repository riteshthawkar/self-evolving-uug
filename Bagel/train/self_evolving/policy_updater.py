# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
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

    prompt_inputs, curr_lens, curr_rope = model.prepare_prompts(
        curr_kvlens=curr_lens,
        curr_rope=curr_rope,
        prompts=[str(prompt or "")],
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

    def state_dict(self) -> Dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_id": int(self.step_id),
            "accum_count": int(self._accum_count),
            "has_grad_window": bool(self._has_grad_window),
        }

    def load_state_dict(self, state: Dict) -> None:
        if not isinstance(state, dict):
            return
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.step_id = int(state.get("step_id", self.step_id))
        self._accum_count = int(state.get("accum_count", self._accum_count))
        self._has_grad_window = bool(state.get("has_grad_window", self._has_grad_window))

    def _reset_grad_window(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self._accum_count = 0
        self._has_grad_window = False

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

        attempt_specs: List[Dict[str, object]] = [{"include_image": True, "policy_edge": int(base_policy_edge)}]
        while len(attempt_specs) < oom_max_retries:
            prev_edge = int(attempt_specs[-1]["policy_edge"])
            next_edge = max(min_policy_edge, int(round(float(prev_edge) * edge_decay)))
            if next_edge >= prev_edge:
                next_edge = max(min_policy_edge, int(prev_edge) - 32)
            if next_edge == prev_edge:
                break
            attempt_specs.append({"include_image": True, "policy_edge": int(next_edge)})
        if text_only_fallback:
            attempt_specs.append({"include_image": False, "policy_edge": 0})

        model = self.runtime.model
        was_training = bool(model.training)
        prev_visual_gen = bool(model.config.visual_gen)
        model.config.visual_gen = False
        model.train(True)

        autocast_enabled = bool(self.cfg.policy_use_bf16)
        token_count = 0

        try:
            for attempt_idx, attempt in enumerate(attempt_specs, start=1):
                include_image = bool(attempt.get("include_image", True))
                policy_edge = int(attempt.get("policy_edge", 0))
                completion_cap = (
                    int(max_completion_tokens)
                    if include_image
                    else int(text_only_max_completion_tokens)
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
                    with use_adapter(self.runtime.model.language_model, self.adapter_name):
                        model.config.visual_und = bool(include_image)
                        with autocast_context(self.runtime.device, enabled=autocast_enabled):
                            outputs = model(**batch)
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
                            loss = ce_mean * float(advantage)

                        if not bool(torch.isfinite(loss.detach()).all().item()):
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
                            ).to_dict()

                        ce_value = float(ce_mean.detach().item())
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
                    ).to_dict()
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    oom_like = ("out of memory" in msg) or ("cuda out of memory" in msg) or ("hip out of memory" in msg)
                    if not oom_like:
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._reset_grad_window()
                    if attempt_idx < len(attempt_specs):
                        next_attempt = attempt_specs[attempt_idx]
                        if bool(next_attempt.get("include_image", True)):
                            next_edge = int(next_attempt.get("policy_edge", 0))
                            print(
                                f"[policy_updater][role={self.role}] OOM at max_vit_edge={policy_edge}; "
                                f"retrying with max_vit_edge={next_edge}."
                            )
                        else:
                            print(
                                f"[policy_updater][role={self.role}] OOM at max_vit_edge={policy_edge}; "
                                "retrying with text-only policy fallback."
                            )
                        continue
                    return PolicyStepResult(
                        skipped=True,
                        reason="cuda_oom",
                        reward=scaled_reward,
                        baseline=scaled_baseline,
                        advantage=advantage,
                        loss=loss_value,
                        ce_loss=ce_value,
                        grad_norm=0.0,
                        optimizer_step_applied=False,
                        token_count=token_count,
                    ).to_dict()
        finally:
            model.config.visual_gen = prev_visual_gen
            model.train(was_training)

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
        ).to_dict()
