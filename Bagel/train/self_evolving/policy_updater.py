# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from PIL import Image

from modeling.bagel.runtime_precision import autocast_context

from .adapter_manager import collect_adapter_parameters, use_adapter
from .config import RolloutConfig
from .model_loader import BagelRuntime


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
) -> Optional[Dict]:
    model = runtime.model
    tokenizer = runtime.tokenizer
    new_token_ids = runtime.new_token_ids

    curr_lens = [0]
    curr_rope = [0]

    image_inputs, curr_lens, curr_rope = model.prepare_vit_images(
        curr_kvlens=curr_lens,
        curr_rope=curr_rope,
        images=[image],
        transforms=runtime.vit_transform,
        new_token_ids=new_token_ids,
    )

    prompt_inputs, curr_lens, curr_rope = model.prepare_prompts(
        curr_kvlens=curr_lens,
        curr_rope=curr_rope,
        prompts=[str(prompt or "")],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )

    completion_ids = tokenizer.encode(str(completion or ""))
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
    tensor_device = image_inputs["packed_text_ids"].device

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

    image_split_len = int(image_inputs["packed_seqlens"][0].item())
    prompt_split_len = int(prompt_inputs["text_token_lens"][0].item())

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

    sequence_length = int(completion_start_idx + completion_len)
    split_lens = [int(image_split_len), int(prompt_split_len), int(completion_len)]
    attn_modes = ["full", "causal", "causal"]

    return {
        "sequence_length": sequence_length,
        "sample_lens": split_lens,
        "split_lens": split_lens,
        "attn_modes": attn_modes,
        "packed_text_ids": packed_text_ids,
        "packed_text_indexes": packed_text_indexes,
        "packed_position_ids": packed_position_ids,
        "packed_vit_tokens": image_inputs["packed_vit_tokens"],
        "packed_vit_token_indexes": image_inputs["packed_vit_token_indexes"],
        "packed_vit_position_ids": image_inputs["packed_vit_position_ids"],
        "vit_token_seqlens": image_inputs["vit_token_seqlens"],
        "ce_loss_indexes": ce_loss_indexes,
        "packed_label_ids": torch.tensor(completion_labels, dtype=torch.long, device=tensor_device),
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

        batch = _build_understanding_train_batch(
            self.runtime,
            image=image,
            prompt=prompt,
            completion=completion_text,
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

        model = self.runtime.model
        was_training = bool(model.training)
        prev_visual_gen = bool(model.config.visual_gen)
        model.config.visual_gen = False
        model.train(True)

        autocast_enabled = bool(self.cfg.policy_use_bf16)
        grad_norm = 0.0
        opt_step = False
        ce_value = 0.0
        loss_value = 0.0

        try:
            with use_adapter(self.runtime.model.language_model, self.adapter_name):
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
                        torch.nn.utils.clip_grad_norm_(self.params, max_norm=float(self.cfg.policy_max_grad_norm)).item()
                    )
                    self.optimizer.step()
                    opt_step = True
                    self._reset_grad_window()
        finally:
            model.config.visual_gen = prev_visual_gen
            model.train(was_training)

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
