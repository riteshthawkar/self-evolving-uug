# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Optional

from PIL import Image
import torch

from .adapter_manager import ROLE_GENERATOR, ROLE_PROPOSER, ROLE_SOLVER, use_adapter
from .model_loader import BagelRuntime
from .prompts import (
    build_generation_spec_prompt,
    build_proposer_multi_prompt,
    build_proposer_prompt,
    build_solver_prompt,
    build_solver_prompt_pps,
)


@dataclass
class GenerationResult:
    text: str
    raw: dict


class BagelRolloutAdapter:
    """Thin adapter around InterleaveInferencer for self-evolving rollouts."""

    def __init__(self, runtime: BagelRuntime) -> None:
        self.runtime = runtime
        self.inferencer = runtime.inferencer
        self._proposer_top_p = self._env_float("BAGEL_PROPOSER_TEXT_TOP_P", self._env_float("BAGEL_TEXT_TOP_P", 0.92))
        self._proposer_top_k = self._env_int("BAGEL_PROPOSER_TEXT_TOP_K", self._env_int("BAGEL_TEXT_TOP_K", 40))
        self._solver_top_p = self._env_float("BAGEL_SOLVER_TEXT_TOP_P", self._env_float("BAGEL_TEXT_TOP_P", 0.92))
        self._solver_top_k = self._env_int("BAGEL_SOLVER_TEXT_TOP_K", self._env_int("BAGEL_TEXT_TOP_K", 40))
        self._gen_spec_top_p = self._env_float("BAGEL_GEN_SPEC_TEXT_TOP_P", self._proposer_top_p)
        self._gen_spec_top_k = self._env_int("BAGEL_GEN_SPEC_TEXT_TOP_K", self._proposer_top_k)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = str(os.environ.get(name, str(default))).strip()
        try:
            return float(raw)
        except Exception:
            return float(default)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = str(os.environ.get(name, str(default))).strip()
        try:
            return int(raw)
        except Exception:
            return int(default)

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return ("out of memory" in msg) or ("cuda out of memory" in msg) or ("hip out of memory" in msg)

    @staticmethod
    def _clear_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _resize_image_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
        max_edge = int(max_edge)
        if max_edge <= 0:
            return image
        width, height = image.size
        longest = max(int(width), int(height))
        if longest <= max_edge:
            return image
        scale = float(max_edge) / float(longest)
        new_width = max(1, int(round(float(width) * scale)))
        new_height = max(1, int(round(float(height) * scale)))
        return image.resize((new_width, new_height), resample=Image.BICUBIC)

    def _build_understanding_retry_plan(self, image: Image.Image) -> list[int]:
        default_edge = self._env_int(
            "BAGEL_UNDERSTANDING_MAX_VIT_EDGE",
            self._env_int("BAGEL_POLICY_MAX_VIT_EDGE", 448),
        )
        min_edge = max(
            128,
            self._env_int("BAGEL_UNDERSTANDING_MIN_VIT_EDGE", self._env_int("BAGEL_POLICY_MIN_VIT_EDGE", 224)),
        )
        decay = self._env_float("BAGEL_UNDERSTANDING_EDGE_DECAY", 0.8)
        if decay <= 0.0 or decay >= 1.0:
            decay = 0.8
        max_attempts = max(1, self._env_int("BAGEL_UNDERSTANDING_OOM_MAX_RETRIES", 3))

        start_edge = max(min_edge, min(max(int(image.size[0]), int(image.size[1])), int(default_edge)))
        plan = [start_edge]
        for _ in range(max_attempts - 1):
            next_edge = max(min_edge, int(round(float(plan[-1]) * decay)))
            if next_edge >= plan[-1]:
                next_edge = max(min_edge, plan[-1] - 32)
            if next_edge < min_edge or next_edge == plan[-1]:
                break
            plan.append(next_edge)
        return plan

    @staticmethod
    def _build_generation_retry_plan(
        *,
        image_size: int,
        num_timesteps: int,
    ) -> list[tuple[int, int]]:
        max_attempts = max(1, int(os.environ.get("BAGEL_GEN_OOM_MAX_RETRIES", "4") or "4"))
        min_image = max(256, int(os.environ.get("BAGEL_GEN_MIN_IMAGE_SIZE", "256") or "256"))
        min_steps = max(8, int(os.environ.get("BAGEL_GEN_MIN_TIMESTEPS", "16") or "16"))
        timestep_decay = float(os.environ.get("BAGEL_GEN_TIMESTEP_DECAY", "0.75") or "0.75")
        if timestep_decay <= 0.0 or timestep_decay >= 1.0:
            timestep_decay = 0.75

        size = max(min_image, int(image_size))
        # Default ROCm safety clamp to reduce OOM risk in VAE decode.
        if getattr(torch.version, "hip", None):
            rocm_cap = max(min_image, int(os.environ.get("BAGEL_GEN_ROCM_MAX_IMAGE_SIZE", "512") or "512"))
            size = min(size, rocm_cap)
        steps = max(min_steps, int(num_timesteps))

        plan: list[tuple[int, int]] = [(size, steps)]
        for _ in range(max_attempts - 1):
            next_size = max(min_image, (size * 4) // 5)  # 20% downscale each retry
            next_steps = max(min_steps, int(round(float(steps) * timestep_decay)))
            # Ensure progress if one dimension cannot shrink further.
            if next_size == size and next_steps == steps:
                if next_steps > min_steps:
                    next_steps -= 1
                elif next_size > min_image:
                    next_size -= 8
                else:
                    break
            size, steps = next_size, next_steps
            if (size, steps) != plan[-1]:
                plan.append((size, steps))
        return plan

    def _adapter_for_role(self, role: str) -> str:
        if not bool(self.runtime.lora_enabled):
            return ""
        return str(self.runtime.adapter_for_role(role))

    def _generate_understanding_text(
        self,
        *,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        text_top_p: float = 1.0,
        text_top_k: int = 0,
    ) -> GenerationResult:
        attempt_edges = self._build_understanding_retry_plan(image)
        last_exc: Optional[BaseException] = None
        for attempt_idx, max_edge in enumerate(attempt_edges, start=1):
            try:
                out = self.inferencer(
                    image=self._resize_image_max_edge(image, max_edge),
                    text=prompt,
                    understanding_output=True,
                    think=False,
                    max_think_token_n=max_new_tokens,
                    do_sample=do_sample,
                    text_temperature=temperature,
                    text_top_p=float(text_top_p),
                    text_top_k=int(text_top_k),
                )
                text = str(out.get("text") or "").strip()
                return GenerationResult(text=text, raw=out)
            except RuntimeError as exc:
                if not self._is_oom_error(exc):
                    raise
                last_exc = exc
                if attempt_idx >= len(attempt_edges):
                    raise
                print(
                    f"[rollout_adapter] OOM during understanding inference; "
                    f"retrying with max_vit_edge={attempt_edges[attempt_idx]}."
                )
                self._clear_memory()
        if last_exc is not None:
            raise last_exc
        return GenerationResult(text="", raw={})

    def propose_question(
        self,
        *,
        image: Image.Image,
        max_new_tokens: int,
        temperature: float,
        target_difficulty: str = "medium",
        do_sample: bool = True,
    ) -> GenerationResult:
        proposer_prompt = build_proposer_prompt(target_difficulty=target_difficulty)
        with use_adapter(self.runtime.model.language_model, self._adapter_for_role(ROLE_PROPOSER)):
            return self._generate_understanding_text(
                image=image,
                prompt=proposer_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=bool(do_sample),
                temperature=temperature,
                text_top_p=self._proposer_top_p,
                text_top_k=self._proposer_top_k,
            )

    def propose_questions(
        self,
        *,
        image: Image.Image,
        max_new_tokens: int,
        temperature: float,
        num_questions: int,
        target_difficulty: str = "medium",
        do_sample: bool = True,
    ) -> GenerationResult:
        n = max(1, int(num_questions))
        if n <= 1:
            return self.propose_question(
                image=image,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                target_difficulty=target_difficulty,
                do_sample=bool(do_sample),
            )
        proposer_prompt = build_proposer_multi_prompt(
            num_questions=n,
            target_difficulty=target_difficulty,
        )
        with use_adapter(self.runtime.model.language_model, self._adapter_for_role(ROLE_PROPOSER)):
            return self._generate_understanding_text(
                image=image,
                prompt=proposer_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=bool(do_sample),
                temperature=temperature,
                text_top_p=self._proposer_top_p,
                text_top_k=self._proposer_top_k,
            )

    def solve_question(
        self,
        *,
        image: Image.Image,
        question: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        template_index: int = 0,
        focus_hint: str = "",
        use_pps: bool = False,
    ) -> GenerationResult:
        if bool(use_pps):
            solver_prompt = build_solver_prompt_pps(
                question_text=question,
                template_index=int(template_index),
                focus_hint=focus_hint,
            )
        else:
            solver_prompt = build_solver_prompt(question_text=question, focus_hint=focus_hint)
        with use_adapter(self.runtime.model.language_model, self._adapter_for_role(ROLE_SOLVER)):
            return self._generate_understanding_text(
                image=image,
                prompt=solver_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                text_top_p=self._solver_top_p,
                text_top_k=self._solver_top_k,
            )

    def intuitive_answer(
        self,
        *,
        image: Image.Image,
        question: str,
        max_new_tokens: int,
    ) -> GenerationResult:
        return self.solve_question(
            image=image,
            question=question,
            max_new_tokens=max_new_tokens,
            temperature=0.01,
            do_sample=False,
        )

    def caption_image(
        self,
        *,
        image: Image.Image,
        max_new_tokens: int = 96,
        temperature: float = 0.4,
        do_sample: bool = False,
    ) -> GenerationResult:
        caption_prompt = "Describe this image in one concise sentence focusing on key visual facts."
        with use_adapter(self.runtime.model.language_model, self._adapter_for_role(ROLE_SOLVER)):
            return self._generate_understanding_text(
                image=image,
                prompt=caption_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=bool(do_sample),
                temperature=float(temperature),
                text_top_p=1.0,
                text_top_k=0,
            )

    def propose_generation_spec(
        self,
        *,
        image: Image.Image,
        max_new_tokens: int,
        temperature: float,
        min_qa_pairs: int,
        do_sample: bool = True,
    ) -> GenerationResult:
        spec_prompt = build_generation_spec_prompt(min_qa_pairs=min_qa_pairs)
        with use_adapter(self.runtime.model.language_model, self._adapter_for_role(ROLE_PROPOSER)):
            return self._generate_understanding_text(
                image=image,
                prompt=spec_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=bool(do_sample),
                temperature=temperature,
                text_top_p=self._gen_spec_top_p,
                text_top_k=self._gen_spec_top_k,
            )

    def generate_image_from_spec(
        self,
        *,
        spec: str,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        num_timesteps: int = 50,
        timestep_shift: float = 3.0,
        image_size: int = 1024,
    ) -> Optional[Image.Image]:
        retry_plan = self._build_generation_retry_plan(
            image_size=int(image_size),
            num_timesteps=int(num_timesteps),
        )

        for attempt_idx, (curr_size, curr_steps) in enumerate(retry_plan, start=1):
            try:
                with use_adapter(self.runtime.model.language_model, self._adapter_for_role(ROLE_GENERATOR)):
                    out = self.inferencer(
                        text=spec,
                        think=False,
                        understanding_output=False,
                        cfg_text_scale=cfg_text_scale,
                        cfg_img_scale=cfg_img_scale,
                        cfg_interval=[0.4, 1.0],
                        timestep_shift=timestep_shift,
                        num_timesteps=int(curr_steps),
                        image_shapes=(int(curr_size), int(curr_size)),
                    )
                image = out.get("image")
                self._clear_memory()
                return image
            except RuntimeError as exc:
                if not self._is_oom_error(exc):
                    raise
                self._clear_memory()
                if attempt_idx >= len(retry_plan):
                    print(
                        "[rollout_adapter] generation OOM after retries; "
                        f"failed at size={curr_size}, steps={curr_steps}. Skipping generation rollout."
                    )
                    return None
                next_size, next_steps = retry_plan[attempt_idx]
                print(
                    "[rollout_adapter] generation OOM; retrying with "
                    f"size={next_size}, steps={next_steps} (prev size={curr_size}, steps={curr_steps})."
                )
        return None
