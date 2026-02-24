# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image

from .model_loader import BagelRuntime
from .prompts import build_proposer_prompt, build_solver_prompt


@dataclass
class GenerationResult:
    text: str
    raw: dict


class BagelRolloutAdapter:
    """Thin adapter around InterleaveInferencer for self-evolving rollouts."""

    def __init__(self, runtime: BagelRuntime) -> None:
        self.runtime = runtime
        self.inferencer = runtime.inferencer

    def _generate_understanding_text(
        self,
        *,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> GenerationResult:
        out = self.inferencer(
            image=image,
            text=prompt,
            understanding_output=True,
            think=False,
            max_think_token_n=max_new_tokens,
            do_sample=do_sample,
            text_temperature=temperature,
        )
        text = str(out.get("text") or "").strip()
        return GenerationResult(text=text, raw=out)

    def propose_question(
        self,
        *,
        image: Image.Image,
        max_new_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        proposer_prompt = build_proposer_prompt()
        return self._generate_understanding_text(
            image=image,
            prompt=proposer_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )

    def solve_question(
        self,
        *,
        image: Image.Image,
        question: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> GenerationResult:
        solver_prompt = build_solver_prompt(question)
        return self._generate_understanding_text(
            image=image,
            prompt=solver_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
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
        out = self.inferencer(
            text=spec,
            think=False,
            understanding_output=False,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=[0.4, 1.0],
            timestep_shift=timestep_shift,
            num_timesteps=num_timesteps,
            image_shapes=(image_size, image_size),
        )
        return out.get("image")

