# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ModelLoadConfig:
    model_path: str
    device: str = "cuda"
    max_latent_size: int = 64
    vit_max_num_patch_per_side: int = 70
    latent_patch_size: int = 2
    connector_act: str = "gelu_pytorch_tanh"
    # Inference transforms used by InterleaveInferencer.
    vae_max_image_size: int = 1024
    vae_min_image_size: int = 512
    vae_stride: int = 16
    vit_max_image_size: int = 980
    vit_min_image_size: int = 224
    vit_stride: int = 14


@dataclass
class RolloutConfig:
    image_dir: str
    output_dir: str
    steps: int = 500
    seed: int = 42
    log_every: int = 10

    # Generation controls
    max_new_tokens_proposer: int = 256
    max_new_tokens_solver: int = 96
    proposer_temperature: float = 0.9
    num_solver_samples: int = 7
    solver_temp_min: float = 0.5
    solver_temp_max: float = 2.0

    # Self-consistency / reward shaping
    proposer_entropy_mu: float = 0.9
    proposer_entropy_sigma: float = 0.25
    solver_unsolvable_maj_threshold: float = 0.20
    zero_entropy_eps: float = 1e-6
    zero_entropy_reward_cap: float = 0.45
    proposer_non_objective_penalty: float = 0.20
    proposer_require_objective: bool = True
    acceptance_require_non_easy: bool = True
    rejected_question_penalty: float = 0.35

    # Persist all raw generations for debugging.
    save_raw_generations: bool = True

    def solver_temperatures(self) -> List[float]:
        n = max(1, int(self.num_solver_samples))
        if n == 1:
            return [float(self.solver_temp_min)]
        tmin = float(self.solver_temp_min)
        tmax = float(self.solver_temp_max)
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        step = (tmax - tmin) / float(n - 1)
        return [tmin + step * float(i) for i in range(n)]

