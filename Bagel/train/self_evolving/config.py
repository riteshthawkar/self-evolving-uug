# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ModelLoadConfig:
    model_path: str
    device: str = "cuda"
    vae_device: str = ""
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

    # Optional LoRA runtime setup (applied on the BAGEL language model only).
    enable_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules_csv: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_role_adapters_csv: str = "proposer,solver,generator"
    lora_default_adapter: str = "proposer"

    def lora_target_modules(self) -> List[str]:
        vals = [v.strip() for v in str(self.lora_target_modules_csv or "").split(",")]
        vals = [v for v in vals if v]
        if vals:
            return vals
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def lora_role_adapters(self) -> List[str]:
        vals = [v.strip() for v in str(self.lora_role_adapters_csv or "").split(",")]
        vals = [v for v in vals if v]
        return vals if vals else ["proposer", "solver", "generator"]


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

    # SUDER-style generation phase (proposer joint reward on generated images).
    suder_generation_enabled: bool = False
    max_new_tokens_gen_spec: int = 384
    gen_spec_temperature: float = 0.9
    gen_spec_min_qa_pairs: int = 2
    proposer_gen_entropy_weight: float = 0.7
    proposer_gen_baseline_momentum: float = 0.6

    # Generation inference controls for SUDER rollout.
    generation_cfg_text_scale: float = 4.0
    generation_cfg_img_scale: float = 1.5
    generation_num_timesteps: int = 50
    generation_timestep_shift: float = 3.0
    generation_image_size: int = 1024
    save_generated_images: bool = False

    # Policy update (phase-2 training) knobs.
    policy_updates_enabled: bool = False
    policy_update_method: str = "reinforce"  # reinforce|grpo
    policy_use_bf16: bool = True
    policy_lr: float = 2e-5
    policy_weight_decay: float = 0.0
    policy_max_grad_norm: float = 1.0
    policy_grad_accum_steps: int = 1
    policy_reward_scale: float = 1.0
    baseline_momentum: float = 0.9
    grpo_eps: float = 1e-6
    solver_reward_mix_gamma: float = 0.7
    solver_skip_easy_updates: bool = True
    solver_easy_update_majority_threshold: float = 0.98
    train_understanding_proposer: bool = True
    train_solver: bool = True
    train_generation_proposer: bool = True
    checkpoint_every: int = 100
    resume_from: str = ""
    save_lora_only: bool = True

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

    def normalized_update_method(self) -> str:
        method = str(self.policy_update_method or "reinforce").strip().lower()
        if method not in {"reinforce", "grpo"}:
            return "reinforce"
        return method
