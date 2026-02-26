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
    experiment_name: str = "understanding_self_evolving"
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

    # Unified scheduler controls (BLIP3o-style alternating phases).
    understanding_steps_per_cycle: int = 3
    generation_steps_per_cycle: int = 2

    # Generation -> understanding feedback loop.
    replay_buffer_size: int = 1000
    replay_min_reward: float = 0.5
    replay_max_staleness: int = 500
    gen_mix_source_mode: str = "buffer"  # buffer|folder
    generated_mix_dir: str = ""
    generated_mix_min_reward: float = 0.5
    generated_mix_max_files: int = 5000
    generated_mix_refresh_every: int = 10
    understanding_generated_only: bool = False
    gen_mix_ratio_start: float = 0.02
    gen_mix_ratio_max: float = 0.25
    gen_mix_ratio_warmup_steps: int = 1000
    reward_ema_momentum: float = 0.95

    # Proposer framework parity knobs (progressively wired in trainer updates).
    proposer_num_candidates: int = 3
    proposer_spot_check_samples: int = 3
    proposer_spot_entropy_min_gate: float = 0.05
    proposer_grpo_gen_group_size: int = 3
    score_grpo_extras: bool = True
    grpo_extra_temp_multiplier: float = 1.5
    solver_token_entropy_enabled: bool = True
    solver_token_entropy_tokens: int = 5
    solver_token_entropy_window_size: int = 128
    solver_token_entropy_sigmoid_alpha: float = 1.5
    solver_token_entropy_sigmoid_beta: float = 2.0
    ste_spot_easy_quantile: float = 0.30
    proposer_ste_primary_weight: float = 0.70
    proposer_sample_entropy_weight: float = 0.30

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

    def normalized_experiment_name(self) -> str:
        exp = str(self.experiment_name or "understanding_self_evolving").strip().lower()
        if exp not in {"understanding_self_evolving", "generation_self_evolving", "unified_self_evolving"}:
            return "understanding_self_evolving"
        return exp

    def normalized_gen_mix_source_mode(self) -> str:
        mode = str(self.gen_mix_source_mode or "buffer").strip().lower()
        if mode not in {"buffer", "folder"}:
            return "buffer"
        return mode

    def cycle_length(self) -> int:
        u = max(0, int(self.understanding_steps_per_cycle))
        g = max(0, int(self.generation_steps_per_cycle))
        return max(1, u + g)

    def current_gen_mix_ratio(self, step: int, start_step: int) -> float:
        start = max(0.0, min(1.0, float(self.gen_mix_ratio_start)))
        mx = max(0.0, min(1.0, float(self.gen_mix_ratio_max)))
        if mx <= 0.0:
            return 0.0
        warmup = max(1, int(self.gen_mix_ratio_warmup_steps))
        elapsed = max(0, int(step) - int(start_step))
        frac = min(1.0, float(elapsed) / float(warmup))
        return float(start + frac * (mx - start))
