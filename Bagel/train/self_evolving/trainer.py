# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from .adapter_manager import ROLE_GENERATOR, ROLE_PROPOSER, ROLE_SOLVER
from .config import RolloutConfig
from .model_loader import BagelRuntime, load_role_lora_checkpoint
from .policy_updater import BagelGeneratorPolicyUpdater, BagelRolePolicyUpdater
from .prompts import (
    build_generation_spec_prompt,
    build_proposer_prompt,
    build_solver_prompt,
    is_objective_question,
    is_well_formed_question,
    parse_answer,
    parse_first_question,
    parse_generation_spec,
)
from .rewards import (
    answer_match_score,
    compute_dual_track_reward,
    compute_suder_joint_reward,
    majority_vote,
    normalize_answer,
    shannon_entropy_nats,
)
from .rollout_adapter import BagelRolloutAdapter


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _list_images(image_dir: str) -> List[str]:
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths: List[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            paths.append(str(p))
    paths.sort()
    if not paths:
        raise RuntimeError(f"No images found under: {image_dir}")
    return paths


def _write_jsonl(path: str, record: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_filename(text: str, max_len: int = 64) -> str:
    val = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text or ""))
    val = "_".join(part for part in val.split("_") if part)
    if not val:
        val = "sample"
    return val[: max(8, int(max_len))]


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class SelfEvolvingUnderstandingTrainer:
    """BAGEL self-evolving trainer.

    Supports both:
    - rollout-only diagnostics (phase-1)
    - LoRA policy updates (phase-2; proposer/solver/generator REINFORCE/GRPO-style)
    """

    def __init__(self, runtime: BagelRuntime, cfg: RolloutConfig) -> None:
        self.runtime = runtime
        self.cfg = cfg
        self.adapter = BagelRolloutAdapter(runtime)
        self.image_paths = _list_images(cfg.image_dir)
        self.output_dir = self._prepare_output_dir(cfg.output_dir)
        self.rollouts_log_path = os.path.join(self.output_dir, "rollouts.jsonl")
        self.generation_rollouts_log_path = os.path.join(self.output_dir, "generation_rollouts.jsonl")
        self.metrics_log_path = os.path.join(self.output_dir, "metrics.jsonl")
        self.status_path = os.path.join(self.output_dir, "status.json")
        self.summary_path = os.path.join(self.output_dir, "summary.json")
        self.config_path = os.path.join(self.output_dir, "config.json")
        self.generated_images_dir = os.path.join(self.output_dir, "generated_images")
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.proposer_gen_baseline = 0.0
        self.generator_baseline = 0.0
        self.proposer_baseline = 0.0
        self.solver_baseline = 0.0
        self.start_step = 1
        self.last_checkpoint_path = ""
        self.last_lora_checkpoint_dir = ""

        self.policy_updates_enabled = bool(cfg.policy_updates_enabled)
        self.proposer_updater: Optional[BagelRolePolicyUpdater] = None
        self.solver_updater: Optional[BagelRolePolicyUpdater] = None
        self.generator_updater: Optional[BagelGeneratorPolicyUpdater] = None

        if self.cfg.save_generated_images:
            os.makedirs(self.generated_images_dir, exist_ok=True)
        if self.policy_updates_enabled:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self._init_policy_updaters()

        if str(self.cfg.resume_from or "").strip():
            self.start_step = self._load_checkpoint(str(self.cfg.resume_from))
            self.start_step = max(1, int(self.start_step))

        self._persist_config()

    def _prepare_output_dir(self, output_root: str) -> str:
        # Output layout can be controlled from launcher scripts:
        # - BAGEL_OUTPUT_DIR_MODE=direct: write logs/checkpoints directly in output_root
        # - BAGEL_OUTPUT_DIR_MODE=timestamp (default): create per-run timestamp folder
        mode = str(os.environ.get("BAGEL_OUTPUT_DIR_MODE", "timestamp")).strip().lower()
        if mode in {"direct", "flat", "inplace"}:
            os.makedirs(output_root, exist_ok=True)
            return output_root

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_root, f"understanding_rollout_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _persist_config(self) -> None:
        payload = asdict(self.cfg)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _write_json_atomic(path: str, payload: Dict) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)

    def _progress_core(self, *, step: int, phase: str, run_started_at: float) -> Dict[str, float]:
        now = float(time.time())
        elapsed_sec = max(1e-9, now - float(run_started_at))
        total = max(1, int(self.cfg.steps) - int(self.start_step) + 1)
        done = max(0, int(step) - int(self.start_step) + 1)
        done = min(total, done)
        progress = float(done) / float(total)
        steps_per_sec = float(done) / float(elapsed_sec) if elapsed_sec > 0.0 else 0.0
        remaining = max(0, total - done)
        eta_sec = float(remaining) / float(steps_per_sec) if steps_per_sec > 0.0 else -1.0
        return {
            "step": int(step),
            "phase": str(phase),
            "steps_total": int(self.cfg.steps),
            "steps_started_from": int(self.start_step),
            "steps_done": int(done),
            "steps_remaining": int(remaining),
            "progress": float(progress),
            "elapsed_sec": float(elapsed_sec),
            "steps_per_sec": float(steps_per_sec),
            "eta_sec": float(eta_sec),
            "timestamp_unix": float(now),
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

    def _append_metrics(self, record: Dict) -> None:
        _write_jsonl(self.metrics_log_path, record)

    def _write_status(self, *, state: str, progress: Dict, metrics: Dict, last_error: str = "") -> None:
        payload = {
            "state": str(state),
            "output_dir": str(self.output_dir),
            "summary_path": str(self.summary_path),
            "rollouts_log_path": str(self.rollouts_log_path),
            "generation_rollouts_log_path": str(self.generation_rollouts_log_path),
            "metrics_log_path": str(self.metrics_log_path),
            "last_error": str(last_error or ""),
            "progress": progress,
            "metrics": metrics,
        }
        self._write_json_atomic(self.status_path, payload)

    def _sample_image_path(self, step: int) -> str:
        idx = (step - 1) % len(self.image_paths)
        return self.image_paths[idx]

    def _load_image(self, path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def _compute_solver_entropy_and_majority(self, answers_norm: List[str]) -> Tuple[float, float, str]:
        maj_answer, maj_count, hist = majority_vote(answers_norm)
        n = max(1, len([a for a in answers_norm if str(a or "").strip()]))
        maj_frac = float(maj_count) / float(n)
        probs = [float(c) / float(n) for c in hist.values()] if hist else [1.0]
        entropy = shannon_entropy_nats(probs)
        return float(entropy), float(maj_frac), str(maj_answer)

    def _init_policy_updaters(self) -> None:
        if not bool(self.runtime.lora_enabled):
            raise RuntimeError(
                "Policy updates were requested but LoRA adapters are not enabled on BAGEL. "
                "Run with --enable_lora and role adapters."
            )

        proposer_adapter = self.runtime.adapter_for_role(ROLE_PROPOSER)
        solver_adapter = self.runtime.adapter_for_role(ROLE_SOLVER)
        generator_adapter = self.runtime.adapter_for_role(ROLE_GENERATOR)

        self.proposer_updater = BagelRolePolicyUpdater(
            runtime=self.runtime,
            cfg=self.cfg,
            role=ROLE_PROPOSER,
            adapter_name=proposer_adapter,
        )
        self.solver_updater = BagelRolePolicyUpdater(
            runtime=self.runtime,
            cfg=self.cfg,
            role=ROLE_SOLVER,
            adapter_name=solver_adapter,
        )
        self.generator_updater = BagelGeneratorPolicyUpdater(
            runtime=self.runtime,
            cfg=self.cfg,
            role=ROLE_GENERATOR,
            adapter_name=generator_adapter,
        )

    def _resolve_checkpoint_path(self, path: str) -> str:
        p = Path(path)
        if p.is_file():
            return str(p)
        if p.is_dir():
            if p.name.endswith("_lora"):
                base_name = p.name[: -len("_lora")]
                sibling = p.with_name(f"{base_name}.pt")
                if sibling.is_file():
                    return str(sibling)
            candidates = sorted(p.glob("step_*.pt"))
            if candidates:
                return str(candidates[-1])
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    def _collect_model_state_for_checkpoint(self) -> Dict[str, torch.Tensor]:
        state = self.runtime.model.language_model.state_dict()
        if not bool(self.cfg.save_lora_only):
            return {k: v.detach().cpu() for k, v in state.items()}

        filtered: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if ("lora_" in key) or ("modules_to_save" in key):
                filtered[key] = value.detach().cpu()
        return filtered

    @staticmethod
    def _adapter_key_matches(key: str, adapter_name: str) -> bool:
        name = str(adapter_name or "").strip()
        if not name:
            return False
        k = str(key)
        return (f".{name}." in k) or (f"lora_{name}" in k) or (
            name == "default" and ("lora_" in k and ".default." in k)
        )

    def _collect_role_adapter_state(self, adapter_name: str) -> Dict[str, torch.Tensor]:
        state = self.runtime.model.language_model.state_dict()
        selected: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if self._adapter_key_matches(key, adapter_name):
                selected[key] = value.detach().cpu()
        return selected

    def _save_role_adapter_checkpoint(self, step: int) -> str:
        if not bool(self.runtime.lora_enabled) or not bool(self.runtime.role_to_adapter):
            return ""

        step_tag = f"step_{int(step):06d}"
        out_dir = Path(self.checkpoint_dir) / f"{step_tag}_lora"
        tmp_dir = Path(self.checkpoint_dir) / f"{step_tag}_lora.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        files_meta: Dict[str, Dict[str, object]] = {}
        for role, adapter_name in sorted(self.runtime.role_to_adapter.items()):
            adapter = str(adapter_name or "").strip()
            if not adapter:
                continue
            role_state = self._collect_role_adapter_state(adapter)
            role_file = f"role_{role}.pt"
            torch.save(
                {
                    "role": str(role),
                    "adapter_name": adapter,
                    "state_dict": role_state,
                },
                tmp_dir / role_file,
            )
            files_meta[str(role)] = {
                "file": role_file,
                "adapter_name": adapter,
                "tensor_count": int(len(role_state)),
            }

        with (tmp_dir / "adapter_roles.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": int(step),
                    "role_to_adapter": {str(k): str(v) for k, v in self.runtime.role_to_adapter.items()},
                    "files": files_meta,
                },
                f,
                indent=2,
            )

        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        os.replace(str(tmp_dir), str(out_dir))
        self.last_lora_checkpoint_dir = str(out_dir)
        return str(out_dir)

    def _checkpoint_extra_state(self) -> Dict[str, object]:
        return {}

    def _load_checkpoint_extra_state(self, state: Dict[str, object]) -> None:
        _ = state

    def _save_checkpoint(self, step: int) -> str:
        if not self.policy_updates_enabled:
            return ""

        role_ckpt_dir = self._save_role_adapter_checkpoint(step)

        payload = {
            "step": int(step),
            "proposer_baseline": float(self.proposer_baseline),
            "solver_baseline": float(self.solver_baseline),
            "proposer_gen_baseline": float(self.proposer_gen_baseline),
            "generator_baseline": float(self.generator_baseline),
            "policy_update_method": self.cfg.normalized_update_method(),
            "model_state": self._collect_model_state_for_checkpoint(),
            "lora_roles_dir": str(role_ckpt_dir),
        }
        if self.proposer_updater is not None:
            payload["proposer_updater"] = self.proposer_updater.state_dict()
        if self.solver_updater is not None:
            payload["solver_updater"] = self.solver_updater.state_dict()
        if self.generator_updater is not None:
            payload["generator_updater"] = self.generator_updater.state_dict()
        extra_state = self._checkpoint_extra_state()
        if isinstance(extra_state, dict) and extra_state:
            payload["extra_state"] = extra_state

        path = os.path.join(self.checkpoint_dir, f"step_{int(step):06d}.pt")
        torch.save(payload, path)
        self.last_checkpoint_path = path
        with open(os.path.join(self.checkpoint_dir, "latest.txt"), "w", encoding="utf-8") as f:
            f.write(path + "\n")
        return path

    def _load_checkpoint(self, path: str) -> int:
        ckpt_path = self._resolve_checkpoint_path(path)
        state = torch.load(ckpt_path, map_location="cpu")

        model_state = state.get("model_state", None)
        loaded_model_state = False
        if isinstance(model_state, dict) and model_state:
            msg = self.runtime.model.language_model.load_state_dict(model_state, strict=False)
            missing = len(getattr(msg, "missing_keys", []) or [])
            unexpected = len(getattr(msg, "unexpected_keys", []) or [])
            print(
                f"[self_evolving] loaded checkpoint model state from {ckpt_path} "
                f"(missing={missing}, unexpected={unexpected})"
            )
            loaded_model_state = True
        elif bool(self.runtime.lora_enabled):
            # Compatibility fallback: if model_state is missing, attempt role-based LoRA folder load.
            lora_dir = str(state.get("lora_roles_dir", "") or "").strip()
            if not lora_dir:
                ckpt_path_obj = Path(ckpt_path)
                fallback_dir = ckpt_path_obj.with_name(f"{ckpt_path_obj.stem}_lora")
                if fallback_dir.is_dir():
                    lora_dir = str(fallback_dir)
            if lora_dir:
                stats = load_role_lora_checkpoint(
                    self.runtime.model.language_model,
                    checkpoint_path=lora_dir,
                    role_to_adapter=self.runtime.role_to_adapter,
                )
                self.last_lora_checkpoint_dir = str(stats.get("source", lora_dir))
                print(
                    f"[self_evolving] loaded role-based LoRA checkpoint from {self.last_lora_checkpoint_dir} "
                    f"(roles_loaded={stats.get('roles_loaded')}, tensors_loaded={stats.get('tensors_loaded')})"
                )

        self.proposer_baseline = float(state.get("proposer_baseline", self.proposer_baseline))
        self.solver_baseline = float(state.get("solver_baseline", self.solver_baseline))
        self.proposer_gen_baseline = float(state.get("proposer_gen_baseline", self.proposer_gen_baseline))
        self.generator_baseline = float(state.get("generator_baseline", self.generator_baseline))

        if self.proposer_updater is not None and isinstance(state.get("proposer_updater"), dict):
            self.proposer_updater.load_state_dict(state["proposer_updater"])
        if self.solver_updater is not None and isinstance(state.get("solver_updater"), dict):
            self.solver_updater.load_state_dict(state["solver_updater"])
        if self.generator_updater is not None and isinstance(state.get("generator_updater"), dict):
            self.generator_updater.load_state_dict(state["generator_updater"])
        self._load_checkpoint_extra_state(state.get("extra_state", {}))

        loaded_step = int(state.get("step", 0))
        self.last_checkpoint_path = ckpt_path
        if loaded_model_state:
            ckpt_path_obj = Path(ckpt_path)
            sibling = ckpt_path_obj.with_name(f"{ckpt_path_obj.stem}_lora")
            if sibling.is_dir():
                self.last_lora_checkpoint_dir = str(sibling)
        print(f"[self_evolving] resumed from checkpoint step={loaded_step} ({ckpt_path})")
        return loaded_step + 1

    def _solver_scalar_reward(self, *, entropy_nats: float, majority_fraction: float, easy_case: bool, unsolvable_case: bool) -> float:
        entropy_cap = max(1e-6, math.log(float(max(2, int(self.cfg.num_solver_samples)))))
        entropy_term = 1.0 - min(1.0, float(entropy_nats) / entropy_cap)
        gamma = _clamp01(self.cfg.solver_reward_mix_gamma)
        reward = gamma * float(majority_fraction) + (1.0 - gamma) * float(entropy_term)
        if easy_case:
            reward = -abs(reward)
        if unsolvable_case:
            reward = 0.0
        return float(max(-1.0, min(1.0, reward)))

    def _run_suder_generation_rollout(
        self,
        *,
        step: int,
        image_path: str,
        image: Image.Image,
        solver_temps: List[float],
    ) -> Dict:
        spec_out = self.adapter.propose_generation_spec(
            image=image,
            max_new_tokens=self.cfg.max_new_tokens_gen_spec,
            temperature=float(self.cfg.gen_spec_temperature),
            min_qa_pairs=int(self.cfg.gen_spec_min_qa_pairs),
        )
        spec = parse_generation_spec(spec_out.text, min_qa_pairs=int(self.cfg.gen_spec_min_qa_pairs))
        spec_retry_attempted = False
        spec_retry_temperature = 0.0
        if spec is None:
            spec_retry_attempted = True
            spec_retry_temperature = max(0.1, min(0.7, float(self.cfg.gen_spec_temperature) * 0.5))
            spec_out_retry = self.adapter.propose_generation_spec(
                image=image,
                max_new_tokens=self.cfg.max_new_tokens_gen_spec,
                temperature=float(spec_retry_temperature),
                min_qa_pairs=int(self.cfg.gen_spec_min_qa_pairs),
            )
            spec_retry = parse_generation_spec(
                spec_out_retry.text,
                min_qa_pairs=int(self.cfg.gen_spec_min_qa_pairs),
            )
            if spec_retry is not None:
                spec = spec_retry
                spec_out = spec_out_retry

        if spec is None:
            return {
                "step": int(step),
                "status": "skipped",
                "skip_reason": "invalid_generation_spec",
                "image_path": image_path,
                "proposer_spec_raw": spec_out.text if self.cfg.save_raw_generations else "",
                "policy_update_attempted": False,
                "policy_update_applied": False,
                "policy_update_reason": "invalid_generation_spec",
                "spec_retry_attempted": bool(spec_retry_attempted),
                "spec_retry_temperature": float(spec_retry_temperature),
            }

        max_spec_prompt_chars = max(
            64,
            int(os.environ.get("BAGEL_MAX_SPEC_PROMPT_CHARS", "384") or "384"),
        )
        gen_spec_prompt = str(spec.prompt or "").strip()
        spec_prompt_truncated = False
        if len(gen_spec_prompt) > max_spec_prompt_chars:
            spec_prompt_truncated = True
            clipped = gen_spec_prompt[:max_spec_prompt_chars]
            clipped_ws = clipped.rsplit(" ", 1)[0].strip()
            gen_spec_prompt = clipped_ws if clipped_ws else clipped.strip()

        generated = self.adapter.generate_image_from_spec(
            spec=gen_spec_prompt,
            cfg_text_scale=float(self.cfg.generation_cfg_text_scale),
            cfg_img_scale=float(self.cfg.generation_cfg_img_scale),
            num_timesteps=int(self.cfg.generation_num_timesteps),
            timestep_shift=float(self.cfg.generation_timestep_shift),
            image_size=int(self.cfg.generation_image_size),
        )
        if generated is None:
            return {
                "step": int(step),
                "status": "skipped",
                "skip_reason": "generation_failed",
                "image_path": image_path,
                "spec_prompt": spec.prompt,
                "qa_pair_count": len(spec.qa_pairs),
                "proposer_spec_raw": spec_out.text if self.cfg.save_raw_generations else "",
                "spec_prompt_truncated": bool(spec_prompt_truncated),
                "policy_update_attempted": False,
                "policy_update_applied": False,
                "policy_update_reason": "generation_failed",
            }

        generated_image_path = ""
        if self.cfg.save_generated_images:
            generated_image_path = os.path.join(
                self.generated_images_dir,
                f"step_{step:06d}_{_safe_filename(Path(image_path).stem)}.png",
            )
            generated.save(generated_image_path)

        qa_logs: List[Dict] = []
        entropy_vals: List[float] = []
        match_vals: List[float] = []
        gen_solver_update_enabled = bool(
            getattr(self.cfg, "gen_step_solver_update_enabled", False)
            and self.policy_updates_enabled
            and self.cfg.train_solver
            and self.solver_updater is not None
        )
        max_gen_solver_updates = max(
            0,
            int(
                os.environ.get(
                    "BAGEL_GEN_SOLVER_POLICY_MAX_SAMPLES",
                    os.environ.get("BAGEL_SOLVER_POLICY_MAX_SAMPLES", "0"),
                )
                or "0"
            ),
        )
        gen_solver_update_attempted = 0
        gen_solver_update_applied = 0
        gen_solver_update_stats: List[Dict] = []
        gen_solver_reward_values: List[float] = []
        for qa_idx, qa in enumerate(spec.qa_pairs):
            solver_outputs_raw: List[str] = []
            solver_answers_norm: List[str] = []
            solver_samples: List[Tuple[str, str]] = []
            for temp in solver_temps:
                out = self.adapter.solve_question(
                    image=generated,
                    question=qa.question,
                    max_new_tokens=self.cfg.max_new_tokens_solver,
                    temperature=float(temp),
                    do_sample=True,
                )
                solver_outputs_raw.append(out.text)
                ans = normalize_answer(parse_answer(out.text))
                if ans:
                    solver_answers_norm.append(ans)
                    solver_samples.append((out.text, ans))
            if not solver_answers_norm:
                qa_logs.append(
                    {
                        "qa_index": int(qa_idx),
                        "status": "skipped",
                        "skip_reason": "empty_solver_answers",
                        "question": qa.question,
                        "expected_answer": normalize_answer(qa.answer),
                        "solver_outputs_raw": solver_outputs_raw if self.cfg.save_raw_generations else [],
                    }
                )
                continue

            entropy_nats, majority_fraction, majority_answer = self._compute_solver_entropy_and_majority(
                solver_answers_norm
            )
            expected_answer = normalize_answer(qa.answer)
            match_score = answer_match_score(majority_answer, expected_answer)
            entropy_vals.append(float(entropy_nats))
            match_vals.append(float(match_score))

            if gen_solver_update_enabled and solver_samples:
                solver_prompt = build_solver_prompt(qa.question)
                group_rewards = [
                    float(answer_match_score(sample_ans, expected_answer))
                    for _, sample_ans in solver_samples
                ]
                for idx, (sample_raw, _) in enumerate(solver_samples):
                    if (
                        max_gen_solver_updates > 0
                        and int(gen_solver_update_attempted) >= int(max_gen_solver_updates)
                    ):
                        break
                    sample_reward = float(group_rewards[idx]) if idx < len(group_rewards) else 0.0
                    stats = self.solver_updater.step(
                        image=generated,
                        prompt=solver_prompt,
                        completion=sample_raw,
                        reward=sample_reward,
                        baseline=self.solver_baseline,
                        group_rewards=group_rewards,
                    )
                    gen_solver_update_stats.append(stats)
                    gen_solver_update_attempted += 1
                    if not bool(stats.get("skipped", True)):
                        gen_solver_update_applied += 1
                    gen_solver_reward_values.append(sample_reward)

            qa_logs.append(
                {
                    "qa_index": int(qa_idx),
                    "status": "ok",
                    "question": qa.question,
                    "expected_answer": expected_answer,
                    "majority_answer": majority_answer,
                    "majority_fraction": float(majority_fraction),
                    "entropy_nats": float(entropy_nats),
                    "match_score": float(match_score),
                    "solver_answers_norm": solver_answers_norm,
                    "solver_outputs_raw": solver_outputs_raw if self.cfg.save_raw_generations else [],
                }
            )

        if gen_solver_update_enabled and gen_solver_reward_values:
            momentum_solver = _clamp01(float(self.cfg.baseline_momentum))
            mean_solver_reward = float(sum(gen_solver_reward_values) / float(len(gen_solver_reward_values)))
            self.solver_baseline = (
                momentum_solver * float(self.solver_baseline)
                + (1.0 - momentum_solver) * mean_solver_reward
            )

        if not entropy_vals:
            return {
                "step": int(step),
                "status": "skipped",
                "skip_reason": "empty_generation_qa_entropy",
                "image_path": image_path,
                "spec_prompt": spec.prompt,
                "qa_pair_count": len(spec.qa_pairs),
                "generated_image_path": generated_image_path,
                "qa_logs": qa_logs,
                "policy_update_attempted": False,
                "policy_update_applied": False,
                "policy_update_reason": "empty_generation_qa_entropy",
                "gen_solver_policy_update_enabled": bool(gen_solver_update_enabled),
                "gen_solver_policy_update_budget": int(max_gen_solver_updates),
                "gen_solver_policy_update_attempts": int(gen_solver_update_attempted),
                "gen_solver_policy_update_applied": int(gen_solver_update_applied),
            }

        mean_entropy = float(sum(entropy_vals) / float(len(entropy_vals)))
        quality_component = float(sum(match_vals) / float(max(1, len(match_vals))))
        joint = compute_suder_joint_reward(
            mean_entropy_nats=mean_entropy,
            quality_component=quality_component,
            entropy_mu=self.cfg.proposer_entropy_mu,
            entropy_sigma=self.cfg.proposer_entropy_sigma,
            entropy_weight_alpha=self.cfg.proposer_gen_entropy_weight,
            zero_entropy_eps=self.cfg.zero_entropy_eps,
            zero_entropy_reward_cap=self.cfg.zero_entropy_reward_cap,
        )

        proposer_baseline_before = float(self.proposer_gen_baseline)
        generator_baseline_before = float(self.generator_baseline)
        momentum = max(0.0, min(1.0, float(self.cfg.proposer_gen_baseline_momentum)))
        self.proposer_gen_baseline = (
            momentum * self.proposer_gen_baseline
            + (1.0 - momentum) * float(joint.reward)
        )
        self.generator_baseline = (
            momentum * self.generator_baseline
            + (1.0 - momentum) * float(joint.reward)
        )

        proposer_policy_update_attempted = False
        proposer_policy_update_applied = False
        proposer_policy_update_reason = "disabled"
        proposer_policy_update_stats: Dict = {}
        if (
            self.policy_updates_enabled
            and self.cfg.train_generation_proposer
            and self.proposer_updater is not None
        ):
            proposer_policy_update_attempted = True
            proposer_policy_update_stats = self.proposer_updater.step(
                image=image,
                prompt=build_generation_spec_prompt(min_qa_pairs=int(self.cfg.gen_spec_min_qa_pairs)),
                completion=spec_out.text,
                reward=float(joint.reward),
                baseline=proposer_baseline_before,
            )
            proposer_policy_update_applied = bool(not proposer_policy_update_stats.get("skipped", True))
            proposer_policy_update_reason = str(proposer_policy_update_stats.get("reason", "unknown"))

        generator_policy_update_attempted = False
        generator_policy_update_applied = False
        generator_policy_update_reason = "disabled"
        generator_policy_update_stats: Dict = {}
        if (
            self.policy_updates_enabled
            and self.cfg.train_generator
            and self.generator_updater is not None
        ):
            generator_policy_update_attempted = True
            generator_policy_update_stats = self.generator_updater.step(
                image=generated,
                prompt=gen_spec_prompt,
                reward=float(joint.reward),
                baseline=generator_baseline_before,
            )
            generator_policy_update_applied = bool(not generator_policy_update_stats.get("skipped", True))
            generator_policy_update_reason = str(generator_policy_update_stats.get("reason", "unknown"))

        policy_update_attempted = bool(
            proposer_policy_update_attempted or generator_policy_update_attempted
        )
        policy_update_applied = bool(
            proposer_policy_update_applied or generator_policy_update_applied
        )
        policy_update_reasons: List[str] = []
        if proposer_policy_update_attempted:
            policy_update_reasons.append(f"proposer:{proposer_policy_update_reason}")
        if generator_policy_update_attempted:
            policy_update_reasons.append(f"generator:{generator_policy_update_reason}")
        policy_update_reason = "ok" if policy_update_applied else (
            ";".join(policy_update_reasons) if policy_update_reasons else "disabled"
        )
        policy_update_stats: Dict = {
            "proposer": proposer_policy_update_stats,
            "generator": generator_policy_update_stats,
        }
        policy_updates_attempted_count = int(proposer_policy_update_attempted) + int(generator_policy_update_attempted)
        policy_updates_applied_count = int(proposer_policy_update_applied) + int(generator_policy_update_applied)

        return {
            "step": int(step),
            "status": "ok",
            "image_path": image_path,
            "spec_prompt": spec.prompt,
            "spec_prompt_for_generation": gen_spec_prompt,
            "spec_prompt_truncated": bool(spec_prompt_truncated),
            "qa_pair_count": len(spec.qa_pairs),
            "qa_pairs": [{"question": qa.question, "answer": normalize_answer(qa.answer)} for qa in spec.qa_pairs],
            "proposer_spec_raw": spec_out.text if self.cfg.save_raw_generations else "",
            "generated_image_path": generated_image_path,
            "qa_logs": qa_logs,
            "mean_entropy_nats": float(joint.mean_entropy_nats),
            "entropy_component": float(joint.entropy_component),
            "quality_component": float(joint.quality_component),
            "proposer_gen_reward": float(joint.reward),
            "entropy_weight_alpha": float(joint.entropy_weight_alpha),
            "proposer_gen_baseline_before": float(proposer_baseline_before),
            "proposer_gen_baseline_after": float(self.proposer_gen_baseline),
            "proposer_gen_advantage": float(joint.reward - proposer_baseline_before),
            "generator_baseline_before": float(generator_baseline_before),
            "generator_baseline_after": float(self.generator_baseline),
            "generator_advantage": float(joint.reward - generator_baseline_before),
            "policy_update_attempted": bool(policy_update_attempted),
            "policy_update_applied": bool(policy_update_applied),
            "policy_update_reason": policy_update_reason,
            "policy_update_stats": policy_update_stats,
            "policy_updates_attempted_count": int(policy_updates_attempted_count),
            "policy_updates_applied_count": int(policy_updates_applied_count),
            "proposer_policy_update_attempted": bool(proposer_policy_update_attempted),
            "proposer_policy_update_applied": bool(proposer_policy_update_applied),
            "proposer_policy_update_reason": str(proposer_policy_update_reason),
            "proposer_policy_update_stats": proposer_policy_update_stats,
            "generator_policy_update_attempted": bool(generator_policy_update_attempted),
            "generator_policy_update_applied": bool(generator_policy_update_applied),
            "generator_policy_update_reason": str(generator_policy_update_reason),
            "generator_policy_update_stats": generator_policy_update_stats,
            "gen_solver_policy_update_enabled": bool(gen_solver_update_enabled),
            "gen_solver_policy_update_budget": int(max_gen_solver_updates),
            "gen_solver_policy_update_attempts": int(gen_solver_update_attempted),
            "gen_solver_policy_update_applied": int(gen_solver_update_applied),
            "gen_solver_policy_update_ce_mean": _mean(
                [
                    float(s.get("ce_loss", 0.0))
                    for s in gen_solver_update_stats
                    if not bool(s.get("skipped", True))
                ]
            ),
            "solver_temperatures": solver_temps,
            "spec_retry_attempted": bool(spec_retry_attempted),
            "spec_retry_temperature": float(spec_retry_temperature),
        }

    def run(self) -> Dict[str, float]:
        random.seed(self.cfg.seed)

        reward_sum = 0.0
        reward_nonzero = 0
        dual_track_disagree = 0
        valid_steps = 0
        skipped_steps = 0
        suder_valid_steps = 0
        suder_skipped_steps = 0
        suder_reward_sum = 0.0
        suder_entropy_sum = 0.0
        suder_quality_sum = 0.0
        policy_updates_attempted = 0
        policy_updates_applied = 0

        solver_temps = self.cfg.solver_temperatures()
        baseline_momentum = _clamp01(self.cfg.baseline_momentum)
        run_started_at = float(time.time())
        last_step = int(self.start_step) - 1

        def _status_metrics(step_time_sec: float) -> Dict[str, float]:
            return {
                "step_time_sec": float(step_time_sec),
                "understanding_steps_valid": int(valid_steps),
                "understanding_steps_skipped": int(skipped_steps),
                "understanding_mean_reward": float(reward_sum / float(max(1, valid_steps))),
                "dual_track_disagree_rate": float(dual_track_disagree / float(max(1, valid_steps))),
                "suder_generation_enabled": bool(self.cfg.suder_generation_enabled),
                "generation_steps_valid": int(suder_valid_steps),
                "generation_steps_skipped": int(suder_skipped_steps),
                "generation_mean_reward": float(suder_reward_sum / float(max(1, suder_valid_steps))),
                "generation_mean_entropy_nats": float(suder_entropy_sum / float(max(1, suder_valid_steps))),
                "generation_mean_quality": float(suder_quality_sum / float(max(1, suder_valid_steps))),
                "policy_updates_attempted": int(policy_updates_attempted),
                "policy_updates_applied": int(policy_updates_applied),
                "proposer_baseline": float(self.proposer_baseline),
                "solver_baseline": float(self.solver_baseline),
                "proposer_gen_baseline": float(self.proposer_gen_baseline),
                "generator_baseline": float(self.generator_baseline),
            }

        def _emit_training_logs(step_id: int, *, phase: str, step_time_sec: float) -> None:
            progress = self._progress_core(
                step=int(step_id),
                phase=str(phase),
                run_started_at=run_started_at,
            )
            metrics = _status_metrics(step_time_sec)
            self._write_status(state="running", progress=progress, metrics=metrics)
            if int(step_id) % max(1, int(self.cfg.log_every)) == 0:
                self._append_metrics({"kind": "heartbeat", **progress, **metrics})

        self._write_status(
            state="running",
            progress=self._progress_core(
                step=int(self.start_step) - 1,
                phase="init",
                run_started_at=run_started_at,
            ),
            metrics=_status_metrics(0.0),
        )

        def _run_and_accumulate_suder(step_id: int, path: str, src_image: Image.Image) -> None:
            nonlocal suder_valid_steps
            nonlocal suder_skipped_steps
            nonlocal suder_reward_sum
            nonlocal suder_entropy_sum
            nonlocal suder_quality_sum
            nonlocal policy_updates_attempted
            nonlocal policy_updates_applied
            if not self.cfg.suder_generation_enabled:
                return
            suder_record = self._run_suder_generation_rollout(
                step=step_id,
                image_path=path,
                image=src_image,
                solver_temps=solver_temps,
            )
            _write_jsonl(self.generation_rollouts_log_path, suder_record)
            if suder_record.get("status") == "ok":
                suder_valid_steps += 1
                suder_reward_sum += float(suder_record.get("proposer_gen_reward", 0.0))
                suder_entropy_sum += float(suder_record.get("mean_entropy_nats", 0.0))
                suder_quality_sum += float(suder_record.get("quality_component", 0.0))
            else:
                suder_skipped_steps += 1

            attempted_count = int(
                suder_record.get(
                    "policy_updates_attempted_count",
                    int(bool(suder_record.get("policy_update_attempted", False))),
                )
            )
            applied_count = int(
                suder_record.get(
                    "policy_updates_applied_count",
                    int(bool(suder_record.get("policy_update_applied", False))),
                )
            )
            if attempted_count > 0:
                policy_updates_attempted += int(attempted_count)
                policy_updates_applied += int(max(0, applied_count))

        for step in range(int(self.start_step), int(self.cfg.steps) + 1):
            step_t0 = float(time.time())
            image_path = self._sample_image_path(step)
            image = self._load_image(image_path)

            proposer = self.adapter.propose_question(
                image=image,
                max_new_tokens=self.cfg.max_new_tokens_proposer,
                temperature=self.cfg.proposer_temperature,
            )
            question = parse_first_question(proposer.text)
            if not question:
                skipped_steps += 1
                _write_jsonl(
                    self.rollouts_log_path,
                    {
                        "step": step,
                        "status": "skipped",
                        "skip_reason": "empty_question",
                        "image_path": image_path,
                        "proposer_raw": proposer.text if self.cfg.save_raw_generations else "",
                    },
                )
                _run_and_accumulate_suder(step, image_path, image)
                _emit_training_logs(step, phase="understanding", step_time_sec=float(time.time() - step_t0))
                continue
            if not is_well_formed_question(question):
                skipped_steps += 1
                _write_jsonl(
                    self.rollouts_log_path,
                    {
                        "step": step,
                        "status": "skipped",
                        "skip_reason": "invalid_question",
                        "image_path": image_path,
                        "question": question,
                        "proposer_raw": proposer.text if self.cfg.save_raw_generations else "",
                    },
                )
                _run_and_accumulate_suder(step, image_path, image)
                _emit_training_logs(step, phase="understanding", step_time_sec=float(time.time() - step_t0))
                continue

            solver_outputs_raw: List[str] = []
            solver_answers_norm: List[str] = []
            solver_samples: List[Tuple[str, str]] = []
            for temp in solver_temps:
                out = self.adapter.solve_question(
                    image=image,
                    question=question,
                    max_new_tokens=self.cfg.max_new_tokens_solver,
                    temperature=float(temp),
                    do_sample=True,
                )
                solver_outputs_raw.append(out.text)
                ans = normalize_answer(parse_answer(out.text))
                if ans:
                    solver_answers_norm.append(ans)
                    solver_samples.append((out.text, ans))

            if not solver_answers_norm:
                skipped_steps += 1
                _write_jsonl(
                    self.rollouts_log_path,
                    {
                        "step": step,
                        "status": "skipped",
                        "skip_reason": "empty_solver_answers",
                        "image_path": image_path,
                        "question": question,
                        "solver_outputs_raw": solver_outputs_raw if self.cfg.save_raw_generations else [],
                    },
                )
                _run_and_accumulate_suder(step, image_path, image)
                _emit_training_logs(step, phase="understanding", step_time_sec=float(time.time() - step_t0))
                continue

            intuitive = self.adapter.intuitive_answer(
                image=image,
                question=question,
                max_new_tokens=self.cfg.max_new_tokens_solver,
            )
            intuitive_norm = normalize_answer(parse_answer(intuitive.text))

            dual = compute_dual_track_reward(
                answers=solver_answers_norm,
                intuitive_answer=intuitive_norm,
                entropy_mu=self.cfg.proposer_entropy_mu,
                entropy_sigma=self.cfg.proposer_entropy_sigma,
                unsolvable_maj_threshold=self.cfg.solver_unsolvable_maj_threshold,
                zero_entropy_eps=self.cfg.zero_entropy_eps,
            )

            reward = dual.reward
            non_objective = not is_objective_question(question)
            if self.cfg.proposer_require_objective and non_objective:
                reward -= float(self.cfg.proposer_non_objective_penalty)
            if self.cfg.acceptance_require_non_easy and dual.easy_case:
                reward -= float(self.cfg.rejected_question_penalty)
            reward = max(-1.0, min(1.0, reward))

            understanding_update_eligible = True
            understanding_update_skip_reason = "ok"
            if self.cfg.proposer_require_objective and non_objective:
                understanding_update_eligible = False
                understanding_update_skip_reason = "non_objective_question"
            elif bool(getattr(self.cfg, "proposer_reject_unsolvable", True)) and bool(dual.unsolvable_case):
                understanding_update_eligible = False
                understanding_update_skip_reason = "unsolvable_case"
            elif bool(getattr(self.cfg, "understanding_update_require_disagreement", True)) and bool(dual.easy_case):
                understanding_update_eligible = False
                understanding_update_skip_reason = "easy_case"

            valid_steps += 1
            reward_sum += reward
            reward_nonzero += int(abs(reward) > 1e-9)
            dual_track_disagree += int(not dual.dual_track_agree)

            proposer_update_stats: Dict = {
                "skipped": True,
                "reason": "disabled",
            }
            proposer_update_attempted = False
            proposer_update_applied = False
            if (
                self.policy_updates_enabled
                and self.cfg.train_understanding_proposer
                and self.proposer_updater is not None
                and understanding_update_eligible
            ):
                proposer_update_attempted = True
                proposer_update_stats = self.proposer_updater.step(
                    image=image,
                    prompt=build_proposer_prompt(),
                    completion=proposer.text,
                    reward=reward,
                    baseline=self.proposer_baseline,
                )
                proposer_update_applied = bool(not proposer_update_stats.get("skipped", True))
                policy_updates_attempted += 1
                policy_updates_applied += int(proposer_update_applied)
            elif self.policy_updates_enabled and self.cfg.train_understanding_proposer and self.proposer_updater is not None:
                proposer_update_stats = {"skipped": True, "reason": f"gated_{understanding_update_skip_reason}"}

            if understanding_update_eligible:
                self.proposer_baseline = (
                    baseline_momentum * self.proposer_baseline
                    + (1.0 - baseline_momentum) * float(reward)
                )

            solver_group_rewards = [
                float(answer_match_score(ans_norm, dual.majority_answer))
                for _, ans_norm in solver_samples
            ]
            if dual.easy_case:
                solver_group_rewards = [-abs(v) for v in solver_group_rewards]
            if dual.unsolvable_case:
                solver_group_rewards = [0.0 for _ in solver_group_rewards]

            solver_scalar_reward = self._solver_scalar_reward(
                entropy_nats=float(dual.entropy_nats),
                majority_fraction=float(dual.majority_fraction),
                easy_case=bool(dual.easy_case),
                unsolvable_case=bool(dual.unsolvable_case),
            )
            solver_skip_update = (
                bool(self.cfg.solver_skip_easy_updates)
                and bool(dual.easy_case)
                and float(dual.majority_fraction) >= float(self.cfg.solver_easy_update_majority_threshold)
            )
            solver_update_stats: List[Dict] = []
            solver_update_reason = "disabled"
            if not understanding_update_eligible and bool(getattr(self.cfg, "understanding_update_require_disagreement", True)):
                solver_skip_update = True
                solver_update_reason = f"gated_{understanding_update_skip_reason}"
            elif bool(getattr(self.cfg, "solver_skip_unsolvable_updates", True)) and bool(dual.unsolvable_case):
                solver_skip_update = True
                solver_update_reason = "unsolvable_case_skip"
            elif solver_skip_update:
                solver_update_reason = "easy_question_skip"
            elif (
                self.policy_updates_enabled
                and self.cfg.train_solver
                and self.solver_updater is not None
            ):
                solver_prompt = build_solver_prompt(question)
                for idx, (sample_raw, _) in enumerate(solver_samples):
                    sample_reward = (
                        float(solver_group_rewards[idx])
                        if idx < len(solver_group_rewards)
                        else float(solver_scalar_reward)
                    )
                    update_stats = self.solver_updater.step(
                        image=image,
                        prompt=solver_prompt,
                        completion=sample_raw,
                        reward=float(sample_reward),
                        baseline=self.solver_baseline,
                        group_rewards=solver_group_rewards,
                    )
                    solver_update_stats.append(update_stats)
                    policy_updates_attempted += 1
                    policy_updates_applied += int(not update_stats.get("skipped", True))
                solver_update_reason = "ok" if solver_update_stats else "no_samples"

            if understanding_update_eligible:
                self.solver_baseline = (
                    baseline_momentum * self.solver_baseline
                    + (1.0 - baseline_momentum) * float(solver_scalar_reward)
                )

            _write_jsonl(
                self.rollouts_log_path,
                {
                    "step": step,
                    "status": "ok",
                    "image_path": image_path,
                    "question": question,
                    "proposer_raw": proposer.text if self.cfg.save_raw_generations else "",
                    "solver_answers_norm": solver_answers_norm,
                    "solver_outputs_raw": solver_outputs_raw if self.cfg.save_raw_generations else [],
                    "intuitive_answer": intuitive_norm,
                    "intuitive_raw": intuitive.text if self.cfg.save_raw_generations else "",
                    "entropy_nats": dual.entropy_nats,
                    "majority_fraction": dual.majority_fraction,
                    "majority_answer": dual.majority_answer,
                    "dual_track_agree": dual.dual_track_agree,
                    "easy_case": dual.easy_case,
                    "unsolvable_case": dual.unsolvable_case,
                    "proposer_reward_raw": dual.reward_raw,
                    "proposer_reward_final": reward,
                    "proposer_non_objective_question": non_objective,
                    "solver_scalar_reward": solver_scalar_reward,
                    "solver_group_rewards": solver_group_rewards,
                    "proposer_baseline": float(self.proposer_baseline),
                    "solver_baseline": float(self.solver_baseline),
                    "understanding_update_eligible": bool(understanding_update_eligible),
                    "understanding_update_skip_reason": str(understanding_update_skip_reason),
                    "policy_updates_enabled": bool(self.policy_updates_enabled),
                    "proposer_policy_update_attempted": bool(proposer_update_attempted),
                    "proposer_policy_update_applied": bool(proposer_update_applied),
                    "proposer_policy_update_stats": proposer_update_stats,
                    "solver_policy_update_skipped": bool(solver_skip_update),
                    "solver_policy_update_reason": solver_update_reason,
                    "solver_policy_update_attempts": int(len(solver_update_stats)),
                    "solver_policy_update_applied": int(
                        sum(int(not s.get("skipped", True)) for s in solver_update_stats)
                    ),
                    "solver_policy_update_ce_mean": _mean(
                        [float(s.get("ce_loss", 0.0)) for s in solver_update_stats if not s.get("skipped", True)]
                    ),
                    "solver_temperatures": solver_temps,
                },
            )

            _run_and_accumulate_suder(step, image_path, image)
            _emit_training_logs(step, phase="understanding", step_time_sec=float(time.time() - step_t0))

            if step % max(1, self.cfg.log_every) == 0:
                mean_reward = reward_sum / float(max(1, valid_steps))
                disagree_rate = dual_track_disagree / float(max(1, valid_steps))
                if self.cfg.suder_generation_enabled:
                    suder_mean_reward = suder_reward_sum / float(max(1, suder_valid_steps))
                    print(
                        f"[self_evolving][step={step}] valid={valid_steps} skipped={skipped_steps} "
                        f"mean_reward={mean_reward:.4f} disagree_rate={disagree_rate:.4f} "
                        f"suder_valid={suder_valid_steps} suder_mean_reward={suder_mean_reward:.4f} "
                        f"policy_updates={policy_updates_applied}/{policy_updates_attempted}"
                    )
                else:
                    print(
                        f"[self_evolving][step={step}] valid={valid_steps} skipped={skipped_steps} "
                        f"mean_reward={mean_reward:.4f} disagree_rate={disagree_rate:.4f} "
                        f"policy_updates={policy_updates_applied}/{policy_updates_attempted}"
                    )

            if (
                self.policy_updates_enabled
                and int(self.cfg.checkpoint_every) > 0
                and step > 0
                and step % int(self.cfg.checkpoint_every) == 0
            ):
                path = self._save_checkpoint(step)
                if path:
                    print(f"[self_evolving] saved checkpoint: {path}")
                    if self.last_lora_checkpoint_dir:
                        print(f"[self_evolving] saved role LoRA checkpoint: {self.last_lora_checkpoint_dir}")

        flushed_optim_steps = 0
        if self.proposer_updater is not None:
            flushed_optim_steps += int(self.proposer_updater.finalize())
        if self.solver_updater is not None:
            flushed_optim_steps += int(self.solver_updater.finalize())
        if self.generator_updater is not None:
            flushed_optim_steps += int(self.generator_updater.finalize())

        if self.policy_updates_enabled:
            final_ckpt = self._save_checkpoint(int(self.cfg.steps))
            if final_ckpt:
                print(f"[self_evolving] final checkpoint: {final_ckpt}")
                if self.last_lora_checkpoint_dir:
                    print(f"[self_evolving] final role LoRA checkpoint: {self.last_lora_checkpoint_dir}")

        summary = {
            "steps_total": int(self.cfg.steps),
            "steps_started_from": int(self.start_step),
            "steps_valid": int(valid_steps),
            "steps_skipped": int(skipped_steps),
            "mean_reward": float(reward_sum / float(max(1, valid_steps))),
            "nonzero_reward_rate": float(reward_nonzero / float(max(1, valid_steps))),
            "dual_track_disagree_rate": float(dual_track_disagree / float(max(1, valid_steps))),
            "output_dir": self.output_dir,
            "rollouts_log_path": self.rollouts_log_path,
            "suder_generation_enabled": bool(self.cfg.suder_generation_enabled),
            "generation_rollouts_log_path": self.generation_rollouts_log_path if self.cfg.suder_generation_enabled else "",
            "suder_steps_valid": int(suder_valid_steps),
            "suder_steps_skipped": int(suder_skipped_steps),
            "suder_mean_reward": float(suder_reward_sum / float(max(1, suder_valid_steps))),
            "suder_mean_entropy_nats": float(suder_entropy_sum / float(max(1, suder_valid_steps))),
            "suder_mean_quality": float(suder_quality_sum / float(max(1, suder_valid_steps))),
            "policy_updates_enabled": bool(self.policy_updates_enabled),
            "policy_updates_attempted": int(policy_updates_attempted),
            "policy_updates_applied": int(policy_updates_applied),
            "proposer_baseline_final": float(self.proposer_baseline),
            "solver_baseline_final": float(self.solver_baseline),
            "proposer_gen_baseline_final": float(self.proposer_gen_baseline),
            "generator_baseline_final": float(self.generator_baseline),
            "optimizer_flush_steps": int(flushed_optim_steps),
            "last_checkpoint_path": str(self.last_checkpoint_path),
            "last_lora_checkpoint_dir": str(self.last_lora_checkpoint_dir),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self._append_metrics(
            {
                "kind": "final_summary",
                **self._progress_core(
                    step=int(self.cfg.steps),
                    phase="completed",
                    run_started_at=run_started_at,
                ),
                **summary,
            }
        )
        self._write_status(
            state="completed",
            progress=self._progress_core(
                step=int(self.cfg.steps),
                phase="completed",
                run_started_at=run_started_at,
            ),
            metrics={k: v for k, v in summary.items() if isinstance(v, (bool, int, float, str))},
        )
        return summary
