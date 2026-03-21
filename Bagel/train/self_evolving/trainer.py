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

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover - distributed optional at runtime
    dist = None

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
    clip_similarity,
    clip_text_similarity,
    compute_dual_track_reward,
    compute_generation_spec_quality,
    gaussian_reward,
    majority_vote,
    normalize_answer,
    per_candidate_diversity_scores,
    shannon_entropy_nats,
    soft_match_score,
    yes_no_polarity,
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
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if world_size > 1:
        per_rank_output = str(os.environ.get("BAGEL_DIST_PER_RANK_OUTPUT", "0")).strip().lower() in {"1", "true", "yes", "on"}
        if not per_rank_output:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or "0")
            if rank != 0:
                return
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
        self.dist_enabled = bool(cfg.dist_enabled) and self._dist_ready()
        self.dist_world_size = max(1, int(cfg.dist_world_size))
        self.dist_rank = max(0, int(cfg.dist_rank))
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

    @staticmethod
    def _dist_ready() -> bool:
        return bool(dist is not None and dist.is_available() and dist.is_initialized())

    def _prepare_output_dir(self, output_root: str) -> str:
        # Output layout can be controlled from launcher scripts:
        # - BAGEL_OUTPUT_DIR_MODE=direct: write logs/checkpoints directly in output_root
        # - BAGEL_OUTPUT_DIR_MODE=timestamp (default): create per-run timestamp folder
        mode = str(os.environ.get("BAGEL_OUTPUT_DIR_MODE", "timestamp")).strip().lower()
        per_rank_output = str(os.environ.get("BAGEL_DIST_PER_RANK_OUTPUT", "0")).strip().lower() in {"1", "true", "yes", "on"}
        rank_suffix = ""
        if self.dist_enabled and self.dist_world_size > 1 and per_rank_output:
            rank_suffix = f"rank_{int(self.dist_rank):02d}"
        if mode in {"direct", "flat", "inplace"}:
            out_dir = os.path.join(output_root, rank_suffix) if rank_suffix else output_root
            os.makedirs(out_dir, exist_ok=True)
            return out_dir

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"understanding_rollout_{ts}"
        if rank_suffix:
            run_name = f"{run_name}_{rank_suffix}"
        run_dir = os.path.join(output_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _should_write_artifacts(self) -> bool:
        if not self.dist_enabled or self.dist_world_size <= 1:
            return True
        per_rank_output = str(os.environ.get("BAGEL_DIST_PER_RANK_OUTPUT", "0")).strip().lower() in {"1", "true", "yes", "on"}
        if per_rank_output:
            return True
        return int(self.dist_rank) == 0

    def _persist_config(self) -> None:
        if not self._should_write_artifacts():
            return
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
        if not self._should_write_artifacts():
            return
        _write_jsonl(self.metrics_log_path, record)

    def _write_status(self, *, state: str, progress: Dict, metrics: Dict, last_error: str = "") -> None:
        if not self._should_write_artifacts():
            return
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
        if self.dist_enabled and self.dist_world_size > 1 and bool(getattr(self.cfg, "dist_data_shard", True)):
            idx = ((int(step) - 1) * int(self.dist_world_size) + int(self.dist_rank)) % len(self.image_paths)
        else:
            idx = (step - 1) % len(self.image_paths)
        return self.image_paths[idx]

    def _sync_scalar(self, value: float) -> float:
        if not self.dist_enabled or self.dist_world_size <= 1:
            return float(value)
        t = torch.tensor([float(value)], dtype=torch.float32, device=self.runtime.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= float(self.dist_world_size)
        return float(t.item())

    def _sync_trainable_adapter_params(self) -> None:
        if not self.dist_enabled or self.dist_world_size <= 1:
            return
        if not bool(self.runtime.lora_enabled):
            return
        language_model = self.runtime.model.language_model
        for param in language_model.parameters():
            if not bool(getattr(param, "requires_grad", False)):
                continue
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= float(self.dist_world_size)

    def _sync_distributed_step_state(self) -> None:
        if not self.dist_enabled or self.dist_world_size <= 1:
            return
        self._sync_trainable_adapter_params()
        self.proposer_baseline = self._sync_scalar(self.proposer_baseline)
        self.solver_baseline = self._sync_scalar(self.solver_baseline)
        self.proposer_gen_baseline = self._sync_scalar(self.proposer_gen_baseline)
        self.generator_baseline = self._sync_scalar(self.generator_baseline)

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

    def _mean_entropy_from_qa_logs(self, qa_logs: List[Dict]) -> float:
        entropy_vals = []
        for qa_log in qa_logs:
            if str(qa_log.get("status", "")) != "ok":
                continue
            entropy = qa_log.get("entropy_nats")
            if entropy is not None:
                entropy_vals.append(float(entropy))
        return _mean(entropy_vals)

    def _qa_confidence_from_logs(self, qa_logs: List[Dict]) -> float:
        confidences = []
        for qa_log in qa_logs:
            if str(qa_log.get("status", "")) != "ok":
                continue
            majority_fraction = qa_log.get("majority_fraction")
            if majority_fraction is not None:
                confidences.append(float(majority_fraction))
        return _mean(confidences)

    def _cycle_reward(self, *, prompt: str, image: Image.Image) -> Tuple[float, str]:
        caption_out = self.adapter.caption_image(
            image=image,
            max_new_tokens=int(getattr(self.cfg, "max_new_tokens_solver", 96)),
            temperature=max(0.2, min(float(self.cfg.proposer_temperature), 0.8)),
            do_sample=False,
        )
        caption = " ".join(str(caption_out.text or "").split())
        caption_sim = clip_text_similarity(prompt, caption) if caption else 0.0
        image_text_sim = clip_similarity(image, prompt)
        score = 0.5 * float(caption_sim) + 0.5 * float(image_text_sim)
        return _clamp01(score), caption

    def _score_generation_candidate(
        self,
        *,
        generated: Image.Image,
        spec,
        solver_temps: List[float],
    ) -> Tuple[float, float, List[Dict]]:
        qa_logs: List[Dict] = []
        score_values: List[float] = []
        contradiction_values: List[float] = []

        for qa_idx, qa in enumerate(spec.qa_pairs):
            solver_outputs_raw: List[str] = []
            solver_answers_norm: List[str] = []
            for temp in solver_temps:
                out = self.adapter.solve_question(
                    image=generated,
                    question=qa.question,
                    max_new_tokens=self.cfg.max_new_tokens_solver,
                    temperature=float(temp),
                    do_sample=True,
                )
                solver_outputs_raw.append(out.text)
                answer = normalize_answer(parse_answer(out.text))
                if answer:
                    solver_answers_norm.append(answer)

            expected_answer = normalize_answer(qa.answer)
            if not solver_answers_norm:
                qa_logs.append(
                    {
                        "qa_index": int(qa_idx),
                        "status": "skipped",
                        "skip_reason": "empty_solver_answers",
                        "question": qa.question,
                        "expected_answer": expected_answer,
                        "solver_outputs_raw": solver_outputs_raw if self.cfg.save_raw_generations else [],
                    }
                )
                continue

            entropy_nats, majority_fraction, majority_answer = self._compute_solver_entropy_and_majority(
                solver_answers_norm
            )
            match_score = soft_match_score(majority_answer, expected_answer)
            combined_score = 0.7 * float(match_score) + 0.3 * float(majority_fraction)
            contradiction = 1.0 if (
                yes_no_polarity(expected_answer) != 0
                and yes_no_polarity(majority_answer) != 0
                and yes_no_polarity(expected_answer) != yes_no_polarity(majority_answer)
            ) else 0.0

            score_values.append(float(combined_score))
            contradiction_values.append(float(contradiction))
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
                    "combined_score": float(combined_score),
                    "contradiction": float(contradiction),
                    "solver_answers_norm": solver_answers_norm,
                    "solver_outputs_raw": solver_outputs_raw if self.cfg.save_raw_generations else [],
                }
            )

        spec_score = _mean(score_values)
        contradiction_score = _mean(contradiction_values)
        return float(spec_score), float(contradiction_score), qa_logs

    def _run_generation_solver_policy_updates(
        self,
        *,
        generated: Image.Image,
        spec,
        solver_temps: List[float],
    ) -> Tuple[int, int, List[Dict], List[float]]:
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
        if not gen_solver_update_enabled:
            return 0, 0, [], []

        attempts = 0
        applied = 0
        update_stats: List[Dict] = []
        reward_values: List[float] = []

        for qa in spec.qa_pairs:
            solver_samples: List[Tuple[str, str]] = []
            for temp in solver_temps:
                out = self.adapter.solve_question(
                    image=generated,
                    question=qa.question,
                    max_new_tokens=self.cfg.max_new_tokens_solver,
                    temperature=float(temp),
                    do_sample=True,
                )
                answer = normalize_answer(parse_answer(out.text))
                if answer:
                    solver_samples.append((out.text, answer))
            if not solver_samples:
                continue

            expected_answer = normalize_answer(qa.answer)
            solver_prompt = build_solver_prompt(qa.question)
            group_rewards = [
                float(answer_match_score(sample_answer, expected_answer))
                for _, sample_answer in solver_samples
            ]
            for idx, (sample_raw, _) in enumerate(solver_samples):
                if max_gen_solver_updates > 0 and int(attempts) >= int(max_gen_solver_updates):
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
                update_stats.append(stats)
                attempts += 1
                if not bool(stats.get("skipped", True)):
                    applied += 1
                reward_values.append(sample_reward)
        return int(attempts), int(applied), update_stats, reward_values

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
        def _read_pointer(pointer_file: Path) -> Optional[str]:
            try:
                raw = pointer_file.read_text(encoding="utf-8").strip()
            except Exception:
                return None
            if not raw:
                return None
            target = Path(raw)
            if not target.is_absolute():
                target = (pointer_file.parent / target).resolve()
            return str(target)

        p = Path(path).expanduser()
        if p.is_file():
            return str(p)
        if p.is_dir():
            trainer_state = p / "trainer_state.pt"
            if trainer_state.is_file():
                return str(trainer_state)
            checkpoint_target = p / "checkpoint_target.json"
            if checkpoint_target.is_file():
                try:
                    payload = json.loads(checkpoint_target.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}
                target = str(payload.get("trainer_state_path", "") or "").strip()
                if target:
                    return self._resolve_checkpoint_path(target)
            for latest_file in (p / "latest.txt", p / "checkpoints" / "latest.txt"):
                if latest_file.is_file():
                    target = _read_pointer(latest_file)
                    if target:
                        return self._resolve_checkpoint_path(target)
            if p.name.endswith("_lora"):
                base_name = p.name[: -len("_lora")]
                sibling = p.with_name(f"{base_name}.pt")
                if sibling.is_file():
                    return str(sibling)
            alias_dirs = sorted(d for d in p.glob("step_*") if d.is_dir())
            if alias_dirs:
                return self._resolve_checkpoint_path(str(alias_dirs[-1]))
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
        if not self._should_write_artifacts():
            return self.last_checkpoint_path

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
        standardized_dir = self._sync_standard_checkpoint_dir(step=step, checkpoint_path=Path(path), lora_dir=Path(role_ckpt_dir) if role_ckpt_dir else None)
        with open(os.path.join(self.checkpoint_dir, "latest.txt"), "w", encoding="utf-8") as f:
            f.write((standardized_dir or path) + "\n")
        return path

    @staticmethod
    def _remove_path(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.exists():
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        try:
            rel_src = os.path.relpath(str(src), str(dst.parent))
            os.symlink(rel_src, str(dst), target_is_directory=src.is_dir())
            return
        except Exception:
            pass

        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    def _sync_standard_checkpoint_dir(
        self,
        *,
        step: int,
        checkpoint_path: Path,
        lora_dir: Optional[Path],
    ) -> str:
        alias_dir = Path(self.checkpoint_dir) / f"step_{int(step):06d}"
        tmp_dir = Path(self.checkpoint_dir) / f".{alias_dir.name}.tmp"
        self._remove_path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        self._link_or_copy(checkpoint_path, tmp_dir / "trainer_state.pt")
        if lora_dir is not None and lora_dir.exists():
            self._link_or_copy(lora_dir, tmp_dir / "lora_roles")
        with (tmp_dir / "checkpoint_target.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": int(step),
                    "trainer_state_path": str(checkpoint_path),
                    "lora_roles_dir": str(lora_dir) if lora_dir is not None else "",
                },
                f,
                indent=2,
            )

        self._remove_path(alias_dir)
        os.replace(str(tmp_dir), str(alias_dir))
        return str(alias_dir)

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
        base_temp = float(self.cfg.gen_spec_temperature)
        retry_temps = [base_temp, max(0.1, min(0.7, base_temp * 0.6)), 0.25]
        dedup_retry_temps: List[float] = []
        for temp in retry_temps:
            t = float(round(temp, 4))
            if t not in dedup_retry_temps:
                dedup_retry_temps.append(t)

        spec = None
        spec_out = None
        spec_retry_attempted = False
        spec_retry_temperature = 0.0
        spec_retry_count = 0
        for attempt_idx, temp in enumerate(dedup_retry_temps):
            if attempt_idx > 0:
                spec_retry_attempted = True
                spec_retry_temperature = float(temp)
                spec_retry_count = int(attempt_idx)
            candidate_out = self.adapter.propose_generation_spec(
                image=image,
                max_new_tokens=self.cfg.max_new_tokens_gen_spec,
                temperature=float(temp),
                min_qa_pairs=int(self.cfg.gen_spec_min_qa_pairs),
                do_sample=bool(attempt_idx < (len(dedup_retry_temps) - 1)),
            )
            if spec_out is None:
                spec_out = candidate_out
            candidate_spec = parse_generation_spec(
                candidate_out.text,
                min_qa_pairs=int(self.cfg.gen_spec_min_qa_pairs),
            )
            if candidate_spec is not None:
                spec = candidate_spec
                spec_out = candidate_out
                break

        if spec is None:
            return {
                "step": int(step),
                "status": "skipped",
                "skip_reason": "invalid_generation_spec",
                "image_path": image_path,
                "proposer_spec_raw": (spec_out.text if (spec_out is not None and self.cfg.save_raw_generations) else ""),
                "policy_update_attempted": False,
                "policy_update_applied": False,
                "policy_update_reason": "invalid_generation_spec",
                "spec_retry_attempted": bool(spec_retry_attempted),
                "spec_retry_temperature": float(spec_retry_temperature),
                "spec_retry_count": int(spec_retry_count),
            }

        spec_quality, spec_quality_details = compute_generation_spec_quality(
            qa_pairs=list(spec.qa_pairs),
            min_spec_qa_pairs=int(getattr(self.cfg, "min_spec_qa_pairs", self.cfg.gen_spec_min_qa_pairs)),
            max_question_words=int(getattr(self.cfg, "max_question_words", 24)),
            max_expected_words=int(getattr(self.cfg, "max_expected_words", 8)),
        )

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

        candidate_images: List[Image.Image] = []
        generation_num_candidates = max(1, int(getattr(self.cfg, "generation_num_candidates", 1)))
        for _ in range(generation_num_candidates):
            generated = self.adapter.generate_image_from_spec(
                spec=gen_spec_prompt,
                cfg_text_scale=float(self.cfg.generation_cfg_text_scale),
                cfg_img_scale=float(self.cfg.generation_cfg_img_scale),
                num_timesteps=int(self.cfg.generation_num_timesteps),
                timestep_shift=float(self.cfg.generation_timestep_shift),
                image_size=int(self.cfg.generation_image_size),
            )
            if generated is not None:
                candidate_images.append(generated)

        if not candidate_images:
            return {
                "step": int(step),
                "status": "skipped",
                "skip_reason": "generation_failed",
                "image_path": image_path,
                "spec_prompt": spec.prompt,
                "qa_pair_count": len(spec.qa_pairs),
                "proposer_spec_raw": spec_out.text if self.cfg.save_raw_generations else "",
                "spec_prompt_truncated": bool(spec_prompt_truncated),
                "spec_quality": float(spec_quality),
                "spec_quality_details": spec_quality_details,
                "policy_update_attempted": False,
                "policy_update_applied": False,
                "policy_update_reason": "generation_failed",
            }

        generated_image_paths: List[str] = []
        if self.cfg.save_generated_images:
            for cand_idx, generated in enumerate(candidate_images):
                generated_image_path = os.path.join(
                    self.generated_images_dir,
                    f"step_{step:06d}_{_safe_filename(Path(image_path).stem)}_cand_{cand_idx:02d}.png",
                )
                generated.save(generated_image_path)
                generated_image_paths.append(generated_image_path)

        diversity_scores = per_candidate_diversity_scores(candidate_images)
        positive_weight_sum = (
            float(getattr(self.cfg, "reward_spec_weight", 0.65))
            + float(getattr(self.cfg, "reward_cycle_weight", 0.20))
            + float(getattr(self.cfg, "reward_diversity_weight", 0.10))
        )
        if positive_weight_sum <= 0.0:
            positive_weight_sum = 1.0
        weight_spec = float(getattr(self.cfg, "reward_spec_weight", 0.65)) / positive_weight_sum
        weight_cycle = float(getattr(self.cfg, "reward_cycle_weight", 0.20)) / positive_weight_sum
        weight_diversity = float(getattr(self.cfg, "reward_diversity_weight", 0.10)) / positive_weight_sum

        scored_candidates: List[Dict] = []
        for cand_idx, generated in enumerate(candidate_images):
            spec_score, contradiction_score, qa_logs = self._score_generation_candidate(
                generated=generated,
                spec=spec,
                solver_temps=solver_temps,
            )
            cycle_score, cycle_caption = self._cycle_reward(
                prompt=gen_spec_prompt,
                image=generated,
            )
            diversity_score = float(diversity_scores[cand_idx]) if cand_idx < len(diversity_scores) else 0.0
            base_reward = (
                weight_spec * float(spec_score)
                + weight_cycle * float(cycle_score)
                + weight_diversity * float(diversity_score)
                - float(getattr(self.cfg, "reward_contradiction_weight", 0.20)) * float(contradiction_score)
            )
            base_reward = _clamp01(base_reward)
            total_reward = _clamp01(float(spec_quality) * float(base_reward))
            scored_candidates.append(
                {
                    "candidate_idx": int(cand_idx),
                    "image": generated,
                    "generated_image_path": (
                        generated_image_paths[cand_idx] if cand_idx < len(generated_image_paths) else ""
                    ),
                    "spec_score": float(spec_score),
                    "contradiction_score": float(contradiction_score),
                    "cycle_score": float(cycle_score),
                    "cycle_caption": cycle_caption,
                    "diversity_score": float(diversity_score),
                    "base_reward": float(base_reward),
                    "total_reward": float(total_reward),
                    "qa_confidence": float(self._qa_confidence_from_logs(qa_logs)),
                    "mean_entropy_nats": float(self._mean_entropy_from_qa_logs(qa_logs)),
                    "qa_logs": qa_logs,
                }
            )

        valid_candidates = [
            candidate for candidate in scored_candidates
            if any(str(log.get("status", "")) == "ok" for log in candidate.get("qa_logs", []))
        ]
        if not valid_candidates:
            return {
                "step": int(step),
                "status": "skipped",
                "skip_reason": "empty_generation_qa_entropy",
                "image_path": image_path,
                "spec_prompt": spec.prompt,
                "qa_pair_count": len(spec.qa_pairs),
                "generated_image_path": generated_image_paths[0] if generated_image_paths else "",
                "generated_image_paths": generated_image_paths,
                "spec_quality": float(spec_quality),
                "spec_quality_details": spec_quality_details,
                "policy_update_attempted": False,
                "policy_update_applied": False,
                "policy_update_reason": "empty_generation_qa_entropy",
                "generation_candidate_rewards": [float(c["total_reward"]) for c in scored_candidates],
            }

        best = max(
            valid_candidates,
            key=lambda candidate: (
                float(candidate.get("total_reward", 0.0)),
                float(candidate.get("spec_score", 0.0)),
                float(candidate.get("cycle_score", 0.0)),
            ),
        )
        best_reward = float(best.get("total_reward", 0.0))
        mean_entropy = float(best.get("mean_entropy_nats", 0.0))
        entropy_component = gaussian_reward(
            mean_entropy,
            float(self.cfg.proposer_entropy_mu),
            float(self.cfg.proposer_entropy_sigma),
        )
        if mean_entropy <= float(self.cfg.zero_entropy_eps):
            entropy_component = -max(0.0, float(self.cfg.zero_entropy_reward_cap))
        quality_component = _clamp01(best_reward)
        alpha = _clamp01(float(self.cfg.proposer_gen_entropy_weight))
        proposer_reward = max(
            -1.0,
            min(1.0, alpha * float(entropy_component) + (1.0 - alpha) * float(quality_component)),
        )

        quality_gate_ok = bool(
            float(spec_quality) >= float(getattr(self.cfg, "min_spec_quality_for_update", 0.35))
            and len(spec.qa_pairs) >= int(getattr(self.cfg, "min_spec_qa_pairs", self.cfg.gen_spec_min_qa_pairs))
        )

        proposer_baseline_before = float(self.proposer_gen_baseline)
        generator_baseline_before = float(self.generator_baseline)
        proposer_baseline_after = float(self.proposer_gen_baseline)
        generator_baseline_after = float(self.generator_baseline)

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
        if quality_gate_ok and gen_solver_update_enabled:
            (
                gen_solver_update_attempted,
                gen_solver_update_applied,
                gen_solver_update_stats,
                gen_solver_reward_values,
            ) = self._run_generation_solver_policy_updates(
                generated=best["image"],
                spec=spec,
                solver_temps=solver_temps,
            )

        if gen_solver_reward_values:
            momentum_solver = _clamp01(float(self.cfg.baseline_momentum))
            mean_solver_reward = float(sum(gen_solver_reward_values) / float(len(gen_solver_reward_values)))
            self.solver_baseline = (
                momentum_solver * float(self.solver_baseline)
                + (1.0 - momentum_solver) * mean_solver_reward
            )

        proposer_policy_update_attempted = False
        proposer_policy_update_applied = False
        proposer_policy_update_reason = "disabled"
        proposer_policy_update_stats: Dict = {}
        if (
            self.policy_updates_enabled
            and self.cfg.train_generation_proposer
            and self.proposer_updater is not None
            and quality_gate_ok
        ):
            proposer_policy_update_attempted = True
            proposer_baseline_after = (
                _clamp01(float(self.cfg.proposer_gen_baseline_momentum)) * float(self.proposer_gen_baseline)
                + (1.0 - _clamp01(float(self.cfg.proposer_gen_baseline_momentum))) * float(proposer_reward)
            )
            self.proposer_gen_baseline = float(proposer_baseline_after)
            proposer_policy_update_stats = self.proposer_updater.step(
                image=image,
                prompt=build_generation_spec_prompt(min_qa_pairs=int(self.cfg.gen_spec_min_qa_pairs)),
                completion=spec_out.text,
                reward=float(proposer_reward),
                baseline=proposer_baseline_before,
            )
            proposer_policy_update_applied = bool(not proposer_policy_update_stats.get("skipped", True))
            proposer_policy_update_reason = str(proposer_policy_update_stats.get("reason", "unknown"))
        elif (
            self.policy_updates_enabled
            and self.cfg.train_generation_proposer
            and self.proposer_updater is not None
            and not quality_gate_ok
        ):
            proposer_policy_update_reason = "low_spec_quality"

        generator_policy_update_attempted = False
        generator_policy_update_applied = False
        generator_policy_update_reason = "disabled"
        generator_policy_update_stats: Dict = {}
        if (
            self.policy_updates_enabled
            and self.cfg.train_generator
            and self.generator_updater is not None
            and quality_gate_ok
        ):
            generator_policy_update_attempted = True
            generator_baseline_after = (
                _clamp01(float(self.cfg.proposer_gen_baseline_momentum)) * float(self.generator_baseline)
                + (1.0 - _clamp01(float(self.cfg.proposer_gen_baseline_momentum))) * float(best_reward)
            )
            self.generator_baseline = float(generator_baseline_after)
            generator_policy_update_stats = self.generator_updater.step(
                image=best["image"],
                prompt=gen_spec_prompt,
                reward=float(best_reward),
                baseline=generator_baseline_before,
                group_rewards=[float(candidate.get("total_reward", 0.0)) for candidate in valid_candidates],
            )
            generator_policy_update_applied = bool(not generator_policy_update_stats.get("skipped", True))
            generator_policy_update_reason = str(generator_policy_update_stats.get("reason", "unknown"))
        elif (
            self.policy_updates_enabled
            and self.cfg.train_generator
            and self.generator_updater is not None
            and not quality_gate_ok
        ):
            generator_policy_update_reason = "low_spec_quality"

        policy_update_attempted = bool(
            proposer_policy_update_attempted or generator_policy_update_attempted
        )
        policy_update_applied = bool(
            proposer_policy_update_applied or generator_policy_update_applied
        )
        policy_update_reasons: List[str] = []
        if proposer_policy_update_attempted or proposer_policy_update_reason == "low_spec_quality":
            policy_update_reasons.append(f"proposer:{proposer_policy_update_reason}")
        if generator_policy_update_attempted or generator_policy_update_reason == "low_spec_quality":
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
            "spec_quality": float(spec_quality),
            "spec_quality_details": spec_quality_details,
            "qa_pair_count": len(spec.qa_pairs),
            "qa_pairs": [{"question": qa.question, "answer": normalize_answer(qa.answer)} for qa in spec.qa_pairs],
            "proposer_spec_raw": spec_out.text if self.cfg.save_raw_generations else "",
            "generated_image_path": str(best.get("generated_image_path", "")),
            "generated_image_paths": generated_image_paths,
            "best_candidate_idx": int(best.get("candidate_idx", 0)),
            "generation_candidate_rewards": [float(candidate.get("total_reward", 0.0)) for candidate in scored_candidates],
            "generation_candidate_details": [
                {
                    "candidate_idx": int(candidate.get("candidate_idx", 0)),
                    "generated_image_path": str(candidate.get("generated_image_path", "")),
                    "spec_score": float(candidate.get("spec_score", 0.0)),
                    "cycle_score": float(candidate.get("cycle_score", 0.0)),
                    "diversity_score": float(candidate.get("diversity_score", 0.0)),
                    "contradiction_score": float(candidate.get("contradiction_score", 0.0)),
                    "base_reward": float(candidate.get("base_reward", 0.0)),
                    "total_reward": float(candidate.get("total_reward", 0.0)),
                    "mean_entropy_nats": float(candidate.get("mean_entropy_nats", 0.0)),
                    "qa_confidence": float(candidate.get("qa_confidence", 0.0)),
                }
                for candidate in scored_candidates
            ],
            "qa_logs": list(best.get("qa_logs", [])),
            "mean_entropy_nats": float(mean_entropy),
            "entropy_component": float(entropy_component),
            "quality_component": float(quality_component),
            "proposer_gen_reward": float(proposer_reward),
            "best_total_reward": float(best_reward),
            "best_spec_score": float(best.get("spec_score", 0.0)),
            "best_cycle_score": float(best.get("cycle_score", 0.0)),
            "best_diversity_score": float(best.get("diversity_score", 0.0)),
            "best_contradiction_score": float(best.get("contradiction_score", 0.0)),
            "best_base_reward": float(best.get("base_reward", 0.0)),
            "best_cycle_caption": str(best.get("cycle_caption", "")),
            "best_qa_confidence": float(best.get("qa_confidence", 0.0)),
            "entropy_weight_alpha": float(alpha),
            "proposer_gen_baseline_before": float(proposer_baseline_before),
            "proposer_gen_baseline_after": float(self.proposer_gen_baseline),
            "proposer_gen_advantage": float(proposer_reward - proposer_baseline_before),
            "generator_baseline_before": float(generator_baseline_before),
            "generator_baseline_after": float(self.generator_baseline),
            "generator_advantage": float(best_reward - generator_baseline_before),
            "quality_gate_ok": bool(quality_gate_ok),
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
            "spec_retry_count": int(spec_retry_count),
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
                do_sample=True,
            )
            question = parse_first_question(proposer.text)
            proposer_retry_attempted = False
            proposer_retry_count = 0
            proposer_retry_temps: List[float] = []
            proposer_retry_recovered = False
            retry_budget = max(0, int(os.environ.get("BAGEL_PROPOSER_PARSE_RETRIES", "2") or "2"))
            retry_decay = float(os.environ.get("BAGEL_PROPOSER_PARSE_RETRY_TEMP_DECAY", "0.70") or "0.70")
            if retry_decay <= 0.0 or retry_decay >= 1.0:
                retry_decay = 0.70
            while (not question) and proposer_retry_count < retry_budget:
                proposer_retry_attempted = True
                proposer_retry_count += 1
                retry_temp = max(
                    0.15,
                    min(4.0, float(self.cfg.proposer_temperature) * (retry_decay ** proposer_retry_count)),
                )
                proposer_retry_temps.append(float(retry_temp))
                proposer_retry = self.adapter.propose_question(
                    image=image,
                    max_new_tokens=self.cfg.max_new_tokens_proposer,
                    temperature=float(retry_temp),
                    do_sample=bool(proposer_retry_count < retry_budget),
                )
                retry_question = parse_first_question(proposer_retry.text)
                if retry_question:
                    proposer = proposer_retry
                    question = retry_question
                    proposer_retry_recovered = True
                    break
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
                        "proposer_retry_attempted": bool(proposer_retry_attempted),
                        "proposer_retry_count": int(proposer_retry_count),
                        "proposer_retry_temps": proposer_retry_temps,
                        "proposer_retry_recovered": bool(proposer_retry_recovered),
                    },
                )
                _run_and_accumulate_suder(step, image_path, image)
                self._sync_distributed_step_state()
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
                        "proposer_retry_attempted": bool(proposer_retry_attempted),
                        "proposer_retry_count": int(proposer_retry_count),
                        "proposer_retry_temps": proposer_retry_temps,
                        "proposer_retry_recovered": bool(proposer_retry_recovered),
                    },
                )
                _run_and_accumulate_suder(step, image_path, image)
                self._sync_distributed_step_state()
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
                self._sync_distributed_step_state()
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
            self._sync_distributed_step_state()
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

        self._sync_distributed_step_state()

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
