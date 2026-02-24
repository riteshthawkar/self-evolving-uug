# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from .config import RolloutConfig
from .model_loader import BagelRuntime
from .prompts import is_objective_question, parse_answer, parse_first_question
from .rewards import compute_dual_track_reward, normalize_answer
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


class SelfEvolvingUnderstandingTrainer:
    """BAGEL self-evolving understanding rollout trainer.

    This first implementation focuses on high-fidelity rollout, reward shaping,
    and diagnostics logging. Update hooks are intentionally separated so policy
    optimization can be added without destabilizing data/metric correctness.
    """

    def __init__(self, runtime: BagelRuntime, cfg: RolloutConfig) -> None:
        self.runtime = runtime
        self.cfg = cfg
        self.adapter = BagelRolloutAdapter(runtime)
        self.image_paths = _list_images(cfg.image_dir)
        self.output_dir = self._prepare_output_dir(cfg.output_dir)
        self.rollouts_log_path = os.path.join(self.output_dir, "rollouts.jsonl")
        self.summary_path = os.path.join(self.output_dir, "summary.json")
        self.config_path = os.path.join(self.output_dir, "config.json")
        self._persist_config()

    def _prepare_output_dir(self, output_root: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_root, f"understanding_rollout_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _persist_config(self) -> None:
        payload = asdict(self.cfg)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _sample_image_path(self, step: int) -> str:
        idx = (step - 1) % len(self.image_paths)
        return self.image_paths[idx]

    def _load_image(self, path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def run(self) -> Dict[str, float]:
        random.seed(self.cfg.seed)

        reward_sum = 0.0
        reward_nonzero = 0
        dual_track_disagree = 0
        valid_steps = 0
        skipped_steps = 0

        solver_temps = self.cfg.solver_temperatures()
        for step in range(1, self.cfg.steps + 1):
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
                continue

            solver_outputs_raw: List[str] = []
            solver_answers_norm: List[str] = []
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

            valid_steps += 1
            reward_sum += reward
            reward_nonzero += int(abs(reward) > 1e-9)
            dual_track_disagree += int(not dual.dual_track_agree)

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
                    "solver_temperatures": solver_temps,
                },
            )

            if step % max(1, self.cfg.log_every) == 0:
                mean_reward = reward_sum / float(max(1, valid_steps))
                disagree_rate = dual_track_disagree / float(max(1, valid_steps))
                print(
                    f"[self_evolving][step={step}] valid={valid_steps} skipped={skipped_steps} "
                    f"mean_reward={mean_reward:.4f} disagree_rate={disagree_rate:.4f}"
                )

        summary = {
            "steps_total": int(self.cfg.steps),
            "steps_valid": int(valid_steps),
            "steps_skipped": int(skipped_steps),
            "mean_reward": float(reward_sum / float(max(1, valid_steps))),
            "nonzero_reward_rate": float(reward_nonzero / float(max(1, valid_steps))),
            "dual_track_disagree_rate": float(dual_track_disagree / float(max(1, valid_steps))),
            "output_dir": self.output_dir,
            "rollouts_log_path": self.rollouts_log_path,
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

