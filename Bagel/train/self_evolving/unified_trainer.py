# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .prompts import (
    build_generation_spec_prompt,
    build_proposer_prompt,
    build_solver_prompt,
    is_objective_question,
    parse_all_questions,
    parse_answer,
    parse_first_question,
)
from .replay_buffer import ReplayBuffer
from .rewards import (
    answer_match_score,
    compute_dual_track_reward,
    normalize_answer,
)
from .trainer import (
    SelfEvolvingUnderstandingTrainer,
    _clamp01,
    _mean,
    _write_jsonl,
)


def _parse_generated_mix_meta(path: Path, min_reward: float) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    reward = float(payload.get("reward", -1.0))
    if reward < float(min_reward):
        return None

    image_path_raw = str(payload.get("image_path", "")).strip()
    if not image_path_raw:
        image_path = path.with_suffix(".png")
    else:
        image_path = Path(image_path_raw)
        if not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()
    if not image_path.exists():
        return None

    questions = payload.get("questions", [])
    answers = payload.get("reference_answers", [])
    if not isinstance(questions, list) or not isinstance(answers, list):
        return None
    n = min(len(questions), len(answers))
    if n <= 0:
        return None
    questions = [str(v).strip() for v in questions[:n] if str(v).strip()]
    answers = [str(v).strip() for v in answers[:n] if str(v).strip()]
    n = min(len(questions), len(answers))
    if n <= 0:
        return None
    questions = questions[:n]
    answers = answers[:n]

    return {
        "meta_path": str(path.resolve()),
        "image_path": str(image_path.resolve()),
        "prompt": str(payload.get("prompt", "")),
        "questions": questions,
        "reference_answers": answers,
        "reward": reward,
        "step_generated": int(payload.get("step_generated", 0)),
    }


class UnifiedSelfEvolvingTrainer(SelfEvolvingUnderstandingTrainer):
    """BAGEL unified self-evolving trainer with alternating U/G schedule."""

    def _prepare_output_dir(self, output_root: str) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_root, f"unified_rollout_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def __init__(self, runtime, cfg) -> None:
        super().__init__(runtime=runtime, cfg=cfg)
        self.ucfg = cfg
        self._gen_mix_source_mode = cfg.normalized_gen_mix_source_mode()
        self._generated_mix_dir = (
            Path(str(cfg.generated_mix_dir).strip()).expanduser().resolve()
            if str(cfg.generated_mix_dir or "").strip()
            else Path(self.output_dir).resolve() / "generated_mix_pool"
        )
        self._generated_mix_cache: List[Dict[str, Any]] = []
        self._generated_mix_last_refresh_step = -10**9
        self._gen_reward_ema = 0.0
        self._gen_reward_ema_initialized = False
        self._ste_window: List[float] = []
        self._ste_window_size = max(8, int(getattr(cfg, "solver_token_entropy_window_size", 128)))

        self.replay_buffer: Optional[ReplayBuffer] = None
        if self._gen_mix_source_mode == "buffer":
            self.replay_buffer = ReplayBuffer(
                max_size=max(1, int(cfg.replay_buffer_size)),
                min_reward=float(cfg.replay_min_reward),
                max_staleness=max(0, int(cfg.replay_max_staleness)),
            )
        if self._gen_mix_source_mode == "folder" or bool(cfg.understanding_generated_only):
            self._generated_mix_dir.mkdir(parents=True, exist_ok=True)

        # Unified loop needs generation outputs to be persisted for folder/buffer reuse.
        if int(cfg.generation_steps_per_cycle) > 0 and not bool(cfg.save_generated_images):
            cfg.save_generated_images = True
            os.makedirs(self.generated_images_dir, exist_ok=True)
            self._persist_config()

    def _spot_solver_temperatures(self, solver_temps: List[float]) -> List[float]:
        if not solver_temps:
            return [float(self.cfg.solver_temp_min)]
        spot_n = max(1, int(getattr(self.cfg, "proposer_spot_check_samples", 3)))
        return list(solver_temps[: min(len(solver_temps), spot_n)])

    def _estimate_ste_difficulty(self, entropy_nats: float, entropy_cap: float) -> float:
        if not bool(getattr(self.cfg, "solver_token_entropy_enabled", True)):
            return 0.0
        raw = max(0.0, float(entropy_nats))
        self._ste_window.append(raw)
        while len(self._ste_window) > int(self._ste_window_size):
            self._ste_window.pop(0)

        if len(self._ste_window) >= 8:
            rank = sum(1 for v in self._ste_window if float(v) < raw)
            return float(rank) / float(max(1, len(self._ste_window)))

        alpha = max(0.1, float(getattr(self.cfg, "solver_token_entropy_sigmoid_alpha", 1.5)))
        beta_cfg = max(0.0, float(getattr(self.cfg, "solver_token_entropy_sigmoid_beta", 2.0)))
        cap = max(1e-6, float(entropy_cap))
        beta = min(beta_cfg, cap)
        z = max(-40.0, min(40.0, alpha * (raw - beta)))
        return 1.0 / (1.0 + math.exp(-z))

    def _evaluate_proposer_candidate(
        self,
        *,
        image: Image.Image,
        question: str,
        spot_temps: List[float],
        candidate_index: int,
    ) -> Dict[str, Any]:
        outputs_raw: List[str] = []
        answers_norm: List[str] = []
        solver_samples: List[Any] = []
        for temp in spot_temps:
            out = self.adapter.solve_question(
                image=image,
                question=question,
                max_new_tokens=self.cfg.max_new_tokens_solver,
                temperature=float(temp),
                do_sample=True,
            )
            outputs_raw.append(out.text)
            ans = normalize_answer(parse_answer(out.text))
            if ans:
                answers_norm.append(ans)
                solver_samples.append((out.text, ans))

        if not answers_norm:
            return {
                "valid": False,
                "candidate_index": int(candidate_index),
                "question": question,
                "completion": f"<question>{question}</question>",
                "solver_outputs_raw": outputs_raw,
                "spot_answers_norm": [],
                "spot_solver_samples": [],
                "entropy_nats": 0.0,
                "majority_fraction": 0.0,
                "easy_case": True,
                "unsolvable_case": False,
                "non_objective": not is_objective_question(question),
                "reward": -1.0,
                "sample_entropy_difficulty": 0.0,
                "ste_difficulty": 0.0,
                "score": -1.0,
                "gate_passed": False,
                "acceptable": False,
            }

        dual = compute_dual_track_reward(
            answers=answers_norm,
            intuitive_answer="",
            entropy_mu=self.cfg.proposer_entropy_mu,
            entropy_sigma=self.cfg.proposer_entropy_sigma,
            unsolvable_maj_threshold=self.cfg.solver_unsolvable_maj_threshold,
            zero_entropy_eps=self.cfg.zero_entropy_eps,
        )

        reward = float(dual.reward)
        non_objective = not is_objective_question(question)
        if self.cfg.proposer_require_objective and non_objective:
            reward -= float(self.cfg.proposer_non_objective_penalty)
        if self.cfg.acceptance_require_non_easy and bool(dual.easy_case):
            reward -= float(self.cfg.rejected_question_penalty)
        reward = max(-1.0, min(1.0, reward))

        entropy_cap = max(1e-6, math.log(float(max(2, len(spot_temps)))))
        sample_entropy_difficulty = max(
            0.0,
            min(1.0, float(dual.entropy_nats) / float(entropy_cap)),
        )
        ste_difficulty = self._estimate_ste_difficulty(
            entropy_nats=float(dual.entropy_nats),
            entropy_cap=float(entropy_cap),
        )
        sample_weight = max(0.0, float(getattr(self.cfg, "proposer_sample_entropy_weight", 0.30)))
        ste_weight = max(0.0, float(getattr(self.cfg, "proposer_ste_primary_weight", 0.70)))
        score = float(reward) + sample_weight * sample_entropy_difficulty + ste_weight * ste_difficulty

        min_gate = max(0.0, float(getattr(self.cfg, "proposer_spot_entropy_min_gate", 0.05)))
        gate_passed = bool(float(dual.entropy_nats) >= min_gate)
        acceptable = bool(gate_passed)
        if self.cfg.proposer_require_objective and non_objective:
            acceptable = False
        if self.cfg.acceptance_require_non_easy and bool(dual.easy_case):
            acceptable = False

        return {
            "valid": True,
            "candidate_index": int(candidate_index),
            "question": question,
            "completion": f"<question>{question}</question>",
            "solver_outputs_raw": outputs_raw,
            "spot_answers_norm": answers_norm,
            "spot_solver_samples": solver_samples,
            "entropy_nats": float(dual.entropy_nats),
            "majority_fraction": float(dual.majority_fraction),
            "easy_case": bool(dual.easy_case),
            "unsolvable_case": bool(dual.unsolvable_case),
            "non_objective": bool(non_objective),
            "reward": float(reward),
            "sample_entropy_difficulty": float(sample_entropy_difficulty),
            "ste_difficulty": float(ste_difficulty),
            "score": float(score),
            "gate_passed": bool(gate_passed),
            "acceptable": bool(acceptable),
        }

    def _build_generation_completion_for_update(self, rec: Dict[str, Any]) -> str:
        raw = str(rec.get("proposer_spec_raw", "")).strip()
        if raw:
            return raw

        prompt = str(rec.get("spec_prompt", "")).strip()
        qa_pairs = rec.get("qa_pairs", [])
        if not prompt or not isinstance(qa_pairs, list):
            return ""
        lines = [f"<prompt>{prompt}</prompt>", "<qa_pairs>"]
        for qa in qa_pairs:
            if not isinstance(qa, dict):
                continue
            q = str(qa.get("question", "")).strip()
            a = str(qa.get("answer", "")).strip()
            if not q or not a:
                continue
            lines.append(f"  <qa><question>{q}</question><answer>{a}</answer></qa>")
        lines.append("</qa_pairs>")
        return "\n".join(lines)

    def _collect_generation_candidate(
        self,
        *,
        step: int,
        image_path: str,
        image: Image.Image,
        solver_temps: List[float],
        spec_temperature: float,
    ) -> Dict[str, Any]:
        prev_policy_enabled = bool(self.policy_updates_enabled)
        prev_temp = float(self.cfg.gen_spec_temperature)
        prev_baseline = float(self.proposer_gen_baseline)
        try:
            self.policy_updates_enabled = False
            self.cfg.gen_spec_temperature = float(spec_temperature)
            rec = self._run_suder_generation_rollout(
                step=step,
                image_path=image_path,
                image=image,
                solver_temps=solver_temps,
            )
        finally:
            self.policy_updates_enabled = prev_policy_enabled
            self.cfg.gen_spec_temperature = prev_temp
            self.proposer_gen_baseline = prev_baseline

        rec["policy_update_attempted"] = False
        rec["policy_update_applied"] = False
        rec["policy_update_reason"] = "deferred_unified_generation_group"
        rec["policy_update_stats"] = {}
        rec["spec_temperature_used"] = float(spec_temperature)
        rec["_completion_for_update"] = self._build_generation_completion_for_update(rec)
        return rec

    def _phase_for_step(self, step: int) -> str:
        u = max(0, int(self.ucfg.understanding_steps_per_cycle))
        g = max(0, int(self.ucfg.generation_steps_per_cycle))
        if u <= 0 and g <= 0:
            return "understanding"
        cycle = max(1, u + g)
        idx = (int(step) - 1) % cycle
        if u > 0 and idx < u:
            return "understanding"
        if g > 0:
            return "generation"
        return "understanding"

    def _current_gen_mix_ratio(self, step: int) -> float:
        return float(self.ucfg.current_gen_mix_ratio(step=int(step), start_step=max(1, int(self.start_step))))

    def _refresh_generated_mix_cache(self, step: int, force: bool = False) -> None:
        refresh_every = max(1, int(self.ucfg.generated_mix_refresh_every))
        if (not force) and (int(step) - int(self._generated_mix_last_refresh_step) < refresh_every):
            return
        entries: List[Dict[str, Any]] = []
        if self._generated_mix_dir.exists():
            for meta_path in sorted(self._generated_mix_dir.glob("*.json")):
                parsed = _parse_generated_mix_meta(
                    meta_path,
                    min_reward=float(self.ucfg.generated_mix_min_reward),
                )
                if parsed is not None:
                    entries.append(parsed)
        max_files = max(1, int(self.ucfg.generated_mix_max_files))
        if len(entries) > max_files:
            entries = sorted(
                entries,
                key=lambda e: (int(e.get("step_generated", 0)), str(e.get("meta_path", ""))),
            )[-max_files:]
        self._generated_mix_cache = entries
        self._generated_mix_last_refresh_step = int(step)

    def _sample_generated_mix_from_folder(self, step: int) -> Optional[Dict[str, Any]]:
        self._refresh_generated_mix_cache(step=step)
        if not self._generated_mix_cache:
            return None
        rng = random.Random(int(self.ucfg.seed) + int(step) * 104729 + 17)
        chosen = self._generated_mix_cache[rng.randint(0, len(self._generated_mix_cache) - 1)]
        try:
            with Image.open(chosen["image_path"]) as img:
                image = img.convert("RGB")
        except Exception:
            return None
        return {
            "image": image,
            "meta": {
                "path": chosen["image_path"],
                "source": "generated_folder",
                "prompt": chosen.get("prompt", ""),
                "questions": chosen.get("questions", []),
                "reference_answers": chosen.get("reference_answers", []),
                "reward": float(chosen.get("reward", 0.0)),
                "step_generated": int(chosen.get("step_generated", 0)),
            },
        }

    def _prune_generated_mix_dir(self) -> None:
        max_files = max(1, int(self.ucfg.generated_mix_max_files))
        meta_files = sorted(self._generated_mix_dir.glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name))
        if len(meta_files) <= max_files:
            return
        remove_count = len(meta_files) - max_files
        for meta_path in meta_files[:remove_count]:
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}
            image_path = Path(str(payload.get("image_path", ""))).expanduser()
            if not image_path.is_absolute():
                image_path = (meta_path.parent / image_path).resolve()
            for p in [image_path, meta_path.with_suffix(".png"), meta_path]:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

    def _store_generated_to_folder(self, step: int, rec: Dict[str, Any]) -> None:
        if not self._generated_mix_dir:
            return
        image_path_raw = str(rec.get("generated_image_path", "")).strip()
        if not image_path_raw:
            return
        image_path = Path(image_path_raw)
        if not image_path.exists():
            return
        qa_pairs = rec.get("qa_pairs", [])
        if not isinstance(qa_pairs, list) or not qa_pairs:
            return
        questions: List[str] = []
        answers: List[str] = []
        for qa in qa_pairs:
            if not isinstance(qa, dict):
                continue
            q = str(qa.get("question", "")).strip()
            a = str(qa.get("answer", "")).strip()
            if q and a:
                questions.append(q)
                answers.append(a)
        n = min(len(questions), len(answers))
        if n <= 0:
            return
        questions = questions[:n]
        answers = answers[:n]

        reward = float(rec.get("proposer_gen_reward", 0.0))
        if reward < float(self.ucfg.generated_mix_min_reward):
            return

        self._generated_mix_dir.mkdir(parents=True, exist_ok=True)
        stem = f"s{int(step):07d}_{int(time.time() * 1000)}_{random.randint(0, 999999):06d}"
        dst_image = self._generated_mix_dir / f"{stem}.png"
        dst_meta = self._generated_mix_dir / f"{stem}.json"
        try:
            with Image.open(image_path) as img:
                img.convert("RGB").save(dst_image, format="PNG")
        except Exception:
            return
        payload = {
            "step_generated": int(step),
            "prompt": str(rec.get("spec_prompt", "")),
            "questions": questions,
            "reference_answers": answers,
            "reward": float(reward),
            "raw_reward": float(reward),
            "image_path": str(dst_image),
            "source": "generation_rollout",
        }
        with dst_meta.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self._generated_mix_last_refresh_step = -10**9
        self._prune_generated_mix_dir()

    def _maybe_add_to_replay_buffer(self, step: int, rec: Dict[str, Any]) -> None:
        if self.replay_buffer is None:
            return
        image_path_raw = str(rec.get("generated_image_path", "")).strip()
        if not image_path_raw:
            return
        image_path = Path(image_path_raw)
        if not image_path.exists():
            return
        reward = float(rec.get("proposer_gen_reward", 0.0))
        qa_pairs = rec.get("qa_pairs", [])
        if not isinstance(qa_pairs, list):
            return
        questions: List[str] = []
        answers: List[str] = []
        for qa in qa_pairs:
            if not isinstance(qa, dict):
                continue
            q = str(qa.get("question", "")).strip()
            a = str(qa.get("answer", "")).strip()
            if q and a:
                questions.append(q)
                answers.append(a)
        n = min(len(questions), len(answers))
        if n <= 0:
            return
        questions = questions[:n]
        answers = answers[:n]
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception:
            return
        self.replay_buffer.add(
            image=image,
            prompt=str(rec.get("spec_prompt", "")),
            questions=questions,
            reference_answers=answers,
            reward=reward,
            step=int(step),
            meta={"image_path": str(image_path), "source": "generation_rollout"},
        )

    def _pick_understanding_image(self, step: int) -> Optional[Dict[str, Any]]:
        ratio = self._current_gen_mix_ratio(step)
        want_generated = bool(self.ucfg.understanding_generated_only)
        if not want_generated and ratio > 0.0:
            rng = random.Random(int(self.ucfg.seed) + int(step) * 7919)
            want_generated = bool(rng.random() < ratio)

        if want_generated:
            if self._gen_mix_source_mode == "folder":
                picked = self._sample_generated_mix_from_folder(step=step)
                if picked is not None:
                    return picked
            elif self.replay_buffer is not None and len(self.replay_buffer) > 0:
                entry = self.replay_buffer.sample()
                if entry is not None:
                    return {
                        "image": entry.image.copy(),
                        "meta": {
                            "path": str(entry.meta.get("image_path", "")) or f"generated://buffer/{step}",
                            "source": "replay_buffer",
                            "prompt": entry.prompt,
                            "questions": entry.questions,
                            "reference_answers": entry.reference_answers,
                            "reward": entry.reward,
                            "step_generated": entry.step_generated,
                        },
                    }

        if bool(self.ucfg.understanding_generated_only):
            return None
        image_path = self._sample_image_path(step)
        image = self._load_image(image_path)
        return {
            "image": image,
            "meta": {"path": image_path, "source": "real"},
        }

    def _run_understanding_step(
        self,
        *,
        step: int,
        image_path: str,
        image: Image.Image,
        solver_temps: List[float],
        baseline_momentum: float,
    ) -> Dict[str, Any]:
        proposer_candidate_count = max(1, int(getattr(self.cfg, "proposer_num_candidates", 1)))
        proposer = self.adapter.propose_questions(
            image=image,
            max_new_tokens=self.cfg.max_new_tokens_proposer,
            temperature=self.cfg.proposer_temperature,
            num_questions=proposer_candidate_count,
        )

        candidate_questions = parse_all_questions(proposer.text)
        if not candidate_questions:
            fallback_q = parse_first_question(proposer.text)
            candidate_questions = [fallback_q] if fallback_q else []
        deduped_questions: List[str] = []
        seen_q = set()
        for q in candidate_questions:
            qq = str(q or "").strip()
            if not qq:
                continue
            key = qq.lower()
            if key in seen_q:
                continue
            seen_q.add(key)
            deduped_questions.append(qq)
        candidate_questions = deduped_questions[:proposer_candidate_count]

        if not candidate_questions:
            record = {
                "step": int(step),
                "phase": "understanding",
                "status": "skipped",
                "skip_reason": "empty_question",
                "image_path": image_path,
                "proposer_raw": proposer.text if self.cfg.save_raw_generations else "",
            }
            return {
                "record": record,
                "valid": 0,
                "skipped": 1,
                "reward_sum": 0.0,
                "reward_nonzero": 0,
                "dual_disagree": 0,
                "policy_attempted": 0,
                "policy_applied": 0,
            }

        spot_temps = self._spot_solver_temperatures(solver_temps)
        candidate_stats = [
            self._evaluate_proposer_candidate(
                image=image,
                question=question,
                spot_temps=spot_temps,
                candidate_index=idx,
            )
            for idx, question in enumerate(candidate_questions)
        ]
        valid_candidates = [c for c in candidate_stats if bool(c.get("valid", False))]
        if not valid_candidates:
            record = {
                "step": int(step),
                "phase": "understanding",
                "status": "skipped",
                "skip_reason": "empty_solver_answers",
                "image_path": image_path,
                "candidate_questions": candidate_questions,
                "proposer_raw": proposer.text if self.cfg.save_raw_generations else "",
            }
            return {
                "record": record,
                "valid": 0,
                "skipped": 1,
                "reward_sum": 0.0,
                "reward_nonzero": 0,
                "dual_disagree": 0,
                "policy_attempted": 0,
                "policy_applied": 0,
            }

        acceptable_candidates = [c for c in valid_candidates if bool(c.get("acceptable", False))]
        selected_candidate = max(
            acceptable_candidates or valid_candidates,
            key=lambda c: (float(c.get("score", -1.0)), float(c.get("reward", -1.0)), float(c.get("entropy_nats", 0.0))),
        )
        question = str(selected_candidate["question"])

        solver_outputs_raw: List[str] = list(selected_candidate.get("solver_outputs_raw", []))
        solver_answers_norm: List[str] = list(selected_candidate.get("spot_answers_norm", []))
        solver_samples: List[Any] = list(selected_candidate.get("spot_solver_samples", []))
        for temp in solver_temps[len(spot_temps):]:
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
            record = {
                "step": int(step),
                "phase": "understanding",
                "status": "skipped",
                "skip_reason": "empty_solver_answers",
                "image_path": image_path,
                "question": question,
                "solver_outputs_raw": solver_outputs_raw if self.cfg.save_raw_generations else [],
            }
            return {
                "record": record,
                "valid": 0,
                "skipped": 1,
                "reward_sum": 0.0,
                "reward_nonzero": 0,
                "dual_disagree": 0,
                "policy_attempted": 0,
                "policy_applied": 0,
            }

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

        reward = float(dual.reward)
        non_objective = not is_objective_question(question)
        if self.cfg.proposer_require_objective and non_objective:
            reward -= float(self.cfg.proposer_non_objective_penalty)
        if self.cfg.acceptance_require_non_easy and dual.easy_case:
            reward -= float(self.cfg.rejected_question_penalty)
        reward = max(-1.0, min(1.0, reward))

        proposer_update_stats: Dict[str, Any] = {"skipped": True, "reason": "disabled"}
        proposer_update_attempted = False
        proposer_update_applied = False
        policy_attempted = 0
        policy_applied = 0
        if (
            self.policy_updates_enabled
            and self.cfg.train_understanding_proposer
            and self.proposer_updater is not None
        ):
            proposer_update_attempted = True
            update_method = self.cfg.normalized_update_method()
            if update_method == "grpo" and len(valid_candidates) > 1:
                group_rewards: List[float] = []
                for cand in valid_candidates:
                    cand_reward = float(cand.get("reward", 0.0))
                    if int(cand.get("candidate_index", -1)) == int(selected_candidate.get("candidate_index", -2)):
                        cand_reward = float(reward)
                    group_rewards.append(cand_reward)

                per_candidate_stats: List[Dict[str, Any]] = []
                applied_count = 0
                max_proposer_updates = max(
                    0,
                    int(os.environ.get("BAGEL_PROPOSER_POLICY_MAX_CANDIDATES", "0") or "0"),
                )
                for cand, cand_reward in zip(valid_candidates, group_rewards):
                    if max_proposer_updates > 0 and len(per_candidate_stats) >= max_proposer_updates:
                        break
                    stats = self.proposer_updater.step(
                        image=image,
                        prompt=build_proposer_prompt(),
                        completion=str(cand.get("completion", "")),
                        reward=float(cand_reward),
                        baseline=self.proposer_baseline,
                        group_rewards=group_rewards,
                    )
                    per_candidate_stats.append(stats)
                    policy_attempted += 1
                    if not bool(stats.get("skipped", True)):
                        applied_count += 1
                        policy_applied += 1

                proposer_update_applied = bool(applied_count > 0)
                proposer_update_stats = {
                    "skipped": not proposer_update_applied,
                    "reason": "ok" if proposer_update_applied else "all_skipped",
                    "update_method": "grpo",
                    "group_size": int(len(group_rewards)),
                    "group_reward_mean": float(_mean(group_rewards)),
                    "group_reward_max": float(max(group_rewards)),
                    "group_reward_min": float(min(group_rewards)),
                    "applied_updates": int(applied_count),
                    "ce_loss_mean": float(
                        _mean(
                            [
                                float(s.get("ce_loss", 0.0))
                                for s in per_candidate_stats
                                if not bool(s.get("skipped", True))
                            ]
                        )
                    ),
                    "selected_candidate_index": int(selected_candidate.get("candidate_index", -1)),
                }
            else:
                proposer_update_stats = self.proposer_updater.step(
                    image=image,
                    prompt=build_proposer_prompt(),
                    completion=str(selected_candidate.get("completion", f"<question>{question}</question>")),
                    reward=float(reward),
                    baseline=self.proposer_baseline,
                )
                proposer_update_applied = bool(not proposer_update_stats.get("skipped", True))
                policy_attempted += 1
                policy_applied += int(proposer_update_applied)

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
        solver_update_stats: List[Dict[str, Any]] = []
        solver_update_reason = "disabled"
        if solver_skip_update:
            solver_update_reason = "easy_question_skip"
        elif self.policy_updates_enabled and self.cfg.train_solver and self.solver_updater is not None:
            solver_prompt = build_solver_prompt(question)
            max_solver_updates = max(
                0,
                int(os.environ.get("BAGEL_SOLVER_POLICY_MAX_SAMPLES", "0") or "0"),
            )
            for idx, (sample_raw, _) in enumerate(solver_samples):
                if max_solver_updates > 0 and idx >= max_solver_updates:
                    break
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
                policy_attempted += 1
                policy_applied += int(not update_stats.get("skipped", True))
            solver_update_reason = "ok" if solver_update_stats else "no_samples"

        self.solver_baseline = (
            baseline_momentum * self.solver_baseline
            + (1.0 - baseline_momentum) * float(solver_scalar_reward)
        )

        candidate_rewards = [float(c.get("reward", 0.0)) for c in valid_candidates]
        candidate_scores = [float(c.get("score", 0.0)) for c in valid_candidates]
        candidate_entropy = [float(c.get("entropy_nats", 0.0)) for c in valid_candidates]
        candidate_ste = [float(c.get("ste_difficulty", 0.0)) for c in valid_candidates]
        candidate_easy = [bool(c.get("easy_case", True)) for c in valid_candidates]

        record = {
            "step": int(step),
            "phase": "understanding",
            "status": "ok",
            "image_path": image_path,
            "question": question,
            "proposer_raw": proposer.text if self.cfg.save_raw_generations else "",
            "proposer_candidate_questions": candidate_questions,
            "proposer_candidate_count_requested": int(proposer_candidate_count),
            "proposer_candidate_count_parsed": int(len(candidate_questions)),
            "proposer_candidate_count_valid": int(len(valid_candidates)),
            "proposer_candidate_count_acceptable": int(len(acceptable_candidates)),
            "proposer_candidate_rewards": candidate_rewards,
            "proposer_candidate_scores": candidate_scores,
            "proposer_candidate_entropy_nats": candidate_entropy,
            "proposer_candidate_ste_difficulty": candidate_ste,
            "proposer_candidate_easy_flags": candidate_easy,
            "proposer_all_easy_candidate_group": bool(candidate_easy and all(candidate_easy)),
            "proposer_selected_candidate_index": int(selected_candidate.get("candidate_index", -1)),
            "proposer_selected_candidate_score": float(selected_candidate.get("score", 0.0)),
            "proposer_selected_candidate_reward_spot": float(selected_candidate.get("reward", 0.0)),
            "proposer_selected_candidate_entropy_nats": float(selected_candidate.get("entropy_nats", 0.0)),
            "proposer_selected_candidate_ste_difficulty": float(selected_candidate.get("ste_difficulty", 0.0)),
            "proposer_spot_check_samples": int(len(spot_temps)),
            "solver_answers_norm": solver_answers_norm,
            "solver_outputs_raw": solver_outputs_raw if self.cfg.save_raw_generations else [],
            "intuitive_answer": intuitive_norm,
            "intuitive_raw": intuitive.text if self.cfg.save_raw_generations else "",
            "entropy_nats": float(dual.entropy_nats),
            "majority_fraction": float(dual.majority_fraction),
            "majority_answer": str(dual.majority_answer),
            "dual_track_agree": bool(dual.dual_track_agree),
            "easy_case": bool(dual.easy_case),
            "unsolvable_case": bool(dual.unsolvable_case),
            "proposer_reward_raw": float(dual.reward_raw),
            "proposer_reward_final": float(reward),
            "proposer_non_objective_question": bool(non_objective),
            "solver_scalar_reward": float(solver_scalar_reward),
            "solver_group_rewards": solver_group_rewards,
            "proposer_baseline": float(self.proposer_baseline),
            "solver_baseline": float(self.solver_baseline),
            "policy_updates_enabled": bool(self.policy_updates_enabled),
            "proposer_policy_update_attempted": bool(proposer_update_attempted),
            "proposer_policy_update_applied": bool(proposer_update_applied),
            "proposer_policy_update_stats": proposer_update_stats,
            "solver_policy_update_skipped": bool(solver_skip_update),
            "solver_policy_update_reason": str(solver_update_reason),
            "solver_policy_update_attempts": int(len(solver_update_stats)),
            "solver_policy_update_applied": int(sum(int(not s.get("skipped", True)) for s in solver_update_stats)),
            "solver_policy_update_ce_mean": _mean(
                [float(s.get("ce_loss", 0.0)) for s in solver_update_stats if not s.get("skipped", True)]
            ),
            "solver_temperatures": solver_temps,
        }
        return {
            "record": record,
            "valid": 1,
            "skipped": 0,
            "reward_sum": float(reward),
            "reward_nonzero": int(abs(reward) > 1e-9),
            "dual_disagree": int(not dual.dual_track_agree),
            "policy_attempted": int(policy_attempted),
            "policy_applied": int(policy_applied),
        }

    def _run_generation_step(
        self,
        *,
        step: int,
        image_path: str,
        image: Image.Image,
        solver_temps: List[float],
    ) -> Dict[str, Any]:
        if not bool(self.cfg.suder_generation_enabled):
            return {
                "record": {
                    "step": int(step),
                    "phase": "generation",
                    "status": "skipped",
                    "skip_reason": "suder_generation_disabled",
                    "image_path": image_path,
                },
                "valid": 0,
                "skipped": 1,
                "reward_sum": 0.0,
                "entropy_sum": 0.0,
                "quality_sum": 0.0,
                "policy_attempted": 0,
                "policy_applied": 0,
            }
        policy_train_ready = bool(
            self.policy_updates_enabled
            and self.cfg.train_generation_proposer
            and self.proposer_updater is not None
        )
        update_method = self.cfg.normalized_update_method()
        group_size = 1
        if policy_train_ready and update_method == "grpo":
            group_size = max(1, int(getattr(self.cfg, "proposer_grpo_gen_group_size", 3)))
            if not bool(getattr(self.cfg, "score_grpo_extras", True)):
                group_size = 1

        base_temp = float(self.cfg.gen_spec_temperature)
        extra_temp_mult = max(0.1, float(getattr(self.cfg, "grpo_extra_temp_multiplier", 1.5)))
        candidates: List[Dict[str, Any]] = []
        for idx in range(group_size):
            temp = base_temp if idx == 0 else base_temp * extra_temp_mult
            src_path = Path(str(image_path))
            candidate_image_path = str(
                src_path.with_name(f"{src_path.stem}_cand{idx}{src_path.suffix}")
            )
            rec = self._collect_generation_candidate(
                step=step,
                image_path=candidate_image_path,
                image=image,
                solver_temps=solver_temps,
                spec_temperature=float(temp),
            )
            rec["source_image_path"] = str(image_path)
            rec["candidate_index"] = int(idx)
            rec["candidate_group_size"] = int(group_size)
            candidates.append(rec)

        valid_candidates = [c for c in candidates if str(c.get("status", "")) == "ok"]
        if not valid_candidates:
            rec = candidates[0] if candidates else {
                "step": int(step),
                "phase": "generation",
                "status": "skipped",
                "skip_reason": "generation_group_empty",
                "image_path": image_path,
            }
            rec["phase"] = "generation"
            rec["source_image_path"] = str(image_path)
            rec["image_path"] = str(image_path)
            rec["generation_candidate_group_size"] = int(group_size)
            rec["generation_candidate_valid_count"] = 0
            rec["generation_candidate_rewards"] = []
            rec["generation_candidate_statuses"] = [str(c.get("status", "skipped")) for c in candidates]
            rec["generation_candidate_temps"] = [float(c.get("spec_temperature_used", base_temp)) for c in candidates]
            rec.pop("_completion_for_update", None)
            return {
                "record": rec,
                "valid": 0,
                "skipped": 1,
                "reward_sum": 0.0,
                "entropy_sum": 0.0,
                "quality_sum": 0.0,
                "policy_attempted": 0,
                "policy_applied": 0,
            }

        selected = max(
            valid_candidates,
            key=lambda c: (
                float(c.get("proposer_gen_reward", -1.0)),
                float(c.get("quality_component", 0.0)),
                float(c.get("mean_entropy_nats", 0.0)),
            ),
        )
        selected_idx = int(selected.get("candidate_index", 0))
        selected_reward = float(selected.get("proposer_gen_reward", 0.0))
        baseline_before = float(self.proposer_gen_baseline)
        baseline_momentum = _clamp01(float(self.cfg.proposer_gen_baseline_momentum))

        policy_attempted = 0
        policy_applied = 0
        policy_update_attempted = False
        policy_update_applied = False
        policy_update_reason = "disabled"
        policy_update_stats: Dict[str, Any] = {}
        prompt = build_generation_spec_prompt(min_qa_pairs=int(self.cfg.gen_spec_min_qa_pairs))

        if policy_train_ready:
            policy_update_attempted = True
            if update_method == "grpo" and len(valid_candidates) > 1:
                group_rewards = [float(c.get("proposer_gen_reward", 0.0)) for c in valid_candidates]
                per_candidate_stats: List[Dict[str, Any]] = []
                for cand, cand_reward in zip(valid_candidates, group_rewards):
                    completion = str(cand.get("_completion_for_update", "")).strip()
                    if not completion:
                        continue
                    stats = self.proposer_updater.step(
                        image=image,
                        prompt=prompt,
                        completion=completion,
                        reward=float(cand_reward),
                        baseline=baseline_before,
                        group_rewards=group_rewards,
                    )
                    per_candidate_stats.append(stats)
                    policy_attempted += 1
                    if not bool(stats.get("skipped", True)):
                        policy_applied += 1
                policy_update_applied = bool(policy_applied > 0)
                policy_update_reason = "ok" if policy_update_applied else "all_skipped"
                policy_update_stats = {
                    "skipped": not policy_update_applied,
                    "reason": policy_update_reason,
                    "update_method": "grpo",
                    "group_size": int(len(group_rewards)),
                    "group_reward_mean": float(_mean(group_rewards)),
                    "group_reward_max": float(max(group_rewards)),
                    "group_reward_min": float(min(group_rewards)),
                    "applied_updates": int(policy_applied),
                    "selected_candidate_index": int(selected_idx),
                    "ce_loss_mean": float(
                        _mean(
                            [
                                float(s.get("ce_loss", 0.0))
                                for s in per_candidate_stats
                                if not bool(s.get("skipped", True))
                            ]
                        )
                    ),
                }
            else:
                completion = str(selected.get("_completion_for_update", "")).strip()
                if completion:
                    stats = self.proposer_updater.step(
                        image=image,
                        prompt=prompt,
                        completion=completion,
                        reward=float(selected_reward),
                        baseline=baseline_before,
                    )
                    policy_attempted = 1
                    policy_applied = int(not bool(stats.get("skipped", True)))
                    policy_update_applied = bool(policy_applied > 0)
                    policy_update_reason = str(stats.get("reason", "unknown"))
                    policy_update_stats = stats
                else:
                    policy_update_applied = False
                    policy_update_reason = "empty_completion"
                    policy_update_stats = {"skipped": True, "reason": "empty_completion"}

        self.proposer_gen_baseline = (
            baseline_momentum * self.proposer_gen_baseline
            + (1.0 - baseline_momentum) * float(selected_reward)
        )

        rec = dict(selected)
        rec["phase"] = "generation"
        rec["generation_candidate_image_path"] = str(rec.get("image_path", image_path))
        rec["source_image_path"] = str(image_path)
        rec["image_path"] = str(image_path)
        rec["generation_candidate_group_size"] = int(group_size)
        rec["generation_candidate_valid_count"] = int(len(valid_candidates))
        rec["generation_candidate_rewards"] = [
            float(c.get("proposer_gen_reward", 0.0)) for c in valid_candidates
        ]
        rec["generation_candidate_statuses"] = [str(c.get("status", "skipped")) for c in candidates]
        rec["generation_candidate_temps"] = [float(c.get("spec_temperature_used", base_temp)) for c in candidates]
        rec["generation_selected_candidate_index"] = int(selected_idx)
        rec["proposer_gen_baseline_before"] = float(baseline_before)
        rec["proposer_gen_baseline_after"] = float(self.proposer_gen_baseline)
        rec["proposer_gen_advantage"] = float(selected_reward - baseline_before)
        rec["policy_update_attempted"] = bool(policy_update_attempted)
        rec["policy_update_applied"] = bool(policy_update_applied)
        rec["policy_update_reason"] = str(policy_update_reason)
        rec["policy_update_stats"] = policy_update_stats

        rec.pop("_completion_for_update", None)
        for cand in candidates:
            cand.pop("_completion_for_update", None)

        self._store_generated_to_folder(step=step, rec=rec)
        self._maybe_add_to_replay_buffer(step=step, rec=rec)
        return {
            "record": rec,
            "valid": 1,
            "skipped": 0,
            "reward_sum": float(selected_reward),
            "entropy_sum": float(rec.get("mean_entropy_nats", 0.0)),
            "quality_sum": float(rec.get("quality_component", 0.0)),
            "policy_attempted": int(policy_attempted),
            "policy_applied": int(policy_applied),
        }

    def run(self) -> Dict[str, float]:
        random.seed(self.cfg.seed)

        steps_valid = 0
        steps_skipped = 0
        reward_sum = 0.0
        reward_nonzero = 0
        dual_track_disagree = 0

        gen_steps_valid = 0
        gen_steps_skipped = 0
        gen_reward_sum = 0.0
        gen_entropy_sum = 0.0
        gen_quality_sum = 0.0

        policy_updates_attempted = 0
        policy_updates_applied = 0
        phase_counts = {"understanding": 0, "generation": 0}

        solver_temps = self.cfg.solver_temperatures()
        baseline_momentum = _clamp01(self.cfg.baseline_momentum)
        run_started_at = float(time.time())

        def _status_metrics(step_time_sec: float) -> Dict[str, Any]:
            replay_size = int(len(self.replay_buffer)) if self.replay_buffer is not None else 0
            return {
                "step_time_sec": float(step_time_sec),
                "phase_counts_understanding": int(phase_counts["understanding"]),
                "phase_counts_generation": int(phase_counts["generation"]),
                "understanding_steps_valid": int(steps_valid),
                "understanding_steps_skipped": int(steps_skipped),
                "understanding_mean_reward": float(reward_sum / float(max(1, steps_valid))),
                "understanding_nonzero_reward_rate": float(reward_nonzero / float(max(1, steps_valid))),
                "dual_track_disagree_rate": float(dual_track_disagree / float(max(1, steps_valid))),
                "generation_steps_valid": int(gen_steps_valid),
                "generation_steps_skipped": int(gen_steps_skipped),
                "generation_mean_reward": float(gen_reward_sum / float(max(1, gen_steps_valid))),
                "generation_mean_entropy_nats": float(gen_entropy_sum / float(max(1, gen_steps_valid))),
                "generation_mean_quality": float(gen_quality_sum / float(max(1, gen_steps_valid))),
                "policy_updates_attempted": int(policy_updates_attempted),
                "policy_updates_applied": int(policy_updates_applied),
                "proposer_baseline": float(self.proposer_baseline),
                "solver_baseline": float(self.solver_baseline),
                "proposer_gen_baseline": float(self.proposer_gen_baseline),
                "generator_reward_ema": float(self._gen_reward_ema) if self._gen_reward_ema_initialized else 0.0,
                "replay_buffer_size": int(replay_size),
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

        for step in range(int(self.start_step), int(self.cfg.steps) + 1):
            step_t0 = float(time.time())
            phase = self._phase_for_step(step)
            phase_counts[phase] = int(phase_counts.get(phase, 0) + 1)

            if phase == "understanding":
                picked = self._pick_understanding_image(step)
                if picked is None:
                    steps_skipped += 1
                    _write_jsonl(
                        self.rollouts_log_path,
                        {
                            "step": int(step),
                            "phase": "understanding",
                            "status": "skipped",
                            "skip_reason": "generated_pool_empty",
                        },
                    )
                else:
                    meta = picked["meta"]
                    image_path = str(meta.get("path", "") or f"generated://missing_path/{step}")
                    stats = self._run_understanding_step(
                        step=step,
                        image_path=image_path,
                        image=picked["image"],
                        solver_temps=solver_temps,
                        baseline_momentum=baseline_momentum,
                    )
                    _write_jsonl(self.rollouts_log_path, stats["record"])
                    steps_valid += int(stats["valid"])
                    steps_skipped += int(stats["skipped"])
                    reward_sum += float(stats["reward_sum"])
                    reward_nonzero += int(stats["reward_nonzero"])
                    dual_track_disagree += int(stats["dual_disagree"])
                    policy_updates_attempted += int(stats["policy_attempted"])
                    policy_updates_applied += int(stats["policy_applied"])
            else:
                image_path = self._sample_image_path(step)
                image = self._load_image(image_path)
                stats = self._run_generation_step(
                    step=step,
                    image_path=image_path,
                    image=image,
                    solver_temps=solver_temps,
                )
                _write_jsonl(self.generation_rollouts_log_path, stats["record"])
                gen_steps_valid += int(stats["valid"])
                gen_steps_skipped += int(stats["skipped"])
                gen_reward_sum += float(stats["reward_sum"])
                gen_entropy_sum += float(stats["entropy_sum"])
                gen_quality_sum += float(stats["quality_sum"])
                policy_updates_attempted += int(stats["policy_attempted"])
                policy_updates_applied += int(stats["policy_applied"])

                if gen_steps_valid > 0:
                    mean_gen_reward = gen_reward_sum / float(max(1, gen_steps_valid))
                    mom = _clamp01(float(self.ucfg.reward_ema_momentum))
                    if not self._gen_reward_ema_initialized:
                        self._gen_reward_ema = float(mean_gen_reward)
                        self._gen_reward_ema_initialized = True
                    else:
                        self._gen_reward_ema = mom * self._gen_reward_ema + (1.0 - mom) * float(mean_gen_reward)

            _emit_training_logs(step, phase=phase, step_time_sec=float(time.time() - step_t0))

            if step % max(1, int(self.cfg.log_every)) == 0:
                mean_reward = reward_sum / float(max(1, steps_valid))
                mean_gen_reward = gen_reward_sum / float(max(1, gen_steps_valid))
                replay_size = int(len(self.replay_buffer)) if self.replay_buffer is not None else 0
                print(
                    f"[self_evolving][step={step}] phase={phase[:1].upper()} "
                    f"U(valid={steps_valid}, skipped={steps_skipped}, mean_reward={mean_reward:.4f}) "
                    f"G(valid={gen_steps_valid}, skipped={gen_steps_skipped}, mean_reward={mean_gen_reward:.4f}) "
                    f"policy_updates={policy_updates_applied}/{policy_updates_attempted} "
                    f"replay_size={replay_size}"
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

        flushed_optim_steps = 0
        if self.proposer_updater is not None:
            flushed_optim_steps += int(self.proposer_updater.finalize())
        if self.solver_updater is not None:
            flushed_optim_steps += int(self.solver_updater.finalize())

        if self.policy_updates_enabled:
            final_ckpt = self._save_checkpoint(int(self.cfg.steps))
            if final_ckpt:
                print(f"[self_evolving] final checkpoint: {final_ckpt}")

        replay_stats = self.replay_buffer.stats() if self.replay_buffer is not None else {
            "replay_buffer_size": 0.0,
            "replay_buffer_mean_reward": 0.0,
            "replay_buffer_min_step": 0.0,
            "replay_buffer_max_step": 0.0,
        }
        summary = {
            "experiment": str(self.cfg.normalized_experiment_name()),
            "steps_total": int(self.cfg.steps),
            "steps_started_from": int(self.start_step),
            "understanding_steps_valid": int(steps_valid),
            "understanding_steps_skipped": int(steps_skipped),
            "understanding_mean_reward": float(reward_sum / float(max(1, steps_valid))),
            "understanding_nonzero_reward_rate": float(reward_nonzero / float(max(1, steps_valid))),
            "dual_track_disagree_rate": float(dual_track_disagree / float(max(1, steps_valid))),
            "generation_steps_valid": int(gen_steps_valid),
            "generation_steps_skipped": int(gen_steps_skipped),
            "generation_mean_reward": float(gen_reward_sum / float(max(1, gen_steps_valid))),
            "generation_mean_entropy_nats": float(gen_entropy_sum / float(max(1, gen_steps_valid))),
            "generation_mean_quality": float(gen_quality_sum / float(max(1, gen_steps_valid))),
            "phase_counts_understanding": int(phase_counts["understanding"]),
            "phase_counts_generation": int(phase_counts["generation"]),
            "output_dir": self.output_dir,
            "rollouts_log_path": self.rollouts_log_path,
            "generation_rollouts_log_path": self.generation_rollouts_log_path,
            "policy_updates_enabled": bool(self.policy_updates_enabled),
            "policy_updates_attempted": int(policy_updates_attempted),
            "policy_updates_applied": int(policy_updates_applied),
            "proposer_baseline_final": float(self.proposer_baseline),
            "solver_baseline_final": float(self.solver_baseline),
            "proposer_gen_baseline_final": float(self.proposer_gen_baseline),
            "generator_reward_ema": float(self._gen_reward_ema) if self._gen_reward_ema_initialized else 0.0,
            "gen_mix_source_mode": str(self._gen_mix_source_mode),
            "generated_mix_dir": str(self._generated_mix_dir),
            "optimizer_flush_steps": int(flushed_optim_steps),
            "last_checkpoint_path": str(self.last_checkpoint_path),
            "replay_buffer_size": float(replay_stats["replay_buffer_size"]),
            "replay_buffer_mean_reward": float(replay_stats["replay_buffer_mean_reward"]),
            "replay_buffer_min_step": float(replay_stats["replay_buffer_min_step"]),
            "replay_buffer_max_step": float(replay_stats["replay_buffer_max_step"]),
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
