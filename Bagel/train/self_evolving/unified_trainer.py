# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter, deque
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from PIL import Image

from .prompts import (
    build_generation_spec_prompt,
    build_proposer_prompt,
    build_solver_prompt_pps,
    build_solver_prompt,
    is_objective_question,
    is_well_formed_question,
    parse_all_questions,
    parse_answer,
    parse_first_question,
    parse_proposer_question_candidates,
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

        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"unified_rollout_{ts}"
        if rank_suffix:
            run_name = f"{run_name}_{rank_suffix}"
        run_dir = os.path.join(output_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def __init__(self, runtime, cfg) -> None:
        self.ucfg = cfg
        self._gen_mix_source_mode = cfg.normalized_gen_mix_source_mode()
        self._generated_mix_cache: List[Dict[str, Any]] = []
        self._generated_mix_last_refresh_step = -10**9
        self._gen_reward_ema = 0.0
        self._gen_reward_ema_initialized = False
        self._ste_window: List[float] = []
        self._ste_window_size = max(8, int(getattr(cfg, "solver_token_entropy_window_size", 128)))
        self._difficulty_window: Deque[str] = deque(
            maxlen=max(8, int(getattr(cfg, "difficulty_sampler_window_size", 256)))
        )
        self._entropy_easy_window: Deque[float] = deque(
            maxlen=max(8, int(getattr(cfg, "entropy_iqr_window_size", 256)))
        )
        warm_exit_window = max(1, int(getattr(cfg, "proposer_warm_start_exit_window", 5)))
        self._warm_start_entropy_window: Deque[float] = deque(maxlen=warm_exit_window)
        self._warm_start_exit_streak = 0
        self._warm_start_completed = False
        self._hardness_debt = 0.0
        self._hardness_debt_cap_streak = 0
        self._hardness_debt_escape_steps_left = 0
        self._forced_explore_steps_left = 0
        self._all_easy_streak = 0
        self._understanding_u_step = 0
        self._proposer_collapse_streak = 0
        self._strategy_hist: Counter[str] = Counter()
        replay_n = max(8, int(getattr(cfg, "proposer_contrastive_replay_size", 256)))
        self._contrastive_pos_replay: Deque[set[str]] = deque(maxlen=replay_n)
        self._contrastive_neg_replay: Deque[set[str]] = deque(maxlen=replay_n)
        failfast_window = max(
            32,
            int(getattr(cfg, "difficulty_sampler_window_size", 256)),
            int(getattr(cfg, "proposer_early_stage2_u_step", 20)) * 4,
        )
        self._candidate_non_easy_window: Deque[float] = deque(maxlen=failfast_window)
        self._all_easy_group_window: Deque[float] = deque(maxlen=failfast_window)
        self._proposer_reward_clipped_window: Deque[float] = deque(maxlen=failfast_window)
        self._selected_non_easy_window: Deque[float] = deque(maxlen=failfast_window)
        self._solver_update_applied_window: Deque[float] = deque(maxlen=failfast_window)
        self.replay_buffer: Optional[ReplayBuffer] = None
        if self._gen_mix_source_mode == "buffer":
            self.replay_buffer = ReplayBuffer(
                max_size=max(1, int(cfg.replay_buffer_size)),
                min_reward=float(cfg.replay_min_reward),
                max_staleness=max(0, int(cfg.replay_max_staleness)),
            )

        super().__init__(runtime=runtime, cfg=cfg)
        self._generated_mix_dir = (
            Path(str(cfg.generated_mix_dir).strip()).expanduser().resolve()
            if str(cfg.generated_mix_dir or "").strip()
            else Path(self.output_dir).resolve() / "generated_mix_pool"
        )
        if self._gen_mix_source_mode == "buffer" and self.replay_buffer is None:
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
        if (
            self.policy_updates_enabled
            and self.cfg.normalized_update_method() == "grpo"
            and bool(getattr(self.cfg, "score_grpo_extras", True))
        ):
            spot_n = max(spot_n, int(getattr(self.cfg, "grpo_extra_sc_samples", spot_n)))
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

    def _question_token_set(self, question: str) -> set[str]:
        toks = [t.strip().lower() for t in str(question or "").replace("?", " ").split()]
        stop = {
            "",
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "to",
            "of",
            "in",
            "on",
            "at",
            "for",
            "or",
            "and",
            "what",
            "which",
            "how",
            "many",
            "much",
        }
        return {t for t in toks if t not in stop}

    def _jaccard_similarity(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = float(len(a.intersection(b)))
        union = float(len(a.union(b)))
        if union <= 0.0:
            return 0.0
        return inter / union

    def _contrastive_replay_adjustment(self, question: str) -> float:
        if not bool(getattr(self.cfg, "proposer_contrastive_replay_enabled", True)):
            return 0.0
        qset = self._question_token_set(question)
        if not qset:
            return 0.0
        pos_bonus = max(0.0, float(getattr(self.cfg, "proposer_contrastive_pos_bonus", 0.08)))
        neg_pen = max(0.0, float(getattr(self.cfg, "proposer_contrastive_neg_penalty", 0.08)))
        max_pos = 0.0
        max_neg = 0.0
        for item in self._contrastive_pos_replay:
            max_pos = max(max_pos, self._jaccard_similarity(qset, item))
        for item in self._contrastive_neg_replay:
            max_neg = max(max_neg, self._jaccard_similarity(qset, item))
        return (pos_bonus * max_pos) - (neg_pen * max_neg)

    def _normalize_bucket_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        w_easy = max(0.0, float(weights.get("easy", 0.0)))
        w_medium = max(0.0, float(weights.get("medium", 0.0)))
        w_hard = max(0.0, float(weights.get("hard", 0.0)))
        total = w_easy + w_medium + w_hard
        if total <= 1e-8:
            return {"easy": 0.2, "medium": 0.6, "hard": 0.2}
        return {
            "easy": w_easy / total,
            "medium": w_medium / total,
            "hard": w_hard / total,
        }

    def _difficulty_target_weights(self) -> Dict[str, float]:
        return self._normalize_bucket_weights(
            {
                "easy": float(getattr(self.cfg, "difficulty_target_easy", 0.10)),
                "medium": float(getattr(self.cfg, "difficulty_target_medium", 0.50)),
                "hard": float(getattr(self.cfg, "difficulty_target_hard", 0.40)),
            }
        )

    def _sample_bucket(self, weights: Dict[str, float]) -> str:
        r = random.random()
        c = 0.0
        for key in ("easy", "medium", "hard"):
            c += float(weights.get(key, 0.0))
            if r <= c:
                return key
        return "medium"

    def _is_proposer_warm_start_active(self, u_step: int) -> bool:
        if not bool(getattr(self.cfg, "proposer_warm_start_enabled", True)):
            return False
        if bool(self._warm_start_completed):
            return False
        max_steps = max(1, int(getattr(self.cfg, "proposer_warm_start_max_steps", 30)))
        return int(u_step) <= max_steps

    def _update_proposer_warm_start_state(self, entropy_nats: float, u_step: int) -> Dict[str, float]:
        if not bool(getattr(self.cfg, "proposer_warm_start_enabled", True)):
            return {
                "enabled": 0.0,
                "active_next": 0.0,
                "completed": 1.0,
                "entropy_mean": 0.0,
                "exit_streak": 0.0,
                "exit_pass": 0.0,
            }
        exit_window = max(1, int(getattr(self.cfg, "proposer_warm_start_exit_window", 5)))
        if int(getattr(self._warm_start_entropy_window, "maxlen", 0) or 0) != exit_window:
            self._warm_start_entropy_window = deque(
                list(self._warm_start_entropy_window)[-exit_window:],
                maxlen=exit_window,
            )
        self._warm_start_entropy_window.append(float(entropy_nats))
        entropy_mean = float(sum(float(x) for x in self._warm_start_entropy_window)) / float(
            max(1, len(self._warm_start_entropy_window))
        )
        exit_thr = max(
            0.0,
            float(getattr(self.cfg, "proposer_warm_start_entropy_exit_threshold", 0.10)),
        )
        exit_pass = bool(
            len(self._warm_start_entropy_window) >= exit_window and entropy_mean >= exit_thr
        )
        if exit_pass:
            self._warm_start_exit_streak += 1
        else:
            self._warm_start_exit_streak = 0
        max_steps = max(1, int(getattr(self.cfg, "proposer_warm_start_max_steps", 30)))
        exit_consecutive = max(
            1, int(getattr(self.cfg, "proposer_warm_start_exit_consecutive", 2))
        )
        if int(u_step) >= max_steps or int(self._warm_start_exit_streak) >= exit_consecutive:
            self._warm_start_completed = True
        return {
            "enabled": 1.0,
            "active_next": 1.0 if self._is_proposer_warm_start_active(int(u_step) + 1) else 0.0,
            "completed": 1.0 if bool(self._warm_start_completed) else 0.0,
            "entropy_mean": float(entropy_mean),
            "exit_streak": float(self._warm_start_exit_streak),
            "exit_pass": 1.0 if exit_pass else 0.0,
        }

    def _update_hardness_debt(self, difficulty_bucket_observed: str) -> Dict[str, float]:
        if not bool(getattr(self.cfg, "hardness_debt_enabled", True)):
            return {
                "enabled": 0.0,
                "debt": 0.0,
                "cap_streak": 0.0,
                "escape_steps_left": 0.0,
                "escape_triggered": 0.0,
            }
        debt = float(self._hardness_debt)
        debt_max = max(1e-6, float(getattr(self.cfg, "hardness_debt_max", 6.0)))
        inc_easy = max(0.0, float(getattr(self.cfg, "hardness_debt_inc_easy", 1.5)))
        dec_non_easy = max(0.0, float(getattr(self.cfg, "hardness_debt_dec_non_easy", 1.0)))
        if str(difficulty_bucket_observed).lower() == "easy":
            debt += inc_easy
        else:
            debt -= dec_non_easy
        debt = max(0.0, min(debt_max, debt))
        cap_streak = int(self._hardness_debt_cap_streak)
        if str(difficulty_bucket_observed).lower() == "easy" and debt >= (debt_max - 1e-8):
            cap_streak += 1
        else:
            cap_streak = 0
        escape_triggered = False
        stale_steps = max(1, int(getattr(self.cfg, "hardness_debt_stale_steps", 8)))
        if cap_streak >= stale_steps:
            reset_to = float(getattr(self.cfg, "hardness_debt_stale_reset_to", 3.0))
            debt = max(0.0, min(debt_max, reset_to))
            escape_steps = max(
                1, int(getattr(self.cfg, "hardness_debt_stale_escape_steps", stale_steps))
            )
            self._hardness_debt_escape_steps_left = max(
                int(self._hardness_debt_escape_steps_left),
                escape_steps,
            )
            cap_streak = 0
            escape_triggered = True
        self._hardness_debt = float(debt)
        self._hardness_debt_cap_streak = int(cap_streak)
        return {
            "enabled": 1.0,
            "debt": float(self._hardness_debt),
            "cap_streak": float(self._hardness_debt_cap_streak),
            "escape_steps_left": float(max(0, int(self._hardness_debt_escape_steps_left))),
            "escape_triggered": 1.0 if escape_triggered else 0.0,
        }

    def _choose_difficulty_target(self) -> Dict[str, Any]:
        enabled = bool(getattr(self.cfg, "difficulty_sampler_enabled", True))
        min_samples = max(4, int(getattr(self.cfg, "difficulty_sampler_min_samples", 32)))
        target = self._difficulty_target_weights()
        history = list(self._difficulty_window)
        observed = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
        mode = "target"
        weights_for_sampling = dict(target)

        if enabled and len(history) >= min_samples:
            for b in history:
                if b in observed:
                    observed[b] += 1.0
            for k in observed:
                observed[k] /= float(max(1, len(history)))
            deficits = {
                k: max(0.0, target[k] - observed[k]) for k in ("easy", "medium", "hard")
            }
            deficit_total = deficits["easy"] + deficits["medium"] + deficits["hard"]
            if deficit_total > 1e-8:
                weights_for_sampling = {
                    k: deficits[k] / deficit_total for k in ("easy", "medium", "hard")
                }
                mode = "deficit"
        elif not enabled:
            mode = "disabled"

        debt = float(self._hardness_debt)
        debt_ratio = 0.0
        debt_escape_active = False
        if bool(getattr(self.cfg, "hardness_debt_enabled", True)):
            weights_for_sampling = self._normalize_bucket_weights(weights_for_sampling)
            if int(self._hardness_debt_escape_steps_left) > 0:
                debt_escape_active = True
                weights_for_sampling = self._normalize_bucket_weights(
                    {
                        "easy": float(getattr(self.cfg, "hardness_debt_stale_easy_weight", 0.05)),
                        "medium": float(getattr(self.cfg, "hardness_debt_stale_medium_weight", 0.55)),
                        "hard": float(getattr(self.cfg, "hardness_debt_stale_hard_weight", 0.40)),
                    }
                )
                self._hardness_debt_escape_steps_left = max(
                    0,
                    int(self._hardness_debt_escape_steps_left) - 1,
                )
                mode = f"{mode}+debt_escape"
            else:
                debt_max = max(1e-6, float(getattr(self.cfg, "hardness_debt_max", 6.0)))
                debt_thr = max(
                    0.0,
                    min(
                        debt_max,
                        float(getattr(self.cfg, "hardness_debt_hard_recovery_threshold", 3.0)),
                    ),
                )
                if debt > debt_thr:
                    debt_ratio = min(1.0, (debt - debt_thr) / max(1e-6, debt_max - debt_thr))
                    recovery_weights = self._normalize_bucket_weights(
                        {
                            "easy": float(getattr(self.cfg, "hardness_debt_recovery_easy_weight", 0.0)),
                            "medium": float(getattr(self.cfg, "hardness_debt_recovery_medium_weight", 0.30)),
                            "hard": float(getattr(self.cfg, "hardness_debt_recovery_hard_weight", 0.70)),
                        }
                    )
                    mixed = {
                        k: ((1.0 - debt_ratio) * float(weights_for_sampling.get(k, 0.0)))
                        + (debt_ratio * float(recovery_weights.get(k, 0.0)))
                        for k in ("easy", "medium", "hard")
                    }
                    weights_for_sampling = self._normalize_bucket_weights(mixed)
                    mode = f"{mode}+debt_recovery"

        if int(self._forced_explore_steps_left) > 0:
            forced_hard = self._normalize_bucket_weights(
                {
                    "easy": 0.0,
                    "medium": float(getattr(self.cfg, "hardness_debt_recovery_medium_weight", 0.30)),
                    "hard": float(getattr(self.cfg, "hardness_debt_recovery_hard_weight", 0.70)),
                }
            )
            weights_for_sampling = forced_hard
            mode = f"{mode}+forced_explore"
            self._forced_explore_steps_left = max(0, int(self._forced_explore_steps_left) - 1)

        desired_bucket = self._sample_bucket(weights_for_sampling) if enabled else "medium"
        return {
            "enabled": enabled,
            "desired_bucket": desired_bucket,
            "mode": mode,
            "history_size": int(len(history)),
            "target_weights": target,
            "observed_weights": observed,
            "sampling_weights": weights_for_sampling,
            "hardness_debt": float(debt),
            "hardness_debt_ratio": float(debt_ratio),
            "hardness_debt_escape_active": bool(debt_escape_active),
        }

    def _quantile(self, values: List[float], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        qq = max(0.0, min(1.0, float(q)))
        sorted_vals = sorted(float(v) for v in values)
        pos = qq * float(len(sorted_vals) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(sorted_vals[lo])
        frac = float(pos - lo)
        return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)

    def _entropy_iqr_filter_state(self, majority_fraction: float = 1.0) -> Dict[str, float]:
        base_thr = max(0.0, float(getattr(self.cfg, "entropy_iqr_min_threshold", 0.02)))
        max_thr = max(base_thr, float(getattr(self.cfg, "entropy_iqr_max_threshold", 1.2)))
        state: Dict[str, float] = {
            "enabled": 1.0 if bool(getattr(self.cfg, "entropy_iqr_filter_enabled", True)) else 0.0,
            "active": 0.0,
            "history_size": float(len(self._entropy_easy_window)),
            "threshold": float(base_thr),
            "q1": float(base_thr),
            "q3": float(base_thr),
            "iqr": 0.0,
        }
        if state["enabled"] <= 0.5:
            return state
        min_samples = max(4, int(getattr(self.cfg, "entropy_iqr_min_samples", 32)))
        min_majority = max(
            0.0,
            min(1.0, float(getattr(self.cfg, "entropy_iqr_filter_min_majority_frac", 0.80))),
        )
        vals = list(self._entropy_easy_window)
        if len(vals) < min_samples or float(majority_fraction) < min_majority:
            return state
        q_easy = max(0.0, min(1.0, float(getattr(self.cfg, "entropy_iqr_easy_quantile", 0.25))))
        q1 = float(self._quantile(vals, q_easy))
        q3 = float(self._quantile(vals, max(q_easy, 0.75)))
        iqr = max(0.0, q3 - q1)
        coef = max(0.0, float(getattr(self.cfg, "entropy_iqr_easy_iqr_coef", 0.25)))
        thr = max(base_thr, min(max_thr, q1 - coef * iqr))
        state.update(
            {
                "active": 1.0,
                "threshold": float(thr),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
            }
        )
        return state

    def _difficulty_bucket_from_outcome(
        self,
        *,
        entropy_nats: float,
        majority_fraction: float,
        entropy_easy_threshold: Optional[float] = None,
    ) -> str:
        easy_thr = (
            max(0.0, float(entropy_easy_threshold))
            if entropy_easy_threshold is not None
            else max(0.0, float(getattr(self.cfg, "entropy_iqr_min_threshold", 0.02)))
        )
        hard_entropy = max(easy_thr, float(getattr(self.cfg, "difficulty_hard_min_entropy", 0.90)))
        hard_margin = max(0.0, float(getattr(self.cfg, "difficulty_hard_max_margin", 0.35)))
        margin = max(0.0, min(1.0, 2.0 * float(majority_fraction) - 1.0))
        if float(entropy_nats) <= easy_thr or float(majority_fraction) >= 0.999:
            return "easy"
        if float(entropy_nats) >= hard_entropy and margin <= hard_margin:
            return "hard"
        return "medium"

    def _apply_grpo_pairwise_ranking(
        self,
        rewards: List[float],
        buckets: List[str],
    ) -> tuple[List[float], List[float]]:
        if (not rewards) or (len(rewards) != len(buckets)):
            return list(rewards), []
        if not bool(getattr(self.cfg, "grpo_pairwise_ranking_enabled", True)):
            return list(rewards), []
        out = [float(r) for r in rewards]
        order = {"easy": 0, "medium": 1, "hard": 2}
        margin = max(0.0, float(getattr(self.cfg, "grpo_pairwise_margin", 0.10)))
        weight = max(0.0, float(getattr(self.cfg, "grpo_pairwise_ranking_weight", 0.15)))
        easy_pen = max(0.0, float(getattr(self.cfg, "grpo_pairwise_easy_penalty", 0.12)))
        rank_deltas: List[float] = []
        n = len(out)
        for i in range(n):
            for j in range(i + 1, n):
                oi = int(order.get(str(buckets[i]).lower(), 0))
                oj = int(order.get(str(buckets[j]).lower(), 0))
                if oi == oj:
                    continue
                sign = 1.0 if oi > oj else -1.0
                actual = float(out[i] - out[j])
                violation = margin - sign * actual
                rank_deltas.append(float(violation))
                if violation > 0.0:
                    adj = weight * violation
                    out[i] += sign * adj
                    out[j] -= sign * adj
                if oi == 0:
                    out[i] -= easy_pen * weight
                if oj == 0:
                    out[j] -= easy_pen * weight
        out = [max(-1.0, min(1.0, float(v))) for v in out]
        return out, rank_deltas

    def _apply_all_easy_relative_negatives(self, rewards: List[float], buckets: List[str]) -> List[float]:
        if (not rewards) or (len(rewards) != len(buckets)):
            return list(rewards)
        if not all(str(b).lower() == "easy" for b in buckets):
            return list(rewards)
        spread = max(0.0, float(getattr(self.cfg, "proposer_all_easy_rank_spread", 0.08)))
        if spread <= 0.0:
            return list(rewards)
        base = float(min(float(r) for r in rewards))
        order = sorted(range(len(rewards)), key=lambda idx: float(rewards[idx]), reverse=True)
        out = [float(v) for v in rewards]
        for rank, idx in enumerate(order):
            out[idx] = base - float(rank) * spread
        return [max(-1.0, min(1.0, float(v))) for v in out]

    def _apply_grpo_degenerate_noise(self, rewards: List[float]) -> tuple[List[float], bool]:
        if not rewards:
            return list(rewards), False
        if not bool(getattr(self.cfg, "grpo_degenerate_noise_enabled", True)):
            return list(rewards), False
        vals = [float(v) for v in rewards]
        mean_v = float(sum(vals) / float(len(vals)))
        var = float(sum((v - mean_v) ** 2 for v in vals) / float(len(vals)))
        std = float(math.sqrt(max(0.0, var)))
        threshold = max(0.0, float(getattr(self.cfg, "grpo_degenerate_noise_std_threshold", 1e-6)))
        if std > threshold:
            return vals, False
        sigma = max(0.0, float(getattr(self.cfg, "grpo_degenerate_noise_sigma", 0.03)))
        noisy = [max(-1.0, min(1.0, v + random.gauss(0.0, sigma))) for v in vals]
        return noisy, True

    def _early_failfast_state(self, *, u_step: int) -> Dict[str, float]:
        state: Dict[str, float] = {
            "enabled": 1.0 if bool(getattr(self.cfg, "proposer_early_failfast_enabled", True)) else 0.0,
            "u_step": float(max(0, int(u_step))),
            "stage1_active": 0.0,
            "stage2_active": 0.0,
            "stage1_pass": 1.0,
            "stage2_pass": 1.0,
            "candidate_non_easy_rate": 0.0,
            "all_easy_group_rate": 0.0,
            "reward_clipped_rate": 0.0,
            "selected_non_easy_rate": 0.0,
            "solver_update_applied_count": 0.0,
            "collapse_streak": float(int(getattr(self, "_proposer_collapse_streak", 0))),
            "max_collapse_streak": float(
                max(0, int(getattr(self.cfg, "proposer_early_max_collapse_streak", 3)))
            ),
            "recovery_armed": 0.0,
            "triggered": 0.0,
            "hard_stop_min_u_step": float(
                max(1, int(getattr(self.cfg, "proposer_early_hard_stop_min_u_step", 80)))
            ),
        }
        if state["enabled"] <= 0.5 or int(u_step) <= 0:
            return state

        state["candidate_non_easy_rate"] = _mean(list(self._candidate_non_easy_window))
        state["all_easy_group_rate"] = _mean(list(self._all_easy_group_window))
        state["reward_clipped_rate"] = _mean(list(self._proposer_reward_clipped_window))
        state["selected_non_easy_rate"] = _mean(list(self._selected_non_easy_window))
        state["solver_update_applied_count"] = float(sum(float(v) for v in self._solver_update_applied_window))

        step1 = max(1, int(getattr(self.cfg, "proposer_early_stage1_u_step", 12)))
        step2 = max(step1, int(getattr(self.cfg, "proposer_early_stage2_u_step", 20)))
        if int(u_step) >= step1:
            state["stage1_active"] = 1.0
            stage1_pass = (
                state["candidate_non_easy_rate"]
                >= float(getattr(self.cfg, "proposer_early_candidate_non_easy_rate_min", 0.08))
                and state["all_easy_group_rate"]
                <= float(getattr(self.cfg, "proposer_early_all_easy_rate_max", 0.93))
                and state["reward_clipped_rate"]
                <= float(getattr(self.cfg, "proposer_early_reward_clipped_rate_max", 0.85))
            )
            state["stage1_pass"] = 1.0 if stage1_pass else 0.0
            if not stage1_pass:
                state["triggered"] = 1.0
        if int(u_step) >= step2:
            state["stage2_active"] = 1.0
            stage2_pass = (
                state["selected_non_easy_rate"]
                >= float(getattr(self.cfg, "proposer_early_selected_non_easy_rate_min", 0.10))
                and state["solver_update_applied_count"]
                >= float(getattr(self.cfg, "proposer_early_solver_updates_min", 1))
            )
            state["stage2_pass"] = 1.0 if stage2_pass else 0.0
            if not stage2_pass:
                state["triggered"] = 1.0

        max_collapse = max(0, int(getattr(self.cfg, "proposer_early_max_collapse_streak", 3)))
        if state["stage1_active"] > 0.5 and int(state["collapse_streak"]) > max_collapse:
            state["triggered"] = 1.0

        if state["triggered"] > 0.5 and bool(getattr(self.cfg, "proposer_early_failfast_recover", True)):
            recover_steps = max(
                1,
                int(getattr(self.cfg, "proposer_early_failfast_recover_steps", 20)),
            )
            self._forced_explore_steps_left = max(int(self._forced_explore_steps_left), recover_steps)
            state["recovery_armed"] = 1.0
        return state

    def _proposer_certificate_score(self, question: str, meta: Dict[str, str]) -> Dict[str, float]:
        if not bool(getattr(self.cfg, "proposer_certificate_enabled", True)):
            return {"score": 1.0, "valid": 1.0}
        q = str(question or "").strip()
        structural = 1.0 if is_well_formed_question(q) else 0.0
        objective = 1.0 if is_objective_question(q) else 0.0
        components: List[float] = [structural, objective]

        visual_target = str(meta.get("visual_target", "") or "").strip()
        if visual_target:
            components.append(1.0)

        reasoning_chain = str(meta.get("reasoning_chain", "") or "").strip()
        if reasoning_chain:
            components.append(1.0 if len(reasoning_chain.split()) >= 4 else 0.0)

        reasoning_domains = str(meta.get("reasoning_domains", "") or "").strip()
        if reasoning_domains:
            domains = [d.strip() for d in reasoning_domains.split(",") if d.strip()]
            components.append(1.0 if len(domains) >= 1 else 0.0)

        two_raw = str(meta.get("two_answer_test", "") or "").strip()
        if two_raw:
            two_opts = [s.strip() for s in two_raw.replace(" vs ", "|").replace("/", "|").split("|") if s.strip()]
            components.append(1.0 if len(two_opts) >= 2 and two_opts[0].lower() != two_opts[1].lower() else 0.0)

        # If metadata is absent (simple schema mode), fall back to question-only quality checks.
        if len(components) <= 2:
            q_tokens = q.split()
            components.append(1.0 if 4 <= len(q_tokens) <= 28 else 0.0)
            components.append(1.0 if any(tok.lower() in {"left", "right", "top", "bottom", "color", "number", "count"} for tok in q_tokens) else 0.0)

        score = float(sum(components)) / float(max(1, len(components)))
        min_score = max(0.0, min(1.0, float(getattr(self.cfg, "proposer_certificate_min_score", 0.55))))
        valid = 1.0 if score >= min_score else 0.0
        return {"score": float(score), "valid": float(valid)}

    def _proposer_text_hardness_bonus(self, meta: Dict[str, str]) -> float:
        strategy = str(meta.get("strategy_used", "") or "").strip().upper()
        if not strategy:
            return 0.0
        if strategy.startswith("H"):
            return 0.16
        if strategy.startswith("M"):
            return 0.10
        return 0.06

    def _proposer_strategy_quota_bonus(self, meta: Dict[str, str]) -> float:
        strategy = str(meta.get("strategy_used", "") or "").strip().upper()
        if not strategy:
            return 0.0
        seen = int(self._strategy_hist.get(strategy, 0))
        return max(-0.05, 0.08 - 0.01 * float(seen))

    def _evaluate_proposer_candidate(
        self,
        *,
        image: Image.Image,
        question: str,
        spot_temps: List[float],
        candidate_index: int,
        meta: Optional[Dict[str, str]] = None,
        desired_bucket: str = "medium",
        entropy_easy_threshold: Optional[float] = None,
        warm_start_active: bool = False,
        penalty_boost: float = 1.0,
    ) -> Dict[str, Any]:
        cand_meta = dict(meta or {})
        question = " ".join(str(question or "").strip().split())
        if not is_well_formed_question(question):
            return {
                "valid": False,
                "candidate_index": int(candidate_index),
                "question": question,
                "completion": f"<question>{question}</question>",
                "meta": cand_meta,
                "solver_outputs_raw": [],
                "spot_answers_norm": [],
                "spot_solver_samples": [],
                "entropy_nats": 0.0,
                "majority_fraction": 0.0,
                "easy_case": True,
                "unsolvable_case": False,
                "non_objective": not is_objective_question(question),
                "reward": -1.0,
                "reward_raw": -1.0,
                "reward_bonus": 0.0,
                "sample_entropy_difficulty": 0.0,
                "ste_difficulty": 0.0,
                "score": -1.0,
                "gate_passed": False,
                "acceptable": False,
                "certificate_score": 0.0,
                "certificate_valid": 0.0,
                "text_hardness_bonus": 0.0,
                "strategy_quota_bonus": 0.0,
                "contrastive_bonus": 0.0,
            }
        focus_hint = str(cand_meta.get("visual_target", "") or "").strip()
        outputs_raw: List[str] = []
        answers_norm: List[str] = []
        solver_samples: List[Any] = []
        for sample_idx, temp in enumerate(spot_temps):
            sample_prompt = build_solver_prompt_pps(
                question_text=question,
                template_index=int(sample_idx + candidate_index),
                focus_hint=focus_hint,
            )
            out = self.adapter.solve_question(
                image=image,
                question=question,
                max_new_tokens=self.cfg.max_new_tokens_solver,
                temperature=float(temp),
                do_sample=True,
                template_index=int(sample_idx + candidate_index),
                focus_hint=focus_hint,
                use_pps=True,
            )
            outputs_raw.append(out.text)
            ans = normalize_answer(parse_answer(out.text))
            if ans:
                answers_norm.append(ans)
                solver_samples.append((out.text, ans, sample_prompt))

        if not answers_norm:
            return {
                "valid": False,
                "candidate_index": int(candidate_index),
                "question": question,
                "completion": f"<question>{question}</question>",
                "meta": cand_meta,
                "solver_outputs_raw": outputs_raw,
                "spot_answers_norm": [],
                "spot_solver_samples": [],
                "entropy_nats": 0.0,
                "majority_fraction": 0.0,
                "easy_case": True,
                "unsolvable_case": False,
                "non_objective": not is_objective_question(question),
                "reward": -1.0,
                "reward_raw": -1.0,
                "reward_bonus": 0.0,
                "sample_entropy_difficulty": 0.0,
                "ste_difficulty": 0.0,
                "score": -1.0,
                "gate_passed": False,
                "acceptable": False,
                "certificate_score": 0.0,
                "certificate_valid": 0.0,
                "text_hardness_bonus": 0.0,
                "strategy_quota_bonus": 0.0,
                "contrastive_bonus": 0.0,
            }

        dual = compute_dual_track_reward(
            answers=answers_norm,
            intuitive_answer="",
            entropy_mu=self.cfg.proposer_entropy_mu,
            entropy_sigma=self.cfg.proposer_entropy_sigma,
            unsolvable_maj_threshold=self.cfg.solver_unsolvable_maj_threshold,
            zero_entropy_eps=self.cfg.zero_entropy_eps,
        )

        reward_raw = float(dual.reward)
        reward = float(reward_raw)
        penalty_scale = max(1.0, float(penalty_boost))
        non_objective = not is_objective_question(question)
        if self.cfg.proposer_require_objective and non_objective:
            reward -= penalty_scale * float(self.cfg.proposer_non_objective_penalty)
        if self.cfg.acceptance_require_non_easy and bool(dual.easy_case):
            easy_penalty_scale = 1.0
            if bool(warm_start_active):
                easy_penalty_scale = max(
                    0.0,
                    float(
                        getattr(
                            self.cfg,
                            "proposer_warm_start_easy_reject_penalty_scale",
                            0.0,
                        )
                    ),
                )
            reward -= penalty_scale * easy_penalty_scale * float(self.cfg.rejected_question_penalty)

        cert = self._proposer_certificate_score(question, cand_meta)
        cert_score = float(cert.get("score", 0.0))
        cert_valid = float(cert.get("valid", 0.0))
        cert_weight = max(0.0, float(getattr(self.cfg, "proposer_certificate_weight", 0.75)))
        cert_min = max(0.0, min(1.0, float(getattr(self.cfg, "proposer_certificate_min_score", 0.55))))
        cert_bonus = cert_weight * (cert_score - cert_min)
        text_hardness_bonus = float(self._proposer_text_hardness_bonus(cand_meta))
        strategy_quota_bonus = float(self._proposer_strategy_quota_bonus(cand_meta))
        contrastive_bonus = float(self._contrastive_replay_adjustment(question))
        bonus_enabled = (not bool(dual.easy_case)) or bool(warm_start_active)
        reward_bonus = 0.0
        if bonus_enabled:
            reward_bonus = (
                float(cert_bonus)
                + float(text_hardness_bonus)
                + float(strategy_quota_bonus)
                + float(contrastive_bonus)
            )
            reward += reward_bonus
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
        ste_easy_quantile = max(
            0.0,
            min(1.0, float(getattr(self.cfg, "ste_spot_easy_quantile", 0.30))),
        )
        ste_primary = bool(float(ste_difficulty) > ste_easy_quantile)
        sample_weight = max(0.0, float(getattr(self.cfg, "proposer_sample_entropy_weight", 0.30)))
        ste_weight = max(0.0, float(getattr(self.cfg, "proposer_ste_primary_weight", 0.70)))
        score = float(reward) + sample_weight * sample_entropy_difficulty + ste_weight * ste_difficulty

        min_gate = max(0.0, float(getattr(self.cfg, "proposer_spot_entropy_min_gate", 0.05)))
        gate_passed = bool(float(dual.entropy_nats) >= min_gate)
        if str(desired_bucket or "").strip().lower() == "hard":
            hard_min_entropy = max(0.0, float(getattr(self.cfg, "difficulty_hard_min_entropy", 0.90)))
            gate_passed = bool(gate_passed and float(dual.entropy_nats) >= hard_min_entropy)
        acceptable = bool(gate_passed)
        if self.cfg.proposer_require_objective and non_objective:
            acceptable = False
        if self.cfg.acceptance_require_non_easy and bool(dual.easy_case):
            acceptable = False
        if bool(getattr(self.cfg, "proposer_reject_unsolvable", True)) and bool(dual.unsolvable_case):
            acceptable = False
        if bool(getattr(self.cfg, "proposer_certificate_enabled", True)) and bool(
            getattr(self.cfg, "proposer_certificate_strict_struct", True)
        ):
            acceptable = bool(acceptable and cert_valid > 0.5)
        difficulty_bucket = self._difficulty_bucket_from_outcome(
            entropy_nats=float(dual.entropy_nats),
            majority_fraction=float(dual.majority_fraction),
            entropy_easy_threshold=entropy_easy_threshold,
        )

        return {
            "valid": True,
            "candidate_index": int(candidate_index),
            "question": question,
            "completion": f"<question>{question}</question>",
            "meta": cand_meta,
            "solver_outputs_raw": outputs_raw,
            "spot_answers_norm": answers_norm,
            "spot_solver_samples": solver_samples,
            "entropy_nats": float(dual.entropy_nats),
            "majority_fraction": float(dual.majority_fraction),
            "easy_case": bool(dual.easy_case),
            "unsolvable_case": bool(dual.unsolvable_case),
            "non_objective": bool(non_objective),
            "reward": float(reward),
            "reward_raw": float(reward_raw),
            "reward_bonus": float(reward_bonus),
            "sample_entropy_difficulty": float(sample_entropy_difficulty),
            "ste_difficulty": float(ste_difficulty),
            "ste_primary": bool(ste_primary),
            "score": float(score),
            "gate_passed": bool(gate_passed),
            "acceptable": bool(acceptable),
            "difficulty_bucket": str(difficulty_bucket),
            "certificate_score": float(cert_score),
            "certificate_valid": float(cert_valid),
            "text_hardness_bonus": float(text_hardness_bonus),
            "strategy_quota_bonus": float(strategy_quota_bonus),
            "contrastive_bonus": float(contrastive_bonus),
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
        prev_generator_baseline = float(self.generator_baseline)
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
            self.generator_baseline = prev_generator_baseline

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

        reward = float(rec.get("best_total_reward", rec.get("proposer_gen_reward", 0.0)))
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
        reward = float(rec.get("best_total_reward", rec.get("proposer_gen_reward", 0.0)))
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
        u_step: int,
        image_path: str,
        image: Image.Image,
        solver_temps: List[float],
        baseline_momentum: float,
    ) -> Dict[str, Any]:
        entropy_iqr_state = self._entropy_iqr_filter_state(majority_fraction=1.0)
        entropy_easy_threshold = float(
            entropy_iqr_state.get(
                "threshold",
                max(0.0, float(getattr(self.cfg, "entropy_iqr_min_threshold", 0.02))),
            )
        )
        difficulty_target_state = self._choose_difficulty_target()
        desired_difficulty_bucket = str(difficulty_target_state.get("desired_bucket", "medium"))
        difficulty_sampler_mode = str(difficulty_target_state.get("mode", "target"))
        proposer_warm_start_active = self._is_proposer_warm_start_active(max(1, int(u_step)))
        proposer_candidate_count = max(1, int(getattr(self.cfg, "proposer_num_candidates", 1)))
        if int(self._forced_explore_steps_left) > 0:
            proposer_candidate_count = max(
                proposer_candidate_count,
                max(1, int(getattr(self.cfg, "all_easy_explore_num_candidates", proposer_candidate_count))),
            )
        proposer_temp = float(self.cfg.proposer_temperature)
        debt_ratio = float(difficulty_target_state.get("hardness_debt_ratio", 0.0))
        debt_temp_boost = max(0.0, float(getattr(self.cfg, "hardness_debt_temp_boost_max", 0.30)))
        penalty_boost = 1.0 + min(1.0, max(0.0, debt_ratio)) * max(
            0.0,
            float(getattr(self.cfg, "hardness_debt_penalty_boost_max", 0.30)),
        )
        if debt_ratio > 0.0:
            proposer_temp *= (1.0 + min(1.0, debt_ratio) * debt_temp_boost)
        if int(self._forced_explore_steps_left) > 0:
            proposer_temp *= (
                1.0 + max(0.0, float(getattr(self.cfg, "all_easy_explore_temp_boost", 1.20)) - 1.0)
            )
            proposer_temp *= (
                1.0 + 0.5 * max(0.0, float(getattr(self.cfg, "all_easy_explore_top_p_boost", 0.15)))
            )
            penalty_boost += max(0.0, float(getattr(self.cfg, "all_easy_explore_penalty_boost", 0.50)))
        proposer_temp = max(0.05, min(4.0, proposer_temp))

        proposer = self.adapter.propose_questions(
            image=image,
            max_new_tokens=self.cfg.max_new_tokens_proposer,
            temperature=proposer_temp,
            num_questions=proposer_candidate_count,
            target_difficulty=desired_difficulty_bucket,
            do_sample=True,
        )

        proposer_retry_attempted = False
        proposer_retry_count = 0
        proposer_retry_temps: List[float] = []
        proposer_retry_recovered = False

        def _extract_candidate_infos(raw_text: str) -> List[Dict[str, str]]:
            candidate_infos_local = parse_proposer_question_candidates(raw_text)
            if not candidate_infos_local:
                candidate_questions_local = parse_all_questions(raw_text)
                fallback_q = parse_first_question(raw_text)
                if fallback_q:
                    candidate_questions_local = candidate_questions_local + [fallback_q]
                candidate_infos_local = [{"text": q} for q in candidate_questions_local if str(q or "").strip()]

            deduped_infos_local: List[Dict[str, str]] = []
            seen_q_local = set()
            for info in candidate_infos_local:
                qq = str(info.get("text", "") or "").strip()
                if not qq:
                    continue
                qq = " ".join(qq.split())
                if not is_well_formed_question(qq):
                    continue
                key = qq.lower()
                if key in seen_q_local:
                    continue
                seen_q_local.add(key)
                clean_info = {k: str(v) for k, v in info.items() if str(v).strip()}
                clean_info["text"] = qq
                deduped_infos_local.append(clean_info)
            return deduped_infos_local[:proposer_candidate_count]

        candidate_infos = _extract_candidate_infos(proposer.text)
        retry_budget = max(0, int(os.environ.get("BAGEL_PROPOSER_PARSE_RETRIES", "2") or "2"))
        retry_decay = float(os.environ.get("BAGEL_PROPOSER_PARSE_RETRY_TEMP_DECAY", "0.70") or "0.70")
        if retry_decay <= 0.0 or retry_decay >= 1.0:
            retry_decay = 0.70
        while (not candidate_infos) and proposer_retry_count < retry_budget:
            proposer_retry_attempted = True
            proposer_retry_count += 1
            retry_temp = max(0.15, min(4.0, proposer_temp * (retry_decay ** proposer_retry_count)))
            proposer_retry_temps.append(float(retry_temp))
            retry_questions = proposer_candidate_count if proposer_retry_count < retry_budget else 1
            proposer_retry = self.adapter.propose_questions(
                image=image,
                max_new_tokens=self.cfg.max_new_tokens_proposer,
                temperature=float(retry_temp),
                num_questions=int(retry_questions),
                target_difficulty=desired_difficulty_bucket,
                do_sample=bool(proposer_retry_count < retry_budget),
            )
            recovered_infos = _extract_candidate_infos(proposer_retry.text)
            if recovered_infos:
                proposer = proposer_retry
                candidate_infos = recovered_infos
                proposer_retry_recovered = True
                break

        candidate_questions = [str(info.get("text", "")) for info in candidate_infos]

        if not candidate_questions:
            record = {
                "step": int(step),
                "phase": "understanding",
                "status": "skipped",
                "skip_reason": "empty_question",
                "image_path": image_path,
                "proposer_raw": proposer.text if self.cfg.save_raw_generations else "",
                "proposer_retry_attempted": bool(proposer_retry_attempted),
                "proposer_retry_count": int(proposer_retry_count),
                "proposer_retry_temps": proposer_retry_temps,
                "proposer_retry_recovered": bool(proposer_retry_recovered),
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
                question=str(info.get("text", "")),
                spot_temps=spot_temps,
                candidate_index=idx,
                meta=info,
                desired_bucket=desired_difficulty_bucket,
                entropy_easy_threshold=entropy_easy_threshold,
                warm_start_active=proposer_warm_start_active,
                penalty_boost=penalty_boost,
            )
            for idx, info in enumerate(candidate_infos)
        ]
        valid_candidates = [c for c in candidate_stats if bool(c.get("valid", False))]
        if not valid_candidates:
            record = {
                "step": int(step),
                "phase": "understanding",
                "status": "skipped",
                "skip_reason": "no_valid_candidates",
                "image_path": image_path,
                "candidate_questions": candidate_questions,
                "proposer_candidate_count_requested": int(proposer_candidate_count),
                "proposer_candidate_count_parsed": int(len(candidate_questions)),
                "proposer_candidate_count_valid": 0,
                "proposer_raw": proposer.text if self.cfg.save_raw_generations else "",
                "proposer_retry_attempted": bool(proposer_retry_attempted),
                "proposer_retry_count": int(proposer_retry_count),
                "proposer_retry_temps": proposer_retry_temps,
                "proposer_retry_recovered": bool(proposer_retry_recovered),
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
        if (not acceptable_candidates) and bool(getattr(self.cfg, "understanding_skip_no_acceptable", True)):
            record = {
                "step": int(step),
                "phase": "understanding",
                "status": "skipped",
                "skip_reason": "no_acceptable_candidates",
                "image_path": image_path,
                "proposer_raw": proposer.text if self.cfg.save_raw_generations else "",
                "proposer_candidate_questions": candidate_questions,
                "proposer_candidate_count_requested": int(proposer_candidate_count),
                "proposer_candidate_count_parsed": int(len(candidate_questions)),
                "proposer_candidate_count_valid": int(len(valid_candidates)),
                "proposer_candidate_count_acceptable": 0,
                "proposer_candidate_rewards": [float(c.get("reward", 0.0)) for c in valid_candidates],
                "proposer_candidate_scores": [float(c.get("score", 0.0)) for c in valid_candidates],
                "proposer_candidate_entropy_nats": [float(c.get("entropy_nats", 0.0)) for c in valid_candidates],
                "proposer_candidate_ste_difficulty": [float(c.get("ste_difficulty", 0.0)) for c in valid_candidates],
                "proposer_candidate_certificate_scores": [float(c.get("certificate_score", 0.0)) for c in valid_candidates],
                "proposer_candidate_easy_flags": [bool(c.get("easy_case", True)) for c in valid_candidates],
                "proposer_candidate_non_easy_rate": float(
                    sum(1 for c in valid_candidates if not bool(c.get("easy_case", True))) / float(max(1, len(valid_candidates)))
                ),
                "proposer_retry_attempted": bool(proposer_retry_attempted),
                "proposer_retry_count": int(proposer_retry_count),
                "proposer_retry_temps": proposer_retry_temps,
                "proposer_retry_recovered": bool(proposer_retry_recovered),
                "policy_update_attempted": False,
                "policy_update_applied": False,
                "policy_update_reason": "gated_no_acceptable_candidates",
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

        if bool(getattr(self.cfg, "understanding_require_acceptable_for_update", True)):
            selected_pool = acceptable_candidates if acceptable_candidates else valid_candidates
        else:
            selected_pool = acceptable_candidates or valid_candidates
        all_easy_candidate_group = bool(valid_candidates and all(bool(c.get("easy_case", True)) for c in valid_candidates))
        if all_easy_candidate_group:
            self._all_easy_streak = int(self._all_easy_streak) + 1
        else:
            self._all_easy_streak = 0
        explore_trigger = max(1, int(getattr(self.cfg, "all_easy_explore_trigger", 2)))
        if int(self._all_easy_streak) >= explore_trigger:
            self._forced_explore_steps_left = max(
                int(self._forced_explore_steps_left),
                max(1, int(getattr(self.cfg, "all_easy_explore_steps", 10))),
            )
            self._all_easy_streak = 0
        selected_candidate = max(
            selected_pool,
            key=lambda c: (float(c.get("score", -1.0)), float(c.get("reward", -1.0)), float(c.get("entropy_nats", 0.0))),
        )
        question = str(selected_candidate["question"])
        selected_meta = dict(selected_candidate.get("meta", {}))

        solver_outputs_raw: List[str] = list(selected_candidate.get("solver_outputs_raw", []))
        solver_answers_norm: List[str] = list(selected_candidate.get("spot_answers_norm", []))
        solver_samples: List[Any] = list(selected_candidate.get("spot_solver_samples", []))
        focus_hint = str(selected_meta.get("visual_target", "") or "").strip()
        for sample_idx, temp in enumerate(solver_temps[len(spot_temps):], start=len(spot_temps)):
            sample_prompt = build_solver_prompt_pps(
                question_text=question,
                template_index=int(sample_idx),
                focus_hint=focus_hint,
            )
            out = self.adapter.solve_question(
                image=image,
                question=question,
                max_new_tokens=self.cfg.max_new_tokens_solver,
                temperature=float(temp),
                do_sample=True,
                template_index=int(sample_idx),
                focus_hint=focus_hint,
                use_pps=True,
            )
            solver_outputs_raw.append(out.text)
            ans = normalize_answer(parse_answer(out.text))
            if ans:
                solver_answers_norm.append(ans)
                solver_samples.append((out.text, ans, sample_prompt))

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

        reward_raw = float(dual.reward)
        reward = float(reward_raw)
        non_objective = not is_objective_question(question)
        if self.cfg.proposer_require_objective and non_objective:
            reward -= float(self.cfg.proposer_non_objective_penalty)
        if self.cfg.acceptance_require_non_easy and dual.easy_case:
            easy_penalty_scale = 1.0
            if bool(proposer_warm_start_active):
                easy_penalty_scale = max(
                    0.0,
                    float(
                        getattr(
                            self.cfg,
                            "proposer_warm_start_easy_reject_penalty_scale",
                            0.0,
                        )
                    ),
                )
            reward -= easy_penalty_scale * float(self.cfg.rejected_question_penalty)

        cert = self._proposer_certificate_score(question, selected_meta)
        cert_score = float(cert.get("score", 0.0))
        cert_valid = float(cert.get("valid", 0.0))
        cert_weight = max(0.0, float(getattr(self.cfg, "proposer_certificate_weight", 0.75)))
        cert_min = max(0.0, min(1.0, float(getattr(self.cfg, "proposer_certificate_min_score", 0.55))))
        cert_bonus = cert_weight * (cert_score - cert_min)
        if bool(proposer_warm_start_active):
            cert_bonus = max(
                cert_bonus,
                max(0.0, float(getattr(self.cfg, "proposer_warm_start_certificate_weight", 0.50)))
                * cert_score,
            )
        text_hardness_bonus = float(self._proposer_text_hardness_bonus(selected_meta))
        strategy_quota_bonus = float(self._proposer_strategy_quota_bonus(selected_meta))
        contrastive_bonus = float(self._contrastive_replay_adjustment(question))
        proposer_bonus_enabled = (not bool(dual.easy_case)) or bool(proposer_warm_start_active)
        proposer_bonus = 0.0
        if proposer_bonus_enabled:
            proposer_bonus = (
                float(cert_bonus)
                + float(text_hardness_bonus)
                + float(strategy_quota_bonus)
                + float(contrastive_bonus)
            )
            reward += proposer_bonus
        reward = max(-1.0, min(1.0, reward))

        selected_candidate_acceptable = bool(selected_candidate.get("acceptable", False))
        entropy_train_threshold = max(
            float(entropy_easy_threshold),
            max(0.0, float(getattr(self.cfg, "proposer_spot_entropy_min_gate", 0.05))),
        )
        understanding_update_eligible = True
        understanding_update_skip_reason = "ok"
        if bool(getattr(self.cfg, "understanding_require_acceptable_for_update", True)) and (
            not selected_candidate_acceptable
        ):
            understanding_update_eligible = False
            understanding_update_skip_reason = "selected_candidate_not_acceptable"
        elif self.cfg.proposer_require_objective and non_objective:
            understanding_update_eligible = False
            understanding_update_skip_reason = "non_objective_question"
        elif bool(getattr(self.cfg, "proposer_reject_unsolvable", True)) and bool(dual.unsolvable_case):
            understanding_update_eligible = False
            understanding_update_skip_reason = "unsolvable_case"
        elif bool(getattr(self.cfg, "understanding_update_require_disagreement", True)):
            if bool(dual.easy_case):
                understanding_update_eligible = False
                understanding_update_skip_reason = "easy_case"
            elif float(dual.entropy_nats) < entropy_train_threshold:
                understanding_update_eligible = False
                understanding_update_skip_reason = "low_disagreement_entropy"
        if bool(getattr(self.cfg, "proposer_certificate_enabled", True)) and bool(
            getattr(self.cfg, "proposer_certificate_strict_struct", True)
        ) and cert_valid <= 0.5:
            understanding_update_eligible = False
            understanding_update_skip_reason = "certificate_invalid"

        proposer_update_stats: Dict[str, Any] = {"skipped": True, "reason": "disabled"}
        proposer_update_attempted = False
        proposer_update_applied = False
        policy_attempted = 0
        policy_applied = 0
        if (
            self.policy_updates_enabled
            and self.cfg.train_understanding_proposer
            and self.proposer_updater is not None
            and understanding_update_eligible
        ):
            proposer_update_attempted = True
            update_method = self.cfg.normalized_update_method()
            proposer_update_candidates = (
                acceptable_candidates
                if bool(getattr(self.cfg, "understanding_require_acceptable_for_update", True))
                else valid_candidates
            )
            if update_method == "grpo" and len(proposer_update_candidates) > 1:
                group_rewards: List[float] = []
                for cand in proposer_update_candidates:
                    cand_reward = float(cand.get("reward", 0.0))
                    if int(cand.get("candidate_index", -1)) == int(selected_candidate.get("candidate_index", -2)):
                        cand_reward = float(reward)
                    group_rewards.append(cand_reward)
                group_buckets = [str(c.get("difficulty_bucket", "easy")) for c in proposer_update_candidates]
                group_rewards, grpo_rank_deltas = self._apply_grpo_pairwise_ranking(
                    group_rewards,
                    group_buckets,
                )
                group_rewards = self._apply_all_easy_relative_negatives(group_rewards, group_buckets)
                group_rewards, grpo_degenerate_noise = self._apply_grpo_degenerate_noise(group_rewards)

                per_candidate_stats: List[Dict[str, Any]] = []
                applied_count = 0
                max_proposer_updates = max(
                    0,
                    int(os.environ.get("BAGEL_PROPOSER_POLICY_MAX_CANDIDATES", "0") or "0"),
                )
                for cand, cand_reward in zip(proposer_update_candidates, group_rewards):
                    if max_proposer_updates > 0 and len(per_candidate_stats) >= max_proposer_updates:
                        break
                    stats = self.proposer_updater.step(
                        image=image,
                        prompt=build_proposer_prompt(target_difficulty=desired_difficulty_bucket),
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
                    "grpo_pairwise_rank_delta_mean": float(_mean(grpo_rank_deltas)),
                    "grpo_pairwise_rank_delta_max": float(max(grpo_rank_deltas) if grpo_rank_deltas else 0.0),
                    "grpo_pairwise_rank_delta_min": float(min(grpo_rank_deltas) if grpo_rank_deltas else 0.0),
                    "grpo_degenerate_noise": bool(grpo_degenerate_noise),
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
                    prompt=build_proposer_prompt(target_difficulty=desired_difficulty_bucket),
                    completion=str(selected_candidate.get("completion", f"<question>{question}</question>")),
                    reward=float(reward),
                    baseline=self.proposer_baseline,
                )
                proposer_update_applied = bool(not proposer_update_stats.get("skipped", True))
                policy_attempted += 1
                policy_applied += int(proposer_update_applied)
        elif self.policy_updates_enabled and self.cfg.train_understanding_proposer and self.proposer_updater is not None:
            proposer_update_stats = {"skipped": True, "reason": f"gated_{understanding_update_skip_reason}"}

        if understanding_update_eligible:
            self.proposer_baseline = (
                baseline_momentum * self.proposer_baseline
                + (1.0 - baseline_momentum) * float(reward)
            )

        solver_group_rewards = [
            float(answer_match_score(ans_norm, dual.majority_answer))
            for _, ans_norm, _ in solver_samples
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
        if not understanding_update_eligible and bool(getattr(self.cfg, "understanding_update_require_disagreement", True)):
            solver_skip_update = True
            solver_update_reason = f"gated_{understanding_update_skip_reason}"
        elif bool(getattr(self.cfg, "solver_skip_unsolvable_updates", True)) and bool(dual.unsolvable_case):
            solver_skip_update = True
            solver_update_reason = "unsolvable_case_skip"
        elif solver_skip_update:
            solver_update_reason = "easy_question_skip"
        elif self.policy_updates_enabled and self.cfg.train_solver and self.solver_updater is not None:
            max_solver_updates = max(
                0,
                int(os.environ.get("BAGEL_SOLVER_POLICY_MAX_SAMPLES", "0") or "0"),
            )
            for idx, (sample_raw, _, sample_prompt) in enumerate(solver_samples):
                if max_solver_updates > 0 and idx >= max_solver_updates:
                    break
                sample_reward = (
                    float(solver_group_rewards[idx])
                    if idx < len(solver_group_rewards)
                    else float(solver_scalar_reward)
                )
                update_stats = self.solver_updater.step(
                    image=image,
                    prompt=sample_prompt,
                    completion=sample_raw,
                    reward=float(sample_reward),
                    baseline=self.solver_baseline,
                    group_rewards=solver_group_rewards,
                )
                solver_update_stats.append(update_stats)
                policy_attempted += 1
                policy_applied += int(not update_stats.get("skipped", True))
            solver_update_reason = "ok" if solver_update_stats else "no_samples"

        if understanding_update_eligible:
            self.solver_baseline = (
                baseline_momentum * self.solver_baseline
                + (1.0 - baseline_momentum) * float(solver_scalar_reward)
            )

        selected_non_easy = int(not bool(dual.easy_case))
        self._candidate_non_easy_window.append(
            float(sum(1 for c in valid_candidates if not bool(c.get("easy_case", True))) / float(max(1, len(valid_candidates))))
        )
        self._all_easy_group_window.append(1.0 if all_easy_candidate_group else 0.0)
        self._proposer_reward_clipped_window.append(1.0 if abs(float(reward)) >= 0.999 else 0.0)
        self._selected_non_easy_window.append(float(selected_non_easy))
        solver_updates_applied_now = int(sum(int(not s.get("skipped", True)) for s in solver_update_stats))
        self._solver_update_applied_window.append(float(solver_updates_applied_now))
        self._entropy_easy_window.append(float(dual.entropy_nats))
        entropy_iqr_state = self._entropy_iqr_filter_state(majority_fraction=float(dual.majority_fraction))
        entropy_easy_threshold = float(
            entropy_iqr_state.get(
                "threshold",
                max(0.0, float(getattr(self.cfg, "entropy_iqr_min_threshold", 0.02))),
            )
        )

        observed_bucket = self._difficulty_bucket_from_outcome(
            entropy_nats=float(dual.entropy_nats),
            majority_fraction=float(dual.majority_fraction),
            entropy_easy_threshold=entropy_easy_threshold,
        )
        if observed_bucket == "easy":
            self._proposer_collapse_streak = int(getattr(self, "_proposer_collapse_streak", 0)) + 1
        else:
            self._proposer_collapse_streak = 0
        self._difficulty_window.append(str(observed_bucket))
        hardness_debt_state = self._update_hardness_debt(observed_bucket)
        warm_start_state = self._update_proposer_warm_start_state(
            entropy_nats=float(dual.entropy_nats),
            u_step=max(1, int(u_step)),
        )
        if str(selected_meta.get("strategy_used", "")).strip():
            self._strategy_hist[str(selected_meta.get("strategy_used", "")).strip().upper()] += 1
        q_tokens = self._question_token_set(question)
        if q_tokens:
            if observed_bucket in {"medium", "hard"} and float(reward) > 0.0:
                self._contrastive_pos_replay.append(set(q_tokens))
            else:
                self._contrastive_neg_replay.append(set(q_tokens))
        early_failfast_state = self._early_failfast_state(u_step=max(1, int(u_step)))
        if (
            early_failfast_state.get("triggered", 0.0) > 0.5
            and bool(getattr(self.cfg, "proposer_early_failfast_stop", False))
            and int(u_step) >= max(1, int(getattr(self.cfg, "proposer_early_hard_stop_min_u_step", 80)))
        ):
            raise RuntimeError(
                "[EarlyFailFast] unhealthy run detected: "
                f"u_step={int(u_step)} "
                f"cand_non_easy_rate={early_failfast_state.get('candidate_non_easy_rate', 0.0):.3f} "
                f"all_easy_rate={early_failfast_state.get('all_easy_group_rate', 0.0):.3f} "
                f"reward_clipped_rate={early_failfast_state.get('reward_clipped_rate', 0.0):.3f} "
                f"selected_non_easy_rate={early_failfast_state.get('selected_non_easy_rate', 0.0):.3f} "
                f"solver_updates={early_failfast_state.get('solver_update_applied_count', 0.0):.1f} "
                f"collapse_streak={early_failfast_state.get('collapse_streak', 0.0):.1f}"
            )

        candidate_rewards = [float(c.get("reward", 0.0)) for c in valid_candidates]
        candidate_scores = [float(c.get("score", 0.0)) for c in valid_candidates]
        candidate_entropy = [float(c.get("entropy_nats", 0.0)) for c in valid_candidates]
        candidate_ste = [float(c.get("ste_difficulty", 0.0)) for c in valid_candidates]
        candidate_easy = [bool(c.get("easy_case", True)) for c in valid_candidates]
        candidate_cert = [float(c.get("certificate_score", 0.0)) for c in valid_candidates]
        candidate_non_easy_rate = float(
            sum(1 for c in valid_candidates if not bool(c.get("easy_case", True)))
            / float(max(1, len(valid_candidates)))
        )

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
            "proposer_candidate_certificate_scores": candidate_cert,
            "proposer_candidate_easy_flags": candidate_easy,
            "proposer_all_easy_candidate_group": bool(all_easy_candidate_group),
            "proposer_candidate_non_easy_rate": float(candidate_non_easy_rate),
            "proposer_selected_candidate_index": int(selected_candidate.get("candidate_index", -1)),
            "proposer_selected_candidate_score": float(selected_candidate.get("score", 0.0)),
            "proposer_selected_candidate_reward_spot": float(selected_candidate.get("reward", 0.0)),
            "proposer_selected_candidate_reward_raw_spot": float(selected_candidate.get("reward_raw", 0.0)),
            "proposer_selected_candidate_reward_bonus_spot": float(selected_candidate.get("reward_bonus", 0.0)),
            "proposer_selected_candidate_entropy_nats": float(selected_candidate.get("entropy_nats", 0.0)),
            "proposer_selected_candidate_ste_difficulty": float(selected_candidate.get("ste_difficulty", 0.0)),
            "proposer_selected_candidate_certificate_score": float(selected_candidate.get("certificate_score", 0.0)),
            "proposer_selected_candidate_certificate_valid": float(selected_candidate.get("certificate_valid", 0.0)),
            "proposer_retry_attempted": bool(proposer_retry_attempted),
            "proposer_retry_count": int(proposer_retry_count),
            "proposer_retry_temps": proposer_retry_temps,
            "proposer_retry_recovered": bool(proposer_retry_recovered),
            "proposer_spot_check_samples": int(len(spot_temps)),
            "proposer_target_difficulty_bucket": str(desired_difficulty_bucket),
            "difficulty_sampler_mode": str(difficulty_sampler_mode),
            "difficulty_bucket_observed": str(observed_bucket),
            "difficulty_target_weights": difficulty_target_state.get("target_weights", {}),
            "difficulty_observed_weights": difficulty_target_state.get("observed_weights", {}),
            "difficulty_sampling_weights": difficulty_target_state.get("sampling_weights", {}),
            "entropy_iqr_filter_enabled": bool(entropy_iqr_state.get("enabled", 0.0) > 0.5),
            "entropy_iqr_filter_active": bool(entropy_iqr_state.get("active", 0.0) > 0.5),
            "entropy_iqr_filter_history_size": int(entropy_iqr_state.get("history_size", 0.0)),
            "entropy_iqr_filter_q1": float(entropy_iqr_state.get("q1", entropy_easy_threshold)),
            "entropy_iqr_filter_q3": float(entropy_iqr_state.get("q3", entropy_easy_threshold)),
            "entropy_iqr_filter_iqr": float(entropy_iqr_state.get("iqr", 0.0)),
            "entropy_easy_threshold": float(entropy_easy_threshold),
            "proposer_reasoning_domains": str(selected_meta.get("reasoning_domains", "")),
            "proposer_reasoning_chain": str(selected_meta.get("reasoning_chain", "")),
            "proposer_strategy_used": str(selected_meta.get("strategy_used", "")),
            "proposer_task_card": str(selected_meta.get("task_card", "")),
            "proposer_visual_target": str(selected_meta.get("visual_target", "")),
            "proposer_two_answer_test": str(selected_meta.get("two_answer_test", "")),
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
            "proposer_reward_raw_base": float(reward_raw),
            "proposer_reward_bonus_total": float(proposer_bonus),
            "proposer_text_hardness_bonus": float(text_hardness_bonus),
            "proposer_strategy_quota_bonus": float(strategy_quota_bonus),
            "proposer_certificate_score": float(cert_score),
            "proposer_certificate_valid": float(cert_valid),
            "proposer_contrastive_replay_bonus": float(contrastive_bonus),
            "proposer_warm_start_active": bool(proposer_warm_start_active),
            "proposer_warm_start_active_next": bool(warm_start_state.get("active_next", 0.0) > 0.5),
            "proposer_warm_start_completed": bool(warm_start_state.get("completed", 0.0) > 0.5),
            "proposer_warm_start_entropy_mean": float(warm_start_state.get("entropy_mean", 0.0)),
            "proposer_warm_start_exit_streak": float(warm_start_state.get("exit_streak", 0.0)),
            "proposer_hardness_debt": float(hardness_debt_state.get("debt", 0.0)),
            "proposer_hardness_debt_cap_streak": float(hardness_debt_state.get("cap_streak", 0.0)),
            "proposer_hardness_debt_escape_steps_left": float(
                hardness_debt_state.get("escape_steps_left", 0.0)
            ),
            "proposer_hardness_debt_escape_triggered": bool(
                hardness_debt_state.get("escape_triggered", 0.0) > 0.5
            ),
            "proposer_all_easy_streak": int(self._all_easy_streak),
            "proposer_collapse_streak": int(getattr(self, "_proposer_collapse_streak", 0)),
            "proposer_forced_explore_steps_left": int(self._forced_explore_steps_left),
            "proposer_early_failfast_enabled": bool(
                early_failfast_state.get("enabled", 0.0) > 0.5
            ),
            "proposer_early_u_step": int(early_failfast_state.get("u_step", 0.0)),
            "proposer_early_stage1_active": bool(
                early_failfast_state.get("stage1_active", 0.0) > 0.5
            ),
            "proposer_early_stage1_pass": bool(
                early_failfast_state.get("stage1_pass", 0.0) > 0.5
            ),
            "proposer_early_stage2_active": bool(
                early_failfast_state.get("stage2_active", 0.0) > 0.5
            ),
            "proposer_early_stage2_pass": bool(
                early_failfast_state.get("stage2_pass", 0.0) > 0.5
            ),
            "proposer_early_triggered": bool(
                early_failfast_state.get("triggered", 0.0) > 0.5
            ),
            "proposer_non_objective_question": bool(non_objective),
            "solver_scalar_reward": float(solver_scalar_reward),
            "solver_group_rewards": solver_group_rewards,
            "proposer_baseline": float(self.proposer_baseline),
            "solver_baseline": float(self.solver_baseline),
            "policy_updates_enabled": bool(self.policy_updates_enabled),
            "understanding_update_eligible": bool(understanding_update_eligible),
            "understanding_update_skip_reason": str(understanding_update_skip_reason),
            "selected_candidate_acceptable": bool(selected_candidate_acceptable),
            "understanding_update_entropy_threshold": float(entropy_train_threshold),
            "proposer_policy_update_attempted": bool(proposer_update_attempted),
            "proposer_policy_update_applied": bool(proposer_update_applied),
            "proposer_policy_update_stats": proposer_update_stats,
            "solver_policy_update_skipped": bool(solver_skip_update),
            "solver_policy_update_reason": str(solver_update_reason),
            "solver_policy_update_attempts": int(len(solver_update_stats)),
            "solver_policy_update_applied": int(solver_updates_applied_now),
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
        proposer_policy_train_ready = bool(
            self.policy_updates_enabled
            and self.cfg.train_generation_proposer
            and self.proposer_updater is not None
        )
        generator_policy_train_ready = bool(
            self.policy_updates_enabled
            and self.cfg.train_generator
            and self.generator_updater is not None
        )
        update_method = self.cfg.normalized_update_method()
        group_size = 1
        if (proposer_policy_train_ready or generator_policy_train_ready) and update_method == "grpo":
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
        selected_generator_reward = float(
            selected.get("best_total_reward", selected.get("total_reward", selected_reward))
        )
        proposer_baseline_before = float(self.proposer_gen_baseline)
        generator_baseline_before = float(self.generator_baseline)
        baseline_momentum = _clamp01(float(self.cfg.proposer_gen_baseline_momentum))

        proposer_policy_attempted = 0
        proposer_policy_applied = 0
        proposer_policy_update_attempted = False
        proposer_policy_update_applied = False
        proposer_policy_update_reason = "disabled"
        proposer_policy_update_stats: Dict[str, Any] = {}
        proposer_prompt = build_generation_spec_prompt(min_qa_pairs=int(self.cfg.gen_spec_min_qa_pairs))

        if proposer_policy_train_ready:
            proposer_policy_update_attempted = True
            if update_method == "grpo" and len(valid_candidates) > 1:
                group_rewards = [float(c.get("proposer_gen_reward", 0.0)) for c in valid_candidates]
                per_candidate_stats: List[Dict[str, Any]] = []
                for cand, cand_reward in zip(valid_candidates, group_rewards):
                    completion = str(cand.get("_completion_for_update", "")).strip()
                    if not completion:
                        continue
                    stats = self.proposer_updater.step(
                        image=image,
                        prompt=proposer_prompt,
                        completion=completion,
                        reward=float(cand_reward),
                        baseline=proposer_baseline_before,
                        group_rewards=group_rewards,
                    )
                    per_candidate_stats.append(stats)
                    proposer_policy_attempted += 1
                    if not bool(stats.get("skipped", True)):
                        proposer_policy_applied += 1
                proposer_policy_update_applied = bool(proposer_policy_applied > 0)
                proposer_policy_update_reason = "ok" if proposer_policy_update_applied else "all_skipped"
                proposer_policy_update_stats = {
                    "skipped": not proposer_policy_update_applied,
                    "reason": proposer_policy_update_reason,
                    "update_method": "grpo",
                    "group_size": int(len(group_rewards)),
                    "group_reward_mean": float(_mean(group_rewards)),
                    "group_reward_max": float(max(group_rewards)),
                    "group_reward_min": float(min(group_rewards)),
                    "applied_updates": int(proposer_policy_applied),
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
                        prompt=proposer_prompt,
                        completion=completion,
                        reward=float(selected_reward),
                        baseline=proposer_baseline_before,
                    )
                    proposer_policy_attempted = 1
                    proposer_policy_applied = int(not bool(stats.get("skipped", True)))
                    proposer_policy_update_applied = bool(proposer_policy_applied > 0)
                    proposer_policy_update_reason = str(stats.get("reason", "unknown"))
                    proposer_policy_update_stats = stats
                else:
                    proposer_policy_update_applied = False
                    proposer_policy_update_reason = "empty_completion"
                    proposer_policy_update_stats = {"skipped": True, "reason": "empty_completion"}

        def _load_generated_candidate_image(candidate: Dict[str, Any]) -> Optional[Image.Image]:
            generated_image_path = str(candidate.get("generated_image_path", "")).strip()
            if not generated_image_path:
                return None
            p = Path(generated_image_path)
            if not p.exists():
                return None
            try:
                with Image.open(p) as _img:
                    return _img.convert("RGB")
            except Exception:
                return None

        generator_policy_attempted = 0
        generator_policy_applied = 0
        generator_policy_update_attempted = False
        generator_policy_update_applied = False
        generator_policy_update_reason = "disabled"
        generator_policy_update_stats: Dict[str, Any] = {}

        max_generator_candidates = max(
            0,
            int(
                os.environ.get(
                    "BAGEL_GENERATOR_POLICY_MAX_CANDIDATES",
                    os.environ.get("BAGEL_PROPOSER_POLICY_MAX_CANDIDATES", "0"),
                )
                or "0"
            ),
        )
        if generator_policy_train_ready:
            generator_policy_update_attempted = True
            if update_method == "grpo" and len(valid_candidates) > 1:
                group_rewards = [
                    float(c.get("best_total_reward", c.get("total_reward", c.get("proposer_gen_reward", 0.0))))
                    for c in valid_candidates
                ]
                per_candidate_stats: List[Dict[str, Any]] = []
                for cand, cand_reward in zip(valid_candidates, group_rewards):
                    if max_generator_candidates > 0 and generator_policy_attempted >= max_generator_candidates:
                        break
                    generated_img = _load_generated_candidate_image(cand)
                    if generated_img is None:
                        continue
                    gen_prompt = str(
                        cand.get("spec_prompt_for_generation", cand.get("spec_prompt", ""))
                    ).strip()
                    if not gen_prompt:
                        continue
                    stats = self.generator_updater.step(
                        image=generated_img,
                        prompt=gen_prompt,
                        reward=float(cand_reward),
                        baseline=generator_baseline_before,
                        group_rewards=group_rewards,
                    )
                    per_candidate_stats.append(stats)
                    generator_policy_attempted += 1
                    if not bool(stats.get("skipped", True)):
                        generator_policy_applied += 1
                generator_policy_update_applied = bool(generator_policy_applied > 0)
                generator_policy_update_reason = "ok" if generator_policy_update_applied else "all_skipped"
                generator_policy_update_stats = {
                    "skipped": not generator_policy_update_applied,
                    "reason": generator_policy_update_reason,
                    "update_method": "grpo",
                    "group_size": int(len(group_rewards)),
                    "group_reward_mean": float(_mean(group_rewards)),
                    "group_reward_max": float(max(group_rewards)),
                    "group_reward_min": float(min(group_rewards)),
                    "applied_updates": int(generator_policy_applied),
                    "selected_candidate_index": int(selected_idx),
                    "mse_loss_mean": float(
                        _mean(
                            [
                                float(s.get("mse_loss", s.get("ce_loss", 0.0)))
                                for s in per_candidate_stats
                                if not bool(s.get("skipped", True))
                            ]
                        )
                    ),
                }
            else:
                generated_img = _load_generated_candidate_image(selected)
                gen_prompt = str(
                    selected.get("spec_prompt_for_generation", selected.get("spec_prompt", ""))
                ).strip()
                if generated_img is not None and gen_prompt:
                    stats = self.generator_updater.step(
                        image=generated_img,
                        prompt=gen_prompt,
                        reward=float(selected_generator_reward),
                        baseline=generator_baseline_before,
                    )
                    generator_policy_attempted = 1
                    generator_policy_applied = int(not bool(stats.get("skipped", True)))
                    generator_policy_update_applied = bool(generator_policy_applied > 0)
                    generator_policy_update_reason = str(stats.get("reason", "unknown"))
                    generator_policy_update_stats = stats
                else:
                    generator_policy_update_applied = False
                    generator_policy_update_reason = "missing_generated_image_or_prompt"
                    generator_policy_update_stats = {
                        "skipped": True,
                        "reason": "missing_generated_image_or_prompt",
                    }

        gen_solver_update_attempted = 0
        gen_solver_update_applied = 0
        gen_solver_update_stats: List[Dict[str, Any]] = []
        gen_solver_reward_values: List[float] = []
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
        gen_solver_enabled = bool(
            getattr(self.cfg, "gen_step_solver_update_enabled", False)
            and self.policy_updates_enabled
            and self.cfg.train_solver
            and self.solver_updater is not None
        )
        if gen_solver_enabled:
            generated_image_path = str(selected.get("generated_image_path", "")).strip()
            generated_img: Optional[Image.Image] = None
            if generated_image_path and Path(generated_image_path).exists():
                try:
                    with Image.open(generated_image_path) as _img:
                        generated_img = _img.convert("RGB")
                except Exception:
                    generated_img = None
            if generated_img is not None:
                for qa in selected.get("qa_logs", []):
                    if (
                        max_gen_solver_updates > 0
                        and int(gen_solver_update_attempted) >= int(max_gen_solver_updates)
                    ):
                        break
                    if not isinstance(qa, dict) or str(qa.get("status", "")) != "ok":
                        continue
                    qa_question = str(qa.get("question", "")).strip()
                    if not qa_question:
                        continue
                    expected = normalize_answer(str(qa.get("expected_answer", "")).strip())
                    parsed_samples: List[tuple[str, str]] = []
                    solver_outputs = qa.get("solver_outputs_raw", [])
                    if isinstance(solver_outputs, list) and solver_outputs:
                        for raw in solver_outputs:
                            raw_text = str(raw or "")
                            ans = normalize_answer(parse_answer(raw_text))
                            if ans:
                                parsed_samples.append((raw_text, ans))
                    if not parsed_samples:
                        remaining = (
                            int(max_gen_solver_updates) - int(gen_solver_update_attempted)
                            if max_gen_solver_updates > 0
                            else len(solver_temps)
                        )
                        if remaining <= 0:
                            break
                        for temp in solver_temps[: max(1, int(remaining))]:
                            out = self.adapter.solve_question(
                                image=generated_img,
                                question=qa_question,
                                max_new_tokens=self.cfg.max_new_tokens_solver,
                                temperature=float(temp),
                                do_sample=True,
                            )
                            raw_text = str(out.text or "")
                            ans = normalize_answer(parse_answer(raw_text))
                            if ans:
                                parsed_samples.append((raw_text, ans))
                    if not parsed_samples:
                        continue
                    group_rewards = [
                        float(answer_match_score(ans, expected)) for _, ans in parsed_samples
                    ]
                    solver_prompt = build_solver_prompt(qa_question)
                    for idx, (sample_raw, _) in enumerate(parsed_samples):
                        if (
                            max_gen_solver_updates > 0
                            and int(gen_solver_update_attempted) >= int(max_gen_solver_updates)
                        ):
                            break
                        sample_reward = (
                            float(group_rewards[idx]) if idx < len(group_rewards) else 0.0
                        )
                        stats = self.solver_updater.step(
                            image=generated_img,
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
            if gen_solver_reward_values:
                self.solver_baseline = (
                    _clamp01(float(self.cfg.baseline_momentum)) * self.solver_baseline
                    + (1.0 - _clamp01(float(self.cfg.baseline_momentum)))
                    * float(_mean(gen_solver_reward_values))
                )

        self.proposer_gen_baseline = (
            baseline_momentum * self.proposer_gen_baseline
            + (1.0 - baseline_momentum) * float(selected_reward)
        )
        self.generator_baseline = (
            baseline_momentum * self.generator_baseline
            + (1.0 - baseline_momentum) * float(selected_generator_reward)
        )

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
        policy_update_stats: Dict[str, Any] = {
            "proposer": proposer_policy_update_stats,
            "generator": generator_policy_update_stats,
        }
        generation_policy_attempted_count = int(proposer_policy_attempted + generator_policy_attempted)
        generation_policy_applied_count = int(proposer_policy_applied + generator_policy_applied)

        rec = dict(selected)
        rec["phase"] = "generation"
        rec["generation_candidate_image_path"] = str(rec.get("image_path", image_path))
        rec["source_image_path"] = str(image_path)
        rec["image_path"] = str(image_path)
        rec["generation_candidate_group_size"] = int(group_size)
        rec["generation_candidate_valid_count"] = int(len(valid_candidates))
        rec["generation_candidate_rewards"] = [
            float(c.get("best_total_reward", c.get("total_reward", c.get("proposer_gen_reward", 0.0))))
            for c in valid_candidates
        ]
        rec["generation_candidate_proposer_rewards"] = [
            float(c.get("proposer_gen_reward", 0.0)) for c in valid_candidates
        ]
        rec["generation_candidate_statuses"] = [str(c.get("status", "skipped")) for c in candidates]
        rec["generation_candidate_temps"] = [float(c.get("spec_temperature_used", base_temp)) for c in candidates]
        rec["generation_selected_candidate_index"] = int(selected_idx)
        rec["proposer_gen_baseline_before"] = float(proposer_baseline_before)
        rec["proposer_gen_baseline_after"] = float(self.proposer_gen_baseline)
        rec["proposer_gen_advantage"] = float(selected_reward - proposer_baseline_before)
        rec["generator_baseline_before"] = float(generator_baseline_before)
        rec["generator_baseline_after"] = float(self.generator_baseline)
        rec["generator_advantage"] = float(selected_generator_reward - generator_baseline_before)
        rec["policy_update_attempted"] = bool(policy_update_attempted)
        rec["policy_update_applied"] = bool(policy_update_applied)
        rec["policy_update_reason"] = str(policy_update_reason)
        rec["policy_update_stats"] = policy_update_stats
        rec["policy_updates_attempted_count"] = int(generation_policy_attempted_count)
        rec["policy_updates_applied_count"] = int(generation_policy_applied_count)
        rec["proposer_policy_update_attempted"] = bool(proposer_policy_update_attempted)
        rec["proposer_policy_update_attempts"] = int(proposer_policy_attempted)
        rec["proposer_policy_update_applied"] = int(proposer_policy_applied)
        rec["proposer_policy_update_reason"] = str(proposer_policy_update_reason)
        rec["proposer_policy_update_stats"] = proposer_policy_update_stats
        rec["generator_policy_update_attempted"] = bool(generator_policy_update_attempted)
        rec["generator_policy_update_attempts"] = int(generator_policy_attempted)
        rec["generator_policy_update_applied"] = int(generator_policy_applied)
        rec["generator_policy_update_reason"] = str(generator_policy_update_reason)
        rec["generator_policy_update_stats"] = generator_policy_update_stats
        rec["gen_solver_policy_update_enabled"] = bool(gen_solver_enabled)
        rec["gen_solver_policy_update_budget"] = int(max_gen_solver_updates)
        rec["gen_solver_policy_update_attempts"] = int(gen_solver_update_attempted)
        rec["gen_solver_policy_update_applied"] = int(gen_solver_update_applied)
        rec["gen_solver_policy_update_ce_mean"] = _mean(
            [float(s.get("ce_loss", 0.0)) for s in gen_solver_update_stats if not s.get("skipped", True)]
        )

        rec.pop("_completion_for_update", None)
        for cand in candidates:
            cand.pop("_completion_for_update", None)

        self._store_generated_to_folder(step=step, rec=rec)
        self._maybe_add_to_replay_buffer(step=step, rec=rec)
        return {
            "record": rec,
            "valid": 1,
            "skipped": 0,
            "reward_sum": float(selected_generator_reward),
            "entropy_sum": float(rec.get("mean_entropy_nats", 0.0)),
            "quality_sum": float(rec.get("quality_component", 0.0)),
            "policy_attempted": int(generation_policy_attempted_count + gen_solver_update_attempted),
            "policy_applied": int(generation_policy_applied_count + gen_solver_update_applied),
        }

    def _checkpoint_extra_state(self) -> Dict[str, object]:
        state = dict(super()._checkpoint_extra_state())
        state.update(
            {
                "unified_gen_reward_ema": float(self._gen_reward_ema),
                "unified_gen_reward_ema_initialized": bool(self._gen_reward_ema_initialized),
                "unified_difficulty_window": list(self._difficulty_window),
                "unified_entropy_easy_window": list(self._entropy_easy_window),
                "unified_warm_start_entropy_window": list(self._warm_start_entropy_window),
                "unified_warm_start_exit_streak": int(self._warm_start_exit_streak),
                "unified_warm_start_completed": bool(self._warm_start_completed),
                "unified_hardness_debt": float(self._hardness_debt),
                "unified_hardness_debt_cap_streak": int(self._hardness_debt_cap_streak),
                "unified_hardness_debt_escape_steps_left": int(self._hardness_debt_escape_steps_left),
                "unified_forced_explore_steps_left": int(self._forced_explore_steps_left),
                "unified_all_easy_streak": int(self._all_easy_streak),
                "unified_understanding_u_step": int(self._understanding_u_step),
                "unified_proposer_collapse_streak": int(self._proposer_collapse_streak),
                "unified_strategy_hist": dict(self._strategy_hist),
                "unified_contrastive_pos_replay": [sorted(list(x)) for x in self._contrastive_pos_replay],
                "unified_contrastive_neg_replay": [sorted(list(x)) for x in self._contrastive_neg_replay],
                "unified_candidate_non_easy_window": list(self._candidate_non_easy_window),
                "unified_all_easy_group_window": list(self._all_easy_group_window),
                "unified_proposer_reward_clipped_window": list(self._proposer_reward_clipped_window),
                "unified_selected_non_easy_window": list(self._selected_non_easy_window),
                "unified_solver_update_applied_window": list(self._solver_update_applied_window),
                "unified_ste_window": [float(v) for v in self._ste_window],
            }
        )
        if self.replay_buffer is not None:
            state["unified_replay_buffer"] = self.replay_buffer.state_dict()
        return state

    def _load_checkpoint_extra_state(self, state: Dict[str, object]) -> None:
        super()._load_checkpoint_extra_state(state)
        if not isinstance(state, dict):
            return

        def _restore_float_deque(name: str, payload_key: str) -> None:
            vals = state.get(payload_key)
            dq = getattr(self, name, None)
            if not isinstance(vals, list) or dq is None:
                return
            dq.clear()
            max_keep = int(getattr(dq, "maxlen", 0) or len(vals))
            for v in vals[-max_keep:]:
                try:
                    dq.append(float(v))
                except Exception:
                    continue

        def _restore_str_deque(name: str, payload_key: str) -> None:
            vals = state.get(payload_key)
            dq = getattr(self, name, None)
            if not isinstance(vals, list) or dq is None:
                return
            dq.clear()
            max_keep = int(getattr(dq, "maxlen", 0) or len(vals))
            for v in vals[-max_keep:]:
                s = str(v).strip().lower()
                if s in {"easy", "medium", "hard"}:
                    dq.append(s)

        _restore_str_deque("_difficulty_window", "unified_difficulty_window")
        _restore_float_deque("_entropy_easy_window", "unified_entropy_easy_window")
        _restore_float_deque("_warm_start_entropy_window", "unified_warm_start_entropy_window")
        _restore_float_deque("_candidate_non_easy_window", "unified_candidate_non_easy_window")
        _restore_float_deque("_all_easy_group_window", "unified_all_easy_group_window")
        _restore_float_deque(
            "_proposer_reward_clipped_window",
            "unified_proposer_reward_clipped_window",
        )
        _restore_float_deque("_selected_non_easy_window", "unified_selected_non_easy_window")
        _restore_float_deque("_solver_update_applied_window", "unified_solver_update_applied_window")
        ste_vals = state.get("unified_ste_window")
        if isinstance(ste_vals, list):
            self._ste_window = []
            max_keep = max(1, int(getattr(self, "_ste_window_size", len(ste_vals) or 1)))
            for v in ste_vals[-max_keep:]:
                try:
                    self._ste_window.append(float(v))
                except Exception:
                    continue

        self._gen_reward_ema = float(state.get("unified_gen_reward_ema", getattr(self, "_gen_reward_ema", 0.0)))
        self._gen_reward_ema_initialized = bool(
            state.get(
                "unified_gen_reward_ema_initialized",
                getattr(self, "_gen_reward_ema_initialized", False),
            )
        )

        self._warm_start_exit_streak = int(
            state.get("unified_warm_start_exit_streak", getattr(self, "_warm_start_exit_streak", 0))
        )
        self._warm_start_completed = bool(
            state.get("unified_warm_start_completed", getattr(self, "_warm_start_completed", False))
        )
        self._hardness_debt = float(
            state.get("unified_hardness_debt", getattr(self, "_hardness_debt", 0.0))
        )
        self._hardness_debt_cap_streak = int(
            state.get("unified_hardness_debt_cap_streak", getattr(self, "_hardness_debt_cap_streak", 0))
        )
        self._hardness_debt_escape_steps_left = int(
            state.get(
                "unified_hardness_debt_escape_steps_left",
                getattr(self, "_hardness_debt_escape_steps_left", 0),
            )
        )
        self._forced_explore_steps_left = int(
            state.get("unified_forced_explore_steps_left", getattr(self, "_forced_explore_steps_left", 0))
        )
        self._all_easy_streak = int(state.get("unified_all_easy_streak", getattr(self, "_all_easy_streak", 0)))
        self._understanding_u_step = int(
            state.get("unified_understanding_u_step", getattr(self, "_understanding_u_step", 0))
        )
        self._proposer_collapse_streak = int(
            state.get("unified_proposer_collapse_streak", getattr(self, "_proposer_collapse_streak", 0))
        )

        strategy_hist = state.get("unified_strategy_hist")
        if isinstance(strategy_hist, dict):
            self._strategy_hist.clear()
            for k, v in strategy_hist.items():
                key = str(k).strip().upper()
                if not key:
                    continue
                try:
                    self._strategy_hist[key] = int(v)
                except Exception:
                    continue

        def _restore_set_deque(name: str, payload_key: str) -> None:
            vals = state.get(payload_key)
            dq = getattr(self, name, None)
            if not isinstance(vals, list) or dq is None:
                return
            dq.clear()
            max_keep = int(getattr(dq, "maxlen", 0) or len(vals))
            for item in vals[-max_keep:]:
                if not isinstance(item, list):
                    continue
                s = {str(t).strip().lower() for t in item if str(t).strip()}
                if s:
                    dq.append(s)

        _restore_set_deque("_contrastive_pos_replay", "unified_contrastive_pos_replay")
        _restore_set_deque("_contrastive_neg_replay", "unified_contrastive_neg_replay")

        replay_state = state.get("unified_replay_buffer")
        if self.replay_buffer is not None and isinstance(replay_state, dict):
            try:
                restored = self.replay_buffer.load_state_dict(replay_state)
                if restored <= 0:
                    self._refresh_generated_mix_cache(step=max(0, int(self.start_step)), force=True)
            except Exception:
                pass

        ste_vals = state.get("unified_ste_window")
        if isinstance(ste_vals, list):
            max_keep = max(8, int(getattr(self.cfg, "solver_token_entropy_window_size", 128)))
            cleaned: List[float] = []
            for v in ste_vals[-max_keep:]:
                try:
                    cleaned.append(float(v))
                except Exception:
                    continue
            self._ste_window = cleaned

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
                "generator_baseline": float(self.generator_baseline),
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
                self._understanding_u_step += 1
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
                        u_step=int(self._understanding_u_step),
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

            self._sync_distributed_step_state()
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
            "generator_baseline_final": float(self.generator_baseline),
            "generator_reward_ema": float(self._gen_reward_ema) if self._gen_reward_ema_initialized else 0.0,
            "gen_mix_source_mode": str(self._gen_mix_source_mode),
            "generated_mix_dir": str(self._generated_mix_dir),
            "optimizer_flush_steps": int(flushed_optim_steps),
            "last_checkpoint_path": str(self.last_checkpoint_path),
            "last_lora_checkpoint_dir": str(self.last_lora_checkpoint_dir),
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
