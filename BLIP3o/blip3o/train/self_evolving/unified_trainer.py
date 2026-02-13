"""
Unified (alternating understanding + generation) self-evolving trainer.

Ported from self_evolving/experiments/generation.py (UnifiedSelfEvolvingTrainer).
Extends GenerationSelfEvolvingTrainer with an interleaved understanding phase.
"""

import gc
import math
import random
import re
import time
import traceback
from collections import deque
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from PIL import Image

from .config import UnifiedSelfEvolvingConfig
from .generation_helpers import GenerationSpec
from .generation_trainer import GenerationSelfEvolvingTrainer
from .prompts import build_proposer_hardening_prompt, build_proposer_prompt, build_solver_prompt
from .replay_buffer import ReplayBuffer
from .utils import (
    HAS_WANDB,
    _json_dump,
    _parse_answer,
    _parse_first_question,
    gaussian_reward,
    majority_vote,
    normalize_answer,
    pre_answer_word_count,
    shannon_entropy_nats,
    strip_tags,
)

_SUBJECTIVE_QUESTION_RE = re.compile(
    r"\b(why|might|could|likely|opinion|feel|emotion|think|believe|suggest|imply|purpose|reason)\b",
    flags=re.IGNORECASE,
)
_OBJECTIVE_QUESTION_RE = re.compile(
    r"\b("
    r"how many|count|number of|what (?:is|are|was|were)|which|compare|difference|ratio|"
    r"total|sum|percent|percentage|value|label|name|color|shape|position|left|right|top|bottom|"
    r"highest|lowest|maximum|minimum"
    r")\b",
    flags=re.IGNORECASE,
)


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    qq = max(0.0, min(1.0, float(q)))
    pos = qq * float(len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    w = pos - float(lo)
    return float(values[lo] * (1.0 - w) + values[hi] * w)


class UnifiedSelfEvolvingTrainer(GenerationSelfEvolvingTrainer):
    """
    Unified self-evolving trainer: alternates understanding and generation steps
    within each cycle.

    Extends GenerationSelfEvolvingTrainer with an interleaved understanding phase.
    """

    def __init__(self, config: UnifiedSelfEvolvingConfig):
        if config.enable_solver_updates and config.solver_update_freq <= 0:
            config.solver_update_freq = max(1, config.synthetic_solver_update_freq)
        # GenerationSelfEvolvingTrainer.__init__ invokes self._maybe_resume_state().
        # Initialize adaptive windows early so resume can safely restore them.
        self.cfg = config
        self._init_adaptive_windows()
        super().__init__(config)
        self.ucfg = config

        # ---- Self-evolving feedback loop state ---- #
        # Always create replay buffer — generated images mix into understanding
        _buf_size = getattr(config, "replay_buffer_size", 1000)
        _buf_min_r = getattr(config, "replay_min_reward", 0.5)
        _buf_stale = getattr(config, "replay_max_staleness", 500)
        self.replay_buffer = ReplayBuffer(
            max_size=_buf_size,
            min_reward=_buf_min_r,
            max_staleness=_buf_stale,
        )

        # Generator reward EMA for monitoring
        self._gen_reward_ema = float(getattr(self, "_gen_reward_ema", 0.0))
        self._gen_reward_ema_initialized = bool(
            getattr(self, "_gen_reward_ema_initialized", False)
        )

    def _phase_local_step_index(self, step: int, phase: str) -> int:
        cycle = max(1, self.cfg.understanding_steps_per_cycle + self.cfg.generation_steps_per_cycle)
        cycle_idx = (step - 1) // cycle
        phase_idx = (step - 1) % cycle
        if phase == "understanding":
            if phase_idx >= self.cfg.understanding_steps_per_cycle:
                return 0
            return cycle_idx * self.cfg.understanding_steps_per_cycle + phase_idx + 1
        if phase == "generation":
            if phase_idx < self.cfg.understanding_steps_per_cycle:
                return 0
            gen_pos = phase_idx - self.cfg.understanding_steps_per_cycle
            return cycle_idx * self.cfg.generation_steps_per_cycle + gen_pos + 1
        raise ValueError(f"Unknown phase: {phase!r}")

    def _is_proposer_update_due(self, step: int, phase: str) -> bool:
        freq = int(getattr(self.cfg, "proposer_update_freq", 0) or 0)
        if freq <= 0:
            return False
        local_idx = self._phase_local_step_index(step, phase)
        if local_idx <= 0:
            return False
        return (local_idx % freq) == 0

    def _solver_top_p_schedule(self) -> List[float]:
        """Vary top_p across solver samples to inject diversity.

        Ranges from solver_top_p_min (default 0.5) to solver_top_p_max (default 1.0).
        Lower top_p forces the model to commit to fewer tokens, producing more
        varied short answers that break unanimous voting on trivially easy questions.
        """
        n = max(1, int(self.cfg.num_solver_samples))
        top_p_min = float(getattr(self.cfg, "solver_top_p_min", 0.5))
        top_p_max = float(getattr(self.cfg, "solver_top_p_max", 1.0))
        if n <= 1:
            return [top_p_max]
        if abs(top_p_max - top_p_min) < 1e-8:
            return [top_p_min] * n
        return [top_p_min + (top_p_max - top_p_min) * (float(i) / float(n - 1)) for i in range(n)]

    def _solver_temperature_schedule(self) -> List[float]:
        n = max(1, int(self.cfg.num_solver_samples))
        base = float(self.cfg.temp)
        if n <= 1 or not bool(getattr(self.cfg, "solver_use_temperature_mix", True)):
            return [base] * n
        tmin = float(getattr(self.cfg, "solver_temp_min", base))
        tmax = float(getattr(self.cfg, "solver_temp_max", base))
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        if abs(tmax - tmin) < 1e-8:
            return [tmin] * n
        return [tmin + (tmax - tmin) * (float(i) / float(n - 1)) for i in range(n)]

    def _update_proposer_entropy_target(self, entropy_nats: float) -> float:
        if not bool(getattr(self.cfg, "adaptive_prop_entropy_target", True)):
            return float(self.cfg.prop_entropy_mu)
        anchor = self._dist_mean(float(entropy_nats))
        prev = float(getattr(self, "proposer_entropy_mu_ema", self.cfg.prop_entropy_mu))
        momentum = float(getattr(self.cfg, "prop_entropy_ema_momentum", 0.95))
        momentum = max(0.0, min(0.9999, momentum))
        ema = momentum * prev + (1.0 - momentum) * anchor
        mu_min = float(getattr(self.cfg, "prop_entropy_mu_min", 0.0))
        mu_max = float(getattr(self.cfg, "prop_entropy_mu_max", 10.0))
        if mu_min > mu_max:
            mu_min, mu_max = mu_max, mu_min
        ema = max(mu_min, min(mu_max, ema))
        self.proposer_entropy_mu_ema = float(ema)
        return float(ema)

    def _is_objective_question(self, question: str) -> bool:
        q = str(question or "").strip()
        if not q:
            return False
        if _SUBJECTIVE_QUESTION_RE.search(q):
            return False
        return bool(_OBJECTIVE_QUESTION_RE.search(q))

    def _init_adaptive_windows(self):
        ent_window_size = max(8, int(getattr(self.cfg, "entropy_iqr_window_size", 256)))
        diff_window_size = max(8, int(getattr(self.cfg, "difficulty_sampler_window_size", 256)))
        self._entropy_window = deque(maxlen=ent_window_size)
        self._difficulty_window = deque(maxlen=diff_window_size)

    def _entropy_iqr_filter_state(self) -> Dict[str, float]:
        static_threshold = float(getattr(self.cfg, "sc_entropy_min", 0.15))
        enabled = bool(getattr(self.cfg, "entropy_iqr_filter_enabled", True))
        min_samples = max(4, int(getattr(self.cfg, "entropy_iqr_min_samples", 32)))
        history = [float(x) for x in self._entropy_window]
        history_size = len(history)
        state: Dict[str, float] = {
            "enabled": 1.0 if enabled else 0.0,
            "active": 0.0,
            "history_size": float(history_size),
            "min_samples": float(min_samples),
            "threshold": float(static_threshold),
            "q1": float(static_threshold),
            "q3": float(static_threshold),
            "iqr": 0.0,
        }
        if (not enabled) or history_size < min_samples:
            return state

        values = sorted(history)
        q = float(getattr(self.cfg, "entropy_iqr_easy_quantile", 0.25))
        q = max(0.01, min(0.49, q))
        q1 = _quantile(values, q)
        q3 = _quantile(values, 1.0 - q)
        iqr = max(0.0, q3 - q1)
        coef = float(getattr(self.cfg, "entropy_iqr_easy_iqr_coef", 0.25))
        threshold = q1 + coef * iqr
        thr_min = float(getattr(self.cfg, "entropy_iqr_min_threshold", 0.02))
        thr_max = float(
            getattr(self.cfg, "entropy_iqr_max_threshold", getattr(self.cfg, "sc_entropy_max", 1.2))
        )
        if thr_min > thr_max:
            thr_min, thr_max = thr_max, thr_min
        threshold = max(thr_min, min(thr_max, threshold))
        state.update(
            {
                "active": 1.0,
                "threshold": float(threshold),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
            }
        )
        return state

    def _difficulty_bucket(
        self,
        entropy_nats: float,
        margin: float,
        majority_fraction: float,
        easy_entropy_threshold: float,
    ) -> str:
        easy_majority = float(getattr(self.cfg, "easy_update_majority_frac_threshold", 0.95))
        hard_min_entropy = float(getattr(self.cfg, "difficulty_hard_min_entropy", 0.90))
        hard_max_margin = float(getattr(self.cfg, "difficulty_hard_max_margin", 0.35))
        if entropy_nats <= easy_entropy_threshold or majority_fraction >= easy_majority:
            return "easy"
        if entropy_nats >= hard_min_entropy and margin <= hard_max_margin:
            return "hard"
        return "medium"

    def _difficulty_target_weights(self) -> Dict[str, float]:
        w_easy = max(0.0, float(getattr(self.cfg, "difficulty_target_easy", 0.20)))
        w_medium = max(0.0, float(getattr(self.cfg, "difficulty_target_medium", 0.60)))
        w_hard = max(0.0, float(getattr(self.cfg, "difficulty_target_hard", 0.20)))
        total = w_easy + w_medium + w_hard
        if total <= 1e-8:
            return {"easy": 0.2, "medium": 0.6, "hard": 0.2}
        return {
            "easy": w_easy / total,
            "medium": w_medium / total,
            "hard": w_hard / total,
        }

    def _sample_bucket(self, weights: Dict[str, float]) -> str:
        r = random.random()
        c = 0.0
        for key in ("easy", "medium", "hard"):
            c += float(weights.get(key, 0.0))
            if r <= c:
                return key
        return "medium"

    def _choose_difficulty_target(self) -> Dict[str, object]:
        enabled = bool(getattr(self.cfg, "difficulty_sampler_enabled", True))
        min_samples = max(4, int(getattr(self.cfg, "difficulty_sampler_min_samples", 32)))
        target = self._difficulty_target_weights()
        history = list(self._difficulty_window)
        history_size = len(history)
        mode = "target"
        observed = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
        weights_for_sampling = dict(target)
        if enabled and history_size >= min_samples:
            for b in history:
                if b in observed:
                    observed[b] += 1.0
            for key in observed:
                observed[key] /= float(history_size)
            deficits = {
                key: max(0.0, target[key] - observed[key]) for key in ("easy", "medium", "hard")
            }
            deficit_total = deficits["easy"] + deficits["medium"] + deficits["hard"]
            if deficit_total > 1e-8:
                weights_for_sampling = {
                    key: deficits[key] / deficit_total for key in ("easy", "medium", "hard")
                }
                mode = "deficit"
            else:
                mode = "target_fallback"
        elif not enabled:
            mode = "disabled"

        desired_bucket = self._sample_bucket(weights_for_sampling) if enabled else "medium"
        return {
            "enabled": enabled,
            "desired_bucket": desired_bucket,
            "mode": mode,
            "history_size": history_size,
            "min_samples": min_samples,
            "target_weights": target,
            "observed_weights": observed,
            "sampling_weights": weights_for_sampling,
        }

    def _understanding_step(self, step: int, image: Image.Image, meta: Dict) -> Dict[str, object]:
        step_t0 = time.perf_counter()
        entropy_min = float(getattr(self.cfg, "sc_entropy_min", 0.15))
        entropy_max = float(getattr(self.cfg, "sc_entropy_max", 1.2))
        if entropy_min > entropy_max:
            entropy_min, entropy_max = entropy_max, entropy_min
        margin_max = float(getattr(self.cfg, "sc_margin_max", 0.9))
        require_objective = bool(getattr(self.cfg, "proposer_require_objective", True))
        harden_on_easy = bool(getattr(self.cfg, "proposer_hardening_on_easy", True))
        max_hardening_retries = max(
            0, int(getattr(self.cfg, "proposer_hardening_max_retries", 0))
        )
        entropy_iqr_state = self._entropy_iqr_filter_state()
        entropy_easy_threshold = float(entropy_iqr_state.get("threshold", entropy_min))
        entropy_iqr_filter_active = bool(entropy_iqr_state.get("active", 0.0) > 0.5)
        difficulty_target_state = self._choose_difficulty_target()
        difficulty_sampler_enabled = bool(difficulty_target_state.get("enabled", False))
        desired_difficulty_bucket = str(difficulty_target_state.get("desired_bucket", "medium"))
        difficulty_sampler_mode = str(difficulty_target_state.get("mode", "target"))
        difficulty_sampler_max_retries = max(
            0, int(getattr(self.cfg, "difficulty_sampler_max_retries", 1))
        )
        solver_temperatures = self._solver_temperature_schedule()
        solver_top_ps = self._solver_top_p_schedule()

        proposer_prompt = build_proposer_prompt()
        proposer_out = ""
        parsed_question = ""
        question = ""
        fallback_used = False
        proposer_rationale = ""
        proposer_non_objective_question = False
        hardening_retries_used = 0
        hardening_reason = ""
        difficulty_sampler_retries_used = 0
        difficulty_bucket_observed = "unknown"

        solver_prompt = ""
        solver_outputs: List[str] = []
        solver_answers_raw: List[str] = []
        solver_answers_norm: List[str] = []
        pre_words: List[int] = []

        while True:
            proposer_out = self._generate(
                image=image,
                prompt=proposer_prompt,
                adapter_name="proposer" if self.cfg.use_lora else None,
                max_new_tokens=self.cfg.max_new_tokens_proposer,
                temperature=self.cfg.temp,
                top_p=self.cfg.top_p,
            )
            parsed_question = _parse_first_question(proposer_out).replace("\n", " ").strip()
            question = parsed_question or "What is the most salient object in the image?"
            fallback_used = not bool(parsed_question)
            proposer_rationale = strip_tags(proposer_out, "rationale")
            proposer_non_objective_question = bool(
                require_objective and (not self._is_objective_question(question))
            )

            solver_prompt = build_solver_prompt(question)
            solver_outputs = []
            solver_answers_raw = []
            solver_answers_norm = []
            pre_words = []

            for sample_idx in range(self.cfg.num_solver_samples):
                solver_temp = (
                    float(solver_temperatures[sample_idx])
                    if sample_idx < len(solver_temperatures)
                    else float(self.cfg.temp)
                )
                solver_top_p = (
                    float(solver_top_ps[sample_idx])
                    if sample_idx < len(solver_top_ps)
                    else float(self.cfg.top_p)
                )
                solver_out = self._generate(
                    image=image,
                    prompt=solver_prompt,
                    adapter_name="default" if self.cfg.use_lora else None,
                    max_new_tokens=self.cfg.max_new_tokens_solver,
                    temperature=solver_temp,
                    top_p=solver_top_p,
                )
                answer_raw = _parse_answer(solver_out)
                solver_outputs.append(solver_out)
                solver_answers_raw.append(answer_raw)
                solver_answers_norm.append(normalize_answer(answer_raw))
                pre_words.append(pre_answer_word_count(solver_out))

            _tmp_hist: Dict[str, int] = {}
            for ans in solver_answers_norm:
                _tmp_hist[ans] = _tmp_hist.get(ans, 0) + 1
            _tmp_probs = [
                count / float(self.cfg.num_solver_samples)
                for count in _tmp_hist.values()
            ]
            _tmp_entropy = shannon_entropy_nats(_tmp_probs)
            _tmp_sorted = sorted(_tmp_probs, reverse=True)
            _tmp_p1 = float(_tmp_sorted[0]) if _tmp_sorted else 0.0
            _tmp_p2 = float(_tmp_sorted[1]) if len(_tmp_sorted) > 1 else 0.0
            _tmp_margin = max(0.0, _tmp_p1 - _tmp_p2)
            _tmp_maj_frac = _tmp_p1
            difficulty_bucket_observed = self._difficulty_bucket(
                _tmp_entropy,
                _tmp_margin,
                _tmp_maj_frac,
                entropy_easy_threshold,
            )
            easy_for_hardening = bool(
                (_tmp_entropy < entropy_easy_threshold) and (_tmp_margin > margin_max)
            )
            retry_due_to_easy = bool(harden_on_easy and easy_for_hardening)
            retry_due_to_non_objective = bool(proposer_non_objective_question)
            retry_due_to_bucket = bool(
                difficulty_sampler_enabled
                and (difficulty_bucket_observed != desired_difficulty_bucket)
            )

            can_retry_hardening = bool(
                hardening_retries_used < max_hardening_retries
                and (retry_due_to_easy or retry_due_to_non_objective)
            )
            can_retry_bucket = bool(
                difficulty_sampler_retries_used < difficulty_sampler_max_retries
                and retry_due_to_bucket
            )
            if can_retry_hardening or can_retry_bucket:
                if retry_due_to_non_objective and retry_due_to_easy:
                    hardening_reason = (
                        "question was subjective/open-ended and produced unanimous easy solver answers"
                    )
                elif retry_due_to_non_objective:
                    hardening_reason = (
                        "question was subjective/open-ended, not objectively verifiable"
                    )
                elif retry_due_to_easy:
                    hardening_reason = (
                        "question was too easy; solver consensus was near-unanimous"
                    )
                else:
                    hardening_reason = (
                        "question did not match requested difficulty bucket "
                        f"(target={desired_difficulty_bucket}, observed={difficulty_bucket_observed})"
                    )
                if can_retry_hardening:
                    hardening_retries_used += 1
                if can_retry_bucket:
                    difficulty_sampler_retries_used += 1
                proposer_prompt = build_proposer_hardening_prompt(question, hardening_reason)
                continue
            break

        maj_answer, maj_count = majority_vote(solver_answers_norm)
        maj_frac = maj_count / float(self.cfg.num_solver_samples)
        hist: Dict[str, int] = {}
        for ans in solver_answers_norm:
            hist[ans] = hist.get(ans, 0) + 1
        probs = [count / float(self.cfg.num_solver_samples) for count in hist.values()]
        entropy_nats = shannon_entropy_nats(probs)

        sorted_probs = sorted(probs, reverse=True)
        p1 = float(sorted_probs[0]) if sorted_probs else 0.0
        p2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = max(0.0, p1 - p2)
        ratio_min = float(getattr(self.cfg, "sc_informative_ratio_min", 0.25))
        ratio_min = max(0.0, min(1.0, ratio_min))
        neg_weight = float(getattr(self.cfg, "sc_negative_weight", 0.25))
        # Informativeness score encourages moderate disagreement and
        # penalizes collapsed unanimity.
        entropy_span = max(1e-6, entropy_max - entropy_min)
        entropy_mid = 0.5 * (entropy_min + entropy_max)
        entropy_sigma = max(1e-6, 0.5 * entropy_span)
        entropy_band_score = math.exp(
            -((entropy_nats - entropy_mid) ** 2) / (2.0 * (entropy_sigma ** 2))
        )
        margin_damp_score = max(0.0, 1.0 - (margin / max(1e-6, margin_max)))
        local_info_score = max(
            0.0,
            min(1.0, 0.5 * entropy_band_score + 0.5 * margin_damp_score),
        )
        solver_informative_local = bool(
            (entropy_min <= entropy_nats <= entropy_max) or (margin <= margin_max)
        )
        informative_ratio = self._dist_mean(1.0 if solver_informative_local else 0.0)
        solver_informative_any = informative_ratio > 0.0
        solver_informative_all = informative_ratio >= (1.0 - 1e-8)
        # NOTE: Updates are per-rank on per-rank images. Use local
        # informativeness for gating; keep global ratio for logging.
        solver_informative_gate = solver_informative_local
        solver_informative_gate_global = informative_ratio >= ratio_min

        sc_signal = max(1e-4, local_info_score)
        # Penalize unanimous, low-entropy, high-margin (trivially easy) cases.
        easy_solver_case = bool((entropy_nats < entropy_easy_threshold) and (margin > margin_max))
        difficulty_bucket_observed = self._difficulty_bucket(
            entropy_nats,
            margin,
            maj_frac,
            entropy_easy_threshold,
        )
        self._entropy_window.append(float(entropy_nats))
        self._difficulty_window.append(difficulty_bucket_observed)
        easy_solver_penalty_scale = float(
            getattr(self.cfg, "easy_solver_penalty_scale", 1.0)
        )
        easy_solver_penalty_scale = max(0.0, easy_solver_penalty_scale)
        if easy_solver_case:
            solver_rewards_raw = [
                (-easy_solver_penalty_scale * sc_signal)
                if ans == maj_answer
                else (neg_weight * sc_signal)
                for ans in solver_answers_norm
            ]
        else:
            solver_rewards_raw = [
                sc_signal if ans == maj_answer else (-neg_weight * sc_signal)
                for ans in solver_answers_norm
            ]
        target_w = max(1, self.cfg.len_penalty_target_words)
        penalties = [min(1.0, max(0.0, (w - target_w) / float(target_w))) for w in pre_words]
        prob_map = {ans: count / float(self.cfg.num_solver_samples) for ans, count in hist.items()}
        solver_probs = [prob_map[ans] for ans in solver_answers_norm]
        solver_rewards_soft = [
            (prob ** self.cfg.solver_soft_gamma) * (1.0 - self.cfg.len_penalty_weight * pen)
            * reward_raw
            for prob, pen, reward_raw in zip(solver_probs, penalties, solver_rewards_raw)
        ]
        proposer_entropy_mu_used = self._update_proposer_entropy_target(entropy_nats)
        proposer_reward_raw = gaussian_reward(
            entropy_nats,
            proposer_entropy_mu_used,
            self.cfg.prop_entropy_sigma,
        )
        proposer_reward = proposer_reward_raw
        # Penalize zero-entropy (unanimous) outcomes — the question was too easy.
        # When all solvers agree perfectly, the proposer gets at most 10% of the
        # Gaussian reward.  Any disagreement (entropy > 0) removes the penalty.
        zero_entropy_cap = float(getattr(self.cfg, "zero_entropy_reward_cap", 0.10))
        zero_entropy_capped = False
        if entropy_nats < 1e-6 and zero_entropy_cap < 1.0:
            proposer_reward = min(proposer_reward, zero_entropy_cap)
            zero_entropy_capped = True
        # Additional easy-question penalty for low-entropy, high-margin cases.
        easy_question_penalty = float(getattr(self.cfg, "easy_question_penalty", 0.15))
        easy_question_detected = bool((entropy_nats < entropy_easy_threshold) and (margin > margin_max))
        if easy_question_detected and easy_question_penalty > 0.0:
            proposer_reward -= easy_question_penalty
        proposer_non_objective_penalty = float(
            getattr(self.cfg, "proposer_non_objective_penalty", 0.0)
        )
        proposer_non_objective_penalty = max(0.0, proposer_non_objective_penalty)
        if proposer_non_objective_question and proposer_non_objective_penalty > 0.0:
            proposer_reward -= proposer_non_objective_penalty
        proposer_reward = max(-1.0, min(1.0, proposer_reward))

        solver_stats_list = []
        solver_update_due = (
            self.solver_updater is not None
            and self.cfg.solver_update_freq > 0
            and (step % self.cfg.solver_update_freq == 0)
        )
        solver_update_applied = bool(solver_update_due)
        solver_update_skip_reason: Optional[str] = None

        skip_uninformative = bool(
            getattr(self.cfg, "skip_solver_update_when_uninformative", True)
        )
        always_scale = bool(
            getattr(self.cfg, "solver_always_update_with_informative_scaling", True)
        )
        min_update_scale = float(getattr(self.cfg, "solver_update_min_scale", 0.20))
        min_update_scale = max(0.0, min(1.0, min_update_scale))
        if always_scale:
            solver_update_scale = max(min_update_scale, local_info_score)
        else:
            solver_update_scale = 1.0
        solver_skip_update_on_easy = bool(
            getattr(self.cfg, "solver_skip_update_on_easy", True)
        )
        easy_update_majority_frac_threshold = float(
            getattr(self.cfg, "easy_update_majority_frac_threshold", 0.95)
        )
        easy_update_majority_frac_threshold = max(
            0.0, min(1.0, easy_update_majority_frac_threshold)
        )
        entropy_iqr_filter_min_majority_frac = float(
            getattr(self.cfg, "entropy_iqr_filter_min_majority_frac", 0.80)
        )
        entropy_iqr_filter_min_majority_frac = max(
            0.0, min(1.0, entropy_iqr_filter_min_majority_frac)
        )
        solver_entropy_iqr_blocked = bool(
            entropy_iqr_filter_active
            and (entropy_nats <= entropy_easy_threshold)
            and (maj_frac >= entropy_iqr_filter_min_majority_frac)
        )
        solver_easy_update_blocked = bool(
            solver_update_due
            and solver_skip_update_on_easy
            and (
                easy_solver_case
                or (maj_frac >= easy_update_majority_frac_threshold)
            )
        )
        if solver_update_applied and solver_entropy_iqr_blocked:
            solver_update_applied = False
            solver_update_skip_reason = "entropy_iqr_filter"
        elif solver_update_applied and solver_easy_update_blocked:
            solver_update_applied = False
            solver_update_skip_reason = "easy_case"
        elif solver_update_applied and (not always_scale) and skip_uninformative and not solver_informative_gate:
            solver_update_applied = False
            solver_update_skip_reason = "uninformative_local"

        if solver_update_applied:
            for sample_idx, (completion, reward) in enumerate(zip(solver_outputs, solver_rewards_soft)):
                local_can_solver_update = bool(str(completion).strip())
                any_rank_can_solver_update = self._dist_any_bool(local_can_solver_update)
                if not any_rank_can_solver_update:
                    self._append_jsonl(
                        self.policy_updates_log_path,
                        {
                            "step": step,
                            "role": "solver",
                            "source": "understanding",
                            "sample_idx": int(sample_idx),
                            "skipped": True,
                            "reason": "all_ranks_empty_solver_completion",
                        },
                    )
                    continue
                baseline_before = self.solver_baseline
                local_skip_update = not local_can_solver_update
                completion_for_update = completion if not local_skip_update else ""
                effective_reward = (
                    reward * solver_update_scale if not local_skip_update else 0.0
                )
                stats = self.solver_updater.step(
                    image=image,
                    prompt=solver_prompt,
                    completion=completion_for_update,
                    reward=effective_reward,
                    baseline=baseline_before if not local_skip_update else 0.0,
                    device=self.device,
                )
                solver_stats_list.append(stats)
                if stats.get("did_step", True):
                    self._policy_update_counts["solver"] += 1
                if not local_skip_update:
                    self._update_baseline("solver", reward)
                self._sync_state_scalars()

                # Aggressive cleanup after each solver update step to avoid OOM
                # on memory-constrained systems (especially with multiple samples).
                del stats
                torch.cuda.empty_cache()
                gc.collect()
            # If every sample was skipped (e.g. no valid completion tokens on all
            # ranks), report this explicitly instead of "applied=true".
            if solver_stats_list:
                all_skipped = all(bool(s.get("skipped_reason")) for s in solver_stats_list)
                if all_skipped:
                    solver_update_applied = False
                    if solver_update_skip_reason is None:
                        solver_update_skip_reason = "all_solver_samples_skipped"
        elif solver_update_due and solver_update_skip_reason is not None:
            self._append_jsonl(
                self.policy_updates_log_path,
                {
                    "step": step,
                    "role": "solver",
                    "source": "understanding",
                    "skipped": True,
                    "reason": solver_update_skip_reason,
                    "solver_margin": margin,
                    "entropy_nats": entropy_nats,
                },
            )

        proposer_stats = None
        proposer_skip_reason: Optional[str] = None
        proposer_update_due = self._is_proposer_update_due(step, phase="understanding")
        if proposer_update_due:
            baseline_before = self.proposer_baseline
            proposer_completion = str(proposer_out or "").strip()
            local_can_proposer_update = bool(proposer_completion)
            any_rank_can_proposer_update = self._dist_any_bool(local_can_proposer_update)
            if any_rank_can_proposer_update:
                completion_for_update = proposer_completion if local_can_proposer_update else ""
                effective_reward = proposer_reward if local_can_proposer_update else 0.0
                proposer_stats = self.proposer_updater.step(
                    image=image,
                    prompt=proposer_prompt,
                    completion=completion_for_update,
                    reward=effective_reward,
                    baseline=baseline_before if local_can_proposer_update else 0.0,
                    device=self.device,
                )
                if proposer_stats.get("did_step", True):
                    self._policy_update_counts["proposer"] += 1
                if local_can_proposer_update:
                    self._update_baseline("proposer", proposer_reward)
            else:
                proposer_skip_reason = "all_ranks_empty_proposer_completion"
                self._append_jsonl(
                    self.policy_updates_log_path,
                    {
                        "step": step,
                        "role": "proposer",
                        "source": "understanding",
                        "skipped": True,
                        "reason": proposer_skip_reason,
                    },
                )
            self._sync_state_scalars()
        else:
            proposer_skip_reason = "update_not_due"

        step_dt = time.perf_counter() - step_t0
        record = {
            "step": step,
            "phase": "understanding",
            "image_path": meta.get("path"),
            "step_time_sec": step_dt,
            "question": question,
            "proposer_out": proposer_out,
            "proposer_rationale": proposer_rationale,
            "fallback_question_used": fallback_used,
            "proposer_non_objective_question": proposer_non_objective_question,
            "proposer_non_objective_penalty": proposer_non_objective_penalty,
            "proposer_hardening_retries_used": hardening_retries_used,
            "proposer_hardening_reason": hardening_reason,
            "solver_answers_raw": solver_answers_raw,
            "solver_answers_norm": solver_answers_norm,
            "solver_rewards_raw": solver_rewards_raw,
            "solver_rewards_soft": solver_rewards_soft,
            "majority_answer": maj_answer,
            "majority_count": maj_count,
            "majority_fraction": maj_frac,
            "solver_top1_prob": p1,
            "solver_top2_prob": p2,
            "solver_margin": margin,
            "entropy_easy_threshold": entropy_easy_threshold,
            "entropy_iqr_filter_enabled": bool(entropy_iqr_state.get("enabled", 0.0)),
            "entropy_iqr_filter_active": entropy_iqr_filter_active,
            "entropy_iqr_filter_history_size": int(entropy_iqr_state.get("history_size", 0.0)),
            "entropy_iqr_filter_q1": entropy_iqr_state.get("q1"),
            "entropy_iqr_filter_q3": entropy_iqr_state.get("q3"),
            "entropy_iqr_filter_iqr": entropy_iqr_state.get("iqr"),
            "solver_informative_local": solver_informative_local,
            "solver_informative_any": solver_informative_any,
            "solver_informative_all": solver_informative_all,
            "solver_informative_ratio": informative_ratio,
            "solver_informative_ratio_min": ratio_min,
            "solver_informative_gate": solver_informative_gate,
            "solver_informative_gate_global": solver_informative_gate_global,
            "solver_margin_score": margin_damp_score,
            "solver_entropy_band_score": entropy_band_score,
            "solver_local_info_score": local_info_score,
            "easy_solver_case": easy_solver_case,
            "easy_solver_penalty_scale": easy_solver_penalty_scale,
            "solver_update_scale": solver_update_scale,
            "solver_temperature_schedule": solver_temperatures,
            "solver_top_p_schedule": solver_top_ps,
            "entropy_nats": entropy_nats,
            "proposer_entropy_mu_used": proposer_entropy_mu_used,
            "proposer_reward_raw": proposer_reward_raw,
            "proposer_reward": proposer_reward,
            "zero_entropy_capped": zero_entropy_capped,
            "zero_entropy_reward_cap": zero_entropy_cap,
            "easy_question_detected": easy_question_detected,
            "easy_question_penalty": easy_question_penalty,
            "solver_skip_update_on_easy": solver_skip_update_on_easy,
            "solver_entropy_iqr_blocked": solver_entropy_iqr_blocked,
            "entropy_iqr_filter_min_majority_frac": entropy_iqr_filter_min_majority_frac,
            "solver_easy_update_blocked": solver_easy_update_blocked,
            "easy_update_majority_frac_threshold": easy_update_majority_frac_threshold,
            "difficulty_sampler_enabled": difficulty_sampler_enabled,
            "difficulty_sampler_mode": difficulty_sampler_mode,
            "difficulty_target_bucket": desired_difficulty_bucket,
            "difficulty_bucket_observed": difficulty_bucket_observed,
            "difficulty_sampler_retries_used": difficulty_sampler_retries_used,
            "difficulty_sampler_max_retries": difficulty_sampler_max_retries,
            "difficulty_target_weights": difficulty_target_state.get("target_weights", {}),
            "difficulty_observed_weights": difficulty_target_state.get("observed_weights", {}),
            "difficulty_sampling_weights": difficulty_target_state.get("sampling_weights", {}),
            "solver_baseline": self.solver_baseline,
            "proposer_baseline": self.proposer_baseline,
            "solver_update_due": solver_update_due,
            "solver_update_applied": solver_update_applied,
            "solver_update_skip_reason": solver_update_skip_reason,
            "solver_stats": solver_stats_list,
            "proposer_update_due": proposer_update_due,
            "proposer_skip_reason": proposer_skip_reason,
            "proposer_stats": proposer_stats,
        }
        self._append_jsonl(self.iter_log_path, record)

        self._append_jsonl(
            self.rewards_log_path,
            {
                "step": step,
                "phase": "understanding",
                "image_path": meta.get("path"),
                "majority_answer": maj_answer,
                "majority_fraction": maj_frac,
                "solver_top1_prob": p1,
                "solver_top2_prob": p2,
                "solver_margin": margin,
                "entropy_easy_threshold": entropy_easy_threshold,
                "entropy_iqr_filter_enabled": bool(entropy_iqr_state.get("enabled", 0.0)),
                "entropy_iqr_filter_active": entropy_iqr_filter_active,
                "entropy_iqr_filter_history_size": int(entropy_iqr_state.get("history_size", 0.0)),
                "entropy_iqr_filter_q1": entropy_iqr_state.get("q1"),
                "entropy_iqr_filter_q3": entropy_iqr_state.get("q3"),
                "entropy_iqr_filter_iqr": entropy_iqr_state.get("iqr"),
                "solver_informative_local": solver_informative_local,
                "solver_informative_any": solver_informative_any,
                "solver_informative_all": solver_informative_all,
                "solver_informative_ratio": informative_ratio,
                "solver_informative_ratio_min": ratio_min,
                "solver_informative_gate": solver_informative_gate,
                "solver_informative_gate_global": solver_informative_gate_global,
                "solver_margin_score": margin_damp_score,
                "solver_entropy_band_score": entropy_band_score,
                "solver_local_info_score": local_info_score,
                "easy_solver_case": easy_solver_case,
                "easy_solver_penalty_scale": easy_solver_penalty_scale,
                "solver_update_scale": solver_update_scale,
                "entropy_nats": entropy_nats,
                "solver_reward_soft_mean": sum(solver_rewards_soft) / max(1, len(solver_rewards_soft)),
                "proposer_entropy_mu_used": proposer_entropy_mu_used,
                "proposer_reward_raw": proposer_reward_raw,
                "proposer_reward": proposer_reward,
                "proposer_non_objective_question": proposer_non_objective_question,
                "proposer_non_objective_penalty": proposer_non_objective_penalty,
                "proposer_hardening_retries_used": hardening_retries_used,
                "proposer_hardening_reason": hardening_reason,
                "zero_entropy_capped": zero_entropy_capped,
                "zero_entropy_reward_cap": zero_entropy_cap,
                "easy_question_detected": easy_question_detected,
                "easy_question_penalty": easy_question_penalty,
                "solver_skip_update_on_easy": solver_skip_update_on_easy,
                "solver_entropy_iqr_blocked": solver_entropy_iqr_blocked,
                "entropy_iqr_filter_min_majority_frac": entropy_iqr_filter_min_majority_frac,
                "solver_easy_update_blocked": solver_easy_update_blocked,
                "easy_update_majority_frac_threshold": easy_update_majority_frac_threshold,
                "difficulty_sampler_enabled": difficulty_sampler_enabled,
                "difficulty_sampler_mode": difficulty_sampler_mode,
                "difficulty_target_bucket": desired_difficulty_bucket,
                "difficulty_bucket_observed": difficulty_bucket_observed,
                "difficulty_sampler_retries_used": difficulty_sampler_retries_used,
                "difficulty_sampler_max_retries": difficulty_sampler_max_retries,
                "difficulty_target_weights": difficulty_target_state.get("target_weights", {}),
                "difficulty_observed_weights": difficulty_target_state.get("observed_weights", {}),
                "difficulty_sampling_weights": difficulty_target_state.get("sampling_weights", {}),
            },
        )

        if self.is_main_process and step % self.cfg.log_every == 0:
            print(
                f"[Step {step:05d}][U] maj={maj_count}/{self.cfg.num_solver_samples} "
                f"maj_frac={maj_frac:.2f} H={entropy_nats:.3f} M={margin:.3f} "
                f"info_local={int(solver_informative_local)} "
                f"info_ratio={informative_ratio:.2f} info_gate={int(solver_informative_gate)} "
                f"up_scale={solver_update_scale:.2f} P_R={proposer_reward:.3f} "
                f"dt={step_dt:.1f}s"
            )
            print(f"  Q: {question}")

        self._update_metric("u_majority_fraction", self._dist_mean(maj_frac))
        self._update_metric("u_entropy_nats", self._dist_mean(entropy_nats))
        self._update_metric("u_solver_margin", self._dist_mean(margin))
        self._update_metric("u_solver_informative", self._dist_mean(informative_ratio))
        self._update_metric("u_entropy_easy_threshold", self._dist_mean(entropy_easy_threshold))
        self._update_metric(
            "u_solver_entropy_iqr_blocked",
            self._dist_mean(1.0 if solver_entropy_iqr_blocked else 0.0),
        )
        self._update_metric(
            "u_difficulty_bucket_easy",
            self._dist_mean(1.0 if difficulty_bucket_observed == "easy" else 0.0),
        )
        self._update_metric(
            "u_difficulty_bucket_medium",
            self._dist_mean(1.0 if difficulty_bucket_observed == "medium" else 0.0),
        )
        self._update_metric(
            "u_difficulty_bucket_hard",
            self._dist_mean(1.0 if difficulty_bucket_observed == "hard" else 0.0),
        )
        self._update_metric("u_proposer_entropy_mu_used", self._dist_mean(proposer_entropy_mu_used))
        self._update_metric("u_proposer_reward", self._dist_mean(proposer_reward))

        return record

    # ---- Checkpoint: save/restore self-evolving state ---- #

    def _trainer_state_dict(self, step: int) -> Dict:
        """Extend parent state dict with self-evolving fields."""
        state = super()._trainer_state_dict(step)
        state["unified_gen_reward_ema"] = self._gen_reward_ema
        state["unified_gen_reward_ema_initialized"] = self._gen_reward_ema_initialized
        state["unified_entropy_window"] = list(self._entropy_window)
        state["unified_difficulty_window"] = list(self._difficulty_window)
        # Replay buffer metadata (not the images — too large for checkpoint;
        # the buffer refills naturally after resume).
        state["unified_replay_buffer_len"] = len(self.replay_buffer)
        return state

    def _maybe_resume_state(self):
        """Restore parent state, then restore self-evolving fields."""
        restored_step = super()._maybe_resume_state()
        if restored_step is None:
            return None

        resume_dir = self._resolve_resume_dir()
        if resume_dir is None:
            return restored_step

        state_path = resume_dir / "trainer_state.pt"
        if not state_path.exists():
            return restored_step

        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")

        if "unified_gen_reward_ema" in state:
            self._gen_reward_ema = float(state["unified_gen_reward_ema"])
            self._gen_reward_ema_initialized = bool(
                state.get("unified_gen_reward_ema_initialized", False)
            )
            entropy_window = state.get("unified_entropy_window")
            if isinstance(entropy_window, list):
                self._entropy_window.clear()
                max_keep = int(self._entropy_window.maxlen or len(entropy_window))
                for value in entropy_window[-max_keep:]:
                    try:
                        self._entropy_window.append(float(value))
                    except Exception:
                        continue
            difficulty_window = state.get("unified_difficulty_window")
            if isinstance(difficulty_window, list):
                self._difficulty_window.clear()
                max_keep = int(self._difficulty_window.maxlen or len(difficulty_window))
                for bucket in difficulty_window[-max_keep:]:
                    b = str(bucket).strip().lower()
                    if b in {"easy", "medium", "hard"}:
                        self._difficulty_window.append(b)
            if self.is_main_process:
                print(
                    f"[Unified] Restored self-evolving state: "
                    f"gen_reward_ema={self._gen_reward_ema:.4f}, "
                    f"replay_buf_was={state.get('unified_replay_buffer_len', 0)}"
                )

        return restored_step

    # ---- Self-evolving: helper methods ---- #

    def _current_gen_mix_ratio(self, step: int) -> float:
        """Compute the generated-image mixing ratio for the understanding step.

        Linearly ramps from ``gen_mix_ratio_start`` to ``gen_mix_ratio_max``
        over ``gen_mix_ratio_warmup_steps`` from the beginning of training.
        Returns 0.0 if the replay buffer is empty (naturally acts as a soft
        warm-up until the first generation steps populate the buffer).
        """
        start = getattr(self.ucfg, "gen_mix_ratio_start", 0.0)
        mx = getattr(self.ucfg, "gen_mix_ratio_max", 0.0)
        warmup = max(1, getattr(self.ucfg, "gen_mix_ratio_warmup_steps", 1))
        if mx <= 0.0:
            return 0.0
        elapsed = max(0, step - self.start_step)
        t = min(1.0, elapsed / warmup)
        return start + t * (mx - start)

    def _update_gen_reward_ema(self, reward_mean: float) -> None:
        """Update the exponential moving average of generator reward."""
        mom = getattr(self.ucfg, "reward_ema_momentum", 0.95)
        if not self._gen_reward_ema_initialized:
            self._gen_reward_ema = reward_mean
            self._gen_reward_ema_initialized = True
        else:
            self._gen_reward_ema = mom * self._gen_reward_ema + (1.0 - mom) * reward_mean

    def train(self):
        cfg = self.ucfg
        if cfg.total_steps <= self.start_step:
            raise ValueError(
                f"total_steps ({cfg.total_steps}) must be greater than start_step ({self.start_step})."
            )
        cycle = max(1, cfg.understanding_steps_per_cycle + cfg.generation_steps_per_cycle)

        if self.is_main_process:
            print(f"[Unified] Starting run at: {self.run_dir}")
            print(f"[Unified] Model: {cfg.model_name}")
            print(f"[Unified] Generator update rule: {cfg.generator_update_rule}")
            print(f"[Unified] Images: {len(self.pool)}")
            print(f"[Unified] Step range: {self.start_step + 1}..{cfg.total_steps}")
            print(
                f"[Unified] Schedule: Ux{cfg.understanding_steps_per_cycle} + Gx{cfg.generation_steps_per_cycle} (cycle={cycle})"
            )
            print(
                f"[Unified] Ref-answer scoring: {getattr(cfg, 'use_ref_answer_scoring', False)}, "
                f"Replay buffer: size={getattr(cfg, 'replay_buffer_size', 0)}, "
                f"Mix ratio: {getattr(cfg, 'gen_mix_ratio_start', 0)}->{getattr(cfg, 'gen_mix_ratio_max', 0)}"
            )

        last_completed_step = self.start_step
        last_attempted_step = self.start_step
        try:
            for step in range(self.start_step + 1, cfg.total_steps + 1):
                step_t0 = time.perf_counter()
                last_attempted_step = step
                image, meta = self._sample_image_for_step(step)
                phase_tag = "U"
                _data_source = "real"  # default; overwritten if replay buffer used

                phase_idx = (step - 1) % cycle
                if phase_idx < cfg.understanding_steps_per_cycle:
                    # ---- Phase 2: optionally mix generated images ---- #
                    # Use a step-seeded RNG so ALL DDP ranks make the same
                    # real-vs-replay decision and pick the same buffer index.
                    # This prevents rank-divergent training data.
                    _gen_mix = self._current_gen_mix_ratio(step)
                    if (
                        _gen_mix > 0
                        and self.replay_buffer
                        and len(self.replay_buffer) > 0
                    ):
                        _step_rng = random.Random(cfg.seed + step)
                        _use_replay = _step_rng.random() < _gen_mix
                        if _use_replay:
                            _buf_idx = _step_rng.randint(0, len(self.replay_buffer) - 1)
                            _entry = self.replay_buffer._entries[_buf_idx]
                            image = _entry.image
                            meta = {
                                "source": "replay_buffer",
                                "prompt": _entry.prompt,
                                "questions": _entry.questions,
                                "reference_answers": _entry.reference_answers,
                                "reward": _entry.reward,
                                "step_generated": _entry.step_generated,
                            }
                            _data_source = "replay_buffer"
                        else:
                            meta["source"] = "real"
                    else:
                        meta["source"] = "real"

                    self._understanding_step(step=step, image=image, meta=meta)
                else:
                    phase_tag = "G"
                    out = self._generation_step(step=step, image=image, meta=meta)
                    source_caption = str(out.get("source_caption", ""))
                    spec: GenerationSpec = out["spec"]
                    scored: List[Dict[str, object]] = out["scored"]
                    spec_quality = float(out.get("spec_quality", 0.0))
                    best_idx = int(out["best_idx"])
                    if cfg.synthetic_solver_update_freq > 0 and step % cfg.synthetic_solver_update_freq == 0:
                        self._solver_synthetic_update_from_best(step, scored[best_idx])

                    rewards = [float(c["total_reward"]) for c in scored]
                    reward_mean = sum(rewards) / max(1, len(rewards))
                    reward_max = max(rewards) if rewards else 0.0
                    reward_min = min(rewards) if rewards else 0.0
                    reward_mean_g = self._dist_mean(reward_mean)
                    reward_max_g = self._dist_mean(reward_max)
                    reward_min_g = self._dist_mean(reward_min)
                    spec_quality_g = self._dist_mean(spec_quality)

                    # Track generator reward EMA for monitoring
                    self._update_gen_reward_ema(reward_mean_g)

                    best = scored[best_idx]
                    best_spec = float(best.get("spec_score", 0.0))
                    best_cycle = float(best.get("cycle_score", 0.0))
                    best_div = float(best.get("diversity_score", 0.0))
                    best_contra = float(best.get("contradiction_score", 0.0))
                    best_spec_g = self._dist_mean(best_spec)
                    best_cycle_g = self._dist_mean(best_cycle)
                    best_div_g = self._dist_mean(best_div)
                    best_contra_g = self._dist_mean(best_contra)

                    self._append_jsonl(
                        self.iter_log_path,
                        {
                            "step": step,
                            "phase": "generation",
                            "image_path": meta.get("path"),
                            "prompt": spec.prompt,
                            "best_idx": best_idx,
                            "best_reward": float(scored[best_idx]["total_reward"]),
                            "spec_quality": spec_quality,
                            "generator_update_rule": self.cfg.generator_update_rule,
                            "generator_update_mode": out.get("generator_update_mode"),
                            "generator_skipped_reason": out.get("generator_skipped_reason"),
                            "proposer_update_due": out.get("proposer_update_due"),
                            "proposer_skip_reason": out.get("proposer_skip_reason"),
                            "proposer_stats": out.get("proposer_stats"),
                            "generator_baseline": self.generator_baseline,
                            "proposer_baseline": self.proposer_baseline,
                            "solver_baseline": self.solver_baseline,
                        },
                    )

                    self._wandb_log_step(
                        step=step,
                        image_path=meta.get("path"),
                        source_caption=source_caption,
                        spec=spec,
                        scored=scored,
                        best_idx=best_idx,
                        spec_quality=spec_quality_g,
                        reward_mean_global=reward_mean_g,
                        reward_max_global=reward_max_g,
                        reward_min_global=reward_min_g,
                        best_spec_global=best_spec_g,
                        best_cycle_global=best_cycle_g,
                        best_diversity_global=best_div_g,
                        best_contradiction_global=best_contra_g,
                        generator_skipped_reason=out.get("generator_skipped_reason"),
                        generator_update_mode=out.get("generator_update_mode"),
                        proposer_stats=out.get("proposer_stats"),
                        generator_stats=out.get("generator_stats"),
                    )

                if self.is_main_process:
                    step_dt = time.perf_counter() - step_t0
                    _src = _data_source if phase_tag == "U" else ""
                    _mix_info = ""
                    if phase_tag == "U" and _src == "replay_buffer":
                        _mix_info = f" [replay_buf, mix={self._current_gen_mix_ratio(step):.2f}]"
                    _ema_info = ""
                    if self._gen_reward_ema_initialized:
                        _ema_info = f" ema_r={self._gen_reward_ema:.4f}"
                    print(
                        f"[Step {step:05d}] phase={phase_tag}"
                        f"{_mix_info}{_ema_info} dt={step_dt:.1f}s"
                    )

                if cfg.save_every > 0 and step % cfg.save_every == 0:
                    self._dist_barrier()
                    self._save_checkpoint(step)
                    self._dist_barrier()

                if (
                    torch.cuda.is_available()
                    and cfg.clear_cache_every > 0
                    and step % cfg.clear_cache_every == 0
                ):
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    gc.collect()

                last_completed_step = step

            if cfg.save_every <= 0 or (cfg.total_steps % cfg.save_every) != 0:
                self._dist_barrier()
                self._save_checkpoint(cfg.total_steps)
                self._dist_barrier()
            self._write_ablation_summary(cfg.total_steps, status="completed")
            if self.is_main_process:
                print(f"[Unified] Training complete. Final checkpoint at step {cfg.total_steps:05d}.")

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            interrupted_step = int(last_attempted_step)
            tb = traceback.format_exc()
            if self.is_main_process:
                print(f"[Unified] Training interrupted at step {interrupted_step}: {error_text}")
                _json_dump(
                    self.run_dir / "interruption.json",
                    {
                        "status": "interrupted",
                        "interrupted_at_step": interrupted_step,
                        "last_completed_step": int(last_completed_step),
                        "error": error_text,
                        "traceback": tb,
                    },
                )

            emergency_step = max(1, interrupted_step)
            try:
                self._dist_barrier()
                self._save_checkpoint(emergency_step)
                self._dist_barrier()
            except Exception:
                pass

            self._write_ablation_summary(
                max(last_completed_step, emergency_step),
                status="interrupted",
                interrupted_at_step=interrupted_step,
                error=error_text,
            )
            raise
        finally:
            if self.wandb_run is not None and HAS_WANDB:
                try:
                    import wandb

                    wandb.finish()
                except Exception:
                    pass
            if self.distributed and dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    pass
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
