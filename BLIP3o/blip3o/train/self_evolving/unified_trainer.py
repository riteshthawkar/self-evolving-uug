"""
Unified (alternating understanding + generation) self-evolving trainer.

Ported from self_evolving/experiments/generation.py (UnifiedSelfEvolvingTrainer).
Extends GenerationSelfEvolvingTrainer with an interleaved understanding phase.
"""

import gc
import json
import math
import pathlib
import random
import re
import time
import traceback
from collections import deque
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from PIL import Image

from .config import UnifiedSelfEvolvingConfig
from .generation_helpers import GenerationSpec
from .generation_trainer import GenerationSelfEvolvingTrainer
from .prompts import (
    build_proposer_multi_prompt,
    build_solver_prompt,
)
from .replay_buffer import ReplayBuffer
from .utils import (
    HAS_WANDB,
    _json_dump,
    _parse_all_questions,
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
_MALFORMED_QUESTION_RE = re.compile(
    r"</?(?:answer|rationale|count|attribute|question)\b|```",
    flags=re.IGNORECASE,
)
_META_PLACEHOLDER_RE = re.compile(
    r"\(\s*[^)]*(?:count|attribute|spatial relation|comparison|number of|color|shape|position)\s*[^)]*\)",
    flags=re.IGNORECASE,
)
_QUESTION_START_RE = re.compile(
    r"^(?:what|which|how|where|when|who|is|are|was|were|does|do|did|can|could|should|would|has|have|had)\b",
    flags=re.IGNORECASE,
)
_EASY_BINARY_START_RE = re.compile(
    r"^(?:is|are|was|were|do|does|did|can|could|should|would|has|have|had)\b",
    flags=re.IGNORECASE,
)
_LOW_INFO_BINARY_TOKEN_RE = re.compile(
    r"\b(?:yes|no|visible|invisible|open|closed|present|absent|clear|murky|"
    r"not visible|unknown|unclear|cannot tell|can't tell)\b",
    flags=re.IGNORECASE,
)
_LATENT_NONVISUAL_RE = re.compile(
    r"\b(?:crispy|soft|texture|tasty|taste|flavor|smell|odor|fresh|stale|"
    r"hot|cold|ripe|unripe)\b",
    flags=re.IGNORECASE,
)
_LOW_SIGNAL_TEMPLATE_RE = re.compile(
    r"\b(?:what type of|what kind of|is there|are there)\b",
    flags=re.IGNORECASE,
)
_TWO_ANSWER_SPLIT_RE = re.compile(
    r"\s*(?:/|\||;|,|\bvs\.?\b|\bor\b)\s*",
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
        # Generated pool mode for understanding mixing.
        _mode = str(getattr(config, "gen_mix_source_mode", "buffer") or "buffer").strip().lower()
        if _mode not in {"buffer", "folder"}:
            _mode = "buffer"
        self._gen_mix_source_mode = _mode
        self._understanding_generated_only = bool(
            getattr(config, "understanding_generated_only", False)
        )
        # Replay buffer is only active in buffer mode.
        if self._gen_mix_source_mode == "buffer":
            _buf_size = getattr(config, "replay_buffer_size", 1000)
            _buf_min_r = getattr(config, "replay_min_reward", 0.5)
            _buf_stale = getattr(config, "replay_max_staleness", 500)
            self.replay_buffer = ReplayBuffer(
                max_size=_buf_size,
                min_reward=_buf_min_r,
                max_staleness=_buf_stale,
            )
        else:
            self.replay_buffer = None

        _generated_dir = getattr(config, "generated_mix_dir", None)
        if _generated_dir:
            self._generated_mix_dir = pathlib.Path(_generated_dir).expanduser().resolve()
        else:
            self._generated_mix_dir = (self.run_dir / "generated_mix_pool").resolve()
        self._generated_mix_cache: List[Dict[str, Any]] = []
        self._generated_mix_last_refresh_step = -10**9
        if self._gen_mix_source_mode == "folder" or self._understanding_generated_only:
            self._generated_mix_dir.mkdir(parents=True, exist_ok=True)

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
        q = " ".join(q.split())
        if _MALFORMED_QUESTION_RE.search(q):
            return False
        if _META_PLACEHOLDER_RE.search(q):
            return False
        if _SUBJECTIVE_QUESTION_RE.search(q):
            return False
        if len(q.split()) < 4:
            return False
        if not _QUESTION_START_RE.search(q):
            return False
        if not q.endswith("?"):
            return False
        return bool(_OBJECTIVE_QUESTION_RE.search(q))

    def _parse_proposer_question_candidates(self, proposer_out: str) -> List[Dict[str, str]]:
        """Parse multi-question proposer XML into structured candidates."""
        text = str(proposer_out or "")
        blocks = re.findall(
            r"<question[^>]*>(.*?)</question>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        candidates: List[Dict[str, str]] = []
        for block in blocks:
            q_text = (strip_tags(block, "text") or _parse_first_question(block) or "").strip()
            q_text = q_text.replace("\n", " ")
            if not q_text:
                continue
            candidates.append(
                {
                    "text": q_text,
                    "strategy_used": (strip_tags(block, "strategy_used") or "").strip(),
                    "two_answer_test": (strip_tags(block, "two_answer_test") or "").strip(),
                    "visual_target": (strip_tags(block, "visual_target") or "").strip(),
                    "rationale": (strip_tags(block, "rationale") or "").strip(),
                }
            )
        if candidates:
            return candidates
        # Fallback to legacy parser to ensure at least one candidate.
        return [{"text": q.replace("\n", " ").strip()} for q in _parse_all_questions(text) if q.strip()]

    def _split_two_answer_test(self, two_answer_test: str) -> List[str]:
        raw = str(two_answer_test or "").strip()
        if not raw:
            return []
        parts = _TWO_ANSWER_SPLIT_RE.split(raw)
        out: List[str] = []
        for p in parts:
            v = normalize_answer(p, max_words=8)
            if v and v not in out:
                out.append(v)
        return out[:4]

    def _question_template_key(self, question: str) -> str:
        q = normalize_answer(str(question or ""), max_words=16)
        if not q:
            return ""
        q = re.sub(r"\b\d+\b", "<num>", q)
        q = re.sub(r"'[^']*'|\"[^\"]*\"", "<quoted>", q)
        q = " ".join(q.split())
        return q

    def _question_repetition_penalty(self, question: str) -> float:
        key = self._question_template_key(question)
        if not key:
            return 0.0
        count = sum(1 for x in self._question_template_window if x == key)
        unit = float(getattr(self.cfg, "proposer_repeat_penalty_unit", 0.04))
        max_pen = float(getattr(self.cfg, "proposer_repeat_penalty_max", 0.25))
        if unit <= 0.0 or max_pen <= 0.0:
            return 0.0
        return max(0.0, min(max_pen, unit * float(count)))

    def _proposer_text_hardness_bonus(
        self,
        question: str,
        strategy_used: str,
        two_answer_test: str,
    ) -> float:
        """Cheap text-only hardness prior (no extra model calls)."""
        q = str(question or "").strip()
        if not q:
            return -0.20
        qn = normalize_answer(q, max_words=20)
        score = 0.0

        strat = normalize_answer(str(strategy_used or ""), max_words=4)
        if strat.startswith("h"):
            score += 0.06
        elif strat.startswith("m"):
            score += 0.02
        elif strat:
            score -= 0.02
        else:
            score -= 0.03

        if _EASY_BINARY_START_RE.search(qn):
            score -= 0.07
        if _LOW_INFO_BINARY_TOKEN_RE.search(qn):
            score -= 0.10
        if _LATENT_NONVISUAL_RE.search(qn):
            score -= 0.12
        if _LOW_SIGNAL_TEMPLATE_RE.search(qn):
            score -= 0.06
        if qn.startswith(("what ", "which ", "how many ")):
            score += 0.03

        alts = self._split_two_answer_test(two_answer_test)
        if len(alts) >= 2:
            score += 0.05
            if len(alts) == 2 and alts[0] == alts[1]:
                score -= 0.18
            if any(_LOW_INFO_BINARY_TOKEN_RE.search(a) for a in alts):
                score -= 0.10
        else:
            score -= 0.12

        if self._is_objective_question(q):
            score += 0.02
        else:
            score -= 0.06

        pos_cap = float(getattr(self.cfg, "proposer_text_bonus_max", 0.20))
        neg_cap = float(getattr(self.cfg, "proposer_text_penalty_max", 0.35))
        if pos_cap < 0.0:
            pos_cap = 0.0
        if neg_cap < 0.0:
            neg_cap = 0.0
        score = max(-neg_cap, min(pos_cap, score))
        if not math.isfinite(score):
            return 0.0
        return float(score)

    def _init_adaptive_windows(self):
        ent_window_size = max(8, int(getattr(self.cfg, "entropy_iqr_window_size", 256)))
        diff_window_size = max(8, int(getattr(self.cfg, "difficulty_sampler_window_size", 256)))
        qhist_window_size = max(32, int(getattr(self.cfg, "proposer_question_history_size", 256)))
        self._entropy_window = deque(maxlen=ent_window_size)
        self._difficulty_window = deque(maxlen=diff_window_size)
        self._question_template_window = deque(maxlen=qhist_window_size)

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
        acceptance_require_non_easy = bool(
            getattr(self.cfg, "acceptance_require_non_easy", True)
        )
        rejected_question_penalty = max(
            0.0, float(getattr(self.cfg, "rejected_question_penalty", 0.0))
        )
        entropy_iqr_state = self._entropy_iqr_filter_state()
        entropy_easy_threshold = float(entropy_iqr_state.get("threshold", entropy_min))
        entropy_iqr_filter_active = bool(entropy_iqr_state.get("active", 0.0) > 0.5)
        difficulty_target_state = self._choose_difficulty_target()
        difficulty_sampler_enabled = bool(difficulty_target_state.get("enabled", False))
        desired_difficulty_bucket = str(difficulty_target_state.get("desired_bucket", "medium"))
        difficulty_sampler_mode = str(difficulty_target_state.get("mode", "target"))
        solver_temperatures = self._solver_temperature_schedule()
        solver_top_ps = self._solver_top_p_schedule()

        # ------------------------------------------------------------------
        # Single-shot multi-question generation (no retry loop).
        #
        # The adversarial proposer generates K candidate questions in one
        # forward pass, ordered hardest-first with explicit chain-of-thought
        # about WHY each question will cause the solver to fail/disagree.
        # We spot-check each candidate with a small solver sample and accept
        # the first one that is non-easy (or the best available if all are easy).
        # This eliminates the while True retry loop entirely.
        # ------------------------------------------------------------------
        num_proposer_candidates = max(
            1, int(getattr(self.cfg, "proposer_num_candidates", 3))
        )
        # How many solver samples to use for the spot-check of each candidate.
        # Fewer samples = faster; default 3 gives ternary entropy outcomes.
        spot_check_samples = max(
            1, int(getattr(self.cfg, "proposer_spot_check_samples", 3))
        )
        # Avoid cold-only spot-checking (e.g. [0.5, 0.83, 1.17]) which tends to
        # classify borderline questions as easy. Start near the 33rd percentile.
        spot_check_offset = max(
            0,
            min(
                len(solver_temperatures) - spot_check_samples,
                len(solver_temperatures) // 3,
            ),
        )

        # Derive image source hint from path so the proposer can apply
        # dataset-appropriate strategies (COCO=natural scenes, TextVQA=text/signs,
        # ChartQA/GQA=charts/graphs/relational).  This is a soft hint only —
        # the proposer still selects the strategy from the library.
        _img_path = str(meta.get("path", "")).lower()
        if "textvqa" in _img_path:
            _src_hint = "textvqa"
        elif "chartqa" in _img_path or "chart" in _img_path:
            _src_hint = "chartqa"
        elif "gqa" in _img_path:
            _src_hint = "gqa"
        else:
            _src_hint = "coco"

        multi_proposer_prompt = build_proposer_multi_prompt(
            target_difficulty=desired_difficulty_bucket,
            num_questions=num_proposer_candidates,
            image_source_hint=_src_hint,
        )

        proposer_out = ""
        parsed_question = ""
        question = ""
        fallback_used = False
        proposer_rationale = ""
        proposer_non_objective_question = False
        difficulty_bucket_observed = "unknown"
        question_rejected = False
        question_reject_reason = ""
        chosen_strategy_used = ""
        chosen_two_answer_test = ""
        proposer_text_hardness_bonus = 0.0
        proposer_repetition_penalty = 0.0

        solver_prompt = ""
        solver_outputs: List[str] = []
        solver_answers_raw: List[str] = []
        solver_answers_norm: List[str] = []
        pre_words: List[int] = []

        # --- Single proposer call: generate all K candidates at once ---
        proposer_out = self._generate(
            image=image,
            prompt=multi_proposer_prompt,
            adapter_name="proposer" if self.cfg.use_lora else None,
            max_new_tokens=self.cfg.max_new_tokens_proposer,
            temperature=self.cfg.temp,
            top_p=self.cfg.top_p,
        )

        candidate_infos = self._parse_proposer_question_candidates(proposer_out)
        candidate_questions = [c.get("text", "").strip() for c in candidate_infos if c.get("text", "").strip()]
        if not candidate_questions:
            candidate_questions = ["What is the most salient object in the image?"]
            fallback_used = True
            candidate_infos = [{"text": candidate_questions[0]}]

        # --- Spot-check each candidate; select best acceptable by score ---
        best_question = ""
        best_outputs: List[str] = []
        best_answers_raw: List[str] = []
        best_answers_norm: List[str] = []
        best_pre_words: List[int] = []
        best_entropy = -1.0
        best_margin = 1.0
        best_bucket = "easy"
        best_meta: Dict[str, str] = {}
        best_pick_score = -1e9
        best_accept_question = ""
        best_accept_outputs: List[str] = []
        best_accept_answers_raw: List[str] = []
        best_accept_answers_norm: List[str] = []
        best_accept_pre_words: List[int] = []
        best_accept_entropy = -1.0
        best_accept_margin = 1.0
        best_accept_meta: Dict[str, str] = {}
        best_accept_pick_score = -1e9

        for cand_idx, cand_q in enumerate(candidate_questions):
            cand_q = cand_q.replace("\n", " ").strip()
            if not cand_q:
                continue
            cand_meta = candidate_infos[cand_idx] if cand_idx < len(candidate_infos) else {"text": cand_q}
            cand_strategy = str(cand_meta.get("strategy_used", "") or "")
            cand_two_answer = str(cand_meta.get("two_answer_test", "") or "")
            cand_text_bonus = self._proposer_text_hardness_bonus(
                cand_q,
                cand_strategy,
                cand_two_answer,
            )

            cand_non_objective = bool(
                require_objective and (not self._is_objective_question(cand_q))
            )

            cand_solver_prompt = build_solver_prompt(cand_q)
            cand_outputs: List[str] = []
            cand_answers_raw: List[str] = []
            cand_answers_norm: List[str] = []
            cand_pre_words: List[int] = []

            # Spot-check with a subset of solver samples for speed.
            for sc_idx in range(spot_check_samples):
                real_idx = spot_check_offset + sc_idx
                sc_temp = (
                    float(solver_temperatures[real_idx])
                    if real_idx < len(solver_temperatures)
                    else float(self.cfg.temp)
                )
                sc_top_p = (
                    float(solver_top_ps[real_idx])
                    if real_idx < len(solver_top_ps)
                    else float(self.cfg.top_p)
                )
                sc_out = self._generate(
                    image=image,
                    prompt=cand_solver_prompt,
                    adapter_name="default" if self.cfg.use_lora else None,
                    max_new_tokens=self.cfg.max_new_tokens_solver,
                    temperature=sc_temp,
                    top_p=sc_top_p,
                )
                sc_ans_raw = _parse_answer(sc_out)
                cand_outputs.append(sc_out)
                cand_answers_raw.append(sc_ans_raw)
                cand_answers_norm.append(normalize_answer(sc_ans_raw))
                cand_pre_words.append(pre_answer_word_count(sc_out))

            sc_hist: Dict[str, int] = {}
            for ans in cand_answers_norm:
                sc_hist[ans] = sc_hist.get(ans, 0) + 1
            sc_probs = [c / float(spot_check_samples) for c in sc_hist.values()]
            sc_entropy = shannon_entropy_nats(sc_probs)
            sc_sorted = sorted(sc_probs, reverse=True)
            sc_p1 = float(sc_sorted[0]) if sc_sorted else 0.0
            sc_p2 = float(sc_sorted[1]) if len(sc_sorted) > 1 else 0.0
            sc_margin = max(0.0, sc_p1 - sc_p2)
            sc_maj_frac = sc_p1
            sc_bucket = self._difficulty_bucket(
                sc_entropy, sc_margin, sc_maj_frac, entropy_easy_threshold
            )
            sc_is_easy = bool(
                (sc_entropy < entropy_easy_threshold) and (sc_margin > margin_max)
            )
            cand_pick_score = sc_entropy + cand_text_bonus

            # Always remember the best candidate seen so far.
            if (
                (cand_pick_score > best_pick_score)
                or (
                    abs(cand_pick_score - best_pick_score) <= 1e-8
                    and (
                        sc_entropy > best_entropy
                        or (abs(sc_entropy - best_entropy) <= 1e-8 and sc_margin < best_margin)
                    )
                )
            ):
                best_pick_score = cand_pick_score
                best_entropy = sc_entropy
                best_margin = sc_margin
                best_bucket = sc_bucket
                best_question = cand_q
                best_outputs = cand_outputs
                best_answers_raw = cand_answers_raw
                best_answers_norm = cand_answers_norm
                best_pre_words = cand_pre_words
                best_meta = dict(cand_meta)

            # Keep the best acceptable (non-easy, objective) candidate instead
            # of taking the first. This removes proposer ordering bias.
            if (not sc_is_easy) and (not cand_non_objective):
                if (
                    (cand_pick_score > best_accept_pick_score)
                    or (
                        abs(cand_pick_score - best_accept_pick_score) <= 1e-8
                        and (
                            sc_entropy > best_accept_entropy
                            or (
                                abs(sc_entropy - best_accept_entropy) <= 1e-8
                                and sc_margin < best_accept_margin
                            )
                        )
                    )
                ):
                    best_accept_pick_score = cand_pick_score
                    best_accept_entropy = sc_entropy
                    best_accept_margin = sc_margin
                    best_accept_question = cand_q
                    best_accept_outputs = cand_outputs
                    best_accept_answers_raw = cand_answers_raw
                    best_accept_answers_norm = cand_answers_norm
                    best_accept_pre_words = cand_pre_words
                    best_accept_meta = dict(cand_meta)

        if best_accept_question:
            question = best_accept_question
            solver_outputs = best_accept_outputs
            solver_answers_raw = best_accept_answers_raw
            solver_answers_norm = best_accept_answers_norm
            pre_words = best_accept_pre_words
            chosen_strategy_used = str(best_accept_meta.get("strategy_used", "") or "")
            chosen_two_answer_test = str(best_accept_meta.get("two_answer_test", "") or "")
        else:
            # No candidate cleared the gate — use the best-entropy one found.
            question = best_question or candidate_questions[0].replace("\n", " ").strip()
            if not question:
                question = "What is the most salient object in the image?"
                fallback_used = True
            solver_outputs = best_outputs
            solver_answers_raw = best_answers_raw
            solver_answers_norm = best_answers_norm
            pre_words = best_pre_words
            chosen_strategy_used = str(best_meta.get("strategy_used", "") or "")
            chosen_two_answer_test = str(best_meta.get("two_answer_test", "") or "")

        # If the chosen candidate came from a spot-check (< num_solver_samples),
        # run the remaining solver samples to reach the full count needed for RL.
        if len(solver_answers_norm) < self.cfg.num_solver_samples:
            solver_prompt = build_solver_prompt(question)
            for sample_idx in range(len(solver_answers_norm), self.cfg.num_solver_samples):
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

        # Ensure solver_prompt is always set (may have been skipped if all
        # solver samples were collected during spot-checking).
        if not solver_prompt and question:
            solver_prompt = build_solver_prompt(question)

        # Derive final question metadata from the accepted question.
        parsed_question = question
        fallback_used = fallback_used or (not bool(parsed_question))
        proposer_rationale = strip_tags(proposer_out, "rationale")
        proposer_non_objective_question = bool(
            require_objective and (not self._is_objective_question(question))
        )
        template_fallback_used = False

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

        # Classify difficulty bucket.
        easy_solver_case = bool((entropy_nats < entropy_easy_threshold) and (margin > margin_max))
        # Unsolvable = all solvers disagree at or below random-chance majority.
        unsolvable_threshold = float(
            getattr(self.cfg, "solver_unsolvable_maj_threshold",
                    1.0 / max(1, self.cfg.num_solver_samples))
        )
        unsolvable_solver_case = bool(
            not easy_solver_case and maj_frac <= unsolvable_threshold
        )
        difficulty_bucket_observed = self._difficulty_bucket(
            entropy_nats,
            margin,
            maj_frac,
            entropy_easy_threshold,
        )
        self._entropy_window.append(float(entropy_nats))
        self._difficulty_window.append(difficulty_bucket_observed)
        easy_solver_penalty_scale = max(
            0.0, float(getattr(self.cfg, "easy_solver_penalty_scale", 1.0))
        )

        # --- Solver rewards ---
        if easy_solver_case:
            solver_rewards_raw = [
                (-easy_solver_penalty_scale * sc_signal)
                if ans == maj_answer
                else (neg_weight * sc_signal)
                for ans in solver_answers_norm
            ]
        elif unsolvable_solver_case:
            solver_rewards_raw = [
                -neg_weight * sc_signal
                for _ in solver_answers_norm
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

        # --- Intuitive answer: one greedy solver call (V-Zero fast track) ---
        # Reference: "V-Zero: Self-Improving Multimodal Reasoning with Zero
        # Annotation" (arXiv:2601.10094)
        _intuitive_answer = ""
        _intuitive_generation_failed = False
        _intuitive_attempted = False
        if question and solver_prompt:
            _intuitive_attempted = True
            try:
                _intuitive_out = self._generate(
                    image=image,
                    prompt=solver_prompt,
                    adapter_name="default" if self.cfg.use_lora else None,
                    max_new_tokens=self.cfg.max_new_tokens_solver,
                    temperature=0.01,   # near-greedy
                    top_p=1.0,
                    do_sample=False,    # truly greedy (V-Zero intuitive track)
                )
                _intuitive_answer = normalize_answer(_parse_answer(_intuitive_out))
            except Exception:
                _intuitive_answer = ""
                _intuitive_generation_failed = True

        # --- Proposer reward: V-Zero dual-track learnability ---
        # The proposer is rewarded when the solver's "intuitive" (greedy)
        # answer DISAGREES with the "reasoned" (majority-voted, multi-temp)
        # answer.  This gives non-zero signal even when the solver is
        # unanimous on the reasoned track — the key to breaking the
        # same-model proposer-solver deadlock.
        proposer_entropy_mu_used = self._update_proposer_entropy_target(entropy_nats)
        proposer_reward_raw = gaussian_reward(
            entropy_nats,
            proposer_entropy_mu_used,
            self.cfg.prop_entropy_sigma,
        )
        proposer_reward = proposer_reward_raw

        zero_entropy_cap = float(getattr(self.cfg, "zero_entropy_reward_cap", 0.10))
        zero_entropy_capped = False
        _confidence_logprob = None  # kept for log-record compatibility
        unsolvable_capped = False

        _tracks_agree = bool(_intuitive_answer == maj_answer) if _intuitive_answer else True

        if unsolvable_solver_case:
            # Unsolvable → zero reward (AZ: r_propose = 0 when mean_solve = 0)
            proposer_reward = 0.0
            unsolvable_capped = True
        elif entropy_nats < 1e-6 or maj_frac >= 1.0:
            # Solver unanimous on reasoned track → check dual-track gap
            if not _tracks_agree and _intuitive_answer:
                # Intuitive ≠ reasoned → "gotcha" question (V-Zero case 2)
                # Reward = 0.5 * confidence (higher confidence → better gotcha)
                proposer_reward = 0.5 * maj_frac
            # else: keep proposer_reward = proposer_reward_raw (Gaussian).
            # At sigma=0.25 and entropy≈0, gaussian ≈ 0.0015 — nearly zero
            # but preserves micro-entropy variance as reward signal instead
            # of discarding to a flat 0.0 that kills all differentiation.
            zero_entropy_capped = True
        # else: non-unanimous → gaussian reward (already set above)

        # Text-only hardness shaping (no extra model calls): pushes proposer
        # away from low-information easy templates even when solver entropy
        # rewards are degenerate.
        proposer_text_hardness_bonus = self._proposer_text_hardness_bonus(
            question,
            chosen_strategy_used,
            chosen_two_answer_test,
        )
        proposer_repetition_penalty = self._question_repetition_penalty(question)
        proposer_reward += proposer_text_hardness_bonus
        proposer_reward -= proposer_repetition_penalty

        # Non-objective penalty.
        proposer_non_objective_penalty = max(
            0.0, float(getattr(self.cfg, "proposer_non_objective_penalty", 0.0))
        )
        if proposer_non_objective_question and proposer_non_objective_penalty > 0.0:
            proposer_reward -= proposer_non_objective_penalty

        # Rejection: non-objective or too-easy bucket.
        easy_question_detected = easy_solver_case
        reject_reasons: List[str] = []
        if require_objective and proposer_non_objective_question:
            reject_reasons.append("non_objective")
        if acceptance_require_non_easy and (difficulty_bucket_observed == "easy"):
            reject_reasons.append("easy_bucket")
        question_rejected = len(reject_reasons) > 0
        question_reject_reason = "|".join(reject_reasons)
        if question_rejected and rejected_question_penalty > 0.0:
            # Scale penalty by how far entropy is from target: fully-easy
            # (entropy=0) gets full penalty, near-target gets none. Creates
            # continuous gradient within the "easy" bucket instead of a flat
            # penalty that makes all easy questions identically bad.
            _rej_entropy = entropy_nats
            for _rr in reject_reasons:
                if _rr == "non_objective":
                    # Non-objective rejection: full penalty regardless of entropy
                    _rej_entropy = 0.0
                    break
            _easy_scale = max(0.0, 1.0 - min(1.0, _rej_entropy / max(1e-6, proposer_entropy_mu_used)))
            proposer_reward -= rejected_question_penalty * _easy_scale
        proposer_reward = max(-1.0, min(1.0, proposer_reward))
        # Track selected question template to discourage repeated easy loops.
        _qkey = self._question_template_key(question)
        if _qkey:
            self._question_template_window.append(_qkey)

        solver_stats_list = []
        solver_update_due = (
            self.solver_updater is not None
            and self.cfg.solver_update_freq > 0
            and (step % self.cfg.solver_update_freq == 0)
        )
        local_solver_update_applied = bool(solver_update_due)
        solver_update_applied = bool(solver_update_due)
        solver_update_skip_reason: Optional[str] = None
        solver_update_skip_reason_local: Optional[str] = None

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
        # NOTE: We intentionally do NOT block the solver update when question_rejected.
        # The solver should train even on rejected (easy/non-objective) questions:
        #   - On easy questions: solver_rewards_raw already assigns a NEGATIVE reward
        #     to the majority answer (penalising unanimous easy agreement) and a small
        #     positive to minority answers.  This is correct supervision — the solver
        #     learns "don't be so confident on easy-looking questions".
        #   - On non-objective questions: the answers are still real outputs with real
        #     rewards; excluding them starves the solver of data.
        # The PROPOSER is penalised for rejected questions (via rejected_question_penalty).
        # The SOLVER should always learn from whatever question it sees.
        # Block is applied only for the easy_case fast-path (solver_easy_update_blocked)
        # which uses the already-gated majority_frac threshold.
        if local_solver_update_applied and solver_entropy_iqr_blocked:
            local_solver_update_applied = False
            solver_update_skip_reason_local = "entropy_iqr_filter"
        elif local_solver_update_applied and solver_easy_update_blocked:
            local_solver_update_applied = False
            solver_update_skip_reason_local = "easy_case"
        # NOTE: With solver_always_update_with_informative_scaling=True (default),
        # `not always_scale` is False, making this branch unreachable.  The solver
        # always gets scaled updates (scale >= min_update_scale=0.20) rather than
        # full skips.  --skip_solver_update_when_uninformative only takes effect
        # when solver_always_update_with_informative_scaling is explicitly False.
        elif local_solver_update_applied and (not always_scale) and skip_uninformative and not solver_informative_gate:
            local_solver_update_applied = False
            solver_update_skip_reason_local = "uninformative_local"

        # DDP safety: if any rank runs solver updates, all ranks must execute
        # the same number of updater.forward() calls.
        solver_update_applied = self._dist_any_bool(local_solver_update_applied)
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
                local_skip_update = (not local_solver_update_applied) or (not local_can_solver_update)
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
                    # Track the SCALED reward that the updater actually receives,
                    # not the raw reward.  Otherwise baseline > effective_reward
                    # when scale < 1, causing systematic negative advantage bias.
                    self._update_baseline("solver", effective_reward)
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
                    if solver_update_skip_reason_local is None:
                        solver_update_skip_reason_local = "all_solver_samples_skipped"
        elif solver_update_due and solver_update_skip_reason_local is not None:
            self._append_jsonl(
                self.policy_updates_log_path,
                {
                    "step": step,
                    "role": "solver",
                    "source": "understanding",
                    "skipped": True,
                    "reason": solver_update_skip_reason_local,
                    "solver_margin": margin,
                    "entropy_nats": entropy_nats,
                },
            )
        if solver_update_applied:
            solver_update_skip_reason = None
        else:
            solver_update_skip_reason = (
                solver_update_skip_reason_local
                if solver_update_skip_reason_local is not None
                else "all_ranks_solver_update_blocked"
            )

        proposer_stats = None
        proposer_skip_reason: Optional[str] = None
        proposer_update_due = self._is_proposer_update_due(step, phase="understanding")
        if proposer_update_due:
            baseline_before = self.proposer_baseline
            # Train on the full proposer output (proposer_out) so that the
            # gradient flows through the rationale and reasoning tokens that
            # actually determine question difficulty — not just the final 8-12
            # question tokens. Fall back to question-only if proposer_out is
            # unavailable (e.g. template fallback path).
            proposer_completion = str(proposer_out or question or "").strip()
            local_can_proposer_update = bool(proposer_completion)
            any_rank_can_proposer_update = self._dist_any_bool(local_can_proposer_update)
            if any_rank_can_proposer_update:
                completion_for_update = proposer_completion if local_can_proposer_update else ""
                effective_reward = proposer_reward if local_can_proposer_update else 0.0

                if self._proposer_uses_grpo and completion_for_update:
                    # ── GRPO + EMA absolute baseline path ───────────────────────────
                    # Vanilla GRPO only sees *relative* reward within the group.
                    # When ALL group members produce easy questions, every member
                    # gets the same penalty → std≈0.21, mean_advantage≈0 →
                    # the gradient is noise and the proposer cannot escape the
                    # easy-question attractor.
                    #
                    # Fix: subtract a cross-step EMA baseline from every reward
                    # BEFORE computing within-group advantages.  This gives an
                    # *absolute* signal: even the "best" candidate in an all-bad
                    # group has a negative baseline-adjusted reward, producing a
                    # consistent push away from the easy attractor.
                    #
                    # After the update, feed the raw mean_reward back into the EMA
                    # so the baseline tracks the proposer's running performance.
                    # This is the approach used in Dr. GRPO / DAPO / REINFORCE++.
                    _grpo_completions = [completion_for_update]
                    _grpo_rewards = [effective_reward]
                    _grpo_images = [image]
                    _grpo_group_size = max(
                        2, int(getattr(self.cfg, "proposer_grpo_gen_group_size", 3))
                    )
                    _score_extras = bool(getattr(self.cfg, "score_grpo_extras", True))
                    _extra_temp_mult = float(getattr(self.cfg, "grpo_extra_temp_multiplier", 2.0))
                    _extra_temp = min(3.0, self.cfg.temp * _extra_temp_mult)
                    # Use a dedicated config for GRPO extras spot-check count,
                    # independent of the candidate-selection spot-check. Extras
                    # need ≥3 samples to produce ternary entropy outcomes
                    # (0, 0.637, 1.099) instead of binary (0 or 0.693), enabling
                    # differential reward signal across the GRPO group.
                    _extra_sc_samples = max(
                        2,
                        int(getattr(self.cfg, "grpo_extra_sc_samples", 3)),
                    )
                    _step_question_templates: set = set()
                    _chosen_qkey = self._question_template_key(question)
                    if _chosen_qkey:
                        _step_question_templates.add(_chosen_qkey)

                    for _gi in range(_grpo_group_size - 1):
                        try:
                            _extra_out = self._generate(
                                image=image,
                                prompt=multi_proposer_prompt,
                                adapter_name="proposer" if self.cfg.use_lora else None,
                                max_new_tokens=self.cfg.max_new_tokens_proposer,
                                temperature=_extra_temp,
                                top_p=self.cfg.top_p,
                            )
                            _extra_comp = str(_extra_out or "").strip()
                            if not _extra_comp:
                                continue

                            _extra_reward = 0.0  # default: neutral / unverified
                            _extra_entropy_val = -1.0
                            _extra_margin_val = 1.0
                            _extra_maj_frac = 0.0
                            _extra_bucket = "unknown"
                            _extra_intuitive_failed = False
                            _extra_intuitive_attempted = False
                            _sc_offset = 0  # temp-offset for extras spot-check
                            _extra_q = ""
                            _extra_strategy_used = ""
                            _extra_two_answer_test = ""
                            _extra_text_bonus = 0.0
                            _extra_repeat_penalty = 0.0

                            # Pick the strongest text-level candidate from this extra output
                            # (zero inference cost), then optionally run solver spot-check.
                            _extra_candidates = self._parse_proposer_question_candidates(_extra_out)
                            if _extra_candidates:
                                _extra_best = max(
                                    _extra_candidates,
                                    key=lambda c: self._proposer_text_hardness_bonus(
                                        c.get("text", ""),
                                        c.get("strategy_used", ""),
                                        c.get("two_answer_test", ""),
                                    ),
                                )
                                _extra_q = str(_extra_best.get("text", "")).replace("\n", " ").strip()
                                _extra_strategy_used = str(_extra_best.get("strategy_used", "") or "")
                                _extra_two_answer_test = str(_extra_best.get("two_answer_test", "") or "")

                            if _score_extras and _extra_q:
                                # ── Score extra candidate with configured solver spot-check ──
                                if _extra_q:
                                    _extra_solver_prompt = build_solver_prompt(_extra_q)
                                    _extra_answers_norm: List[str] = []
                                    # Offset extras spot-check to use hotter solver
                                    # temperatures. With schedule [0.5..2.5] and 3
                                    # spot-check samples, indices [0,1,2] give temps
                                    # [0.5, 0.83, 1.17] — too cold to break unanimity
                                    # on easy questions. Offset to ~33rd percentile so
                                    # samples use temps like [1.17, 1.5, 1.83] where
                                    # borderline questions are more likely to split.
                                    _sc_offset = max(
                                        0,
                                        min(
                                            len(solver_temperatures) - _extra_sc_samples,
                                            len(solver_temperatures) // 3,
                                        ),
                                    )
                                    for _sc_idx in range(_extra_sc_samples):
                                        _real_idx = _sc_offset + _sc_idx
                                        _sc_temp = (
                                            float(solver_temperatures[_real_idx])
                                            if _real_idx < len(solver_temperatures)
                                            else float(self.cfg.temp)
                                        )
                                        _sc_top_p = (
                                            float(solver_top_ps[_real_idx])
                                            if _real_idx < len(solver_top_ps)
                                            else float(self.cfg.top_p)
                                        )
                                        try:
                                            _sc_out = self._generate(
                                                image=image,
                                                prompt=_extra_solver_prompt,
                                                adapter_name="default" if self.cfg.use_lora else None,
                                                max_new_tokens=self.cfg.max_new_tokens_solver,
                                                temperature=_sc_temp,
                                                top_p=_sc_top_p,
                                            )
                                            _sc_ans = normalize_answer(_parse_answer(_sc_out))
                                            _extra_answers_norm.append(_sc_ans)
                                        except Exception:
                                            pass

                                    if _extra_answers_norm:
                                        _extra_hist: Dict[str, int] = {}
                                        for _ans in _extra_answers_norm:
                                            _extra_hist[_ans] = _extra_hist.get(_ans, 0) + 1
                                        _extra_probs = [
                                            c / float(len(_extra_answers_norm))
                                            for c in _extra_hist.values()
                                        ]
                                        _extra_entropy_val = shannon_entropy_nats(_extra_probs)
                                        _extra_sorted_probs = sorted(_extra_probs, reverse=True)
                                        _extra_p1 = float(_extra_sorted_probs[0]) if _extra_sorted_probs else 0.0
                                        _extra_p2 = float(_extra_sorted_probs[1]) if len(_extra_sorted_probs) > 1 else 0.0
                                        _extra_margin_val = max(0.0, _extra_p1 - _extra_p2)
                                        _extra_maj_frac = _extra_p1
                                        _extra_bucket = self._difficulty_bucket(
                                            _extra_entropy_val,
                                            _extra_margin_val,
                                            _extra_maj_frac,
                                            entropy_easy_threshold,
                                        )

                                        # Compute reward using same logic as chosen candidate.
                                        _extra_reward_raw = gaussian_reward(
                                            _extra_entropy_val,
                                            proposer_entropy_mu_used,
                                            self.cfg.prop_entropy_sigma,
                                        )
                                        _extra_reward = _extra_reward_raw
                                        # V-Zero dual-track for extras
                                        if _extra_entropy_val < 1e-6:
                                            # Unanimous on spot-check → get intuitive answer
                                            _extra_intuitive = ""
                                            _extra_intuitive_attempted = True
                                            try:
                                                _ei_out = self._generate(
                                                    image=image,
                                                    prompt=_extra_solver_prompt,
                                                    adapter_name="default" if self.cfg.use_lora else None,
                                                    max_new_tokens=self.cfg.max_new_tokens_solver,
                                                    temperature=0.01,
                                                    top_p=1.0,
                                                    do_sample=False,  # truly greedy (V-Zero intuitive track)
                                                )
                                                _extra_intuitive = normalize_answer(
                                                    _parse_answer(_ei_out)
                                                )
                                            except Exception:
                                                _extra_intuitive_failed = True
                                            _extra_maj = max(
                                                set(_extra_answers_norm),
                                                key=_extra_answers_norm.count,
                                            ) if _extra_answers_norm else ""
                                            if _extra_intuitive and _extra_maj and _extra_intuitive != _extra_maj:
                                                # Dual-track disagree → gotcha reward (consistent with chosen: 0.5 * maj_frac)
                                                _extra_reward = 0.5 * _extra_maj_frac
                                            # else: keep _extra_reward = _extra_reward_raw (set at L1042).
                                            # Preserves Gaussian micro-signal instead of discarding to flat 0.0.

                                    # ── Penalties for extras (mirror chosen-candidate objective) ──
                                    _extra_non_objective = bool(
                                        require_objective and (not self._is_objective_question(_extra_q))
                                    )
                                    _extra_reject_reasons: List[str] = []
                                    if _extra_non_objective:
                                        _extra_reject_reasons.append("non_objective")
                                    if (
                                        acceptance_require_non_easy
                                        and _extra_entropy_val >= 0.0
                                        and _extra_bucket == "easy"
                                    ):
                                        _extra_reject_reasons.append("easy_bucket")

                                    if _extra_non_objective and proposer_non_objective_penalty > 0.0:
                                        _extra_reward -= proposer_non_objective_penalty
                                    if _extra_reject_reasons and rejected_question_penalty > 0.0:
                                        _extra_rej_entropy = max(0.0, _extra_entropy_val)
                                        if "non_objective" in _extra_reject_reasons:
                                            _extra_rej_entropy = 0.0
                                        _extra_easy_scale = max(
                                            0.0,
                                            1.0 - min(
                                                1.0,
                                                _extra_rej_entropy / max(1e-6, proposer_entropy_mu_used),
                                            ),
                                        )
                                        _extra_reward -= rejected_question_penalty * _extra_easy_scale

                            if _extra_q:
                                # Deterministic text-level shaping for extras.
                                _extra_text_bonus = self._proposer_text_hardness_bonus(
                                    _extra_q,
                                    _extra_strategy_used,
                                    _extra_two_answer_test,
                                )
                                _extra_repeat_penalty = self._question_repetition_penalty(_extra_q)
                                _extra_qkey = self._question_template_key(_extra_q)
                                if _extra_qkey:
                                    if _extra_qkey in _step_question_templates:
                                        _extra_repeat_penalty += float(
                                            getattr(self.cfg, "proposer_text_step_dup_penalty", 0.08)
                                        )
                                    _step_question_templates.add(_extra_qkey)
                                _extra_reward += _extra_text_bonus
                                _extra_reward -= _extra_repeat_penalty

                            _extra_reward = max(-1.0, min(1.0, _extra_reward))

                            _grpo_completions.append(_extra_comp)
                            _grpo_rewards.append(_extra_reward)
                            _grpo_images.append(image)

                            # Log extra candidate stats for diagnostics.
                            if proposer_stats is None:
                                proposer_stats = {}
                            proposer_stats[f"grpo_extra_{_gi}_reward"] = _extra_reward
                            proposer_stats[f"grpo_extra_{_gi}_entropy"] = _extra_entropy_val
                            proposer_stats[f"grpo_extra_{_gi}_intuitive_attempted"] = _extra_intuitive_attempted
                            proposer_stats[f"grpo_extra_{_gi}_intuitive_failed"] = _extra_intuitive_failed
                            proposer_stats[f"grpo_extra_{_gi}_sc_samples"] = _extra_sc_samples
                            proposer_stats[f"grpo_extra_{_gi}_sc_offset"] = _sc_offset
                            proposer_stats[f"grpo_extra_{_gi}_margin"] = _extra_margin_val
                            proposer_stats[f"grpo_extra_{_gi}_bucket"] = _extra_bucket
                            proposer_stats[f"grpo_extra_{_gi}_strategy"] = _extra_strategy_used
                            proposer_stats[f"grpo_extra_{_gi}_two_answer_test"] = _extra_two_answer_test
                            proposer_stats[f"grpo_extra_{_gi}_text_bonus"] = _extra_text_bonus
                            proposer_stats[f"grpo_extra_{_gi}_repeat_penalty"] = _extra_repeat_penalty
                        except Exception:
                            pass

                    # ── Degenerate-group exploration noise ──────────────
                    # When ALL GRPO candidates receive identical pre-shift
                    # rewards (std ≈ 0), the baseline-shifted advantage path
                    # produces uniform advantages (e.g. [-1, -1, -1]).  This
                    # provides zero directional signal and accelerates mode
                    # collapse by uniformly reducing policy entropy.
                    #
                    # Fix: inject micro-noise to break the tie, creating a
                    # random exploration gradient.  Over many steps the random
                    # directions average out *except* when one direction
                    # accidentally produces a harder question — that step gets
                    # a real (non-noisy) reward signal and reinforces the move.
                    _pre_shift_std = (
                        torch.tensor(_grpo_rewards, dtype=torch.float64)
                        .std(correction=0)
                        .item()
                    )
                    _noise_enabled = bool(getattr(self.cfg, "grpo_degenerate_noise_enabled", True))
                    _noise_std_threshold = max(
                        0.0, float(getattr(self.cfg, "grpo_degenerate_noise_std_threshold", 1e-6))
                    )
                    _noise_sigma = max(0.0, float(getattr(self.cfg, "grpo_degenerate_noise_sigma", 0.03)))
                    if (
                        _noise_enabled
                        and _noise_sigma > 0.0
                        and _pre_shift_std < _noise_std_threshold
                        and len(_grpo_rewards) > 1
                    ):
                        _grpo_rewards = [r + random.gauss(0.0, _noise_sigma) for r in _grpo_rewards]
                        if proposer_stats is None:
                            proposer_stats = {}
                        proposer_stats["grpo_degenerate_noise"] = True
                        proposer_stats["grpo_degenerate_noise_sigma"] = _noise_sigma
                    else:
                        if proposer_stats is not None:
                            proposer_stats["grpo_degenerate_noise"] = False

                    # Apply EMA absolute baseline shift to all group rewards.
                    # self.proposer_baseline tracks the chosen candidate's reward;
                    # when the whole group underperforms the baseline the advantage
                    # of every candidate is negative → consistent push away.
                    _ema_baseline = float(self.proposer_baseline)
                    _grpo_rewards_shifted = [r - _ema_baseline for r in _grpo_rewards]

                    proposer_stats_grpo = self.proposer_updater.step(
                        prompt=multi_proposer_prompt,
                        completions=_grpo_completions,
                        rewards=_grpo_rewards_shifted,
                        device=self.device,
                        images=_grpo_images,
                        baseline_shifted=True,
                    )
                    if proposer_stats_grpo is not None:
                        if proposer_stats is None:
                            proposer_stats = {}
                        proposer_stats.update(proposer_stats_grpo)

                    # Update EMA baseline from the CHOSEN candidate's reward only.
                    # Previously this tracked the group mean (including unverified
                    # extras at 0.0), which made shifted rewards sum to zero at
                    # equilibrium → GRPO loss = 0 (mathematical deadlock).
                    # Tracking chosen-only means baseline → effective_reward at
                    # equilibrium; scored extras then get shifted ≠ 0 → non-zero loss.
                    self._update_baseline("proposer", effective_reward)
                    # Log the baseline shift for visibility.
                    if proposer_stats is not None:
                        proposer_stats["grpo_ema_baseline"] = _ema_baseline
                        proposer_stats["grpo_baseline_input"] = effective_reward
                        # Debug: log valid completions to diagnose GRPO loss=0
                        proposer_stats["grpo_valid_completions"] = proposer_stats.get("valid_completions", -1)
                else:
                    # ── REINFORCE path (legacy / proposer_update_rule="reinforce") ──
                    # Use the raw baseline without clamping. The previous clamp
                    # (min(baseline, reward) when reward < 0) caused the advantage
                    # to collapse to exactly 0.0 at equilibrium (when baseline ≈
                    # reward) — eliminating the learning signal entirely. Standard
                    # REINFORCE advantage = reward - baseline handles negative rewards
                    # correctly without any clamping.
                    effective_baseline = baseline_before if local_can_proposer_update else 0.0
                    proposer_stats = self.proposer_updater.step(
                        image=image,
                        prompt=multi_proposer_prompt,
                        completion=completion_for_update,
                        reward=effective_reward,
                        baseline=effective_baseline,
                        device=self.device,
                    )
                    if local_can_proposer_update:
                        self._update_baseline("proposer", proposer_reward)

                if proposer_stats and proposer_stats.get("did_step", True):
                    self._policy_update_counts["proposer"] += 1
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
            "proposer_template_fallback_used": template_fallback_used,
            "proposer_non_objective_question": proposer_non_objective_question,
            "proposer_non_objective_penalty": proposer_non_objective_penalty,
            "question_rejected": question_rejected,
            "question_reject_reason": question_reject_reason,
            "rejected_question_penalty": rejected_question_penalty,
            "acceptance_require_non_easy": acceptance_require_non_easy,
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
            "proposer_spot_check_samples": spot_check_samples,
            "proposer_spot_check_offset": spot_check_offset,
            "entropy_nats": entropy_nats,
            "proposer_entropy_mu_used": proposer_entropy_mu_used,
            "proposer_reward_raw": proposer_reward_raw,
            "proposer_reward": proposer_reward,
            "proposer_text_hardness_bonus": proposer_text_hardness_bonus,
            "proposer_repetition_penalty": proposer_repetition_penalty,
            "proposer_strategy_used": chosen_strategy_used,
            "proposer_two_answer_test": chosen_two_answer_test,
            "zero_entropy_capped": zero_entropy_capped,
            "zero_entropy_reward_cap": zero_entropy_cap,
            "intuitive_answer": _intuitive_answer,
            "dual_track_agree": _tracks_agree,
            "intuitive_attempted": _intuitive_attempted,
            "intuitive_generation_failed": _intuitive_generation_failed,
            "unsolvable_solver_case": unsolvable_solver_case,
            "unsolvable_capped": unsolvable_capped,
            "easy_question_detected": easy_question_detected,
            "solver_skip_update_on_easy": solver_skip_update_on_easy,
            "solver_entropy_iqr_blocked": solver_entropy_iqr_blocked,
            "entropy_iqr_filter_min_majority_frac": entropy_iqr_filter_min_majority_frac,
            "solver_easy_update_blocked": solver_easy_update_blocked,
            "easy_update_majority_frac_threshold": easy_update_majority_frac_threshold,
            "difficulty_sampler_enabled": difficulty_sampler_enabled,
            "difficulty_sampler_mode": difficulty_sampler_mode,
            "difficulty_target_bucket": desired_difficulty_bucket,
            "difficulty_bucket_observed": difficulty_bucket_observed,
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
                "proposer_spot_check_samples": spot_check_samples,
                "proposer_spot_check_offset": spot_check_offset,
                "entropy_nats": entropy_nats,
                "solver_reward_soft_mean": sum(solver_rewards_soft) / max(1, len(solver_rewards_soft)),
                "proposer_entropy_mu_used": proposer_entropy_mu_used,
                "proposer_reward_raw": proposer_reward_raw,
                "proposer_reward": proposer_reward,
                "proposer_text_hardness_bonus": proposer_text_hardness_bonus,
                "proposer_repetition_penalty": proposer_repetition_penalty,
                "proposer_strategy_used": chosen_strategy_used,
                "proposer_two_answer_test": chosen_two_answer_test,
                "proposer_non_objective_question": proposer_non_objective_question,
                "proposer_non_objective_penalty": proposer_non_objective_penalty,
                "question_rejected": question_rejected,
                "question_reject_reason": question_reject_reason,
                "rejected_question_penalty": rejected_question_penalty,
                "acceptance_require_non_easy": acceptance_require_non_easy,
                "zero_entropy_capped": zero_entropy_capped,
                "zero_entropy_reward_cap": zero_entropy_cap,
                "intuitive_answer": _intuitive_answer,
                "dual_track_agree": _tracks_agree,
                "intuitive_attempted": _intuitive_attempted,
                "intuitive_generation_failed": _intuitive_generation_failed,
                "unsolvable_solver_case": unsolvable_solver_case,
                "unsolvable_capped": unsolvable_capped,
                "easy_question_detected": easy_question_detected,
                "solver_skip_update_on_easy": solver_skip_update_on_easy,
                "solver_entropy_iqr_blocked": solver_entropy_iqr_blocked,
                "entropy_iqr_filter_min_majority_frac": entropy_iqr_filter_min_majority_frac,
                "solver_easy_update_blocked": solver_easy_update_blocked,
                "easy_update_majority_frac_threshold": easy_update_majority_frac_threshold,
                "difficulty_sampler_enabled": difficulty_sampler_enabled,
                "difficulty_sampler_mode": difficulty_sampler_mode,
                "difficulty_target_bucket": desired_difficulty_bucket,
                "difficulty_bucket_observed": difficulty_bucket_observed,
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
                f"T_B={proposer_text_hardness_bonus:.3f} R_P={proposer_repetition_penalty:.3f} "
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
        self._update_metric("u_proposer_text_bonus", self._dist_mean(proposer_text_hardness_bonus))
        self._update_metric("u_proposer_repeat_penalty", self._dist_mean(proposer_repetition_penalty))

        return record

    # ---- Checkpoint: save/restore self-evolving state ---- #

    def _trainer_state_dict(self, step: int) -> Dict:
        """Extend parent state dict with self-evolving fields."""
        state = super()._trainer_state_dict(step)
        state["unified_gen_reward_ema"] = self._gen_reward_ema
        state["unified_gen_reward_ema_initialized"] = self._gen_reward_ema_initialized
        state["unified_entropy_window"] = list(self._entropy_window)
        state["unified_difficulty_window"] = list(self._difficulty_window)
        state["unified_question_template_window"] = list(self._question_template_window)
        # Replay buffer metadata (not the images — too large for checkpoint;
        # the buffer refills naturally after resume).
        state["unified_replay_buffer_len"] = len(self.replay_buffer) if self.replay_buffer is not None else 0
        state["unified_gen_mix_source_mode"] = self._gen_mix_source_mode
        state["unified_understanding_generated_only"] = self._understanding_generated_only
        state["unified_generated_mix_dir"] = str(self._generated_mix_dir)
        state["unified_generated_mix_cache_len"] = len(self._generated_mix_cache)
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
            q_template_window = state.get("unified_question_template_window")
            if isinstance(q_template_window, list):
                self._question_template_window.clear()
                max_keep = int(self._question_template_window.maxlen or len(q_template_window))
                for key in q_template_window[-max_keep:]:
                    k = str(key).strip()
                    if k:
                        self._question_template_window.append(k)
            # When reset_proposer_baseline=True (set in parent _load_state),
            # also wipe the entropy/difficulty history so the IQR filter
            # re-warms from scratch instead of staying locked at IQR=0.
            if bool(getattr(self.cfg, "reset_proposer_baseline", False)):
                if self.is_main_process:
                    print(
                        "[Unified] reset_proposer_baseline=True: clearing entropy "
                        "and difficulty windows so IQR filter re-warms from scratch"
                    )
                self._entropy_window.clear()
                self._difficulty_window.clear()
                self._question_template_window.clear()
            if self.is_main_process:
                print(
                    f"[Unified] Restored self-evolving state: "
                    f"gen_reward_ema={self._gen_reward_ema:.4f}, "
                    f"replay_buf_was={state.get('unified_replay_buffer_len', 0)}, "
                    f"generated_mix_cache_was={state.get('unified_generated_mix_cache_len', 0)}"
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

    @staticmethod
    def _normalized_mix_reward(raw_reward: float, use_ref_scoring: bool) -> float:
        """Normalize reward used for generated-image quality gating."""
        if not use_ref_scoring:
            return float(raw_reward)
        # Same mapping used by replay-buffer integration in generation_trainer.
        try:
            return float(1.0 / (1.0 + math.exp(-(float(raw_reward) + 2.0))))
        except OverflowError:
            return 0.0 if float(raw_reward) < 0.0 else 1.0

    def _generated_mix_min_reward(self) -> float:
        return float(getattr(self.ucfg, "generated_mix_min_reward", 0.5))

    def _read_generated_mix_meta(self, meta_path: pathlib.Path) -> Optional[Dict[str, Any]]:
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None

        reward = float(payload.get("reward", -1.0))
        if reward < self._generated_mix_min_reward():
            return None

        image_path_raw = str(payload.get("image_path", "")).strip()
        if not image_path_raw:
            image_path = meta_path.with_suffix(".png")
        else:
            image_path = pathlib.Path(image_path_raw)
            if not image_path.is_absolute():
                image_path = (meta_path.parent / image_path).resolve()

        if not image_path.exists():
            return None

        questions = payload.get("questions", [])
        reference_answers = payload.get("reference_answers", [])
        if not isinstance(questions, list) or not isinstance(reference_answers, list):
            return None
        if not questions or not reference_answers:
            return None

        n = min(len(questions), len(reference_answers))
        if n <= 0:
            return None
        questions = [str(q).strip() for q in questions[:n]]
        reference_answers = [str(a).strip() for a in reference_answers[:n]]
        if not any(questions) or not any(reference_answers):
            return None

        return {
            "meta_path": str(meta_path.resolve()),
            "image_path": str(image_path.resolve()),
            "prompt": str(payload.get("prompt", "")),
            "questions": questions,
            "reference_answers": reference_answers,
            "reward": reward,
            "step_generated": int(payload.get("step_generated", 0)),
        }

    def _refresh_generated_mix_cache(self, step: int, force: bool = False) -> None:
        refresh_every = max(1, int(getattr(self.ucfg, "generated_mix_refresh_every", 10)))
        if (not force) and (step - self._generated_mix_last_refresh_step < refresh_every):
            return

        entries: List[Dict[str, Any]] = []
        if self._generated_mix_dir.exists():
            for meta_path in sorted(self._generated_mix_dir.glob("*.json")):
                parsed = self._read_generated_mix_meta(meta_path)
                if parsed is not None:
                    entries.append(parsed)

        max_files = max(1, int(getattr(self.ucfg, "generated_mix_max_files", 5000)))
        if len(entries) > max_files:
            entries = sorted(
                entries,
                key=lambda e: (int(e.get("step_generated", 0)), str(e.get("meta_path", ""))),
            )[-max_files:]

        self._generated_mix_cache = entries
        self._generated_mix_last_refresh_step = int(step)

    def _sample_generated_mix_from_folder(self, step: int) -> Optional[Dict[str, Any]]:
        self._refresh_generated_mix_cache(step=step)
        local_count = len(self._generated_mix_cache)
        shared_count = self._dist_min_int(local_count)
        if shared_count <= 0:
            return None

        chosen_meta_path: Optional[str]
        if self.is_main_process:
            rng = random.Random(int(self.cfg.seed) + int(step) * 104729 + 17)
            chosen_idx = rng.randint(0, shared_count - 1)
            chosen_meta_path = str(self._generated_mix_cache[chosen_idx]["meta_path"])
        else:
            chosen_meta_path = None

        if self.distributed and dist.is_initialized():
            obj = [chosen_meta_path]
            dist.broadcast_object_list(obj, src=0)
            chosen_meta_path = str(obj[0]) if obj[0] else None
        if not chosen_meta_path:
            return None

        parsed: Optional[Dict[str, Any]] = None
        all_ok = False
        # Mitigate short NFS visibility lag across DDP ranks.
        for _ in range(3):
            parsed = self._read_generated_mix_meta(pathlib.Path(chosen_meta_path))
            local_ok = parsed is not None
            all_ok = self._dist_all_bool(local_ok)
            if all_ok and parsed is not None:
                break
            time.sleep(0.05)
        if not all_ok or parsed is None:
            return None

        try:
            with Image.open(parsed["image_path"]) as img:
                image = img.convert("RGB")
        except Exception:
            return None

        meta = {
            "path": parsed["image_path"],
            "source": "generated_folder",
            "prompt": parsed.get("prompt", ""),
            "questions": parsed.get("questions", []),
            "reference_answers": parsed.get("reference_answers", []),
            "reward": float(parsed.get("reward", 0.0)),
            "step_generated": int(parsed.get("step_generated", 0)),
        }
        return {"image": image, "meta": meta}

    def _prune_generated_mix_dir(self) -> None:
        max_files = max(1, int(getattr(self.ucfg, "generated_mix_max_files", 5000)))
        meta_files = sorted(self._generated_mix_dir.glob("*.json"))
        if len(meta_files) <= max_files:
            return
        # Oldest first by mtime.
        meta_files = sorted(meta_files, key=lambda p: (p.stat().st_mtime, p.name))
        remove_count = max(0, len(meta_files) - max_files)
        for meta_path in meta_files[:remove_count]:
            image_candidates = [meta_path.with_suffix(".png")]
            parsed = self._read_generated_mix_meta(meta_path)
            if parsed is not None:
                image_candidates.insert(0, pathlib.Path(str(parsed["image_path"])))
            for image_path in image_candidates:
                try:
                    if image_path.exists():
                        image_path.unlink()
                except Exception:
                    pass
            try:
                if meta_path.exists():
                    meta_path.unlink()
            except Exception:
                pass

    def _store_best_generated_to_folder(
        self,
        *,
        step: int,
        spec: GenerationSpec,
        scored: List[Dict[str, object]],
        best_idx: int,
        reference_questions: Optional[List[str]] = None,
        reference_answers: Optional[List[str]] = None,
    ) -> None:
        if not self.is_main_process:
            return
        if best_idx < 0 or best_idx >= len(scored):
            return

        best = scored[best_idx]
        image = best.get("image")
        if not isinstance(image, Image.Image):
            return

        if isinstance(reference_questions, list) and isinstance(reference_answers, list):
            paired = [
                (str(q).strip(), str(a).strip())
                for q, a in zip(reference_questions, reference_answers)
            ]
            paired = [(q, a) for q, a in paired if q and a]
            questions = [q for q, _ in paired]
            answers = [a for _, a in paired]
        else:
            questions = [str(qa.question).strip() for qa in spec.qa_pairs if str(qa.question).strip()]
            answers = [str(qa.expected).strip() for qa in spec.qa_pairs if str(qa.expected).strip()]
        n = min(len(questions), len(answers))
        if n <= 0:
            return
        questions = questions[:n]
        answers = answers[:n]

        use_ref_scoring = bool(getattr(self.ucfg, "use_ref_answer_scoring", False))
        raw_reward = float(best.get("total_reward", 0.0))
        reward = self._normalized_mix_reward(raw_reward, use_ref_scoring)
        if reward < self._generated_mix_min_reward():
            return

        self._generated_mix_dir.mkdir(parents=True, exist_ok=True)
        stem = f"s{int(step):07d}_{int(time.time() * 1000)}_{random.randint(0, 999999):06d}"
        image_path = self._generated_mix_dir / f"{stem}.png"
        meta_path = self._generated_mix_dir / f"{stem}.json"

        try:
            image.convert("RGB").save(image_path, format="PNG")
        except Exception:
            return

        _json_dump(
            meta_path,
            {
                "step_generated": int(step),
                "prompt": str(spec.prompt),
                "questions": questions,
                "reference_answers": answers,
                "reward": float(reward),
                "raw_reward": float(raw_reward),
                "use_ref_answer_scoring": use_ref_scoring,
                "best_idx": int(best_idx),
                "num_candidates": int(len(scored)),
                "image_path": str(image_path),
            },
        )
        self._generated_mix_last_refresh_step = -10**9
        self._prune_generated_mix_dir()

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
                f"Replay buffer: size={len(self.replay_buffer) if self.replay_buffer is not None else 0}, "
                f"Mix ratio: {getattr(cfg, 'gen_mix_ratio_start', 0)}->{getattr(cfg, 'gen_mix_ratio_max', 0)}"
            )
            print(
                f"[Unified] Gen-mix source mode: {self._gen_mix_source_mode}, "
                f"generated_only={self._understanding_generated_only}, "
                f"generated_mix_dir={self._generated_mix_dir}"
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
                    # ---- Understanding phase: optional generated-image mix ---- #
                    _gen_mix = self._current_gen_mix_ratio(step)
                    _step_rng = random.Random(cfg.seed + step)
                    _want_generated = bool(self._understanding_generated_only)
                    if not _want_generated and _gen_mix > 0.0:
                        _want_generated = bool(_step_rng.random() < _gen_mix)

                    _used_generated = False
                    if _want_generated and self._gen_mix_source_mode == "folder":
                        folder_sample = self._sample_generated_mix_from_folder(step=step)
                        if folder_sample is not None:
                            image = folder_sample["image"]
                            meta = folder_sample["meta"]
                            _data_source = "generated_folder"
                            _used_generated = True
                    elif (
                        _want_generated
                        and self._gen_mix_source_mode == "buffer"
                        and self.replay_buffer
                        and len(self.replay_buffer) > 0
                    ):
                        _buf_idx = _step_rng.randint(0, len(self.replay_buffer) - 1)
                        _entry = self.replay_buffer._entries[_buf_idx]
                        image = _entry.image
                        meta = {
                            "path": None,
                            "source": "replay_buffer",
                            "prompt": _entry.prompt,
                            "questions": _entry.questions,
                            "reference_answers": _entry.reference_answers,
                            "reward": _entry.reward,
                            "step_generated": _entry.step_generated,
                        }
                        _data_source = "replay_buffer"
                        _used_generated = True

                    if self._understanding_generated_only and not _used_generated:
                        _data_source = "generated_pool_empty_skip"
                        self._append_jsonl(
                            self.iter_log_path,
                            {
                                "step": step,
                                "phase": "understanding",
                                "image_path": meta.get("path"),
                                "skip_reason": "generated_pool_empty",
                                "gen_mix_source_mode": self._gen_mix_source_mode,
                                "understanding_generated_only": True,
                            },
                        )
                    else:
                        if not _used_generated:
                            meta["source"] = "real"
                            _data_source = "real"
                        self._understanding_step(step=step, image=image, meta=meta)
                else:
                    phase_tag = "G"
                    # Sample the curriculum difficulty target for this generation
                    # step using the same sampler as the understanding phase.
                    # This closes the curriculum loop: the difficulty sampler
                    # tracks which buckets are under-represented in the history
                    # and upweights them — now both phases respond to it.
                    gen_difficulty_state = self._choose_difficulty_target()
                    gen_target_difficulty = str(gen_difficulty_state.get("desired_bucket", "medium"))
                    # In imageless proposer mode (E5), pass image=None so the
                    # proposer generates specs from topics instead of images.
                    _gen_image = None if bool(getattr(cfg, "imageless_proposer_mode", False)) else image
                    out = self._generation_step(
                        step=step,
                        image=_gen_image,
                        meta=meta,
                        target_difficulty=gen_target_difficulty,
                    )
                    source_caption = str(out.get("source_caption", ""))
                    spec: GenerationSpec = out["spec"]
                    scored: List[Dict[str, object]] = out["scored"]
                    spec_quality = float(out.get("spec_quality", 0.0))
                    best_idx = int(out["best_idx"])
                    if self._gen_mix_source_mode == "folder":
                        self._store_best_generated_to_folder(
                            step=step,
                            spec=spec,
                            scored=scored,
                            best_idx=best_idx,
                            reference_questions=out.get("reference_questions"),
                            reference_answers=out.get("reference_answers"),
                        )
                    if cfg.synthetic_solver_update_freq > 0 and step % cfg.synthetic_solver_update_freq == 0:
                        self._solver_synthetic_update_from_best(step, scored[best_idx])
                    # Joint step: also train the solver on the generated image every
                    # generation step. The solver already ran on it for scoring — this
                    # reuses those rollouts to turn every G-step into a U-step on
                    # synthetic data, effectively doubling understanding supervision.
                    # Only active when gen_step_solver_update_enabled=True.
                    elif bool(getattr(cfg, "gen_step_solver_update_enabled", False)):
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
                            "dit_update_due": out.get("dit_update_due"),
                            "dit_skip_reason": out.get("dit_skip_reason"),
                            "dit_stats": out.get("dit_stats"),
                            "unicorn_spec_meta": out.get("unicorn_spec_meta"),
                            "unicorn_reconstruction": out.get("unicorn_reconstruction"),
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
                        dit_stats=out.get("dit_stats"),
                        unicorn_spec_meta=out.get("unicorn_spec_meta"),
                        unicorn_reconstruction=out.get("unicorn_reconstruction"),
                    )

                if self.is_main_process:
                    step_dt = time.perf_counter() - step_t0
                    _src = _data_source if phase_tag == "U" else ""
                    _mix_info = ""
                    if phase_tag == "U":
                        if _src == "replay_buffer":
                            _mix_info = f" [replay_buf, mix={self._current_gen_mix_ratio(step):.2f}]"
                        elif _src == "generated_folder":
                            _mix_info = f" [generated_folder, mix={self._current_gen_mix_ratio(step):.2f}]"
                        elif _src == "generated_pool_empty_skip":
                            _mix_info = " [generated_pool_empty -> U-skip]"
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
