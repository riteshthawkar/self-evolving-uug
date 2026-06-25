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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from PIL import Image

from .config import UnifiedSelfEvolvingConfig
from .generation_helpers import GenerationSpec
from .generation_trainer import GenerationSelfEvolvingTrainer
from .prompts import (
    build_proposer_multi_prompt,
    build_solver_prompt,
    build_solver_prompt_pps,
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
    r"total|sum|percent|percentage|value|label|name|color|shape|position|pattern|material|made of|"
    r"left|right|top|bottom|above|below|inside|outside|"
    r"walking|standing|sitting|running|open|closed|attached|hanging|resting|supported|unsupported|"
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
    r"hot|cold|ripe|unripe|about to|just|recently|trying to|intent|stable|unstable|"
    r"supported|unsupported)\b",
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
_REASONING_DOMAIN_SPLIT_RE = re.compile(
    r"\s*(?:,|/|\||;|->|\+|\band\b)\s*",
    flags=re.IGNORECASE,
)
_VISUAL_TARGET_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:[hmcd]\d+(?:=[a-z0-9_\-]+)?|strategy|target)\s*$",
    flags=re.IGNORECASE,
)
_TWO_ANSWER_TOKEN_PLACEHOLDER_RE = re.compile(
    r"token[_\s-]?\d+",
    flags=re.IGNORECASE,
)
_BINARY_FORCED_CHOICE_RE = re.compile(
    r"\b(?:or|vs\.?)\b",
    flags=re.IGNORECASE,
)
_BINARY_ALT_TAIL_RE = re.compile(
    r"\b([a-z0-9][a-z0-9\-\s]{1,40})\s+(?:or|vs\.?)\s+([a-z0-9][a-z0-9\-\s]{1,40})\s*\??$",
    flags=re.IGNORECASE,
)
_BINARY_LOW_INFO_ALT_RE = re.compile(
    r"^(?:yes|no|true|false|unknown|unclear|not visible|cannot tell|can't tell)$",
    flags=re.IGNORECASE,
)
_OBJECTIVE_BINARY_HINT_RE = re.compile(
    r"\b("
    r"material|made of|color|shape|pattern|text|word|token|count|number|left|right|above|below|"
    r"inside|outside|walking|standing|sitting|running|open|closed|attached|hanging|resting|"
    r"supported|unsupported|before|after"
    r")\b",
    flags=re.IGNORECASE,
)
_QUESTION_TEMPLATE_LITERAL_RE = re.compile(
    r"\b(?:"
    r"stable-by-contact|pre-?event|post-?event|preparing-for-x|recovering-from-x|"
    r"token[_\s-]?\d+|lower count|higher count|plausible token [ab]|"
    r"support-state|event-phase|agent-intent"
    r")\b",
    flags=re.IGNORECASE,
)
_TWO_ANSWER_META_ALT_RE = re.compile(
    r"\b(?:"
    r"lower count|higher count|pre-?event|post-?event|token[_\s-]?\d+|"
    r"plausible token [ab]|option [ab]|answer [ab]|stable-by-contact|unsupported"
    r")\b",
    flags=re.IGNORECASE,
)
_LOW_INFO_ANSWER_RE = re.compile(
    r"^(?:"
    r"(?:yes|no)(?:\b.*)?|"
    r"supported|unsupported|present|absent|"
    r"unknown|unclear|not visible|cannot tell|can't tell|n/?a|none|"
    r"something|anything|object|item|thing"
    r")$",
    flags=re.IGNORECASE,
)
_VAGUE_COUNT_ANSWER_RE = re.compile(
    r"\b(?:many|too many|a lot|lots|several|some|few|numerous|multiple)\b",
    flags=re.IGNORECASE,
)
_NUMERIC_ANSWER_RE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
    flags=re.IGNORECASE,
)
_COUNT_QUESTION_RE = re.compile(
    r"\b(?:how many|number of|count)\b",
    flags=re.IGNORECASE,
)
_BINARY_QUESTION_RE = re.compile(
    r"^(?:is|are|was|were|do|does|did|can|could|should|has|have|had)\b",
    flags=re.IGNORECASE,
)
_NUM_WORD_TO_INT = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_YES_SET = {"yes", "y", "true", "present", "exists", "visible"}
_NO_SET = {"no", "n", "false", "absent", "missing", "none"}
_GENERIC_TARGET_TOKEN_RE = re.compile(
    r"^(?:"
    r"person|people|man|woman|child|player|car|vehicle|bike|bicycle|shoe|wheel|"
    r"road|street|tree|building|house|dog|cat|animal|object|item|thing"
    r")$",
    flags=re.IGNORECASE,
)
_EASY_ARCHETYPE_RE = re.compile(
    r"^(?:"
    r"what material is (?:the )?.+"
    r"|what color is (?:the )?.+"
    r"|where is (?:the )?.+ located"
    r"|is (?:the )?.+ to the (?:left|right|above|below|top|bottom) or "
    r"(?:left|right|above|below|top|bottom)(?: of .+)?"
    r"|is (?:the )?.+ (?:tied|untied|smooth|rough)"
    r")\??$",
    flags=re.IGNORECASE,
)
_LATENT_NON_OBSERVABLE_RE = re.compile(
    r"\b(?:"
    r"about to|just|recently|likely|probably|appears to|seems to|"
    r"pre-?event|post-?event|preparing-for-x|recovering-from-x|"
    r"stable-by-contact|unsupported|support-state|agent-intent|event-phase"
    r")\b",
    flags=re.IGNORECASE,
)
_DOMAIN_ALIAS_TO_CANONICAL = {
    "d1": "relation",
    "relation": "relation",
    "spatial": "relation",
    "multi-hop-relation": "relation",
    "support": "relation",
    "contact": "relation",
    "d2": "attribute",
    "attribute": "attribute",
    "attributes": "attribute",
    "object": "attribute",
    "objectidentity": "attribute",
    "identity": "attribute",
    "color": "attribute",
    "pattern": "attribute",
    "d3": "count",
    "count": "count",
    "counting": "count",
    "set-size": "count",
    "setsize": "count",
    "number": "count",
    "quantity": "count",
    "d4": "text",
    "ocr": "text",
    "text": "text",
    "symbol": "text",
    "symbols": "text",
    "reading": "text",
    "d5": "action",
    "action": "action",
    "observableaction": "action",
    "state": "action",
    "observablestate": "action",
    "pose": "action",
    "behavior": "action",
    "d6": "material",
    "material": "material",
    "materials": "material",
    "texture": "material",
    "surface": "material",
    "part-whole": "material",
    "partwhole": "material",
    "component": "material",
    "scientific": "material",
    "d7": "comparison",
    "comparison": "comparison",
    "comparative": "comparison",
    "ranking": "comparison",
    "ordering": "comparison",
    "depth": "comparison",
    "front-back": "comparison",
    "frontback": "comparison",
    # Legacy labels from earlier prompt versions.  Keep them parseable on resume,
    # but map them into observable benchmark skills rather than encouraging
    # hidden-state or external-knowledge reasoning.
    "temporal": "action",
    "event-phase": "action",
    "physics": "relation",
    "mechanics": "relation",
    "intent": "action",
    "agent-intent": "action",
    "commonsense": "attribute",
    "world-knowledge": "attribute",
    "realworld": "attribute",
    "causal": "comparison",
    "counterfactual": "comparison",
}


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
        if bool(getattr(config, "strict_imageless_mode", False)):
            config.imageless_proposer_mode = True
            config.understanding_generated_only = True
            if bool(getattr(config, "use_ref_answer_scoring", False)):
                config.use_ref_answer_scoring = False
                print(
                    "[Unified] strict_imageless_mode=True: disabling Solver-derived reference-answer scoring "
                    "(requires a real reference image)."
                )
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

    def _phase_for_step(self, step: int) -> Tuple[str, int]:
        """Return (phase, phase_local_index) for the given global step."""
        cfg = self.cfg
        u_steps = max(0, int(getattr(cfg, "understanding_steps_per_cycle", 0)))
        g_steps = max(0, int(getattr(cfg, "generation_steps_per_cycle", 0)))
        if u_steps == 0 and g_steps == 0:
            raise ValueError("Both understanding_steps_per_cycle and generation_steps_per_cycle are 0.")

        bootstrap = max(0, int(getattr(cfg, "bootstrap_generated_pool_steps", 0)))
        if step <= bootstrap:
            # Bootstrap phase is generation-only, regardless of cycle order.
            return "generation", int(step)

        rel_step = step - bootstrap
        cycle = max(1, u_steps + g_steps)
        cycle_idx = (rel_step - 1) // cycle
        phase_idx = (rel_step - 1) % cycle
        starts_with_generation = bool(getattr(cfg, "cycle_starts_with_generation", False))

        if starts_with_generation:
            if phase_idx < g_steps:
                local = bootstrap + cycle_idx * g_steps + phase_idx + 1
                return "generation", local
            u_pos = phase_idx - g_steps
            local = cycle_idx * u_steps + u_pos + 1
            return "understanding", local

        if phase_idx < u_steps:
            local = cycle_idx * u_steps + phase_idx + 1
            return "understanding", local
        g_pos = phase_idx - u_steps
        local = bootstrap + cycle_idx * g_steps + g_pos + 1
        return "generation", local

    def _phase_local_step_index(self, step: int, phase: str) -> int:
        phase_name, phase_local = self._phase_for_step(step)
        return phase_local if phase_name == phase else 0

    def _is_proposer_update_due(self, step: int, phase: str) -> bool:
        freq = int(getattr(self.cfg, "proposer_update_freq", 0) or 0)
        if freq <= 0:
            return False
        local_idx = self._phase_local_step_index(step, phase)
        if local_idx <= 0:
            return False
        return (local_idx % freq) == 0

    def _solver_top_p_schedule(self) -> List[float]:
        """Vary top_p across solver samples with a stratified ladder.

        Uses widely separated quantiles first (0.0, 1.0, 0.2, 0.8, ...),
        then fills with low-discrepancy points. This avoids near-duplicate
        decode regimes across samples.
        """
        n = max(1, int(self.cfg.num_solver_samples))
        top_p_min = float(getattr(self.cfg, "solver_top_p_min", 0.5))
        top_p_max = float(getattr(self.cfg, "solver_top_p_max", 1.0))
        if n <= 1:
            return [top_p_max]
        if abs(top_p_max - top_p_min) < 1e-8:
            return [top_p_min] * n
        q = self._solver_mix_quantiles(n)
        # Anti-correlate with temperature quantiles to increase decode diversity.
        return [top_p_min + (top_p_max - top_p_min) * (1.0 - qi) for qi in q]

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
        q = self._solver_mix_quantiles(n)
        return [tmin + (tmax - tmin) * qi for qi in q]

    def _solver_mix_quantiles(self, n: int) -> List[float]:
        n = max(1, int(n))
        if n <= 1:
            return [1.0]
        anchors = [0.0, 1.0, 0.2, 0.8, 0.4, 0.6, 0.1, 0.9, 0.3, 0.7, 0.5]
        if n <= len(anchors):
            return anchors[:n]

        out = list(anchors)
        k = 1
        while len(out) < n:
            # Van der Corput sequence (base-2) for stable low-discrepancy fill.
            x = 0.0
            denom = 1.0
            kk = k
            while kk > 0:
                kk, rem = divmod(kk, 2)
                denom *= 2.0
                x += rem / denom
            k += 1
            if all(abs(x - y) > 1e-9 for y in out):
                out.append(x)
        return out[:n]

    def _update_proposer_entropy_target(self, entropy_nats: float) -> float:
        if not bool(getattr(self.cfg, "adaptive_prop_entropy_target", True)):
            return float(self.cfg.prop_entropy_mu)
        prev = float(getattr(self, "proposer_entropy_mu_ema", self.cfg.prop_entropy_mu))
        # Only incorporate observations with meaningful entropy into the EMA.
        # Near-zero entropy (all solvers agree) carries no useful difficulty
        # signal — incorporating it would drag the target toward zero,
        # eliminating the gradient that pushes the proposer toward harder
        # questions.  When entropy is near-zero, keep the target unchanged.
        _ent_incorporate_floor = max(
            0.0, float(getattr(self.cfg, "prop_entropy_incorporate_floor", 0.05))
        )
        if float(entropy_nats) > _ent_incorporate_floor:
            anchor = self._dist_mean(float(entropy_nats))
            momentum = float(getattr(self.cfg, "prop_entropy_ema_momentum", 0.95))
            momentum = max(0.0, min(0.9999, momentum))
            ema = momentum * prev + (1.0 - momentum) * anchor
        else:
            ema = prev
        mu_min = float(getattr(self.cfg, "prop_entropy_mu_min", 0.40))
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
        q_lower = q.lower()
        subjective_hit = bool(_SUBJECTIVE_QUESTION_RE.search(q))
        if _MALFORMED_QUESTION_RE.search(q):
            return False
        if _META_PLACEHOLDER_RE.search(q):
            return False
        if len(q.split()) < 4:
            return False
        if not _QUESTION_START_RE.search(q):
            return False
        if not q.endswith("?"):
            return False
        # Reject latent-state binary forms that are weakly grounded in a single frame.
        if _EASY_BINARY_START_RE.search(q) and _LATENT_NONVISUAL_RE.search(q_lower):
            return False
        # Reject any question with a concrete forced-choice tail ("X or Y?"
        # / "X: A or B?").  These collapse solver entropy to zero and
        # prevent any learning signal, regardless of other quality signals.
        # Only reject when both alternatives are short (<=3 words each) —
        # longer matches indicate natural "or" usage, not forced-choice.
        _fc_match = _BINARY_ALT_TAIL_RE.search(q)
        if _fc_match:
            _fc_left = _fc_match.group(1).strip()
            _fc_right = _fc_match.group(2).strip()
            if len(_fc_left.split()) <= 3 and len(_fc_right.split()) <= 3:
                return False
        if _OBJECTIVE_QUESTION_RE.search(q):
            return True
        if subjective_hit:
            return False
        return False

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
                    "task_card": (strip_tags(block, "task_card") or "").strip(),
                    "reasoning_domains": (strip_tags(block, "reasoning_domains") or "").strip(),
                    "reasoning_chain": (strip_tags(block, "reasoning_chain") or "").strip(),
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

    def _is_easy_archetype_question(self, question: str) -> bool:
        qn = normalize_answer(str(question or ""), max_words=32)
        if not qn:
            return True
        if _EASY_ARCHETYPE_RE.match(qn):
            return True
        if _QUESTION_TEMPLATE_LITERAL_RE.search(qn):
            return True
        if _LATENT_NON_OBSERVABLE_RE.search(qn):
            return True
        return False

    def _normalize_visual_target(self, visual_target: str) -> str:
        t = normalize_answer(str(visual_target or ""), max_words=10)
        if not t:
            return ""
        t_compact = re.sub(r"\s+", "", t)
        if _VISUAL_TARGET_PLACEHOLDER_RE.match(t_compact):
            return ""
        t = re.sub(r"^(?:the|a|an)\s+", "", t).strip()
        if not t:
            return ""
        if len(t.split()) == 1 and _GENERIC_TARGET_TOKEN_RE.match(t):
            return ""
        return t

    def _extract_target_from_question(self, question_text: str) -> str:
        qn = normalize_answer(str(question_text or ""), max_words=32)
        if not qn:
            return ""
        patterns = [
            r"^how many ([a-z0-9' -]{2,48}) (?:are|is)\b",
            r"^what (?:material|color|pattern|word|text|token) is (?:the )?([a-z0-9' -]{2,48})\b",
            r"^is (?:the )?([a-z0-9' -]{2,48}) (?:to|in|on|at|near|behind|above|below|left|right)\b",
            r"^what is (?:the )?([a-z0-9' -]{2,48}) (?:made of|doing|holding)\b",
        ]
        for pat in patterns:
            m = re.search(pat, qn)
            if not m:
                continue
            target = self._normalize_visual_target(m.group(1))
            if target:
                return target
        return ""

    def _synthesize_grounded_question(
        self,
        question_text: str,
        meta: Dict[str, str],
        target: str,
        alts: List[str],
    ) -> str:
        qn = normalize_answer(str(question_text or ""), max_words=28)
        t = target or self._normalize_visual_target(meta.get("visual_target", ""))
        if not t:
            t = self._extract_target_from_question(question_text)
        task_card = normalize_answer(str(meta.get("task_card", "") or ""), max_words=2)
        domains = self._parse_reasoning_domains(str(meta.get("reasoning_domains", "") or ""))

        if t:
            if (task_card == "c6") or qn.startswith("how many") or ("count" in qn):
                return f"How many {t} are partially visible?"
            if (task_card == "c5") or ("text" in qn) or ("word" in qn) or ("token" in qn):
                return f"What exact text is visible on the {t}?"
            if len(alts) >= 2:
                # Keep question open-ended — NEVER inject the two_answer_test
                # alternatives.  Forced-choice kills solver entropy.
                return f"What is the {t}?"
            if "relation" in domains:
                return f"What is immediately beside the {t}?"
            return f"What detail on the {t} is most clearly visible?"

        # Last-resort synthesized question when target extraction fails.
        return "How many partially visible objects are near the center?"

    def _solver_focus_hint(self, sample_idx: int) -> str:
        hints = [
            "global scene layout",
            "fine text and symbols",
            "occlusion boundaries",
            "left-right spatial relations",
            "counting visible instances",
            "color and texture evidence",
            "object interaction cues",
        ]
        idx = int(sample_idx) if sample_idx >= 0 else 0
        return hints[idx % len(hints)]

    def _compile_question_from_slots(
        self,
        question_text: str,
        meta: Dict[str, str],
    ) -> Tuple[str, bool, str]:
        q = str(question_text or "").replace("\n", " ").strip()
        if not bool(getattr(self.cfg, "proposer_slot_compiler_enabled", True)):
            return q, True, "disabled"
        strict = bool(getattr(self.cfg, "proposer_slot_compiler_strict", True))
        target = self._normalize_visual_target(meta.get("visual_target", ""))
        if not target:
            target = self._extract_target_from_question(q)
        qn = normalize_answer(q, max_words=28)
        alts = []
        for opt in self._split_two_answer_test(meta.get("two_answer_test", "")):
            v = normalize_answer(opt, max_words=8)
            if not v:
                continue
            if _BINARY_LOW_INFO_ALT_RE.match(v):
                continue
            if _TWO_ANSWER_META_ALT_RE.search(v):
                continue
            if _VAGUE_COUNT_ANSWER_RE.search(v):
                continue
            alts.append(v)
        if len(alts) >= 2 and alts[0] == alts[1]:
            alts = []

        compiled = q
        is_count = bool(
            qn.startswith("how many")
            or ("number of" in qn)
            or ("count" in qn)
        )
        if target and is_count:
            compiled = f"How many {target} are visible?"
        elif target and len(alts) >= 2:
            # Keep the original proposer question open-ended.  The
            # two_answer_test alternatives stay as a HIDDEN validator and
            # are NOT injected into the question text.  Forced-choice
            # questions ("A or B?") collapse solver entropy to zero and
            # prevent any learning signal.
            if q.endswith("?") and not _BINARY_FORCED_CHOICE_RE.search(q):
                compiled = q
            else:
                # Original already contains "or" / "vs" options or is
                # malformed — rewrite as open-ended about the target.
                compiled = f"What is the {target}?"
        elif target and qn.startswith("what color"):
            compiled = f"What color is the {target}?"
        elif target and qn.startswith("what pattern"):
            compiled = f"What pattern is on the {target}?"
        elif target and qn.startswith("which"):
            compiled = f"Which {target} is correct?"

        if strict and (not target):
            repaired = self._synthesize_grounded_question(q, meta, target, alts).strip()
            if repaired:
                compiled = repaired
            else:
                return "", False, "target_missing_or_generic"

        compiled = compiled.strip()
        if compiled and not compiled.endswith("?"):
            compiled = compiled + "?"

        if strict and (self._is_easy_archetype_question(compiled) or (not self._is_objective_question(compiled))):
            repaired = self._synthesize_grounded_question(q, meta, target, alts).strip()
            if repaired:
                compiled = repaired
                if compiled and not compiled.endswith("?"):
                    compiled = compiled + "?"

        if not compiled:
            return "", False, "empty_compiled"
        if strict and self._is_easy_archetype_question(compiled):
            return "", False, "easy_archetype"
        if strict and (not self._is_objective_question(compiled)):
            return "", False, "non_objective"
        return compiled, True, "ok"

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

    def _extract_forced_choice_options(self, two_answer_test: str) -> Tuple[str, str]:
        """Extract concrete A/B alternatives from two_answer_test.

        Returns ("", "") when alternatives are missing, placeholders, or low-signal.
        """
        opts = self._split_two_answer_test(two_answer_test)
        cleaned: List[str] = []
        for opt in opts:
            v = normalize_answer(opt, max_words=8)
            if not v:
                continue
            v = re.sub(r"^(?:option|answer)\s*[ab]\s*[:.)-]?\s*", "", v).strip()
            v = re.sub(r"^[ab]\s*[:.)-]\s*", "", v).strip()
            if not v:
                continue
            if v in {"a", "b", "option a", "option b", "answer a", "answer b"}:
                continue
            if _TWO_ANSWER_TOKEN_PLACEHOLDER_RE.search(v):
                continue
            if _TWO_ANSWER_META_ALT_RE.search(v):
                continue
            if _BINARY_LOW_INFO_ALT_RE.match(v):
                continue
            if v not in cleaned:
                cleaned.append(v)
        if len(cleaned) >= 2 and cleaned[0] != cleaned[1]:
            return cleaned[0], cleaned[1]
        return "", ""

    def _parse_forced_choice_answer(
        self,
        answer_raw: str,
        option_a: str,
        option_b: str,
    ) -> str:
        """Map free-form solver output to canonical 'a'/'b' for choice prompts."""
        raw = str(answer_raw or "").strip()
        if not raw:
            return ""
        v = normalize_answer(raw, max_words=12)
        if not v:
            return ""
        if re.search(r"\b(?:option|answer)?\s*a\b", v) and not re.search(
            r"\b(?:option|answer)?\s*b\b", v
        ):
            return "a"
        if re.search(r"\b(?:option|answer)?\s*b\b", v) and not re.search(
            r"\b(?:option|answer)?\s*a\b", v
        ):
            return "b"

        a = normalize_answer(option_a, max_words=8)
        b = normalize_answer(option_b, max_words=8)
        if a and (a in v) and (b not in v):
            return "a"
        if b and (b in v) and (a not in v):
            return "b"

        vtoks = set(re.findall(r"[a-z0-9]+", v))
        atoks = set(re.findall(r"[a-z0-9]+", a))
        btoks = set(re.findall(r"[a-z0-9]+", b))
        sa = (len(vtoks & atoks) / float(max(1, len(vtoks | atoks)))) if atoks else 0.0
        sb = (len(vtoks & btoks) / float(max(1, len(vtoks | btoks)))) if btoks else 0.0
        if sa > sb and sa > 0.05:
            return "a"
        if sb > sa and sb > 0.05:
            return "b"
        return ""

    def _parse_reasoning_domains(self, raw: str) -> List[str]:
        text = str(raw or "").strip().lower()
        if not text:
            return []
        parts = _REASONING_DOMAIN_SPLIT_RE.split(text)
        out: List[str] = []
        for p in parts:
            key = normalize_answer(p, max_words=4).replace(" ", "")
            if not key:
                continue
            canonical = _DOMAIN_ALIAS_TO_CANONICAL.get(key, "")
            if canonical and canonical not in out:
                out.append(canonical)
        return out

    def _question_token_set(self, question: str) -> set:
        q = normalize_answer(str(question or ""), max_words=32)
        if not q:
            return set()
        toks = [t for t in re.findall(r"[a-z0-9]+", q) if len(t) > 2]
        stop = {
            "what",
            "which",
            "where",
            "when",
            "who",
            "many",
            "much",
            "color",
            "type",
            "kind",
            "this",
            "that",
            "there",
            "with",
            "from",
            "about",
            "into",
            "than",
            "then",
        }
        return {t for t in toks if t not in stop}

    def _jaccard_similarity(self, a: set, b: set) -> float:
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

    def _proposer_certificate_score(self, question: str, meta: Dict[str, str]) -> Dict[str, float]:
        if not bool(getattr(self.cfg, "proposer_certificate_enabled", True)):
            return {
                "score": 1.0,
                "valid": 1.0,
                "domains": 1.0,
                "non_relation": 1.0,
                "chain": 1.0,
                "two_answer": 1.0,
                "target": 1.0,
            }

        q = str(question or "").strip()
        domains = self._parse_reasoning_domains(meta.get("reasoning_domains", ""))
        min_domains = max(1, int(getattr(self.cfg, "proposer_reasoning_min_domains", 2)))
        require_non_rel = bool(getattr(self.cfg, "proposer_reasoning_require_non_relation", True))
        min_chain_words = max(1, int(getattr(self.cfg, "proposer_reasoning_min_chain_words", 8)))
        chain_words = len(str(meta.get("reasoning_chain", "") or "").split())
        two_ans = self._split_two_answer_test(meta.get("two_answer_test", ""))
        two_answer_raw = str(meta.get("two_answer_test", "") or "")
        visual_target = str(meta.get("visual_target", "") or "").strip()
        strategy_used = str(meta.get("strategy_used", "") or "").strip()
        compiler_valid = str(meta.get("_compiler_valid", "1")).strip() != "0"
        question_template_literal = bool(_QUESTION_TEMPLATE_LITERAL_RE.search(q))
        easy_archetype = self._is_easy_archetype_question(q)
        generic_target = (self._normalize_visual_target(visual_target) == "")
        objective = 1.0 if self._is_objective_question(q) else 0.0
        has_domains = 1.0 if len(domains) >= min_domains else 0.0
        has_non_rel = 1.0 if any(d != "relation" for d in domains) else 0.0
        if not require_non_rel:
            has_non_rel = 1.0
        has_chain = 1.0 if chain_words >= min_chain_words else 0.0
        two_answer_placeholder = bool(_TWO_ANSWER_TOKEN_PLACEHOLDER_RE.search(two_answer_raw))
        if not two_answer_placeholder:
            for opt in two_ans[:2]:
                opt_norm = normalize_answer(opt, max_words=6)
                if opt_norm in {
                    "a",
                    "b",
                    "option a",
                    "option b",
                    "answer a",
                    "answer b",
                    "token",
                    "token 1",
                    "token 2",
                    "token1",
                    "token2",
                }:
                    two_answer_placeholder = True
                    break
        two_answer_meta_placeholder = bool(_TWO_ANSWER_META_ALT_RE.search(two_answer_raw))
        if not two_answer_meta_placeholder:
            for opt in two_ans:
                if _TWO_ANSWER_META_ALT_RE.search(opt):
                    two_answer_meta_placeholder = True
                    break
        two_answer_valid = (
            1.0
            if (
                len(two_ans) >= 2
                and two_ans[0] != two_ans[1]
                and (not two_answer_placeholder)
                and (not two_answer_meta_placeholder)
            )
            else 0.0
        )
        target_compact = re.sub(r"\s+", "", visual_target.lower())
        strategy_compact = re.sub(r"\s+", "", strategy_used.lower())
        target_placeholder = bool(_VISUAL_TARGET_PLACEHOLDER_RE.match(target_compact))
        if strategy_compact and target_compact == strategy_compact:
            target_placeholder = True
        has_target = 1.0 if (len(visual_target.split()) >= 2 and (not target_placeholder)) else 0.0

        score = (
            0.24 * objective
            + 0.20 * has_domains
            + 0.18 * has_non_rel
            + 0.16 * has_chain
            + 0.14 * two_answer_valid
            + 0.08 * has_target
        )
        if not compiler_valid:
            score -= 0.20
        if question_template_literal:
            score -= 0.20
        if easy_archetype:
            score -= 0.20
        if generic_target:
            score -= 0.12
        score = float(max(0.0, min(1.0, score)))
        min_score = max(0.0, min(1.0, float(getattr(self.cfg, "proposer_certificate_min_score", 0.55))))
        strict_struct = bool(getattr(self.cfg, "proposer_certificate_strict_struct", True))
        if strict_struct:
            valid = 1.0 if (
                score >= min_score
                and objective > 0.5
                and has_domains > 0.5
                and has_non_rel > 0.5
                and has_chain > 0.5
                and two_answer_valid > 0.5
                and has_target > 0.5
                and (not question_template_literal)
                and (not two_answer_meta_placeholder)
                and (not easy_archetype)
                and compiler_valid
                and (not generic_target)
            ) else 0.0
        else:
            valid = 1.0 if score >= min_score else 0.0
        return {
            "score": score,
            "valid": valid,
            "domains": has_domains,
            "non_relation": has_non_rel,
            "chain": has_chain,
            "two_answer": two_answer_valid,
            "target": has_target,
            "target_placeholder": 1.0 if target_placeholder else 0.0,
            "two_answer_placeholder": 1.0 if two_answer_placeholder else 0.0,
            "two_answer_meta_placeholder": 1.0 if two_answer_meta_placeholder else 0.0,
            "question_template_literal": 1.0 if question_template_literal else 0.0,
            "easy_archetype": 1.0 if easy_archetype else 0.0,
            "generic_target": 1.0 if generic_target else 0.0,
            "compiler_valid": 1.0 if compiler_valid else 0.0,
        }

    def _is_low_info_majority_answer(self, question: str, majority_answer: str) -> bool:
        q = normalize_answer(str(question or ""), max_words=24)
        a = normalize_answer(str(majority_answer or ""), max_words=12)
        if not a:
            return True
        if a == "ood":
            return True
        if a in {"yes", "no"} or a.startswith("yes ") or a.startswith("no "):
            return True
        if _LOW_INFO_ANSWER_RE.match(a):
            return True
        if _QUESTION_TEMPLATE_LITERAL_RE.search(q):
            return True
        if (
            q.startswith("how many")
            or ("number of" in q)
            or ("count" in q)
        ):
            if _VAGUE_COUNT_ANSWER_RE.search(a) and (not _NUMERIC_ANSWER_RE.search(a)):
                return True
        return False

    def _classify_answer_family(self, question: str, majority_answer: str) -> str:
        q = normalize_answer(str(question or ""), max_words=24)
        a = normalize_answer(str(majority_answer or ""), max_words=12)
        if not a:
            return "empty"
        if a == "ood":
            return "ood"
        if a in {"yes", "no"} or a.startswith("yes ") or a.startswith("no "):
            return "yesno"
        if _LOW_INFO_ANSWER_RE.match(a):
            return "low_info"
        if q.startswith("how many") or ("number of" in q) or ("count" in q):
            if _VAGUE_COUNT_ANSWER_RE.search(a) and (not _NUMERIC_ANSWER_RE.search(a)):
                return "count_vague"
            if _NUMERIC_ANSWER_RE.search(a):
                return "count_numeric"
            return "count_other"
        if "made of" in q or "material" in q:
            return "material"
        if "color" in q:
            return "color"
        if re.search(r"\b(left|right|above|below|top|bottom|inside|outside)\b", q):
            return "direction"
        return "other"

    def _answer_family_penalty(self, family: str) -> float:
        fam = str(family or "").strip().lower()
        if not fam:
            return 0.0
        hist = list(getattr(self, "_answer_family_window", []))
        rep_target = max(
            0.0,
            min(1.0, float(getattr(self.cfg, "proposer_answer_family_repeat_target", 0.25))),
        )
        rep_pen_weight = max(
            0.0, float(getattr(self.cfg, "proposer_answer_family_repeat_penalty", 0.25))
        )
        repeat_pen = 0.0
        if hist and rep_pen_weight > 0.0:
            share = float(sum(1 for x in hist if x == fam)) / float(len(hist))
            if share > rep_target:
                repeat_pen = rep_pen_weight * min(
                    1.0, (share - rep_target) / max(1e-6, 1.0 - rep_target)
                )
        trivial_pen = 0.0
        if fam in {"yesno", "low_info", "ood", "count_vague", "material", "direction"}:
            trivial_pen = max(
                0.0, float(getattr(self.cfg, "proposer_trivial_archetype_penalty", 0.25))
            )
        return float(repeat_pen + trivial_pen)

    def _question_answer_type(self, question: str) -> str:
        q = normalize_answer(str(question or ""), max_words=28)
        if not q:
            return "other"
        if _COUNT_QUESTION_RE.search(q):
            return "count"
        # Treat only pure yes/no style prompts as binary. Many "is ... A or B?"
        # prompts are forced-choice attribute questions, not yes/no.
        if _BINARY_QUESTION_RE.search(q) and (not _BINARY_FORCED_CHOICE_RE.search(q)):
            return "binary"
        if any(k in q for k in ("text", "word", "token", "letter", "sign")):
            return "text"
        if any(k in q for k in ("left", "right", "above", "below", "between", "beside")):
            return "spatial"
        if any(k in q for k in ("color", "material", "pattern", "shape", "type")):
            return "attribute"
        return "other"

    def _normalize_answer_for_type(
        self,
        answer_norm: str,
        answer_type: str,
    ) -> Tuple[str, bool, bool]:
        a = normalize_answer(str(answer_norm or ""), max_words=12)
        if not a:
            return "ood", True, True
        low_info = bool(_LOW_INFO_ANSWER_RE.match(a) or _VAGUE_COUNT_ANSWER_RE.search(a))
        noncanonical = False
        out = a
        at = str(answer_type or "other").strip().lower()

        if at == "count":
            m = re.search(r"\b\d+\b", a)
            if m:
                out = str(int(m.group(0)))
            elif a in _NUM_WORD_TO_INT:
                out = str(int(_NUM_WORD_TO_INT[a]))
            else:
                out = "ood"
                noncanonical = True
            if low_info:
                noncanonical = True
        elif at == "binary":
            tok = a.split()[0]
            if tok in _YES_SET:
                out = "yes"
            elif tok in _NO_SET:
                out = "no"
            else:
                out = "ood"
                noncanonical = True
            if low_info:
                noncanonical = True
        else:
            if low_info:
                noncanonical = True

        return out, low_info, noncanonical

    def _arm_domain_bucket(self, reasoning_domains_raw: str) -> str:
        domains = self._parse_reasoning_domains(reasoning_domains_raw)
        for d in domains:
            if d != "relation":
                return d
        return domains[0] if domains else "relation"

    def _arm_key(self, task_card: str, domain_bucket: str, answer_type: str) -> str:
        tc = normalize_answer(str(task_card or ""), max_words=2).replace(" ", "")
        db = normalize_answer(str(domain_bucket or ""), max_words=2).replace(" ", "")
        at = normalize_answer(str(answer_type or ""), max_words=2).replace(" ", "")
        if not tc:
            tc = "c0"
        if not db:
            db = "relation"
        if not at:
            at = "other"
        return f"{tc}|{db}|{at}"

    def _arm_from_key(self, key: str) -> Tuple[str, str, str]:
        raw = str(key or "").strip().lower()
        parts = raw.split("|")
        if len(parts) != 3:
            return "c0", "relation", "other"
        tc, db, at = (parts[0].strip(), parts[1].strip(), parts[2].strip())
        return tc or "c0", db or "relation", at or "other"

    def _default_curriculum_arms(self) -> List[str]:
        return [
            self._arm_key("C6", "count", "count"),
            self._arm_key("C5", "text", "text"),
            self._arm_key("C4", "material", "attribute"),
            self._arm_key("C2", "action", "attribute"),
            self._arm_key("C8", "attribute", "attribute"),
            self._arm_key("C7", "comparison", "spatial"),
            self._arm_key("C1", "relation", "attribute"),
        ]

    def _curriculum_arm_score(self, arm_key: str) -> float:
        stats = dict(getattr(self, "_curriculum_arm_stats", {}).get(arm_key, {}))
        if not stats:
            return 0.0
        all_counts = float(
            sum(float(v.get("count", 0.0)) for v in getattr(self, "_curriculum_arm_stats", {}).values())
        )
        my_count = float(stats.get("count", 0.0))
        share = my_count / max(1e-6, all_counts) if all_counts > 0.0 else 0.0
        n_arms = max(1, len(getattr(self, "_curriculum_arm_stats", {})))
        target_share = 1.0 / float(n_arms)
        under = max(0.0, target_share - share)
        progress = float(stats.get("progress_ema", 0.0))
        non_easy = float(stats.get("non_easy_ema", 0.0))
        solver_gain = float(stats.get("solver_gain_ema", 0.0))
        score = (
            float(getattr(self.cfg, "curriculum_arm_progress_weight", 0.20)) * progress
            + float(getattr(self.cfg, "curriculum_arm_underuse_weight", 0.12)) * under
            - float(getattr(self.cfg, "curriculum_arm_easy_penalty_weight", 0.15)) * max(0.0, 1.0 - non_easy)
            + float(getattr(self.cfg, "curriculum_arm_solver_gain_weight", 0.10)) * solver_gain
        )
        return float(score)

    def _sample_curriculum_arm(self) -> Dict[str, object]:
        enabled = bool(getattr(self.cfg, "curriculum_arm_enabled", True))
        arms = set(self._default_curriculum_arms())
        stats_map = getattr(self, "_curriculum_arm_stats", {})
        for key in stats_map.keys():
            if str(key).strip():
                arms.add(str(key).strip().lower())
        arm_list = sorted(arms)
        if not arm_list:
            return {"enabled": enabled, "key": "", "score": 0.0, "hint": ""}
        # Ensure stats entries exist so undersampled arms can be selected.
        for key in arm_list:
            if key not in stats_map:
                stats_map[key] = {
                    "count": 0.0,
                    "non_easy_ema": 0.0,
                    "solver_gain_ema": 0.0,
                    "prev_solver_gain_ema": 0.0,
                    "progress_ema": 0.0,
                }
        self._curriculum_arm_stats = stats_map
        if not enabled:
            tc, db, at = self._arm_from_key(arm_list[0])
            return {
                "enabled": False,
                "key": arm_list[0],
                "score": 0.0,
                "hint": f"task_card={tc.upper()}, domain={db}, answer_type={at}",
            }
        scores = [self._curriculum_arm_score(k) for k in arm_list]
        temp = max(1e-3, float(getattr(self.cfg, "curriculum_arm_prompt_temp", 0.60)))
        m = max(scores) if scores else 0.0
        exps = [math.exp((s - m) / temp) for s in scores]
        total = sum(exps)
        probs = [e / total for e in exps] if total > 0 else [1.0 / len(arm_list)] * len(arm_list)
        r = random.random()
        c = 0.0
        pick_idx = 0
        for i, p in enumerate(probs):
            c += float(p)
            if r <= c:
                pick_idx = i
                break
        key = arm_list[pick_idx]
        tc, db, at = self._arm_from_key(key)
        hint = f"task_card={tc.upper()}, domain={db}, answer_type={at}"
        return {"enabled": True, "key": key, "score": float(scores[pick_idx]), "hint": hint}

    def _update_curriculum_arm_stats(
        self,
        arm_key: str,
        non_easy: float,
        solver_gain: float,
    ) -> None:
        if not arm_key:
            return
        if not bool(getattr(self.cfg, "curriculum_arm_enabled", True)):
            return
        alpha = max(0.0, min(0.9999, float(getattr(self.cfg, "curriculum_arm_ema_momentum", 0.90))))
        stats = dict(getattr(self, "_curriculum_arm_stats", {}).get(arm_key, {}))
        prev_gain_ema = float(stats.get("solver_gain_ema", 0.0))
        new_gain_ema = alpha * prev_gain_ema + (1.0 - alpha) * float(max(0.0, solver_gain))
        progress_now = new_gain_ema - prev_gain_ema
        prev_prog = float(stats.get("progress_ema", 0.0))
        stats["count"] = float(stats.get("count", 0.0)) + 1.0
        stats["non_easy_ema"] = alpha * float(stats.get("non_easy_ema", 0.0)) + (1.0 - alpha) * float(
            max(0.0, min(1.0, non_easy))
        )
        stats["prev_solver_gain_ema"] = prev_gain_ema
        stats["solver_gain_ema"] = new_gain_ema
        stats["progress_ema"] = alpha * prev_prog + (1.0 - alpha) * float(progress_now)
        self._curriculum_arm_stats[arm_key] = stats

    def _seed_anchor_exemplars(self) -> None:
        """Pre-populate anchor replay with diverse hard question pattern seeds.

        These cover the major VLM capability dimensions — spatial reasoning,
        depth/3D, materials, actions/states, compositional reasoning,
        counting, part-whole, and fine-grained attributes.  This diversity
        ensures the proposer explores MANY types of hard questions from
        step 0, not just counting or OCR.

        Real high-STE questions from training will naturally displace these
        seeds within ~10-15 steps because they have low priority scores.
        """
        _seeds = [
            # --- Spatial reasoning ---
            ("What is immediately to the left of the tallest object on the table?",
             "H4", 0.30, 0.60),
            # --- Depth & 3D understanding ---
            ("Which object is closer to the camera: the item on the left or the one on the right side of the surface?",
             "H11", 0.30, 0.60),
            # --- Material / texture recognition ---
            ("What material does the floor in the background appear to be made of?",
             "H10", 0.30, 0.60),
            # --- Action / state of non-dominant subject ---
            ("What is the person furthest from the camera doing with their hands?",
             "H9", 0.30, 0.60),
            # --- Object state / condition ---
            ("Is the container in the background open, closed, or partially open?",
             "H14", 0.28, 0.58),
            # --- Compositional multi-hop reference ---
            ("What color is the item being held by the person who is closest to the right edge?",
             "H13", 0.30, 0.60),
            # --- Part-whole relationship ---
            ("What is mounted on or attached to the top of the vertical structure on the left?",
             "H12", 0.28, 0.58),
            # --- Counting under occlusion ---
            ("How many partially visible objects are on the shelf behind the main subject?",
             "H1", 0.28, 0.58),
            # --- Fine-grained attribute of background object ---
            ("What pattern or design is visible on the non-dominant object in the background?",
             "M4", 0.25, 0.55),
            # --- Comparison / relative judgment ---
            ("Which is larger: the object on the far left or the object nearest the center?",
             "H2", 0.25, 0.55),
        ]
        for q, strat, priority, sted in _seeds:
            self._proposer_anchor_replay.append({
                "question": q,
                "priority": priority,
                "reward": 0.15,
                "step": 0,
                "ste_difficulty": sted,
                "entropy": 0.0,
                "source": "seed",
                "qkey": "",
                "q_tokens": [],
                "strategy": strat,
                "bucket": "easy",
            })

    def _top_replay_anchor_hints(self, k: int = 3) -> List[str]:
        if not bool(getattr(self.cfg, "replay_priority_enabled", True)):
            return []
        if not self._proposer_anchor_replay:
            return []
        kk = max(0, int(k))
        if kk <= 0:
            return []
        ranked = sorted(
            list(self._proposer_anchor_replay),
            key=lambda x: (float(x.get("priority", 0.0)), float(x.get("reward", 0.0)), float(x.get("step", 0.0))),
            reverse=True,
        )
        out: List[str] = []
        seen_questions: set = set()
        for item in ranked:
            q = str(item.get("question", "") or "").strip()
            if not q or q in seen_questions:
                continue
            seen_questions.add(q)
            # Enrich the hint with WHY this question was hard.
            src = str(item.get("source", "")).strip()
            ent = float(item.get("entropy", 0.0))
            sted = float(item.get("ste_difficulty", 0.0))
            if ent > 0.05:
                reason = f"(caused solver split, H={ent:.2f})"
            elif sted > 0.5:
                reason = f"(high model uncertainty, SD={sted:.2f})"
            else:
                reason = "(effective)"
            hint = f"{q} {reason}"
            out.append(hint)
            if len(out) >= kk:
                break
        return out

    def _replay_anchor_priority(
        self,
        question: str,
        hardness_score: float,
        solver_gain: float,
    ) -> float:
        if not bool(getattr(self.cfg, "replay_priority_enabled", True)):
            return 0.0
        w_hard = max(0.0, float(getattr(self.cfg, "replay_priority_hardness_weight", 0.50)))
        w_gain = max(0.0, float(getattr(self.cfg, "replay_priority_update_weight", 0.30)))
        w_novel = max(0.0, float(getattr(self.cfg, "replay_priority_novelty_weight", 0.20)))
        qset = self._question_token_set(question)
        max_sim = 0.0
        if qset:
            for item in self._proposer_anchor_replay:
                toks = set(item.get("q_tokens", []) or [])
                if not toks:
                    toks = self._question_token_set(str(item.get("question", "") or ""))
                max_sim = max(max_sim, self._jaccard_similarity(qset, toks))
        novelty = max(0.0, 1.0 - max_sim)
        hard = max(0.0, min(1.0, float(hardness_score)))
        gain = max(0.0, min(1.0, float(solver_gain)))
        return float(w_hard * hard + w_gain * gain + w_novel * novelty)

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

    def _normalize_strategy_key(self, strategy_used: str) -> str:
        s = str(strategy_used or "").strip().lower()
        if not s:
            return ""
        s = s.split("=", 1)[0].strip()
        s = re.sub(r"[^a-z0-9_]+", "", s)
        return s

    def _difficulty_rank(self, bucket: str) -> int:
        b = str(bucket or "").strip().lower()
        if b == "hard":
            return 2
        if b == "medium":
            return 1
        return 0

    def _strategy_quota_adjustment(self, strategy_used: str) -> float:
        key = self._normalize_strategy_key(strategy_used)
        if not key:
            return 0.0
        values = [self._normalize_strategy_key(x) for x in self._strategy_window if str(x).strip()]
        values = [x for x in values if x]
        if not values:
            return 0.0
        count = sum(1 for x in values if x == key)
        share = float(count) / float(len(values))
        target = float(getattr(self.cfg, "proposer_strategy_target_share", 0.16))
        target = max(0.01, min(1.0, target))
        over_pen = max(0.0, float(getattr(self.cfg, "proposer_strategy_overuse_penalty", 0.12)))
        under_bonus = max(0.0, float(getattr(self.cfg, "proposer_strategy_underuse_bonus", 0.04)))
        if share > target:
            return -over_pen * min(1.0, (share - target) / max(1e-6, 1.0 - target))
        if share < 0.5 * target:
            return under_bonus * min(1.0, ((0.5 * target) - share) / max(1e-6, 0.5 * target))
        return 0.0

    def _template_cooldown_penalty(self, question: str) -> float:
        key = self._question_template_key(question)
        if not key:
            return 0.0
        cooldown_steps = max(0, int(getattr(self.cfg, "proposer_template_cooldown_steps", 8)))
        if cooldown_steps <= 0:
            return 0.0
        max_pen = max(0.0, float(getattr(self.cfg, "proposer_template_cooldown_penalty", 0.20)))
        if max_pen <= 0.0:
            return 0.0
        recent = list(self._question_template_window)
        for back_idx, prev_key in enumerate(reversed(recent)):
            if prev_key != key:
                continue
            if back_idx >= cooldown_steps:
                return 0.0
            scale = 1.0 - (float(back_idx) / float(cooldown_steps))
            return max_pen * max(0.0, scale)
        return 0.0

    def _anchor_replay_bonus(self, question: str, strategy_used: str) -> float:
        if not self._proposer_anchor_replay:
            return 0.0
        qkey = self._question_template_key(question)
        skey = self._normalize_strategy_key(strategy_used)
        strategy_bonus = max(0.0, float(getattr(self.cfg, "proposer_anchor_strategy_bonus", 0.06)))
        template_bonus = max(0.0, float(getattr(self.cfg, "proposer_anchor_template_bonus", 0.04)))
        bonus = 0.0
        if skey and any(item.get("strategy") == skey for item in self._proposer_anchor_replay):
            bonus += strategy_bonus
        if qkey and any(item.get("qkey") == qkey for item in self._proposer_anchor_replay):
            bonus += template_bonus
        return float(bonus)

    def _proposer_base_reward(self, entropy_nats: float, local_info_score: float, entropy_mu: float) -> float:
        gaussian_raw = gaussian_reward(
            entropy_nats,
            entropy_mu,
            self.cfg.prop_entropy_sigma,
        )
        band_raw = max(-1.0, min(1.0, (2.0 * float(local_info_score)) - 1.0))
        mode = str(getattr(self.cfg, "proposer_reward_mode", "hybrid") or "hybrid").strip().lower()
        if mode == "gaussian":
            return float(max(-1.0, min(1.0, gaussian_raw)))
        if mode == "band":
            return float(band_raw)
        band_w = float(getattr(self.cfg, "proposer_band_reward_weight", 0.70))
        band_w = max(0.0, min(1.0, band_w))
        mixed = band_w * band_raw + (1.0 - band_w) * gaussian_raw
        return float(max(-1.0, min(1.0, mixed)))

    def _get_proposer_bucket_baseline(self, bucket: str) -> float:
        b = str(bucket or "").strip().lower()
        baseline = float(self._proposer_bucket_baselines.get(b, self.proposer_baseline))
        if b == "easy":
            # Prevent baseline collapse: when ALL questions are easy the
            # easy baseline tracks the easy reward floor (≈ -0.35), making
            # advantage ≈ 0 and killing the gradient.  Cap the easy baseline
            # so that easy questions always have a meaningfully negative
            # advantage, giving the proposer a consistent downward push.
            easy_baseline_cap = float(
                getattr(self.cfg, "proposer_easy_baseline_cap", -0.05)
            )
            baseline = min(baseline, easy_baseline_cap)
        return baseline

    def _update_proposer_bucket_baseline(self, bucket: str, reward: float):
        b = str(bucket or "").strip().lower()
        if b not in {"easy", "medium", "hard"}:
            return
        m = float(getattr(self.cfg, "baseline_momentum", 0.6))
        m = max(0.0, min(0.9999, m))
        prev = float(self._proposer_bucket_baselines.get(b, self.proposer_baseline))
        self._proposer_bucket_baselines[b] = m * prev + (1.0 - m) * float(reward)

    def _update_easy_constraint(self, is_easy: bool) -> Dict[str, float]:
        obs = self._dist_mean(1.0 if is_easy else 0.0)
        mom = float(getattr(self.cfg, "easy_rate_ema_momentum", 0.97))
        mom = max(0.0, min(0.9999, mom))
        prev = float(getattr(self, "_easy_rate_ema", 0.0))
        self._easy_rate_ema = mom * prev + (1.0 - mom) * obs
        if bool(getattr(self.cfg, "easy_constraint_enabled", True)):
            target = float(getattr(self.cfg, "easy_constraint_target_rate", 0.35))
            lr = float(getattr(self.cfg, "easy_constraint_lr", 0.05))
            lam_max = max(0.0, float(getattr(self.cfg, "easy_constraint_lambda_max", 1.5)))
            lam = float(getattr(self, "_easy_lagrange_lambda", 0.0))
            lam = lam + lr * (self._easy_rate_ema - target)
            self._easy_lagrange_lambda = max(0.0, min(lam_max, lam))
        else:
            self._easy_lagrange_lambda = 0.0
        return {
            "easy_rate_ema": float(self._easy_rate_ema),
            "easy_lambda": float(self._easy_lagrange_lambda),
        }

    def _proposer_controller_state(self) -> Dict[str, float]:
        easy_rate = float(getattr(self, "_easy_rate_ema", 0.0))
        temp_boost = 0.0
        top_p_boost = 0.0
        penalty_boost = 1.0
        num_candidates_override = 0
        if bool(getattr(self.cfg, "adaptive_exploration_enabled", True)):
            thr = float(getattr(self.cfg, "exploration_easy_rate_threshold", 0.75))
            if easy_rate > thr:
                ratio = min(1.0, (easy_rate - thr) / max(1e-6, 1.0 - thr))
                temp_boost = float(getattr(self.cfg, "exploration_temp_boost_max", 0.60)) * ratio
                top_p_boost = float(getattr(self.cfg, "exploration_top_p_boost_max", 0.10)) * ratio
                penalty_boost += float(getattr(self.cfg, "exploration_penalty_boost_max", 1.0)) * ratio
        if bool(getattr(self.cfg, "hardness_debt_enabled", True)):
            debt = float(getattr(self, "_hardness_debt", 0.0))
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
                temp_boost = max(
                    temp_boost,
                    float(getattr(self.cfg, "hardness_debt_temp_boost_max", 0.30))
                    * debt_ratio,
                )
                penalty_boost += float(
                    getattr(self.cfg, "hardness_debt_penalty_boost_max", 0.30)
                ) * debt_ratio
        trigger = max(1, int(getattr(self.cfg, "collapse_streak_trigger", 8)))
        collapse_active = bool(getattr(self, "_collapse_streak", 0) >= trigger)
        if collapse_active:
            temp_boost = max(temp_boost, 0.50)
            top_p_boost = max(top_p_boost, 0.10)
            penalty_boost += max(
                0.0, float(getattr(self.cfg, "collapse_cooldown_penalty_boost", 0.10))
            )
        all_easy_trigger = max(1, int(getattr(self.cfg, "all_easy_explore_trigger", 3)))
        all_easy_steps = max(1, int(getattr(self.cfg, "all_easy_explore_steps", 10)))
        if self._all_easy_streak() >= all_easy_trigger:
            self._forced_explore_steps_left = max(
                int(getattr(self, "_forced_explore_steps_left", 0)),
                all_easy_steps,
            )
        forced_explore_active = int(getattr(self, "_forced_explore_steps_left", 0)) > 0
        if forced_explore_active:
            temp_boost = max(
                temp_boost, float(getattr(self.cfg, "all_easy_explore_temp_boost", 0.90))
            )
            top_p_boost = max(
                top_p_boost, float(getattr(self.cfg, "all_easy_explore_top_p_boost", 0.15))
            )
            penalty_boost += max(
                0.0, float(getattr(self.cfg, "all_easy_explore_penalty_boost", 0.50))
            )
            num_candidates_override = max(
                int(getattr(self.cfg, "all_easy_explore_num_candidates", 6)),
                int(getattr(self.cfg, "proposer_num_candidates", 3)),
            )
            self._forced_explore_steps_left = max(
                0, int(getattr(self, "_forced_explore_steps_left", 0)) - 1
            )
        dyn_temp = min(2.5, float(self.cfg.temp) * (1.0 + temp_boost))
        dyn_top_p = min(1.0, float(self.cfg.top_p) + top_p_boost)
        return {
            "easy_rate_ema": easy_rate,
            "temp_boost": temp_boost,
            "top_p_boost": top_p_boost,
            "penalty_boost": penalty_boost,
            "collapse_active": 1.0 if collapse_active else 0.0,
            "forced_explore_active": 1.0 if forced_explore_active else 0.0,
            "forced_explore_steps_left": float(
                max(0, int(getattr(self, "_forced_explore_steps_left", 0)))
            ),
            "num_candidates_override": float(max(0, int(num_candidates_override))),
            "proposer_temp": dyn_temp,
            "proposer_top_p": dyn_top_p,
        }

    def _all_easy_streak(self) -> int:
        streak = 0
        for v in reversed(self._all_easy_group_window):
            if float(v) >= 0.5:
                streak += 1
            else:
                break
        return int(streak)

    def _update_collapse_state(self, difficulty_bucket_observed: str, proposer_stats: Optional[Dict[str, Any]]) -> Dict[str, float]:
        has_std = False
        std_reward = 0.0
        if isinstance(proposer_stats, dict):
            try:
                std_reward = float(proposer_stats.get("std_reward", 0.0) or 0.0)
                has_std = True
            except Exception:
                std_reward = 0.0
        if has_std and math.isfinite(std_reward):
            self._grpo_std_window.append(std_reward)
        mean_std = (
            float(sum(self._grpo_std_window) / max(1, len(self._grpo_std_window)))
            if self._grpo_std_window
            else 0.0
        )
        collapse_enabled = bool(getattr(self.cfg, "collapse_detector_enabled", True))
        easy_thr = float(getattr(self.cfg, "collapse_easy_rate_threshold", 0.85))
        std_thr = max(0.0, float(getattr(self.cfg, "collapse_std_threshold", 0.06)))
        is_easy = str(difficulty_bucket_observed).strip().lower() == "easy"
        collapse_hit = (
            collapse_enabled
            and has_std
            and is_easy
            and float(getattr(self, "_easy_rate_ema", 0.0)) >= easy_thr
            and mean_std <= std_thr
        )
        if has_std:
            if collapse_hit:
                self._collapse_streak = int(getattr(self, "_collapse_streak", 0)) + 1
            else:
                self._collapse_streak = max(0, int(getattr(self, "_collapse_streak", 0)) - 1)

        trigger = max(1, int(getattr(self.cfg, "collapse_streak_trigger", 8)))
        if (
            collapse_enabled
            and self._collapse_streak >= trigger
            and bool(getattr(self.cfg, "easy_constraint_enabled", True))
        ):
            lam = float(getattr(self, "_easy_lagrange_lambda", 0.0))
            lam += max(0.0, float(getattr(self.cfg, "collapse_lambda_boost", 0.10)))
            lam_max = max(0.0, float(getattr(self.cfg, "easy_constraint_lambda_max", 1.5)))
            self._easy_lagrange_lambda = min(lam_max, lam)

        return {
            "collapse_streak": float(self._collapse_streak),
            "collapse_mean_std": float(mean_std),
            "collapse_hit": 1.0 if collapse_hit else 0.0,
        }

    def _sync_proposer_framework_state(self):
        self._easy_rate_ema = self._dist_mean(float(getattr(self, "_easy_rate_ema", 0.0)))
        self._easy_lagrange_lambda = self._dist_mean(
            float(getattr(self, "_easy_lagrange_lambda", 0.0))
        )
        for b in ("easy", "medium", "hard"):
            self._proposer_bucket_baselines[b] = self._dist_mean(
                float(self._proposer_bucket_baselines.get(b, self.proposer_baseline))
            )
        self._collapse_streak = int(
            round(self._dist_mean(float(getattr(self, "_collapse_streak", 0))))
        )
        self._hardness_debt = self._dist_mean(float(getattr(self, "_hardness_debt", 0.0)))
        self._hardness_debt_cap_streak = int(
            round(self._dist_mean(float(getattr(self, "_hardness_debt_cap_streak", 0))))
        )
        self._hardness_debt_escape_steps_left = int(
            round(
                self._dist_mean(float(getattr(self, "_hardness_debt_escape_steps_left", 0)))
            )
        )
        self._warm_start_exit_streak = int(
            round(self._dist_mean(float(getattr(self, "_warm_start_exit_streak", 0))))
        )
        self._warm_start_completed = bool(
            self._dist_mean(1.0 if bool(getattr(self, "_warm_start_completed", False)) else 0.0)
            > 0.5
        )

    def _apply_grpo_pairwise_ranking(
        self,
        rewards: List[float],
        buckets: List[str],
    ) -> Tuple[List[float], List[float]]:
        if len(rewards) <= 1:
            return list(rewards), [0.0 for _ in rewards]
        if not bool(getattr(self.cfg, "grpo_pairwise_ranking_enabled", True)):
            return list(rewards), [0.0 for _ in rewards]
        rank_w = max(0.0, float(getattr(self.cfg, "grpo_pairwise_ranking_weight", 0.08)))
        margin = max(0.0, float(getattr(self.cfg, "grpo_pairwise_margin", 0.05)))
        easy_pen = max(0.0, float(getattr(self.cfg, "grpo_pairwise_easy_penalty", 0.05)))
        if rank_w <= 0.0:
            return list(rewards), [0.0 for _ in rewards]
        adjusted = list(float(r) for r in rewards)
        deltas = [0.0 for _ in adjusted]
        n = len(adjusted)
        for i in range(n):
            for j in range(i + 1, n):
                ri = self._difficulty_rank(buckets[i] if i < len(buckets) else "easy")
                rj = self._difficulty_rank(buckets[j] if j < len(buckets) else "easy")
                if ri == rj:
                    continue
                pref = i if ri > rj else j
                other = j if pref == i else i
                gap = adjusted[pref] - adjusted[other]
                target = margin * float(abs(ri - rj))
                if gap < target:
                    boost = rank_w * (target - gap)
                    deltas[pref] += boost
                    deltas[other] -= boost
        for i, b in enumerate(buckets):
            if str(b).strip().lower() == "easy":
                deltas[i] -= easy_pen
        adjusted = [max(-1.0, min(1.0, r + d)) for r, d in zip(adjusted, deltas)]
        return adjusted, deltas

    def _apply_all_easy_relative_negatives(
        self,
        rewards: List[float],
        buckets: List[str],
    ) -> Tuple[List[float], List[float], bool]:
        if len(rewards) <= 1:
            return list(rewards), [0.0 for _ in rewards], False
        labels = [str(b).strip().lower() for b in buckets]
        if any(b != "easy" for b in labels):
            return list(rewards), [0.0 for _ in rewards], False
        easy_floor = min(
            -1e-6, float(getattr(self.cfg, "proposer_easy_reward_floor", -0.35))
        )
        spread = max(
            0.01, float(getattr(self.cfg, "proposer_all_easy_rank_spread", 0.08))
        )
        adjusted = list(float(r) for r in rewards)
        deltas = [0.0 for _ in adjusted]
        order = sorted(range(len(adjusted)), key=lambda i: adjusted[i], reverse=True)
        denom = max(1, len(order) - 1)
        for rank, idx in enumerate(order):
            target = easy_floor - spread * (float(rank) / float(denom))
            if adjusted[idx] > target:
                deltas[idx] += (target - adjusted[idx])
            else:
                # Keep existing stronger-negative signal but enforce floor.
                deltas[idx] += min(0.0, easy_floor - adjusted[idx])
        adjusted = [max(-1.0, min(1.0, r + d)) for r, d in zip(adjusted, deltas)]
        return adjusted, deltas, True

    def _init_adaptive_windows(self):
        ent_window_size = max(8, int(getattr(self.cfg, "entropy_iqr_window_size", 256)))
        diff_window_size = max(8, int(getattr(self.cfg, "difficulty_sampler_window_size", 256)))
        qhist_window_size = max(32, int(getattr(self.cfg, "proposer_question_history_size", 256)))
        strategy_window_size = max(32, int(getattr(self.cfg, "proposer_strategy_window_size", 256)))
        anchor_replay_size = max(16, int(getattr(self.cfg, "proposer_anchor_replay_size", 256)))
        contrastive_replay_size = max(
            32, int(getattr(self.cfg, "proposer_contrastive_replay_size", 256))
        )
        std_window_size = max(8, int(getattr(self.cfg, "collapse_std_window_size", 32)))
        health_window_size = max(32, int(getattr(self.cfg, "proposer_health_window_size", 256)))
        self._entropy_window = deque(maxlen=ent_window_size)
        self._difficulty_window = deque(maxlen=diff_window_size)
        self._question_template_window = deque(maxlen=qhist_window_size)
        self._answer_family_window = deque(maxlen=qhist_window_size)
        self._strategy_window = deque(maxlen=strategy_window_size)
        self._proposer_anchor_replay = deque(maxlen=anchor_replay_size)
        self._seed_anchor_exemplars()
        self._contrastive_pos_replay = deque(maxlen=contrastive_replay_size)
        self._contrastive_neg_replay = deque(maxlen=contrastive_replay_size)
        self._grpo_std_window = deque(maxlen=std_window_size)
        self._candidate_non_easy_window = deque(maxlen=health_window_size)
        self._all_easy_group_window = deque(maxlen=health_window_size)
        self._proposer_reward_clipped_window = deque(maxlen=health_window_size)
        self._selected_non_easy_window = deque(maxlen=health_window_size)
        self._solver_update_applied_window = deque(maxlen=health_window_size)
        self._curriculum_arm_stats = {}
        self._last_curriculum_arm_key = ""
        self._proposer_bucket_baselines = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
        self._easy_rate_ema = 0.0
        self._easy_lagrange_lambda = 0.0
        self._collapse_streak = 0
        self._forced_explore_steps_left = 0
        warm_exit_window = max(1, int(getattr(self.cfg, "proposer_warm_start_exit_window", 5)))
        self._warm_start_entropy_window = deque(maxlen=warm_exit_window)
        self._warm_start_exit_streak = 0
        self._warm_start_completed = False
        self._hardness_debt = 0.0
        self._hardness_debt_cap_streak = 0
        self._hardness_debt_escape_steps_left = 0
        # Solver Token Entropy (STE) rolling window for quantile normalization.
        _ste_window_size = max(
            8, int(getattr(self.cfg, "solver_token_entropy_window_size", 128))
        )
        self._ste_window: List[float] = []

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

    def _is_proposer_warm_start_active(self, u_step: int) -> bool:
        if not bool(getattr(self.cfg, "proposer_warm_start_enabled", True)):
            return False
        if bool(getattr(self, "_warm_start_completed", False)):
            return False
        max_steps = max(1, int(getattr(self.cfg, "proposer_warm_start_max_steps", 30)))
        return int(u_step) <= max_steps

    def _update_proposer_warm_start_state(self, entropy_nats: float, u_step: int) -> Dict[str, float]:
        enabled = bool(getattr(self.cfg, "proposer_warm_start_enabled", True))
        if not enabled:
            return {
                "enabled": 0.0,
                "active_next": 0.0,
                "completed": 1.0,
                "entropy_mean": 0.0,
                "exit_streak": 0.0,
                "exit_pass": 0.0,
            }
        max_steps = max(1, int(getattr(self.cfg, "proposer_warm_start_max_steps", 30)))
        exit_window = max(1, int(getattr(self.cfg, "proposer_warm_start_exit_window", 5)))
        exit_consecutive = max(
            1, int(getattr(self.cfg, "proposer_warm_start_exit_consecutive", 2))
        )
        exit_thr = max(
            0.0, float(getattr(self.cfg, "proposer_warm_start_entropy_exit_threshold", 0.10))
        )
        if int(getattr(self._warm_start_entropy_window, "maxlen", 0) or 0) != exit_window:
            self._warm_start_entropy_window = deque(
                list(self._warm_start_entropy_window)[-exit_window:], maxlen=exit_window
            )
        self._warm_start_entropy_window.append(float(entropy_nats))
        entropy_mean = (
            float(sum(float(x) for x in self._warm_start_entropy_window))
            / float(max(1, len(self._warm_start_entropy_window)))
        )
        exit_pass = bool(
            len(self._warm_start_entropy_window) >= exit_window and entropy_mean >= exit_thr
        )
        if exit_pass:
            self._warm_start_exit_streak = int(getattr(self, "_warm_start_exit_streak", 0)) + 1
        else:
            self._warm_start_exit_streak = 0
        if (
            int(u_step) >= max_steps
            or int(getattr(self, "_warm_start_exit_streak", 0)) >= exit_consecutive
        ):
            self._warm_start_completed = True
        active_next = self._is_proposer_warm_start_active(int(u_step) + 1)
        return {
            "enabled": 1.0,
            "active_next": 1.0 if active_next else 0.0,
            "completed": 1.0 if bool(getattr(self, "_warm_start_completed", False)) else 0.0,
            "entropy_mean": float(entropy_mean),
            "exit_streak": float(getattr(self, "_warm_start_exit_streak", 0)),
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
        debt = float(getattr(self, "_hardness_debt", 0.0))
        debt_max = max(1e-6, float(getattr(self.cfg, "hardness_debt_max", 6.0)))
        inc_easy = max(0.0, float(getattr(self.cfg, "hardness_debt_inc_easy", 1.5)))
        dec_non_easy = max(0.0, float(getattr(self.cfg, "hardness_debt_dec_non_easy", 1.0)))
        bucket = str(difficulty_bucket_observed or "").strip().lower()
        if bucket == "easy":
            debt += inc_easy
        else:
            debt -= dec_non_easy
        debt = max(0.0, min(debt_max, debt))
        cap_streak = int(getattr(self, "_hardness_debt_cap_streak", 0))
        if bucket == "easy" and debt >= (debt_max - 1e-8):
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
                int(getattr(self, "_hardness_debt_escape_steps_left", 0)),
                escape_steps,
            )
            cap_streak = 0
            escape_triggered = True
        self._hardness_debt = debt
        self._hardness_debt_cap_streak = cap_streak
        return {
            "enabled": 1.0,
            "debt": float(debt),
            "cap_streak": float(cap_streak),
            "escape_steps_left": float(
                max(0, int(getattr(self, "_hardness_debt_escape_steps_left", 0)))
            ),
            "escape_triggered": 1.0 if escape_triggered else 0.0,
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

        debt = float(getattr(self, "_hardness_debt", 0.0))
        debt_ratio = 0.0
        debt_escape_active = False
        if bool(getattr(self.cfg, "hardness_debt_enabled", True)):
            weights_for_sampling = self._normalize_bucket_weights(weights_for_sampling)
            if int(getattr(self, "_hardness_debt_escape_steps_left", 0)) > 0:
                debt_escape_active = True
                weights_for_sampling = self._normalize_bucket_weights(
                    {
                        "easy": float(getattr(self.cfg, "hardness_debt_stale_easy_weight", 0.05)),
                        "medium": float(
                            getattr(self.cfg, "hardness_debt_stale_medium_weight", 0.55)
                        ),
                        "hard": float(getattr(self.cfg, "hardness_debt_stale_hard_weight", 0.40)),
                    }
                )
                self._hardness_debt_escape_steps_left = max(
                    0, int(getattr(self, "_hardness_debt_escape_steps_left", 0)) - 1
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
                            "easy": float(
                                getattr(self.cfg, "hardness_debt_recovery_easy_weight", 0.0)
                            ),
                            "medium": float(
                                getattr(self.cfg, "hardness_debt_recovery_medium_weight", 0.30)
                            ),
                            "hard": float(
                                getattr(self.cfg, "hardness_debt_recovery_hard_weight", 0.70)
                            ),
                        }
                    )
                    mixed = {
                        key: ((1.0 - debt_ratio) * float(weights_for_sampling.get(key, 0.0)))
                        + (debt_ratio * float(recovery_weights.get(key, 0.0)))
                        for key in ("easy", "medium", "hard")
                    }
                    weights_for_sampling = self._normalize_bucket_weights(mixed)
                    mode = f"{mode}+debt_recovery"

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
            "hardness_debt": float(debt),
            "hardness_debt_ratio": float(debt_ratio),
            "hardness_debt_escape_active": bool(debt_escape_active),
        }

    def _mean_recent(self, values: deque, count: int) -> float:
        if not values:
            return 0.0
        n = max(1, min(int(count), len(values)))
        tail = list(values)[-n:]
        return float(sum(float(v) for v in tail) / float(n))

    def _sum_recent(self, values: deque, count: int) -> float:
        if not values:
            return 0.0
        n = max(1, min(int(count), len(values)))
        tail = list(values)[-n:]
        return float(sum(float(v) for v in tail))

    def _early_failfast_state(self, step: int, collapse_state: Dict[str, float]) -> Dict[str, float]:
        enabled = bool(getattr(self.cfg, "proposer_early_failfast_enabled", True))
        state: Dict[str, float] = {
            "enabled": 1.0 if enabled else 0.0,
            "u_step": 0.0,
            "stage1_pass": 1.0,
            "stage2_pass": 1.0,
            "stage1_active": 0.0,
            "stage2_active": 0.0,
            "candidate_non_easy_rate": 0.0,
            "all_easy_group_rate": 0.0,
            "reward_clipped_rate": 0.0,
            "selected_non_easy_rate": 0.0,
            "solver_update_applied_count": 0.0,
            "recovery_armed": 0.0,
            "triggered": 0.0,
            "hard_stop_min_u_step": 0.0,
        }
        if not enabled:
            return state

        u_step = self._phase_local_step_index(step, "understanding")
        state["u_step"] = float(max(0, u_step))
        if u_step <= 0:
            return state

        step1 = max(1, int(getattr(self.cfg, "proposer_early_step1", 12)))
        step2 = max(step1, int(getattr(self.cfg, "proposer_early_step2", 20)))
        cand_non_easy_min = max(
            0.0, min(1.0, float(getattr(self.cfg, "proposer_early_candidate_non_easy_min", 0.25)))
        )
        all_easy_max = max(
            0.0, min(1.0, float(getattr(self.cfg, "proposer_early_all_easy_rate_max", 0.70)))
        )
        clipped_max = max(
            0.0, min(1.0, float(getattr(self.cfg, "proposer_early_reward_clipped_rate_max", 0.60)))
        )
        selected_non_easy_min = max(
            0.0, min(1.0, float(getattr(self.cfg, "proposer_early_selected_non_easy_min", 0.15)))
        )
        solver_updates_min = max(0, int(getattr(self.cfg, "proposer_early_solver_updates_min", 2)))
        collapse_max = max(0, int(getattr(self.cfg, "proposer_early_collapse_streak_max", 5)))
        hard_stop_min_u_step = max(
            0, int(getattr(self.cfg, "proposer_early_hard_stop_min_u_step", 80))
        )
        state["hard_stop_min_u_step"] = float(hard_stop_min_u_step)

        state["candidate_non_easy_rate"] = self._dist_mean(
            self._mean_recent(self._candidate_non_easy_window, u_step)
        )
        state["all_easy_group_rate"] = self._dist_mean(
            self._mean_recent(self._all_easy_group_window, u_step)
        )
        state["reward_clipped_rate"] = self._dist_mean(
            self._mean_recent(self._proposer_reward_clipped_window, u_step)
        )
        state["selected_non_easy_rate"] = self._dist_mean(
            self._mean_recent(self._selected_non_easy_window, u_step)
        )
        state["solver_update_applied_count"] = self._dist_mean(
            self._sum_recent(self._solver_update_applied_window, u_step)
        )

        collapse_streak = int(
            round(self._dist_mean(float(collapse_state.get("collapse_streak", 0.0))))
        )

        if u_step >= step1:
            state["stage1_active"] = 1.0
            stage1_pass = (
                state["candidate_non_easy_rate"] >= cand_non_easy_min
                and state["all_easy_group_rate"] <= all_easy_max
                and state["reward_clipped_rate"] <= clipped_max
            )
            state["stage1_pass"] = 1.0 if stage1_pass else 0.0
            if not stage1_pass:
                state["triggered"] = 1.0

        if u_step >= step2:
            state["stage2_active"] = 1.0
            stage2_pass = (
                state["selected_non_easy_rate"] >= selected_non_easy_min
                and state["solver_update_applied_count"] >= float(solver_updates_min)
                and collapse_streak <= collapse_max
            )
            state["stage2_pass"] = 1.0 if stage2_pass else 0.0
            if not stage2_pass:
                state["triggered"] = 1.0

        if state["triggered"] > 0.5:
            recover_enabled = bool(
                getattr(self.cfg, "proposer_early_failfast_recover", True)
            )
            if recover_enabled:
                recover_steps = max(
                    1,
                    int(getattr(self.cfg, "proposer_early_failfast_recover_steps", 20)),
                )
                self._forced_explore_steps_left = max(
                    int(getattr(self, "_forced_explore_steps_left", 0)),
                    recover_steps,
                )
                state["recovery_armed"] = 1.0
            if (
                bool(getattr(self.cfg, "proposer_early_failfast_stop", True))
                and (not recover_enabled)
                and (u_step >= hard_stop_min_u_step)
            ):
                msg = (
                    "[EarlyFailFast] unhealthy run detected: "
                    f"u_step={u_step} cand_non_easy_rate={state['candidate_non_easy_rate']:.3f} "
                    f"all_easy_rate={state['all_easy_group_rate']:.3f} "
                    f"reward_clipped_rate={state['reward_clipped_rate']:.3f} "
                    f"selected_non_easy_rate={state['selected_non_easy_rate']:.3f} "
                    f"solver_updates={state['solver_update_applied_count']:.1f} "
                    f"collapse_streak={collapse_streak}"
                )
                raise RuntimeError(msg)

        return state

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
        u_step = max(0, self._phase_local_step_index(step, "understanding"))
        proposer_warm_start_active = self._is_proposer_warm_start_active(max(1, u_step))
        warm_start_state: Dict[str, float] = {
            "enabled": 1.0
            if bool(getattr(self.cfg, "proposer_warm_start_enabled", True))
            else 0.0,
            "active_next": 1.0 if proposer_warm_start_active else 0.0,
            "completed": 0.0 if proposer_warm_start_active else 1.0,
            "entropy_mean": 0.0,
            "exit_streak": 0.0,
            "exit_pass": 0.0,
        }
        hardness_debt_state: Dict[str, float] = {
            "enabled": 1.0 if bool(getattr(self.cfg, "hardness_debt_enabled", True)) else 0.0,
            "debt": float(getattr(self, "_hardness_debt", 0.0)),
            "cap_streak": float(getattr(self, "_hardness_debt_cap_streak", 0)),
            "escape_steps_left": float(getattr(self, "_hardness_debt_escape_steps_left", 0)),
            "escape_triggered": 0.0,
        }
        solver_temperatures = self._solver_temperature_schedule()
        solver_top_ps = self._solver_top_p_schedule()
        controller_state = self._proposer_controller_state()
        proposer_temp_ctrl = float(controller_state.get("proposer_temp", self.cfg.temp))
        proposer_top_p_ctrl = float(controller_state.get("proposer_top_p", self.cfg.top_p))
        proposer_penalty_boost = max(1.0, float(controller_state.get("penalty_boost", 1.0)))
        controller_num_candidates = max(
            0, int(round(float(controller_state.get("num_candidates_override", 0.0))))
        )
        easy_selection_scale = max(
            0.0, float(getattr(self.cfg, "easy_constraint_selection_scale", 0.20))
        )
        easy_constraint_enabled = bool(getattr(self.cfg, "easy_constraint_enabled", True))
        easy_reward_floor = min(
            -1e-6, float(getattr(self.cfg, "proposer_easy_reward_floor", -0.35))
        )

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
        if controller_num_candidates > 0:
            num_proposer_candidates = max(num_proposer_candidates, controller_num_candidates)
        # How many solver samples to use for the spot-check of each candidate.
        # 3 samples give ternary entropy outcomes (0, 0.637, 1.099).
        spot_check_samples = max(
            1, int(getattr(self.cfg, "proposer_spot_check_samples", 3))
        )
        _pps_enabled = bool(getattr(self.cfg, "solver_pps_enabled", True))
        spot_entropy_min_gate = max(
            0.0, float(getattr(self.cfg, "proposer_spot_entropy_min_gate", 0.05))
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
        solver_use_forced_choice_from_proposer = bool(
            getattr(self.cfg, "solver_use_forced_choice_from_proposer", False)
        )
        curriculum_arm_state = self._sample_curriculum_arm()
        curriculum_arm_key = str(curriculum_arm_state.get("key", "") or "")
        curriculum_arm_hint = (
            str(curriculum_arm_state.get("hint", "") or "")
            if bool(getattr(self.cfg, "curriculum_arm_prompt_enabled", True))
            else ""
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
        # Always inject anchor hints (not just during easy streaks).
        # The proposer needs exemplars of what hard questions look like
        # from step 1.  STE-qualified questions fill the buffer even when
        # entropy is 0, so hints are available from early training.
        replay_anchor_hints: List[str] = self._top_replay_anchor_hints(
            int(getattr(self.cfg, "replay_anchor_inject_k", 3))
        )

        multi_proposer_prompt = build_proposer_multi_prompt(
            target_difficulty=desired_difficulty_bucket,
            num_questions=num_proposer_candidates,
            image_source_hint=_src_hint,
            curriculum_arm_hint=curriculum_arm_hint,
            replay_anchor_hints=replay_anchor_hints,
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
        chosen_reasoning_domains = ""
        chosen_reasoning_chain = ""
        chosen_task_card = ""
        proposer_text_hardness_bonus = 0.0
        proposer_strategy_quota_bonus = 0.0
        proposer_anchor_bonus = 0.0
        proposer_contrastive_replay_bonus = 0.0
        proposer_repetition_penalty = 0.0
        proposer_cooldown_penalty = 0.0
        easy_constraint_penalty = 0.0

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
            temperature=proposer_temp_ctrl,
            top_p=proposer_top_p_ctrl,
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
        best_choice_mode = False
        best_choice_option_a = ""
        best_choice_option_b = ""
        best_accept_question = ""
        best_accept_outputs: List[str] = []
        best_accept_answers_raw: List[str] = []
        best_accept_answers_norm: List[str] = []
        best_accept_pre_words: List[int] = []
        best_accept_entropy = -1.0
        best_accept_margin = 1.0
        best_accept_meta: Dict[str, str] = {}
        best_accept_pick_score = -1e9
        best_accept_choice_mode = False
        best_accept_choice_option_a = ""
        best_accept_choice_option_b = ""
        best_valid_question = ""
        best_valid_outputs: List[str] = []
        best_valid_answers_raw: List[str] = []
        best_valid_answers_norm: List[str] = []
        best_valid_pre_words: List[int] = []
        best_valid_entropy = -1.0
        best_valid_margin = 1.0
        best_valid_meta: Dict[str, str] = {}
        best_valid_pick_score = -1e9
        best_valid_choice_mode = False
        best_valid_choice_option_a = ""
        best_valid_choice_option_b = ""
        candidate_bucket_list: List[str] = []
        candidate_valid_list: List[float] = []
        candidate_cert_list: List[float] = []
        candidate_low_info_rate_list: List[float] = []
        candidate_noncanonical_rate_list: List[float] = []
        candidate_non_easy_rate = 0.0
        candidate_struct_valid_rate = 0.0
        all_easy_candidate_group = True
        spot_all_easy_low_entropy = False
        strict_slot_compiler = bool(getattr(self.cfg, "proposer_slot_compiler_strict", True))

        for cand_idx, cand_q in enumerate(candidate_questions):
            cand_q = cand_q.replace("\n", " ").strip()
            if not cand_q:
                continue
            cand_meta = candidate_infos[cand_idx] if cand_idx < len(candidate_infos) else {"text": cand_q}
            cand_q_compiled, cand_compile_ok, cand_compile_reason = self._compile_question_from_slots(
                cand_q,
                cand_meta,
            )
            cand_meta = dict(cand_meta)
            cand_meta["_compiler_valid"] = "1" if cand_compile_ok else "0"
            cand_meta["_compiler_reason"] = cand_compile_reason
            if cand_compile_ok and cand_q_compiled:
                cand_q = cand_q_compiled
            cand_strategy = str(cand_meta.get("strategy_used", "") or "")
            cand_two_answer = str(cand_meta.get("two_answer_test", "") or "")
            cand_text_bonus = self._proposer_text_hardness_bonus(
                cand_q,
                cand_strategy,
                cand_two_answer,
            )
            cand_strategy_bonus = self._strategy_quota_adjustment(cand_strategy)
            cand_anchor_bonus = self._anchor_replay_bonus(cand_q, cand_strategy)
            cand_cooldown_penalty = self._template_cooldown_penalty(cand_q) * proposer_penalty_boost

            cand_non_objective = bool(
                require_objective and (not self._is_objective_question(cand_q))
            )
            cand_cert = self._proposer_certificate_score(cand_q, cand_meta)
            cand_cert_score = float(cand_cert.get("score", 0.0))
            cand_cert_valid = bool(cand_cert.get("valid", 0.0) > 0.5)
            if (not cand_compile_ok) and bool(getattr(self.cfg, "proposer_slot_compiler_strict", True)):
                cand_cert_valid = False
                cand_cert_score = min(cand_cert_score, 0.0)
            cert_weight = max(0.0, float(getattr(self.cfg, "proposer_certificate_weight", 0.75)))
            cert_entropy_floor = max(
                0.0, min(1.0, float(getattr(self.cfg, "proposer_certificate_entropy_floor", 0.10)))
            )
            cand_contrastive_replay = self._contrastive_replay_adjustment(cand_q)
            cand_option_a, cand_option_b = self._extract_forced_choice_options(cand_two_answer)
            cand_choice_mode = bool(
                solver_use_forced_choice_from_proposer and cand_option_a and cand_option_b
            )
            cand_answer_type = self._question_answer_type(cand_q)
            cand_outputs: List[str] = []
            cand_answers_raw: List[str] = []
            cand_answers_norm: List[str] = []
            cand_vote_labels: List[str] = []
            cand_low_info_flags: List[bool] = []
            cand_noncanonical_flags: List[bool] = []
            cand_pre_words: List[int] = []

            # ---------------------------------------------------------
            # STE-first spot-check: ONE greedy call with token entropy
            # instead of multiple temperature samples.  STE provides a
            # continuous difficulty signal that has no dead zone, unlike
            # sample entropy which is almost always 0 for easy questions.
            # This reduces spot-check cost from 3 calls → 1 call per
            # candidate while giving a BETTER difficulty ranking.
            # ---------------------------------------------------------
            _ste_spot_prompt = build_solver_prompt(cand_q, focus_hint=self._solver_focus_hint(0))
            _ste_spot_tokens = max(1, int(getattr(self.cfg, "solver_token_entropy_tokens", 5)))
            _ste_spot_out, _ste_spot_conf = self._generate_with_confidence(
                image=image,
                prompt=_ste_spot_prompt,
                adapter_name="default" if self.cfg.use_lora else None,
                max_new_tokens=self.cfg.max_new_tokens_solver,
                margin_tokens=_ste_spot_tokens,
            )
            sc_ans_raw = _parse_answer(_ste_spot_out)
            sc_ans_text = normalize_answer(sc_ans_raw)
            sc_vote, sc_low_info, sc_noncanonical = self._normalize_answer_for_type(
                sc_ans_text, cand_answer_type,
            )
            cand_outputs.append(_ste_spot_out)
            cand_answers_raw.append(sc_ans_raw)
            cand_answers_norm.append(sc_ans_text)
            cand_vote_labels.append(sc_vote)
            cand_low_info_flags.append(sc_low_info)
            cand_noncanonical_flags.append(sc_noncanonical)
            cand_pre_words.append(pre_answer_word_count(_ste_spot_out))

            # STE difficulty: use max token entropy as difficulty proxy.
            # Higher max_entropy → model is more uncertain → harder question.
            cand_ste_max = float(_ste_spot_conf.get("max_entropy", 0.0))
            cand_ste_mean = float(_ste_spot_conf.get("mean_entropy", 0.0))

            # Map STE to pseudo-entropy for compatibility with existing
            # bucket classification.  Use the STE window for calibration
            # if available, otherwise use a sigmoid.
            _ste_window_vals = list(getattr(self, "_ste_window", []))
            if len(_ste_window_vals) >= 8:
                _ste_rank = sum(1 for e in _ste_window_vals if e < cand_ste_max)
                _ste_quantile = float(_ste_rank) / float(len(_ste_window_vals))
            else:
                import math as _math
                _sig_a = float(getattr(self.cfg, "solver_token_entropy_sigmoid_alpha", 1.5))
                _sig_b = float(getattr(self.cfg, "solver_token_entropy_sigmoid_beta", 2.0))
                _ste_quantile = 1.0 / (1.0 + _math.exp(-_sig_a * (cand_ste_max - _sig_b)))

            # Use STE quantile to determine easy/non-easy.
            # Candidates above the 30th percentile are considered non-easy.
            _ste_easy_threshold = max(
                0.0, float(getattr(self.cfg, "ste_spot_easy_quantile", 0.30))
            )
            sc_entropy = cand_ste_max  # use raw STE as entropy proxy for scoring
            sc_margin = 1.0 - _ste_quantile  # invert: high quantile → low margin
            sc_maj_frac = 1.0  # single sample → always unanimous
            cand_low_info_rate = 1.0 if sc_low_info else 0.0
            cand_noncanonical_rate = 1.0 if sc_noncanonical else 0.0
            cand_quality_penalty = (
                cand_noncanonical_rate
                * max(0.0, float(getattr(self.cfg, "proposer_candidate_noncanonical_penalty", 0.12)))
                + cand_low_info_rate
                * max(0.0, float(getattr(self.cfg, "proposer_candidate_low_info_penalty", 0.10)))
            )
            cand_meta["_spot_low_info_rate"] = f"{cand_low_info_rate:.6f}"
            cand_meta["_spot_noncanonical_rate"] = f"{cand_noncanonical_rate:.6f}"
            cand_meta["_ste_max_entropy"] = f"{cand_ste_max:.6f}"
            cand_meta["_ste_quantile"] = f"{_ste_quantile:.6f}"

            # Classify difficulty using STE quantile instead of sample entropy.
            sc_bucket = "easy" if _ste_quantile < _ste_easy_threshold else (
                "hard" if _ste_quantile > 0.70 else "medium"
            )
            sc_is_easy = bool(_ste_quantile < _ste_easy_threshold)
            cand_bucket_bonus = 0.06 if sc_bucket == desired_difficulty_bucket else 0.0
            if sc_bucket == "easy":
                cand_bucket_bonus -= 0.03
            cand_bonus_enabled = (sc_bucket in {"medium", "hard"}) or proposer_warm_start_active
            cand_text_bonus_used = cand_text_bonus if cand_bonus_enabled else 0.0
            cand_strategy_bonus_used = cand_strategy_bonus if cand_bonus_enabled else 0.0
            cand_anchor_bonus_used = cand_anchor_bonus if cand_bonus_enabled else 0.0
            cand_cert_weight_used = cert_weight if cand_bonus_enabled else 0.0
            cand_easy_constraint_penalty = 0.0
            if easy_constraint_enabled and sc_bucket == "easy":
                cand_easy_constraint_penalty = (
                    float(getattr(self, "_easy_lagrange_lambda", 0.0)) * easy_selection_scale
                )
            cand_sc_core = (
                max(cert_entropy_floor, sc_entropy)
                + cand_text_bonus_used
                + cand_strategy_bonus_used
                + cand_anchor_bonus_used
                + cand_bucket_bonus
                + cand_contrastive_replay
                - cand_cooldown_penalty
                - cand_easy_constraint_penalty
                - cand_quality_penalty
            )
            cand_pick_score = cand_sc_core * (1.0 + cand_cert_weight_used * cand_cert_score)
            candidate_bucket_list.append(sc_bucket)
            candidate_valid_list.append(1.0 if cand_cert_valid else 0.0)
            candidate_cert_list.append(cand_cert_score)
            candidate_low_info_rate_list.append(cand_low_info_rate)
            candidate_noncanonical_rate_list.append(cand_noncanonical_rate)

            # Always remember the best candidate seen so far.
            if (not strict_slot_compiler) or cand_cert_valid:
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
                    best_choice_mode = cand_choice_mode
                    best_choice_option_a = cand_option_a
                    best_choice_option_b = cand_option_b

            # Keep the best acceptable (non-easy, objective) candidate instead
            # of taking the first. This removes proposer ordering bias.
            if (not sc_is_easy) and (not cand_non_objective) and cand_cert_valid:
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
                    best_accept_choice_mode = cand_choice_mode
                    best_accept_choice_option_a = cand_option_a
                    best_accept_choice_option_b = cand_option_b

            # Keep the best structurally valid candidate even when all are easy.
            if (not cand_non_objective) and cand_cert_valid:
                if (
                    (cand_pick_score > best_valid_pick_score)
                    or (
                        abs(cand_pick_score - best_valid_pick_score) <= 1e-8
                        and (
                            sc_entropy > best_valid_entropy
                            or (
                                abs(sc_entropy - best_valid_entropy) <= 1e-8
                                and sc_margin < best_valid_margin
                            )
                        )
                    )
                ):
                    best_valid_pick_score = cand_pick_score
                    best_valid_entropy = sc_entropy
                    best_valid_margin = sc_margin
                    best_valid_question = cand_q
                    best_valid_outputs = cand_outputs
                    best_valid_answers_raw = cand_answers_raw
                    best_valid_answers_norm = cand_answers_norm
                    best_valid_pre_words = cand_pre_words
                    best_valid_meta = dict(cand_meta)
                    best_valid_choice_mode = cand_choice_mode
                    best_valid_choice_option_a = cand_option_a
                    best_valid_choice_option_b = cand_option_b

        if candidate_bucket_list:
            candidate_non_easy_rate = float(
                sum(1.0 for b in candidate_bucket_list if b != "easy")
                / float(len(candidate_bucket_list))
            )
            candidate_struct_valid_rate = float(
                sum(candidate_valid_list) / float(len(candidate_valid_list))
            )
            all_easy_candidate_group = bool(all(b == "easy" for b in candidate_bucket_list))
        else:
            all_easy_candidate_group = True
        candidate_low_info_rate_mean = (
            float(sum(candidate_low_info_rate_list) / max(1, len(candidate_low_info_rate_list)))
            if candidate_low_info_rate_list
            else 0.0
        )
        candidate_noncanonical_rate_mean = (
            float(sum(candidate_noncanonical_rate_list) / max(1, len(candidate_noncanonical_rate_list)))
            if candidate_noncanonical_rate_list
            else 0.0
        )

        # Minimum spot-check entropy gate: if all candidates are easy and even
        # the best spot-check entropy is near zero, force exploration for
        # upcoming steps to avoid repeatedly selecting "best-of-easy".
        if (
            all_easy_candidate_group
            and best_entropy >= 0.0
            and best_entropy < spot_entropy_min_gate
        ):
            spot_all_easy_low_entropy = True
            candidate_non_easy_rate = 0.0
            self._forced_explore_steps_left = max(
                int(getattr(self, "_forced_explore_steps_left", 0)),
                max(1, int(getattr(self.cfg, "all_easy_explore_steps", 10))),
            )

        selected_choice_mode = False
        selected_choice_option_a = ""
        selected_choice_option_b = ""
        selected_visual_target = ""
        if best_accept_question:
            question = best_accept_question
            solver_outputs = best_accept_outputs
            solver_answers_raw = best_accept_answers_raw
            solver_answers_norm = best_accept_answers_norm
            pre_words = best_accept_pre_words
            chosen_strategy_used = str(best_accept_meta.get("strategy_used", "") or "")
            chosen_two_answer_test = str(best_accept_meta.get("two_answer_test", "") or "")
            chosen_reasoning_domains = str(best_accept_meta.get("reasoning_domains", "") or "")
            chosen_reasoning_chain = str(best_accept_meta.get("reasoning_chain", "") or "")
            chosen_task_card = str(best_accept_meta.get("task_card", "") or "")
            selected_visual_target = str(best_accept_meta.get("visual_target", "") or "")
            selected_choice_mode = best_accept_choice_mode
            selected_choice_option_a = best_accept_choice_option_a
            selected_choice_option_b = best_accept_choice_option_b
        elif best_valid_question:
            question = best_valid_question
            solver_outputs = best_valid_outputs
            solver_answers_raw = best_valid_answers_raw
            solver_answers_norm = best_valid_answers_norm
            pre_words = best_valid_pre_words
            chosen_strategy_used = str(best_valid_meta.get("strategy_used", "") or "")
            chosen_two_answer_test = str(best_valid_meta.get("two_answer_test", "") or "")
            chosen_reasoning_domains = str(best_valid_meta.get("reasoning_domains", "") or "")
            chosen_reasoning_chain = str(best_valid_meta.get("reasoning_chain", "") or "")
            chosen_task_card = str(best_valid_meta.get("task_card", "") or "")
            selected_visual_target = str(best_valid_meta.get("visual_target", "") or "")
            selected_choice_mode = best_valid_choice_mode
            selected_choice_option_a = best_valid_choice_option_a
            selected_choice_option_b = best_valid_choice_option_b
        else:
            # No candidate cleared the gate — use the best-entropy one found.
            fallback_used = True
            if best_question:
                question = best_question
            else:
                seed_q = candidate_questions[0].replace("\n", " ").strip() if candidate_questions else ""
                seed_meta = candidate_infos[0] if candidate_infos else {}
                seed_alts = self._split_two_answer_test(str(seed_meta.get("two_answer_test", "") or ""))
                question = self._synthesize_grounded_question(seed_q, seed_meta, "", seed_alts)
                if (not question) and seed_q:
                    question = seed_q
            if question and (not question.endswith("?")):
                question = question + "?"
            if not question:
                question = "How many partially visible objects are near the center?"
                fallback_used = True
            solver_outputs = best_outputs
            solver_answers_raw = best_answers_raw
            solver_answers_norm = best_answers_norm
            pre_words = best_pre_words
            chosen_strategy_used = str(best_meta.get("strategy_used", "") or "")
            chosen_two_answer_test = str(best_meta.get("two_answer_test", "") or "")
            chosen_reasoning_domains = str(best_meta.get("reasoning_domains", "") or "")
            chosen_reasoning_chain = str(best_meta.get("reasoning_chain", "") or "")
            chosen_task_card = str(best_meta.get("task_card", "") or "")
            selected_visual_target = str(best_meta.get("visual_target", "") or "")
            selected_choice_mode = best_choice_mode
            selected_choice_option_a = best_choice_option_a
            selected_choice_option_b = best_choice_option_b

        selected_cert = self._proposer_certificate_score(
            question,
            {
                "strategy_used": chosen_strategy_used,
                "two_answer_test": chosen_two_answer_test,
                "visual_target": selected_visual_target,
                "reasoning_domains": chosen_reasoning_domains,
                "reasoning_chain": chosen_reasoning_chain,
            },
        )
        selected_cert_score = float(selected_cert.get("score", 0.0))
        selected_domain_bucket = self._arm_domain_bucket(chosen_reasoning_domains)
        selected_answer_type = self._question_answer_type(question)
        selected_arm_key = self._arm_key(
            chosen_task_card,
            selected_domain_bucket,
            selected_answer_type,
        )
        if (not str(chosen_task_card or "").strip()) and curriculum_arm_key:
            selected_arm_key = curriculum_arm_key
        curriculum_arm_score = self._curriculum_arm_score(selected_arm_key)
        curriculum_arm_reward_bonus = (
            max(0.0, float(getattr(self.cfg, "curriculum_arm_reward_scale", 0.10)))
            * curriculum_arm_score
        )

        # Solver is always prompted in free-form mode.
        # Proposer two-answer alternatives are used only as a hidden scorer.
        solver_prompt = build_solver_prompt(question)
        # --- Prompt-Perturbed Sampling (PPS) ---
        # When PPS is enabled, each of the N solver samples uses a DIFFERENT
        # prompt template (same question, different preamble framing).  This
        # makes entropy measure ROBUSTNESS of understanding rather than
        # stochastic variation from temperature sampling.
        if len(solver_answers_norm) < self.cfg.num_solver_samples:
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
                if _pps_enabled:
                    # PPS: each sample gets a different prompt template
                    sample_solver_prompt = build_solver_prompt_pps(
                        question,
                        template_index=sample_idx,
                        focus_hint=self._solver_focus_hint(sample_idx),
                    )
                else:
                    # Fallback: original single-template behavior
                    sample_solver_prompt = build_solver_prompt(
                        question,
                        focus_hint=self._solver_focus_hint(sample_idx),
                    )
                solver_out = self._generate(
                    image=image,
                    prompt=sample_solver_prompt,
                    adapter_name="default" if self.cfg.use_lora else None,
                    max_new_tokens=self.cfg.max_new_tokens_solver,
                    temperature=solver_temp,
                    top_p=solver_top_p,
                )
                answer_raw = _parse_answer(solver_out)
                answer_norm = normalize_answer(answer_raw)
                solver_outputs.append(solver_out)
                solver_answers_raw.append(answer_raw)
                solver_answers_norm.append(answer_norm)
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

        solver_choice_mode = bool(
            solver_use_forced_choice_from_proposer
            and selected_choice_option_a
            and selected_choice_option_b
        )
        solver_answer_type = self._question_answer_type(question)
        solver_vote_labels: List[str] = []
        solver_low_info_flags: List[bool] = []
        solver_noncanonical_flags: List[bool] = []
        for ans_norm in solver_answers_norm:
            if solver_choice_mode:
                vote_label = ans_norm
                low_info_flag = False
                noncanonical_flag = False
            else:
                vote_label, low_info_flag, noncanonical_flag = self._normalize_answer_for_type(
                    ans_norm,
                    solver_answer_type,
                )
            solver_vote_labels.append(vote_label or "ood")
            solver_low_info_flags.append(bool(low_info_flag))
            solver_noncanonical_flags.append(bool(noncanonical_flag))
        raw_majority_answer, _ = majority_vote(solver_answers_norm)
        canonical_majority_answer, _ = majority_vote(solver_vote_labels)
        maj_answer_vote, maj_count = majority_vote(solver_vote_labels)
        _vote_count = max(1, len(solver_vote_labels))
        maj_frac = maj_count / float(_vote_count)
        hist: Dict[str, int] = {}
        for ans in solver_vote_labels:
            hist[ans] = hist.get(ans, 0) + 1
        probs = [count / float(_vote_count) for count in hist.values()]
        entropy_nats = shannon_entropy_nats(probs)

        maj_answer = maj_answer_vote
        solver_low_info_rate = (
            float(sum(1.0 for x in solver_low_info_flags if x)) / float(max(1, len(solver_low_info_flags)))
        )
        solver_noncanonical_rate = (
            float(sum(1.0 for x in solver_noncanonical_flags if x))
            / float(max(1, len(solver_noncanonical_flags)))
        )

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
        solver_low_info_majority = self._is_low_info_majority_answer(
            question,
            canonical_majority_answer or raw_majority_answer,
        )
        if solver_low_info_rate >= 0.5:
            solver_low_info_majority = True
        self._entropy_window.append(float(entropy_nats))
        self._difficulty_window.append(difficulty_bucket_observed)
        hardness_debt_state = self._update_hardness_debt(difficulty_bucket_observed)
        warm_start_state = self._update_proposer_warm_start_state(
            entropy_nats=entropy_nats,
            u_step=max(1, u_step),
        )

        easy_solver_penalty_scale = max(
            0.0, float(getattr(self.cfg, "easy_solver_penalty_scale", 1.0))
        )

        # --- Solver rewards ---
        if easy_solver_case:
            easy_majority_penalty = easy_solver_penalty_scale * sc_signal
            if solver_low_info_majority:
                easy_majority_penalty *= max(
                    1.0,
                    float(getattr(self.cfg, "solver_low_info_easy_penalty_scale", 2.0)),
                )
            solver_rewards_raw = [
                (-easy_majority_penalty)
                if ans == maj_answer_vote
                else (neg_weight * sc_signal)
                for ans in solver_vote_labels
            ]
        elif unsolvable_solver_case:
            solver_rewards_raw = [
                -neg_weight * sc_signal
                for _ in solver_vote_labels
            ]
        else:
            solver_rewards_raw = [
                sc_signal if ans == maj_answer_vote else (-neg_weight * sc_signal)
                for ans in solver_vote_labels
            ]
        solver_noncanonical_answer_penalty = max(
            0.0, float(getattr(self.cfg, "solver_noncanonical_answer_penalty", 0.10))
        )
        solver_low_info_answer_penalty = max(
            0.0, float(getattr(self.cfg, "solver_low_info_answer_penalty", 0.08))
        )
        solver_quality_penalties = [
            (solver_noncanonical_answer_penalty if noncanon else 0.0)
            + (solver_low_info_answer_penalty if low_info else 0.0)
            for noncanon, low_info in zip(solver_noncanonical_flags, solver_low_info_flags)
        ]
        solver_rewards_raw = [
            float(r) - float(p) for r, p in zip(solver_rewards_raw, solver_quality_penalties)
        ]

        target_w = max(1, self.cfg.len_penalty_target_words)
        penalties = [min(1.0, max(0.0, (w - target_w) / float(target_w))) for w in pre_words]
        prob_map = {ans: count / float(self.cfg.num_solver_samples) for ans, count in hist.items()}
        solver_probs = [prob_map[ans] for ans in solver_vote_labels]
        solver_rewards_soft = [
            (prob ** self.cfg.solver_soft_gamma) * (1.0 - self.cfg.len_penalty_weight * pen)
            * reward_raw
            for prob, pen, reward_raw in zip(solver_probs, penalties, solver_rewards_raw)
        ]

        # --- Intuitive answer: one greedy solver call (V-Zero fast track) ---
        # Reference: "V-Zero: Self-Improving Multimodal Reasoning with Zero
        # Annotation" (arXiv:2601.10094)
        #
        # Logit-Margin Difficulty Signal (LMDS): the greedy call now also
        # --- V-Zero intuitive track with Solver Token Entropy (STE) ---
        # The greedy generation extracts FULL token-level entropy at each
        # answer token.  Unlike logit margin (top1-top2 gap), STE captures
        # genuine multi-way uncertainty across the entire vocabulary:
        #   forced-choice "A or B?" → H ≈ ln(2) ≈ 0.69  (binary)
        #   genuinely hard question → H >> 1.0  (multi-way)
        # STE is naturally resistant to forced-choice gaming.
        _intuitive_answer_raw = ""
        _intuitive_answer = ""
        _intuitive_answer_vote = ""
        _intuitive_generation_failed = False
        _intuitive_attempted = False
        _intuitive_logit_min_margin = 999.0
        _intuitive_logit_mean_margin = 999.0
        _intuitive_token_entropy_max = 0.0
        _intuitive_token_entropy_mean = 0.0
        _ste_enabled = bool(getattr(self.cfg, "solver_token_entropy_enabled", True))
        _ste_tokens = max(1, int(getattr(self.cfg, "solver_token_entropy_tokens", 5)))
        if question and solver_prompt:
            _intuitive_attempted = True
            try:
                if _ste_enabled:
                    _intuitive_out, _conf_info = self._generate_with_confidence(
                        image=image,
                        prompt=solver_prompt,
                        adapter_name="default" if self.cfg.use_lora else None,
                        max_new_tokens=self.cfg.max_new_tokens_solver,
                        margin_tokens=_ste_tokens,
                    )
                    _intuitive_logit_min_margin = float(
                        _conf_info.get("min_margin", 999.0)
                    )
                    _intuitive_logit_mean_margin = float(
                        _conf_info.get("mean_margin", 999.0)
                    )
                    _intuitive_token_entropy_max = float(
                        _conf_info.get("max_entropy", 0.0)
                    )
                    _intuitive_token_entropy_mean = float(
                        _conf_info.get("mean_entropy", 0.0)
                    )
                else:
                    _intuitive_out = self._generate(
                        image=image,
                        prompt=solver_prompt,
                        adapter_name="default" if self.cfg.use_lora else None,
                        max_new_tokens=self.cfg.max_new_tokens_solver,
                        temperature=0.01,
                        top_p=1.0,
                        do_sample=False,
                    )
                _intuitive_answer_raw = normalize_answer(_parse_answer(_intuitive_out))
                _intuitive_answer = _intuitive_answer_raw
                if solver_choice_mode:
                    _intuitive_answer_vote = self._parse_forced_choice_answer(
                        _parse_answer(_intuitive_out),
                        selected_choice_option_a,
                        selected_choice_option_b,
                    )
                    if not _intuitive_answer_vote:
                        _intuitive_answer_vote = "ood"
                else:
                    _intuitive_answer_vote, _, _ = self._normalize_answer_for_type(
                        _intuitive_answer_raw,
                        solver_answer_type,
                    )
            except Exception:
                _intuitive_answer_raw = ""
                _intuitive_answer = ""
                _intuitive_answer_vote = ""
                _intuitive_generation_failed = True
                _intuitive_logit_min_margin = 999.0
                _intuitive_logit_mean_margin = 999.0
                _intuitive_token_entropy_max = 0.0
                _intuitive_token_entropy_mean = 0.0

        # --- Compute Solver Token Entropy (STE) difficulty ---
        # STE uses the MAX token-level entropy across the first K answer
        # tokens as the raw signal.  High max-entropy = the model had at
        # least one token where it was genuinely uncertain across many
        # vocabulary items (not just 2 forced-choice options).
        #
        # Self-calibrating: uses quantile rank in a rolling window so the
        # signal adapts to the model's evolving entropy distribution.
        # Falls back to sigmoid when the window is too small.
        #
        # KEY ADVANTAGE over LMDS: forced-choice "A or B?" produces token
        # entropy ≈ ln(2) ≈ 0.69.  Genuinely hard open-ended questions
        # produce entropy >> 1.0.  No explicit forced-choice discount needed.
        _ste_difficulty = 0.0
        _ste_aggregation = str(
            getattr(self.cfg, "solver_token_entropy_aggregation", "max")
        ).strip().lower()
        if _ste_aggregation == "mean":
            _ste_raw_value = _intuitive_token_entropy_mean
        else:
            _ste_aggregation = "max"
            _ste_raw_value = _intuitive_token_entropy_max
        if _ste_enabled and _ste_raw_value > 1e-6:
            self._ste_window.append(_ste_raw_value)
            _ste_window_size = max(
                8,
                int(getattr(self.cfg, "solver_token_entropy_window_size", 128)),
            )
            # Trim window
            while len(self._ste_window) > _ste_window_size:
                self._ste_window.pop(0)
            _ste_n = len(self._ste_window)
            if _ste_n >= 8:
                # Quantile-based: fraction of window entries with LOWER
                # entropy (= more confident = easier).  High quantile = this
                # question triggered more uncertainty than most recent ones.
                _ste_rank = sum(
                    1 for e in self._ste_window
                    if e < _ste_raw_value
                )
                _ste_difficulty = float(_ste_rank) / float(_ste_n)
            else:
                # Sigmoid fallback for cold window.  Note: the sigmoid is
                # in ENTROPY space (higher = harder), so we DON'T negate.
                import math as _math
                _sig_alpha = float(
                    getattr(self.cfg, "solver_token_entropy_sigmoid_alpha", 1.5)
                )
                _sig_beta = float(
                    getattr(self.cfg, "solver_token_entropy_sigmoid_beta", 2.0)
                )
                _ste_difficulty = 1.0 / (
                    1.0 + _math.exp(-_sig_alpha * (_ste_raw_value - _sig_beta))
                )

        # --- Proposer reward: V-Zero dual-track learnability ---
        # The proposer is rewarded when the solver's "intuitive" (greedy)
        # answer DISAGREES with the "reasoned" (majority-voted, multi-temp)
        # answer.  This gives non-zero signal even when the solver is
        # unanimous on the reasoned track — the key to breaking the
        # same-model proposer-solver deadlock.
        proposer_entropy_mu_used = self._update_proposer_entropy_target(entropy_nats)
        proposer_reward_raw_gaussian = gaussian_reward(
            entropy_nats,
            proposer_entropy_mu_used,
            self.cfg.prop_entropy_sigma,
        )
        proposer_reward_raw = self._proposer_base_reward(
            entropy_nats=entropy_nats,
            local_info_score=local_info_score,
            entropy_mu=proposer_entropy_mu_used,
        )
        proposer_reward = proposer_reward_raw

        zero_entropy_cap = float(getattr(self.cfg, "zero_entropy_reward_cap", 0.10))
        zero_entropy_capped = False
        _confidence_logprob = None  # kept for log-record compatibility
        unsolvable_capped = False

        _tracks_agree = (
            bool(_intuitive_answer_vote == maj_answer_vote)
            if _intuitive_answer_vote
            else True
        )

        if unsolvable_solver_case:
            # Unsolvable → zero reward (AZ: r_propose = 0 when mean_solve = 0)
            proposer_reward = 0.0
            unsolvable_capped = True
        elif entropy_nats < 1e-6 or maj_frac >= 1.0:
            # Solver unanimous on reasoned track → check dual-track gap
            if not _tracks_agree and _intuitive_answer_vote:
                # Intuitive ≠ reasoned → "gotcha" question (V-Zero case 2)
                # Reward = 0.5 * confidence (higher confidence → better gotcha)
                proposer_reward = 0.5 * maj_frac
            # else: keep proposer_reward = proposer_reward_raw.
            zero_entropy_capped = True
        # else: non-unanimous → band/hybrid reward (already set above)

        # Text-only hardness shaping (no extra model calls): pushes proposer
        # away from low-information easy templates even when solver entropy
        # rewards are degenerate.
        proposer_text_hardness_bonus = self._proposer_text_hardness_bonus(
            question,
            chosen_strategy_used,
            chosen_two_answer_test,
        )
        proposer_strategy_quota_bonus = self._strategy_quota_adjustment(chosen_strategy_used)
        proposer_anchor_bonus = self._anchor_replay_bonus(question, chosen_strategy_used)
        proposer_contrastive_replay_bonus = self._contrastive_replay_adjustment(question)
        proposer_bonus_enabled = difficulty_bucket_observed in {"medium", "hard"}
        # -----------------------------------------------------------
        # STE-primary reward architecture.
        # STE is the PRIMARY difficulty signal — it provides a continuous
        # gradient even when sample entropy is dead (all solvers agree).
        # Sample entropy is a complementary bonus when it's non-zero.
        # This eliminates the fundamental dead-zone problem: STE always
        # differentiates easy from hard questions via token-level
        # uncertainty, costing only 1 greedy call vs 5-7 samples.
        # -----------------------------------------------------------
        _ste_is_primary = bool(
            _ste_enabled and _ste_difficulty > 0.0
        )
        proposer_bonus_warm_enabled = proposer_bonus_enabled or proposer_warm_start_active or _ste_is_primary
        if _ste_is_primary:
            # STE is always the dominant reward component.
            _ste_primary_weight = max(
                0.0,
                float(
                    getattr(self.cfg, "proposer_ste_primary_weight", 0.70)
                ),
            )
            # Blend: STE dominates, sample entropy provides bonus when available
            _sample_entropy_bonus = 0.0
            if entropy_nats > 1e-6:
                _sample_weight = max(
                    0.0,
                    float(
                        getattr(self.cfg, "proposer_sample_entropy_weight", 0.30)
                    ),
                )
                _sample_entropy_bonus = _sample_weight * proposer_reward_raw
            proposer_reward = _ste_primary_weight * _ste_difficulty + _sample_entropy_bonus
        elif _ste_enabled and _ste_difficulty > 0.0:
            # Fallback: STE available but not dominant — use as complement
            _ste_weight = max(
                0.0,
                float(
                    getattr(self.cfg, "proposer_ste_reward_weight", 0.30)
                ),
            )
            proposer_reward += _ste_weight * _ste_difficulty
        if not proposer_bonus_warm_enabled:
            proposer_text_hardness_bonus = 0.0
            proposer_strategy_quota_bonus = 0.0
            proposer_anchor_bonus = 0.0
        proposer_repetition_penalty = (
            self._question_repetition_penalty(question) * proposer_penalty_boost
        )
        proposer_cooldown_penalty = (
            self._template_cooldown_penalty(question) * proposer_penalty_boost
        )
        proposer_reward += proposer_text_hardness_bonus
        proposer_reward += proposer_strategy_quota_bonus
        proposer_reward += proposer_anchor_bonus
        proposer_certificate_weight_used = 0.0
        if proposer_bonus_warm_enabled:
            proposer_certificate_weight_used = max(
                0.0, float(getattr(self.cfg, "proposer_certificate_weight", 0.75))
            )
            if proposer_warm_start_active and (not proposer_bonus_enabled):
                proposer_certificate_weight_used = max(
                    0.0,
                    float(getattr(self.cfg, "proposer_warm_start_certificate_weight", 0.50)),
                )
            proposer_reward += (
                proposer_certificate_weight_used * selected_cert_score
            )
        if float(selected_cert.get("valid", 0.0)) < 0.5:
            proposer_reward -= 0.10
        proposer_reward += proposer_contrastive_replay_bonus
        proposer_reward += curriculum_arm_reward_bonus
        proposer_reward -= proposer_repetition_penalty
        proposer_reward -= proposer_cooldown_penalty

        # NOTE: Forced-choice penalty removed — STE naturally penalizes
        # forced-choice questions because binary "A or B?" uncertainty
        # produces token entropy ≈ ln(2) ≈ 0.69, which ranks LOW in the
        # quantile window compared to genuinely hard open-ended questions
        # with entropy >> 1.0.  No explicit discount needed.

        # Non-objective penalty.
        proposer_non_objective_penalty = max(
            0.0, float(getattr(self.cfg, "proposer_non_objective_penalty", 0.0))
        )
        if proposer_non_objective_question and proposer_non_objective_penalty > 0.0:
            proposer_reward -= proposer_non_objective_penalty
        if solver_low_info_majority:
            proposer_reward -= max(
                0.0,
                float(getattr(self.cfg, "proposer_low_info_majority_penalty", 0.30)),
            )
        proposer_answer_family = self._classify_answer_family(
            question,
            raw_majority_answer,
        )
        proposer_answer_family_penalty = self._answer_family_penalty(
            proposer_answer_family
        )
        proposer_reward -= proposer_answer_family_penalty

        # Rejection: non-objective or too-easy bucket.
        easy_question_detected = easy_solver_case
        reject_reasons: List[str] = []
        if require_objective and proposer_non_objective_question:
            reject_reasons.append("non_objective")
        if acceptance_require_non_easy and (difficulty_bucket_observed == "easy"):
            reject_reasons.append("easy_bucket")
        if acceptance_require_non_easy and spot_all_easy_low_entropy:
            reject_reasons.append("all_candidates_easy")
        if solver_low_info_majority:
            reject_reasons.append("low_info_majority")
        question_rejected = len(reject_reasons) > 0
        question_reject_reason = "|".join(reject_reasons)
        proposer_reject_penalty_scale_used = 1.0
        warm_start_easy_only_reject = (
            (proposer_warm_start_active or _ste_is_primary)
            and len(reject_reasons) > 0
            and all(r in {"easy_bucket", "all_candidates_easy"} for r in reject_reasons)
        )
        if warm_start_easy_only_reject:
            proposer_reject_penalty_scale_used = max(
                0.0,
                float(
                    getattr(self.cfg, "proposer_warm_start_easy_reject_penalty_scale", 0.0)
                ),
            )
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
            proposer_reward -= (
                rejected_question_penalty
                * proposer_reject_penalty_scale_used
                * _easy_scale
            )

        easy_constraint_penalty = 0.0
        if (
            easy_constraint_enabled
            and difficulty_bucket_observed == "easy"
            and (not proposer_warm_start_active)
            and (not _ste_is_primary)
        ):
            easy_constraint_penalty = (
                float(getattr(self, "_easy_lagrange_lambda", 0.0))
                * max(0.0, float(getattr(self.cfg, "easy_constraint_penalty_scale", 0.30)))
            )
            proposer_reward -= easy_constraint_penalty

        if difficulty_bucket_observed == "easy" and (not proposer_warm_start_active) and (not _ste_is_primary):
            proposer_reward = min(proposer_reward, easy_reward_floor)

        easy_update_majority_frac_threshold = float(
            getattr(self.cfg, "easy_update_majority_frac_threshold", 0.95)
        )
        easy_update_majority_frac_threshold = max(
            0.0, min(1.0, easy_update_majority_frac_threshold)
        )

        # Final safety cap for collapsed/easy questions.  STE and structural
        # certificates are useful auxiliary signals, but they must not override
        # the observed self-consistency result: if all solver passes agree, the
        # proposer did not produce useful exploration signal for the solver.
        # A verified dual-track disagreement is allowed a higher cap, but
        # rejected/all-easy candidate groups stay under the conservative cap.
        proposer_easy_reward_cap_applied = False
        proposer_easy_reward_cap_value = None
        proposer_easy_reward_cap_reason = ""
        easy_cap = max(
            -1.0,
            min(1.0, float(getattr(self.cfg, "proposer_easy_reward_cap", 0.20))),
        )
        gotcha_cap = max(
            easy_cap,
            min(1.0, float(getattr(self.cfg, "proposer_easy_gotcha_reward_cap", 0.50))),
        )
        collapsed_easy_for_reward_cap = bool(
            difficulty_bucket_observed == "easy"
            or easy_solver_case
            or (entropy_nats < 1e-6)
            or (maj_frac >= easy_update_majority_frac_threshold)
            or question_rejected
            or all_easy_candidate_group
        )
        verified_dual_track_gap = bool(
            (not _tracks_agree)
            and bool(_intuitive_answer_vote)
            and not question_rejected
            and not all_easy_candidate_group
            and solver_informative_gate
        )
        if collapsed_easy_for_reward_cap:
            proposer_easy_reward_cap_reason = (
                "dual_track_gap" if verified_dual_track_gap else "collapsed_easy"
            )
            cap_value = gotcha_cap if verified_dual_track_gap else easy_cap
            if question_rejected or all_easy_candidate_group or not solver_informative_gate:
                cap_value = min(cap_value, easy_cap)
                proposer_easy_reward_cap_reason = "rejected_or_all_easy"
            proposer_easy_reward_cap_value = cap_value
            if proposer_reward > cap_value:
                proposer_reward = cap_value
                proposer_easy_reward_cap_applied = True

        proposer_reward_pre_clip = float(proposer_reward)
        proposer_reward = max(-1.0, min(1.0, proposer_reward))
        proposer_reward_clipped = bool(abs(proposer_reward - proposer_reward_pre_clip) > 1e-8)
        # Track selected question template to discourage repeated easy loops.
        _qkey = self._question_template_key(question)
        _qtoken_set = self._question_token_set(question)
        if _qkey:
            self._question_template_window.append(_qkey)
            if difficulty_bucket_observed in {"medium", "hard"}:
                if _qtoken_set:
                    self._contrastive_pos_replay.append(_qtoken_set)
            elif difficulty_bucket_observed == "easy":
                if _qtoken_set:
                    self._contrastive_neg_replay.append(_qtoken_set)
        if proposer_answer_family:
            self._answer_family_window.append(proposer_answer_family)
        _strategy_key = self._normalize_strategy_key(chosen_strategy_used)
        if _strategy_key:
            self._strategy_window.append(_strategy_key)

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
        entropy_iqr_filter_min_majority_frac = float(
            getattr(self.cfg, "entropy_iqr_filter_min_majority_frac", 0.80)
        )
        entropy_iqr_filter_min_majority_frac = max(
            0.0, min(1.0, entropy_iqr_filter_min_majority_frac)
        )
        solver_entropy_iqr_blocked = bool(
            getattr(self.cfg, "solver_skip_update_on_easy", True)
            and
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
            and not (
                bool(getattr(self.cfg, "solver_update_on_low_info_easy", True))
                and solver_low_info_majority
            )
        )
        if local_solver_update_applied and question_rejected:
            local_solver_update_applied = False
            solver_update_skip_reason_local = (
                question_reject_reason or "question_rejected"
            )
        elif local_solver_update_applied and solver_entropy_iqr_blocked:
            local_solver_update_applied = False
            solver_update_skip_reason_local = "entropy_iqr_filter"
        elif local_solver_update_applied and solver_easy_update_blocked:
            local_solver_update_applied = False
            solver_update_skip_reason_local = "easy_case"
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
        selected_non_easy = 1.0 if difficulty_bucket_observed in {"medium", "hard"} else 0.0
        arm_solver_gain = float(solver_update_scale) if solver_update_applied else 0.0
        self._update_curriculum_arm_stats(
            selected_arm_key,
            non_easy=selected_non_easy,
            solver_gain=arm_solver_gain,
        )
        self._last_curriculum_arm_key = selected_arm_key
        # --- Anchor replay: capture questions that showed genuine difficulty ---
        # Original gate: medium/hard bucket + reward threshold.
        # NEW: Also capture questions with HIGH STE difficulty (SD > 0.60),
        # which means the solver had high token-level uncertainty even if
        # sample-level entropy was 0 (unanimous).  This fills the anchor
        # buffer from step 1, giving the proposer exemplars of what
        # triggers real model uncertainty — solving the cold-start problem
        # for the replay buffer.
        _anchor_qualify_bucket = difficulty_bucket_observed in {"medium", "hard"}
        _anchor_qualify_ste = bool(
            _ste_enabled and _ste_difficulty >= 0.60 and question
        )
        _anchor_qualify_entropy = bool(entropy_nats > 0.05 and question)
        _anchor_min_reward = float(getattr(self.cfg, "proposer_anchor_min_reward", 0.20))
        if (
            (_anchor_qualify_bucket and proposer_reward >= _anchor_min_reward)
            or _anchor_qualify_ste
            or _anchor_qualify_entropy
        ):
            # Priority score: blend of hardness, solver gain, novelty.
            # For STE-qualified entries, use ste_difficulty as hardness.
            _effective_hardness = local_info_score
            if _anchor_qualify_ste and _ste_difficulty > _effective_hardness:
                _effective_hardness = _ste_difficulty
            if _anchor_qualify_entropy and entropy_nats > 0:
                _effective_hardness = max(_effective_hardness, min(1.0, entropy_nats / 1.5))
            anchor_priority = self._replay_anchor_priority(
                question,
                hardness_score=_effective_hardness,
                solver_gain=arm_solver_gain,
            )
            # Tag with the source so we can present richer hints.
            _anchor_source = "bucket"
            if _anchor_qualify_entropy:
                _anchor_source = "entropy"
            elif _anchor_qualify_ste:
                _anchor_source = "ste"
            self._proposer_anchor_replay.append(
                {
                    "qkey": _qkey,
                    "q_tokens": sorted(list(_qtoken_set)) if _qtoken_set else [],
                    "question": question,
                    "strategy": _strategy_key,
                    "bucket": difficulty_bucket_observed,
                    "reward": float(proposer_reward),
                    "priority": float(anchor_priority),
                    "step": int(step),
                    "ste_difficulty": float(_ste_difficulty),
                    "entropy": float(entropy_nats),
                    "source": _anchor_source,
                }
            )
        self._candidate_non_easy_window.append(float(candidate_non_easy_rate))
        self._all_easy_group_window.append(1.0 if all_easy_candidate_group else 0.0)
        self._proposer_reward_clipped_window.append(1.0 if proposer_reward_clipped else 0.0)
        self._selected_non_easy_window.append(float(selected_non_easy))
        self._solver_update_applied_window.append(1.0 if solver_update_applied else 0.0)

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
                    _grpo_buckets = [difficulty_bucket_observed]
                    _grpo_group_size = max(
                        2, int(getattr(self.cfg, "proposer_grpo_gen_group_size", 3))
                    )
                    _score_extras = bool(getattr(self.cfg, "score_grpo_extras", True))
                    _extra_temp_mult = float(getattr(self.cfg, "grpo_extra_temp_multiplier", 2.0))
                    _extra_temp = min(3.0, proposer_temp_ctrl * _extra_temp_mult)
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
                                top_p=proposer_top_p_ctrl,
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
                            _extra_strategy_bonus = 0.0
                            _extra_anchor_bonus = 0.0
                            _extra_contrastive_bonus = 0.0
                            _extra_cert_score = 0.0
                            _extra_cert_valid = 0.0
                            _extra_repeat_penalty = 0.0
                            _extra_cooldown_penalty = 0.0

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
                                    )
                                    + max(
                                        0.0,
                                        float(getattr(self.cfg, "proposer_certificate_weight", 0.75)),
                                    )
                                    * float(
                                        self._proposer_certificate_score(
                                            str(c.get("text", "") or ""),
                                            c,
                                        ).get("score", 0.0)
                                    ),
                                )
                                _extra_q = str(_extra_best.get("text", "")).replace("\n", " ").strip()
                                _extra_q_compiled, _extra_compile_ok, _extra_compile_reason = (
                                    self._compile_question_from_slots(_extra_q, _extra_best)
                                )
                                _extra_best = dict(_extra_best)
                                _extra_best["_compiler_valid"] = "1" if _extra_compile_ok else "0"
                                _extra_best["_compiler_reason"] = _extra_compile_reason
                                if _extra_compile_ok and _extra_q_compiled:
                                    _extra_q = _extra_q_compiled
                                _extra_strategy_used = str(_extra_best.get("strategy_used", "") or "")
                                _extra_two_answer_test = str(_extra_best.get("two_answer_test", "") or "")
                                _extra_cert = self._proposer_certificate_score(_extra_q, _extra_best)
                                _extra_cert_score = float(_extra_cert.get("score", 0.0))
                                _extra_cert_valid = float(_extra_cert.get("valid", 0.0))
                                if (not _extra_compile_ok) and bool(
                                    getattr(self.cfg, "proposer_slot_compiler_strict", True)
                                ):
                                    _extra_cert_valid = 0.0
                                    _extra_cert_score = min(_extra_cert_score, 0.0)

                            if _score_extras and _extra_q:
                                # ── Score extra candidate with configured solver spot-check ──
                                if _extra_q:
                                    _extra_opt_a, _extra_opt_b = self._extract_forced_choice_options(
                                        _extra_two_answer_test
                                    )
                                    _extra_choice_mode = bool(
                                        solver_use_forced_choice_from_proposer
                                        and _extra_opt_a
                                        and _extra_opt_b
                                    )
                                    _extra_answer_type = self._question_answer_type(_extra_q)
                                    _extra_answers_norm: List[str] = []
                                    _extra_vote_labels: List[str] = []
                                    _extra_low_info_flags: List[bool] = []
                                    _extra_noncanonical_flags: List[bool] = []
                                    _extra_raw_majority = ""
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
                                        if _pps_enabled:
                                            _extra_solver_prompt = build_solver_prompt_pps(
                                                _extra_q,
                                                template_index=_real_idx,
                                                focus_hint=self._solver_focus_hint(_real_idx),
                                            )
                                        else:
                                            _extra_solver_prompt = build_solver_prompt(
                                                _extra_q,
                                                focus_hint=self._solver_focus_hint(_real_idx),
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
                                            _sc_ans_raw = _parse_answer(_sc_out)
                                            _sc_ans_text = normalize_answer(_sc_ans_raw)
                                            if _extra_choice_mode:
                                                _sc_vote = self._parse_forced_choice_answer(
                                                    _sc_ans_raw, _extra_opt_a, _extra_opt_b
                                                )
                                                if not _sc_vote:
                                                    _sc_vote = "ood"
                                                _sc_low_info = False
                                                _sc_noncanonical = False
                                            else:
                                                _sc_vote, _sc_low_info, _sc_noncanonical = (
                                                    self._normalize_answer_for_type(
                                                        _sc_ans_text,
                                                        _extra_answer_type,
                                                    )
                                                )
                                            _extra_answers_norm.append(_sc_ans_text)
                                            _extra_vote_labels.append(_sc_vote)
                                            _extra_low_info_flags.append(bool(_sc_low_info))
                                            _extra_noncanonical_flags.append(bool(_sc_noncanonical))
                                        except Exception:
                                            pass

                                    _extra_low_info_rate = 0.0
                                    _extra_noncanonical_rate = 0.0
                                    if _extra_answers_norm:
                                        _extra_raw_majority, _ = majority_vote(_extra_answers_norm)
                                        _extra_hist: Dict[str, int] = {}
                                        for _ans in _extra_vote_labels:
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
                                        _extra_entropy_band = math.exp(
                                            -(((_extra_entropy_val - entropy_mid) ** 2) / (2.0 * (entropy_sigma ** 2)))
                                        )
                                        _extra_margin_damp = max(
                                            0.0, 1.0 - (_extra_margin_val / max(1e-6, margin_max))
                                        )
                                        _extra_local_info = max(
                                            0.0, min(1.0, 0.5 * _extra_entropy_band + 0.5 * _extra_margin_damp)
                                        )
                                        _extra_low_info_rate = float(
                                            sum(1.0 for x in _extra_low_info_flags if x)
                                        ) / float(max(1, len(_extra_low_info_flags)))
                                        _extra_noncanonical_rate = float(
                                            sum(1.0 for x in _extra_noncanonical_flags if x)
                                        ) / float(max(1, len(_extra_noncanonical_flags)))

                                        # Compute reward using same logic as chosen candidate.
                                        _extra_reward_raw = self._proposer_base_reward(
                                            entropy_nats=_extra_entropy_val,
                                            local_info_score=_extra_local_info,
                                            entropy_mu=proposer_entropy_mu_used,
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
                                                _extra_intuitive_raw = _parse_answer(_ei_out)
                                                _extra_intuitive = (
                                                    self._parse_forced_choice_answer(
                                                        _extra_intuitive_raw, _extra_opt_a, _extra_opt_b
                                                    )
                                                    if _extra_choice_mode
                                                    else self._normalize_answer_for_type(
                                                        normalize_answer(_extra_intuitive_raw),
                                                        _extra_answer_type,
                                                    )[0]
                                                )
                                                if not _extra_intuitive:
                                                    _extra_intuitive = (
                                                        "ood"
                                                        if _extra_choice_mode
                                                        else self._normalize_answer_for_type(
                                                            normalize_answer(_extra_intuitive_raw),
                                                            _extra_answer_type,
                                                        )[0]
                                                    )
                                            except Exception:
                                                _extra_intuitive_failed = True
                                            _extra_maj = max(
                                                set(_extra_vote_labels),
                                                key=_extra_vote_labels.count,
                                            ) if _extra_vote_labels else ""
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
                                    if easy_constraint_enabled and _extra_bucket == "easy":
                                        _extra_reward -= (
                                            float(getattr(self, "_easy_lagrange_lambda", 0.0))
                                            * max(
                                                0.0,
                                                float(getattr(self.cfg, "easy_constraint_penalty_scale", 0.30)),
                                            )
                                        )

                            if _extra_q:
                                # Deterministic text-level shaping for extras.
                                _extra_text_bonus = self._proposer_text_hardness_bonus(
                                    _extra_q,
                                    _extra_strategy_used,
                                    _extra_two_answer_test,
                                )
                                _extra_strategy_bonus = self._strategy_quota_adjustment(_extra_strategy_used)
                                _extra_anchor_bonus = self._anchor_replay_bonus(_extra_q, _extra_strategy_used)
                                _extra_contrastive_bonus = self._contrastive_replay_adjustment(_extra_q)
                                _extra_bonus_enabled = _extra_bucket in {"medium", "hard"}
                                if not _extra_bonus_enabled:
                                    _extra_text_bonus = 0.0
                                    _extra_strategy_bonus = 0.0
                                    _extra_anchor_bonus = 0.0
                                _extra_repeat_penalty = (
                                    self._question_repetition_penalty(_extra_q) * proposer_penalty_boost
                                )
                                _extra_cooldown_penalty = (
                                    self._template_cooldown_penalty(_extra_q) * proposer_penalty_boost
                                )
                                _extra_qkey = self._question_template_key(_extra_q)
                                if _extra_qkey:
                                    if _extra_qkey in _step_question_templates:
                                        _extra_repeat_penalty += float(
                                            getattr(self.cfg, "proposer_text_step_dup_penalty", 0.08)
                                        )
                                    _step_question_templates.add(_extra_qkey)
                                _extra_reward += _extra_text_bonus
                                _extra_reward += _extra_strategy_bonus
                                _extra_reward += _extra_anchor_bonus
                                _extra_reward += _extra_contrastive_bonus
                                _extra_reward -= (
                                    _extra_noncanonical_rate
                                    * max(
                                        0.0,
                                        float(
                                            getattr(
                                                self.cfg,
                                                "proposer_candidate_noncanonical_penalty",
                                                0.12,
                                            )
                                        ),
                                    )
                                    + _extra_low_info_rate
                                    * max(
                                        0.0,
                                        float(
                                            getattr(
                                                self.cfg,
                                                "proposer_candidate_low_info_penalty",
                                                0.10,
                                            )
                                        ),
                                    )
                                )
                                if _extra_bonus_enabled:
                                    _extra_reward += (
                                        max(
                                            0.0,
                                            float(getattr(self.cfg, "proposer_certificate_weight", 0.75)),
                                        )
                                        * _extra_cert_score
                                    )
                                if _extra_cert_valid < 0.5:
                                    _extra_reward -= 0.10
                                _extra_answer_family = self._classify_answer_family(
                                    _extra_q,
                                    _extra_raw_majority,
                                )
                                _extra_reward -= self._answer_family_penalty(_extra_answer_family)
                                _extra_reward -= _extra_repeat_penalty
                                _extra_reward -= _extra_cooldown_penalty

                            if _extra_bucket == "easy":
                                _extra_reward = min(_extra_reward, easy_reward_floor)

                            _extra_reward = max(-1.0, min(1.0, _extra_reward))

                            _grpo_completions.append(_extra_comp)
                            _grpo_rewards.append(_extra_reward)
                            _grpo_images.append(image)
                            _grpo_buckets.append(_extra_bucket if _extra_bucket in {"easy", "medium", "hard"} else difficulty_bucket_observed)

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
                            proposer_stats[f"grpo_extra_{_gi}_strategy_bonus"] = _extra_strategy_bonus
                            proposer_stats[f"grpo_extra_{_gi}_anchor_bonus"] = _extra_anchor_bonus
                            proposer_stats[f"grpo_extra_{_gi}_contrastive_bonus"] = _extra_contrastive_bonus
                            proposer_stats[f"grpo_extra_{_gi}_cert_score"] = _extra_cert_score
                            proposer_stats[f"grpo_extra_{_gi}_cert_valid"] = _extra_cert_valid
                            proposer_stats[f"grpo_extra_{_gi}_repeat_penalty"] = _extra_repeat_penalty
                            proposer_stats[f"grpo_extra_{_gi}_cooldown_penalty"] = _extra_cooldown_penalty
                            proposer_stats[f"grpo_extra_{_gi}_low_info_rate"] = _extra_low_info_rate
                            proposer_stats[f"grpo_extra_{_gi}_noncanonical_rate"] = _extra_noncanonical_rate
                        except Exception:
                            pass

                    # Pairwise ranking signal: enforce medium/hard > easy
                    # ordering inside each GRPO group without extra inference.
                    _grpo_rewards, _grpo_rank_deltas = self._apply_grpo_pairwise_ranking(
                        _grpo_rewards, _grpo_buckets
                    )
                    _grpo_rewards, _grpo_all_easy_deltas, _grpo_all_easy_applied = (
                        self._apply_all_easy_relative_negatives(_grpo_rewards, _grpo_buckets)
                    )
                    if proposer_stats is None:
                        proposer_stats = {}
                    proposer_stats["grpo_pairwise_rank_delta_mean"] = (
                        float(sum(_grpo_rank_deltas) / max(1, len(_grpo_rank_deltas)))
                        if _grpo_rank_deltas
                        else 0.0
                    )
                    proposer_stats["grpo_pairwise_rank_delta_max"] = (
                        float(max(_grpo_rank_deltas)) if _grpo_rank_deltas else 0.0
                    )
                    proposer_stats["grpo_pairwise_rank_delta_min"] = (
                        float(min(_grpo_rank_deltas)) if _grpo_rank_deltas else 0.0
                    )
                    proposer_stats["grpo_all_easy_rank_applied"] = bool(
                        _grpo_all_easy_applied
                    )
                    proposer_stats["grpo_all_easy_rank_delta_mean"] = (
                        float(sum(_grpo_all_easy_deltas) / max(1, len(_grpo_all_easy_deltas)))
                        if _grpo_all_easy_deltas
                        else 0.0
                    )

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

                    # Apply bucket-stratified EMA baselines to all group rewards.
                    _grpo_rewards_shifted = [
                        r - self._get_proposer_bucket_baseline(b)
                        for r, b in zip(_grpo_rewards, _grpo_buckets)
                    ]

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
                    self._update_proposer_bucket_baseline(
                        difficulty_bucket_observed,
                        effective_reward,
                    )
                    # Log the baseline shift for visibility.
                    if proposer_stats is not None:
                        proposer_stats["grpo_ema_baseline"] = float(self.proposer_baseline)
                        proposer_stats["grpo_bucket_baseline_easy"] = self._get_proposer_bucket_baseline("easy")
                        proposer_stats["grpo_bucket_baseline_medium"] = self._get_proposer_bucket_baseline("medium")
                        proposer_stats["grpo_bucket_baseline_hard"] = self._get_proposer_bucket_baseline("hard")
                        proposer_stats["grpo_baseline_input"] = effective_reward
                        # Record valid completion count for GRPO diagnostics.
                        proposer_stats["grpo_valid_completions"] = proposer_stats.get("valid_completions", -1)
                        proposer_stats["grpo_bucket_labels"] = list(_grpo_buckets)
                else:
                    # ── REINFORCE path (legacy / proposer_update_rule="reinforce") ──
                    # Use the raw baseline without clamping. The previous clamp
                    # (min(baseline, reward) when reward < 0) caused the advantage
                    # to collapse to exactly 0.0 at equilibrium (when baseline ≈
                    # reward) — eliminating the learning signal entirely. Standard
                    # REINFORCE advantage = reward - baseline handles negative rewards
                    # correctly without any clamping.
                    effective_baseline = (
                        self._get_proposer_bucket_baseline(difficulty_bucket_observed)
                        if local_can_proposer_update
                        else 0.0
                    )
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
                        self._update_proposer_bucket_baseline(
                            difficulty_bucket_observed,
                            proposer_reward,
                        )

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

        easy_constraint_state = self._update_easy_constraint(
            difficulty_bucket_observed == "easy"
        )
        collapse_state = self._update_collapse_state(
            difficulty_bucket_observed=difficulty_bucket_observed,
            proposer_stats=proposer_stats,
        )
        early_failfast_state = self._early_failfast_state(
            step=step,
            collapse_state=collapse_state,
        )
        self._sync_proposer_framework_state()

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
            "solver_vote_labels": solver_vote_labels,
            "solver_answer_type": solver_answer_type,
            "solver_low_info_rate": solver_low_info_rate,
            "solver_noncanonical_rate": solver_noncanonical_rate,
            "solver_noncanonical_answer_penalty": solver_noncanonical_answer_penalty,
            "solver_low_info_answer_penalty": solver_low_info_answer_penalty,
            "solver_rewards_raw": solver_rewards_raw,
            "solver_rewards_soft": solver_rewards_soft,
            "majority_answer": maj_answer,
            "majority_answer_vote": maj_answer_vote,
            "majority_answer_raw": raw_majority_answer,
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
            "solver_low_info_majority": solver_low_info_majority,
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
            "proposer_spot_entropy_min_gate": spot_entropy_min_gate,
            "proposer_spot_all_easy_low_entropy": spot_all_easy_low_entropy,
            "entropy_nats": entropy_nats,
            "proposer_entropy_mu_used": proposer_entropy_mu_used,
            "proposer_reward_raw_gaussian": proposer_reward_raw_gaussian,
            "proposer_reward_raw": proposer_reward_raw,
            "proposer_reward": proposer_reward,
            "proposer_reward_pre_clip": proposer_reward_pre_clip,
            "proposer_reward_clipped": proposer_reward_clipped,
            "proposer_easy_reward_cap_applied": proposer_easy_reward_cap_applied,
            "proposer_easy_reward_cap_value": proposer_easy_reward_cap_value,
            "proposer_easy_reward_cap_reason": proposer_easy_reward_cap_reason,
            "proposer_bonus_enabled": proposer_bonus_enabled,
            "proposer_bonus_warm_enabled": proposer_bonus_warm_enabled,
            "proposer_certificate_weight_used": proposer_certificate_weight_used,
            "proposer_reject_penalty_scale_used": proposer_reject_penalty_scale_used,
            "proposer_text_hardness_bonus": proposer_text_hardness_bonus,
            "proposer_strategy_quota_bonus": proposer_strategy_quota_bonus,
            "proposer_anchor_bonus": proposer_anchor_bonus,
            "proposer_contrastive_replay_bonus": proposer_contrastive_replay_bonus,
            "proposer_answer_family": proposer_answer_family,
            "proposer_answer_family_penalty": proposer_answer_family_penalty,
            "proposer_repetition_penalty": proposer_repetition_penalty,
            "proposer_cooldown_penalty": proposer_cooldown_penalty,
            "proposer_easy_constraint_penalty": easy_constraint_penalty,
            "proposer_strategy_used": chosen_strategy_used,
            "proposer_two_answer_test": chosen_two_answer_test,
            "solver_choice_mode": solver_choice_mode,
            "solver_choice_option_a": selected_choice_option_a,
            "solver_choice_option_b": selected_choice_option_b,
            "proposer_reasoning_domains": chosen_reasoning_domains,
            "proposer_reasoning_chain": chosen_reasoning_chain,
            "proposer_task_card": chosen_task_card,
            "proposer_certificate_selected": selected_cert_score,
            "proposer_candidate_certificate_best": max(candidate_cert_list) if candidate_cert_list else 0.0,
            "proposer_candidate_certificate_mean": (
                float(sum(candidate_cert_list) / max(1, len(candidate_cert_list)))
                if candidate_cert_list
                else 0.0
            ),
            "proposer_candidate_non_easy_rate": candidate_non_easy_rate,
            "proposer_candidate_struct_valid_rate": candidate_struct_valid_rate,
            "proposer_candidate_low_info_rate_mean": candidate_low_info_rate_mean,
            "proposer_candidate_noncanonical_rate_mean": candidate_noncanonical_rate_mean,
            "proposer_all_easy_candidate_group": all_easy_candidate_group,
            "curriculum_arm_sampled_key": curriculum_arm_key,
            "curriculum_arm_sampled_hint": curriculum_arm_hint,
            "curriculum_arm_sampled_score": float(curriculum_arm_state.get("score", 0.0)),
            "curriculum_arm_selected_key": selected_arm_key,
            "curriculum_arm_selected_score": curriculum_arm_score,
            "curriculum_arm_reward_bonus": curriculum_arm_reward_bonus,
            "replay_anchor_hints": replay_anchor_hints,
            "replay_anchor_hints_count": len(replay_anchor_hints),
            "replay_anchor_queue_size": len(self._proposer_anchor_replay),
            "proposer_controller_temp": proposer_temp_ctrl,
            "proposer_controller_top_p": proposer_top_p_ctrl,
            "proposer_controller_penalty_boost": proposer_penalty_boost,
            "proposer_controller_forced_explore_active": bool(
                controller_state.get("forced_explore_active", 0.0) > 0.5
            ),
            "proposer_controller_forced_explore_steps_left": int(
                round(float(controller_state.get("forced_explore_steps_left", 0.0)))
            ),
            "proposer_controller_num_candidates": int(num_proposer_candidates),
            "proposer_warm_start_active": bool(proposer_warm_start_active),
            "proposer_warm_start_active_next": bool(
                warm_start_state.get("active_next", 0.0) > 0.5
            ),
            "proposer_warm_start_completed": bool(
                warm_start_state.get("completed", 0.0) > 0.5
            ),
            "proposer_warm_start_entropy_mean": warm_start_state.get("entropy_mean", 0.0),
            "proposer_warm_start_exit_streak": warm_start_state.get("exit_streak", 0.0),
            "proposer_warm_start_exit_pass": bool(
                warm_start_state.get("exit_pass", 0.0) > 0.5
            ),
            "proposer_logit_margin_min": float(_intuitive_logit_min_margin),
            "proposer_logit_margin_mean": float(_intuitive_logit_mean_margin),
            "proposer_token_entropy_max": float(_intuitive_token_entropy_max),
            "proposer_token_entropy_mean": float(_intuitive_token_entropy_mean),
            "proposer_ste_aggregation": _ste_aggregation,
            "proposer_ste_raw_value": float(_ste_raw_value),
            "proposer_ste_difficulty": float(_ste_difficulty),
            "proposer_ste_window_size": len(self._ste_window),
            "proposer_pps_enabled": bool(_pps_enabled),
            "proposer_hardness_debt": hardness_debt_state.get("debt", 0.0),
            "proposer_hardness_debt_cap_streak": hardness_debt_state.get("cap_streak", 0.0),
            "proposer_hardness_debt_escape_steps_left": hardness_debt_state.get(
                "escape_steps_left", 0.0
            ),
            "proposer_hardness_debt_escape_triggered": bool(
                hardness_debt_state.get("escape_triggered", 0.0) > 0.5
            ),
            "proposer_easy_rate_ema": easy_constraint_state.get("easy_rate_ema"),
            "proposer_easy_lagrange_lambda": easy_constraint_state.get("easy_lambda"),
            "proposer_easy_reward_floor": easy_reward_floor,
            "proposer_collapse_streak": collapse_state.get("collapse_streak"),
            "proposer_collapse_mean_std": collapse_state.get("collapse_mean_std"),
            "proposer_collapse_hit": collapse_state.get("collapse_hit"),
            "proposer_bucket_baseline_easy": self._get_proposer_bucket_baseline("easy"),
            "proposer_bucket_baseline_medium": self._get_proposer_bucket_baseline("medium"),
            "proposer_bucket_baseline_hard": self._get_proposer_bucket_baseline("hard"),
            "zero_entropy_capped": zero_entropy_capped,
            "zero_entropy_reward_cap": zero_entropy_cap,
            "intuitive_answer": _intuitive_answer,
            "intuitive_answer_raw": _intuitive_answer_raw,
            "intuitive_answer_vote": _intuitive_answer_vote,
            "dual_track_agree": _tracks_agree,
            "intuitive_attempted": _intuitive_attempted,
            "intuitive_generation_failed": _intuitive_generation_failed,
            "unsolvable_solver_case": unsolvable_solver_case,
            "unsolvable_capped": unsolvable_capped,
            "easy_question_detected": easy_question_detected,
            "solver_skip_update_on_easy": solver_skip_update_on_easy,
            "solver_update_on_low_info_easy": bool(
                getattr(self.cfg, "solver_update_on_low_info_easy", True)
            ),
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
            "difficulty_hardness_debt": difficulty_target_state.get("hardness_debt", 0.0),
            "difficulty_hardness_debt_ratio": difficulty_target_state.get(
                "hardness_debt_ratio", 0.0
            ),
            "difficulty_hardness_debt_escape_active": bool(
                difficulty_target_state.get("hardness_debt_escape_active", False)
            ),
            "solver_baseline": self.solver_baseline,
            "proposer_baseline": self.proposer_baseline,
            "solver_update_due": solver_update_due,
            "solver_update_applied": solver_update_applied,
            "solver_update_skip_reason": solver_update_skip_reason,
            "solver_stats": solver_stats_list,
            "proposer_update_due": proposer_update_due,
            "proposer_skip_reason": proposer_skip_reason,
            "proposer_stats": proposer_stats,
            "proposer_early_failfast_enabled": bool(early_failfast_state.get("enabled", 0.0)),
            "proposer_early_u_step": int(early_failfast_state.get("u_step", 0.0)),
            "proposer_early_hard_stop_min_u_step": int(
                early_failfast_state.get("hard_stop_min_u_step", 0.0)
            ),
            "proposer_early_stage1_active": bool(early_failfast_state.get("stage1_active", 0.0)),
            "proposer_early_stage1_pass": bool(early_failfast_state.get("stage1_pass", 1.0)),
            "proposer_early_stage2_active": bool(early_failfast_state.get("stage2_active", 0.0)),
            "proposer_early_stage2_pass": bool(early_failfast_state.get("stage2_pass", 1.0)),
            "proposer_early_recovery_armed": bool(
                early_failfast_state.get("recovery_armed", 0.0)
            ),
            "proposer_early_triggered": bool(early_failfast_state.get("triggered", 0.0)),
        }
        self._append_jsonl(self.iter_log_path, record)
        self._monitor_understanding_record(record)

        self._append_jsonl(
            self.rewards_log_path,
            {
                "step": step,
                "phase": "understanding",
                "image_path": meta.get("path"),
                "majority_answer": maj_answer,
                "majority_answer_vote": maj_answer_vote,
                "majority_answer_raw": raw_majority_answer,
                "solver_answer_type": solver_answer_type,
                "solver_low_info_rate": solver_low_info_rate,
                "solver_noncanonical_rate": solver_noncanonical_rate,
                "solver_noncanonical_answer_penalty": solver_noncanonical_answer_penalty,
                "solver_low_info_answer_penalty": solver_low_info_answer_penalty,
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
                "solver_low_info_majority": solver_low_info_majority,
                "solver_margin_score": margin_damp_score,
                "solver_entropy_band_score": entropy_band_score,
                "solver_local_info_score": local_info_score,
                "easy_solver_case": easy_solver_case,
                "easy_solver_penalty_scale": easy_solver_penalty_scale,
                "solver_update_scale": solver_update_scale,
                "proposer_spot_check_samples": spot_check_samples,
                "proposer_spot_check_offset": spot_check_offset,
                "proposer_spot_entropy_min_gate": spot_entropy_min_gate,
                "proposer_spot_all_easy_low_entropy": spot_all_easy_low_entropy,
                "entropy_nats": entropy_nats,
                "solver_reward_soft_mean": sum(solver_rewards_soft) / max(1, len(solver_rewards_soft)),
                "proposer_entropy_mu_used": proposer_entropy_mu_used,
                "proposer_reward_raw_gaussian": proposer_reward_raw_gaussian,
                "proposer_reward_raw": proposer_reward_raw,
                "proposer_reward": proposer_reward,
                "proposer_reward_pre_clip": proposer_reward_pre_clip,
                "proposer_reward_clipped": proposer_reward_clipped,
                "proposer_easy_reward_cap_applied": proposer_easy_reward_cap_applied,
                "proposer_easy_reward_cap_value": proposer_easy_reward_cap_value,
                "proposer_easy_reward_cap_reason": proposer_easy_reward_cap_reason,
                "proposer_bonus_enabled": proposer_bonus_enabled,
                "proposer_bonus_warm_enabled": proposer_bonus_warm_enabled,
                "proposer_certificate_weight_used": proposer_certificate_weight_used,
                "proposer_reject_penalty_scale_used": proposer_reject_penalty_scale_used,
                "proposer_text_hardness_bonus": proposer_text_hardness_bonus,
                "proposer_strategy_quota_bonus": proposer_strategy_quota_bonus,
                "proposer_anchor_bonus": proposer_anchor_bonus,
                "proposer_contrastive_replay_bonus": proposer_contrastive_replay_bonus,
                "proposer_answer_family": proposer_answer_family,
                "proposer_answer_family_penalty": proposer_answer_family_penalty,
                "proposer_repetition_penalty": proposer_repetition_penalty,
                "proposer_cooldown_penalty": proposer_cooldown_penalty,
                "proposer_easy_constraint_penalty": easy_constraint_penalty,
                "proposer_strategy_used": chosen_strategy_used,
                "proposer_two_answer_test": chosen_two_answer_test,
                "solver_choice_mode": solver_choice_mode,
                "solver_choice_option_a": selected_choice_option_a,
                "solver_choice_option_b": selected_choice_option_b,
                "proposer_reasoning_domains": chosen_reasoning_domains,
                "proposer_reasoning_chain": chosen_reasoning_chain,
                "proposer_task_card": chosen_task_card,
                "proposer_certificate_selected": selected_cert_score,
                "proposer_candidate_certificate_best": max(candidate_cert_list) if candidate_cert_list else 0.0,
                "proposer_candidate_certificate_mean": (
                    float(sum(candidate_cert_list) / max(1, len(candidate_cert_list)))
                    if candidate_cert_list
                    else 0.0
                ),
                "proposer_candidate_non_easy_rate": candidate_non_easy_rate,
                "proposer_candidate_struct_valid_rate": candidate_struct_valid_rate,
                "proposer_candidate_low_info_rate_mean": candidate_low_info_rate_mean,
                "proposer_candidate_noncanonical_rate_mean": candidate_noncanonical_rate_mean,
                "proposer_all_easy_candidate_group": all_easy_candidate_group,
                "curriculum_arm_sampled_key": curriculum_arm_key,
                "curriculum_arm_sampled_hint": curriculum_arm_hint,
                "curriculum_arm_sampled_score": float(curriculum_arm_state.get("score", 0.0)),
                "curriculum_arm_selected_key": selected_arm_key,
                "curriculum_arm_selected_score": curriculum_arm_score,
                "curriculum_arm_reward_bonus": curriculum_arm_reward_bonus,
                "replay_anchor_hints": replay_anchor_hints,
                "replay_anchor_hints_count": len(replay_anchor_hints),
                "replay_anchor_queue_size": len(self._proposer_anchor_replay),
                "proposer_controller_temp": proposer_temp_ctrl,
                "proposer_controller_top_p": proposer_top_p_ctrl,
                "proposer_controller_penalty_boost": proposer_penalty_boost,
                "proposer_controller_forced_explore_active": bool(
                    controller_state.get("forced_explore_active", 0.0) > 0.5
                ),
                "proposer_controller_forced_explore_steps_left": int(
                    round(float(controller_state.get("forced_explore_steps_left", 0.0)))
                ),
                "proposer_controller_num_candidates": int(num_proposer_candidates),
                "proposer_warm_start_active": bool(proposer_warm_start_active),
                "proposer_warm_start_active_next": bool(
                    warm_start_state.get("active_next", 0.0) > 0.5
                ),
                "proposer_warm_start_completed": bool(
                    warm_start_state.get("completed", 0.0) > 0.5
                ),
                "proposer_warm_start_entropy_mean": warm_start_state.get("entropy_mean", 0.0),
                "proposer_warm_start_exit_streak": warm_start_state.get("exit_streak", 0.0),
                "proposer_warm_start_exit_pass": bool(
                    warm_start_state.get("exit_pass", 0.0) > 0.5
                ),
                "proposer_hardness_debt": hardness_debt_state.get("debt", 0.0),
                "proposer_hardness_debt_cap_streak": hardness_debt_state.get("cap_streak", 0.0),
                "proposer_hardness_debt_escape_steps_left": hardness_debt_state.get(
                    "escape_steps_left", 0.0
                ),
                "proposer_hardness_debt_escape_triggered": bool(
                    hardness_debt_state.get("escape_triggered", 0.0) > 0.5
                ),
                "proposer_easy_rate_ema": easy_constraint_state.get("easy_rate_ema"),
                "proposer_easy_lagrange_lambda": easy_constraint_state.get("easy_lambda"),
                "proposer_easy_reward_floor": easy_reward_floor,
                "proposer_collapse_streak": collapse_state.get("collapse_streak"),
                "proposer_collapse_mean_std": collapse_state.get("collapse_mean_std"),
                "proposer_collapse_hit": collapse_state.get("collapse_hit"),
                "proposer_bucket_baseline_easy": self._get_proposer_bucket_baseline("easy"),
                "proposer_bucket_baseline_medium": self._get_proposer_bucket_baseline("medium"),
                "proposer_bucket_baseline_hard": self._get_proposer_bucket_baseline("hard"),
                "proposer_non_objective_question": proposer_non_objective_question,
                "proposer_non_objective_penalty": proposer_non_objective_penalty,
                "question_rejected": question_rejected,
                "question_reject_reason": question_reject_reason,
                "rejected_question_penalty": rejected_question_penalty,
                "acceptance_require_non_easy": acceptance_require_non_easy,
                "zero_entropy_capped": zero_entropy_capped,
                "zero_entropy_reward_cap": zero_entropy_cap,
                "intuitive_answer": _intuitive_answer,
                "intuitive_answer_raw": _intuitive_answer_raw,
                "intuitive_answer_vote": _intuitive_answer_vote,
                "dual_track_agree": _tracks_agree,
                "intuitive_attempted": _intuitive_attempted,
                "intuitive_generation_failed": _intuitive_generation_failed,
                "unsolvable_solver_case": unsolvable_solver_case,
                "unsolvable_capped": unsolvable_capped,
                "easy_question_detected": easy_question_detected,
                "solver_skip_update_on_easy": solver_skip_update_on_easy,
                "solver_update_on_low_info_easy": bool(
                    getattr(self.cfg, "solver_update_on_low_info_easy", True)
                ),
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
                "difficulty_hardness_debt": difficulty_target_state.get("hardness_debt", 0.0),
                "difficulty_hardness_debt_ratio": difficulty_target_state.get(
                    "hardness_debt_ratio", 0.0
                ),
                "difficulty_hardness_debt_escape_active": bool(
                    difficulty_target_state.get("hardness_debt_escape_active", False)
                ),
                "proposer_early_u_step": int(early_failfast_state.get("u_step", 0.0)),
                "proposer_early_hard_stop_min_u_step": int(
                    early_failfast_state.get("hard_stop_min_u_step", 0.0)
                ),
                "proposer_early_stage1_active": bool(early_failfast_state.get("stage1_active", 0.0)),
                "proposer_early_stage1_pass": bool(early_failfast_state.get("stage1_pass", 1.0)),
                "proposer_early_stage2_active": bool(early_failfast_state.get("stage2_active", 0.0)),
                "proposer_early_stage2_pass": bool(early_failfast_state.get("stage2_pass", 1.0)),
                "proposer_early_recovery_armed": bool(
                    early_failfast_state.get("recovery_armed", 0.0)
                ),
                "proposer_early_triggered": bool(early_failfast_state.get("triggered", 0.0)),
            },
        )

        if self.is_main_process and step % self.cfg.log_every == 0:
            print(
                f"[Step {step:05d}][U] maj={maj_count}/{self.cfg.num_solver_samples} "
                f"maj_frac={maj_frac:.2f} H={entropy_nats:.3f} M={margin:.3f} "
                f"info_local={int(solver_informative_local)} "
                f"info_ratio={informative_ratio:.2f} info_gate={int(solver_informative_gate)} "
                f"li_maj={int(solver_low_info_majority)} li_rate={solver_low_info_rate:.2f} "
                f"nc_rate={solver_noncanonical_rate:.2f} "
                f"ch={int(solver_choice_mode)} "
                f"up_scale={solver_update_scale:.2f} P_R={proposer_reward:.3f} "
                f"T_B={proposer_text_hardness_bonus:.3f} R_P={proposer_repetition_penalty:.3f} "
                f"C_NE={candidate_non_easy_rate:.2f} C_V={candidate_struct_valid_rate:.2f} "
                f"ARM={curriculum_arm_score:.2f} "
                f"WS={int(proposer_warm_start_active)} "
                f"TE[{_ste_aggregation}]={_ste_raw_value:.2f} "
                f"SD={_ste_difficulty:.3f} "
                f"D={float(hardness_debt_state.get('debt', 0.0)):.2f} "
                f"E_L={easy_constraint_state.get('easy_lambda', 0.0):.3f} "
                f"E_R={easy_constraint_state.get('easy_rate_ema', 0.0):.2f} "
                f"C_ST={int(collapse_state.get('collapse_streak', 0.0))} "
                f"EF1={int(bool(early_failfast_state.get('stage1_pass', 1.0)))} "
                f"EF2={int(bool(early_failfast_state.get('stage2_pass', 1.0)))} "
                f"EFR={int(bool(early_failfast_state.get('recovery_armed', 0.0)))} "
                f"dt={step_dt:.1f}s"
            )
            print(f"  Q: {question}")

        self._update_metric("u_majority_fraction", self._dist_mean(maj_frac))
        self._update_metric("u_entropy_nats", self._dist_mean(entropy_nats))
        self._update_metric("u_solver_margin", self._dist_mean(margin))
        self._update_metric("u_solver_low_info_rate", self._dist_mean(solver_low_info_rate))
        self._update_metric("u_solver_noncanonical_rate", self._dist_mean(solver_noncanonical_rate))
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
        self._update_metric("u_proposer_strategy_bonus", self._dist_mean(proposer_strategy_quota_bonus))
        self._update_metric("u_proposer_anchor_bonus", self._dist_mean(proposer_anchor_bonus))
        self._update_metric("u_proposer_contrastive_bonus", self._dist_mean(proposer_contrastive_replay_bonus))
        self._update_metric(
            "u_proposer_answer_family_penalty",
            self._dist_mean(proposer_answer_family_penalty),
        )
        self._update_metric("u_proposer_repeat_penalty", self._dist_mean(proposer_repetition_penalty))
        self._update_metric("u_proposer_cooldown_penalty", self._dist_mean(proposer_cooldown_penalty))
        self._update_metric("u_proposer_easy_penalty", self._dist_mean(easy_constraint_penalty))
        self._update_metric("u_proposer_reward_clipped", self._dist_mean(1.0 if proposer_reward_clipped else 0.0))
        self._update_metric("u_candidate_non_easy_rate", self._dist_mean(candidate_non_easy_rate))
        self._update_metric("u_candidate_struct_valid_rate", self._dist_mean(candidate_struct_valid_rate))
        self._update_metric(
            "u_candidate_low_info_rate",
            self._dist_mean(candidate_low_info_rate_mean),
        )
        self._update_metric(
            "u_candidate_noncanonical_rate",
            self._dist_mean(candidate_noncanonical_rate_mean),
        )
        self._update_metric("u_candidate_all_easy_group", self._dist_mean(1.0 if all_easy_candidate_group else 0.0))
        self._update_metric("u_selected_cert_score", self._dist_mean(selected_cert_score))
        self._update_metric("u_curriculum_arm_score", self._dist_mean(curriculum_arm_score))
        self._update_metric(
            "u_proposer_warm_start_active",
            self._dist_mean(1.0 if proposer_warm_start_active else 0.0),
        )
        self._update_metric(
            "u_proposer_warm_start_entropy_mean",
            self._dist_mean(float(warm_start_state.get("entropy_mean", 0.0))),
        )
        self._update_metric(
            "u_proposer_token_entropy_max",
            self._dist_mean(float(_intuitive_token_entropy_max)),
        )
        self._update_metric(
            "u_proposer_ste_difficulty",
            self._dist_mean(float(_ste_difficulty)),
        )
        self._update_metric(
            "u_proposer_hardness_debt",
            self._dist_mean(float(hardness_debt_state.get("debt", 0.0))),
        )
        self._update_metric(
            "u_proposer_hardness_debt_escape",
            self._dist_mean(1.0 if hardness_debt_state.get("escape_triggered", 0.0) > 0.5 else 0.0),
        )
        self._update_metric(
            "u_early_stage1_pass",
            self._dist_mean(float(early_failfast_state.get("stage1_pass", 1.0))),
        )
        self._update_metric(
            "u_early_stage2_pass",
            self._dist_mean(float(early_failfast_state.get("stage2_pass", 1.0))),
        )
        self._update_metric(
            "u_early_recovery_armed",
            self._dist_mean(float(early_failfast_state.get("recovery_armed", 0.0))),
        )
        self._update_metric(
            "u_proposer_easy_rate_ema",
            self._dist_mean(float(easy_constraint_state.get("easy_rate_ema", 0.0))),
        )
        self._update_metric(
            "u_proposer_easy_lambda",
            self._dist_mean(float(easy_constraint_state.get("easy_lambda", 0.0))),
        )
        self._update_metric(
            "u_proposer_collapse_streak",
            self._dist_mean(float(collapse_state.get("collapse_streak", 0.0))),
        )
        self._update_metric(
            "u_proposer_collapse_mean_std",
            self._dist_mean(float(collapse_state.get("collapse_mean_std", 0.0))),
        )

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
        state["unified_answer_family_window"] = list(self._answer_family_window)
        state["unified_strategy_window"] = list(self._strategy_window)
        state["unified_proposer_anchor_replay"] = list(self._proposer_anchor_replay)
        state["unified_contrastive_pos_replay"] = [sorted(list(x)) for x in self._contrastive_pos_replay]
        state["unified_contrastive_neg_replay"] = [sorted(list(x)) for x in self._contrastive_neg_replay]
        state["unified_grpo_std_window"] = list(self._grpo_std_window)
        state["unified_candidate_non_easy_window"] = list(self._candidate_non_easy_window)
        state["unified_all_easy_group_window"] = list(self._all_easy_group_window)
        state["unified_proposer_reward_clipped_window"] = list(self._proposer_reward_clipped_window)
        state["unified_selected_non_easy_window"] = list(self._selected_non_easy_window)
        state["unified_solver_update_applied_window"] = list(self._solver_update_applied_window)
        state["unified_curriculum_arm_stats"] = dict(getattr(self, "_curriculum_arm_stats", {}))
        state["unified_last_curriculum_arm_key"] = str(
            getattr(self, "_last_curriculum_arm_key", "")
        )
        state["unified_easy_rate_ema"] = float(getattr(self, "_easy_rate_ema", 0.0))
        state["unified_easy_lagrange_lambda"] = float(getattr(self, "_easy_lagrange_lambda", 0.0))
        state["unified_collapse_streak"] = int(getattr(self, "_collapse_streak", 0))
        state["unified_forced_explore_steps_left"] = int(
            getattr(self, "_forced_explore_steps_left", 0)
        )
        state["unified_warm_start_entropy_window"] = list(
            getattr(self, "_warm_start_entropy_window", [])
        )
        state["unified_warm_start_exit_streak"] = int(
            getattr(self, "_warm_start_exit_streak", 0)
        )
        state["unified_warm_start_completed"] = bool(
            getattr(self, "_warm_start_completed", False)
        )
        state["unified_hardness_debt"] = float(getattr(self, "_hardness_debt", 0.0))
        state["unified_hardness_debt_cap_streak"] = int(
            getattr(self, "_hardness_debt_cap_streak", 0)
        )
        state["unified_hardness_debt_escape_steps_left"] = int(
            getattr(self, "_hardness_debt_escape_steps_left", 0)
        )
        state["unified_ste_window"] = list(
            getattr(self, "_ste_window", [])
        )
        state["unified_proposer_bucket_baselines"] = dict(self._proposer_bucket_baselines)
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
            answer_family_window = state.get("unified_answer_family_window")
            if isinstance(answer_family_window, list):
                self._answer_family_window.clear()
                max_keep = int(self._answer_family_window.maxlen or len(answer_family_window))
                for key in answer_family_window[-max_keep:]:
                    k = str(key).strip().lower()
                    if k:
                        self._answer_family_window.append(k)
            strategy_window = state.get("unified_strategy_window")
            if isinstance(strategy_window, list):
                self._strategy_window.clear()
                max_keep = int(self._strategy_window.maxlen or len(strategy_window))
                for key in strategy_window[-max_keep:]:
                    k = self._normalize_strategy_key(str(key))
                    if k:
                        self._strategy_window.append(k)
            anchor_replay = state.get("unified_proposer_anchor_replay")
            if isinstance(anchor_replay, list):
                self._proposer_anchor_replay.clear()
                max_keep = int(self._proposer_anchor_replay.maxlen or len(anchor_replay))
                for item in anchor_replay[-max_keep:]:
                    if isinstance(item, dict):
                        self._proposer_anchor_replay.append(dict(item))
            contrastive_pos = state.get("unified_contrastive_pos_replay")
            if isinstance(contrastive_pos, list):
                self._contrastive_pos_replay.clear()
                max_keep = int(self._contrastive_pos_replay.maxlen or len(contrastive_pos))
                for item in contrastive_pos[-max_keep:]:
                    if isinstance(item, (list, tuple, set)):
                        s = {str(x).strip() for x in item if str(x).strip()}
                        if s:
                            self._contrastive_pos_replay.append(s)
            contrastive_neg = state.get("unified_contrastive_neg_replay")
            if isinstance(contrastive_neg, list):
                self._contrastive_neg_replay.clear()
                max_keep = int(self._contrastive_neg_replay.maxlen or len(contrastive_neg))
                for item in contrastive_neg[-max_keep:]:
                    if isinstance(item, (list, tuple, set)):
                        s = {str(x).strip() for x in item if str(x).strip()}
                        if s:
                            self._contrastive_neg_replay.append(s)
            std_window = state.get("unified_grpo_std_window")
            if isinstance(std_window, list):
                self._grpo_std_window.clear()
                max_keep = int(self._grpo_std_window.maxlen or len(std_window))
                for v in std_window[-max_keep:]:
                    try:
                        self._grpo_std_window.append(float(v))
                    except Exception:
                        continue
            cand_non_easy_window = state.get("unified_candidate_non_easy_window")
            if isinstance(cand_non_easy_window, list):
                self._candidate_non_easy_window.clear()
                max_keep = int(self._candidate_non_easy_window.maxlen or len(cand_non_easy_window))
                for v in cand_non_easy_window[-max_keep:]:
                    try:
                        self._candidate_non_easy_window.append(float(v))
                    except Exception:
                        continue
            all_easy_window = state.get("unified_all_easy_group_window")
            if isinstance(all_easy_window, list):
                self._all_easy_group_window.clear()
                max_keep = int(self._all_easy_group_window.maxlen or len(all_easy_window))
                for v in all_easy_window[-max_keep:]:
                    try:
                        self._all_easy_group_window.append(float(v))
                    except Exception:
                        continue
            clipped_window = state.get("unified_proposer_reward_clipped_window")
            if isinstance(clipped_window, list):
                self._proposer_reward_clipped_window.clear()
                max_keep = int(self._proposer_reward_clipped_window.maxlen or len(clipped_window))
                for v in clipped_window[-max_keep:]:
                    try:
                        self._proposer_reward_clipped_window.append(float(v))
                    except Exception:
                        continue
            selected_non_easy_window = state.get("unified_selected_non_easy_window")
            if isinstance(selected_non_easy_window, list):
                self._selected_non_easy_window.clear()
                max_keep = int(self._selected_non_easy_window.maxlen or len(selected_non_easy_window))
                for v in selected_non_easy_window[-max_keep:]:
                    try:
                        self._selected_non_easy_window.append(float(v))
                    except Exception:
                        continue
            solver_applied_window = state.get("unified_solver_update_applied_window")
            if isinstance(solver_applied_window, list):
                self._solver_update_applied_window.clear()
                max_keep = int(self._solver_update_applied_window.maxlen or len(solver_applied_window))
                for v in solver_applied_window[-max_keep:]:
                    try:
                        self._solver_update_applied_window.append(float(v))
                    except Exception:
                        continue
            curr_stats = state.get("unified_curriculum_arm_stats")
            if isinstance(curr_stats, dict):
                self._curriculum_arm_stats = {}
                for key, val in curr_stats.items():
                    k = str(key).strip().lower()
                    if not k:
                        continue
                    if isinstance(val, dict):
                        self._curriculum_arm_stats[k] = {
                            "count": float(val.get("count", 0.0) or 0.0),
                            "non_easy_ema": float(val.get("non_easy_ema", 0.0) or 0.0),
                            "solver_gain_ema": float(val.get("solver_gain_ema", 0.0) or 0.0),
                            "prev_solver_gain_ema": float(val.get("prev_solver_gain_ema", 0.0) or 0.0),
                            "progress_ema": float(val.get("progress_ema", 0.0) or 0.0),
                        }
            self._last_curriculum_arm_key = str(
                state.get("unified_last_curriculum_arm_key", getattr(self, "_last_curriculum_arm_key", ""))
            )
            self._easy_rate_ema = float(state.get("unified_easy_rate_ema", self._easy_rate_ema))
            self._easy_lagrange_lambda = float(
                state.get("unified_easy_lagrange_lambda", self._easy_lagrange_lambda)
            )
            self._collapse_streak = int(state.get("unified_collapse_streak", self._collapse_streak))
            self._forced_explore_steps_left = int(
                state.get("unified_forced_explore_steps_left", self._forced_explore_steps_left)
            )
            warm_start_window = state.get("unified_warm_start_entropy_window")
            if isinstance(warm_start_window, list):
                self._warm_start_entropy_window.clear()
                max_keep = int(
                    self._warm_start_entropy_window.maxlen or len(warm_start_window)
                )
                for v in warm_start_window[-max_keep:]:
                    try:
                        self._warm_start_entropy_window.append(float(v))
                    except Exception:
                        continue
            self._warm_start_exit_streak = int(
                state.get("unified_warm_start_exit_streak", self._warm_start_exit_streak)
            )
            self._warm_start_completed = bool(
                state.get("unified_warm_start_completed", self._warm_start_completed)
            )
            self._hardness_debt = float(
                state.get("unified_hardness_debt", self._hardness_debt)
            )
            self._hardness_debt_cap_streak = int(
                state.get(
                    "unified_hardness_debt_cap_streak",
                    self._hardness_debt_cap_streak,
                )
            )
            self._hardness_debt_escape_steps_left = int(
                state.get(
                    "unified_hardness_debt_escape_steps_left",
                    self._hardness_debt_escape_steps_left,
                )
            )
            _restored_ste_window = state.get("unified_ste_window")
            if isinstance(_restored_ste_window, list):
                self._ste_window = [float(x) for x in _restored_ste_window]
            bucket_baselines = state.get("unified_proposer_bucket_baselines")
            if isinstance(bucket_baselines, dict):
                for b in ("easy", "medium", "hard"):
                    if b in bucket_baselines:
                        try:
                            self._proposer_bucket_baselines[b] = float(bucket_baselines[b])
                        except Exception:
                            continue
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
                self._answer_family_window.clear()
                self._strategy_window.clear()
                self._proposer_anchor_replay.clear()
                # Re-seed with hard question exemplars after reset so
                # the proposer has guidance from step 0 of the new run.
                self._seed_anchor_exemplars()
                self._contrastive_pos_replay.clear()
                self._contrastive_neg_replay.clear()
                self._grpo_std_window.clear()
                self._candidate_non_easy_window.clear()
                self._all_easy_group_window.clear()
                self._proposer_reward_clipped_window.clear()
                self._selected_non_easy_window.clear()
                self._solver_update_applied_window.clear()
                self._curriculum_arm_stats = {}
                self._last_curriculum_arm_key = ""
                self._easy_rate_ema = 0.0
                self._easy_lagrange_lambda = 0.0
                self._collapse_streak = 0
                self._forced_explore_steps_left = 0
                self._warm_start_entropy_window.clear()
                self._warm_start_exit_streak = 0
                self._warm_start_completed = False
                self._hardness_debt = 0.0
                self._hardness_debt_cap_streak = 0
                self._hardness_debt_escape_steps_left = 0
                self._ste_window = []
                self._proposer_bucket_baselines = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
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
        bootstrap_steps = max(0, int(getattr(cfg, "bootstrap_generated_pool_steps", 0)))
        cycle_order = (
            "G->U" if bool(getattr(cfg, "cycle_starts_with_generation", False)) else "U->G"
        )

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
                f"[Unified] Cycle order: {cycle_order}, bootstrap_generated_pool_steps={bootstrap_steps}"
            )
            print(
                f"[Unified] Solver-derived reference-answer scoring: {getattr(cfg, 'use_ref_answer_scoring', False)}, "
                f"Replay buffer: size={len(self.replay_buffer) if self.replay_buffer is not None else 0}, "
                f"Mix ratio: {getattr(cfg, 'gen_mix_ratio_start', 0)}->{getattr(cfg, 'gen_mix_ratio_max', 0)}"
            )
            print(
                f"[Unified] Gen-mix source mode: {self._gen_mix_source_mode}, "
                f"generated_only={self._understanding_generated_only}, "
                f"generated_mix_dir={self._generated_mix_dir}"
            )

        run_started_at = float(time.time())
        last_completed_step = self.start_step
        last_attempted_step = self.start_step

        def _emit_training_logs(step_id: int, *, phase: str, step_time_sec: float):
            progress = self._progress_core(
                step=int(step_id),
                phase=str(phase),
                run_started_at=run_started_at,
            )
            metrics = self._release_metrics(step_time_sec=step_time_sec)
            if self._gen_reward_ema_initialized:
                metrics["generator_reward_ema"] = float(self._gen_reward_ema)
            metrics["replay_buffer_size"] = int(len(self.replay_buffer) if self.replay_buffer is not None else 0)
            self._write_status(state="running", progress=progress, metrics=metrics)
            if int(step_id) % max(1, int(cfg.log_every)) == 0:
                self._append_metrics({"kind": "heartbeat", **progress, **metrics})

        init_metrics = self._release_metrics(step_time_sec=0.0)
        if self._gen_reward_ema_initialized:
            init_metrics["generator_reward_ema"] = float(self._gen_reward_ema)
        init_metrics["replay_buffer_size"] = int(len(self.replay_buffer) if self.replay_buffer is not None else 0)
        self._write_status(
            state="running",
            progress=self._progress_core(
                step=int(self.start_step),
                phase="init",
                run_started_at=run_started_at,
            ),
            metrics=init_metrics,
        )
        try:
            for step in range(self.start_step + 1, cfg.total_steps + 1):
                step_t0 = time.perf_counter()
                last_attempted_step = step
                phase_name, _phase_local_idx = self._phase_for_step(step)
                phase_tag = "U" if phase_name == "understanding" else "G"
                _data_source = "real"
                image: Optional[Image.Image]
                meta: Dict[str, Any]

                if phase_name == "understanding":
                    # ---- Understanding phase: optional generated-image mix ---- #
                    image = None
                    meta = {"path": None, "source": "generated_pool"}
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
                        skip_record = {
                            "step": step,
                            "phase": "understanding",
                            "image_path": meta.get("path"),
                            "skip_reason": "generated_pool_empty",
                            "gen_mix_source_mode": self._gen_mix_source_mode,
                            "understanding_generated_only": True,
                        }
                        self._append_jsonl(self.iter_log_path, skip_record)
                        self._append_training_monitor(
                            {
                                "step": int(step),
                                "phase": "understanding",
                                "health": "skipped_or_waiting",
                                "image_path": meta.get("path"),
                                "solver_skip": "generated_pool_empty",
                            }
                        )
                    else:
                        if not _used_generated:
                            image, meta = self._sample_image_for_step(step)
                            meta["source"] = "real"
                            _data_source = "real"
                        self._understanding_step(step=step, image=image, meta=meta)
                else:
                    phase_tag = "G"
                    if bool(getattr(cfg, "imageless_proposer_mode", False)):
                        image = None
                        meta = {"path": None, "source": "imageless_topic"}
                        _data_source = "imageless"
                    else:
                        image, meta = self._sample_image_for_step(step)
                        _data_source = "real"
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
                _emit_training_logs(step, phase=phase_name, step_time_sec=time.perf_counter() - step_t0)

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
            summary = self._write_ablation_summary(cfg.total_steps, status="completed")
            final_progress = self._progress_core(
                step=int(cfg.total_steps),
                phase="completed",
                run_started_at=run_started_at,
            )
            final_metrics = self._release_metrics(step_time_sec=0.0)
            if self._gen_reward_ema_initialized:
                final_metrics["generator_reward_ema"] = float(self._gen_reward_ema)
            final_metrics["replay_buffer_size"] = int(len(self.replay_buffer) if self.replay_buffer is not None else 0)
            self._append_metrics({"kind": "final_summary", **final_progress, **summary})
            self._write_status(state="completed", progress=final_progress, metrics=final_metrics)
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

            summary = self._write_ablation_summary(
                max(last_completed_step, emergency_step),
                status="interrupted",
                interrupted_at_step=interrupted_step,
                error=error_text,
            )
            interrupted_progress = self._progress_core(
                step=max(last_completed_step, emergency_step),
                phase="interrupted",
                run_started_at=run_started_at,
            )
            interrupted_metrics = self._release_metrics(step_time_sec=0.0)
            if self._gen_reward_ema_initialized:
                interrupted_metrics["generator_reward_ema"] = float(self._gen_reward_ema)
            interrupted_metrics["replay_buffer_size"] = int(len(self.replay_buffer) if self.replay_buffer is not None else 0)
            self._append_metrics({"kind": "interrupted", **interrupted_progress, **summary})
            self._write_status(
                state="interrupted",
                progress=interrupted_progress,
                metrics=interrupted_metrics,
                last_error=error_text,
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
