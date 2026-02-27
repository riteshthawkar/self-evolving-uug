"""
SelfEvolvingTrainer: Main training loop for the self-evolving framework on VARGPT.

Inherits from HF Trainer but overrides train() with the custom multi-phase
proposer-solver-generator loop:
  - U-step: proposer proposes questions, solver answers, both get GRPO rewards
  - G-step: proposer proposes specs, generator creates images, GRPO on discrete tokens

Ported from BLIP3o's unified_trainer.py with VARGPT-specific adaptations.
"""

import collections
import gc
import json
import logging
import math
import os
import pathlib
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from PIL import Image
from transformers import Trainer

from .adapter_manager import (
    ROLE_GENERATOR,
    ROLE_PROPOSER,
    ROLE_SOLVER,
    use_role,
    collect_role_params,
    get_role_optimizer,
)
from .config import SelfEvolvingConfig
from .generation_helpers import (
    GenerationSpec,
    _ensure_pil_image,
    _parse_generation_spec,
)
from .gen_policy_updater import VARImageGenPolicyUpdater
from .policy_updater import RolePolicyUpdater
from .prompts import (
    _sample_imageless_topic,
    build_generation_spec_prompt,
    build_imageless_spec_prompt,
    build_proposer_multi_prompt,
    build_proposer_prompt,
    build_solver_prompt,
    build_generator_prompt,
)
from .replay_buffer import ReplayBuffer
from .rewards import score_generated_image
from .utils import (
    _build_chat_text,
    _build_text_only_chat,
    _decode_tokens,
    _json_dump,
    _prepare_mm_inputs,
    _prepare_text_only_inputs,
    _set_global_seed,
    _unwrap_model,
    gaussian_reward,
    majority_vote,
    normalize_answer,
    shannon_entropy_nats,
    use_adapter,
    _parse_answer,
    _parse_all_questions,
    _parse_first_question,
)


logger = logging.getLogger(__name__)


class SelfEvolvingTrainer(Trainer):
    """Self-evolving proposer-solver-generator trainer for VARGPT v1.1.

    Overrides HF Trainer's train() with a custom multi-phase loop.
    Reuses HF Trainer for: model save/load, logging, callbacks, DDP setup.
    """

    def __init__(
        self,
        model,
        args,
        se_config: SelfEvolvingConfig,
        finetuning_args=None,
        **kwargs,
    ):
        self._processor = kwargs.pop("processor", None)
        super().__init__(model=model, args=args, **kwargs)
        self.se_config = se_config
        self.finetuning_args = finetuning_args

        # Get processor/tokenizer
        self.processor = self._processor or self.tokenizer

        # Device
        self.device = (
            torch.device(f"cuda:{args.local_rank}")
            if args.local_rank >= 0 and torch.cuda.is_available()
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── Role-specific updaters ──────────────────────────────────────
        self.proposer_updater = RolePolicyUpdater(
            model=model,
            processor=self.processor,
            config=se_config,
            adapter_name=ROLE_PROPOSER,
        )
        self.solver_updater = RolePolicyUpdater(
            model=model,
            processor=self.processor,
            config=se_config,
            adapter_name=ROLE_SOLVER,
        )
        self.generator_updater = VARImageGenPolicyUpdater(
            model=model,
            tokenizer=self.processor,
            config=se_config,
        )

        # ── Replay buffer ───────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(
            max_size=se_config.replay_buffer_size,
            min_reward=se_config.replay_min_reward,
            max_staleness=se_config.replay_max_staleness,
        )

        # ── Baselines & tracking ────────────────────────────────────────
        self.proposer_baseline = 0.0
        self.solver_baseline = 0.0
        self.generator_baseline = 0.0
        self.proposer_gen_baseline = 0.0
        self.reward_ema = 0.0
        self.global_step = 0

        # ── Adaptive entropy target (EMA) ──────────────────────────────
        self.proposer_entropy_mu_ema = float(se_config.prop_entropy_mu)

        # ── Entropy & difficulty tracking windows ──────────────────────
        self._entropy_window: collections.deque = collections.deque(
            maxlen=se_config.entropy_iqr_window_size,
        )
        self._difficulty_window: collections.deque = collections.deque(
            maxlen=se_config.difficulty_sampler_window_size,
        )

        # ── Proposer controller / failfast state ────────────────────────
        failfast_window = 128
        self._candidate_non_easy_window: collections.deque = collections.deque(maxlen=failfast_window)
        self._all_easy_group_window: collections.deque = collections.deque(maxlen=failfast_window)
        self._proposer_reward_clipped_window: collections.deque = collections.deque(maxlen=failfast_window)
        self._selected_non_easy_window: collections.deque = collections.deque(maxlen=failfast_window)
        self._solver_update_applied_window: collections.deque = collections.deque(maxlen=failfast_window)
        self._entropy_easy_window: collections.deque = collections.deque(
            maxlen=max(32, int(getattr(se_config, "entropy_iqr_window_size", 256)))
        )

        self._all_easy_streak: int = 0
        self._forced_explore_steps_left: int = 0
        self._proposer_collapse_streak: int = 0
        self._u_step_counter: int = 0

        # Warm-start and hardness-debt state
        ws_window = max(1, int(getattr(se_config, "proposer_warm_start_exit_window", 5)))
        self._warm_start_entropy_window: collections.deque = collections.deque(maxlen=ws_window)
        self._warm_start_exit_streak: int = 0
        self._warm_start_completed: bool = False
        self._hardness_debt: float = 0.0
        self._hardness_debt_cap_streak: int = 0
        self._hardness_debt_escape_steps_left: int = 0

        # ── DDP detection ────────────────────────────────────────────────
        self._is_ddp = dist.is_available() and dist.is_initialized()

        # ── Image folder mode ────────────────────────────────────────────
        # When se_image_folder is set, scan the folder for images at init
        # so _sample_image() can pick random images without a JSON dataset.
        self._image_folder_paths: List[str] = []
        if se_config.image_folder:
            self._image_folder_paths = self._scan_image_folder(
                se_config.image_folder
            )
            logger.info(
                f"[SelfEvolvingTrainer] Image folder mode: "
                f"found {len(self._image_folder_paths)} images in "
                f"{se_config.image_folder}"
            )

        logger.info(
            f"[SelfEvolvingTrainer] Initialized with "
            f"U={se_config.understanding_steps_per_cycle}, "
            f"G={se_config.generation_steps_per_cycle}, "
            f"total_steps={se_config.total_steps}"
        )

    # ── Main training loop ──────────────────────────────────────────────

    def train(self, resume_from_checkpoint=None, **kwargs):
        """Override HF Trainer's train() with the self-evolving multi-phase loop."""
        cfg = self.se_config

        # Set seed
        _set_global_seed(cfg.seed)

        # Resume
        start_step = cfg.start_step
        if resume_from_checkpoint:
            start_step = self._load_se_checkpoint(resume_from_checkpoint)

        # Create output directory
        output_dir = pathlib.Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        _json_dump(output_dir / "se_config.json", {
            k: str(v) if not isinstance(v, (int, float, bool, str, type(None)))
            else v
            for k, v in cfg.__dict__.items()
        })

        logger.info(f"[SelfEvolvingTrainer] Starting training from step {start_step}")

        cycle_len = cfg.understanding_steps_per_cycle + cfg.generation_steps_per_cycle
        if cycle_len <= 0:
            raise ValueError("Cycle length must be > 0")

        self.model.train()

        for step in range(start_step, cfg.total_steps):
            self.global_step = step
            step_start = time.time()

            # Determine phase
            phase_in_cycle = step % cycle_len
            is_u_step = phase_in_cycle < cfg.understanding_steps_per_cycle

            try:
                if is_u_step:
                    step_stats = self._understanding_step(step)
                    step_stats["phase"] = "understanding"
                else:
                    step_stats = self._generation_step(step)
                    step_stats["phase"] = "generation"
            except Exception as e:
                logger.error(f"[SelfEvolvingTrainer] Step {step} failed: {e}")
                import traceback
                traceback.print_exc()
                step_stats = {"phase": "error", "error": str(e)}

            step_stats["step"] = step
            step_stats["step_time"] = time.time() - step_start

            # ── Logging ─────────────────────────────────────────────────
            if step % cfg.log_every == 0:
                self._log_step(step, step_stats)

            # ── Checkpointing ───────────────────────────────────────────
            if step > 0 and step % cfg.save_every == 0:
                self._save_se_checkpoint(step, output_dir)

            # ── Memory management ───────────────────────────────────────
            if cfg.clear_cache_every > 0 and step % cfg.clear_cache_every == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final save
        self._save_se_checkpoint(cfg.total_steps, output_dir)
        logger.info("[SelfEvolvingTrainer] Training complete.")

    # ── Understanding Step ──────────────────────────────────────────────

    def _understanding_step(self, step: int) -> Dict:
        """U-step: proposer proposes questions, solver answers, both update.

        When ``acceptance_require_non_easy`` is enabled, the proposer is given
        up to ``_max_easy_retries`` attempts to generate a non-trivial question.
        If all attempts produce easy questions, the last attempt is used with
        the easy-question penalty applied to the proposer reward.
        """
        cfg = self.se_config
        stats = {}
        self._u_step_counter += 1
        u_step = int(self._u_step_counter)
        max_retries = 3 if cfg.acceptance_require_non_easy else 1

        # ── 1. Sample image ─────────────────────────────────────────────
        image, source = self._sample_image(step)
        if image is None:
            return {"u_skipped": True, "reason": "no_image"}
        stats["image_source"] = source

        question_text = None
        proposer_completion = None
        answers = []
        solver_completions = []
        candidate_records: List[Dict[str, object]] = []
        selected_candidate: Optional[Dict[str, object]] = None
        selected_candidate_idx: int = -1
        selected_meta: Dict[str, str] = {}
        controller_state = self._choose_difficulty_target()
        desired_difficulty_bucket = str(controller_state.get("desired_bucket", "medium"))
        warm_start_active = bool(self._is_proposer_warm_start_active(u_step))
        controller_mode = str(controller_state.get("mode", "target"))
        debt_ratio = float(controller_state.get("hardness_debt_ratio", 0.0))
        controller_penalty_boost = 1.0 + debt_ratio * max(
            0.0, float(getattr(cfg, "hardness_debt_penalty_boost_max", 0.30))
        )
        proposer_temp = float(cfg.temp) + debt_ratio * max(
            0.0, float(getattr(cfg, "hardness_debt_temp_boost_max", 0.30))
        )
        proposer_top_p = float(cfg.top_p)
        proposer_num_candidates = max(1, int(cfg.proposer_num_candidates))
        if "forced_explore" in controller_mode:
            proposer_num_candidates = max(
                proposer_num_candidates,
                int(getattr(cfg, "all_easy_explore_num_candidates", proposer_num_candidates)),
            )
            proposer_temp += max(0.0, float(getattr(cfg, "all_easy_explore_temp_boost", 1.20)))
            proposer_top_p = min(
                1.0,
                proposer_top_p + max(0.0, float(getattr(cfg, "all_easy_explore_top_p_boost", 0.20))),
            )
            controller_penalty_boost += max(
                0.0,
                float(getattr(cfg, "all_easy_explore_penalty_boost", 0.70)),
            )
        proposer_temp = max(0.05, min(3.5, proposer_temp))
        proposer_top_p = max(0.05, min(1.0, proposer_top_p))
        attempt = 0

        for attempt in range(max_retries):
            # ── 2. Proposer generates question candidates ───────────────
            with use_role(self.model, ROLE_PROPOSER):
                candidates, proposer_raw = self._generate_proposer_candidates(
                    image,
                    step,
                    target_difficulty=desired_difficulty_bucket,
                    num_candidates=proposer_num_candidates,
                    temperature=proposer_temp,
                    top_p=proposer_top_p,
                )
            if not candidates:
                continue

            spot_n = max(1, min(int(cfg.proposer_spot_check_samples), int(cfg.num_solver_samples)))
            current_candidates: List[Dict[str, object]] = []
            entropy_mu_for_spot = (
                float(self.proposer_entropy_mu_ema)
                if bool(cfg.adaptive_prop_entropy_target)
                else float(cfg.prop_entropy_mu)
            )

            # ── 3. Spot-check each candidate with reduced solver budget ─
            for cand_idx, cand in enumerate(candidates):
                q = str(cand.get("question", "")).strip()
                if not q:
                    continue
                with use_role(self.model, ROLE_SOLVER):
                    ans_spot, sc_spot = self._generate_solver_answers(
                        image, q, num_samples=spot_n,
                    )
                if not ans_spot:
                    continue

                norm = [normalize_answer(a) for a in ans_spot]
                norm = [n for n in norm if n]
                if not norm:
                    continue

                _, mc = majority_vote(norm)
                mf = mc / len(norm)
                counts_spot: Dict[str, int] = {}
                for a in norm:
                    counts_spot[a] = counts_spot.get(a, 0) + 1
                probs_spot = [c / len(norm) for c in counts_spot.values()]
                ent_spot = shannon_entropy_nats(probs_spot)
                margin_spot = mf - self._second_highest_frac(norm)
                easy_spot = (ent_spot < cfg.sc_entropy_min) and (margin_spot > cfg.sc_margin_max)

                # Spot reward used only for candidate ranking/group update seed.
                spot_reward = gaussian_reward(
                    ent_spot, entropy_mu_for_spot, cfg.prop_entropy_sigma,
                )
                if ent_spot < 0.01:
                    spot_reward = -cfg.zero_entropy_reward_cap
                elif easy_spot:
                    spot_reward = min(spot_reward, -cfg.zero_entropy_reward_cap * 0.5)
                if cfg.acceptance_require_non_easy and easy_spot:
                    spot_reward -= controller_penalty_boost * float(cfg.rejected_question_penalty)

                meta = dict(cand.get("meta", {}))
                objective_ok = bool(self._is_objective_question(q))
                if cfg.proposer_require_objective and not objective_ok:
                    spot_reward -= controller_penalty_boost * float(cfg.proposer_non_objective_penalty)

                cert = self._proposer_certificate_score(q, meta)
                cert_score = float(cert.get("score", 0.0))
                cert_valid = float(cert.get("valid", 0.0))
                cert_bonus = 0.0
                if bool(getattr(cfg, "proposer_certificate_enabled", True)):
                    cert_weight_cfg = float(
                        getattr(
                            cfg,
                            "proposer_warm_start_certificate_weight"
                            if warm_start_active
                            else "proposer_certificate_weight",
                            0.50 if warm_start_active else 0.75,
                        )
                    )
                    cert_weight = max(0.0, cert_weight_cfg)
                    cert_min = max(
                        0.0,
                        min(
                            1.0,
                            min(
                                float(getattr(cfg, "proposer_certificate_min_score", 0.55)),
                                0.50 if warm_start_active else 1.0,
                            ),
                        ),
                    )
                    cert_bonus = cert_weight * (cert_score - cert_min)
                    spot_reward += cert_bonus

                current_candidates.append(
                    {
                        "candidate_index": int(cand_idx),
                        "question": q,
                        "completion": str(cand.get("completion", "")),
                        "meta": meta,
                        "spot_answers_raw": list(ans_spot),
                        "spot_solver_completions": list(sc_spot),
                        "spot_entropy": float(ent_spot),
                        "spot_margin": float(margin_spot),
                        "spot_majority_frac": float(mf),
                        "easy_spot": bool(easy_spot),
                        "objective_ok": bool(objective_ok),
                        "certificate_score": float(cert_score),
                        "certificate_valid": float(cert_valid),
                        "certificate_bonus": float(cert_bonus),
                        "spot_reward": float(spot_reward),
                    }
                )

            if not current_candidates:
                continue

            # Select best candidate: prefer non-easy, then stronger spot reward, then entropy.
            selected_now = max(
                current_candidates,
                key=lambda c: (
                    1.0 if not bool(c.get("easy_spot", True)) else 0.0,
                    float(c.get("certificate_valid", 0.0)),
                    float(c.get("certificate_score", 0.0)),
                    float(c.get("spot_reward", -1e9)),
                    float(c.get("spot_entropy", 0.0)),
                ),
            )

            candidate_records = current_candidates
            selected_candidate = selected_now
            selected_candidate_idx = int(selected_now.get("candidate_index", -1))
            selected_meta = dict(selected_now.get("meta", {}))
            question_text = str(selected_now.get("question", "")).strip()
            proposer_completion = str(selected_now.get("completion", ""))
            if not proposer_completion:
                proposer_completion = f"<question>{question_text}</question>"

            if (
                question_text
                and (
                    (not cfg.acceptance_require_non_easy)
                    or (not bool(selected_now.get("easy_spot", True)))
                )
            ):
                break
            # Otherwise retry with a new proposer pass

        if not question_text:
            return {"u_skipped": True, "reason": "no_question"}

        # ── 4. Full solver rollout on selected candidate ────────────────
        with use_role(self.model, ROLE_SOLVER):
            answers, solver_completions = self._generate_solver_answers(
                image, question_text, num_samples=cfg.num_solver_samples,
            )
        if not answers:
            return {"u_skipped": True, "reason": "no_answers"}

        stats["question"] = question_text[:100]
        stats["num_answers"] = len(answers)
        stats["proposer_retries"] = attempt
        stats["u_step"] = u_step
        stats["difficulty_target_bucket"] = desired_difficulty_bucket
        stats["difficulty_sampler_enabled"] = 1.0 if bool(controller_state.get("enabled", False)) else 0.0
        stats["difficulty_sampler_mode"] = str(controller_state.get("mode", "target"))
        stats["difficulty_target_weights"] = dict(controller_state.get("target_weights", {}))
        stats["difficulty_observed_weights"] = dict(controller_state.get("observed_weights", {}))
        stats["difficulty_sampling_weights"] = dict(controller_state.get("sampling_weights", {}))
        stats["proposer_controller_temp"] = float(proposer_temp)
        stats["proposer_controller_top_p"] = float(proposer_top_p)
        stats["proposer_controller_penalty_boost"] = float(controller_penalty_boost)
        stats["proposer_warm_start_active"] = bool(warm_start_active)
        stats["proposer_candidate_count"] = len(candidate_records)
        stats["proposer_selected_candidate_index"] = selected_candidate_idx
        if candidate_records:
            stats["proposer_candidate_non_easy_rate"] = (
                sum(1 for c in candidate_records if not bool(c.get("easy_spot", True)))
                / float(len(candidate_records))
            )
        else:
            stats["proposer_candidate_non_easy_rate"] = 0.0

        # ── 5. Compute rewards ──────────────────────────────────────────
        # Normalize answers for voting
        norm_answers = [normalize_answer(a) for a in answers]
        if not norm_answers:
            return {"u_skipped": True, "reason": "no_answers"}

        # Majority vote and entropy
        majority_answer, majority_count = majority_vote(norm_answers)
        majority_frac = majority_count / len(norm_answers)

        # Entropy
        counts = {}
        for a in norm_answers:
            counts[a] = counts.get(a, 0) + 1
        probs = [c / len(norm_answers) for c in counts.values()]
        entropy = shannon_entropy_nats(probs)

        # ── Difficulty classification ────────────────────────────────
        margin = majority_frac - self._second_highest_frac(norm_answers)
        easy_question = (entropy < cfg.sc_entropy_min) and (margin > cfg.sc_margin_max)

        # ── Track entropy & difficulty for adaptive thresholds ────
        self._entropy_window.append(entropy)
        diff_bucket = self._difficulty_bucket(entropy, margin, majority_frac)
        self._difficulty_window.append(diff_bucket)

        # ── Adaptive entropy target ──────────────────────────────
        prop_entropy_mu_used = self._update_proposer_entropy_target(entropy)

        # Proposer reward: entropy reward in steady state; text/certificate bootstrap in warm-start.
        proposer_reward = gaussian_reward(
            entropy,
            prop_entropy_mu_used,
            cfg.prop_entropy_sigma,
        )
        proposer_reward_raw = float(proposer_reward)
        non_objective_question = not self._is_objective_question(question_text)

        cert_final = self._proposer_certificate_score(question_text, selected_meta)
        cert_score_final = float(cert_final.get("score", 0.0))
        cert_valid_final = float(cert_final.get("valid", 0.0))
        cert_bonus_final = 0.0
        if bool(getattr(cfg, "proposer_certificate_enabled", True)):
            cert_weight_cfg = float(
                getattr(
                    cfg,
                    "proposer_warm_start_certificate_weight"
                    if warm_start_active
                    else "proposer_certificate_weight",
                    0.50 if warm_start_active else 0.75,
                )
            )
            cert_weight = max(0.0, cert_weight_cfg)
            cert_min = max(
                0.0,
                min(
                    1.0,
                    min(
                        float(getattr(cfg, "proposer_certificate_min_score", 0.55)),
                        0.50 if warm_start_active else 1.0,
                    ),
                ),
            )
            cert_bonus_final = cert_weight * (cert_score_final - cert_min)
        if warm_start_active:
            qn = normalize_answer(question_text)
            lexical_bonus = 0.0
            if qn:
                if len(question_text.split()) >= 8:
                    lexical_bonus += 0.05
                if any(
                    key in qn
                    for key in (
                        "how many",
                        "partially",
                        "behind",
                        "between",
                        "compared",
                        "left of",
                        "right of",
                        "second",
                        "third",
                        "closest",
                        "farthest",
                    )
                ):
                    lexical_bonus += 0.08
                if (" or " in qn) and not qn.startswith(
                    ("is ", "are ", "was ", "were ", "do ", "does ", "can ", "could ")
                ):
                    lexical_bonus += 0.04
            strategy_used = str(selected_meta.get("strategy_used", "") or "").strip().upper()
            strategy_bonus = 0.0
            if strategy_used.startswith("H"):
                strategy_bonus = 0.10
            elif strategy_used.startswith("M"):
                strategy_bonus = 0.05
            domains = str(selected_meta.get("reasoning_domains", "") or "")
            domain_count = len([d for d in domains.split(",") if d.strip()])
            structure_bonus = 0.02 if domain_count >= 2 else 0.0
            proposer_reward = cert_bonus_final + lexical_bonus + strategy_bonus + structure_bonus
        else:
            # Hard negative penalties once warm-start is over.
            if entropy < 0.01:
                proposer_reward = -cfg.zero_entropy_reward_cap
            elif easy_question:
                proposer_reward = min(proposer_reward, -cfg.zero_entropy_reward_cap * 0.5)
            proposer_reward += cert_bonus_final

        if cfg.proposer_require_objective and non_objective_question:
            proposer_reward -= controller_penalty_boost * float(cfg.proposer_non_objective_penalty)
        if cfg.acceptance_require_non_easy and easy_question:
            easy_pen_scale = (
                float(getattr(cfg, "proposer_warm_start_easy_reject_penalty_scale", 0.0))
                if warm_start_active
                else 1.0
            )
            proposer_reward -= (
                controller_penalty_boost * easy_pen_scale * float(cfg.rejected_question_penalty)
            )
        proposer_reward_pre_clip = float(proposer_reward)
        proposer_reward = max(-1.0, min(1.0, float(proposer_reward)))
        proposer_reward_clipped = bool(abs(proposer_reward_pre_clip - proposer_reward) > 1e-8)

        # Solver reward: penalize easy questions to avoid reinforcing
        # overconfident unanimous answers on trivial questions.
        if easy_question:
            # Negative reward: punish solver for easy unanimous agreement
            solver_reward = -(cfg.easy_solver_penalty_scale * (
                cfg.solver_soft_gamma * majority_frac + (
                    1.0 - cfg.solver_soft_gamma
                ) * (1.0 - min(1.0, entropy / max(cfg.sc_entropy_max, 0.01)))
            ))
        else:
            solver_reward = cfg.solver_soft_gamma * majority_frac + (
                1.0 - cfg.solver_soft_gamma
            ) * (1.0 - min(1.0, entropy / max(cfg.sc_entropy_max, 0.01)))

        stats.update({
            "entropy": entropy,
            "majority_frac": majority_frac,
            "majority_answer": majority_answer[:50],
            "proposer_reward": proposer_reward,
            "proposer_reward_raw": proposer_reward_raw,
            "proposer_reward_pre_clip": proposer_reward_pre_clip,
            "proposer_reward_clipped": bool(proposer_reward_clipped),
            "proposer_non_objective_question": bool(non_objective_question),
            "proposer_certificate_score": float(cert_score_final),
            "proposer_certificate_valid": float(cert_valid_final),
            "proposer_certificate_bonus": float(cert_bonus_final),
            "solver_reward": solver_reward,
            "easy_question": easy_question,
            "margin": margin,
            "prop_entropy_mu_used": prop_entropy_mu_used,
            "difficulty_bucket": diff_bucket,
        })

        # ── 6. Update proposer ──────────────────────────────────────────
        proposer_prompt = build_proposer_prompt(target_difficulty=desired_difficulty_bucket)
        update_rule = str(getattr(cfg, "proposer_update_rule", "reinforce")).strip().lower()
        if update_rule == "grpo" and len(candidate_records) > 1:
            group_rewards: List[float] = []
            for c in candidate_records:
                r = float(c.get("spot_reward", 0.0))
                if int(c.get("candidate_index", -1)) == int(selected_candidate_idx):
                    r = float(proposer_reward)
                group_rewards.append(r)
            if group_rewards:
                mean_r = sum(group_rewards) / float(len(group_rewards))
                std_r = math.sqrt(
                    sum((r - mean_r) ** 2 for r in group_rewards) / float(max(1, len(group_rewards)))
                )
            else:
                mean_r, std_r = 0.0, 0.0

            if std_r > 1e-8:
                group_advantages = [(r - mean_r) / (std_r + 1e-8) for r in group_rewards]
            else:
                n = len(group_rewards)
                if n > 1:
                    order = sorted(range(n), key=lambda i: group_rewards[i])
                    group_advantages = [0.0] * n
                    for rank, idx in enumerate(order):
                        group_advantages[idx] = ((rank / float(n - 1)) - 0.5) * 0.10
                else:
                    group_advantages = [0.0]

            group_stats: List[Dict[str, float]] = []
            for cand, adv in zip(candidate_records, group_advantages):
                comp = str(cand.get("completion", "")).strip()
                if not comp:
                    comp = f"<question>{str(cand.get('question', ''))}</question>"
                st = self.proposer_updater.step(
                    image=image,
                    prompt=proposer_prompt,
                    completion=comp,
                    reward=float(adv),
                    baseline=0.0,
                    device=self.device,
                    ddp_no_sync=self._is_ddp,
                )
                group_stats.append(st)

            applied = [s for s in group_stats if not bool(s.get("skipped_reason"))]
            stats["prop_update_rule"] = "grpo"
            stats["prop_group_size"] = len(group_rewards)
            stats["prop_group_reward_mean"] = float(mean_r)
            stats["prop_group_reward_std"] = float(std_r)
            stats["prop_applied_updates"] = len(applied)
            if group_stats:
                stats["prop_ce_loss_mean"] = float(
                    sum(float(s.get("ce_loss", 0.0)) for s in group_stats if not math.isnan(float(s.get("ce_loss", 0.0))))
                    / max(1, len([s for s in group_stats if not math.isnan(float(s.get("ce_loss", 0.0)))]))
                )
        else:
            prop_stats = self.proposer_updater.step(
                image=image,
                prompt=proposer_prompt,
                completion=proposer_completion,
                reward=proposer_reward,
                baseline=self.proposer_baseline,
                device=self.device,
                ddp_no_sync=self._is_ddp,
            )
            stats.update({f"prop_{k}": v for k, v in prop_stats.items()})
            stats["prop_update_rule"] = "reinforce"

        self.proposer_baseline = (
            cfg.baseline_momentum * self.proposer_baseline
            + (1 - cfg.baseline_momentum) * proposer_reward
        )

        # ── 7. Update solver ────────────────────────────────────────────
        # Skip solver update on easy questions to avoid wasting gradient
        # budget on trivial cases (the solver already knows the answer).
        skip_solver = (
            easy_question
            and cfg.solver_skip_update_on_easy
            and majority_frac >= cfg.easy_update_majority_frac_threshold
        )
        stats["solver_update_skipped"] = skip_solver
        solver_update_applied = False

        if not skip_solver:
            solver_prompt = build_solver_prompt(question_text)
            solver_completion = f"\n<answer>{majority_answer}</answer>"
            sol_stats = self.solver_updater.step(
                image=image,
                prompt=solver_prompt,
                completion=solver_completion,
                reward=solver_reward,
                baseline=self.solver_baseline,
                device=self.device,
                ddp_no_sync=self._is_ddp,
            )
            self.solver_baseline = (
                cfg.baseline_momentum * self.solver_baseline
                + (1 - cfg.baseline_momentum) * solver_reward
            )
            stats.update({f"sol_{k}": v for k, v in sol_stats.items()})
            solver_update_applied = not bool(sol_stats.get("skipped_reason"))
        stats["solver_update_applied"] = bool(solver_update_applied)

        # ── 8. Update controller state & fail-fast diagnostics ─────────
        candidate_non_easy_rate = float(stats.get("proposer_candidate_non_easy_rate", 0.0))
        all_easy_group = 1.0 if candidate_non_easy_rate <= 0.0 else 0.0
        selected_non_easy = 0.0 if easy_question else 1.0
        self._candidate_non_easy_window.append(candidate_non_easy_rate)
        self._all_easy_group_window.append(all_easy_group)
        self._proposer_reward_clipped_window.append(1.0 if proposer_reward_clipped else 0.0)
        self._selected_non_easy_window.append(selected_non_easy)
        self._solver_update_applied_window.append(1.0 if solver_update_applied else 0.0)
        self._entropy_easy_window.append(1.0 if easy_question else 0.0)

        if all_easy_group > 0.5:
            self._all_easy_streak += 1
        else:
            self._all_easy_streak = 0
        if easy_question:
            self._proposer_collapse_streak += 1
        else:
            self._proposer_collapse_streak = 0

        if self._all_easy_streak >= max(1, int(getattr(cfg, "all_easy_explore_trigger", 2))):
            self._forced_explore_steps_left = max(
                int(self._forced_explore_steps_left),
                max(1, int(getattr(cfg, "all_easy_explore_steps", 16))),
            )

        debt_state = self._update_hardness_debt(diff_bucket)
        warm_state = self._update_proposer_warm_start_state(entropy, u_step)
        early_state = self._early_failfast_state(u_step=u_step)

        stats.update({
            "proposer_all_easy_streak": float(self._all_easy_streak),
            "proposer_forced_explore_steps_left": float(max(0, int(self._forced_explore_steps_left))),
            "proposer_collapse_streak": float(self._proposer_collapse_streak),
            "proposer_hardness_debt": float(debt_state.get("debt", 0.0)),
            "proposer_hardness_debt_cap_streak": float(debt_state.get("cap_streak", 0.0)),
            "proposer_hardness_debt_escape_steps_left": float(
                debt_state.get("escape_steps_left", 0.0)
            ),
            "proposer_hardness_debt_escape_triggered": bool(
                debt_state.get("escape_triggered", 0.0) > 0.5
            ),
            "proposer_warm_start_entropy_mean": float(warm_state.get("entropy_mean", 0.0)),
            "proposer_warm_start_exit_streak": float(warm_state.get("exit_streak", 0.0)),
            "proposer_warm_start_exit_pass": bool(warm_state.get("exit_pass", 0.0) > 0.5),
            "proposer_warm_start_completed": bool(warm_state.get("completed", 0.0) > 0.5),
            "proposer_early_failfast_enabled": bool(early_state.get("enabled", 0.0) > 0.5),
            "proposer_early_u_step": int(early_state.get("u_step", float(u_step))),
            "proposer_early_stage1_active": bool(early_state.get("stage1_active", 0.0) > 0.5),
            "proposer_early_stage1_pass": bool(early_state.get("stage1_pass", 1.0) > 0.5),
            "proposer_early_stage2_active": bool(early_state.get("stage2_active", 0.0) > 0.5),
            "proposer_early_stage2_pass": bool(early_state.get("stage2_pass", 1.0) > 0.5),
            "proposer_early_triggered": bool(early_state.get("triggered", 0.0) > 0.5),
        })

        if (
            bool(early_state.get("triggered", 0.0) > 0.5)
            and bool(getattr(cfg, "proposer_early_failfast_stop", False))
            and u_step >= int(getattr(cfg, "proposer_early_hard_stop_min_u_step", 80))
        ):
            msg = (
                "[EarlyFailFast] unhealthy run detected: "
                f"u_step={u_step} "
                f"cand_non_easy_rate={float(early_state.get('candidate_non_easy_rate', 0.0)):.3f} "
                f"all_easy_rate={float(early_state.get('all_easy_group_rate', 0.0)):.3f} "
                f"reward_clipped_rate={float(early_state.get('reward_clipped_rate', 0.0)):.3f} "
                f"selected_non_easy_rate={float(early_state.get('selected_non_easy_rate', 0.0)):.3f} "
                f"solver_updates={float(early_state.get('solver_update_applied_count', 0.0)):.1f} "
                f"collapse_streak={int(self._proposer_collapse_streak)}"
            )
            raise RuntimeError(msg)

        return stats

    # ── Generation Step ─────────────────────────────────────────────────

    def _generation_step(self, step: int) -> Dict:
        """G-step: proposer proposes specs, generator creates images, GRPO update."""
        cfg = self.se_config
        stats = {}

        # ── 1. Proposer generates spec ──────────────────────────────────
        topic = None
        image = None
        if cfg.imageless_proposer_mode:
            topic = _sample_imageless_topic(step, cfg.seed)
            spec, spec_completion = self._generate_imageless_spec(topic, step)
            stats["topic"] = topic[:80]
        else:
            # Sample a source image for spec generation
            image, source = self._sample_image(step)
            if image is None:
                return {"g_skipped": True, "reason": "no_source_image"}
            spec, spec_completion = self._generate_spec(image, step)
            stats["image_source"] = source

        if spec is None or not spec.prompt:
            return {"g_skipped": True, "reason": "no_spec"}

        stats["gen_prompt"] = spec.prompt[:100]
        stats["num_qa_pairs"] = len(spec.qa_pairs)

        # ── 2. Generate K candidate images ──────────────────────────────
        K = cfg.num_generations
        candidates = []  # List of (PIL Image, pixel_gen_tensor)
        with use_role(self.model, ROLE_GENERATOR):
            for k in range(K):
                try:
                    gen_image, gen_tensor = self._generate_image(spec.prompt)
                    if gen_image is not None:
                        candidates.append((gen_image, gen_tensor))
                except Exception as e:
                    logger.warning(f"[SelfEvolvingTrainer] Generation {k} failed: {e}")

        if not candidates:
            return {"g_skipped": True, "reason": "no_candidates"}
        stats["num_candidates"] = len(candidates)

        # Save generated images for early-step sanity checks.
        # Only rank-0 writes files to avoid DDP duplication.
        if step < 50 and self.is_world_process_zero():
            preview_dir = pathlib.Path(self.args.output_dir) / "checkpoints" / "generated_first50"
            preview_dir.mkdir(parents=True, exist_ok=True)
            for cand_idx, (gen_image, _) in enumerate(candidates):
                out_path = preview_dir / f"step_{step:05d}_cand_{cand_idx:02d}.png"
                try:
                    gen_image.save(out_path)
                except Exception as e:
                    logger.warning(f"[SelfEvolvingTrainer] Failed to save preview image {out_path}: {e}")

        # ── 3. Score candidates ─────────────────────────────────────────
        questions = [qa.question for qa in spec.qa_pairs]
        expected_answers = [qa.expected for qa in spec.qa_pairs]

        rewards = []
        reward_details_list = []
        for gen_image, _ in candidates:
            reward, details = score_generated_image(
                model=self.model,
                processor=self.processor,
                image=gen_image,
                prompt=spec.prompt,
                questions=questions,
                expected_answers=expected_answers,
                device=self.device,
                config=cfg,
            )
            rewards.append(reward)
            reward_details_list.append(details)

        stats["gen_rewards"] = rewards
        stats["gen_reward_mean"] = sum(rewards) / len(rewards)
        stats["gen_reward_max"] = max(rewards)

        # ── 4. GRPO update on generator ─────────────────────────────────
        if len(candidates) >= 2:
            # Prepare inputs for GRPO
            gen_prompt = build_generator_prompt(spec.prompt)
            chat_text = _build_text_only_chat(self.processor, gen_prompt)
            text_inputs = _prepare_text_only_inputs(
                self.processor, self.device, chat_text,
            )
            input_ids = text_inputs["input_ids"]
            attention_mask = text_inputs["attention_mask"]
            labels = input_ids.clone()

            pixel_gen_values_list = [[c[1]] for c in candidates]

            gen_stats = self.generator_updater.step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_gen_values_list=pixel_gen_values_list,
                rewards=rewards,
                device=self.device,
                ddp_no_sync=self._is_ddp,
            )
            stats.update({f"gen_{k}": v for k, v in gen_stats.items()})

        # ── 5. Update proposer with generation quality ──────────────────
        if cfg.proposer_gen_reward_enabled:
            gen_quality = max(rewards)
            prop_gen_reward = (
                cfg.proposer_gen_entropy_weight * 0.5  # placeholder: no entropy here
                + (1 - cfg.proposer_gen_entropy_weight) * gen_quality
            )

            if cfg.imageless_proposer_mode:
                prop_prompt = build_imageless_spec_prompt(
                    topic or "",
                    target_difficulty=self._current_difficulty(step),
                )
                prop_stats = self.proposer_updater.step(
                    image=None,
                    prompt=prop_prompt,
                    completion=spec_completion,
                    reward=prop_gen_reward,
                    baseline=self.proposer_gen_baseline,
                    device=self.device,
                    ddp_no_sync=self._is_ddp,
                )
            else:
                prop_prompt = build_generation_spec_prompt(
                    target_difficulty=self._current_difficulty(step),
                )
                prop_stats = self.proposer_updater.step(
                    image=image,
                    prompt=prop_prompt,
                    completion=spec_completion,
                    reward=prop_gen_reward,
                    baseline=self.proposer_gen_baseline,
                    device=self.device,
                    ddp_no_sync=self._is_ddp,
                )

            self.proposer_gen_baseline = (
                cfg.proposer_gen_baseline_momentum * self.proposer_gen_baseline
                + (1 - cfg.proposer_gen_baseline_momentum) * prop_gen_reward
            )
            stats.update({f"prop_gen_{k}": v for k, v in prop_stats.items()})

        # ── 6. Best image → replay buffer ───────────────────────────────
        best_idx = rewards.index(max(rewards))
        best_image = candidates[best_idx][0]

        added = self.replay_buffer.add(
            image=best_image,
            prompt=spec.prompt,
            questions=questions,
            reference_answers=expected_answers,
            reward=max(rewards),
            step=step,
        )
        stats["replay_buffer_added"] = added
        stats["replay_buffer_size"] = len(self.replay_buffer)

        # Update reward EMA
        self.reward_ema = (
            cfg.reward_ema_momentum * self.reward_ema
            + (1 - cfg.reward_ema_momentum) * max(rewards)
        )
        stats["reward_ema"] = self.reward_ema

        return stats

    # ── Helper Methods ──────────────────────────────────────────────────

    @staticmethod
    def _scan_image_folder(folder_path: str) -> List[str]:
        """Recursively scan a folder for image files. Returns sorted list of paths."""
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
        folder = pathlib.Path(folder_path)
        if not folder.is_dir():
            logger.warning(f"[SelfEvolvingTrainer] image_folder not found: {folder_path}")
            return []
        paths = []
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(p))
        paths.sort()
        return paths

    def _sample_image(self, step: int) -> Tuple[Optional[Image.Image], str]:
        """Sample an image: image folder, replay buffer, or dataset."""
        cfg = self.se_config

        # Gen-mix ratio: linearly ramp from start to max
        warmup = max(1, cfg.gen_mix_ratio_warmup_steps)
        ratio = cfg.gen_mix_ratio_start + (
            cfg.gen_mix_ratio_max - cfg.gen_mix_ratio_start
        ) * min(1.0, step / warmup)

        # Try replay buffer first (for generated image mixing)
        if random.random() < ratio and self.replay_buffer:
            entry = self.replay_buffer.sample()
            if entry is not None:
                return entry.image, "replay_buffer"

        # ── Image folder mode (preferred when set) ────────────────────
        if self._image_folder_paths:
            path = random.choice(self._image_folder_paths)
            try:
                pil_img = Image.open(path).convert("RGB")
                return pil_img, "image_folder"
            except Exception as e:
                logger.warning(
                    f"[SelfEvolvingTrainer] Failed to load {path}: {e}"
                )
                return None, "none"

        # ── Fallback: LLaMA-Factory dataset ───────────────────────────
        try:
            ds_len = len(self.train_dataset) if self.train_dataset is not None else 0
            if ds_len > 0:
                idx = random.randint(0, ds_len - 1)
                sample = self.train_dataset[idx]
                image_obj = None
                for key in ("images", "image", "pixel_values"):
                    if key in sample and sample[key] is not None:
                        image_obj = sample[key]
                        if isinstance(image_obj, (list, tuple)) and image_obj:
                            image_obj = image_obj[0]
                        break
                if image_obj is not None:
                    if isinstance(image_obj, Image.Image):
                        return image_obj, "dataset"
                    if isinstance(image_obj, str):
                        try:
                            pil_img = Image.open(image_obj).convert("RGB")
                            return pil_img, "dataset"
                        except Exception as e_open:
                            logger.warning(
                                f"[SelfEvolvingTrainer] Failed to open "
                                f"'{image_obj}': {e_open}"
                            )
                            return None, "none"
                    if isinstance(image_obj, bytes):
                        from io import BytesIO
                        pil_img = Image.open(BytesIO(image_obj)).convert("RGB")
                        return pil_img, "dataset"
                    return _ensure_pil_image(image_obj), "dataset"
        except Exception as e:
            logger.warning(f"[SelfEvolvingTrainer] Dataset sampling failed: {e}")

        return None, "none"

    def _generate_proposer_question(
        self, image: Image.Image, step: int, target_difficulty: str = ""
    ) -> Tuple[str, str]:
        """Generate a question from the proposer."""
        candidates, completion = self._generate_proposer_candidates(
            image=image, step=step, target_difficulty=target_difficulty
        )
        if not candidates:
            return "", completion
        first = candidates[0]
        return str(first.get("question", "")), str(first.get("completion", completion))

    def _generate_proposer_candidates(
        self,
        image: Image.Image,
        step: int,
        target_difficulty: str = "",
        num_candidates: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[List[Dict[str, object]], str]:
        """Generate and parse proposer candidates from a single proposer call.

        Returns a list of candidate dicts:
          - question: parsed question text
          - completion: candidate-local completion text (raw block when available)
          - meta: parsed auxiliary tags (best effort)
        """
        cfg = self.se_config
        difficulty = str(target_difficulty or self._current_difficulty(step))
        n_candidates = int(num_candidates) if num_candidates is not None else int(cfg.proposer_num_candidates)
        dec_temp = float(temperature) if temperature is not None else float(cfg.temp)
        dec_top_p = float(top_p) if top_p is not None else float(cfg.top_p)
        n_candidates = max(1, n_candidates)
        dec_temp = max(0.05, dec_temp)
        dec_top_p = max(0.05, min(1.0, dec_top_p))

        if n_candidates > 1:
            prompt = build_proposer_multi_prompt(
                target_difficulty=difficulty,
                num_questions=n_candidates,
            )
        else:
            prompt = build_proposer_prompt(target_difficulty=difficulty)

        chat_text = _build_chat_text(self.processor, image, prompt)
        mm_inputs = _prepare_mm_inputs(
            self.processor, self.device, image, chat_text, model=self.model
        )

        base_model = _unwrap_model(self.model)
        with torch.no_grad():
            gen_ids = base_model.generate(
                **mm_inputs,
                max_new_tokens=cfg.max_new_tokens_proposer,
                do_sample=True,
                temperature=dec_temp,
                top_p=dec_top_p,
            )

        input_len = mm_inputs["input_ids"].shape[1]
        new_ids = gen_ids[0, input_len:]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        completion = _decode_tokens(tokenizer, new_ids)

        return self._parse_proposer_candidates(completion), completion

    def _parse_proposer_candidates(self, completion: str) -> List[Dict[str, object]]:
        """Parse candidate question blocks with best-effort metadata."""
        text = str(completion or "")
        candidates: List[Dict[str, object]] = []

        blocks = list(re.finditer(r"<question[^>]*>.*?</question>", text, flags=re.IGNORECASE | re.DOTALL))
        for idx, match in enumerate(blocks):
            block = match.group(0)
            inner = re.sub(r"^<question[^>]*>|</question>$", "", block, flags=re.IGNORECASE | re.DOTALL).strip()

            def _tag_value(tag: str) -> str:
                m = re.search(rf"<{tag}>(.*?)</{tag}>", inner, flags=re.IGNORECASE | re.DOTALL)
                return (m.group(1).strip() if m else "")

            q_text = _tag_value("text")
            if not q_text:
                q_text = _parse_first_question(inner)
            q_text = str(q_text).strip()
            if not q_text:
                continue
            candidates.append(
                {
                    "candidate_index": int(idx),
                    "question": q_text,
                    "completion": block.strip(),
                    "meta": {
                        "task_card": _tag_value("task_card"),
                        "reasoning_domains": _tag_value("reasoning_domains"),
                        "reasoning_chain": _tag_value("reasoning_chain"),
                        "strategy_used": _tag_value("strategy_used"),
                        "visual_target": _tag_value("visual_target"),
                        "two_answer_test": _tag_value("two_answer_test"),
                        "rationale": _tag_value("rationale"),
                    },
                }
            )

        if not candidates:
            qs = _parse_all_questions(text)
            for idx, q in enumerate(qs):
                q_text = str(q).strip()
                if not q_text:
                    continue
                candidates.append(
                    {
                        "candidate_index": int(idx),
                        "question": q_text,
                        "completion": f"<question>{q_text}</question>",
                        "meta": {},
                    }
                )

        # De-duplicate by normalized question text while preserving order.
        seen = set()
        deduped: List[Dict[str, object]] = []
        for cand in candidates:
            key = normalize_answer(str(cand.get("question", "")))
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(cand)
        return deduped

    def _generate_solver_answers(
        self,
        image: Image.Image,
        question: str,
        num_samples: int = 5,
    ) -> Tuple[List[str], List[str]]:
        """Generate solver answers for a given question.

        Uses a temperature schedule across samples to encourage answer
        diversity. Lower temperatures produce more confident answers while
        higher temperatures explore alternatives — this is critical for
        producing non-zero entropy in the solver answer distribution.
        """
        cfg = self.se_config
        prompt = build_solver_prompt(question)
        chat_text = _build_chat_text(self.processor, image, prompt)
        mm_inputs = _prepare_mm_inputs(
            self.processor, self.device, image, chat_text, model=self.model
        )

        base_model = _unwrap_model(self.model)
        answers = []
        completions = []

        # Build temperature and top_p schedules across samples
        if cfg.solver_use_temperature_mix and num_samples > 1:
            temp_schedule = [
                cfg.solver_temp_min + (cfg.solver_temp_max - cfg.solver_temp_min)
                * i / (num_samples - 1)
                for i in range(num_samples)
            ]
            top_p_schedule = [
                cfg.solver_top_p_min + (cfg.solver_top_p_max - cfg.solver_top_p_min)
                * i / (num_samples - 1)
                for i in range(num_samples)
            ]
        else:
            temp_schedule = [cfg.temp] * num_samples
            top_p_schedule = [cfg.top_p] * num_samples

        with torch.no_grad():
            for i in range(num_samples):
                try:
                    gen_ids = base_model.generate(
                        **mm_inputs,
                        max_new_tokens=cfg.max_new_tokens_solver,
                        do_sample=True,
                        temperature=temp_schedule[i],
                        top_p=top_p_schedule[i],
                    )
                    input_len = mm_inputs["input_ids"].shape[1]
                    new_ids = gen_ids[0, input_len:]
                    tokenizer = getattr(self.processor, "tokenizer", self.processor)
                    comp = _decode_tokens(tokenizer, new_ids)
                    completions.append(comp)
                    answers.append(_parse_answer(comp))
                except Exception:
                    pass

        return answers, completions

    def _generate_imageless_spec(
        self, topic: str, step: int
    ) -> Tuple[Optional[GenerationSpec], str]:
        """Generate a spec from topic text (no image)."""
        cfg = self.se_config
        difficulty = self._current_difficulty(step)
        prompt = build_imageless_spec_prompt(topic, target_difficulty=difficulty)

        chat_text = _build_text_only_chat(self.processor, prompt)
        text_inputs = _prepare_text_only_inputs(
            self.processor, self.device, chat_text,
        )

        base_model = _unwrap_model(self.model)
        with use_role(self.model, ROLE_PROPOSER):
            with torch.no_grad():
                gen_ids = base_model.generate(
                    **text_inputs,
                    max_new_tokens=cfg.max_new_tokens_proposer,
                    do_sample=True,
                    temperature=cfg.temp,
                    top_p=cfg.top_p,
                )

        input_len = text_inputs["input_ids"].shape[1]
        new_ids = gen_ids[0, input_len:]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        completion = _decode_tokens(tokenizer, new_ids)

        spec = _parse_generation_spec(completion)
        if spec.fallback_used:
            return None, completion

        return spec, completion

    def _generate_spec(
        self, image: Image.Image, step: int
    ) -> Tuple[Optional[GenerationSpec], str]:
        """Generate a spec from a source image."""
        cfg = self.se_config
        difficulty = self._current_difficulty(step)
        prompt = build_generation_spec_prompt(target_difficulty=difficulty)

        chat_text = _build_chat_text(self.processor, image, prompt)
        mm_inputs = _prepare_mm_inputs(
            self.processor, self.device, image, chat_text, model=self.model
        )

        base_model = _unwrap_model(self.model)
        with use_role(self.model, ROLE_PROPOSER):
            with torch.no_grad():
                gen_ids = base_model.generate(
                    **mm_inputs,
                    max_new_tokens=cfg.max_new_tokens_proposer,
                    do_sample=True,
                    temperature=cfg.temp,
                    top_p=cfg.top_p,
                )

        input_len = mm_inputs["input_ids"].shape[1]
        new_ids = gen_ids[0, input_len:]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        completion = _decode_tokens(tokenizer, new_ids)

        spec = _parse_generation_spec(completion)
        if spec.fallback_used:
            return None, completion

        return spec, completion

    @staticmethod
    def _get_actual_model(model):
        """Navigate through DDP + PEFT wrappers to get the actual model.

        PeftModel wraps: PeftModel → .base_model (LoraModel) → .model (actual)
        DDP wraps:       DDP → .module (PeftModel or actual)

        The actual model is VargptQwen2VLForConditionalGeneration, which holds
        ``past_hidden_states`` and the ``forward(inference_image_gen=...)`` logic.
        """
        m = model.module if hasattr(model, "module") else model  # unwrap DDP
        # Unwrap PEFT: PeftModel.base_model is the tuner, tuner.model is actual
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            return m.base_model.model
        return m

    def _generate_image(
        self, prompt: str
    ) -> Tuple[Optional[Image.Image], Optional[torch.Tensor]]:
        """Generate a single image from a text prompt using VARGPT.

        Uses the model's autoregressive_infer_cfg() method for image generation.

        Key insights for VARGPT image generation:
          1. The model must be in **eval mode** because forward() only stores
             ``past_hidden_states`` when ``not self.model.training`` (line 2417).
          2. ``past_hidden_states`` lives on the **actual** model instance
             (VargptQwen2VLForConditionalGeneration), not the PEFT wrapper.
          3. Two-step forward: first populate ``past_hidden_states``, then call
             with ``inference_image_gen=True`` passing ``past_key_values`` so that
             the reset guard at line 2246 is skipped.

        Returns
        -------
        image : PIL Image or None
        tensor : torch.Tensor (the raw pixel tensor for GRPO training)
        """
        cfg = self.se_config
        peft_model = _unwrap_model(self.model)       # PeftModel (DDP-unwrapped)
        actual_model = self._get_actual_model(self.model)  # VargptQwen2VLForConditionalGeneration
        _gen_modules = []
        _gen_module_orig_dtypes = {}
        _past_hidden_states_orig_dtype = None

        def _module_fp_dtype(module):
            for p in module.parameters():
                if p.is_floating_point():
                    return p.dtype
            for b in module.buffers():
                if b.is_floating_point():
                    return b.dtype
            return None

        try:
            # Build generation prompt with special tokens
            gen_prompt = build_generator_prompt(prompt)

            # Tokenize
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            inputs = tokenizer(gen_prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Temporarily switch to eval mode so that forward() stores
            # past_hidden_states (guarded by ``if not self.model.training``).
            peft_model.eval()

            with torch.no_grad():
                # Step 1: Run a normal forward pass to populate past_hidden_states
                actual_model.past_hidden_states = None
                outputs = peft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                # past_hidden_states is now set on actual_model (lines 2418-2421)

                if actual_model.past_hidden_states is None:
                    logger.warning(
                        "[SelfEvolvingTrainer] past_hidden_states still None "
                        "after step-1 forward; image generation cannot proceed."
                    )
                    peft_model.train()
                    return None, None

                # Step 2: Call forward with inference_image_gen=True
                #
                # CRITICAL: Switch back to TRAINING mode before step 2.
                # The model's forward() line 2246 resets past_hidden_states when:
                #   (past_key_values is None or len(past_key_values)==0) AND not self.model.training
                # Even though we pass past_kv, len() can be 0 due to PEFT/cache
                # interactions. By switching to train mode, the "not training"
                # guard is False → past_hidden_states is preserved.
                # The inference_image_gen branch (line 2345) runs regardless of
                # training mode, so this is safe.
                peft_model.train()

                # Cast VAR generation modules to float32.
                # The VAR model (vargpt_gen, image_gen_projector, etc.) was
                # designed for float32, but the whole model is loaded in bf16.
                # autoregressive_infer_cfg does .float() on inputs before
                # passing to these modules, causing dtype mismatches.
                for name in ("vargpt_gen", "image_gen_projector",
                             "image_gen_projector_out", "vae_local"):
                    mod = getattr(actual_model, name, None)
                    if mod is not None:
                        _gen_modules.append((name, mod))
                        _gen_module_orig_dtypes[name] = _module_fp_dtype(mod)
                        mod.float()

                # Also cast past_hidden_states to float32 — it was stored
                # during step 1 in bf16, but now flows into the float32
                # VAR modules via get_ca_kv_cross → image_gen_projector_out.
                if actual_model.past_hidden_states is not None:
                    _past_hidden_states_orig_dtype = actual_model.past_hidden_states.dtype
                    actual_model.past_hidden_states = actual_model.past_hidden_states.float()

                gen_result = peft_model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    inference_image_gen=True,
                )

                if gen_result is not None:
                    if isinstance(gen_result, torch.Tensor):
                        img_tensor = gen_result
                    elif isinstance(gen_result, (list, tuple)):
                        img_tensor = gen_result[0] if gen_result else None
                    else:
                        img_tensor = None
                        # Handle ModelOutput-style returns from VARGPT forward.
                        for key in ("generated_image", "image", "images", "img", "output_image"):
                            if hasattr(gen_result, key):
                                value = getattr(gen_result, key)
                                if isinstance(value, (list, tuple)):
                                    img_tensor = value[0] if value else None
                                else:
                                    img_tensor = value
                                if img_tensor is not None:
                                    break
                        # Dict-like fallback.
                        if img_tensor is None and isinstance(gen_result, dict):
                            for key in ("generated_image", "image", "images", "img", "output_image"):
                                if key in gen_result:
                                    value = gen_result[key]
                                    if isinstance(value, (list, tuple)):
                                        img_tensor = value[0] if value else None
                                    else:
                                        img_tensor = value
                                    break

                    if img_tensor is not None:
                        pil_image = _ensure_pil_image(img_tensor)
                        peft_model.train()
                        return pil_image, img_tensor

        except Exception as e:
            logger.warning(f"[SelfEvolvingTrainer] Image generation failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore original dtypes even if generation path throws.
            for name, mod in _gen_modules:
                target_dtype = _gen_module_orig_dtypes.get(name, None)
                if target_dtype is not None:
                    mod.to(dtype=target_dtype)
            if actual_model.past_hidden_states is not None and _past_hidden_states_orig_dtype is not None:
                actual_model.past_hidden_states = actual_model.past_hidden_states.to(dtype=_past_hidden_states_orig_dtype)
            # Always restore training mode
            peft_model.train()

        return None, None

    def _is_objective_question(self, question: str) -> bool:
        """Best-effort objective question validator."""
        q = str(question or "").strip()
        if not q:
            return False
        if "?" not in q:
            return False
        qn = normalize_answer(q)
        if not qn:
            return False
        if re.search(r"\b(why|might|could|likely|opinion|feel|believe|think)\b", qn):
            return False
        if re.search(r"\b(something|anything|stuff|thing)\b", qn):
            return False
        if "<" in q or ">" in q:
            return False
        return True

    def _proposer_certificate_score(self, question: str, meta: Dict[str, str]) -> Dict[str, float]:
        """Compute lightweight structural validity score for proposer output."""
        if not bool(getattr(self.se_config, "proposer_certificate_enabled", True)):
            return {"score": 0.0, "valid": 1.0}

        objective = 1.0 if self._is_objective_question(question) else 0.0
        visual_target = str(meta.get("visual_target", "") or "").strip()
        strategy_used = str(meta.get("strategy_used", "") or "").strip()
        reasoning_chain = str(meta.get("reasoning_chain", "") or "").strip()
        reasoning_domains = str(meta.get("reasoning_domains", "") or "").strip()
        rationale = str(meta.get("rationale", "") or "").strip()
        two_answer_test = str(meta.get("two_answer_test", "") or "").strip()

        strategy_ok = 1.0 if strategy_used else 0.0
        target_ok = 1.0 if visual_target else 0.0
        chain_ok = 1.0 if reasoning_chain and ("->" in reasoning_chain or len(reasoning_chain.split()) >= 4) else 0.0
        rationale_ok = 1.0 if len(rationale.split()) >= 6 else 0.0

        domains = [d.strip().lower() for d in reasoning_domains.split(",") if d.strip()]
        domains_ok = 1.0 if len(domains) >= 2 else 0.0

        has_split = (" vs " in two_answer_test.lower()) or ("/" in two_answer_test)
        two_ok = 1.0 if has_split and len(two_answer_test.split()) >= 3 else 0.0

        # Use fields present in both old and new prompt templates.
        structural_mid = max(target_ok, strategy_ok)
        context_mid = max(chain_ok, rationale_ok)
        score = float(objective + structural_mid + context_mid + domains_ok + two_ok) / 5.0
        min_score = max(
            0.0,
            min(1.0, float(getattr(self.se_config, "proposer_certificate_min_score", 0.55))),
        )
        strict_struct = bool(getattr(self.se_config, "proposer_certificate_strict_struct", True))
        valid = 1.0 if score >= min_score else 0.0
        if strict_struct and (objective < 0.5 or two_ok < 0.5 or structural_mid < 0.5):
            valid = 0.0
        return {"score": float(score), "valid": float(valid)}

    def _update_proposer_entropy_target(self, entropy_nats: float) -> float:
        """Adaptively shift the Gaussian reward center based on observed entropy.

        Ported from BLIP3o understanding_trainer.py:220-234.

        When ``adaptive_prop_entropy_target`` is False, returns the fixed
        ``prop_entropy_mu`` from config. Otherwise, maintains an EMA of
        observed entropy and shifts the Gaussian center toward it, clamped
        to [prop_entropy_mu_min, prop_entropy_mu_max].
        """
        cfg = self.se_config
        if not cfg.adaptive_prop_entropy_target:
            return float(cfg.prop_entropy_mu)

        momentum = max(0.0, min(0.9999, cfg.prop_entropy_ema_momentum))
        prev = self.proposer_entropy_mu_ema
        ema = momentum * prev + (1.0 - momentum) * float(entropy_nats)

        mu_min = float(cfg.prop_entropy_mu_min)
        mu_max = float(cfg.prop_entropy_mu_max)
        if mu_min > mu_max:
            mu_min, mu_max = mu_max, mu_min
        ema = max(mu_min, min(mu_max, ema))

        self.proposer_entropy_mu_ema = float(ema)
        return float(ema)

    def _difficulty_bucket(
        self, entropy_nats: float, margin: float, majority_frac: float,
    ) -> str:
        """Classify observed difficulty into easy/medium/hard.

        Ported from BLIP3o understanding_trainer.py:306-320.
        """
        cfg = self.se_config
        easy_majority = cfg.easy_update_majority_frac_threshold
        hard_min_entropy = cfg.difficulty_hard_min_entropy
        hard_max_margin = cfg.difficulty_hard_max_margin

        if entropy_nats <= cfg.sc_entropy_min or majority_frac >= easy_majority:
            return "easy"
        if entropy_nats >= hard_min_entropy and margin <= hard_max_margin:
            return "hard"
        return "medium"

    @staticmethod
    def _second_highest_frac(norm_answers: List[str]) -> float:
        """Return the fraction of the second-most-common answer (0 if only one unique)."""
        counts: Dict[str, int] = {}
        for a in norm_answers:
            counts[a] = counts.get(a, 0) + 1
        if len(counts) < 2:
            return 0.0
        sorted_counts = sorted(counts.values(), reverse=True)
        return sorted_counts[1] / len(norm_answers)

    @staticmethod
    def _sample_bucket(weights: Dict[str, float]) -> str:
        keys = ["easy", "medium", "hard"]
        probs = [max(0.0, float(weights.get(k, 0.0))) for k in keys]
        total = sum(probs)
        if total <= 0.0:
            return "medium"
        probs = [p / total for p in probs]
        return random.choices(keys, weights=probs, k=1)[0]

    @staticmethod
    def _normalize_bucket_weights(weights: Dict[str, float]) -> Dict[str, float]:
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
                "easy": float(getattr(self.se_config, "difficulty_target_easy", 0.0)),
                "medium": float(getattr(self.se_config, "difficulty_target_medium", 0.7)),
                "hard": float(getattr(self.se_config, "difficulty_target_hard", 0.3)),
            }
        )

    def _is_proposer_warm_start_active(self, u_step: int) -> bool:
        cfg = self.se_config
        if not bool(getattr(cfg, "proposer_warm_start_enabled", True)):
            return False
        if bool(getattr(self, "_warm_start_completed", False)):
            return False
        max_steps = max(1, int(getattr(cfg, "proposer_warm_start_max_steps", 30)))
        return int(u_step) <= max_steps

    def _update_proposer_warm_start_state(self, entropy_nats: float, u_step: int) -> Dict[str, float]:
        cfg = self.se_config
        if not bool(getattr(cfg, "proposer_warm_start_enabled", True)):
            return {
                "enabled": 0.0,
                "active_next": 0.0,
                "completed": 1.0,
                "entropy_mean": 0.0,
                "exit_streak": 0.0,
                "exit_pass": 0.0,
            }
        exit_window = max(1, int(getattr(cfg, "proposer_warm_start_exit_window", 5)))
        if int(getattr(self._warm_start_entropy_window, "maxlen", 0) or 0) != exit_window:
            self._warm_start_entropy_window = collections.deque(
                list(self._warm_start_entropy_window)[-exit_window:],
                maxlen=exit_window,
            )
        self._warm_start_entropy_window.append(float(entropy_nats))
        entropy_mean = float(sum(float(x) for x in self._warm_start_entropy_window)) / float(
            max(1, len(self._warm_start_entropy_window))
        )
        exit_thr = max(
            0.0,
            float(getattr(cfg, "proposer_warm_start_entropy_exit_threshold", 0.10)),
        )
        exit_pass = bool(
            len(self._warm_start_entropy_window) >= exit_window and entropy_mean >= exit_thr
        )
        if exit_pass:
            self._warm_start_exit_streak += 1
        else:
            self._warm_start_exit_streak = 0
        max_steps = max(1, int(getattr(cfg, "proposer_warm_start_max_steps", 30)))
        exit_consecutive = max(
            1, int(getattr(cfg, "proposer_warm_start_exit_consecutive", 2))
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
        cfg = self.se_config
        if not bool(getattr(cfg, "hardness_debt_enabled", True)):
            return {
                "enabled": 0.0,
                "debt": 0.0,
                "cap_streak": 0.0,
                "escape_steps_left": 0.0,
                "escape_triggered": 0.0,
            }

        debt = float(self._hardness_debt)
        debt_max = max(1e-6, float(getattr(cfg, "hardness_debt_max", 6.0)))
        inc_easy = max(0.0, float(getattr(cfg, "hardness_debt_inc_easy", 1.5)))
        dec_non_easy = max(0.0, float(getattr(cfg, "hardness_debt_dec_non_easy", 1.0)))
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
        stale_steps = max(1, int(getattr(cfg, "hardness_debt_stale_steps", 8)))
        if cap_streak >= stale_steps:
            reset_to = float(getattr(cfg, "hardness_debt_stale_reset_to", 3.0))
            debt = max(0.0, min(debt_max, reset_to))
            escape_steps = max(
                1, int(getattr(cfg, "hardness_debt_stale_escape_steps", stale_steps))
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

    def _choose_difficulty_target(self) -> Dict[str, object]:
        cfg = self.se_config
        enabled = bool(getattr(cfg, "difficulty_sampler_enabled", True))
        min_samples = max(4, int(getattr(cfg, "difficulty_sampler_min_samples", 8)))
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
        if bool(getattr(cfg, "hardness_debt_enabled", True)):
            weights_for_sampling = self._normalize_bucket_weights(weights_for_sampling)
            if int(self._hardness_debt_escape_steps_left) > 0:
                debt_escape_active = True
                weights_for_sampling = self._normalize_bucket_weights(
                    {
                        "easy": float(getattr(cfg, "hardness_debt_stale_easy_weight", 0.05)),
                        "medium": float(getattr(cfg, "hardness_debt_stale_medium_weight", 0.55)),
                        "hard": float(getattr(cfg, "hardness_debt_stale_hard_weight", 0.40)),
                    }
                )
                self._hardness_debt_escape_steps_left = max(
                    0,
                    int(self._hardness_debt_escape_steps_left) - 1,
                )
                mode = f"{mode}+debt_escape"
            else:
                debt_max = max(1e-6, float(getattr(cfg, "hardness_debt_max", 6.0)))
                debt_thr = max(
                    0.0,
                    min(
                        debt_max,
                        float(getattr(cfg, "hardness_debt_hard_recovery_threshold", 3.0)),
                    ),
                )
                if debt > debt_thr:
                    debt_ratio = min(1.0, (debt - debt_thr) / max(1e-6, debt_max - debt_thr))
                    recovery_weights = self._normalize_bucket_weights(
                        {
                            "easy": float(getattr(cfg, "hardness_debt_recovery_easy_weight", 0.0)),
                            "medium": float(getattr(cfg, "hardness_debt_recovery_medium_weight", 0.30)),
                            "hard": float(getattr(cfg, "hardness_debt_recovery_hard_weight", 0.70)),
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
                    "medium": float(getattr(cfg, "hardness_debt_recovery_medium_weight", 0.30)),
                    "hard": float(getattr(cfg, "hardness_debt_recovery_hard_weight", 0.70)),
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

    @staticmethod
    def _mean_recent(values: collections.deque) -> float:
        if not values:
            return 0.0
        vals = [float(v) for v in values]
        return float(sum(vals) / float(max(1, len(vals))))

    def _early_failfast_state(self, *, u_step: int) -> Dict[str, float]:
        cfg = self.se_config
        state: Dict[str, float] = {
            "enabled": 1.0 if bool(getattr(cfg, "proposer_early_failfast_enabled", True)) else 0.0,
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
                max(0, int(getattr(cfg, "proposer_early_max_collapse_streak", 3)))
            ),
            "recovery_armed": 0.0,
            "triggered": 0.0,
            "hard_stop_min_u_step": float(
                max(1, int(getattr(cfg, "proposer_early_hard_stop_min_u_step", 80)))
            ),
        }
        if state["enabled"] <= 0.5 or int(u_step) <= 0:
            return state

        state["candidate_non_easy_rate"] = self._mean_recent(self._candidate_non_easy_window)
        state["all_easy_group_rate"] = self._mean_recent(self._all_easy_group_window)
        state["reward_clipped_rate"] = self._mean_recent(self._proposer_reward_clipped_window)
        state["selected_non_easy_rate"] = self._mean_recent(self._selected_non_easy_window)
        state["solver_update_applied_count"] = float(sum(float(v) for v in self._solver_update_applied_window))

        step1 = max(1, int(getattr(cfg, "proposer_early_stage1_u_step", 12)))
        step2 = max(step1, int(getattr(cfg, "proposer_early_stage2_u_step", 24)))
        if int(u_step) >= step1:
            state["stage1_active"] = 1.0
            stage1_pass = (
                state["candidate_non_easy_rate"]
                >= float(getattr(cfg, "proposer_early_candidate_non_easy_rate_min", 0.08))
                and state["all_easy_group_rate"]
                <= float(getattr(cfg, "proposer_early_all_easy_rate_max", 0.93))
                and state["reward_clipped_rate"]
                <= float(getattr(cfg, "proposer_early_reward_clipped_rate_max", 0.85))
            )
            state["stage1_pass"] = 1.0 if stage1_pass else 0.0
            if not stage1_pass:
                state["triggered"] = 1.0
        if int(u_step) >= step2:
            state["stage2_active"] = 1.0
            stage2_pass = (
                state["selected_non_easy_rate"]
                >= float(getattr(cfg, "proposer_early_selected_non_easy_rate_min", 0.10))
                and state["solver_update_applied_count"]
                >= float(getattr(cfg, "proposer_early_solver_updates_min", 1))
            )
            state["stage2_pass"] = 1.0 if stage2_pass else 0.0
            if not stage2_pass:
                state["triggered"] = 1.0

        max_collapse = max(0, int(getattr(cfg, "proposer_early_max_collapse_streak", 3)))
        if state["stage1_active"] > 0.5 and int(state["collapse_streak"]) > max_collapse:
            state["triggered"] = 1.0

        if state["triggered"] > 0.5 and bool(getattr(cfg, "proposer_early_failfast_recover", True)):
            recover_steps = max(
                1,
                int(getattr(cfg, "proposer_early_failfast_recover_steps", 20)),
            )
            self._forced_explore_steps_left = max(int(self._forced_explore_steps_left), recover_steps)
            state["recovery_armed"] = 1.0
        return state

    def _current_difficulty(self, step: int) -> str:
        """Choose target difficulty using deficit-based sampling."""
        cfg = self.se_config
        target = {
            "easy": max(0.0, float(cfg.difficulty_target_easy)),
            "medium": max(0.0, float(cfg.difficulty_target_medium)),
            "hard": max(0.0, float(cfg.difficulty_target_hard)),
        }
        t_sum = sum(target.values())
        if t_sum <= 0.0:
            target = {"easy": 0.0, "medium": 0.7, "hard": 0.3}
            t_sum = 1.0
        target = {k: v / t_sum for k, v in target.items()}

        if not bool(cfg.difficulty_sampler_enabled):
            if target.get("hard", 0.0) >= target.get("medium", 0.0):
                return "hard"
            return "medium"

        min_samples = max(1, int(cfg.difficulty_sampler_min_samples))
        hist = list(self._difficulty_window)
        if len(hist) < min_samples:
            return self._sample_bucket(target)

        observed = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
        for b in hist:
            if b in observed:
                observed[b] += 1.0
        h_sum = max(1.0, float(len(hist)))
        observed = {k: observed[k] / h_sum for k in observed}

        deficits = {k: max(0.0, target[k] - observed.get(k, 0.0)) for k in target}
        d_sum = sum(deficits.values())
        if d_sum <= 1e-8:
            return self._sample_bucket(target)
        return self._sample_bucket(deficits)

    # ── Checkpoint Management ───────────────────────────────────────────

    def _save_se_checkpoint(self, step: int, output_dir: pathlib.Path):
        """Save self-evolving specific state."""
        ckpt_dir = output_dir / f"se_checkpoint_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "step": step,
            "proposer_baseline": self.proposer_baseline,
            "solver_baseline": self.solver_baseline,
            "generator_baseline": self.generator_baseline,
            "proposer_gen_baseline": self.proposer_gen_baseline,
            "reward_ema": self.reward_ema,
            "proposer_entropy_mu_ema": float(self.proposer_entropy_mu_ema),
            "u_step_counter": int(self._u_step_counter),
            "all_easy_streak": int(self._all_easy_streak),
            "forced_explore_steps_left": int(self._forced_explore_steps_left),
            "proposer_collapse_streak": int(self._proposer_collapse_streak),
            "warm_start_exit_streak": int(self._warm_start_exit_streak),
            "warm_start_completed": bool(self._warm_start_completed),
            "hardness_debt": float(self._hardness_debt),
            "hardness_debt_cap_streak": int(self._hardness_debt_cap_streak),
            "hardness_debt_escape_steps_left": int(self._hardness_debt_escape_steps_left),
            "entropy_window": list(self._entropy_window),
            "difficulty_window": list(self._difficulty_window),
            "candidate_non_easy_window": list(self._candidate_non_easy_window),
            "all_easy_group_window": list(self._all_easy_group_window),
            "proposer_reward_clipped_window": list(self._proposer_reward_clipped_window),
            "selected_non_easy_window": list(self._selected_non_easy_window),
            "solver_update_applied_window": list(self._solver_update_applied_window),
            "entropy_easy_window": list(self._entropy_easy_window),
            "warm_start_entropy_window": list(self._warm_start_entropy_window),
            "proposer_updater": self.proposer_updater.state_dict(),
            "solver_updater": self.solver_updater.state_dict(),
            "generator_updater": self.generator_updater.state_dict(),
            "replay_buffer_stats": self.replay_buffer.stats(),
        }

        torch.save(state, ckpt_dir / "se_state.pt")

        # Save model adapters
        try:
            self.save_model(str(ckpt_dir / "model"))
        except Exception as e:
            logger.warning(f"[SelfEvolvingTrainer] Model save failed: {e}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(output_dir, keep=self.se_config.max_checkpoints)

        logger.info(f"[SelfEvolvingTrainer] Saved checkpoint at step {step}")

    def _load_se_checkpoint(self, checkpoint_path: str) -> int:
        """Load self-evolving checkpoint. Returns the step to resume from."""
        ckpt_path = pathlib.Path(checkpoint_path)

        se_state_path = ckpt_path / "se_state.pt"
        if not se_state_path.exists():
            # Try looking for se_state.pt in parent
            se_state_path = ckpt_path.parent / "se_state.pt"

        if se_state_path.exists():
            state = torch.load(se_state_path, map_location="cpu")
            self.proposer_baseline = state.get("proposer_baseline", 0.0)
            self.solver_baseline = state.get("solver_baseline", 0.0)
            self.generator_baseline = state.get("generator_baseline", 0.0)
            self.proposer_gen_baseline = state.get("proposer_gen_baseline", 0.0)
            self.reward_ema = state.get("reward_ema", 0.0)
            self.proposer_entropy_mu_ema = float(
                state.get("proposer_entropy_mu_ema", self.se_config.prop_entropy_mu)
            )
            self._u_step_counter = int(state.get("u_step_counter", 0))
            self._all_easy_streak = int(state.get("all_easy_streak", 0))
            self._forced_explore_steps_left = int(state.get("forced_explore_steps_left", 0))
            self._proposer_collapse_streak = int(state.get("proposer_collapse_streak", 0))
            self._warm_start_exit_streak = int(state.get("warm_start_exit_streak", 0))
            self._warm_start_completed = bool(state.get("warm_start_completed", False))
            self._hardness_debt = float(state.get("hardness_debt", 0.0))
            self._hardness_debt_cap_streak = int(state.get("hardness_debt_cap_streak", 0))
            self._hardness_debt_escape_steps_left = int(
                state.get("hardness_debt_escape_steps_left", 0)
            )

            self._entropy_window = collections.deque(
                list(state.get("entropy_window", [])),
                maxlen=self.se_config.entropy_iqr_window_size,
            )
            self._difficulty_window = collections.deque(
                list(state.get("difficulty_window", [])),
                maxlen=self.se_config.difficulty_sampler_window_size,
            )
            self._candidate_non_easy_window = collections.deque(
                list(state.get("candidate_non_easy_window", [])),
                maxlen=self._candidate_non_easy_window.maxlen,
            )
            self._all_easy_group_window = collections.deque(
                list(state.get("all_easy_group_window", [])),
                maxlen=self._all_easy_group_window.maxlen,
            )
            self._proposer_reward_clipped_window = collections.deque(
                list(state.get("proposer_reward_clipped_window", [])),
                maxlen=self._proposer_reward_clipped_window.maxlen,
            )
            self._selected_non_easy_window = collections.deque(
                list(state.get("selected_non_easy_window", [])),
                maxlen=self._selected_non_easy_window.maxlen,
            )
            self._solver_update_applied_window = collections.deque(
                list(state.get("solver_update_applied_window", [])),
                maxlen=self._solver_update_applied_window.maxlen,
            )
            self._entropy_easy_window = collections.deque(
                list(state.get("entropy_easy_window", [])),
                maxlen=self._entropy_easy_window.maxlen,
            )
            self._warm_start_entropy_window = collections.deque(
                list(state.get("warm_start_entropy_window", [])),
                maxlen=self._warm_start_entropy_window.maxlen,
            )

            if "proposer_updater" in state:
                self.proposer_updater.load_state_dict(state["proposer_updater"])
            if "solver_updater" in state:
                self.solver_updater.load_state_dict(state["solver_updater"])
            if "generator_updater" in state:
                self.generator_updater.load_state_dict(state["generator_updater"])

            step = state.get("step", 0)
            logger.info(f"[SelfEvolvingTrainer] Resumed from step {step}")
            return step

        logger.warning(
            f"[SelfEvolvingTrainer] No se_state.pt found at {checkpoint_path}"
        )
        return 0

    def _cleanup_old_checkpoints(self, output_dir: pathlib.Path, keep: int = 5):
        """Remove old checkpoints, keeping only the most recent `keep`."""
        import glob
        ckpt_dirs = sorted(
            output_dir.glob("se_checkpoint_*"),
            key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0,
        )
        while len(ckpt_dirs) > keep:
            old_dir = ckpt_dirs.pop(0)
            try:
                import shutil
                shutil.rmtree(old_dir)
            except Exception:
                pass

    # ── Logging ─────────────────────────────────────────────────────────

    def _log_step(self, step: int, stats: Dict):
        """Log step statistics."""
        phase = stats.get("phase", "unknown")
        step_time = stats.get("step_time", 0.0)

        msg_parts = [f"step={step}", f"phase={phase}", f"time={step_time:.1f}s"]

        if phase == "understanding":
            msg_parts.append(f"entropy={stats.get('entropy', 0.0):.3f}")
            msg_parts.append(f"prop_r={stats.get('proposer_reward', 0.0):.3f}")
            msg_parts.append(f"sol_r={stats.get('solver_reward', 0.0):.3f}")
            msg_parts.append(f"easy={stats.get('easy_question', '?')}")
            msg_parts.append(f"margin={stats.get('margin', 0.0):.2f}")
            if "proposer_candidate_non_easy_rate" in stats:
                msg_parts.append(
                    f"cand_non_easy={float(stats.get('proposer_candidate_non_easy_rate', 0.0)):.2f}"
                )
            if "proposer_hardness_debt" in stats:
                msg_parts.append(
                    f"debt={float(stats.get('proposer_hardness_debt', 0.0)):.2f}"
                )
            if "proposer_warm_start_active" in stats:
                msg_parts.append(
                    f"warm={1 if bool(stats.get('proposer_warm_start_active')) else 0}"
                )
            if bool(stats.get("proposer_early_triggered", False)):
                msg_parts.append("early=triggered")
            retries = stats.get("proposer_retries", 0)
            if retries > 0:
                msg_parts.append(f"retries={retries}")
            if stats.get("solver_update_skipped"):
                msg_parts.append("sol_skip=True")
            diff_bucket = stats.get("difficulty_bucket", "")
            if diff_bucket:
                msg_parts.append(f"diff={diff_bucket}")
        elif phase == "generation":
            msg_parts.append(f"reward_mean={stats.get('gen_reward_mean', 0.0):.3f}")
            msg_parts.append(f"reward_max={stats.get('gen_reward_max', 0.0):.3f}")
            msg_parts.append(f"replay_sz={stats.get('replay_buffer_size', 0)}")

        logger.info(f"[SE] {' | '.join(msg_parts)}")

        # Log to W&B if available
        try:
            import wandb
            if wandb.run is not None:
                log_dict = {
                    f"se/{k}": v for k, v in stats.items()
                    if isinstance(v, (int, float)) and not math.isnan(v)
                    and not math.isinf(v)
                }
                wandb.log(log_dict, step=step)
        except Exception:
            pass
