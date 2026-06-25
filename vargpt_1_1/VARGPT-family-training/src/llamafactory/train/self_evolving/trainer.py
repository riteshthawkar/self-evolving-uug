"""
SelfEvolvingTrainer: Main training loop for the self-evolving framework on VARGPT.

Inherits from HF Trainer but overrides train() with the custom multi-phase
proposer-solver-generator loop:
  - U-step: proposer proposes questions, solver answers, both get GRPO rewards
  - G-step: proposer proposes specs, generator creates images, GRPO on discrete tokens

Ported from BLIP3o's unified_trainer.py with VARGPT-specific adaptations.
"""

import collections
import datetime as dt
import gc
import json
import logging
import math
import os
import pathlib
import random
import re
import shutil
import time
import traceback
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
    sanitize_and_score_generation_spec,
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
from .rewards import score_generated_candidates
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

        # ── Runtime safety / health state ────────────────────────────────
        self._consecutive_step_errors: int = 0
        self._total_step_errors: int = 0
        self._generation_consecutive_unhealthy: int = 0
        g_window = max(1, int(getattr(se_config, "generation_failfast_window", 20)))
        self._generation_health_window: collections.deque = collections.deque(maxlen=g_window)

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

        # Step-level logging paths (initialized in train once output_dir is ready)
        self.run_dir: Optional[pathlib.Path] = None
        self.logs_dir: Optional[pathlib.Path] = None
        self.iter_log_path: Optional[pathlib.Path] = None
        self.understanding_log_path: Optional[pathlib.Path] = None
        self.generation_log_path: Optional[pathlib.Path] = None
        self.error_log_path: Optional[pathlib.Path] = None
        self.release_rollouts_log_path: Optional[pathlib.Path] = None
        self.release_generation_rollouts_log_path: Optional[pathlib.Path] = None
        self.metrics_log_path: Optional[pathlib.Path] = None
        self.status_path: Optional[pathlib.Path] = None
        self.summary_path: Optional[pathlib.Path] = None
        self.config_path: Optional[pathlib.Path] = None
        self.checkpoint_root: Optional[pathlib.Path] = None
        self.last_checkpoint_dir: str = ""
        self._metric_stats: Dict[str, Dict[str, float]] = {}
        self._phase_counts: Dict[str, int] = {"understanding": 0, "generation": 0, "error": 0}

        logger.info(
            f"[SelfEvolvingTrainer] Initialized with "
            f"U={se_config.understanding_steps_per_cycle}, "
            f"G={se_config.generation_steps_per_cycle}, "
            f"total_steps={se_config.total_steps}"
        )

    @staticmethod
    def _dist_mean_scalar(value: float, *, device: Optional[torch.device] = None) -> float:
        if not (dist.is_available() and dist.is_initialized()):
            return float(value)
        dev = device
        if dev is None:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
                dev = torch.device(f"cuda:{local_rank}")
            else:
                dev = torch.device("cpu")
        tensor = torch.tensor([float(value)], dtype=torch.float32, device=dev)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(dist.get_world_size())
        return float(tensor.item())

    @staticmethod
    def _dist_all_true(flag: bool, *, device: Optional[torch.device] = None) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return bool(flag)
        dev = device
        if dev is None:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
                dev = torch.device(f"cuda:{local_rank}")
            else:
                dev = torch.device("cpu")
        tensor = torch.tensor([1 if flag else 0], dtype=torch.int32, device=dev)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return bool(int(tensor.item()) == 1)

    @staticmethod
    def _dist_any_true(flag: bool, *, device: Optional[torch.device] = None) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return bool(flag)
        dev = device
        if dev is None:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
                dev = torch.device(f"cuda:{local_rank}")
            else:
                dev = torch.device("cpu")
        tensor = torch.tensor([1 if flag else 0], dtype=torch.int32, device=dev)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return bool(int(tensor.item()) == 1)

    def _sync_distributed_step_state(self) -> None:
        if not self._is_ddp:
            return
        self.proposer_baseline = self._dist_mean_scalar(self.proposer_baseline, device=self.device)
        self.solver_baseline = self._dist_mean_scalar(self.solver_baseline, device=self.device)
        self.generator_baseline = self._dist_mean_scalar(self.generator_baseline, device=self.device)
        self.proposer_gen_baseline = self._dist_mean_scalar(self.proposer_gen_baseline, device=self.device)
        self.reward_ema = self._dist_mean_scalar(self.reward_ema, device=self.device)
        self.proposer_entropy_mu_ema = self._dist_mean_scalar(
            self.proposer_entropy_mu_ema,
            device=self.device,
        )
        if self.proposer_updater is not None:
            self.proposer_updater.kl_coef = self._dist_mean_scalar(
                self.proposer_updater.kl_coef,
                device=self.device,
            )
        if self.solver_updater is not None:
            self.solver_updater.kl_coef = self._dist_mean_scalar(
                self.solver_updater.kl_coef,
                device=self.device,
            )
        if self.generator_updater is not None:
            self.generator_updater.kl_coef = self._dist_mean_scalar(
                self.generator_updater.kl_coef,
                device=self.device,
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
        self._init_step_log_paths(output_dir)

        # Save config
        config_payload = {
            k: str(v) if not isinstance(v, (int, float, bool, str, type(None)))
            else v
            for k, v in cfg.__dict__.items()
        }
        _json_dump(output_dir / "se_config.json", config_payload)
        if self.config_path is not None:
            _json_dump(self.config_path, config_payload)

        logger.info(f"[SelfEvolvingTrainer] Starting training from step {start_step}")

        cycle_len = cfg.understanding_steps_per_cycle + cfg.generation_steps_per_cycle
        if cycle_len <= 0:
            raise ValueError("Cycle length must be > 0")

        self.model.train()
        run_started_at = float(time.time())
        last_completed_step = max(start_step - 1, cfg.start_step - 1)
        last_attempted_step = last_completed_step

        def _emit_training_logs(step_id: int, *, phase: str, step_time_sec: float):
            progress = self._progress_core(
                step=int(step_id),
                phase=str(phase),
                run_started_at=run_started_at,
            )
            metrics = self._release_metrics(step_time_sec=step_time_sec)
            self._write_status(state="running", progress=progress, metrics=metrics)
            if int(step_id) % max(1, int(cfg.log_every)) == 0:
                self._append_metrics({"kind": "heartbeat", **progress, **metrics})

        self._write_status(
            state="running",
            progress=self._progress_core(
                step=int(last_completed_step),
                phase="init",
                run_started_at=run_started_at,
            ),
            metrics=self._release_metrics(step_time_sec=0.0),
        )

        try:
            for step in range(start_step, cfg.total_steps):
                self.global_step = step
                step_start = time.time()
                last_attempted_step = step

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
                    traceback.print_exc()
                    step_stats = {"phase": "error", "error": str(e)}
                    self._consecutive_step_errors += 1
                    self._total_step_errors += 1
                    if bool(getattr(cfg, "fail_on_step_error", True)):
                        max_consecutive = max(0, int(getattr(cfg, "max_consecutive_step_errors", 0)))
                        max_total = max(0, int(getattr(cfg, "max_total_step_errors", 0)))
                        if (
                            self._consecutive_step_errors > max_consecutive
                            or self._total_step_errors > max_total
                        ):
                            raise RuntimeError(
                                "[SelfEvolvingTrainer] Aborting due to step errors: "
                                f"consecutive={self._consecutive_step_errors} (limit={max_consecutive}), "
                                f"total={self._total_step_errors} (limit={max_total}), step={step}"
                            ) from e

                step_stats["step"] = step
                step_stats["step_time"] = time.time() - step_start
                if step_stats.get("phase") != "error":
                    self._consecutive_step_errors = 0
                    self._sync_distributed_step_state()

                phase_name = str(step_stats.get("phase", "unknown"))
                if phase_name in self._phase_counts:
                    self._phase_counts[phase_name] += 1

                for key, value in step_stats.items():
                    if isinstance(value, bool):
                        self._update_metric(str(key), 1.0 if value else 0.0)
                    elif isinstance(value, (int, float)):
                        numeric = float(value)
                        if math.isfinite(numeric):
                            self._update_metric(str(key), numeric)

                # Persist full per-step records (BLIP3o-like traceability).
                step_record = dict(step_stats)
                step_record["timestamp_unix"] = time.time()
                self._append_jsonl(self.iter_log_path, step_record)
                if phase_name == "understanding":
                    self._append_jsonl(self.understanding_log_path, step_record)
                    self._append_jsonl(self.release_rollouts_log_path, step_record)
                elif phase_name == "generation":
                    self._append_jsonl(self.generation_log_path, step_record)
                    self._append_jsonl(self.release_generation_rollouts_log_path, step_record)
                elif phase_name == "error":
                    self._append_jsonl(self.error_log_path, step_record)

                _emit_training_logs(step, phase=phase_name, step_time_sec=float(step_stats.get("step_time", 0.0)))

                # ── Logging ─────────────────────────────────────────────────
                if step % cfg.log_every == 0:
                    self._log_step(step, step_stats)

                # ── Generation health fail-fast ─────────────────────────────
                if step_stats.get("phase") == "generation":
                    self._update_generation_health(step, step_stats)

                # ── Checkpointing ───────────────────────────────────────────
                if step > 0 and step % cfg.save_every == 0:
                    self._save_se_checkpoint(step, output_dir)

                # ── Memory management ───────────────────────────────────────
                if cfg.clear_cache_every > 0 and step % cfg.clear_cache_every == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                last_completed_step = step

            # Final save
            self._save_se_checkpoint(cfg.total_steps, output_dir)
            summary = {
                "run_dir": str(self.run_dir) if self.run_dir is not None else str(output_dir),
                "final_step": int(cfg.total_steps),
                "start_step": int(start_step),
                "status": "completed",
                "phase_counts": dict(self._phase_counts),
                "metrics": self._metrics_summary(),
                "rollouts_log_path": str(self.release_rollouts_log_path) if self.release_rollouts_log_path is not None else "",
                "generation_rollouts_log_path": str(self.release_generation_rollouts_log_path) if self.release_generation_rollouts_log_path is not None else "",
                "metrics_log_path": str(self.metrics_log_path) if self.metrics_log_path is not None else "",
                "last_checkpoint_dir": str(self.last_checkpoint_dir),
            }
            if self.summary_path is not None:
                _json_dump(self.summary_path, summary)
            self._append_metrics(
                {
                    "kind": "final_summary",
                    **self._progress_core(
                        step=int(cfg.total_steps),
                        phase="completed",
                        run_started_at=run_started_at,
                    ),
                    **summary,
                }
            )
            self._write_status(
                state="completed",
                progress=self._progress_core(
                    step=int(cfg.total_steps),
                    phase="completed",
                    run_started_at=run_started_at,
                ),
                metrics=self._release_metrics(step_time_sec=0.0),
            )
            logger.info("[SelfEvolvingTrainer] Training complete.")
            return summary
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            interrupted_step = int(last_attempted_step)
            if self.is_world_process_zero():
                logger.error(
                    f"[SelfEvolvingTrainer] Training interrupted at step {interrupted_step}: {error_text}"
                )
            try:
                self._save_se_checkpoint(max(0, interrupted_step), output_dir)
            except Exception as save_exc:
                logger.warning(f"[SelfEvolvingTrainer] Emergency checkpoint failed: {save_exc}")
            summary = {
                "run_dir": str(self.run_dir) if self.run_dir is not None else str(output_dir),
                "final_step": int(max(last_completed_step, interrupted_step)),
                "start_step": int(start_step),
                "status": "interrupted",
                "interrupted_at_step": int(interrupted_step),
                "error": error_text,
                "phase_counts": dict(self._phase_counts),
                "metrics": self._metrics_summary(),
                "rollouts_log_path": str(self.release_rollouts_log_path) if self.release_rollouts_log_path is not None else "",
                "generation_rollouts_log_path": str(self.release_generation_rollouts_log_path) if self.release_generation_rollouts_log_path is not None else "",
                "metrics_log_path": str(self.metrics_log_path) if self.metrics_log_path is not None else "",
                "last_checkpoint_dir": str(self.last_checkpoint_dir),
            }
            if self.summary_path is not None:
                _json_dump(self.summary_path, summary)
            interrupted_progress = self._progress_core(
                step=int(max(last_completed_step, interrupted_step)),
                phase="interrupted",
                run_started_at=run_started_at,
            )
            self._append_metrics({"kind": "interrupted", **interrupted_progress, **summary})
            self._write_status(
                state="interrupted",
                progress=interrupted_progress,
                metrics=self._release_metrics(step_time_sec=0.0),
                last_error=error_text,
            )
            raise

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
        selected_choice_mode = False
        selected_choice_option_a = ""
        selected_choice_option_b = ""
        proposer_reject_counts: collections.Counter = collections.Counter()
        proposer_raw_previews: List[str] = []
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
        spot_entropy_min_gate = max(
            0.0, float(getattr(cfg, "proposer_spot_entropy_min_gate", 0.05))
        )
        attempt = 0
        spot_all_easy_low_entropy = False

        for attempt in range(max_retries):
            # ── 2. Proposer generates question candidates ───────────────
            with use_role(self.model, ROLE_PROPOSER):
                candidates, proposer_raw = self._generate_proposer_candidates(
                    image,
                    step,
                    target_difficulty=desired_difficulty_bucket,
                    image_source_hint=source,
                    num_candidates=proposer_num_candidates,
                    temperature=proposer_temp,
                    top_p=proposer_top_p,
                )
            proposer_raw_previews.append(str(proposer_raw or "")[:500])
            if not candidates:
                proposer_reject_counts["parser_no_candidates"] += 1
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
                meta = dict(cand.get("meta", {}))
                q_compiled, compile_ok, compile_reason = self._compile_question_from_slots(q, meta)
                meta["_compiler_valid"] = "1" if compile_ok else "0"
                meta["_compiler_reason"] = str(compile_reason)
                if q_compiled:
                    q = q_compiled
                if not q:
                    proposer_reject_counts["empty_question"] += 1
                    continue
                quality = self._question_quality_score(q, meta)
                structural_score = float(quality.get("score", 0.0))
                structural_valid = bool(quality.get("valid", False))
                structural_min = float(getattr(cfg, "proposer_question_structural_min_score", 0.60))
                if (not structural_valid) or structural_score < structural_min:
                    proposer_reject_counts["structural_quality"] += 1
                    continue
                judge_score = 1.0
                judge_raw = "disabled"
                if bool(getattr(cfg, "proposer_question_model_judge_enabled", True)):
                    with use_role(self.model, ROLE_SOLVER):
                        judge_score, judge_raw = self._judge_visual_question(image, q)
                judge_weight = max(
                    0.0,
                    min(1.0, float(getattr(cfg, "proposer_question_model_judge_weight", 0.35))),
                )
                quality_score = (1.0 - judge_weight) * structural_score + judge_weight * float(judge_score)
                quality_min = float(getattr(cfg, "proposer_question_quality_min_score", 0.70))
                if quality_score < quality_min:
                    proposer_reject_counts["combined_quality"] += 1
                    continue
                opt_a, opt_b = self._extract_forced_choice_options(
                    str(meta.get("two_answer_test", "") or "")
                )
                choice_mode = bool(
                    getattr(cfg, "solver_use_forced_choice_from_proposer", False)
                    and opt_a
                    and opt_b
                )
                solver_q = q
                if choice_mode:
                    solver_q = (
                        f"{q}\n"
                        f"Choose exactly one option.\n"
                        f"Option A: {opt_a}\n"
                        f"Option B: {opt_b}\n"
                        "Respond with only 'A' or 'B'."
                    )
                with use_role(self.model, ROLE_SOLVER):
                    ans_spot, sc_spot = self._generate_solver_answers(
                        image, solver_q, num_samples=spot_n,
                    )
                if not ans_spot:
                    proposer_reject_counts["no_spot_answers"] += 1
                    continue

                if choice_mode:
                    norm = []
                    for a in ans_spot:
                        vote = self._parse_forced_choice_answer(a, opt_a, opt_b)
                        if vote == "a":
                            norm.append(normalize_answer(opt_a))
                        elif vote == "b":
                            norm.append(normalize_answer(opt_b))
                else:
                    norm = [normalize_answer(a) for a in ans_spot]
                norm = [n for n in norm if n]
                if not norm:
                    proposer_reject_counts["empty_spot_answers"] += 1
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

                objective_ok = bool(quality_score >= quality_min)
                if cfg.proposer_require_objective and not objective_ok:
                    spot_reward -= controller_penalty_boost * float(cfg.proposer_non_objective_penalty)

                cert = self._proposer_certificate_score(q, meta)
                cert_score = float(cert.get("score", 0.0))
                cert_valid = float(cert.get("valid", 0.0))
                if (not compile_ok) and bool(getattr(cfg, "proposer_slot_compiler_strict", True)):
                    cert_valid = 0.0
                    cert_score = min(cert_score, 0.0)
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
                        "question_quality_score": float(quality_score),
                        "question_structural_score": float(structural_score),
                        "question_model_judge_score": float(judge_score),
                        "question_model_judge_raw": str(judge_raw)[:80],
                        "certificate_score": float(cert_score),
                        "certificate_valid": float(cert_valid),
                        "certificate_bonus": float(cert_bonus),
                        "spot_reward": float(spot_reward),
                        "choice_mode": bool(choice_mode),
                        "choice_option_a": str(opt_a),
                        "choice_option_b": str(opt_b),
                    }
                )

            if not current_candidates:
                proposer_reject_counts["no_rankable_candidates"] += 1
                continue

            def _cand_key(cand: Dict[str, object]) -> Tuple[float, float, float, float]:
                return (
                    1.0 if not bool(cand.get("easy_spot", True)) else 0.0,
                    float(cand.get("certificate_valid", 0.0)),
                    float(cand.get("spot_reward", -1e9)),
                    float(cand.get("spot_entropy", 0.0)),
                )

            # Candidate selection fallback priority:
            # 1) non-easy + objective + cert-valid
            # 2) objective + cert-valid
            # 3) best available candidate
            best_any = max(current_candidates, key=_cand_key)
            acceptable = [
                c for c in current_candidates
                if (not bool(c.get("easy_spot", True)))
                and bool(c.get("objective_ok", False))
                and (float(c.get("certificate_valid", 0.0)) > 0.5)
            ]
            valid = [
                c for c in current_candidates
                if bool(c.get("objective_ok", False))
                and (float(c.get("certificate_valid", 0.0)) > 0.5)
            ]
            if acceptable:
                selected_now = max(acceptable, key=_cand_key)
            elif valid:
                selected_now = max(valid, key=_cand_key)
            else:
                selected_now = best_any

            cand_non_easy_rate_attempt = (
                sum(1 for c in current_candidates if not bool(c.get("easy_spot", True)))
                / float(max(1, len(current_candidates)))
            )
            if (
                cand_non_easy_rate_attempt <= 0.0
                and float(selected_now.get("spot_entropy", 0.0)) < spot_entropy_min_gate
            ):
                spot_all_easy_low_entropy = True
                self._forced_explore_steps_left = max(
                    int(self._forced_explore_steps_left),
                    max(1, int(getattr(cfg, "all_easy_explore_steps", 10))),
                )

            candidate_records = current_candidates
            selected_candidate = selected_now
            selected_candidate_idx = int(selected_now.get("candidate_index", -1))
            selected_meta = dict(selected_now.get("meta", {}))
            selected_choice_mode = bool(selected_now.get("choice_mode", False))
            selected_choice_option_a = str(selected_now.get("choice_option_a", "") or "")
            selected_choice_option_b = str(selected_now.get("choice_option_b", "") or "")
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

        have_question = bool(question_text)
        if self._is_ddp and not self._dist_all_true(have_question, device=self.device):
            return {"u_skipped": True, "reason": "ddp_question_mismatch"}
        if not have_question:
            return {
                "u_skipped": True,
                "reason": "no_question",
                "proposer_candidate_reject_counts": dict(proposer_reject_counts),
                "proposer_raw_previews": proposer_raw_previews[-3:],
            }

        # ── 4. Full solver rollout on selected candidate ────────────────
        solver_rollout_question = question_text
        if selected_choice_mode and selected_choice_option_a and selected_choice_option_b:
            solver_rollout_question = (
                f"{question_text}\n"
                f"Choose exactly one option.\n"
                f"Option A: {selected_choice_option_a}\n"
                f"Option B: {selected_choice_option_b}\n"
                "Respond with only 'A' or 'B'."
            )
        with use_role(self.model, ROLE_SOLVER):
            answers, solver_completions = self._generate_solver_answers(
                image, solver_rollout_question, num_samples=cfg.num_solver_samples,
            )
        have_answers = bool(answers)
        if self._is_ddp and not self._dist_all_true(have_answers, device=self.device):
            return {"u_skipped": True, "reason": "ddp_answers_mismatch"}
        if not have_answers:
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
        stats["proposer_candidate_reject_counts"] = dict(proposer_reject_counts)
        stats["proposer_raw_previews"] = proposer_raw_previews[-3:]
        stats["proposer_candidate_count"] = len(candidate_records)
        stats["proposer_selected_candidate_index"] = selected_candidate_idx
        if selected_candidate:
            stats["proposer_question_quality_score"] = float(
                selected_candidate.get("question_quality_score", 0.0)
            )
            stats["proposer_question_structural_score"] = float(
                selected_candidate.get("question_structural_score", 0.0)
            )
            stats["proposer_question_model_judge_score"] = float(
                selected_candidate.get("question_model_judge_score", 0.0)
            )
            stats["proposer_question_model_judge_raw"] = str(
                selected_candidate.get("question_model_judge_raw", "")
            )[:80]
        if candidate_records:
            stats["proposer_candidate_non_easy_rate"] = (
                sum(1 for c in candidate_records if not bool(c.get("easy_spot", True)))
                / float(len(candidate_records))
            )
        else:
            stats["proposer_candidate_non_easy_rate"] = 0.0
        stats["proposer_spot_entropy_min_gate"] = float(spot_entropy_min_gate)
        stats["proposer_spot_all_easy_low_entropy"] = bool(spot_all_easy_low_entropy)

        # ── 5. Compute rewards ──────────────────────────────────────────
        # Normalize answers for voting
        if selected_choice_mode and selected_choice_option_a and selected_choice_option_b:
            norm_answers = []
            for a in answers:
                vote = self._parse_forced_choice_answer(
                    a, selected_choice_option_a, selected_choice_option_b
                )
                if vote == "a":
                    norm_answers.append(normalize_answer(selected_choice_option_a))
                elif vote == "b":
                    norm_answers.append(normalize_answer(selected_choice_option_b))
        else:
            norm_answers = [normalize_answer(a) for a in answers]
        have_norm_answers = bool(norm_answers)
        if self._is_ddp and not self._dist_all_true(have_norm_answers, device=self.device):
            return {"u_skipped": True, "reason": "ddp_normalized_answers_mismatch"}
        if not have_norm_answers:
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
        use_prop_grpo = update_rule == "grpo" and len(candidate_records) > 1
        if self._is_ddp and (update_rule == "grpo"):
            use_prop_grpo = self._dist_all_true(use_prop_grpo, device=self.device)

        if use_prop_grpo:
            group_rewards: List[float] = []
            group_buckets: List[str] = []
            for c in candidate_records:
                r = float(c.get("spot_reward", 0.0))
                c_easy = bool(c.get("easy_spot", True))
                c_entropy = float(c.get("spot_entropy", 0.0))
                c_margin = float(c.get("spot_margin", 1.0))
                if c_easy:
                    bucket = "easy"
                elif (
                    c_entropy >= float(getattr(cfg, "difficulty_hard_min_entropy", 0.90))
                    and c_margin <= float(getattr(cfg, "difficulty_hard_max_margin", 0.35))
                ):
                    bucket = "hard"
                else:
                    bucket = "medium"
                if int(c.get("candidate_index", -1)) == int(selected_candidate_idx):
                    r = float(proposer_reward)
                    bucket = str(diff_bucket)
                group_rewards.append(r)
                group_buckets.append(bucket)

            group_rewards, group_rank_deltas = self._apply_grpo_pairwise_ranking(
                group_rewards, group_buckets
            )
            group_rewards, group_all_easy_deltas, group_all_easy_applied = (
                self._apply_all_easy_relative_negatives(group_rewards, group_buckets)
            )
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
            stats["grpo_pairwise_rank_delta_mean"] = (
                float(sum(group_rank_deltas) / max(1, len(group_rank_deltas)))
                if group_rank_deltas
                else 0.0
            )
            stats["grpo_pairwise_rank_delta_max"] = (
                float(max(group_rank_deltas)) if group_rank_deltas else 0.0
            )
            stats["grpo_pairwise_rank_delta_min"] = (
                float(min(group_rank_deltas)) if group_rank_deltas else 0.0
            )
            stats["grpo_all_easy_rank_applied"] = bool(group_all_easy_applied)
            stats["grpo_all_easy_rank_delta_mean"] = (
                float(sum(group_all_easy_deltas) / max(1, len(group_all_easy_deltas)))
                if group_all_easy_deltas
                else 0.0
            )
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
            stats["prop_update_rule"] = (
                "reinforce_ddp_fallback" if update_rule == "grpo" and len(candidate_records) > 1 else "reinforce"
            )

        self.proposer_baseline = (
            cfg.baseline_momentum * self.proposer_baseline
            + (1 - cfg.baseline_momentum) * proposer_reward
        )

        # ── 7. Update solver ────────────────────────────────────────────
        # Skip solver update on easy questions to avoid wasting gradient
        # budget on trivial cases (the solver already knows the answer).
        skip_solver = False
        solver_skip_reason = None
        if cfg.proposer_require_objective and non_objective_question:
            skip_solver = True
            solver_skip_reason = "non_objective_question"
        elif (
            easy_question
            and cfg.solver_skip_update_on_easy
            and majority_frac >= cfg.easy_update_majority_frac_threshold
        ):
            skip_solver = True
            solver_skip_reason = "easy_question"
        stats["solver_update_skipped"] = skip_solver
        if solver_skip_reason is not None:
            stats["solver_update_skipped_reason"] = solver_skip_reason
        solver_update_applied = False

        do_solver_update = not skip_solver
        if self._is_ddp:
            do_solver_update = self._dist_all_true(do_solver_update, device=self.device)
            if (not do_solver_update) and (not skip_solver):
                stats["solver_update_skipped_reason"] = "ddp_solver_skip_mismatch"
        if do_solver_update:
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
        elif skip_solver and solver_skip_reason is not None:
            stats["solver_update_skipped_reason"] = solver_skip_reason
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
            have_source_image = image is not None
            if self._is_ddp and not self._dist_all_true(have_source_image, device=self.device):
                return {"g_skipped": True, "reason": "ddp_source_image_mismatch", **stats}
            if image is None:
                return {"g_skipped": True, "reason": "no_source_image"}
            spec, spec_completion = self._generate_spec(image, step)
            stats["image_source"] = source

        have_spec = spec is not None and bool(getattr(spec, "prompt", ""))
        if self._is_ddp and not self._dist_all_true(have_spec, device=self.device):
            return {"g_skipped": True, "reason": "ddp_spec_mismatch", **stats}
        if not have_spec:
            return {"g_skipped": True, "reason": "no_spec"}

        spec, spec_quality, spec_quality_details = sanitize_and_score_generation_spec(
            spec,
            min_spec_qa_pairs=int(getattr(cfg, "min_spec_qa_pairs", 2)),
            max_question_words=int(getattr(cfg, "max_question_words", 24)),
            max_expected_words=int(getattr(cfg, "max_expected_words", 8)),
        )
        have_valid_spec = bool(spec.prompt) and bool(spec.qa_pairs)
        if self._is_ddp and not self._dist_all_true(have_valid_spec, device=self.device):
            return {"g_skipped": True, "reason": "ddp_sanitized_spec_mismatch", **stats}
        if not have_valid_spec:
            return {"g_skipped": True, "reason": "invalid_spec_after_sanitize"}

        stats["gen_prompt"] = spec.prompt[:100]
        stats["num_qa_pairs"] = len(spec.qa_pairs)
        stats["spec_quality"] = float(spec_quality)
        stats["spec_quality_details"] = spec_quality_details

        # ── 2. Generate K candidate images ──────────────────────────────
        K = cfg.num_generations
        stats["generation_attempted"] = int(K)
        candidates = []  # List of (PIL Image, pixel_gen_tensor)
        generation_failures = 0
        with use_role(self.model, ROLE_GENERATOR):
            for k in range(K):
                try:
                    gen_image, gen_tensor = self._generate_image(spec.prompt)
                    if gen_image is not None and gen_tensor is not None:
                        candidates.append((gen_image, gen_tensor))
                    else:
                        generation_failures += 1
                except Exception as e:
                    generation_failures += 1
                    logger.warning(f"[SelfEvolvingTrainer] Generation {k} failed: {e}")

        stats["generation_succeeded"] = int(len(candidates))
        stats["generation_failures"] = int(generation_failures)
        stats["generation_success_rate"] = (
            float(len(candidates)) / float(max(1, K))
        )
        have_candidates = bool(candidates)
        if self._is_ddp and not self._dist_all_true(have_candidates, device=self.device):
            return {"g_skipped": True, "reason": "ddp_generation_candidate_mismatch", **stats}
        if not candidates:
            return {
                "g_skipped": True,
                "reason": "no_candidates",
                **stats,
            }
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
        candidate_images = [cand[0] for cand in candidates]
        reward_details_list = score_generated_candidates(
            model=self.model,
            processor=self.processor,
            images=candidate_images,
            prompt=spec.prompt,
            questions=questions,
            expected_answers=expected_answers,
            device=self.device,
            config=cfg,
            spec_quality=float(spec_quality),
        )
        rewards = [float(details.get("total_reward", 0.0)) for details in reward_details_list]
        have_rewards = bool(rewards)
        if self._is_ddp and not self._dist_all_true(have_rewards, device=self.device):
            return {"g_skipped": True, "reason": "ddp_reward_scoring_mismatch", **stats}
        if not rewards:
            return {"g_skipped": True, "reason": "reward_scoring_failed", **stats}

        valid_reward_indices = [
            idx
            for idx, details in enumerate(reward_details_list)
            if any(str(log.get("status", "")) == "ok" for log in details.get("qa_logs", []))
        ]
        have_valid_reward_indices = bool(valid_reward_indices)
        if self._is_ddp and not self._dist_all_true(have_valid_reward_indices, device=self.device):
            return {"g_skipped": True, "reason": "ddp_valid_reward_mismatch", **stats}
        if not valid_reward_indices:
            return {"g_skipped": True, "reason": "empty_generation_qa_entropy", **stats}

        best_idx = max(valid_reward_indices, key=lambda idx: rewards[idx])
        best_reward_details = reward_details_list[best_idx]
        selected_total_reward = float(best_reward_details.get("total_reward", 0.0))
        mean_entropy = float(best_reward_details.get("mean_entropy_nats", 0.0))
        entropy_component = gaussian_reward(
            mean_entropy,
            float(getattr(cfg, "prop_entropy_mu", 0.90)),
            float(getattr(cfg, "prop_entropy_sigma", 0.35)),
        )
        if mean_entropy <= 1e-6:
            entropy_component = -max(0.0, float(getattr(cfg, "zero_entropy_reward_cap", 0.10)))
        quality_component = max(0.0, min(1.0, float(best_reward_details.get("total_reward", 0.0))))
        entropy_alpha = max(0.0, min(1.0, float(getattr(cfg, "proposer_gen_entropy_weight", 0.7))))
        proposer_gen_reward = max(
            -1.0,
            min(
                1.0,
                entropy_alpha * float(entropy_component)
                + (1.0 - entropy_alpha) * float(quality_component),
            ),
        )
        quality_gate_ok = bool(
            float(spec_quality) >= float(getattr(cfg, "min_spec_quality_for_update", 0.35))
            and len(spec.qa_pairs) >= int(getattr(cfg, "min_spec_qa_pairs", 2))
        )

        stats["gen_rewards"] = rewards
        stats["gen_reward_mean"] = sum(rewards) / len(rewards)
        stats["gen_reward_max"] = max(rewards)
        stats["best_candidate_idx"] = int(best_idx)
        stats["best_spec_score"] = float(best_reward_details.get("spec_score", 0.0))
        stats["best_cycle_score"] = float(best_reward_details.get("cycle_score", 0.0))
        stats["best_diversity_score"] = float(best_reward_details.get("diversity_score", 0.0))
        stats["best_contradiction_score"] = float(best_reward_details.get("contradiction_score", 0.0))
        stats["best_base_reward"] = float(best_reward_details.get("base_reward", 0.0))
        stats["best_total_reward"] = float(best_reward_details.get("total_reward", 0.0))
        stats["best_cycle_caption"] = str(best_reward_details.get("cycle_caption", ""))
        stats["best_qa_confidence"] = float(best_reward_details.get("qa_confidence", 0.0))
        stats["mean_entropy_nats"] = float(mean_entropy)
        stats["entropy_component"] = float(entropy_component)
        stats["quality_component"] = float(quality_component)
        stats["proposer_gen_reward"] = float(proposer_gen_reward)
        stats["entropy_weight_alpha"] = float(entropy_alpha)
        stats["quality_gate_ok"] = bool(quality_gate_ok)
        stats["gen_candidate_details"] = [
            {
                "candidate_idx": int(details.get("candidate_idx", idx)),
                "spec_score": float(details.get("spec_score", 0.0)),
                "cycle_score": float(details.get("cycle_score", 0.0)),
                "diversity_score": float(details.get("diversity_score", 0.0)),
                "contradiction_score": float(details.get("contradiction_score", 0.0)),
                "base_reward": float(details.get("base_reward", 0.0)),
                "total_reward": float(details.get("total_reward", 0.0)),
                "mean_entropy_nats": float(details.get("mean_entropy_nats", 0.0)),
                "qa_confidence": float(details.get("qa_confidence", 0.0)),
            }
            for idx, details in enumerate(reward_details_list)
        ]

        # ── 4. GRPO update on generator ─────────────────────────────────
        local_gen_update_ready = (
            len(candidates) >= 2
            and quality_gate_ok
            and (len(rewards) >= 2)
            and (float(torch.tensor(rewards, dtype=torch.float32).std().item()) >= float(self.generator_updater.config.grpo_min_group_std))
        )
        gen_update_ready = local_gen_update_ready
        if self._is_ddp:
            gen_update_ready = self._dist_all_true(local_gen_update_ready, device=self.device)

        if gen_update_ready:
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
        elif len(candidates) < 2:
            stats["gen_update_skipped_reason"] = "insufficient_candidates"
        elif self._is_ddp and not local_gen_update_ready:
            stats["gen_update_skipped_reason"] = "ddp_generator_update_mismatch"
        elif len(rewards) >= 2 and float(torch.tensor(rewards, dtype=torch.float32).std().item()) < float(self.generator_updater.config.grpo_min_group_std):
            stats["gen_update_skipped_reason"] = "low_reward_std"
        else:
            stats["gen_update_skipped_reason"] = "low_spec_quality"

        # ── 5. Update proposer with generation quality ──────────────────
        if cfg.proposer_gen_reward_enabled:
            prop_gen_update_ready = bool(quality_gate_ok)
            if self._is_ddp:
                prop_gen_update_ready = self._dist_all_true(prop_gen_update_ready, device=self.device)
            if prop_gen_update_ready:
                if cfg.imageless_proposer_mode:
                    prop_prompt = build_imageless_spec_prompt(
                        topic or "",
                        target_difficulty=self._current_difficulty(step),
                    )
                    prop_stats = self.proposer_updater.step(
                        image=None,
                        prompt=prop_prompt,
                        completion=spec_completion,
                        reward=proposer_gen_reward,
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
                        reward=proposer_gen_reward,
                        baseline=self.proposer_gen_baseline,
                        device=self.device,
                        ddp_no_sync=self._is_ddp,
                    )

                self.proposer_gen_baseline = (
                    cfg.proposer_gen_baseline_momentum * self.proposer_gen_baseline
                    + (1 - cfg.proposer_gen_baseline_momentum) * proposer_gen_reward
                )
                stats.update({f"prop_gen_{k}": v for k, v in prop_stats.items()})
            elif self._is_ddp and quality_gate_ok:
                stats["prop_gen_skipped_reason"] = "ddp_proposer_gen_gate_mismatch"
            else:
                stats["prop_gen_skipped_reason"] = "low_spec_quality"

        # ── 6. Best image → replay buffer ───────────────────────────────
        best_image = candidates[best_idx][0]

        added = self.replay_buffer.add(
            image=best_image,
            prompt=spec.prompt,
            questions=questions,
            reference_answers=expected_answers,
            reward=selected_total_reward,
            step=step,
        )
        stats["replay_buffer_added"] = added
        stats["replay_buffer_size"] = len(self.replay_buffer)

        # Update reward EMA
        self.reward_ema = (
            cfg.reward_ema_momentum * self.reward_ema
            + (1 - cfg.reward_ema_momentum) * selected_total_reward
        )
        stats["reward_ema"] = self.reward_ema

        return stats

    def _update_generation_health(self, step: int, stats: Dict) -> None:
        """Fail fast when generation pipeline is unhealthy for sustained periods."""
        cfg = self.se_config
        if not bool(getattr(cfg, "generation_failfast_enabled", True)):
            return

        attempted = int(stats.get("generation_attempted", 0) or 0)
        succeeded = int(stats.get("generation_succeeded", 0) or 0)
        success_rate = float(stats.get("generation_success_rate", 1.0))
        skipped = bool(stats.get("g_skipped", False))
        reason = str(stats.get("reason", ""))

        unhealthy = False
        if attempted > 0:
            min_rate = max(0.0, min(1.0, float(getattr(cfg, "generation_failfast_min_success_rate", 0.10))))
            if succeeded <= 0 or success_rate < min_rate:
                unhealthy = True
        elif skipped and reason in {"no_candidates"}:
            unhealthy = True

        self._generation_health_window.append(0.0 if unhealthy else 1.0)
        if unhealthy:
            self._generation_consecutive_unhealthy += 1
        else:
            self._generation_consecutive_unhealthy = 0

        max_consecutive = max(1, int(getattr(cfg, "generation_failfast_consecutive_skips", 5)))
        window_success = (
            float(sum(self._generation_health_window)) / float(len(self._generation_health_window))
            if len(self._generation_health_window) > 0
            else 1.0
        )
        min_rate = max(0.0, min(1.0, float(getattr(cfg, "generation_failfast_min_success_rate", 0.10))))
        if self._generation_consecutive_unhealthy >= max_consecutive:
            raise RuntimeError(
                "[SelfEvolvingTrainer] Generation fail-fast triggered: "
                f"step={step}, consecutive_unhealthy={self._generation_consecutive_unhealthy}, "
                f"window_success_rate={window_success:.3f}, last_reason={reason or 'n/a'}"
            )
        if (
            len(self._generation_health_window) == int(self._generation_health_window.maxlen)
            and window_success < min_rate
        ):
            raise RuntimeError(
                "[SelfEvolvingTrainer] Generation fail-fast triggered by window health: "
                f"step={step}, window={len(self._generation_health_window)}, "
                f"window_success_rate={window_success:.3f}, min_required={min_rate:.3f}"
            )

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
        step_rng = random.Random(int(cfg.seed) + int(step) * 104729)

        # Gen-mix ratio: linearly ramp from start to max
        warmup = max(1, cfg.gen_mix_ratio_warmup_steps)
        ratio = cfg.gen_mix_ratio_start + (
            cfg.gen_mix_ratio_max - cfg.gen_mix_ratio_start
        ) * min(1.0, step / warmup)

        # Try replay buffer first (for generated image mixing)
        if step_rng.random() < ratio and self.replay_buffer:
            entry = self.replay_buffer.sample(rng=step_rng)
            if entry is not None:
                return entry.image, "replay_buffer"

        # ── Image folder mode (preferred when set) ────────────────────
        if self._image_folder_paths:
            path = self._image_folder_paths[step_rng.randrange(len(self._image_folder_paths))]
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
                idx = step_rng.randrange(ds_len)
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
        image_source_hint: str = "",
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
                image_source_hint=image_source_hint,
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

        def _clean_question_text(candidate: str) -> str:
            q = str(candidate or "").replace("\n", " ").strip()
            if not q:
                return ""
            q = re.sub(r"</?[^>]+>", " ", q)
            q = re.sub(
                r"^\s*(?:q(?:uestion)?\s*)?\d+\s*[\).:\-\s]*",
                "",
                q,
                flags=re.IGNORECASE,
            )
            q = re.sub(r"^\s*(?:question|text)\s*[:=-]\s*", "", q, flags=re.IGNORECASE)
            q = re.sub(
                r"^\s*(?:[a-d]|option\s*[a-d]|answer\s*[a-d])\s*[\).:-]\s*",
                "",
                q,
                flags=re.IGNORECASE,
            )
            q = " ".join(q.strip(" \"'`").split())
            if "?" in q:
                q = q[: q.find("?") + 1].strip()
            if not q.endswith("?"):
                return ""
            qn = normalize_answer(q, max_words=0)
            if not qn:
                return ""
            if re.match(r"^(?:a|b|c|d|option [a-d]|answer [a-d])\b", qn):
                return ""
            q_quality = self._question_quality_score(q)
            if (not bool(q_quality.get("valid", False))) or float(q_quality.get("score", 0.0)) < 0.50:
                return ""
            return q

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
            q_text = _clean_question_text(q_text)
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
                q_text = _clean_question_text(str(q).strip())
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
            q_text = _clean_question_text(str(cand.get("question", "")))
            if not q_text:
                continue
            cand["question"] = q_text
            key = normalize_answer(q_text, max_words=0)
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

    @staticmethod
    def _collect_image_payload_candidates(gen_result) -> List[object]:
        """Collect likely image payload objects from heterogeneous model outputs."""
        preferred_keys = (
            "generated_image",
            "image",
            "images",
            "img",
            "output_image",
            "img_list",
        )
        out: List[object] = []
        seen: set = set()

        def _walk(obj) -> None:
            oid = id(obj)
            if oid in seen:
                return
            seen.add(oid)
            if obj is None:
                return
            if torch.is_tensor(obj) or isinstance(obj, Image.Image):
                out.append(obj)
                return
            if isinstance(obj, dict):
                before = len(out)
                for k in preferred_keys:
                    if k in obj:
                        _walk(obj[k])
                # Fallback scan only when preferred keys did not yield candidates.
                if len(out) == before:
                    for idx, v in enumerate(obj.values()):
                        if idx >= 16:
                            break
                        _walk(v)
                return
            if isinstance(obj, (list, tuple)):
                for idx, item in enumerate(obj):
                    if idx >= 16:
                        break
                    _walk(item)
                return
            for k in preferred_keys:
                if hasattr(obj, k):
                    try:
                        _walk(getattr(obj, k))
                    except Exception:
                        continue

        _walk(gen_result)
        return out

    @staticmethod
    def _looks_like_image_tensor(t: torch.Tensor) -> bool:
        if not torch.is_tensor(t):
            return False
        x = t
        if x.ndim == 4:
            if x.shape[0] < 1:
                return False
            x = x[0]
        if x.ndim != 3:
            return False
        c_first = int(x.shape[0]) in (1, 3, 4)
        c_last = int(x.shape[-1]) in (1, 3, 4)
        if not (c_first or c_last):
            return False
        try:
            if c_first:
                h, w = int(x.shape[1]), int(x.shape[2])
            else:
                h, w = int(x.shape[0]), int(x.shape[1])
            ratio = max(h, w) / float(max(1, min(h, w)))
            return ratio <= 6.0
        except Exception:
            return False

    def _extract_generated_image_tensor(self, gen_result) -> Optional[torch.Tensor]:
        """Select a real image tensor from model output robustly."""
        candidates = self._collect_image_payload_candidates(gen_result)
        for cand in candidates:
            if torch.is_tensor(cand) and self._looks_like_image_tensor(cand):
                return cand
            if isinstance(cand, Image.Image):
                # Convert PIL to tensor as a fallback path.
                try:
                    import numpy as np  # local import to avoid hard dependency

                    arr = np.asarray(cand.convert("RGB"), dtype=np.uint8)
                    ten = torch.from_numpy(arr).to(self.device)
                    return ten
                except Exception:
                    continue
        return None

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
                    img_tensor = self._extract_generated_image_tensor(gen_result)

                    if img_tensor is not None:
                        # VARGPT inference path returns uint8 HWC in BGR order.
                        # Convert to RGB for PIL conversion and downstream training.
                        if torch.is_tensor(img_tensor) and img_tensor.dtype == torch.uint8:
                            if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
                                img_tensor = img_tensor[..., [2, 1, 0]].contiguous()
                            elif img_tensor.ndim == 4 and img_tensor.shape[-1] == 3:
                                img_tensor = img_tensor[..., [2, 1, 0]].contiguous()
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

    @staticmethod
    def _content_tokens(text: str) -> List[str]:
        """Return non-function tokens for generic question-quality checks."""
        stop = {
            "a", "an", "the", "this", "that", "these", "those",
            "is", "are", "was", "were", "be", "being", "been",
            "do", "does", "did", "can", "could", "has", "have", "had",
            "what", "which", "who", "where", "when", "how", "many",
            "of", "to", "in", "on", "at", "by", "for", "from", "with",
            "and", "or", "as", "than", "visible", "shown", "seen",
        }
        norm = normalize_answer(str(text or ""), max_words=0)
        return [t for t in re.findall(r"[a-z0-9]+", norm) if t not in stop]

    def _question_quality_score(
        self,
        question: str,
        meta: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        """General rubric score for whether a candidate is trainable."""
        meta = meta or {}
        q = " ".join(str(question or "").replace("\n", " ").strip().split())
        tokens = re.findall(r"[a-z0-9]+", q.lower())
        qn = " ".join(tokens)
        issues: List[str] = []

        format_ok = bool(q) and q.rstrip().endswith("?")
        if (not format_ok) or re.search(r"</?[^>]+>|[{}\[\]|]", q):
            issues.append("format")
            format_ok = False
        if "..." in q or re.search(r"\b[a-z]+_[a-z0-9_]+\b", q.lower()):
            issues.append("artifact")
            format_ok = False

        length_ok = 4 <= len(tokens) <= 24
        if not length_ok:
            issues.append("length")

        question_start_re = (
            r"^(?:what|which|who|where|when|how|is|are|was|were|do|does|did|"
            r"can|could|has|have|will|would|on which|in which|at which|"
            r"under which|above which|behind which|beside which|next to which)\b"
        )
        start_ok = bool(re.match(question_start_re, qn))
        if not start_ok:
            issues.append("question_start")

        grammar_ok = True
        if re.search(r"\b([a-z0-9]+)\s+\1\b", qn):
            grammar_ok = False
        if re.search(r"\b(?:a|an|the)\s+(?:a|an|the)\b", qn):
            grammar_ok = False
        dangling_tokens = {
            "of", "to", "by", "with", "from", "on", "in", "at", "for",
            "or", "and", "the", "a", "an",
        }
        if tokens and tokens[-1] in dangling_tokens:
            grammar_ok = False
        if not grammar_ok:
            issues.append("grammar")

        content = self._content_tokens(q)
        content_ok = bool(content)
        if content_ok and not qn.startswith("how many") and len(set(content)) < 2:
            content_ok = False
        if not content_ok:
            issues.append("content")

        schema_values = [str(v or "").strip() for v in meta.values()]
        has_schema = any(schema_values)
        target = str(meta.get("visual_target", "") or "").strip()
        target_content = self._content_tokens(target)
        target_ok = True
        if has_schema:
            target_ok = bool(target_content) and len(target_content) <= 6
            if target_ok:
                target_ok = bool(set(content).intersection(target_content))
            if not target_ok:
                issues.append("target")

        checks = [format_ok, length_ok, start_ok, grammar_ok, content_ok]
        if has_schema:
            checks.append(target_ok)
        score = sum(1.0 for ok in checks if ok) / float(max(1, len(checks)))
        mandatory_ok = bool(format_ok and length_ok and start_ok and grammar_ok and content_ok)
        return {
            "score": float(score),
            "valid": float(mandatory_ok and score >= 0.50),
            "issues": issues,
        }

    def _judge_visual_question(self, image: Image.Image, question: str) -> Tuple[float, str]:
        """Use the current VLM as a general rubric judge for question quality."""
        prompt = (
            "You are validating a candidate training question for a vision-language model.\n"
            "Answer yes only if the question is grammatical, complete, objective, "
            "answerable from the image alone, and asks about concrete visible evidence. "
            "Answer no if it is malformed, template-like, vague, meta-text, subjective, "
            "or not visually grounded.\n"
            "Return exactly one XML tag: <answer>yes</answer> or <answer>no</answer>.\n"
            f"Candidate question: {question}"
        )
        try:
            chat_text = _build_chat_text(self.processor, image, prompt)
            mm_inputs = _prepare_mm_inputs(
                self.processor, self.device, image, chat_text, model=self.model
            )
            base_model = _unwrap_model(self.model)
            with torch.no_grad():
                gen_ids = base_model.generate(
                    **mm_inputs,
                    max_new_tokens=8,
                    do_sample=False,
                )
            input_len = mm_inputs["input_ids"].shape[1]
            new_ids = gen_ids[0, input_len:]
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            completion = _decode_tokens(tokenizer, new_ids)
            parsed = normalize_answer(_parse_answer(completion) or completion, max_words=4)
            if re.search(r"\byes\b", parsed) and not re.search(r"\bno\b", parsed):
                return 1.0, completion
            if re.search(r"\bno\b", parsed):
                return 0.0, completion
            return 0.5, completion
        except Exception as exc:
            return 0.5, f"judge_error:{type(exc).__name__}"

    def _is_objective_question(self, question: str) -> bool:
        """General objective-question validator."""
        quality = self._question_quality_score(question)
        if float(quality.get("score", 0.0)) < 0.70:
            return False
        q = str(question or "").strip()
        qn = normalize_answer(q, max_words=0)
        if not qn:
            return False
        if re.search(r"\b(why|might|could|likely|opinion|feel|believe|think)\b", qn):
            return False
        if re.search(r"\b(something|anything|stuff|thing)\b", qn):
            return False
        return True

    def _extract_target_from_question(self, question: str) -> str:
        q = normalize_answer(str(question or ""), max_words=40)
        if not q:
            return ""
        patterns = (
            r"(?:of|on|in|behind|beside|near|under|above|left of|right of)\s+the\s+([a-z0-9\- ]+?)(?:\?|$)",
            r"the\s+([a-z0-9\- ]+?)\s+(?:is|are|was|were)\b",
        )
        for pat in patterns:
            m = re.search(pat, q)
            if m:
                t = normalize_answer(m.group(1), max_words=8)
                if t:
                    return t
        tokens = [t for t in re.findall(r"[a-z0-9]+", q) if t]
        if tokens:
            return " ".join(tokens[-2:]) if len(tokens) >= 2 else tokens[-1]
        return ""

    def _compile_question_from_slots(
        self,
        question_text: str,
        meta: Dict[str, str],
    ) -> Tuple[str, bool, str]:
        q = str(question_text or "").replace("\n", " ").strip()
        if not bool(getattr(self.se_config, "proposer_slot_compiler_enabled", True)):
            return q, True, "disabled"
        strict = bool(getattr(self.se_config, "proposer_slot_compiler_strict", True))
        target_from_meta = normalize_answer(str(meta.get("visual_target", "") or ""), max_words=8)
        target = target_from_meta
        if not target:
            target = self._extract_target_from_question(q)
        target = re.sub(r"^(?:a|an|the)\s+", "", target).strip()
        target_invalid = bool(target) and not bool(self._content_tokens(target))

        compiled = q
        qn = normalize_answer(q, max_words=32)
        # Only rewrite the candidate when the proposer supplied an explicit
        # target. Inferred targets can be partial phrases; rewriting with them
        # can damage otherwise valid questions.
        rewrite_target = re.sub(r"^(?:a|an|the)\s+", "", target_from_meta).strip()
        if " or " in qn and rewrite_target:
            # Rewrite forced-choice question into open-ended form.
            if qn.startswith("how many"):
                compiled = f"How many {rewrite_target} are visible?"
            else:
                compiled = f"What is the {rewrite_target}?"
        elif qn.startswith("how many") and rewrite_target:
            compiled = f"How many {rewrite_target} are visible?"

        compiled = compiled.strip()
        if compiled and not compiled.endswith("?"):
            compiled = compiled + "?"

        if strict and (not target):
            return "", False, "target_missing"
        if strict and target_invalid:
            return "", False, "target_invalid"
        compiled_quality = self._question_quality_score(
            compiled,
            {**dict(meta), "visual_target": target},
        )
        if strict and float(compiled_quality.get("score", 0.0)) < float(
            getattr(self.se_config, "proposer_question_structural_min_score", 0.60)
        ):
            return "", False, "non_objective"
        if not compiled:
            return "", False, "empty_compiled"
        return compiled, True, "ok"

    def _extract_forced_choice_options(self, two_answer_test: str) -> Tuple[str, str]:
        raw = normalize_answer(str(two_answer_test or ""), max_words=30)
        if not raw:
            return "", ""
        parts = [p.strip() for p in re.split(r"\s*(?:vs\.?|or|/|,|\||;)\s*", raw) if p.strip()]
        cleaned: List[str] = []
        for p in parts:
            v = re.sub(r"^(?:option|answer)\s*[ab]\s*[:.)-]?\s*", "", p).strip()
            v = re.sub(r"^[ab]\s*[:.)-]\s*", "", v).strip()
            if not v:
                continue
            if v in {"a", "b", "option a", "option b", "answer a", "answer b"}:
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
        v = normalize_answer(str(answer_raw or ""), max_words=16)
        if not v:
            return ""
        if re.search(r"\b(?:option|answer)?\s*a\b", v) and not re.search(r"\b(?:option|answer)?\s*b\b", v):
            return "a"
        if re.search(r"\b(?:option|answer)?\s*b\b", v) and not re.search(r"\b(?:option|answer)?\s*a\b", v):
            return "b"
        a = normalize_answer(option_a, max_words=8)
        b = normalize_answer(option_b, max_words=8)
        if a and a in v and (not b or b not in v):
            return "a"
        if b and b in v and (not a or a not in v):
            return "b"

        vtoks = set(re.findall(r"[a-z0-9]+", v))
        atoks = set(re.findall(r"[a-z0-9]+", a))
        btoks = set(re.findall(r"[a-z0-9]+", b))
        sa = (len(vtoks & atoks) / float(max(1, len(vtoks | atoks)))) if atoks else 0.0
        sb = (len(vtoks & btoks) / float(max(1, len(vtoks | btoks)))) if btoks else 0.0
        if max(sa, sb) < 0.20:
            return ""
        return "a" if sa >= sb else "b"

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

        has_aux_meta = any(
            bool(v)
            for v in (
                visual_target,
                strategy_used,
                reasoning_chain,
                reasoning_domains,
                rationale,
                two_answer_test,
            )
        )
        if not has_aux_meta:
            quality = self._question_quality_score(
                question,
                {"visual_target": self._extract_target_from_question(question)},
            )
            score = float(quality.get("score", 0.0))
            valid = 1.0 if bool(quality.get("valid", False)) and score >= 0.50 else 0.0
            return {"score": score, "valid": valid}

        strategy_ok = 1.0 if strategy_used else 0.0
        target_ok = 1.0 if visual_target else 0.0
        chain_ok = 1.0 if reasoning_chain and ("->" in reasoning_chain or len(reasoning_chain.split()) >= 4) else 0.0
        rationale_ok = 1.0 if len(rationale.split()) >= 6 else 0.0

        domains = [d.strip().lower() for d in reasoning_domains.split(",") if d.strip()]
        min_domains = max(1, int(getattr(self.se_config, "proposer_reasoning_min_domains", 2)))
        domains_ok = 1.0 if len(domains) >= min_domains else 0.0
        non_rel_ok = 1.0
        if bool(getattr(self.se_config, "proposer_reasoning_require_non_relation", True)):
            non_rel_ok = 1.0 if any(d not in {"d1", "relation", "spatial", "d1=relation/spatial"} for d in domains) else 0.0

        two_parts = [p.strip() for p in re.split(r"\s*(?:vs\.?|or|/|,|\||;)\s*", two_answer_test.lower()) if p.strip()]
        two_parts = [p for p in two_parts if p not in {"a", "b", "option a", "option b", "answer a", "answer b"}]
        two_ok = 1.0 if len(two_parts) >= 2 and two_parts[0] != two_parts[1] else 0.0

        # Use fields present in both old and new prompt templates.
        structural_mid = max(target_ok, strategy_ok)
        context_mid = max(chain_ok, rationale_ok)
        score = float(objective + structural_mid + context_mid + domains_ok + non_rel_ok + two_ok) / 6.0
        min_score = max(
            0.0,
            min(1.0, float(getattr(self.se_config, "proposer_certificate_min_score", 0.55))),
        )
        strict_struct = bool(getattr(self.se_config, "proposer_certificate_strict_struct", True))
        valid = 1.0 if score >= min_score else 0.0
        if strict_struct and (
            objective < 0.5
            or two_ok < 0.5
            or structural_mid < 0.5
            or domains_ok < 0.5
            or non_rel_ok < 0.5
        ):
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
    def _difficulty_rank(bucket: str) -> int:
        b = str(bucket or "").strip().lower()
        if b == "hard":
            return 2
        if b == "medium":
            return 1
        return 0

    def _apply_grpo_pairwise_ranking(
        self,
        rewards: List[float],
        buckets: List[str],
    ) -> Tuple[List[float], List[float]]:
        if len(rewards) <= 1:
            return list(rewards), [0.0 for _ in rewards]
        if not bool(getattr(self.se_config, "grpo_pairwise_ranking_enabled", True)):
            return list(rewards), [0.0 for _ in rewards]
        rank_w = max(
            0.0, float(getattr(self.se_config, "grpo_pairwise_ranking_weight", 0.15))
        )
        margin = max(0.0, float(getattr(self.se_config, "grpo_pairwise_margin", 0.10)))
        easy_pen = max(
            0.0, float(getattr(self.se_config, "grpo_pairwise_easy_penalty", 0.12))
        )
        if rank_w <= 0.0:
            return list(rewards), [0.0 for _ in rewards]

        adjusted = [float(r) for r in rewards]
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
            -1e-6, float(getattr(self.se_config, "proposer_easy_reward_floor", -0.35))
        )
        spread = max(
            0.01, float(getattr(self.se_config, "proposer_all_easy_rank_spread", 0.08))
        )
        adjusted = [float(r) for r in rewards]
        deltas = [0.0 for _ in adjusted]
        order = sorted(range(len(adjusted)), key=lambda i: adjusted[i], reverse=True)
        denom = max(1, len(order) - 1)
        for rank, idx in enumerate(order):
            target = easy_floor - spread * (float(rank) / float(denom))
            if adjusted[idx] > target:
                deltas[idx] += (target - adjusted[idx])
            else:
                deltas[idx] += min(0.0, easy_floor - adjusted[idx])
        adjusted = [max(-1.0, min(1.0, r + d)) for r, d in zip(adjusted, deltas)]
        return adjusted, deltas, True

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
        if not self.is_world_process_zero():
            return

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
        with (ckpt_dir / "SAVE_OK").open("w", encoding="utf-8") as f:
            f.write("ok\n")

        # Save model adapters
        try:
            model_dir = ckpt_dir / "model"
            model_ref = _unwrap_model(self.model)
            generator_adapter_name = self._resolve_generator_adapter_name(model_ref)
            if hasattr(model_ref, "save_pretrained"):
                peft_cfg = getattr(model_ref, "peft_config", {})
                selected_adapters: List[str] = []
                if isinstance(peft_cfg, dict) and peft_cfg:
                    preferred = [generator_adapter_name, ROLE_PROPOSER, ROLE_SOLVER]
                    selected_adapters = [name for name in preferred if name in peft_cfg]
                    if not selected_adapters:
                        selected_adapters = [str(k) for k in peft_cfg.keys()]
                    # Preserve order while removing duplicates.
                    selected_adapters = list(dict.fromkeys(selected_adapters))

                save_kwargs = {
                    "safe_serialization": bool(getattr(self.args, "save_safetensors", True)),
                }
                prev_active = getattr(model_ref, "active_adapter", None)
                if isinstance(prev_active, (list, tuple)):
                    prev_active = prev_active[0] if prev_active else None
                try:
                    if hasattr(model_ref, "set_adapter") and generator_adapter_name:
                        model_ref.set_adapter(generator_adapter_name)
                    if selected_adapters:
                        model_ref.save_pretrained(
                            str(model_dir),
                            selected_adapters=selected_adapters,
                            **save_kwargs,
                        )
                    else:
                        model_ref.save_pretrained(str(model_dir), **save_kwargs)
                except TypeError:
                    # Backward compatibility for older PEFT versions.
                    model_ref.save_pretrained(str(model_dir), **save_kwargs)
                finally:
                    if prev_active is not None and hasattr(model_ref, "set_adapter"):
                        try:
                            model_ref.set_adapter(prev_active)
                        except Exception:
                            pass
            else:
                self.save_model(str(model_dir))
            self._save_adapter_manifest(model_dir, generator_adapter_name)
        except Exception as e:
            logger.warning(f"[SelfEvolvingTrainer] Model save failed: {e}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(output_dir, keep=self.se_config.max_checkpoints)
        self._sync_standard_checkpoint_dir(step=step, source_dir=ckpt_dir)

        logger.info(f"[SelfEvolvingTrainer] Saved checkpoint at step {step}")

    def _resolve_generator_adapter_name(self, model_ref: torch.nn.Module) -> str:
        peft_cfg = getattr(model_ref, "peft_config", {})
        if not isinstance(peft_cfg, dict) or not peft_cfg:
            return ROLE_GENERATOR
        if ROLE_GENERATOR in peft_cfg:
            return ROLE_GENERATOR
        for name in peft_cfg.keys():
            s = str(name)
            if s not in {ROLE_PROPOSER, ROLE_SOLVER}:
                return s
        return str(next(iter(peft_cfg.keys())))

    def _save_adapter_manifest(self, model_dir: pathlib.Path, generator_adapter_name: str) -> None:
        """Persist adapter metadata to make resume robust across PEFT versions."""
        model_ref = _unwrap_model(self.model)
        peft_cfg = getattr(model_ref, "peft_config", {})
        adapter_names: List[str] = []
        if isinstance(peft_cfg, dict):
            adapter_names = [str(k) for k in peft_cfg.keys()]

        active_adapter = getattr(model_ref, "active_adapter", None)
        if isinstance(active_adapter, (list, tuple)):
            active_adapter = active_adapter[0] if active_adapter else None

        _json_dump(
            model_dir / "se_adapter_manifest.json",
            {
                "adapters": adapter_names,
                "active_adapter": active_adapter,
                "role_adapters": {
                    "generator": generator_adapter_name,
                    "proposer": ROLE_PROPOSER,
                    "solver": ROLE_SOLVER,
                },
            },
        )

    @staticmethod
    def _parse_checkpoint_step(name: str) -> int:
        if not name.startswith("se_checkpoint_"):
            return -1
        suffix = name.split("_")[-1]
        return int(suffix) if suffix.isdigit() else -1

    def _resolve_se_checkpoint_dir(self, checkpoint_path: str) -> Optional[pathlib.Path]:
        """Resolve resume input into a concrete se_checkpoint_* directory."""
        path = pathlib.Path(checkpoint_path).expanduser()
        if not path.exists():
            return None

        # Passed directly: /.../se_checkpoint_XXXX/se_state.pt
        if path.is_file() and path.name == "se_state.pt":
            return path.parent

        # Passed directly: /.../se_checkpoint_XXXX
        if path.is_dir() and (path / "se_state.pt").exists():
            return path

        # Passed model dir: /.../se_checkpoint_XXXX/model
        if path.is_dir() and path.name == "model" and (path.parent / "se_state.pt").exists():
            return path.parent

        # Passed run dir: /.../output_dir
        if path.is_dir():
            candidates = [
                p for p in path.glob("se_checkpoint_*")
                if p.is_dir() and (p / "se_state.pt").exists()
            ]
            if candidates:
                return max(candidates, key=lambda p: self._parse_checkpoint_step(p.name))
            if path.name == "checkpoints" or (path / "checkpoints").is_dir():
                checkpoint_root = path if path.name == "checkpoints" else (path / "checkpoints")
                alias_candidates = [
                    p for p in checkpoint_root.glob("step_*")
                    if p.is_dir() and (p / "se_state.pt").exists()
                ]
                if alias_candidates:
                    return max(
                        alias_candidates,
                        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
                    )

        return None

    def _resolve_adapter_model_dir(
        self,
        checkpoint_dir: pathlib.Path,
        original_resume_path: pathlib.Path,
    ) -> Optional[pathlib.Path]:
        """Locate adapter directory for a checkpoint."""
        candidate = checkpoint_dir / "model"
        if candidate.is_dir():
            return candidate

        # Allow resume_from pointing directly to model adapter folder.
        if original_resume_path.is_dir() and (
            (original_resume_path / "adapter_config.json").exists()
            or (original_resume_path / "adapter_model.bin").exists()
            or (original_resume_path / "adapter_model.safetensors").exists()
        ):
            return original_resume_path

        return None

    def _load_adapter_checkpoint(self, model_dir: pathlib.Path) -> bool:
        """Best-effort adapter restore (generator/proposer/solver)."""
        model_ref = _unwrap_model(self.model)
        if not hasattr(model_ref, "load_adapter"):
            logger.warning("[SelfEvolvingTrainer] Model has no load_adapter(); skipping adapter restore.")
            return False

        manifest_path = model_dir / "se_adapter_manifest.json"
        adapter_names: List[str] = []
        role_adapters: Dict[str, str] = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                adapter_names = [str(x) for x in manifest.get("adapters", []) if str(x)]
                role_adapters = {
                    str(k): str(v) for k, v in dict(manifest.get("role_adapters", {})).items()
                }
            except Exception as e:
                logger.warning(f"[SelfEvolvingTrainer] Failed to read adapter manifest: {e}")

        # Fallback discovery if manifest is absent.
        generator_name = str(
            role_adapters.get("generator", self._resolve_generator_adapter_name(model_ref))
        )
        if not adapter_names:
            adapter_names = [generator_name, ROLE_PROPOSER, ROLE_SOLVER]

        existing_before: set = set()
        peft_cfg_before = getattr(model_ref, "peft_config", {})
        if isinstance(peft_cfg_before, dict):
            existing_before = {str(k) for k in peft_cfg_before.keys()}

        loaded: List[str] = []
        topology_changed = False
        for adapter_name in adapter_names:
            if adapter_name == generator_name:
                adapter_path = model_dir
            else:
                adapter_path = model_dir / adapter_name

            if not adapter_path.exists():
                continue

            def _try_load() -> None:
                try:
                    model_ref.load_adapter(str(adapter_path), adapter_name, is_trainable=True)
                except TypeError:
                    # Older PEFT signatures may not accept is_trainable.
                    model_ref.load_adapter(str(adapter_path), adapter_name)

            try:
                _try_load()
                if adapter_name not in existing_before:
                    topology_changed = True
                loaded.append(adapter_name)
                continue
            except Exception as first_exc:
                msg = str(first_exc).lower()
                if "already exists" in msg and hasattr(model_ref, "delete_adapter"):
                    try:
                        active = getattr(model_ref, "active_adapter", None)
                        if isinstance(active, (list, tuple)):
                            active = active[0] if active else None
                        if active == adapter_name and hasattr(model_ref, "set_adapter"):
                            peft_cfg = getattr(model_ref, "peft_config", {})
                            if isinstance(peft_cfg, dict):
                                fallback = next((n for n in peft_cfg.keys() if n != adapter_name), None)
                                if fallback is not None:
                                    model_ref.set_adapter(fallback)
                        model_ref.delete_adapter(adapter_name)
                        topology_changed = True
                        _try_load()
                        loaded.append(adapter_name)
                        continue
                    except Exception as retry_exc:
                        logger.warning(
                            f"[SelfEvolvingTrainer] Failed to reload adapter '{adapter_name}' "
                            f"from {adapter_path}: {retry_exc}"
                        )
                        continue

                logger.warning(
                    f"[SelfEvolvingTrainer] Failed to load adapter '{adapter_name}' "
                    f"from {adapter_path}: {first_exc}"
                )

        if hasattr(model_ref, "set_adapter"):
            target = generator_name if generator_name in loaded else (loaded[0] if loaded else None)
            if target is not None:
                try:
                    model_ref.set_adapter(target)
                except Exception as e:
                    logger.warning(f"[SelfEvolvingTrainer] Failed to set active adapter '{target}': {e}")

        if loaded:
            logger.info(
                f"[SelfEvolvingTrainer] Restored adapters from checkpoint: {','.join(loaded)}"
            )
        else:
            logger.warning(
                f"[SelfEvolvingTrainer] No adapter weights restored from {model_dir}; "
                "continuing with current in-memory adapters."
            )
        return topology_changed or bool(loaded)

    def _rebuild_updaters_for_resume(self) -> None:
        """Recreate role updaters so optimizers bind to current adapter params."""
        self.proposer_updater = RolePolicyUpdater(
            model=self.model,
            processor=self.processor,
            config=self.se_config,
            adapter_name=ROLE_PROPOSER,
        )
        self.solver_updater = RolePolicyUpdater(
            model=self.model,
            processor=self.processor,
            config=self.se_config,
            adapter_name=ROLE_SOLVER,
        )
        self.generator_updater = VARImageGenPolicyUpdater(
            model=self.model,
            tokenizer=self.processor,
            config=self.se_config,
        )

    def _load_se_checkpoint(self, checkpoint_path: str) -> int:
        """Load self-evolving checkpoint. Returns the step to resume from."""
        ckpt_dir = self._resolve_se_checkpoint_dir(checkpoint_path)
        if ckpt_dir is None:
            logger.warning(
                f"[SelfEvolvingTrainer] Could not resolve checkpoint directory from: {checkpoint_path}"
            )
            return 0

        resume_path = pathlib.Path(checkpoint_path).expanduser()
        model_dir = self._resolve_adapter_model_dir(ckpt_dir, resume_path)
        adapters_restored = False
        if model_dir is not None:
            adapters_restored = self._load_adapter_checkpoint(model_dir)
        else:
            logger.warning(
                f"[SelfEvolvingTrainer] No model adapter directory found under checkpoint: {ckpt_dir}"
            )

        se_state_path = ckpt_dir / "se_state.pt"
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

            if adapters_restored:
                self._rebuild_updaters_for_resume()
            if "proposer_updater" in state:
                self.proposer_updater.load_state_dict(state["proposer_updater"])
            if "solver_updater" in state:
                self.solver_updater.load_state_dict(state["solver_updater"])
            if "generator_updater" in state:
                self.generator_updater.load_state_dict(state["generator_updater"])

            saved_step = int(state.get("step", 0))
            next_step = max(0, saved_step + 1)
            logger.info(
                f"[SelfEvolvingTrainer] Resumed from {ckpt_dir} "
                f"(saved_step={saved_step}, next_step={next_step})"
            )
            return next_step

        logger.warning(
            f"[SelfEvolvingTrainer] No se_state.pt found at resolved path: {ckpt_dir}"
        )
        return 0

    def _cleanup_old_checkpoints(self, output_dir: pathlib.Path, keep: int = 5):
        """Remove old checkpoints, keeping only the most recent `keep`."""
        ckpt_dirs = sorted(
            output_dir.glob("se_checkpoint_*"),
            key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0,
        )
        while len(ckpt_dirs) > keep:
            old_dir = ckpt_dirs.pop(0)
            try:
                shutil.rmtree(old_dir)
            except Exception:
                pass
        keep_steps = {
            int(p.name.split("_")[-1]) for p in ckpt_dirs if p.name.split("_")[-1].isdigit()
        }
        checkpoint_root = output_dir / "checkpoints"
        if checkpoint_root.is_dir():
            for alias_dir in checkpoint_root.glob("step_*"):
                try:
                    alias_step = int(alias_dir.name.split("_")[-1])
                except Exception:
                    continue
                if alias_step not in keep_steps:
                    self._remove_path(alias_dir)

    def _init_step_log_paths(self, output_dir: pathlib.Path):
        """Initialize run/log directories and JSONL log file paths."""
        self.run_dir = output_dir
        self.logs_dir = output_dir / "logs"
        self.iter_log_path = output_dir / "iter_log.jsonl"
        self.understanding_log_path = self.logs_dir / "understanding_steps.jsonl"
        self.generation_log_path = self.logs_dir / "generation_steps.jsonl"
        self.error_log_path = self.logs_dir / "error_steps.jsonl"
        self.release_rollouts_log_path = output_dir / "rollouts.jsonl"
        self.release_generation_rollouts_log_path = output_dir / "generation_rollouts.jsonl"
        self.metrics_log_path = output_dir / "metrics.jsonl"
        self.status_path = output_dir / "status.json"
        self.summary_path = output_dir / "summary.json"
        self.config_path = output_dir / "config.json"
        self.checkpoint_root = output_dir / "checkpoints"
        if self.is_world_process_zero():
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_root.mkdir(parents=True, exist_ok=True)

    def _to_jsonable(self, value):
        """Convert nested training stats into JSON-safe values."""
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        if isinstance(value, torch.Tensor):
            return {
                "_type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, Image.Image):
            return {
                "_type": "image",
                "size": list(value.size),
                "mode": value.mode,
            }
        return str(value)

    def _append_jsonl(self, path: Optional[pathlib.Path], record: Dict):
        """Append one JSON record to a JSONL log file (main process only)."""
        if path is None or not self.is_world_process_zero():
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(self._to_jsonable(record), ensure_ascii=True) + "\n")
        except Exception as exc:
            logger.warning(f"[SelfEvolvingTrainer] Failed to append log {path}: {exc}")

    def _update_metric(self, name: str, value: float):
        stat = self._metric_stats.setdefault(
            name,
            {"count": 0.0, "sum": 0.0, "sum_sq": 0.0, "min": value, "max": value},
        )
        stat["count"] += 1.0
        stat["sum"] += value
        stat["sum_sq"] += value * value
        stat["min"] = min(stat["min"], value)
        stat["max"] = max(stat["max"], value)

    def _metrics_summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for name, stat in self._metric_stats.items():
            count = max(1.0, stat["count"])
            mean = stat["sum"] / count
            variance = max(0.0, (stat["sum_sq"] / count) - (mean * mean))
            summary[name] = {
                "count": int(stat["count"]),
                "mean": mean,
                "std": math.sqrt(variance),
                "min": stat["min"],
                "max": stat["max"],
            }
        return summary

    @staticmethod
    def _write_json_atomic(path: pathlib.Path, payload: Dict) -> None:
        tmp_path = path.with_name(f"{path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(str(tmp_path), str(path))

    def _progress_core(self, *, step: int, phase: str, run_started_at: float) -> Dict[str, float]:
        now = float(time.time())
        elapsed_sec = max(1e-9, now - float(run_started_at))
        total = max(1, int(self.se_config.total_steps) - int(self.se_config.start_step))
        done = max(0, int(step) - int(self.se_config.start_step) + 1)
        done = min(total, done)
        progress = float(done) / float(total)
        steps_per_sec = float(done) / float(elapsed_sec) if elapsed_sec > 0.0 else 0.0
        remaining = max(0, total - done)
        eta_sec = float(remaining) / float(steps_per_sec) if steps_per_sec > 0.0 else -1.0
        return {
            "step": int(step),
            "phase": str(phase),
            "steps_total": int(self.se_config.total_steps),
            "steps_started_from": int(self.se_config.start_step),
            "steps_done": int(done),
            "steps_remaining": int(remaining),
            "progress": float(progress),
            "elapsed_sec": float(elapsed_sec),
            "steps_per_sec": float(steps_per_sec),
            "eta_sec": float(eta_sec),
            "timestamp_unix": float(now),
            "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

    def _append_metrics(self, record: Dict) -> None:
        self._append_jsonl(self.metrics_log_path, record)

    def _release_metrics(self, *, step_time_sec: float) -> Dict[str, float]:
        summary = self._metrics_summary()

        def _mean(name: str, default: float = 0.0) -> float:
            return float(summary.get(name, {}).get("mean", default))

        return {
            "step_time_sec": float(step_time_sec),
            "understanding_steps": int(self._phase_counts.get("understanding", 0)),
            "generation_steps": int(self._phase_counts.get("generation", 0)),
            "error_steps": int(self._phase_counts.get("error", 0)),
            "proposer_reward": _mean("proposer_reward"),
            "solver_reward": _mean("solver_reward"),
            "entropy": _mean("entropy"),
            "gen_reward_mean": _mean("gen_reward_mean"),
            "gen_reward_max": _mean("gen_reward_max"),
            "generation_success_rate": _mean("generation_success_rate"),
            "proposer_baseline": float(self.proposer_baseline),
            "solver_baseline": float(self.solver_baseline),
            "generator_baseline": float(self.generator_baseline),
            "proposer_gen_baseline": float(self.proposer_gen_baseline),
            "reward_ema": float(self.reward_ema),
            "replay_buffer_size": int(len(self.replay_buffer)),
            "last_checkpoint_dir": str(self.last_checkpoint_dir),
        }

    def _write_status(self, *, state: str, progress: Dict, metrics: Dict, last_error: str = "") -> None:
        if self.status_path is None or not self.is_world_process_zero():
            return
        payload = {
            "state": str(state),
            "output_dir": str(self.run_dir) if self.run_dir is not None else "",
            "summary_path": str(self.summary_path) if self.summary_path is not None else "",
            "rollouts_log_path": str(self.release_rollouts_log_path) if self.release_rollouts_log_path is not None else "",
            "generation_rollouts_log_path": (
                str(self.release_generation_rollouts_log_path)
                if self.release_generation_rollouts_log_path is not None
                else ""
            ),
            "metrics_log_path": str(self.metrics_log_path) if self.metrics_log_path is not None else "",
            "last_error": str(last_error or ""),
            "progress": progress,
            "metrics": metrics,
        }
        self._write_json_atomic(self.status_path, payload)

    @staticmethod
    def _remove_path(path: pathlib.Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.exists():
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _link_or_copy_path(src: pathlib.Path, dst: pathlib.Path) -> None:
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

    def _sync_standard_checkpoint_dir(self, *, step: int, source_dir: pathlib.Path) -> pathlib.Path:
        if self.checkpoint_root is None:
            raise RuntimeError("checkpoint_root is not initialized")
        alias_dir = self.checkpoint_root / f"step_{int(step):06d}"
        tmp_dir = self.checkpoint_root / f".{alias_dir.name}.tmp"
        self._remove_path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for child in source_dir.iterdir():
            self._link_or_copy_path(child, tmp_dir / child.name)
        _json_dump(
            tmp_dir / "checkpoint_target.json",
            {
                "step": int(step),
                "source_checkpoint_dir": str(source_dir),
            },
        )
        self._remove_path(alias_dir)
        os.replace(str(tmp_dir), str(alias_dir))
        self.last_checkpoint_dir = str(alias_dir)
        with (self.checkpoint_root / "latest.txt").open("w", encoding="utf-8") as f:
            f.write(str(alias_dir) + "\n")
        return alias_dir

    # ── Logging ─────────────────────────────────────────────────────────

    def _log_step(self, step: int, stats: Dict):
        """Log step statistics."""
        if not self.is_world_process_zero():
            return

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

        # BLIP3o-style concise step lines to stdout for easy terminal tracking.
        if phase == "understanding":
            num_answers = int(stats.get("num_answers", 0))
            maj_frac = float(stats.get("majority_frac", 0.0))
            maj_count = int(round(maj_frac * num_answers)) if num_answers > 0 else 0
            entropy = float(stats.get("entropy", 0.0))
            margin = float(stats.get("margin", 0.0))
            info_local = 1 if (not bool(stats.get("easy_question", True))) else 0
            debt = float(stats.get("proposer_hardness_debt", 0.0))
            warm = 1 if bool(stats.get("proposer_warm_start_active", False)) else 0
            q_text = str(stats.get("question", "")).strip()
            print(
                f"[Step {step:05d}][U] maj={maj_count}/{num_answers} "
                f"maj_frac={maj_frac:.2f} H={entropy:.3f} M={margin:.3f} "
                f"info_local={info_local} P_R={float(stats.get('proposer_reward', 0.0)):.3f} "
                f"S_R={float(stats.get('solver_reward', 0.0)):.3f} "
                f"debt={debt:.2f} warm={warm} dt={step_time:.1f}s",
                flush=True,
            )
            if q_text:
                print(f"  Q: {q_text}", flush=True)
        elif phase == "generation":
            print(
                f"[Step {step:05d}][G] K={int(stats.get('generation_attempted', 0))} "
                f"ok={int(stats.get('generation_succeeded', 0))} "
                f"fail={int(stats.get('generation_failures', 0))} "
                f"succ={float(stats.get('generation_success_rate', 0.0)):.2f} "
                f"R_mean={float(stats.get('gen_reward_mean', 0.0)):.3f} "
                f"R_max={float(stats.get('gen_reward_max', 0.0)):.3f} "
                f"replay={int(stats.get('replay_buffer_size', 0))} "
                f"dt={step_time:.1f}s",
                flush=True,
            )
        elif phase == "error":
            print(
                f"[Step {step:05d}][E] error={str(stats.get('error', 'unknown'))} "
                f"dt={step_time:.1f}s",
                flush=True,
            )

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
