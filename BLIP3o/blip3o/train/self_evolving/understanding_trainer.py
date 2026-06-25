"""
Understanding-only self-evolving trainer.

Ported from self_evolving/experiments/understanding.py.
Uses native BLIP3o model loading instead of _load_model_with_fallback().
"""

import dataclasses
import datetime as dt
import gc
import json
import math
from collections import deque
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
from transformers import AutoProcessor

from .config import UnderstandingSelfEvolvingConfig
from .checkpoint_adapters import sanitize_peft_adapter_dir
from .image_pool import ImagePool, ImagePoolConfig
from .model_api import _load_blip3o_model
from .policy_updater import RolePolicyUpdater
from .prompts import (
    build_proposer_multi_prompt,
    build_solver_prompt,
)
from .utils import (
    HAS_PEFT,
    HAS_WANDB,
    _append_training_monitor_record,
    _append_training_watch_record,
    _build_chat_text,
    _collect_git_info,
    _decode_tokens,
    _infer_primary_device,
    _json_dump,
    _parse_all_questions,
    _parse_answer,
    _parse_first_question,
    _prepare_mm_inputs,
    _resolve_attn_implementation,
    _safe_dtype,
    _save_code_run_registry,
    _set_global_seed,
    gaussian_reward,
    majority_vote,
    normalize_answer,
    pre_answer_word_count,
    shannon_entropy_nats,
    strip_tags,
    use_adapter,
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
_VISUAL_BRIDGE_TARGET_MARKERS = ("mm_projector", "visual.merger", "merger.mlp")


def _target_tuple(value) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def _dedupe_targets(*groups: Tuple[str, ...]) -> Tuple[str, ...]:
    seen = set()
    merged = []
    for group in groups:
        for target in group:
            if target and target not in seen:
                seen.add(target)
                merged.append(target)
    return tuple(merged)


def _text_only_lora_targets(targets: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(
        target
        for target in targets
        if not any(marker in target for marker in _VISUAL_BRIDGE_TARGET_MARKERS)
    )
_QUESTION_START_RE = re.compile(
    r"^(?:what|which|how|where|when|who|is|are|was|were|does|do|did|can|could|should|would|has|have|had)\b",
    flags=re.IGNORECASE,
)


def _quantile(values: List[float], q: float) -> float:
    """Linear-interpolated quantile over a pre-sorted list."""
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


class UnderstandingSelfEvolvingTrainer:
    """Understanding-only self-evolving trainer with native BLIP3o loading."""

    # -------------------------------------------------------------------
    # Distributed helpers
    # -------------------------------------------------------------------
    def _setup_distributed(self):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.cfg.cuda_device)))
        self.distributed = self.world_size > 1
        self.is_main_process = self.rank == 0

        if self.distributed:
            if not dist.is_available():
                raise RuntimeError("torch.distributed is not available in this environment.")
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                backend = "nccl"
            else:
                backend = "gloo"
            if not dist.is_initialized():
                init_kwargs = {"backend": backend, "init_method": "env://"}
                if backend == "nccl":
                    init_kwargs["device_id"] = self.local_rank
                try:
                    dist.init_process_group(**init_kwargs)
                except TypeError:
                    init_kwargs.pop("device_id", None)
                    dist.init_process_group(**init_kwargs)
            self.is_main_process = dist.get_rank() == 0
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            print(
                f"[DDP] Initialized rank={self.rank}/{self.world_size} "
                f"local_rank={self.local_rank} backend={backend}"
            )
        elif torch.cuda.is_available():
            torch.cuda.set_device(self.cfg.cuda_device)

    def _dist_barrier(self):
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def _dist_mean(self, value: float) -> float:
        if not (self.distributed and dist.is_initialized()):
            return float(value)
        dev = (
            torch.device(f"cuda:{self.local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        tensor = torch.tensor([float(value)], dtype=torch.float64, device=dev)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float((tensor / float(self.world_size)).item())

    def _dist_all_bool(self, value: bool) -> bool:
        if not (self.distributed and dist.is_initialized()):
            return bool(value)
        dev = (
            torch.device(f"cuda:{self.local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        tensor = torch.tensor([1 if value else 0], dtype=torch.int32, device=dev)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return bool(int(tensor.item()) == 1)

    def _dist_any_bool(self, value: bool) -> bool:
        if not (self.distributed and dist.is_initialized()):
            return bool(value)
        dev = (
            torch.device(f"cuda:{self.local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        tensor = torch.tensor([1 if value else 0], dtype=torch.int32, device=dev)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return bool(int(tensor.item()) == 1)

    def _sync_state_scalars(self):
        if not (self.distributed and dist.is_initialized()):
            return
        self.solver_baseline = self._dist_mean(self.solver_baseline)
        self.proposer_baseline = self._dist_mean(self.proposer_baseline)
        self.proposer_entropy_mu_ema = self._dist_mean(self.proposer_entropy_mu_ema)
        self.solver_updater.kl_coef = self._dist_mean(self.solver_updater.kl_coef)
        self.proposer_updater.kl_coef = self._dist_mean(self.proposer_updater.kl_coef)

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

    def _solver_top_p_schedule(self) -> List[float]:
        n = max(1, int(self.cfg.num_solver_samples))
        base = float(self.cfg.top_p)
        if n <= 1 or not bool(getattr(self.cfg, "solver_use_temperature_mix", True)):
            return [base] * n
        pmin = float(getattr(self.cfg, "solver_top_p_min", base))
        pmax = float(getattr(self.cfg, "solver_top_p_max", base))
        if pmin > pmax:
            pmin, pmax = pmax, pmin
        pmin = max(1e-6, min(1.0, pmin))
        pmax = max(1e-6, min(1.0, pmax))
        if abs(pmax - pmin) < 1e-8:
            return [pmin] * n
        return [pmin + (pmax - pmin) * (float(i) / float(n - 1)) for i in range(n)]

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

    # -------------------------------------------------------------------
    # Init
    # -------------------------------------------------------------------
    def __init__(self, config: UnderstandingSelfEvolvingConfig):
        self.cfg = config
        self._setup_distributed()
        _set_global_seed(config.seed + self.rank, deterministic=config.deterministic)

        if not config.data_dir:
            raise ValueError("`data_dir` is required for understanding self-evolving training")

        self.run_dir = self._build_run_dir()
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.iter_log_path = self.run_dir / "iter_log.jsonl"
        self.questions_log_path = self.logs_dir / "questions.jsonl"
        self.rollouts_log_path = self.logs_dir / "solver_rollouts.jsonl"
        self.rewards_log_path = self.logs_dir / "rewards.jsonl"
        self.policy_updates_log_path = self.logs_dir / "policy_updates.jsonl"
        self.summary_path = self.run_dir / "ablation_summary.json"
        self.release_rollouts_log_path = self.run_dir / "rollouts.jsonl"
        self.metrics_log_path = self.run_dir / "metrics.jsonl"
        self.monitor_log_path = self.logs_dir / "training_monitor.jsonl"
        self.monitor_tsv_path = self.logs_dir / "training_monitor.tsv"
        self.watch_log_path = self.logs_dir / "training_watch.log"
        self.status_path = self.run_dir / "status.json"
        self.release_summary_path = self.run_dir / "summary.json"
        self.checkpoint_root = self.run_dir / "checkpoints"
        self.last_checkpoint_dir = ""
        self.code_run_registry_entry = ""
        self._save_run_metadata()

        self.model, self.processor = self._load_model()
        fallback_dev = self.local_rank if self.distributed else config.cuda_device
        self.device = _infer_primary_device(self.model, fallback_cuda_device=fallback_dev)

        pool_cfg = ImagePoolConfig(
            data_dir=config.data_dir,
            include_subfolders=list(config.include_subfolders) if config.include_subfolders else None,
            split=None if config.data_split == "all" else config.data_split,
            prefer_manifest=False,
            max_images=config.max_images,
            seed=config.seed,
        )
        self.pool = ImagePool(pool_cfg)

        reference_model = None
        if not config.use_lora:
            reference_model = _load_blip3o_model(
                config.model_name,
                torch_dtype=_safe_dtype(config.dtype),
                device_map={"": fallback_dev} if self.device.type == "cuda" else "cpu",
                attn_implementation=_resolve_attn_implementation(config.attn_implementation),
            )
            reference_model.eval()
            for p in reference_model.parameters():
                p.requires_grad_(False)

        self.train_model = self.model
        if self.distributed:
            ddp_kwargs = {
                "find_unused_parameters": True,
                # Reuse DDP bucket storage for gradients to lower peak memory.
                "gradient_as_bucket_view": True,
            }
            if torch.cuda.is_available():
                ddp_kwargs["device_ids"] = [self.local_rank]
                ddp_kwargs["output_device"] = self.local_rank
            self.train_model = torch.nn.parallel.DistributedDataParallel(
                self.model, **ddp_kwargs
            )

        self.solver_updater = RolePolicyUpdater(
            model=self.train_model,
            processor=self.processor,
            config=config,
            adapter_name="default" if config.use_lora else None,
            reference_model=reference_model,
        )
        self.proposer_updater = RolePolicyUpdater(
            model=self.train_model,
            processor=self.processor,
            config=config,
            adapter_name="proposer" if config.use_lora else None,
            reference_model=reference_model,
        )

        self.solver_baseline = 0.0
        self.proposer_baseline = 0.0
        self.proposer_entropy_mu_ema = float(config.prop_entropy_mu)
        self._init_adaptive_windows()
        self.start_step = max(0, int(config.start_step))

        self._metric_stats: Dict[str, Dict[str, float]] = {}
        self._policy_update_counts: Dict[str, int] = {"solver": 0, "proposer": 0}
        self.wandb_run = self._init_wandb()

        loaded_resume_step = self._maybe_resume_state()
        if loaded_resume_step is not None:
            self.start_step = max(self.start_step, int(loaded_resume_step))

    # -------------------------------------------------------------------
    # W&B
    # -------------------------------------------------------------------
    def _init_wandb(self):
        if not self.is_main_process:
            return None
        mode = (self.cfg.wandb_mode or "disabled").strip().lower()
        if mode == "disabled":
            return None
        if not HAS_WANDB:
            print("[W&B] wandb package not available; disabling W&B logging.")
            return None

        import wandb

        token = os.environ.get("WANDB_API_KEY", "").strip()
        if token:
            try:
                wandb.login(key=token, relogin=False)
            except Exception as exc:
                print(f"[W&B] login failed using WANDB_API_KEY: {exc}")

        run_name = self.cfg.wandb_run_name or self.cfg.run_name or self.run_dir.name
        kwargs = {
            "project": self.cfg.wandb_project,
            "name": run_name,
            "mode": mode,
            "config": dataclasses.asdict(self.cfg),
            "dir": str(self.run_dir),
        }
        if self.cfg.wandb_entity:
            kwargs["entity"] = self.cfg.wandb_entity
        try:
            run = wandb.init(**kwargs)
            print(f"[W&B] Initialized run: {run_name} (mode={mode})")
            return run
        except Exception as exc:
            print(f"[W&B] init failed; continuing without W&B: {exc}")
            return None

    # -------------------------------------------------------------------
    # Run directory + metadata
    # -------------------------------------------------------------------
    def _build_run_dir(self) -> pathlib.Path:
        base_dir = pathlib.Path(self.cfg.output_dir).expanduser().resolve()
        if self.distributed and dist.is_initialized():
            obj = [None]
            if self.is_main_process:
                timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                run_name = self.cfg.run_name or f"{self.cfg.experiment_name}_{timestamp}"
                run_dir = base_dir / run_name
                if run_dir.exists() and any(run_dir.iterdir()) and not self.cfg.resume_from:
                    run_dir = base_dir / f"{run_name}_{timestamp}"
                run_dir.mkdir(parents=True, exist_ok=True)
                obj[0] = str(run_dir)
            dist.broadcast_object_list(obj, src=0)
            run_dir = pathlib.Path(obj[0]).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            self._dist_barrier()
            return run_dir

        timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_name = self.cfg.run_name or f"{self.cfg.experiment_name}_{timestamp}"
        run_dir = base_dir / run_name
        if run_dir.exists() and any(run_dir.iterdir()) and not self.cfg.resume_from:
            run_dir = base_dir / f"{run_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _save_run_metadata(self):
        if not self.is_main_process:
            self._dist_barrier()
            return
        repo_root = pathlib.Path(__file__).resolve().parents[4]
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        config_payload = dataclasses.asdict(self.cfg)
        git_payload = _collect_git_info(repo_root)
        env_payload = {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "rank": self.rank,
            "world_size": self.world_size,
            "distributed": self.distributed,
        }
        _json_dump(self.run_dir / "config.json", config_payload)
        _json_dump(self.run_dir / "git_info.json", git_payload)
        _json_dump(self.run_dir / "environment.json", env_payload)
        if bool(getattr(self.cfg, "code_run_registry_enabled", True)):
            try:
                registry_entry = _save_code_run_registry(
                    run_dir=self.run_dir,
                    config=config_payload,
                    git_info=git_payload,
                    environment=env_payload,
                    registry_dir=getattr(self.cfg, "code_run_registry_dir", None),
                )
                self.code_run_registry_entry = str(registry_entry)
            except Exception as exc:
                self.code_run_registry_entry = ""
                print(f"[Understanding] WARNING: failed to write code run registry: {exc}")
        self._dist_barrier()

    # -------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------
    def _append_jsonl(self, path: pathlib.Path, record: Dict):
        if not self.is_main_process:
            return
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _monitor_float(value, default=None):
        try:
            val = float(value)
        except Exception:
            return default
        return val if math.isfinite(val) else default

    @classmethod
    def _monitor_stat(cls, stats, key: str, default=None):
        if not isinstance(stats, dict):
            return default
        return cls._monitor_float(stats.get(key), default=default)

    @staticmethod
    def _monitor_bool(stats, key: str, default=False) -> bool:
        if not isinstance(stats, dict) or key not in stats:
            return bool(default)
        return bool(stats.get(key))

    @staticmethod
    def _monitor_skip(stats) -> str:
        if isinstance(stats, dict):
            reason = stats.get("skipped_reason")
            if reason:
                return str(reason)
        return ""

    @staticmethod
    def _monitor_nonfinite_fields(record: Dict) -> List[str]:
        fields: List[str] = []

        def visit(prefix: str, value):
            if isinstance(value, float):
                if not math.isfinite(value):
                    fields.append(prefix)
            elif isinstance(value, dict):
                for k, v in value.items():
                    visit(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(value, (list, tuple)):
                for idx, v in enumerate(value[:16]):
                    visit(f"{prefix}[{idx}]", v)

        visit("", record)
        return fields

    @staticmethod
    def _finite_mean_from_stats(stats_list: List[Dict], key: str) -> Optional[float]:
        vals: List[float] = []
        for stats in stats_list:
            if not isinstance(stats, dict):
                continue
            try:
                val = float(stats.get(key))
            except Exception:
                continue
            if math.isfinite(val):
                vals.append(val)
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def _append_training_monitor(self, record: Dict):
        if not self.is_main_process:
            return
        payload = dict(record)
        nonfinite_fields = self._monitor_nonfinite_fields(payload)
        prior_fields = [
            part.strip()
            for part in str(payload.get("nonfinite_fields", "") or "").split(",")
            if part.strip()
        ]
        merged_fields = prior_fields + [f for f in nonfinite_fields if f not in prior_fields]
        payload["nan_detected"] = bool(payload.get("nan_detected")) or bool(merged_fields)
        payload["nonfinite_fields"] = ",".join(merged_fields[:16])
        if not payload.get("health"):
            if merged_fields:
                payload["health"] = "nonfinite_detected"
            elif any(payload.get(k) for k in ("proposer_skip", "solver_skip")):
                payload["health"] = "skipped_or_waiting"
            elif any(bool(payload.get(k)) for k in ("proposer_did_step", "solver_did_step")):
                payload["health"] = "optimizer_step"
            else:
                payload["health"] = "observed"
        _append_training_monitor_record(self.monitor_log_path, self.monitor_tsv_path, payload)
        _append_training_watch_record(self.watch_log_path, payload)

    def _monitor_understanding_record(self, record: Dict):
        solver_stats = record.get("solver_stats_per_sample") or record.get("solver_stats") or []
        if not isinstance(solver_stats, list):
            solver_stats = []
        proposer_stats = record.get("proposer_stats")
        solver_skip_reasons = sorted(
            {
                str(s.get("skipped_reason"))
                for s in solver_stats
                if isinstance(s, dict) and s.get("skipped_reason")
            }
        )
        solver_did_step = any(
            bool(s.get("did_step", False)) for s in solver_stats if isinstance(s, dict)
        )
        forced_tail = max(
            [self._monitor_float(s.get("forced_tail_tokens"), 0.0) or 0.0 for s in solver_stats if isinstance(s, dict)]
            + [self._monitor_stat(proposer_stats, "forced_tail_tokens", 0.0) or 0.0]
        )
        raw_nonfinite = self._monitor_nonfinite_fields(
            {
                "solver_stats": solver_stats,
                "proposer_stats": proposer_stats,
            }
        )
        self._append_training_monitor(
            {
                "step": int(record.get("step", 0)),
                "phase": "understanding",
                "image_path": record.get("image_path"),
                "solver_reward_raw_mean": self._monitor_float(record.get("solver_rewards_raw_mean")),
                "solver_reward_soft_mean": self._monitor_float(record.get("solver_rewards_soft_mean")),
                "proposer_reward": self._monitor_float(record.get("proposer_reward")),
                "entropy_nats": self._monitor_float(record.get("entropy_nats")),
                "majority_fraction": self._monitor_float(record.get("majority_fraction")),
                "proposer_did_step": self._monitor_bool(proposer_stats, "did_step"),
                "proposer_skip": self._monitor_skip(proposer_stats),
                "proposer_ce_loss": self._monitor_stat(proposer_stats, "ce_loss"),
                "proposer_kl_loss": self._monitor_stat(proposer_stats, "kl_loss"),
                "proposer_total_loss": self._monitor_stat(proposer_stats, "total_loss"),
                "proposer_valid_tokens": self._monitor_stat(proposer_stats, "valid_token_count"),
                "proposer_kl_coef": self._monitor_stat(proposer_stats, "kl_coef_after", record.get("proposer_kl_coef")),
                "solver_did_step": solver_did_step,
                "solver_skip": ",".join(solver_skip_reasons),
                "solver_ce_loss": self._finite_mean_from_stats(solver_stats, "ce_loss"),
                "solver_kl_loss": self._finite_mean_from_stats(solver_stats, "kl_loss"),
                "solver_total_loss": self._finite_mean_from_stats(solver_stats, "total_loss"),
                "solver_valid_tokens": self._finite_mean_from_stats(solver_stats, "valid_token_count"),
                "solver_kl_coef": self._monitor_float(record.get("solver_kl_coef"), getattr(self.solver_updater, "kl_coef", None)),
                "forced_tail_tokens": forced_tail,
                "proposer_baseline": self._monitor_float(record.get("proposer_baseline_after", self.proposer_baseline)),
                "solver_baseline": self._monitor_float(record.get("solver_baseline_after", self.solver_baseline)),
                "step_duration_sec": self._monitor_float(record.get("step_duration_sec")),
                "nan_detected": bool(raw_nonfinite),
                "nonfinite_fields": ",".join(raw_nonfinite[:16]),
            }
        )

    def _update_metric(self, name: str, value: float):
        try:
            value = float(value)
        except Exception:
            return
        if not math.isfinite(value):
            return
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
    def _write_json_atomic(path: pathlib.Path, payload: Dict):
        tmp_path = path.with_name(f"{path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(str(tmp_path), str(path))

    def _progress_core(self, *, step: int, phase: str, run_started_at: float) -> Dict[str, float]:
        now = float(time.time())
        elapsed_sec = max(1e-9, now - float(run_started_at))
        total = max(1, int(self.cfg.total_steps) - int(self.start_step))
        done = max(0, int(step) - int(self.start_step))
        done = min(total, done)
        progress = float(done) / float(total)
        steps_per_sec = float(done) / float(elapsed_sec) if elapsed_sec > 0.0 else 0.0
        remaining = max(0, total - done)
        eta_sec = float(remaining) / float(steps_per_sec) if steps_per_sec > 0.0 else -1.0
        return {
            "step": int(step),
            "phase": str(phase),
            "steps_total": int(self.cfg.total_steps),
            "steps_started_from": int(self.start_step),
            "steps_done": int(done),
            "steps_remaining": int(remaining),
            "progress": float(progress),
            "elapsed_sec": float(elapsed_sec),
            "steps_per_sec": float(steps_per_sec),
            "eta_sec": float(eta_sec),
            "timestamp_unix": float(now),
            "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

    def _append_metrics(self, record: Dict):
        self._append_jsonl(self.metrics_log_path, record)

    def _release_metrics(self, *, step_time_sec: float) -> Dict[str, float]:
        summary = self._metrics_summary()

        def _mean(name: str, default: float = 0.0) -> float:
            return float(summary.get(name, {}).get("mean", default))

        return {
            "step_time_sec": float(step_time_sec),
            "solver_reward_raw_mean": _mean("solver_reward_raw_mean"),
            "solver_reward_soft_mean": _mean("solver_reward_soft_mean"),
            "proposer_reward": _mean("proposer_reward"),
            "entropy_nats": _mean("entropy_nats"),
            "majority_fraction": _mean("majority_fraction"),
            "pre_answer_words_mean": _mean("pre_answer_words_mean"),
            "solver_kl_coef": float(self.solver_updater.kl_coef),
            "proposer_kl_coef": float(self.proposer_updater.kl_coef),
            "solver_baseline": float(self.solver_baseline),
            "proposer_baseline": float(self.proposer_baseline),
            "last_checkpoint_dir": str(self.last_checkpoint_dir),
        }

    def _write_status(self, *, state: str, progress: Dict, metrics: Dict, last_error: str = ""):
        if not self.is_main_process:
            return
        payload = {
            "state": str(state),
            "output_dir": str(self.run_dir),
            "summary_path": str(self.release_summary_path),
            "rollouts_log_path": str(self.release_rollouts_log_path),
            "metrics_log_path": str(self.metrics_log_path),
            "training_monitor_log_path": str(self.monitor_log_path),
            "training_monitor_tsv_path": str(self.monitor_tsv_path),
            "training_watch_log_path": str(self.watch_log_path),
            "code_run_registry_entry": str(getattr(self, "code_run_registry_entry", "")),
            "last_error": str(last_error or ""),
            "progress": progress,
            "metrics": metrics,
        }
        self._write_json_atomic(self.status_path, payload)

    @staticmethod
    def _remove_path(path: pathlib.Path):
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.exists():
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _link_or_copy_path(src: pathlib.Path, dst: pathlib.Path):
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

    # -------------------------------------------------------------------
    # Resume / checkpoint
    # -------------------------------------------------------------------
    def _resolve_resume_dir(self) -> Optional[pathlib.Path]:
        if not self.cfg.resume_from:
            return None
        candidate = pathlib.Path(self.cfg.resume_from).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"resume_from path does not exist: {candidate}")

        if candidate.is_dir() and (candidate / "trainer_state.pt").exists():
            return candidate
        if candidate.is_file() and candidate.name == "trainer_state.pt":
            return candidate.parent
        if candidate.is_dir() and candidate.name.startswith("step_"):
            return candidate

        step_dirs = [
            p
            for p in candidate.glob("step_*")
            if p.is_dir() and (p / "trainer_state.pt").exists()
        ]
        if not step_dirs:
            raise FileNotFoundError(
                f"No checkpoint with trainer_state.pt found under resume_from path: {candidate}"
            )
        return sorted(step_dirs, key=lambda p: p.name)[-1]

    def _maybe_resume_state(self) -> Optional[int]:
        resume_dir = self._resolve_resume_dir()
        if resume_dir is None:
            return None

        state_path = resume_dir / "trainer_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(
                f"trainer_state.pt not found in resume checkpoint: {resume_dir}"
            )

        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")
        if "solver_updater" in state:
            self.solver_updater.load_state_dict(state["solver_updater"])
        elif "solver_opt" in state:
            self.solver_updater.load_state_dict(
                {
                    "optimizer": state["solver_opt"],
                    "kl_coef": state.get("solver_kl_coef"),
                    "step_id": state.get("solver_updater_step", state.get("step", 0)),
                }
            )

        if "proposer_updater" in state:
            self.proposer_updater.load_state_dict(state["proposer_updater"])
        elif "proposer_opt" in state:
            self.proposer_updater.load_state_dict(
                {
                    "optimizer": state["proposer_opt"],
                    "kl_coef": state.get("proposer_kl_coef"),
                    "step_id": state.get("proposer_updater_step", state.get("step", 0)),
                }
            )

        self.solver_baseline = float(state.get("solver_baseline", self.solver_baseline))
        self.proposer_baseline = float(state.get("proposer_baseline", self.proposer_baseline))
        self.proposer_entropy_mu_ema = float(
            state.get("proposer_entropy_mu_ema", self.proposer_entropy_mu_ema)
        )
        entropy_window = state.get("entropy_window")
        if isinstance(entropy_window, list):
            self._entropy_window.clear()
            max_keep = int(self._entropy_window.maxlen or len(entropy_window))
            for value in entropy_window[-max_keep:]:
                try:
                    self._entropy_window.append(float(value))
                except Exception:
                    continue
        difficulty_window = state.get("difficulty_window")
        if isinstance(difficulty_window, list):
            self._difficulty_window.clear()
            max_keep = int(self._difficulty_window.maxlen or len(difficulty_window))
            for bucket in difficulty_window[-max_keep:]:
                b = str(bucket).strip().lower()
                if b in {"easy", "medium", "hard"}:
                    self._difficulty_window.append(b)
        # Allow a poisoned/locked proposer baseline to be reset on resume
        # (e.g. after the baseline-clamp bug fix). When set, also clears the
        # entropy and difficulty history windows so the IQR filter re-warms
        # from scratch rather than staying locked at IQR=0 from a stale run.
        if bool(getattr(self.cfg, "reset_proposer_baseline", False)):
            if self.is_main_process:
                print(
                    f"[Understanding] reset_proposer_baseline=True: resetting proposer_baseline "
                    f"{self.proposer_baseline:.4f} → 0.0, clearing entropy/difficulty windows"
                )
            self.proposer_baseline = 0.0
            self._entropy_window.clear()
            self._difficulty_window.clear()

        py_state = state.get("py_random_state")
        if py_state is not None:
            random.setstate(py_state)
        torch_state = state.get("torch_rng_state")
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        cuda_states = state.get("torch_cuda_rng_state_all")
        if torch.cuda.is_available() and cuda_states is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass

        restored_step = int(state.get("step", 0))
        if self.is_main_process:
            print(f"[Understanding] Resumed trainer state from: {state_path} (step={restored_step})")
            _json_dump(
                self.run_dir / "resume_info.json",
                {
                    "resume_from": str(resume_dir),
                    "restored_step": restored_step,
                    "restored_solver_baseline": self.solver_baseline,
                    "restored_proposer_baseline": self.proposer_baseline,
                    "restored_proposer_entropy_mu_ema": self.proposer_entropy_mu_ema,
                },
            )
        self._dist_barrier()
        return restored_step

    def _trainer_state_dict(self, step: int) -> Dict:
        state = {
            "step": int(step),
            "solver_updater": self.solver_updater.state_dict(),
            "proposer_updater": self.proposer_updater.state_dict(),
            "solver_baseline": float(self.solver_baseline),
            "proposer_baseline": float(self.proposer_baseline),
            "proposer_entropy_mu_ema": float(self.proposer_entropy_mu_ema),
            "entropy_window": list(self._entropy_window),
            "difficulty_window": list(self._difficulty_window),
            "py_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        return state

    def _is_complete_checkpoint(self, step_dir: pathlib.Path) -> bool:
        if not step_dir.is_dir():
            return False
        if not (step_dir / "SAVE_OK").exists():
            return False
        if self.cfg.use_lora:
            return (step_dir / "solver").is_dir() and (step_dir / "proposer").is_dir()
        return (step_dir / "model").is_dir()

    def _list_complete_checkpoints(self) -> List[pathlib.Path]:
        checkpoints = [
            p for p in self.run_dir.glob("step_*") if self._is_complete_checkpoint(p)
        ]
        return sorted(checkpoints, key=lambda p: p.name)

    # -------------------------------------------------------------------
    # Model loading (native BLIP3o)
    # -------------------------------------------------------------------
    def _load_model(self):
        if self.cfg.use_lora and not HAS_PEFT:
            raise RuntimeError("PEFT is required for role-specific LoRA adapters")

        dtype = _safe_dtype(self.cfg.dtype)
        attn_impl = _resolve_attn_implementation(self.cfg.attn_implementation)

        if self.distributed:
            if self.cfg.device_map == "auto" and self.is_main_process:
                print(
                    "[Understanding] Distributed run detected; overriding device_map=auto "
                    "to per-rank single-device mapping."
                )
            device_map = {"": self.local_rank} if torch.cuda.is_available() else "cpu"
        elif self.cfg.device_map == "single":
            device_map = {"": self.cfg.cuda_device} if torch.cuda.is_available() else "cpu"
        elif self.cfg.device_map == "cpu":
            device_map = "cpu"
        else:
            device_map = "auto"

        processor = AutoProcessor.from_pretrained(self.cfg.model_name, trust_remote_code=True)
        model = _load_blip3o_model(
            self.cfg.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
            load_in_4bit=bool(getattr(self.cfg, "load_in_4bit", False)),
            bnb_4bit_quant_type=str(getattr(self.cfg, "bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(getattr(self.cfg, "bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=str(getattr(self.cfg, "bnb_4bit_compute_dtype", "bfloat16")),
        )
        if self.is_main_process:
            print(
                f"[Understanding] Load options: dtype={dtype}, device_map={device_map}, "
                f"attn_implementation={attn_impl or 'default'}"
            )

        if self.cfg.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            try:
                from peft import prepare_model_for_kbit_training
            except Exception:
                prepare_model_for_kbit_training = None

            if bool(getattr(self.cfg, "load_in_4bit", False)):
                if prepare_model_for_kbit_training is None:
                    raise RuntimeError("QLoRA requires peft.prepare_model_for_kbit_training")
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

            requested_targets = _target_tuple(self.cfg.lora_target_modules)
            text_targets = _text_only_lora_targets(requested_targets)
            if not text_targets:
                raise ValueError(
                    "No text LoRA targets configured. `lora_target_modules` should include Qwen targets "
                    "such as q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj."
                )
            solver_merger_enabled = bool(getattr(self.cfg, "solver_merger_lora_enabled", False))
            solver_merger_targets = (
                _target_tuple(getattr(self.cfg, "solver_merger_lora_target_modules", tuple()))
                if solver_merger_enabled
                else tuple()
            )
            solver_targets = _dedupe_targets(text_targets, solver_merger_targets)
            lcfg_kwargs = {
                "r": self.cfg.lora_r,
                "lora_alpha": self.cfg.lora_alpha,
                "lora_dropout": self.cfg.lora_dropout,
                "target_modules": list(solver_targets),
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM,
            }
            if solver_merger_targets:
                lcfg_kwargs["rank_pattern"] = {
                    target: int(getattr(self.cfg, "solver_merger_lora_r", self.cfg.lora_r))
                    for target in solver_merger_targets
                }
                lcfg_kwargs["alpha_pattern"] = {
                    target: int(getattr(self.cfg, "solver_merger_lora_alpha", self.cfg.lora_alpha))
                    for target in solver_merger_targets
                }
            try:
                lcfg = LoraConfig(**lcfg_kwargs)
            except TypeError:
                lcfg_kwargs.pop("rank_pattern", None)
                lcfg_kwargs.pop("alpha_pattern", None)
                lcfg = LoraConfig(**lcfg_kwargs)
            proposer_cfg = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=list(text_targets),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lcfg)
            # default adapter is solver; create proposer adapter explicitly
            if hasattr(model, "add_adapter"):
                try:
                    model.add_adapter("proposer", proposer_cfg)
                except Exception as exc:
                    raise RuntimeError(f"Failed to add proposer adapter: {exc}") from exc

            # Keep training restricted to role adapters only.
            for name, param in model.named_parameters():
                if "lora_" in name and (".default." in name or ".proposer." in name):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

            if self.is_main_process:
                print(
                    "[Understanding] Role LoRA targets: "
                    f"text={list(text_targets)}; "
                    f"solver_merger_enabled={solver_merger_enabled}; "
                    f"solver_merger={list(solver_merger_targets)}"
                )
                model.print_trainable_parameters()

        # Activation checkpointing significantly reduces training-time memory.
        gc_enabled = os.environ.get("SE_USE_GRADIENT_CHECKPOINTING", "1").strip().lower() not in {"0", "false", "no"}
        gc_use_reentrant_env = os.environ.get("SE_GRADIENT_CHECKPOINT_USE_REENTRANT", "").strip().lower()
        if gc_use_reentrant_env:
            gc_use_reentrant = gc_use_reentrant_env in {"1", "true", "yes", "on"}
        else:
            # Non-reentrant checkpointing is DDP-safe for multi-adapter LoRA training.
            gc_use_reentrant = False
        if gc_use_reentrant and (self.distributed or self.cfg.use_lora):
            # Reentrant checkpointing is incompatible with this trainer's
            # DDP + multi-adapter LoRA update pattern and can trigger:
            # - "mark variable ready twice" (DDP reducer error)
            # - no-grad checkpoint warnings for frozen-base LoRA tuning
            if self.is_main_process:
                print(
                    "[Understanding] Forcing gradient checkpointing use_reentrant=False "
                    "(DDP/LoRA compatibility)."
                )
            gc_use_reentrant = False
        if self.is_main_process:
            print(
                "[Understanding] Gradient checkpointing config: "
                f"enabled={gc_enabled} env_use_reentrant="
                f"{gc_use_reentrant_env if gc_use_reentrant_env else '<unset>'} "
                f"effective_use_reentrant={gc_use_reentrant}"
            )
        if gc_enabled and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": gc_use_reentrant}
                )
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                if self.is_main_process:
                    print(
                        f"[Understanding] Enabled gradient checkpointing "
                        f"(use_reentrant={gc_use_reentrant})."
                    )
            except TypeError:
                # Older transformers versions don't accept gradient_checkpointing_kwargs.
                # In DDP+LoRA, avoid silently enabling unknown/default reentrant mode.
                if self.distributed or self.cfg.use_lora:
                    if self.is_main_process:
                        print(
                            "[Understanding] Skipping gradient checkpointing: "
                            "current transformers build does not expose "
                            "gradient_checkpointing_kwargs (cannot guarantee "
                            "use_reentrant=False safely under DDP/LoRA)."
                        )
                else:
                    model.gradient_checkpointing_enable()
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    if self.is_main_process:
                        print("[Understanding] Enabled gradient checkpointing.")
            except Exception:
                pass
        elif self.is_main_process and not gc_enabled:
            print("[Understanding] Gradient checkpointing disabled via SE_USE_GRADIENT_CHECKPOINTING=0.")

        model.eval()
        return model, processor

    # -------------------------------------------------------------------
    # Generation / sampling
    # -------------------------------------------------------------------
    def _sample_image_for_step(self, step: int) -> Tuple[Image.Image, Dict]:
        if self.distributed:
            global_offset = (step - 1) * self.world_size + self.rank
            shuffled_idx = self.pool.indices[global_offset % len(self.pool.indices)]
        else:
            shuffled_idx = self.pool.indices[(step - 1) % len(self.pool.indices)]
        return self.pool.get_image(shuffled_idx)

    def _generate(
        self,
        image: Image.Image,
        prompt: str,
        adapter_name: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool = True,
    ) -> str:
        chat_text = _build_chat_text(self.processor, image, prompt)
        inputs = _prepare_mm_inputs(self.processor, self.device, image, chat_text, model=self.model)

        from .model_api import _extract_tokenizer_from_processor
        _tok = _extract_tokenizer_from_processor(self.processor)
        _pad_id = getattr(_tok, "eos_token_id", None) if _tok is not None else None

        with torch.no_grad():
            with use_adapter(self.model, adapter_name):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=_pad_id,
                )

        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        completion_ids = outputs[0, input_len:]
        text = _decode_tokens(self.processor, completion_ids)
        return text.strip()

    def _update_baseline(self, which: str, reward: float):
        m = self.cfg.baseline_momentum
        if which == "solver":
            self.solver_baseline = m * self.solver_baseline + (1 - m) * reward
        else:
            self.proposer_baseline = m * self.proposer_baseline + (1 - m) * reward

    def _append_iter_record(self, record: Dict):
        payload = dict(record)
        payload.setdefault("phase", "understanding")
        self._append_jsonl(self.iter_log_path, payload)
        self._append_jsonl(self.release_rollouts_log_path, payload)
        self._monitor_understanding_record(payload)

    # -------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------
    def _save_checkpoint(self, step: int):
        if not self.is_main_process:
            return
        step_dir = self.run_dir / f"step_{step:05d}"
        tmp_dir = self.run_dir / f"step_{step:05d}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.use_lora:
            for adapter_name, sub_name in (("default", "solver"), ("proposer", "proposer")):
                subdir = tmp_dir / sub_name
                subdir.mkdir(parents=True, exist_ok=True)
                saved = False
                try:
                    self.model.save_pretrained(subdir, selected_adapters=[adapter_name])
                    saved = True
                except TypeError:
                    saved = False
                except Exception:
                    saved = False
                if not saved:
                    with use_adapter(self.model, adapter_name):
                        self.model.save_pretrained(subdir)
                try:
                    sanitize_peft_adapter_dir(
                        subdir,
                        in_place=True,
                        backup=False,
                        log=lambda msg: print(f"[Understanding] {msg}"),
                    )
                except Exception as exc:
                    print(f"[Understanding] WARNING: failed to sanitize {sub_name} adapter checkpoint: {exc}")
                if sub_name == "solver":
                    try:
                        self.processor.save_pretrained(subdir)
                    except Exception:
                        pass
        else:
            self.model.save_pretrained(tmp_dir / "model")
            try:
                self.processor.save_pretrained(tmp_dir / "model")
            except Exception:
                pass

        torch.save(self._trainer_state_dict(step), tmp_dir / "trainer_state.pt")

        _json_dump(
            tmp_dir / "trainer_state.json",
            {
                "step": step,
                "solver_baseline": self.solver_baseline,
                "proposer_baseline": self.proposer_baseline,
                "solver_kl_coef": self.solver_updater.kl_coef,
                "proposer_kl_coef": self.proposer_updater.kl_coef,
                "solver_updater_step": self.solver_updater.step_id,
                "proposer_updater_step": self.proposer_updater.step_id,
            },
        )
        with (tmp_dir / "SAVE_OK").open("w", encoding="utf-8") as f:
            f.write("ok\n")

        if step_dir.exists():
            shutil.rmtree(step_dir, ignore_errors=True)
        os.replace(str(tmp_dir), str(step_dir))
        self._sync_standard_checkpoint_dir(step=step, source_dir=step_dir)

        self._prune_checkpoints()

    def _prune_checkpoints(self):
        if not self.is_main_process:
            return
        keep = max(1, int(self.cfg.max_checkpoints))
        checkpoints = self._list_complete_checkpoints()
        if len(checkpoints) <= keep:
            return
        keep_steps = {int(path.name.split("_")[-1]) for path in checkpoints[-keep:]}
        for path in checkpoints[:-keep]:
            shutil.rmtree(path, ignore_errors=True)
        for alias_dir in self.checkpoint_root.glob("step_*"):
            try:
                alias_step = int(alias_dir.name.split("_")[-1])
            except Exception:
                continue
            if alias_step not in keep_steps:
                self._remove_path(alias_dir)

    def _write_ablation_summary(
        self,
        final_step: int,
        *,
        status: str = "completed",
        interrupted_at_step: Optional[int] = None,
        error: Optional[str] = None,
    ):
        if not self.is_main_process:
            return
        payload = {
            "experiment": self.cfg.experiment_name,
            "run_dir": str(self.run_dir),
            "final_step": int(final_step),
            "start_step": int(self.start_step),
            "status": status,
            "interrupted_at_step": interrupted_at_step,
            "error": error,
            "policy_update_counts": self._policy_update_counts,
            "metrics": self._metrics_summary(),
            "rollouts_log_path": str(self.release_rollouts_log_path),
            "metrics_log_path": str(self.metrics_log_path),
            "last_checkpoint_dir": str(self.last_checkpoint_dir),
        }
        _json_dump(self.summary_path, payload)
        _json_dump(self.release_summary_path, payload)
        return payload

    # -------------------------------------------------------------------
    # W&B step logging
    # -------------------------------------------------------------------
    def _wandb_log_step(
        self,
        *,
        step: int,
        image: Optional[Image.Image],
        image_path: Optional[str],
        question: str,
        proposer_out: str,
        solver_answers_raw: List[str],
        maj_answer: str,
        maj_count: int,
        maj_frac: float,
        entropy_nats: float,
        proposer_reward: float,
        solver_rewards_raw: List[float],
        solver_rewards_soft: List[float],
        pre_words_mean: float,
        solver_stats_mean: Dict[str, float],
        proposer_stats: Optional[Dict[str, float]],
    ):
        if not self.is_main_process or self.wandb_run is None:
            return

        import wandb

        metrics: Dict[str, object] = {
            "train/step": step,
            "train/maj_count": maj_count,
            "train/maj_frac": maj_frac,
            "train/num_solver_samples": self.cfg.num_solver_samples,
            "train/solver_reward_mean_raw": sum(solver_rewards_raw) / max(1, len(solver_rewards_raw)),
            "train/solver_reward_mean_soft": sum(solver_rewards_soft) / max(1, len(solver_rewards_soft)),
            "train/proposer_reward_gauss": proposer_reward,
            "train/solver_baseline": self.solver_baseline,
            "train/proposer_baseline": self.proposer_baseline,
            "train/entropy_nats": entropy_nats,
            "train/pre_words_mean": pre_words_mean,
            "text/question": question,
            "text/maj_answer": maj_answer,
            "text/solver_answers": ", ".join(solver_answers_raw),
            "text/proposer_out": proposer_out,
            "kl/solver_beta": self.solver_updater.kl_coef,
            "kl/proposer_beta": self.proposer_updater.kl_coef,
        }
        if image_path:
            metrics["data/image_path"] = image_path

        if solver_stats_mean:
            metrics.update(
                {
                    "solver/ce_loss_mean": solver_stats_mean.get("ce_loss_mean"),
                    "solver/kl_loss_mean": solver_stats_mean.get("kl_loss_mean"),
                    "solver/advantage_mean": solver_stats_mean.get("advantage_mean"),
                }
            )
        if proposer_stats:
            metrics.update(
                {
                    "proposer/ce_loss": proposer_stats.get("ce_loss"),
                    "proposer/kl_loss": proposer_stats.get("kl_loss"),
                    "proposer/advantage": proposer_stats.get("advantage"),
                    "proposer/kl_coef": proposer_stats.get("kl_coef_after"),
                }
            )

        if (
            self.cfg.wandb_log_images_every > 0
            and image is not None
            and (step % self.cfg.wandb_log_images_every) == 0
        ):
            try:
                metrics["vis/image"] = wandb.Image(image, caption=f"step={step}")
            except Exception:
                pass

        try:
            wandb.log(metrics, step=step)
        except Exception as exc:
            print(f"[W&B] log failed at step {step}: {exc}")

    # -------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        if cfg.total_steps <= self.start_step:
            raise ValueError(
                f"total_steps ({cfg.total_steps}) must be greater than start_step ({self.start_step})."
            )

        if self.is_main_process:
            print(f"[Understanding] Starting run at: {self.run_dir}")
            print(f"[Understanding] Model: {cfg.model_name}")
            print(f"[Understanding] Images: {len(self.pool)}")
            print(f"[Understanding] Step range: {self.start_step + 1}..{cfg.total_steps}")
            if self.distributed:
                print(
                    f"[Understanding] Distributed mode: world_size={self.world_size}, "
                    f"effective_batch_per_step={self.world_size}"
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
            self._write_status(state="running", progress=progress, metrics=metrics)
            if int(step_id) % max(1, int(cfg.log_every)) == 0:
                self._append_metrics({"kind": "heartbeat", **progress, **metrics})

        self._write_status(
            state="running",
            progress=self._progress_core(
                step=int(self.start_step),
                phase="init",
                run_started_at=run_started_at,
            ),
            metrics=self._release_metrics(step_time_sec=0.0),
        )
        try:
            for step in range(self.start_step + 1, cfg.total_steps + 1):
                last_attempted_step = step
                step_t0 = time.perf_counter()
                image, meta = self._sample_image_for_step(step)

                # --- Proposer: single multi-candidate call, pick hardest ---
                entropy_min = float(getattr(cfg, "sc_entropy_min", 0.15))
                entropy_max = float(getattr(cfg, "sc_entropy_max", 1.2))
                if entropy_min > entropy_max:
                    entropy_min, entropy_max = entropy_max, entropy_min
                margin_max = float(getattr(cfg, "sc_margin_max", 0.9))
                require_objective = bool(getattr(cfg, "proposer_require_objective", True))
                acceptance_require_non_easy = bool(
                    getattr(cfg, "acceptance_require_non_easy", True)
                )
                rejected_question_penalty = max(
                    0.0, float(getattr(cfg, "rejected_question_penalty", 0.0))
                )
                entropy_iqr_state = self._entropy_iqr_filter_state()
                entropy_easy_threshold = float(entropy_iqr_state.get("threshold", entropy_min))
                entropy_iqr_filter_active = bool(entropy_iqr_state.get("active", 0.0) > 0.5)
                difficulty_target_state = self._choose_difficulty_target()
                difficulty_sampler_enabled = bool(difficulty_target_state.get("enabled", False))
                desired_difficulty_bucket = str(
                    difficulty_target_state.get("desired_bucket", "medium")
                )
                difficulty_sampler_mode = str(difficulty_target_state.get("mode", "target"))

                solver_temperatures = self._solver_temperature_schedule()
                solver_top_ps = self._solver_top_p_schedule()

                num_candidates = max(1, int(getattr(cfg, "proposer_num_candidates", 3)))
                spot_check_samples = max(1, int(getattr(cfg, "proposer_spot_check_samples", 2)))

                # Single proposer call produces K candidates with adversarial
                # solver-failure reasoning baked in.
                proposer_out = self._generate(
                    image=image,
                    prompt=build_proposer_multi_prompt(desired_difficulty_bucket, num_candidates),
                    adapter_name="proposer" if cfg.use_lora else None,
                    max_new_tokens=cfg.max_new_tokens_proposer,
                    temperature=cfg.temp,
                    top_p=cfg.top_p,
                )

                # Parse all K candidate question strings.
                raw_candidates = _parse_all_questions(proposer_out)
                candidates = [c.replace("\n", " ").strip() for c in raw_candidates if c.strip()]
                if not candidates:
                    candidates = ["What is the most salient object in the image?"]

                # Spot-check each candidate with a small number of solver samples
                # and pick the one with the highest entropy (hardest for solver).
                best_question = candidates[0]
                best_spot_entropy = -1.0
                for cand in candidates:
                    if not self._is_objective_question(cand):
                        # Skip non-objective candidates entirely; don't waste
                        # full solver budget on them.
                        continue
                    _spot_answers: List[str] = []
                    for _si in range(spot_check_samples):
                        _st = (
                            float(solver_temperatures[_si])
                            if _si < len(solver_temperatures)
                            else float(cfg.temp)
                        )
                        _sp = (
                            float(solver_top_ps[_si])
                            if _si < len(solver_top_ps)
                            else float(cfg.top_p)
                        )
                        _sout = self._generate(
                            image=image,
                            prompt=build_solver_prompt(cand),
                            adapter_name="default" if cfg.use_lora else None,
                            max_new_tokens=cfg.max_new_tokens_solver,
                            temperature=_st,
                            top_p=_sp,
                        )
                        _spot_answers.append(normalize_answer(_parse_answer(_sout)))
                    _spot_hist: Dict[str, int] = {}
                    for _a in _spot_answers:
                        _spot_hist[_a] = _spot_hist.get(_a, 0) + 1
                    _spot_probs = [v / float(spot_check_samples) for v in _spot_hist.values()]
                    _spot_entropy = shannon_entropy_nats(_spot_probs)
                    if _spot_entropy > best_spot_entropy:
                        best_spot_entropy = _spot_entropy
                        best_question = cand

                question = best_question
                # Check if the selected question is objective.
                proposer_non_objective_question = bool(
                    require_objective and (not self._is_objective_question(question))
                )
                proposer_rationale = strip_tags(proposer_out, "rationale")
                fallback_question_used = question not in raw_candidates

                # --- Full solver rollout on the selected question ---
                solver_prompt = build_solver_prompt(question)
                solver_outputs: List[str] = []
                solver_answers_raw: List[str] = []
                solver_answers_norm: List[str] = []
                pre_words: List[int] = []

                for sample_idx in range(cfg.num_solver_samples):
                    solver_temp = (
                        float(solver_temperatures[sample_idx])
                        if sample_idx < len(solver_temperatures)
                        else float(cfg.temp)
                    )
                    solver_top_p = (
                        float(solver_top_ps[sample_idx])
                        if sample_idx < len(solver_top_ps)
                        else float(cfg.top_p)
                    )
                    solver_out = self._generate(
                        image=image,
                        prompt=solver_prompt,
                        adapter_name="default" if cfg.use_lora else None,
                        max_new_tokens=cfg.max_new_tokens_solver,
                        temperature=solver_temp,
                        top_p=solver_top_p,
                    )
                    answer_raw = _parse_answer(solver_out)
                    solver_outputs.append(solver_out)
                    solver_answers_raw.append(answer_raw)
                    solver_answers_norm.append(normalize_answer(answer_raw))
                    pre_words.append(pre_answer_word_count(solver_out))

                # --- Reward computation ---
                maj_answer, maj_count = majority_vote(solver_answers_norm)
                maj_frac = maj_count / float(cfg.num_solver_samples)
                hist: Dict[str, int] = {}
                for ans in solver_answers_norm:
                    hist[ans] = hist.get(ans, 0) + 1
                probs = [count / float(cfg.num_solver_samples) for count in hist.values()]
                entropy_nats = shannon_entropy_nats(probs)

                sorted_probs = sorted(probs, reverse=True)
                p1 = float(sorted_probs[0]) if sorted_probs else 0.0
                p2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
                margin = max(0.0, p1 - p2)
                ratio_min = float(getattr(cfg, "sc_informative_ratio_min", 0.25))
                ratio_min = max(0.0, min(1.0, ratio_min))
                neg_weight = float(getattr(cfg, "sc_negative_weight", 0.25))

                # Informativeness score: Gaussian centred on the target entropy band.
                entropy_span = max(1e-6, entropy_max - entropy_min)
                entropy_mid = 0.5 * (entropy_min + entropy_max)
                entropy_sigma = max(1e-6, 0.5 * entropy_span)
                entropy_band_score = math.exp(
                    -((entropy_nats - entropy_mid) ** 2) / (2.0 * (entropy_sigma ** 2))
                )
                margin_damp_score = max(0.0, 1.0 - (margin / max(1e-6, margin_max)))
                local_info_score = max(0.0, min(1.0, 0.5 * entropy_band_score + 0.5 * margin_damp_score))
                solver_informative_local = bool(
                    (entropy_min <= entropy_nats <= entropy_max) or (margin <= margin_max)
                )
                informative_ratio = self._dist_mean(1.0 if solver_informative_local else 0.0)
                solver_informative_any = informative_ratio > 0.0
                solver_informative_all = informative_ratio >= (1.0 - 1e-8)
                solver_informative_gate = solver_informative_local
                solver_informative_gate_global = informative_ratio >= ratio_min

                sc_signal = max(1e-4, local_info_score)

                # Classify difficulty bucket.
                difficulty_bucket_observed = self._difficulty_bucket(
                    entropy_nats,
                    margin,
                    maj_frac,
                    entropy_easy_threshold,
                )
                self._entropy_window.append(float(entropy_nats))
                self._difficulty_window.append(difficulty_bucket_observed)

                # --- Solver rewards ---
                # Too easy  (entropy ≈ 0, all solvers agree): penalise solvers for
                #   answering correctly — this case should not reinforce them.
                # Too hard  (all solvers wrong / pure noise): also penalise, because
                #   the question is unlearnable.
                # Sweet-spot (moderate disagreement): reward the majority answer.
                easy_solver_case = bool(
                    entropy_nats < entropy_easy_threshold and margin > margin_max
                )
                # "Unsolvable" = every solver gave a *different* answer AND the
                # majority fraction is at or below random-chance level.  With
                # num_solver_samples=5 that is ≤ 1/5 = 0.20.
                unsolvable_threshold = float(
                    getattr(cfg, "solver_unsolvable_maj_threshold", 1.0 / max(1, cfg.num_solver_samples))
                )
                unsolvable_solver_case = bool(
                    not easy_solver_case and maj_frac <= unsolvable_threshold
                )
                easy_solver_penalty_scale = max(
                    0.0, float(getattr(cfg, "easy_solver_penalty_scale", 1.0))
                )
                if easy_solver_case:
                    # Proposer asked something trivially easy — penalise solvers
                    # that got it right to discourage memorising easy answers.
                    solver_rewards_raw = [
                        (-easy_solver_penalty_scale * sc_signal)
                        if ans == maj_answer
                        else (neg_weight * sc_signal)
                        for ans in solver_answers_norm
                    ]
                elif unsolvable_solver_case:
                    # Question is pure noise to the solver — everyone gets penalised
                    # equally (small negative) to signal the question is useless.
                    solver_rewards_raw = [
                        -neg_weight * sc_signal
                        for _ in solver_answers_norm
                    ]
                else:
                    solver_rewards_raw = [
                        sc_signal if ans == maj_answer else (-neg_weight * sc_signal)
                        for ans in solver_answers_norm
                    ]

                target_w = max(1, cfg.len_penalty_target_words)
                penalties = [
                    min(1.0, max(0.0, (w - target_w) / float(target_w))) for w in pre_words
                ]
                prob_map = {
                    ans: count / float(cfg.num_solver_samples) for ans, count in hist.items()
                }
                solver_probs = [prob_map[ans] for ans in solver_answers_norm]
                solver_rewards_soft = [
                    (prob ** cfg.solver_soft_gamma)
                    * (1.0 - cfg.len_penalty_weight * pen)
                    * reward_raw
                    for prob, pen, reward_raw in zip(solver_probs, penalties, solver_rewards_raw)
                ]

                # --- Proposer reward: symmetric penalty for too-easy AND too-hard ---
                #
                # Target zone: entropy_min ≤ entropy_nats ≤ entropy_max.
                # The Gaussian reward (gaussian_reward) already gives ~1 at the
                # target mu and decays for deviations in both directions, so it
                # naturally penalises both extremes.  We add explicit hard floors
                # for the two degenerate cases so the gradient signal is strong
                # enough to overcome the model's prior.
                #
                #   easy:       entropy ≈ 0   → all solvers unanimously correct
                #   unsolvable: entropy >> max → all solvers give different wrong answers
                #
                proposer_entropy_mu_used = self._update_proposer_entropy_target(entropy_nats)
                proposer_reward_raw = gaussian_reward(
                    entropy_nats, proposer_entropy_mu_used, cfg.prop_entropy_sigma
                )
                proposer_reward = proposer_reward_raw

                # NOTE: This hard-negative reward logic is NOT used by current
                # experiments (E1-E6), which all run via UnifiedSelfEvolvingTrainer
                # with V-Zero dual-track shaping (see unified_trainer.py L728-741).
                # This code path only activates with --experiment understanding_self_evolving.
                #
                # Hard NEGATIVE for trivially easy questions (entropy ≈ 0).
                # Using min() gave a positive floor (+0.10), making net penalty only -0.25.
                # Assigning a hard negative makes trivially-easy a genuine punishment.
                zero_entropy_capped = False
                zero_entropy_cap = float(getattr(cfg, "zero_entropy_reward_cap", 0.10))
                if entropy_nats < 1e-6:
                    proposer_reward = -zero_entropy_cap  # hard negative, not a positive floor
                    zero_entropy_capped = True

                # Hard NEGATIVE for unsolvable questions (solver cannot learn anything).
                unsolvable_capped = False
                unsolvable_reward_cap = float(
                    getattr(cfg, "proposer_unsolvable_reward_cap", zero_entropy_cap)
                )
                if unsolvable_solver_case and not zero_entropy_capped:
                    proposer_reward = -unsolvable_reward_cap  # hard negative, not a positive floor
                    unsolvable_capped = True

                # Non-objective question penalty.
                proposer_non_objective_penalty = max(
                    0.0, float(getattr(cfg, "proposer_non_objective_penalty", 0.0))
                )
                if proposer_non_objective_question and proposer_non_objective_penalty > 0.0:
                    proposer_reward -= proposer_non_objective_penalty

                # Rejection: non-objective or too-easy bucket.
                reject_reasons: List[str] = []
                if require_objective and proposer_non_objective_question:
                    reject_reasons.append("non_objective")
                if acceptance_require_non_easy and (difficulty_bucket_observed == "easy"):
                    reject_reasons.append("easy_bucket")
                question_rejected = len(reject_reasons) > 0
                question_reject_reason = "|".join(reject_reasons)
                if question_rejected and rejected_question_penalty > 0.0:
                    proposer_reward -= rejected_question_penalty

                proposer_reward = max(-1.0, min(1.0, proposer_reward))

                fallback_used = fallback_question_used
                template_fallback_used = False
                easy_question_detected = easy_solver_case

                # --- Solver policy updates ---
                solver_baseline_before_step = self.solver_baseline
                solver_step_stats = []
                solver_update_due = True
                local_solver_update_applied = True
                solver_update_applied = True
                solver_update_skip_reason = None
                solver_update_skip_reason_local = None
                skip_uninformative = bool(
                    getattr(cfg, "skip_solver_update_when_uninformative", True)
                )
                always_scale = bool(
                    getattr(cfg, "solver_always_update_with_informative_scaling", True)
                )
                min_update_scale = float(getattr(cfg, "solver_update_min_scale", 0.20))
                min_update_scale = max(0.0, min(1.0, min_update_scale))
                if always_scale:
                    solver_update_scale = max(min_update_scale, local_info_score)
                else:
                    solver_update_scale = 1.0
                solver_skip_update_on_easy = bool(
                    getattr(cfg, "solver_skip_update_on_easy", True)
                )
                easy_update_majority_frac_threshold = float(
                    getattr(cfg, "easy_update_majority_frac_threshold", 0.95)
                )
                easy_update_majority_frac_threshold = max(
                    0.0, min(1.0, easy_update_majority_frac_threshold)
                )
                entropy_iqr_filter_min_majority_frac = float(
                    getattr(cfg, "entropy_iqr_filter_min_majority_frac", 0.80)
                )
                entropy_iqr_filter_min_majority_frac = max(
                    0.0, min(1.0, entropy_iqr_filter_min_majority_frac)
                )
                solver_entropy_iqr_blocked = bool(
                    getattr(cfg, "solver_skip_update_on_easy", True)
                    and
                    entropy_iqr_filter_active
                    and (entropy_nats <= entropy_easy_threshold)
                    and (maj_frac >= entropy_iqr_filter_min_majority_frac)
                )
                solver_easy_update_blocked = bool(
                    solver_skip_update_on_easy
                    and (
                        easy_solver_case
                        or (maj_frac >= easy_update_majority_frac_threshold)
                    )
                )
                if local_solver_update_applied and question_rejected:
                    local_solver_update_applied = False
                    solver_update_skip_reason_local = (
                        "question_rejected"
                        if not question_reject_reason
                        else f"question_rejected:{question_reject_reason}"
                    )
                elif local_solver_update_applied and solver_entropy_iqr_blocked:
                    local_solver_update_applied = False
                    solver_update_skip_reason_local = "entropy_iqr_filter"
                elif local_solver_update_applied and solver_easy_update_blocked:
                    local_solver_update_applied = False
                    solver_update_skip_reason_local = "easy_case"
                elif (
                    local_solver_update_applied
                    and (not always_scale)
                    and skip_uninformative
                    and not solver_informative_gate
                ):
                    local_solver_update_applied = False
                    solver_update_skip_reason_local = "uninformative_local"

                # DDP safety: if any rank will run solver updates, all ranks must
                # execute the same number of updater.forward() calls.
                solver_update_applied = self._dist_any_bool(local_solver_update_applied)
                if solver_update_applied:
                    for sample_idx, (
                        completion,
                        reward,
                        reward_raw,
                        answer_raw,
                        answer_norm,
                        prob,
                        penalty,
                        words,
                        temp,
                        sample_top_p,
                    ) in enumerate(
                        zip(
                            solver_outputs,
                            solver_rewards_soft,
                            solver_rewards_raw,
                            solver_answers_raw,
                            solver_answers_norm,
                            solver_probs,
                            penalties,
                            pre_words,
                            solver_temperatures,
                            solver_top_ps,
                        ),
                        start=1,
                    ):
                        local_can_solver_update = bool(str(completion).strip())
                        any_rank_can_solver_update = self._dist_any_bool(local_can_solver_update)
                        if not any_rank_can_solver_update:
                            self._append_jsonl(
                                self.policy_updates_log_path,
                                {
                                    "step": step,
                                    "role": "solver",
                                    "sample_idx": sample_idx,
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
                        solver_step_stats.append(stats)
                        if stats.get("did_step", True):
                            self._policy_update_counts["solver"] += 1
                        if not local_skip_update:
                            # Track the SCALED reward that the updater actually receives,
                            # not the raw reward.  Otherwise baseline > effective_reward
                            # when scale < 1, causing systematic negative advantage bias.
                            self._update_baseline("solver", effective_reward)
                        self._sync_state_scalars()
                        baseline_after = self.solver_baseline

                        self._append_jsonl(
                            self.rollouts_log_path,
                            {
                                "step": step,
                                "sample_idx": sample_idx,
                                "image_path": meta.get("path"),
                                "solver_prompt": solver_prompt,
                                "solver_completion": completion,
                                "answer_raw": answer_raw,
                                "answer_norm": answer_norm,
                                "answer_probability": prob,
                                "solver_temperature": temp,
                                "solver_top_p": sample_top_p,
                                "reward_raw": reward_raw,
                                "reward_soft": reward,
                                "solver_update_scale": solver_update_scale,
                                "reward_effective": effective_reward,
                                "length_penalty": penalty,
                                "pre_answer_word_count": words,
                            },
                        )
                        self._append_jsonl(
                            self.policy_updates_log_path,
                            {
                                "step": step,
                                "role": "solver",
                                "sample_idx": sample_idx,
                                "reward": reward,
                                "baseline_before": baseline_before,
                                "baseline_after": baseline_after,
                                "stats": stats,
                            },
                        )
                    if solver_step_stats:
                        all_skipped = all(bool(s.get("skipped_reason")) for s in solver_step_stats)
                        if all_skipped:
                            solver_update_applied = False
                            if solver_update_skip_reason_local is None:
                                solver_update_skip_reason_local = "all_solver_samples_skipped"
                else:
                    self._append_jsonl(
                        self.policy_updates_log_path,
                        {
                            "step": step,
                            "role": "solver",
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
                solver_baseline_after_step = self.solver_baseline

                # --- Proposer policy update ---
                proposer_baseline_before_step = self.proposer_baseline
                proposer_baseline_after_step = proposer_baseline_before_step
                proposer_stats = None
                if step % cfg.proposer_update_freq == 0:
                    # Train on the full proposer output (proposer_out) so that
                    # the gradient flows through the rationale and reasoning
                    # tokens that actually determine question difficulty — not
                    # just the final 8-12 question tokens. Fall back to
                    # question-only if proposer_out is unavailable.
                    proposer_completion = str(proposer_out or question or "").strip()
                    local_can_proposer_update = bool(proposer_completion)
                    any_rank_can_proposer_update = self._dist_any_bool(local_can_proposer_update)
                    if any_rank_can_proposer_update:
                        completion_for_update = proposer_completion if local_can_proposer_update else ""
                        effective_reward = proposer_reward if local_can_proposer_update else 0.0
                        # Use the multi-candidate prompt as the conditioning context.
                        proposer_prompt_for_update = build_proposer_multi_prompt(
                            desired_difficulty_bucket,
                            max(1, int(getattr(cfg, "proposer_num_candidates", 3))),
                        )
                        # Use the raw baseline without clamping. The previous
                        # clamp (min(baseline, reward) when reward < 0) caused
                        # the advantage to collapse to exactly 0.0 at
                        # equilibrium — eliminating the learning signal. Standard
                        # REINFORCE advantage = reward - baseline handles negative
                        # rewards correctly without any clamping.
                        effective_baseline = proposer_baseline_before_step if local_can_proposer_update else 0.0
                        proposer_stats = self.proposer_updater.step(
                            image=image,
                            prompt=proposer_prompt_for_update,
                            completion=completion_for_update,
                            reward=effective_reward,
                            baseline=effective_baseline,
                            device=self.device,
                        )
                        if proposer_stats.get("did_step", True):
                            self._policy_update_counts["proposer"] += 1
                        self._append_jsonl(
                            self.policy_updates_log_path,
                            {
                                "step": step,
                                "role": "proposer",
                                "reward": proposer_reward,
                                "baseline_before": proposer_baseline_before_step,
                                "stats": proposer_stats,
                            },
                        )
                        if local_can_proposer_update:
                            self._update_baseline("proposer", proposer_reward)
                    else:
                        self._append_jsonl(
                            self.policy_updates_log_path,
                            {
                                "step": step,
                                "role": "proposer",
                                "skipped": True,
                                "reason": "all_ranks_empty_proposer_completion",
                                "baseline_before": proposer_baseline_before_step,
                            },
                        )
                    self._sync_state_scalars()
                    proposer_baseline_after_step = self.proposer_baseline

                # --- Aggregated metrics ---
                solver_raw_mean = sum(solver_rewards_raw) / len(solver_rewards_raw)
                solver_soft_mean = sum(solver_rewards_soft) / len(solver_rewards_soft)
                pre_words_mean = sum(pre_words) / len(pre_words)
                step_duration_sec = time.perf_counter() - step_t0
                solver_raw_mean_global = self._dist_mean(solver_raw_mean)
                solver_soft_mean_global = self._dist_mean(solver_soft_mean)
                proposer_reward_global = self._dist_mean(proposer_reward)
                entropy_nats_global = self._dist_mean(entropy_nats)
                maj_frac_global = self._dist_mean(maj_frac)
                pre_words_mean_global = self._dist_mean(pre_words_mean)
                step_duration_sec_global = self._dist_mean(step_duration_sec)

                solver_ce_mean = self._finite_mean_from_stats(solver_step_stats, "ce_loss")
                solver_kl_mean = self._finite_mean_from_stats(solver_step_stats, "kl_loss")
                solver_adv_mean = self._finite_mean_from_stats(solver_step_stats, "advantage")
                solver_nonfinite_count = sum(
                    1
                    for s in solver_step_stats
                    if isinstance(s, dict)
                    and self._monitor_float(s.get("ce_loss")) is None
                    and "ce_loss" in s
                )
                solver_stats_mean = {
                    "ce_loss_mean": solver_ce_mean,
                    "kl_loss_mean": solver_kl_mean,
                    "advantage_mean": solver_adv_mean,
                    "finite_solver_updates": sum(
                        1
                        for s in solver_step_stats
                        if isinstance(s, dict) and self._monitor_float(s.get("ce_loss")) is not None
                    ),
                    "nonfinite_solver_updates": solver_nonfinite_count,
                }

                if self.is_main_process and step % cfg.log_every == 0:
                    print(
                        f"[Step {step:05d}] maj={maj_count}/{cfg.num_solver_samples} "
                        f"maj_frac={maj_frac_global:.2f} H={entropy_nats_global:.3f} "
                        f"M={margin:.3f} info_local={int(solver_informative_local)} "
                        f"info_ratio={informative_ratio:.2f} info_gate={int(solver_informative_gate)} "
                        f"up_scale={solver_update_scale:.2f} "
                        f"P_R={proposer_reward_global:.3f} "
                        f"S_R_raw={solver_raw_mean_global:.3f} S_R_soft={solver_soft_mean_global:.3f} "
                        f"pre_words={pre_words_mean_global:.2f}"
                    )
                    print(f"  Q: {question}")
                    print(f"  A: [{', '.join(solver_answers_raw)}] | MAJ: {maj_answer}")

                # --- Logging ---
                self._append_jsonl(
                    self.questions_log_path,
                    {
                        "step": step,
                        "image_path": meta.get("path"),
                        "proposer_output": proposer_out,
                        "proposer_rationale": proposer_rationale,
                        "final_question": question,
                        "fallback_question_used": fallback_used,
                        "proposer_non_objective_question": proposer_non_objective_question,
                        "question_rejected": question_rejected,
                        "question_reject_reason": question_reject_reason,
                        "acceptance_require_non_easy": acceptance_require_non_easy,
                        "difficulty_sampler_enabled": difficulty_sampler_enabled,
                        "difficulty_sampler_mode": difficulty_sampler_mode,
                        "difficulty_target_bucket": desired_difficulty_bucket,
                        "difficulty_bucket_observed": difficulty_bucket_observed,
                        "difficulty_target_weights": difficulty_target_state.get("target_weights", {}),
                        "difficulty_observed_weights": difficulty_target_state.get("observed_weights", {}),
                        "difficulty_sampling_weights": difficulty_target_state.get("sampling_weights", {}),
                    },
                )
                self._append_jsonl(
                    self.rewards_log_path,
                    {
                        "step": step,
                        "image_path": meta.get("path"),
                        "majority_answer": maj_answer,
                        "majority_count": maj_count,
                        "majority_fraction": maj_frac,
                        "answer_histogram": hist,
                        "answer_probabilities": prob_map,
                        "solver_top1_prob": p1,
                        "solver_top2_prob": p2,
                        "solver_margin": margin,
                        "solver_margin_score": margin_damp_score,
                        "solver_entropy_band_score": entropy_band_score,
                        "solver_local_info_score": local_info_score,
                        "entropy_easy_threshold": entropy_easy_threshold,
                        "entropy_iqr_filter_enabled": bool(entropy_iqr_state.get("enabled", 0.0)),
                        "entropy_iqr_filter_active": entropy_iqr_filter_active,
                        "entropy_iqr_filter_history_size": int(entropy_iqr_state.get("history_size", 0.0)),
                        "entropy_iqr_filter_q1": entropy_iqr_state.get("q1"),
                        "entropy_iqr_filter_q3": entropy_iqr_state.get("q3"),
                        "entropy_iqr_filter_iqr": entropy_iqr_state.get("iqr"),
                        "easy_solver_case": easy_solver_case,
                        "easy_solver_penalty_scale": easy_solver_penalty_scale,
                        "solver_update_scale": solver_update_scale,
                        "solver_informative_local": solver_informative_local,
                        "solver_informative_any": solver_informative_any,
                        "solver_informative_all": solver_informative_all,
                        "solver_informative_ratio": informative_ratio,
                        "solver_informative_ratio_min": ratio_min,
                        "solver_informative_gate": solver_informative_gate,
                        "solver_informative_gate_global": solver_informative_gate_global,
                        "entropy_nats": entropy_nats,
                        "solver_rewards_raw": solver_rewards_raw,
                        "solver_rewards_soft": solver_rewards_soft,
                        "solver_rewards_raw_mean": solver_raw_mean,
                        "solver_rewards_soft_mean": solver_soft_mean,
                        "solver_temperature_schedule": solver_temperatures,
                        "solver_top_p_schedule": solver_top_ps,
                        "solver_update_due": solver_update_due,
                        "solver_update_applied": solver_update_applied,
                        "solver_update_skip_reason": solver_update_skip_reason,
                        "proposer_entropy_mu_used": proposer_entropy_mu_used,
                        "proposer_reward_raw": proposer_reward_raw,
                        "proposer_reward": proposer_reward,
                        "proposer_non_objective_question": proposer_non_objective_question,
                        "proposer_non_objective_penalty": proposer_non_objective_penalty,
                        "question_rejected": question_rejected,
                        "question_reject_reason": question_reject_reason,
                        "rejected_question_penalty": rejected_question_penalty,
                        "acceptance_require_non_easy": acceptance_require_non_easy,
                        "zero_entropy_capped": zero_entropy_capped,
                        "zero_entropy_reward_cap": zero_entropy_cap,
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
                        "solver_baseline_before": solver_baseline_before_step,
                        "solver_baseline_after": solver_baseline_after_step,
                        "proposer_baseline_before": proposer_baseline_before_step,
                        "proposer_baseline_after": proposer_baseline_after_step,
                    },
                )

                self._append_iter_record(
                    {
                        "step": step,
                        "image_path": meta.get("path"),
                        "question": question,
                        "proposer_out": proposer_out,
                        "proposer_rationale": proposer_rationale,
                        "fallback_question_used": fallback_used,
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
                        "solver_temperature_schedule": solver_temperatures,
                        "solver_probs": solver_probs,
                        "solver_len_penalties": penalties,
                        "pre_answer_word_counts": pre_words,
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
                        "easy_solver_case": easy_solver_case,
                        "unsolvable_solver_case": unsolvable_solver_case,
                        "easy_solver_penalty_scale": easy_solver_penalty_scale,
                        "solver_informative_local": solver_informative_local,
                        "solver_informative_any": solver_informative_any,
                        "solver_informative_all": solver_informative_all,
                        "solver_informative_ratio": informative_ratio,
                        "solver_informative_gate": solver_informative_gate,
                        "solver_informative_gate_global": solver_informative_gate_global,
                        "solver_update_scale": solver_update_scale,
                        "entropy_nats": entropy_nats,
                        "proposer_entropy_mu_used": proposer_entropy_mu_used,
                        "proposer_reward_raw": proposer_reward_raw,
                        "proposer_reward": proposer_reward,
                        "zero_entropy_capped": zero_entropy_capped,
                        "zero_entropy_reward_cap": zero_entropy_cap,
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
                        "solver_baseline_before": solver_baseline_before_step,
                        "solver_baseline_after": solver_baseline_after_step,
                        "proposer_baseline_before": proposer_baseline_before_step,
                        "proposer_baseline_after": proposer_baseline_after_step,
                        "solver_update_due": solver_update_due,
                        "solver_update_applied": solver_update_applied,
                        "solver_update_skip_reason": solver_update_skip_reason,
                        "solver_kl_coef": self.solver_updater.kl_coef,
                        "proposer_kl_coef": self.proposer_updater.kl_coef,
                        "solver_stats_per_sample": solver_step_stats,
                        "solver_stats_mean": solver_stats_mean,
                        "proposer_stats": proposer_stats,
                        "step_duration_sec": step_duration_sec,
                    }
                )

                self._wandb_log_step(
                    step=step,
                    image=image,
                    image_path=meta.get("path"),
                    question=question,
                    proposer_out=proposer_out,
                    solver_answers_raw=solver_answers_raw,
                    maj_answer=maj_answer,
                    maj_count=maj_count,
                    maj_frac=maj_frac_global,
                    entropy_nats=entropy_nats_global,
                    proposer_reward=proposer_reward_global,
                    solver_rewards_raw=solver_rewards_raw,
                    solver_rewards_soft=solver_rewards_soft,
                    pre_words_mean=pre_words_mean_global,
                    solver_stats_mean=solver_stats_mean,
                    proposer_stats=proposer_stats,
                )

                self._update_metric("solver_reward_raw_mean", solver_raw_mean_global)
                self._update_metric("solver_reward_soft_mean", solver_soft_mean_global)
                self._update_metric("proposer_reward", proposer_reward_global)
                self._update_metric("entropy_nats", entropy_nats_global)
                self._update_metric("solver_margin", self._dist_mean(margin))
                self._update_metric("solver_informative", self._dist_mean(informative_ratio))
                self._update_metric("proposer_entropy_mu_used", self._dist_mean(proposer_entropy_mu_used))
                self._update_metric("majority_fraction", maj_frac_global)
                self._update_metric("entropy_easy_threshold", self._dist_mean(entropy_easy_threshold))
                self._update_metric(
                    "solver_entropy_iqr_blocked", self._dist_mean(1.0 if solver_entropy_iqr_blocked else 0.0)
                )
                self._update_metric(
                    "difficulty_bucket_easy",
                    self._dist_mean(1.0 if difficulty_bucket_observed == "easy" else 0.0),
                )
                self._update_metric(
                    "difficulty_bucket_medium",
                    self._dist_mean(1.0 if difficulty_bucket_observed == "medium" else 0.0),
                )
                self._update_metric(
                    "difficulty_bucket_hard",
                    self._dist_mean(1.0 if difficulty_bucket_observed == "hard" else 0.0),
                )
                self._update_metric("pre_answer_words_mean", pre_words_mean_global)
                self._update_metric("solver_kl_coef", float(self.solver_updater.kl_coef))
                self._update_metric("proposer_kl_coef", float(self.proposer_updater.kl_coef))
                self._update_metric("step_duration_sec", step_duration_sec_global)
                self._update_metric("fallback_question_used", 1.0 if fallback_used else 0.0)
                _emit_training_logs(step, phase="understanding", step_time_sec=step_duration_sec_global)

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

            # --- Final checkpoint ---
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
            self._append_metrics({"kind": "final_summary", **final_progress, **summary})
            self._write_status(
                state="completed",
                progress=final_progress,
                metrics=self._release_metrics(step_time_sec=0.0),
            )
            if self.is_main_process:
                print(
                    f"[Understanding] Training complete. Final checkpoint at step {cfg.total_steps:05d}."
                )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            interrupted_step = int(last_attempted_step)
            tb = traceback.format_exc()
            if self.is_main_process:
                print(
                    f"[Understanding] Training interrupted at step {interrupted_step}: {error_text}"
                )
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
                if self.is_main_process:
                    print(
                        f"[Understanding] Emergency checkpoint saved at step {emergency_step:05d}."
                    )
                    _json_dump(
                        self.run_dir / "resume_hint.json",
                        {
                            "resume_from": str(self.checkpoint_root / f"step_{emergency_step:06d}"),
                            "start_step": emergency_step,
                            "total_steps": cfg.total_steps,
                        },
                    )
            except Exception as save_exc:
                if self.is_main_process:
                    print(f"[Understanding] Emergency checkpoint failed: {save_exc}")

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
            self._append_metrics({"kind": "interrupted", **interrupted_progress, **summary})
            self._write_status(
                state="interrupted",
                progress=interrupted_progress,
                metrics=self._release_metrics(step_time_sec=0.0),
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
