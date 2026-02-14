"""
Generation-only and unified self-evolving trainers.

Ported from self_evolving/experiments/generation.py.
Uses native BLIP3o model loading instead of the workaround-heavy path.
"""

import dataclasses
import datetime as dt
import gc
import inspect
import json
import math
import os
import pathlib
import random
import re
import shutil
import time
import traceback
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoProcessor

from .config import GenerationSelfEvolvingConfig
from .image_pool import ImagePool, ImagePoolConfig
from .policy_updater import RolePolicyUpdater
from .prompts import (
    build_generation_spec_prompt,
    build_generation_spec_retry_prompt,
    build_proposer_prompt,
    build_solver_prompt,
)
from .utils import (
    HAS_PEFT,
    HAS_WANDB,
    _build_chat_text,
    _clip_grad_norm_multi_device,
    _collect_git_info,
    _collect_trainable_params,
    _decode_tokens,
    _infer_primary_device,
    _json_dump,
    _parse_answer,
    _parse_first_question,
    _prepare_mm_inputs,
    _resolve_attn_implementation,
    _safe_dtype,
    _set_global_seed,
    _unwrap_model,
    gaussian_reward,
    majority_vote,
    normalize_answer,
    pre_answer_word_count,
    shannon_entropy_nats,
    strip_tags,
    use_adapter,
)

if HAS_PEFT:
    from peft import LoraConfig, TaskType, get_peft_model
else:
    LoraConfig = None
    TaskType = None
    get_peft_model = None

if HAS_WANDB:
    import wandb

from .diffusion_pipeline import (
    _build_original_blip3o_diffusion_pipeline,
    _collect_pipeline_device_mismatches,
    _decode_blip3o_generate_image_output,
    _ensure_pipeline_device_placement,
    _is_original_blip3o_model_name,
    _resolve_multimodal_encoder_for_pipeline,
)
from .generation_helpers import (
    GEN_CYCLE_CAPTION_PROMPT,
    GEN_PROMPT_TEMPLATE,
    GENERATOR_PROXY_CAPTION_PROMPT,
    GenerationQAPair,
    GenerationSpec,
    SOURCE_CAPTION_PROMPT,
    _ensure_pil_image,
    _image_diversity_score,
    _per_candidate_diversity_scores,
    _latent_tensor_to_pil,
    _parse_generation_spec,
    _prepare_text_inputs,
    _soft_match,
    _tokenize_words,
    _yes_no_polarity,
)
from .generation_policy_updater import TextPolicyUpdater, TextPreferenceDPOUpdater, TextGRPOUpdater
from .model_api import (
    _adapt_mm_generate_inputs,
    _collect_image_token_ids,
    _count_image_tokens_in_inputs,
    _extract_tokenizer_from_processor,
    _find_callable_object,
    _find_generation_callable,
    _load_blip3o_model,
    _parse_unused_model_kwargs_from_error,
)


# ---------------------------------------------------------------------------
# GenerationSelfEvolvingTrainer
# ---------------------------------------------------------------------------


class GenerationSelfEvolvingTrainer:
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
                f"[DDP] Initialized rank={self.rank}/{self.world_size} local_rank={self.local_rank} backend={backend}"
            )
        elif torch.cuda.is_available():
            torch.cuda.set_device(self.cfg.cuda_device)

    def _dist_barrier(self):
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def _dist_mean(self, value: float) -> float:
        if not (self.distributed and dist.is_initialized()):
            return float(value)
        dev = torch.device(f"cuda:{self.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor([float(value)], dtype=torch.float64, device=dev)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float((tensor / float(self.world_size)).item())

    def _dist_all_bool(self, value: bool) -> bool:
        if not (self.distributed and dist.is_initialized()):
            return bool(value)
        dev = torch.device(f"cuda:{self.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor([1 if value else 0], dtype=torch.int32, device=dev)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return bool(int(tensor.item()) == 1)

    def _dist_any_bool(self, value: bool) -> bool:
        """Return True if ANY rank has value=True."""
        if not (self.distributed and dist.is_initialized()):
            return bool(value)
        dev = torch.device(f"cuda:{self.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor([1 if value else 0], dtype=torch.int32, device=dev)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return bool(int(tensor.item()) == 1)

    def _dist_min_int(self, value: int) -> int:
        if not (self.distributed and dist.is_initialized()):
            return int(value)
        dev = torch.device(f"cuda:{self.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor([int(value)], dtype=torch.int64, device=dev)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return int(tensor.item())

    def _dist_max_int(self, value: int) -> int:
        if not (self.distributed and dist.is_initialized()):
            return int(value)
        dev = torch.device(f"cuda:{self.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor([int(value)], dtype=torch.int64, device=dev)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return int(tensor.item())

    def _distributed_update_ready(
        self,
        local_ready: bool,
        local_reason: Optional[str],
        *,
        peer_reason: str,
    ) -> Tuple[bool, Optional[str]]:
        if not (self.distributed and dist.is_initialized()):
            return bool(local_ready), local_reason
        all_ready = self._dist_all_bool(bool(local_ready))
        if all_ready:
            return True, local_reason
        if local_reason:
            return False, local_reason
        return False, peer_reason

    def _expected_pipeline_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    @staticmethod
    def _is_diffusion_device_mismatch_error(exc: BaseException) -> bool:
        msg = str(exc)
        return (
            "Expected all tensors to be on the same device" in msg
            or "wrapper_CUDA__native_group_norm" in msg
            or "found at least two devices" in msg
        )

    def _rebuild_diffusion_pipeline(self):
        if not _is_original_blip3o_model_name(self.cfg.model_name):
            raise RuntimeError(
                "Diffusion pipeline rebuild requested for non-BLIP3o-original model. "
                f"model_name={self.cfg.model_name}"
            )
        dtype = _safe_dtype(self.cfg.dtype)
        pipeline_device = self._expected_pipeline_device()
        pipe_encoder = _resolve_multimodal_encoder_for_pipeline(self.model)
        self._blip3o_diffusion_pipe = _build_original_blip3o_diffusion_pipeline(
            self.cfg.model_name,
            multimodal_encoder=pipe_encoder,
            processor=self.processor,
            torch_dtype=dtype,
            device=pipeline_device,
        )
        if self.is_main_process:
            print(
                "[Generation] Rebuilt diffusion pipeline after placement failure "
                f"(device={pipeline_device}, dtype={dtype})."
            )

    def _run_diffusion_pipeline_with_repair(self, **kwargs):
        if self._blip3o_diffusion_pipe is None:
            raise RuntimeError("Diffusion pipeline is not initialized.")

        repair_device = self._expected_pipeline_device()
        repair_dtype = _safe_dtype(self.cfg.dtype)

        def _preflight_has_mismatch() -> bool:
            try:
                mismatches = _collect_pipeline_device_mismatches(self._blip3o_diffusion_pipe, repair_device)
            except Exception:
                return False
            if mismatches and self.is_main_process:
                preview = ", ".join(mismatches[:6])
                print(
                    "[Generation] Detected diffusion device drift before generation call "
                    f"(expected={repair_device}): {preview}"
                )
            return bool(mismatches)

        def _repair_pipeline_placement():
            self._blip3o_diffusion_pipe = _ensure_pipeline_device_placement(
                self._blip3o_diffusion_pipe,
                device=repair_device,
                torch_dtype=repair_dtype,
            )
            self._diffusion_repair_count = int(getattr(self, "_diffusion_repair_count", 0)) + 1

        if _preflight_has_mismatch():
            _repair_pipeline_placement()

        try:
            return self._blip3o_diffusion_pipe(**kwargs)
        except RuntimeError as exc:
            if not self._is_diffusion_device_mismatch_error(exc):
                raise
            first_error = exc

        # First repair attempt: re-place all components and retry once.
        try:
            _repair_pipeline_placement()
            return self._blip3o_diffusion_pipe(**kwargs)
        except Exception as repair_exc:
            if not self._is_diffusion_device_mismatch_error(repair_exc):
                raise
            second_error = repair_exc

        # Second repair attempt: rebuild full pipeline and retry once.
        try:
            self._rebuild_diffusion_pipeline()
            return self._blip3o_diffusion_pipe(**kwargs)
        except Exception as rebuild_exc:
            raise RuntimeError(
                "Diffusion pipeline failed after placement-repair and rebuild attempts. "
                f"first_error={type(first_error).__name__}: {first_error} | "
                f"second_error={type(second_error).__name__}: {second_error} | "
                f"rebuild_error={type(rebuild_exc).__name__}: {rebuild_exc}"
            ) from rebuild_exc

    def _sync_state_scalars(self):
        if not (self.distributed and dist.is_initialized()):
            return
        self.generator_baseline = self._dist_mean(self.generator_baseline)
        self.proposer_baseline = self._dist_mean(self.proposer_baseline)
        self.proposer_entropy_mu_ema = self._dist_mean(self.proposer_entropy_mu_ema)
        if self.solver_updater is not None:
            self.solver_baseline = self._dist_mean(self.solver_baseline)
            self.solver_updater.kl_coef = self._dist_mean(self.solver_updater.kl_coef)
        self.proposer_updater.kl_coef = self._dist_mean(self.proposer_updater.kl_coef)
        self.generator_updater.kl_coef = self._dist_mean(self.generator_updater.kl_coef)

    def __init__(self, config: GenerationSelfEvolvingConfig):
        self.cfg = config
        self.cfg.generator_update_rule = str(self.cfg.generator_update_rule or "reinforce").strip().lower()
        if self.cfg.generator_update_rule not in {"reinforce", "dpo", "grpo"}:
            raise ValueError(
                f"Unsupported generator_update_rule={self.cfg.generator_update_rule!r}. "
                "Expected one of: reinforce, dpo, grpo."
            )
        self.cfg.dpo_pair_selection = str(
            getattr(self.cfg, "dpo_pair_selection", "best_worst") or "best_worst"
        ).strip().lower()
        if self.cfg.dpo_pair_selection not in {"best_worst", "best_hard_negative"}:
            raise ValueError(
                f"Unsupported dpo_pair_selection={self.cfg.dpo_pair_selection!r}. "
                "Expected one of: best_worst, best_hard_negative."
            )
        self.cfg.generator_proxy_max_ratio = float(getattr(self.cfg, "generator_proxy_max_ratio", 1.0))
        if not (0.0 <= self.cfg.generator_proxy_max_ratio <= 1.0):
            raise ValueError(
                "generator_proxy_max_ratio must be in [0, 1]. "
                f"Got {self.cfg.generator_proxy_max_ratio}."
            )
        self.cfg.unicorn_target_difficulty = str(
            getattr(self.cfg, "unicorn_target_difficulty", "medium") or "medium"
        ).strip().lower()
        if self.cfg.unicorn_target_difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(
                "unicorn_target_difficulty must be one of: easy, medium, hard. "
                f"Got {self.cfg.unicorn_target_difficulty!r}."
            )
        self.cfg.unicorn_spec_max_retries = max(0, int(getattr(self.cfg, "unicorn_spec_max_retries", 2)))
        self.cfg.unicorn_reconstruction_buffer_size = max(
            1, int(getattr(self.cfg, "unicorn_reconstruction_buffer_size", 512))
        )
        self.cfg.unicorn_reconstruction_step_freq = max(
            1, int(getattr(self.cfg, "unicorn_reconstruction_step_freq", 1))
        )
        self.cfg.unicorn_reconstruction_updates_per_step = max(
            1, int(getattr(self.cfg, "unicorn_reconstruction_updates_per_step", 2))
        )
        self._setup_distributed()
        _set_global_seed(config.seed + self.rank, deterministic=config.deterministic)

        if not config.data_dir:
            raise ValueError("`data_dir` is required for generation self-evolving training")

        self.run_dir = self._build_run_dir()
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir = self.run_dir / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        self.iter_log_path = self.run_dir / "iter_log.jsonl"
        self.prompts_log_path = self.logs_dir / "proposer_prompts.jsonl"
        self.candidates_log_path = self.logs_dir / "generation_candidates.jsonl"
        self.rewards_log_path = self.logs_dir / "rewards.jsonl"
        self.policy_updates_log_path = self.logs_dir / "policy_updates.jsonl"
        self.dpo_pairs_log_path = self.logs_dir / "dpo_pairs.jsonl"
        self.unicorn_spec_log_path = self.logs_dir / "unicorn_spec_attempts.jsonl"
        self.unicorn_reconstruction_log_path = self.logs_dir / "unicorn_reconstruction.jsonl"
        self.summary_path = self.run_dir / "ablation_summary.json"
        self._save_run_metadata()

        self._blip3o_diffusion_pipe = None
        self._generation_api_name = None
        self._generation_api_obj = None
        self._generation_api_path = None
        self._warned_latent_fallback = False
        self.model, self.processor = self._load_model()
        fallback_dev = self.local_rank if self.distributed else config.cuda_device
        self.device = _infer_primary_device(self.model, fallback_cuda_device=fallback_dev)
        self._generation_api_name, self._generation_api_obj, self._generation_api_path, inspected = _find_generation_callable(
            _unwrap_model(self.model)
        )
        if self._generation_api_name is None and self._blip3o_diffusion_pipe is None:
            inspected_text = "; ".join(inspected[:10]) if inspected else "none"
            raise RuntimeError(
                "Loaded model does not expose a supported image generation API for "
                f"`{config.experiment_name}`.\n"
                f"model_name={config.model_name}\n"
                f"inspected_wrappers={inspected_text}\n"
                "Expected one of: generate_images(...), generate_image(...), or a BLIP3o diffusion-decoder pipeline.\n"
                "Use a generation-capable model (e.g., BLIP3o family) for generation/unified experiments."
            )
        if (
            _is_original_blip3o_model_name(config.model_name)
            and config.require_decoder_for_blip3o
            and self._blip3o_diffusion_pipe is None
        ):
            raise RuntimeError(
                "Original BLIP3o scientific runs require a working diffusion decoder pipeline, "
                "but decoder initialization failed.\n"
                f"model_name={config.model_name}\n"
                "Set BLIP3O_DIFFUSION_REPO to a checkpoint that contains `diffusion-decoder` "
                "(e.g., BLIP3o/BLIP3o-Model), ensure enough HF cache disk, and rerun.\n"
                "If you intentionally want debug-only latent visualization, pass "
                "`--allow_missing_decoder_for_blip3o --allow_latent_visualization_fallback`."
            )
        if self._generation_api_name is not None and self.is_main_process:
            print(
                f"[Generation] Using generation backend `{self._generation_api_name}` "
                f"from `{self._generation_api_path}` ({type(self._generation_api_obj).__name__})"
            )
        elif self._blip3o_diffusion_pipe is not None and self.is_main_process:
            print("[Generation] Using generation backend `diffusion_pipeline` (original BLIP3o decoder).")
            if not self.cfg.strict_require_generation_tokens:
                print(
                    "[Generation] Note: token traces may be unavailable with diffusion pipeline backend; "
                    "generator updates can be skipped when no completion trace is returned."
                )

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
        if not config.use_lora and self.is_main_process:
            import warnings
            warnings.warn(
                "[self-evolving] use_lora=False: embedding-based cycle-consistency "
                "reward uses the training model's own weights as a reference frame. "
                "Since base weights drift during full-model training, the reward "
                "signal is NOT a stable anchor. Consider enabling LoRA for a "
                "frozen-backbone reference or be aware of this limitation.",
                stacklevel=1,
            )
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
            self.train_model = torch.nn.parallel.DistributedDataParallel(self.model, **ddp_kwargs)

        self.solver_updater: Optional[RolePolicyUpdater] = None
        if config.enable_solver_updates and config.solver_update_freq > 0:
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

        if self.cfg.generator_update_rule == "dpo":
            self.generator_updater = TextPreferenceDPOUpdater(
                model=self.train_model,
                processor=self.processor,
                config=config,
                adapter_name="generator" if config.use_lora else None,
                reference_model=reference_model,
            )
        elif self.cfg.generator_update_rule == "grpo":
            self.generator_updater = TextGRPOUpdater(
                model=self.train_model,
                processor=self.processor,
                config=config,
                adapter_name="generator" if config.use_lora else None,
                reference_model=reference_model,
            )
        else:
            self.generator_updater = TextPolicyUpdater(
                model=self.train_model,
                processor=self.processor,
                config=config,
                adapter_name="generator" if config.use_lora else None,
                reference_model=reference_model,
            )

        self.generator_baseline = 0.0
        self.proposer_baseline = 0.0
        self.solver_baseline = 0.0
        self.proposer_entropy_mu_ema = float(config.prop_entropy_mu)
        self.start_step = max(0, int(config.start_step))

        self._metric_stats: Dict[str, Dict[str, float]] = {}
        self._policy_update_counts: Dict[str, int] = {"solver": 0, "proposer": 0, "generator": 0}
        self._generator_update_mode_counts: Dict[str, int] = {
            "token_trace": 0,
            "proxy_caption": 0,
            "skipped": 0,
        }
        self._unicorn_reconstruction_buffer = deque(
            maxlen=self.cfg.unicorn_reconstruction_buffer_size
        )
        self._unicorn_reconstruction_update_counts: Dict[str, int] = {
            "proposer": 0,
            "generator": 0,
            "skipped": 0,
        }
        self._diffusion_repair_count: int = 0
        self.wandb_run = self._init_wandb()

        loaded_resume_step = self._maybe_resume_state()
        if loaded_resume_step is not None:
            self.start_step = max(self.start_step, int(loaded_resume_step))

    def _init_wandb(self):
        if not self.is_main_process:
            return None
        mode = (self.cfg.wandb_mode or "disabled").strip().lower()
        if mode == "disabled":
            return None
        if not HAS_WANDB:
            print("[W&B] wandb package not available; disabling W&B logging.")
            return None

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
        _json_dump(self.run_dir / "config.json", dataclasses.asdict(self.cfg))
        _json_dump(self.run_dir / "git_info.json", _collect_git_info(repo_root))
        _json_dump(
            self.run_dir / "environment.json",
            {
                "python": os.sys.version,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "rank": self.rank,
                "world_size": self.world_size,
                "distributed": self.distributed,
            },
        )
        self._dist_barrier()

    def _append_jsonl(self, path: pathlib.Path, record: Dict):
        if not self.is_main_process:
            return
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

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

        step_dirs = [p for p in candidate.glob("step_*") if p.is_dir() and (p / "trainer_state.pt").exists()]
        if not step_dirs:
            raise FileNotFoundError(
                f"No checkpoint with trainer_state.pt found under resume_from path: {candidate}"
            )
        return sorted(step_dirs, key=lambda p: p.name)[-1]

    def _maybe_resume_state(self) -> Optional[int]:
        resume_dir = self._resolve_resume_dir()
        if resume_dir is None:
            return None

        trainable_path = resume_dir / "trainable_adapters.pt"
        if trainable_path.exists():
            try:
                adapter_state = torch.load(trainable_path, map_location="cpu")
                model_ref = _unwrap_model(self.model)
                missing, unexpected = model_ref.load_state_dict(adapter_state, strict=False)
                if self.is_main_process:
                    print(
                        f"[Generation] Restored trainable adapter weights from {trainable_path} "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})"
                    )
            except Exception as exc:
                if self.is_main_process:
                    print(f"[Generation] WARNING: failed to restore trainable adapter weights: {exc}")

        state_path = resume_dir / "trainer_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"trainer_state.pt not found in resume checkpoint: {resume_dir}")

        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")

        if self.solver_updater is not None:
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

        if "generator_updater" in state:
            self.generator_updater.load_state_dict(state["generator_updater"])
        elif "generator_opt" in state:
            self.generator_updater.load_state_dict(
                {
                    "optimizer": state["generator_opt"],
                    "kl_coef": state.get("generator_kl_coef"),
                    "step_id": state.get("generator_updater_step", state.get("step", 0)),
                }
            )

        self.solver_baseline = float(state.get("solver_baseline", self.solver_baseline))
        self.proposer_baseline = float(state.get("proposer_baseline", self.proposer_baseline))
        self.generator_baseline = float(state.get("generator_baseline", self.generator_baseline))
        self.proposer_entropy_mu_ema = float(
            state.get("proposer_entropy_mu_ema", self.proposer_entropy_mu_ema)
        )
        recon_counts = state.get("unicorn_reconstruction_update_counts")
        if isinstance(recon_counts, dict):
            merged_counts = dict(self._unicorn_reconstruction_update_counts)
            for key in ("proposer", "generator", "skipped"):
                if key in recon_counts:
                    try:
                        merged_counts[key] = int(recon_counts[key])
                    except Exception:
                        pass
            self._unicorn_reconstruction_update_counts = merged_counts

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
            print(f"[Generation] Resumed trainer state from: {state_path} (step={restored_step})")
            _json_dump(
                self.run_dir / "resume_info.json",
                {
                    "resume_from": str(resume_dir),
                    "restored_step": restored_step,
                    "restored_solver_baseline": self.solver_baseline,
                    "restored_proposer_baseline": self.proposer_baseline,
                    "restored_generator_baseline": self.generator_baseline,
                },
            )
        self._dist_barrier()
        return restored_step

    def _trainer_state_dict(self, step: int) -> Dict:
        state = {
            "step": int(step),
            "proposer_updater": self.proposer_updater.state_dict(),
            "generator_updater": self.generator_updater.state_dict(),
            "proposer_baseline": float(self.proposer_baseline),
            "generator_baseline": float(self.generator_baseline),
            "proposer_entropy_mu_ema": float(self.proposer_entropy_mu_ema),
            "unicorn_reconstruction_update_counts": dict(self._unicorn_reconstruction_update_counts),
            "py_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if self.solver_updater is not None:
            state["solver_updater"] = self.solver_updater.state_dict()
            state["solver_baseline"] = float(self.solver_baseline)
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
            return (
                (step_dir / "solver").is_dir()
                and (step_dir / "proposer").is_dir()
                and (step_dir / "generator").is_dir()
            )
        return (step_dir / "model").is_dir()

    def _list_complete_checkpoints(self) -> List[pathlib.Path]:
        checkpoints = [p for p in self.run_dir.glob("step_*") if self._is_complete_checkpoint(p)]
        return sorted(checkpoints, key=lambda p: p.name)

    def _load_model(self):
        if self.cfg.use_lora and (not HAS_PEFT or LoraConfig is None or get_peft_model is None or TaskType is None):
            raise RuntimeError("PEFT is required for role-specific LoRA adapters")

        dtype = _safe_dtype(self.cfg.dtype)
        attn_impl = _resolve_attn_implementation(self.cfg.attn_implementation)

        if self.distributed:
            if self.cfg.device_map == "auto" and self.is_main_process:
                print("[Generation] Distributed run detected; overriding device_map=auto to per-rank single-device mapping.")
            device_map = {"": self.local_rank} if torch.cuda.is_available() else "cpu"
        elif self.cfg.device_map == "single":
            device_map = {"": self.cfg.cuda_device} if torch.cuda.is_available() else "cpu"
        elif self.cfg.device_map == "cpu":
            device_map = "cpu"
        else:
            device_map = "auto"

        from transformers import AutoProcessor

        # Load model using native BLIP3o classes
        model = _load_blip3o_model(
            self.cfg.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
        )

        # Load processor
        processor = AutoProcessor.from_pretrained(self.cfg.model_name, trust_remote_code=True)

        if self.is_main_process:
            print(f"[Generation] Loaded model: dtype={dtype}, device_map={device_map}, attn_implementation={attn_impl or 'default'}")

        if self.cfg.use_lora:
            lcfg = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=list(self.cfg.lora_target_modules),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lcfg)
            if hasattr(model, "add_adapter"):
                try:
                    model.add_adapter("proposer", lcfg)
                except Exception:
                    pass
                try:
                    model.add_adapter("generator", lcfg)
                except Exception:
                    pass

            for name, param in model.named_parameters():
                if "lora_" in name and (
                    ".default." in name or ".proposer." in name or ".generator." in name
                ):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

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
                    "[Generation] Forcing gradient checkpointing use_reentrant=False "
                    "(DDP/LoRA compatibility)."
                )
            gc_use_reentrant = False
        if self.is_main_process:
            print(
                "[Generation] Gradient checkpointing config: "
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
                        f"[Generation] Enabled gradient checkpointing "
                        f"(use_reentrant={gc_use_reentrant})."
                    )
            except TypeError:
                # Older transformers versions don't accept gradient_checkpointing_kwargs.
                # In DDP+LoRA, we avoid silently enabling unknown/default reentrant mode.
                if self.distributed or self.cfg.use_lora:
                    if self.is_main_process:
                        print(
                            "[Generation] Skipping gradient checkpointing: "
                            "current transformers build does not expose "
                            "gradient_checkpointing_kwargs (cannot guarantee "
                            "use_reentrant=False safely under DDP/LoRA)."
                        )
                else:
                    model.gradient_checkpointing_enable()
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    if self.is_main_process:
                        print("[Generation] Enabled gradient checkpointing.")
            except Exception:
                pass
        elif self.is_main_process and not gc_enabled:
            print("[Generation] Gradient checkpointing disabled via SE_USE_GRADIENT_CHECKPOINTING=0.")

        is_original_blip3o = _is_original_blip3o_model_name(self.cfg.model_name)
        if is_original_blip3o:
            try:
                pipeline_device = (
                    torch.device(f"cuda:{self.local_rank if self.distributed else self.cfg.cuda_device}")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                pipe_encoder = _resolve_multimodal_encoder_for_pipeline(model)
                self._blip3o_diffusion_pipe = _build_original_blip3o_diffusion_pipeline(
                    self.cfg.model_name,
                    multimodal_encoder=pipe_encoder,
                    processor=processor,
                    torch_dtype=dtype,
                    device=pipeline_device,
                )
            except Exception as exc:
                if self.is_main_process:
                    print(
                        "[Generation] WARNING: failed to initialize original BLIP3o diffusion "
                        f"decoder pipeline: {repr(exc)}"
                    )

        model.eval()
        return model, processor

    def _sample_image_for_step(self, step: int) -> Tuple[Image.Image, Dict]:
        if self.distributed:
            global_offset = (step - 1) * self.world_size + self.rank
            shuffled_idx = self.pool.indices[global_offset % len(self.pool.indices)]
        else:
            shuffled_idx = self.pool.indices[(step - 1) % len(self.pool.indices)]
        return self.pool.get_image(shuffled_idx)

    # ------------------------------------------------------------------
    # Self-model embedding helpers (text & image-text similarity)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _text_embedding(self, text: str) -> torch.Tensor:
        """Get a mean-pooled hidden-state embedding for *text* from the LLM backbone.

        Uses the base model (no adapter) so the representation is stable
        and doesn't drift with adapter training.  Returns a 1-D float tensor
        on the model's device.

        WARNING: When ``use_lora=False`` the base weights ARE the training
        weights, so embeddings drift during training.  The cycle-consistency
        signal will still work but is no longer a *stable* reference frame.
        """
        inputs = _prepare_text_inputs(self.processor, self.device, text)
        with use_adapter(self.model, None):
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )
        # Last hidden state: [1, seq_len, hidden_dim]
        hidden = outputs.hidden_states[-1]
        # Mean-pool over non-padding positions
        mask = inputs.get("attention_mask")
        if mask is not None:
            mask = mask.unsqueeze(-1).to(hidden.dtype)
            embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            embedding = hidden.mean(dim=1)
        return torch.nn.functional.normalize(embedding.squeeze(0), dim=-1)

    @torch.no_grad()
    def _image_text_embedding(self, image: Image.Image, text: str) -> torch.Tensor:
        """Get a mean-pooled hidden-state embedding for an *image+text* pair.

        The image tokens and text tokens are jointly encoded, giving a
        representation that captures the interaction between modalities.
        """
        chat_text = _build_chat_text(self.processor, image, text)
        inputs = _prepare_mm_inputs(self.processor, self.device, image, chat_text, model=self.model)
        # Filter out generate()-only keys for forward() call
        forward_inputs = {k: v for k, v in inputs.items()
                          if k not in ("images", "image_sizes")}
        with use_adapter(self.model, None):
            outputs = self.model(
                **forward_inputs,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = outputs.hidden_states[-1]
        mask = inputs.get("attention_mask")
        if mask is not None:
            mask = mask.unsqueeze(-1).to(hidden.dtype)
            embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            embedding = hidden.mean(dim=1)
        return torch.nn.functional.normalize(embedding.squeeze(0), dim=-1)

    def _embedding_similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between two texts using the model's own embeddings."""
        emb1 = self._text_embedding(text1)
        emb2 = self._text_embedding(text2)
        return float(torch.dot(emb1, emb2).item())

    def _image_text_similarity(self, image: Image.Image, text: str) -> float:
        """Cosine similarity between image-text pair and text-only embeddings.

        Compares how well the generated *image* (when combined with a
        neutral caption prompt) aligns with the original generation *text*.
        This gives a richer cycle-consistency signal than caption-level
        Jaccard overlap because it operates in the model's semantic space.
        """
        # Image side: encode "describe this image" + image jointly
        img_emb = self._image_text_embedding(image, "Describe this image briefly.")
        # Text side: encode the original prompt as pure text
        txt_emb = self._text_embedding(text)
        return float(torch.dot(img_emb, txt_emb).item())

    def _generate(
        self,
        image: Image.Image,
        prompt: str,
        adapter_name: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        chat_text = _build_chat_text(self.processor, image, prompt)
        inputs = _prepare_mm_inputs(self.processor, self.device, image, chat_text, model=self.model)

        has_image_feats = ("pixel_values" in inputs) or ("images" in inputs)
        if has_image_feats and "input_ids" in inputs:
            image_token_ids = _collect_image_token_ids(self.model)
            token_count = _count_image_tokens_in_inputs(inputs["input_ids"], image_token_ids)
            if token_count == 0:
                mm_proc = getattr(self.processor, "multimodal_processor", None)
                if mm_proc is not None and hasattr(mm_proc, "apply_chat_template"):
                    try:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ]
                        chat_text_mm = mm_proc.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        inputs = mm_proc(
                            text=[chat_text_mm],
                            images=[image],
                            return_tensors="pt",
                            padding=True,
                        ).to(self.device)
                    except Exception:
                        pass

        gen_inputs = _adapt_mm_generate_inputs(self.model, dict(inputs))

        # Fail fast on corrupted / incompatible token IDs before hitting
        # device-side asserts inside HIP kernels.
        model_cfg = getattr(_unwrap_model(self.model), "config", None)
        vocab_size = getattr(model_cfg, "vocab_size", None)
        input_ids = gen_inputs.get("input_ids")
        if torch.is_tensor(input_ids) and isinstance(vocab_size, int) and vocab_size > 0:
            min_id = int(input_ids.min().item())
            max_id = int(input_ids.max().item())
            if min_id < 0 or max_id >= vocab_size:
                raise RuntimeError(
                    "Invalid token ids prepared for generation: "
                    f"min_id={min_id}, max_id={max_id}, vocab_size={vocab_size}. "
                    "This indicates tokenizer/model mismatch in multimodal input preparation."
                )

        # Extract pad_token_id robustly — processor may BE the tokenizer
        _tok = _extract_tokenizer_from_processor(self.processor)
        _pad_id = getattr(_tok, "eos_token_id", None) if _tok is not None else None

        def _run_generate(curr_inputs: Dict[str, torch.Tensor]):
            base_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "pad_token_id": _pad_id,
                # Stabilize multinomial sampling on mixed precision by
                # sanitizing invalid logits and re-normalizing probabilities.
                "remove_invalid_values": True,
                "renormalize_logits": True,
            }
            try:
                return self.model.generate(**curr_inputs, **base_kwargs)
            except TypeError as exc:
                msg = str(exc)
                if ("remove_invalid_values" in msg) or ("renormalize_logits" in msg):
                    base_kwargs.pop("remove_invalid_values", None)
                    base_kwargs.pop("renormalize_logits", None)
                    return self.model.generate(**curr_inputs, **base_kwargs)
                raise

        with torch.no_grad():
            with use_adapter(self.model, adapter_name):
                try:
                    outputs = _run_generate(gen_inputs)
                except ValueError as exc:
                    unused = _parse_unused_model_kwargs_from_error(exc)
                    if not unused:
                        raise
                    retry_inputs = dict(gen_inputs)
                    if "images" in unused and "images" in retry_inputs and "pixel_values" not in retry_inputs:
                        retry_inputs["pixel_values"] = retry_inputs["images"]
                    if "pixel_values" in unused and "pixel_values" in retry_inputs and "images" not in retry_inputs:
                        retry_inputs["images"] = retry_inputs["pixel_values"]
                    if "image_grid_thw" in unused and "image_grid_thw" in retry_inputs and "grid_thw" not in retry_inputs:
                        retry_inputs["grid_thw"] = retry_inputs["image_grid_thw"]
                    if "grid_thw" in unused and "grid_thw" in retry_inputs and "image_grid_thw" not in retry_inputs:
                        retry_inputs["image_grid_thw"] = retry_inputs["grid_thw"]
                    for key in unused:
                        retry_inputs.pop(key, None)
                    if retry_inputs == gen_inputs:
                        raise
                    outputs = _run_generate(retry_inputs)
                    gen_inputs = retry_inputs

        input_len = gen_inputs["input_ids"].shape[1] if "input_ids" in gen_inputs else 0
        completion_ids = outputs[0, input_len:]
        text = _decode_tokens(self.processor, completion_ids)
        return text.strip()

    def _caption_image(self, image: Image.Image) -> str:
        caption = self._generate(
            image=image,
            prompt=SOURCE_CAPTION_PROMPT,
            adapter_name="default" if self.cfg.use_lora else None,
            max_new_tokens=self.cfg.max_new_tokens_caption,
            temperature=max(0.2, min(self.cfg.temp, 0.8)),
            top_p=1.0,
        )
        caption = " ".join(caption.split())
        if not caption:
            caption = "An image with multiple visual elements."
        return caption

    def _propose_generation_spec(
        self,
        image: Image.Image,
        *,
        proposer_prompt: Optional[str] = None,
    ) -> GenerationSpec:
        prompt_text = str(proposer_prompt or GEN_PROMPT_TEMPLATE)
        raw = self._generate(
            image=image,
            prompt=prompt_text,
            adapter_name="proposer" if self.cfg.use_lora else None,
            max_new_tokens=self.cfg.max_new_tokens_proposer,
            temperature=self.cfg.temp,
            top_p=self.cfg.top_p,
        )
        spec = _parse_generation_spec(raw)
        return spec

    def _is_question_like_prompt(self, text: str) -> bool:
        s = " ".join((text or "").split())
        if not s:
            return True
        if "?" in s:
            return True
        lower = s.lower()
        starters = (
            "what ",
            "which ",
            "who ",
            "where ",
            "when ",
            "why ",
            "how ",
            "is ",
            "are ",
            "do ",
            "does ",
            "did ",
            "can ",
            "could ",
            "would ",
            "should ",
        )
        return lower.startswith(starters)

    def _strip_instruction_prefix(self, text: str) -> str:
        s = " ".join((text or "").split()).strip()
        if not s:
            return s

        s_l = s.lower()
        fixed_prefixes = (
            "create an image variation of:",
            "create an image of:",
            "generate an image of:",
            "generate image of:",
            "draw an image of:",
            "an image of ",
            "image of ",
        )
        for prefix in fixed_prefixes:
            if s_l.startswith(prefix):
                s = s[len(prefix):].strip()
                s_l = s.lower()
                break

        verb_prefixes = (
            "describe ",
            "explain ",
            "illustrate ",
            "show ",
            "depict ",
            "create ",
            "generate ",
            "draw ",
            "compare ",
            "analyze ",
            "summarize ",
            "outline ",
        )
        for prefix in verb_prefixes:
            if s_l.startswith(prefix):
                s = s[len(prefix):].strip()
                s_l = s.lower()
                break

        # Drop common instruction wrappers that make prompts awkward.
        s = re.sub(r"^(in (this|the) (image|figure|diagram),?\s*)", "", s, flags=re.IGNORECASE).strip()
        s = s.rstrip(" .")
        return s

    def _question_to_fact_label(self, question: str) -> str:
        q = " ".join((question or "").replace("?", "").split()).strip()
        if not q:
            return ""
        q_l = q.lower()

        direct_rules = (
            ("how many ", "count of "),
            ("how much ", "amount of "),
            ("how long ", "length of "),
            ("how far ", "distance of "),
            ("how high ", "height of "),
            ("what is the ", "the "),
            ("what was the ", "the "),
            ("what are the ", "the "),
            ("what were the ", "the "),
            ("what is ", ""),
            ("what was ", ""),
            ("what are ", ""),
            ("what were ", ""),
            ("which ", ""),
            ("identify ", ""),
            ("name the ", "the "),
            ("is there ", "presence of "),
            ("are there ", "presence of "),
            ("does ", "whether "),
            ("do ", "whether "),
            ("did ", "whether "),
            ("can ", "whether "),
        )
        for src, dst in direct_rules:
            if q_l.startswith(src):
                q = (dst + q[len(src):]).strip()
                break

        q = self._strip_instruction_prefix(q)
        q = q.strip(" :.")
        if not q:
            q = "visual detail"
        return q

    def _compose_generation_prompt(
        self,
        raw_prompt: str,
        source_caption: str,
        qa_pairs: Tuple[GenerationQAPair, ...],
    ) -> str:
        raw = " ".join((raw_prompt or "").split())
        caption = " ".join((source_caption or "").split())
        prompt_was_question = self._is_question_like_prompt(raw)

        if raw and not prompt_was_question:
            subject = raw.rstrip(".")
        elif caption:
            subject = caption.rstrip(".")
        elif raw:
            # Last resort: use raw text but make it declarative.
            subject = raw.replace("?", "").rstrip(".")
        else:
            subject = "a coherent scene with clear objects and relationships"

        subject = self._strip_instruction_prefix(subject)
        if self._is_question_like_prompt(subject):
            subject = "a scene matching the verified visual facts"
        if not subject:
            subject = "a coherent scene with clear objects and relationships"

        facts: List[str] = []
        for qa in qa_pairs[:3]:
            q = " ".join((qa.question or "").replace("?", "").split())
            e = " ".join((qa.expected or "").replace("?", "").split())
            if q and e:
                fact_label = self._question_to_fact_label(q)
                facts.append(f"{fact_label}: {e}")
            elif e:
                facts.append(e)

        parts: List[str] = [f"A detailed, realistic scene showing {subject}."]
        if facts:
            parts.append(
                "Visible details include " + "; ".join(facts) + "."
            )
        parts.append(
            "The composition should be coherent, with readable text or numbers when present."
        )

        composed = " ".join(parts).replace("?", "")
        composed = " ".join(composed.split())
        max_words = 96
        words = composed.split()
        if len(words) > max_words:
            composed = " ".join(words[:max_words])
        return composed

    def _sanitize_and_score_spec(
        self,
        spec: GenerationSpec,
        source_caption: str = "",
    ) -> Tuple[GenerationSpec, float, Dict[str, float]]:
        filtered: List[GenerationQAPair] = []
        seen_questions = set()
        valid_count = 0

        for qa in spec.qa_pairs:
            q = " ".join((qa.question or "").split())
            e = " ".join((qa.expected or "").split())
            if q and not q.endswith("?"):
                q = f"{q}?"

            q_words = len(_tokenize_words(q))
            e_words = len(_tokenize_words(e))
            is_valid = bool(q and e and q_words <= self.cfg.max_question_words and 1 <= e_words <= self.cfg.max_expected_words)
            if not is_valid:
                continue

            q_key = normalize_answer(q)
            if q_key in seen_questions:
                continue
            seen_questions.add(q_key)
            valid_count += 1
            filtered.append(GenerationQAPair(question=q, expected=e))

        filtered = filtered[:3]
        qa_count = len(filtered)
        raw_count = len(spec.qa_pairs)

        count_score = min(1.0, qa_count / float(max(1, self.cfg.min_spec_qa_pairs)))
        validity_score = qa_count / float(max(1, raw_count))
        uniqueness_score = len({normalize_answer(qa.question) for qa in filtered}) / float(max(1, qa_count))
        all_yes_no = qa_count > 0 and all(_yes_no_polarity(qa.expected) != 0 for qa in filtered)
        yes_no_penalty = 0.2 if all_yes_no and qa_count >= self.cfg.min_spec_qa_pairs else 0.0

        quality = 0.5 * count_score + 0.3 * validity_score + 0.2 * uniqueness_score - yes_no_penalty
        quality = float(max(0.0, min(1.0, quality)))

        prompt_was_question = self._is_question_like_prompt(spec.prompt)
        composed_prompt = self._compose_generation_prompt(
            raw_prompt=spec.prompt,
            source_caption=source_caption,
            qa_pairs=tuple(filtered),
        )

        sanitized = GenerationSpec(
            prompt=composed_prompt,
            qa_pairs=tuple(filtered),
            raw_output=spec.raw_output,
            fallback_used=spec.fallback_used or (qa_count < self.cfg.min_spec_qa_pairs),
        )
        details = {
            "raw_qa_count": float(raw_count),
            "filtered_qa_count": float(qa_count),
            "count_score": float(count_score),
            "validity_score": float(validity_score),
            "uniqueness_score": float(uniqueness_score),
            "yes_no_penalty": float(yes_no_penalty),
            "spec_quality": float(quality),
            "prompt_was_question": 1.0 if prompt_was_question else 0.0,
            "raw_prompt_words": float(len(_tokenize_words(spec.prompt))),
            "sanitized_prompt_words": float(len(_tokenize_words(composed_prompt))),
        }
        return sanitized, quality, details

    def _unicorn_spec_attempt(
        self,
        image: Image.Image,
        source_caption: str,
        proposer_prompt: str,
        *,
        attempt_idx: int,
        max_attempts: int,
        step: Optional[int] = None,
        verbose: bool = False,
        force_alignment_eval: bool = False,
    ) -> Dict[str, object]:
        raw_spec = self._propose_generation_spec(image=image, proposer_prompt=proposer_prompt)
        if raw_spec.fallback_used and source_caption:
            raw_spec = GenerationSpec(
                prompt=f"Create an image variation of: {source_caption}",
                qa_pairs=raw_spec.qa_pairs,
                raw_output=raw_spec.raw_output,
                fallback_used=True,
            )

        sanitized, spec_quality, spec_quality_details = self._sanitize_and_score_spec(
            raw_spec,
            source_caption=source_caption,
        )
        qa_count = len(sanitized.qa_pairs)
        min_pairs = max(1, int(self.cfg.min_spec_qa_pairs))
        min_quality = float(getattr(self.cfg, "unicorn_spec_min_quality", 0.55))
        min_alignment = float(getattr(self.cfg, "unicorn_spec_min_alignment", 0.55))

        # Run solver-alignment judge only when quality pre-gate passes or when forced.
        should_eval_alignment = bool(
            force_alignment_eval
            or (qa_count >= min_pairs and spec_quality >= min_quality)
            or (attempt_idx >= (max_attempts - 1))
        )
        alignment = 0.0
        contradiction = 0.0
        if should_eval_alignment and qa_count > 0:
            alignment, contradiction, _ = self._score_spec(
                image=image,
                qa_pairs=sanitized.qa_pairs,
                step=step if verbose else None,
                verbose=False,
            )

        reject_reason = ""
        if qa_count < min_pairs:
            reject_reason = "insufficient_qa_pairs"
        elif spec_quality < min_quality:
            reject_reason = "low_spec_quality"
        elif should_eval_alignment and alignment < min_alignment:
            reject_reason = "low_self_alignment"

        accepted = (reject_reason == "")
        combined_score = 0.65 * float(spec_quality) + 0.35 * float(alignment)
        return {
            "spec": sanitized,
            "spec_quality": float(spec_quality),
            "spec_quality_details": spec_quality_details,
            "alignment": float(alignment),
            "contradiction": float(contradiction),
            "accepted": bool(accepted),
            "reject_reason": reject_reason,
            "attempt_idx": int(attempt_idx),
            "max_attempts": int(max_attempts),
            "combined_score": float(combined_score),
            "proposer_prompt": proposer_prompt,
            "fallback_used": bool(sanitized.fallback_used),
        }

    def _select_generation_spec_with_unicorn(
        self,
        image: Image.Image,
        source_caption: str,
        *,
        step: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[GenerationSpec, float, Dict[str, float], Dict[str, object]]:
        unicorn_enabled = bool(getattr(self.cfg, "unicorn_generation_enabled", True))
        if not unicorn_enabled:
            raw_spec = self._propose_generation_spec(
                image=image,
                proposer_prompt=GEN_PROMPT_TEMPLATE,
            )
            if raw_spec.fallback_used and source_caption:
                raw_spec = GenerationSpec(
                    prompt=f"Create an image variation of: {source_caption}",
                    qa_pairs=raw_spec.qa_pairs,
                    raw_output=raw_spec.raw_output,
                    fallback_used=True,
                )
            spec, quality, details = self._sanitize_and_score_spec(
                raw_spec,
                source_caption=source_caption,
            )
            details.update(
                {
                    "unicorn_enabled": 0.0,
                    "unicorn_rejection_enabled": 0.0,
                    "unicorn_spec_attempts": 1.0,
                    "unicorn_spec_retries_used": 0.0,
                    "unicorn_spec_alignment": 0.0,
                    "unicorn_spec_contradiction": 0.0,
                    "unicorn_spec_selected_accepted": 1.0,
                }
            )
            return spec, float(quality), details, {
                "enabled": False,
                "rejection_enabled": False,
                "attempts": 1,
                "retries_used": 0,
                "selected_accepted": True,
                "selected_reject_reason": "",
                "selected_alignment": 0.0,
                "selected_contradiction": 0.0,
                "selected_quality": float(quality),
                "attempt_logs": [
                    {
                        "attempt_idx": 0,
                        "max_attempts": 1,
                        "accepted": True,
                        "reject_reason": "",
                        "spec_quality": float(quality),
                        "alignment": 0.0,
                        "contradiction": 0.0,
                        "combined_score": float(quality),
                        "fallback_used": bool(spec.fallback_used),
                    }
                ],
            }

        rejection_enabled = bool(getattr(self.cfg, "unicorn_spec_rejection_enabled", True))
        retries = int(getattr(self.cfg, "unicorn_spec_max_retries", 0))
        max_attempts = 1 + (retries if (unicorn_enabled and rejection_enabled) else 0)
        max_attempts = max(1, max_attempts)

        target_diff = str(getattr(self.cfg, "unicorn_target_difficulty", "medium") or "medium")
        proposer_prompt = (
            build_generation_spec_prompt(target_difficulty=target_diff)
            if unicorn_enabled
            else GEN_PROMPT_TEMPLATE
        )

        attempts: List[Dict[str, object]] = []
        selected: Optional[Dict[str, object]] = None
        best_seen: Optional[Dict[str, object]] = None

        for attempt_idx in range(max_attempts):
            force_alignment_eval = (attempt_idx >= max_attempts - 1)
            attempt = self._unicorn_spec_attempt(
                image=image,
                source_caption=source_caption,
                proposer_prompt=proposer_prompt,
                attempt_idx=attempt_idx,
                max_attempts=max_attempts,
                step=step,
                verbose=verbose,
                force_alignment_eval=force_alignment_eval,
            )
            attempts.append(attempt)

            if best_seen is None or float(attempt["combined_score"]) > float(best_seen["combined_score"]):
                best_seen = attempt

            if bool(attempt["accepted"]):
                selected = attempt
                break

            if attempt_idx < (max_attempts - 1):
                retry_reason = str(attempt["reject_reason"] or "spec did not meet quality gate")
                proposer_prompt = build_generation_spec_retry_prompt(
                    previous_prompt=str(attempt["spec"].prompt),
                    reason=retry_reason,
                    target_difficulty=target_diff,
                )

        if selected is None:
            selected = best_seen if best_seen is not None else attempts[-1]

        retries_used = max(0, len(attempts) - 1)
        selected_quality = float(selected["spec_quality"])
        selected_alignment = float(selected["alignment"])
        selected_contradiction = float(selected["contradiction"])
        selected_spec: GenerationSpec = selected["spec"]

        details = dict(selected["spec_quality_details"])
        details.update(
            {
                "unicorn_enabled": 1.0 if unicorn_enabled else 0.0,
                "unicorn_rejection_enabled": 1.0 if rejection_enabled else 0.0,
                "unicorn_spec_attempts": float(len(attempts)),
                "unicorn_spec_retries_used": float(retries_used),
                "unicorn_spec_alignment": float(selected_alignment),
                "unicorn_spec_contradiction": float(selected_contradiction),
                "unicorn_spec_selected_accepted": 1.0 if bool(selected.get("accepted", False)) else 0.0,
            }
        )

        unicorn_meta = {
            "enabled": bool(unicorn_enabled),
            "rejection_enabled": bool(rejection_enabled),
            "attempts": len(attempts),
            "retries_used": retries_used,
            "selected_accepted": bool(selected.get("accepted", False)),
            "selected_reject_reason": str(selected.get("reject_reason", "")),
            "selected_alignment": selected_alignment,
            "selected_contradiction": selected_contradiction,
            "selected_quality": selected_quality,
            "attempt_logs": [
                {
                    "attempt_idx": int(a["attempt_idx"]),
                    "max_attempts": int(a["max_attempts"]),
                    "accepted": bool(a["accepted"]),
                    "reject_reason": str(a["reject_reason"]),
                    "spec_quality": float(a["spec_quality"]),
                    "alignment": float(a["alignment"]),
                    "contradiction": float(a["contradiction"]),
                    "combined_score": float(a["combined_score"]),
                    "fallback_used": bool(a["fallback_used"]),
                }
                for a in attempts
            ],
        }
        return selected_spec, selected_quality, details, unicorn_meta

    def _enqueue_unicorn_reconstruction_tasks(
        self,
        *,
        step: int,
        image: Image.Image,
        spec: GenerationSpec,
        best: Dict[str, object],
        spec_quality: float,
    ) -> int:
        if not bool(getattr(self.cfg, "unicorn_reconstruction_sft_enabled", True)):
            return 0
        if spec_quality < float(getattr(self.cfg, "unicorn_reconstruction_min_quality", 0.55)):
            return 0

        enqueued = 0
        target_diff = str(getattr(self.cfg, "unicorn_target_difficulty", "medium") or "medium")
        if bool(getattr(self.cfg, "unicorn_reconstruction_enable_proposer", True)):
            proposer_completion = str(spec.raw_output or "").strip()
            if proposer_completion:
                self._unicorn_reconstruction_buffer.append(
                    {
                        "role": "proposer",
                        "step": int(step),
                        "prompt": build_generation_spec_prompt(target_difficulty=target_diff),
                        "completion": proposer_completion,
                        "image": image,
                        "completion_token_ids": None,
                        "task": "spec_reconstruction",
                    }
                )
                enqueued += 1

        if bool(getattr(self.cfg, "unicorn_reconstruction_enable_generator", True)):
            completion = str(best.get("policy_completion", "")).strip()
            completion_token_ids = best.get("policy_completion_ids")
            if not isinstance(completion_token_ids, list):
                completion_token_ids = None
            prompt = str(best.get("policy_prompt", spec.prompt))
            update_image: Optional[Image.Image] = None
            task = "generator_trace_reconstruction"

            if not completion:
                best_image = best.get("image")
                if isinstance(best_image, Image.Image):
                    proxy_completion = self._proxy_generator_completion(best_image)
                    if proxy_completion:
                        completion = proxy_completion
                        prompt = GENERATOR_PROXY_CAPTION_PROMPT
                        update_image = best_image
                        completion_token_ids = None
                        task = "generator_proxy_reconstruction"
            if completion:
                self._unicorn_reconstruction_buffer.append(
                    {
                        "role": "generator",
                        "step": int(step),
                        "prompt": prompt,
                        "completion": completion,
                        "image": update_image,
                        "completion_token_ids": completion_token_ids,
                        "task": task,
                    }
                )
                enqueued += 1

        return enqueued

    def _unicorn_has_task_for_role(self, role: str) -> bool:
        return any(str(task.get("role", "")) == role for task in self._unicorn_reconstruction_buffer)

    def _unicorn_pop_task_for_role(self, role: str) -> Optional[Dict[str, object]]:
        if not self._unicorn_reconstruction_buffer:
            return None
        retained: List[Dict[str, object]] = []
        selected: Optional[Dict[str, object]] = None
        while self._unicorn_reconstruction_buffer:
            item = self._unicorn_reconstruction_buffer.pop()
            if selected is None and str(item.get("role", "")) == role:
                selected = item
                break
            retained.append(item)
        while retained:
            self._unicorn_reconstruction_buffer.append(retained.pop())
        return selected

    def _run_unicorn_reconstruction_sft(self, step: int) -> Dict[str, object]:
        info: Dict[str, object] = {
            "enabled": bool(getattr(self.cfg, "unicorn_reconstruction_sft_enabled", True)),
            "queued": int(len(self._unicorn_reconstruction_buffer)),
            "attempted_updates": 0,
            "applied_updates": 0,
            "skipped_updates": 0,
            "update_records": [],
        }
        if not bool(info["enabled"]):
            return info
        if step % int(getattr(self.cfg, "unicorn_reconstruction_step_freq", 1)) != 0:
            return info
        if len(self._unicorn_reconstruction_buffer) == 0:
            return info

        max_updates = int(getattr(self.cfg, "unicorn_reconstruction_updates_per_step", 2))
        for update_idx in range(max_updates):
            role_order = ("proposer", "generator") if (update_idx % 2 == 0) else ("generator", "proposer")
            selected_role: Optional[str] = None
            for role in role_order:
                local_has_role = self._unicorn_has_task_for_role(role)
                has_role_all = local_has_role
                if self.distributed and dist.is_initialized():
                    has_role_all = self._dist_all_bool(local_has_role)
                if has_role_all:
                    selected_role = role
                    break
            if selected_role is None:
                break

            info["attempted_updates"] += 1
            task = self._unicorn_pop_task_for_role(selected_role)
            if task is None:
                info["skipped_updates"] += 1
                self._unicorn_reconstruction_update_counts["skipped"] += 1
                info["update_records"].append(
                    {
                        "role": selected_role,
                        "task": "unknown",
                        "skipped": True,
                        "reason": "role_task_missing_local",
                    }
                )
                continue
            role = str(task.get("role", ""))
            completion = str(task.get("completion", "")).strip()
            prompt = str(task.get("prompt", ""))
            update_image = task.get("image")
            completion_token_ids = task.get("completion_token_ids")
            if not isinstance(completion_token_ids, list):
                completion_token_ids = None

            local_ready = bool(prompt and completion)
            can_update, skip_reason = self._distributed_update_ready(
                local_ready,
                None if local_ready else "empty_prompt_or_completion",
                peer_reason="distributed_peer_unicorn_skip",
            )
            if not can_update:
                info["skipped_updates"] += 1
                self._unicorn_reconstruction_update_counts["skipped"] += 1
                info["update_records"].append(
                    {"role": role, "task": task.get("task"), "skipped": True, "reason": skip_reason}
                )
                continue

            stats: Optional[Dict[str, float]] = None
            if role == "proposer":
                if not isinstance(update_image, Image.Image):
                    skip_reason = "proposer_task_missing_image"
                else:
                    stats = self.proposer_updater.step(
                        image=update_image,
                        prompt=prompt,
                        completion=completion,
                        reward=1.0,
                        baseline=0.0,
                        device=self.device,
                    )
            elif role == "generator":
                sft_fn = getattr(self.generator_updater, "sft_step", None)
                if not callable(sft_fn):
                    skip_reason = "generator_updater_missing_sft_step"
                else:
                    stats = sft_fn(
                        prompt=prompt,
                        completion=completion,
                        device=self.device,
                        image=update_image if isinstance(update_image, Image.Image) else None,
                        completion_token_ids=completion_token_ids,
                    )
            else:
                skip_reason = f"unsupported_unicorn_role:{role}"

            if stats is None:
                info["skipped_updates"] += 1
                self._unicorn_reconstruction_update_counts["skipped"] += 1
                info["update_records"].append(
                    {"role": role, "task": task.get("task"), "skipped": True, "reason": skip_reason}
                )
                continue

            did_step = bool(stats.get("did_step", True))
            if did_step:
                info["applied_updates"] += 1
                self._policy_update_counts[role] = self._policy_update_counts.get(role, 0) + 1
                self._unicorn_reconstruction_update_counts[role] = (
                    self._unicorn_reconstruction_update_counts.get(role, 0) + 1
                )
            else:
                info["skipped_updates"] += 1
                self._unicorn_reconstruction_update_counts["skipped"] += 1

            info["update_records"].append(
                {
                    "role": role,
                    "task": task.get("task"),
                    "did_step": did_step,
                    "stats": stats,
                }
            )

            self._append_jsonl(
                self.unicorn_reconstruction_log_path,
                {
                    "step": int(step),
                    "role": role,
                    "task": task.get("task"),
                    "did_step": did_step,
                    "stats": stats,
                },
            )

        self._sync_state_scalars()
        return info

    def _generate_image_candidate(self, inputs: str, **kwargs) -> Dict[str, Any]:
        prompt = inputs
        api_name = self._generation_api_name
        api_obj = self._generation_api_obj
        if api_name is None or api_obj is None:
            api_name, api_obj, _api_path, inspected = _find_generation_callable(_unwrap_model(self.model))
            self._generation_api_name = api_name
            self._generation_api_obj = api_obj
            self._generation_api_path = _api_path
            if (api_name is None or api_obj is None) and self._blip3o_diffusion_pipe is None:
                inspected_text = "; ".join(inspected[:10]) if inspected else "none"
                raise RuntimeError(
                    "Model does not expose a supported image generation API. "
                    f"model_name={self.cfg.model_name} inspected_wrappers={inspected_text}. "
                    "Expected `generate_images(...)`, `generate_image(...)`, or BLIP3o diffusion pipeline."
                )

        if (api_name is None or api_obj is None) and self._blip3o_diffusion_pipe is not None:
            with torch.no_grad():
                with use_adapter(self.model, "generator" if self.cfg.use_lora else None):
                    out = self._run_diffusion_pipeline_with_repair(
                        prompt=prompt,
                        guidance_scale=self.cfg.generation_guidance_scale,
                        num_inference_steps=self.cfg.generation_num_inference_steps,
                        height=self.cfg.generation_height,
                        width=self.cfg.generation_width,
                    )
            images = getattr(out, "images", None)
            if not images:
                raise RuntimeError("BLIP3o diffusion pipeline returned no images.")
            return {
                "image": _ensure_pil_image(images[0]),
                "policy_prompt": prompt,
                "policy_completion": "",
                "policy_completion_ids": None,
                "backend": "diffusion_pipeline",
            }

        # Path 1: BLIP3o-style API with token trace.
        if api_name == "generate_images":
            text_inputs = _prepare_text_inputs(self.processor, self.device, prompt)
            with torch.no_grad():
                with use_adapter(self.model, "generator" if self.cfg.use_lora else None):
                    gen_fn = getattr(api_obj, "generate_images")
                    try:
                        first_param = next(iter(inspect.signature(gen_fn).parameters.values()), None)
                        first_name = first_param.name if first_param is not None else ""
                    except Exception:
                        first_name = ""
                    try:
                        out = gen_fn(
                            text_inputs.get("input_ids"),
                            attention_mask=text_inputs.get("attention_mask"),
                            max_new_tokens=self.cfg.max_new_tokens_generator,
                            temperature=self.cfg.temp,
                            top_p=self.cfg.top_p,
                            num_inference_steps=self.cfg.generation_num_inference_steps,
                            guidance_scale=self.cfg.generation_guidance_scale,
                            return_tensor=False,
                            enable_progress_bar=False,
                        )
                    except TypeError:
                        if first_name in {"inputs", "input_ids"}:
                            out = gen_fn(
                                text_inputs.get("input_ids"),
                                text_inputs.get("attention_mask"),
                                max_new_tokens=self.cfg.max_new_tokens_generator,
                                temperature=self.cfg.temp,
                                top_p=self.cfg.top_p,
                                num_inference_steps=self.cfg.generation_num_inference_steps,
                                guidance_scale=self.cfg.generation_guidance_scale,
                            )
                        else:
                            out = gen_fn(
                                prompt=prompt,
                                max_new_tokens=self.cfg.max_new_tokens_generator,
                                temperature=self.cfg.temp,
                                top_p=self.cfg.top_p,
                                num_inference_steps=self.cfg.generation_num_inference_steps,
                                guidance_scale=self.cfg.generation_guidance_scale,
                            )

            token_completion = ""
            token_completion_ids = None
            image_out = None

            if isinstance(out, tuple) and len(out) >= 2:
                gen_ids, images = out[0], out[1]
                if isinstance(images, list) and images:
                    image_out = images[0]
                elif isinstance(images, Image.Image):
                    image_out = images

                try:
                    if isinstance(gen_ids, torch.Tensor) and gen_ids.ndim == 2 and text_inputs.get("input_ids") is not None:
                        prompt_len = text_inputs["input_ids"].shape[1]
                        completion_ids = gen_ids[0, prompt_len:]
                        token_completion_ids = completion_ids.detach().cpu().tolist()
                        token_completion = _decode_tokens(self.processor, completion_ids).strip()
                except Exception:
                    token_completion = ""
                    token_completion_ids = None
            else:
                images = out
                if isinstance(images, list) and images:
                    image_out = images[0]
                elif isinstance(images, Image.Image):
                    image_out = images

            if image_out is None:
                raise RuntimeError("generate_images returned no image output.")

            backend_name = "generate_images"
            try:
                pil_img = _ensure_pil_image(image_out)
            except Exception:
                if not self.cfg.allow_latent_visualization_fallback:
                    out_shape = tuple(image_out.shape) if torch.is_tensor(image_out) else None
                    out_dtype = str(image_out.dtype) if torch.is_tensor(image_out) else None
                    raise RuntimeError(
                        "Generation backend returned non-image output, and latent visualization fallback is disabled. "
                        "For scientific runs, this indicates missing decoder integration.\n"
                        f"type={type(image_out).__name__} shape={out_shape} dtype={out_dtype}"
                    )
                pil_img = _latent_tensor_to_pil(
                    image_out,
                    target_size=(self.cfg.generation_width, self.cfg.generation_height),
                )
                if pil_img is None:
                    raise
                backend_name = "generate_images_latent_vis"
                if self.is_main_process and not self._warned_latent_fallback:
                    print(
                        "[Generation] WARNING: using latent-visualization fallback for generated outputs "
                        "(decoder pipeline unavailable)."
                    )
                    self._warned_latent_fallback = True

            return {
                "image": pil_img,
                "policy_prompt": prompt,
                "policy_completion": token_completion,
                "policy_completion_ids": token_completion_ids,
                "backend": backend_name,
            }

        # Path 2: generic single-image API (native BLIP3o generate_image).
        if api_name == "generate_image":
            if self._blip3o_diffusion_pipe is not None:
                with torch.no_grad():
                    with use_adapter(self.model, "generator" if self.cfg.use_lora else None):
                        pipe_out = self._run_diffusion_pipeline_with_repair(
                            inputs=prompt,
                            guidance_scale=self.cfg.generation_guidance_scale,
                            num_inference_steps=self.cfg.generation_num_inference_steps,
                            height=self.cfg.generation_height,
                            width=self.cfg.generation_width,
                        )
                images = getattr(pipe_out, "images", None)
                if images is None:
                    if hasattr(pipe_out, "image"):
                        images = pipe_out.image
                    else:
                        images = pipe_out
                if not isinstance(images, (list, tuple)):
                    images = [images]
                if not images:
                    raise RuntimeError(f"BLIP3o diffusion pipeline returned no images. Output type: {type(pipe_out)}")
                return {
                    "image": _ensure_pil_image(images[0]),
                    "policy_prompt": prompt,
                    "policy_completion": "",
                    "policy_completion_ids": None,
                    "backend": "diffusion_pipeline",
                }

            with torch.no_grad():
                with use_adapter(self.model, "generator" if self.cfg.use_lora else None):
                    gen_fn = getattr(api_obj, "generate_image")
                    fn_sig = None
                    fn_params = set()
                    has_var_kw = False
                    try:
                        fn_sig = inspect.signature(gen_fn)
                        for p in fn_sig.parameters.values():
                            if p.kind == inspect.Parameter.VAR_KEYWORD:
                                has_var_kw = True
                            else:
                                fn_params.add(p.name)
                    except Exception:
                        fn_sig = None

                    tokenizer = _extract_tokenizer_from_processor(self.processor)
                    try:
                        if (("text" in fn_params) or (fn_sig is None)) and tokenizer is not None:
                            call_kwargs = {"text": [prompt], "tokenizer": tokenizer}
                            if "pixel_values" in fn_params or has_var_kw:
                                call_kwargs["pixel_values"] = None
                            if "image_grid_thw" in fn_params or has_var_kw:
                                call_kwargs["image_grid_thw"] = None
                            image_out = gen_fn(**call_kwargs)
                        elif "prompt" in fn_params or has_var_kw:
                            image_out = gen_fn(
                                prompt=prompt,
                                num_inference_steps=self.cfg.generation_num_inference_steps,
                                guidance_scale=self.cfg.generation_guidance_scale,
                                height=self.cfg.generation_height,
                                width=self.cfg.generation_width,
                            )
                        elif tokenizer is not None:
                            image_out = gen_fn([prompt], tokenizer)
                        else:
                            image_out = gen_fn(prompt)
                    except TypeError:
                        if tokenizer is not None:
                            try:
                                image_out = gen_fn(text=[prompt], tokenizer=tokenizer)
                            except TypeError:
                                image_out = gen_fn([prompt], tokenizer)
                        else:
                            image_out = gen_fn(prompt)
            try:
                pil_image = _ensure_pil_image(image_out)
            except Exception:
                decode_obj = api_obj
                if not callable(getattr(decode_obj, "decode_latents", None)):
                    decode_obj, _ = _find_callable_object(_unwrap_model(self.model), "decode_latents")
                pil_image = _decode_blip3o_generate_image_output(decode_obj, image_out) if decode_obj is not None else None
                if pil_image is None:
                    if not self.cfg.allow_latent_visualization_fallback:
                        out_shape = tuple(image_out.shape) if torch.is_tensor(image_out) else None
                        out_dtype = str(image_out.dtype) if torch.is_tensor(image_out) else None
                        raise RuntimeError(
                            "generate_image returned non-image output and decoder path failed. "
                            "Latent visualization fallback is disabled for scientific runs.\n"
                            f"type={type(image_out).__name__} shape={out_shape} dtype={out_dtype}"
                        )
                    pil_image = _latent_tensor_to_pil(
                        image_out,
                        target_size=(self.cfg.generation_width, self.cfg.generation_height),
                    )
                    if pil_image is not None:
                        if self.is_main_process and not self._warned_latent_fallback:
                            print(
                                "[Generation] WARNING: using latent-visualization fallback for generated outputs "
                                "(decoder pipeline unavailable)."
                            )
                            self._warned_latent_fallback = True
                        return {
                            "image": pil_image,
                            "policy_prompt": prompt,
                            "policy_completion": "",
                            "policy_completion_ids": None,
                            "backend": "generate_image_latent_vis",
                        }
                    out_shape = tuple(image_out.shape) if torch.is_tensor(image_out) else None
                    out_dtype = str(image_out.dtype) if torch.is_tensor(image_out) else None
                    raise RuntimeError(
                        "generate_image returned a non-image output and no diffusion/latent decode path succeeded. "
                        f"type={type(image_out).__name__} shape={out_shape} dtype={out_dtype}"
                    )
            return {
                "image": pil_image,
                "policy_prompt": prompt,
                "policy_completion": "",
                "policy_completion_ids": None,
                "backend": "generate_image",
            }

        raise RuntimeError(f"Unsupported generation API mode resolved: {api_name}")

    def _solve_question_with_rollouts(self, image: Image.Image, question: str) -> Dict[str, object]:
        solver_prompt = build_solver_prompt(question)
        rollouts = []
        answers_norm: List[str] = []
        adapter_name: Optional[str] = None
        if self.cfg.use_lora:
            adapter_name = "default"

        for _ in range(self.cfg.num_solver_samples_spec):
            completion = self._generate(
                image=image,
                prompt=solver_prompt,
                adapter_name=adapter_name,
                max_new_tokens=self.cfg.max_new_tokens_solver,
                temperature=self.cfg.temp,
                top_p=self.cfg.top_p,
            )
            answer_raw = _parse_answer(completion)
            answer_norm = normalize_answer(answer_raw)
            rollouts.append(
                {
                    "completion": completion,
                    "answer_raw": answer_raw,
                    "answer_norm": answer_norm,
                    "pre_answer_word_count": pre_answer_word_count(completion),
                }
            )
            answers_norm.append(answer_norm)

        maj_answer, maj_count = majority_vote(answers_norm)
        maj_frac = maj_count / float(max(1, len(answers_norm)))

        hist: Dict[str, int] = {}
        for a in answers_norm:
            hist[a] = hist.get(a, 0) + 1
        probs = [count / float(max(1, len(answers_norm))) for count in hist.values()]
        entropy_nats = shannon_entropy_nats(probs)

        return {
            "solver_prompt": solver_prompt,
            "verification_adapter": "default" if adapter_name == "default" else "reference",
            "rollouts": rollouts,
            "majority_answer": maj_answer,
            "majority_count": maj_count,
            "majority_fraction": maj_frac,
            "entropy_nats": entropy_nats,
            "histogram": hist,
        }

    def _score_spec(
        self,
        image: Image.Image,
        qa_pairs: Tuple[GenerationQAPair, ...],
        *,
        step: Optional[int] = None,
        candidate_idx: Optional[int] = None,
        candidate_count: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[float, float, List[Dict[str, object]]]:
        if not qa_pairs:
            return 0.5, 0.0, []

        qa_logs: List[Dict[str, object]] = []
        score_values = []
        contradiction_values = []

        for qa_idx, qa in enumerate(qa_pairs):
            qa_t0 = time.perf_counter()
            if verbose and self.is_main_process and step is not None:
                if candidate_idx is not None and candidate_count is not None:
                    print(
                        f"[Step {step:05d}][G] scoring cand {candidate_idx + 1}/{candidate_count} "
                        f"qa {qa_idx + 1}/{len(qa_pairs)}"
                    )
                else:
                    print(f"[Step {step:05d}][G] scoring qa {qa_idx + 1}/{len(qa_pairs)}")
            solved = self._solve_question_with_rollouts(image=image, question=qa.question)
            mode_answer = str(solved["majority_answer"])
            maj_frac = float(solved["majority_fraction"])

            match_score = _soft_match(mode_answer, qa.expected)
            combined = 0.7 * match_score + 0.3 * maj_frac

            epol = _yes_no_polarity(qa.expected)
            apol = _yes_no_polarity(mode_answer)
            contradiction = 1.0 if (epol != 0 and apol != 0 and epol != apol) else 0.0

            score_values.append(combined)
            contradiction_values.append(contradiction)

            qa_logs.append(
                {
                    "question": qa.question,
                    "expected": qa.expected,
                    "majority_answer": mode_answer,
                    "majority_fraction": maj_frac,
                    "match_score": match_score,
                    "combined_score": combined,
                    "contradiction": contradiction,
                    "solver": solved,
                }
            )
            if verbose and self.is_main_process and step is not None:
                qa_dt = time.perf_counter() - qa_t0
                print(
                    f"[Step {step:05d}][G]   qa {qa_idx + 1}/{len(qa_pairs)} done in {qa_dt:.1f}s "
                    f"(maj_frac={maj_frac:.2f}, match={match_score:.2f})"
                )

        spec_score = float(sum(score_values) / max(1, len(score_values)))
        contradiction_score = float(sum(contradiction_values) / max(1, len(contradiction_values)))
        return spec_score, contradiction_score, qa_logs

    def _cycle_reward(self, prompt: str, image: Image.Image) -> Tuple[float, str]:
        """Compute cycle-consistency reward using self-model embeddings.

        Two complementary signals are combined:
        1. **Caption embedding similarity** — caption the generated image,
           then compare caption vs. original prompt in the model's own
           embedding space (replaces the old Jaccard token overlap).
        2. **Direct image-text similarity** — encode the generated image
           jointly with a neutral probe and compare against the prompt
           embedding.  This captures visual-semantic alignment without
           relying on captioning quality.

        Both scores use the *base model* (no adapter) so they are
        stable reference signals that don't co-drift with training.
        """
        # 1) Generate a caption of the produced image (solver adapter)
        caption = self._generate(
            image=image,
            prompt=GEN_CYCLE_CAPTION_PROMPT,
            adapter_name="default" if self.cfg.use_lora else None,
            max_new_tokens=self.cfg.max_new_tokens_caption,
            temperature=max(0.2, min(self.cfg.temp, 0.8)),
            top_p=1.0,
        )
        caption = " ".join(caption.split())
        if not caption:
            caption = ""

        # 2) Embedding-based caption ↔ prompt similarity
        try:
            caption_sim = self._embedding_similarity(prompt, caption)
        except Exception:
            # Fallback to Jaccard if embedding fails (e.g. OOM on very long text)
            caption_sim = _jaccard_similarity(prompt, caption)

        # 3) Direct image ↔ prompt similarity (visual-semantic alignment)
        try:
            image_text_sim = self._image_text_similarity(image, prompt)
        except Exception:
            image_text_sim = caption_sim  # graceful fallback

        # Combine: caption similarity anchors semantic fidelity,
        # image-text similarity captures what captioning may miss.
        score = 0.5 * caption_sim + 0.5 * image_text_sim
        # Clamp to [0, 1] — cosine similarity can be negative
        score = max(0.0, min(1.0, score))
        return score, caption

    def _score_candidates(
        self,
        prompt: str,
        qa_pairs: Tuple[GenerationQAPair, ...],
        candidates: List[Dict[str, object]],
        spec_quality: float,
        *,
        step: Optional[int] = None,
        verbose: bool = False,
    ) -> List[Dict[str, object]]:
        images = [cand["image"] for cand in candidates]
        diversity_scores = _per_candidate_diversity_scores(images)

        scored: List[Dict[str, object]] = []
        for idx, cand in enumerate(candidates):
            cand_t0 = time.perf_counter()
            if verbose and self.is_main_process and step is not None:
                backend = str(cand.get("backend", "unknown"))
                print(
                    f"[Step {step:05d}][G] evaluating candidate {idx + 1}/{len(candidates)} "
                    f"(backend={backend})"
                )
            image = cand["image"]
            spec_score, contradiction_score, qa_logs = self._score_spec(
                image=image,
                qa_pairs=qa_pairs,
                step=step,
                candidate_idx=idx,
                candidate_count=len(candidates),
                verbose=verbose,
            )
            qa_confidence = self._qa_confidence_from_logs(qa_logs)
            cycle_score, cycle_caption = self._cycle_reward(prompt=prompt, image=image)

            pos_sum = (
                self.cfg.reward_spec_weight
                + self.cfg.reward_cycle_weight
                + self.cfg.reward_diversity_weight
            )
            if pos_sum <= 0:
                pos_sum = 1.0
            w_spec = self.cfg.reward_spec_weight / pos_sum
            w_cycle = self.cfg.reward_cycle_weight / pos_sum
            w_div = self.cfg.reward_diversity_weight / pos_sum

            base_reward = (
                w_spec * spec_score
                + w_cycle * cycle_score
                + w_div * diversity_scores[idx]
                - self.cfg.reward_contradiction_weight * contradiction_score
            )
            base_reward = max(0.0, min(1.0, base_reward))
            total_reward = spec_quality * base_reward
            scored.append(
                {
                    "candidate_idx": idx,
                    "backend": cand.get("backend"),
                    "policy_prompt": cand.get("policy_prompt", prompt),
                    "policy_completion": cand.get("policy_completion", ""),
                    "spec_score": spec_score,
                    "contradiction_score": contradiction_score,
                    "cycle_score": cycle_score,
                    "cycle_caption": cycle_caption,
                    "diversity_score": diversity_scores[idx],
                    "base_reward": base_reward,
                    "spec_quality": spec_quality,
                    "total_reward": total_reward,
                    "qa_confidence": qa_confidence,
                    "qa_logs": qa_logs,
                    "image": image,
                }
            )
            if verbose and self.is_main_process and step is not None:
                cand_dt = time.perf_counter() - cand_t0
                print(
                    f"[Step {step:05d}][G] candidate {idx + 1}/{len(candidates)} done in {cand_dt:.1f}s "
                    f"(spec={spec_score:.3f}, cycle={cycle_score:.3f}, total={total_reward:.3f})"
                )
        return scored

    # ---- Phase 2: reference-answer log-prob scoring ---- #

    @torch.no_grad()
    def _compute_ref_answer_logp(
        self,
        image: Image.Image,
        question: str,
        reference_answer: str,
        device: torch.device,
    ) -> float:
        """Compute log P(reference_answer | image, question) under solver.

        Returns the *mean* per-token log-probability of the reference answer
        conditioned on the generated image and the question.  This is used
        as a continuous reward signal for ranking candidate images.

        Runs under ``torch.no_grad()`` — inference only, no gradient.
        Uses the solver adapter ("default") on the *wrapped* model to stay
        consistent with ``_generate`` and the rest of the codebase.
        """
        import torch.nn.functional as F
        from .generation_policy_updater import _aligned_prompt_prefix_len

        ref_ans_stripped = reference_answer.strip()
        if not ref_ans_stripped:
            return -10.0

        solver_prompt = build_solver_prompt(question)
        solver_prompt_chat = _build_chat_text(self.processor, image, solver_prompt)
        # The chat template ends with the assistant generation prompt; the
        # reference answer follows as if the model had generated it.
        full_text = solver_prompt_chat + ref_ans_stripped

        # Use the trained solver LoRA to score candidates so that
        # understanding improvements flow into generation scoring
        # (mutual supervision).  The solver is grounded by majority-vote
        # training on real images, preventing co-adaptation.
        model = self.train_model if hasattr(self, "train_model") else self.model
        if self.cfg.use_lora:
            adapter = "default"  # trained solver adapter
        else:
            adapter = None
        was_training = model.training

        try:
            with use_adapter(model, adapter):
                model.eval()
                prompt_inputs = _prepare_mm_inputs(
                    self.processor, device, image, solver_prompt_chat,
                    model=_unwrap_model(model),
                )
                full_inputs = _prepare_mm_inputs(
                    self.processor, device, image, full_text,
                    model=_unwrap_model(model),
                )

                out = model(**full_inputs)

                # Robust prompt length via token alignment (handles edge cases
                # where prompt/full tokenization differs).
                prompt_len = _aligned_prompt_prefix_len(
                    prompt_inputs["input_ids"],
                    full_inputs["input_ids"],
                    ref_ans_stripped,
                )
                labels = full_inputs["input_ids"].clone()
                labels[:, :prompt_len] = -100
                shift_labels = labels[:, 1:]
                valid_mask = shift_labels != -100

                logp = F.log_softmax(out.logits[:, :-1, :], dim=-1)
                gathered = logp.gather(
                    -1, shift_labels.clamp_min(0).unsqueeze(-1)
                ).squeeze(-1)

                valid_count = int(valid_mask.sum().item())
                if valid_count > 0:
                    return float(gathered[valid_mask].mean().item())
                else:
                    return -10.0
        finally:
            # Restore original training/eval state
            model.train(was_training)

    def _score_candidates_ref_answer(
        self,
        real_image: Image.Image,
        spec: "GenerationSpec",
        candidates: List[Dict[str, object]],
        *,
        step: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[List[Dict[str, object]], List[str], List[str]]:
        """Phase 2 scoring: reference-answer log-prob on generated images.

        1. Extract questions from ``spec.qa_pairs``.
        2. Solver answers each question looking at ``real_image`` → reference answers.
        3. For each candidate, compute mean log P(ref_answer | candidate, question).
        4. That mean log-prob is the ``total_reward``.

        Returns
        -------
        scored : list of dicts (same schema as ``_score_candidates``)
        questions : list of question strings
        reference_answers : list of solver answers on the real image
        """
        questions = [qa.question for qa in spec.qa_pairs if qa.question.strip()]
        if not questions:
            # Fallback: empty scoring — all zeros
            scored = [
                {
                    "candidate_idx": idx,
                    "total_reward": 0.0,
                    "ref_answer_logps": [],
                    "image": cand.get("image"),
                    "policy_prompt": cand.get("policy_prompt", spec.prompt),
                    "policy_completion": cand.get("policy_completion", ""),
                    "policy_completion_ids": cand.get("policy_completion_ids"),
                    "backend": cand.get("backend"),
                }
                for idx, cand in enumerate(candidates)
            ]
            return scored, [], []

        # Step 1: Solver generates reference answers on the REAL image.
        # Uses the trained solver LoRA — as solver improves through
        # understanding training, it provides better reference answers,
        # which means harder/more accurate scoring for generation (mutual supervision).
        device = self.device
        cfg = self.cfg
        _solver_adapter = "default" if cfg.use_lora else None

        reference_answers: List[str] = []
        temp = max(0.2, min(0.8, cfg.temp))
        for q in questions:
            ref_ans = self._generate(
                image=real_image,
                prompt=build_solver_prompt(q),
                adapter_name=_solver_adapter,
                max_new_tokens=cfg.max_new_tokens_solver,
                temperature=temp,
                top_p=cfg.top_p,
            )
            reference_answers.append(ref_ans.strip())

        if verbose and self.is_main_process and step is not None:
            for i, (q, a) in enumerate(zip(questions, reference_answers)):
                print(f"[Step {step:05d}][G-ref] Q{i}: {q}")
                print(f"[Step {step:05d}][G-ref] A{i}: {a}")

        # Step 2: Score each candidate via log-prob
        scored: List[Dict[str, object]] = []
        for idx, cand in enumerate(candidates):
            cand_image = cand.get("image")
            if not isinstance(cand_image, Image.Image):
                scored.append(
                    {
                        "candidate_idx": idx,
                        "total_reward": -10.0,
                        "ref_answer_logps": [],
                        "image": cand_image,
                        "policy_prompt": cand.get("policy_prompt", spec.prompt),
                        "policy_completion": cand.get("policy_completion", ""),
                        "policy_completion_ids": cand.get("policy_completion_ids"),
                        "backend": cand.get("backend"),
                    }
                )
                continue

            logps: List[float] = []
            for q, ref_ans in zip(questions, reference_answers):
                if not ref_ans:
                    continue
                lp = self._compute_ref_answer_logp(
                    image=cand_image,
                    question=q,
                    reference_answer=ref_ans,
                    device=device,
                )
                logps.append(lp)

            reward = sum(logps) / len(logps) if logps else -10.0

            scored.append(
                {
                    "candidate_idx": idx,
                    "total_reward": reward,
                    "ref_answer_logps": logps,
                    "image": cand_image,
                    "policy_prompt": cand.get("policy_prompt", spec.prompt),
                    "policy_completion": cand.get("policy_completion", ""),
                    "policy_completion_ids": cand.get("policy_completion_ids"),
                    "backend": cand.get("backend"),
                    # Compat keys for logging (not used in ref-answer mode)
                    "spec_score": 0.0,
                    "cycle_score": 0.0,
                    "diversity_score": 0.0,
                    "contradiction_score": 0.0,
                    "base_reward": reward,
                    "spec_quality": 1.0,
                    "qa_confidence": 0.0,
                    "qa_logs": [],
                }
            )

            if verbose and self.is_main_process and step is not None:
                print(
                    f"[Step {step:05d}][G-ref] candidate {idx + 1}/{len(candidates)} "
                    f"reward={reward:.4f} logps={[f'{lp:.3f}' for lp in logps]}"
                )

        return scored, questions, reference_answers

    def _update_baseline(self, which: str, reward: float):
        m = self.cfg.baseline_momentum
        if which == "generator":
            self.generator_baseline = m * self.generator_baseline + (1.0 - m) * reward
        elif which == "proposer":
            self.proposer_baseline = m * self.proposer_baseline + (1.0 - m) * reward
        else:
            self.solver_baseline = m * self.solver_baseline + (1.0 - m) * reward

    def _save_candidate_images(self, step: int, scored: List[Dict[str, object]], best_idx: int):
        if self.cfg.save_generated_images_every <= 0:
            return
        if (step % self.cfg.save_generated_images_every) != 0:
            return
        step_dir = self.generated_dir / f"step_{step:05d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for i, cand in enumerate(scored):
            image = cand.get("image")
            if not isinstance(image, Image.Image):
                continue
            flag = "best" if i == best_idx else "cand"
            reward = cand.get("total_reward", 0.0)
            path = step_dir / f"{flag}_{i:02d}_r{reward:.4f}.png"
            try:
                image.save(path)
            except Exception:
                pass

    def _proxy_generator_completion(self, image: Image.Image) -> str:
        completion = self._generate(
            image=image,
            prompt=GENERATOR_PROXY_CAPTION_PROMPT,
            adapter_name="generator" if self.cfg.use_lora else None,
            max_new_tokens=self.cfg.max_new_tokens_caption,
            temperature=max(0.2, min(self.cfg.temp, 0.8)),
            top_p=1.0,
        )
        return " ".join(completion.split())

    def _current_proxy_ratio(self) -> float:
        token_updates = float(self._generator_update_mode_counts.get("token_trace", 0))
        proxy_updates = float(self._generator_update_mode_counts.get("proxy_caption", 0))
        denom = token_updates + proxy_updates
        if denom <= 0:
            return 0.0
        return proxy_updates / denom

    def _proxy_updates_allowed(self) -> bool:
        max_ratio = float(getattr(self.cfg, "generator_proxy_max_ratio", 1.0))
        max_ratio = max(0.0, min(1.0, max_ratio))
        if max_ratio >= 1.0:
            return True
        return self._current_proxy_ratio() < max_ratio

    @staticmethod
    def _qa_confidence_from_logs(qa_logs: List[Dict[str, object]]) -> float:
        values: List[float] = []
        for qa in qa_logs:
            try:
                frac = float(qa.get("majority_fraction", 0.0))
            except Exception:
                continue
            if math.isfinite(frac):
                values.append(max(0.0, min(1.0, frac)))
        if not values:
            return 0.0
        return float(sum(values) / max(1, len(values)))

    def _select_dpo_pair_indices(self, scored: List[Dict[str, object]], best_idx: int) -> Optional[Tuple[int, int]]:
        if len(scored) < 2:
            return None
        candidate_indices = [i for i in range(len(scored)) if i != best_idx]
        if not candidate_indices:
            return None
        mode = str(getattr(self.cfg, "dpo_pair_selection", "best_worst") or "best_worst").strip().lower()
        if mode == "best_hard_negative":
            # Hard negative: strongest non-winning candidate (closest competitor).
            rejected_idx = max(candidate_indices, key=lambda i: float(scored[i]["total_reward"]))
        elif mode == "best_worst":
            rejected_idx = min(candidate_indices, key=lambda i: float(scored[i]["total_reward"]))
        else:
            raise ValueError(
                f"Unsupported dpo_pair_selection={self.cfg.dpo_pair_selection!r}. "
                "Expected one of: best_worst, best_hard_negative."
            )
        return int(best_idx), int(rejected_idx)

    def _save_checkpoint(self, step: int):
        if not self.is_main_process:
            return
        step_dir = self.run_dir / f"step_{step:05d}"
        tmp_dir = self.run_dir / f"step_{step:05d}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.use_lora:
            adapter_map = (("default", "solver"), ("proposer", "proposer"), ("generator", "generator"))
            for adapter_name, sub_name in adapter_map:
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

        model_ref = _unwrap_model(self.model)
        trainable_state = {
            name: param.detach().cpu()
            for name, param in model_ref.named_parameters()
            if param.requires_grad
        }
        torch.save(trainable_state, tmp_dir / "trainable_adapters.pt")

        torch.save(self._trainer_state_dict(step), tmp_dir / "trainer_state.pt")

        _json_dump(
            tmp_dir / "trainer_state.json",
            {
                "step": step,
                "solver_baseline": self.solver_baseline,
                "proposer_baseline": self.proposer_baseline,
                "generator_baseline": self.generator_baseline,
                "solver_kl_coef": self.solver_updater.kl_coef if self.solver_updater is not None else None,
                "proposer_kl_coef": self.proposer_updater.kl_coef,
                "generator_kl_coef": self.generator_updater.kl_coef,
                "solver_updater_step": self.solver_updater.step_id if self.solver_updater is not None else None,
                "proposer_updater_step": self.proposer_updater.step_id,
                "generator_updater_step": self.generator_updater.step_id,
            },
        )
        with (tmp_dir / "SAVE_OK").open("w", encoding="utf-8") as f:
            f.write("ok\n")

        if step_dir.exists():
            shutil.rmtree(step_dir, ignore_errors=True)
        os.replace(str(tmp_dir), str(step_dir))

        self._prune_checkpoints()

    def _prune_checkpoints(self):
        if not self.is_main_process:
            return
        keep = max(1, int(self.cfg.max_checkpoints))
        checkpoints = self._list_complete_checkpoints()
        if len(checkpoints) <= keep:
            return
        for path in checkpoints[:-keep]:
            shutil.rmtree(path, ignore_errors=True)

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
        _json_dump(
            self.summary_path,
            {
                "experiment": self.cfg.experiment_name,
                "run_dir": str(self.run_dir),
                "generator_update_rule": self.cfg.generator_update_rule,
                "final_step": int(final_step),
                "start_step": int(self.start_step),
                "status": status,
                "interrupted_at_step": interrupted_at_step,
                "error": error,
                "policy_update_counts": self._policy_update_counts,
                "generator_update_mode_counts": self._generator_update_mode_counts,
                "unicorn_reconstruction_update_counts": self._unicorn_reconstruction_update_counts,
                "unicorn_reconstruction_buffer_size": int(len(self._unicorn_reconstruction_buffer)),
                "diffusion_repair_count": int(self._diffusion_repair_count),
                "metrics": self._metrics_summary(),
            },
        )

    def _wandb_log_step(
        self,
        *,
        step: int,
        image_path: Optional[str],
        source_caption: str,
        spec: GenerationSpec,
        scored: List[Dict[str, object]],
        best_idx: int,
        spec_quality: float,
        reward_mean_global: float,
        reward_max_global: float,
        reward_min_global: float,
        best_spec_global: float,
        best_cycle_global: float,
        best_diversity_global: float,
        best_contradiction_global: float,
        generator_skipped_reason: Optional[str],
        generator_update_mode: Optional[str],
        proposer_stats: Optional[Dict[str, float]],
        generator_stats: Optional[Dict[str, float]],
        unicorn_spec_meta: Optional[Dict[str, object]] = None,
        unicorn_reconstruction: Optional[Dict[str, object]] = None,
    ):
        if not self.is_main_process or self.wandb_run is None:
            return

        best = scored[best_idx]
        metrics: Dict[str, object] = {
            "train/step": step,
            "train/source_caption": source_caption,
            "train/spec_fallback_used": 1.0 if spec.fallback_used else 0.0,
            "train/spec_quality": float(spec_quality),
            "train/spec_qa_count": float(len(spec.qa_pairs)),
            "train/candidate_reward_mean": float(reward_mean_global),
            "train/candidate_reward_max": float(reward_max_global),
            "train/candidate_reward_min": float(reward_min_global),
            "train/best_spec_score": float(best_spec_global),
            "train/best_cycle_score": float(best_cycle_global),
            "train/best_diversity_score": float(best_diversity_global),
            "train/best_contradiction_score": float(best_contradiction_global),
            "train/best_qa_confidence": float(best.get("qa_confidence", 0.0)),
            "train/generator_baseline": self.generator_baseline,
            "train/proposer_baseline": self.proposer_baseline,
            "train/generator_update_skipped": 1.0 if generator_skipped_reason else 0.0,
            "train/generator_update_mode_token_trace": 1.0 if generator_update_mode == "token_trace" else 0.0,
            "train/generator_update_mode_proxy_caption": 1.0 if generator_update_mode == "proxy_caption" else 0.0,
            "train/generator_update_rule_reinforce": 1.0 if self.cfg.generator_update_rule == "reinforce" else 0.0,
            "train/generator_update_rule_dpo": 1.0 if self.cfg.generator_update_rule == "dpo" else 0.0,
            "train/generator_update_rule_grpo": 1.0 if self.cfg.generator_update_rule == "grpo" else 0.0,
            "train/generator_proxy_update_ratio": float(self._current_proxy_ratio()),
            "train/diffusion_repair_count": float(self._diffusion_repair_count),
            "kl/generator_beta": self.generator_updater.kl_coef,
            "kl/proposer_beta": self.proposer_updater.kl_coef,
            "text/prompt": spec.prompt,
            "text/proposer_raw": spec.raw_output,
            "text/best_cycle_caption": best.get("cycle_caption", ""),
        }
        metrics["train/generator_update_rule"] = self.cfg.generator_update_rule
        if generator_update_mode:
            metrics["train/generator_update_mode"] = generator_update_mode
        if generator_skipped_reason:
            metrics["train/generator_skip_reason"] = generator_skipped_reason
        if image_path:
            metrics["data/image_path"] = image_path
        if unicorn_spec_meta:
            metrics["train/unicorn_spec_attempts"] = float(unicorn_spec_meta.get("attempts", 1.0))
            metrics["train/unicorn_spec_retries_used"] = float(unicorn_spec_meta.get("retries_used", 0.0))
            metrics["train/unicorn_spec_alignment"] = float(unicorn_spec_meta.get("selected_alignment", 0.0))
            metrics["train/unicorn_spec_selected_accepted"] = (
                1.0 if bool(unicorn_spec_meta.get("selected_accepted", False)) else 0.0
            )
        if unicorn_reconstruction:
            metrics["train/unicorn_recon_enqueued"] = float(unicorn_reconstruction.get("enqueued_this_step", 0.0))
            metrics["train/unicorn_recon_attempted"] = float(unicorn_reconstruction.get("attempted_updates", 0.0))
            metrics["train/unicorn_recon_applied"] = float(unicorn_reconstruction.get("applied_updates", 0.0))
            metrics["train/unicorn_recon_skipped"] = float(unicorn_reconstruction.get("skipped_updates", 0.0))
            metrics["train/unicorn_recon_buffer_size"] = float(
                unicorn_reconstruction.get("buffer_size_after_step", 0.0)
            )

        if proposer_stats:
            metrics.update(
                {
                    "proposer/ce_loss": proposer_stats.get("ce_loss"),
                    "proposer/kl_loss": proposer_stats.get("kl_loss"),
                    "proposer/advantage": proposer_stats.get("advantage"),
                }
            )

        if generator_stats:
            if "dpo_loss" in generator_stats:
                metrics.update(
                    {
                        "generator/dpo_loss": generator_stats.get("dpo_loss"),
                        "generator/dpo_margin": generator_stats.get("preference_margin"),
                        "generator/dpo_pi_gap": generator_stats.get("pi_gap"),
                        "generator/dpo_ref_gap": generator_stats.get("ref_gap"),
                        "generator/dpo_beta": generator_stats.get("dpo_beta"),
                    }
                )
            else:
                metrics.update(
                    {
                        "generator/ce_loss": generator_stats.get("ce_loss"),
                        "generator/kl_loss": generator_stats.get("kl_loss"),
                        "generator/advantage": generator_stats.get("advantage"),
                    }
                )

        if (
            self.cfg.wandb_log_images_every > 0
            and (step % self.cfg.wandb_log_images_every) == 0
            and isinstance(best.get("image"), Image.Image)
        ):
            try:
                metrics["vis/best_generated_image"] = wandb.Image(best["image"], caption=f"step={step}")
            except Exception:
                pass

        try:
            wandb.log(metrics, step=step)
        except Exception as exc:
            print(f"[W&B] log failed at step {step}: {exc}")

    def _generation_step(self, step: int, image: Image.Image, meta: Dict) -> Dict[str, object]:
        verbose = self.is_main_process and (step % self.cfg.log_every == 0)
        step_t0 = time.perf_counter()
        if verbose:
            print(f"[Step {step:05d}][G] generation phase start")

        source_caption = self._caption_image(image)
        spec, spec_quality, spec_quality_details, unicorn_spec_meta = self._select_generation_spec_with_unicorn(
            image=image,
            source_caption=source_caption,
            step=step,
            verbose=verbose,
        )
        self._append_jsonl(
            self.unicorn_spec_log_path,
            {
                "step": int(step),
                "image_path": meta.get("path"),
                "selected_prompt": spec.prompt,
                "spec_quality": float(spec_quality),
                "spec_quality_details": spec_quality_details,
                "unicorn_spec_meta": unicorn_spec_meta,
            },
        )
        if verbose:
            print(
                f"[Step {step:05d}][G] spec ready: qa_pairs={len(spec.qa_pairs)} "
                f"quality={spec_quality:.3f} fallback={int(spec.fallback_used)} "
                f"attempts={int(unicorn_spec_meta.get('attempts', 1))} "
                f"align={float(unicorn_spec_meta.get('selected_alignment', 0.0)):.3f}"
            )

        candidates: List[Dict[str, object]] = []
        for cand_idx in range(self.cfg.num_generations):
            cand_t0 = time.perf_counter()
            if verbose:
                print(
                    f"[Step {step:05d}][G] generating candidate {cand_idx + 1}/{self.cfg.num_generations}"
                )
            cand = self._generate_image_candidate(inputs=spec.prompt)
            candidates.append(cand)
            if verbose:
                backend = str(cand.get("backend", "unknown"))
                cand_dt = time.perf_counter() - cand_t0
                print(
                    f"[Step {step:05d}][G] generated candidate {cand_idx + 1}/{self.cfg.num_generations} "
                    f"in {cand_dt:.1f}s (backend={backend})"
                )
        # ---- Score candidates (Phase 1 vs Phase 2 scoring) ---- #
        _use_ref_scoring = getattr(self.cfg, "use_ref_answer_scoring", False)
        _ref_questions: Optional[List[str]] = None
        _ref_answers: Optional[List[str]] = None

        if _use_ref_scoring:
            scored, _ref_questions, _ref_answers = self._score_candidates_ref_answer(
                real_image=image,
                spec=spec,
                candidates=candidates,
                step=step,
                verbose=verbose,
            )
        else:
            scored = self._score_candidates(
                prompt=spec.prompt,
                qa_pairs=spec.qa_pairs,
                candidates=candidates,
                spec_quality=spec_quality,
                step=step,
                verbose=verbose,
            )
        best_idx = max(range(len(scored)), key=lambda i: float(scored[i]["total_reward"]))
        best = scored[best_idx]

        # ---- Store best candidate in replay buffer ---- #
        # Best generated image enters the replay buffer for mixing into
        # understanding training. The buffer's quality gate (min_reward)
        # ensures only good images are kept.
        #
        # For ref-answer scoring (MODE B), total_reward is a log-prob (negative).
        # Normalize to [0, 1] so the replay buffer quality gate works uniformly:
        #   sigmoid(logp) maps (-inf, 0] → (0, 0.5], typical range [-5, 0] → [0.007, 0.5]
        #   We use sigmoid(logp + 2) to shift the useful range up, so:
        #     logp = -4 → 0.12,  logp = -2 → 0.5,  logp = -1 → 0.73,  logp = 0 → 0.88
        _replay_buf = getattr(self, "replay_buffer", None)
        if (
            _replay_buf is not None
            and isinstance(best.get("image"), Image.Image)
        ):
            _rb_questions = _ref_questions or [qa.question for qa in spec.qa_pairs]
            _rb_answers = _ref_answers or [qa.expected for qa in spec.qa_pairs]
            _raw_reward = float(best["total_reward"])
            if _use_ref_scoring:
                # Normalize log-prob to [0, 1] for replay buffer compatibility
                _rb_reward = 1.0 / (1.0 + math.exp(-(_raw_reward + 2.0)))
            else:
                _rb_reward = _raw_reward
            _replay_buf.add(
                image=best["image"],
                prompt=spec.prompt,
                questions=_rb_questions,
                reference_answers=_rb_answers,
                reward=_rb_reward,
                step=step,
                meta={"best_idx": best_idx, "num_candidates": len(candidates),
                       "raw_reward": _raw_reward},
            )

        proposer_stats = None
        generator_stats = None
        generator_skipped_reason = None
        generator_update_mode = None

        def _global_update_ready(
            local_ready: bool,
            local_reason: Optional[str],
            *,
            peer_reason: str,
        ) -> Tuple[bool, Optional[str]]:
            if not (self.distributed and dist.is_initialized()):
                return bool(local_ready), local_reason
            all_ready = self._dist_all_bool(bool(local_ready))
            if all_ready:
                return True, local_reason
            if local_reason:
                return False, local_reason
            return False, peer_reason

        generator_update_due = self.cfg.generator_update_freq > 0 and (step % self.cfg.generator_update_freq == 0)
        local_quality_ok = spec_quality >= self.cfg.min_spec_quality_for_update
        quality_ok_all = local_quality_ok
        if generator_update_due and self.distributed and dist.is_initialized():
            quality_ok_all = self._dist_all_bool(local_quality_ok)

        if generator_update_due and quality_ok_all:
            baseline_before = self.generator_baseline
            generator_reward = float(best["total_reward"])
            update_rule = self.cfg.generator_update_rule

            if update_rule == "dpo":
                pair = self._select_dpo_pair_indices(scored, best_idx)
                chosen_idx: Optional[int] = None
                rejected_idx: Optional[int] = None
                chosen_reward = 0.0
                rejected_reward = 0.0
                reward_gap = 0.0
                chosen_spec = 0.0
                rejected_spec = 0.0
                spec_gap = 0.0
                chosen_confidence = 0.0
                rejected_confidence = 0.0
                confidence_gap = 0.0
                chosen_contradiction = 0.0
                rejected_contradiction = 0.0
                contradiction_max = 0.0
                chosen: Optional[Dict[str, object]] = None
                rejected: Optional[Dict[str, object]] = None

                local_pair_ready = False
                local_pair_skip_reason: Optional[str] = None
                if pair is None:
                    local_pair_skip_reason = "dpo_requires_two_candidates"
                else:
                    chosen_idx, rejected_idx = pair
                    chosen = scored[chosen_idx]
                    rejected = scored[rejected_idx]
                    chosen_reward = float(chosen["total_reward"])
                    rejected_reward = float(rejected["total_reward"])
                    reward_gap = chosen_reward - rejected_reward
                    chosen_spec = float(chosen.get("spec_score", 0.0))
                    rejected_spec = float(rejected.get("spec_score", 0.0))
                    spec_gap = chosen_spec - rejected_spec
                    chosen_confidence = float(chosen.get("qa_confidence", 0.0))
                    rejected_confidence = float(rejected.get("qa_confidence", 0.0))
                    confidence_gap = chosen_confidence - rejected_confidence
                    chosen_contradiction = float(chosen.get("contradiction_score", 0.0))
                    rejected_contradiction = float(rejected.get("contradiction_score", 0.0))
                    contradiction_max = max(chosen_contradiction, rejected_contradiction)

                    pair_gate_reason: Optional[str] = None
                    if reward_gap < float(self.cfg.dpo_min_reward_gap):
                        pair_gate_reason = "dpo_reward_gap_too_small"
                    elif spec_gap < float(getattr(self.cfg, "dpo_min_spec_gap", 0.0)):
                        pair_gate_reason = "dpo_spec_gap_too_small"
                    elif confidence_gap < float(getattr(self.cfg, "dpo_min_confidence_gap", 0.0)):
                        pair_gate_reason = "dpo_confidence_gap_too_small"
                    elif contradiction_max > float(getattr(self.cfg, "dpo_max_contradiction", 1.0)):
                        pair_gate_reason = "dpo_contradiction_too_high"

                    if pair_gate_reason is not None:
                        local_pair_skip_reason = pair_gate_reason
                    else:
                        local_pair_ready = True

                pair_ready, generator_skipped_reason = _global_update_ready(
                    local_pair_ready,
                    local_pair_skip_reason,
                    peer_reason="distributed_peer_skip",
                )

                if pair_ready:
                    assert chosen is not None and rejected is not None
                    assert chosen_idx is not None and rejected_idx is not None
                    chosen_completion = str(chosen.get("policy_completion", "")).strip()
                    rejected_completion = str(rejected.get("policy_completion", "")).strip()

                    chosen_token_ids = chosen.get("policy_completion_ids")
                    if not isinstance(chosen_token_ids, list):
                        chosen_token_ids = None
                    rejected_token_ids = rejected.get("policy_completion_ids")
                    if not isinstance(rejected_token_ids, list):
                        rejected_token_ids = None

                    update_prompt = str(chosen.get("policy_prompt", spec.prompt))
                    chosen_image: Optional[Image.Image] = None
                    rejected_image: Optional[Image.Image] = None

                    if chosen_completion and rejected_completion:
                        generator_update_mode = "token_trace"
                    else:
                        strategy = (self.cfg.generator_missing_trace_strategy or "proxy").strip().lower()
                        if self.cfg.strict_require_generation_tokens or strategy == "error":
                            generator_skipped_reason = "missing_generation_token_trace_strict"
                            strategy = "skip"
                        if strategy == "proxy":
                            if not self._proxy_updates_allowed():
                                generator_skipped_reason = "proxy_budget_exceeded"
                                strategy = "skip"
                            chosen_raw_image = chosen.get("image")
                            rejected_raw_image = rejected.get("image")
                            if (
                                generator_skipped_reason is None
                                and isinstance(chosen_raw_image, Image.Image)
                                and isinstance(rejected_raw_image, Image.Image)
                            ):
                                chosen_proxy = self._proxy_generator_completion(chosen_raw_image)
                                rejected_proxy = self._proxy_generator_completion(rejected_raw_image)
                                if chosen_proxy and rejected_proxy:
                                    chosen_completion = chosen_proxy
                                    rejected_completion = rejected_proxy
                                    chosen_token_ids = None
                                    rejected_token_ids = None
                                    update_prompt = GENERATOR_PROXY_CAPTION_PROMPT
                                    chosen_image = chosen_raw_image
                                    rejected_image = rejected_raw_image
                                    generator_update_mode = "proxy_caption"
                                else:
                                    generator_skipped_reason = "dpo_proxy_empty_completion"
                            elif generator_skipped_reason is None:
                                generator_skipped_reason = "dpo_proxy_missing_image"
                        elif strategy == "skip":
                            generator_skipped_reason = "missing_generation_token_trace"
                        else:
                            raise ValueError(
                                "Unsupported generator_missing_trace_strategy="
                                f"{self.cfg.generator_missing_trace_strategy!r}. Expected one of: proxy, skip, error."
                            )

                    local_can_update = bool(chosen_completion and rejected_completion and generator_skipped_reason is None)
                    can_update, generator_skipped_reason = _global_update_ready(
                        local_can_update,
                        generator_skipped_reason,
                        peer_reason="distributed_peer_skip",
                    )

                    if can_update:
                        generator_stats = self.generator_updater.step(
                            prompt=update_prompt,
                            chosen_completion=chosen_completion,
                            rejected_completion=rejected_completion,
                            device=self.device,
                            chosen_image=chosen_image,
                            rejected_image=rejected_image,
                            chosen_completion_token_ids=chosen_token_ids,
                            rejected_completion_token_ids=rejected_token_ids,
                        )
                        if generator_stats.get("did_step", True):
                            self._policy_update_counts["generator"] += 1
                            self._generator_update_mode_counts[generator_update_mode] = (
                                self._generator_update_mode_counts.get(generator_update_mode, 0) + 1
                            )

                        self._append_jsonl(
                            self.policy_updates_log_path,
                            {
                                "step": step,
                                "role": "generator",
                                "update_rule": "dpo",
                                "reward": generator_reward,
                                "baseline_before": baseline_before,
                                "baseline_after": baseline_before,
                                "stats": generator_stats,
                                "update_mode": generator_update_mode,
                                "update_prompt": update_prompt,
                                "used_image_conditioning": chosen_image is not None and rejected_image is not None,
                                "chosen_candidate_idx": int(chosen_idx),
                                "rejected_candidate_idx": int(rejected_idx),
                                "chosen_reward": chosen_reward,
                                "rejected_reward": rejected_reward,
                                "reward_gap": reward_gap,
                                "chosen_spec": chosen_spec,
                                "rejected_spec": rejected_spec,
                                "spec_gap": spec_gap,
                                "chosen_confidence": chosen_confidence,
                                "rejected_confidence": rejected_confidence,
                                "confidence_gap": confidence_gap,
                                "chosen_contradiction": chosen_contradiction,
                                "rejected_contradiction": rejected_contradiction,
                                "spec_quality": spec_quality,
                            },
                        )
                        self._append_jsonl(
                            self.dpo_pairs_log_path,
                            {
                                "step": step,
                                "chosen_candidate_idx": int(chosen_idx),
                                "rejected_candidate_idx": int(rejected_idx),
                                "chosen_reward": chosen_reward,
                                "rejected_reward": rejected_reward,
                                "reward_gap": reward_gap,
                                "chosen_spec": chosen_spec,
                                "rejected_spec": rejected_spec,
                                "spec_gap": spec_gap,
                                "chosen_confidence": chosen_confidence,
                                "rejected_confidence": rejected_confidence,
                                "confidence_gap": confidence_gap,
                                "chosen_contradiction": chosen_contradiction,
                                "rejected_contradiction": rejected_contradiction,
                                "update_mode": generator_update_mode,
                                "prompt": update_prompt,
                                "chosen_completion_char_len": len(chosen_completion),
                                "rejected_completion_char_len": len(rejected_completion),
                                "chosen_completion_token_count": len(chosen_token_ids) if chosen_token_ids else None,
                                "rejected_completion_token_count": len(rejected_token_ids) if rejected_token_ids else None,
                                "stats": generator_stats,
                            },
                        )
                    else:
                        if generator_skipped_reason is None:
                            generator_skipped_reason = "dpo_missing_completion"
                        self._generator_update_mode_counts["skipped"] = (
                            self._generator_update_mode_counts.get("skipped", 0) + 1
                        )
                        self._append_jsonl(
                            self.policy_updates_log_path,
                            {
                                "step": step,
                                "role": "generator",
                                "update_rule": "dpo",
                                "skipped": True,
                                "reason": generator_skipped_reason,
                                "candidate_idx": int(best_idx),
                                "spec_quality": spec_quality,
                            },
                        )
                        self._append_jsonl(
                            self.dpo_pairs_log_path,
                            {
                                "step": step,
                                "skipped": True,
                                "reason": generator_skipped_reason,
                                "candidate_count": len(scored),
                                "best_idx": int(best_idx),
                            },
                        )
                else:
                    if generator_skipped_reason is None:
                        generator_skipped_reason = "dpo_pair_not_ready"
                    self._generator_update_mode_counts["skipped"] = (
                        self._generator_update_mode_counts.get("skipped", 0) + 1
                    )
                    policy_skip_payload: Dict[str, object] = {
                        "step": step,
                        "role": "generator",
                        "update_rule": "dpo",
                        "skipped": True,
                        "reason": generator_skipped_reason,
                        "candidate_count": len(scored),
                        "spec_quality": spec_quality,
                    }
                    pair_skip_payload: Dict[str, object] = {
                        "step": step,
                        "skipped": True,
                        "reason": generator_skipped_reason,
                        "candidate_count": len(scored),
                        "best_idx": int(best_idx),
                    }
                    if chosen_idx is not None and rejected_idx is not None:
                        policy_skip_payload.update(
                            {
                                "chosen_candidate_idx": int(chosen_idx),
                                "rejected_candidate_idx": int(rejected_idx),
                                "chosen_reward": chosen_reward,
                                "rejected_reward": rejected_reward,
                                "reward_gap": reward_gap,
                                "min_reward_gap": float(self.cfg.dpo_min_reward_gap),
                                "chosen_spec": chosen_spec,
                                "rejected_spec": rejected_spec,
                                "spec_gap": spec_gap,
                                "min_spec_gap": float(getattr(self.cfg, "dpo_min_spec_gap", 0.0)),
                                "chosen_confidence": chosen_confidence,
                                "rejected_confidence": rejected_confidence,
                                "confidence_gap": confidence_gap,
                                "min_confidence_gap": float(getattr(self.cfg, "dpo_min_confidence_gap", 0.0)),
                                "chosen_contradiction": chosen_contradiction,
                                "rejected_contradiction": rejected_contradiction,
                                "max_contradiction": contradiction_max,
                                "dpo_max_contradiction": float(getattr(self.cfg, "dpo_max_contradiction", 1.0)),
                            }
                        )
                        pair_skip_payload.update(
                            {
                                "chosen_candidate_idx": int(chosen_idx),
                                "rejected_candidate_idx": int(rejected_idx),
                                "chosen_reward": chosen_reward,
                                "rejected_reward": rejected_reward,
                                "reward_gap": reward_gap,
                                "min_reward_gap": float(self.cfg.dpo_min_reward_gap),
                                "chosen_spec": chosen_spec,
                                "rejected_spec": rejected_spec,
                                "spec_gap": spec_gap,
                                "min_spec_gap": float(getattr(self.cfg, "dpo_min_spec_gap", 0.0)),
                                "chosen_confidence": chosen_confidence,
                                "rejected_confidence": rejected_confidence,
                                "confidence_gap": confidence_gap,
                                "min_confidence_gap": float(getattr(self.cfg, "dpo_min_confidence_gap", 0.0)),
                                "chosen_contradiction": chosen_contradiction,
                                "rejected_contradiction": rejected_contradiction,
                                "max_contradiction": contradiction_max,
                                "dpo_max_contradiction": float(getattr(self.cfg, "dpo_max_contradiction", 1.0)),
                            }
                        )
                    self._append_jsonl(self.policy_updates_log_path, policy_skip_payload)
                    self._append_jsonl(self.dpo_pairs_log_path, pair_skip_payload)
            elif update_rule == "grpo":
                # GRPO path: use ALL scored candidates, not just best/worst pair.
                # To avoid mixing token-trace and proxy-caption completions under
                # different prompt contexts, we force a single mode for the whole
                # group: if ANY candidate lacks a token trace, generate proxy
                # captions for ALL candidates so the prompt is consistent.
                grpo_completions: list = []
                grpo_rewards: list = []
                grpo_images: list = []
                grpo_token_ids: list = []

                any_needs_proxy = any(
                    not str(sc.get("policy_completion", "")).strip()
                    for sc in scored
                )
                use_proxy_for_all = any_needs_proxy and (
                    (self.cfg.generator_missing_trace_strategy or "proxy").strip().lower() == "proxy"
                )

                if use_proxy_for_all:
                    generator_update_mode = "proxy_caption"
                    for sc in scored:
                        img_i = sc.get("image")
                        if isinstance(img_i, Image.Image) and self._proxy_updates_allowed():
                            proxy_comp = self._proxy_generator_completion(img_i)
                            if proxy_comp:
                                grpo_completions.append(proxy_comp)
                                grpo_rewards.append(float(sc["total_reward"]))
                                grpo_images.append(img_i)
                                grpo_token_ids.append(None)
                else:
                    # All candidates have token traces — use them directly
                    generator_update_mode = "token_trace"
                    for sc in scored:
                        comp_i = str(sc.get("policy_completion", "")).strip()
                        if comp_i:
                            grpo_completions.append(comp_i)
                            grpo_rewards.append(float(sc["total_reward"]))
                            img_i = sc.get("image")
                            grpo_images.append(img_i if isinstance(img_i, Image.Image) else None)
                            tid_i = sc.get("policy_completion_ids")
                            grpo_token_ids.append(tid_i if isinstance(tid_i, list) else None)

                local_can_update = len(grpo_completions) >= 2
                grpo_skip_reason: Optional[str] = None
                if not local_can_update:
                    grpo_skip_reason = "grpo_too_few_completions"

                can_update, generator_skipped_reason = _global_update_ready(
                    local_can_update,
                    grpo_skip_reason,
                    peer_reason="distributed_peer_skip",
                )

                if can_update:
                    # generator_update_mode is already set above (proxy_caption or token_trace)
                    grpo_update_prompt = GENERATOR_PROXY_CAPTION_PROMPT if generator_update_mode == "proxy_caption" else str(spec.prompt)
                    generator_stats = self.generator_updater.step(
                        prompt=grpo_update_prompt,
                        completions=grpo_completions,
                        rewards=grpo_rewards,
                        device=self.device,
                        images=grpo_images,
                        completion_token_ids=grpo_token_ids,
                    )
                    if generator_stats.get("did_step", True):
                        self._policy_update_counts["generator"] += 1
                        self._generator_update_mode_counts[generator_update_mode] = (
                            self._generator_update_mode_counts.get(generator_update_mode, 0) + 1
                        )
                    self._update_baseline("generator", generator_reward)

                    self._append_jsonl(
                        self.policy_updates_log_path,
                        {
                            "step": step,
                            "role": "generator",
                            "update_rule": "grpo",
                            "reward": generator_reward,
                            "baseline_before": baseline_before,
                            "baseline_after": self.generator_baseline,
                            "stats": generator_stats,
                            "update_mode": generator_update_mode,
                            "update_prompt": grpo_update_prompt,
                            "group_size": len(grpo_completions),
                            "group_rewards": grpo_rewards,
                            "best_idx": int(best_idx),
                            "spec_quality": spec_quality,
                        },
                    )
                else:
                    if generator_skipped_reason is None:
                        generator_skipped_reason = "grpo_update_failed"
                    self._generator_update_mode_counts["skipped"] = (
                        self._generator_update_mode_counts.get("skipped", 0) + 1
                    )
                    self._append_jsonl(
                        self.policy_updates_log_path,
                        {
                            "step": step,
                            "role": "generator",
                            "update_rule": "grpo",
                            "skipped": True,
                            "reason": generator_skipped_reason,
                            "valid_completions": len(grpo_completions),
                            "candidate_count": len(scored),
                            "spec_quality": spec_quality,
                        },
                    )
            else:
                # REINFORCE path
                completion = str(best.get("policy_completion", "")).strip()
                completion_token_ids = best.get("policy_completion_ids")
                if not isinstance(completion_token_ids, list):
                    completion_token_ids = None
                update_prompt = str(best.get("policy_prompt", spec.prompt))
                update_image: Optional[Image.Image] = None

                if not completion:
                    strategy = (self.cfg.generator_missing_trace_strategy or "proxy").strip().lower()
                    if self.cfg.strict_require_generation_tokens or strategy == "error":
                        generator_skipped_reason = "missing_generation_token_trace_strict"
                        strategy = "skip"

                    if strategy == "proxy":
                        if not self._proxy_updates_allowed():
                            generator_skipped_reason = "proxy_budget_exceeded"
                            strategy = "skip"
                        best_image = best.get("image")
                        if generator_skipped_reason is None and isinstance(best_image, Image.Image):
                            proxy_completion = self._proxy_generator_completion(best_image)
                            if proxy_completion:
                                completion = proxy_completion
                                completion_token_ids = None
                                update_prompt = GENERATOR_PROXY_CAPTION_PROMPT
                                update_image = best_image
                                generator_update_mode = "proxy_caption"
                            else:
                                generator_skipped_reason = "missing_trace_proxy_empty_completion"
                        elif generator_skipped_reason is None:
                            generator_skipped_reason = "missing_trace_proxy_missing_image"
                    elif strategy == "skip":
                        generator_skipped_reason = "missing_generation_token_trace"
                    else:
                        raise ValueError(
                            "Unsupported generator_missing_trace_strategy="
                            f"{self.cfg.generator_missing_trace_strategy!r}. Expected one of: proxy, skip, error."
                        )

                local_can_update = bool(completion)
                can_update, generator_skipped_reason = _global_update_ready(
                    local_can_update,
                    generator_skipped_reason,
                    peer_reason="distributed_peer_skip",
                )

                if can_update:
                    if generator_update_mode is None:
                        generator_update_mode = "token_trace"
                    generator_stats = self.generator_updater.step(
                        prompt=update_prompt,
                        completion=completion,
                        reward=generator_reward,
                        baseline=baseline_before,
                        device=self.device,
                        image=update_image,
                        completion_token_ids=completion_token_ids,
                    )
                    if generator_stats.get("did_step", True):
                        self._policy_update_counts["generator"] += 1
                        self._generator_update_mode_counts[generator_update_mode] = (
                            self._generator_update_mode_counts.get(generator_update_mode, 0) + 1
                        )
                    self._update_baseline("generator", generator_reward)

                    self._append_jsonl(
                        self.policy_updates_log_path,
                        {
                            "step": step,
                            "role": "generator",
                            "update_rule": "reinforce",
                            "reward": generator_reward,
                            "baseline_before": baseline_before,
                            "baseline_after": self.generator_baseline,
                            "stats": generator_stats,
                            "update_mode": generator_update_mode,
                            "update_prompt": update_prompt,
                            "used_image_conditioning": update_image is not None,
                            "completion_char_len": len(completion),
                            "completion_token_count": len(completion_token_ids) if completion_token_ids else None,
                            "candidate_idx": int(best_idx),
                            "spec_quality": spec_quality,
                        },
                    )
                else:
                    if generator_skipped_reason is None:
                        generator_skipped_reason = "missing_generation_token_trace"
                    self._generator_update_mode_counts["skipped"] = (
                        self._generator_update_mode_counts.get("skipped", 0) + 1
                    )
                    self._append_jsonl(
                        self.policy_updates_log_path,
                        {
                            "step": step,
                            "role": "generator",
                            "update_rule": "reinforce",
                            "skipped": True,
                            "reason": generator_skipped_reason,
                            "candidate_idx": int(best_idx),
                            "spec_quality": spec_quality,
                        },
                    )
            self._sync_state_scalars()
        elif generator_update_due:
            if local_quality_ok and not quality_ok_all:
                generator_skipped_reason = "distributed_peer_low_spec_quality"
            else:
                generator_skipped_reason = "low_spec_quality"
            self._generator_update_mode_counts["skipped"] = (
                self._generator_update_mode_counts.get("skipped", 0) + 1
            )
            self._append_jsonl(
                self.policy_updates_log_path,
                {
                    "step": step,
                    "role": "generator",
                    "update_rule": self.cfg.generator_update_rule,
                    "skipped": True,
                    "reason": generator_skipped_reason,
                    "spec_quality": spec_quality,
                    "min_spec_quality_for_update": self.cfg.min_spec_quality_for_update,
                },
            )
            self._sync_state_scalars()

        proposer_reward = float(best["total_reward"])
        proposer_skip_reason = None
        proposer_update_due = False
        if self.cfg.proposer_update_freq > 0:
            phase_due_fn = getattr(self, "_is_proposer_update_due", None)
            if callable(phase_due_fn):
                try:
                    proposer_update_due = bool(phase_due_fn(step, phase="generation"))
                except Exception:
                    proposer_update_due = bool(step % self.cfg.proposer_update_freq == 0)
            else:
                proposer_update_due = bool(step % self.cfg.proposer_update_freq == 0)
        if proposer_update_due:
            baseline_before = self.proposer_baseline
            proposer_completion = str(spec.raw_output or "").strip()
            local_proposer_can_update = bool(proposer_completion)
            proposer_can_update, proposer_skip_reason = _global_update_ready(
                local_proposer_can_update,
                None if local_proposer_can_update else "proposer_empty_completion",
                peer_reason="distributed_peer_proposer_skip",
            )

            if proposer_can_update:
                proposer_stats = self.proposer_updater.step(
                    image=image,
                    prompt=build_generation_spec_prompt(
                        target_difficulty=str(getattr(self.cfg, "unicorn_target_difficulty", "medium"))
                    ),
                    completion=proposer_completion,
                    reward=proposer_reward,
                    baseline=baseline_before,
                    device=self.device,
                )
                if proposer_stats.get("did_step", True):
                    self._policy_update_counts["proposer"] += 1
                self._update_baseline("proposer", proposer_reward)
                self._append_jsonl(
                    self.policy_updates_log_path,
                    {
                        "step": step,
                        "role": "proposer",
                        "reward": proposer_reward,
                        "baseline_before": baseline_before,
                        "baseline_after": self.proposer_baseline,
                        "stats": proposer_stats,
                    },
                )
            else:
                self._append_jsonl(
                    self.policy_updates_log_path,
                    {
                        "step": step,
                        "role": "proposer",
                        "skipped": True,
                        "reason": proposer_skip_reason,
                        "baseline_before": baseline_before,
                    },
                )
            self._sync_state_scalars()
        else:
            proposer_skip_reason = "update_not_due"

        unicorn_recon_enqueued = self._enqueue_unicorn_reconstruction_tasks(
            step=step,
            image=image,
            spec=spec,
            best=best,
            spec_quality=float(spec_quality),
        )
        unicorn_reconstruction = self._run_unicorn_reconstruction_sft(step)
        unicorn_reconstruction["enqueued_this_step"] = int(unicorn_recon_enqueued)
        unicorn_reconstruction["buffer_size_after_step"] = int(len(self._unicorn_reconstruction_buffer))

        self._sync_state_scalars()

        self._save_candidate_images(step=step, scored=scored, best_idx=best_idx)

        if verbose:
            step_dt = time.perf_counter() - step_t0
            print(
                f"[Step {step:05d}][G] generation phase done in {step_dt:.1f}s "
                f"(best_idx={best_idx}, best_reward={float(best['total_reward']):.3f})"
            )

        self._append_jsonl(
            self.prompts_log_path,
            {
                "step": step,
                "image_path": meta.get("path"),
                "source_caption": source_caption,
                "prompt": spec.prompt,
                "qa_pairs": [dataclasses.asdict(qa) for qa in spec.qa_pairs],
                "fallback_used": spec.fallback_used,
                "spec_quality": spec_quality,
                "spec_quality_details": spec_quality_details,
                "raw_output": spec.raw_output,
            },
        )

        for cand in scored:
            qa_logs = cand["qa_logs"]
            self._append_jsonl(
                self.candidates_log_path,
                {
                    "step": step,
                    "image_path": meta.get("path"),
                    "candidate_idx": cand["candidate_idx"],
                    "is_best": cand["candidate_idx"] == best_idx,
                    "backend": cand.get("backend"),
                    "policy_prompt": cand.get("policy_prompt"),
                    "policy_completion": cand.get("policy_completion"),
                    "spec_score": cand["spec_score"],
                    "contradiction_score": cand["contradiction_score"],
                    "cycle_score": cand["cycle_score"],
                    "cycle_caption": cand["cycle_caption"],
                    "diversity_score": cand["diversity_score"],
                    "qa_confidence": cand.get("qa_confidence", 0.0),
                    "base_reward": cand["base_reward"],
                    "spec_quality": cand["spec_quality"],
                    "total_reward": cand["total_reward"],
                    "qa_logs": qa_logs,
                },
            )

        self._append_jsonl(
            self.rewards_log_path,
            {
                "step": step,
                "image_path": meta.get("path"),
                "prompt": spec.prompt,
                "reward_components": {
                    "spec_weight": self.cfg.reward_spec_weight,
                    "cycle_weight": self.cfg.reward_cycle_weight,
                    "diversity_weight": self.cfg.reward_diversity_weight,
                    "contradiction_weight": self.cfg.reward_contradiction_weight,
                },
                "spec_quality": spec_quality,
                "spec_quality_details": spec_quality_details,
                "candidate_rewards": [float(c["total_reward"]) for c in scored],
                "best_idx": int(best_idx),
                "best_reward": float(best["total_reward"]),
                "best_spec_score": float(best["spec_score"]),
                "best_cycle_score": float(best["cycle_score"]),
                "best_diversity_score": float(best["diversity_score"]),
                "best_contradiction_score": float(best["contradiction_score"]),
                "best_qa_confidence": float(best.get("qa_confidence", 0.0)),
                "generator_baseline": self.generator_baseline,
                "proposer_baseline": self.proposer_baseline,
                "generator_update_rule": self.cfg.generator_update_rule,
                "generator_skipped_reason": generator_skipped_reason,
                "generator_update_mode": generator_update_mode,
                "generator_proxy_ratio": float(self._current_proxy_ratio()),
                "unicorn_spec_meta": unicorn_spec_meta,
                "unicorn_reconstruction": unicorn_reconstruction,
                "proposer_update_due": proposer_update_due,
                "proposer_skip_reason": proposer_skip_reason,
                "proposer_reward": proposer_reward,
                "proposer_stats": proposer_stats,
                "generator_update_stats": generator_stats,
            },
        )

        return {
            "source_caption": source_caption,
            "spec": spec,
            "spec_quality": spec_quality,
            "spec_quality_details": spec_quality_details,
            "scored": scored,
            "best_idx": best_idx,
            "reference_questions": _ref_questions,
            "reference_answers": _ref_answers,
            "proposer_stats": proposer_stats,
            "generator_stats": generator_stats,
            "generator_update_rule": self.cfg.generator_update_rule,
            "generator_skipped_reason": generator_skipped_reason,
            "generator_update_mode": generator_update_mode,
            "generator_proxy_ratio": float(self._current_proxy_ratio()),
            "unicorn_spec_meta": unicorn_spec_meta,
            "unicorn_reconstruction": unicorn_reconstruction,
            "proposer_update_due": proposer_update_due,
            "proposer_skip_reason": proposer_skip_reason,
            "proposer_reward": proposer_reward,
        }

    def _solver_synthetic_update_from_best(self, step: int, best: Dict[str, object]):
        if self.solver_updater is None:
            return
        if self.cfg.solver_update_freq <= 0 or step % self.cfg.solver_update_freq != 0:
            return

        image = best.get("image")
        if not isinstance(image, Image.Image):
            return

        qa_logs = best.get("qa_logs", [])
        valid_qas: List[Dict[str, object]] = []
        hard_only = bool(getattr(self.cfg, "synthetic_solver_hard_only", False))
        min_entropy = float(getattr(self.cfg, "solver_hardness_min_entropy", 0.2))
        for qa in qa_logs:
            question = str(qa.get("question", "")).strip()
            if not question:
                continue
            if hard_only:
                entropy = None
                solver_info = qa.get("solver")
                if isinstance(solver_info, dict):
                    try:
                        entropy = float(solver_info.get("entropy_nats", 0.0))
                    except Exception:
                        entropy = None
                if entropy is None or entropy < min_entropy:
                    continue
            valid_qas.append(qa)

        shared_qa_count = len(valid_qas)
        if self.distributed and dist.is_initialized():
            shared_qa_count = self._dist_max_int(shared_qa_count)

        for qa_idx in range(shared_qa_count):
            has_local_qa = qa_idx < len(valid_qas)
            any_rank_has_qa = self._dist_any_bool(has_local_qa)
            if not any_rank_has_qa:
                continue

            qa = valid_qas[qa_idx] if has_local_qa else {}
            question = str(qa.get("question", "")).strip()

            if has_local_qa and question:
                completion = self._generate(
                    image=image,
                    prompt=build_solver_prompt(question),
                    adapter_name="default" if self.cfg.use_lora else None,
                    max_new_tokens=self.cfg.max_new_tokens_solver,
                    temperature=self.cfg.temp,
                    top_p=self.cfg.top_p,
                ).strip()
            else:
                completion = ""
            local_has_completion = bool(completion)
            if not local_has_completion:
                self._append_jsonl(
                    self.policy_updates_log_path,
                    {
                        "step": step,
                        "role": "solver",
                        "source": "synthetic_generation",
                        "skipped": True,
                        "reason": "empty_solver_completion_local",
                        "qa_idx": int(qa_idx),
                        "question": question,
                    },
                )
            effective_completion = completion if local_has_completion else ""
            reward = float(qa.get("combined_score", 0.0)) if (has_local_qa and local_has_completion) else 0.0

            baseline_before = self.solver_baseline
            stats = self.solver_updater.step(
                image=image,
                prompt=build_solver_prompt(question),
                completion=effective_completion,
                reward=reward,
                baseline=baseline_before if local_has_completion else 0.0,
                device=self.device,
            )
            if stats.get("did_step", True):
                self._policy_update_counts["solver"] += 1
            if local_has_completion:
                self._update_baseline("solver", reward)
            self._sync_state_scalars()

            self._append_jsonl(
                self.policy_updates_log_path,
                {
                    "step": step,
                    "role": "solver",
                    "source": "synthetic_generation",
                    "question": question,
                    "reward": reward,
                    "baseline_before": baseline_before,
                    "baseline_after": self.solver_baseline,
                    "stats": stats,
                },
            )

    def train(self):
        cfg = self.cfg
        if cfg.total_steps <= self.start_step:
            raise ValueError(
                f"total_steps ({cfg.total_steps}) must be greater than start_step ({self.start_step})."
            )

        if self.is_main_process:
            print(f"[Generation] Starting run at: {self.run_dir}")
            print(f"[Generation] Model: {cfg.model_name}")
            print(f"[Generation] Generator update rule: {cfg.generator_update_rule}")
            print(f"[Generation] Images: {len(self.pool)}")
            print(f"[Generation] Step range: {self.start_step + 1}..{cfg.total_steps}")
            if self.distributed:
                print(
                    f"[Generation] Distributed mode: world_size={self.world_size}, "
                    f"effective_batch_per_step={self.world_size}"
                )

        last_completed_step = self.start_step
        last_attempted_step = self.start_step
        try:
            for step in range(self.start_step + 1, cfg.total_steps + 1):
                last_attempted_step = step
                step_t0 = time.perf_counter()

                image, meta = self._sample_image_for_step(step)
                out = self._generation_step(step=step, image=image, meta=meta)
                spec: GenerationSpec = out["spec"]
                spec_quality = float(out["spec_quality"])
                scored: List[Dict[str, object]] = out["scored"]
                best_idx = int(out["best_idx"])

                if self.solver_updater is not None:
                    self._solver_synthetic_update_from_best(step, scored[best_idx])

                rewards = [float(c["total_reward"]) for c in scored]
                reward_mean = sum(rewards) / max(1, len(rewards))
                reward_max = max(rewards) if rewards else 0.0
                reward_min = min(rewards) if rewards else 0.0
                step_duration_sec = time.perf_counter() - step_t0

                reward_mean_g = self._dist_mean(reward_mean)
                reward_max_g = self._dist_mean(reward_max)
                reward_min_g = self._dist_mean(reward_min)
                step_duration_g = self._dist_mean(step_duration_sec)
                spec_quality_g = self._dist_mean(spec_quality)

                best = scored[best_idx]
                best_spec = float(best["spec_score"])
                best_cycle = float(best["cycle_score"])
                best_div = float(best["diversity_score"])
                best_contra = float(best["contradiction_score"])

                best_spec_g = self._dist_mean(best_spec)
                best_cycle_g = self._dist_mean(best_cycle)
                best_div_g = self._dist_mean(best_div)
                best_contra_g = self._dist_mean(best_contra)

                if self.is_main_process and step % cfg.log_every == 0:
                    print(
                        f"[Step {step:05d}] R_mean={reward_mean_g:.3f} R_max={reward_max_g:.3f} "
                        f"spec={best_spec_g:.3f} cycle={best_cycle_g:.3f} div={best_div_g:.3f} contra={best_contra_g:.3f}"
                    )
                    print(f"  Prompt: {spec.prompt}")

                self._append_jsonl(
                    self.iter_log_path,
                    {
                        "step": step,
                        "phase": "generation",
                        "image_path": meta.get("path"),
                        "prompt": spec.prompt,
                        "qa_pairs": [dataclasses.asdict(qa) for qa in spec.qa_pairs],
                        "fallback_used": spec.fallback_used,
                        "spec_quality": spec_quality,
                        "spec_quality_details": out.get("spec_quality_details"),
                        "candidate_rewards": rewards,
                        "best_idx": best_idx,
                        "best_metrics": {
                            "spec_score": best_spec,
                            "cycle_score": best_cycle,
                            "diversity_score": best_div,
                            "contradiction_score": best_contra,
                            "total_reward": float(best["total_reward"]),
                        },
                        "generator_baseline": self.generator_baseline,
                        "proposer_baseline": self.proposer_baseline,
                        "solver_baseline": self.solver_baseline,
                        "generator_update_rule": self.cfg.generator_update_rule,
                        "generator_kl_coef": self.generator_updater.kl_coef,
                        "proposer_kl_coef": self.proposer_updater.kl_coef,
                        "solver_kl_coef": self.solver_updater.kl_coef if self.solver_updater is not None else None,
                        "generator_skipped_reason": out.get("generator_skipped_reason"),
                        "unicorn_spec_meta": out.get("unicorn_spec_meta"),
                        "unicorn_reconstruction": out.get("unicorn_reconstruction"),
                        "step_duration_sec": step_duration_sec,
                    },
                )

                self._wandb_log_step(
                    step=step,
                    image_path=meta.get("path"),
                    source_caption=str(out["source_caption"]),
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
                    proposer_stats=out["proposer_stats"],
                    generator_stats=out["generator_stats"],
                    unicorn_spec_meta=out.get("unicorn_spec_meta"),
                    unicorn_reconstruction=out.get("unicorn_reconstruction"),
                )

                self._update_metric("reward_mean", reward_mean_g)
                self._update_metric("reward_max", reward_max_g)
                self._update_metric("reward_min", reward_min_g)
                self._update_metric("best_spec_score", best_spec_g)
                self._update_metric("best_cycle_score", best_cycle_g)
                self._update_metric("best_diversity_score", best_div_g)
                self._update_metric("best_contradiction_score", best_contra_g)
                self._update_metric("spec_quality", spec_quality_g)
                self._update_metric("generator_kl_coef", float(self.generator_updater.kl_coef))
                self._update_metric("proposer_kl_coef", float(self.proposer_updater.kl_coef))
                self._update_metric("step_duration_sec", step_duration_g)
                self._update_metric("spec_fallback_used", 1.0 if spec.fallback_used else 0.0)
                unicorn_meta = out.get("unicorn_spec_meta") or {}
                self._update_metric("unicorn_spec_attempts", float(unicorn_meta.get("attempts", 1.0)))
                self._update_metric("unicorn_spec_alignment", float(unicorn_meta.get("selected_alignment", 0.0)))
                self._update_metric(
                    "unicorn_spec_selected_accepted",
                    1.0 if bool(unicorn_meta.get("selected_accepted", False)) else 0.0,
                )
                unicorn_recon = out.get("unicorn_reconstruction") or {}
                self._update_metric(
                    "unicorn_reconstruction_applied_updates",
                    float(unicorn_recon.get("applied_updates", 0.0)),
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
                print(f"[Generation] Training complete. Final checkpoint at step {cfg.total_steps:05d}.")

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            interrupted_step = int(last_attempted_step)
            tb = traceback.format_exc()
            if self.is_main_process:
                print(f"[Generation] Training interrupted at step {interrupted_step}: {error_text}")
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
                    print(f"[Generation] Emergency checkpoint saved at step {emergency_step:05d}.")
                    _json_dump(
                        self.run_dir / "resume_hint.json",
                        {
                            "resume_from": str(self.run_dir / f"step_{emergency_step:05d}"),
                            "start_step": emergency_step,
                            "total_steps": cfg.total_steps,
                            "command_example": (
                                "python BLIP3o/blip3o/train/train_self_evolving.py "
                                f"--experiment {cfg.experiment_name} --data_dir {cfg.data_dir} "
                                f"--output_dir {cfg.output_dir} --run_name {self.run_dir.name} "
                                f"--resume_from {self.run_dir / f'step_{emergency_step:05d}'} "
                                f"--start_step {emergency_step} --total_steps {cfg.total_steps}"
                            ),
                        },
                    )
            except Exception as save_exc:
                if self.is_main_process:
                    print(f"[Generation] Emergency checkpoint failed: {save_exc}")

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
