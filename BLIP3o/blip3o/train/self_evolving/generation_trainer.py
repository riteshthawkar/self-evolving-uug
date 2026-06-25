"""
Generation-only and unified self-evolving trainers.

Ported from self_evolving/experiments/generation.py.
Uses native BLIP3o model loading instead of the workaround-heavy path.
"""

import contextlib
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
from .checkpoint_adapters import sanitize_peft_adapter_dir
from .dit_updater import DiTUpdater
from .image_pool import ImagePool, ImagePoolConfig
from .policy_updater import RolePolicyUpdater
from .prompts import (
    build_generation_spec_prompt,
    build_generation_spec_retry_prompt,
    build_imageless_spec_prompt,
    build_proposer_prompt,
    build_solver_prompt,
    _sample_imageless_topic,
)
from .utils import (
    HAS_PEFT,
    HAS_WANDB,
    _append_training_monitor_record,
    _append_training_watch_record,
    _build_chat_text,
    _build_text_only_chat,
    _clip_grad_norm_multi_device,
    _collect_git_info,
    _collect_trainable_params,
    _decode_tokens,
    _infer_primary_device,
    _json_dump,
    _parse_answer,
    _parse_first_question,
    _prepare_mm_inputs,
    _prepare_text_only_inputs,
    _resolve_attn_implementation,
    _safe_dtype,
    _save_code_run_registry,
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
    try:
        from peft import prepare_model_for_kbit_training
    except Exception:
        prepare_model_for_kbit_training = None
else:
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

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
    _jaccard_similarity,
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


_VISUAL_BRIDGE_TARGET_MARKERS = ("mm_projector", "visual.merger", "merger.mlp")


def _target_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def _dedupe_targets(*groups: Tuple[str, ...]) -> Tuple[str, ...]:
    seen = set()
    merged: List[str] = []
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


def _is_dit_param_name(name: str) -> bool:
    return ".dit." in name or name.startswith("dit.")


def _disable_gradient_checkpointing_for_peft_wrap(module: torch.nn.Module) -> bool:
    """Disable checkpointing before wrapping a non-text DiT with PEFT.

    PEFT checks ``is_gradient_checkpointing`` during ``get_peft_model`` and, for
    PreTrainedModel instances, installs an input-embedding grad hook.  BLIP3o's
    DiT is a PreTrainedModel wrapper around a diffusion transformer and does not
    implement text ``get_input_embeddings()``, so that hook path raises
    NotImplementedError.  Disabling checkpointing before the adapter injection
    avoids the text-only hook while preserving the same LoRA parameters and
    denoising objective.
    """
    disabled_any = False
    try:
        disable = getattr(module, "gradient_checkpointing_disable", None)
        if callable(disable):
            disable()
            disabled_any = True
    except Exception:
        pass

    for submodule in module.modules():
        for attr in ("gradient_checkpointing", "_gradient_checkpointing"):
            if not hasattr(submodule, attr):
                continue
            try:
                if bool(getattr(submodule, attr)):
                    disabled_any = True
                setattr(submodule, attr, False)
            except Exception:
                pass
        cfg = getattr(submodule, "config", None)
        if cfg is not None and hasattr(cfg, "_gradient_checkpointing"):
            try:
                if bool(getattr(cfg, "_gradient_checkpointing")):
                    disabled_any = True
                setattr(cfg, "_gradient_checkpointing", False)
            except Exception:
                pass
    return disabled_any


def _build_lora_config(
    *,
    r: int,
    alpha: int,
    dropout: float,
    targets: Tuple[str, ...],
    task_type: Optional[Any] = None,
    rank_pattern: Optional[Dict[str, int]] = None,
    alpha_pattern: Optional[Dict[str, int]] = None,
):
    kwargs = {
        "r": int(r),
        "lora_alpha": int(alpha),
        "lora_dropout": float(dropout),
        "target_modules": list(targets),
        "bias": "none",
    }
    if task_type is not None:
        kwargs["task_type"] = task_type
    if rank_pattern:
        kwargs["rank_pattern"] = dict(rank_pattern)
    if alpha_pattern:
        kwargs["alpha_pattern"] = dict(alpha_pattern)
    try:
        return LoraConfig(**kwargs)
    except TypeError:
        kwargs.pop("rank_pattern", None)
        kwargs.pop("alpha_pattern", None)
        return LoraConfig(**kwargs)


def _count_lora_trainables(model: torch.nn.Module) -> Dict[str, int]:
    counts = {
        "solver": 0,
        "solver_merger": 0,
        "proposer": 0,
        "generator": 0,
        "dit": 0,
    }
    for name, param in model.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        n_params = int(param.numel())
        if _is_dit_param_name(name):
            counts["dit"] += n_params
        elif ".default." in name:
            counts["solver"] += n_params
            if any(marker in name for marker in ("visual.merger", "merger.mlp", "mm_projector")):
                counts["solver_merger"] += n_params
        elif ".proposer." in name:
            counts["proposer"] += n_params
        elif ".generator." in name:
            counts["generator"] += n_params
    return counts


# Strict-imageless fallback when no external image pool is available.
class _SyntheticImagePool:
    def __init__(self, seed: int = 42):
        self.indices = [0]
        self._seed = int(seed)

    def __len__(self) -> int:
        return 1

    def get_image(self, idx: int) -> Tuple[Image.Image, Dict]:
        rng = random.Random(self._seed + int(idx))
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        image = Image.new("RGB", (224, 224), color=color)
        meta = {
            "path": None,
            "dataset": "synthetic",
            "split": "train",
            "subfolder": "synthetic",
            "filename": f"synthetic_{int(idx)}.png",
            "source": "synthetic_pool",
        }
        return image, meta


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
        self.proposer_gen_baseline = self._dist_mean(self.proposer_gen_baseline)
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
        self.cfg.generator_missing_trace_strategy = str(
            getattr(self.cfg, "generator_missing_trace_strategy", "skip") or "skip"
        ).strip().lower()
        if self.cfg.generator_missing_trace_strategy not in {"proxy", "skip", "error"}:
            raise ValueError(
                "Unsupported generator_missing_trace_strategy="
                f"{self.cfg.generator_missing_trace_strategy!r}. Expected one of: proxy, skip, error."
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
        self.cfg.dit_update_enabled = bool(getattr(self.cfg, "dit_update_enabled", False))
        self.cfg.require_dit_update = bool(getattr(self.cfg, "require_dit_update", False))
        if self.cfg.require_dit_update and not self.cfg.dit_update_enabled:
            raise ValueError("--require_dit_update requires --dit_update_enabled.")
        self.cfg.dit_update_freq = max(1, int(getattr(self.cfg, "dit_update_freq", 1)))
        self.cfg.dit_grad_accum_steps = max(1, int(getattr(self.cfg, "dit_grad_accum_steps", 1)))
        self.cfg.dit_lr = float(getattr(self.cfg, "dit_lr", self.cfg.lr))
        self.cfg.dit_weight_decay = float(
            getattr(self.cfg, "dit_weight_decay", self.cfg.weight_decay)
        )
        self.cfg.dit_grad_clip = float(getattr(self.cfg, "dit_grad_clip", self.cfg.grad_clip))
        self.cfg.dit_conditioning_dropout = float(
            getattr(self.cfg, "dit_conditioning_dropout", 0.10)
        )
        self.cfg.dit_loss_weight = float(getattr(self.cfg, "dit_loss_weight", 1.0))
        self.cfg.dit_prompt_suffix_token_id = int(
            getattr(self.cfg, "dit_prompt_suffix_token_id", 151665)
        )
        # Generator reward EMA for monitoring
        self._gen_reward_ema = float(getattr(self, "_gen_reward_ema", 0.0))
        self._gen_reward_ema_initialized = bool(
            getattr(self, "_gen_reward_ema_initialized", False)
        )
        self._setup_distributed()

        # Exp 3: Frozen Judge (Unicorn)
        # NOTE: initialize after model/updater/resume setup so cloning uses
        # fully loaded weights and avoids deepcopying a partially-built trainer.
        self.judge = None
        # Previously this block forced generator_missing_trace_strategy=skip when
        # dit_update_enabled=True. That was overly conservative — DiT SFT and GRPO
        # proxy-caption updates operate on different model components (DiT weights vs
        # generator LoRA text adapter) and do not interfere. Removed so that both can
        # run simultaneously when explicitly configured via --generator_missing_trace_strategy proxy.
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
        self.release_rollouts_log_path = self.run_dir / "rollouts.jsonl"
        self.release_generation_rollouts_log_path = self.run_dir / "generation_rollouts.jsonl"
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
        try:
            self.pool = ImagePool(pool_cfg)
        except Exception as exc:
            if bool(getattr(config, "strict_imageless_mode", False)):
                if self.is_main_process:
                    print(
                        "[Generation] strict_imageless_mode=True and image pool unavailable; "
                        f"using synthetic fallback pool ({type(exc).__name__}: {exc})."
                    )
                self.pool = _SyntheticImagePool(seed=config.seed)
            else:
                raise

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

        # Proposer updater: REINFORCE (default) or GRPO.
        # "grpo" is preferred for conference reviewers — group-normalized advantages
        # are lower variance and don't need a separate EMA baseline network.
        _proposer_rule = str(getattr(config, "proposer_update_rule", "grpo") or "grpo").lower()
        if _proposer_rule == "grpo":
            self.proposer_updater = TextGRPOUpdater(
                model=self.train_model,
                processor=self.processor,
                config=config,
                adapter_name="proposer" if config.use_lora else None,
                reference_model=reference_model,
            )
            self._proposer_uses_grpo = True
        else:
            self.proposer_updater = RolePolicyUpdater(
                model=self.train_model,
                processor=self.processor,
                config=config,
                adapter_name="proposer" if config.use_lora else None,
                reference_model=reference_model,
            )
            self._proposer_uses_grpo = False

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
        self.dit_updater: Optional[DiTUpdater] = None
        if bool(getattr(self.cfg, "dit_update_enabled", False)):
            try:
                self.dit_updater = DiTUpdater(
                    model=self.train_model,
                    processor=self.processor,
                    config=config,
                )
                if self.is_main_process:
                    print(
                        "[Generation] DiT updater active: "
                        f"freq={self.cfg.dit_update_freq}, lr={self.cfg.dit_lr:g}, "
                        f"grad_accum={self.cfg.dit_grad_accum_steps}, "
                        f"lora_enabled={bool(getattr(self.cfg, 'dit_lora_enabled', True))}, "
                        f"trainable_params={getattr(self.dit_updater, 'trainable_param_count', 'unknown')}"
                    )
            except Exception as exc:
                self.dit_updater = None
                self.cfg.dit_update_enabled = False
                if bool(getattr(self.cfg, "require_dit_update", False)):
                    raise RuntimeError(
                        "DiT updates are required for this run, but DiTUpdater "
                        f"failed to initialize: {type(exc).__name__}: {exc}"
                    ) from exc
                if self.is_main_process:
                    print(f"[Generation] WARNING: failed to initialize DiT updater, disabling it: {exc}")

        self.generator_baseline = 0.0
        self.proposer_baseline = 0.0
        self.proposer_gen_baseline = 0.0  # separate EMA for generation-phase proposer reward
        self.solver_baseline = 0.0
        self.proposer_entropy_mu_ema = float(config.prop_entropy_mu)
        self.start_step = max(0, int(config.start_step))

        self._metric_stats: Dict[str, Dict[str, float]] = {}
        self._policy_update_counts: Dict[str, int] = {"solver": 0, "proposer": 0, "generator": 0, "dit": 0}
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

        if getattr(config, "enable_frozen_judge", False):
            from .judge import FrozenJudge
            if self.is_main_process:
                print(f"[GenerationTrainer] Initializing FrozenJudge (ema_decay={config.judge_ema_decay})...")
            self.judge = FrozenJudge(self, ema_decay=config.judge_ema_decay)

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
        resume_run_dir = self._infer_resume_run_dir()
        if self.distributed and dist.is_initialized():
            obj = [None]
            if self.is_main_process:
                if resume_run_dir is not None:
                    run_dir = resume_run_dir
                else:
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

        if resume_run_dir is not None:
            run_dir = resume_run_dir
        else:
            timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            run_name = self.cfg.run_name or f"{self.cfg.experiment_name}_{timestamp}"
            run_dir = base_dir / run_name
            if run_dir.exists() and any(run_dir.iterdir()) and not self.cfg.resume_from:
                run_dir = base_dir / f"{run_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _infer_resume_run_dir(self) -> Optional[pathlib.Path]:
        """Resolve the run directory that owns `resume_from` checkpoint path.

        This keeps resumed logs/checkpoints in the original run folder instead of
        creating a sibling run directory based on output_dir/run_name.
        """
        if not self.cfg.resume_from:
            return None

        candidate = pathlib.Path(self.cfg.resume_from).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"resume_from path does not exist: {candidate}")

        if candidate.is_file() and candidate.name == "trainer_state.pt":
            step_dir = candidate.parent
            if step_dir.name.startswith("step_"):
                return step_dir.parent.resolve()
            return step_dir.resolve()

        if candidate.is_dir() and candidate.name.startswith("step_"):
            return candidate.parent.resolve()

        if candidate.is_dir() and (candidate / "trainer_state.pt").exists():
            if candidate.name.startswith("step_"):
                return candidate.parent.resolve()
            return candidate.resolve()

        if candidate.is_dir():
            step_dirs = [
                p for p in candidate.glob("step_*")
                if p.is_dir() and (p / "trainer_state.pt").exists()
            ]
            if step_dirs:
                return candidate.resolve()

        return None

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
                print(f"[Generation] WARNING: failed to write code run registry: {exc}")
        self._dist_barrier()

    def _append_jsonl(self, path: pathlib.Path, record: Dict):
        if not self.is_main_process:
            return
        serialized = json.dumps(record, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(serialized + "\n")
        if path == self.iter_log_path:
            phase = str(record.get("phase", "generation")).strip().lower()
            mirror_path = (
                self.release_rollouts_log_path
                if phase == "understanding"
                else self.release_generation_rollouts_log_path
            )
            with mirror_path.open("a", encoding="utf-8") as f:
                f.write(serialized + "\n")

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
            elif any(payload.get(k) for k in ("generator_skip", "proposer_skip", "solver_skip", "dit_skip")):
                payload["health"] = "skipped_or_waiting"
            elif any(bool(payload.get(k)) for k in ("generator_did_step", "proposer_did_step", "solver_did_step", "dit_did_step")):
                payload["health"] = "optimizer_step"
            else:
                payload["health"] = "observed"
        _append_training_monitor_record(self.monitor_log_path, self.monitor_tsv_path, payload)
        _append_training_watch_record(self.watch_log_path, payload)

    @classmethod
    def _finite_mean_from_stats(cls, stats_list: List[Dict], key: str) -> Optional[float]:
        vals: List[float] = []
        for stats in stats_list:
            if not isinstance(stats, dict):
                continue
            val = cls._monitor_float(stats.get(key))
            if val is not None:
                vals.append(val)
        if not vals:
            return None
        return float(sum(vals) / len(vals))

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
        forced_tail_values = [
            self._monitor_float(s.get("forced_tail_tokens"), 0.0) or 0.0
            for s in solver_stats
            if isinstance(s, dict)
        ]
        forced_tail_values.append(self._monitor_stat(proposer_stats, "forced_tail_tokens", 0.0) or 0.0)
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
                "solver_reward_raw_mean": self._monitor_float(
                    record.get("solver_rewards_raw_mean", record.get("solver_reward_raw_mean"))
                ),
                "solver_reward_soft_mean": self._monitor_float(
                    record.get("solver_rewards_soft_mean", record.get("solver_reward_soft_mean"))
                ),
                "proposer_reward": self._monitor_float(record.get("proposer_reward")),
                "proposer_easy_reward_cap_applied": bool(
                    record.get("proposer_easy_reward_cap_applied", False)
                ),
                "proposer_easy_reward_cap_value": self._monitor_float(
                    record.get("proposer_easy_reward_cap_value")
                ),
                "proposer_easy_reward_cap_reason": str(
                    record.get("proposer_easy_reward_cap_reason") or ""
                ),
                "entropy_nats": self._monitor_float(record.get("entropy_nats")),
                "majority_fraction": self._monitor_float(record.get("majority_fraction")),
                "proposer_did_step": self._monitor_bool(proposer_stats, "did_step"),
                "proposer_skip": str(record.get("proposer_skip_reason") or self._monitor_skip(proposer_stats) or ""),
                "proposer_ce_loss": self._monitor_stat(proposer_stats, "ce_loss"),
                "proposer_kl_loss": self._monitor_stat(proposer_stats, "kl_loss"),
                "proposer_total_loss": self._monitor_stat(proposer_stats, "total_loss"),
                "proposer_valid_tokens": self._monitor_stat(proposer_stats, "valid_token_count"),
                "proposer_kl_coef": self._monitor_stat(
                    proposer_stats,
                    "kl_coef_after",
                    record.get("proposer_kl_coef", getattr(self.proposer_updater, "kl_coef", None)),
                ),
                "solver_did_step": solver_did_step,
                "solver_skip": str(record.get("solver_update_skip_reason") or ",".join(solver_skip_reasons)),
                "solver_ce_loss": self._finite_mean_from_stats(solver_stats, "ce_loss"),
                "solver_kl_loss": self._finite_mean_from_stats(solver_stats, "kl_loss"),
                "solver_total_loss": self._finite_mean_from_stats(solver_stats, "total_loss"),
                "solver_valid_tokens": self._finite_mean_from_stats(solver_stats, "valid_token_count"),
                "solver_kl_coef": self._monitor_float(
                    record.get("solver_kl_coef"),
                    getattr(self.solver_updater, "kl_coef", None) if self.solver_updater is not None else None,
                ),
                "forced_tail_tokens": max(forced_tail_values) if forced_tail_values else 0.0,
                "proposer_baseline": self._monitor_float(record.get("proposer_baseline_after", self.proposer_baseline)),
                "solver_baseline": self._monitor_float(record.get("solver_baseline_after", self.solver_baseline)),
                "step_duration_sec": self._monitor_float(record.get("step_duration_sec")),
                "nan_detected": bool(raw_nonfinite),
                "nonfinite_fields": ",".join(raw_nonfinite[:16]),
            }
        )

    def _monitor_generation_step(
        self,
        *,
        step: int,
        meta: Dict,
        scored: List[Dict[str, object]],
        best_idx: int,
        spec_quality: float,
        generator_stats,
        generator_update_mode,
        generator_effective_objective: str,
        generator_skipped_reason,
        dit_stats,
        dit_skip_reason,
        proposer_stats,
        proposer_skip_reason,
        proposer_reward,
        step_duration_sec: float,
    ):
        rewards = [self._monitor_float(c.get("total_reward")) for c in scored]
        rewards = [r for r in rewards if r is not None]
        best = scored[best_idx] if scored and 0 <= best_idx < len(scored) else {}
        forced_tail = max(
            self._monitor_stat(generator_stats, "forced_tail_tokens", 0.0) or 0.0,
            self._monitor_stat(proposer_stats, "forced_tail_tokens", 0.0) or 0.0,
        )
        raw_nonfinite = self._monitor_nonfinite_fields(
            {
                "generator_stats": generator_stats,
                "dit_stats": dit_stats,
                "proposer_stats": proposer_stats,
            }
        )
        self._append_training_monitor(
            {
                "step": int(step),
                "phase": "generation",
                "image_path": meta.get("path"),
                "reward_mean": sum(rewards) / max(1, len(rewards)),
                "reward_max": max(rewards) if rewards else None,
                "reward_min": min(rewards) if rewards else None,
                "best_reward": self._monitor_float(best.get("total_reward")),
                "spec_quality": self._monitor_float(spec_quality),
                "generator_objective": generator_effective_objective,
                "generator_mode": generator_update_mode,
                "generator_did_step": self._monitor_bool(generator_stats, "did_step"),
                "generator_skip": str(generator_skipped_reason or self._monitor_skip(generator_stats) or ""),
                "generator_ce_loss": self._monitor_stat(generator_stats, "ce_loss"),
                "generator_kl_loss": self._monitor_stat(generator_stats, "kl_loss"),
                "generator_total_loss": self._monitor_stat(generator_stats, "total_loss"),
                "generator_valid_tokens": self._monitor_stat(generator_stats, "valid_token_count"),
                "generator_kl_coef": self._monitor_stat(generator_stats, "kl_coef_after", getattr(self.generator_updater, "kl_coef", None)),
                "dit_did_step": self._monitor_bool(dit_stats, "did_step"),
                "dit_skip": str(dit_skip_reason or self._monitor_skip(dit_stats) or ""),
                "dit_loss": self._monitor_stat(dit_stats, "loss"),
                "dit_objective": str(dit_stats.get("objective", "")) if isinstance(dit_stats, dict) else "",
                "dit_reward": self._monitor_stat(dit_stats, "reward"),
                "proposer_did_step": self._monitor_bool(proposer_stats, "did_step"),
                "proposer_skip": str(proposer_skip_reason or self._monitor_skip(proposer_stats) or ""),
                "proposer_reward": self._monitor_float(proposer_reward),
                "proposer_ce_loss": self._monitor_stat(proposer_stats, "ce_loss"),
                "proposer_kl_loss": self._monitor_stat(proposer_stats, "kl_loss"),
                "proposer_total_loss": self._monitor_stat(proposer_stats, "total_loss"),
                "proposer_valid_tokens": self._monitor_stat(proposer_stats, "valid_token_count"),
                "proposer_kl_coef": self._monitor_stat(proposer_stats, "kl_coef_after", getattr(self.proposer_updater, "kl_coef", None)),
                "forced_tail_tokens": forced_tail,
                "generator_baseline": self._monitor_float(self.generator_baseline),
                "proposer_baseline": self._monitor_float(self.proposer_baseline),
                "solver_baseline": self._monitor_float(self.solver_baseline),
                "step_duration_sec": self._monitor_float(step_duration_sec),
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
            "reward_mean": _mean("reward_mean"),
            "reward_max": _mean("reward_max"),
            "reward_min": _mean("reward_min"),
            "best_spec_score": _mean("best_spec_score"),
            "best_cycle_score": _mean("best_cycle_score"),
            "best_diversity_score": _mean("best_diversity_score"),
            "best_contradiction_score": _mean("best_contradiction_score"),
            "spec_quality": _mean("spec_quality"),
            "generator_kl_coef": float(self.generator_updater.kl_coef),
            "proposer_kl_coef": float(self.proposer_updater.kl_coef),
            "solver_kl_coef": (
                float(self.solver_updater.kl_coef) if self.solver_updater is not None else None
            ),
            "generator_baseline": float(self.generator_baseline),
            "proposer_baseline": float(self.proposer_baseline),
            "proposer_gen_baseline": float(self.proposer_gen_baseline),
            "solver_baseline": float(self.solver_baseline),
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
            "generation_rollouts_log_path": str(self.release_generation_rollouts_log_path),
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
        else:
            dit_index_path = resume_dir / "dit_trainable_index.json"
            dit_dir = resume_dir / "dit_trainable"
            if dit_index_path.exists() and dit_dir.is_dir():
                loaded = 0
                failed = 0
                missing = 0
                model_ref = _unwrap_model(self.model)
                try:
                    with dit_index_path.open("r", encoding="utf-8") as f:
                        payload = json.load(f)
                    items = payload.get("params", {}) if isinstance(payload, dict) else {}
                    if isinstance(items, dict):
                        for param_name, file_name in items.items():
                            shard_path = (dit_dir / str(file_name)).resolve()
                            if not shard_path.exists():
                                missing += 1
                                continue
                            try:
                                shard = torch.load(shard_path, map_location="cpu")
                                missing_keys, unexpected_keys = model_ref.load_state_dict(
                                    {str(param_name): shard},
                                    strict=False,
                                )
                                if unexpected_keys:
                                    failed += len(unexpected_keys)
                                elif missing_keys and str(param_name) in missing_keys:
                                    failed += 1
                                else:
                                    loaded += 1
                            except Exception:
                                failed += 1
                    if self.is_main_process:
                        print(
                            f"[Generation] Restored DiT shards from {dit_dir} "
                            f"(loaded={loaded}, missing={missing}, failed={failed})"
                        )
                except Exception as exc:
                    if self.is_main_process:
                        print(f"[Generation] WARNING: failed to restore DiT shard weights: {exc}")

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
        if self.dit_updater is not None and "dit_updater" in state:
            self.dit_updater.load_state_dict(state["dit_updater"])

        self.solver_baseline = float(state.get("solver_baseline", self.solver_baseline))
        self.proposer_baseline = float(state.get("proposer_baseline", self.proposer_baseline))
        self.proposer_gen_baseline = float(state.get("proposer_gen_baseline", self.proposer_gen_baseline))
        self.generator_baseline = float(state.get("generator_baseline", self.generator_baseline))
        self.proposer_entropy_mu_ema = float(
            state.get("proposer_entropy_mu_ema", self.proposer_entropy_mu_ema)
        )
        # Allow a poisoned/locked proposer baseline to be reset on resume
        # (e.g. after the baseline-clamp bug fix). When set, also clears the
        # entropy history so the IQR filter re-warms from scratch rather than
        # staying locked at IQR=0 from a stale run.
        if bool(getattr(self.cfg, "reset_proposer_baseline", False)):
            if self.is_main_process:
                print(
                    f"[Generation] reset_proposer_baseline=True: resetting proposer_baseline "
                    f"{self.proposer_baseline:.4f} → 0.0, proposer_gen_baseline "
                    f"{self.proposer_gen_baseline:.4f} → 0.0"
                )
            self.proposer_baseline = 0.0
            self.proposer_gen_baseline = 0.0
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
            "proposer_gen_baseline": float(self.proposer_gen_baseline),
            "generator_baseline": float(self.generator_baseline),
            "proposer_entropy_mu_ema": float(self.proposer_entropy_mu_ema),
            "unicorn_reconstruction_update_counts": dict(self._unicorn_reconstruction_update_counts),
            "py_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if self.solver_updater is not None:
            state["solver_updater"] = self.solver_updater.state_dict()
            state["solver_baseline"] = float(self.solver_baseline)
        if self.dit_updater is not None:
            save_dit_opt_state = os.environ.get("SE_SAVE_DIT_OPT_STATE", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if save_dit_opt_state:
                state["dit_updater"] = self.dit_updater.state_dict()
            else:
                state["dit_updater"] = {
                    "step_id": int(getattr(self.dit_updater, "step_id", 0)),
                    "accum_count": int(getattr(self.dit_updater, "_accum_count", 0)),
                    "has_real_grad_in_window": bool(
                        getattr(self.dit_updater, "_has_real_grad_in_window", False)
                    ),
                    "dit_lora_enabled": bool(getattr(self.dit_updater, "dit_lora_enabled", False)),
                    "trainable_param_count": int(getattr(self.dit_updater, "trainable_param_count", 0)),
                    "optimizer": None,
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
        dit_lora_requested = bool(getattr(self.cfg, "dit_update_enabled", False)) and bool(
            getattr(self.cfg, "dit_lora_enabled", True)
        )
        if (self.cfg.use_lora or dit_lora_requested) and (
            not HAS_PEFT or LoraConfig is None or get_peft_model is None or TaskType is None
        ):
            raise RuntimeError("PEFT is required for role-specific LoRA adapters and DiT LoRA")

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
            load_in_4bit=bool(getattr(self.cfg, "load_in_4bit", False)),
            bnb_4bit_quant_type=str(getattr(self.cfg, "bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(getattr(self.cfg, "bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=str(getattr(self.cfg, "bnb_4bit_compute_dtype", "bfloat16")),
        )

        # Load processor
        processor = AutoProcessor.from_pretrained(self.cfg.model_name, trust_remote_code=True)

        if self.is_main_process:
            print(f"[Generation] Loaded model: dtype={dtype}, device_map={device_map}, attn_implementation={attn_impl or 'default'}")

        if self.cfg.use_lora:
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
            solver_rank_pattern = {
                target: int(getattr(self.cfg, "solver_merger_lora_r", self.cfg.lora_r))
                for target in solver_merger_targets
            }
            solver_alpha_pattern = {
                target: int(getattr(self.cfg, "solver_merger_lora_alpha", self.cfg.lora_alpha))
                for target in solver_merger_targets
            }

            solver_cfg = _build_lora_config(
                r=self.cfg.lora_r,
                alpha=self.cfg.lora_alpha,
                dropout=self.cfg.lora_dropout,
                targets=solver_targets,
                task_type=TaskType.CAUSAL_LM,
                rank_pattern=solver_rank_pattern,
                alpha_pattern=solver_alpha_pattern,
            )
            text_cfg = _build_lora_config(
                r=self.cfg.lora_r,
                alpha=self.cfg.lora_alpha,
                dropout=self.cfg.lora_dropout,
                targets=text_targets,
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, solver_cfg)
            if hasattr(model, "add_adapter"):
                try:
                    model.add_adapter("proposer", text_cfg)
                except Exception:
                    pass
                try:
                    model.add_adapter("generator", text_cfg)
                except Exception:
                    pass

            for name, param in model.named_parameters():
                if (
                    "lora_" in name
                    and not _is_dit_param_name(name)
                    and (".default." in name or ".proposer." in name or ".generator." in name)
                ):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

            if self.is_main_process:
                print(
                    "[Generation] Role LoRA targets: "
                    f"text={list(text_targets)}; "
                    f"solver_merger_enabled={solver_merger_enabled}; "
                    f"solver_merger={list(solver_merger_targets)}"
                )
                try:
                    model.print_trainable_parameters()
                except Exception:
                    pass

        if bool(getattr(self.cfg, "dit_update_enabled", False)):
            try:
                core_model = _unwrap_model(model).get_model()
                dit_module = getattr(core_model, "dit", None)
                if dit_module is None:
                    if bool(getattr(self.cfg, "require_dit_update", False)):
                        raise RuntimeError(
                            "--dit_update_enabled was requested with --require_dit_update, "
                            "but the loaded model core does not expose `dit`."
                        )
                    if self.is_main_process:
                        print("[Generation] WARNING: --dit_update_enabled set but model has no `dit`; disabling DiT updates.")
                    self.cfg.dit_update_enabled = False
                elif bool(getattr(self.cfg, "dit_lora_enabled", True)):
                    dit_targets = _target_tuple(getattr(self.cfg, "dit_lora_target_modules", tuple()))
                    if not dit_targets:
                        raise ValueError("dit_lora_enabled=True but no dit_lora_target_modules were provided.")
                    dit_lora_cfg = _build_lora_config(
                        r=int(getattr(self.cfg, "dit_lora_r", 16)),
                        alpha=int(getattr(self.cfg, "dit_lora_alpha", 32)),
                        dropout=float(getattr(self.cfg, "dit_lora_dropout", 0.0)),
                        targets=dit_targets,
                    )
                    dit_gc_disabled = _disable_gradient_checkpointing_for_peft_wrap(dit_module)
                    core_model.dit = get_peft_model(dit_module, dit_lora_cfg)
                    for name, param in core_model.dit.named_parameters():
                        param.requires_grad_("lora_" in name)
                    if self.is_main_process:
                        counts = _count_lora_trainables(model)
                        print(
                            "[Generation] DiT LoRA enabled before DDP: "
                            f"targets={list(dit_targets)}, "
                            f"r={int(getattr(self.cfg, 'dit_lora_r', 16))}, "
                            f"alpha={int(getattr(self.cfg, 'dit_lora_alpha', 32))}, "
                            f"trainable_dit_lora_params={counts['dit']}"
                        )
                        if dit_gc_disabled:
                            print(
                                "[Generation] DiT gradient checkpointing disabled before PEFT wrapping "
                                "to avoid text-embedding gradient hooks on the diffusion transformer."
                            )
                        if counts["dit"] <= 0:
                            print("[Generation] WARNING: no trainable DiT LoRA parameters matched the configured targets.")
                else:
                    for param in dit_module.parameters():
                        param.requires_grad_(True)
                    if self.is_main_process:
                        print(
                            "[Generation] DiT module detected. "
                            "Legacy full-DiT update path is enabled (--disable_dit_lora)."
                        )
            except Exception as exc:
                if bool(getattr(self.cfg, "require_dit_update", False)):
                    raise RuntimeError(
                        "DiT updates are required for this run, but DiT training "
                        f"parameters could not be configured: {type(exc).__name__}: {exc}"
                    ) from exc
                if self.is_main_process:
                    print(f"[Generation] WARNING: failed to configure DiT training params: {exc}")
                self.cfg.dit_update_enabled = False

        if self.is_main_process and self.cfg.use_lora:
            counts = _count_lora_trainables(model)
            print(
                "[Generation] Trainable LoRA parameters by role: "
                f"solver={counts['solver']} "
                f"(solver_merger={counts['solver_merger']}), "
                f"proposer={counts['proposer']}, "
                f"generator={counts['generator']}, "
                f"dit={counts['dit']}"
            )
            if bool(getattr(self.cfg, "solver_merger_lora_enabled", False)) and counts["solver_merger"] <= 0:
                print(
                    "[Generation] WARNING: solver_merger_lora is enabled but no merger LoRA parameters "
                    "were created. Check --solver_merger_lora_targets against the model module names."
                )

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
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is not None and len(hidden_states) > 0:
                hidden = hidden_states[-1]
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                hidden = outputs[0]
            else:
                model_ref = _unwrap_model(self.model)
                embed_layer = None
                if hasattr(model_ref, "get_input_embeddings"):
                    try:
                        embed_layer = model_ref.get_input_embeddings()
                    except Exception:
                        embed_layer = None
                if embed_layer is None:
                    embed_layer = getattr(getattr(model_ref, "model", None), "embed_tokens", None)
                if embed_layer is None:
                    raise RuntimeError("Model forward did not return hidden states for text embedding.")
                hidden = embed_layer(inputs["input_ids"])
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
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is not None and len(hidden_states) > 0:
                hidden = hidden_states[-1]
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                hidden = outputs[0]
            else:
                return self._text_embedding(text)
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
        do_sample: bool = True,
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
                "do_sample": do_sample,
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

    def _generate_with_confidence(
        self,
        image: Image.Image,
        prompt: str,
        adapter_name: Optional[str],
        max_new_tokens: int,
        margin_tokens: int = 5,
    ) -> Tuple[str, Dict[str, float]]:
        """Greedy generation that returns Solver Token Entropy (STE) info.

        This method performs a **greedy** (do_sample=False) generation and
        computes the full softmax entropy of the logit distribution at each
        of the first *margin_tokens* generated tokens.  Unlike the logit
        margin (gap between top-1 and top-2), full token entropy captures
        genuine multi-way uncertainty across the entire vocabulary:

        - Forced-choice "A or B?" → mass on 2 tokens → entropy ≈ ln(2) ≈ 0.69
        - Genuinely hard question → mass spread across many answers → entropy >> 1.0

        This makes STE **naturally resistant to forced-choice gaming** because
        binary uncertainty always produces less entropy than multi-way uncertainty.

        Returns
        -------
        (text, confidence_info)
            text : str — decoded answer (same as ``_generate``)
            confidence_info : dict with keys
                ``min_margin``     — minimum logit margin (top1-top2) across K tokens
                ``mean_margin``    — mean logit margin across K tokens
                ``margins``        — list of per-token logit margins
                ``min_entropy``    — minimum token-level entropy across K tokens
                ``mean_entropy``   — mean token-level entropy across K tokens
                ``max_entropy``    — maximum token-level entropy across K tokens
                ``token_entropies``— list of per-token entropies (nats)
        """
        _NO_CONFIDENCE = {
            "min_margin": 999.0, "mean_margin": 999.0, "margins": [],
            "min_entropy": 0.0, "mean_entropy": 0.0, "max_entropy": 0.0,
            "token_entropies": [],
        }

        chat_text = _build_chat_text(self.processor, image, prompt)
        inputs = _prepare_mm_inputs(
            self.processor, self.device, image, chat_text, model=self.model,
        )

        has_image_feats = ("pixel_values" in inputs) or ("images" in inputs)
        if has_image_feats and "input_ids" in inputs:
            image_token_ids = _collect_image_token_ids(self.model)
            token_count = _count_image_tokens_in_inputs(
                inputs["input_ids"], image_token_ids,
            )
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
                            messages, tokenize=False, add_generation_prompt=True,
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

        # Fail fast on corrupted token IDs.
        model_cfg = getattr(_unwrap_model(self.model), "config", None)
        vocab_size = getattr(model_cfg, "vocab_size", None)
        input_ids = gen_inputs.get("input_ids")
        if (
            torch.is_tensor(input_ids)
            and isinstance(vocab_size, int)
            and vocab_size > 0
        ):
            min_id = int(input_ids.min().item())
            max_id = int(input_ids.max().item())
            if min_id < 0 or max_id >= vocab_size:
                raise RuntimeError(
                    "Invalid token ids prepared for confidence generation: "
                    f"min_id={min_id}, max_id={max_id}, vocab_size={vocab_size}."
                )

        _tok = _extract_tokenizer_from_processor(self.processor)
        _pad_id = getattr(_tok, "eos_token_id", None) if _tok is not None else None

        def _run_generate_conf(curr_inputs: Dict[str, torch.Tensor]):
            base_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,          # always greedy for confidence
                "temperature": 1.0,          # unused when do_sample=False
                "top_p": 1.0,                # unused when do_sample=False
                "pad_token_id": _pad_id,
                "remove_invalid_values": True,
                "renormalize_logits": False,  # keep raw logits for margin
                "output_scores": True,
                "return_dict_in_generate": True,
            }
            try:
                return self.model.generate(**curr_inputs, **base_kwargs)
            except TypeError as exc:
                msg = str(exc)
                # Some model versions may not support all kwargs.
                for kw in (
                    "remove_invalid_values", "renormalize_logits",
                    "output_scores", "return_dict_in_generate",
                ):
                    if kw in msg:
                        base_kwargs.pop("remove_invalid_values", None)
                        base_kwargs.pop("renormalize_logits", None)
                        # If output_scores/return_dict fails, fall back to
                        # plain generation (no margin info).
                        if "output_scores" in msg or "return_dict" in msg:
                            base_kwargs.pop("output_scores", None)
                            base_kwargs.pop("return_dict_in_generate", None)
                        return self.model.generate(**curr_inputs, **base_kwargs)
                raise

        with torch.no_grad():
            with use_adapter(self.model, adapter_name):
                try:
                    outputs = _run_generate_conf(gen_inputs)
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
                    outputs = _run_generate_conf(retry_inputs)
                    gen_inputs = retry_inputs

        # --- Extract text ---
        # outputs may be GenerateOutput (dict-like) or plain tensor.
        if hasattr(outputs, "sequences"):
            sequences = outputs.sequences
        else:
            sequences = outputs

        input_len = (
            gen_inputs["input_ids"].shape[1] if "input_ids" in gen_inputs else 0
        )
        completion_ids = sequences[0, input_len:]
        text = _decode_tokens(self.processor, completion_ids)

        # --- Extract logit margins AND full token entropy ---
        scores = getattr(outputs, "scores", None)
        if scores is None or len(scores) == 0:
            return text.strip(), _NO_CONFIDENCE

        K = min(max(1, margin_tokens), len(scores))
        margins = []
        token_entropies = []
        for i in range(K):
            logits_i = scores[i][0]  # [vocab_size] for batch element 0
            if logits_i.numel() < 2:
                continue
            # Logit margin (kept for backward compatibility / logging)
            top2 = torch.topk(logits_i, k=2)
            margin_val = float((top2.values[0] - top2.values[1]).item())
            margins.append(margin_val)
            # Full token entropy: H = -sum(p * log(p)) over entire vocab
            # This is the KEY metric — captures genuine multi-way uncertainty.
            # Forced-choice gives H ≈ ln(2) ≈ 0.69; genuinely hard gives H >> 1.0
            probs_i = torch.softmax(logits_i.float(), dim=0)
            # Clamp to avoid log(0)
            log_probs_i = torch.log(probs_i.clamp(min=1e-12))
            entropy_i = float(-(probs_i * log_probs_i).sum().item())
            token_entropies.append(entropy_i)

        # Free score tensors immediately to save GPU memory.
        del scores
        if hasattr(outputs, "scores"):
            outputs.scores = None

        if not margins:
            return text.strip(), _NO_CONFIDENCE

        confidence_info = {
            "min_margin": min(margins),
            "mean_margin": sum(margins) / len(margins),
            "margins": margins,
            "min_entropy": min(token_entropies) if token_entropies else 0.0,
            "mean_entropy": (
                sum(token_entropies) / len(token_entropies)
                if token_entropies else 0.0
            ),
            "max_entropy": max(token_entropies) if token_entropies else 0.0,
            "token_entropies": token_entropies,
        }
        return text.strip(), confidence_info

    def _generate_text_only(
        self,
        prompt: str,
        adapter_name: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool = True,
    ) -> str:
        """Generate text from a text-only prompt (no image input).

        Used by the imageless proposer mode (E5) where the proposer receives
        a topic description and generates a generation spec without visual input.
        """
        chat_text = _build_text_only_chat(self.processor, prompt)
        inputs = _prepare_text_only_inputs(self.processor, self.device, chat_text)

        gen_inputs = dict(inputs)

        # Fail fast on corrupted / incompatible token IDs
        model_cfg = getattr(_unwrap_model(self.model), "config", None)
        vocab_size = getattr(model_cfg, "vocab_size", None)
        input_ids = gen_inputs.get("input_ids")
        if torch.is_tensor(input_ids) and isinstance(vocab_size, int) and vocab_size > 0:
            min_id = int(input_ids.min().item())
            max_id = int(input_ids.max().item())
            if min_id < 0 or max_id >= vocab_size:
                raise RuntimeError(
                    "Invalid token ids prepared for text-only generation: "
                    f"min_id={min_id}, max_id={max_id}, vocab_size={vocab_size}."
                )

        # Extract pad_token_id
        _tok = _extract_tokenizer_from_processor(self.processor)
        _pad_id = getattr(_tok, "eos_token_id", None) if _tok is not None else None

        def _run_generate(curr_inputs: Dict[str, torch.Tensor]):
            base_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "pad_token_id": _pad_id,
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
                outputs = _run_generate(gen_inputs)

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

    def _propose_imageless_generation_spec(
        self,
        *,
        proposer_prompt: str,
    ) -> GenerationSpec:
        """Generate a generation spec from text-only input (no image).

        Used by imageless proposer mode (E5): the proposer receives a topic/theme
        as text and generates a prompt + QA pairs purely from text reasoning.
        """
        raw = self._generate_text_only(
            prompt=proposer_prompt,
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

    def _compose_generation_prompt(
        self,
        raw_prompt: str,
        source_caption: str,
    ) -> str:
        """Build the final prompt string passed to the DiT image generator.

        Design principle: the proposer LLM is explicitly instructed (in its
        system prompt) to write a <prompt> that already embeds all verifiable
        details from its own QA pairs.  For example, if QA asks
        "How many players?" → "Two", the proposer must write "two players" in
        the <prompt> itself.  We therefore use spec.prompt as-is and only
        perform light cleanup:

          1. Strip any instruction-style prefixes the LLM might accidentally
             prepend (e.g. "Generate an image of ..." or "Create a scene...").
          2. Strip question marks (DiT tokenizers handle them poorly).
          3. Fall back to source_caption when the prompt is missing or looks
             like a question rather than a description.
          4. Hard-cap at 96 words to stay within typical CLIP token limits.

        We deliberately do NOT reconstruct the prompt from QA pairs — the
        proposer owns that responsibility and is trained on it.
        """
        raw = " ".join((raw_prompt or "").split())
        raw = self._strip_instruction_prefix(raw).strip()

        caption = " ".join((source_caption or "").split())

        # Choose the best prompt base
        if raw and not self._is_question_like_prompt(raw):
            prompt = raw
        elif caption:
            # Proposer prompt was missing or unusable; fall back to caption
            prompt = caption
        elif raw:
            # Last resort: strip question marks and use raw text
            prompt = raw.replace("?", "")
        else:
            prompt = "a coherent scene with clear objects and relationships"

        # Final sanitisation
        prompt = prompt.replace("?", "").strip()
        prompt = " ".join(prompt.split())

        # Hard word cap (CLIP text encoder limit is ~77 tokens ≈ 60-96 words)
        max_words = 96
        words = prompt.split()
        if len(words) > max_words:
            prompt = " ".join(words[:max_words])

        return prompt

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
        target_difficulty: str = "medium",
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

        # Use the curriculum-sampled difficulty passed in from the trainer loop.
        # Fall back to the static config value only if the caller didn't provide one.
        target_diff = str(target_difficulty or getattr(self.cfg, "unicorn_target_difficulty", "medium") or "medium")
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
        image: Optional[Image.Image],
        spec: GenerationSpec,
        best: Dict[str, object],
        spec_quality: float,
        target_difficulty: str = "medium",
    ) -> int:
        if not bool(getattr(self.cfg, "unicorn_reconstruction_sft_enabled", True)):
            return 0
        # In imageless mode, skip reconstruction (requires source image for SFT)
        if image is None:
            return 0
        if spec_quality < float(getattr(self.cfg, "unicorn_reconstruction_min_quality", 0.55)):
            return 0

        enqueued = 0
        target_diff = str(target_difficulty or getattr(self.cfg, "unicorn_target_difficulty", "medium") or "medium")
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

            role_can_update = True
            role_skip_reason: Optional[str] = None
            generator_sft_fn = None
            if role == "proposer":
                if not isinstance(update_image, Image.Image):
                    role_can_update = False
                    role_skip_reason = "proposer_task_missing_image"
            elif role == "generator":
                generator_sft_fn = getattr(self.generator_updater, "sft_step", None)
                if not callable(generator_sft_fn):
                    role_can_update = False
                    role_skip_reason = "generator_updater_missing_sft_step"
            else:
                role_can_update = False
                role_skip_reason = f"unsupported_unicorn_role:{role}"

            role_can_update, role_skip_reason = self._distributed_update_ready(
                role_can_update,
                role_skip_reason,
                peer_reason="distributed_peer_unicorn_role_skip",
            )
            if not role_can_update:
                info["skipped_updates"] += 1
                self._unicorn_reconstruction_update_counts["skipped"] += 1
                info["update_records"].append(
                    {"role": role, "task": task.get("task"), "skipped": True, "reason": role_skip_reason}
                )
                continue

            stats: Optional[Dict[str, float]] = None
            if role == "proposer":
                # Unicorn reconstruction is always an SFT-style update (reward=+1).
                # GRPO's group normalization is degenerate for a single sample (advantage=0),
                # so we use sft_step regardless of proposer_update_rule.
                _proposer_sft_fn = getattr(self.proposer_updater, "sft_step", None)
                if callable(_proposer_sft_fn):
                    stats = _proposer_sft_fn(
                        prompt=prompt,
                        completion=completion,
                        device=self.device,
                        image=update_image if isinstance(update_image, Image.Image) else None,
                    )
                elif hasattr(self.proposer_updater, "step") and not self._proposer_uses_grpo:
                    stats = self.proposer_updater.step(
                        image=update_image,
                        prompt=prompt,
                        completion=completion,
                        reward=1.0,
                        baseline=0.0,
                        device=self.device,
                    )
            elif role == "generator":
                stats = generator_sft_fn(
                    prompt=prompt,
                    completion=completion,
                    device=self.device,
                    image=update_image if isinstance(update_image, Image.Image) else None,
                    completion_token_ids=completion_token_ids,
                )
            else:
                role_skip_reason = f"unsupported_unicorn_role:{role}"

            if stats is None:
                info["skipped_updates"] += 1
                self._unicorn_reconstruction_update_counts["skipped"] += 1
                info["update_records"].append(
                    {"role": role, "task": task.get("task"), "skipped": True, "reason": role_skip_reason}
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

    def verify_with_spec(
        self,
        image: Image.Image,
        spec: GenerationSpec,
        n_samples: int = 5,
    ) -> Tuple[float, List[Dict[str, object]]]:
        """
        Verify an image against a generation spec (Unicorn-style).
        Public interface used by FrozenJudge.
        """
        if not spec.qa_pairs:
            return 0.5, []

        per_question_scores = []
        score_values = []

        # Temporarily override num_solver_samples if requested
        original_n = self.cfg.num_solver_samples_spec
        self.cfg.num_solver_samples_spec = n_samples

        try:
            for qa in spec.qa_pairs:
                solved = self._solve_question_with_rollouts(image=image, question=qa.question)
                mode_answer = str(solved["majority_answer"])
                maj_frac = float(solved["majority_fraction"])

                match_score = _soft_match(mode_answer, qa.expected)
                combined = 0.7 * match_score + 0.3 * maj_frac

                epol = _yes_no_polarity(qa.expected)
                apol = _yes_no_polarity(mode_answer)
                contradiction = 1.0 if (epol != 0 and apol != 0 and epol != apol) else 0.0

                score_values.append(combined)
                
                per_question_scores.append({
                    "question": qa.question,
                    "expected": qa.expected,
                    "majority_answer": mode_answer,
                    "majority_fraction": maj_frac,
                    "match_score": match_score,
                    "combined_score": combined,
                    "contradiction": contradiction,
                    "solver": solved,
                })
        finally:
            self.cfg.num_solver_samples_spec = original_n

        # Aggregate per-question scores (mean)
        final_score = sum(score_values) / max(1, len(score_values))
        return final_score, per_question_scores

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
            # Delegate core verification logic to new public method
            score_vals, per_q_vals = self.verify_with_spec(
                image=image,
                spec=GenerationSpec(
                    prompt="",
                    qa_pairs=(qa,),
                    raw_output="",
                    fallback_used=False,
                ),
                n_samples=self.cfg.num_solver_samples_spec
            )
            
            # Unpack single-QA result
            combined = score_vals
            solved = per_q_vals[0]["solver"]
            match_score = per_q_vals[0]["match_score"]
            contradiction = per_q_vals[0]["contradiction"]
            maj_frac = solved["majority_fraction"]
            mode_answer = str(solved["majority_answer"])

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
        candidate_specs: Optional[List[GenerationSpec]] = None,
        candidate_spec_qualities: Optional[List[float]] = None,
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
            # Support per-candidate specs (Exp 2: Diversity in Prompts)
            current_prompt = prompt
            current_quality = spec_quality
            if candidate_specs is not None and idx < len(candidate_specs):
                current_prompt = candidate_specs[idx].prompt
                if candidate_spec_qualities is not None and idx < len(candidate_spec_qualities):
                    current_quality = candidate_spec_qualities[idx]
                # Re-score spec match against the specific QA pairs for this diverse prompt
                if self.judge is not None:
                    # Use Frozen Judge for scoring (Exp 3)
                    spec_score, qa_results = self.judge.evaluate(
                        image=image, spec=candidate_specs[idx], n_samples=self.cfg.num_solver_samples_spec
                    )
                    contradiction_score = sum(r["contradiction"] for r in qa_results) / max(1, len(qa_results))
                    qa_logs = qa_results
                else:
                    # Use active solver (baseline)
                    spec_score, contradiction_score, qa_logs = self._score_spec(
                        image=image,
                        qa_pairs=candidate_specs[idx].qa_pairs,
                        step=step,
                        candidate_idx=idx,
                        candidate_count=len(candidates),
                        verbose=verbose,
                    )
            else:
                # Shared spec (Exp 2 off or base behavior)
                current_prompt = prompt
                current_quality = spec_quality
                
                if self.judge is not None:
                     # Use Frozen Judge for scoring (Exp 3)
                     # We create a temporary spec object since the signature needs one
                     temp_spec = GenerationSpec(
                        prompt=prompt,
                        qa_pairs=qa_pairs,
                        raw_output="",
                        fallback_used=False,
                     )
                     spec_score, qa_results = self.judge.evaluate(
                        image=image, spec=temp_spec, n_samples=self.cfg.num_solver_samples_spec
                     )
                     contradiction_score = sum(r["contradiction"] for r in qa_results) / max(1, len(qa_results))
                     qa_logs = qa_results
                else:
                     # Use active solver
                     spec_score, contradiction_score, qa_logs = self._score_spec(
                        image=image,
                        qa_pairs=qa_pairs,
                        step=step,
                        candidate_idx=idx,
                        candidate_count=len(candidates),
                        verbose=verbose,
                    )
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

            qa_confidence = self._qa_confidence_from_logs(qa_logs)
            cycle_score, cycle_caption = self._cycle_reward(prompt=current_prompt, image=image)

            base_reward = (
                w_spec * spec_score
                + w_cycle * cycle_score
                + w_div * diversity_scores[idx]
                - self.cfg.reward_contradiction_weight * contradiction_score
            )
            base_reward = max(0.0, min(1.0, base_reward))
            total_reward = current_quality * base_reward
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
                    "spec_quality": current_quality,
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

    def _score_candidates_self_clip(
        self,
        prompt: str,
        candidates: List[Dict[str, object]],
        *,
        step: Optional[int] = None,
        verbose: bool = False,
    ) -> List[Dict[str, object]]:
        """Score generated candidates with internal CLIP-style image-text similarity.

        The similarity is computed from this model's own frozen embedding path
        (no external reward model checkpoint).
        """
        scored: List[Dict[str, object]] = []
        for idx, cand in enumerate(candidates):
            cand_image = cand.get("image")
            cand_prompt = str(cand.get("policy_prompt", prompt))
            reward = 0.0
            raw_cosine = -1.0
            if isinstance(cand_image, Image.Image):
                try:
                    raw_cosine = float(self._image_text_similarity(cand_image, cand_prompt))
                    reward = max(0.0, min(1.0, 0.5 * (raw_cosine + 1.0)))
                except Exception:
                    reward = 0.0
                    raw_cosine = -1.0

            scored.append(
                {
                    "candidate_idx": idx,
                    "backend": cand.get("backend"),
                    "policy_prompt": cand_prompt,
                    "policy_completion": cand.get("policy_completion", ""),
                    "policy_completion_ids": cand.get("policy_completion_ids"),
                    "spec_score": 0.0,
                    "contradiction_score": 0.0,
                    "cycle_score": reward,  # compatibility key for existing logs/metrics
                    "cycle_caption": "",
                    "diversity_score": 0.0,
                    "base_reward": reward,
                    "spec_quality": 1.0,
                    "qa_confidence": 0.0,
                    "qa_logs": [],
                    "self_clip_score": reward,
                    "self_clip_raw_cosine": raw_cosine,
                    "total_reward": reward,
                    "image": cand_image,
                }
            )

            if verbose and self.is_main_process and step is not None:
                print(
                    f"[Step {step:05d}][G-clip] candidate {idx + 1}/{len(candidates)} "
                    f"score={reward:.4f} raw_cos={raw_cosine:.4f}"
                )

        return scored

    # ---- Phase 2: Solver-derived reference-answer log-prob scoring ---- #

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
        conditioned on the generated image and the question. The reference
        answer is Solver-derived from the real image, not a dataset label or
        human annotation. This is used as a continuous reward signal for
        ranking candidate images.

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
        full_text = solver_prompt_chat + ref_ans_stripped

        def _filter_forward_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:
            return {
                k: v
                for k, v in payload.items()
                if k not in ("images", "image_sizes")
            }

        # IMPORTANT: use the local (unwrapped) model for no-grad scoring.
        # Using DDP-wrapped forward here can deadlock in ref-scoring paths
        # where per-rank forward counts/timings diverge.
        model = self.model
        adapter = "default" if self.cfg.use_lora else None
        was_training = model.training

        try:
            with use_adapter(model, adapter):
                model.eval()
                prompt_inputs = _prepare_mm_inputs(
                    self.processor,
                    device,
                    image,
                    solver_prompt_chat,
                    model=_unwrap_model(model),
                )
                full_inputs = _prepare_mm_inputs(
                    self.processor,
                    device,
                    image,
                    full_text,
                    model=_unwrap_model(model),
                )

                prompt_len = _aligned_prompt_prefix_len(
                    prompt_inputs["input_ids"],
                    full_inputs["input_ids"],
                    ref_ans_stripped,
                )
                target_ids = full_inputs["input_ids"][:, prompt_len:]
                target_count = int(target_ids.numel())
                if target_count <= 0:
                    return -10.0

                prompt_forward = _filter_forward_kwargs(prompt_inputs)
                try:
                    out = model(
                        **prompt_forward,
                        use_cache=True,
                        return_dict=True,
                    )
                    logits = getattr(out, "logits", None)
                    if logits is None:
                        raise RuntimeError("Solver forward returned no logits.")
                    past_key_values = getattr(out, "past_key_values", None)

                    running_attention = prompt_forward.get("attention_mask")
                    logp_values: List[torch.Tensor] = []
                    one_mask: Optional[torch.Tensor] = None

                    for tok_idx in range(target_ids.shape[1]):
                        tok = target_ids[:, tok_idx : tok_idx + 1]
                        token_logp = F.log_softmax(logits[:, -1, :], dim=-1).gather(-1, tok).squeeze(-1)
                        logp_values.append(token_logp)

                        if tok_idx >= (target_ids.shape[1] - 1):
                            break

                        if past_key_values is None:
                            raise RuntimeError("Solver forward did not return past_key_values.")

                        if running_attention is not None:
                            if one_mask is None:
                                one_mask = torch.ones(
                                    (running_attention.shape[0], 1),
                                    dtype=running_attention.dtype,
                                    device=running_attention.device,
                                )
                            running_attention = torch.cat([running_attention, one_mask], dim=1)

                        out = model(
                            input_ids=tok,
                            attention_mask=running_attention,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
                        logits = getattr(out, "logits", None)
                        if logits is None:
                            raise RuntimeError("Solver forward returned no logits.")
                        past_key_values = getattr(out, "past_key_values", None)

                    if logp_values:
                        return float(torch.stack(logp_values).mean().item())
                except Exception:
                    # Fallback path for model variants that do not support
                    # stable cache-based scoring.
                    out = model(**_filter_forward_kwargs(full_inputs), use_cache=False, return_dict=True)
                    logits = getattr(out, "logits", None)
                    if logits is None:
                        return -10.0
                    labels = full_inputs["input_ids"].clone()
                    labels[:, :prompt_len] = -100
                    shift_labels = labels[:, 1:]
                    valid_mask = shift_labels != -100
                    valid_count = int(valid_mask.sum().item())
                    if valid_count <= 0:
                        return -10.0
                    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
                    gathered = logp.gather(
                        -1, shift_labels.clamp_min(0).unsqueeze(-1)
                    ).squeeze(-1)
                    return float(gathered[valid_mask].mean().item())

                return -10.0
        finally:
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
        """Phase 2 scoring: Solver-derived reference-answer log-prob on generated images.

        1. Extract questions from ``spec.qa_pairs``.
        2. Solver answers each question looking at ``real_image`` to produce
           Solver-derived reference answers.
        3. For each candidate, compute mean log P(ref_answer | candidate, question).
        4. That mean log-prob is the ``total_reward``.

        Returns
        -------
        scored : list of dicts (same schema as ``_score_candidates``)
        questions : list of question strings
        reference_answers : list of Solver-derived answers on the real image
        """
        questions = [qa.question for qa in spec.qa_pairs if qa.question.strip()]
        if not questions:
            # Fallback: empty scoring — all zeros
            scored = [
                {
                    "candidate_idx": idx,
                    "backend": cand.get("backend"),
                    "policy_prompt": cand.get("policy_prompt", spec.prompt),
                    "policy_completion": cand.get("policy_completion", ""),
                    "policy_completion_ids": cand.get("policy_completion_ids"),
                    "spec_score": 0.0,
                    "contradiction_score": 0.0,
                    "cycle_score": 0.0,
                    "cycle_caption": "",
                    "diversity_score": 0.0,
                    "base_reward": 0.0,
                    "spec_quality": 1.0,
                    "qa_confidence": 0.0,
                    "qa_logs": [],
                    "total_reward": 0.0,
                    "ref_answer_logps": [],
                    "image": cand.get("image"),
                }
                for idx, cand in enumerate(candidates)
            ]
            return scored, [], []

        # Step 1: Solver generates Solver-derived reference answers on the REAL image.
        # Uses the trained solver LoRA — as solver improves through
        # understanding training, it provides better Solver-derived reference answers,
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
                        "backend": cand.get("backend"),
                        "policy_prompt": cand.get("policy_prompt", spec.prompt),
                        "policy_completion": cand.get("policy_completion", ""),
                        "policy_completion_ids": cand.get("policy_completion_ids"),
                        "spec_score": 0.0,
                        "contradiction_score": 0.0,
                        "cycle_score": 0.0,
                        "cycle_caption": "",
                        "diversity_score": 0.0,
                        "base_reward": -10.0,
                        "spec_quality": 1.0,
                        "qa_confidence": 0.0,
                        "qa_logs": [],
                        "total_reward": -10.0,
                        "ref_answer_logps": [],
                        "image": cand_image,
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
                    "cycle_caption": "",
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
        if not self.is_main_process:
            return
        if self.cfg.save_generated_images_every <= 0:
            return
        if (step % self.cfg.save_generated_images_every) != 0:
            return
        step_dir = self.generated_dir / f"step_{step:05d}"
        # On resume/restart, do not overwrite already-saved image artifacts for
        # completed steps.
        if step_dir.exists() and any(step_dir.glob("*.png")):
            return
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

    def _archive_existing_step_checkpoint(self, step_dir: pathlib.Path) -> Optional[pathlib.Path]:
        if not step_dir.exists():
            return None
        backup_root = self.run_dir / "checkpoint_backups"
        backup_root.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_dir = backup_root / f"{step_dir.name}_{ts}"
        suffix = 1
        while backup_dir.exists():
            backup_dir = backup_root / f"{step_dir.name}_{ts}_{suffix:02d}"
            suffix += 1
        shutil.move(str(step_dir), str(backup_dir))
        return backup_dir

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
                try:
                    sanitize_peft_adapter_dir(
                        subdir,
                        in_place=True,
                        backup=False,
                        log=lambda msg: print(f"[Generation] {msg}"),
                    )
                except Exception as exc:
                    print(f"[Generation] WARNING: failed to sanitize {sub_name} adapter checkpoint: {exc}")
                if sub_name == "solver":
                    try:
                        self.processor.save_pretrained(subdir)
                    except Exception:
                        pass
            if bool(getattr(self.cfg, "dit_update_enabled", False)) and bool(getattr(self.cfg, "dit_lora_enabled", True)):
                try:
                    core_model = _unwrap_model(self.model).get_model()
                    dit_module = getattr(core_model, "dit", None)
                    if dit_module is not None and hasattr(dit_module, "save_pretrained"):
                        dit_dir = tmp_dir / "dit_lora"
                        dit_module.save_pretrained(dit_dir)
                        _json_dump(
                            tmp_dir / "dit_lora_metadata.json",
                            {
                                "dit_lora_enabled": True,
                                "dit_lora_r": int(getattr(self.cfg, "dit_lora_r", 16)),
                                "dit_lora_alpha": int(getattr(self.cfg, "dit_lora_alpha", 32)),
                                "dit_lora_dropout": float(getattr(self.cfg, "dit_lora_dropout", 0.0)),
                                "dit_lora_target_modules": list(
                                    _target_tuple(getattr(self.cfg, "dit_lora_target_modules", tuple()))
                                ),
                            },
                        )
                except Exception as exc:
                    print(f"[Generation] WARNING: failed to save DiT LoRA adapter: {exc}")
        else:
            self.model.save_pretrained(tmp_dir / "model")
            try:
                self.processor.save_pretrained(tmp_dir / "model")
            except Exception:
                pass

        model_ref = _unwrap_model(self.model)
        save_trainable_snapshot = not (
            self.cfg.use_lora
            and bool(getattr(self.cfg, "dit_update_enabled", False))
            and not bool(getattr(self.cfg, "dit_lora_enabled", True))
        )
        if save_trainable_snapshot:
            trainable_state = {
                name: param.detach().cpu()
                for name, param in model_ref.named_parameters()
                if param.requires_grad
            }
            torch.save(trainable_state, tmp_dir / "trainable_adapters.pt")
        elif bool(getattr(self.cfg, "dit_update_enabled", False)):
            dit_dir = tmp_dir / "dit_trainable"
            dit_dir.mkdir(parents=True, exist_ok=True)
            dit_index: Dict[str, str] = {}
            for idx, (name, param) in enumerate(model_ref.named_parameters()):
                if not param.requires_grad:
                    continue
                if ".dit." not in name and not name.startswith("dit."):
                    continue
                shard_name = f"dit_param_{idx:06d}.pt"
                shard_path = dit_dir / shard_name
                torch.save(param.detach().cpu(), shard_path)
                dit_index[name] = shard_name
            _json_dump(
                tmp_dir / "dit_trainable_index.json",
                {
                    "count": int(len(dit_index)),
                    "params": dit_index,
                },
            )

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
                "dit_updater_step": self.dit_updater.step_id if self.dit_updater is not None else None,
                "dit_lora_enabled": bool(getattr(self.cfg, "dit_lora_enabled", True)),
            },
        )
        with (tmp_dir / "SAVE_OK").open("w", encoding="utf-8") as f:
            f.write("ok\n")

        if step_dir.exists():
            archived = self._archive_existing_step_checkpoint(step_dir)
            if archived is not None and self.is_main_process:
                print(
                    f"[Generation] Preserved existing checkpoint {step_dir.name} "
                    f"at {archived}"
                )
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
            "rollouts_log_path": str(self.release_rollouts_log_path),
            "generation_rollouts_log_path": str(self.release_generation_rollouts_log_path),
            "metrics_log_path": str(self.metrics_log_path),
            "last_checkpoint_dir": str(self.last_checkpoint_dir),
        }
        _json_dump(self.summary_path, payload)
        _json_dump(self.release_summary_path, payload)
        return payload

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
        dit_stats: Optional[Dict[str, float]] = None,
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
        if dit_stats is not None:
            metrics["train/diffusion_generator_update"] = 1.0 if bool(dit_stats.get("did_step", False)) else 0.0
            if dit_stats.get("objective"):
                metrics["train/generator_effective_objective"] = str(dit_stats.get("objective"))
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
        if dit_stats:
            metrics["train/dit_update_skipped"] = 1.0 if dit_stats.get("skipped_reason") else 0.0
            metrics["train/dit_update_applied"] = 1.0 if bool(dit_stats.get("did_step", False)) else 0.0
            metrics["dit/loss"] = float(dit_stats.get("loss", 0.0))
            metrics["dit/valid_latent_tokens"] = float(dit_stats.get("valid_latent_tokens", 0.0))
            metrics["dit/lora_enabled"] = 1.0 if bool(dit_stats.get("lora_enabled", False)) else 0.0
            metrics["dit/trainable_params"] = float(dit_stats.get("trainable_params", 0.0))
            if dit_stats.get("skipped_reason"):
                metrics["train/dit_skip_reason"] = str(dit_stats.get("skipped_reason"))

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

    def _generation_step(
        self,
        step: int,
        image: Optional[Image.Image],
        meta: Dict,
        target_difficulty: str = "medium",
    ) -> Dict[str, object]:
        verbose = self.is_main_process and (step % self.cfg.log_every == 0)
        step_t0 = time.perf_counter()
        _imageless = bool(getattr(self.cfg, "imageless_proposer_mode", False))
        if verbose:
            print(
                f"[Step {step:05d}][G] generation phase start "
                f"(target_difficulty={target_difficulty}, imageless={_imageless})"
            )

        if _imageless:
            # ── Imageless proposer path (E5): topic → text-only proposer → spec ──
            topic = _sample_imageless_topic(step=step, seed=self.cfg.seed)
            source_caption = topic  # use topic as pseudo-caption for logging
            if verbose:
                print(f"[Step {step:05d}][G] imageless mode: topic='{topic[:80]}...'")

            proposer_prompt = build_imageless_spec_prompt(
                topic=topic,
                target_difficulty=target_difficulty,
            )
            raw_spec = self._propose_imageless_generation_spec(
                proposer_prompt=proposer_prompt,
            )
            # Sanitize & score spec (quality only — no alignment since no source image)
            spec, spec_quality, spec_quality_details = self._sanitize_and_score_spec(
                raw_spec,
                source_caption=source_caption,
            )
            unicorn_spec_meta = {
                "enabled": False,
                "rejection_enabled": False,
                "imageless_mode": True,
                "topic": topic,
                "attempts": 1,
                "retries_used": 0,
                "selected_accepted": True,
                "selected_reject_reason": "",
                "selected_alignment": 0.0,
                "selected_contradiction": 0.0,
                "selected_quality": float(spec_quality),
                "attempt_logs": [],
            }
        else:
            # ── Standard image-based proposer path ──
            source_caption = self._caption_image(image)
            spec, spec_quality, spec_quality_details, unicorn_spec_meta = self._select_generation_spec_with_unicorn(
                image=image,
                source_caption=source_caption,
                step=step,
                verbose=verbose,
                target_difficulty=target_difficulty,
            )

        if self.judge is not None:
            self.judge.update(self)

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

        use_diverse_prompts = bool(getattr(self.cfg, "use_diverse_prompts", False))
        # Diverse prompts require image-based spec generation — disable in imageless mode
        if _imageless:
            use_diverse_prompts = False
        candidates: List[Dict[str, object]] = []
        candidate_specs = None
        candidate_spec_qualities = None

        if use_diverse_prompts:
            if verbose:
                print(f"[Step {step:05d}][G] diverse-prompts mode enabled")
            candidate_specs = [spec]
            candidate_spec_qualities = [spec_quality]
            for cand_idx in range(self.cfg.num_generations):
                if cand_idx == 0:
                    curr_spec = spec
                    curr_quality = spec_quality
                else:
                    curr_spec, curr_quality, _, _ = self._select_generation_spec_with_unicorn(
                        image=image,
                        source_caption=source_caption,
                        step=step,
                        verbose=False,
                    )
                    candidate_specs.append(curr_spec)
                    candidate_spec_qualities.append(curr_quality)
                cand_t0 = time.perf_counter()
                if verbose:
                    print(
                        f"[Step {step:05d}][G] generating candidate {cand_idx + 1}/{self.cfg.num_generations} "
                        f"(spec_q={curr_quality:.3f})"
                    )
                cand = self._generate_image_candidate(inputs=curr_spec.prompt)
                candidates.append(cand)
                if verbose:
                    backend = str(cand.get("backend", "unknown"))
                    cand_dt = time.perf_counter() - cand_t0
                    print(
                        f"[Step {step:05d}][G] generated candidate {cand_idx + 1}/{self.cfg.num_generations} "
                        f"in {cand_dt:.1f}s (backend={backend})"
                    )
        else:
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
        _use_ref_scoring = bool(getattr(self.cfg, "use_ref_answer_scoring", False))
        _use_self_clip_scoring = bool(getattr(self.cfg, "use_self_clip_reward_scoring", False))
        # In imageless mode, Solver-derived reference-answer scoring requires a real image.
        # Fall back to spec-based scoring which uses the generated images + QA pairs.
        if _imageless and _use_ref_scoring:
            _use_ref_scoring = False
            if verbose:
                print(f"[Step {step:05d}][G] imageless mode: disabling Solver-derived reference-answer scoring (no real image)")
        _ref_questions: Optional[List[str]] = None
        _ref_answers: Optional[List[str]] = None

        if _use_self_clip_scoring:
            if _use_ref_scoring and verbose and self.is_main_process:
                print(
                    f"[Step {step:05d}][G] both self-clip and Solver-derived reference-answer scoring enabled; "
                    "using self-clip scoring."
                )
            scored = self._score_candidates_self_clip(
                prompt=spec.prompt,
                candidates=candidates,
                step=step,
                verbose=verbose,
            )
        elif _use_ref_scoring:
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
                candidate_specs=candidate_specs,
                candidate_spec_qualities=candidate_spec_qualities,
            )
        best_idx = max(range(len(scored)), key=lambda i: float(scored[i]["total_reward"]))
        best = scored[best_idx]

        # ---- Store best candidate in replay buffer ---- #
        # Best generated image enters the replay buffer for mixing into
        # understanding training. The buffer's quality gate (min_reward)
        # ensures only good images are kept.
        #
        # For Solver-derived reference-answer scoring (MODE B), total_reward is a log-prob (negative).
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
        dit_stats = None
        dit_update_due = False
        dit_skip_reason = None

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
                        # Update the generator baseline so subsequent steps get
                        # a meaningful advantage signal. Missing this (as in the
                        # original code) caused the baseline to freeze while the
                        # REINFORCE and PPO paths correctly updated it.
                        self._update_baseline("generator", generator_reward)

                        self._append_jsonl(
                            self.policy_updates_log_path,
                            {
                                "step": step,
                                "role": "generator",
                                "update_rule": "dpo",
                                "reward": generator_reward,
                                "baseline_before": baseline_before,
                                "baseline_after": self.generator_baseline,
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

                missing_trace_count = sum(
                    1
                    for sc in scored
                    if not str(sc.get("policy_completion", "")).strip()
                )
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
                    # For BLIP3o's diffusion backend, image candidates are
                    # produced by the DiT/flow decoder and carry no discrete
                    # generator token trace.  This is not a generic "too few"
                    # failure; it is the expected signal to route the Generator
                    # update to reward-weighted denoising below.
                    if missing_trace_count == len(scored) and any_needs_proxy:
                        grpo_skip_reason = "missing_generation_token_trace"
                    else:
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

        # ── Proposer joint-reward update (SUDER-style) ──────────────────────────
        # The proposer wrote the spec that led to the generated images. We reward
        # it with a JOINT signal that combines two objectives:
        #
        #   1. Entropy reward  (α):  gaussian_reward(mean_entropy_on_generated_image)
        #      — Identical objective to the understanding phase: reward the proposer
        #        for writing specs whose QA pairs are hard for the solver to answer on
        #        the generated image.  The solver already ran on the generated image
        #        for scoring; entropy_nats is already in best["qa_logs"][i]["solver"].
        #        Using the same gaussian_reward + mu/sigma as understanding phase
        #        unifies both proposers around one coherent difficulty objective.
        #
        #   2. Image-quality reward  (1-α):  spec_score + cycle_score + diversity
        #      — Keeps a quality floor: the proposer shouldn't write specs that are
        #        hard BECAUSE the image is garbage (unscorable).  A small weight (0.3)
        #        ensures the generated image is actually faithful to the spec.
        #
        #   reward = α * gaussian_reward(entropy) + (1-α) * image_quality
        #
        # KEY SAFETY: separate baseline EMA (proposer_gen_baseline) is never mixed
        # with self.proposer_baseline (understanding phase).  The proposer LoRA
        # weights are shared, but reward signals are tracked independently.
        proposer_update_due = False
        proposer_skip_reason = "disabled_in_generation_phase"
        proposer_reward = None
        proposer_stats = None

        _proposer_gen_reward_enabled = bool(
            getattr(self.cfg, "proposer_gen_reward_enabled", False)
        )
        if _proposer_gen_reward_enabled and best is not None:
            # ── Component 1: entropy reward from solver on generated image ────────
            # Extract entropy_nats from qa_logs (computed during candidate scoring).
            # Each entry in qa_logs["solver"]["entropy_nats"] is the Shannon entropy
            # of the solver's answer distribution on that QA pair.
            _qa_logs = best.get("qa_logs") or []
            _entropy_vals = []
            for _ql in _qa_logs:
                _solver_info = _ql.get("solver") if isinstance(_ql, dict) else None
                if isinstance(_solver_info, dict):
                    _e = _solver_info.get("entropy_nats")
                    if _e is not None:
                        _entropy_vals.append(float(_e))
            _mean_entropy = float(sum(_entropy_vals) / len(_entropy_vals)) if _entropy_vals else 0.0

            # Same gaussian_reward as understanding phase — unified difficulty objective.
            _proposer_entropy_mu = float(getattr(self.cfg, "prop_entropy_mu", 0.693))
            _proposer_entropy_sigma = float(getattr(self.cfg, "prop_entropy_sigma", 0.25))
            _entropy_component = gaussian_reward(_mean_entropy, _proposer_entropy_mu, _proposer_entropy_sigma)

            # Hard negative for zero entropy: spec produced a trivially easy image.
            _zero_entropy_cap = float(getattr(self.cfg, "zero_entropy_reward_cap", 0.10))
            if _mean_entropy < 1e-6:
                _entropy_component = -_zero_entropy_cap

            # ── Component 2: image-quality reward ────────────────────────────────
            # Normalize total_reward to [0,1] across all scoring modes:
            #   Mode A (weighted):   already in [0,1]
            #   Mode B (ref logp):   in (-inf,0] → sigmoid(logp+2) → [0,0.88]
            #   Mode C (self-clip):  already in [0,1]
            _raw_gen_reward = float(best.get("total_reward", 0.0))
            if _use_ref_scoring:
                _quality_component = 1.0 / (1.0 + math.exp(-(_raw_gen_reward + 2.0)))
            else:
                _quality_component = max(0.0, min(1.0, _raw_gen_reward))

            # ── Blend: α * entropy + (1-α) * quality ─────────────────────────────
            _alpha = float(getattr(self.cfg, "proposer_gen_entropy_weight", 0.7))
            _alpha = max(0.0, min(1.0, _alpha))
            _gen_proposer_reward = _alpha * _entropy_component + (1.0 - _alpha) * _quality_component
            _gen_proposer_reward = max(-1.0, min(1.0, _gen_proposer_reward))

            # Build completion: full proposer XML output.
            _proposer_gen_completion = str(spec.raw_output or spec.prompt or "").strip()
            _proposer_gen_prompt = str(getattr(spec, "proposer_prompt", spec.prompt) or "").strip()
            _grpo_completions: Optional[List[str]] = None  # set in GRPO branch for logging

            _local_proposer_ready = False
            _local_proposer_skip_reason: Optional[str] = None
            if not _proposer_gen_completion:
                _local_proposer_skip_reason = "empty_proposer_completion"
            elif spec_quality < self.cfg.min_spec_quality_for_update:
                _local_proposer_skip_reason = "low_spec_quality_for_proposer_gen"
            else:
                _local_proposer_ready = True

            proposer_ready, proposer_skip_reason = _global_update_ready(
                _local_proposer_ready,
                _local_proposer_skip_reason,
                peer_reason="distributed_peer_proposer_skip",
            )

            if proposer_ready:
                try:
                    if self._proposer_uses_grpo:
                        # ── GRPO path ────────────────────────────────────────────────────
                        # For the generation phase we need a group of completions.
                        # We already have one spec; sample (group_size - 1) additional specs
                        # from the proposer and blend their rewards the same way.
                        # These extra specs are NOT used for image generation — only for
                        # providing variance in the reward group so GRPO can normalize.
                        _grpo_group_size = max(
                            2, int(getattr(self.cfg, "proposer_grpo_gen_group_size", 3))
                        )
                        _grpo_completions = [_proposer_gen_completion]
                        _grpo_rewards = [_gen_proposer_reward]
                        _grpo_images = [image]  # None in imageless mode — OK

                        # Sample extra specs (lightweight — proposer only, no image gen)
                        # Unverified extras must NEVER outrank the verified chosen spec.
                        # Use a small margin below chosen reward so GRPO has a stable
                        # ordering signal without expensive extra image scoring.
                        _extra_margin = max(
                            2.0 * float(getattr(self.cfg, "grpo_min_group_std", 1e-6)),
                            float(getattr(self.cfg, "proposer_grpo_unverified_extra_margin", 0.02)),
                        )
                        for _gi in range(_grpo_group_size - 1):
                            try:
                                if _imageless:
                                    _extra_spec = self._propose_imageless_generation_spec(
                                        proposer_prompt=_proposer_gen_prompt or "",
                                    )
                                else:
                                    _extra_spec = self._propose_generation_spec(
                                        image=image,
                                        proposer_prompt=_proposer_gen_prompt or None,
                                    )
                                _extra_comp = str(_extra_spec.raw_output or _extra_spec.prompt or "").strip()
                                if not _extra_comp:
                                    continue
                                # Use a strict proxy lower than the chosen reward.
                                # This prevents unverified extras from tying/outperforming
                                # the verified candidate in baseline-shifted GRPO.
                                _extra_reward = max(
                                    -1.0,
                                    min(1.0, _gen_proposer_reward - _extra_margin),
                                )
                                _grpo_completions.append(_extra_comp)
                                _grpo_rewards.append(_extra_reward)
                                _grpo_images.append(image)
                            except Exception:
                                pass

                        # Apply EMA absolute baseline shift (same as understanding phase).
                        # Use proposer_gen_baseline (separate EMA for generation reward)
                        # so understanding and generation phases don't contaminate each other.
                        _gen_ema_baseline = float(self.proposer_gen_baseline)
                        _grpo_rewards_shifted = [r - _gen_ema_baseline for r in _grpo_rewards]

                        # When a generator GRPO backward already fired this step,
                        # the proposer backward is the 2nd DDP backward.  Pass
                        # ddp_no_sync=True to avoid DDP reducer confusion.
                        _prop_no_sync = (
                            generator_update_due
                            and generator_skipped_reason is None
                        )
                        proposer_stats = self.proposer_updater.step(
                            prompt=_proposer_gen_prompt,
                            completions=_grpo_completions,
                            rewards=_grpo_rewards_shifted,
                            device=self.device,
                            images=_grpo_images,
                            ddp_no_sync=_prop_no_sync,
                            baseline_shifted=True,
                        )
                        _gen_advantage = float(proposer_stats.get("mean_advantage", 0.0))

                        # Update the gen-phase EMA baseline from the CHOSEN
                        # candidate's reward only (not the group mean including
                        # unverified extras at 0.0).  Tracking chosen-only prevents
                        # shifted rewards from summing to zero at EMA equilibrium.
                        _gen_baseline_momentum = float(
                            getattr(self.cfg, "proposer_gen_baseline_momentum", 0.6)
                        )
                        self.proposer_gen_baseline = (
                            _gen_baseline_momentum * self.proposer_gen_baseline
                            + (1.0 - _gen_baseline_momentum) * _gen_proposer_reward
                        )
                        if proposer_stats is not None:
                            proposer_stats["grpo_ema_baseline"] = _gen_ema_baseline
                            proposer_stats["grpo_baseline_input"] = _gen_proposer_reward
                            proposer_stats["grpo_unverified_extra_margin"] = _extra_margin
                    else:
                        # ── REINFORCE path (legacy) ───────────────────────────────────────
                        _gen_baseline_momentum = float(
                            getattr(self.cfg, "proposer_gen_baseline_momentum", 0.6)
                        )
                        self.proposer_gen_baseline = (
                            _gen_baseline_momentum * self.proposer_gen_baseline
                            + (1.0 - _gen_baseline_momentum) * _gen_proposer_reward
                        )
                        _gen_advantage = _gen_proposer_reward - self.proposer_gen_baseline
                        _prop_no_sync_rf = (
                            generator_update_due
                            and generator_skipped_reason is None
                        )
                        proposer_stats = self.proposer_updater.step(
                            image=image,
                            prompt=_proposer_gen_prompt,
                            completion=_proposer_gen_completion,
                            reward=_gen_proposer_reward,
                            baseline=self.proposer_gen_baseline,
                            device=self.device,
                            ddp_no_sync=_prop_no_sync_rf,
                        )

                    proposer_update_due = True
                    proposer_skip_reason = None
                    proposer_reward = _gen_proposer_reward
                    if bool(proposer_stats.get("did_step", False)):
                        self._policy_update_counts["proposer"] = (
                            self._policy_update_counts.get("proposer", 0) + 1
                        )
                    _logged_group_size = (
                        len(_grpo_completions) if _grpo_completions is not None
                        else int(proposer_stats.get("group_size", 1))
                    )
                    self._append_jsonl(
                        self.policy_updates_log_path,
                        {
                            "step": step,
                            "role": "proposer_gen",
                            "update_due": True,
                            "update_rule": "grpo" if self._proposer_uses_grpo else "reinforce",
                            "reward": _gen_proposer_reward,
                            "entropy_component": float(_entropy_component),
                            "quality_component": float(_quality_component),
                            "mean_entropy_on_generated": float(_mean_entropy),
                            "entropy_weight_alpha": float(_alpha),
                            "grpo_group_size": _logged_group_size,
                            "advantage": float(_gen_advantage),
                            "spec_quality": float(spec_quality),
                            "stats": proposer_stats,
                        },
                    )
                except Exception as _prop_exc:
                    proposer_skip_reason = f"proposer_gen_update_error:{type(_prop_exc).__name__}"
                    proposer_update_due = False

        unicorn_recon_enqueued = self._enqueue_unicorn_reconstruction_tasks(
            step=step,
            image=image,
            spec=spec,
            best=best,
            spec_quality=float(spec_quality),
            target_difficulty=target_difficulty,
        )
        unicorn_reconstruction = self._run_unicorn_reconstruction_sft(step)
        unicorn_reconstruction["enqueued_this_step"] = int(unicorn_recon_enqueued)
        unicorn_reconstruction["buffer_size_after_step"] = int(len(self._unicorn_reconstruction_buffer))

        # ── Unified image-generator update ──────────────────────────────────────
        # Architecture-agnostic design: "train the module that produces the image"
        # maps to two different implementations depending on generation paradigm:
        #
        #   Paradigm A — discrete-token AR (Janus, VARGPT, LlamaGen, etc.):
        #     The LLM backbone directly generates discrete image token IDs.
        #     policy_completion_ids is non-None → generator GRPO ran above with
        #     real token traces → the correct module was already trained.
        #
        #   Paradigm B — continuous latent + diffusion/flow decoder (BLIP3o, BAGEL, etc.):
        #     The LLM encodes a spec prompt into continuous features; a separate
        #     DiT/flow denoiser produces the image from those features.
        #     policy_completion_ids is always None (no discrete image tokens exist).
        #     Generator GRPO would only train caption-writing, NOT image generation.
        #     The correct Generator adapter objective is reward-weighted denoising:
        #     DiT/flow LoRA and, when enabled, text-conditioning LoRA receive gradients
        #     through the denoising loss while base weights remain frozen.
        #
        # Routing logic:
        #   • generator_update_freq controls BOTH paths (single unified knob).
        #   • If generator GRPO fired successfully (token traces) → DiT skips.
        #   • If generator GRPO was skipped due to missing token traces AND a DiT
        #     updater exists → route to the diffusion Generator objective.
        #   • dit_update_enabled / dit_update_freq still work as explicit overrides
        #     for running DiT updates independently of generator_update_freq.
        #
        # This means: set generator_update_freq=1 for ALL architectures.
        # The framework automatically routes to GRPO (Janus/VARGPT) or DiT (BLIP3o).
        _generator_grpo_fired = (
            generator_update_due
            and generator_skipped_reason is None
            and generator_stats is not None
        )
        _missing_diffusion_trace_for_generator = bool(
            generator_update_due
            and not _generator_grpo_fired
            and (
                generator_skipped_reason in (
                    "missing_generation_token_trace",
                    "missing_generation_token_trace_strict",
                    "missing_trace_proxy_empty_completion",
                    "missing_trace_proxy_missing_image",
                    "dpo_proxy_empty_completion",
                    "dpo_proxy_missing_image",
                )
                or (
                    generator_skipped_reason == "grpo_too_few_completions"
                    and scored
                    and all(
                        not str(sc.get("policy_completion", "")).strip()
                        for sc in scored
                    )
                )
            )
        )
        _dit_should_fallback = (
            _missing_diffusion_trace_for_generator
            and self.dit_updater is not None
        )
        if (
            _missing_diffusion_trace_for_generator
            and self.dit_updater is None
            and bool(getattr(self.cfg, "require_dit_update", False))
        ):
            raise RuntimeError(
                "Generator update has no discrete token traces, so this BLIP3o "
                "run must route to DiT denoising, but no active DiT updater exists. "
                "Check earlier initialization logs for the exact DiT setup failure."
            )

        # Determine if the DiT should fire this step. Two triggers:
        #   1. Explicit dit_update_freq path (legacy/override, independent frequency).
        #   2. Unified diffusion route: generator_update_freq fired but no token
        #      traces exist, so the Generator update is the denoising objective.
        # Avoid double updates: if both triggers fire on the same step, run once.
        if self.dit_updater is not None:
            _explicit_dit_due = (
                self.cfg.dit_update_freq > 0
                and bool(step % int(self.cfg.dit_update_freq) == 0)
            )
            dit_update_due = _explicit_dit_due or _dit_should_fallback
            _dit_trigger = (
                "diffusion_generator_route" if _dit_should_fallback
                else "explicit_dit_update_freq" if _explicit_dit_due
                else None
            )
            if dit_update_due:
                # Pass best-candidate reward for Reward-Weighted Regression (RWR).
                # When dit_reward_loss_weight > 0, the DiT loss is scaled by
                # (1 + w * reward) so high-reward generations are reinforced more.
                # Pass None when reward weighting is disabled to keep original behaviour.
                _dit_reward = None
                if float(getattr(self.cfg, "dit_reward_loss_weight", 0.0)) > 0.0 and best is not None:
                    _raw_dit_reward = float(best.get("total_reward", 0.0))
                    # Normalize reward to [0, 1] before passing to RWR so that the
                    # loss scale (1 + w * reward) stays in [1.0, 1+w] regardless of
                    # the scoring mode:
                    #   Mode A (weighted score):   already in [0, 1] → identity
                    #   Mode B (ref-answer logp):  in (-inf, 0] → sigmoid(logp + 2)
                    #     logp = -4 → 0.12, -2 → 0.50, -1 → 0.73, 0 → 0.88
                    #   Mode C (self-clip cosine):  already in [0, 1] → identity
                    # Without normalization, Mode B values like -6.76 produce
                    # scale = 1 + 0.5*(-6.76) = -2.38 → clamped to 0 → no gradient.
                    if _use_ref_scoring:
                        _dit_reward = 1.0 / (1.0 + math.exp(-(_raw_dit_reward + 2.0)))
                    else:
                        _dit_reward = max(0.0, min(1.0, _raw_dit_reward))
                # In imageless mode (E5), image is None — use the best
                # generated candidate image for DiT denoising training instead.
                _dit_image = image
                if _dit_image is None and best is not None:
                    _dit_image = best.get("image")
                dit_stats = self.dit_updater.step(
                    image=_dit_image,
                    prompt=str(spec.prompt),
                    device=self.device,
                    reward=_dit_reward,
                )
                dit_skip_reason = (
                    str(dit_stats.get("skipped_reason"))
                    if dit_stats.get("skipped_reason") is not None
                    else None
                )
                if bool(dit_stats.get("did_step", False)):
                    self._policy_update_counts["dit"] += 1
                self._append_jsonl(
                    self.policy_updates_log_path,
                    {
                        "step": step,
                        "role": "dit",
                        "role_family": "generator",
                        "submodule": "dit",
                        "objective": (
                            "reward_weighted_denoising"
                            if float(getattr(self.cfg, "dit_reward_loss_weight", 0.0)) > 0.0
                            else "denoising"
                        ),
                        "trigger": _dit_trigger,
                        "explicit_dit_due": bool(_explicit_dit_due),
                        "missing_trace_route": bool(_dit_should_fallback),
                        "update_due": True,
                        "skipped": bool(dit_skip_reason),
                        "reason": dit_skip_reason,
                        "prompt": spec.prompt,
                        "stats": dit_stats,
                    },
                )
            else:
                dit_skip_reason = "update_not_due"

        self._sync_state_scalars()

        generator_effective_objective = "none"
        if generator_stats is not None and generator_skipped_reason is None:
            generator_effective_objective = f"text_{generator_update_mode or self.cfg.generator_update_rule}"
        if dit_stats is not None and bool(dit_stats.get("did_step", False)):
            generator_effective_objective = str(dit_stats.get("objective") or "diffusion_denoising")

        self._save_candidate_images(step=step, scored=scored, best_idx=best_idx)

        if verbose:
            step_dt = time.perf_counter() - step_t0
            print(
                f"[Step {step:05d}][G] generation phase done in {step_dt:.1f}s "
                f"(best_idx={best_idx}, best_reward={float(best['total_reward']):.3f})"
            )
        step_duration_sec = time.perf_counter() - step_t0

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
            qa_logs = cand.get("qa_logs", [])
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
                    "spec_score": cand.get("spec_score", 0.0),
                    "contradiction_score": cand.get("contradiction_score", 0.0),
                    "cycle_score": cand.get("cycle_score", 0.0),
                    "cycle_caption": cand.get("cycle_caption", ""),
                    "diversity_score": cand.get("diversity_score", 0.0),
                    "qa_confidence": cand.get("qa_confidence", 0.0),
                    "base_reward": cand.get("base_reward", 0.0),
                    "spec_quality": cand.get("spec_quality", 1.0),
                    "total_reward": cand.get("total_reward", 0.0),
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
                "generator_effective_objective": generator_effective_objective,
                "generator_proxy_ratio": float(self._current_proxy_ratio()),
                "unicorn_spec_meta": unicorn_spec_meta,
                "unicorn_reconstruction": unicorn_reconstruction,
                "dit_update_due": dit_update_due,
                "dit_skip_reason": dit_skip_reason,
                "dit_stats": dit_stats,
                "proposer_update_due": proposer_update_due,
                "proposer_skip_reason": proposer_skip_reason,
                "proposer_reward": proposer_reward,
                "proposer_stats": proposer_stats,
                "generator_update_stats": generator_stats,
            },
        )

        self._monitor_generation_step(
            step=step,
            meta=meta,
            scored=scored,
            best_idx=best_idx,
            spec_quality=spec_quality,
            generator_stats=generator_stats,
            generator_update_mode=generator_update_mode,
            generator_effective_objective=generator_effective_objective,
            generator_skipped_reason=generator_skipped_reason,
            dit_stats=dit_stats,
            dit_skip_reason=dit_skip_reason,
            proposer_stats=proposer_stats,
            proposer_skip_reason=proposer_skip_reason,
            proposer_reward=proposer_reward,
            step_duration_sec=step_duration_sec,
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
            "generator_effective_objective": generator_effective_objective,
            "generator_proxy_ratio": float(self._current_proxy_ratio()),
            "unicorn_spec_meta": unicorn_spec_meta,
            "unicorn_reconstruction": unicorn_reconstruction,
            "dit_update_due": dit_update_due,
            "dit_skip_reason": dit_skip_reason,
            "dit_stats": dit_stats,
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
            # When gen_step_solver_update_enabled is True, this solver backward
            # is the 2nd (or 3rd) backward through the DDP model in the same
            # G-step (after generator GRPO + optional proposer update).
            # DDP's reducer gets confused by multiple forward+backward cycles,
            # raising "gradient which is undefined, but still allreduced".
            # Fix: pass ddp_no_sync=True so the updater uses no_sync() and
            # manually allreduces only this adapter's gradients before opt.step.
            _solver_no_sync = bool(getattr(self.cfg, "gen_step_solver_update_enabled", False))
            stats = self.solver_updater.step(
                image=image,
                prompt=build_solver_prompt(question),
                completion=effective_completion,
                reward=reward,
                baseline=baseline_before if local_has_completion else 0.0,
                device=self.device,
                ddp_no_sync=_solver_no_sync,
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
        run_started_at = float(time.time())

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
                        "dit_update_due": out.get("dit_update_due"),
                        "dit_skip_reason": out.get("dit_skip_reason"),
                        "dit_stats": out.get("dit_stats"),
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
                    dit_stats=out.get("dit_stats"),
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
                dit_stats = out.get("dit_stats") or {}
                self._update_metric(
                    "dit_update_applied",
                    1.0 if bool(dit_stats.get("did_step", False)) else 0.0,
                )
                if "loss" in dit_stats:
                    try:
                        dit_loss_val = float(dit_stats.get("loss", 0.0))
                        if math.isfinite(dit_loss_val):
                            self._update_metric("dit_loss", dit_loss_val)
                    except Exception:
                        pass
                _emit_training_logs(step, phase="generation", step_time_sec=step_duration_g)

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
            self._append_metrics({"kind": "final_summary", **final_progress, **summary})
            self._write_status(
                state="completed",
                progress=final_progress,
                metrics=self._release_metrics(step_time_sec=0.0),
            )
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
                            "resume_from": str(self.checkpoint_root / f"step_{emergency_step:06d}"),
                            "start_step": emergency_step,
                            "total_steps": cfg.total_steps,
                            "command_example": (
                                "python BLIP3o/blip3o/train/train_self_evolving.py "
                                f"--experiment {cfg.experiment_name} --data_dir {cfg.data_dir} "
                                f"--output_dir {cfg.output_dir} --run_name {self.run_dir.name} "
                                f"--resume_from {self.checkpoint_root / f'step_{emergency_step:06d}'} "
                                f"--start_step {emergency_step} --total_steps {cfg.total_steps}"
                            ),
                        },
                    )
            except Exception as save_exc:
                if self.is_main_process:
                    print(f"[Generation] Emergency checkpoint failed: {save_exc}")

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
