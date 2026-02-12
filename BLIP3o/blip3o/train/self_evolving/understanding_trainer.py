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
import os
import pathlib
import random
import shutil
import time
import traceback
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoProcessor

from .config import UnderstandingSelfEvolvingConfig
from .image_pool import ImagePool, ImagePoolConfig
from .model_api import _load_blip3o_model
from .policy_updater import RolePolicyUpdater
from .prompts import build_proposer_prompt, build_solver_prompt
from .utils import (
    HAS_PEFT,
    HAS_WANDB,
    _build_chat_text,
    _collect_git_info,
    _decode_tokens,
    _infer_primary_device,
    _json_dump,
    _parse_answer,
    _parse_first_question,
    _prepare_mm_inputs,
    _resolve_attn_implementation,
    _safe_dtype,
    _set_global_seed,
    gaussian_reward,
    majority_vote,
    normalize_answer,
    pre_answer_word_count,
    shannon_entropy_nats,
    strip_tags,
    use_adapter,
)

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

    # -------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------
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
        )
        if self.is_main_process:
            print(
                f"[Understanding] Load options: dtype={dtype}, device_map={device_map}, "
                f"attn_implementation={attn_impl or 'default'}"
            )

        if self.cfg.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            lcfg = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=list(self.cfg.lora_target_modules),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lcfg)
            # default adapter is solver; create proposer adapter explicitly
            if hasattr(model, "add_adapter"):
                try:
                    model.add_adapter("proposer", lcfg)
                except Exception as exc:
                    raise RuntimeError(f"Failed to add proposer adapter: {exc}") from exc

            # Keep training restricted to role adapters only.
            for name, param in model.named_parameters():
                if "lora_" in name and (".default." in name or ".proposer." in name):
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
                    do_sample=True,
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
        self._append_jsonl(self.iter_log_path, record)

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
                "final_step": int(final_step),
                "start_step": int(self.start_step),
                "status": status,
                "interrupted_at_step": interrupted_at_step,
                "error": error,
                "policy_update_counts": self._policy_update_counts,
                "metrics": self._metrics_summary(),
            },
        )

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
        last_completed_step = self.start_step
        last_attempted_step = self.start_step
        try:
            for step in range(self.start_step + 1, cfg.total_steps + 1):
                last_attempted_step = step
                step_t0 = time.perf_counter()
                image, meta = self._sample_image_for_step(step)

                # --- Proposer ---
                proposer_prompt = build_proposer_prompt()
                proposer_out = self._generate(
                    image=image,
                    prompt=proposer_prompt,
                    adapter_name="proposer" if cfg.use_lora else None,
                    max_new_tokens=cfg.max_new_tokens_proposer,
                    temperature=cfg.temp,
                    top_p=cfg.top_p,
                )
                parsed_question = _parse_first_question(proposer_out).replace("\n", " ").strip()
                question = parsed_question
                if not question:
                    question = "What is the most salient object in the image?"
                fallback_used = not bool(parsed_question)
                proposer_rationale = strip_tags(proposer_out, "rationale")

                # --- Solver ---
                solver_prompt = build_solver_prompt(question)
                solver_outputs: List[str] = []
                solver_answers_raw: List[str] = []
                solver_answers_norm: List[str] = []
                pre_words: List[int] = []

                solver_temperatures = self._solver_temperature_schedule()
                solver_top_ps = self._solver_top_p_schedule()
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
                entropy_min = float(getattr(cfg, "sc_entropy_min", 0.15))
                entropy_max = float(getattr(cfg, "sc_entropy_max", 1.2))
                if entropy_min > entropy_max:
                    entropy_min, entropy_max = entropy_max, entropy_min
                margin_max = float(getattr(cfg, "sc_margin_max", 0.9))
                ratio_min = float(getattr(cfg, "sc_informative_ratio_min", 0.25))
                ratio_min = max(0.0, min(1.0, ratio_min))
                neg_weight = float(getattr(cfg, "sc_negative_weight", 0.25))
                # Informativeness score encourages moderate disagreement and
                # penalizes collapsed unanimity.
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
                solver_informative_gate = informative_ratio >= ratio_min

                sc_signal = max(1e-4, local_info_score)
                # Penalize unanimous, low-entropy, high-margin (trivially easy) cases.
                easy_solver_case = bool((entropy_nats < entropy_min) and (margin > margin_max))
                easy_solver_penalty_scale = float(
                    getattr(cfg, "easy_solver_penalty_scale", 1.0)
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
                proposer_entropy_mu_used = self._update_proposer_entropy_target(entropy_nats)
                proposer_reward_raw = gaussian_reward(
                    entropy_nats, proposer_entropy_mu_used, cfg.prop_entropy_sigma
                )
                proposer_reward = proposer_reward_raw
                # Penalize zero-entropy (unanimous) outcomes — question was too easy.
                zero_entropy_cap = float(getattr(cfg, "zero_entropy_reward_cap", 0.10))
                zero_entropy_capped = False
                if entropy_nats < 1e-6 and zero_entropy_cap < 1.0:
                    proposer_reward = min(proposer_reward, zero_entropy_cap)
                    zero_entropy_capped = True
                # Additional easy-question penalty for low-entropy, high-margin cases.
                easy_question_penalty = float(getattr(cfg, "easy_question_penalty", 0.15))
                easy_question_detected = bool(
                    (entropy_nats < entropy_min) and (margin > margin_max)
                )
                if easy_question_detected and easy_question_penalty > 0.0:
                    proposer_reward -= easy_question_penalty
                proposer_reward = max(-1.0, min(1.0, proposer_reward))

                # --- Solver policy updates ---
                solver_baseline_before_step = self.solver_baseline
                solver_step_stats = []
                solver_update_due = True
                solver_update_applied = True
                solver_update_skip_reason = None
                skip_uninformative = bool(
                    getattr(cfg, "skip_solver_update_when_uninformative", True)
                )
                always_scale = bool(
                    getattr(cfg, "solver_always_update_with_informative_scaling", True)
                )
                min_update_scale = float(getattr(cfg, "solver_update_min_scale", 0.20))
                min_update_scale = max(0.0, min(1.0, min_update_scale))
                if always_scale:
                    solver_update_scale = max(min_update_scale, informative_ratio)
                else:
                    solver_update_scale = 1.0
                    if skip_uninformative and not solver_informative_gate:
                        solver_update_applied = False
                        solver_update_skip_reason = (
                            "uninformative_ratio_below_threshold"
                        )

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
                        solver_step_stats.append(stats)
                        if stats.get("did_step", True):
                            self._policy_update_counts["solver"] += 1
                        if not local_skip_update:
                            self._update_baseline("solver", reward)
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
                            if solver_update_skip_reason is None:
                                solver_update_skip_reason = "all_solver_samples_skipped"
                else:
                    self._append_jsonl(
                        self.policy_updates_log_path,
                        {
                            "step": step,
                            "role": "solver",
                            "skipped": True,
                            "reason": solver_update_skip_reason,
                            "solver_margin": margin,
                            "entropy_nats": entropy_nats,
                        },
                    )
                solver_baseline_after_step = self.solver_baseline

                # --- Proposer policy update ---
                proposer_baseline_before_step = self.proposer_baseline
                proposer_baseline_after_step = proposer_baseline_before_step
                proposer_stats = None
                if step % cfg.proposer_update_freq == 0:
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
                            baseline=proposer_baseline_before_step if local_can_proposer_update else 0.0,
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

                solver_ce_mean = (
                    sum(s["ce_loss"] for s in solver_step_stats) / max(1, len(solver_step_stats))
                )
                solver_kl_mean = (
                    sum(s["kl_loss"] for s in solver_step_stats) / max(1, len(solver_step_stats))
                )
                solver_adv_mean = (
                    sum(s["advantage"] for s in solver_step_stats)
                    / max(1, len(solver_step_stats))
                )
                solver_stats_mean = {
                    "ce_loss_mean": solver_ce_mean,
                    "kl_loss_mean": solver_kl_mean,
                    "advantage_mean": solver_adv_mean,
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
                        "proposer_prompt": proposer_prompt,
                        "proposer_output": proposer_out,
                        "proposer_rationale": proposer_rationale,
                        "parsed_question": parsed_question,
                        "final_question": question,
                        "fallback_question_used": fallback_used,
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
                        "easy_solver_case": easy_solver_case,
                        "easy_solver_penalty_scale": easy_solver_penalty_scale,
                        "solver_update_scale": solver_update_scale,
                        "solver_informative_local": solver_informative_local,
                        "solver_informative_any": solver_informative_any,
                        "solver_informative_all": solver_informative_all,
                        "solver_informative_ratio": informative_ratio,
                        "solver_informative_ratio_min": ratio_min,
                        "solver_informative_gate": solver_informative_gate,
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
                        "zero_entropy_capped": zero_entropy_capped,
                        "zero_entropy_reward_cap": zero_entropy_cap,
                        "easy_question_detected": easy_question_detected,
                        "easy_question_penalty": easy_question_penalty,
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
                        "easy_solver_case": easy_solver_case,
                        "easy_solver_penalty_scale": easy_solver_penalty_scale,
                        "solver_informative_local": solver_informative_local,
                        "solver_informative_any": solver_informative_any,
                        "solver_informative_all": solver_informative_all,
                        "solver_informative_ratio": informative_ratio,
                        "solver_informative_gate": solver_informative_gate,
                        "solver_update_scale": solver_update_scale,
                        "entropy_nats": entropy_nats,
                        "proposer_entropy_mu_used": proposer_entropy_mu_used,
                        "proposer_reward_raw": proposer_reward_raw,
                        "proposer_reward": proposer_reward,
                        "zero_entropy_capped": zero_entropy_capped,
                        "zero_entropy_reward_cap": zero_entropy_cap,
                        "easy_question_detected": easy_question_detected,
                        "easy_question_penalty": easy_question_penalty,
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
                self._update_metric("pre_answer_words_mean", pre_words_mean_global)
                self._update_metric("solver_kl_coef", float(self.solver_updater.kl_coef))
                self._update_metric("proposer_kl_coef", float(self.proposer_updater.kl_coef))
                self._update_metric("step_duration_sec", step_duration_sec_global)
                self._update_metric("fallback_question_used", 1.0 if fallback_used else 0.0)

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
            self._write_ablation_summary(cfg.total_steps, status="completed")
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
                            "resume_from": str(self.run_dir / f"step_{emergency_step:05d}"),
                            "start_step": emergency_step,
                            "total_steps": cfg.total_steps,
                        },
                    )
            except Exception as save_exc:
                if self.is_main_process:
                    print(f"[Understanding] Emergency checkpoint failed: {save_exc}")

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
