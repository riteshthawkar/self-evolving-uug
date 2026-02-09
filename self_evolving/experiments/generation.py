"""
Generation-only and unified self-evolving experiments.

This module implements:
- generation_self_evolving
- unified_self_evolving (alternating understanding + generation)

Design goals:
- Single-codebase reproducibility (metadata, checkpoints, resumability)
- Role-specific adapters (solver / proposer / generator)
- Detailed ablation logs for proposer/spec/reward internals
"""

import dataclasses
import datetime as dt
import gc
import json
import math
import os
import pathlib
import random
import re
import shutil
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

from self_evolving.data.image_pool import ImagePool, ImagePoolConfig
import self_evolving.experiments.understanding as _u

DEFAULT_LORA_TARGETS = _u.DEFAULT_LORA_TARGETS
HAS_PEFT = _u.HAS_PEFT
HAS_WANDB = _u.HAS_WANDB
RolePolicyUpdater = _u.RolePolicyUpdater
_build_chat_text = _u._build_chat_text
_clip_grad_norm_multi_device = _u._clip_grad_norm_multi_device
_collect_git_info = _u._collect_git_info
_collect_trainable_params = _u._collect_trainable_params
_decode_tokens = _u._decode_tokens
_infer_primary_device = _u._infer_primary_device
_json_dump = _u._json_dump
_load_model_with_fallback = _u._load_model_with_fallback
_parse_answer = _u._parse_answer
_parse_first_question = _u._parse_first_question
_prepare_mm_inputs = _u._prepare_mm_inputs
_resolve_attn_implementation = _u._resolve_attn_implementation
_safe_dtype = _u._safe_dtype
_set_global_seed = _u._set_global_seed
_unwrap_model = _u._unwrap_model
build_proposer_prompt = _u.build_proposer_prompt
build_solver_prompt = _u.build_solver_prompt
gaussian_reward = _u.gaussian_reward
majority_vote = _u.majority_vote
normalize_answer = _u.normalize_answer
pre_answer_word_count = _u.pre_answer_word_count
shannon_entropy_nats = _u.shannon_entropy_nats
strip_tags = _u.strip_tags
use_adapter = _u.use_adapter
LoraConfig = getattr(_u, "LoraConfig", None)
TaskType = getattr(_u, "TaskType", None)
get_peft_model = getattr(_u, "get_peft_model", None)

if HAS_WANDB:
    import wandb


GEN_PROMPT_TEMPLATE = (
    "You are a generation-spec proposer for self-evolving training.\n"
    "Given the source image, propose one new text-to-image prompt and verification QA pairs.\n"
    "Rules:\n"
    "- Prompt must be image-grounded but not a trivial copy.\n"
    "- QA pairs must be short-answer and visually verifiable.\n"
    "- Expected answers must be concise.\n"
    "Output XML only:\n"
    "<prompt>...</prompt>\n"
    "<spec>\n"
    "  <qa><question>...</question><expected>...</expected></qa>\n"
    "  <qa><question>...</question><expected>...</expected></qa>\n"
    "  <qa><question>...</question><expected>...</expected></qa>\n"
    "</spec>"
)

SOURCE_CAPTION_PROMPT = (
    "Describe this image in one concise sentence with key entities, attributes, and scene context."
)

GEN_CYCLE_CAPTION_PROMPT = (
    "Describe this image in one concise sentence focusing on key visual facts."
)


@dataclass
class GenerationQAPair:
    question: str
    expected: str


@dataclass
class GenerationSpec:
    prompt: str
    qa_pairs: Tuple[GenerationQAPair, ...]
    raw_output: str
    fallback_used: bool


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _jaccard_similarity(a: str, b: str) -> float:
    ta = set(_tokenize_words(a))
    tb = set(_tokenize_words(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return float(inter) / float(max(1, union))


def _parse_float_safe(text: str) -> Optional[float]:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _soft_match(pred: str, expected: str) -> float:
    p = normalize_answer(pred)
    e = normalize_answer(expected)
    if not e:
        return 0.5
    if p == e:
        return 1.0
    if p and e and (p in e or e in p):
        return 0.8

    pn = _parse_float_safe(p)
    en = _parse_float_safe(e)
    if pn is not None and en is not None:
        den = max(abs(en), 1.0)
        rel = abs(pn - en) / den
        if rel <= 0.01:
            return 1.0
        if rel <= 0.05:
            return 0.7
        if rel <= 0.20:
            return 0.4
        return 0.0

    return _jaccard_similarity(p, e)


def _yes_no_polarity(text: str) -> int:
    t = normalize_answer(text)
    if t.startswith("yes") or t in {"true", "1", "present"}:
        return 1
    if t.startswith("no") or t in {"false", "0", "absent"}:
        return -1
    return 0


def _image_diversity_score(images: List[Image.Image]) -> float:
    if len(images) <= 1:
        return 0.5
    if not HAS_NUMPY:
        return 0.5

    vectors = []
    for image in images:
        arr = np.asarray(image.convert("RGB").resize((48, 48)), dtype=np.float32) / 255.0
        vectors.append(arr.reshape(-1))

    dists = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            d = float(np.mean(np.abs(vectors[i] - vectors[j])))
            dists.append(d)
    if not dists:
        return 0.5

    # Typical mean abs RGB diff lies near [0, ~0.35]. Scale to [0,1].
    scaled = min(1.0, max(0.0, (sum(dists) / len(dists)) / 0.25))
    return scaled


def _ensure_pil_image(image_obj) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj
    if HAS_NUMPY and isinstance(image_obj, np.ndarray):
        arr = image_obj
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
            return Image.fromarray(arr)
    if torch.is_tensor(image_obj):
        tensor = image_obj.detach().cpu()
        if tensor.ndim == 3:
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)
            arr = tensor.numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
            return Image.fromarray(arr)
    raise TypeError(f"Unsupported generated image type: {type(image_obj)}")


def _parse_generation_spec(raw_text: str) -> GenerationSpec:
    prompt_text = (strip_tags(raw_text, "prompt") or "").strip()

    qa_pairs: List[GenerationQAPair] = []
    pattern = re.compile(
        r"<qa>\s*<question>(.*?)</question>\s*<expected>(.*?)</expected>\s*</qa>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for q, e in pattern.findall(raw_text):
        qn = " ".join(q.strip().split())
        en = " ".join(e.strip().split())
        if qn:
            qa_pairs.append(GenerationQAPair(question=qn, expected=en))

    # Fallback parsing for line-based outputs.
    if not qa_pairs:
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        if not prompt_text:
            for line in lines:
                if line.lower().startswith("prompt:"):
                    prompt_text = line.split(":", 1)[1].strip()
                    break

        pending_q: Optional[str] = None
        for line in lines:
            lower = line.lower()
            if lower.startswith("q") and ":" in line:
                pending_q = line.split(":", 1)[1].strip()
            elif lower.startswith("a") and ":" in line and pending_q:
                ans = line.split(":", 1)[1].strip()
                qa_pairs.append(GenerationQAPair(question=pending_q, expected=ans))
                pending_q = None

    fallback = False
    if not prompt_text:
        fallback = True
        prompt_text = "Generate a realistic image with clear salient objects and readable details."

    if not qa_pairs:
        fallback = True

    return GenerationSpec(
        prompt=prompt_text,
        qa_pairs=tuple(qa_pairs[:3]),
        raw_output=raw_text,
        fallback_used=fallback,
    )


def _prepare_text_inputs(processor, device: torch.device, text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    return inputs.to(device)


class TextPolicyUpdater:
    """
    KL-regularized REINFORCE updater for text-only trajectories (generator role).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor,
        config,
        adapter_name: Optional[str],
        reference_model: Optional[torch.nn.Module] = None,
    ):
        self.model = model
        self.processor = processor
        self.config = config
        self.adapter_name = adapter_name
        self.reference_model = reference_model
        self.kl_coef = config.kl_coef
        self.step_id = 0

        params = list(_collect_trainable_params(model, adapter_name))
        if not params:
            raise RuntimeError(f"No trainable parameters found for adapter={adapter_name!r}")
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    def state_dict(self) -> Dict:
        return {
            "optimizer": self.opt.state_dict(),
            "kl_coef": float(self.kl_coef),
            "step_id": int(self.step_id),
        }

    def load_state_dict(self, state: Dict):
        if not isinstance(state, dict):
            return
        if "optimizer" in state:
            self.opt.load_state_dict(state["optimizer"])
        if "kl_coef" in state:
            self.kl_coef = float(state["kl_coef"])
        if "step_id" in state:
            self.step_id = int(state["step_id"])

    def _adapt_beta(self, kl_val: float):
        target = max(self.config.kl_target, 1e-8)
        delta = (kl_val - target) / target
        beta = self.kl_coef * math.exp(self.config.kl_adapt_rate * delta)
        beta = max(self.config.kl_min, min(self.config.kl_max, beta))
        self.kl_coef = float(beta)

    def step(
        self,
        prompt: str,
        completion: str,
        reward: float,
        baseline: float,
        device: torch.device,
    ) -> Dict[str, float]:
        completion = completion or "\n<image>"
        self.step_id += 1

        text_prompt = prompt
        text_full = prompt + completion

        inputs_prompt = _prepare_text_inputs(self.processor, device, text_prompt)
        inputs_full = _prepare_text_inputs(self.processor, device, text_full)

        input_ids = inputs_full["input_ids"]
        labels = input_ids.clone()
        prompt_len = inputs_prompt["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        valid_mask = labels[:, 1:] != -100

        self.model.train(True)
        policy_inputs = dict(inputs_full)
        policy_inputs["labels"] = labels
        with use_adapter(self.model, self.adapter_name):
            out_pi = self.model(**policy_inputs)
        ce_loss = out_pi.loss
        logp_pi = F.log_softmax(out_pi.logits, dim=-1)

        ref_inputs = dict(inputs_full)
        if self.reference_model is not None:
            with torch.no_grad():
                out_ref = self.reference_model(**ref_inputs)
        else:
            with torch.no_grad():
                with use_adapter(self.model, None):
                    out_ref = self.model(**ref_inputs)
        logp_ref = F.log_softmax(out_ref.logits, dim=-1)

        logp_pi_shift = logp_pi[:, :-1, :]
        logp_ref_shift = logp_ref[:, :-1, :]
        p_pi_shift = logp_pi_shift.exp()
        kl_per_tok = (p_pi_shift * (logp_pi_shift - logp_ref_shift)).sum(dim=-1)
        if valid_mask.any():
            kl_loss = kl_per_tok[valid_mask].mean()
        else:
            kl_loss = torch.tensor(0.0, device=ce_loss.device)

        advantage = float(reward - baseline)
        beta_before = float(self.kl_coef)
        total_loss = advantage * ce_loss + beta_before * kl_loss

        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()
        _clip_grad_norm_multi_device(self.params, self.config.grad_clip)
        self.opt.step()
        self.model.train(False)

        kl_val = float(kl_loss.item())
        self._adapt_beta(kl_val)

        try:
            del inputs_prompt, inputs_full, input_ids, labels, policy_inputs
            del out_pi, out_ref, logp_pi, logp_ref, logp_pi_shift, logp_ref_shift
            del p_pi_shift, kl_per_tok, valid_mask
        except Exception:
            pass

        if (
            torch.cuda.is_available()
            and self.config.clear_cache_every > 0
            and self.step_id % self.config.clear_cache_every == 0
        ):
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            gc.collect()

        return {
            "ce_loss": float(ce_loss.item()),
            "kl_loss": kl_val,
            "advantage": advantage,
            "kl_coef_before": beta_before,
            "kl_coef_after": float(self.kl_coef),
            "total_loss": float(total_loss.item()),
        }


@dataclass
class GenerationSelfEvolvingConfig:
    experiment_name: str = "generation_self_evolving"
    run_name: Optional[str] = None
    output_dir: str = "./runs"

    # Data
    data_dir: str = ""
    data_split: str = "all"
    include_subfolders: Optional[Tuple[str, ...]] = None
    max_images: Optional[int] = None

    # Model
    model_name: str = "BLIP3o/BLIP3o-NEXT-4B"
    dtype: str = "bfloat16"
    cuda_device: int = 0
    device_map: str = "single"
    attn_implementation: str = "auto"

    # Optimization
    total_steps: int = 100
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    proposer_update_freq: int = 5
    generator_update_freq: int = 1
    enable_solver_updates: bool = False
    solver_update_freq: int = 0

    # Decoding
    temp: float = 1.0
    top_p: float = 1.0
    max_new_tokens_solver: int = 128
    max_new_tokens_proposer: int = 256
    max_new_tokens_caption: int = 96
    max_new_tokens_generator: int = 768
    num_solver_samples: int = 5
    num_generations: int = 4

    # Generation backend
    generation_num_inference_steps: int = 30
    generation_guidance_scale: float = 2.0
    generation_height: int = 1024
    generation_width: int = 1024
    strict_require_generation_tokens: bool = False

    # Reward shaping
    solver_soft_gamma: float = 0.7
    len_penalty_weight: float = 0.10
    len_penalty_target_words: int = 6
    prop_entropy_mu: float = 0.90
    prop_entropy_sigma: float = 0.35
    reward_spec_weight: float = 0.65
    reward_cycle_weight: float = 0.20
    reward_diversity_weight: float = 0.10
    reward_contradiction_weight: float = 0.20

    # KL control
    kl_coef: float = 1e-3
    kl_target: float = 0.02
    kl_adapt_rate: float = 0.10
    kl_min: float = 1e-8
    kl_max: float = 1e2

    # Baselines
    baseline_momentum: float = 0.9

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = DEFAULT_LORA_TARGETS

    # Repro + logging
    seed: int = 42
    deterministic: bool = True
    log_every: int = 1
    save_every: int = 50
    max_checkpoints: int = 3
    clear_cache_every: int = 25
    save_generated_images_every: int = 0

    # W&B
    wandb_mode: str = "disabled"
    wandb_project: str = "self-evolving-uug"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_images_every: int = 0

    # Resume
    resume_from: Optional[str] = None
    start_step: int = 0


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
                dist.init_process_group(backend=backend, init_method="env://")
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

    def _sync_state_scalars(self):
        if not (self.distributed and dist.is_initialized()):
            return
        self.generator_baseline = self._dist_mean(self.generator_baseline)
        self.proposer_baseline = self._dist_mean(self.proposer_baseline)
        if self.solver_updater is not None:
            self.solver_baseline = self._dist_mean(self.solver_baseline)
            self.solver_updater.kl_coef = self._dist_mean(self.solver_updater.kl_coef)
        self.proposer_updater.kl_coef = self._dist_mean(self.proposer_updater.kl_coef)
        self.generator_updater.kl_coef = self._dist_mean(self.generator_updater.kl_coef)

    def __init__(self, config: GenerationSelfEvolvingConfig):
        self.cfg = config
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
            reference_model = _load_model_with_fallback(
                config.model_name,
                torch_dtype=_safe_dtype(config.dtype),
                device_map={"": fallback_dev} if self.device.type == "cuda" else "cpu",
                trust_remote_code=True,
                attn_implementation=_resolve_attn_implementation(config.attn_implementation),
            )
            reference_model.eval()
            for p in reference_model.parameters():
                p.requires_grad_(False)

        self.train_model = self.model
        if self.distributed:
            ddp_kwargs = {"find_unused_parameters": True}
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
        self.start_step = max(0, int(config.start_step))

        self._metric_stats: Dict[str, Dict[str, float]] = {}
        self._policy_update_counts: Dict[str, int] = {"solver": 0, "proposer": 0, "generator": 0}
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
        repo_root = pathlib.Path(__file__).resolve().parents[2]
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

        # Load trainable parameter snapshot to restore adapter weights without rebinding modules.
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
                and (step_dir / "trainable_adapters.pt").is_file()
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
            device_map = {"": self.local_rank} if torch.cuda.is_available() else "cpu"
        elif self.cfg.device_map == "single":
            device_map = {"": self.cfg.cuda_device} if torch.cuda.is_available() else "cpu"
        elif self.cfg.device_map == "cpu":
            device_map = "cpu"
        else:
            device_map = "auto"

        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(self.cfg.model_name, trust_remote_code=True)
        model = _load_model_with_fallback(
            self.cfg.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

        if self.is_main_process:
            print(
                f"[Generation] Load options: dtype={dtype}, device_map={device_map}, "
                f"attn_implementation={attn_impl or 'default'}"
            )

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

        model.eval()
        return model, processor

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
        inputs = _prepare_mm_inputs(self.processor, self.device, image, chat_text)
        with torch.no_grad():
            with use_adapter(self.model, adapter_name):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None),
                )

        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
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

    def _propose_generation_spec(self, image: Image.Image) -> GenerationSpec:
        raw = self._generate(
            image=image,
            prompt=GEN_PROMPT_TEMPLATE,
            adapter_name="proposer" if self.cfg.use_lora else None,
            max_new_tokens=self.cfg.max_new_tokens_proposer,
            temperature=self.cfg.temp,
            top_p=self.cfg.top_p,
        )
        spec = _parse_generation_spec(raw)
        return spec

    def _generate_image_candidate(self, prompt: str) -> Dict[str, object]:
        model_ref = _unwrap_model(self.model)

        # Path 1: BLIP3o style API with token trace.
        if hasattr(model_ref, "generate_images"):
            text_inputs = _prepare_text_inputs(self.processor, self.device, prompt)
            with torch.no_grad():
                with use_adapter(self.model, "generator" if self.cfg.use_lora else None):
                    out = model_ref.generate_images(
                        input_ids=text_inputs.get("input_ids"),
                        attention_mask=text_inputs.get("attention_mask"),
                        max_new_tokens=self.cfg.max_new_tokens_generator,
                        temperature=self.cfg.temp,
                        top_p=self.cfg.top_p,
                        num_inference_steps=self.cfg.generation_num_inference_steps,
                        guidance_scale=self.cfg.generation_guidance_scale,
                        return_tensor=False,
                        enable_progress_bar=False,
                    )

            token_completion = ""
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
                        token_completion = _decode_tokens(self.processor, completion_ids).strip()
                except Exception:
                    token_completion = ""
            else:
                images = out
                if isinstance(images, list) and images:
                    image_out = images[0]
                elif isinstance(images, Image.Image):
                    image_out = images

            if image_out is None:
                raise RuntimeError("generate_images returned no image output.")

            return {
                "image": _ensure_pil_image(image_out),
                "policy_prompt": prompt,
                "policy_completion": token_completion,
                "backend": "generate_images",
            }

        # Path 2: generic single-image API.
        if hasattr(model_ref, "generate_image"):
            with torch.no_grad():
                with use_adapter(self.model, "generator" if self.cfg.use_lora else None):
                    image_out = model_ref.generate_image(
                        prompt=prompt,
                        num_inference_steps=self.cfg.generation_num_inference_steps,
                        guidance_scale=self.cfg.generation_guidance_scale,
                        height=self.cfg.generation_height,
                        width=self.cfg.generation_width,
                    )
            return {
                "image": _ensure_pil_image(image_out),
                "policy_prompt": prompt,
                "policy_completion": "\n<image>",
                "backend": "generate_image",
            }

        raise RuntimeError(
            "Model does not expose a supported image generation API. "
            "Expected `generate_images(...)` or `generate_image(...)`."
        )

    def _solve_question_with_rollouts(self, image: Image.Image, question: str) -> Dict[str, object]:
        solver_prompt = build_solver_prompt(question)
        rollouts = []
        answers_norm: List[str] = []

        for _ in range(self.cfg.num_solver_samples):
            completion = self._generate(
                image=image,
                prompt=solver_prompt,
                adapter_name="default" if self.cfg.use_lora else None,
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
    ) -> Tuple[float, float, List[Dict[str, object]]]:
        if not qa_pairs:
            return 0.5, 0.0, []

        qa_logs: List[Dict[str, object]] = []
        score_values = []
        contradiction_values = []

        for qa in qa_pairs:
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

        spec_score = float(sum(score_values) / max(1, len(score_values)))
        contradiction_score = float(sum(contradiction_values) / max(1, len(contradiction_values)))
        return spec_score, contradiction_score, qa_logs

    def _cycle_reward(self, prompt: str, image: Image.Image) -> Tuple[float, str]:
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
        score = _jaccard_similarity(prompt, caption)
        return score, caption

    def _score_candidates(
        self,
        prompt: str,
        qa_pairs: Tuple[GenerationQAPair, ...],
        candidates: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        images = [cand["image"] for cand in candidates]
        diversity = _image_diversity_score(images)

        scored: List[Dict[str, object]] = []
        for idx, cand in enumerate(candidates):
            image = cand["image"]
            spec_score, contradiction_score, qa_logs = self._score_spec(image=image, qa_pairs=qa_pairs)
            cycle_score, cycle_caption = self._cycle_reward(prompt=prompt, image=image)

            total_reward = (
                self.cfg.reward_spec_weight * spec_score
                + self.cfg.reward_cycle_weight * cycle_score
                + self.cfg.reward_diversity_weight * diversity
                - self.cfg.reward_contradiction_weight * contradiction_score
            )
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
                    "diversity_score": diversity,
                    "total_reward": total_reward,
                    "qa_logs": qa_logs,
                    "image": image,
                }
            )
        return scored

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
                "final_step": int(final_step),
                "start_step": int(self.start_step),
                "status": status,
                "interrupted_at_step": interrupted_at_step,
                "error": error,
                "policy_update_counts": self._policy_update_counts,
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
        proposer_stats: Optional[Dict[str, float]],
        generator_stats: Optional[Dict[str, float]],
    ):
        if not self.is_main_process or self.wandb_run is None:
            return

        best = scored[best_idx]
        rewards = [float(c["total_reward"]) for c in scored]
        metrics: Dict[str, object] = {
            "train/step": step,
            "train/source_caption": source_caption,
            "train/spec_fallback_used": 1.0 if spec.fallback_used else 0.0,
            "train/spec_qa_count": float(len(spec.qa_pairs)),
            "train/candidate_reward_mean": sum(rewards) / max(1, len(rewards)),
            "train/candidate_reward_max": max(rewards) if rewards else 0.0,
            "train/candidate_reward_min": min(rewards) if rewards else 0.0,
            "train/best_spec_score": float(best["spec_score"]),
            "train/best_cycle_score": float(best["cycle_score"]),
            "train/best_diversity_score": float(best["diversity_score"]),
            "train/best_contradiction_score": float(best["contradiction_score"]),
            "train/generator_baseline": self.generator_baseline,
            "train/proposer_baseline": self.proposer_baseline,
            "kl/generator_beta": self.generator_updater.kl_coef,
            "kl/proposer_beta": self.proposer_updater.kl_coef,
            "text/prompt": spec.prompt,
            "text/proposer_raw": spec.raw_output,
            "text/best_cycle_caption": best.get("cycle_caption", ""),
        }
        if image_path:
            metrics["data/image_path"] = image_path

        if proposer_stats:
            metrics.update(
                {
                    "proposer/ce_loss": proposer_stats.get("ce_loss"),
                    "proposer/kl_loss": proposer_stats.get("kl_loss"),
                    "proposer/advantage": proposer_stats.get("advantage"),
                }
            )

        if generator_stats:
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
        source_caption = self._caption_image(image)
        spec = self._propose_generation_spec(image)
        if spec.fallback_used and source_caption:
            # Ground the fallback prompt to source content.
            spec = GenerationSpec(
                prompt=f"Create an image variation of: {source_caption}",
                qa_pairs=spec.qa_pairs,
                raw_output=spec.raw_output,
                fallback_used=True,
            )

        candidates = [self._generate_image_candidate(spec.prompt) for _ in range(self.cfg.num_generations)]
        scored = self._score_candidates(prompt=spec.prompt, qa_pairs=spec.qa_pairs, candidates=candidates)
        best_idx = max(range(len(scored)), key=lambda i: float(scored[i]["total_reward"]))
        best = scored[best_idx]

        proposer_stats = None
        generator_stats = None

        if step % self.cfg.generator_update_freq == 0:
            baseline_before = self.generator_baseline
            completion = str(best.get("policy_completion", ""))
            if self.cfg.strict_require_generation_tokens and not completion.strip():
                raise RuntimeError(
                    "No generation token trace was returned by the model backend. "
                    "Set --strict_require_generation_tokens false or use a backend exposing token traces."
                )
            generator_stats = self.generator_updater.step(
                prompt=str(best.get("policy_prompt", spec.prompt)),
                completion=completion,
                reward=float(best["total_reward"]),
                baseline=baseline_before,
                device=self.device,
            )
            self._policy_update_counts["generator"] += 1
            self._update_baseline("generator", float(best["total_reward"]))
            self._sync_state_scalars()

            self._append_jsonl(
                self.policy_updates_log_path,
                {
                    "step": step,
                    "role": "generator",
                    "reward": float(best["total_reward"]),
                    "baseline_before": baseline_before,
                    "baseline_after": self.generator_baseline,
                    "stats": generator_stats,
                    "candidate_idx": int(best_idx),
                },
            )

        proposer_reward = float(sum(float(c["total_reward"]) for c in scored) / max(1, len(scored)))
        if step % self.cfg.proposer_update_freq == 0:
            baseline_before = self.proposer_baseline
            proposer_stats = self.proposer_updater.step(
                image=image,
                prompt=GEN_PROMPT_TEMPLATE,
                completion=spec.raw_output,
                reward=proposer_reward,
                baseline=baseline_before,
                device=self.device,
            )
            self._policy_update_counts["proposer"] += 1
            self._update_baseline("proposer", proposer_reward)
            self._sync_state_scalars()

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

        self._save_candidate_images(step=step, scored=scored, best_idx=best_idx)

        self._append_jsonl(
            self.prompts_log_path,
            {
                "step": step,
                "image_path": meta.get("path"),
                "source_caption": source_caption,
                "prompt": spec.prompt,
                "qa_pairs": [dataclasses.asdict(qa) for qa in spec.qa_pairs],
                "fallback_used": spec.fallback_used,
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
                "candidate_rewards": [float(c["total_reward"]) for c in scored],
                "best_idx": int(best_idx),
                "best_reward": float(best["total_reward"]),
                "best_spec_score": float(best["spec_score"]),
                "best_cycle_score": float(best["cycle_score"]),
                "best_diversity_score": float(best["diversity_score"]),
                "best_contradiction_score": float(best["contradiction_score"]),
                "generator_baseline": self.generator_baseline,
                "proposer_baseline": self.proposer_baseline,
            },
        )

        return {
            "source_caption": source_caption,
            "spec": spec,
            "scored": scored,
            "best_idx": best_idx,
            "proposer_stats": proposer_stats,
            "generator_stats": generator_stats,
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
        for qa in qa_logs:
            solver_blob = qa.get("solver", {}) if isinstance(qa, dict) else {}
            rollouts = solver_blob.get("rollouts", []) if isinstance(solver_blob, dict) else []
            if not rollouts:
                continue
            completion = str(rollouts[0].get("completion", "")).strip()
            if not completion:
                continue

            question = str(qa.get("question", "")).strip()
            if not question:
                continue
            reward = float(qa.get("combined_score", 0.0))

            baseline_before = self.solver_baseline
            stats = self.solver_updater.step(
                image=image,
                prompt=build_solver_prompt(question),
                completion=completion,
                reward=reward,
                baseline=baseline_before,
                device=self.device,
            )
            self._policy_update_counts["solver"] += 1
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
                        "generator_kl_coef": self.generator_updater.kl_coef,
                        "proposer_kl_coef": self.proposer_updater.kl_coef,
                        "solver_kl_coef": self.solver_updater.kl_coef if self.solver_updater is not None else None,
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
                    proposer_stats=out["proposer_stats"],
                    generator_stats=out["generator_stats"],
                )

                self._update_metric("reward_mean", reward_mean_g)
                self._update_metric("reward_max", reward_max_g)
                self._update_metric("reward_min", reward_min_g)
                self._update_metric("best_spec_score", best_spec_g)
                self._update_metric("best_cycle_score", best_cycle_g)
                self._update_metric("best_diversity_score", best_div_g)
                self._update_metric("best_contradiction_score", best_contra_g)
                self._update_metric("generator_kl_coef", float(self.generator_updater.kl_coef))
                self._update_metric("proposer_kl_coef", float(self.proposer_updater.kl_coef))
                self._update_metric("step_duration_sec", step_duration_g)
                self._update_metric("spec_fallback_used", 1.0 if spec.fallback_used else 0.0)

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
                                "python self_evolving/run_experiment.py "
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


@dataclass
class UnifiedSelfEvolvingConfig(GenerationSelfEvolvingConfig):
    experiment_name: str = "unified_self_evolving"
    understanding_steps_per_cycle: int = 3
    generation_steps_per_cycle: int = 2
    synthetic_solver_update_freq: int = 1


class UnifiedSelfEvolvingTrainer(GenerationSelfEvolvingTrainer):
    def __init__(self, config: UnifiedSelfEvolvingConfig):
        if config.enable_solver_updates is False:
            config.enable_solver_updates = True
        if config.solver_update_freq <= 0:
            config.solver_update_freq = max(1, config.synthetic_solver_update_freq)
        super().__init__(config)
        self.ucfg = config

    def _understanding_step(self, step: int, image: Image.Image, meta: Dict) -> Dict[str, object]:
        proposer_prompt = build_proposer_prompt()
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

        solver_prompt = build_solver_prompt(question)
        solver_outputs: List[str] = []
        solver_answers_raw: List[str] = []
        solver_answers_norm: List[str] = []
        pre_words: List[int] = []

        for _ in range(self.cfg.num_solver_samples):
            solver_out = self._generate(
                image=image,
                prompt=solver_prompt,
                adapter_name="default" if self.cfg.use_lora else None,
                max_new_tokens=self.cfg.max_new_tokens_solver,
                temperature=self.cfg.temp,
                top_p=self.cfg.top_p,
            )
            answer_raw = _parse_answer(solver_out)
            solver_outputs.append(solver_out)
            solver_answers_raw.append(answer_raw)
            solver_answers_norm.append(normalize_answer(answer_raw))
            pre_words.append(pre_answer_word_count(solver_out))

        maj_answer, maj_count = majority_vote(solver_answers_norm)
        maj_frac = maj_count / float(self.cfg.num_solver_samples)
        hist: Dict[str, int] = {}
        for ans in solver_answers_norm:
            hist[ans] = hist.get(ans, 0) + 1
        probs = [count / float(self.cfg.num_solver_samples) for count in hist.values()]
        entropy_nats = shannon_entropy_nats(probs)

        solver_rewards_raw = [1.0 if ans == maj_answer else 0.0 for ans in solver_answers_norm]
        target_w = max(1, self.cfg.len_penalty_target_words)
        penalties = [min(1.0, max(0.0, (w - target_w) / float(target_w))) for w in pre_words]
        prob_map = {ans: count / float(self.cfg.num_solver_samples) for ans, count in hist.items()}
        solver_probs = [prob_map[ans] for ans in solver_answers_norm]
        solver_rewards_soft = [
            (prob ** self.cfg.solver_soft_gamma) * (1.0 - self.cfg.len_penalty_weight * pen)
            for prob, pen in zip(solver_probs, penalties)
        ]
        proposer_reward = gaussian_reward(entropy_nats, self.cfg.prop_entropy_mu, self.cfg.prop_entropy_sigma)

        solver_stats_list = []
        if self.solver_updater is not None:
            for completion, reward in zip(solver_outputs, solver_rewards_soft):
                baseline_before = self.solver_baseline
                stats = self.solver_updater.step(
                    image=image,
                    prompt=solver_prompt,
                    completion=completion,
                    reward=reward,
                    baseline=baseline_before,
                    device=self.device,
                )
                solver_stats_list.append(stats)
                self._policy_update_counts["solver"] += 1
                self._update_baseline("solver", reward)
                self._sync_state_scalars()

        proposer_stats = None
        if step % self.cfg.proposer_update_freq == 0:
            baseline_before = self.proposer_baseline
            proposer_stats = self.proposer_updater.step(
                image=image,
                prompt=proposer_prompt,
                completion=proposer_out,
                reward=proposer_reward,
                baseline=baseline_before,
                device=self.device,
            )
            self._policy_update_counts["proposer"] += 1
            self._update_baseline("proposer", proposer_reward)
            self._sync_state_scalars()

        record = {
            "step": step,
            "phase": "understanding",
            "image_path": meta.get("path"),
            "question": question,
            "proposer_out": proposer_out,
            "solver_answers_raw": solver_answers_raw,
            "solver_answers_norm": solver_answers_norm,
            "solver_rewards_raw": solver_rewards_raw,
            "solver_rewards_soft": solver_rewards_soft,
            "majority_answer": maj_answer,
            "majority_count": maj_count,
            "majority_fraction": maj_frac,
            "entropy_nats": entropy_nats,
            "proposer_reward": proposer_reward,
            "solver_baseline": self.solver_baseline,
            "proposer_baseline": self.proposer_baseline,
            "solver_stats": solver_stats_list,
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
                "entropy_nats": entropy_nats,
                "solver_reward_soft_mean": sum(solver_rewards_soft) / max(1, len(solver_rewards_soft)),
                "proposer_reward": proposer_reward,
            },
        )

        if self.is_main_process and step % self.cfg.log_every == 0:
            print(
                f"[Step {step:05d}][U] maj={maj_count}/{self.cfg.num_solver_samples} "
                f"maj_frac={maj_frac:.2f} H={entropy_nats:.3f} P_R={proposer_reward:.3f}"
            )
            print(f"  Q: {question}")

        self._update_metric("u_majority_fraction", self._dist_mean(maj_frac))
        self._update_metric("u_entropy_nats", self._dist_mean(entropy_nats))
        self._update_metric("u_proposer_reward", self._dist_mean(proposer_reward))

        return record

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
            print(f"[Unified] Images: {len(self.pool)}")
            print(f"[Unified] Step range: {self.start_step + 1}..{cfg.total_steps}")
            print(
                f"[Unified] Schedule: Ux{cfg.understanding_steps_per_cycle} + Gx{cfg.generation_steps_per_cycle} (cycle={cycle})"
            )

        last_completed_step = self.start_step
        last_attempted_step = self.start_step
        try:
            for step in range(self.start_step + 1, cfg.total_steps + 1):
                last_attempted_step = step
                image, meta = self._sample_image_for_step(step)

                phase_idx = (step - 1) % cycle
                if phase_idx < cfg.understanding_steps_per_cycle:
                    self._understanding_step(step=step, image=image, meta=meta)
                else:
                    out = self._generation_step(step=step, image=image, meta=meta)
                    spec: GenerationSpec = out["spec"]
                    scored: List[Dict[str, object]] = out["scored"]
                    best_idx = int(out["best_idx"])
                    if cfg.synthetic_solver_update_freq > 0 and step % cfg.synthetic_solver_update_freq == 0:
                        self._solver_synthetic_update_from_best(step, scored[best_idx])

                    self._append_jsonl(
                        self.iter_log_path,
                        {
                            "step": step,
                            "phase": "generation",
                            "image_path": meta.get("path"),
                            "prompt": spec.prompt,
                            "best_idx": best_idx,
                            "best_reward": float(scored[best_idx]["total_reward"]),
                            "generator_baseline": self.generator_baseline,
                            "proposer_baseline": self.proposer_baseline,
                            "solver_baseline": self.solver_baseline,
                        },
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
