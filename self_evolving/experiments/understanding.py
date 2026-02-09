"""
Understanding-only self-evolving experiment.

This module ports the working EvoLMM-style understanding loop into the
unified self_evolving codebase:
- Proposer generates image-grounded questions
- Solver answers each question multiple times
- Continuous self-rewards are computed from solver agreement/entropy
- Both roles are optimized via KL-regularized REINFORCE with adaptive beta
"""

import contextlib
import dataclasses
import datetime as dt
import gc
import importlib.util
import json
import math
import os
import pathlib
import random
import re
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen2VLForConditionalGeneration
except Exception:
    Qwen2VLForConditionalGeneration = None

from self_evolving.data.image_pool import ImagePool, ImagePoolConfig

try:
    from peft import LoraConfig, TaskType, get_peft_model

    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

try:
    import wandb

    HAS_WANDB = True
except Exception:
    HAS_WANDB = False


DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "mm_projector",
)


def _safe_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if dtype == "float16" and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _resolve_attn_implementation(requested: str) -> Optional[str]:
    choice = (requested or "auto").strip().lower()
    if choice in {"none", "off", "disable", "disabled"}:
        return None
    if choice in {"sdpa", "eager", "flash_attention_2"}:
        return choice
    if choice != "auto":
        return None

    if not torch.cuda.is_available():
        return None
    if getattr(torch.version, "hip", None):
        # On ROCm, SDPA is the most stable default backend.
        return "sdpa"
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def strip_tags(text: str, tag: str) -> Optional[str]:
    lt = f"<{tag}>"
    rt = f"</{tag}>"
    if lt in text and rt in text:
        return text.split(lt, 1)[1].split(rt, 1)[0].strip()
    return None


def normalize_answer(ans: str) -> str:
    s = ans.strip().lower()
    s = s.replace(",", " ")
    s = " ".join(s.split())
    return s.strip(" .,:;!?\"'")


def majority_vote(answers: List[str]) -> Tuple[str, int]:
    counts: Dict[str, int] = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    return max(counts.items(), key=lambda x: x[1])


def shannon_entropy_nats(probs: List[float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(max(p, eps)) for p in probs if p > 0.0)


def pre_answer_word_count(text: str) -> int:
    idx = text.lower().find("<answer>")
    prefix = text if idx == -1 else text[:idx]
    return len(prefix.strip().split())


def gaussian_reward(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return math.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma))


def _parse_first_question(text: str) -> str:
    tagged = strip_tags(text, "question")
    if tagged:
        return tagged
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines:
        line = re.sub(r"^\d+[\).\-\s]*", "", line).strip()
        if line.endswith("?"):
            return line
    if lines:
        return lines[0]
    return ""


def _parse_answer(text: str) -> str:
    tagged = strip_tags(text, "answer")
    if tagged:
        return tagged
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else "unknown"


def build_proposer_prompt() -> str:
    return (
        "You are a Question Proposer.\n"
        "Given the image, generate exactly one short-answer question that can be answered from the image alone.\n"
        "Rules:\n"
        "- Avoid ambiguity and external knowledge.\n"
        "- Keep the question concise and specific.\n"
        "- Output XML only:\n"
        "<question>...</question>\n"
        "<rationale>...</rationale>"
    )


def build_solver_prompt(question_text: str) -> str:
    return (
        "You are a precise vision-language solver.\n"
        "Answer the question using only the provided image.\n"
        "Return only the final answer inside XML:\n"
        "<answer>...</answer>\n"
        f"Question: {question_text}"
    )


def _set_global_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _json_dump(path: pathlib.Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _collect_git_info(repo_root: pathlib.Path) -> Dict[str, Optional[str]]:
    def run_git(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(
                ["git"] + args,
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out
        except Exception:
            return None

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "is_dirty": run_git(["status", "--porcelain"]) not in (None, ""),
    }


def _infer_primary_device(model: torch.nn.Module, fallback_cuda_device: int) -> torch.device:
    model_ref = _unwrap_model(model)
    hf_device_map = getattr(model_ref, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        cuda_devs = [value for value in hf_device_map.values() if isinstance(value, str) and value.startswith("cuda")]
        if cuda_devs:
            try:
                idx = min(int(item.split(":")[1]) for item in cuda_devs)
                return torch.device(f"cuda:{idx}")
            except Exception:
                pass
    if torch.cuda.is_available():
        return torch.device(f"cuda:{fallback_cuda_device}")
    return torch.device("cpu")


@contextlib.contextmanager
def use_adapter(model: torch.nn.Module, adapter_name: Optional[str]):
    model_ref = _unwrap_model(model)
    if not hasattr(model_ref, "set_adapter"):
        yield
        return

    if adapter_name is None and hasattr(model_ref, "disable_adapter"):
        with model_ref.disable_adapter():
            yield
        return

    prev_adapter = getattr(model_ref, "active_adapter", None)
    switched = False
    if adapter_name is not None:
        try:
            model_ref.set_adapter(adapter_name)
            switched = True
        except Exception:
            switched = False
    try:
        yield
    finally:
        if switched and prev_adapter is not None:
            try:
                model_ref.set_adapter(prev_adapter)
            except Exception:
                pass


def _decode_tokens(processor, token_ids: torch.Tensor) -> str:
    if hasattr(processor, "decode"):
        return processor.decode(token_ids, skip_special_tokens=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor does not expose decode/tokenizer.decode")
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _build_chat_text(processor, image: Image.Image, prompt: str) -> str:
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _prepare_mm_inputs(processor, device: torch.device, image: Image.Image, chat_text: str):
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt", padding=True)
    return inputs.to(device)


def _clip_grad_norm_multi_device(params: Iterable[torch.nn.Parameter], max_norm: float):
    grouped: Dict[torch.device, List[torch.nn.Parameter]] = {}
    for p in params:
        if p.grad is None:
            continue
        grouped.setdefault(p.grad.device, []).append(p)
    for group in grouped.values():
        torch.nn.utils.clip_grad_norm_(group, max_norm)


def _load_model_with_fallback(
    model_name: str,
    *,
    torch_dtype: torch.dtype,
    device_map,
    trust_remote_code: bool,
    attn_implementation: Optional[str] = None,
):
    errors: Dict[str, str] = {}

    model_classes: List[type] = []
    model_name_lower = (model_name or "").lower()

    # Prefer explicit model classes for Qwen VL where available.
    if "qwen2.5-vl" in model_name_lower and Qwen2_5_VLForConditionalGeneration is not None:
        model_classes.append(Qwen2_5_VLForConditionalGeneration)
    if "qwen2-vl" in model_name_lower and Qwen2VLForConditionalGeneration is not None:
        model_classes.append(Qwen2VLForConditionalGeneration)

    # Generic multimodal auto classes across transformer versions.
    for cls in (AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM):
        if cls is not None and cls not in model_classes:
            model_classes.append(cls)

    def _try_load(model_cls):
        base_kwargs = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        attn_attempts = [attn_implementation] if attn_implementation else [None]
        if attn_implementation is not None:
            attn_attempts.append(None)
        seen = set()
        for attn_value in attn_attempts:
            key = attn_value or "__none__"
            if key in seen:
                continue
            seen.add(key)
            kwargs = dict(base_kwargs)
            if attn_value is not None:
                kwargs["attn_implementation"] = attn_value

            for dtype_key in ("dtype", "torch_dtype"):
                try:
                    return model_cls.from_pretrained(
                        model_name,
                        **kwargs,
                        **{dtype_key: torch_dtype},
                    )
                except TypeError:
                    continue
                except Exception as exc:
                    errors[f"{model_cls.__name__}[attn={attn_value or 'none'}|{dtype_key}]"] = repr(exc)
                    break
        return None

    for cls in model_classes:
        loaded = _try_load(cls)
        if loaded is not None:
            return loaded

    details = "; ".join(f"{name}: {err}" for name, err in errors.items())
    raise RuntimeError(f"Failed to load model '{model_name}' with supported loaders. {details}")


def _collect_trainable_params(
    model: torch.nn.Module,
    adapter_name: Optional[str],
) -> Iterable[torch.nn.Parameter]:
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if adapter_name is None:
        return [p for _, p in trainable]

    selected = [p for n, p in trainable if (f".{adapter_name}." in n) or (f"{adapter_name}." in n)]
    if not selected:
        preview = [name for name, _ in trainable[:20]]
        raise RuntimeError(
            f"No trainable parameters matched adapter '{adapter_name}'. "
            f"Trainable preview: {preview}"
        )
    return selected


class RolePolicyUpdater:
    """
    KL-regularized REINFORCE updater for a role adapter.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor,
        config: "UnderstandingSelfEvolvingConfig",
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
        image: Image.Image,
        prompt: str,
        completion: str,
        reward: float,
        baseline: float,
        device: torch.device,
    ) -> Dict[str, float]:
        self.step_id += 1

        chat_prompt = _build_chat_text(self.processor, image, prompt)
        chat_full = chat_prompt + completion

        inputs_prompt = _prepare_mm_inputs(self.processor, device, image, chat_prompt)
        inputs_full = _prepare_mm_inputs(self.processor, device, image, chat_full)

        input_ids = inputs_full["input_ids"]
        attention_mask = inputs_full.get("attention_mask")
        labels = input_ids.clone()
        prompt_len = inputs_prompt["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        valid_mask = labels[:, 1:] != -100

        self.model.train(True)
        policy_inputs = dict(inputs_full)
        policy_inputs["labels"] = labels
        # Avoid allocating KV cache during training forwards; this materially lowers
        # memory usage for long multimodal sequences.
        policy_inputs["use_cache"] = False
        with use_adapter(self.model, self.adapter_name):
            out_pi = self.model(**policy_inputs)
        ce_loss = out_pi.loss
        logp_pi = F.log_softmax(out_pi.logits, dim=-1)

        ref_inputs = dict(inputs_full)
        ref_inputs["use_cache"] = False
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
            del inputs_prompt, inputs_full, input_ids, attention_mask, labels, policy_inputs
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
class UnderstandingSelfEvolvingConfig:
    # Experiment identity
    experiment_name: str = "understanding_self_evolving"
    run_name: Optional[str] = None
    output_dir: str = "./runs"

    # Data
    data_dir: str = ""
    data_split: str = "all"  # train|val|test|all
    include_subfolders: Optional[Tuple[str, ...]] = None
    max_images: Optional[int] = None

    # Model
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    dtype: str = "bfloat16"
    cuda_device: int = 0
    device_map: str = "single"  # single|auto|cpu
    attn_implementation: str = "auto"  # auto|sdpa|eager|flash_attention_2|none

    # Optimization
    total_steps: int = 100
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    proposer_update_freq: int = 5

    # Decoding
    temp: float = 1.0
    top_p: float = 1.0
    max_new_tokens_solver: int = 128
    max_new_tokens_proposer: int = 128
    num_solver_samples: int = 5

    # Reward shaping
    solver_soft_gamma: float = 0.7
    len_penalty_weight: float = 0.10
    len_penalty_target_words: int = 6
    prop_entropy_mu: float = 0.90
    prop_entropy_sigma: float = 0.35

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

    # W&B
    wandb_mode: str = "disabled"  # online|offline|disabled
    wandb_project: str = "self-evolving-uug"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_images_every: int = 0

    # Resume
    resume_from: Optional[str] = None
    start_step: int = 0


class UnderstandingSelfEvolvingTrainer:
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

    def _sync_state_scalars(self):
        if not (self.distributed and dist.is_initialized()):
            return
        self.solver_baseline = self._dist_mean(self.solver_baseline)
        self.proposer_baseline = self._dist_mean(self.proposer_baseline)
        self.solver_updater.kl_coef = self._dist_mean(self.solver_updater.kl_coef)
        self.proposer_updater.kl_coef = self._dist_mean(self.proposer_updater.kl_coef)

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
        self.start_step = max(0, int(config.start_step))

        self._metric_stats: Dict[str, Dict[str, float]] = {}
        self._policy_update_counts: Dict[str, int] = {"solver": 0, "proposer": 0}
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

        # Accept either step_XXXXX directory, direct trainer_state.pt directory, or run directory.
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

        state_path = resume_dir / "trainer_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"trainer_state.pt not found in resume checkpoint: {resume_dir}")

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
        checkpoints = [p for p in self.run_dir.glob("step_*") if self._is_complete_checkpoint(p)]
        return sorted(checkpoints, key=lambda p: p.name)

    def _load_model(self):
        if self.cfg.use_lora and not HAS_PEFT:
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
                f"[Understanding] Load options: dtype={dtype}, device_map={device_map}, "
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

    def _update_baseline(self, which: str, reward: float):
        m = self.cfg.baseline_momentum
        if which == "solver":
            self.solver_baseline = m * self.solver_baseline + (1 - m) * reward
        else:
            self.proposer_baseline = m * self.proposer_baseline + (1 - m) * reward

    def _append_iter_record(self, record: Dict):
        self._append_jsonl(self.iter_log_path, record)

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

                solver_prompt = build_solver_prompt(question)
                solver_outputs: List[str] = []
                solver_answers_raw: List[str] = []
                solver_answers_norm: List[str] = []
                pre_words: List[int] = []

                for _ in range(cfg.num_solver_samples):
                    solver_out = self._generate(
                        image=image,
                        prompt=solver_prompt,
                        adapter_name="default" if cfg.use_lora else None,
                        max_new_tokens=cfg.max_new_tokens_solver,
                        temperature=cfg.temp,
                        top_p=cfg.top_p,
                    )
                    answer_raw = _parse_answer(solver_out)
                    solver_outputs.append(solver_out)
                    solver_answers_raw.append(answer_raw)
                    solver_answers_norm.append(normalize_answer(answer_raw))
                    pre_words.append(pre_answer_word_count(solver_out))

                maj_answer, maj_count = majority_vote(solver_answers_norm)
                maj_frac = maj_count / float(cfg.num_solver_samples)
                hist: Dict[str, int] = {}
                for ans in solver_answers_norm:
                    hist[ans] = hist.get(ans, 0) + 1
                probs = [count / float(cfg.num_solver_samples) for count in hist.values()]
                entropy_nats = shannon_entropy_nats(probs)

                solver_rewards_raw = [1.0 if ans == maj_answer else 0.0 for ans in solver_answers_norm]
                target_w = max(1, cfg.len_penalty_target_words)
                penalties = [min(1.0, max(0.0, (w - target_w) / float(target_w))) for w in pre_words]
                prob_map = {ans: count / float(cfg.num_solver_samples) for ans, count in hist.items()}
                solver_probs = [prob_map[ans] for ans in solver_answers_norm]
                solver_rewards_soft = [
                    (prob ** cfg.solver_soft_gamma) * (1.0 - cfg.len_penalty_weight * pen)
                    for prob, pen in zip(solver_probs, penalties)
                ]
                proposer_reward = gaussian_reward(entropy_nats, cfg.prop_entropy_mu, cfg.prop_entropy_sigma)

                solver_baseline_before_step = self.solver_baseline
                solver_step_stats = []
                for sample_idx, (completion, reward, reward_raw, answer_raw, answer_norm, prob, penalty, words) in enumerate(
                    zip(
                        solver_outputs,
                        solver_rewards_soft,
                        solver_rewards_raw,
                        solver_answers_raw,
                        solver_answers_norm,
                        solver_probs,
                        penalties,
                        pre_words,
                    ),
                    start=1,
                ):
                    baseline_before = self.solver_baseline
                    stats = self.solver_updater.step(
                        image=image,
                        prompt=solver_prompt,
                        completion=completion,
                        reward=reward,
                        baseline=baseline_before,
                        device=self.device,
                    )
                    solver_step_stats.append(stats)
                    self._policy_update_counts["solver"] += 1
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
                            "reward_raw": reward_raw,
                            "reward_soft": reward,
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
                solver_baseline_after_step = self.solver_baseline

                proposer_baseline_before_step = self.proposer_baseline
                proposer_baseline_after_step = proposer_baseline_before_step
                proposer_stats = None
                if step % cfg.proposer_update_freq == 0:
                    proposer_stats = self.proposer_updater.step(
                        image=image,
                        prompt=proposer_prompt,
                        completion=proposer_out,
                        reward=proposer_reward,
                        baseline=proposer_baseline_before_step,
                        device=self.device,
                    )
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
                    self._update_baseline("proposer", proposer_reward)
                    self._sync_state_scalars()
                    proposer_baseline_after_step = self.proposer_baseline

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

                solver_ce_mean = sum(s["ce_loss"] for s in solver_step_stats) / max(1, len(solver_step_stats))
                solver_kl_mean = sum(s["kl_loss"] for s in solver_step_stats) / max(1, len(solver_step_stats))
                solver_adv_mean = sum(s["advantage"] for s in solver_step_stats) / max(1, len(solver_step_stats))
                solver_stats_mean = {
                    "ce_loss_mean": solver_ce_mean,
                    "kl_loss_mean": solver_kl_mean,
                    "advantage_mean": solver_adv_mean,
                }

                if self.is_main_process and step % cfg.log_every == 0:
                    print(
                        f"[Step {step:05d}] maj={maj_count}/{cfg.num_solver_samples} "
                        f"maj_frac={maj_frac_global:.2f} H={entropy_nats_global:.3f} "
                        f"P_R={proposer_reward_global:.3f} "
                        f"S_R_raw={solver_raw_mean_global:.3f} S_R_soft={solver_soft_mean_global:.3f} "
                        f"pre_words={pre_words_mean_global:.2f}"
                    )
                    print(f"  Q: {question}")
                    print(f"  A: [{', '.join(solver_answers_raw)}] | MAJ: {maj_answer}")

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
                        "entropy_nats": entropy_nats,
                        "solver_rewards_raw": solver_rewards_raw,
                        "solver_rewards_soft": solver_rewards_soft,
                        "solver_rewards_raw_mean": solver_raw_mean,
                        "solver_rewards_soft_mean": solver_soft_mean,
                        "proposer_reward": proposer_reward,
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
                        "solver_probs": solver_probs,
                        "solver_len_penalties": penalties,
                        "pre_answer_word_counts": pre_words,
                        "majority_answer": maj_answer,
                        "majority_count": maj_count,
                        "majority_fraction": maj_frac,
                        "entropy_nats": entropy_nats,
                        "proposer_reward": proposer_reward,
                        "solver_baseline_before": solver_baseline_before_step,
                        "solver_baseline_after": solver_baseline_after_step,
                        "proposer_baseline_before": proposer_baseline_before_step,
                        "proposer_baseline_after": proposer_baseline_after_step,
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

            if cfg.save_every <= 0 or (cfg.total_steps % cfg.save_every) != 0:
                self._dist_barrier()
                self._save_checkpoint(cfg.total_steps)
                self._dist_barrier()
            self._write_ablation_summary(cfg.total_steps, status="completed")
            if self.is_main_process:
                print(f"[Understanding] Training complete. Final checkpoint at step {cfg.total_steps:05d}.")
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            interrupted_step = int(last_attempted_step)
            tb = traceback.format_exc()
            if self.is_main_process:
                print(f"[Understanding] Training interrupted at step {interrupted_step}: {error_text}")
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
                    print(f"[Understanding] Emergency checkpoint saved at step {emergency_step:05d}.")
                    _json_dump(
                        self.run_dir / "resume_hint.json",
                        {
                            "resume_from": str(self.run_dir / f"step_{emergency_step:05d}"),
                            "start_step": emergency_step,
                            "total_steps": cfg.total_steps,
                            "command_example": (
                                "python self_evolving/run_experiment.py "
                                f"--experiment understanding_self_evolving --data_dir {cfg.data_dir} "
                                f"--output_dir {cfg.output_dir} --run_name {self.run_dir.name} "
                                f"--resume_from {self.run_dir / f'step_{emergency_step:05d}'} "
                                f"--start_step {emergency_step} --total_steps {cfg.total_steps}"
                            ),
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
