"""
Multi-adapter LoRA management for the self-evolving framework on VARGPT.

LLaMA-Factory's adapter.py (line 166) merges all-but-last adapters and only
trains the last one. The self-evolving framework needs 3 simultaneous adapters
(proposer, solver, generator) with runtime switching.

This module provides:
  - setup_multi_adapter(): After initial model loading, adds proposer + solver
    adapters alongside the default generator adapter.
  - use_role(): Context manager that switches the active adapter AND manages
    trainability of vargpt_gen / image_gen_projector params.
  - collect_role_params(): Returns the trainable parameters for a given role.

Key constraint: vargpt_gen and image_gen_projector are always-trainable in the
base adapter.py. For self-evolving, we control their requires_grad per-role:
  - generator role: vargpt_gen + image_gen_projector trainable
  - proposer/solver roles: vargpt_gen + image_gen_projector frozen
"""

import contextlib
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .utils import _unwrap_model


logger = logging.getLogger(__name__)


# ── Module-level constants ───────────────────────────────────────────────

# Modules that should ONLY be trainable during generator role updates.
# These match the unconditional requires_grad=True in adapter.py lines 252-258.
GENERATOR_ONLY_MODULES = ("vargpt_gen", "image_gen_projector")

# Adapter role names
ROLE_PROPOSER = "proposer"
ROLE_SOLVER = "solver"
ROLE_GENERATOR = "default"  # The initial adapter from load_model()

ALL_ROLES = (ROLE_PROPOSER, ROLE_SOLVER, ROLE_GENERATOR)


# ── Setup ────────────────────────────────────────────────────────────────


def setup_multi_adapter(
    model: nn.Module,
    finetuning_args,
    se_config,
) -> nn.Module:
    """After ``load_model()`` applies a single LoRA adapter, add two more
    named adapters for proposer and solver roles.

    The initial adapter becomes the **generator** role (name = "default").
    We add "proposer" and "solver" with the same LoRA config.

    Parameters
    ----------
    model : nn.Module
        Model after ``load_model()`` has applied the initial LoRA adapter.
    finetuning_args : FinetuningArguments
        Contains lora_rank, lora_alpha, lora_dropout, lora_target, etc.
    se_config : SelfEvolvingConfig
        Self-evolving training config.

    Returns
    -------
    nn.Module
        The model with all three adapters added.
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError:
        raise ImportError(
            "PEFT is required for multi-adapter setup. "
            "Install it with: pip install peft"
        )

    base_model = _unwrap_model(model)

    # Verify initial adapter exists
    if not hasattr(base_model, "peft_config"):
        raise RuntimeError(
            "Model does not have PEFT adapters. Ensure finetuning_type='lora' "
            "is set and load_model() was called first."
        )

    existing_adapters = list(base_model.peft_config.keys())
    logger.info(f"[AdapterManager] Existing adapters after load_model: {existing_adapters}")

    # The initial adapter name (typically "default")
    generator_adapter_name = existing_adapters[0] if existing_adapters else "default"
    logger.info(f"[AdapterManager] Generator adapter: {generator_adapter_name}")

    # Build LoRA config matching the existing adapter.
    # IMPORTANT: existing_config.target_modules may be a regex string
    # (from patch_target_modules in visual.py) like
    # "^(?!.*visual).*(?:q_proj|k_proj|...).*"
    # After PEFT wraps the first adapter, the module tree now contains
    # ModuleDict (for dropout) and LoRA-wrapped layers. Reusing the regex
    # causes PEFT's add_adapter() to match these wrapped modules and fail
    # with "ModuleDict is not supported".
    # Fix: use an explicit list of module short-names instead.
    existing_config = base_model.peft_config[generator_adapter_name]

    # Extract clean module names: if target_modules is a regex, parse out
    # the module names from the (?:...) group; otherwise use as-is.
    raw_targets = existing_config.target_modules
    if isinstance(raw_targets, str):
        import re
        match = re.search(r'\(\?:([^)]+)\)', raw_targets)
        if match:
            target_modules = list(set(match.group(1).split('|')))
        else:
            # Fallback: use the well-known LLM linear module names
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]
    elif isinstance(raw_targets, set):
        target_modules = list(raw_targets)
    else:
        target_modules = list(raw_targets) if raw_targets else [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    logger.info(f"[AdapterManager] target_modules for new adapters: {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=existing_config.r,
        lora_alpha=existing_config.lora_alpha,
        lora_dropout=existing_config.lora_dropout,
        target_modules=target_modules,
    )

    # Add proposer and solver adapters
    for role in (ROLE_PROPOSER, ROLE_SOLVER):
        if role not in existing_adapters:
            logger.info(f"[AdapterManager] Adding adapter: {role}")
            base_model.add_adapter(role, lora_config)
        else:
            logger.info(f"[AdapterManager] Adapter already exists: {role}")

    # Set the generator adapter as active by default
    base_model.set_adapter(generator_adapter_name)

    # Initially freeze vargpt_gen / image_gen_projector
    # (they'll be unfrozen only during generator role updates)
    _freeze_generator_only_modules(base_model)

    # Log adapter info
    all_adapters = list(base_model.peft_config.keys())
    logger.info(f"[AdapterManager] All adapters after setup: {all_adapters}")

    return model


# ── Per-Role Trainability ────────────────────────────────────────────────


def _freeze_generator_only_modules(model: nn.Module):
    """Freeze vargpt_gen and image_gen_projector params."""
    for name, param in model.named_parameters():
        if any(mod in name for mod in GENERATOR_ONLY_MODULES):
            param.requires_grad = False


def _unfreeze_generator_only_modules(model: nn.Module):
    """Unfreeze vargpt_gen and image_gen_projector params."""
    for name, param in model.named_parameters():
        if any(mod in name for mod in GENERATOR_ONLY_MODULES):
            param.requires_grad = True


@contextlib.contextmanager
def use_role(model: nn.Module, role: str):
    """Context manager to switch to a specific role's adapter.

    Handles:
      1. Switching the active LoRA adapter.
      2. Freezing/unfreezing vargpt_gen and image_gen_projector:
         - generator: unfrozen (these modules are part of image generation)
         - proposer/solver: frozen (these modules should not update)

    Parameters
    ----------
    model : nn.Module
        The model (possibly DDP-wrapped).
    role : str
        One of "proposer", "solver", "default" (generator).
    """
    base_model = _unwrap_model(model)

    # Save previous state
    prev_adapter = getattr(base_model, "active_adapter", None)
    # Handle PEFT returning list for active_adapter
    if isinstance(prev_adapter, (list, tuple)):
        prev_adapter = prev_adapter[0] if prev_adapter else None

    # Determine adapter name
    adapter_name = role if role != "generator" else ROLE_GENERATOR

    # Switch adapter
    try:
        base_model.set_adapter(adapter_name)
    except Exception as e:
        logger.warning(f"[AdapterManager] Failed to set adapter '{adapter_name}': {e}")

    # Manage generator-only module trainability
    is_generator = (role == ROLE_GENERATOR or role == "generator")
    if is_generator:
        _unfreeze_generator_only_modules(base_model)
    else:
        _freeze_generator_only_modules(base_model)

    try:
        yield
    finally:
        # Restore previous adapter
        if prev_adapter is not None:
            try:
                base_model.set_adapter(prev_adapter)
            except Exception:
                pass
        # Restore frozen state (conservative: freeze by default)
        _freeze_generator_only_modules(base_model)


@contextlib.contextmanager
def use_base_model(model: nn.Module):
    """Context manager to use the base model (no adapters) for reference
    log-prob computation.

    Disables all adapters and freezes generator-only modules.
    Restores the previous adapter on exit.
    """
    base_model = _unwrap_model(model)

    prev_adapter = getattr(base_model, "active_adapter", None)
    if isinstance(prev_adapter, (list, tuple)):
        prev_adapter = prev_adapter[0] if prev_adapter else None

    if hasattr(base_model, "disable_adapter"):
        ctx = base_model.disable_adapter()
    else:
        ctx = contextlib.nullcontext()

    _freeze_generator_only_modules(base_model)

    try:
        with ctx:
            yield
    finally:
        if prev_adapter is not None:
            try:
                base_model.set_adapter(prev_adapter)
            except Exception:
                pass


# ── Parameter Collection ─────────────────────────────────────────────────


def collect_role_params(
    model: nn.Module,
    role: str,
    include_generator_modules: bool = False,
) -> List[torch.nn.Parameter]:
    """Collect trainable parameters for a specific role.

    Parameters
    ----------
    model : nn.Module
        The model with PEFT adapters.
    role : str
        One of "proposer", "solver", "default" (generator).
    include_generator_modules : bool
        If True, also include vargpt_gen and image_gen_projector params.
        Should be True only for the generator role.

    Returns
    -------
    List of parameters for this role's optimizer.
    """
    base_model = _unwrap_model(model)
    adapter_name = role if role != "generator" else ROLE_GENERATOR

    params = []
    for name, param in base_model.named_parameters():
        # LoRA params for this adapter
        if f".{adapter_name}." in name or f"lora_{adapter_name}" in name:
            params.append(param)
            continue

        # For default adapter, also check without explicit adapter name
        # (PEFT may name it "default" or have no adapter name in path)
        if adapter_name == ROLE_GENERATOR:
            # Check for default adapter pattern
            if "lora_" in name and not any(
                f".{r}." in name for r in (ROLE_PROPOSER, ROLE_SOLVER)
            ):
                # This is a lora param that doesn't belong to proposer/solver
                # → it belongs to the default (generator) adapter
                if ".default." in name or (
                    f".{ROLE_PROPOSER}." not in name
                    and f".{ROLE_SOLVER}." not in name
                ):
                    params.append(param)
                    continue

        # Generator-only modules
        if include_generator_modules:
            if any(mod in name for mod in GENERATOR_ONLY_MODULES):
                params.append(param)

    # Deduplicate (by data_ptr)
    seen = set()
    unique_params = []
    for p in params:
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            unique_params.append(p)

    if not unique_params:
        all_names = [n for n, _ in base_model.named_parameters()]
        logger.warning(
            f"[AdapterManager] No params found for role '{role}'. "
            f"Available param names (first 10): {all_names[:10]}"
        )

    return unique_params


def get_role_optimizer(
    model: nn.Module,
    role: str,
    lr: float,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Create an AdamW optimizer for a specific role's parameters.

    The generator role includes vargpt_gen and image_gen_projector.
    Proposer and solver only include their LoRA params.
    """
    is_generator = (role == ROLE_GENERATOR or role == "generator")
    params = collect_role_params(
        model, role, include_generator_modules=is_generator
    )

    if not params:
        raise RuntimeError(
            f"No trainable parameters found for role '{role}'. "
            f"Cannot create optimizer."
        )

    logger.info(
        f"[AdapterManager] Creating optimizer for role '{role}': "
        f"{len(params)} parameters, lr={lr}"
    )

    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
