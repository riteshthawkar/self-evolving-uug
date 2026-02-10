"""
Self-evolving training pipeline for BLIP3o.

This package implements the EvoLMM self-evolving training loop
(Proposer → Solver → Reward → Update) with native BLIP3o model loading.

Experiment modes:
- understanding_self_evolving: VQA-based understanding training
- generation_self_evolving: Image generation training with cycle-consistency
- unified_self_evolving: Alternating understanding + generation

Usage::

    from blip3o.train.self_evolving import (
        UnderstandingSelfEvolvingConfig,
        UnderstandingSelfEvolvingTrainer,
        GenerationSelfEvolvingConfig,
        GenerationSelfEvolvingTrainer,
        UnifiedSelfEvolvingConfig,
        UnifiedSelfEvolvingTrainer,
    )
"""

from .config import (
    GenerationSelfEvolvingConfig,
    UnderstandingSelfEvolvingConfig,
    UnifiedSelfEvolvingConfig,
)
from .understanding_trainer import UnderstandingSelfEvolvingTrainer

# Generation and unified trainers have heavier dependencies; import lazily
# to avoid import-time failures when diffusion/generation backends are
# unavailable (e.g., understanding-only environments).


def _get_generation_trainer():
    from .generation_trainer import GenerationSelfEvolvingTrainer
    return GenerationSelfEvolvingTrainer


def _get_unified_trainer():
    from .unified_trainer import UnifiedSelfEvolvingTrainer
    return UnifiedSelfEvolvingTrainer


__all__ = [
    "UnderstandingSelfEvolvingConfig",
    "UnderstandingSelfEvolvingTrainer",
    "GenerationSelfEvolvingConfig",
    "GenerationSelfEvolvingTrainer",
    "UnifiedSelfEvolvingConfig",
    "UnifiedSelfEvolvingTrainer",
]


def __getattr__(name):
    if name == "GenerationSelfEvolvingTrainer":
        return _get_generation_trainer()
    if name == "UnifiedSelfEvolvingTrainer":
        return _get_unified_trainer()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
