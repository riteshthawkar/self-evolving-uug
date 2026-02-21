"""
Self-evolving proposer-solver-generator training framework for VARGPT.

Ported from the BLIP3o implementation to demonstrate model-agnostic
generality across diffusion-based (BLIP3o) and autoregressive discrete-token
(VARGPT v1.1) unified understanding-generation architectures.
"""

from .workflow import run_self_evolving


__all__ = ["run_self_evolving"]
