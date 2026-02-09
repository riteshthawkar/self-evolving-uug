# Self-Evolving BLIP3o-NEXT
# EvoLMM-style fully unsupervised self-evolution on BLIP3o-NEXT

from .data.image_pool import ImagePool, ImagePoolConfig
from .roles.proposer import ProposerRole, VerificationSpec
from .roles.solver import SolverRole
from .roles.generator import GeneratorRole
from .judge import FrozenJudge
from .grpo_trainer import SelfEvolvingGRPOTrainer, InternalRewardConfig
from .rl_controller import RLController, RLControllerConfig, EvoLMMReward
from .experiments import (
    GenerationSelfEvolvingConfig,
    GenerationSelfEvolvingTrainer,
    UnderstandingSelfEvolvingConfig,
    UnderstandingSelfEvolvingTrainer,
    UnifiedSelfEvolvingConfig,
    UnifiedSelfEvolvingTrainer,
)

__version__ = "0.1.0"
