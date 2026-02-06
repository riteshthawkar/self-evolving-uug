# Role definitions for self-evolving training
from .proposer import ProposerRole, VerificationSpec
from .solver import SolverRole
from .generator import GeneratorRole
from .editor import EditorRole, EditInstruction, EditResult, EditReward
