"""
Verification Reward for self-evolving generation training.
Uses Proposer specs + Solver answers to compute grounded verification.
"""

from typing import Dict, List, Tuple, Optional
from PIL import Image
import math


def verification_reward(
    image: Image.Image,
    spec: "VerificationSpec",  # From proposer.py
    solver,  # SolverRole instance
    n_samples: int = 5,
    match_threshold: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute verification reward based on spec compliance.
    
    Args:
        image: Generated image to verify
        spec: Verification spec with questions and expected answers
        solver: Solver role for answering questions
        n_samples: Samples per question
        match_threshold: Threshold for considering a match
        
    Returns:
        Tuple of (overall reward, per-question rewards)
    """
    return solver.verify_with_spec(image, spec, n_samples)


class VerificationReward:
    """
    Stateful verification reward using frozen judge.
    
    Anti-hacking strategy:
    - Uses EMA-updated frozen judge
    - Multiple judge snapshots for cross-play evaluation
    """
    
    def __init__(
        self,
        solver,
        n_samples: int = 5,
        use_frozen_judge: bool = True,
        ema_decay: float = 0.99,
    ):
        """
        Initialize verification reward.
        
        Args:
            solver: Solver role for verification
            n_samples: Samples per question
            use_frozen_judge: Whether to use EMA-frozen judge
            ema_decay: EMA decay for frozen judge
        """
        self.solver = solver
        self.n_samples = n_samples
        self.use_frozen_judge = use_frozen_judge
        self.ema_decay = ema_decay
        
        # Frozen judge (EMA copy)
        if use_frozen_judge:
            from ..judge import FrozenJudge
            self.frozen_judge = FrozenJudge(solver, ema_decay)
        else:
            self.frozen_judge = None
    
    def __call__(
        self,
        image: Image.Image,
        spec: "VerificationSpec",
    ) -> Tuple[float, Dict[str, float]]:
        """Compute verification reward."""
        # Use frozen judge if available
        judge = self.frozen_judge.solver if self.frozen_judge else self.solver
        
        return verification_reward(
            image=image,
            spec=spec,
            solver=judge,
            n_samples=self.n_samples,
        )
    
    def update_frozen_judge(self):
        """Update frozen judge with current solver weights (EMA)."""
        if self.frozen_judge:
            self.frozen_judge.update(self.solver)


def paired_contrast_check(
    image: Image.Image,
    positive_question: str,
    negative_question: str,
    solver,
    n_samples: int = 5,
) -> float:
    """
    Anti-collusion check using paired contrast questions.
    
    Example:
        positive: "Is it nighttime?"
        negative: "Is it daytime?"
    
    Penalizes contradictory answers (both yes or both no).
    
    Args:
        image: Image to check
        positive_question: Question expecting "yes"
        negative_question: Contrasting question expecting "no"
        solver: Solver for answering
        n_samples: Samples per question
        
    Returns:
        Penalty (higher = more contradictions)
    """
    pos_answers, pos_agreement = solver.solve(image, positive_question, n_samples)
    neg_answers, neg_agreement = solver.solve(image, negative_question, n_samples)
    
    # Check for contradictions
    # If both have high agreement for same answer type, that's suspicious
    
    def is_affirmative(answer: str) -> bool:
        answer = answer.lower().strip()
        return answer.startswith("yes") or answer in ["true", "correct", "1"]
    
    pos_mode = max(set(pos_answers), key=pos_answers.count)
    neg_mode = max(set(neg_answers), key=neg_answers.count)
    
    pos_is_yes = is_affirmative(pos_mode)
    neg_is_yes = is_affirmative(neg_mode)
    
    # Both yes or both no = contradiction
    if pos_is_yes == neg_is_yes:
        # Penalty proportional to agreement
        return (pos_agreement + neg_agreement) / 2
    
    # Consistent = no penalty
    return 0.0


def negative_control_check(
    image: Image.Image,
    negative_question: str,
    solver,
    n_samples: int = 5,
) -> float:
    """
    Check that negative control questions are correctly rejected.
    
    Example: "Is there a purple elephant in the image?" should be "no"
    
    Args:
        image: Image to check
        negative_question: Question that should be answered "no"
        solver: Solver for answering
        n_samples: Samples per question
        
    Returns:
        Reward (higher = correctly rejected)
    """
    answers, agreement = solver.solve(image, negative_question, n_samples)
    
    # Check that answers are negative
    def is_negative(answer: str) -> bool:
        answer = answer.lower().strip()
        return answer.startswith("no") or answer in ["false", "incorrect", "0"]
    
    negative_count = sum(1 for a in answers if is_negative(a))
    return negative_count / len(answers)
