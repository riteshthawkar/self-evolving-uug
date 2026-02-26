"""
FrozenJudge: EMA-updated judge for anti-reward-hacking.
Prevents generator-judge co-adaptation.
Ported from self_evolving/judge.py.
"""

import copy
import torch
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .utils import GenerationSpec


def _clone_solver_lightweight(solver):
    """Clone a solver without deepcopying non-model state.

    For large training objects, ``copy.deepcopy(solver)`` can duplicate
    processor/tokenizer/runtime fields that the judge does not need.
    We keep a shallow solver copy but deep-copy only ``solver.model`` so
    the frozen judge never shares train-time parameters with the live model.
    """
    if not hasattr(solver, "model") or not hasattr(solver.model, "state_dict"):
        return copy.deepcopy(solver)

    # Shallow-copy the solver object (fields like processor are shared — fine)
    cloned = copy.copy(solver)

    # Deep-copy only the model to avoid parameter aliasing with live training.
    # If this fails for a custom module, fall back to full deepcopy for safety.
    try:
        cloned_model = copy.deepcopy(solver.model)
    except Exception:
        return copy.deepcopy(solver)

    src_param_ids = {id(param) for param in solver.model.parameters()}
    if any(id(param) in src_param_ids for param in cloned_model.parameters()):
        return copy.deepcopy(solver)

    cloned.model = cloned_model
    cloned.model.eval()
    for param in cloned.model.parameters():
        param.requires_grad_(False)
    return cloned


class FrozenJudge:
    """
    EMA-updated frozen judge for anti-reward-hacking.

    The judge evaluates generated images using an EMA copy of the solver,
    preventing the generator and judge from co-adapting into degenerate
    "always pass" equilibria.

    Key features:
    - EMA updates (slow-moving, stable)
    - Optional cross-play with multiple snapshots
    - Periodic hard resets to reference
    """

    def __init__(
        self,
        solver,
        ema_decay: float = 0.99,
        keep_snapshots: int = 3,
    ):
        self.ema_decay = ema_decay
        self.keep_snapshots = keep_snapshots

        # Create frozen copy (lightweight)
        self.solver = _clone_solver_lightweight(solver)

        # Historical snapshots for cross-play
        self.snapshots = []

        # Track updates
        self.update_count = 0

    def update(self, new_solver):
        """Update frozen judge with EMA from new solver."""
        if not hasattr(new_solver, "model") or not hasattr(self.solver, "model"):
            return

        # EMA update
        for p_ema, p_new in zip(
            self.solver.model.parameters(), new_solver.model.parameters()
        ):
            p_ema.data = self.ema_decay * p_ema.data + (1 - self.ema_decay) * p_new.data

        self.update_count += 1

        # Periodically save snapshot for cross-play
        if self.update_count % 100 == 0:
            self._save_snapshot()

    def _save_snapshot(self):
        """Save current state as a lightweight snapshot."""
        if len(self.snapshots) >= self.keep_snapshots:
            removed = self.snapshots.pop(0)
            del removed

        snapshot = _clone_solver_lightweight(self.solver)
        self.snapshots.append(snapshot)

    def evaluate(
        self,
        image,
        spec,
        n_samples: int = 5,
        use_cross_play: bool = True,
        aggregation: str = "min",
    ):
        """
        Evaluate an image against a spec.

        Returns:
            Tuple of (score, per_question_scores)
        """
        scores = []

        # Evaluate with current frozen judge
        score, per_q = self.solver.verify_with_spec(image, spec, n_samples)
        scores.append(score)

        # Cross-play evaluation with historical snapshots
        if use_cross_play and self.snapshots:
            for snapshot in self.snapshots:
                snap_score, _ = snapshot.verify_with_spec(image, spec, n_samples)
                scores.append(snap_score)

        # Aggregate
        if aggregation == "min":
            final_score = min(scores)
        elif aggregation == "median":
            scores.sort()
            final_score = scores[len(scores) // 2]
        elif aggregation == "mean":
            final_score = sum(scores) / len(scores)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return final_score, per_q

    def reset_to_reference(self, reference_solver):
        """Hard reset frozen judge to reference solver."""
        self.solver = _clone_solver_lightweight(reference_solver)

        for snap in self.snapshots:
            del snap
        self.snapshots = []
        self.update_count = 0


class CrossPlayEvaluator:
    """
    Full cross-play evaluation for anti-reward-hacking.

    Cross-play evaluates generated images with multiple solver versions
    to detect and prevent reward hacking.
    """

    def __init__(
        self,
        snapshots_dir: Optional[str] = None,
        max_snapshots: int = 5,
        divergence_threshold: float = 0.3,
    ):
        self.snapshots_dir = snapshots_dir
        self.max_snapshots = max_snapshots
        self.divergence_threshold = divergence_threshold
        self.snapshots = []

        self.divergence_history = []
        self.evaluation_count = 0

    def add_snapshot(self, solver, step: int, name: Optional[str] = None):
        """Add a solver snapshot."""
        if len(self.snapshots) >= self.max_snapshots:
            self.snapshots.pop(0)

        snapshot = {
            "solver": _clone_solver_lightweight(solver),
            "step": step,
            "name": name or f"snapshot_{step}",
        }

        self.snapshots.append(snapshot)

        if self.snapshots_dir:
            self._save_snapshot(snapshot)

    def _save_snapshot(self, snapshot):
        """Save snapshot to disk."""
        import os

        os.makedirs(self.snapshots_dir, exist_ok=True)

        path = os.path.join(self.snapshots_dir, f"{snapshot['name']}.pt")

        if hasattr(snapshot["solver"], "model"):
            state = {
                "step": snapshot["step"],
                "name": snapshot["name"],
                "model_state": snapshot["solver"].model.state_dict(),
            }
            torch.save(state, path)

    def load_snapshots(self, solver_template):
        """Load snapshots from disk."""
        import os
        import glob

        if not self.snapshots_dir or not os.path.exists(self.snapshots_dir):
            return

        for path in sorted(glob.glob(os.path.join(self.snapshots_dir, "*.pt"))):
            state = torch.load(path)

            solver_copy = _clone_solver_lightweight(solver_template)
            if hasattr(solver_copy, "model"):
                solver_copy.model.load_state_dict(state["model_state"])

            self.snapshots.append(
                {
                    "solver": solver_copy,
                    "step": state["step"],
                    "name": state["name"],
                }
            )

    def evaluate_cross_play(
        self,
        image,
        spec,
        current_solver,
        n_samples: int = 5,
        aggregation: str = "min",
    ):
        """Evaluate using current + historical solvers."""
        scores = []

        current_score, per_q = current_solver.verify_with_spec(image, spec, n_samples)
        scores.append(("current", current_score))

        historical_scores = []
        for snap in self.snapshots:
            snap_score, _ = snap["solver"].verify_with_spec(image, spec, n_samples)
            scores.append((snap["name"], snap_score))
            historical_scores.append(snap_score)

        all_scores = [s[1] for s in scores]
        if aggregation == "min":
            final_score = min(all_scores) if all_scores else 0.0
        elif aggregation == "mean":
            final_score = (
                sum(all_scores) / len(all_scores) if all_scores else 0.0
            )
        elif aggregation == "median":
            all_scores.sort()
            final_score = (
                all_scores[len(all_scores) // 2] if all_scores else 0.0
            )
        else:
            final_score = min(all_scores) if all_scores else 0.0

        divergence_flag = False
        if historical_scores:
            avg_historical = sum(historical_scores) / len(historical_scores)
            divergence = current_score - avg_historical

            if divergence > self.divergence_threshold:
                divergence_flag = True
                self.divergence_history.append(
                    {
                        "step": self.evaluation_count,
                        "divergence": divergence,
                        "current": current_score,
                        "historical_avg": avg_historical,
                    }
                )

        self.evaluation_count += 1

        return final_score, scores, divergence_flag

    def get_divergence_rate(self, last_n: int = 100) -> float:
        if self.evaluation_count == 0:
            return 0.0
        recent_divergences = [
            d
            for d in self.divergence_history
            if d["step"] >= self.evaluation_count - last_n
        ]
        return len(recent_divergences) / min(last_n, self.evaluation_count)

    def should_reset_judge(self) -> bool:
        divergence_rate = self.get_divergence_rate()
        return divergence_rate > 0.3

    def get_stats(self) -> dict:
        return {
            "num_snapshots": len(self.snapshots),
            "total_evaluations": self.evaluation_count,
            "divergence_count": len(self.divergence_history),
            "divergence_rate": self.get_divergence_rate(),
        }
