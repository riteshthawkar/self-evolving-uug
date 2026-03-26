"""
Replay buffer for the self-evolving pipeline.

Stores the best generated images along with their scoring metadata
(prompt, questions, reference answers) so they can be mixed into
the understanding training step.  This closes the
*generation → understanding* supervision loop.

Design principles:
    - FIFO eviction keeps the buffer bounded.
    - Quality gate (min_reward) ensures only good images are kept.
    - Staleness eviction ensures the solver/proposer always train on
      images from the *current* generator quality level.
    - Thread-safety is **not** required (one buffer per DDP rank,
      single-threaded training loop).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from PIL import Image


# ---- data container ------------------------------------------------------- #

@dataclass
class ReplayEntry:
    """One stored generated image with its scoring metadata."""

    image: Image.Image
    prompt: str                         # generation prompt
    questions: List[str]                # proposer-generated questions
    reference_answers: List[str]        # solver answers on the *real* image
    reward: float                       # total_reward at generation time
    step_generated: int                 # training step when image was created
    meta: Dict = field(default_factory=dict)   # arbitrary extra info


# ---- buffer --------------------------------------------------------------- #

class ReplayBuffer:
    """Fixed-size FIFO buffer of :class:`ReplayEntry` items.

    Parameters
    ----------
    max_size : int
        Maximum number of entries.  When full, the oldest entry is evicted.
    min_reward : float
        Quality gate – entries with ``reward < min_reward`` are rejected.
    max_staleness : int
        Entries older than ``current_step - max_staleness`` are evicted on
        every :meth:`add` call.  Set to 0 to disable staleness eviction.
    """

    def __init__(
        self,
        max_size: int = 1000,
        min_reward: float = 0.5,
        max_staleness: int = 500,
    ) -> None:
        self.max_size = max(1, max_size)
        self.min_reward = min_reward
        self.max_staleness = max_staleness
        self._entries: List[ReplayEntry] = []

    # ---- core API --------------------------------------------------------- #

    def add(
        self,
        *,
        image: Image.Image,
        prompt: str,
        questions: List[str],
        reference_answers: List[str],
        reward: float,
        step: int,
        meta: Optional[Dict] = None,
    ) -> bool:
        """Attempt to add an entry.  Returns ``True`` if accepted.

        The entry is rejected (returns ``False``) when:
        * ``reward < self.min_reward``
        * ``questions`` or ``reference_answers`` is empty
        """
        if reward < self.min_reward:
            return False
        if not questions or not reference_answers:
            return False
        if len(questions) != len(reference_answers):
            # Defensive: truncate to the shorter list
            n = min(len(questions), len(reference_answers))
            questions = questions[:n]
            reference_answers = reference_answers[:n]
        if not questions:
            return False

        entry = ReplayEntry(
            image=image,
            prompt=prompt,
            questions=list(questions),
            reference_answers=list(reference_answers),
            reward=reward,
            step_generated=step,
            meta=dict(meta) if meta else {},
        )

        # Staleness eviction first (makes room)
        if self.max_staleness > 0:
            self._evict_stale(step)

        # FIFO eviction if at capacity
        while len(self._entries) >= self.max_size:
            self._entries.pop(0)

        self._entries.append(entry)
        return True

    def sample(self, rng: Optional[random.Random] = None) -> Optional[ReplayEntry]:
        """Return a uniformly random entry, or ``None`` if empty."""
        if not self._entries:
            return None
        chooser = rng if rng is not None else random
        return chooser.choice(self._entries)

    def sample_batch(self, n: int) -> List[ReplayEntry]:
        """Return up to *n* random entries (without replacement)."""
        if not self._entries:
            return []
        k = min(n, len(self._entries))
        return random.sample(self._entries, k)

    # ---- housekeeping ----------------------------------------------------- #

    def _evict_stale(self, current_step: int) -> int:
        """Remove entries older than ``max_staleness`` steps. Returns count."""
        if self.max_staleness <= 0:
            return 0
        cutoff = current_step - self.max_staleness
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.step_generated >= cutoff]
        return before - len(self._entries)

    def evict_stale(self, current_step: int) -> int:
        """Public wrapper for staleness eviction."""
        return self._evict_stale(current_step)

    # ---- introspection ---------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return len(self._entries) > 0

    def stats(self) -> Dict[str, float]:
        """Return summary statistics for logging / W&B."""
        if not self._entries:
            return {
                "replay_buffer_size": 0,
                "replay_buffer_mean_reward": 0.0,
                "replay_buffer_min_step": 0,
                "replay_buffer_max_step": 0,
            }
        rewards = [e.reward for e in self._entries]
        steps = [e.step_generated for e in self._entries]
        return {
            "replay_buffer_size": len(self._entries),
            "replay_buffer_mean_reward": sum(rewards) / len(rewards),
            "replay_buffer_min_step": min(steps),
            "replay_buffer_max_step": max(steps),
        }
