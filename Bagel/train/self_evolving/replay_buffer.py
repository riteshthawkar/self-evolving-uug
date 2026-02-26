# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from PIL import Image


@dataclass
class ReplayEntry:
    image: Image.Image
    prompt: str
    questions: List[str]
    reference_answers: List[str]
    reward: float
    step_generated: int
    meta: Dict = field(default_factory=dict)


class ReplayBuffer:
    """Fixed-size FIFO replay buffer for generated supervision examples."""

    def __init__(
        self,
        *,
        max_size: int = 1000,
        min_reward: float = 0.5,
        max_staleness: int = 500,
    ) -> None:
        self.max_size = max(1, int(max_size))
        self.min_reward = float(min_reward)
        self.max_staleness = max(0, int(max_staleness))
        self._entries: List[ReplayEntry] = []

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
        if float(reward) < self.min_reward:
            return False
        if not questions or not reference_answers:
            return False
        n = min(len(questions), len(reference_answers))
        if n <= 0:
            return False
        questions = [str(v).strip() for v in questions[:n] if str(v).strip()]
        reference_answers = [str(v).strip() for v in reference_answers[:n] if str(v).strip()]
        n = min(len(questions), len(reference_answers))
        if n <= 0:
            return False

        if self.max_staleness > 0:
            self.evict_stale(int(step))
        while len(self._entries) >= self.max_size:
            self._entries.pop(0)

        self._entries.append(
            ReplayEntry(
                image=image,
                prompt=str(prompt or ""),
                questions=questions[:n],
                reference_answers=reference_answers[:n],
                reward=float(reward),
                step_generated=int(step),
                meta=dict(meta or {}),
            )
        )
        return True

    def sample(self) -> Optional[ReplayEntry]:
        if not self._entries:
            return None
        return random.choice(self._entries)

    def evict_stale(self, current_step: int) -> int:
        if self.max_staleness <= 0:
            return 0
        cutoff = int(current_step) - int(self.max_staleness)
        prev = len(self._entries)
        self._entries = [e for e in self._entries if int(e.step_generated) >= cutoff]
        return prev - len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> Dict[str, float]:
        if not self._entries:
            return {
                "replay_buffer_size": 0.0,
                "replay_buffer_mean_reward": 0.0,
                "replay_buffer_min_step": 0.0,
                "replay_buffer_max_step": 0.0,
            }
        rewards = [float(e.reward) for e in self._entries]
        steps = [int(e.step_generated) for e in self._entries]
        return {
            "replay_buffer_size": float(len(self._entries)),
            "replay_buffer_mean_reward": float(sum(rewards) / float(len(rewards))),
            "replay_buffer_min_step": float(min(steps)),
            "replay_buffer_max_step": float(max(steps)),
        }
