from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Transition:
    state: dict = field(default_factory=dict)
    action: str = ""
    reward: float = 0.0
    next_state: dict = field(default_factory=dict)
    done: bool = False
    log_prob: float = 0.0


class ReplayBuffer:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: list[Transition] = []

    @property
    def size(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def push_many(self, transitions: list[Transition]) -> None:
        for t in transitions:
            self.push(t)

    def sample(self, batch_size: int) -> list[Transition]:
        if not self.buffer:
            return []
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def clear(self) -> None:
        self.buffer.clear()

    def latest(self, n: int = 10) -> list[Transition]:
        return self.buffer[-n:]

    def to_dict_list(self) -> list[dict]:
        return [
            {
                "state": t.state,
                "action": t.action,
                "reward": t.reward,
                "next_state": t.next_state,
                "done": t.done,
            }
            for t in self.buffer
        ]
