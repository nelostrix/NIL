from __future__ import annotations

import random
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Transition:
    state: dict = field(default_factory=dict)
    action: str = ""
    reward: float = 0.0
    next_state: dict = field(default_factory=dict)
    done: bool = False
    log_prob: float = 0.0
    terminated: bool | None = None
    truncated: bool = False

    def __post_init__(self):
        if not math.isfinite(self.reward) or not math.isfinite(self.log_prob):
            raise ValueError("Transition rewards/log probabilities must be finite")
        self.truncated = bool(self.truncated)
        self.terminated = bool(self.done and not self.truncated) if self.terminated is None else bool(self.terminated)
        self.done = self.terminated or self.truncated


class ReplayBuffer:
    def __init__(self, max_size: int = 10000, *, seed=None):
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
            raise ValueError("max_size must be a positive integer")
        self.max_size = max_size
        self._items: list[Transition] = []
        self._cursor = 0
        self._rng = random.Random(seed)

    @property
    def buffer(self):
        """Chronological list snapshot; Transition objects remain shared."""
        return self.latest(self.size)

    @property
    def size(self) -> int:
        return len(self._items)

    def push(self, transition: Transition) -> None:
        if not isinstance(transition, Transition):
            raise TypeError("ReplayBuffer accepts Transition objects")
        if self.size < self.max_size:
            self._items.append(transition)
        else:
            self._items[self._cursor] = transition
        self._cursor = (self._cursor + 1) % self.max_size

    def push_many(self, transitions: list[Transition]) -> None:
        for t in transitions:
            self.push(t)

    def sample(self, batch_size: int) -> list[Transition]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not self._items:
            return []
        batch_size = min(batch_size, self.size)
        return self._rng.sample(self._items, batch_size)

    def clear(self) -> None:
        self._items.clear()
        self._cursor = 0

    def latest(self, n: int = 10) -> list[Transition]:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        count = min(n, self.size)
        if not count:
            return []
        return [self._items[(self._cursor - count + index) % self.size] for index in range(count)]

    def to_dict_list(self) -> list[dict]:
        return [asdict(transition) for transition in self.buffer]

    def state_dict(self):
        return {"schema_version": 1, "max_size": self.max_size, "cursor": self._cursor,
                "items": [asdict(transition) for transition in self._items], "rng": self._rng.getstate()}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or type(state.get("schema_version")) is not int or state.get("schema_version") != 1 or type(state.get("max_size")) is not int or state.get("max_size") != self.max_size:
            raise ValueError("Replay state schema/capacity mismatch")
        items = state.get("items")
        cursor = state.get("cursor")
        if not isinstance(items, list) or len(items) > self.max_size or type(cursor) is not int or not 0 <= cursor < self.max_size or len(items) < self.max_size and cursor != len(items):
            raise ValueError("Invalid replay contents/cursor")
        restored = [Transition(**item) for item in items]

        def tuples(value):
            return tuple(tuples(part) for part in value) if isinstance(value, (list, tuple)) else value

        rng = random.Random()
        rng.setstate(tuples(state["rng"]))
        self._items, self._cursor, self._rng = restored, cursor, rng
        return self
