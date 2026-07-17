from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field

from nil.models.state_space import StateKey

logger = logging.getLogger("nil.models.transitions")
# usecase to Log system information
ActionKey = str


@dataclass
class TransitionOutcome:
    next_state: StateKey = ""
    confidence: float = 0.0
    alternative_states: list[tuple[StateKey, float]] = field(default_factory=list)

    @staticmethod
    def unknown() -> TransitionOutcome:
        return TransitionOutcome(confidence=0.0)


@dataclass
class TransitionModel:
    transitions: dict[tuple[StateKey, ActionKey], dict[StateKey, float]] = field(
        default_factory=dict
    )
    raw_counts: dict[tuple[StateKey, ActionKey], Counter] = field(default_factory=dict)
    total_transitions: int = 0

    def record(
        self,
        from_state: StateKey,
        action: ActionKey,
        to_state: StateKey,
        count: int = 1,
    ) -> None:
        key = (from_state, action)
        if key not in self.raw_counts:
            self.raw_counts[key] = Counter()
        self.raw_counts[key][to_state] += count
        self.total_transitions += count
        self._recompute(key)

    def record_batch(
        self, transitions: list[tuple[StateKey, ActionKey, StateKey, int]]
    ) -> None:
        for from_state, action, to_state, count in transitions:
            self.record(from_state, action, to_state, count)

    def _recompute(self, key: tuple[StateKey, ActionKey]) -> None:
        counter = self.raw_counts.get(key)
        if not counter or not counter:
            self.transitions.pop(key, None)
            return
        total = sum(counter.values())
        self.transitions[key] = {state: cnt / total for state, cnt in counter.items()}

    def prob(
        self, from_state: StateKey, action: ActionKey, to_state: StateKey
    ) -> float:
        probs = self.transitions.get((from_state, action), {})
        return probs.get(to_state, 0.0)

    def predict(
        self, state: StateKey, action: ActionKey
    ) -> list[tuple[StateKey, float]]:
        probs = self.transitions.get((state, action), {})
        return sorted(probs.items(), key=lambda x: x[1], reverse=True)

    def expected_outcome(self, state: StateKey, action: ActionKey) -> TransitionOutcome:
        predictions = self.predict(state, action)
        if not predictions:
            return TransitionOutcome.unknown()
        return TransitionOutcome(
            next_state=predictions[0][0],
            confidence=predictions[0][1],
            alternative_states=predictions[1:],
        )

    def uncertainty(self, state: StateKey, action: ActionKey) -> float:
        probs = list(self.transitions.get((state, action), {}).values())
        if not probs:
            return 1.0
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        max_entropy = math.log(max(len(probs), 2))
        return entropy / max_entropy if max_entropy > 0 else 1.0

    def known_actions(self, state: StateKey) -> list[ActionKey]:
        return [action for (s, action) in self.transitions if s == state]

    def merge(self, other: TransitionModel) -> None:
        for (state, action), counter in other.raw_counts.items():
            for to_state, count in counter.items():
                self.record(state, action, to_state, count)

    def to_dict(self) -> dict:
        serializable = {}
        for (state, action), probs in self.transitions.items():
            key = f"{state}|{action}"
            serializable[key] = probs
        return {
            "transitions": serializable,
            "total_transitions": self.total_transitions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TransitionModel:
        model = cls(total_transitions=d.get("total_transitions", 0))
        for key, probs in d.get("transitions", {}).items():
            parts = key.split("|", 1)
            if len(parts) == 2:
                state, action = parts
                model.transitions[(state, action)] = probs
        return model
