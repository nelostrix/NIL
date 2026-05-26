from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.models.state_space import StateKey

logger = logging.getLogger("nil.models.observations")


@dataclass
class ObservationModel:
    observation_likelihoods: dict[tuple[StateKey, str], dict[str, float]] = field(default_factory=dict)
    raw_obs_counts: dict[tuple[StateKey, str], Counter] = field(default_factory=dict)
    total_observations: int = 0

    def record(self, state: StateKey, action: str, observation: str, count: int = 1) -> None:
        key = (state, action)
        if key not in self.raw_obs_counts:
            self.raw_obs_counts[key] = Counter()
        self.raw_obs_counts[key][observation] += count
        self.total_observations += count
        self._recompute(key)

    def _recompute(self, key: tuple[StateKey, str]) -> None:
        counter = self.raw_obs_counts.get(key)
        if not counter:
            return
        total = sum(counter.values())
        if total == 0:
            return
        self.observation_likelihoods[key] = {
            obs: cnt / total for obs, cnt in counter.items()
        }

    def prob(self, observation: str, state: StateKey, action: str) -> float:
        probs = self.observation_likelihoods.get((state, action), {})
        return probs.get(observation, 1e-6)

    def most_likely_state(self, observation: str, action: str, candidates: list[StateKey]) -> list[tuple[StateKey, float]]:
        scored = []
        for state in candidates:
            p = self.prob(observation, state, action)
            scored.append((state, p))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def diagnostic_value(self, observation: str, action: str, states: list[StateKey]) -> float:
        probs = [self.prob(observation, s, action) for s in states]
        if not probs:
            return 0.0
        total = sum(probs)
        if total == 0:
            return 0.0
        normalized = [p / total for p in probs]
        entropy = -sum(p * math.log(p + 1e-10) for p in normalized)
        max_entropy = math.log(len(states))
        return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

    def to_dict(self) -> dict:
        serializable = {}
        for (state, action), probs in self.observation_likelihoods.items():
            key = f"{state}|{action}"
            serializable[key] = probs
        return {
            "observation_likelihoods": serializable,
            "total_observations": self.total_observations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ObservationModel:
        model = cls(total_observations=d.get("total_observations", 0))
        for key, probs in d.get("observation_likelihoods", {}).items():
            parts = key.split("|", 1)
            if len(parts) == 2:
                state, action = parts
                model.observation_likelihoods[(state, action)] = probs
        return model
