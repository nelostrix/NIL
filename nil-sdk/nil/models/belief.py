from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.models.state_space import StateKey, StateSpace
from nil.models.transitions import TransitionModel
from nil.models.observations import ObservationModel

logger = logging.getLogger("nil.models.belief")


@dataclass
class BeliefTracker:
    state_space: StateSpace
    transition_model: TransitionModel = field(default_factory=TransitionModel)
    observation_model: ObservationModel = field(default_factory=ObservationModel)
    belief: dict[StateKey, float] = field(default_factory=dict)
    prior: dict[StateKey, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.belief:
            self.reset()

    def reset(self) -> None:
        keys = self.state_space.all_keys()
        if keys:
            uniform = 1.0 / len(keys)
            self.belief = {k: uniform for k in keys}
        else:
            self.belief = {}
        self.prior = dict(self.belief)

    def incorporate_task(self, task: str, context: dict | None = None) -> None:
        context = context or {}
        keys = self.state_space.all_keys()
        if not keys:
            return

        task_lower = task.lower()
        weighted: dict[StateKey, float] = {}
        for key in keys:
            score = 1.0
            if "error" in task_lower or "fix" in task_lower:
                if "degraded" in key or '"degraded"' in key:
                    score *= 2.0
            if "deploy" in task_lower:
                if 'deploy' in key:
                    score *= 2.0
            weighted[key] = score

        total = sum(weighted.values())
        if total > 0:
            self.belief = {k: v / total for k, v in weighted.items()}

    def update(self, action: str, observation: str) -> dict[StateKey, float]:
        if not self.belief:
            return {}

        predicted: dict[StateKey, float] = {}
        for s_prime_key in self.state_space.all_keys():
            total = 0.0
            for s_key, prob in self.belief.items():
                trans_prob = self.transition_model.prob(s_key, action, s_prime_key)
                total += trans_prob * prob

            if total == 0 and s_prime_key in self.belief:
                total = self.belief[s_prime_key] * 0.01

            predicted[s_prime_key] = total

        pred_total = sum(predicted.values())
        if pred_total > 0:
            predicted = {k: v / pred_total for k, v in predicted.items()}

        unnormalized: dict[StateKey, float] = {}
        for s_key, prob in predicted.items():
            obs_prob = self.observation_model.prob(observation, s_key, action)
            unnormalized[s_key] = obs_prob * prob

        total = sum(unnormalized.values())
        if total > 0:
            self.belief = {k: v / total for k, v in unnormalized.items()}
        else:
            self.belief = {k: 0.0 for k in predicted}

        self.prior = dict(predicted)
        return self.belief

    @property
    def most_likely_state(self) -> Optional[str]:
        if not self.belief:
            return None
        return max(self.belief, key=self.belief.get)

    @property
    def most_likely_state_probability(self) -> float:
        mls = self.most_likely_state
        return self.belief.get(mls, 0.0) if mls else 0.0

    @property
    def uncertainty(self) -> float:
        return -sum(p * math.log(p + 1e-10) for p in self.belief.values()) if self.belief else 1.0

    @property
    def normalized_uncertainty(self) -> float:
        u = self.uncertainty
        max_entropy = math.log(max(len(self.belief), 2))
        return u / max_entropy if max_entropy > 0 else 1.0

    def top_beliefs(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.belief.items(), key=lambda x: x[1], reverse=True)[:n]

    def to_dict(self) -> dict:
        return {
            "belief": self.belief,
            "most_likely": self.most_likely_state,
            "uncertainty": self.uncertainty,
            "normalized_uncertainty": self.normalized_uncertainty,
        }
