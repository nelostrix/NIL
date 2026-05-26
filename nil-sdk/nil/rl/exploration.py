from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


class ExplorationScheduler:
    def __init__(self):
        self.action_values: dict[str, float] = defaultdict(float)
        self.action_counts: dict[str, int] = defaultdict(int)
        self.success_counts: dict[str, int] = defaultdict(int)
        self.failure_counts: dict[str, int] = defaultdict(int)
        self.state_visits: dict[str, int] = defaultdict(int)
        self.total_actions: int = 0

        self.uncertainty_threshold: float = 0.6
        self.min_visits: int = 3
        self.base_exploration_rate: float = 0.1
        self.max_uncertainty: float = 1.0

    def ucb1(self, action: str) -> float:
        Q = self.action_values.get(action, 0.0)
        n = self.action_counts.get(action, 1)
        return Q + math.sqrt(2 * math.log(self.total_actions + 1) / n)

    def thompson_sample(self) -> str:
        actions = list(set(list(self.success_counts.keys()) + list(self.failure_counts.keys())))
        if not actions:
            return "explore"
        best_action = actions[0]
        best_score = -float("inf")
        for a in actions:
            alpha = self.success_counts.get(a, 1)
            beta = self.failure_counts.get(a, 1)
            sample = random.betavariate(alpha, beta)
            if sample > best_score:
                best_score = sample
                best_action = a
        return best_action

    def should_explore(self, state: dict) -> bool:
        if state.get("uncertainty_level", 1.0) > self.uncertainty_threshold:
            return True
        state_key = state.get("key", json.dumps(state, sort_keys=True))
        if self.state_visits.get(state_key, 0) < self.min_visits:
            return True
        epsilon = self._adaptive_epsilon(state)
        return random.random() < epsilon

    def sample_action(self, available_actions: list[str]) -> str:
        if not available_actions:
            return "explore"
        scores = [(a, self.ucb1(a)) for a in available_actions]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def record_action(self, action: str, reward: float, success: bool) -> None:
        self.total_actions += 1
        self.action_counts[action] += 1
        n = self.action_counts[action]
        old = self.action_values[action]
        self.action_values[action] = old + (reward - old) / n

        if success:
            self.success_counts[action] += 1
        else:
            self.failure_counts[action] += 1

    def record_state_visit(self, state_key: str) -> None:
        self.state_visits[state_key] += 1

    def _adaptive_epsilon(self, state: dict) -> float:
        uncertainty = state.get("uncertainty_level", 0.5)
        uncertainty_ratio = uncertainty / max(self.max_uncertainty, 0.01)
        decay_factor = 1.0 / (1.0 + self.total_actions * 0.01)
        return max(0.01, min(0.5, self.base_exploration_rate * uncertainty_ratio * decay_factor))
