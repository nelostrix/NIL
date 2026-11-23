from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


class ValueFunction:
    def __init__(self, state_dim: int = 4, hidden_dim: int = 16):
        if type(state_dim) is not int or not 1 <= state_dim <= 4 or type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("Agent value features support 1–4 state dimensions and positive hidden_dim")
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self._weights: list[list[float]] = [
            [0.1] * hidden_dim for _ in range(state_dim)
        ]
        self._hidden_weights: list[float] = [0.1] * hidden_dim
        self._bias: float = 0.0
        self.learning_rate: float = 0.01

    def predict(self, state: dict) -> float:
        features = self._extract_features(state)
        hidden = [
            sum(f * w[i] for f, w in zip(features, self._weights))
            for i in range(self.hidden_dim)
        ]
        hidden_activated = [max(0.0, h) for h in hidden]
        return sum(h * w for h, w in zip(hidden_activated, self._hidden_weights)) + self._bias

    def update(self, states: list[dict], returns: list[float]) -> float:
        if len(states) != len(returns) or not all(math.isfinite(value) for value in returns):
            raise ValueError("States/returns must align and returns must be finite")
        total_loss = 0.0
        for state, G in zip(states, returns):
            predicted = self.predict(state)
            loss = (predicted - G) ** 2
            total_loss += loss

            grad = 2 * (predicted - G)
            self._bias -= self.learning_rate * grad

            features = self._extract_features(state)
            hidden = [
                sum(f * w[i] for f, w in zip(features, self._weights))
                for i in range(self.hidden_dim)
            ]
            hidden_activated = [max(0.0, h) for h in hidden]

            for i in range(self.hidden_dim):
                previous_weight = self._hidden_weights[i]
                self._hidden_weights[i] -= self.learning_rate * grad * hidden_activated[i]
                h_grad = grad * previous_weight if hidden_activated[i] > 0 else 0.0
                for j in range(self.state_dim):
                    self._weights[j][i] -= self.learning_rate * h_grad * features[j]

        return total_loss / max(len(states), 1)

    def _extract_features(self, state: dict) -> list[float]:
        return [
            1.0 if state.get("service_health") == "healthy" else 0.0,
            _stage_value(state.get("investigation_stage", "not_started")),
            state.get("steps_taken", 0) / 50.0,
            1.0 - min(state.get("uncertainty_level", 1.0), 1.0),
        ][:self.state_dim]


def _stage_value(stage: str) -> float:
    stages = [
        "not_started", "gathering_data", "analyzing",
        "diagnosed", "fixing", "verifying", "resolved",
    ]
    try:
        return stages.index(stage) / max(len(stages) - 1, 1)
    except ValueError:
        return 0.0


class AdvantageEstimator:
    def compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        gamma: float = 0.99,
        lam: float = 0.95,
        *, terminated=None, truncated=None, next_values=None, last_value=0.0,
    ) -> list[float]:
        rewards, values, terminated, truncated, next_values = _rollout(
            rewards, values, terminated, truncated, next_values, last_value, gamma, lam)
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * (0.0 if terminated[t] else next_values[t]) - values[t]
            gae = delta + gamma * lam * (not (terminated[t] or truncated[t])) * gae
            advantages.append(gae)

        return advantages[::-1]

    def compute_monte_carlo(self, rewards: list[float], gamma: float = 0.99, *,
                            terminated=None, truncated=None, next_values=None, last_value=0.0) -> list[float]:
        rewards, _, terminated, truncated, next_values = _rollout(
            rewards, [0.0] * len(rewards), terminated, truncated, next_values, last_value, gamma, 1.0)
        returns = []
        G = last_value
        for index in reversed(range(len(rewards))):
            if terminated[index]:
                G = 0.0
            elif truncated[index]:
                G = next_values[index]
            G = rewards[index] + gamma * G
            returns.append(G)
        return returns[::-1]


def _rollout(rewards, values, terminated, truncated, next_values, last_value, gamma, lam):
    if any(isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1 for value in (gamma, lam)):
        raise ValueError("gamma/lam must be finite and in [0, 1]")
    rewards, values = list(rewards), list(values)
    if len(rewards) != len(values) or not all(math.isfinite(value) for value in [*rewards, *values, last_value]):
        raise ValueError("Rewards/values must align and be finite")
    n = len(rewards)
    terminated = [False] * n if terminated is None else list(terminated)
    truncated = [False] * n if truncated is None else list(truncated)
    if len(terminated) != n or len(truncated) != n:
        raise ValueError("Termination/truncation masks must align with rewards")
    if any(value not in (False, True) for value in [*terminated, *truncated]):
        raise ValueError("Termination/truncation masks must contain booleans")
    if next_values is None:
        if any(truncated):
            raise ValueError("Truncations require next_values from final observations, not reset observations")
        next_values = values[1:] + ([last_value] if n else [])
    else:
        next_values = list(next_values)
    if len(next_values) != n or not all(math.isfinite(value) for value in next_values):
        raise ValueError("next_values must align and be finite")
    return rewards, values, terminated, truncated, next_values
