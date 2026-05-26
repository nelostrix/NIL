from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


class ValueFunction:
    def __init__(self, state_dim: int = 4, hidden_dim: int = 16):
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
                self._hidden_weights[i] -= self.learning_rate * grad * hidden_activated[i]
                h_grad = grad * self._hidden_weights[i] if hidden_activated[i] > 0 else 0.0
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
    ) -> list[float]:
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)

        return advantages

    def compute_monte_carlo(self, rewards: list[float], gamma: float = 0.99) -> list[float]:
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns
