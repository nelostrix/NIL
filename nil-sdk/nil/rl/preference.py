from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.rl.trajectory import EloTrajectoryRating


class DirectPreferenceOptimizer:
    def __init__(self, beta: float = 0.1, learning_rate: float = 0.01):
        self.beta = beta
        self.lr = learning_rate
        self.action_logits: dict[str, float] = {}

    def optimize(
        self,
        preferences: list[tuple[str, str, float]],
        action_space: list[str] = None,
        n_epochs: int = 10,
    ) -> dict[str, float]:
        action_space = action_space or []
        if not self.action_logits and action_space:
            self.action_logits = {a: 0.0 for a in action_space}

        for _ in range(n_epochs):
            for winner, loser, strength in preferences:
                log_p_winner = self.action_logits.get(winner, 0.0)
                log_p_loser = self.action_logits.get(loser, 0.0)

                diff = self.beta * (log_p_winner - log_p_loser)
                sigmoid = 1.0 / (1.0 + math.exp(-diff))
                loss = -math.log(max(sigmoid, 1e-10))

                grad = strength * (sigmoid - 1.0) * self.beta

                self.action_logits[winner] = self.action_logits.get(winner, 0.0) - self.lr * grad * 1.0
                self.action_logits[loser] = self.action_logits.get(loser, 0.0) - self.lr * grad * (-1.0)

        return dict(self.action_logits)

    def preference_from_scores(
        self,
        scores: list[tuple[str, float]],
        n_pairs: int = 10,
    ) -> list[tuple[str, str, float]]:
        preferences = []
        sorted_items = sorted(scores, key=lambda x: x[1], reverse=True)

        for _ in range(n_pairs):
            if len(sorted_items) < 2:
                break
            high_idx = random.randint(0, max(0, len(sorted_items) // 3))
            low_idx = random.randint(len(sorted_items) // 2, len(sorted_items) - 1)
            if high_idx < low_idx:
                winner_id = sorted_items[high_idx][0]
                loser_id = sorted_items[low_idx][0]
                score_diff = sorted_items[high_idx][1] - sorted_items[low_idx][1]
                strength = min(1.0, max(0.1, score_diff * 2))
                preferences.append((winner_id, loser_id, strength))

        return preferences
