from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.rl.rewards import StepRecord


@dataclass
class TrajectoryScore:
    steps_efficiency: float = 0.0
    token_efficiency: float = 0.0
    cost_efficiency: float = 0.0
    correctness: float = 0.0
    completeness: float = 0.0
    robustness: float = 0.0
    novelty: float = 0.0
    information_gain: float = 0.0
    exploration_value: float = 0.0
    composite: float = 0.0

    def __post_init__(self):
        self.composite = round((
            self.steps_efficiency * 0.20
            + self.token_efficiency * 0.15
            + self.cost_efficiency * 0.15
            + self.correctness * 0.20
            + self.completeness * 0.10
            + self.robustness * 0.10
            + self.novelty * 0.05
            + self.information_gain * 0.05
        ), 4)


class TrajectoryScorer:
    def score(self, trace: dict) -> TrajectoryScore:
        steps = trace.get("steps", [])
        success = trace.get("outcome_success", False)
        total_cost = trace.get("total_cost", 0)
        total_tokens = trace.get("total_tokens_in", 0) + trace.get("total_tokens_out", 0)

        return TrajectoryScore(
            steps_efficiency=self._step_efficiency(steps, success),
            token_efficiency=self._token_efficiency(steps, total_tokens, success),
            cost_efficiency=self._cost_efficiency(total_cost, success),
            correctness=1.0 if success else 0.0,
            completeness=self._completeness(steps),
            robustness=self._robustness(steps),
            novelty=self._novelty(trace),
            information_gain=self._information_gain(trace),
        )

    def rank(self, trajectories: list[dict]) -> list[tuple[dict, TrajectoryScore]]:
        scored = [(t, self.score(t)) for t in trajectories]
        return sorted(scored, key=lambda x: x[1].composite, reverse=True)

    def _step_efficiency(self, steps: list, success: bool) -> float:
        if not steps:
            return 0.0
        n = len(steps)
        if n <= 3:
            return 1.0 if success else 0.3
        return max(0.0, 1.0 - (n - 3) * 0.05)

    def _token_efficiency(self, steps: list, total_tokens: int, success: bool) -> float:
        if not steps or total_tokens == 0:
            return 0.0
        tokens_per_step = total_tokens / len(steps)
        if tokens_per_step <= 100:
            return 1.0 if success else 0.2
        return max(0.0, 1.0 - (tokens_per_step - 100) * 0.002)

    def _cost_efficiency(self, total_cost: float, success: bool) -> float:
        if total_cost <= 0.001:
            return 1.0 if success else 0.2
        return max(0.0, 1.0 - total_cost * 50)

    def _completeness(self, steps: list) -> float:
        if not steps:
            return 0.0
        has_output = sum(1 for s in steps if isinstance(s, dict) and s.get("tool_output"))
        return has_output / max(len(steps), 1)

    def _robustness(self, steps: list) -> float:
        if not steps:
            return 0.0
        errors = sum(1 for s in steps if isinstance(s, dict) and s.get("error"))
        return 1.0 - (errors / max(len(steps), 1))

    def _novelty(self, trace: dict) -> float:
        return 0.1

    def _information_gain(self, trace: dict) -> float:
        return 0.1


class EloTrajectoryRating:
    def __init__(self, k_factor: int = 32, initial_rating: int = 1500):
        self.ratings: dict[str, float] = {}
        self.k = k_factor
        self.initial_rating = initial_rating

    def compare(self, winner_id: str, loser_id: str) -> tuple[float, float]:
        R_A = self.ratings.get(winner_id, self.initial_rating)
        R_B = self.ratings.get(loser_id, self.initial_rating)

        E_A = 1.0 / (1.0 + 10 ** ((R_B - R_A) / 400.0))
        E_B = 1.0 - E_A

        self.ratings[winner_id] = R_A + self.k * (1.0 - E_A)
        self.ratings[loser_id] = R_B + self.k * (0.0 - E_B)

        return self.ratings[winner_id], self.ratings[loser_id]

    def get_rating(self, trace_id: str) -> float:
        return self.ratings.get(trace_id, self.initial_rating)

    def leaderboard(self, top_n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
