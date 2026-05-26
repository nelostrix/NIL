from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nil.models.state_space import EnvironmentState, StateKey, INVESTIGATION_STAGES

logger = logging.getLogger("nil.models.rewards")


@dataclass
class RewardModel:
    domain: str = ""
    cost_sensitivity: float = 1.0
    time_sensitivity: float = 0.5
    exploration_bonus: float = 0.1
    success_reward: float = 10.0
    failure_penalty: float = -5.0
    progress_reward_per_stage: float = 0.2

    def reward(self, state: EnvironmentState, action_name: str, action_cost: float, action_duration: float, next_state: EnvironmentState) -> float:
        progress = self._progress_reward(state, next_state)
        success_reward = self._terminal_reward(next_state)
        cost_penalty = -self.cost_sensitivity * action_cost
        time_penalty = -self.time_sensitivity * action_duration
        exploration = self.exploration_bonus if self._is_novel(state, action_name) else 0.0
        return progress + success_reward + cost_penalty + time_penalty + exploration

    def _progress_reward(self, state: EnvironmentState, next_state: EnvironmentState) -> float:
        try:
            s_idx = INVESTIGATION_STAGES.index(state.investigation_stage)
            ns_idx = INVESTIGATION_STAGES.index(next_state.investigation_stage)
            return (ns_idx - s_idx) * self.progress_reward_per_stage
        except (ValueError, IndexError):
            return 0.0

    def _terminal_reward(self, state: EnvironmentState) -> float:
        if state.is_terminal_success:
            return self.success_reward
        if state.is_terminal_failure:
            return self.failure_penalty
        return 0.0

    def _is_novel(self, state: EnvironmentState, action: str) -> bool:
        return action not in state.tools_used

    def expected(self, state: StateKey, action: str) -> float:
        return 0.0

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "cost_sensitivity": self.cost_sensitivity,
            "time_sensitivity": self.time_sensitivity,
            "exploration_bonus": self.exploration_bonus,
            "success_reward": self.success_reward,
            "failure_penalty": self.failure_penalty,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RewardModel:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RewardObservation:
    from_state: StateKey
    action: str
    to_state: StateKey
    cost: float
    duration: float
    success: bool
    error: str = ""


@dataclass
class RewardLearner:
    cost_sensitivity_range: tuple[float, float] = (0.1, 5.0)
    time_sensitivity_range: tuple[float, float] = (0.1, 2.0)
    observations: list[RewardObservation] = field(default_factory=list)

    def observe(self, obs: RewardObservation) -> None:
        self.observations.append(obs)

    def learn(self, domain: str = "") -> RewardModel:
        if not self.observations:
            return RewardModel(domain=domain)

        successful = [o for o in self.observations if o.success]
        failed = [o for o in self.observations if not o.success]

        avg_cost_success = sum(o.cost for o in successful) / max(len(successful), 1)
        avg_cost_failure = sum(o.cost for o in failed) / max(len(failed), 1)

        avg_duration_success = sum(o.duration for o in successful) / max(len(successful), 1)
        avg_duration_failure = sum(o.duration for o in failed) / max(len(failed), 1)

        cost_sensitivity = 1.0
        if avg_cost_success > 0 and avg_cost_failure > 0:
            cost_ratio = avg_cost_failure / avg_cost_success
            cost_sensitivity = max(0.5, min(3.0, cost_ratio))

        time_sensitivity = 0.5
        if avg_duration_success > 0 and avg_duration_failure > 0:
            time_ratio = avg_duration_failure / avg_duration_success
            time_sensitivity = max(0.3, min(1.5, time_ratio))

        return RewardModel(
            domain=domain,
            cost_sensitivity=round(cost_sensitivity, 2),
            time_sensitivity=round(time_sensitivity, 2),
        )
