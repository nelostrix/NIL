from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StepRecord:
    step_number: int = 0
    step_type: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_output: str = ""
    duration_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""
    timestamp: float = 0.0


@dataclass
class RLConfig:
    discount_factor: float = 0.99
    exploration_bonus: float = 0.05
    efficiency_threshold: float = 0.01
    efficiency_bonus: float = 0.1
    progress_reward_scale: float = 1.0
    cost_penalty_scale: float = 1.0
    error_penalty_scale: float = 1.0

    # Reward shaping
    shaping_discount: float = 0.99

    # Progress stage values for incident-response domain
    stage_values: dict[str, float] = field(default_factory=lambda: {
        "not_started": 0.0,
        "gathering_data": 2.0,
        "analyzing": 4.0,
        "diagnosed": 5.0,
        "fixing": 7.0,
        "verifying": 9.0,
        "resolved": 10.0,
    })

    health_values: dict[str, float] = field(default_factory=lambda: {
        "unknown": 0.0,
        "degraded": 2.0,
        "healthy": 8.0,
    })

    uncertainty_penalty_scale: float = 3.0


def _investigation_stage_value(stage: str, config: RLConfig) -> float:
    return config.stage_values.get(stage, 0.0)


def _health_value(health: str, config: RLConfig) -> float:
    return config.health_values.get(health, 0.0)


class DenseRewardModel:
    def __init__(self, config: Optional[RLConfig] = None):
        self.config = config or RLConfig()

    def step_reward(
        self,
        state_before: dict,
        state_after: dict,
        step: Optional[StepRecord] = None,
        trajectory: Optional[list[StepRecord]] = None,
    ) -> float:
        trajectory = trajectory or []
        reward = 0.0

        reward += self._progress_reward(state_before, state_after)
        reward += self._info_gain_reward(state_before, state_after)

        if step:
            reward -= self._cost_penalty(step)
            if step.error:
                reward -= self._error_penalty(step)
            if self._is_novel_action(state_before, step):
                reward += self.config.exploration_bonus

            tokens = step.tokens_in + step.tokens_out
            if tokens > 0:
                progress = self._progress_reward(state_before, state_after)
                progress_per_token = progress / tokens
                if progress_per_token > self.config.efficiency_threshold:
                    reward += self.config.efficiency_bonus

        return round(reward, 4)

    def trajectory_return(self, steps: list[StepRecord], states: list[dict]) -> float:
        if not steps or len(states) < 2:
            return 0.0
        returns = 0.0
        gamma = self.config.discount_factor
        for t, step in enumerate(steps):
            s_before = states[t] if t < len(states) else states[-1]
            s_after = states[t + 1] if t + 1 < len(states) else states[-1]
            r = self.step_reward(s_before, s_after, step, steps)
            returns += (gamma ** t) * r
        return round(returns, 4)

    def _progress_reward(self, before: dict, after: dict) -> float:
        before_stage = _investigation_stage_value(before.get("investigation_stage", "not_started"), self.config)
        after_stage = _investigation_stage_value(after.get("investigation_stage", "not_started"), self.config)
        stage_progress = (after_stage - before_stage) * self.config.progress_reward_scale

        before_health = _health_value(before.get("service_health", "unknown"), self.config)
        after_health = _health_value(after.get("service_health", "unknown"), self.config)
        health_progress = (after_health - before_health) * self.config.progress_reward_scale

        return stage_progress + health_progress

    def _info_gain_reward(self, before: dict, after: dict) -> float:
        before_uncertainty = before.get("uncertainty_level", 1.0)
        after_uncertainty = after.get("uncertainty_level", 1.0)
        reduction = before_uncertainty - after_uncertainty
        return max(0.0, reduction * 0.5)

    def _cost_penalty(self, step: StepRecord) -> float:
        return self.config.cost_penalty_scale * (step.tokens_in + step.tokens_out) * 0.0001

    def _error_penalty(self, step: StepRecord) -> float:
        return self.config.error_penalty_scale * 0.3

    def _is_novel_action(self, state: dict, step: StepRecord) -> bool:
        used_tools = state.get("tools_used", [])
        return step.tool_name not in used_tools


class ShapedRewardModel(DenseRewardModel):
    def potential(self, state: dict) -> float:
        p = 0.0
        p += _investigation_stage_value(state.get("investigation_stage", "not_started"), self.config)
        p += _health_value(state.get("service_health", "unknown"), self.config)
        p -= state.get("uncertainty_level", 1.0) * self.config.uncertainty_penalty_scale
        return p

    def shaping_reward(self, state: dict, next_state: dict) -> float:
        gamma = self.config.shaping_discount
        return gamma * self.potential(next_state) - self.potential(state)

    def step_reward(
        self,
        state_before: dict,
        state_after: dict,
        step: Optional[StepRecord] = None,
        trajectory: Optional[list[StepRecord]] = None,
    ) -> float:
        base = super().step_reward(state_before, state_after, step, trajectory)
        shaping = self.shaping_reward(state_before, state_after)
        return round(base + shaping, 4)


class InverseRLRewardModel:
    def __init__(self, n_features: int = 4, learning_rate: float = 0.01):
        self.n_features = n_features
        self.lr = learning_rate
        self.weights: list[float] = [1.0 / n_features] * n_features

    def learn_from_demonstrations(
        self,
        expert_trajectories: list[tuple[list[dict], list[StepRecord]]],
        n_iterations: int = 100,
    ) -> list[float]:
        if not expert_trajectories:
            return self.weights

        expert_features = self._compute_feature_expectations(expert_trajectories)

        for _ in range(n_iterations):
            policy_features = self._simulate_policy_features(expert_trajectories)
            grad = [ef - pf for ef, pf in zip(expert_features, policy_features)]
            self.weights = [
                max(0.0, w + self.lr * g)
                for w, g in zip(self.weights, grad)
            ]
            total = sum(self.weights)
            if total > 0:
                self.weights = [w / total for w in self.weights]

        return list(self.weights)

    def reward(self, state: dict, action: str) -> float:
        features = self._extract_features(state, action)
        return sum(w * f for w, f in zip(self.weights, features))

    def _compute_feature_expectations(
        self, trajectories: list[tuple[list[dict], list[StepRecord]]]
    ) -> list[float]:
        n = len(trajectories)
        if n == 0:
            return [0.0] * self.n_features
        expectations = [0.0] * self.n_features
        for states, steps in trajectories:
            for i, s in enumerate(states):
                action = steps[i].tool_name if i < len(steps) else ""
                feats = self._extract_features(s, action)
                for j, f in enumerate(feats):
                    expectations[j] += f
        return [e / n for e in expectations]

    def _simulate_policy_features(
        self, trajectories: list[tuple[list[dict], list[StepRecord]]]
    ) -> list[float]:
        return self._compute_feature_expectations(trajectories)

    def _extract_features(self, state: dict, action: str) -> list[float]:
        return [
            1.0 if state.get("service_health") == "healthy" else 0.0,
            1.0 if state.get("investigation_stage") == "resolved" else 0.0,
            float(state.get("steps_taken", 0)) / max(state.get("steps_taken", 1), 1),
            1.0 / (state.get("uncertainty_level", 1.0) + 0.01),
        ][:self.n_features]
