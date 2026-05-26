from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.models.belief import BeliefTracker
from nil.models.observations import ObservationModel
from nil.models.rewards import RewardLearner, RewardModel, RewardObservation
from nil.models.state_space import EnvironmentState, StateSpace, StateSpaceLearner
from nil.models.transitions import TransitionModel

logger = logging.getLogger("nil.models.learning")


@dataclass
class ModelUpdate:
    new_states_discovered: int = 0
    transitions_added: int = 0
    observations_added: int = 0
    model_refined: bool = False
    uncertainty_reduction: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dataclass_fields__.items()}


class OnlineModelLearner:
    def __init__(
        self,
        state_learner: StateSpaceLearner | None = None,
        transition_model: TransitionModel | None = None,
        observation_model: ObservationModel | None = None,
        reward_learner: RewardLearner | None = None,
        state_space: StateSpace | None = None,
    ):
        self.state_learner = state_learner or StateSpaceLearner()
        self.transition_model = transition_model or TransitionModel()
        self.observation_model = observation_model or ObservationModel()
        self.reward_learner = reward_learner or RewardLearner()
        self.state_space = state_space or StateSpace()
        self._total_updates = 0

    @property
    def total_updates(self) -> int:
        return self._total_updates

    def update(self, trace: dict) -> ModelUpdate:
        start = time.time()
        update = ModelUpdate()

        pre_state_count = self.state_space.size

        states_from_trace = self.state_learner.discover_states(
            [trace], existing_space=self.state_space
        )
        update.new_states_discovered = self.state_space.size - pre_state_count

        trace_states = self.state_learner._extract_transition_states(trace)
        trace_state_keys = [s.to_key() for s in trace_states]
        trace_state_ids = [self.state_space.get_or_create(k) for k in trace_state_keys]

        steps = trace.get("steps", [])
        for i in range(min(len(trace_state_ids) - 1, len(steps))):
            if i < len(steps):
                step = steps[i]
                action = step.get("tool_name", "") or step.get("step_type", "unknown")
                s_from = self.state_space.get_key(trace_state_ids[i])
                s_to = self.state_space.get_key(trace_state_ids[i + 1])
                if s_from and s_to:
                    self.transition_model.record(s_from, action, s_to)
                    update.transitions_added += 1

                    observation_text = self._extract_observation(step)
                    self.observation_model.record(s_to, action, observation_text)
                    update.observations_added += 1

        success = trace.get("outcome_success", False)

        if trace_state_keys:
            reward_obs = RewardObservation(
                from_state=trace_state_keys[0] if trace_state_keys else "",
                action=trace.get("step_type", "unknown"),
                to_state=trace_state_keys[-1] if trace_state_keys else "",
                cost=len(steps) * 0.1,
                duration=sum(
                    s.get("duration_ms", 0.0) for s in steps if isinstance(s, dict)
                ),
                success=bool(success),
                error=str(trace.get("error", "") or ""),
            )
            self.reward_learner.observe(reward_obs)

        update.model_refined = self._total_updates > 0
        self._total_updates += 1
        update.duration_seconds = time.time() - start
        return update

    def update_batch(self, traces: list[dict]) -> ModelUpdate:
        combined = ModelUpdate()
        for t in traces:
            u = self.update(t)
            combined.new_states_discovered += u.new_states_discovered
            combined.transitions_added += u.transitions_added
            combined.observations_added += u.observations_added
            combined.duration_seconds += u.duration_seconds
        combined.model_refined = True
        return combined

    def get_belief_tracker(self) -> BeliefTracker:
        return BeliefTracker(
            state_space=self.state_space,
            transition_model=self.transition_model,
            observation_model=self.observation_model,
        )

    def _extract_observation(self, step: dict) -> str:
        result = step.get("result", "")
        if isinstance(result, str) and result:
            return result[:100]
        error = step.get("error", "")
        if error:
            return f"error:{str(error)[:80]}"
        return f"executed_{step.get('tool_name', step.get('step_type', 'unknown'))}"

    def to_dict(self) -> dict:
        return {
            "state_space": self.state_space.to_dict(),
            "transition_model": self.transition_model.to_dict(),
            "observation_model": self.observation_model.to_dict(),
            "total_updates": self._total_updates,
        }
