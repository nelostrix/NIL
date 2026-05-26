from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.models.belief import BeliefTracker
from nil.models.learning import ModelUpdate, OnlineModelLearner
from nil.models.observations import ObservationModel
from nil.models.planners import HeuristicPolicy, MCTSPlanner, PolicyFn, ValuePlanner
from nil.models.rewards import RewardLearner, RewardModel
from nil.models.state_space import EnvironmentState, StateKey, StateSpace, StateSpaceLearner
from nil.models.transitions import TransitionModel

logger = logging.getLogger("nil.models.framework")


class EnvironmentModel:
    def __init__(
        self,
        domain: str = "",
        planner_type: str = "mcts",
        gamma: float = 0.95,
        exploration_bonus: float = 0.1,
        mcts_iterations: int = 50,
        value_iteration_threshold: float = 1e-4,
    ):
        self.domain = domain
        self.state_space = StateSpace()
        self.transition_model = TransitionModel()
        self.observation_model = ObservationModel()
        self.reward_model = RewardModel(
            domain=domain,
            exploration_bonus=exploration_bonus,
        )
        self.reward_learner = RewardLearner()
        self.state_learner = StateSpaceLearner()
        self.belief_tracker = BeliefTracker(self.state_space)
        self.learner = OnlineModelLearner(
            state_learner=self.state_learner,
            transition_model=self.transition_model,
            observation_model=self.observation_model,
            reward_learner=self.reward_learner,
            state_space=self.state_space,
        )
        self.planner_type = planner_type
        self.gamma = gamma
        self.mcts_iterations = mcts_iterations
        self.value_iteration_threshold = value_iteration_threshold
        self._planner: ValuePlanner | MCTSPlanner | HeuristicPolicy | None = None
        self._current_policy: dict[StateKey, str] = {}
        self._total_traces_ingested: int = 0
        self._last_update: ModelUpdate | None = None

    @property
    def total_traces(self) -> int:
        return self._total_traces_ingested

    @property
    def state_count(self) -> int:
        return self.state_space.size

    def ingest(self, trace: dict) -> ModelUpdate:
        update = self.learner.update(trace)
        self._total_traces_ingested += 1
        self._last_update = update

        if self._total_traces_ingested >= 3:
            self.reward_model = self.reward_learner.learn(domain=self.domain)
            self._replan()

        return update

    def ingest_batch(self, traces: list[dict]) -> ModelUpdate:
        combined = ModelUpdate()
        for t in traces:
            u = self.ingest(t)
            combined.new_states_discovered += u.new_states_discovered
            combined.transitions_added += u.transitions_added
            combined.observations_added += u.observations_added
        combined.model_refined = True
        return combined

    def get_belief(self, task: str = "", context: dict = None) -> BeliefTracker:
        bt = self.learner.get_belief_tracker()
        if task:
            bt.incorporate_task(task, context)
        return bt

    def suggest_action(
        self,
        belief_tracker: BeliefTracker | None = None,
        task: str = "",
        context: dict = None,
    ) -> dict:
        if belief_tracker is None:
            belief_tracker = self.get_belief(task, context)

        mls = belief_tracker.most_likely_state
        if not mls:
            return {"action": "explore", "confidence": 0.0}

        policy = self._get_policy()

        action = policy.get_action(mls)
        confidence = belief_tracker.most_likely_state_probability

        outcomes = self.transition_model.expected_outcome(mls, action)

        return {
            "action": action,
            "confidence": round(confidence, 3),
            "mls": mls,
            "mls_probability": confidence,
            "expected_next_state": outcomes.next_state or "",
            "uncertainty": round(belief_tracker.normalized_uncertainty, 3),
            "top_beliefs": belief_tracker.top_beliefs(3),
        }

    def evaluate_state(self, state: StateKey) -> float:
        return self.reward_model.expected(state, "")

    def state_info(self, state: EnvironmentState) -> dict:
        key = state.to_key()
        uncertainty = self.transition_model.uncertainty(key, "")
        actions = self.transition_model.known_actions(key)
        return {
            "key": key,
            "uncertainty": round(uncertainty, 3),
            "known_actions": actions,
            "known_transitions": len(actions),
        }

    def summarize(self) -> dict:
        return {
            "domain": self.domain,
            "total_traces": self._total_traces_ingested,
            "states": self.state_space.size,
            "transitions": self.transition_model.total_transitions,
            "observations": self.observation_model.total_observations,
            "total_reward_observations": len(self.reward_learner.observations),
            "planner_type": self.planner_type,
            "last_update": self._last_update.to_dict() if self._last_update else None,
        }

    def _get_policy(self) -> ValuePlanner | MCTSPlanner | HeuristicPolicy:
        if self._planner is None:
            self._replan()
        return self._planner

    def _replan(self) -> None:
        states = self.state_space.all_keys()
        terminal_states = set()

        if not states:
            self._planner = HeuristicPolicy.action_for_stage("not_started")
            return

        def reward_fn(s: StateKey, a: str) -> float:
            return self.reward_model.expected(s, a)

        if self.planner_type == "value_iteration":
            planner = ValuePlanner(
                gamma=self.gamma,
                theta=self.value_iteration_threshold,
            )
            self._current_policy = planner.plan(
                states=states,
                transition_model=self.transition_model,
                reward_fn=reward_fn,
                terminal_states=terminal_states,
            )
            self._planner = planner
        elif self.planner_type == "mcts":
            planner = MCTSPlanner(
                iterations=self.mcts_iterations,
                gamma=self.gamma,
            )
            self._current_policy = planner.plan(
                states=states,
                transition_model=self.transition_model,
                reward_fn=reward_fn,
                terminal_states=terminal_states,
            )
            self._planner = planner
        else:
            self._planner = HeuristicPolicy.action_for_stage("gathering_data")

    def to_dict(self) -> dict:
        return {
            "state_space": self.state_space.to_dict(),
            "transition_model": self.transition_model.to_dict(),
            "observation_model": self.observation_model.to_dict(),
            "reward_model": self.reward_model.to_dict(),
            "domain": self.domain,
            "planner_type": self.planner_type,
            "total_traces": self._total_traces_ingested,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EnvironmentModel:
        model = cls(domain=d.get("domain", ""), planner_type=d.get("planner_type", "mcts"))
        model.state_space = StateSpace.from_dict(d.get("state_space", {}))
        model.transition_model = TransitionModel.from_dict(d.get("transition_model", {}))
        model.observation_model = ObservationModel.from_dict(d.get("observation_model", {}))
        model.reward_model = RewardModel.from_dict(d.get("reward_model", {}))
        model._total_traces_ingested = d.get("total_traces", 0)
        model.belief_tracker = model.learner.get_belief_tracker()
        return model


class ModelAwareAgent:
    def __init__(
        self,
        env_model: EnvironmentModel | None = None,
        domain: str = "",
        planner_type: str = "mcts",
    ):
        self.env_model = env_model or EnvironmentModel(
            domain=domain, planner_type=planner_type
        )
        self._current_task: str = ""
        self._action_history: list[dict] = []
        self._current_belief: BeliefTracker | None = None

    def start_task(self, task: str, context: dict = None) -> dict:
        self._current_task = task
        self._current_belief = self.env_model.get_belief(task, context)
        suggestion = self.env_model.suggest_action(
            self._current_belief, task, context
        )
        self._current_suggestion = suggestion
        return suggestion

    def observe_outcome(self, step: dict) -> dict:
        action = step.get("tool_name", "") or step.get("step_type", "unknown")
        result = step.get("result", "")
        error = step.get("error", "")

        observation = result or error or f"executed_{action}"
        observation_text = str(observation)[:100]

        if self._current_belief:
            self._current_belief.update(action, observation_text)

        self._action_history.append({
            "step": step,
            "observation": observation_text,
            "belief": self._current_belief.to_dict() if self._current_belief else {},
        })

        ucb = self.env_model.transition_model.uncertainty(
            self._current_belief.most_likely_state or "",
            action,
        ) if self._current_belief else 1.0

        self._current_suggestion = {
            "action": action,
            "confidence": round(self._current_belief.most_likely_state_probability, 3) if self._current_belief else 0.0,
            "uncertainty": round(
                self._current_belief.normalized_uncertainty if self._current_belief else 1.0, 3
            ),
            "ucb": round(ucb, 3),
        }
        return self._current_suggestion

    def finish_task(self, trace: dict) -> ModelUpdate:
        update = self.env_model.ingest(trace)
        self._current_task = ""
        self._current_belief = None
        return update
