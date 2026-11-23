from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from nil.rl.exploration import ExplorationScheduler
from nil.rl.policy import BehavioralCloning, Policy
from nil.rl.preference import DirectPreferenceOptimizer, EloTrajectoryRating
from nil.rl.replay import ReplayBuffer, Transition
from nil.rl.rewards import DenseRewardModel, StepRecord
from nil.rl.trajectory import TrajectoryScorer
from nil.rl.value import AdvantageEstimator, ValueFunction

@dataclass
class RLConfig:
    algorithm: str = "advantage_weights"
    batch_size: int = 64
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_frequency: int = 5
    use_mcts: bool = False

    def __post_init__(self):
        if self.algorithm not in {"advantage_weights", "bc", "gae"}:
            raise ValueError("Agent policies use advantage_weights/bc; use nil.rl.create for neural PPO and other control algorithms")
        if type(self.batch_size) is not int or self.batch_size < 1 or type(self.update_frequency) is not int or self.update_frequency < 1:
            raise ValueError("batch_size/update_frequency must be positive integers")


class RLEnabledAgent:
    def __init__(self, domain: str = ""):
        self.domain = domain
        self.replay_buffer = ReplayBuffer(max_size=10000)
        self.reward_model = DenseRewardModel()
        self.value_function = ValueFunction(state_dim=4)
        self.advantage_estimator = AdvantageEstimator()
        self.exploration = ExplorationScheduler()
        self.policy = Policy()
        self.trajectory_scorer = TrajectoryScorer()
        self.elo = EloTrajectoryRating()
        self.dpo = DirectPreferenceOptimizer()
        self.bc = BehavioralCloning()

        self.online_rl_config = RLConfig()
        self._current_trajectory: list[StepRecord] = []
        self._current_states: list[dict] = []
        self._step_count: int = 0

    def select_action(self, state: dict, available_actions: list[str] = None) -> str:
        available_actions = available_actions or []

        if self.exploration.should_explore(state):
            return self.exploration.sample_action(available_actions)

        return self.policy.act(state, self.domain, available_actions)

    def on_step_start(self, state: dict) -> None:
        self._current_states.append(dict(state))

    def on_step_complete(self, step: StepRecord, state_before: dict, state_after: dict) -> None:
        self._current_trajectory.append(step)
        self._step_count += 1

        reward = self.reward_model.step_reward(state_before, state_after, step, self._current_trajectory)

        self.replay_buffer.push(Transition(
            state=state_before,
            action=step.tool_name or step.step_type,
            reward=reward,
            next_state=state_after,
            done=state_after.get("is_terminal_success", False) or state_after.get("is_terminal_failure", False),
        ))

        self.exploration.record_action(
            step.tool_name or step.step_type,
            reward,
            success=state_after.get("is_terminal_success", False),
        )

        if self.replay_buffer.size >= self.online_rl_config.batch_size and self._step_count % self.online_rl_config.update_frequency == 0:
            self._update_policy()

    def finish_trajectory(self, trace: dict) -> None:
        scored = self.trajectory_scorer.score(trace)
        trace_id = trace.get("run_id", "")
        if trace_id and scored.correctness > 0:
            self.elo.ratings[trace_id] = scored.composite * 2000

        if scored.composite > 0.7:
            self.bc.clone([trace], self.policy)
        self._current_trajectory.clear()
        self._current_states.clear()

    def _update_policy(self) -> None:
        # GAE requires a contiguous ordered rollout, never shuffled transitions.
        batch = self.replay_buffer.latest(self.online_rl_config.batch_size)

        if self.online_rl_config.algorithm == "bc":
            states = [t.state for t in batch]
            actions = [t.action for t in batch]
            for s, a in zip(states, actions):
                self.policy.update_weight(a, 0.01)

        elif self.online_rl_config.algorithm in ("advantage_weights", "gae"):
            rewards = [t.reward for t in batch]
            values = [self.value_function.predict(t.state) for t in batch]

            advantages = self.advantage_estimator.compute_gae(
                rewards, values,
                gamma=self.online_rl_config.discount_factor,
                lam=self.online_rl_config.gae_lambda,
                terminated=[t.terminated for t in batch], truncated=[t.truncated for t in batch],
                next_values=[self.value_function.predict(t.next_state) for t in batch],
            )

            for t, adv in zip(batch, advantages):
                if adv > 0:
                    self.policy.update_weight(t.action, self.online_rl_config.learning_rate * adv)

            returns = self.advantage_estimator.compute_monte_carlo(
                rewards, gamma=self.online_rl_config.discount_factor,
                terminated=[t.terminated for t in batch], truncated=[t.truncated for t in batch],
                next_values=[self.value_function.predict(t.next_state) for t in batch],
                last_value=self.value_function.predict(batch[-1].next_state) if batch else 0.0,
            )
            states_for_update = [t.state for t in batch]
            self.value_function.update(states_for_update, returns)
        else:
            raise ValueError("Agent action-weight updates are not PPO/CQL; use nil.rl.create for neural control")
