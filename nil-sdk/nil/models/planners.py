from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nil.models.belief import BeliefTracker
from nil.models.state_space import EnvironmentState, StateKey
from nil.models.transitions import ActionKey, TransitionModel

logger = logging.getLogger("nil.models.planners")

PolicyFn = Callable[[StateKey], str]


@dataclass
class ValuePlanner:
    gamma: float = 0.95
    theta: float = 1e-4
    max_iterations: int = 1000
    _values: dict[StateKey, float] = field(default_factory=dict)
    _policy: dict[StateKey, str] = field(default_factory=dict)

    def plan(
        self,
        states: list[StateKey],
        transition_model: TransitionModel,
        reward_fn: Callable[[StateKey, str], float],
        terminal_states: set[StateKey] = None,
    ) -> dict[StateKey, str]:
        terminal_states = terminal_states or set()
        self._values = {s: 0.0 for s in states}

        for _ in range(self.max_iterations):
            delta = 0.0
            for s in states:
                if s in terminal_states:
                    continue
                v = self._values[s]
                actions = transition_model.known_actions(s) or ["default"]
                q_values = []
                for a in actions:
                    q = reward_fn(s, a)
                    for s_next, prob in transition_model.predict(s, a):
                        next_val = self._values.get(s_next, 0.0)
                        q += prob * self.gamma * next_val
                    q_values.append((a, q))
                self._values[s] = max(q_values, key=lambda x: x[1])[1] if q_values else 0.0
                delta = max(delta, abs(v - self._values[s]))
            if delta < self.theta:
                break

        for s in states:
            actions = transition_model.known_actions(s) or ["default"]
            best_action = max(
                actions,
                key=lambda a: reward_fn(s, a) + self.gamma * sum(
                    prob * self._values.get(s_next, 0.0)
                    for s_next, prob in transition_model.predict(s, a)
                ),
                default="default",
            )
            self._policy[s] = best_action

        return dict(self._policy)

    def get_action(self, state: StateKey) -> str:
        return self._policy.get(state, "default")

    def get_value(self, state: StateKey) -> float:
        return self._values.get(state, 0.0)


@dataclass
class MCTSNode:
    state: StateKey
    action: str = ""
    visits: int = 0
    total_value: float = 0.0
    children: dict[str, MCTSNode] = field(default_factory=dict)
    parent: Optional[MCTSNode] = None
    untried_actions: list[str] = field(default_factory=list)

    def ucb1(self, exploration_constant: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.total_value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits + 1) / (self.visits + 1e-6)
        )
        return exploitation + exploration

    def best_child(self, exploration_constant: float = 1.41) -> MCTSNode:
        return max(
            self.children.values(),
            key=lambda c: c.ucb1(exploration_constant),
        )


@dataclass
class MCTSPlanner:
    iterations: int = 50
    exploration_constant: float = 1.41
    gamma: float = 0.95
    max_depth: int = 20
    _policy: dict[StateKey, str] = field(default_factory=dict)

    def plan(
        self,
        states: list[StateKey],
        transition_model: TransitionModel,
        reward_fn: Callable[[StateKey, str], float],
        terminal_states: set[StateKey] = None,
    ) -> dict[StateKey, str]:
        terminal_states = terminal_states or set()
        self._policy = {}

        for s in states:
            if s in terminal_states:
                self._policy[s] = "_terminal"
                continue
            root = MCTSNode(state=s)
            actions = transition_model.known_actions(s) or ["default"]
            root.untried_actions = list(actions)

            for _ in range(self.iterations):
                node = self._select(root)
                node = self._expand(node)
                value = self._simulate(node, transition_model, reward_fn, terminal_states, 0)
                self._backpropagate(node, value)

            if root.children:
                self._policy[s] = max(
                    root.children.values(),
                    key=lambda c: c.visits,
                ).action
            else:
                self._policy[s] = "default"

        return dict(self._policy)

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.untried_actions == [] and node.children:
            node = node.best_child(self.exploration_constant)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if not node.untried_actions:
            return node
        action = node.untried_actions.pop()
        child = MCTSNode(state=node.state, action=action, parent=node)
        node.children[action] = child
        return child

    def _simulate(
        self,
        node: MCTSNode,
        transition_model: TransitionModel,
        reward_fn: Callable[[StateKey, str], float],
        terminal_states: set[StateKey],
        depth: int,
    ) -> float:
        if depth >= self.max_depth:
            return 0.0
        if node.state in terminal_states:
            return 0.0

        actions = transition_model.known_actions(node.state) or ["default"]
        if not actions:
            return reward_fn(node.state, "default")

        action = actions[0]
        predictions = transition_model.predict(node.state, action)
        if not predictions:
            return reward_fn(node.state, action)

        s_next = predictions[0][0]
        immediate = reward_fn(node.state, action)
        return immediate + self.gamma * self._simulate(
            MCTSNode(state=s_next),
            transition_model,
            reward_fn,
            terminal_states,
            depth + 1,
        )

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        while node:
            node.visits += 1
            node.total_value += value
            node = node.parent

    def get_action(self, state: StateKey) -> str:
        return self._policy.get(state, "default")


@dataclass
class HeuristicPolicy:
    rules: list[tuple[Callable[[StateKey], bool], str]] = field(default_factory=list)

    def add_rule(self, condition: Callable[[StateKey], bool], action: str) -> None:
        self.rules.append((condition, action))

    def plan(
        self,
        states: list[StateKey],
        transition_model: TransitionModel | None = None,
        reward_fn: Callable | None = None,
        terminal_states: set[StateKey] = None,
    ) -> dict[StateKey, str]:
        policy = {}
        for s in states:
            action = self.get_action(s)
            policy[s] = action
        return policy

    def get_action(self, state: StateKey) -> str:
        for condition, action in self.rules:
            if condition(state):
                return action
        return "explore"

    @staticmethod
    def action_for_stage(stage: str) -> HeuristicPolicy:
        policy = HeuristicPolicy()
        policy.add_rule(
            lambda s: '"not_started"' in s or "gathering_data" in s,
            "investigate",
        )
        policy.add_rule(
            lambda s: "analyzing" in s,
            "analyze",
        )
        policy.add_rule(
            lambda s: "diagnosed" in s or "fixing" in s,
            "execute_fix",
        )
        policy.add_rule(
            lambda s: "verifying" in s,
            "verify",
        )
        policy.add_rule(
            lambda s: '"degraded"' in s or '"down"' in s,
            "investigate",
        )
        return policy
