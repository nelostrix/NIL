from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Policy:
    temperature: float = 0.0
    action_weights: dict[str, float] = field(default_factory=dict)
    in_context_examples: list[dict] = field(default_factory=list)
    system_prompt_template: str = ""

    def act(self, state: dict, task: str, available_actions: list[str] | None = None) -> str:
        available_actions = available_actions or list(self.action_weights.keys())
        if not available_actions:
            return "explore"

        import random
        if self.temperature == 0.0:
            return max(available_actions, key=lambda a: self.action_weights.get(a, 0.0))

        weights = [self.action_weights.get(a, 1.0) for a in available_actions]
        total = sum(weights)
        if total == 0:
            probs = [1.0 / len(available_actions)] * len(available_actions)
        else:
            probs = [w / total for w in weights]
        return random.choices(available_actions, weights=probs, k=1)[0]

    def log_probability(self, action: str) -> float:
        import math
        total = sum(self.action_weights.values()) or 1.0
        prob = self.action_weights.get(action, 0.0) / total
        return math.log(max(prob, 1e-10))

    def log_probability_of_trace(self, trace: dict) -> float:
        log_prob = 0.0
        for step in trace.get("steps", []):
            action = step.get("tool_name", "")
            if action:
                log_prob += self.log_probability(action)
        return log_prob / max(len(trace.get("steps", [])), 1)

    def update_weight(self, action: str, delta: float) -> None:
        self.action_weights[action] = self.action_weights.get(action, 0.0) + delta

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "action_weights": dict(self.action_weights),
            "system_prompt_template": self.system_prompt_template,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Policy:
        return cls(
            temperature=d.get("temperature", 0.0),
            action_weights=d.get("action_weights", {}),
            system_prompt_template=d.get("system_prompt_template", ""),
        )


class BehavioralCloning:
    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 10):
        self.lr = learning_rate
        self.n_epochs = n_epochs

    def clone(self, demonstrations: list[dict], policy: Optional[Policy] = None) -> Policy:
        policy = policy or Policy()

        action_counts: dict[str, int] = defaultdict(int)
        total_steps = 0
        for trace in demonstrations:
            for step in trace.get("steps", []):
                action = step.get("tool_name", "")
                if action:
                    action_counts[action] += 1
                    total_steps += 1

        if total_steps == 0:
            return policy

        for _ in range(self.n_epochs):
            for action, count in action_counts.items():
                prob = count / total_steps
                current = policy.action_weights.get(action, 0.0)
                policy.action_weights[action] = current + self.lr * (prob - current)

        total = sum(policy.action_weights.values())
        if total > 0:
            policy.action_weights = {
                k: v / total for k, v in policy.action_weights.items()
            }

        return policy
