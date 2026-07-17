from __future__ import annotations

import random
from dataclasses import dataclass, field

from nil.benchmarks.base import (
    Agent,
    HorizonResult,
    Outcome,
    std,
)


class LongHorizonBenchmark:
    task_lengths: list[int] = [10, 25, 50, 100]
    n_tasks_per_length: int = 20

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def run(self, agent: Agent) -> dict[int, HorizonResult]:
        results = {}
        for length in self.task_lengths:
            scores = []
            error_recovery = []
            for _ in range(self.n_tasks_per_length):
                task = self._generate_horizon_task(length)
                outcome = agent.run(task.description)

                step_correctness = self._score_step_accuracy(outcome, task)
                recovery = self._score_error_recovery(outcome, task)

                scores.append(step_correctness)
                error_recovery.append(recovery)

            results[length] = HorizonResult(
                task_length=length,
                mean_accuracy=sum(scores) / max(len(scores), 1),
                std_accuracy=std(scores),
                error_recovery_rate=sum(error_recovery) / max(len(error_recovery), 1),
                completion_rate=sum(1 for s in scores if s > 0.8) / max(len(scores), 1),
            )
        return results

    def _generate_horizon_task(self, length: int) -> _Task:
        steps = [f"step_{i}" for i in range(length)]
        return _Task(
            description=f"Execute a {length}-step procedure to resolve issue",
            required_steps=steps,
        )

    def _score_step_accuracy(self, outcome: Outcome, task: _Task) -> float:
        if not task.required_steps:
            return 0.0
        correct = 0
        for required_step in task.required_steps:
            if self._step_matches(outcome, required_step):
                correct += 1
        return correct / len(task.required_steps)

    def _score_error_recovery(self, outcome: Outcome, task: _Task) -> float:
        if not outcome.had_errors:
            return 1.0

        recovery_score = 0.0
        for error_step in outcome.error_steps:
            subsequent = outcome.steps_after(error_step)
            if self._reached_goal(subsequent, task):
                recovery_score += 1.0
            elif self._changed_strategy(subsequent, error_step):
                recovery_score += 0.7
            elif self._retried_same(subsequent, error_step):
                recovery_score += 0.3

        return recovery_score / max(len(outcome.error_steps), 1)

    def _step_matches(self, outcome: Outcome, step_name: str) -> bool:
        return any(
            s.get("tool_name") == step_name or s.get("step_type") == step_name
            for s in outcome.steps
        )

    def _reached_goal(self, steps: list[dict], task: _Task) -> bool:
        return any(s.get("success") for s in steps)

    def _changed_strategy(self, steps: list[dict], error_step: dict) -> bool:
        if not steps:
            return False
        err_tool = error_step.get("tool_name", "")
        return any(s.get("tool_name") != err_tool for s in steps)

    def _retried_same(self, steps: list[dict], error_step: dict) -> bool:
        if not steps:
            return False
        err_tool = error_step.get("tool_name", "")
        return any(s.get("tool_name") == err_tool for s in steps)


@dataclass
class _Task:
    description: str = ""
    required_steps: list[str] = field(default_factory=list)
