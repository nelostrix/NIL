from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nil.benchmarks.base import (
    Agent,
    ForgettingCurve,
    ForgettingPoint,
    InterferenceResult,
    Outcome,
)


class MemoryRetentionBenchmark:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def forgetting_curve(
        self,
        agent: Agent,
        learn_task: Callable,
        test_task: Callable,
        delays: list[int] = None,
    ) -> ForgettingCurve:
        delays = delays or [1, 5, 10, 25, 50, 100]

        for _ in range(5):
            agent.run(learn_task())

        results: list[ForgettingPoint] = []
        for delay in delays:
            for _ in range(delay):
                agent.run(self._distractor_task())

            test_outcomes = [agent.run(test_task()) for _ in range(3)]
            accuracy = sum(o.success for o in test_outcomes) / max(len(test_outcomes), 1)
            results.append(ForgettingPoint(delay=delay, accuracy=accuracy))

        half_life = self._estimate_half_life(results)
        retention_at_100 = results[-1].accuracy if len(results) >= 6 else None

        return ForgettingCurve(
            task=learn_task.__name__ if hasattr(learn_task, "__name__") else "learn",
            points=results,
            half_life=half_life,
            retention_at_100=retention_at_100 or 0.0,
        )

    def interference_test(
        self,
        agent: Agent,
        pattern_a: Callable,
        pattern_b: Callable,
        test: Callable,
    ) -> InterferenceResult:
        for _ in range(10):
            agent.run(pattern_a())
        accuracy_a_before = self._measure_accuracy(agent, test, n=5)

        for _ in range(10):
            agent.run(pattern_b())
        accuracy_b = self._measure_accuracy(agent, lambda: pattern_b(), n=5)

        accuracy_a_after = self._measure_accuracy(agent, test, n=5)

        interference = accuracy_a_before - accuracy_a_after

        return InterferenceResult(
            accuracy_a_before=accuracy_a_before,
            accuracy_a_after=accuracy_a_after,
            interference_drop=interference,
            interference_ratio=interference / accuracy_a_before if accuracy_a_before > 0 else 1.0,
        )

    def _distractor_task(self) -> str:
        return f"Perform unrelated distractor task_{self.rng.randint(0, 100)}"

    def _estimate_half_life(self, points: list[ForcingPoint]) -> float:
        if len(points) < 2:
            return float("inf")
        initial = points[0].accuracy
        if initial == 0:
            return float("inf")
        half = initial / 2
        for i in range(1, len(points)):
            if points[i].accuracy <= half:
                prev = points[i - 1]
                curr = points[i]
                if curr.accuracy == prev.accuracy:
                    return float(curr.delay)
                t = prev.delay + (half - prev.accuracy) * (curr.delay - prev.delay) / (curr.accuracy - prev.accuracy)
                return t
        return float(points[-1].delay * 2)

    def _measure_accuracy(self, agent: Agent, task_fn: Callable, n: int = 5) -> float:
        outcomes = [agent.run(task_fn()) for _ in range(n)]
        return sum(o.success for o in outcomes) / max(len(outcomes), 1)
