from __future__ import annotations

import random
from collections import defaultdict

from nil.benchmarks.base import (
    Agent,
    Outcome,
    RecoveryReport,
    TestResult,
)


class RecoveryBenchmark:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def run(self, agent: Agent) -> RecoveryReport:
        results = {}
        results["tool_failure"] = self._test_tool_failure(agent)
        results["tool_degradation"] = self._test_tool_degradation(agent)
        results["distribution_shift"] = self._test_distribution_shift(agent)
        results["delayed_rewards"] = self._test_delayed_rewards(agent)

        return RecoveryReport(
            overall_recovery_score=self._composite(results),
            per_test=results,
        )

    def _test_tool_failure(self, agent: Agent) -> TestResult:
        outcomes = []
        for _ in range(20):
            task = "Investigate API error and resolve the issue"
            outcome = agent.run(task)
            outcomes.append(outcome)

        completion_rate = sum(o.success for o in outcomes) / max(len(outcomes), 1)
        tool_switch_latency = self._avg_steps_before_tool_switch(outcomes)
        unnecessary_retries = sum(self._count_unnecessary_retries(o) for o in outcomes)

        return TestResult(
            completion_rate=completion_rate,
            tool_switch_latency=tool_switch_latency,
            unnecessary_retries=unnecessary_retries,
        )

    def _test_tool_degradation(self, agent: Agent) -> TestResult:
        outcomes = []
        for _ in range(20):
            task = f"Diagnose system issue type_{self.rng.randint(0, 3)}"
            outcome = agent.run(task)
            outcomes.append(outcome)

        return TestResult(
            completion_rate=sum(o.success for o in outcomes) / max(len(outcomes), 1),
        )

    def _test_distribution_shift(self, agent: Agent) -> TestResult:
        phase1_tasks = [f"Perform type_a task number {i}" for i in range(20)]
        phase2_tasks = [f"Perform type_b task number {i}" for i in range(20)]

        phase1_scores = []
        for task in phase1_tasks:
            phase1_scores.append(agent.run(task).success)

        phase2_scores = []
        for task in phase2_tasks:
            phase2_scores.append(agent.run(task).success)

        plateau_1 = sum(phase1_scores[-5:]) / 5 if phase1_scores else 0.0
        drop = plateau_1 - (phase2_scores[0] if phase2_scores else 0.0)
        recovery_runs = self._runs_to_recover(phase2_scores, 0.8 * plateau_1)
        post_shift = sum(phase2_scores[-5:]) / 5 if phase2_scores else 0.0

        return TestResult(
            pre_shift_accuracy=plateau_1,
            shift_drop=drop,
            recovery_runs=recovery_runs,
            post_shift_accuracy=post_shift,
        )

    def _test_delayed_rewards(self, agent: Agent) -> TestResult:
        outcomes = []
        for _ in range(20):
            task = f"Execute multi-step diagnostic task_{self.rng.randint(0, 5)}"
            outcome = agent.run(task)
            outcomes.append(outcome)

        return TestResult(
            completion_rate=sum(o.success for o in outcomes) / max(len(outcomes), 1),
        )

    def _avg_steps_before_tool_switch(self, outcomes: list[Outcome]) -> float:
        steps_before_switch = []
        for o in outcomes:
            seen_tools = set()
            for i, s in enumerate(o.steps):
                tool = s.get("tool_name", "")
                if tool and tool in seen_tools:
                    steps_before_switch.append(i)
                    break
                seen_tools.add(tool)
        if steps_before_switch:
            return sum(steps_before_switch) / len(steps_before_switch)
        return 0.0

    def _count_unnecessary_retries(self, outcome: Outcome) -> int:
        tool_counts = defaultdict(int)
        for s in outcome.steps:
            tool = s.get("tool_name", "")
            if tool and s.get("error"):
                tool_counts[tool] += 1
        return sum(c - 1 for c in tool_counts.values() if c > 1)

    def _runs_to_recover(self, scores: list[float], threshold: float) -> int:
        for i, s in enumerate(scores):
            if s >= threshold:
                return i + 1
        return len(scores)

    def _composite(self, results: dict[str, TestResult]) -> float:
        if not results:
            return 0.0
        return sum(r.score for r in results.values()) / len(results)
