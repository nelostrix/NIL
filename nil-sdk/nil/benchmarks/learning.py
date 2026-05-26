from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nil.benchmarks.base import (
    Agent,
    ForgettingEvent,
    LearningReport,
    Outcome,
)

_SEED = 42
_rng = random.Random(_SEED)


class LearningVelocity:
    def compute(self, run_scores: list[float]) -> LearningReport:
        n = len(run_scores)
        if n < 3:
            return LearningReport(insufficient_data=True)

        initial = run_scores[0]

        window = min(30, n)
        early_scores = run_scores[:window]
        x = list(range(window))
        slope, _ = self._linear_regression(x, early_scores)

        plateau_window = min(10, max(1, n // 5))
        plateau = sum(run_scores[-plateau_window:]) / plateau_window

        threshold = 0.9 * plateau
        runs_to_plateau = n
        for i, score in enumerate(run_scores):
            if score >= threshold:
                runs_to_plateau = i + 1
                break

        total_improvement = plateau - initial
        sample_efficiency = total_improvement / max(n, 1)

        drops = self._detect_forgetting_events(run_scores)

        return LearningReport(
            initial_accuracy=round(initial, 4),
            final_accuracy=round(plateau, 4),
            learning_velocity=round(slope, 6),
            runs_to_plateau=runs_to_plateau,
            sample_efficiency=round(sample_efficiency, 6),
            plateau_accuracy=round(plateau, 4),
            forgetting_events=drops,
            total_runs=n,
        )

    def _detect_forgetting_events(self, scores: list[float]) -> list[ForgettingEvent]:
        events = []
        i = 0
        while i < len(scores) - 3:
            if scores[i + 1] < scores[i] * 0.85:
                drop_start = i
                drop_end = i + 1
                while drop_end < len(scores) and scores[drop_end] < scores[i] * 0.85:
                    drop_end += 1
                if drop_end - drop_start >= 3:
                    events.append(ForgettingEvent(
                        run=drop_start,
                        drop=scores[i] - min(scores[drop_start:drop_end]),
                        duration=drop_end - drop_start,
                    ))
                i = drop_end
            else:
                i += 1
        return events

    @staticmethod
    def _linear_regression(x: list[float], y: list[float]) -> tuple[float, float]:
        n = len(x)
        if n == 0:
            return 0.0, 0.0
        sx = sum(x)
        sy = sum(y)
        sxx = sum(xi * xi for xi in x)
        sxy = sum(xi * yi for xi, yi in zip(x, y))
        denom = n * sxx - sx * sx
        if denom == 0:
            return 0.0, sy / n
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        return slope, intercept


class _BaseBenchmark:
    def _measure_accuracy(self, agent: Agent, tasks: list[str], n: int = 10) -> float:
        outcomes = [agent.run(task) for task in tasks[:n]]
        return sum(o.success for o in outcomes) / max(len(outcomes), 1)


@dataclass
class ToolSelectionBenchmark:
    name: str = "learn_which_tool_works"
    description: str = "Agent must learn which of 5 tools solves each task type"
    n_tasks: int = 50
    n_tools: int = 5
    n_adaptation_runs: int = 30
    seed: int = 42

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def run(self, agent: Agent) -> LearningReport:
        correct_tool = self.rng.randint(0, self.n_tools - 1)
        scores = []
        for i in range(self.n_tasks):
            task = f"select correct tool to fix issue type_{i % 3}"
            outcome = agent.run(task)
            tools_used = [s.get("tool_name", "") for s in outcome.steps if s.get("tool_name")]

            correct = 0.0
            if outcome.success:
                if tools_used and tools_used[0] == f"tool_{correct_tool}":
                    correct = 1.0
                elif tools_used:
                    correct = 0.5
            scores.append(correct)

        return LearningVelocity().compute(scores)


@dataclass
class SequentialReasoningBenchmark:
    name: str = "learn_multi_step_sequence"
    description: str = "Agent must learn the correct 5-step tool sequence"
    n_tasks: int = 50
    sequence_length: int = 5
    n_alternatives: int = 3
    seed: int = 42

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def run(self, agent: Agent) -> LearningReport:
        correct_seq = [f"step_{i}" for i in range(self.sequence_length)]
        scores = []
        for i in range(self.n_tasks):
            task = f"execute the correct sequence to resolve incident type_{i % 4}"
            outcome = agent.run(task)
            seq = outcome.tool_sequence
            if not seq:
                scores.append(0.0)
                continue
            matches = sum(
                1 for j, s in enumerate(seq[:self.sequence_length])
                if j < len(correct_seq) and s == correct_seq[j]
            )
            scores.append(matches / self.sequence_length)

        return LearningVelocity().compute(scores)


@dataclass
class ContextDependenceBenchmark:
    name: str = "learn_context_dependent_action"
    description: str = "Correct action depends on context the agent must learn"
    n_tasks: int = 50
    n_contexts: int = 4
    n_adaptation_runs: int = 30
    seed: int = 42

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def run(self, agent: Agent) -> LearningReport:
        context_map = {i: f"correct_action_{i}" for i in range(self.n_contexts)}
        scores = []
        for i in range(self.n_tasks):
            ctx = i % self.n_contexts
            task = f"handle {ctx=}: resolve the issue appropriately"
            outcome = agent.run(task)
            tools = outcome.tool_sequence
            if tools and context_map[ctx] in tools:
                scores.append(1.0)
            elif outcome.success:
                scores.append(0.5)
            else:
                scores.append(0.0)

        return LearningVelocity().compute(scores)


class LearningCurveBenchmarks:
    def __init__(self):
        self.benchmarks = {
            "tool_selection": ToolSelectionBenchmark(),
            "sequential_reasoning": SequentialReasoningBenchmark(),
            "context_dependence": ContextDependenceBenchmark(),
        }

    def run(self, agent: Agent) -> dict[str, LearningReport]:
        results = {}
        for name, bench in self.benchmarks.items():
            results[name] = bench.run(agent)
        return results
