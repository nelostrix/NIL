"""
Autonomous Curriculum — self-directed capability growth.

The agent detects its own weaknesses, generates synthetic training tasks,
practices on past failures, benchmarks itself, and drives capability growth.

Pipeline:
  1. WeaknessAnalyzer identifies skill gaps from trace analysis
  2. SyntheticTaskGenerator creates targeted training tasks
  3. PracticeEngine runs the agent on targeted exercises
  4. BenchmarkSuite measures improvement before/after
  5. CurriculumEngine orchestrates the full cycle
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("nil.training.curriculum")


def _trace_success(t: Any) -> bool:
    if isinstance(t, dict):
        return bool(t.get("outcome_success", False))
    return bool(getattr(t, "outcome_success", False))


# ---------------------------------------------------------------------------
# Phase 6a — Weakness Analysis
# ---------------------------------------------------------------------------


@dataclass
class WeaknessProfile:
    domain: str = ""
    success_rate: float = 1.0
    failing_tools: list[tuple[str, float, int]] = field(default_factory=list)
    avoiding_tools: list[str] = field(default_factory=list)
    weak_task_types: list[tuple[str, float, int]] = field(default_factory=list)
    missing_transitions: list[tuple[str, str, int]] = field(default_factory=list)
    high_cost_tools: list[tuple[str, float, float]] = field(default_factory=list)
    low_frequency_tools: list[tuple[str, int]] = field(default_factory=list)
    total_traces: int = 0
    total_successes: int = 0
    total_failures: int = 0

    @property
    def has_weaknesses(self) -> bool:
        return (
            len(self.failing_tools) > 0
            or len(self.weak_task_types) > 0
            or self.success_rate < 0.6
        )

    @property
    def weakness_count(self) -> int:
        return (
            len(self.failing_tools)
            + len(self.weak_task_types)
            + (1 if self.success_rate < 0.6 else 0)
        )

    def summary(self) -> list[str]:
        lines = []
        if self.success_rate < 0.6:
            lines.append(f"Overall low success rate: {self.success_rate:.0%}")
        for tool, rate, count in self.failing_tools[:3]:
            lines.append(f"Low success with '{tool}': {rate:.0%} ({count} uses)")
        for task, rate, count in self.weak_task_types[:3]:
            lines.append(f"Weak at '{task}' tasks: {rate:.0%} ({count} runs)")
        for tool, cost, avg in self.high_cost_tools[:3]:
            lines.append(f"High cost tool '{tool}': ${cost:.4f}/call")
        return lines


class WeaknessAnalyzer:
    def __init__(self, min_samples: int = 3):
        self.min_samples = min_samples

    def analyze(
        self,
        domain: str,
        traces: list[Any],
        tool_names: Optional[list[str]] = None,
    ) -> WeaknessProfile:
        profile = WeaknessProfile(domain=domain)

        if not traces:
            return profile

        profile.total_traces = len(traces)
        profile.total_successes = sum(1 for t in traces if _trace_success(t))
        profile.total_failures = profile.total_traces - profile.total_successes
        profile.success_rate = (
            profile.total_successes / profile.total_traces
            if profile.total_traces > 0
            else 1.0
        )

        tool_counts: dict[str, int] = defaultdict(int)
        tool_success: dict[str, int] = defaultdict(int)
        tool_cost: dict[str, list[float]] = defaultdict(list)
        tool_transitions: dict[tuple[str, str], int] = defaultdict(int)

        task_types: dict[str, list[bool]] = defaultdict(list)

        for t in traces:
            if isinstance(t, dict):
                seq = t.get("tool_sequence", [])
                success = t.get("outcome_success", False)
                task = t.get("task", "")
                cost = t.get("total_cost", 0)
            else:
                seq = getattr(t, "tool_sequence", [])
                success = getattr(t, "outcome_success", True)
                task = getattr(t, "task", "")
                cost = getattr(t, "total_cost", 0)

            task_type = self._classify_task_type(task)
            task_types[task_type].append(success)

            for tool in seq:
                tool_counts[tool] += 1
                if success:
                    tool_success[tool] += 1

            if cost and seq:
                per_step = cost / max(len(seq), 1)
                for tool in seq:
                    tool_cost[tool].append(per_step)

            for i in range(len(seq) - 1):
                tool_transitions[(seq[i], seq[i + 1])] += 1

        all_tools = tool_names or list(tool_counts.keys())

        for tool in all_tools:
            count = tool_counts.get(tool, 0)
            if count >= self.min_samples:
                success_count = tool_success.get(tool, 0)
                rate = success_count / count
                if rate < 0.6:
                    profile.failing_tools.append((tool, rate, count))
                costs = tool_cost.get(tool, [])
                if costs:
                    avg_cost = sum(costs) / len(costs)
                    if avg_cost > 0.02:
                        profile.high_cost_tools.append((tool, avg_cost, count))
            elif tool not in tool_counts:
                profile.avoiding_tools.append(tool)
            elif count < self.min_samples:
                profile.low_frequency_tools.append((tool, count))

        for task_type, outcomes in task_types.items():
            if len(outcomes) >= self.min_samples:
                rate = sum(outcomes) / len(outcomes)
                if rate < 0.5:
                    profile.weak_task_types.append((task_type, rate, len(outcomes)))

        profile.failing_tools.sort(key=lambda x: x[1])
        profile.weak_task_types.sort(key=lambda x: x[1])
        profile.high_cost_tools.sort(key=lambda x: -x[1])

        return profile

    def _classify_task_type(self, task: str) -> str:
        if not task:
            return "unknown"
        task_lower = task.lower()
        if any(kw in task_lower for kw in ["search", "find", "lookup", "query"]):
            return "search"
        if any(kw in task_lower for kw in ["deploy", "release", "publish"]):
            return "deployment"
        if any(kw in task_lower for kw in ["test", "verify", "check"]):
            return "testing"
        if any(kw in task_lower for kw in ["debug", "fix", "repair", "error"]):
            return "debugging"
        if any(kw in task_lower for kw in ["analyze", "analyze", "report"]):
            return "analysis"
        if any(kw in task_lower for kw in ["generate", "create", "write", "build"]):
            return "generation"
        return "general"


# ---------------------------------------------------------------------------
# Phase 6b — Synthetic Task Generation
# ---------------------------------------------------------------------------


@dataclass
class SyntheticTask:
    task: str
    target_weakness: str
    difficulty: float
    expected_tools: list[str] = field(default_factory=list)
    task_type: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "target_weakness": self.target_weakness,
            "difficulty": self.difficulty,
            "expected_tools": self.expected_tools,
            "task_type": self.task_type,
        }


class SyntheticTaskGenerator:
    def __init__(self):
        self._task_count = 0

    def generate_for_weakness(self, weakness: WeaknessProfile) -> list[SyntheticTask]:
        tasks: list[SyntheticTask] = []

        for tool, rate, count in weakness.failing_tools:
            tasks.append(self._make_practice_task(tool))
            if len(tasks) >= 3:
                break

        for task_type, rate, count in weakness.weak_task_types:
            tasks.append(self._make_type_task(task_type))
            if len(tasks) >= 2:
                break

        if weakness.success_rate < 0.6 and not tasks:
            tasks.append(self._make_general_task())

        return tasks

    def generate_from_failure(
        self, failed_trace: dict[str, Any]
    ) -> Optional[SyntheticTask]:
        task = (
            failed_trace.get("task")
            if isinstance(failed_trace, dict)
            else getattr(failed_trace, "task", None)
        )
        if not task:
            return None
        seq = (
            failed_trace.get("tool_sequence", [])
            if isinstance(failed_trace, dict)
            else getattr(failed_trace, "tool_sequence", [])
        )
        return SyntheticTask(
            task=task,
            target_weakness=f"past_failure_{task[:40]}",
            difficulty=0.6,
            expected_tools=seq,
            task_type="retry",
        )

    def _make_practice_task(self, tool: str) -> SyntheticTask:
        self._task_count += 1
        return SyntheticTask(
            task=f"Practice using {tool}: perform a task that requires the {tool} tool effectively.",
            target_weakness=f"low_success_{tool}",
            difficulty=0.5,
            expected_tools=[tool],
            task_type="practice",
        )

    def _make_type_task(self, task_type: str) -> SyntheticTask:
        self._task_count += 1
        return SyntheticTask(
            task=f"Practice {task_type} skills: complete a {task_type} task efficiently and correctly.",
            target_weakness=f"weak_{task_type}",
            difficulty=0.6,
            task_type=task_type,
        )

    def _make_general_task(self) -> SyntheticTask:
        self._task_count += 1
        return SyntheticTask(
            task="Complete a multi-step task using the available tools in the correct order.",
            target_weakness="general_capability",
            difficulty=0.5,
            task_type="general",
        )


# ---------------------------------------------------------------------------
# Phase 6c — Practice Engine
# ---------------------------------------------------------------------------


@dataclass
class PracticeSession:
    domain: str
    tasks: list[SyntheticTask] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    success_count: int = 0
    total_count: int = 0
    improvement_score: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.total_count, 1)

    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    def summary(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "tasks": len(self.tasks),
            "completed": self.total_count,
            "success_rate": self.success_rate,
            "improvement": self.improvement_score,
            "duration_s": round(self.duration, 2),
        }


class PracticeEngine:
    def __init__(
        self,
        run_fn: Optional[Callable[[str, dict], Any]] = None,
        max_tasks_per_session: int = 5,
    ):
        self._run_fn = run_fn
        self.max_tasks_per_session = max_tasks_per_session
        self._session_history: list[dict[str, Any]] = []

    def run_session(
        self,
        domain: str,
        tasks: list[SyntheticTask],
        before_score: Optional[float] = None,
    ) -> PracticeSession:
        session = PracticeSession(domain=domain, tasks=tasks[: self.max_tasks_per_session])

        logger.info(
            "Practice session starting for '%s' with %d tasks",
            domain,
            len(session.tasks),
        )

        for synthetic in session.tasks:
            try:
                if self._run_fn:
                    outcome = self._run_fn(synthetic.task, {})
                    success = bool(
                        getattr(outcome, "success", False)
                        if hasattr(outcome, "success")
                        else getattr(outcome, "outcome_success", False)
                    )
                else:
                    success = False

                session.results.append(
                    {
                        "task": synthetic.task,
                        "target": synthetic.target_weakness,
                        "success": success,
                        "difficulty": synthetic.difficulty,
                    }
                )
                session.total_count += 1
                if success:
                    session.success_count += 1

            except Exception as e:
                logger.warning("Practice task failed: %s", e)
                session.results.append(
                    {
                        "task": synthetic.task,
                        "target": synthetic.target_weakness,
                        "success": False,
                        "error": str(e),
                    }
                )
                session.total_count += 1

        session.end_time = time.time()

        if before_score is not None:
            after_score = session.success_rate
            session.improvement_score = after_score - before_score

        self._session_history.append(session.summary())

        logger.info(
            "Practice session complete: %d/%d successes (%.0f%%)",
            session.success_count,
            session.total_count,
            session.success_rate * 100,
        )

        return session

    def session_history(self) -> list[dict[str, Any]]:
        return list(self._session_history)


# ---------------------------------------------------------------------------
# Phase 6d — Benchmark Suite
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    domain: str
    passed: int = 0
    total: int = 0
    task_results: list[dict[str, Any]] = field(default_factory=list)
    score_before: Optional[float] = None
    score_after: Optional[float] = None
    improvement: float = 0.0
    duration: float = 0.0

    @property
    def score(self) -> float:
        return self.passed / max(self.total, 1)


class BenchmarkSuite:
    def __init__(
        self,
        run_fn: Optional[Callable[[str, dict], Any]] = None,
    ):
        self._run_fn = run_fn
        self._benchmarks: dict[str, list[dict]] = defaultdict(list)

    def register_benchmark_task(
        self,
        task: str,
        category: str = "general",
        expected_tools: Optional[list[str]] = None,
    ) -> None:
        self._benchmarks[category].append(
            {
                "task": task,
                "expected_tools": expected_tools or [],
            }
        )

    def run(
        self,
        domain: str,
        tasks: Optional[list[str]] = None,
    ) -> BenchmarkResult:
        start = time.time()
        result = BenchmarkResult(domain=domain)

        all_tasks = []
        if tasks:
            all_tasks = [{"task": t, "expected_tools": []} for t in tasks]
        else:
            for cat_tasks in self._benchmarks.values():
                all_tasks.extend(cat_tasks)

        result.total = len(all_tasks)

        for task_info in all_tasks:
            try:
                if self._run_fn:
                    outcome = self._run_fn(task_info["task"], {})
                    success = bool(
                        getattr(outcome, "success", False)
                        if hasattr(outcome, "success")
                        else getattr(outcome, "outcome_success", False)
                    )
                else:
                    success = False

                result.task_results.append(
                    {
                        "task": task_info["task"],
                        "success": success,
                    }
                )
                if success:
                    result.passed += 1

            except Exception as e:
                result.task_results.append(
                    {
                        "task": task_info["task"],
                        "success": False,
                        "error": str(e),
                    }
                )

        result.duration = time.time() - start
        return result

    def compare(
        self,
        domain: str,
        before_traces: list[Any],
        after_traces: list[Any],
    ) -> dict[str, Any]:
        def rate(traces: list) -> float:
            if not traces:
                return 0.0
            success = sum(1 for t in traces if _trace_success(t))
            return success / len(traces)

        before_rate = rate(before_traces)
        after_rate = rate(after_traces)

        return {
            "before_rate": round(before_rate, 4),
            "after_rate": round(after_rate, 4),
            "improvement": round(after_rate - before_rate, 4),
            "before_count": len(before_traces),
            "after_count": len(after_traces),
        }


# ---------------------------------------------------------------------------
# Phase 6e — Curriculum Engine (Orchestrator)
# ---------------------------------------------------------------------------


@dataclass
class CurriculumResult:
    domain: str
    weakness_profile: Optional[WeaknessProfile] = None
    synthetic_tasks: list[SyntheticTask] = field(default_factory=list)
    practice_session: Optional[PracticeSession] = None
    benchmark_before: Optional[BenchmarkResult] = None
    benchmark_after: Optional[BenchmarkResult] = None
    comparison: Optional[dict[str, Any]] = None
    had_impact: bool = False
    duration: float = 0.0
    error: Optional[str] = None

    def summary(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "had_impact": self.had_impact,
            "weaknesses": self.weakness_profile.weakness_count if self.weakness_profile else 0,
            "tasks_generated": len(self.synthetic_tasks),
            "practice_success_rate": self.practice_session.success_rate if self.practice_session else 0.0,
            "benchmark_before": round(self.benchmark_before.score, 3) if self.benchmark_before else None,
            "benchmark_after": round(self.benchmark_after.score, 3) if self.benchmark_after else None,
            "improvement": self.comparison["improvement"] if self.comparison else 0.0,
            "duration_s": round(self.duration, 2),
        }


class CurriculumEngine:
    def __init__(
        self,
        run_fn: Optional[Callable[[str, dict], Any]] = None,
        min_traces: int = 10,
        auto_practice: bool = True,
    ):
        self.weakness_analyzer = WeaknessAnalyzer()
        self.task_generator = SyntheticTaskGenerator()
        self.practice_engine = PracticeEngine(run_fn=run_fn)
        self.benchmark_suite = BenchmarkSuite(run_fn=run_fn)
        self.min_traces = min_traces
        self.auto_practice = auto_practice
        self._curriculum_history: list[dict[str, Any]] = []
        self._practice_count = 0

    def run(
        self,
        domain: str,
        traces: list[Any],
        tool_names: Optional[list[str]] = None,
    ) -> CurriculumResult:
        start = time.time()
        result = CurriculumResult(domain=domain)

        try:
            if len(traces) < self.min_traces:
                logger.info(
                    "Curriculum skip for '%s': %d traces (need %d)",
                    domain,
                    len(traces),
                    self.min_traces,
                )
                result.duration = time.time() - start
                return result

            logger.info(
                "Curriculum cycle starting for '%s' (%d traces, practice #%d)",
                domain,
                len(traces),
                self._practice_count + 1,
            )

            result.weakness_profile = self.weakness_analyzer.analyze(
                domain, traces, tool_names=tool_names
            )

            if not result.weakness_profile.has_weaknesses:
                logger.info("No weaknesses found for '%s'", domain)
                result.duration = time.time() - start
                return result

            result.synthetic_tasks = self.task_generator.generate_for_weakness(
                result.weakness_profile
            )

            failed_traces = [t for t in traces if not _trace_success(t)]
            for ft in failed_traces[:3]:
                retry_task = self.task_generator.generate_from_failure(ft)
                if retry_task:
                    result.synthetic_tasks.append(retry_task)

            if self.auto_practice and result.synthetic_tasks:
                bench_before = self.benchmark_suite.run(domain)
                result.benchmark_before = bench_before

                weak_tasks = [
                    t
                    for w in result.weakness_profile.weak_task_types[:2]
                    for t in result.synthetic_tasks
                    if t.task_type == w[0]
                ]
                if not weak_tasks:
                    weak_tasks = result.synthetic_tasks[:3]

                result.practice_session = self.practice_engine.run_session(
                    domain,
                    weak_tasks,
                    before_score=bench_before.score,
                )
                self._practice_count += 1

                result.benchmark_after = self.benchmark_suite.run(domain)

                result.comparison = self.benchmark_suite.compare(
                    domain,
                    [t for t in traces if _trace_success(t)],
                    [r for r in (result.benchmark_after.task_results if result.benchmark_after else []) if r.get("success")],
                )

                improvement = result.comparison.get("improvement", 0.0)
                result.had_impact = improvement > 0.05

                logger.info(
                    "Curriculum impact for '%s': benchmark %.0f%% -> %.0f%% (improvement %+.0f%%)",
                    domain,
                    (result.benchmark_before.score * 100) if result.benchmark_before else 0,
                    (result.benchmark_after.score * 100) if result.benchmark_after else 0,
                    improvement * 100,
                )

        except Exception as e:
            logger.error("Curriculum cycle failed: %s", e)
            result.error = str(e)

        result.duration = time.time() - start
        self._curriculum_history.append(result.summary())
        return result

    def curriculum_history(self) -> list[dict[str, Any]]:
        return list(self._curriculum_history)
