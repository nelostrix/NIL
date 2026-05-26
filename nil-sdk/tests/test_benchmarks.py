from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from nil.benchmarks.base import (
    Agent,
    BenchmarkReport,
    CompositeScore,
    ForgettingCurve,
    ForgettingEvent,
    ForgettingPoint,
    HorizonResult,
    InterferenceResult,
    LearningReport,
    Outcome,
    RecoveryReport,
    SystemReport,
    TestResult,
    TransferReport,
    TransferResult,
    percentile,
    std,
)
from nil.benchmarks.horizon import LongHorizonBenchmark
from nil.benchmarks.learning import (
    ContextDependenceBenchmark,
    LearningCurveBenchmarks,
    LearningVelocity,
    SequentialReasoningBenchmark,
    ToolSelectionBenchmark,
)
from nil.benchmarks.memory import MemoryRetentionBenchmark
from nil.benchmarks.recovery import RecoveryBenchmark
from nil.benchmarks.runner import NilBenchmarkSuite
from nil.benchmarks.system import SystemBenchmark
from nil.benchmarks.transfer import TransferBenchmark


# ---------------------------------------------------------------------------
# Mock Agent
# ---------------------------------------------------------------------------

class _MockAgent:
    def __init__(self, success_rate: float = 1.0, tool: str = "tool_0"):
        self.success_rate = success_rate
        self.tool = tool
        self.run_count = 0
        self.tasks_seen: list[str] = []

    def run(self, task: str) -> Outcome:
        self.run_count += 1
        self.tasks_seen.append(task)
        return Outcome(
            success=self.run_count % 100 < self.success_rate * 100,
            steps=[
                {"step_number": 1, "tool_name": self.tool, "step_type": "tool_call"},
                {"step_number": 2, "tool_name": "tool_verify", "step_type": "tool_call"},
            ],
            total_cost=0.001,
            duration=0.5,
            had_errors=self.run_count % 5 == 0,
            error_steps=[{"tool_name": self.tool, "error": "fail"}] if self.run_count % 5 == 0 else [],
        )


# ===========================================================================
# Base / Data class tests
# ===========================================================================

class TestOutcome:
    def test_tool_sequence(self):
        o = Outcome(
            success=True,
            steps=[
                {"tool_name": "a"},
                {"tool_name": "b"},
                {"step_type": "think"},
            ],
        )
        assert o.tool_sequence == ["a", "b"]

    def test_steps_after(self):
        step_x = {"tool_name": "x"}
        o = Outcome(steps=[{"tool_name": "a"}, step_x, {"tool_name": "c"}])
        after = o.steps_after(step_x)
        assert len(after) == 1
        assert after[0]["tool_name"] == "c"


class TestLearningReport:
    def test_composite_on_sufficient_data(self):
        lr = LearningReport(initial_accuracy=0.2, plateau_accuracy=0.9)
        assert lr.composite == 0.9

    def test_no_composite_on_insufficient(self):
        lr = LearningReport(insufficient_data=True)
        assert lr.composite == 0.0


class TestCompositeScore:
    def test_total_weighted(self):
        cs = CompositeScore(
            learning_score=0.8,
            horizon_score=0.7,
            recovery_score=0.9,
            memory_score=0.6,
            transfer_score=0.5,
            system_score=0.4,
        )
        expected = 0.8 * 0.25 + 0.7 * 0.20 + 0.9 * 0.20 + 0.6 * 0.15 + 0.5 * 0.10 + 0.4 * 0.10
        assert cs.total == pytest.approx(expected)

    def test_to_dict(self):
        cs = CompositeScore(learning_score=0.5)
        d = cs.to_dict()
        assert "total" in d
        assert d["learning_score"] == 0.5


class TestPercentile:
    def test_median(self):
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == pytest.approx(3.0)

    def test_p95(self):
        data = list(range(100))
        assert percentile(data, 0.95) == pytest.approx(94.05)

    def test_empty(self):
        assert percentile([], 0.5) == 0.0

    def test_single(self):
        assert percentile([5.0], 0.5) == 5.0


class TestStd:
    def test_basic(self):
        assert std([1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_single_element(self):
        assert std([5.0]) == 0.0

    def test_empty(self):
        assert std([]) == 0.0


class TestBenchmarkReport:
    def test_summary(self):
        report = BenchmarkReport(agent_name="TestAgent")
        report.composite = CompositeScore(learning_score=1.0)
        s = report.summary()
        assert "TestAgent" in s
        assert "0.25" in s or "Composite" in s

    def test_to_dict(self):
        report = BenchmarkReport(agent_name="A")
        report.composite = CompositeScore(learning_score=0.5)
        d = report.to_dict()
        assert d["agent_name"] == "A"


# ===========================================================================
# Learning Velocity
# ===========================================================================

class TestLearningVelocity:
    def test_compute_improving_scores(self):
        scores = [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.93]
        lv = LearningVelocity()
        report = lv.compute(scores)
        assert report.insufficient_data is False
        assert report.initial_accuracy == 0.2
        assert report.plateau_accuracy > 0.8
        assert report.learning_velocity > 0
        assert report.runs_to_plateau > 0
        assert report.sample_efficiency > 0

    def test_insufficient_data(self):
        lv = LearningVelocity()
        report = lv.compute([0.5])
        assert report.insufficient_data is True

    def test_detect_forgetting_events(self):
        scores = [0.9, 0.9, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9]
        lv = LearningVelocity()
        events = lv._detect_forgetting_events(scores)
        assert len(events) >= 1

    def test_linear_regression(self):
        slope, intercept = LearningVelocity._linear_regression([0, 1, 2], [1, 3, 5])
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(1.0)

    def test_no_drops_no_events(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        lv = LearningVelocity()
        events = lv._detect_forgetting_events(scores)
        assert events == []


# ===========================================================================
# Learning Benchmarks
# ===========================================================================

class TestToolSelectionBenchmark:
    def test_run_returns_learning_report(self):
        agent = _MockAgent(success_rate=0.8, tool="tool_2")
        bench = ToolSelectionBenchmark()
        report = bench.run(agent)
        assert isinstance(report, LearningReport)


class TestSequentialReasoningBenchmark:
    def test_run_returns_learning_report(self):
        agent = _MockAgent(success_rate=0.9)
        bench = SequentialReasoningBenchmark()
        report = bench.run(agent)
        assert isinstance(report, LearningReport)


class TestContextDependenceBenchmark:
    def test_run_returns_learning_report(self):
        agent = _MockAgent(success_rate=0.85)
        bench = ContextDependenceBenchmark()
        report = bench.run(agent)
        assert isinstance(report, LearningReport)


class TestLearningCurveBenchmarks:
    def test_run_all_benchmarks(self):
        agent = _MockAgent(success_rate=0.8)
        lcb = LearningCurveBenchmarks()
        results = lcb.run(agent)
        assert "tool_selection" in results
        assert "sequential_reasoning" in results
        assert "context_dependence" in results
        for name, report in results.items():
            assert isinstance(report, LearningReport), f"{name} failed"


# ===========================================================================
# Long Horizon Benchmark
# ===========================================================================

class _SimpleTask:
    def __init__(self, required_steps):
        self.required_steps = required_steps
        self.description = "test"

class TestLongHorizonBenchmark:
    def test_run_returns_results_by_length(self):
        agent = _MockAgent(success_rate=0.9)
        bench = LongHorizonBenchmark()
        results = bench.run(agent)
        assert len(results) > 0
        for length, hr in results.items():
            assert isinstance(hr, HorizonResult)
            assert hr.mean_accuracy >= 0

    def test_score_step_accuracy(self):
        bench = LongHorizonBenchmark()
        outcome = Outcome(steps=[
            {"tool_name": "step_0"},
            {"tool_name": "step_1"},
        ])
        task = _SimpleTask(required_steps=["step_0", "step_1", "step_2"])
        acc = bench._score_step_accuracy(outcome, task)
        assert acc == pytest.approx(2 / 3)

    def test_no_errors_means_perfect_recovery(self):
        bench = LongHorizonBenchmark()
        outcome = Outcome(had_errors=False)
        task = _SimpleTask(required_steps=[])
        assert bench._score_error_recovery(outcome, task) == 1.0

    def test_recovery_with_subsequent_goal(self):
        bench = LongHorizonBenchmark()
        error_step = {"tool_name": "fail_tool"}
        outcome = Outcome(
            had_errors=True,
            error_steps=[error_step],
            steps=[error_step, {"tool_name": "fix_tool", "success": True}],
        )
        task = _SimpleTask(required_steps=[])
        score = bench._score_error_recovery(outcome, task)
        assert score == 1.0


# ===========================================================================
# Recovery Benchmark
# ===========================================================================

class TestRecoveryBenchmark:
    def test_run_returns_recovery_report(self):
        agent = _MockAgent(success_rate=0.8)
        bench = RecoveryBenchmark()
        report = bench.run(agent)
        assert isinstance(report, RecoveryReport)
        assert "tool_failure" in report.per_test
        assert "distribution_shift" in report.per_test

    def test_tool_failure_test(self):
        agent = _MockAgent(success_rate=0.75)
        bench = RecoveryBenchmark()
        result = bench._test_tool_failure(agent)
        assert isinstance(result, TestResult)
        assert 0 <= result.completion_rate <= 1

    def test_distribution_shift_test(self):
        agent = _MockAgent(success_rate=0.8)
        bench = RecoveryBenchmark()
        result = bench._test_distribution_shift(agent)
        assert isinstance(result, TestResult)
        assert result.pre_shift_accuracy >= 0

    def test_count_unnecessary_retries(self):
        bench = RecoveryBenchmark()
        outcome = Outcome(steps=[
            {"tool_name": "a", "error": "fail"},
            {"tool_name": "a", "error": "fail"},
            {"tool_name": "a", "error": "fail"},
        ])
        assert bench._count_unnecessary_retries(outcome) == 2

    def test_runs_to_recover(self):
        bench = RecoveryBenchmark()
        assert bench._runs_to_recover([0.2, 0.3, 0.8, 0.9], 0.8) == 3
        assert bench._runs_to_recover([0.1, 0.1], 0.8) == 2


# ===========================================================================
# Memory Retention Benchmark
# ===========================================================================

class TestMemoryRetentionBenchmark:
    def test_forgetting_curve(self):
        agent = _MockAgent(success_rate=0.9)

        def learn_task():
            return "Learn: pattern ABC"

        def test_task():
            return "Test: recall pattern ABC"

        bench = MemoryRetentionBenchmark()
        curve = bench.forgetting_curve(agent, learn_task, test_task, delays=[1, 5])
        assert isinstance(curve, ForgettingCurve)
        assert len(curve.points) == 2

    def test_interference_test(self):
        agent = _MockAgent(success_rate=0.8)

        def pattern_a():
            return "Pattern A: use tool_x"

        def pattern_b():
            return "Pattern B: use tool_y (conflicts with A)"

        def test_a():
            return "Test: which tool?"

        bench = MemoryRetentionBenchmark()
        result = bench.interference_test(agent, pattern_a, pattern_b, test_a)
        assert isinstance(result, InterferenceResult)
        assert 0 <= result.accuracy_a_before <= 1

    def test_half_life_estimate(self):
        bench = MemoryRetentionBenchmark()
        points = [
            ForgettingPoint(delay=0, accuracy=1.0),
            ForgettingPoint(delay=5, accuracy=0.5),
            ForgettingPoint(delay=10, accuracy=0.3),
        ]
        hl = bench._estimate_half_life(points)
        assert hl == pytest.approx(5.0, rel=0.1)

    def test_half_life_no_decay(self):
        bench = MemoryRetentionBenchmark()
        points = [ForgettingPoint(delay=0, accuracy=1.0)]
        hl = bench._estimate_half_life(points)
        assert hl == float("inf")


# ===========================================================================
# Transfer Benchmark
# ===========================================================================

class TestTransferBenchmark:
    def test_run_returns_transfer_report(self):
        agent = _MockAgent(success_rate=0.8)
        bench = TransferBenchmark()
        report = bench.run(agent)
        assert isinstance(report, TransferReport)
        assert "same_new_task" in report.transfer_results
        assert "similar_domain" in report.transfer_results

    def test_transfer_result_has_few_shot(self):
        agent = _MockAgent(success_rate=0.8)
        bench = TransferBenchmark()
        report = bench.run(agent)
        for name, tr in report.transfer_results.items():
            assert isinstance(tr, TransferResult)
            if tr.few_shot_accuracy:
                assert 1 in tr.few_shot_accuracy


# ===========================================================================
# System Benchmark
# ===========================================================================

class TestSystemBenchmark:
    def test_run_returns_system_report(self):
        agent = _MockAgent(success_rate=0.8)
        bench = SystemBenchmark()
        report = bench.run(agent, n_runs=10)
        assert isinstance(report, SystemReport)
        assert report.cost_per_run > 0
        assert report.latency_per_run > 0


# ===========================================================================
# NilBenchmarkSuite Integration
# ===========================================================================

class TestNilBenchmarkSuite:
    def test_run_returns_report(self):
        agent = _MockAgent(success_rate=0.9)
        suite = NilBenchmarkSuite()
        report = suite.run(agent, output_dir="/tmp/benchmark_test")
        assert isinstance(report, BenchmarkReport)
        assert report.agent_name != ""

    def test_report_has_all_sections(self):
        agent = _MockAgent(success_rate=0.85)
        suite = NilBenchmarkSuite()
        report = suite.run(agent, output_dir="/tmp/benchmark_test")
        assert len(report.learning_curves) > 0
        assert len(report.horizon_results) > 0
        assert report.recovery_results is not None
        assert len(report.memory_results) > 0
        assert report.transfer_results is not None
        assert report.system_metrics is not None
        assert report.total_duration > 0

    def test_composite_score_computed(self):
        agent = _MockAgent(success_rate=0.9)
        suite = NilBenchmarkSuite()
        report = suite.run(agent, output_dir="/tmp/benchmark_test")
        assert report.composite is not None
        assert report.composite.total > 0
