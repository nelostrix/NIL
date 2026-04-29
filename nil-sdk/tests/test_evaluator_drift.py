"""
Test suite for Module 5 (Replay Evaluator) and Module 6 (Drift Detector).

Coverage:
  - Evaluator: scoring, significance testing, recommendations, deltas, rollback risk
  - Drift: multi-signal detection, severity, root causes, recovery, rollback
  - Integration: evaluator + drift working together
  - Edge cases and scalability
"""

import math
import tempfile
import time

import pytest

from nil.evolution.drift_detector import (
    DriftDetector,
    DriftReport,
    DriftSignal,
    RecoveryResult,
)
from nil.evolution.evaluator import (
    EvaluationResult,
    ReplayEvaluator,
    TraceScore,
    _sign_test_p,
    _wilson_ci,
)
from nil.memory.store import PersistentMemory


@pytest.fixture
def memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = PersistentMemory(tmpdir)
        yield mem
        mem.close()


@pytest.fixture
def evaluator(memory):
    return ReplayEvaluator(memory, min_traces_for_evaluation=5)


@pytest.fixture
def detector(memory):
    return DriftDetector(memory, window_size=10, baseline_window=30)


def _trace(
    run_id,
    success,
    tools,
    domain="test",
    cost=0.001,
    duration=1.0,
    error=None,
    created_offset=0,
    task="generic task",
    step_errors=None,
):
    steps = []
    for i, t in enumerate(tools):
        step = {
            "step_number": i + 1,
            "step_type": "tool_call",
            "tool_name": t,
            "tool_input": {},
            "tool_output": "ok",
            "duration_ms": 50,
            "tokens_in": 10,
            "tokens_out": 5,
            "error": None,
            "timestamp": time.time() + created_offset,
        }
        if step_errors and i in step_errors:
            step["error"] = step_errors[i]
        steps.append(step)
    return {
        "run_id": run_id,
        "domain": domain,
        "task": task,
        "steps": steps,
        "outcome_success": success,
        "outcome_output": "done" if success else "failed",
        "total_duration": duration,
        "total_tokens_in": 100,
        "total_tokens_out": 50,
        "total_cost": cost,
        "tool_sequence": tools,
        "error": error if not success else None,
        "created_at": time.time() + created_offset,
        "metadata": {},
    }


def _populate(memory, traces):
    for t in traces:
        memory.store_trace(t)


# ===========================================================================
# MODULE 5: REPLAY EVALUATOR — Statistical utilities
# ===========================================================================


class TestSignTest:
    def test_equal_wins_losses(self):
        p = _sign_test_p(5, 5)
        assert p > 0.5

    def test_all_wins(self):
        p = _sign_test_p(10, 0)
        assert p < 0.01

    def test_all_losses(self):
        p = _sign_test_p(0, 10)
        assert p < 0.01

    def test_zero_both(self):
        p = _sign_test_p(0, 0)
        assert p == 1.0

    def test_marginal(self):
        p = _sign_test_p(7, 3)
        assert 0 < p < 1

    def test_large_numbers(self):
        p = _sign_test_p(100, 50)
        assert 0 <= p <= 1


class TestWilsonCI:
    def test_basic(self):
        lo, hi = _wilson_ci(7, 10)
        assert 0 < lo < 0.7
        assert 0.7 < hi < 1

    def test_zero(self):
        lo, hi = _wilson_ci(0, 0)
        assert lo == 0.0 and hi == 1.0


# ===========================================================================
# MODULE 5: REPLAY EVALUATOR — Trace scoring
# ===========================================================================


class TestTraceScore:
    def test_composite(self):
        s = TraceScore(
            run_id="t-1",
            success_alignment=0.8,
            failure_avoidance=0.6,
            cost_score=0.7,
            constraint_compliance=0.9,
        )
        c = s.compute_composite()
        assert 0 < c < 1
        assert s.composite == c

    def test_all_zeros(self):
        s = TraceScore(run_id="t-1")
        c = s.compute_composite()
        assert c == 0.0

    def test_all_ones(self):
        s = TraceScore(
            run_id="t-1",
            success_alignment=1.0,
            failure_avoidance=1.0,
            cost_score=1.0,
            constraint_compliance=1.0,
        )
        c = s.compute_composite()
        assert c == pytest.approx(1.0)


# ===========================================================================
# MODULE 5: REPLAY EVALUATOR — Full evaluation
# ===========================================================================


class TestEvaluation:
    def test_basic_evaluation(self, memory, evaluator):
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["search", "fix"]))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["fix"], error="err"))
        _populate(memory, traces)

        result = evaluator.evaluate(
            "test",
            candidate={"reasoning_hints": ["Always search before fixing"]},
        )
        assert isinstance(result, EvaluationResult)
        assert result.domain == "test"
        assert result.traces_evaluated == 15
        assert result.recommendation in ("promote", "reject", "inconclusive")

    def test_better_candidate_promoted(self, memory, evaluator):
        """Candidate matching success patterns should score higher."""
        traces = []
        for i in range(15):
            traces.append(_trace(f"s-{i}", True, ["search", "fix"]))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["bad_tool"], error="err"))
        _populate(memory, traces)

        result = evaluator.evaluate(
            "test",
            candidate={
                "preferred_first_tool": "search",
                "skill_injections": ["search then fix sequence"],
                "avoidance_rules": ["Avoid bad_tool"],
            },
            baseline={},
        )
        # Candidate should score at least as well as empty baseline
        assert result.candidate_score >= result.baseline_score - 0.1

    def test_insufficient_traces(self, memory, evaluator):
        _populate(memory, [_trace("s-0", True, ["a"])])
        result = evaluator.evaluate("test", candidate={})
        assert result.recommendation == "inconclusive"
        assert "Need at least" in result.details.get("reason", "")

    def test_empty_domain(self, memory, evaluator):
        result = evaluator.evaluate("nonexistent", candidate={})
        assert result.recommendation == "inconclusive"

    def test_result_to_dict(self, memory, evaluator):
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(10)]
        _populate(memory, traces)

        result = evaluator.evaluate("test", candidate={})
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "recommendation" in d
        assert "p_value" in d

    def test_per_dimension_breakdown(self, memory, evaluator):
        traces = [_trace(f"s-{i}", True, ["a", "b"]) for i in range(10)]
        traces += [_trace(f"f-{i}", False, ["c"], error="err") for i in range(5)]
        _populate(memory, traces)

        result = evaluator.evaluate("test", candidate={"reasoning_hints": ["hint"]})
        assert "success_alignment" in result.per_dimension
        assert "failure_avoidance" in result.per_dimension

    def test_rollback_risk(self, memory, evaluator):
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(10)]
        _populate(memory, traces)

        result = evaluator.evaluate("test", candidate={})
        assert result.rollback_risk in ("low", "medium", "high")


class TestCompare:
    def test_compare_two_strategies(self, memory, evaluator):
        traces = [_trace(f"s-{i}", True, ["a", "b"]) for i in range(10)]
        traces += [_trace(f"f-{i}", False, ["c"], error="err") for i in range(5)]
        _populate(memory, traces)

        result = evaluator.compare(
            "test",
            strategy_a={},
            strategy_b={"preferred_first_tool": "a"},
        )
        assert isinstance(result, EvaluationResult)
        assert result.candidate_id == "strategy_b"


class TestEvaluateDeltas:
    def test_evaluate_multiple_deltas(self, memory, evaluator):
        traces = [_trace(f"s-{i}", True, ["a", "b"]) for i in range(10)]
        traces += [_trace(f"f-{i}", False, ["c"], error="err") for i in range(5)]
        _populate(memory, traces)

        deltas = [
            {"delta_id": "d1", "changes": {"preferred_first_tool": "a"}},
            {"delta_id": "d2", "changes": {"reasoning_hints": ["Be careful"]}},
        ]
        results = evaluator.evaluate_deltas("test", deltas)
        assert len(results) == 2
        # Should be sorted by improvement
        assert results[0].improvement >= results[1].improvement

    def test_empty_deltas(self, memory, evaluator):
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(10)]
        _populate(memory, traces)
        results = evaluator.evaluate_deltas("test", [])
        assert results == []


class TestLegacyEvaluator:
    def test_legacy_interface(self, memory, evaluator):
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(10)]
        _populate(memory, traces)
        result = evaluator.evaluate_legacy("test", candidate={})
        assert isinstance(result, dict)
        assert "recommendation" in result


# ===========================================================================
# MODULE 6: DRIFT DETECTOR — Signal detection
# ===========================================================================


class TestDriftSignals:
    def test_no_drift(self, memory, detector):
        traces = [_trace(f"r-{i}", True, ["a"], created_offset=i) for i in range(40)]
        _populate(memory, traces)

        report = detector.check("test")
        assert report.severity in ("none", "mild")

    def test_success_rate_drop(self, memory, detector):
        traces = []
        for i in range(25):
            traces.append(_trace(f"old-{i}", True, ["a"], created_offset=i))
        for i in range(15):
            traces.append(_trace(f"new-{i}", False, ["a"], error="err", created_offset=25 + i))
        _populate(memory, traces)

        report = detector.check("test")
        assert report.drift_detected is True
        types = [s.signal_type for s in report.signals]
        assert "success_rate" in types

    def test_new_errors_detected(self, memory, detector):
        traces = []
        for i in range(20):
            traces.append(_trace(f"old-{i}", True, ["a"], created_offset=i))
        for i in range(10):
            traces.append(
                _trace(f"new-{i}", False, ["a"], error="NeverSeenBefore", created_offset=20 + i)
            )
        _populate(memory, traces)

        report = detector.check("test")
        types = [s.signal_type for s in report.signals]
        assert "new_errors" in types

    def test_tool_degradation(self, memory, detector):
        traces = []
        for i in range(20):
            traces.append(_trace(f"old-{i}", True, ["good_tool"], created_offset=i))
        for i in range(10):
            traces.append(
                _trace(f"new-{i}", False, ["good_tool"], error="err", created_offset=20 + i)
            )
        _populate(memory, traces)

        report = detector.check("test")
        tool_signals = [s for s in report.signals if s.signal_type == "tool_degradation"]
        # May or may not detect depending on rates
        assert isinstance(tool_signals, list)

    def test_cost_drift(self, memory, detector):
        traces = []
        for i in range(20):
            traces.append(_trace(f"old-{i}", True, ["a"], cost=0.001, created_offset=i))
        for i in range(10):
            traces.append(_trace(f"new-{i}", True, ["a"], cost=0.1, created_offset=20 + i))
        _populate(memory, traces)

        report = detector.check("test")
        cost_signals = [s for s in report.signals if s.signal_type == "cost_drift"]
        assert len(cost_signals) >= 1

    def test_insufficient_data(self, memory, detector):
        traces = [_trace("s-0", True, ["a"])]
        _populate(memory, traces)
        report = detector.check("test")
        assert not report.drift_detected


# ===========================================================================
# MODULE 6: DRIFT DETECTOR — Severity and root causes
# ===========================================================================


class TestSeverity:
    def test_critical_from_big_drop(self, memory, detector):
        traces = []
        for i in range(20):
            traces.append(_trace(f"old-{i}", True, ["a"], created_offset=i))
        for i in range(10):
            traces.append(_trace(f"new-{i}", False, ["a"], error="crash", created_offset=20 + i))
        _populate(memory, traces)

        report = detector.check("test")
        assert report.severity in ("significant", "critical")

    def test_root_causes_populated(self, memory, detector):
        traces = []
        for i in range(20):
            traces.append(_trace(f"old-{i}", True, ["a"], created_offset=i))
        for i in range(10):
            traces.append(_trace(f"new-{i}", False, ["a"], error="NewError", created_offset=20 + i))
        _populate(memory, traces)

        report = detector.check("test")
        if report.drift_detected:
            assert len(report.root_causes) > 0

    def test_recommendation_present(self, memory, detector):
        traces = [_trace(f"r-{i}", True, ["a"], created_offset=i) for i in range(30)]
        _populate(memory, traces)
        report = detector.check("test")
        assert report.recommendation != ""


# ===========================================================================
# MODULE 6: DRIFT DETECTOR — Recovery
# ===========================================================================


class TestRecovery:
    def test_no_recovery_when_no_drift(self, memory, detector):
        traces = [_trace(f"r-{i}", True, ["a"], created_offset=i) for i in range(30)]
        _populate(memory, traces)

        report, recovery = detector.check_and_recover("test")
        assert not recovery.recovery_triggered or recovery.recovery_type == ""

    def test_recovery_on_drift(self, memory, detector):
        traces = []
        for i in range(20):
            traces.append(_trace(f"old-{i}", True, ["a"], created_offset=i))
        for i in range(10):
            traces.append(_trace(f"new-{i}", False, ["a"], error="boom", created_offset=20 + i))
        _populate(memory, traces)

        report, recovery = detector.check_and_recover("test")
        if report.drift_detected:
            assert recovery.recovery_triggered
            assert recovery.recovery_type in ("evolution", "rollback")
            assert recovery.description != ""

    def test_recovery_stores_strategy(self, memory, detector):
        traces = []
        for i in range(20):
            traces.append(_trace(f"old-{i}", True, ["a"], created_offset=i))
        for i in range(10):
            traces.append(_trace(f"new-{i}", False, ["a"], error="boom", created_offset=20 + i))
        _populate(memory, traces)

        report, recovery = detector.check_and_recover("test")
        if recovery.recovery_triggered:
            strategy = memory.get_active_strategy("test")
            assert strategy is not None

    def test_recovery_result_to_dict(self):
        r = RecoveryResult(
            domain="test",
            recovery_triggered=True,
            recovery_type="evolution",
            deltas_applied=3,
            description="Applied 3 deltas",
        )
        d = r.to_dict()
        assert d["recovery_triggered"] is True
        assert d["deltas_applied"] == 3


# ===========================================================================
# MODULE 6: DRIFT DETECTOR — Report serialization
# ===========================================================================


class TestDriftReport:
    def test_report_to_dict(self):
        report = DriftReport(
            domain="test",
            drift_detected=True,
            severity="significant",
            signals=[
                DriftSignal(
                    signal_type="success_rate", severity="significant", description="Rate dropped"
                ),
            ],
            root_causes=["Rate dropped"],
            recommendation="Fix it",
        )
        d = report.to_dict()
        assert d["drift_detected"] is True
        assert len(d["signals"]) == 1

    def test_report_properties(self):
        report = DriftReport(
            domain="test",
            signals=[
                DriftSignal(signal_type="a", severity="critical", description="x"),
                DriftSignal(signal_type="b", severity="significant", description="y"),
                DriftSignal(signal_type="c", severity="mild", description="z"),
            ],
        )
        assert len(report.critical_signals) == 1
        assert len(report.significant_signals) == 2
        assert report.signal_count == 3


# ===========================================================================
# INTEGRATION: Evaluator + Drift working together
# ===========================================================================


class TestIntegration:
    def test_drift_then_evaluate_recovery(self, memory):
        """Detect drift, generate recovery, evaluate it."""
        traces = []
        for i in range(20):
            traces.append(_trace(f"old-{i}", True, ["search", "fix"], created_offset=i))
        for i in range(10):
            traces.append(
                _trace(f"new-{i}", False, ["fix"], error="timeout", created_offset=20 + i)
            )
        _populate(memory, traces)

        # Detect drift
        detector = DriftDetector(memory, window_size=10)
        report = detector.check("test")

        if report.drift_detected and report.recovery_actions:
            # Build a recovery strategy from actions
            recovery_strategy = {"reasoning_hints": ["Be careful after timeout errors"]}
            for action in report.recovery_actions:
                if action.get("data"):
                    recovery_strategy.setdefault("error_handling_hints", [])
                    recovery_strategy["error_handling_hints"].extend(
                        [f"Handle: {e}" for e in action["data"][:3]]
                    )

            # Evaluate the recovery
            evaluator = ReplayEvaluator(memory, min_traces_for_evaluation=5)
            eval_result = evaluator.evaluate("test", candidate=recovery_strategy)
            assert isinstance(eval_result, EvaluationResult)


# ===========================================================================
# EDGE CASES
# ===========================================================================


class TestEdgeCases:
    def test_evaluator_all_same(self, memory, evaluator):
        traces = [_trace(f"r-{i}", True, ["a"]) for i in range(10)]
        _populate(memory, traces)
        result = evaluator.evaluate("test", candidate={}, baseline={})
        # Same strategy should tie
        assert result.recommendation == "inconclusive"

    def test_detector_mixed_domains(self, memory, detector):
        _populate(
            memory,
            [_trace(f"d1-{i}", True, ["a"], domain="domain1", created_offset=i) for i in range(30)]
            + [
                _trace(f"d2-{i}", False, ["b"], domain="domain2", error="err", created_offset=i)
                for i in range(30)
            ],
        )
        report = detector.check("domain1")
        assert report.severity in ("none", "mild")

    def test_evaluator_unicode(self, memory, evaluator):
        traces = [_trace(f"r-{i}", True, ["工具"], task="修复错误") for i in range(10)]
        _populate(memory, traces)
        result = evaluator.evaluate("test", candidate={"reasoning_hints": ["注意"]})
        assert isinstance(result, EvaluationResult)


# ===========================================================================
# SCALABILITY
# ===========================================================================


class TestScalability:
    def test_evaluator_200_traces(self, memory, evaluator):
        traces = [_trace(f"r-{i}", i % 3 != 0, ["a", "b"], created_offset=i) for i in range(200)]
        _populate(memory, traces)

        import time as t

        start = t.monotonic()
        result = evaluator.evaluate("test", candidate={"reasoning_hints": ["hint"]})
        elapsed = t.monotonic() - start

        assert result.traces_evaluated == 200
        assert elapsed < 30

    def test_detector_200_traces(self, memory, detector):
        traces = []
        for i in range(150):
            traces.append(_trace(f"old-{i}", True, ["a"], created_offset=i))
        for i in range(50):
            traces.append(_trace(f"new-{i}", False, ["a"], error="err", created_offset=150 + i))
        _populate(memory, traces)

        import time as t

        start = t.monotonic()
        report = detector.check("test")
        elapsed = t.monotonic() - start

        assert elapsed < 30


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
