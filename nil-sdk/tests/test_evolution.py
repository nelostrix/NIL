"""
Comprehensive test suite for nil.evolution.analyzer — Causal Trace Analyzer.

Tests are organized by analysis module with edge cases, scalability
tests, statistical verification, and boundary conditions.

Coverage:
  - Statistical utilities (Fisher, Wilson CI, Phi, decay)
  - Causal pattern mining (happy path, edge cases, subsequences)
  - Step-level attribution (necessity, sufficiency, direction)
  - Failure clustering (normalization, systemic detection)
  - Transition matrix (dead ends, start/end markers)
  - Counterfactual analysis (impact scoring, verdicts)
  - Performance trend detection (improving, degrading, stable)
  - Full analysis pipeline
  - Scalability with large trace volumes
  - Empty/degenerate inputs
  - Legacy compatibility
"""

import math
import tempfile
import time
from collections import Counter

import pytest

from nil.evolution.analyzer import (
    AnalysisReport,
    CausalPattern,
    FailureCluster,
    StepAttribution,
    TraceAnalyzer,
    TransitionEdge,
    exponential_decay_weight,
    fisher_exact_p,
    lcs_multi,
    lcs_two,
    normalize_error,
    phi_coefficient,
    wilson_ci,
)
from nil.evolution.drift_detector import DriftDetector
from nil.evolution.skill_extractor import ExtractionResult, SkillExtractor
from nil.memory.store import PersistentMemory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = PersistentMemory(tmpdir)
        yield mem
        mem.close()


@pytest.fixture
def analyzer(memory):
    return TraceAnalyzer(memory, min_sample_size=3)


def _trace(
    run_id: str,
    success: bool,
    tools: list[str],
    domain: str = "test",
    created_offset: float = 0,
    cost: float = 0.001,
    duration: float = 1.0,
    error: str = None,
    step_errors: dict[int, str] = None,
):
    """Build a synthetic trace dict."""
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
        "task": f"task-{run_id}",
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
    """Store a list of traces."""
    for t in traces:
        memory.store_trace(t)


# ===========================================================================
# UNIT TESTS: Statistical utilities
# ===========================================================================


class TestFisherExact:
    """Tests for the Fisher's exact test implementation."""

    def test_perfect_association(self):
        # Tool always appears in success, never in failure
        p = fisher_exact_p(10, 0, 0, 10)
        assert p < 0.001

    def test_no_association(self):
        # Tool appears equally in success and failure
        p = fisher_exact_p(5, 5, 5, 5)
        assert p > 0.3

    def test_negative_association(self):
        # Tool appears mostly in failures — p-value still valid range
        p = fisher_exact_p(1, 9, 9, 1)
        assert 0 <= p <= 1.0

    def test_empty_table(self):
        p = fisher_exact_p(0, 0, 0, 0)
        assert p == 1.0

    def test_single_cell_nonzero(self):
        p = fisher_exact_p(1, 0, 0, 0)
        assert 0 <= p <= 1.0

    def test_all_success(self):
        # Everyone succeeds, tool present in some
        p = fisher_exact_p(5, 0, 5, 0)
        assert 0 <= p <= 1.0

    def test_all_failure(self):
        p = fisher_exact_p(0, 5, 0, 5)
        assert p == 1.0

    def test_large_numbers(self):
        """Should not overflow with large counts."""
        p = fisher_exact_p(500, 100, 100, 500)
        assert 0 <= p <= 1.0

    def test_symmetry(self):
        """Swapping rows produces same or different p-value depending on table."""
        p1 = fisher_exact_p(8, 2, 2, 8)
        p2 = fisher_exact_p(2, 8, 8, 2)
        # Both are valid probabilities
        assert 0 <= p1 <= 1.0
        assert 0 <= p2 <= 1.0
        # Strong positive association should have small p
        assert p1 < 0.05


class TestWilsonCI:
    def test_zero_total(self):
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_all_successes(self):
        lo, hi = wilson_ci(10, 10)
        assert lo > 0.5
        assert hi <= 1.0

    def test_no_successes(self):
        lo, hi = wilson_ci(0, 10)
        assert lo >= 0.0
        assert hi < 0.5

    def test_half_success(self):
        lo, hi = wilson_ci(50, 100)
        assert 0.3 < lo < 0.5
        assert 0.5 < hi < 0.7

    def test_single_sample(self):
        lo, hi = wilson_ci(1, 1)
        assert 0 <= lo <= hi <= 1

    def test_ci_narrows_with_more_data(self):
        lo1, hi1 = wilson_ci(5, 10)
        lo2, hi2 = wilson_ci(50, 100)
        assert (hi1 - lo1) > (hi2 - lo2)


class TestPhiCoefficient:
    def test_perfect_positive(self):
        phi = phi_coefficient(10, 0, 0, 10)
        assert phi == pytest.approx(1.0)

    def test_perfect_negative(self):
        phi = phi_coefficient(0, 10, 10, 0)
        assert phi == pytest.approx(-1.0)

    def test_no_association(self):
        phi = phi_coefficient(5, 5, 5, 5)
        assert phi == pytest.approx(0.0)

    def test_zero_denominator(self):
        phi = phi_coefficient(0, 0, 0, 0)
        assert phi == 0.0

    def test_partial_association(self):
        phi = phi_coefficient(8, 2, 3, 7)
        assert 0 < phi < 1


class TestExponentialDecay:
    def test_zero_age(self):
        w = exponential_decay_weight(100.0, 100.0)
        assert w == pytest.approx(1.0)

    def test_one_half_life(self):
        half_life = 86400 * 7
        w = exponential_decay_weight(0.0, half_life, half_life)
        assert w == pytest.approx(0.5, abs=0.01)

    def test_very_old(self):
        w = exponential_decay_weight(0.0, 86400 * 365, 86400 * 7)
        assert w < 0.001

    def test_future_timestamp(self):
        """Future created_at should get weight 1.0."""
        w = exponential_decay_weight(200.0, 100.0)
        assert w == pytest.approx(1.0)


# ===========================================================================
# UNIT TESTS: Helper functions
# ===========================================================================


class TestNormalizeError:
    def test_strips_line_numbers(self):
        assert "line N" in normalize_error("Error at line 42")

    def test_strips_addresses(self):
        assert "0xADDR" in normalize_error("Segfault at 0xDEADBEEF")

    def test_strips_timestamps(self):
        assert "TIMESTAMP" in normalize_error("Error at 2024-01-15T10:30:00")

    def test_strips_uuids(self):
        assert "UUID" in normalize_error("Failed for 550e8400-e29b-41d4-a716-446655440000")

    def test_strips_long_ids(self):
        assert "ID" in normalize_error("Record 123456789 not found")

    def test_truncates(self):
        long_error = "x" * 500
        assert len(normalize_error(long_error)) <= 200

    def test_empty_string(self):
        assert normalize_error("") == ""

    def test_collapses_whitespace(self):
        assert "  " not in normalize_error("error   at   line")


class TestLCS:
    def test_identical(self):
        assert lcs_two(["a", "b", "c"], ["a", "b", "c"]) == ["a", "b", "c"]

    def test_no_common(self):
        assert lcs_two(["a"], ["b"]) == []

    def test_partial(self):
        assert lcs_two(["a", "b", "c"], ["a", "c"]) == ["a", "c"]

    def test_empty(self):
        assert lcs_two([], ["a"]) == []
        assert lcs_two(["a"], []) == []
        assert lcs_two([], []) == []

    def test_multi_empty(self):
        assert lcs_multi([]) == []

    def test_multi_single(self):
        assert lcs_multi([["a", "b"]]) == ["a", "b"]

    def test_multi_three(self):
        result = lcs_multi([["a", "b", "c"], ["a", "c"], ["b", "a", "c"]])
        assert "a" in result and "c" in result


# ===========================================================================
# INTEGRATION TESTS: Causal Pattern Mining
# ===========================================================================


class TestCausalPatterns:
    def test_finds_dominant_pattern(self, memory, analyzer):
        """A sequence that always succeeds should have high causal score."""
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["diagnose", "fix", "verify"]))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["fix"]))
        _populate(memory, traces)

        patterns = analyzer.find_causal_patterns("test")
        assert len(patterns) > 0
        top = patterns[0]
        assert top.success_rate > 0.8
        assert top.causal_score > 0

    def test_no_patterns_below_threshold(self, memory, analyzer):
        """Patterns below min_frequency should not appear."""
        _populate(
            memory,
            [
                _trace("s-0", True, ["a", "b"]),
                _trace("s-1", True, ["c", "d"]),
            ],
        )
        patterns = analyzer.find_causal_patterns("test", min_frequency=3)
        assert len(patterns) == 0

    def test_subsequence_mining(self, memory, analyzer):
        """Should find common subsequences across different full sequences."""
        traces = []
        for i in range(5):
            traces.append(_trace(f"s-{i}", True, ["setup", "diagnose", "fix", "verify"]))
        for i in range(5):
            traces.append(_trace(f"s2-{i}", True, ["diagnose", "fix"]))
        _populate(memory, traces)

        patterns = analyzer.find_causal_patterns("test", min_frequency=3)
        subseqs = [tuple(p.sequence) for p in patterns]
        # Should find ["diagnose", "fix"] as a subsequence
        assert ("diagnose", "fix") in subseqs

    def test_confidence_interval_present(self, memory, analyzer):
        traces = [_trace(f"s-{i}", True, ["a", "b"]) for i in range(10)]
        traces += [_trace(f"f-{i}", False, ["a", "b"]) for i in range(2)]
        _populate(memory, traces)

        patterns = analyzer.find_causal_patterns("test")
        for p in patterns:
            lo, hi = p.confidence_interval
            assert 0 <= lo <= hi <= 1

    def test_temporal_trend_detection(self, memory, analyzer):
        """Early failures + late successes should show 'rising' trend."""
        traces = []
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["a", "b"], created_offset=i))
        for i in range(5):
            traces.append(_trace(f"s-{i}", True, ["a", "b"], created_offset=10 + i))
        _populate(memory, traces)

        patterns = analyzer.find_causal_patterns("test")
        ab_patterns = [p for p in patterns if p.sequence == ["a", "b"]]
        if ab_patterns:
            assert ab_patterns[0].temporal_trend in ("rising", "stable")

    def test_cost_efficiency_ranking(self, memory, analyzer):
        """Cheaper successful patterns should rank higher on cost_efficiency."""
        traces = []
        for i in range(5):
            traces.append(_trace(f"cheap-{i}", True, ["fast"], cost=0.001))
        for i in range(5):
            traces.append(_trace(f"expensive-{i}", True, ["slow"], cost=1.0))
        _populate(memory, traces)

        patterns = analyzer.find_causal_patterns("test")
        fast = [p for p in patterns if p.sequence == ["fast"]]
        slow = [p for p in patterns if p.sequence == ["slow"]]
        if fast and slow:
            assert fast[0].cost_efficiency > slow[0].cost_efficiency

    def test_empty_domain(self, memory, analyzer):
        assert analyzer.find_causal_patterns("nonexistent") == []

    def test_all_failures_no_patterns(self, memory, analyzer):
        """Pure failure traces should produce no causal patterns (s_count=0 filtered)."""
        traces = [_trace(f"f-{i}", False, ["a"]) for i in range(10)]
        _populate(memory, traces)
        patterns = analyzer.find_causal_patterns("test")
        assert len(patterns) == 0

    def test_empty_tool_sequences_ignored(self, memory, analyzer):
        """Traces with no tool calls should not crash."""
        traces = [_trace(f"s-{i}", True, []) for i in range(10)]
        _populate(memory, traces)
        patterns = analyzer.find_causal_patterns("test")
        assert patterns == []

    def test_single_tool_pattern(self, memory, analyzer):
        """A single-tool sequence should still be captured."""
        traces = [_trace(f"s-{i}", True, ["only_tool"]) for i in range(5)]
        traces += [_trace(f"f-{i}", False, ["other"]) for i in range(5)]
        _populate(memory, traces)

        patterns = analyzer.find_causal_patterns("test")
        names = [p.sequence for p in patterns]
        assert ["only_tool"] in names

    def test_source_run_ids_capped(self, memory, analyzer):
        """Source run IDs should be capped at 20."""
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(50)]
        _populate(memory, traces)
        patterns = analyzer.find_causal_patterns("test")
        for p in patterns:
            assert len(p.source_run_ids) <= 20


# ===========================================================================
# INTEGRATION TESTS: Step Attribution
# ===========================================================================


class TestStepAttribution:
    def test_necessary_tool(self, memory, analyzer):
        """A tool in >90% of successes should be marked necessary."""
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["critical_tool", "other"]))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["other"]))
        _populate(memory, traces)

        attrs = analyzer.compute_step_attributions("test")
        critical = [a for a in attrs if a.tool_name == "critical_tool"]
        assert len(critical) == 1
        assert critical[0].is_necessary is True

    def test_sufficient_tool(self, memory, analyzer):
        """A tool that succeeds >90% when present should be marked sufficient."""
        traces = []
        for i in range(9):
            traces.append(_trace(f"s-{i}", True, ["magic_tool"]))
        traces.append(_trace("f-0", False, ["magic_tool"]))
        for i in range(5):
            traces.append(_trace(f"f2-{i}", False, ["other"]))
        _populate(memory, traces)

        attrs = analyzer.compute_step_attributions("test")
        magic = [a for a in attrs if a.tool_name == "magic_tool"]
        assert len(magic) == 1
        assert magic[0].conditional_success_rate >= 0.9

    def test_positive_direction(self, memory, analyzer):
        """Tool strongly correlated with success → positive direction."""
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["good_tool"]))
        for i in range(10):
            traces.append(_trace(f"f-{i}", False, ["bad_tool"]))
        _populate(memory, traces)

        attrs = analyzer.compute_step_attributions("test")
        good = [a for a in attrs if a.tool_name == "good_tool"]
        assert good[0].causal_direction == "positive"

    def test_negative_direction(self, memory, analyzer):
        """Tool strongly correlated with failure → negative direction."""
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["helper"]))
        for i in range(10):
            traces.append(_trace(f"f-{i}", False, ["poison_tool", "helper"]))
        _populate(memory, traces)

        attrs = analyzer.compute_step_attributions("test")
        poison = [a for a in attrs if a.tool_name == "poison_tool"]
        if poison:
            assert poison[0].causal_direction in ("negative", "neutral")

    def test_neutral_tool(self, memory, analyzer):
        """Tool appearing equally in success/failure → neutral."""
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["neutral_tool"]))
        for i in range(10):
            traces.append(_trace(f"f-{i}", False, ["neutral_tool"]))
        _populate(memory, traces)

        attrs = analyzer.compute_step_attributions("test")
        neutral = [a for a in attrs if a.tool_name == "neutral_tool"]
        assert neutral[0].causal_direction == "neutral"

    def test_below_min_sample_filtered(self, memory, analyzer):
        """Tools appearing fewer than min_sample_size times are excluded."""
        traces = [
            _trace("s-0", True, ["rare_tool"]),
            _trace("s-1", True, ["common"]),
            _trace("s-2", True, ["common"]),
            _trace("s-3", True, ["common"]),
        ]
        _populate(memory, traces)

        attrs = analyzer.compute_step_attributions("test")
        names = [a.tool_name for a in attrs]
        assert "rare_tool" not in names
        assert "common" in names

    def test_empty_traces(self, memory, analyzer):
        assert analyzer.compute_step_attributions("empty") == []

    def test_lift_calculation(self, memory, analyzer):
        """Lift > 1 means tool boosts success above baseline."""
        traces = []
        for i in range(8):
            traces.append(_trace(f"s-{i}", True, ["booster"]))
        for i in range(2):
            traces.append(_trace(f"f-{i}", False, ["booster"]))
        for i in range(5):
            traces.append(_trace(f"f2-{i}", False, ["other"]))
        _populate(memory, traces)

        attrs = analyzer.compute_step_attributions("test")
        booster = [a for a in attrs if a.tool_name == "booster"]
        assert booster[0].lift > 1.0


# ===========================================================================
# INTEGRATION TESTS: Failure Clustering
# ===========================================================================


class TestFailureClustering:
    def test_groups_same_error(self, memory, analyzer):
        """Same error message → same cluster."""
        traces = []
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["a"], error="ConnectionTimeout"))
        _populate(memory, traces)

        clusters = analyzer.cluster_failures("test")
        assert len(clusters) == 1
        assert clusters[0].count == 5
        assert clusters[0].is_systemic is True

    def test_separates_different_errors(self, memory, analyzer):
        """Different errors → different clusters."""
        traces = [
            _trace("f-0", False, ["a"], error="ConnectionTimeout"),
            _trace("f-1", False, ["a"], error="ConnectionTimeout"),
            _trace("f-2", False, ["a"], error="ConnectionTimeout"),
            _trace("f-3", False, ["b"], error="AuthFailure"),
            _trace("f-4", False, ["b"], error="AuthFailure"),
            _trace("f-5", False, ["b"], error="AuthFailure"),
        ]
        _populate(memory, traces)

        clusters = analyzer.cluster_failures("test")
        assert len(clusters) == 2

    def test_normalizes_errors(self, memory, analyzer):
        """Errors with different line numbers should cluster together."""
        traces = [
            _trace("f-0", False, ["a"], error="Error at line 42 in module"),
            _trace("f-1", False, ["a"], error="Error at line 99 in module"),
            _trace("f-2", False, ["a"], error="Error at line 7 in module"),
        ]
        _populate(memory, traces)

        clusters = analyzer.cluster_failures("test")
        assert len(clusters) == 1

    def test_systemic_threshold(self, memory, analyzer):
        """Below threshold → not systemic."""
        traces = [
            _trace("f-0", False, ["a"], error="rare error"),
            _trace("f-1", False, ["a"], error="rare error"),
        ]
        _populate(memory, traces)

        clusters = analyzer.cluster_failures("test", systemic_threshold=3)
        assert clusters[0].is_systemic is False

    def test_common_subsequence_found(self, memory, analyzer):
        """Failures with shared tool prefix → common subsequence."""
        traces = [
            _trace("f-0", False, ["setup", "connect", "query"], error="timeout"),
            _trace("f-1", False, ["setup", "connect", "insert"], error="timeout"),
            _trace("f-2", False, ["setup", "connect"], error="timeout"),
        ]
        _populate(memory, traces)

        clusters = analyzer.cluster_failures("test")
        assert (
            "setup" in clusters[0].common_subsequence or "connect" in clusters[0].common_subsequence
        )

    def test_step_error_extraction(self, memory, analyzer):
        """Error from step-level should be used when no top-level error."""
        traces = [
            _trace(f"f-{i}", False, ["a", "b"], step_errors={1: "step_error"}) for i in range(3)
        ]
        _populate(memory, traces)

        clusters = analyzer.cluster_failures("test")
        assert len(clusters) >= 1

    def test_no_failures_empty(self, memory, analyzer):
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(5)]
        _populate(memory, traces)
        assert analyzer.cluster_failures("test") == []

    def test_avg_step_where_failed(self, memory, analyzer):
        traces = [
            _trace(f"f-{i}", False, ["a", "b", "c"], step_errors={2: "boom"}) for i in range(3)
        ]
        _populate(memory, traces)

        clusters = analyzer.cluster_failures("test")
        assert clusters[0].avg_step_where_failed > 0


# ===========================================================================
# INTEGRATION TESTS: Transition Matrix
# ===========================================================================


class TestTransitionMatrix:
    def test_basic_transitions(self, memory, analyzer):
        traces = [_trace(f"s-{i}", True, ["a", "b", "c"]) for i in range(5)]
        _populate(memory, traces)

        edges = analyzer.build_transition_matrix("test")
        from_to = {(e.from_tool, e.to_tool) for e in edges}
        assert ("__START__", "a") in from_to
        assert ("a", "b") in from_to
        assert ("b", "c") in from_to
        assert ("c", "__END__") in from_to

    def test_dead_end_detection(self, memory, analyzer):
        """A transition that never leads to success → dead end."""
        traces = []
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["a", "dead_tool"]))
        for i in range(5):
            traces.append(_trace(f"s-{i}", True, ["a", "good_tool"]))
        _populate(memory, traces)

        edges = analyzer.build_transition_matrix("test")
        dead = [e for e in edges if e.from_tool == "a" and e.to_tool == "dead_tool"]
        if dead:
            assert dead[0].is_dead_end is True

    def test_success_rate_per_edge(self, memory, analyzer):
        traces = []
        for i in range(8):
            traces.append(_trace(f"s-{i}", True, ["x", "y"]))
        for i in range(2):
            traces.append(_trace(f"f-{i}", False, ["x", "y"]))
        _populate(memory, traces)

        edges = analyzer.build_transition_matrix("test")
        xy = [e for e in edges if e.from_tool == "x" and e.to_tool == "y"]
        assert xy[0].success_rate == pytest.approx(0.8, abs=0.01)

    def test_rare_transitions_filtered(self, memory, analyzer):
        """Transitions seen only once should be filtered out."""
        traces = [_trace("s-0", True, ["rare_a", "rare_b"])]
        _populate(memory, traces)

        edges = analyzer.build_transition_matrix("test")
        rare = [e for e in edges if e.from_tool == "rare_a"]
        assert len(rare) == 0  # count=1 < min threshold of 2

    def test_empty(self, memory, analyzer):
        assert analyzer.build_transition_matrix("empty") == []


# ===========================================================================
# INTEGRATION TESTS: Counterfactual Analysis
# ===========================================================================


class TestCounterfactual:
    def test_essential_tool(self, memory, analyzer):
        """Tool with high impact → 'essential' verdict."""
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["key_tool", "helper"]))
        for i in range(10):
            traces.append(_trace(f"f-{i}", False, ["helper"]))
        _populate(memory, traces)

        results = analyzer.counterfactual_step_importance("test")
        key = [r for r in results if r["tool"] == "key_tool"]
        assert len(key) == 1
        assert key[0]["verdict"] in ("essential", "helpful")
        assert key[0]["causal_impact"] > 0

    def test_harmful_tool(self, memory, analyzer):
        """Tool that lowers success rate → 'harmful' or 'neutral' verdict."""
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["good"]))
        for i in range(10):
            traces.append(_trace(f"f-{i}", False, ["bad_tool", "good"]))
        _populate(memory, traces)

        results = analyzer.counterfactual_step_importance("test")
        bad = [r for r in results if r["tool"] == "bad_tool"]
        if bad:
            assert bad[0]["causal_impact"] < 0

    def test_cost_delta(self, memory, analyzer):
        """Cost delta should reflect actual cost difference."""
        traces = []
        for i in range(5):
            traces.append(_trace(f"cheap-{i}", True, ["fast"], cost=0.001))
        for i in range(5):
            traces.append(_trace(f"no-fast-{i}", True, [], cost=0.01))
        _populate(memory, traces)

        results = analyzer.counterfactual_step_importance("test")
        fast = [r for r in results if r["tool"] == "fast"]
        if fast:
            assert fast[0]["cost_delta"] < 0  # cheaper with tool

    def test_insufficient_data(self, memory, analyzer):
        traces = [_trace("s-0", True, ["a"])]
        _populate(memory, traces)
        results = analyzer.counterfactual_step_importance("test")
        assert results == []

    def test_all_same_tools_no_results(self, memory, analyzer):
        """If every trace uses the same tools, no counterfactual possible."""
        traces = [_trace(f"r-{i}", i < 5, ["same"]) for i in range(10)]
        _populate(memory, traces)
        results = analyzer.counterfactual_step_importance("test")
        assert results == []  # no traces without "same"


# ===========================================================================
# INTEGRATION TESTS: Performance Trend
# ===========================================================================


class TestPerformanceTrend:
    def test_improving(self, memory, analyzer):
        traces = []
        for i in range(20):
            traces.append(_trace(f"old-{i}", False, ["a"], created_offset=i))
        for i in range(20):
            traces.append(_trace(f"new-{i}", True, ["a"], created_offset=20 + i))
        _populate(memory, traces)

        trend = analyzer.detect_performance_trend("test", window=15)
        assert trend["status"] == "improving"

    def test_degrading(self, memory, analyzer):
        traces = []
        for i in range(20):
            traces.append(_trace(f"old-{i}", True, ["a"], created_offset=i))
        for i in range(20):
            traces.append(_trace(f"new-{i}", False, ["a"], created_offset=20 + i))
        _populate(memory, traces)

        trend = analyzer.detect_performance_trend("test", window=15)
        assert trend["status"] == "degrading"

    def test_stable(self, memory, analyzer):
        traces = []
        for i in range(40):
            traces.append(_trace(f"r-{i}", i % 2 == 0, ["a"], created_offset=i))
        _populate(memory, traces)

        trend = analyzer.detect_performance_trend("test", window=15)
        assert trend["status"] == "stable"

    def test_insufficient_data(self, memory, analyzer):
        traces = [_trace("r-0", True, ["a"])]
        _populate(memory, traces)
        trend = analyzer.detect_performance_trend("test", window=20)
        assert trend["status"] == "insufficient_data"

    def test_slope_present(self, memory, analyzer):
        traces = [_trace(f"r-{i}", True, ["a"], created_offset=i) for i in range(40)]
        _populate(memory, traces)
        trend = analyzer.detect_performance_trend("test", window=15)
        assert "slope" in trend


# ===========================================================================
# INTEGRATION TESTS: Full Analysis Pipeline
# ===========================================================================


class TestFullAnalysis:
    def test_report_structure(self, memory, analyzer):
        traces = []
        for i in range(20):
            traces.append(_trace(f"s-{i}", True, ["diagnose", "fix"], created_offset=i))
        for i in range(10):
            traces.append(_trace(f"f-{i}", False, ["fix"], error="timeout", created_offset=20 + i))
        _populate(memory, traces)

        report = analyzer.full_analysis("test")
        assert isinstance(report, AnalysisReport)
        assert report.total_traces == 30
        assert report.total_successes == 20
        assert report.total_failures == 10
        assert 0.6 < report.baseline_success_rate < 0.7
        assert len(report.causal_patterns) > 0
        assert len(report.step_attributions) > 0
        assert len(report.failure_clusters) > 0
        assert len(report.transitions) > 0
        assert "status" in report.performance_trend

    def test_report_properties(self, memory, analyzer):
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["good_tool"]))
        for i in range(10):
            traces.append(_trace(f"f-{i}", False, ["bad_tool"], error="err"))
        _populate(memory, traces)

        report = analyzer.full_analysis("test")
        assert isinstance(report.significant_patterns, list)
        assert isinstance(report.positive_tools, list)
        assert isinstance(report.negative_tools, list)
        assert isinstance(report.systemic_failures, list)
        assert isinstance(report.dead_end_transitions, list)

    def test_empty_domain(self, memory, analyzer):
        report = analyzer.full_analysis("nonexistent")
        assert report.total_traces == 0
        assert report.causal_patterns == []


# ===========================================================================
# SCALABILITY TESTS
# ===========================================================================


class TestScalability:
    def test_500_traces(self, memory, analyzer):
        """Should handle 500 traces without timeout."""
        traces = []
        for i in range(500):
            success = i % 3 != 0
            tools = ["search", "analyze", "fix"] if success else ["search"]
            traces.append(_trace(f"r-{i}", success, tools, created_offset=i))
        _populate(memory, traces)

        import time as t

        start = t.monotonic()
        report = analyzer.full_analysis("test")
        elapsed = t.monotonic() - start

        assert report.total_traces == 500
        assert elapsed < 30  # should finish well within 30 seconds

    def test_many_unique_sequences(self, memory, analyzer):
        """100 different tool sequences should not crash."""
        traces = []
        for i in range(100):
            tools = [f"tool_{j}" for j in range(i % 5 + 1)]
            traces.append(_trace(f"r-{i}", i % 2 == 0, tools, created_offset=i))
        _populate(memory, traces)

        report = analyzer.full_analysis("test")
        assert report.total_traces == 100

    def test_long_tool_sequences(self, memory, analyzer):
        """Sequences with 10+ tools should work."""
        long_seq = [f"step_{i}" for i in range(12)]
        traces = [_trace(f"r-{i}", True, long_seq, created_offset=i) for i in range(5)]
        _populate(memory, traces)

        patterns = analyzer.find_causal_patterns("test")
        assert len(patterns) > 0


# ===========================================================================
# EDGE CASES
# ===========================================================================


class TestEdgeCases:
    def test_all_same_trace(self, memory, analyzer):
        """All traces identical → should still produce a report."""
        traces = [_trace(f"r-{i}", True, ["a", "b"]) for i in range(10)]
        _populate(memory, traces)

        report = analyzer.full_analysis("test")
        assert report.baseline_success_rate == 1.0

    def test_100_percent_failure(self, memory, analyzer):
        """All failures → patterns empty, clusters populated."""
        traces = [_trace(f"f-{i}", False, ["a"], error="boom") for i in range(10)]
        _populate(memory, traces)

        report = analyzer.full_analysis("test")
        assert report.baseline_success_rate == 0.0
        assert len(report.causal_patterns) == 0
        assert len(report.failure_clusters) > 0

    def test_mixed_domains(self, memory, analyzer):
        """Analysis for one domain should not include another."""
        _populate(
            memory,
            [
                _trace("d1-0", True, ["a"], domain="domain1"),
                _trace("d1-1", True, ["a"], domain="domain1"),
                _trace("d1-2", True, ["a"], domain="domain1"),
                _trace("d2-0", False, ["b"], domain="domain2"),
                _trace("d2-1", False, ["b"], domain="domain2"),
                _trace("d2-2", False, ["b"], domain="domain2"),
            ],
        )

        report = analyzer.full_analysis("domain1")
        assert report.total_traces == 3
        assert report.baseline_success_rate == 1.0

    def test_unicode_tool_names(self, memory, analyzer):
        traces = [_trace(f"r-{i}", True, ["工具_alpha", "инструмент"]) for i in range(5)]
        _populate(memory, traces)
        patterns = analyzer.find_causal_patterns("test")
        assert len(patterns) > 0

    def test_duplicate_tools_in_sequence(self, memory, analyzer):
        """Same tool called multiple times in one trace."""
        traces = [_trace(f"r-{i}", True, ["retry", "retry", "retry"]) for i in range(5)]
        _populate(memory, traces)
        patterns = analyzer.find_causal_patterns("test")
        assert len(patterns) > 0

    def test_very_long_error_strings(self, memory, analyzer):
        long_err = "Error: " + "a" * 1000
        traces = [_trace(f"f-{i}", False, ["a"], error=long_err) for i in range(3)]
        _populate(memory, traces)
        clusters = analyzer.cluster_failures("test")
        assert clusters[0].error_signature is not None
        assert len(clusters[0].error_signature) <= 200


# ===========================================================================
# LEGACY COMPATIBILITY
# ===========================================================================


class TestLegacyCompat:
    def test_find_winning_patterns(self, memory, analyzer):
        """Legacy method should return dicts."""
        traces = [_trace(f"s-{i}", True, ["a", "b"]) for i in range(5)]
        _populate(memory, traces)

        patterns = analyzer.find_winning_patterns("test", min_frequency=3)
        assert isinstance(patterns, list)
        if patterns:
            assert isinstance(patterns[0], dict)
            assert "sequence" in patterns[0]
            assert "success_rate" in patterns[0]

    def test_find_redundant_steps(self, memory, analyzer):
        """Legacy method should return dicts."""
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["neutral"]))
        for i in range(10):
            traces.append(_trace(f"f-{i}", False, ["neutral"]))
        _populate(memory, traces)

        results = analyzer.find_redundant_steps("test")
        assert isinstance(results, list)


# ===========================================================================
# CROSS-MODULE: Skill Extractor + Drift Detector still work
# ===========================================================================


class TestCrossModule:
    def test_skill_extractor_uses_analyzer(self, memory):
        """SkillExtractor should work with the upgraded analyzer."""
        for i in range(6):
            memory.store_trace(_trace(f"s-{i}", True, ["diagnose", "fix"]))
        for i in range(2):
            memory.store_trace(_trace(f"f-{i}", False, ["fix"]))

        extractor = SkillExtractor(memory)
        result = extractor.extract("test", min_frequency=3, min_confidence=0.5)
        # Should not crash; returns ExtractionResult
        assert isinstance(result, ExtractionResult)

    def test_drift_detector_uses_analyzer(self, memory):
        """DriftDetector should work with the upgraded analyzer."""
        for i in range(40):
            memory.store_trace(_trace(f"r-{i}", True, ["a"], created_offset=i))

        detector = DriftDetector(memory)
        result = detector.check("test")
        assert "drift_detected" in result
        assert "severity" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
