"""
Comprehensive test suite for nil.evolution.gradient_engine.

Coverage:
  - Data structures (TextualGradient, StrategyDelta, GradientBatch)
  - StubGradientProvider (all heuristics, edge cases)
  - TextualGradientEngine pipeline (end-to-end, partial, empty)
  - Critique → gradient → delta conversion chain
  - Analysis-driven deltas (dead ends, significant patterns, systemic failures)
  - Strategy application and merging
  - Edge cases (empty traces, all success, all failure, single trace)
  - Trace selection (recency, cost, diversity)
  - Scalability
"""

import tempfile
import time

import pytest

from nil.evolution.analyzer import TraceAnalyzer
from nil.evolution.gradient_engine import (
    GradientBatch,
    GradientProvider,
    GradientType,
    StrategyDelta,
    StubGradientProvider,
    TextualGradient,
    TextualGradientEngine,
)
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
def engine(memory):
    return TextualGradientEngine(memory, min_failures_for_gradient=3)


@pytest.fixture
def provider():
    return StubGradientProvider()


def _trace(
    run_id,
    success,
    tools,
    domain="test",
    cost=0.001,
    duration=1.0,
    error=None,
    created_offset=0,
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
    for t in traces:
        memory.store_trace(t)


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================


class TestDataStructures:
    def test_gradient_type_values(self):
        assert GradientType.PROMPT_REWRITE.value == "prompt_rewrite"
        assert GradientType.TOOL_REORDER.value == "tool_reorder"
        assert GradientType.SKILL_INJECTION.value == "skill_injection"

    def test_textual_gradient_to_dict(self):
        g = TextualGradient(
            gradient_id="g-1",
            gradient_type=GradientType.PROMPT_REWRITE,
            critique="Agent failed to check logs",
            proposal="Add log-checking step",
            target="system_prompt",
            confidence=0.8,
        )
        d = g.to_dict()
        assert d["gradient_id"] == "g-1"
        assert d["gradient_type"] == "prompt_rewrite"
        assert d["confidence"] == 0.8

    def test_strategy_delta_to_dict(self):
        d = StrategyDelta(
            delta_id="d-1",
            source_gradient="g-1",
            mutation_type="reasoning_hint",
            changes={"reasoning_hints": ["Always check logs first"]},
            description="Add log hint",
            expected_impact=0.7,
        )
        dd = d.to_dict()
        assert dd["delta_id"] == "d-1"
        assert dd["changes"]["reasoning_hints"] == ["Always check logs first"]

    def test_gradient_batch_properties(self):
        batch = GradientBatch(
            batch_id="b-1",
            domain="test",
            gradients=[
                TextualGradient(
                    gradient_id="g-1",
                    gradient_type=GradientType.PROMPT_REWRITE,
                    critique="test",
                    proposal="test",
                    target="prompt",
                )
            ],
            deltas=[
                StrategyDelta(
                    delta_id="d-1",
                    source_gradient="g-1",
                    mutation_type="prompt_rewrite",
                )
            ],
        )
        assert batch.gradient_count == 1
        assert batch.has_actionable_deltas is True

    def test_empty_batch(self):
        batch = GradientBatch(batch_id="b-0", domain="test")
        assert batch.gradient_count == 0
        assert batch.has_actionable_deltas is False


# ===========================================================================
# STUB GRADIENT PROVIDER
# ===========================================================================


class TestStubProvider:
    def test_detects_missing_tools(self, provider):
        """Tools in successes but absent from failures → missing_tool critique."""
        successes = [
            _trace("s-0", True, ["diagnose", "fix", "verify"]),
            _trace("s-1", True, ["diagnose", "fix", "verify"]),
        ]
        failures = [
            _trace("f-0", False, ["fix"]),
            _trace("f-1", False, ["fix"]),
        ]
        critiques = provider.critique_traces(failures, successes, {"name": "test"})
        types = [c["type"] for c in critiques]
        assert "missing_tool" in types
        missing = [c for c in critiques if c["type"] == "missing_tool"]
        assert "diagnose" in missing[0]["affected_tools"]

    def test_detects_harmful_tools(self, provider):
        successes = [_trace("s-0", True, ["good"])]
        failures = [_trace("f-0", False, ["good", "poison"])]
        critiques = provider.critique_traces(failures, successes, {"name": "test"})
        types = [c["type"] for c in critiques]
        assert "harmful_tool" in types

    def test_detects_recurring_errors(self, provider):
        failures = [
            _trace("f-0", False, ["a"], error="ConnectionTimeout"),
            _trace("f-1", False, ["a"], error="ConnectionTimeout"),
        ]
        critiques = provider.critique_traces(failures, [], {"name": "test"})
        types = [c["type"] for c in critiques]
        assert "recurring_error" in types

    def test_detects_wrong_start(self, provider):
        successes = [
            _trace("s-0", True, ["check", "fix"]),
            _trace("s-1", True, ["check", "fix"]),
        ]
        failures = [
            _trace("f-0", False, ["fix", "check"]),
            _trace("f-1", False, ["fix", "check"]),
        ]
        critiques = provider.critique_traces(failures, successes, {"name": "test"})
        types = [c["type"] for c in critiques]
        assert "wrong_start" in types

    def test_detects_over_tooling(self, provider):
        successes = [_trace("s-0", True, ["a", "b"])]
        failures = [_trace("f-0", False, ["a", "b", "c", "d", "e", "f"])]
        critiques = provider.critique_traces(failures, successes, {"name": "test"})
        types = [c["type"] for c in critiques]
        assert "over_tooling" in types

    def test_fallback_critique(self, provider):
        """When no pattern is detected, produces a general critique."""
        failures = [_trace("f-0", False, ["a"])]
        critiques = provider.critique_traces(failures, [], {"name": "test"})
        assert len(critiques) >= 1

    def test_no_failures_no_critiques(self, provider):
        critiques = provider.critique_traces([], [], {"name": "test"})
        assert critiques == []

    def test_empty_tool_sequences(self, provider):
        failures = [_trace("f-0", False, [])]
        successes = [_trace("s-0", True, [])]
        critiques = provider.critique_traces(failures, successes, {"name": "test"})
        # Should not crash, may produce general critique
        assert isinstance(critiques, list)

    def test_proposals_from_missing_tool(self, provider):
        critiques = [
            {
                "type": "missing_tool",
                "critique": "Missing diagnose tool",
                "severity": "high",
                "affected_tools": ["diagnose"],
                "confidence": 0.8,
            }
        ]
        proposals = provider.propose_improvements(critiques, {}, {"name": "test"})
        assert len(proposals) >= 1
        assert proposals[0]["gradient_type"] == "reasoning_hint"
        assert "changes" in proposals[0]

    def test_proposals_from_harmful_tool(self, provider):
        critiques = [
            {
                "type": "harmful_tool",
                "critique": "Poison tool detected",
                "affected_tools": ["poison"],
                "confidence": 0.6,
            }
        ]
        proposals = provider.propose_improvements(critiques, {}, {"name": "test"})
        assert len(proposals) >= 1
        assert proposals[0]["gradient_type"] == "failure_avoidance"

    def test_proposals_from_wrong_start(self, provider):
        critiques = [
            {
                "type": "wrong_start",
                "critique": "Start with check",
                "recommended_start": "check",
                "confidence": 0.7,
            }
        ]
        proposals = provider.propose_improvements(critiques, {}, {"name": "test"})
        assert len(proposals) >= 1
        assert proposals[0]["changes"]["preferred_first_tool"] == "check"

    def test_proposals_from_over_tooling(self, provider):
        critiques = [
            {
                "type": "over_tooling",
                "critique": "Too many calls",
                "confidence": 0.5,
            }
        ]
        proposals = provider.propose_improvements(critiques, {}, {"name": "test"})
        assert len(proposals) >= 1
        assert proposals[0]["gradient_type"] == "cost_optimization"

    def test_proposals_from_recurring_error(self, provider):
        critiques = [
            {
                "type": "recurring_error",
                "critique": "Timeout errors",
                "error_patterns": ["ConnectionTimeout", "ReadTimeout"],
                "confidence": 0.7,
            }
        ]
        proposals = provider.propose_improvements(critiques, {}, {"name": "test"})
        assert len(proposals) >= 1
        assert "error_handling_hints" in proposals[0]["changes"]

    def test_multiple_critiques_multiple_proposals(self, provider):
        critiques = [
            {"type": "missing_tool", "critique": "a", "affected_tools": ["x"], "confidence": 0.8},
            {"type": "harmful_tool", "critique": "b", "affected_tools": ["y"], "confidence": 0.6},
            {"type": "wrong_start", "critique": "c", "recommended_start": "z", "confidence": 0.7},
        ]
        proposals = provider.propose_improvements(critiques, {}, {"name": "test"})
        assert len(proposals) == 3


# ===========================================================================
# TEXTUAL GRADIENT ENGINE — END TO END
# ===========================================================================


class TestGradientEngineE2E:
    def test_full_pipeline(self, memory, engine):
        """Complete pipeline: traces → critiques → gradients → deltas."""
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["diagnose", "fix", "verify"]))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["fix"], error="timeout"))
        _populate(memory, traces)

        batch = engine.compute_gradients("test")
        assert batch.domain == "test"
        assert batch.gradient_count > 0
        assert batch.has_actionable_deltas
        assert batch.analysis_summary["total_traces"] == 15
        assert batch.analysis_summary["failures"] == 5

    def test_insufficient_failures(self, memory, engine):
        """Below min_failures_for_gradient → empty batch."""
        traces = [
            _trace("s-0", True, ["a"]),
            _trace("s-1", True, ["a"]),
            _trace("f-0", False, ["b"], error="err"),
        ]
        _populate(memory, traces)

        batch = engine.compute_gradients("test")
        assert batch.gradient_count == 0
        assert batch.analysis_summary.get("reason") == "insufficient_failures"

    def test_empty_domain(self, memory, engine):
        batch = engine.compute_gradients("nonexistent")
        assert batch.gradient_count == 0
        assert not batch.has_actionable_deltas

    def test_all_successes(self, memory, engine):
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(10)]
        _populate(memory, traces)
        batch = engine.compute_gradients("test")
        assert batch.analysis_summary.get("reason") == "insufficient_failures"

    def test_all_failures(self, memory, engine):
        traces = [_trace(f"f-{i}", False, ["a"], error="boom") for i in range(10)]
        _populate(memory, traces)
        batch = engine.compute_gradients("test")
        assert batch.gradient_count > 0

    def test_batch_id_unique(self, memory, engine):
        traces = [_trace(f"f-{i}", False, ["a"], error="err") for i in range(5)]
        traces += [_trace(f"s-{i}", True, ["b"]) for i in range(5)]
        _populate(memory, traces)

        b1 = engine.compute_gradients("test")
        b2 = engine.compute_gradients("test")
        assert b1.batch_id != b2.batch_id

    def test_gradients_have_source_traces(self, memory, engine):
        traces = [_trace(f"f-{i}", False, ["a"], error="err") for i in range(5)]
        traces += [_trace(f"s-{i}", True, ["b"]) for i in range(5)]
        _populate(memory, traces)

        batch = engine.compute_gradients("test")
        for g in batch.gradients:
            assert isinstance(g.source_traces, list)

    def test_deltas_have_changes(self, memory, engine):
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["diagnose", "fix"]))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["fix"], error="timeout"))
        _populate(memory, traces)

        batch = engine.compute_gradients("test")
        for d in batch.deltas:
            assert isinstance(d.changes, dict)
            assert d.mutation_type != ""


# ===========================================================================
# ANALYSIS-DRIVEN DELTAS
# ===========================================================================


class TestAnalysisDrivenDeltas:
    def test_dead_end_avoidance(self, memory, engine):
        """Dead-end transitions should produce avoidance deltas."""
        traces = []
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["a", "dead_tool"]))
        for i in range(5):
            traces.append(_trace(f"s-{i}", True, ["a", "good_tool"]))
        _populate(memory, traces)

        batch = engine.compute_gradients("test")
        avoidance_deltas = [
            d
            for d in batch.deltas
            if d.mutation_type == "failure_avoidance" and "avoidance_rules" in d.changes
        ]
        assert len(avoidance_deltas) >= 1

    def test_skill_injection_from_patterns(self, memory, engine):
        """Significant causal patterns should produce skill injection deltas."""
        traces = []
        for i in range(15):
            traces.append(_trace(f"s-{i}", True, ["diagnose", "fix", "verify"]))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["random"], error="err"))
        _populate(memory, traces)

        batch = engine.compute_gradients("test")
        skill_deltas = [d for d in batch.deltas if d.mutation_type == "skill_injection"]
        assert len(skill_deltas) >= 1
        if skill_deltas:
            assert "skill_injections" in skill_deltas[0].changes

    def test_systemic_failure_handling(self, memory, engine):
        """Systemic failures should produce error handling deltas."""
        traces = []
        for i in range(5):
            traces.append(_trace(f"s-{i}", True, ["a"]))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["a"], error="DatabaseLocked"))
        _populate(memory, traces)

        batch = engine.compute_gradients("test")
        error_deltas = [d for d in batch.deltas if "error_handling_hints" in d.changes]
        assert len(error_deltas) >= 1


# ===========================================================================
# STRATEGY APPLICATION
# ===========================================================================


class TestStrategyApplication:
    def test_apply_single_delta(self, memory, engine):
        delta = StrategyDelta(
            delta_id="d-1",
            source_gradient="g-1",
            mutation_type="reasoning_hint",
            changes={"reasoning_hints": ["Always check logs first"]},
        )
        result = engine.apply_deltas("test", [delta])
        assert "reasoning_hints" in result
        assert "Always check logs first" in result["reasoning_hints"]

    def test_apply_multiple_deltas_merge(self, memory, engine):
        """Multiple deltas with list values should merge, not overwrite."""
        d1 = StrategyDelta(
            delta_id="d-1",
            source_gradient="g-1",
            mutation_type="reasoning_hint",
            changes={"reasoning_hints": ["Hint 1"]},
        )
        d2 = StrategyDelta(
            delta_id="d-2",
            source_gradient="g-2",
            mutation_type="reasoning_hint",
            changes={"reasoning_hints": ["Hint 2"]},
        )
        result = engine.apply_deltas("test", [d1, d2])
        assert len(result["reasoning_hints"]) == 2
        assert "Hint 1" in result["reasoning_hints"]
        assert "Hint 2" in result["reasoning_hints"]

    def test_apply_deduplicates(self, memory, engine):
        """Same hint applied twice should not duplicate."""
        d1 = StrategyDelta(
            delta_id="d-1",
            source_gradient="g-1",
            mutation_type="reasoning_hint",
            changes={"reasoning_hints": ["Same hint"]},
        )
        d2 = StrategyDelta(
            delta_id="d-2",
            source_gradient="g-2",
            mutation_type="reasoning_hint",
            changes={"reasoning_hints": ["Same hint"]},
        )
        result = engine.apply_deltas("test", [d1, d2])
        assert result["reasoning_hints"].count("Same hint") == 1

    def test_apply_preserves_existing_strategy(self, memory, engine):
        """Deltas should merge into existing strategy, not replace it."""
        memory.store_strategy("test", {"existing_key": "existing_value"})

        delta = StrategyDelta(
            delta_id="d-1",
            source_gradient="g-1",
            mutation_type="reasoning_hint",
            changes={"new_key": "new_value"},
        )
        result = engine.apply_deltas("test", [delta])
        assert result["existing_key"] == "existing_value"
        assert result["new_key"] == "new_value"

    def test_apply_scalar_overwrites(self, memory, engine):
        """Scalar values should overwrite, not merge."""
        delta = StrategyDelta(
            delta_id="d-1",
            source_gradient="g-1",
            mutation_type="tool_reorder",
            changes={"preferred_first_tool": "diagnose"},
        )
        result = engine.apply_deltas("test", [delta])
        assert result["preferred_first_tool"] == "diagnose"

    def test_apply_stores_in_memory(self, memory, engine):
        delta = StrategyDelta(
            delta_id="d-1",
            source_gradient="g-1",
            mutation_type="prompt_rewrite",
            changes={"reasoning_hints": ["Check logs"]},
        )
        engine.apply_deltas("test", [delta])
        stored = memory.get_active_strategy("test")
        assert stored is not None
        assert "reasoning_hints" in stored

    def test_apply_empty_deltas(self, memory, engine):
        result = engine.apply_deltas("test", [])
        assert isinstance(result, dict)

    def test_apply_versions_increment(self, memory, engine):
        d1 = StrategyDelta(
            delta_id="d-1",
            source_gradient="g-1",
            mutation_type="prompt_rewrite",
            changes={"v": "1"},
        )
        d2 = StrategyDelta(
            delta_id="d-2",
            source_gradient="g-2",
            mutation_type="prompt_rewrite",
            changes={"v": "2"},
        )
        engine.apply_deltas("test", [d1])
        engine.apply_deltas("test", [d2])
        stored = memory.get_active_strategy("test")
        assert stored["v"] == "2"


# ===========================================================================
# TRACE SELECTION
# ===========================================================================


class TestTraceSelection:
    def test_caps_at_max(self, memory, engine):
        traces = [_trace(f"t-{i}", False, ["a"]) for i in range(50)]
        selected = engine._select_traces_for_critique(traces)
        assert len(selected) <= engine.max_traces_per_critique

    def test_small_set_unchanged(self, memory, engine):
        traces = [_trace(f"t-{i}", False, ["a"]) for i in range(3)]
        selected = engine._select_traces_for_critique(traces)
        assert len(selected) == 3

    def test_includes_recent(self, memory, engine):
        traces = []
        for i in range(20):
            traces.append(_trace(f"t-{i}", False, ["a"], created_offset=i))
        selected = engine._select_traces_for_critique(traces)
        selected_ids = {t["run_id"] for t in selected}
        # Most recent trace should be included
        assert "t-19" in selected_ids

    def test_includes_high_cost(self, memory, engine):
        traces = [_trace(f"cheap-{i}", False, ["a"], cost=0.001) for i in range(15)]
        traces.append(_trace("expensive", False, ["a"], cost=100.0, created_offset=-1000))
        selected = engine._select_traces_for_critique(traces)
        selected_ids = {t["run_id"] for t in selected}
        assert "expensive" in selected_ids


# ===========================================================================
# EDGE CASES
# ===========================================================================


class TestEdgeCases:
    def test_single_failure_trace(self, memory):
        """One failure below threshold → no gradient."""
        engine = TextualGradientEngine(memory, min_failures_for_gradient=3)
        _populate(memory, [_trace("f-0", False, ["a"], error="err")])
        batch = engine.compute_gradients("test")
        assert batch.gradient_count == 0

    def test_mixed_domains(self, memory, engine):
        """Gradient for domain1 should not see domain2 traces."""
        _populate(
            memory,
            [_trace(f"d1-s-{i}", True, ["a"], domain="domain1") for i in range(5)]
            + [_trace(f"d1-f-{i}", False, ["b"], domain="domain1", error="err") for i in range(5)]
            + [_trace(f"d2-{i}", True, ["c"], domain="domain2") for i in range(5)],
        )
        batch = engine.compute_gradients("domain1")
        assert batch.analysis_summary["total_traces"] == 10

    def test_unicode_errors(self, memory, engine):
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(5)] + [
            _trace(f"f-{i}", False, ["a"], error="错误: 连接超时") for i in range(5)
        ]
        _populate(memory, traces)
        batch = engine.compute_gradients("test")
        assert isinstance(batch, GradientBatch)

    def test_step_level_errors_only(self, memory, engine):
        """Traces with errors only in steps (not top-level) should work."""
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(5)]
        traces += [
            _trace(f"f-{i}", False, ["a", "b"], step_errors={1: "step_boom"}) for i in range(5)
        ]
        _populate(memory, traces)
        batch = engine.compute_gradients("test")
        assert batch.gradient_count > 0

    def test_custom_provider(self, memory):
        """Custom provider should be used instead of stub."""

        class FakeProvider(GradientProvider):
            def __init__(self):
                self.called = False

            def critique_traces(self, failed_traces, successful_traces, domain_context):
                self.called = True
                return [{"type": "general", "critique": "test", "confidence": 0.5}]

            def propose_improvements(self, critiques, current_strategy, domain_context):
                return [
                    {
                        "gradient_type": "prompt_rewrite",
                        "target": "prompt",
                        "proposal": "test",
                        "confidence": 0.5,
                        "changes": {},
                    }
                ]

        provider = FakeProvider()
        engine = TextualGradientEngine(memory, provider=provider)
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(5)]
        traces += [_trace(f"f-{i}", False, ["b"], error="err") for i in range(5)]
        _populate(memory, traces)

        batch = engine.compute_gradients("test")
        assert provider.called is True

    def test_no_tool_sequences_at_all(self, memory, engine):
        """Traces with empty tool sequences everywhere."""
        traces = [_trace(f"s-{i}", True, []) for i in range(5)]
        traces += [_trace(f"f-{i}", False, [], error="err") for i in range(5)]
        _populate(memory, traces)
        batch = engine.compute_gradients("test")
        assert isinstance(batch, GradientBatch)


# ===========================================================================
# SCALABILITY
# ===========================================================================


class TestScalability:
    def test_200_traces(self, memory, engine):
        traces = []
        for i in range(150):
            traces.append(_trace(f"s-{i}", True, ["a", "b", "c"], created_offset=i))
        for i in range(50):
            traces.append(_trace(f"f-{i}", False, ["a"], error="err", created_offset=150 + i))
        _populate(memory, traces)

        import time as t

        start = t.monotonic()
        batch = engine.compute_gradients("test")
        elapsed = t.monotonic() - start

        assert batch.analysis_summary["total_traces"] == 200
        assert elapsed < 30

    def test_many_error_types(self, memory, engine):
        """50 different error types should not crash clustering."""
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(5)]
        for i in range(50):
            traces.append(_trace(f"f-{i}", False, ["a"], error=f"Error_{i}"))
        _populate(memory, traces)

        batch = engine.compute_gradients("test")
        assert batch.gradient_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
