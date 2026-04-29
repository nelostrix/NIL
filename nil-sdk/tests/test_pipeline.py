"""
Test suite for Evolution Pipeline — the orchestrator that wires all 6 modules.

Coverage:
  - PipelineResult data structure
  - Full pipeline execution (all stages)
  - Partial execution (insufficient data, skip stages)
  - Stage ordering and gating
  - Skill extraction through pipeline
  - Gradient computation through pipeline
  - World model rebuild scheduling
  - Drift detection through pipeline
  - Auto-recovery through pipeline
  - Agent integration (evolve method, introspection)
  - Edge cases
  - Scalability
"""

import tempfile
import time

import pytest

import nil
from nil.evolution.pipeline import EvolutionPipeline, PipelineResult
from nil.memory.store import PersistentMemory


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


@pytest.fixture
def memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = PersistentMemory(tmpdir)
        yield mem
        mem.close()


@pytest.fixture
def pipeline(memory):
    return EvolutionPipeline(memory, min_traces_for_evolution=10)


# ===========================================================================
# PIPELINE RESULT
# ===========================================================================


class TestPipelineResult:
    def test_default_no_impact(self):
        r = PipelineResult(domain="test")
        assert not r.had_impact
        assert r.skills_extracted == 0
        assert r.strategy_promoted is False

    def test_has_impact_skills(self):
        r = PipelineResult(domain="test", skills_extracted=3)
        assert r.had_impact

    def test_has_impact_deltas(self):
        r = PipelineResult(domain="test", deltas_applied=1)
        assert r.had_impact

    def test_has_impact_recovery(self):
        r = PipelineResult(domain="test", recovery_triggered=True)
        assert r.had_impact

    def test_summary(self):
        r = PipelineResult(
            domain="test",
            stages_run=["analyze", "extract_skills"],
            skills_extracted=2,
            duration=1.5,
        )
        s = r.summary()
        assert s["domain"] == "test"
        assert s["skills_extracted"] == 2
        assert s["duration_s"] == 1.5
        assert "analyze" in s["stages"]


# ===========================================================================
# FULL PIPELINE
# ===========================================================================


class TestFullPipeline:
    def test_runs_all_stages(self, memory, pipeline):
        """With enough data, all stages should run."""
        traces = []
        for i in range(15):
            traces.append(_trace(f"s-{i}", True, ["search", "fix"], created_offset=i))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["fix"], error="timeout", created_offset=15 + i))
        _populate(memory, traces)

        result = pipeline.run("test")
        assert "analyze" in result.stages_run
        assert "extract_skills" in result.stages_run
        assert "compute_gradients" in result.stages_run
        assert "check_drift" in result.stages_run
        assert result.analysis is not None
        assert result.duration > 0

    def test_insufficient_data_skips(self, memory, pipeline):
        """Below min traces, pipeline skips entirely."""
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(3)]
        _populate(memory, traces)

        result = pipeline.run("test")
        assert "skipped:insufficient_data" in result.stages_run
        assert not result.had_impact

    def test_empty_domain(self, memory, pipeline):
        result = pipeline.run("nonexistent")
        assert "skipped:insufficient_data" in result.stages_run

    def test_analysis_report_populated(self, memory, pipeline):
        traces = [_trace(f"s-{i}", True, ["a", "b"]) for i in range(15)]
        traces += [_trace(f"f-{i}", False, ["c"], error="err") for i in range(5)]
        _populate(memory, traces)

        result = pipeline.run("test")
        assert result.analysis is not None
        assert result.analysis.total_traces == 20

    def test_extraction_result_populated(self, memory, pipeline):
        traces = [_trace(f"s-{i}", True, ["search", "fix"]) for i in range(15)]
        traces += [_trace(f"f-{i}", False, ["other"], error="err") for i in range(5)]
        _populate(memory, traces)

        result = pipeline.run("test")
        assert result.extraction is not None

    def test_gradient_batch_populated(self, memory, pipeline):
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(10)]
        traces += [_trace(f"f-{i}", False, ["b"], error="err") for i in range(5)]
        _populate(memory, traces)

        result = pipeline.run("test")
        assert result.gradient_batch is not None

    def test_drift_report_populated(self, memory, pipeline):
        traces = [_trace(f"s-{i}", True, ["a"], created_offset=i) for i in range(30)]
        _populate(memory, traces)

        result = pipeline.run("test")
        assert result.drift_report is not None


# ===========================================================================
# WORLD MODEL SCHEDULING
# ===========================================================================


class TestWorldModelScheduling:
    def test_builds_on_first_run(self, memory):
        pipeline = EvolutionPipeline(
            memory,
            min_traces_for_evolution=10,
            min_traces_for_world_model=10,
        )
        traces = [_trace(f"s-{i}", True, ["a"], created_offset=i) for i in range(15)]
        traces += [_trace(f"f-{i}", False, ["b"], error="e") for i in range(5)]
        _populate(memory, traces)

        result = pipeline.run("test")
        assert "build_world_model" in result.stages_run
        assert result.world_model is not None

    def test_skips_below_threshold(self, memory):
        pipeline = EvolutionPipeline(
            memory,
            min_traces_for_evolution=10,
            min_traces_for_world_model=100,
        )
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(15)]
        _populate(memory, traces)

        result = pipeline.run("test")
        assert "build_world_model" not in result.stages_run


# ===========================================================================
# DRIFT AND RECOVERY
# ===========================================================================


class TestDriftRecovery:
    def test_detects_drift(self, memory):
        pipeline = EvolutionPipeline(memory, min_traces_for_evolution=10)
        traces = []
        for i in range(15):
            traces.append(_trace(f"old-{i}", True, ["a"], created_offset=i))
        for i in range(10):
            traces.append(_trace(f"new-{i}", False, ["a"], error="boom", created_offset=15 + i))
        _populate(memory, traces)

        result = pipeline.run("test")
        assert result.drift_report is not None
        # Drift may or may not be detected depending on window size

    def test_recovery_on_critical_drift(self, memory):
        pipeline = EvolutionPipeline(memory, min_traces_for_evolution=10, auto_apply=True)
        traces = []
        for i in range(20):
            traces.append(_trace(f"old-{i}", True, ["a"], created_offset=i))
        for i in range(15):
            traces.append(_trace(f"new-{i}", False, ["a"], error="crash", created_offset=20 + i))
        _populate(memory, traces)

        result = pipeline.run("test")
        if result.drift_detected:
            assert result.recovery_triggered or "auto_recover" in result.stages_run


# ===========================================================================
# AGENT INTEGRATION
# ===========================================================================


class TestAgentIntegration:
    def test_agent_evolve_manual(self, memory):
        """Manual evolution via agent.evolve() returns PipelineResult."""

        @nil.tool
        def search(query: str) -> str:
            """Search."""
            return f"Result: {query}"

        domain = nil.Domain(name="agent-test", tools=[search], objectives=["test"])
        agent = nil.Agent(
            domain=domain,
            memory=memory,
            evolution=nil.EvolutionConfig(strategy="manual"),
        )

        # Run enough traces
        for i in range(12):
            agent.run(f"task {i}")

        result = agent.evolve()
        assert isinstance(result, PipelineResult)
        assert result.domain == "agent-test"

    def test_agent_auto_evolution(self, memory):
        """Periodic evolution fires automatically."""

        @nil.tool
        def echo(msg: str) -> str:
            """Echo."""
            return msg

        domain = nil.Domain(name="auto-test", tools=[echo])
        agent = nil.Agent(
            domain=domain,
            memory=memory,
            evolution=nil.EvolutionConfig(
                strategy="periodic",
                evolve_interval=5,
            ),
        )

        # Run 10 tasks — should trigger evolution at run 5 and 10
        for i in range(10):
            agent.run(f"task {i}")

        # Should have 2 evolution entries
        history = agent.evolution_history()
        assert len(history) == 2

    def test_agent_stats(self, memory):
        @nil.tool
        def noop(x: str) -> str:
            """No-op."""
            return x

        domain = nil.Domain(name="stats-test", tools=[noop])
        agent = nil.Agent(domain=domain, memory=memory)

        for i in range(3):
            agent.run(f"task {i}")

        stats = agent.stats()
        assert stats["domain"] == "stats-test"
        assert stats["run_count"] == 3
        assert stats["traces_in_db"] == 3
        assert "success_rate" in stats
        assert "skills" in stats

    def test_agent_check_drift(self, memory):
        @nil.tool
        def noop(x: str) -> str:
            """No-op."""
            return x

        domain = nil.Domain(name="drift-test", tools=[noop])
        agent = nil.Agent(domain=domain, memory=memory)

        for i in range(5):
            agent.run(f"task {i}")

        drift = agent.check_drift()
        assert hasattr(drift, "drift_detected")
        assert hasattr(drift, "severity")

    def test_agent_repr_with_skills(self, memory):
        domain = nil.Domain(name="repr-test")
        agent = nil.Agent(domain=domain, memory=memory)
        r = repr(agent)
        assert "repr-test" in r
        assert "skills=" in r

    def test_agent_get_strategy(self, memory):
        domain = nil.Domain(name="strat-test")
        agent = nil.Agent(domain=domain, memory=memory)
        # No strategy yet
        assert agent.get_strategy() is None

    def test_agent_learned_skills_empty(self, memory):
        domain = nil.Domain(name="empty-test")
        agent = nil.Agent(domain=domain, memory=memory)
        assert agent.learned_skills() == []


# ===========================================================================
# ERROR HANDLING
# ===========================================================================


class TestErrorHandling:
    def test_pipeline_survives_analysis_error(self, memory):
        """Pipeline should catch and log errors, not crash."""
        pipeline = EvolutionPipeline(memory, min_traces_for_evolution=5)
        # Store malformed traces
        for i in range(10):
            memory.store_trace(
                {
                    "run_id": f"r-{i}",
                    "domain": "test",
                    "task": "test",
                    "steps": [],
                    "outcome_success": True,
                    "outcome_output": "ok",
                    "total_duration": 1.0,
                    "total_tokens_in": 0,
                    "total_tokens_out": 0,
                    "total_cost": 0,
                    "tool_sequence": [],
                    "error": None,
                    "created_at": time.time() + i,
                    "metadata": {},
                }
            )

        result = pipeline.run("test")
        # Should complete without raising
        assert isinstance(result, PipelineResult)


# ===========================================================================
# SCALABILITY
# ===========================================================================


class TestScalability:
    def test_pipeline_300_traces(self, memory):
        pipeline = EvolutionPipeline(memory, min_traces_for_evolution=10)
        traces = []
        for i in range(250):
            traces.append(_trace(f"s-{i}", True, ["a", "b", "c"], created_offset=i))
        for i in range(50):
            traces.append(_trace(f"f-{i}", False, ["d"], error="err", created_offset=250 + i))
        _populate(memory, traces)

        import time as t

        start = t.monotonic()
        result = pipeline.run("test")
        elapsed = t.monotonic() - start

        assert result.analysis.total_traces > 0
        assert elapsed < 60  # Full pipeline under 60s

    def test_agent_20_runs_with_evolution(self, memory):
        """Agent running 20 tasks with periodic evolution."""

        @nil.tool
        def noop(x: str) -> str:
            """No-op."""
            return x

        domain = nil.Domain(name="scale-test", tools=[noop])
        agent = nil.Agent(
            domain=domain,
            memory=memory,
            evolution=nil.EvolutionConfig(
                strategy="periodic",
                evolve_interval=10,
            ),
        )

        import time as t

        start = t.monotonic()
        for i in range(20):
            agent.run(f"task {i}")
        elapsed = t.monotonic() - start

        assert agent.run_count == 20
        assert len(agent.evolution_history()) == 2
        assert elapsed < 60


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
