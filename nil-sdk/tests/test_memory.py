"""Tests for nil.kernel — runner, tracer, tool_manager, outcome."""

import pytest

from nil.config import NilConfig
from nil.domain import Domain
from nil.kernel.outcome import RunOutcome, StepRecord
from nil.kernel.tool_manager import ToolManager
from nil.kernel.tracer import Trace, Tracer
from nil.memory.store import PersistentMemory
from nil.tools.base import Tool, tool

# ---- Tool tests ----


def test_tool_decorator():
    @tool
    def greet(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}!"

    assert greet.name == "greet"
    assert greet.description == "Say hello."
    assert "name" in greet.parameters
    result = greet.invoke({"name": "Shobhit"})
    assert result == "Hello, Shobhit!"


def test_tool_schema():
    @tool
    def search(query: str, limit: int = 10) -> str:
        """Search for things."""
        return f"Results for {query}"

    schema = search.to_schema()
    assert schema["function"]["name"] == "search"
    assert "query" in schema["function"]["parameters"]["properties"]
    assert "query" in schema["function"]["parameters"]["required"]


def test_tool_manager_register_and_execute():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    mgr = ToolManager([add])
    assert "add" in mgr
    assert mgr.names == ["add"]

    result = mgr.execute("add", {"a": 3, "b": 5})
    assert result["output"] == 8
    assert result["error"] is None
    assert result["duration_ms"] >= 0


def test_tool_manager_missing_tool():
    mgr = ToolManager([])
    result = mgr.execute("nonexistent", {})
    assert result["error"] is not None
    assert result["output"] is None


# ---- Outcome tests ----


def test_run_outcome_basics():
    outcome = RunOutcome(
        run_id="test-1",
        task="do something",
        output="done",
        success=True,
        steps=[
            StepRecord(step_number=1, step_type="tool_call", tool_name="search"),
            StepRecord(step_number=2, step_type="tool_call", tool_name="analyze"),
        ],
        duration=1.5,
    )
    assert outcome.step_count == 2
    assert outcome.tool_sequence == ["search", "analyze"]
    assert not outcome.had_errors

    summary = outcome.summary()
    assert summary["success"] is True
    assert summary["steps"] == 2


def test_run_outcome_cost():
    step = StepRecord(
        step_number=1,
        step_type="llm_call",
        tokens_in=1000,
        tokens_out=500,
    )
    assert step.cost_estimate > 0


# ---- Tracer tests ----


def test_tracer_records_steps():
    tracer = Tracer(run_id="t-1", domain="test")
    tracer.record_llm_call(prompt="hello", response="world", tokens_in=10, tokens_out=5)
    tracer.record_tool_call(
        name="search", tool_input={"q": "test"}, tool_output="results"
    )

    assert tracer.step_count == 2

    outcome = RunOutcome(run_id="t-1", task="test task", output="done", success=True)
    trace = tracer.finalize(outcome)

    assert isinstance(trace, Trace)
    assert trace.run_id == "t-1"
    assert trace.domain == "test"
    assert len(trace.steps) == 2
    assert trace.total_duration > 0


def test_trace_serialization():
    tracer = Tracer(run_id="t-2", domain="test")
    tracer.record_llm_call(prompt="hi", tokens_in=5, tokens_out=3)
    outcome = RunOutcome(run_id="t-2", task="test", success=True)
    trace = tracer.finalize(outcome)

    d = trace.to_dict()
    assert d["run_id"] == "t-2"
    assert isinstance(d["steps"], list)
    assert "created_at" in d


# ---- Domain tests ----


def test_domain_creation():
    d = Domain(name="Test Domain", objectives=["win"])
    assert d.name == "test-domain"  # normalized
    assert d.objectives == ["win"]


def test_domain_empty_name_raises():
    with pytest.raises(ValueError):
        Domain(name="")


def test_domain_evaluate_default():
    d = Domain(name="test")
    assert d.evaluate(None) is True  # no signal = optimistic


def test_domain_evaluate_custom():
    d = Domain(
        name="test",
        success_signal=lambda outcome: outcome.get("score", 0) > 50,
    )
    assert d.evaluate({"score": 80}) is True
    assert d.evaluate({"score": 30}) is False


# ---- Integration: stub provider ----


def test_agent_run_with_stub():
    """Full integration test using the stub provider."""
    import os
    import tempfile

    @tool
    def echo(message: str) -> str:
        """Echo a message back."""
        return f"echo: {message}"

    domain = Domain(
        name="test-integration",
        tools=[echo],
        objectives=["echo things"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        memory = PersistentMemory(tmpdir)

        from nil.agent import Agent

        agent = Agent(
            domain=domain,
            base_model="stub-model",
            memory=memory,
        )

        result = agent.run("Say hello")
        assert result.run_id != ""
        assert result.duration > 0

        # Verify trace was stored
        assert memory.get_trace_count("test-integration") == 1

        # Release SQLite file lock for Windows cleanup
        agent.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
