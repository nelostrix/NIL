"""
End-to-end test: full agent lifecycle from creation to evolution.

This test exercises the complete pipeline:
    Agent creation → Run → Trace storage → Analysis → Skill extraction → Drift detection
"""

import tempfile
import time

import pytest

import nil
from nil.evolution.analyzer import TraceAnalyzer
from nil.evolution.drift_detector import DriftDetector
from nil.evolution.skill_extractor import SkillExtractor
from nil.memory.experience import ExperienceQuery


@nil.tool
def diagnose(symptom: str) -> str:
    """Diagnose the root cause of a problem."""
    return f"Root cause identified for '{symptom}': database connection pool exhausted"


@nil.tool
def fix(action: str) -> str:
    """Apply a fix."""
    return f"Fix applied: {action}"


@nil.tool
def verify(check: str) -> str:
    """Verify the fix worked."""
    return f"Verification passed: {check}"


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestFullLifecycle:
    """Test the complete agent lifecycle end-to-end."""

    def test_create_agent(self, workspace):
        domain = nil.Domain(
            name="e2e-test",
            tools=[diagnose, fix, verify],
            objectives=["resolve incidents"],
        )
        memory = nil.PersistentMemory(workspace)
        agent = nil.Agent(
            domain=domain,
            base_model="stub",
            memory=memory,
        )
        assert agent.run_count == 0
        assert repr(agent).startswith("Agent(")

    def test_run_stores_trace(self, workspace):
        domain = nil.Domain(name="e2e-test", tools=[diagnose])
        memory = nil.PersistentMemory(workspace)
        agent = nil.Agent(domain=domain, memory=memory)

        result = agent.run("server is slow")

        assert result.run_id != ""
        assert result.duration > 0
        assert memory.get_trace_count("e2e-test") == 1

    def test_multiple_runs_accumulate(self, workspace):
        domain = nil.Domain(name="e2e-test", tools=[diagnose, fix])
        memory = nil.PersistentMemory(workspace)
        agent = nil.Agent(domain=domain, memory=memory)

        for i in range(5):
            agent.run(f"incident #{i}")

        assert agent.run_count == 5
        assert memory.get_trace_count("e2e-test") == 5

    def test_experience_query_on_real_runs(self, workspace):
        domain = nil.Domain(name="e2e-test", tools=[diagnose, fix, verify])
        memory = nil.PersistentMemory(workspace)
        agent = nil.Agent(domain=domain, memory=memory)

        for i in range(10):
            agent.run(f"incident #{i}")

        eq = ExperienceQuery(memory)
        stats = eq.stats("e2e-test")

        assert stats.total_runs == 10
        assert stats.avg_duration > 0

    def test_analyzer_on_real_runs(self, workspace):
        domain = nil.Domain(name="e2e-test", tools=[diagnose, fix])
        memory = nil.PersistentMemory(workspace)
        agent = nil.Agent(domain=domain, memory=memory)

        for i in range(15):
            agent.run(f"incident #{i}")

        analyzer = TraceAnalyzer(memory)
        trend = analyzer.detect_performance_trend("e2e-test", window=7)
        assert trend["status"] in ("stable", "improving", "degrading", "insufficient_data")

    def test_drift_detector_no_drift(self, workspace):
        domain = nil.Domain(name="e2e-test", tools=[diagnose])
        memory = nil.PersistentMemory(workspace)
        agent = nil.Agent(domain=domain, memory=memory)

        for i in range(40):
            agent.run(f"task #{i}")

        detector = DriftDetector(memory)
        result = detector.check("e2e-test")
        assert "drift_detected" in result
        assert "severity" in result

    def test_evolution_config_manual(self, workspace):
        domain = nil.Domain(name="e2e-test", tools=[diagnose])
        memory = nil.PersistentMemory(workspace)
        agent = nil.Agent(
            domain=domain,
            memory=memory,
            evolution=nil.EvolutionConfig(strategy="manual"),
        )

        for i in range(5):
            agent.run(f"task #{i}")

        # Manual evolution should not crash
        agent.evolve()

    def test_evolution_config_continuous(self, workspace):
        domain = nil.Domain(name="e2e-test", tools=[diagnose])
        memory = nil.PersistentMemory(workspace)
        agent = nil.Agent(
            domain=domain,
            memory=memory,
            evolution=nil.EvolutionConfig(strategy="continuous"),
        )

        # Continuous evolution triggers after every run — should not crash
        for i in range(3):
            agent.run(f"task #{i}")

        assert agent.run_count == 3

    def test_learned_skills_initially_empty(self, workspace):
        domain = nil.Domain(name="e2e-test", tools=[diagnose])
        memory = nil.PersistentMemory(workspace)
        agent = nil.Agent(domain=domain, memory=memory)

        assert agent.learned_skills() == []

    def test_skill_extraction_stores_to_memory(self, workspace):
        """Manually run skill extraction and verify it persists."""
        memory = nil.PersistentMemory(workspace)

        # Manually insert traces with consistent success pattern
        for i in range(6):
            memory.store_trace(
                {
                    "run_id": f"manual-{i}",
                    "domain": "e2e-test",
                    "task": f"task-{i}",
                    "steps": [
                        {
                            "step_number": 1,
                            "step_type": "tool_call",
                            "tool_name": "diagnose",
                            "tool_input": {},
                            "tool_output": "ok",
                            "duration_ms": 10,
                            "tokens_in": 0,
                            "tokens_out": 0,
                            "error": None,
                            "timestamp": time.time(),
                        },
                        {
                            "step_number": 2,
                            "step_type": "tool_call",
                            "tool_name": "fix",
                            "tool_input": {},
                            "tool_output": "ok",
                            "duration_ms": 10,
                            "tokens_in": 0,
                            "tokens_out": 0,
                            "error": None,
                            "timestamp": time.time(),
                        },
                    ],
                    "outcome_success": True,
                    "outcome_output": "done",
                    "total_duration": 1.0,
                    "total_tokens_in": 100,
                    "total_tokens_out": 50,
                    "total_cost": 0.001,
                    "tool_sequence": ["diagnose", "fix"],
                    "error": None,
                    "created_at": time.time() + i,
                    "metadata": {},
                }
            )

        # Add failure traces with different sequence (needed for Fisher significance)
        for i in range(4):
            memory.store_trace(
                {
                    "run_id": f"fail-{i}",
                    "domain": "e2e-test",
                    "task": f"fail-task-{i}",
                    "steps": [
                        {
                            "step_number": 1,
                            "step_type": "tool_call",
                            "tool_name": "other",
                            "tool_input": {},
                            "tool_output": "err",
                            "duration_ms": 10,
                            "tokens_in": 0,
                            "tokens_out": 0,
                            "error": "failed",
                            "timestamp": time.time(),
                        },
                    ],
                    "outcome_success": False,
                    "outcome_output": "failed",
                    "total_duration": 1.0,
                    "total_tokens_in": 50,
                    "total_tokens_out": 20,
                    "total_cost": 0.001,
                    "tool_sequence": ["other"],
                    "error": "something broke",
                    "created_at": time.time() + 10 + i,
                    "metadata": {},
                }
            )

        extractor = SkillExtractor(memory)
        result = extractor.extract("e2e-test", min_frequency=3, min_confidence=0.5)

        assert result.new_skill_count >= 1
        assert "diagnose" in result.extracted[0].name

        # Verify skill persisted to memory
        stored_skills = memory.list_skills("e2e-test")
        assert len(stored_skills) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
