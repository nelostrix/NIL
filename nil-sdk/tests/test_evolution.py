"""Tests for nil.evolution — analyzer, skill_extractor, drift_detector."""

import tempfile
import time

import pytest

from nil.evolution.analyzer import TraceAnalyzer
from nil.evolution.drift_detector import DriftDetector
from nil.evolution.skill_extractor import SkillExtractor
from nil.memory.store import PersistentMemory


@pytest.fixture
def memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = PersistentMemory(tmpdir)
        yield mem
        mem.close()


def _trace(run_id, success, tools, domain="test", created_offset=0):
    return {
        "run_id": run_id,
        "domain": domain,
        "task": f"task-{run_id}",
        "steps": [
            {
                "step_number": i + 1,
                "step_type": "tool_call",
                "tool_name": t,
                "tool_input": {},
                "tool_output": "ok",
                "duration_ms": 50,
                "tokens_in": 0,
                "tokens_out": 0,
                "error": None,
                "timestamp": time.time(),
            }
            for i, t in enumerate(tools)
        ],
        "outcome_success": success,
        "outcome_output": "done",
        "total_duration": 1.0,
        "total_tokens_in": 100,
        "total_tokens_out": 50,
        "total_cost": 0.001,
        "tool_sequence": tools,
        "error": None if success else "failed",
        "created_at": time.time() + created_offset,
        "metadata": {},
    }


class TestAnalyzer:
    def test_find_winning_patterns(self, memory):
        # 5 successful runs with same sequence
        for i in range(5):
            memory.store_trace(_trace(f"s-{i}", True, ["search", "fix", "verify"]))
        # 2 failed runs with different sequence
        for i in range(2):
            memory.store_trace(_trace(f"f-{i}", False, ["fix"]))

        analyzer = TraceAnalyzer(memory)
        patterns = analyzer.find_winning_patterns("test", min_frequency=3)
        assert len(patterns) >= 1
        assert patterns[0]["sequence"] == ["search", "fix", "verify"]
        assert patterns[0]["success_rate"] > 0.7

    def test_detect_stable_trend(self, memory):
        for i in range(40):
            memory.store_trace(
                _trace(f"r-{i}", success=(i % 3 != 0), tools=["a"], created_offset=i)
            )

        analyzer = TraceAnalyzer(memory)
        trend = analyzer.detect_performance_trend("test", window=20)
        assert trend["status"] in ("stable", "improving", "degrading")

    def test_insufficient_data(self, memory):
        memory.store_trace(_trace("r-0", True, ["a"]))
        analyzer = TraceAnalyzer(memory)
        trend = analyzer.detect_performance_trend("test", window=20)
        assert trend["status"] == "insufficient_data"


class TestSkillExtractor:
    def test_extract_skill(self, memory):
        for i in range(5):
            memory.store_trace(_trace(f"s-{i}", True, ["diagnose", "patch", "test"]))
        for i in range(1):
            memory.store_trace(_trace(f"f-{i}", False, ["patch"]))

        extractor = SkillExtractor(memory)
        skills = extractor.extract("test", min_frequency=3, min_confidence=0.7)
        assert len(skills) >= 1
        assert "diagnose" in skills[0].name

    def test_no_extraction_below_threshold(self, memory):
        # Only 1 successful run — below min_frequency
        memory.store_trace(_trace("s-0", True, ["a", "b"]))

        extractor = SkillExtractor(memory)
        skills = extractor.extract("test", min_frequency=3)
        assert len(skills) == 0


class TestDriftDetector:
    def test_no_drift(self, memory):
        for i in range(40):
            memory.store_trace(
                _trace(f"r-{i}", success=True, tools=["a"], created_offset=i)
            )

        detector = DriftDetector(memory)
        result = detector.check("test")
        assert result["drift_detected"] is False
        assert result["severity"] == "none"

    def test_drift_detected(self, memory):
        # Old runs: all successful
        for i in range(20):
            memory.store_trace(
                _trace(f"old-{i}", success=True, tools=["a"], created_offset=i)
            )
        # Recent runs: mostly failing
        for i in range(20):
            memory.store_trace(
                _trace(f"new-{i}", success=(i > 17), tools=["a"], created_offset=20 + i)
            )

        detector = DriftDetector(memory)
        result = detector.check("test")
        assert result["drift_detected"] is True
        assert result["severity"] in ("significant", "critical")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
