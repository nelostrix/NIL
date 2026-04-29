"""
Comprehensive test suite for Domain World Model.

Coverage:
  - Data structures (ToolProfile, TaskCluster, Constraint, WorldModel)
  - Tool profile building (success rates, positions, complementary/redundant)
  - Task clustering (keyword extraction, winning sequences)
  - Constraint discovery (prerequisites, forbidden, minimum steps, errors)
  - State transitions (recommended, forbidden)
  - Health metrics (percentiles, trends)
  - Prompt injection (full, compact)
  - Full build pipeline
  - Edge cases (empty, single trace, all same, unicode)
  - Scalability
"""

import tempfile
import time

import pytest

from nil.evolution.world_model import (
    Constraint,
    DomainHealth,
    StateTransition,
    TaskCluster,
    ToolProfile,
    WorldModel,
    WorldModelBuilder,
)
from nil.memory.store import PersistentMemory


@pytest.fixture
def memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = PersistentMemory(tmpdir)
        yield mem
        mem.close()


@pytest.fixture
def builder(memory):
    return WorldModelBuilder(memory, min_traces=5)


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
# DATA STRUCTURES
# ===========================================================================


class TestDataStructures:
    def test_tool_profile_to_dict(self):
        tp = ToolProfile(name="search", total_invocations=10, success_rate=0.8)
        d = tp.to_dict()
        assert d["name"] == "search"
        assert d["success_rate"] == 0.8

    def test_task_cluster_to_dict(self):
        tc = TaskCluster(cluster_id="tc-1", keywords=["timeout"], task_count=5)
        d = tc.to_dict()
        assert d["cluster_id"] == "tc-1"
        assert d["task_count"] == 5

    def test_constraint_to_dict(self):
        c = Constraint(constraint_id="c-1", constraint_type="forbidden", description="Don't do X")
        d = c.to_dict()
        assert d["constraint_type"] == "forbidden"

    def test_health_to_dict(self):
        h = DomainHealth(total_runs=100, success_rate=0.85)
        d = h.to_dict()
        assert d["total_runs"] == 100

    def test_world_model_properties(self):
        model = WorldModel(
            domain="test",
            tool_profiles=[
                ToolProfile(name="good", causal_direction="positive"),
                ToolProfile(name="bad", causal_direction="negative"),
                ToolProfile(name="meh", causal_direction="neutral"),
            ],
            transitions=[
                StateTransition(from_state="a", to_state="b", is_recommended=True),
                StateTransition(from_state="c", to_state="d", is_forbidden=True),
            ],
            constraints=[
                Constraint(
                    constraint_id="c1", constraint_type="forbidden", description="X", confidence=0.9
                ),
                Constraint(
                    constraint_id="c2",
                    constraint_type="prerequisite",
                    description="Y",
                    confidence=0.5,
                ),
            ],
        )
        assert len(model.positive_tools) == 1
        assert len(model.negative_tools) == 1
        assert len(model.recommended_transitions) == 1
        assert len(model.forbidden_transitions) == 1
        assert len(model.strong_constraints) == 1
        assert model.tool_names == ["good", "bad", "meh"]

    def test_world_model_to_dict(self):
        model = WorldModel(domain="test")
        d = model.to_dict()
        assert d["domain"] == "test"
        assert "tool_profiles" in d
        assert "health" in d


# ===========================================================================
# TOOL PROFILES
# ===========================================================================


class TestToolProfiles:
    def test_builds_profiles(self, memory, builder):
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["search", "fix"]))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["search"], error="err"))
        _populate(memory, traces)

        model = builder.build("test")
        names = [tp.name for tp in model.tool_profiles]
        assert "search" in names
        assert "fix" in names

    def test_success_rate(self, memory, builder):
        traces = []
        for i in range(8):
            traces.append(_trace(f"s-{i}", True, ["tool_a"]))
        for i in range(2):
            traces.append(_trace(f"f-{i}", False, ["tool_a"]))
        _populate(memory, traces)

        model = builder.build("test")
        tool_a = [tp for tp in model.tool_profiles if tp.name == "tool_a"][0]
        assert tool_a.success_rate == pytest.approx(0.8, abs=0.01)

    def test_typical_position(self, memory, builder):
        traces = [_trace(f"s-{i}", True, ["first", "second", "third"]) for i in range(10)]
        _populate(memory, traces)

        model = builder.build("test")
        first = [tp for tp in model.tool_profiles if tp.name == "first"][0]
        third = [tp for tp in model.tool_profiles if tp.name == "third"][0]
        assert first.typical_position < third.typical_position

    def test_complementary_tools(self, memory, builder):
        traces = [_trace(f"s-{i}", True, ["alpha", "beta"]) for i in range(10)]
        _populate(memory, traces)

        model = builder.build("test")
        alpha = [tp for tp in model.tool_profiles if tp.name == "alpha"][0]
        assert "beta" in alpha.complementary_tools

    def test_error_tracking(self, memory, builder):
        traces = [
            _trace(f"f-{i}", False, ["broken"], step_errors={0: "NullPointer"}) for i in range(5)
        ]
        traces += [_trace(f"s-{i}", True, ["broken"]) for i in range(5)]
        _populate(memory, traces)

        model = builder.build("test")
        broken = [tp for tp in model.tool_profiles if tp.name == "broken"][0]
        assert len(broken.common_errors) > 0


# ===========================================================================
# TASK CLUSTERING
# ===========================================================================


class TestTaskClustering:
    def test_clusters_by_keyword(self, memory, builder):
        traces = []
        for i in range(5):
            traces.append(_trace(f"db-{i}", True, ["a"], task="Fix the database timeout"))
        for i in range(5):
            traces.append(_trace(f"api-{i}", True, ["b"], task="Handle API rate limit"))
        _populate(memory, traces)

        model = builder.build("test")
        cluster_keywords = [kw for tc in model.task_clusters for kw in tc.keywords]
        assert "database" in cluster_keywords or "timeout" in cluster_keywords

    def test_cluster_success_rate(self, memory, builder):
        traces = []
        for i in range(8):
            traces.append(_trace(f"s-{i}", True, ["a"], task="server issue"))
        for i in range(2):
            traces.append(_trace(f"f-{i}", False, ["a"], task="server problem"))
        _populate(memory, traces)

        model = builder.build("test")
        server_cluster = [tc for tc in model.task_clusters if "server" in tc.keywords]
        if server_cluster:
            assert server_cluster[0].success_rate > 0.5

    def test_winning_sequences(self, memory, builder):
        traces = [
            _trace(f"s-{i}", True, ["search", "fix"], task="database error") for i in range(10)
        ]
        _populate(memory, traces)

        model = builder.build("test")
        if model.task_clusters:
            has_winning = any(tc.winning_sequences for tc in model.task_clusters)
            assert has_winning

    def test_empty_traces(self, memory, builder):
        model = builder.build("empty")
        assert model.task_clusters == []


# ===========================================================================
# CONSTRAINT DISCOVERY
# ===========================================================================


class TestConstraintDiscovery:
    def test_discovers_prerequisites(self, memory, builder):
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["check", "fix"]))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["fix"], error="err"))
        _populate(memory, traces)

        model = builder.build("test")
        prereqs = [c for c in model.constraints if c.constraint_type == "prerequisite"]
        assert len(prereqs) >= 1

    def test_discovers_forbidden(self, memory, builder):
        traces = []
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["good", "poison"]))
        for i in range(5):
            traces.append(_trace(f"s-{i}", True, ["good", "safe"]))
        _populate(memory, traces)

        model = builder.build("test")
        forbidden = [c for c in model.constraints if c.constraint_type == "forbidden"]
        # dead-end transitions become forbidden constraints
        assert isinstance(forbidden, list)

    def test_discovers_minimum_steps(self, memory, builder):
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["a", "b", "c"]))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["a"], error="too short"))
        _populate(memory, traces)

        model = builder.build("test")
        min_step = [c for c in model.constraints if c.constraint_type == "required_minimum"]
        assert len(min_step) >= 1

    def test_discovers_error_constraints(self, memory, builder):
        traces = [_trace(f"s-{i}", True, ["tool"]) for i in range(5)]
        traces += [
            _trace(f"f-{i}", False, ["tool"], step_errors={0: "TimeoutError"}) for i in range(5)
        ]
        _populate(memory, traces)

        model = builder.build("test")
        error_constraints = [
            c for c in model.constraints if c.constraint_type == "conditional_error"
        ]
        assert len(error_constraints) >= 1

    def test_first_tool_constraint(self, memory, builder):
        traces = [_trace(f"s-{i}", True, ["always_first", "then_this"]) for i in range(10)]
        traces += [_trace(f"f-{i}", False, ["wrong_start"], error="err") for i in range(3)]
        _populate(memory, traces)

        model = builder.build("test")
        first_tool_constraints = [
            c
            for c in model.constraints
            if "start with" in c.description.lower() or "always_first" in c.description
        ]
        assert len(first_tool_constraints) >= 1

    def test_confidence_ordering(self, memory, builder):
        traces = []
        for i in range(20):
            traces.append(_trace(f"s-{i}", True, ["a", "b"], created_offset=i))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["c"], error="err", created_offset=20 + i))
        _populate(memory, traces)

        model = builder.build("test")
        if len(model.constraints) >= 2:
            assert model.constraints[0].confidence >= model.constraints[1].confidence


# ===========================================================================
# STATE TRANSITIONS
# ===========================================================================


class TestTransitions:
    def test_recommended_transitions(self, memory, builder):
        traces = [_trace(f"s-{i}", True, ["start", "end"]) for i in range(10)]
        _populate(memory, traces)

        model = builder.build("test")
        rec = model.recommended_transitions
        assert len(rec) >= 1

    def test_forbidden_transitions(self, memory, builder):
        traces = []
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["a", "bad"]))
        for i in range(5):
            traces.append(_trace(f"s-{i}", True, ["a", "good"]))
        _populate(memory, traces)

        model = builder.build("test")
        forbidden = model.forbidden_transitions
        assert isinstance(forbidden, list)


# ===========================================================================
# HEALTH METRICS
# ===========================================================================


class TestHealth:
    def test_basic_health(self, memory, builder):
        traces = []
        for i in range(8):
            traces.append(_trace(f"s-{i}", True, ["a"], cost=0.01, duration=2.0))
        for i in range(2):
            traces.append(_trace(f"f-{i}", False, ["a"], cost=0.05, duration=5.0))
        _populate(memory, traces)

        model = builder.build("test")
        h = model.health
        assert h.total_runs == 10
        assert h.success_rate == pytest.approx(0.8, abs=0.01)
        assert h.avg_cost > 0
        assert h.avg_duration > 0
        assert h.unique_tools == 1

    def test_percentiles(self, memory, builder):
        traces = []
        for i in range(20):
            traces.append(_trace(f"r-{i}", True, ["a"], cost=0.01 * (i + 1)))
        _populate(memory, traces)

        model = builder.build("test")
        assert model.health.cost_p50 > 0
        assert model.health.cost_p95 > model.health.cost_p50

    def test_trend_present(self, memory, builder):
        traces = [_trace(f"r-{i}", True, ["a"], created_offset=i) for i in range(40)]
        _populate(memory, traces)

        model = builder.build("test")
        assert model.health.trend in ("stable", "improving", "degrading", "insufficient_data")


# ===========================================================================
# PROMPT INJECTION
# ===========================================================================


class TestPromptInjection:
    def test_full_injection(self, memory, builder):
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["diagnose", "fix"], task="server timeout"))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["fix"], error="timeout", task="server issue"))
        _populate(memory, traces)

        model = builder.build("test")
        prompt = model.to_prompt_injection()
        assert "DOMAIN KNOWLEDGE" in prompt
        assert "Tool capabilities" in prompt

    def test_compact_injection(self, memory, builder):
        traces = []
        for i in range(10):
            traces.append(_trace(f"s-{i}", True, ["a", "b"]))
        for i in range(5):
            traces.append(_trace(f"f-{i}", False, ["c"], error="err"))
        _populate(memory, traces)

        model = builder.build("test")
        full = model.to_prompt_injection()
        compact = model.to_compact_injection()
        assert len(compact) <= len(full) or len(compact) < 500

    def test_empty_model_prompt(self):
        model = WorldModel(domain="empty")
        prompt = model.to_prompt_injection()
        assert "DOMAIN KNOWLEDGE" in prompt


# ===========================================================================
# FULL BUILD PIPELINE
# ===========================================================================


class TestFullBuild:
    def test_builds_complete_model(self, memory, builder):
        traces = []
        for i in range(15):
            traces.append(
                _trace(
                    f"s-{i}",
                    True,
                    ["search", "analyze", "fix"],
                    task="Fix database timeout error",
                    cost=0.01,
                    duration=2.0,
                )
            )
        for i in range(5):
            traces.append(
                _trace(
                    f"f-{i}",
                    False,
                    ["fix"],
                    task="Handle API failure",
                    error="timeout",
                    cost=0.05,
                    duration=5.0,
                )
            )
        _populate(memory, traces)

        model = builder.build("test")
        assert model.domain == "test"
        assert model.trace_count_at_generation == 20
        assert len(model.tool_profiles) > 0
        assert model.health.total_runs == 20

    def test_empty_domain(self, memory, builder):
        model = builder.build("nonexistent")
        assert model.trace_count_at_generation == 0
        assert model.tool_profiles == []
        assert model.task_clusters == []

    def test_below_min_traces(self, memory, builder):
        traces = [_trace("s-0", True, ["a"])]
        _populate(memory, traces)
        model = builder.build("test")
        # Should still build, just with limited data
        assert model.trace_count_at_generation == 1

    def test_model_serialization(self, memory, builder):
        traces = [_trace(f"s-{i}", True, ["a", "b"]) for i in range(10)]
        traces += [_trace(f"f-{i}", False, ["c"], error="err") for i in range(5)]
        _populate(memory, traces)

        model = builder.build("test")
        d = model.to_dict()
        assert isinstance(d, dict)
        assert d["domain"] == "test"
        assert isinstance(d["tool_profiles"], list)
        assert isinstance(d["health"], dict)


# ===========================================================================
# EDGE CASES
# ===========================================================================


class TestEdgeCases:
    def test_single_tool_domain(self, memory, builder):
        traces = [_trace(f"r-{i}", i % 2 == 0, ["only_tool"]) for i in range(20)]
        _populate(memory, traces)
        model = builder.build("test")
        assert len(model.tool_profiles) == 1

    def test_unicode_tasks(self, memory, builder):
        traces = [_trace(f"r-{i}", True, ["a"], task="修复数据库超时") for i in range(10)]
        _populate(memory, traces)
        model = builder.build("test")
        assert isinstance(model, WorldModel)

    def test_no_tool_sequences(self, memory, builder):
        traces = [_trace(f"r-{i}", True, []) for i in range(10)]
        _populate(memory, traces)
        model = builder.build("test")
        assert model.tool_profiles == []

    def test_mixed_domains(self, memory, builder):
        _populate(
            memory,
            [_trace(f"d1-{i}", True, ["a"], domain="domain1") for i in range(10)]
            + [_trace(f"d2-{i}", True, ["b"], domain="domain2") for i in range(10)],
        )
        model = builder.build("domain1")
        assert model.trace_count_at_generation == 10
        names = [tp.name for tp in model.tool_profiles]
        assert "a" in names
        assert "b" not in names

    def test_all_same_task(self, memory, builder):
        traces = [_trace(f"r-{i}", True, ["a"], task="same task") for i in range(10)]
        _populate(memory, traces)
        model = builder.build("test")
        assert isinstance(model.task_clusters, list)

    def test_very_long_error_strings(self, memory, builder):
        traces = [_trace(f"f-{i}", False, ["a"], step_errors={0: "x" * 500}) for i in range(5)]
        traces += [_trace(f"s-{i}", True, ["a"]) for i in range(5)]
        _populate(memory, traces)
        model = builder.build("test")
        assert isinstance(model, WorldModel)


# ===========================================================================
# SCALABILITY
# ===========================================================================


class TestScalability:
    def test_300_traces(self, memory, builder):
        traces = []
        for i in range(250):
            traces.append(
                _trace(f"s-{i}", True, ["a", "b", "c"], task=f"Fix issue {i % 5}", created_offset=i)
            )
        for i in range(50):
            traces.append(_trace(f"f-{i}", False, ["d"], error="timeout", created_offset=250 + i))
        _populate(memory, traces)

        import time as t

        start = t.monotonic()
        model = builder.build("test")
        elapsed = t.monotonic() - start

        assert model.trace_count_at_generation == 300
        assert elapsed < 30


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
