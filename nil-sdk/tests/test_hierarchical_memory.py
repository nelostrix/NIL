"""Tests for the hierarchical memory system."""

import json
import os
import tempfile
import time

import pytest

from nil.memory.hierarchical import HierarchicalMemory
from nil.memory.episodic import Episode, RetentionTier
from nil.memory.semantic import Fact, FactType
from nil.memory.working import ContentType
from nil.memory.social_memory import UserProfile, Correction
from nil.memory.meta_memory import KnowledgeGap, Contradiction
from nil.memory.skill_registry import TriggerCondition


# ---- Fixtures ----


@pytest.fixture
def memory():
    """Create a hierarchical memory instance in a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hm = HierarchicalMemory(os.path.join(tmpdir, "test_db"))
        yield hm
        hm.close()


@pytest.fixture
def sample_episode():
    return Episode(
        episode_id="ep-1",
        domain="test-domain",
        task="deploy the service",
        steps=[
            {"step_number": 1, "step_type": "tool_call", "tool_name": "build"},
            {"step_number": 2, "step_type": "tool_call", "tool_name": "test"},
            {"step_number": 3, "step_type": "tool_call", "tool_name": "deploy"},
        ],
        outcome={"success": True, "output": "Deployed successfully"},
        importance=0.8,
        session_id="session-1",
    )


@pytest.fixture
def sample_fact():
    return Fact(
        fact_id="fact-1",
        subject="tool:build",
        predicate="precedes",
        object="tool:test",
        confidence=0.9,
        fact_type=FactType.RELATIONSHIP,
        domain="test-domain",
    )


# ---- Episodic Memory Tests ----


class TestEpisodicMemory:
    def test_store_and_recall_episode(self, memory, sample_episode):
        memory.episodic.store_episode(sample_episode)
        recalled = memory.episodic.get_episode("ep-1")
        assert recalled is not None
        assert recalled.task == "deploy the service"
        assert recalled.domain == "test-domain"
        assert len(recalled.steps) == 3

    def test_search_episodes(self, memory):
        ep1 = Episode(episode_id="e1", domain="d", task="fix bug", outcome={"success": True})
        ep2 = Episode(episode_id="e2", domain="d", task="add feature", outcome={"success": False})
        memory.episodic.store_episode(ep1)
        memory.episodic.store_episode(ep2)

        results = memory.episodic.search_episodes(domain="d", limit=10)
        assert len(results) == 2

    def test_timeline(self, memory):
        now = time.time()
        for i in range(5):
            ep = Episode(
                episode_id=f"t-{i}", domain="d", task=f"task-{i}",
                created_at=now - (5 - i) * 10,
            )
            memory.episodic.store_episode(ep)

        results = memory.episodic.timeline("d", start=now - 60, end=now + 60)
        assert len(results) == 5

    def test_importance_computation(self, memory, sample_episode):
        score = memory.episodic.compute_importance(sample_episode)
        assert score > 0.3  # success + 3 steps

    def test_retention_flow(self, memory):
        for i in range(20):
            ep = Episode(
                episode_id=f"r-{i}", domain="d", task=f"task-{i}",
                importance=0.1 + (i * 0.04),
                retention_tier=RetentionTier.RECENT,
            )
            memory.episodic.store_episode(ep)

        forgotten = memory.episodic.apply_retention("d", {
            "max_recent": 10, "max_summarized": 5, "max_compressed": 3,
            "importance_threshold": 0.3,
        })

        recent = memory.episodic.search_episodes("d", tier=RetentionTier.RECENT, limit=100)
        assert len(recent) <= 10

    def test_session_tracking(self, memory):
        session_id = memory.create_session("d", "big task")
        assert session_id != ""

        memory.update_session_activity(session_id)
        memory.close_session(session_id)

    def test_trace_backward_compat(self, memory):
        class FakeTrace:
            def to_dict(self):
                return {
                    "run_id": "trace-1",
                    "domain": "test-domain",
                    "task": "backward compat",
                    "steps": [],
                    "outcome_success": True,
                    "error": None,
                    "total_cost": 0.5,
                    "total_duration": 1.0,
                    "created_at": time.time(),
                }

        memory.store_trace(FakeTrace())
        traces = memory.get_traces("test-domain")
        assert len(traces) == 1
        assert traces[0]["task"] == "backward compat"


# ---- Semantic Memory Tests ----


class TestSemanticMemory:
    def test_observe_and_query_facts(self, memory, sample_fact):
        memory.semantic.observe_fact(sample_fact, source_episode_id="ep-1")
        facts = memory.semantic.query_facts(subject="tool:build")
        assert len(facts) == 1
        assert facts[0].object == "tool:test"

    def test_belief(self, memory):
        fact = Fact(
            subject="error:timeout", predicate="resolved_by",
            object="action:retry", confidence=0.85,
            fact_type=FactType.HEURISTIC, domain="d",
        )
        memory.semantic.observe_fact(fact)
        conf = memory.semantic.belief("error:timeout resolved_by action:retry", domain="d")
        assert conf == 0.85

    def test_relationships(self, memory):
        f1 = Fact(subject="tool:a", predicate="precedes", object="tool:b",
                   confidence=0.8, domain="d")
        f2 = Fact(subject="tool:a", predicate="precedes", object="tool:c",
                   confidence=0.6, domain="d")
        memory.semantic.observe_fact(f1)
        memory.semantic.observe_fact(f2)

        rels = memory.semantic.relationships("tool:a", domain="d")
        assert len(rels) == 2

    def test_knowledge_graph(self, memory):
        f = Fact(subject="tool:x", predicate="precedes", object="tool:y",
                  confidence=0.9, domain="d")
        memory.semantic.observe_fact(f)
        g = memory.semantic.get_knowledge_graph("d")
        assert "tool:x" in g.entities
        assert "tool:y" in g.entities

    def test_decay_confidence(self, memory):
        fact = Fact(subject="s", predicate="p", object="o",
                     confidence=0.9, domain="d",
                     last_confirmed_at=time.time() - 86400 * 5)
        memory.semantic.observe_fact(fact)
        count = memory.semantic.decay_confidence("d", decay_rate=0.05)
        assert count > 0

        post = memory.semantic.query_facts(subject="s", domain="d")
        assert post[0].confidence < 0.9

    def test_entity_tracking(self, memory):
        memory.semantic.track_entity("pagerduty", "service",
                                      {"status": "healthy"}, domain="d")
        memory.semantic.track_entity("pagerduty", "service",
                                      {"status": "degraded"}, domain="d")


# ---- Working Memory Tests ----


class TestWorkingMemory:
    def test_create_and_destroy_session(self, memory):
        sid = memory.working.create_session("fix stuff")
        assert memory.working.session_exists(sid)
        memory.working.destroy_session(sid)
        assert not memory.working.session_exists(sid)

    def test_push_and_pop_context(self, memory):
        sid = memory.working.create_session("test")
        ref = memory.working.push_context(
            sid, "important observation",
            content_type=ContentType.OBSERVATION,
            importance=0.9, tokens=50,
        )
        assert ref != ""

        block = memory.working.pop_context(sid)
        assert block is not None
        assert block.content == "important observation"

    def test_goal_and_subgoal(self, memory):
        sid = memory.working.create_session("complex task")
        memory.working.set_goal(sid, "main goal")
        g = memory.working.add_subgoal(sid, "step 1")
        assert g.description == "step 1"

        memory.working.complete_subgoal(sid, g.goal_id)

    def test_scratchpad(self, memory):
        sid = memory.working.create_session("think")
        memory.working.update_scratchpad(sid, "I think...")
        assert memory.working.get_scratchpad(sid) == "I think..."

    def test_eviction(self, memory):
        memory.working.default_token_budget = 200
        sid = memory.working.create_session("test evict")
        session = memory.working.get_session(sid)
        session.token_budget = 200
        for i in range(20):
            memory.working.push_context(
                sid, f"block-{i}", tokens=50, importance=0.1,
            )
        usage = memory.working.current_token_usage(sid)
        assert usage <= 200

    def test_attend_to(self, memory):
        sid = memory.working.create_session("attend")
        ref = memory.working.push_context(sid, "important", importance=0.9)
        memory.working.attend_to(sid, ref)

    def test_to_episode_dict(self, memory):
        sid = memory.working.create_session("save me")
        memory.working.set_goal(sid, "goal")
        d = memory.working.to_episode_dict(sid)
        assert d["task"] == "save me"
        assert d["current_goal"] == "goal"


# ---- World State Tests ----


class TestWorldStateMemory:
    def test_update_and_get_state(self, memory):
        snap = memory.world_state.update_state("incident-response",
                                                {"status": "healthy", "incidents": 2})
        assert snap.domain == "incident-response"
        assert snap.state_snapshot["status"] == "healthy"

        retrieved = memory.world_state.get_state("incident-response")
        assert retrieved is not None
        assert retrieved.state_snapshot["incidents"] == 2

    def test_state_delta_tracking(self, memory):
        memory.world_state.update_state("d", {"status": "ok"})
        memory.world_state.update_state("d", {"status": "degraded", "incidents": 3})
        memory.world_state.update_state("d", {"status": "ok", "incidents": 0})

        deltas = memory.world_state.get_delta_log("d", field="status")
        assert len(deltas) >= 2
        assert deltas[0].field == "status"

    def test_verify_state_match(self, memory):
        memory.world_state.update_state("d", {"status": "healthy"})
        result = memory.world_state.verify_state("d", {"status": "healthy"})
        assert result["matches"] is True

    def test_verify_state_mismatch(self, memory):
        memory.world_state.update_state("d", {"status": "healthy"})
        result = memory.world_state.verify_state("d", {"status": "degraded"})
        assert result["matches"] is False
        assert "status" in result["discrepancies"]

    def test_predict_state(self, memory):
        pred = memory.world_state.predict_state("d", "restart")
        assert "action" in pred
        assert "predicted_state" in pred

    def test_confidence_decay_on_stale(self, memory):
        memory.world_state.update_state("d", {"status": "healthy"})
        snap = memory.world_state.get_state("d")
        assert snap.confidence == 1.0


# ---- Social Memory Tests ----


class TestSocialMemory:
    def test_get_or_create_user(self, memory):
        profile = memory.social.get_or_create_user("alice", "Alice")
        assert profile.user_id == "alice"
        assert profile.name == "Alice"

    def test_record_interaction(self, memory):
        memory.social.record_interaction("bob", "incident-response", request="help")
        profile = memory.social.get_or_create_user("bob")
        assert profile.interaction_count >= 1
        assert "help" in profile.common_requests

    def test_record_correction(self, memory):
        memory.social.get_or_create_user("carol", "Carol")
        memory.social.record_correction(
            "carol",
            Correction(episode_id="ep-1", original="wrong", corrected="right", domain="d"),
        )

        correction_history = memory.social.get_correction_history("carol")
        assert len(correction_history) == 1

    def test_adapt_to_user(self, memory):
        profile = memory.social.get_or_create_user("dave")
        original = "This is a very long response with lots of detail about the system"
        adapted = memory.social.adapt_to_user("dave", original)
        assert isinstance(adapted, str)


# ---- Meta Memory Tests ----


class TestMetaMemory:
    def test_identify_gap(self, memory):
        gap = memory.meta.identify_gap(
            "tool:unknown_relationships",
            evidence="No data for tool X",
            domain="d",
        )
        assert gap.area == "tool:unknown_relationships"
        assert gap.domain == "d"

    def test_knowledge_gaps(self, memory):
        memory.meta.identify_gap("area1", domain="d")
        memory.meta.identify_gap("area2", evidence="missing data", domain="d")
        gaps = memory.meta.knowledge_gaps(domain="d")
        assert len(gaps) >= 2

    def test_detect_contradictions(self, memory):
        f1 = Fact(subject="tool:a", predicate="precedes", object="tool:b",
                   confidence=0.8, fact_type=FactType.RELATIONSHIP, domain="d")
        f2 = Fact(subject="tool:a", predicate="precedes", object="tool:c",
                   confidence=0.7, fact_type=FactType.RELATIONSHIP, domain="d")
        memory.semantic.observe_fact(f1)
        memory.semantic.observe_fact(f2)

        contras = memory.meta.detect_contradictions("d")
        assert len(contras) == 0  # no contradiction (different objects)

        f3 = Fact(subject="tool:x", predicate="status", object="healthy",
                   confidence=0.8, domain="d")
        f4 = Fact(subject="tool:x", predicate="status", object="degraded",
                   confidence=0.7, domain="d")
        memory.semantic.observe_fact(f3)
        memory.semantic.observe_fact(f4)

        contras = memory.meta.detect_contradictions("d")
        assert len(contras) >= 1

    def test_resolve_contradiction(self, memory):
        f1 = Fact(subject="s", predicate="p", object="a", confidence=0.8, domain="d")
        f2 = Fact(subject="s", predicate="p", object="b", confidence=0.7, domain="d")
        memory.semantic.observe_fact(f1)
        memory.semantic.observe_fact(f2)
        contras = memory.meta.detect_contradictions("d")
        if contras:
            memory.meta.resolve_contradiction(contras[0].contradiction_id, "both valid")
            remaining = memory.meta.contradictions(domain="d", unresolved_only=True)
            assert len(remaining) == 0

    def test_knowledge_summary(self, memory):
        summary = memory.meta.knowledge_summary("d")
        assert "known_facts_count" in summary
        assert "knowledge_gaps" in summary

    def test_audit(self, memory):
        result = memory.meta.audit("d")
        assert "domain" in result
        assert "gaps_found" in result


# ---- Procedural Memory Tests ----


class TestProceduralMemory:
    def test_store_and_retrieve_procedure(self, memory):
        from nil.memory.procedural import Procedure
        proc = Procedure(
            procedure_id="proc-1",
            domain="d",
            name="build_test_deploy",
            steps=[
                {"tool": "build", "type": "tool_call"},
                {"tool": "test", "type": "tool_call"},
            ],
            trigger_conditions=TriggerCondition(keywords=["deploy"]),
            success_rate=0.8,
        )
        memory.procedural.store_procedure(proc)

        retrieved = memory.procedural.get_procedure("proc-1")
        assert retrieved is not None
        assert retrieved.name == "build_test_deploy"

    def test_applicable_procedures(self, memory):
        from nil.memory.procedural import Procedure
        p1 = Procedure(
            procedure_id="p1", domain="d", name="fix_bug",
            trigger_conditions=TriggerCondition(keywords=["bug", "fix"]),
            success_rate=0.9,
        )
        p2 = Procedure(
            procedure_id="p2", domain="d", name="deploy",
            trigger_conditions=TriggerCondition(keywords=["deploy"]),
            success_rate=0.7,
        )
        memory.procedural.store_procedure(p1)
        memory.procedural.store_procedure(p2)

        applicable = memory.procedural.applicable_procedures("d", task="fix a bug")
        assert len(applicable) >= 1
        assert applicable[0].procedure_id == "p1"

    def test_record_usage(self, memory):
        from nil.memory.procedural import Procedure
        proc = Procedure(procedure_id="pu-1", domain="d", name="test_usage")
        memory.procedural.store_procedure(proc)
        memory.procedural.record_usage("pu-1", success=True, cost=0.5, duration=1.0)
        memory.procedural.record_usage("pu-1", success=False, cost=0.3, duration=2.0)

        updated = memory.procedural.get_procedure("pu-1")
        assert updated.usage_count == 2
        assert updated.success_rate == 0.5

    def test_record_refinement(self, memory):
        from nil.memory.procedural import RefinementEvent
        event = RefinementEvent(
            procedure_id="pr-1", previous_version=1, new_version=2,
            success_rate_delta=0.1, cost_delta=-0.05,
        )
        memory.procedural.record_refinement(event)


# ---- Hierarchical Memory Integration Tests ----


class TestHierarchicalMemoryAPI:
    def test_trace_creates_episode(self, memory):
        class FakeTrace:
            def to_dict(self):
                return {
                    "run_id": "hm-1", "domain": "d", "task": "integration",
                    "steps": [{"step_number": 1, "step_type": "tool_call", "tool_name": "search"}],
                    "outcome_success": True, "error": None,
                    "total_cost": 0.1, "total_duration": 0.5,
                    "created_at": time.time(),
                }

        memory.store_trace(FakeTrace())
        assert memory.get_trace_count("d") == 1
        assert memory.get_success_rate("d") == 1.0

    def test_store_and_list_skills(self, memory):
        memory.store_skill("d", "search_diagnostics", "Search logs for errors")
        memory.store_skill("d", "apply_fix", "Apply a remediation", confidence=0.8)
        skills = memory.list_skills("d")
        assert len(skills) >= 2

    def test_strategy_storage(self, memory):
        version = memory.store_strategy("d", {"prompt": "be careful"}, score=0.9)
        assert version == 1

        strategy = memory.get_active_strategy("d")
        assert strategy is not None
        assert strategy["prompt"] == "be careful"

    def test_full_consolidation(self, memory):
        class FakeTrace:
            def to_dict(self):
                return {
                    "run_id": f"c-{i}", "domain": "d", "task": f"task-{i}",
                    "steps": [
                        {"step_number": 1, "step_type": "tool_call", "tool_name": "build"},
                        {"step_number": 2, "step_type": "tool_call", "tool_name": "test"},
                    ],
                    "outcome_success": True, "error": None,
                    "total_cost": 0.1, "total_duration": 0.5,
                    "created_at": time.time(),
                }

        for i in range(5):
            memory.store_trace(FakeTrace())

        report = memory.consolidate("d")
        assert report.facts_extracted >= 0
        assert isinstance(report.duration_ms, float)

    def test_cross_tier_flow(self, memory):
        session_id = memory.create_session("d", "big investigation")
        assert session_id != ""

        sid = memory.create_working_session("live debug")
        memory.push_context(sid, "found error X", ContentType.OBSERVATION, importance=0.9)

        memory.update_state("d", {"status": "degraded", "error_rate": 0.05})
        snap = memory.get_state("d")
        assert snap is not None
        assert snap.state_snapshot["status"] == "degraded"

        profile = memory.user_profile("alice")
        assert profile is not None

        gap = memory.identify_gap("tool:unknown", domain="d")
        assert gap is not None

    def test_fact_extraction_during_consolidation(self, memory):
        for i in range(3):
            ep = Episode(
                episode_id=f"fe-{i}", domain="d", task="deploy",
                steps=[
                    {"step_number": 1, "step_type": "tool_call", "tool_name": "build"},
                    {"step_number": 2, "step_type": "tool_call", "tool_name": "deploy"},
                ],
                outcome={"success": True},
            )
            memory.episodic.store_episode(ep)

        memory.consolidate("d")
        facts = memory.query_facts(domain="d")
        assert len(facts) > 0
        has_precedes = any(
            f.subject == "tool:build" and f.predicate == "precedes"
            for f in facts
        )
        assert has_precedes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
