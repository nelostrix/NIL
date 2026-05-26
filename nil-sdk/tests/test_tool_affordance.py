from __future__ import annotations

import pytest

from nil.tools.base import Tool, tool
from nil.tools.embeddings import ToolEmbedding, ToolEmbeddingGenerator, _cosine_sim, _text_to_embedding
from nil.tools.affordance import ToolAffordance, AffordanceLearner, ToolAffordanceGraph, _TraceStub
from nil.tools.composition import CompositeTool, ToolCompositionSynthesizer
from nil.tools.selection import LatentToolSelector, LatentAwarePromptBuilder, ScoredTool


class TestCosineSim:
    def test_identical(self):
        assert _cosine_sim([1, 0], [1, 0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine_sim([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_empty(self):
        assert _cosine_sim([], [1]) == 0.0
        assert _cosine_sim([1], []) == 0.0

    def test_zero_vector(self):
        assert _cosine_sim([0, 0], [1, 0]) == 0.0


class TestTextToEmbedding:
    def test_returns_list(self):
        emb = _text_to_embedding("hello")
        assert isinstance(emb, list)
        assert len(emb) > 0

    def test_deterministic(self):
        a = _text_to_embedding("hello")
        b = _text_to_embedding("hello")
        assert a == b

    def test_different_inputs_differ(self):
        a = _text_to_embedding("hello")
        b = _text_to_embedding("world")
        assert a != b


class TestToolEmbedding:
    def test_similarity_self(self):
        emb = ToolEmbedding(tool_name="test", full_embedding=[1.0, 0.0])
        assert emb.similarity(emb) == pytest.approx(1.0)

    def test_to_dict_roundtrip(self):
        emb = ToolEmbedding(
            tool_name="test",
            full_embedding=[0.5, 0.5],
            name_embedding=[1.0, 0.0],
        )
        d = emb.to_dict()
        restored = ToolEmbedding.from_dict(d)
        assert restored.tool_name == "test"
        assert restored.full_embedding == [0.5, 0.5]
        assert restored.name_embedding == [1.0, 0.0]


class TestToolEmbeddingGenerator:
    def test_initialize_from_tool(self):
        gen = ToolEmbeddingGenerator(embedding_dim=16)
        tool = Tool(name="search", description="search the web", parameters={"q": {"type": "string"}})
        emb = gen.initialize(tool)
        assert emb.tool_name == "search"
        assert len(emb.full_embedding) == 16

    def test_update_from_usage(self):
        gen = ToolEmbeddingGenerator(embedding_dim=16)
        tool = Tool(name="deploy", description="deploy service")
        emb = gen.initialize(tool)
        cooccurrence = [0.1] * 16
        outcome = [0.2] * 16
        transition = [0.3] * 16
        updated = gen.update_from_usage(emb, cooccurrence, outcome, transition)
        assert len(updated.usage_embedding) == 16
        assert abs(sum(updated.usage_embedding)) > 0


class TestToolAffordance:
    def test_to_dict_roundtrip(self):
        aff = ToolAffordance(
            tool_name="search",
            enabled_transitions=[("unknown", "investigating", 0.9)],
            substitutes=[("query_metrics", 0.8)],
        )
        d = aff.to_dict()
        restored = ToolAffordance.from_dict(d)
        assert restored.tool_name == "search"
        assert restored.enabled_transitions == [("unknown", "investigating", 0.9)]
        assert restored.substitutes == [("query_metrics", 0.8)]


class TestAffordanceLearner:
    def test_empty(self):
        learner = AffordanceLearner()
        tool = Tool(name="search", description="search")
        aff = learner.learn_affordances(tool)
        assert aff.tool_name == "search"

    def test_learn_from_traces(self):
        learner = AffordanceLearner()
        tool_a = Tool(name="search", description="search")
        tool_b = Tool(name="analyze", description="analyze")
        learner.register_tool(tool_a)
        learner.register_tool(tool_b)
        learner._traces = [
            _TraceStub(
                tool_sequence=["search", "analyze"],
                success=True,
                state_before={"status": "unknown"},
                state_after={"status": "investigated"},
                task="find bug",
            ),
            _TraceStub(
                tool_sequence=["search"],
                success=False,
                error="timeout",
                state_before={"status": "unknown"},
                state_after={"status": "error"},
                task="find bug",
            ),
        ]
        aff = learner.learn_affordances(tool_a)
        assert len(aff.enabled_transitions) > 0
        assert len(aff.associated_goals) > 0

    def test_substitute_discovery(self):
        learner = AffordanceLearner()
        tool_a = Tool(name="search", description="search")
        tool_b = Tool(name="query", description="query")
        learner.register_tool(tool_a)
        learner.register_tool(tool_b)

    def test_failure_contexts(self):
        learner = AffordanceLearner()
        tool = Tool(name="deploy", description="deploy")
        learner.register_tool(tool)
        learner._traces = [
            _TraceStub(tool_sequence=["deploy"], success=False, error="timeout"),
            _TraceStub(tool_sequence=["deploy"], success=False, error="timeout"),
            _TraceStub(tool_sequence=["deploy"], success=True),
        ]
        aff = learner.learn_affordances(tool)
        assert len(aff.failure_contexts) >= 1


class TestToolAffordanceGraph:
    def test_add_and_query(self):
        graph = ToolAffordanceGraph()
        aff = ToolAffordance(
            tool_name="search",
            substitutes=[("query", 0.8)],
            complements=[("analyze", 0.9)],
            associated_goals=[("find bug", 0.9)],
        )
        graph.add_tool(aff)
        subs = graph.substitutes("search")
        assert len(subs) == 1
        assert subs[0][0] == "query"

    def test_downstream(self):
        graph = ToolAffordanceGraph()
        aff = ToolAffordance(tool_name="search", complements=[("analyze", 0.9)])
        graph.add_tool(aff)
        downstream = graph.downstream_tools("search")
        assert len(downstream) == 1
        assert downstream[0][0] == "analyze"

    def test_tools_for_goal(self):
        graph = ToolAffordanceGraph()
        aff = ToolAffordance(tool_name="search", associated_goals=[("find bug", 0.9)])
        graph.add_tool(aff)
        results = graph.tools_for_goal("bug")
        assert len(results) == 1
        assert results[0][0] == "search"

    def test_to_dict_roundtrip(self):
        graph = ToolAffordanceGraph()
        aff = ToolAffordance(tool_name="search", substitutes=[("query", 0.8)])
        graph.add_tool(aff)
        d = graph.to_dict()
        restored = ToolAffordanceGraph.from_dict(d)
        assert len(restored._nodes) == 1
        assert restored._nodes["search"].tool_name == "search"


class TestCompositeTool:
    def test_to_tool_conversion(self):
        ct = CompositeTool(name="search_then_analyze", tools=["search", "analyze"], success_rate=0.9)
        tool = ct.to_tool()
        assert tool.name == "search_then_analyze"
        assert "search" in tool.description
        assert "analyze" in tool.description

    def test_to_dict_roundtrip(self):
        ct = CompositeTool(name="test", tools=["a", "b"], success_rate=0.8, avg_cost=2.0)
        d = ct.to_dict()
        restored = CompositeTool.from_dict(d)
        assert restored.name == "test"
        assert restored.tools == ["a", "b"]
        assert restored.success_rate == 0.8
        assert restored.avg_cost == 2.0


class TestToolCompositionSynthesizer:
    def test_discover_compositions(self):
        synth = ToolCompositionSynthesizer()
        synth.tool_costs = {"a": 1.0, "b": 1.0, "c": 1.0}
        traces = []
        for _ in range(10):
            traces.append(_TraceStub(tool_sequence=["a", "b", "c"], success=True))
        composites = synth.discover_compositions(traces, min_occurrences=5, min_success_rate=0.0)
        assert len(composites) >= 1

    def test_synthesize_composite(self):
        synth = ToolCompositionSynthesizer()
        ct = synth.synthesize_composite_tool(["build", "deploy"])
        assert ct.name == "build_deploy"
        assert ct.tools == ["build", "deploy"]

    def test_generate_composite_name(self):
        synth = ToolCompositionSynthesizer()
        assert synth._generate_composite_name(["a", "b"]) == "a_b"
        assert synth._generate_composite_name(["a", "b", "c"]) == "a_thru_c"


class TestScoredTool:
    def test_to_dict_roundtrip(self):
        st = ScoredTool(tool_name="search", score=0.9, similarity=0.8, affordance_score=0.7, exploration_bonus=0.1)
        d = st.to_dict()
        assert d["tool_name"] == "search"
        assert d["score"] == 0.9


class TestLatentToolSelector:
    def test_select_tools(self):
        selector = LatentToolSelector()
        emb = ToolEmbedding(tool_name="search", full_embedding=[1.0, 0.0])
        selector.add_tool_embedding(emb)
        results = selector.select_tools("search for something", top_k=3)
        assert len(results) >= 1

    def test_precondition_gating(self):
        selector = LatentToolSelector()
        emb = ToolEmbedding(tool_name="deploy", full_embedding=[0.5, 0.5])
        selector.add_tool_embedding(emb)
        aff = ToolAffordance(tool_name="deploy", preconditions=[("status=healthy", 0.9)])
        selector.add_affordance(aff)
        results = selector.select_tools("deploy service", top_k=3)
        assert results == []

    def test_exploration_bonus(self):
        selector = LatentToolSelector()
        for name in ["a", "b"]:
            emb = ToolEmbedding(tool_name=name, full_embedding=[0.5, 0.5])
            selector.add_tool_embedding(emb)
        selector.record_selection("a")
        bonus_a = selector._get_exploration_bonus("a")
        bonus_b = selector._get_exploration_bonus("b")
        assert bonus_b > bonus_a


class TestLatentAwarePromptBuilder:
    def test_build_tool_section(self):
        builder = LatentAwarePromptBuilder()
        tool = Tool(name="search", description="search the web for information")
        builder.register_tool(tool)
        emb = ToolEmbedding(tool_name="search", full_embedding=[1.0, 0.0])
        builder.selector.add_tool_embedding(emb)
        section = builder.build_tool_section("search for something")
        assert "search" in section
        assert "Available tools" in section

    def test_no_tools_section(self):
        builder = LatentAwarePromptBuilder()
        section = builder.build_tool_section("anything")
        assert section == "No tools available."


class TestEndToEnd:
    def test_initialize_learn_select_chain(self):
        gen = ToolEmbeddingGenerator(embedding_dim=16)
        tool = Tool(name="search", description="search the web", parameters={"q": {"type": "string"}})

        emb = gen.initialize(tool)

        selector = LatentToolSelector()
        selector.add_tool_embedding(emb)

        results = selector.select_tools("search for something", top_k=3)
        assert len(results) >= 1
        assert results[0].tool_name == "search"
