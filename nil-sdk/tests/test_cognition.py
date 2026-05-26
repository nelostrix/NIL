from __future__ import annotations

import pytest

from nil.cognition.index import StoredItem, ScoredResult, RetrievalIndex
from nil.cognition.query import RetrievalQuery, RetrievalStrategy, RetrievalStrategySelector
from nil.cognition.retrieval import HybridRetriever
from nil.cognition.scoring import RelevanceScorer
from nil.cognition.sources import MultiSourceRetriever
from nil.cognition.assembler import ContextAssembler
from nil.cognition.feedback import RetrievalFeedback
from nil.cognition.runner import RACEngine, CognitiveResult


class TestStoredItem:
    def test_default_id_generated(self):
        item = StoredItem(source="skills", content="test")
        assert item.id != ""
        assert len(item.id) == 12

    def test_manual_id(self):
        item = StoredItem(id="myid", source="skills", content="test")
        assert item.id == "myid"

    def test_to_dict_roundtrip(self):
        item = StoredItem(source="skills", content="hello", keywords=["kw"], score=0.9)
        d = item.to_dict()
        restored = StoredItem.from_dict(d)
        assert restored.source == "skills"
        assert restored.content == "hello"
        assert restored.keywords == ["kw"]
        assert restored.score == 0.9


class TestScoredResult:
    def test_to_dict_roundtrip(self):
        item = StoredItem(source="skills", content="test")
        sr = ScoredResult(item=item, score=0.8, source="skills", rank=1)
        d = sr.to_dict()
        restored = ScoredResult.from_dict(d)
        assert restored.score == 0.8
        assert restored.source == "skills"
        assert restored.rank == 1
        assert restored.item.content == "test"

    def test_was_used(self):
        sr = ScoredResult(score=0.5)
        assert sr.was_used_in_response({}) is True
        sr.score = 0.2
        assert sr.was_used_in_response({}) is False


class TestRetrievalIndex:
    def test_upsert_and_search_sparse(self):
        idx = RetrievalIndex()
        item = StoredItem(source="skills", content="deploy with kubernetes", keywords=["deploy", "k8s"])
        idx.upsert(item)
        results = idx.search_sparse("deploy kubernetes")
        assert len(results) >= 1
        assert results[0].item.id == item.id

    def test_search_dense(self):
        idx = RetrievalIndex()
        item = StoredItem(
            source="skills", content="test",
            embedding=[0.1, 0.2, 0.3],
        )
        idx.upsert(item)
        results = idx.search_dense([0.1, 0.2, 0.3])
        assert len(results) >= 1

    def test_delete(self):
        idx = RetrievalIndex()
        item = StoredItem(source="skills", content="test", keywords=["kw"])
        idx.upsert(item)
        idx.delete(item.id)
        assert idx.size() == 0

    def test_constraint_search(self):
        idx = RetrievalIndex()
        idx.add_constraint({"deploy", "rollback"}, "never deploy on friday")
        results = idx.search_constraints(["deploy", "rollback"])
        assert len(results) == 1
        assert "friday" in results[0].item.content

    def test_cosine_sim(self):
        idx = RetrievalIndex()
        sim = idx._cosine_sim([1, 0], [1, 0])
        assert sim == pytest.approx(1.0)
        sim = idx._cosine_sim([1, 0], [0, 1])
        assert sim == pytest.approx(0.0)


class TestRetrievalQuery:
    def test_composite_text(self):
        q = RetrievalQuery(task="fix bug", last_error="timeout", domain="web")
        text = q.composite_text
        assert "fix bug" in text
        assert "timeout" in text

    def test_composite_text_with_tools(self):
        q = RetrievalQuery(task="deploy", current_tool_state=["kubectl", "docker"])
        text = q.composite_text
        assert "kubectl" in text
        assert "docker" in text


class TestRetrievalStrategySelector:
    def test_error_selects_failure(self):
        selector = RetrievalStrategySelector()
        q = RetrievalQuery(task="test", last_error="crash")
        s = selector.select(q)
        assert s.prioritize_failures is True

    def test_early_exploration(self):
        selector = RetrievalStrategySelector()
        q = RetrievalQuery(task="test", current_step=1)
        s = selector.select(q)
        assert s.prioritize_facts is True

    def test_late_minimal(self):
        selector = RetrievalStrategySelector()
        q = RetrievalQuery(task="test", current_step=15)
        s = selector.select(q)
        assert s.minimal is True

    def test_balanced_default(self):
        selector = RetrievalStrategySelector()
        q = RetrievalQuery(task="test", current_step=5)
        s = selector.select(q)
        assert s.name == "balanced"


class TestHybridRetriever:
    def test_search_returns_results(self):
        retriever = HybridRetriever()
        idx = RetrievalIndex()
        item = StoredItem(source="skills", content="deploy with docker compose", keywords=["deploy", "docker"])
        idx.upsert(item)
        query = RetrievalQuery(task="deploy docker")
        strategy = RetrievalStrategy(name="test", top_k=5)
        results = retriever.search(idx, query, strategy)
        assert len(results) >= 1

    def test_empty_index(self):
        retriever = HybridRetriever()
        idx = RetrievalIndex()
        query = RetrievalQuery(task="something")
        strategy = RetrievalStrategy(top_k=5)
        results = retriever.search(idx, query, strategy)
        assert results == []


class TestRelevanceScorer:
    def test_score_item(self):
        scorer = RelevanceScorer()
        item = StoredItem(source="skills", content="deploy with kubernetes", keywords=["deploy", "k8s"])
        sr = ScoredResult(item=item, score=0.5, source="skills")
        query = RetrievalQuery(task="deploy")
        score = scorer.score_item(sr, query)
        assert score >= 0

    def test_diversity_penalty(self):
        scorer = RelevanceScorer()
        item1 = StoredItem(source="skills", content="deploy", keywords=["deploy"])
        sr1 = ScoredResult(item=item1, score=0.5, source="skills")
        query = RetrievalQuery(task="deploy")
        score_first = scorer.score_item(sr1, query)
        score_with_penalty = scorer.score_item(sr1, query, already_selected=["item:deploy"])
        assert score_with_penalty <= score_first


class TestMultiSourceRetriever:
    def test_index_and_retrieve(self):
        msr = MultiSourceRetriever()
        item = StoredItem(source="skills", content="deploy containers", keywords=["deploy"])
        msr.index_skill(item)
        query = RetrievalQuery(task="deploy")
        strategy = RetrievalStrategy(name="test")
        results = msr.retrieve(query, strategy, top_k=5)
        assert len(results) >= 1
        assert results[0].source == "skills"

    def test_failure_boost_on_error(self):
        msr = MultiSourceRetriever()
        item = StoredItem(source="failures", content="timeout occurred during deploy", keywords=["timeout"])
        msr.index_failure(item)
        query = RetrievalQuery(task="deploy", last_error="timeout")
        strategy = RetrievalStrategy(name="test", min_score=0.0)
        results = msr.retrieve(query, strategy, top_k=5)
        assert len(results) >= 1

    def test_clear(self):
        msr = MultiSourceRetriever()
        msr.index_skill(StoredItem(content="test"))
        msr.clear()
        query = RetrievalQuery(task="test")
        strategy = RetrievalStrategy(name="test", min_score=0.0)
        assert msr.retrieve(query, strategy) == []


class TestContextAssembler:
    def test_assemble_empty(self):
        assembler = ContextAssembler()
        context = assembler.assemble([])
        assert "RELEVANT PAST EXPERIENCE" in context

    def test_assemble_with_results(self):
        assembler = ContextAssembler()
        item = StoredItem(source="skills", content="deploy with kubernetes", keywords=["deploy"])
        sr = ScoredResult(item=item, score=0.9, source="skills")
        context = assembler.assemble([sr])
        assert "deploy with kubernetes" in context
        assert "Skills" in context

    def test_compact_format_when_budget_exceeded(self):
        assembler = ContextAssembler(max_tokens=10)
        item = StoredItem(source="skills", content="x" * 500, keywords=["test"])
        sr = ScoredResult(item=item, score=0.5, source="skills")
        context = assembler.assemble([sr])
        assert len(context) > 0


class TestRetrievalFeedback:
    def test_record_success(self):
        fb = RetrievalFeedback()
        item = StoredItem(source="skills", content="test")
        sr = ScoredResult(item=item, score=0.9, source="skills")
        fb.record_success("test query", [sr], "success")
        assert len(fb.signals) == 1
        assert fb.signals[0].outcome == "success"

    def test_record_failure(self):
        fb = RetrievalFeedback()
        item = StoredItem(source="skills", content="test")
        sr = ScoredResult(item=item, score=0.2, source="skills")
        fb.record_failure("test query", [sr], "error occurred")
        assert len(fb.signals) == 1
        assert "error" in fb.signals[0].context

    def test_max_signals(self):
        fb = RetrievalFeedback(max_signals=3)
        for i in range(5):
            fb.record_feedback(f"q{i}", "failure", 0.1, "ctx")
        assert len(fb.signals) == 3

    def test_positive_negative(self):
        fb = RetrievalFeedback()
        fb.record_feedback("q1", "success", 0.8, "good")
        fb.record_feedback("q2", "failure", 0.2, "bad")
        assert len(fb.positive_signals()) == 1
        assert len(fb.negative_signals()) == 1

    def test_to_dict_roundtrip(self):
        fb = RetrievalFeedback(max_signals=100)
        fb.record_feedback("q1", "success", 0.8, "good")
        d = fb.to_dict()
        restored = RetrievalFeedback.from_dict(d)
        assert len(restored.signals) == 1


class TestRACEngine:
    def test_add_and_search_skills(self):
        engine = RACEngine(top_k=5, min_score=0.0)
        engine.add_skill("deploy containers with kubernetes", ["deploy", "k8s"])
        results = engine.search("deploy containers")
        assert len(results) >= 1

    def test_add_and_search_failures(self):
        engine = RACEngine(top_k=5, min_score=0.0)
        engine.add_failure("timeout during deploy", ["timeout"])
        results = engine.search("deploy timeout")
        assert len(results) >= 1

    def test_add_and_search_trajectories(self):
        engine = RACEngine(top_k=5, min_score=0.0)
        engine.add_trajectory("step1: build, step2: deploy", ["build", "deploy"])
        results = engine.search("build deploy")
        assert len(results) >= 1

    def test_add_and_search_constraints(self):
        engine = RACEngine(top_k=5, min_score=0.0)
        engine.add_constraint("never deploy on friday", ["deploy", "friday"])
        results = engine.search("deploy friday")
        assert len(results) >= 1

    def test_add_and_search_facts(self):
        engine = RACEngine(top_k=5, min_score=0.0)
        engine.add_fact("kubernetes version is 1.28", ["k8s"])
        results = engine.search("kubernetes version")
        assert len(results) >= 1

    def test_run_returns_cognitive_result(self):
        engine = RACEngine(top_k=5, min_score=0.1)
        engine.add_skill("monitor cpu usage with prometheus", ["monitor", "prometheus"])
        result = engine.run("cpu monitoring")
        assert isinstance(result, CognitiveResult)
        assert "RELEVANT PAST EXPERIENCE" in result.context

    def test_run_with_feedback_success(self):
        engine = RACEngine(top_k=5, min_score=0.0)
        engine.add_skill("fix memory leak", ["memory", "leak"])
        result = engine.run_with_feedback("memory leak fix", outcome="success")
        assert result.signal is None

    def test_run_with_feedback_failure(self):
        engine = RACEngine(top_k=5, min_score=0.0)
        engine.add_skill("fix memory leak", ["memory", "leak"])
        result = engine.run_with_feedback("memory leak", outcome="failure", error="not found")
        assert result.signal is None

    def test_search_relevance_filtering(self):
        engine = RACEngine(top_k=10, min_score=0.5)
        engine.add_skill("unrelated content about weather", ["weather"])
        engine.add_skill("deploy with docker", ["deploy"])
        results = engine.search("deploy")
        assert all(r.score >= 0.5 for r in results)

    def test_to_dict_roundtrip(self):
        engine = RACEngine(top_k=5, min_score=0.1)
        engine.add_skill("test skill", ["test"])
        engine.run_with_feedback("test", outcome="success")
        d = engine.to_dict()
        restored = RACEngine.from_dict(d)
        assert restored.top_k == 5
        assert restored.min_score == 0.1
        assert len(restored.feedback.signals) >= 0


class TestCognitiveResult:
    def test_to_dict_roundtrip(self):
        item = StoredItem(source="skills", content="test")
        sr = ScoredResult(item=item, score=0.5, source="skills")
        cr = CognitiveResult(context="hello", results=[sr])
        d = cr.to_dict()
        restored = CognitiveResult.from_dict(d)
        assert restored.context == "hello"
        assert len(restored.results) == 1
        assert restored.results[0].score == 0.5

    def test_with_signal(self):
        cr = CognitiveResult(context="test", results=[], signal={"type": "feedback"})
        d = cr.to_dict()
        restored = CognitiveResult.from_dict(d)
        assert restored.signal == {"type": "feedback"}
