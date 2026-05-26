from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from nil.cognition.index import RetrievalIndex, ScoredResult
from nil.cognition.query import RetrievalQuery, RetrievalStrategySelector
from nil.cognition.retrieval import HybridRetriever
from nil.cognition.sources import MultiSourceRetriever
from nil.cognition.scoring import RelevanceScorer
from nil.cognition.assembler import ContextAssembler
from nil.cognition.feedback import RetrievalFeedback


@dataclass
class CognitiveResult:
    context: str
    results: list[ScoredResult]
    signal: Optional[dict] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "results": [r.to_dict() for r in self.results],
            "signal": self.signal,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CognitiveResult:
        return cls(
            context=d["context"],
            results=[ScoredResult.from_dict(r) for r in d["results"]],
            signal=d.get("signal"),
        )


class RACEngine:
    def __init__(
        self,
        top_k: int = 10,
        min_score: float = 0.1,
    ):
        self.index = RetrievalIndex()
        self.sources = MultiSourceRetriever()
        self.retriever = HybridRetriever()
        self.scorer = RelevanceScorer()
        self.assembler = ContextAssembler()
        self.feedback = RetrievalFeedback()
        self.strategy_selector = RetrievalStrategySelector()
        self.top_k = top_k
        self.min_score = min_score

    def search(self, query_text: str, top_k: Optional[int] = None) -> list[ScoredResult]:
        query = RetrievalQuery(task=query_text)
        strategy = self.strategy_selector.select(query)

        indexed_results = self.retriever.search(self.index, query, strategy)
        source_results = self.sources.retrieve(query, strategy, top_k=top_k or self.top_k)

        combined = indexed_results + source_results
        seen: set[str] = set()
        deduped = []
        for r in combined:
            if r.item.id not in seen:
                seen.add(r.item.id)
                deduped.append(r)

        deduped.sort(key=lambda x: x.score, reverse=True)
        k = top_k or self.top_k
        return [r for r in deduped if r.score >= self.min_score][:k]

    def run(self, query_text: str) -> CognitiveResult:
        results = self.search(query_text)
        context = self.assembler.assemble(results)
        return CognitiveResult(context=context, results=results)

    def run_with_feedback(
        self,
        query_text: str,
        outcome: str,
        error: Optional[str] = None,
    ) -> CognitiveResult:
        result = self.run(query_text)

        if outcome == "success":
            self.feedback.record_success(query_text, result.results, outcome)
        elif error:
            self.feedback.record_failure(query_text, result.results, error)

        return result

    def add_skill(self, content: str, keywords: Optional[list[str]] = None) -> None:
        from nil.cognition.index import StoredItem
        item = StoredItem(
            source="skills",
            content=content,
            keywords=keywords or [],
        )
        self.sources.index_skill(item)
        self.index.upsert(item)

    def add_failure(self, content: str, keywords: Optional[list[str]] = None) -> None:
        from nil.cognition.index import StoredItem
        item = StoredItem(
            source="failures",
            content=content,
            keywords=keywords or [],
        )
        self.sources.index_failure(item)
        self.index.upsert(item)

    def add_trajectory(self, content: str, keywords: Optional[list[str]] = None) -> None:
        from nil.cognition.index import StoredItem
        item = StoredItem(
            source="trajectories",
            content=content,
            keywords=keywords or [],
        )
        self.sources.index_trajectory(item)
        self.index.upsert(item)

    def add_constraint(self, content: str, keywords: Optional[list[str]] = None) -> None:
        from nil.cognition.index import StoredItem
        item = StoredItem(
            source="constraints",
            content=content,
            keywords=keywords or [],
        )
        self.sources.index_constraint(item)
        self.index.upsert(item)

    def add_fact(self, content: str, keywords: Optional[list[str]] = None) -> None:
        from nil.cognition.index import StoredItem
        item = StoredItem(
            source="facts",
            content=content,
            keywords=keywords or [],
        )
        self.sources.index_fact(item)
        self.index.upsert(item)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback": self.feedback.to_dict(),
            "top_k": self.top_k,
            "min_score": self.min_score,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RACEngine:
        engine = cls(
            top_k=d.get("top_k", 10),
            min_score=d.get("min_score", 0.1),
        )
        engine.feedback = RetrievalFeedback.from_dict(d.get("feedback", {"signals": []}))
        return engine
