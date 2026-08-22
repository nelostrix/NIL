from __future__ import annotations

import logging

from nil.cognition.index import ScoredResult, StoredItem
from nil.cognition.query import RetrievalQuery, RetrievalStrategy

logger = logging.getLogger("nil.cognition.sources")


class MultiSourceRetriever:
    def __init__(self):
        self.skill_store: dict[str, StoredItem] = {}
        self.failure_store: dict[str, StoredItem] = {}
        self.trajectory_store: dict[str, StoredItem] = {}
        self.constraint_store: dict[str, StoredItem] = {}
        self.fact_store: dict[str, StoredItem] = {}

    def index_skill(self, item: StoredItem) -> None:
        item.source = "skills"
        self.skill_store[item.id] = item

    def index_failure(self, item: StoredItem) -> None:
        item.source = "failures"
        self.failure_store[item.id] = item

    def index_trajectory(self, item: StoredItem) -> None:
        item.source = "trajectories"
        self.trajectory_store[item.id] = item

    def index_constraint(self, item: StoredItem) -> None:
        item.source = "constraints"
        self.constraint_store[item.id] = item

    def index_fact(self, item: StoredItem) -> None:
        item.source = "facts"
        self.fact_store[item.id] = item

    def retrieve(
        self,
        query: RetrievalQuery,
        strategy: RetrievalStrategy,
        top_k: int = 5,
    ) -> list[ScoredResult]:
        results: list[ScoredResult] = []
        sources = {
            "skills": self.skill_store,
            "failures": self.failure_store,
            "trajectories": self.trajectory_store,
            "constraints": self.constraint_store,
            "facts": self.fact_store,
        }

        query_words = set(query.composite_text.lower().split())

        for source_name, store in sources.items():
            weight = strategy.source_weights.get(source_name, 0.0)
            source_top_k = max(1, int(top_k * weight * 3))

            scored = []
            for item_id, item in store.items():
                score = self._match_item(item, query_words, query)
                if score > strategy.min_score:
                    scored.append(ScoredResult(
                        item=item, score=round(score, 4), source=source_name,
                    ))

            scored.sort(key=lambda x: x.score, reverse=True)
            results.extend(scored[:source_top_k])

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _match_item(self, item: StoredItem, query_words: set[str], query: RetrievalQuery) -> float:
        content_words = set(item.content.lower().split()) if item.content else set()
        kw_words = set(item.keywords)
        all_words = content_words | kw_words

        if not all_words or not query_words:
            return 0.0

        overlap = len(query_words & all_words)
        max_len = max(len(query_words), 1)
        base_score = overlap / max_len

        if query.last_error and item.source == "failures":
            if query.last_error.lower() in item.content.lower():
                base_score += 0.3

        if item.source == "skills" and query.current_tool_state:
            for tool in query.current_tool_state:
                if tool in kw_words:
                    base_score += 0.2

        return min(1.0, base_score)

    def clear(self) -> None:
        self.skill_store.clear()
        self.failure_store.clear()
        self.trajectory_store.clear()
        self.constraint_store.clear()
        self.fact_store.clear()
