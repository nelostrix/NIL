from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from nil.cognition.index import RetrievalIndex, ScoredResult, StoredItem
from nil.cognition.query import RetrievalQuery, RetrievalStrategy


class HybridRetriever:
    def __init__(self):
        self.retrieval_weights = {"dense": 0.5, "sparse": 0.3, "structure": 0.2}

    def search(
        self,
        index: RetrievalIndex,
        query: RetrievalQuery,
        strategy: RetrievalStrategy,
    ) -> list[ScoredResult]:
        query_embedding = self._encode_dense(query)
        dense_results = index.search_dense(query_embedding, top_k=strategy.top_k * 2)
        sparse_results = index.search_sparse(query.composite_text, top_k=strategy.top_k * 2)
        structure_results = index.search_constraints(query.current_tool_state, top_k=strategy.top_k)

        fused = self._fuse(
            dense_results,
            sparse_results,
            structure_results,
            weights=self.retrieval_weights,
        )

        reranked = self._rerank(query, fused)
        return reranked[:strategy.top_k]

    def _encode_dense(self, query: RetrievalQuery) -> list[float]:
        text = query.composite_text
        import hashlib
        h = hashlib.sha256(text.encode())
        return [int(h.hexdigest()[i:i + 2], 16) / 255.0 for i in range(0, 32, 2)]

    def _fuse(
        self,
        dense: list[ScoredResult],
        sparse: list[ScoredResult],
        structure: list[ScoredResult],
        weights: dict[str, float],
    ) -> list[ScoredResult]:
        score_map: dict[str, dict] = {}

        for r in dense:
            score_map[r.item.id] = {"item": r.item, "dense": r.score, "sparse": 0.0, "structure": 0.0}
        for r in sparse:
            if r.item.id not in score_map:
                score_map[r.item.id] = {"item": r.item, "dense": 0.0, "sparse": 0.0, "structure": 0.0}
            score_map[r.item.id]["sparse"] = r.score
        for r in structure:
            if r.item.id not in score_map:
                score_map[r.item.id] = {"item": r.item, "dense": 0.0, "sparse": 0.0, "structure": 0.0}
            score_map[r.item.id]["structure"] = r.score

        results = []
        for item_id, scores in score_map.items():
            combined = (
                scores["dense"] * weights.get("dense", 0.5)
                + scores["sparse"] * weights.get("sparse", 0.3)
                + scores["structure"] * weights.get("structure", 0.2)
            )
            results.append(ScoredResult(
                item=scores["item"],
                score=round(combined, 4),
                source=scores["item"].source,
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _rerank(self, query: RetrievalQuery, results: list[ScoredResult]) -> list[ScoredResult]:
        for i, r in enumerate(results):
            r.rank = i
            query_words = set(query.composite_text.lower().split())
            content_words = set(r.item.content.lower().split()) if r.item.content else set()
            overlap = len(query_words & content_words)
            if overlap > 0 and len(query_words) > 0:
                r.score = r.score * 0.7 + (overlap / max(len(query_words), 1)) * 0.3
                r.score = round(r.score, 4)
        results.sort(key=lambda x: x.score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i
        return results
