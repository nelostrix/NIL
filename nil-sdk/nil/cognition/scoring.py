from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.cognition.index import ScoredResult
from nil.cognition.query import RetrievalQuery


class RelevanceScorer:
    def score_item(
        self,
        item: ScoredResult,
        query: RetrievalQuery,
        already_selected: list[str] = None,
    ) -> float:
        already_selected = already_selected or []
        semantic = item.score * 0.35
        recency = self._recency_score(item) * 0.15
        specificity = self._specificity_score(item, query) * 0.20
        confidence = self._confidence_score(item) * 0.15
        diversity = self._diversity_penalty(item, already_selected) * 0.15

        return round(semantic + recency + specificity + confidence + diversity, 4)

    def _recency_score(self, item: ScoredResult) -> float:
        age = time.time() - item.item.created_at
        return max(0.0, 1.0 - age / (86400 * 30))

    def _specificity_score(self, item: ScoredResult, query: RetrievalQuery) -> float:
        query_words = set(query.composite_text.lower().split())
        content_words = set(item.item.content.lower().split()) if item.item.content else set()
        kw_words = set(item.item.keywords)
        all_words = content_words | kw_words
        if not all_words or not query_words:
            return 0.0
        overlap = len(query_words & all_words)
        return overlap / max(len(query_words), 1)

    def _confidence_score(self, item: ScoredResult) -> float:
        return min(1.0, item.item.score)

    def _diversity_penalty(self, item: ScoredResult, already_selected: list[str]) -> float:
        if not already_selected:
            return 1.0
        item_kw = set(item.item.keywords)
        if not item_kw:
            return 0.8
        max_overlap = 0
        for selected_id in already_selected:
            selected_kw = set()
            parts = selected_id.split(":")
            if len(parts) > 1:
                selected_kw = set(parts[1].split(","))
            overlap = len(item_kw & selected_kw)
            max_overlap = max(max_overlap, overlap)
        return max(0.1, 1.0 - max_overlap * 0.3)
