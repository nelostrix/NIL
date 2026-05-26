from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StoredItem:
    id: str = ""
    source: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    score: float = 1.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "keywords": self.keywords,
            "score": self.score,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StoredItem:
        return cls(
            id=d.get("id", ""),
            source=d.get("source", ""),
            content=d.get("content", ""),
            metadata=d.get("metadata", {}),
            embedding=d.get("embedding", []),
            keywords=d.get("keywords", []),
            score=d.get("score", 1.0),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
        )


@dataclass
class ScoredResult:
    item: StoredItem = field(default_factory=StoredItem)
    score: float = 0.0
    source: str = ""
    rank: int = 0

    def was_used_in_response(self, outcome: dict) -> bool:
        return self.score > 0.3

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "score": self.score,
            "source": self.source,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScoredResult:
        return cls(
            item=StoredItem.from_dict(d.get("item", {})),
            score=d.get("score", 0.0),
            source=d.get("source", ""),
            rank=d.get("rank", 0),
        )


class RetrievalIndex:
    def __init__(self):
        self.items: dict[str, StoredItem] = {}
        self.dense_items: list[tuple[str, list[float]]] = []
        self.sparse_index: dict[str, list[str]] = {}
        self.constraint_rules: list[tuple[set[str], str]] = []

    def upsert(self, item: StoredItem) -> None:
        item.updated_at = time.time()
        self.items[item.id] = item
        if item.embedding:
            self.dense_items.append((item.id, item.embedding))
        for kw in item.keywords:
            kw_lower = kw.lower()
            if kw_lower not in self.sparse_index:
                self.sparse_index[kw_lower] = []
            if item.id not in self.sparse_index[kw_lower]:
                self.sparse_index[kw_lower].append(item.id)

    def delete(self, item_id: str) -> None:
        self.items.pop(item_id, None)
        self.dense_items = [(iid, emb) for iid, emb in self.dense_items if iid != item_id]
        for kw in list(self.sparse_index):
            self.sparse_index[kw] = [iid for iid in self.sparse_index[kw] if iid != item_id]
            if not self.sparse_index[kw]:
                del self.sparse_index[kw]

    def search_dense(self, query_embedding: list[float], top_k: int = 10) -> list[ScoredResult]:
        if not self.dense_items or not query_embedding:
            return []
        scored = []
        for item_id, emb in self.dense_items:
            sim = self._cosine_sim(query_embedding, emb)
            scored.append((item_id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            ScoredResult(item=self.items[sid], score=s, source=self.items[sid].source)
            for sid, s in scored[:top_k] if sid in self.items
        ]

    def search_sparse(self, query_text: str, top_k: int = 10) -> list[ScoredResult]:
        query_words = set(query_text.lower().split())
        if not query_words:
            return []
        scores: dict[str, float] = {}
        for word in query_words:
            for item_id in self.sparse_index.get(word, []):
                scores[item_id] = scores.get(item_id, 0.0) + 1.0
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            ScoredResult(item=self.items[sid], score=s / max(len(query_words), 1), source=self.items[sid].source)
            for sid, s in ranked[:top_k] if sid in self.items
        ]

    def search_constraints(self, tools: list[str], top_k: int = 10) -> list[ScoredResult]:
        tool_set = set(tools)
        results = []
        for trigger_tools, constraint_text in self.constraint_rules:
            if trigger_tools and trigger_tools.issubset(tool_set):
                item = StoredItem(
                    id=f"rule-{uuid.uuid4().hex[:8]}",
                    source="constraints",
                    content=constraint_text,
                    keywords=list(trigger_tools),
                )
                results.append(ScoredResult(item=item, score=1.0, source="constraints"))
        return results[:top_k]

    def add_constraint(self, trigger_tools: set[str], constraint_text: str) -> None:
        self.constraint_rules.append((trigger_tools, constraint_text))

    def refresh(self) -> None:
        pass

    def size(self) -> int:
        return len(self.items)

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
