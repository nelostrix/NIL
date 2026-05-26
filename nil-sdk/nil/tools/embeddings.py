from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.tools.base import Tool


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _text_to_embedding(text: str, dim: int = 128) -> list[float]:
    h = hashlib.sha256(text.encode())
    digest = h.hexdigest()
    return [int(digest[i:i + 2], 16) / 255.0 for i in range(0, min(dim * 2, len(digest)), 2)]


@dataclass
class ToolEmbedding:
    tool_name: str = ""
    name_embedding: list[float] = field(default_factory=list)
    description_embedding: list[float] = field(default_factory=list)
    parameter_embedding: list[float] = field(default_factory=list)
    usage_embedding: list[float] = field(default_factory=list)
    outcome_embedding: list[float] = field(default_factory=list)
    transition_embedding: list[float] = field(default_factory=list)
    full_embedding: list[float] = field(default_factory=list)

    def similarity(self, other: ToolEmbedding) -> float:
        return _cosine_sim(self.full_embedding, other.full_embedding)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "name_embedding": self.name_embedding,
            "description_embedding": self.description_embedding,
            "parameter_embedding": self.parameter_embedding,
            "usage_embedding": self.usage_embedding,
            "outcome_embedding": self.outcome_embedding,
            "transition_embedding": self.transition_embedding,
            "full_embedding": self.full_embedding,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolEmbedding:
        return cls(
            tool_name=d.get("tool_name", ""),
            name_embedding=d.get("name_embedding", []),
            description_embedding=d.get("description_embedding", []),
            parameter_embedding=d.get("parameter_embedding", []),
            usage_embedding=d.get("usage_embedding", []),
            outcome_embedding=d.get("outcome_embedding", []),
            transition_embedding=d.get("transition_embedding", []),
            full_embedding=d.get("full_embedding", []),
        )


class ToolEmbeddingGenerator:
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

    def initialize(self, tool: Tool) -> ToolEmbedding:
        name_emb = _text_to_embedding(tool.name, self.embedding_dim)
        desc_emb = _text_to_embedding(tool.description, self.embedding_dim)
        param_text = " ".join(
            f"{k}:{v.get('type', 'string')}" for k, v in tool.parameters.items()
        )
        param_emb = _text_to_embedding(param_text, self.embedding_dim)

        full = self._combine([name_emb, desc_emb, param_emb], weights=[0.3, 0.5, 0.2])

        return ToolEmbedding(
            tool_name=tool.name,
            name_embedding=name_emb,
            description_embedding=desc_emb,
            parameter_embedding=param_emb,
            full_embedding=full,
        )

    def update_from_usage(
        self,
        embedding: ToolEmbedding,
        cooccurrence_vector: Optional[list[float]] = None,
        outcome_vector: Optional[list[float]] = None,
        transition_vector: Optional[list[float]] = None,
    ) -> ToolEmbedding:
        if cooccurrence_vector is not None:
            embedding.usage_embedding = self._fuse(
                cooccurrence_vector,
                outcome_vector or [0.0] * self.embedding_dim,
                transition_vector or [0.0] * self.embedding_dim,
            )

        if outcome_vector is not None:
            embedding.outcome_embedding = outcome_vector

        if transition_vector is not None:
            embedding.transition_embedding = transition_vector

        embedding.full_embedding = self._combine(
            [
                embedding.name_embedding,
                embedding.description_embedding,
                embedding.parameter_embedding,
                embedding.usage_embedding if embedding.usage_embedding else [0.0] * self.embedding_dim,
            ],
            weights=[0.15, 0.25, 0.10, 0.50],
        )

        return embedding

    def _combine(self, vectors: list[list[float]], weights: list[float]) -> list[float]:
        dim = len(vectors[0]) if vectors else self.embedding_dim
        result = [0.0] * dim
        for vec, weight in zip(vectors, weights):
            for i in range(min(len(vec), dim)):
                result[i] += vec[i] * weight
        return result

    def _fuse(self, *vectors: list[float]) -> list[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        result = [0.0] * dim
        for vec in vectors:
            for i in range(min(len(vec), dim)):
                result[i] += vec[i]
        norm = math.sqrt(sum(x * x for x in result)) or 1.0
        return [x / norm for x in result]
