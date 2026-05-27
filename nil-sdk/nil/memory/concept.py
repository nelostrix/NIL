from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ConceptType(str, Enum):
    CAUSAL = "causal"
    ORDERING = "ordering"
    CONDITIONAL = "conditional"
    PATTERN = "pattern"
    POLICY = "policy"


class AbstractionLevel(str, Enum):
    RAW = "raw"
    CLUSTERED = "clustered"
    COMPRESSED = "compressed"
    SYMBOLIC = "symbolic"
    POLICY = "policy"


@dataclass
class Concept:
    concept_id: str = ""
    domain: str = ""
    concept_type: ConceptType = ConceptType.PATTERN
    abstraction_level: AbstractionLevel = AbstractionLevel.SYMBOLIC
    description: str = ""
    pattern_data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    support_count: int = 0
    source_traces: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id or str(uuid.uuid4()),
            "domain": self.domain,
            "concept_type": self.concept_type.value,
            "abstraction_level": self.abstraction_level.value,
            "description": self.description,
            "pattern_data": self.pattern_data,
            "confidence": round(self.confidence, 4),
            "support_count": self.support_count,
            "source_traces": self.source_traces,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Concept:
        return cls(
            concept_id=d.get("concept_id", str(uuid.uuid4())),
            domain=d.get("domain", ""),
            concept_type=ConceptType(d.get("concept_type", "pattern")),
            abstraction_level=AbstractionLevel(d.get("abstraction_level", "symbolic")),
            description=d.get("description", ""),
            pattern_data=d.get("pattern_data", {}),
            confidence=d.get("confidence", 0.0),
            support_count=d.get("support_count", 0),
            source_traces=d.get("source_traces", []),
            created_at=d.get("created_at", time.time()),
            last_updated=d.get("last_updated", time.time()),
            tags=d.get("tags", []),
        )

    def short_summary(self) -> str:
        return f"[{self.concept_type.value}] {self.description} (conf={self.confidence:.2f}, n={self.support_count})"


@dataclass
class ConceptGraph:
    concepts: dict[str, Concept] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)

    def add_concept(self, concept: Concept) -> None:
        cid = concept.concept_id or str(uuid.uuid4())
        concept.concept_id = cid
        self.concepts[cid] = concept

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related",
        weight: float = 0.5,
    ) -> None:
        self.edges.append({
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "weight": weight,
        })

    def concepts_by_type(self, concept_type: ConceptType) -> list[Concept]:
        return [c for c in self.concepts.values() if c.concept_type == concept_type]

    def concepts_by_level(self, level: AbstractionLevel) -> list[Concept]:
        return [c for c in self.concepts.values() if c.abstraction_level == level]

    def high_confidence(self, min_conf: float = 0.7) -> list[Concept]:
        return [c for c in self.concepts.values() if c.confidence >= min_conf]
