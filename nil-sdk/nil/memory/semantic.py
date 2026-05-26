from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("nil.memory.semantic")


class FactType(str, Enum):
    RULE = "rule"
    RELATIONSHIP = "relationship"
    CONCEPT = "concept"
    CONSTRAINT = "constraint"
    HEURISTIC = "heuristic"


@dataclass
class Fact:
    fact_id: str = ""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    confidence: float = 0.0
    evidence_count: int = 0
    first_observed_at: float = field(default_factory=time.time)
    last_confirmed_at: float = field(default_factory=time.time)
    source_episodes: list[str] = field(default_factory=list)
    fact_type: FactType = FactType.RELATIONSHIP
    domain: str = ""

    def statement(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "first_observed_at": self.first_observed_at,
            "last_confirmed_at": self.last_confirmed_at,
            "source_episodes": self.source_episodes,
            "fact_type": self.fact_type.value,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Fact:
        return cls(
            fact_id=d.get("fact_id", str(uuid.uuid4())),
            subject=d["subject"],
            predicate=d["predicate"],
            object=d["object"],
            confidence=d.get("confidence", 0.0),
            evidence_count=d.get("evidence_count", 0),
            first_observed_at=d.get("first_observed_at", 0.0),
            last_confirmed_at=d.get("last_confirmed_at", 0.0),
            source_episodes=d.get("source_episodes", []),
            fact_type=FactType(d.get("fact_type", "relationship")),
            domain=d.get("domain", ""),
        )


@dataclass
class EntityGraph:
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    relationships: list[dict[str, Any]] = field(default_factory=list)

    def add_entity(self, name: str, entity_type: str = "", properties: dict = None):
        if name not in self.entities:
            self.entities[name] = {
                "name": name,
                "type": entity_type,
                "properties": properties or {},
                "first_seen": time.time(),
                "last_seen": time.time(),
            }
        self.entities[name]["last_seen"] = time.time()
        if properties:
            self.entities[name]["properties"].update(properties)


class SemanticMemory:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None

    def _get_conn(self):
        import sqlite3
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def init_tables(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                evidence_count INTEGER DEFAULT 0,
                first_observed_at REAL NOT NULL,
                last_confirmed_at REAL NOT NULL,
                source_episodes TEXT DEFAULT '[]',
                fact_type TEXT DEFAULT 'relationship',
                domain TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_facts_domain ON facts(domain);
            CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
            CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate);
            CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object);
            CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence DESC);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_spo ON facts(subject, predicate, object, domain);

            CREATE TABLE IF NOT EXISTS entities (
                name TEXT NOT NULL,
                entity_type TEXT DEFAULT '',
                properties TEXT DEFAULT '{}',
                domain TEXT DEFAULT '',
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                PRIMARY KEY (name, domain)
            );
        """)
        conn.commit()

    def observe_fact(self, fact: Fact, source_episode_id: str = "") -> None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM facts WHERE subject = ? AND predicate = ? AND object = ? AND domain = ?",
            (fact.subject, fact.predicate, fact.object, fact.domain),
        ).fetchone()

        now = time.time()
        if row:
            existing = json.loads(row["source_episodes"])
            if source_episode_id and source_episode_id not in existing:
                existing.append(source_episode_id)
            conn.execute(
                """UPDATE facts SET confidence = ?, evidence_count = evidence_count + 1,
                   last_confirmed_at = ?, source_episodes = ?
                   WHERE fact_id = ?""",
                (fact.confidence, now, json.dumps(existing), row["fact_id"]),
            )
        else:
            if source_episode_id:
                fact.source_episodes = [source_episode_id]
            fact.fact_id = fact.fact_id or str(uuid.uuid4())
            last_confirmed = fact.last_confirmed_at or now
            first_observed = fact.first_observed_at or now
            conn.execute(
                """INSERT INTO facts
                   (fact_id, subject, predicate, object, confidence, evidence_count,
                    first_observed_at, last_confirmed_at, source_episodes, fact_type, domain)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fact.fact_id,
                    fact.subject,
                    fact.predicate,
                    fact.object,
                    fact.confidence,
                    1,
                    first_observed,
                    last_confirmed,
                    json.dumps(fact.source_episodes),
                    fact.fact_type.value,
                    fact.domain,
                ),
            )
        conn.commit()

    def query_facts(
        self,
        domain: str = "",
        subject: str = "",
        predicate: str = "",
        object: str = "",
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[Fact]:
        conn = self._get_conn()
        conditions = []
        params: list[Any] = []

        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if subject:
            conditions.append("subject = ?")
            params.append(subject)
        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)
        if object:
            conditions.append("object = ?")
            params.append(object)
        if min_confidence > 0:
            conditions.append("confidence >= ?")
            params.append(min_confidence)

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = conn.execute(
            f"SELECT * FROM facts WHERE {where} ORDER BY confidence DESC, last_confirmed_at DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [Fact.from_dict(dict(r)) for r in rows]

    def belief(self, statement: str, domain: str = "") -> float:
        parts = statement.split(" ", 2)
        if len(parts) < 3:
            return 0.0
        subject, predicate, object_ = parts[0], parts[1], " ".join(parts[2:])
        facts = self.query_facts(
            domain=domain, subject=subject, predicate=predicate, object=object_
        )
        if not facts:
            return 0.0
        return max(f.confidence for f in facts)

    def relationships(self, entity: str, domain: str = "") -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM facts
               WHERE (subject = ? OR object = ?) AND domain = ?
               ORDER BY confidence DESC LIMIT 50""",
            (entity, entity, domain),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_knowledge_graph(self, domain: str) -> EntityGraph:
        g = EntityGraph()
        facts = self.query_facts(domain=domain, limit=5000)
        for f in facts:
            g.add_entity(f.subject, entity_type="concept")
            g.add_entity(f.object, entity_type="concept")
            g.relationships.append({
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "confidence": f.confidence,
                "type": f.fact_type.value,
            })
        return g

    def decay_confidence(self, domain: str, decay_rate: float = 0.05) -> int:
        conn = self._get_conn()
        now = time.time()
        rows = conn.execute(
            "SELECT * FROM facts WHERE domain = ?", (domain,)
        ).fetchall()
        updated = 0
        for row in rows:
            age_days = (now - row["last_confirmed_at"]) / 86400
            if age_days > 1:
                decay = min(decay_rate * age_days, 0.5)
                new_conf = max(row["confidence"] - decay, 0.0)
                conn.execute(
                    "UPDATE facts SET confidence = ? WHERE fact_id = ?",
                    (new_conf, row["fact_id"]),
                )
                updated += 1
        if updated:
            conn.commit()
        return updated

    def track_entity(
        self, name: str, entity_type: str = "", properties: dict = None, domain: str = ""
    ) -> None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM entities WHERE name = ? AND domain = ?", (name, domain)
        ).fetchone()
        if row:
            existing_props = json.loads(row["properties"])
            if properties:
                existing_props.update(properties)
            conn.execute(
                "UPDATE entities SET last_seen = ?, properties = ? WHERE name = ? AND domain = ?",
                (time.time(), json.dumps(existing_props), name, domain),
            )
        else:
            conn.execute(
                "INSERT INTO entities (name, entity_type, properties, domain, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
                (name, entity_type, json.dumps(properties or {}), domain, time.time(), time.time()),
            )
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
