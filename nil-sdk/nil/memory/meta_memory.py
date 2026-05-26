from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.memory.semantic import FactType, SemanticMemory

logger = logging.getLogger("nil.memory.meta")


@dataclass
class KnowledgeGap:
    gap_id: str
    area: str
    evidence: str = ""
    exploration_priority: float = 0.0
    suggested_task: str = ""
    domain: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "gap_id": self.gap_id,
            "area": self.area,
            "evidence": self.evidence,
            "exploration_priority": self.exploration_priority,
            "suggested_task": self.suggested_task,
            "domain": self.domain,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> KnowledgeGap:
        return cls(
            gap_id=d.get("gap_id", ""),
            area=d.get("area", ""),
            evidence=d.get("evidence", ""),
            exploration_priority=d.get("exploration_priority", 0.0),
            suggested_task=d.get("suggested_task", ""),
            domain=d.get("domain", ""),
            created_at=d.get("created_at", 0.0),
        )


@dataclass
class Contradiction:
    contradiction_id: str
    fact_a: dict
    fact_b: dict
    description: str = ""
    resolution: str = ""
    resolved: bool = False
    domain: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "contradiction_id": self.contradiction_id,
            "fact_a": self.fact_a,
            "fact_b": self.fact_b,
            "description": self.description,
            "resolution": self.resolution,
            "resolved": self.resolved,
            "domain": self.domain,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Contradiction:
        return cls(
            contradiction_id=d.get("contradiction_id", ""),
            fact_a=d.get("fact_a", {}),
            fact_b=d.get("fact_b", {}),
            description=d.get("description", ""),
            resolution=d.get("resolution", ""),
            resolved=d.get("resolved", False),
            domain=d.get("domain", ""),
            created_at=d.get("created_at", 0.0),
        )


class MetaMemory:
    def __init__(self, db_path: str, semantic: Optional[SemanticMemory] = None):
        self.db_path = db_path
        self.semantic = semantic
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
            CREATE TABLE IF NOT EXISTS knowledge_gaps (
                gap_id TEXT PRIMARY KEY,
                area TEXT NOT NULL,
                evidence TEXT DEFAULT '',
                exploration_priority REAL DEFAULT 0.0,
                suggested_task TEXT DEFAULT '',
                domain TEXT DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_gaps_domain ON knowledge_gaps(domain);
            CREATE INDEX IF NOT EXISTS idx_gaps_priority ON knowledge_gaps(exploration_priority DESC);

            CREATE TABLE IF NOT EXISTS contradictions (
                contradiction_id TEXT PRIMARY KEY,
                fact_a TEXT NOT NULL,
                fact_b TEXT NOT NULL,
                description TEXT DEFAULT '',
                resolution TEXT DEFAULT '',
                resolved INTEGER DEFAULT 0,
                domain TEXT DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_contradictions_domain ON contradictions(domain);
            CREATE INDEX IF NOT EXISTS idx_contradictions_resolved ON contradictions(resolved);

            CREATE TABLE IF NOT EXISTS meta_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                audit_type TEXT NOT NULL,
                result TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            );
        """)
        conn.commit()

    def identify_gap(
        self, area: str, evidence: str = "", domain: str = ""
    ) -> KnowledgeGap:
        gap = KnowledgeGap(
            gap_id=str(uuid.uuid4()),
            area=area,
            evidence=evidence,
            domain=domain,
        )
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT * FROM knowledge_gaps WHERE area = ? AND domain = ?",
            (area, domain),
        ).fetchone()
        if existing:
            gap.gap_id = existing["gap_id"]
            gap.exploration_priority = existing["exploration_priority"] + 0.1

        conn.execute(
            """INSERT OR REPLACE INTO knowledge_gaps
               (gap_id, area, evidence, exploration_priority, suggested_task, domain, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (gap.gap_id, area, evidence, gap.exploration_priority,
             gap.suggested_task, domain, time.time()),
        )
        conn.commit()
        return gap

    def knowledge_gaps(self, domain: str = "", min_priority: float = 0.0) -> list[KnowledgeGap]:
        conn = self._get_conn()
        params: list[Any] = []
        sql = "SELECT * FROM knowledge_gaps"
        conditions = []
        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if min_priority > 0:
            conditions.append("exploration_priority >= ?")
            params.append(min_priority)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY exploration_priority DESC LIMIT 50"
        rows = conn.execute(sql, params).fetchall()
        return [KnowledgeGap.from_dict(dict(r)) for r in rows]

    def detect_contradictions(self, domain: str) -> list[Contradiction]:
        if not self.semantic:
            return []
        facts = self.semantic.query_facts(domain=domain, limit=5000)
        contradictions = []
        non_conflict_predicates = {"precedes", "followed_by", "related_to",
                                   "similar_to", "has_tag", "mentioned_with"}

        for i, a in enumerate(facts):
            for b in facts[i + 1 :]:
                if a.subject == b.subject and a.predicate == b.predicate:
                    if a.object != b.object and a.confidence > 0.3 and b.confidence > 0.3:
                        if a.predicate in non_conflict_predicates:
                            continue
                        contradictions.append(Contradiction(
                            contradiction_id=str(uuid.uuid4()),
                            fact_a=a.to_dict(),
                            fact_b=b.to_dict(),
                            description=f"'{a.subject} {a.predicate}' conflicts: "
                                        f"'{a.object}' vs '{b.object}'",
                            domain=domain,
                        ))

        conn = self._get_conn()
        for c in contradictions:
            existing = conn.execute(
                "SELECT * FROM contradictions WHERE fact_a = ? AND fact_b = ? AND resolved = 0",
                (json.dumps(c.fact_a), json.dumps(c.fact_b)),
            ).fetchone()
            if not existing:
                conn.execute(
                    """INSERT INTO contradictions
                       (contradiction_id, fact_a, fact_b, description, resolved, domain, created_at)
                       VALUES (?, ?, ?, ?, 0, ?, ?)""",
                    (c.contradiction_id, json.dumps(c.fact_a), json.dumps(c.fact_b),
                     c.description, domain, time.time()),
                )
        conn.commit()
        return contradictions

    def contradictions(self, domain: str = "", unresolved_only: bool = True) -> list[Contradiction]:
        conn = self._get_conn()
        params: list[Any] = []
        sql = "SELECT * FROM contradictions"
        conditions = []
        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if unresolved_only:
            conditions.append("resolved = 0")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC LIMIT 50"
        rows = conn.execute(sql, params).fetchall()
        return [Contradiction.from_dict(dict(r)) for r in rows]

    def resolve_contradiction(self, contradiction_id: str, resolution: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE contradictions SET resolved = 1, resolution = ? WHERE contradiction_id = ?",
            (resolution, contradiction_id),
        )
        conn.commit()

    def confidence_in(self, statement: str, domain: str = "") -> float:
        if not self.semantic:
            return 0.0
        return self.semantic.belief(statement, domain=domain)

    def knowledge_summary(self, domain: str) -> dict[str, Any]:
        conn = self._get_conn()
        gap_count = conn.execute(
            "SELECT COUNT(*) as c FROM knowledge_gaps WHERE domain = ?", (domain,)
        ).fetchone()["c"]
        contra_count = conn.execute(
            "SELECT COUNT(*) as c FROM contradictions WHERE domain = ? AND resolved = 0",
            (domain,),
        ).fetchone()["c"]

        fact_count = 0
        calibration_score = 0.0
        if self.semantic:
            facts = self.semantic.query_facts(domain=domain, limit=10000)
            fact_count = len(facts)
            if facts:
                confs = [f.confidence for f in facts]
                calibration_score = sum(confs) / len(confs)

        return {
            "domain": domain,
            "known_facts_count": fact_count,
            "knowledge_gaps": gap_count,
            "unresolved_contradictions": contra_count,
            "calibration_score": round(calibration_score, 3),
        }

    def audit(self, domain: str) -> dict[str, Any]:
        gaps = self.knowledge_gaps(domain=domain)
        contras = self.detect_contradictions(domain=domain)
        summary = self.knowledge_summary(domain=domain)

        result = {
            "domain": domain,
            "gaps_found": len(gaps),
            "contradictions_found": len(contras),
            "summary": summary,
            "high_priority_gaps": [g.to_dict() for g in gaps if g.exploration_priority > 0.5],
        }

        conn = self._get_conn()
        conn.execute(
            "INSERT INTO meta_audit_log (domain, audit_type, result, created_at) VALUES (?, 'full_audit', ?, ?)",
            (domain, json.dumps(result), time.time()),
        )
        conn.commit()
        return result

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
