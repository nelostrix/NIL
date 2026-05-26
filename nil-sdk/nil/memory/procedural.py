from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.memory.skill_registry import Skill, SkillStatus, SkillOrigin, TriggerCondition

logger = logging.getLogger("nil.memory.procedural")


@dataclass
class Procedure:
    procedure_id: str
    domain: str
    name: str
    steps: list[dict] = field(default_factory=list)
    trigger_conditions: TriggerCondition = field(default_factory=TriggerCondition)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    success_rate: float = 0.0
    avg_cost: float = 0.0
    avg_duration: float = 0.0
    usage_count: int = 0
    version: int = 1
    parent_procedure: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    skill_id: str = ""

    def to_dict(self) -> dict:
        return {
            "procedure_id": self.procedure_id,
            "domain": self.domain,
            "name": self.name,
            "steps": self.steps,
            "trigger_conditions": self.trigger_conditions.to_dict(),
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "success_rate": self.success_rate,
            "avg_cost": self.avg_cost,
            "avg_duration": self.avg_duration,
            "usage_count": self.usage_count,
            "version": self.version,
            "parent_procedure": self.parent_procedure,
            "created_at": self.created_at,
            "skill_id": self.skill_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Procedure:
        return cls(
            procedure_id=d.get("procedure_id", ""),
            domain=d.get("domain", ""),
            name=d.get("name", ""),
            steps=d.get("steps", []),
            trigger_conditions=TriggerCondition.from_dict(d.get("trigger_conditions", {})),
            preconditions=d.get("preconditions", []),
            postconditions=d.get("postconditions", []),
            success_rate=d.get("success_rate", 0.0),
            avg_cost=d.get("avg_cost", 0.0),
            avg_duration=d.get("avg_duration", 0.0),
            usage_count=d.get("usage_count", 0),
            version=d.get("version", 1),
            parent_procedure=d.get("parent_procedure"),
            created_at=d.get("created_at", 0.0),
            skill_id=d.get("skill_id", ""),
        )


@dataclass
class RefinementEvent:
    procedure_id: str
    previous_version: int
    new_version: int
    success_rate_delta: float
    cost_delta: float
    failure_reason: str = ""
    applied_at: float = field(default_factory=time.time)


class ProceduralMemory:
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
            CREATE TABLE IF NOT EXISTS procedures (
                procedure_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                name TEXT NOT NULL,
                procedure_data TEXT NOT NULL,
                skill_id TEXT,
                version INTEGER DEFAULT 1,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_procedures_domain ON procedures(domain);
            CREATE INDEX IF NOT EXISTS idx_procedures_skill ON procedures(skill_id);
            CREATE INDEX IF NOT EXISTS idx_procedures_success ON procedures(domain, success_rate DESC);

            CREATE TABLE IF NOT EXISTS refinement_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                procedure_id TEXT NOT NULL,
                previous_version INTEGER NOT NULL,
                new_version INTEGER NOT NULL,
                success_rate_delta REAL DEFAULT 0.0,
                cost_delta REAL DEFAULT 0.0,
                failure_reason TEXT DEFAULT '',
                applied_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS skill_chains (
                chain_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                name TEXT NOT NULL,
                skill_ids TEXT NOT NULL,
                trigger_description TEXT DEFAULT '',
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_skill_chains_domain ON skill_chains(domain);
        """)
        conn.commit()

    def store_skill(self, skill: Skill) -> None:
        from nil.memory.store import PersistentMemory
        conn = self._get_conn()
        source_runs = json.dumps(skill.source_runs) if skill.source_runs else "[]"
        row = conn.execute(
            "SELECT * FROM skills WHERE domain = ? AND name = ?",
            (skill.name.split(":")[0] if ":" in skill.name else "", skill.name),
        ).fetchone()
        if row:
            conn.execute(
                """UPDATE skills SET description = ?, pattern = ?, source_runs = ?,
                   confidence = ?, usage_count = usage_count + ?
                   WHERE id = ?""",
                (skill.description, json.dumps(skill.to_dict()),
                 source_runs, skill.confidence, 0, row["id"]),
            )
        else:
            domain = skill.name.split(":")[0] if ":" in skill.name else ""
            conn.execute(
                """INSERT INTO skills (domain, name, description, pattern, source_runs, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (domain, skill.name, skill.description,
                 json.dumps(skill.to_dict()), source_runs, skill.confidence, time.time()),
            )
        conn.commit()

    def store_procedure(self, procedure: Procedure) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO procedures
               (procedure_id, domain, name, procedure_data, skill_id, version, usage_count, success_rate, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                procedure.procedure_id,
                procedure.domain,
                procedure.name,
                json.dumps(procedure.to_dict()),
                procedure.skill_id,
                procedure.version,
                procedure.usage_count,
                procedure.success_rate,
                procedure.created_at,
            ),
        )
        conn.commit()

    def get_procedure(self, procedure_id: str) -> Optional[Procedure]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT procedure_data FROM procedures WHERE procedure_id = ?", (procedure_id,)
        ).fetchone()
        return Procedure.from_dict(json.loads(row["procedure_data"])) if row else None

    def applicable_procedures(
        self, domain: str, task: str = "", tools_called: list[str] = None, limit: int = 10
    ) -> list[Procedure]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT procedure_data FROM procedures
               WHERE domain = ? ORDER BY success_rate DESC, usage_count DESC LIMIT ?""",
            (domain, limit * 2),
        ).fetchall()
        procedures = [Procedure.from_dict(json.loads(r["procedure_data"])) for r in rows]
        tools_called = tools_called or []

        scored = []
        for p in procedures:
            score = p.trigger_conditions.matches(
                task=task, tools_called=tools_called
            )
            if score > 0:
                scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:limit]]

    def record_refinement(self, event: RefinementEvent) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO refinement_history
               (procedure_id, previous_version, new_version, success_rate_delta, cost_delta, failure_reason, applied_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event.procedure_id,
                event.previous_version,
                event.new_version,
                event.success_rate_delta,
                event.cost_delta,
                event.failure_reason,
                event.applied_at,
            ),
        )
        conn.commit()

    def record_usage(self, procedure_id: str, success: bool, cost: float, duration: float) -> None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT procedure_data FROM procedures WHERE procedure_id = ?", (procedure_id,)
        ).fetchone()
        if row:
            proc = Procedure.from_dict(json.loads(row["procedure_data"]))
            old_total = proc.success_rate * proc.usage_count
            proc.usage_count += 1
            new_successes = old_total + (1.0 if success else 0.0)
            proc.success_rate = new_successes / proc.usage_count
            proc.avg_cost = ((proc.avg_cost * (proc.usage_count - 1)) + cost) / proc.usage_count
            proc.avg_duration = ((proc.avg_duration * (proc.usage_count - 1)) + duration) / proc.usage_count
            conn.execute(
                "UPDATE procedures SET procedure_data = ?, usage_count = ?, success_rate = ? WHERE procedure_id = ?",
                (json.dumps(proc.to_dict()), proc.usage_count, proc.success_rate, procedure_id),
            )
            conn.commit()

    def compose_plan(self, goal: str, skill_names: list[str]) -> list[Procedure]:
        conn = self._get_conn()
        procedures = []
        for name in skill_names:
            row = conn.execute(
                "SELECT procedure_data FROM procedures WHERE name = ? ORDER BY version DESC LIMIT 1",
                (name,),
            ).fetchone()
            if row:
                procedures.append(Procedure.from_dict(json.loads(row["procedure_data"])))
        return procedures

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
