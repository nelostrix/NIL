from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("nil.memory.world_state")


@dataclass
class StateDelta:
    timestamp: float
    field: str
    from_value: Any
    to_value: Any

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "field": self.field,
            "from": self.from_value,
            "to": self.to_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StateDelta:
        return cls(
            timestamp=d.get("timestamp", 0.0),
            field=d.get("field", ""),
            from_value=d.get("from"),
            to_value=d.get("to"),
        )


@dataclass
class WorldSnapshot:
    world_id: str
    domain: str
    state_snapshot: dict = field(default_factory=dict)
    state_schema: dict = field(default_factory=dict)
    last_verified_at: float = field(default_factory=time.time)
    confidence: float = 1.0
    delta_count: int = 0

    def to_dict(self) -> dict:
        return {
            "world_id": self.world_id,
            "domain": self.domain,
            "state_snapshot": self.state_snapshot,
            "state_schema": self.state_schema,
            "last_verified_at": self.last_verified_at,
            "confidence": self.confidence,
            "delta_count": self.delta_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> WorldSnapshot:
        return cls(
            world_id=d.get("world_id", ""),
            domain=d.get("domain", ""),
            state_snapshot=d.get("state_snapshot", {}),
            state_schema=d.get("state_schema", {}),
            last_verified_at=d.get("last_verified_at", 0.0),
            confidence=d.get("confidence", 1.0),
            delta_count=d.get("delta_count", 0),
        )


class WorldStateMemory:
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
            CREATE TABLE IF NOT EXISTS world_snapshots (
                world_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL UNIQUE,
                snapshot_data TEXT NOT NULL,
                last_verified_at REAL NOT NULL,
                confidence REAL DEFAULT 1.0,
                delta_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS state_deltas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                field TEXT NOT NULL,
                from_value TEXT,
                to_value TEXT,
                timestamp REAL NOT NULL,
                episode_id TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_deltas_domain ON state_deltas(domain);
            CREATE INDEX IF NOT EXISTS idx_deltas_field ON state_deltas(domain, field);
        """)
        conn.commit()

    def get_state(self, domain: str) -> Optional[WorldSnapshot]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM world_snapshots WHERE domain = ?", (domain,)
        ).fetchone()
        if row is None:
            return None
        snap = WorldSnapshot.from_dict(json.loads(row["snapshot_data"]))
        snap.last_verified_at = row["last_verified_at"]
        snap.confidence = row["confidence"]
        return snap

    def update_state(
        self, domain: str, updates: dict, episode_id: str = ""
    ) -> WorldSnapshot:
        conn = self._get_conn()
        existing = self.get_state(domain)
        now = time.time()

        if existing:
            old_state = dict(existing.state_snapshot)
            new_state = dict(old_state)
            new_state.update(updates)

            for key, new_val in updates.items():
                old_val = old_state.get(key)
                if old_val != new_val:
                    conn.execute(
                        """INSERT INTO state_deltas (domain, field, from_value, to_value, timestamp, episode_id)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (domain, key, json.dumps(old_val), json.dumps(new_val), now, episode_id),
                    )

            existing.state_snapshot = new_state
            existing.last_verified_at = now
            existing.delta_count += len(updates)
            existing.confidence = min(existing.confidence + 0.05, 1.0)

            conn.execute(
                """UPDATE world_snapshots SET snapshot_data = ?, last_verified_at = ?,
                   confidence = ?, delta_count = ? WHERE domain = ?""",
                (json.dumps(existing.to_dict()), now, existing.confidence,
                 existing.delta_count, domain),
            )
        else:
            snap = WorldSnapshot(
                world_id=str(uuid.uuid4()),
                domain=domain,
                state_snapshot=updates,
                last_verified_at=now,
            )
            conn.execute(
                """INSERT INTO world_snapshots (world_id, domain, snapshot_data, last_verified_at, confidence, delta_count)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (snap.world_id, domain, json.dumps(snap.to_dict()), now, snap.confidence, 0),
            )
            existing = snap

        conn.commit()
        return existing

    def verify_state(self, domain: str, observed: dict) -> dict[str, Any]:
        existing = self.get_state(domain)
        if existing is None:
            return {"matches": False, "reason": "no_state"}

        discrepancies = {}
        for key, observed_val in observed.items():
            stored_val = existing.state_snapshot.get(key)
            if stored_val is not None and stored_val != observed_val:
                discrepancies[key] = {"expected": stored_val, "observed": observed_val}

        if discrepancies:
            existing.confidence = max(existing.confidence - 0.1, 0.0)
            conn = self._get_conn()
            conn.execute(
                "UPDATE world_snapshots SET confidence = ? WHERE domain = ?",
                (existing.confidence, domain),
            )
            conn.commit()
        else:
            conn = self._get_conn()
            existing.last_verified_at = time.time()
            existing.confidence = min(existing.confidence + 0.05, 1.0)
            conn.execute(
                "UPDATE world_snapshots SET confidence = ?, last_verified_at = ? WHERE domain = ?",
                (existing.confidence, existing.last_verified_at, domain),
            )
            conn.commit()

        return {
            "matches": len(discrepancies) == 0,
            "discrepancies": discrepancies,
        }

    def get_delta_log(
        self, domain: str, field: str = "", limit: int = 100
    ) -> list[StateDelta]:
        conn = self._get_conn()
        params: list[Any] = [domain]
        sql = "SELECT * FROM state_deltas WHERE domain = ?"
        if field:
            sql += " AND field = ?"
            params.append(field)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return [
            StateDelta(
                timestamp=r["timestamp"],
                field=r["field"],
                from_value=json.loads(r["from_value"]) if r["from_value"] else None,
                to_value=json.loads(r["to_value"]) if r["to_value"] else None,
            )
            for r in rows
        ]

    def predict_state(self, domain: str, action: str) -> dict:
        conn = self._get_conn()
        similar_deltas = conn.execute(
            """SELECT * FROM state_deltas WHERE domain = ? ORDER BY timestamp DESC LIMIT 20""",
            (domain,),
        ).fetchall()

        predictions = {}
        field_transitions: dict[str, list[Any]] = {}
        for d in similar_deltas:
            field = d["field"]
            to_val = json.loads(d["to_value"]) if d["to_value"] else None
            if field not in field_transitions:
                field_transitions[field] = []
            field_transitions[field].append(to_val)

        for field, values in field_transitions.items():
            if values:
                predictions[field] = values[-1]

        return {"action": action, "predicted_state": predictions, "confidence": 0.3}

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
