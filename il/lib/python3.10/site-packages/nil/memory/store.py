"""
PersistentMemory — the agent's long-term memory.

Uses SQLite for zero-dependency, zero-cost persistent storage.
Stores execution traces, extracted skills, and strategy versions.

This is what makes Nil agents different from goldfish agents:
they remember what they've done, what worked, and what didn't.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("nil.memory")


class PersistentMemory:
    """
    SQLite-backed persistent memory for agent traces and skills.

    Usage:
        memory = PersistentMemory("./agent_store")
        memory.store_trace(trace)
        traces = memory.get_traces("my-domain", limit=10)
        skills = memory.list_skills("my-domain")
    """

    def __init__(self, path: str = "./nil_store") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / "nil.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS traces (
                run_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                task TEXT,
                trace_data TEXT NOT NULL,
                success INTEGER DEFAULT 0,
                duration REAL DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0,
                tool_sequence TEXT,
                error TEXT,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_traces_domain
                ON traces(domain);
            CREATE INDEX IF NOT EXISTS idx_traces_success
                ON traces(domain, success);
            CREATE INDEX IF NOT EXISTS idx_traces_created
                ON traces(domain, created_at DESC);

            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                pattern TEXT,
                source_runs TEXT,
                confidence REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                UNIQUE(domain, name)
            );

            CREATE TABLE IF NOT EXISTS strategy_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                version INTEGER NOT NULL,
                strategy_data TEXT NOT NULL,
                score REAL DEFAULT 0.0,
                is_active INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                UNIQUE(domain, version)
            );
        """)
        conn.commit()
        logger.debug("Memory store initialized at %s", self.db_path)

    # ---- Traces ----

    def store_trace(self, trace: Any) -> None:
        """Store a completed trace."""
        conn = self._get_conn()
        trace_dict = trace.to_dict() if hasattr(trace, "to_dict") else trace
        conn.execute(
            """INSERT OR REPLACE INTO traces
               (run_id, domain, task, trace_data, success, duration,
                total_tokens, total_cost, tool_sequence, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace_dict["run_id"],
                trace_dict["domain"],
                trace_dict.get("task", ""),
                json.dumps(trace_dict),
                1 if trace_dict.get("outcome_success") else 0,
                trace_dict.get("total_duration", 0),
                trace_dict.get("total_tokens_in", 0)
                + trace_dict.get("total_tokens_out", 0),
                trace_dict.get("total_cost", 0),
                json.dumps(trace_dict.get("tool_sequence", [])),
                trace_dict.get("error"),
                trace_dict.get("created_at", 0),
            ),
        )
        conn.commit()

    def get_traces(
        self,
        domain: str,
        limit: int = 50,
        success_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve traces for a domain, newest first."""
        conn = self._get_conn()
        query = "SELECT trace_data FROM traces WHERE domain = ?"
        params: list[Any] = [domain]
        if success_only:
            query += " AND success = 1"
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [json.loads(row["trace_data"]) for row in rows]

    def get_trace_count(self, domain: str) -> int:
        """Count total traces for a domain."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM traces WHERE domain = ?", (domain,)
        ).fetchone()
        return row["cnt"] if row else 0

    def get_success_rate(self, domain: str, last_n: int = 50) -> float:
        """Calculate success rate over the last N runs."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT success FROM traces
               WHERE domain = ?
               ORDER BY created_at DESC LIMIT ?""",
            (domain, last_n),
        ).fetchall()
        if not rows:
            return 0.0
        return sum(r["success"] for r in rows) / len(rows)

    # ---- Skills ----

    def store_skill(
        self,
        domain: str,
        name: str,
        description: str = "",
        pattern: Optional[dict[str, Any]] = None,
        source_runs: Optional[list[str]] = None,
        confidence: float = 0.0,
    ) -> None:
        """Store an extracted skill."""
        import time

        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO skills
               (domain, name, description, pattern, source_runs, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                domain,
                name,
                description,
                json.dumps(pattern) if pattern else None,
                json.dumps(source_runs) if source_runs else None,
                confidence,
                time.time(),
            ),
        )
        conn.commit()

    def list_skills(self, domain: str) -> list[str]:
        """Return skill names for a domain."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT name FROM skills WHERE domain = ? ORDER BY confidence DESC",
            (domain,),
        ).fetchall()
        return [row["name"] for row in rows]

    def get_skill(self, domain: str, name: str) -> Optional[dict[str, Any]]:
        """Get full skill details."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM skills WHERE domain = ? AND name = ?",
            (domain, name),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    # ---- Strategy Versions ----

    def store_strategy(
        self,
        domain: str,
        strategy_data: dict[str, Any],
        score: float = 0.0,
    ) -> int:
        """Store a new strategy version. Returns the version number."""
        import time

        conn = self._get_conn()

        # Get next version number
        row = conn.execute(
            "SELECT MAX(version) as max_v FROM strategy_versions WHERE domain = ?",
            (domain,),
        ).fetchone()
        next_version = (row["max_v"] or 0) + 1

        # Deactivate previous active strategy
        conn.execute(
            "UPDATE strategy_versions SET is_active = 0 WHERE domain = ? AND is_active = 1",
            (domain,),
        )

        # Insert new active strategy
        conn.execute(
            """INSERT INTO strategy_versions
               (domain, version, strategy_data, score, is_active, created_at)
               VALUES (?, ?, ?, ?, 1, ?)""",
            (domain, next_version, json.dumps(strategy_data), score, time.time()),
        )
        conn.commit()
        return next_version

    def get_active_strategy(self, domain: str) -> Optional[dict[str, Any]]:
        """Get the currently active strategy for a domain."""
        conn = self._get_conn()
        row = conn.execute(
            """SELECT strategy_data FROM strategy_versions
               WHERE domain = ? AND is_active = 1""",
            (domain,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["strategy_data"])

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
