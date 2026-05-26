from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("nil.memory.episodic")


class RetentionTier(str, Enum):
    RECENT = "recent"
    SUMMARIZED = "summarized"
    COMPRESSED = "compressed"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"


@dataclass
class Episode:
    episode_id: str
    domain: str
    task: str
    steps: list[dict] = field(default_factory=list)
    outcome: dict = field(default_factory=dict)
    session_id: str = ""
    importance: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    retention_tier: RetentionTier = RetentionTier.RECENT
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "domain": self.domain,
            "task": self.task,
            "steps": self.steps,
            "outcome": self.outcome,
            "session_id": self.session_id,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "retention_tier": self.retention_tier.value,
            "summary": self.summary,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Episode:
        return cls(
            episode_id=d["episode_id"],
            domain=d.get("domain", ""),
            task=d.get("task", ""),
            steps=d.get("steps", []),
            outcome=d.get("outcome", {}),
            session_id=d.get("session_id", ""),
            importance=d.get("importance", 0.0),
            created_at=d.get("created_at", 0.0),
            last_accessed_at=d.get("last_accessed_at", 0.0),
            access_count=d.get("access_count", 0),
            retention_tier=RetentionTier(d.get("retention_tier", "recent")),
            summary=d.get("summary", ""),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
        )


class EpisodicMemory:
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
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                task TEXT,
                episode_data TEXT NOT NULL,
                session_id TEXT,
                importance REAL DEFAULT 0.0,
                retention_tier TEXT DEFAULT 'recent',
                created_at REAL NOT NULL,
                last_accessed_at REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_domain ON episodes(domain);
            CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_retention ON episodes(domain, retention_tier);
            CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(domain, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(domain, importance DESC);

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                task TEXT,
                episode_count INTEGER DEFAULT 0,
                started_at REAL NOT NULL,
                last_activity_at REAL NOT NULL,
                status TEXT DEFAULT 'active',
                metadata TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_domain ON sessions(domain);
        """)
        conn.commit()

    def store_episode(self, episode: Episode) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO episodes
               (episode_id, domain, task, episode_data, session_id, importance,
                retention_tier, created_at, last_accessed_at, access_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.episode_id,
                episode.domain,
                episode.task,
                json.dumps(episode.to_dict()),
                episode.session_id,
                episode.importance,
                episode.retention_tier.value,
                episode.created_at,
                episode.last_accessed_at,
                episode.access_count,
            ),
        )
        conn.commit()

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT episode_data FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE episodes SET access_count = access_count + 1, last_accessed_at = ? WHERE episode_id = ?",
            (time.time(), episode_id),
        )
        conn.commit()
        return Episode.from_dict(json.loads(row["episode_data"]))

    def search_episodes(
        self, domain: str, query: str = "", limit: int = 10, tier: Optional[RetentionTier] = None
    ) -> list[Episode]:
        conn = self._get_conn()
        params: list[Any] = [domain]
        sql = "SELECT episode_data FROM episodes WHERE domain = ?"
        if tier:
            sql += " AND retention_tier = ?"
            params.append(tier.value)
        sql += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()

        episodes = [Episode.from_dict(json.loads(r["episode_data"])) for r in rows]
        if query:
            q = query.lower()
            episodes = [e for e in episodes if q in e.task.lower() or q in e.summary.lower()]
        return episodes

    def timeline(
        self, domain: str, start: float, end: float, limit: int = 100
    ) -> list[Episode]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT episode_data FROM episodes
               WHERE domain = ? AND created_at >= ? AND created_at <= ?
               ORDER BY created_at DESC LIMIT ?""",
            (domain, start, end, limit),
        ).fetchall()
        return [Episode.from_dict(json.loads(r["episode_data"])) for r in rows]

    def create_session(self, domain: str, task: str = "") -> str:
        session_id = str(uuid.uuid4())
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO sessions
               (session_id, domain, task, started_at, last_activity_at)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, domain, task, time.time(), time.time()),
        )
        conn.commit()
        return session_id

    def update_session_activity(self, session_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET last_activity_at = ?, episode_count = episode_count + 1 WHERE session_id = ?",
            (time.time(), session_id),
        )
        conn.commit()

    def close_session(self, session_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET status = 'closed', last_activity_at = ? WHERE session_id = ?",
            (time.time(), session_id),
        )
        conn.commit()

    def compute_importance(self, episode: Episode) -> float:
        score = 0.0
        if episode.outcome.get("success"):
            score += 0.3
        else:
            score += 0.5
        step_count = len(episode.steps)
        score += min(step_count / 10, 0.2)
        if episode.outcome.get("error"):
            score += 0.2
        if episode.tags:
            score += min(len(episode.tags) * 0.05, 0.1)
        return min(score, 1.0)

    def apply_retention(self, domain: str, config: dict) -> int:
        conn = self._get_conn()
        max_recent = config.get("max_recent", 100)
        max_summarized = config.get("max_summarized", 500)
        max_compressed = config.get("max_compressed", 2000)
        importance_threshold = config.get("importance_threshold", 0.3)
        forgotten_count = 0

        rows = conn.execute(
            "SELECT episode_id, episode_data FROM episodes WHERE domain = ? AND retention_tier = 'recent' ORDER BY importance DESC",
            (domain,),
        ).fetchall()

        if len(rows) > max_recent:
            to_demote = rows[max_recent:]
            for row in to_demote:
                ep = Episode.from_dict(json.loads(row["episode_data"]))
                if ep.importance >= importance_threshold:
                    tier = RetentionTier.SUMMARIZED
                else:
                    tier = RetentionTier.COMPRESSED
                ep.retention_tier = tier
                conn.execute(
                    "UPDATE episodes SET retention_tier = ?, episode_data = ? WHERE episode_id = ?",
                    (tier.value, json.dumps(ep.to_dict()), ep.episode_id),
                )

        for tier_name, max_count in [
            (RetentionTier.SUMMARIZED, max_summarized),
            (RetentionTier.COMPRESSED, max_compressed),
        ]:
            rows = conn.execute(
                "SELECT episode_id FROM episodes WHERE domain = ? AND retention_tier = ? ORDER BY importance ASC",
                (domain, tier_name.value),
            ).fetchall()
            if len(rows) > max_count:
                to_forget = rows[: len(rows) - max_count]
                for row in to_forget:
                    conn.execute(
                        "UPDATE episodes SET retention_tier = 'archived' WHERE episode_id = ?",
                        (row["episode_id"],),
                    )

        archive_rows = conn.execute(
            "SELECT episode_id FROM episodes WHERE domain = ? AND retention_tier = 'archived' AND created_at < ?",
            (domain, time.time() - config.get("archive_ttl_days", 90) * 86400),
        ).fetchall()
        for row in archive_rows:
            conn.execute("DELETE FROM episodes WHERE episode_id = ?", (row["episode_id"],))
            forgotten_count += 1

        conn.commit()
        return forgotten_count

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
