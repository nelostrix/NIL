from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("nil.memory.social")


class VerbosityLevel(str, Enum):
    CONCISE = "concise"
    NORMAL = "normal"
    DETAILED = "detailed"


class CommunicationStyle(str, Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"


@dataclass
class Correction:
    episode_id: str
    original: str
    corrected: str
    domain: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "original": self.original,
            "corrected": self.corrected,
            "domain": self.domain,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Correction:
        return cls(
            episode_id=d.get("episode_id", ""),
            original=d.get("original", ""),
            corrected=d.get("corrected", ""),
            domain=d.get("domain", ""),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class UserProfile:
    user_id: str
    name: str = ""
    interaction_count: int = 0
    preferred_verbosity: VerbosityLevel = VerbosityLevel.NORMAL
    communication_style: CommunicationStyle = CommunicationStyle.CASUAL
    domain_expertise: dict[str, float] = field(default_factory=dict)
    common_requests: list[str] = field(default_factory=list)
    trust_per_domain: dict[str, float] = field(default_factory=dict)
    last_interaction: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "interaction_count": self.interaction_count,
            "preferred_verbosity": self.preferred_verbosity.value,
            "communication_style": self.communication_style.value,
            "domain_expertise": self.domain_expertise,
            "common_requests": self.common_requests,
            "trust_per_domain": self.trust_per_domain,
            "last_interaction": self.last_interaction,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> UserProfile:
        return cls(
            user_id=d.get("user_id", ""),
            name=d.get("name", ""),
            interaction_count=d.get("interaction_count", 0),
            preferred_verbosity=VerbosityLevel(d.get("preferred_verbosity", "normal")),
            communication_style=CommunicationStyle(d.get("communication_style", "casual")),
            domain_expertise=d.get("domain_expertise", {}),
            common_requests=d.get("common_requests", []),
            trust_per_domain=d.get("trust_per_domain", {}),
            last_interaction=d.get("last_interaction", 0.0),
            created_at=d.get("created_at", 0.0),
        )


class SocialMemory:
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
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_data TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                episode_id TEXT,
                original TEXT,
                corrected TEXT,
                domain TEXT,
                timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_corrections_user ON corrections(user_id);
            CREATE INDEX IF NOT EXISTS idx_corrections_domain ON corrections(domain);
        """)
        conn.commit()

    def get_or_create_user(self, user_id: str, name: str = "") -> UserProfile:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT profile_data FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row:
            return UserProfile.from_dict(json.loads(row["profile_data"]))
        profile = UserProfile(user_id=user_id, name=name or user_id)
        conn.execute(
            "INSERT INTO user_profiles (user_id, profile_data) VALUES (?, ?)",
            (user_id, json.dumps(profile.to_dict())),
        )
        conn.commit()
        return profile

    def update_user(self, profile: UserProfile) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO user_profiles (user_id, profile_data) VALUES (?, ?)",
            (profile.user_id, json.dumps(profile.to_dict())),
        )
        conn.commit()

    def record_interaction(self, user_id: str, domain: str, request: str = "") -> None:
        profile = self.get_or_create_user(user_id, name=user_id)
        conn = self._get_conn()
        profile.interaction_count += 1
        profile.last_interaction = time.time()
        if request and request not in profile.common_requests:
            profile.common_requests.append(request)
            if len(profile.common_requests) > 20:
                profile.common_requests = profile.common_requests[-20:]
        conn.execute(
            "UPDATE user_profiles SET profile_data = ? WHERE user_id = ?",
            (json.dumps(profile.to_dict()), user_id),
        )
        conn.commit()

    def record_correction(
        self, user_id: str, correction: Correction
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO corrections (user_id, episode_id, original, corrected, domain, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, correction.episode_id, correction.original,
             correction.corrected, correction.domain, correction.timestamp),
        )

        row = conn.execute(
            "SELECT profile_data FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row:
            profile = UserProfile.from_dict(json.loads(row["profile_data"]))
            domain = correction.domain
            current_trust = profile.trust_per_domain.get(domain, 0.5)
            profile.trust_per_domain[domain] = max(current_trust - 0.05, 0.0)
            conn.execute(
                "UPDATE user_profiles SET profile_data = ? WHERE user_id = ?",
                (json.dumps(profile.to_dict()), user_id),
            )
        conn.commit()

    def adapt_to_user(
        self, user_id: str, response: str, domain: str = ""
    ) -> str:
        profile = self.get_or_create_user(user_id)
        style = profile.communication_style
        verbosity = profile.preferred_verbosity

        if verbosity == VerbosityLevel.CONCISE:
            lines = response.strip().split("\n")
            response = "\n".join(
                line for line in lines
                if not line.startswith("  ") and len(line.strip()) > 10
            )[:2000]
        elif verbosity == VerbosityLevel.DETAILED:
            pass

        if style == CommunicationStyle.TECHNICAL and domain:
            if domain not in response.lower():
                response = f"[{domain.upper()}]\n{response}"
        elif style == CommunicationStyle.FORMAL:
            if not response.startswith(("Dear", "Hello", "Hi")):
                pass

        return response

    def get_correction_history(
        self, user_id: str, domain: str = "", limit: int = 20
    ) -> list[Correction]:
        conn = self._get_conn()
        params: list[Any] = [user_id]
        sql = "SELECT * FROM corrections WHERE user_id = ?"
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return [Correction.from_dict(dict(r)) for r in rows]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
