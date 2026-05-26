from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from nil.memory.consolidation import ConsolidationPipeline, ConsolidationReport
from nil.memory.episodic import EpisodicMemory, Episode, RetentionTier
from nil.memory.meta_memory import MetaMemory, KnowledgeGap, Contradiction
from nil.memory.procedural import ProceduralMemory, Procedure
from nil.memory.semantic import SemanticMemory, Fact, EntityGraph
from nil.memory.social_memory import SocialMemory, UserProfile, Correction
from nil.memory.store import PersistentMemory
from nil.memory.working import WorkingMemory, ContentBlock, ContentType, Goal
from nil.memory.world_state import WorldStateMemory, WorldSnapshot, StateDelta

logger = logging.getLogger("nil.memory.hierarchical")


class HierarchicalMemory:
    def __init__(self, path: str = "./nil_store", config: Optional[dict] = None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.path / "nil.db")
        self.config = config or {}

        self.episodic = EpisodicMemory(self.db_path)
        self.semantic = SemanticMemory(self.db_path)
        self.procedural = ProceduralMemory(self.db_path)
        self.working = WorkingMemory(
            token_budget=self.config.get("working_token_budget", 8000)
        )
        self.world_state = WorldStateMemory(self.db_path)
        self.social = SocialMemory(self.db_path)
        self.meta = MetaMemory(self.db_path, semantic=self.semantic)
        self.consolidation = ConsolidationPipeline(
            episodic=self.episodic,
            semantic=self.semantic,
            procedural=self.procedural,
            meta=self.meta,
            config=self.config,
        )

        self._init_all_tables()

    def _init_all_tables(self) -> None:
        self.episodic.init_tables()
        self.semantic.init_tables()
        self.procedural.init_tables()
        self.world_state.init_tables()
        self.social.init_tables()
        self.meta.init_tables()
        self._init_legacy_tables()

    def _init_legacy_tables(self) -> None:
        import sqlite3
        conn = sqlite3.connect(self.db_path)
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
            CREATE INDEX IF NOT EXISTS idx_traces_domain ON traces(domain);

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
        conn.close()

    # --- Trace compatibility (backward-compatible with PersistentMemory) ---

    def store_trace(self, trace: Any) -> None:
        trace_dict = trace.to_dict() if hasattr(trace, "to_dict") else trace
        ep = Episode(
            episode_id=trace_dict.get("run_id", str(uuid.uuid4())),
            domain=trace_dict.get("domain", ""),
            task=trace_dict.get("task", ""),
            steps=trace_dict.get("steps", []),
            outcome={
                "success": trace_dict.get("outcome_success", False),
                "output": trace_dict.get("outcome_output"),
                "error": trace_dict.get("error"),
                "cost": trace_dict.get("total_cost", 0),
                "duration": trace_dict.get("total_duration", 0),
            },
            importance=0.5,
            created_at=trace_dict.get("created_at", time.time()),
        )
        ep.importance = self.episodic.compute_importance(ep)
        self.episodic.store_episode(ep)

    def get_traces(
        self, domain: str, limit: int = 50, success_only: bool = False
    ) -> list[dict]:
        episodes = self.episodic.search_episodes(domain=domain, limit=limit)
        if success_only:
            episodes = [e for e in episodes if e.outcome.get("success")]
        return [e.to_dict() for e in episodes]

    def get_trace_count(self, domain: str) -> int:
        return len(self.episodic.search_episodes(domain=domain, limit=100000))

    def get_success_rate(self, domain: str, last_n: int = 50) -> float:
        episodes = self.episodic.search_episodes(domain=domain, limit=last_n)
        if not episodes:
            return 0.0
        successes = sum(1 for e in episodes if e.outcome.get("success"))
        return successes / len(episodes)

    def store_skill(
        self,
        domain: str,
        name: str,
        description: str = "",
        pattern: Optional[dict] = None,
        source_runs: Optional[list[str]] = None,
        confidence: float = 0.0,
    ) -> None:
        from nil.memory.skill_registry import Skill, TriggerCondition, SkillOrigin, SkillStatus
        skill = Skill(
            name=f"{domain}:{name}",
            description=description,
            compressed_instruction=description,
            confidence=confidence,
            source_runs=source_runs or [],
            status=SkillStatus.ACTIVE,
            origin=SkillOrigin.PATTERN_EXTRACTION,
        )
        self.procedural.store_skill(skill)

    def list_skills(self, domain: str) -> list[str]:
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT name FROM skills WHERE domain = ? ORDER BY confidence DESC",
                (domain,),
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def get_skill(self, domain: str, name: str) -> Optional[dict]:
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT * FROM skills WHERE domain = ? AND name = ?",
                (domain, name),
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def store_strategy(self, domain: str, strategy_data: dict, score: float = 0.0) -> int:
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT MAX(version) as max_v FROM strategy_versions WHERE domain = ?",
                (domain,),
            ).fetchone()
            next_version = (row[0] or 0) + 1
            conn.execute(
                "UPDATE strategy_versions SET is_active = 0 WHERE domain = ? AND is_active = 1",
                (domain,),
            )
            conn.execute(
                """INSERT INTO strategy_versions
                   (domain, version, strategy_data, score, is_active, created_at)
                   VALUES (?, ?, ?, ?, 1, ?)""",
                (domain, next_version, json.dumps(strategy_data), score, time.time()),
            )
            conn.commit()
            return next_version
        finally:
            conn.close()

    def get_active_strategy(self, domain: str) -> Optional[dict]:
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT strategy_data FROM strategy_versions WHERE domain = ? AND is_active = 1",
                (domain,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])
        finally:
            conn.close()

    def close(self) -> None:
        self.episodic.close()
        self.semantic.close()
        self.procedural.close()
        self.world_state.close()
        self.social.close()
        self.meta.close()

    # --- Episodic ---

    def recall_episode(self, episode_id: str) -> Optional[Episode]:
        return self.episodic.get_episode(episode_id)

    def search_episodes(self, query: str = "", domain: str = "", limit: int = 10) -> list[Episode]:
        return self.episodic.search_episodes(domain=domain, query=query, limit=limit)

    def timeline(self, domain: str, start: float, end: float) -> list[Episode]:
        return self.episodic.timeline(domain, start, end)

    def create_session(self, domain: str, task: str = "") -> str:
        return self.episodic.create_session(domain, task)

    def update_session_activity(self, session_id: str) -> None:
        self.episodic.update_session_activity(session_id)

    def close_session(self, session_id: str) -> None:
        self.episodic.close_session(session_id)

    # --- Semantic ---

    def query_facts(
        self, subject: str = "", predicate: str = "", domain: str = ""
    ) -> list[Fact]:
        return self.semantic.query_facts(
            domain=domain, subject=subject, predicate=predicate
        )

    def belief(self, statement: str, domain: str = "") -> float:
        return self.semantic.belief(statement, domain=domain)

    def relationships(self, entity: str, domain: str = "") -> list[dict]:
        return self.semantic.relationships(entity, domain=domain)

    def get_knowledge_graph(self, domain: str) -> EntityGraph:
        return self.semantic.get_knowledge_graph(domain)

    def observe_fact(self, fact: Fact, source_episode_id: str = "") -> None:
        self.semantic.observe_fact(fact, source_episode_id=source_episode_id)

    def track_entity(
        self, name: str, entity_type: str = "", properties: dict = None, domain: str = ""
    ) -> None:
        self.semantic.track_entity(name, entity_type, properties, domain)

    # --- Procedural ---

    def applicable_skills(self, task: str = "", domain: str = "", tools_called: list[str] = None) -> list[Procedure]:
        return self.procedural.applicable_procedures(
            domain=domain, task=task, tools_called=tools_called
        )

    def compose_plan(self, goal: str, skill_names: list[str]) -> list[Procedure]:
        return self.procedural.compose_plan(goal, skill_names)

    def record_refinement(self, procedure_id: str, previous_version: int, new_version: int,
                           success_delta: float, cost_delta: float, reason: str = "") -> None:
        from nil.memory.procedural import RefinementEvent
        self.procedural.record_refinement(RefinementEvent(
            procedure_id=procedure_id,
            previous_version=previous_version,
            new_version=new_version,
            success_rate_delta=success_delta,
            cost_delta=cost_delta,
            failure_reason=reason,
        ))

    # --- Working ---

    def create_working_session(self, task: str) -> str:
        return self.working.create_session(task)

    def push_context(
        self, session_id: str, content: Any, content_type: ContentType,
        importance: float = 0.5, tokens: int = 0,
    ) -> str:
        return self.working.push_context(session_id, content, content_type, importance, tokens)

    def pop_context(self, session_id: str) -> Optional[ContentBlock]:
        return self.working.pop_context(session_id)

    def attend_to(self, session_id: str, ref: str) -> None:
        self.working.attend_to(session_id, ref)

    def update_scratchpad(self, session_id: str, text: str) -> None:
        self.working.update_scratchpad(session_id, text)

    def get_scratchpad(self, session_id: str) -> str:
        return self.working.get_scratchpad(session_id)

    def get_context(self, session_id: str) -> list[ContentBlock]:
        return self.working.get_context(session_id)

    def set_goal(self, session_id: str, goal: str) -> None:
        self.working.set_goal(session_id, goal)

    def add_subgoal(self, session_id: str, description: str) -> Goal:
        return self.working.add_subgoal(session_id, description)

    def complete_subgoal(self, session_id: str, goal_id: str) -> None:
        self.working.complete_subgoal(session_id, goal_id)

    def destroy_working_session(self, session_id: str) -> None:
        self.working.destroy_session(session_id)

    # --- World State ---

    def get_state(self, domain: str) -> Optional[WorldSnapshot]:
        return self.world_state.get_state(domain)

    def update_state(self, domain: str, updates: dict, episode_id: str = "") -> WorldSnapshot:
        return self.world_state.update_state(domain, updates, episode_id)

    def verify_state(self, domain: str, observed: dict) -> dict:
        return self.world_state.verify_state(domain, observed)

    def predict_state(self, domain: str, action: str) -> dict:
        return self.world_state.predict_state(domain, action)

    def get_delta_log(self, domain: str, field: str = "", limit: int = 100) -> list[StateDelta]:
        return self.world_state.get_delta_log(domain, field, limit)

    # --- Social/User ---

    def user_profile(self, user_id: str) -> UserProfile:
        return self.social.get_or_create_user(user_id)

    def record_interaction(self, user_id: str, domain: str, request: str = "") -> None:
        self.social.record_interaction(user_id, domain, request)

    def record_correction(self, user_id: str, correction: Correction) -> None:
        self.social.record_correction(user_id, correction)

    def adapt_to_user(self, user_id: str, response: str, domain: str = "") -> str:
        return self.social.adapt_to_user(user_id, response, domain)

    # --- Meta ---

    def knowledge_gaps(self, task: str = "", domain: str = "") -> list[KnowledgeGap]:
        gaps = self.meta.knowledge_gaps(domain=domain)
        if task:
            task_lower = task.lower()
            gaps = [g for g in gaps if g.area.lower() in task_lower or task_lower in g.area.lower()]
        return gaps

    def confidence_in(self, claim: str, domain: str = "") -> float:
        return self.meta.confidence_in(claim, domain=domain)

    def contradictions(self, domain: str = "") -> list[Contradiction]:
        return self.meta.contradictions(domain=domain)

    def resolve_contradiction(self, contradiction_id: str, resolution: str) -> None:
        self.meta.resolve_contradiction(contradiction_id, resolution)

    def knowledge_summary(self, domain: str) -> dict:
        return self.meta.knowledge_summary(domain)

    def audit(self, domain: str) -> dict:
        return self.meta.audit(domain)

    def identify_gap(self, area: str, evidence: str = "", domain: str = "") -> KnowledgeGap:
        return self.meta.identify_gap(area, evidence, domain)

    # --- Consolidation ---

    def consolidate(self, domain: str = "") -> ConsolidationReport:
        return self.consolidation.run(domain)
