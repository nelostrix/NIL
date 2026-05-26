from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("nil.memory.working")


class ContentType(str, Enum):
    GOAL = "goal"
    SUBGOAL = "subgoal"
    OBSERVATION = "observation"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"
    SYSTEM = "system"
    USER_INPUT = "user_input"


@dataclass
class ContentBlock:
    ref: str
    content_type: ContentType
    content: Any
    importance: float = 0.5
    tokens: int = 0
    created_at: float = field(default_factory=time.time)
    last_referenced_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def touch(self) -> None:
        self.last_referenced_at = time.time()


@dataclass
class Goal:
    goal_id: str
    description: str
    completed: bool = False
    subgoals: list[Goal] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class WorkingMemoryState:
    session_id: str
    task: str
    current_goal: str = ""
    subgoals: list[Goal] = field(default_factory=list)
    context_window: list[ContentBlock] = field(default_factory=list)
    scratchpad: str = ""
    token_budget: int = 8000
    created_at: float = field(default_factory=time.time)


class WorkingMemory:
    def __init__(self, token_budget: int = 8000):
        self._sessions: dict[str, WorkingMemoryState] = {}
        self.default_token_budget = token_budget

    def create_session(self, task: str, session_id: str = "") -> str:
        session_id = session_id or str(uuid.uuid4())
        self._sessions[session_id] = WorkingMemoryState(
            session_id=session_id,
            task=task,
            token_budget=self.default_token_budget,
        )
        return session_id

    def get_session(self, session_id: str) -> Optional[WorkingMemoryState]:
        return self._sessions.get(session_id)

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def destroy_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def set_goal(self, session_id: str, goal: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session.current_goal = goal

    def add_subgoal(self, session_id: str, description: str) -> Goal:
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"No session: {session_id}")
        goal = Goal(
            goal_id=str(uuid.uuid4()),
            description=description,
        )
        session.subgoals.append(goal)
        return goal

    def complete_subgoal(self, session_id: str, goal_id: str) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return

        def _find(g: Goal) -> bool:
            if g.goal_id == goal_id:
                g.completed = True
                return True
            for sg in g.subgoals:
                if _find(sg):
                    return True
            return False

        for g in session.subgoals:
            if _find(g):
                break

    def push_context(
        self,
        session_id: str,
        content: Any,
        content_type: ContentType = ContentType.OBSERVATION,
        importance: float = 0.5,
        tokens: int = 0,
    ) -> str:
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"No session: {session_id}")

        ref = str(uuid.uuid4())
        block = ContentBlock(
            ref=ref,
            content_type=content_type,
            content=content,
            importance=importance,
            tokens=tokens,
        )
        session.context_window.append(block)
        self._evict_if_needed(session)
        return ref

    def pop_context(self, session_id: str) -> Optional[ContentBlock]:
        session = self._sessions.get(session_id)
        if session and session.context_window:
            return session.context_window.pop()
        return None

    def get_context(self, session_id: str) -> list[ContentBlock]:
        session = self._sessions.get(session_id)
        return list(session.context_window) if session else []

    def attend_to(self, session_id: str, ref: str) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        for block in session.context_window:
            if block.ref == ref:
                block.touch()
                break

    def update_scratchpad(self, session_id: str, text: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session.scratchpad = text

    def get_scratchpad(self, session_id: str) -> str:
        session = self._sessions.get(session_id)
        return session.scratchpad if session else ""

    def current_token_usage(self, session_id: str) -> int:
        session = self._sessions.get(session_id)
        if not session:
            return 0
        return sum(b.tokens for b in session.context_window)

    def _evict_if_needed(self, session: WorkingMemoryState) -> None:
        total = sum(b.tokens for b in session.context_window)
        if total <= session.token_budget:
            return

        def _eviction_score(block: ContentBlock) -> float:
            score = block.importance
            if block.content_type in (ContentType.GOAL, ContentType.SUBGOAL):
                score += 0.3
            age = time.time() - block.last_referenced_at
            score -= min(age / 3600, 0.2)
            return score

        while total > session.token_budget * 0.9 and len(session.context_window) > 1:
            candidates = [
                b for b in session.context_window
                if b.content_type not in (ContentType.GOAL, ContentType.SUBGOAL)
            ]
            if not candidates:
                break
            worst = min(candidates, key=lambda b: _eviction_score(b))
            session.context_window.remove(worst)
            total -= worst.tokens

    def to_episode_dict(self, session_id: str) -> dict:
        session = self._sessions.get(session_id)
        if not session:
            return {}
        return {
            "session_id": session.session_id,
            "task": session.task,
            "current_goal": session.current_goal,
            "subgoals": [
                {"goal_id": g.goal_id, "description": g.description, "completed": g.completed}
                for g in session.subgoals
            ],
            "context_summary": [
                {"ref": b.ref, "type": b.content_type.value, "importance": b.importance}
                for b in session.context_window
            ],
            "scratchpad": session.scratchpad[:500] if session.scratchpad else "",
        }
