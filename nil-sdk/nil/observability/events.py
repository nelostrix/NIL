from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL_COMPLETED = "llm_call_completed"
    LLM_CALL_FAILED = "llm_call_failed"
    LLM_TOKEN_USAGE = "llm_token_usage"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    MEMORY_RETRIEVAL = "memory_retrieval"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    RETRIEVAL_QUERY = "retrieval_query"
    RETRIEVAL_RESULT = "retrieval_result"
    EVOLUTION_TRIGGERED = "evolution_triggered"
    ANALYSIS_COMPLETED = "analysis_completed"
    SKILL_EXTRACTED = "skill_extracted"
    GRADIENT_COMPUTED = "gradient_computed"
    DELTA_APPLIED = "delta_applied"
    STRATEGY_PROMOTED = "strategy_promoted"
    DRIFT_DETECTED = "drift_detected"
    RECOVERY_TRIGGERED = "recovery_triggered"


@dataclass
class ObservabilityEvent:
    event_id: str = ""
    event_type: EventType = EventType.RUN_STARTED
    timestamp: float = 0.0
    agent_id: str = ""
    domain: str = ""
    run_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    parent_event_id: Optional[str] = None
    span_id: str = ""
    trace_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = uuid.uuid4().hex[:16]
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.trace_id:
            self.trace_id = self.run_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "domain": self.domain,
            "run_id": self.run_id,
            "payload": dict(self.payload),
            "parent_event_id": self.parent_event_id,
            "span_id": self.span_id,
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ObservabilityEvent:
        return cls(
            event_id=d.get("event_id", ""),
            event_type=EventType(d.get("event_type", "run_started")),
            timestamp=d.get("timestamp", 0.0),
            agent_id=d.get("agent_id", ""),
            domain=d.get("domain", ""),
            run_id=d.get("run_id", ""),
            payload=d.get("payload", {}),
            parent_event_id=d.get("parent_event_id"),
            span_id=d.get("span_id", ""),
            trace_id=d.get("trace_id", ""),
        )


class ObservabilityStore:
    def __init__(self):
        self._events: list[ObservabilityEvent] = []

    def write(self, event: ObservabilityEvent) -> None:
        self._events.append(event)

    def get_events(
        self,
        run_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        agent_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> list[ObservabilityEvent]:
        results = list(self._events)
        if run_id:
            results = [e for e in results if e.run_id == run_id]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if agent_id:
            results = [e for e in results if e.agent_id == agent_id]
        if domain:
            results = [e for e in results if e.domain == domain]
        return results

    def clear(self) -> None:
        self._events.clear()

    def size(self) -> int:
        return len(self._events)

    def to_dict(self) -> dict[str, Any]:
        return {"events": [e.to_dict() for e in self._events]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ObservabilityStore:
        store = cls()
        store._events = [ObservabilityEvent.from_dict(e) for e in d.get("events", [])]
        return store


class EventRecorder:
    def __init__(self, store: Optional[ObservabilityStore] = None):
        self.store = store or ObservabilityStore()
        self.agent_id: str = ""
        self.domain: str = ""
        self._current_span_stack: list[str] = []

    def record(
        self,
        event_type: EventType,
        run_id: str,
        payload: Optional[dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> ObservabilityEvent:
        event = ObservabilityEvent(
            event_type=event_type,
            agent_id=self.agent_id,
            domain=self.domain,
            run_id=run_id,
            payload=payload or {},
            parent_event_id=parent_id,
            span_id=self._current_span_stack[-1] if self._current_span_stack else "",
            trace_id=run_id,
        )
        self.store.write(event)
        return event

    def start_span(self, span_name: str) -> str:
        span_id = f"{span_name}-{uuid.uuid4().hex[:8]}"
        self._current_span_stack.append(span_id)
        return span_id

    def end_span(self) -> None:
        if self._current_span_stack:
            self._current_span_stack.pop()

    def to_dict(self) -> dict[str, Any]:
        return {
            "store": self.store.to_dict(),
            "agent_id": self.agent_id,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EventRecorder:
        recorder = cls(store=ObservabilityStore.from_dict(d.get("store", {"events": []})))
        recorder.agent_id = d.get("agent_id", "")
        recorder.domain = d.get("domain", "")
        return recorder
