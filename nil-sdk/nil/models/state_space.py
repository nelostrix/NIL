from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("nil.models.state_space")

INVESTIGATION_STAGES = [
    "not_started", "gathering_data", "analyzing",
    "diagnosed", "fixing", "verifying", "resolved",
]

StateKey = str


@dataclass
class EnvironmentState:
    service_health: str = "unknown"
    incident_count: int = 0
    last_deploy_status: str = "unknown"
    investigation_stage: str = "not_started"
    current_hypothesis: str = ""
    hypothesis_confidence: float = 0.0
    tools_used: list[str] = field(default_factory=list)
    errors_encountered: list[str] = field(default_factory=list)
    steps_taken: int = 0
    budget_remaining: float = 1.0
    uncertainty_level: float = 1.0
    unobserved_factors: list[str] = field(default_factory=list)
    is_terminal_success: bool = False
    is_terminal_failure: bool = False

    def to_key(self) -> StateKey:
        return json.dumps({
            "health": self.service_health,
            "stage": self.investigation_stage,
            "hypothesis": self.current_hypothesis[:30] if self.current_hypothesis else "",
            "errors": sorted(self.errors_encountered)[:3],
        }, sort_keys=True)

    def to_dict(self) -> dict:
        return {
            "service_health": self.service_health,
            "incident_count": self.incident_count,
            "last_deploy_status": self.last_deploy_status,
            "investigation_stage": self.investigation_stage,
            "current_hypothesis": self.current_hypothesis,
            "hypothesis_confidence": self.hypothesis_confidence,
            "tools_used": list(self.tools_used),
            "errors_encountered": list(self.errors_encountered),
            "steps_taken": self.steps_taken,
            "budget_remaining": self.budget_remaining,
            "uncertainty_level": self.uncertainty_level,
            "unobserved_factors": list(self.unobserved_factors),
            "is_terminal_success": self.is_terminal_success,
            "is_terminal_failure": self.is_terminal_failure,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EnvironmentState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_trace(cls, trace_dict: dict) -> EnvironmentState:
        steps = trace_dict.get("steps", [])
        errors = []
        tools = []
        for s in steps:
            if isinstance(s, dict):
                if s.get("step_type") == "tool_call" and s.get("tool_name"):
                    tools.append(s["tool_name"])
                if s.get("error"):
                    errors.append(str(s["error"])[:60])

        success = trace_dict.get("outcome_success", False)
        health = "healthy" if success else "degraded"

        stage = "resolved" if success else "fixing"

        task = trace_dict.get("task", "")
        hypothesis = task[:40] if task else ""

        return cls(
            service_health=health,
            investigation_stage=stage,
            tools_used=tools,
            errors_encountered=errors,
            steps_taken=len(steps),
            is_terminal_success=bool(success),
            is_terminal_failure=trace_dict.get("error") is not None,
            current_hypothesis=hypothesis,
        )


@dataclass
class StateSpace:
    states: dict[StateKey, int] = field(default_factory=dict)
    state_id_to_key: dict[int, StateKey] = field(default_factory=dict)
    next_id: int = 0
    dimensions: list[str] = field(default_factory=lambda: [
        "service_health", "investigation_stage", "errors",
    ])

    @property
    def size(self) -> int:
        return len(self.states)

    def get_or_create(self, state_key: StateKey) -> int:
        if state_key in self.states:
            return self.states[state_key]
        sid = self.next_id
        self.states[state_key] = sid
        self.state_id_to_key[sid] = state_key
        self.next_id += 1
        return sid

    def get_id(self, state_key: StateKey) -> Optional[int]:
        return self.states.get(state_key)

    def get_key(self, state_id: int) -> Optional[StateKey]:
        return self.state_id_to_key.get(state_id)

    def add_state(self, state: EnvironmentState) -> int:
        return self.get_or_create(state.to_key())

    def all_keys(self) -> list[StateKey]:
        return list(self.states.keys())

    def all_ids(self) -> list[int]:
        return list(self.state_id_to_key.keys())

    def to_dict(self) -> dict:
        return {
            "states": {k: v for k, v in self.states.items()},
            "dimensions": self.dimensions,
            "next_id": self.next_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StateSpace:
        ss = cls(dimensions=d.get("dimensions", []), next_id=d.get("next_id", 0))
        ss.states = {k: v for k, v in d.get("states", {}).items()}
        ss.state_id_to_key = {v: k for k, v in ss.states.items()}
        return ss


class StateSpaceLearner:
    def __init__(self, max_states: int = 100):
        self.max_states = max_states

    def discover_states(
        self,
        traces: list[dict],
        existing_space: Optional[StateSpace] = None,
    ) -> StateSpace:
        space = existing_space or StateSpace()
        for trace in traces:
            state = EnvironmentState.from_trace(trace)
            key = state.to_key()
            if key not in space.states and space.size < self.max_states:
                space.get_or_create(key)
            transitions = self._extract_transition_states(trace)
            for s in transitions:
                key = s.to_key()
                if key not in space.states and space.size < self.max_states:
                    space.get_or_create(key)
        return space

    def _extract_transition_states(self, trace: dict) -> list[EnvironmentState]:
        steps = trace.get("steps", [])
        states = []
        current = EnvironmentState.from_trace(trace)

        for i, step in enumerate(steps):
            if isinstance(step, dict):
                next_state = EnvironmentState(
                    service_health=current.service_health,
                    investigation_stage=self._infer_stage(step, i, len(steps)),
                    steps_taken=i + 1,
                    tools_used=current.tools_used + (
                        [step.get("tool_name", "")] if step.get("tool_name") else []
                    ),
                    errors_encountered=current.errors_encountered + (
                        [str(step.get("error", ""))[:60]] if step.get("error") else []
                    ),
                )
                states.append(next_state)
                current = next_state
        return states

    def _infer_stage(self, step: dict, idx: int, total: int) -> str:
        step_type = step.get("step_type", "")
        error = step.get("error")
        if error:
            return "fixing"
        if step_type == "tool_call":
            if idx < total * 0.3:
                return "gathering_data"
            elif idx < total * 0.6:
                return "analyzing"
            else:
                return "fixing"
        return "analyzing"

    def build_transition_counts(
        self, traces: list[dict], space: StateSpace
    ) -> dict[tuple[int, str], Counter]:
        counts: dict[tuple[int, str], Counter] = defaultdict(Counter)

        for trace in traces:
            states = self._extract_transition_states(trace)
            state_ids = [space.get_or_create(s.to_key()) for s in states]
            steps = trace.get("steps", [])

            for i in range(min(len(state_ids) - 1, len(steps))):
                if i < len(steps):
                    step = steps[i]
                    action = step.get("tool_name", "") or step.get("step_type", "unknown")
                    s_from = state_ids[i]
                    s_to = state_ids[i + 1]
                    counts[(s_from, action)][s_to] += 1

            outcome = trace.get("outcome", trace)
            if outcome.get("success") or outcome.get("outcome_success"):
                counts[(state_ids[-1], "_success")]["_terminal"] += 1
            elif outcome.get("error"):
                counts[(state_ids[-1], "_failure")]["_terminal"] += 1

        return dict(counts)


def _sid() -> str:
    return uuid.uuid4().hex[:8]
