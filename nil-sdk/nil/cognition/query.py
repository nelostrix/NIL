from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RetrievalQuery:
    task: str = ""
    current_tool_state: list[str] = field(default_factory=list)
    last_error: str = ""
    current_step: int = 0
    domain: str = ""
    recent_outputs: list[str] = field(default_factory=list)
    active_hypothesis: str = ""

    @property
    def composite_text(self) -> str:
        parts = [self.task]
        if self.current_tool_state:
            parts.append("tools: " + " ".join(self.current_tool_state))
        if self.last_error:
            parts.append("error: " + self.last_error)
        if self.active_hypothesis:
            parts.append("hypothesis: " + self.active_hypothesis)
        return " | ".join(parts)


class RetrievalStrategy:
    def __init__(
        self,
        name: str = "balanced",
        source_weights: dict[str, float] = None,
        top_k: int = 5,
        min_score: float = 0.0,
        prioritize_failures: bool = False,
        prioritize_trajectories: bool = False,
        prioritize_facts: bool = False,
        minimal: bool = False,
    ):
        self.name = name
        self.source_weights = source_weights or {
            "skills": 0.30,
            "trajectories": 0.25,
            "failures": 0.20,
            "constraints": 0.15,
            "facts": 0.10,
        }
        self.top_k = top_k
        self.min_score = min_score
        self.prioritize_failures = prioritize_failures
        self.prioritize_trajectories = prioritize_trajectories
        self.prioritize_facts = prioritize_facts
        self.minimal = minimal


_FAILURE_STRATEGY = RetrievalStrategy(
    name="failure_recovery",
    source_weights={"failures": 0.40, "constraints": 0.25, "trajectories": 0.20, "skills": 0.10, "facts": 0.05},
    top_k=7,
    prioritize_failures=True,
)

_EXPLORATION_STRATEGY = RetrievalStrategy(
    name="exploration",
    source_weights={"facts": 0.30, "trajectories": 0.25, "constraints": 0.20, "skills": 0.15, "failures": 0.10},
    top_k=8,
    prioritize_facts=True,
)

_MINIMAL_STRATEGY = RetrievalStrategy(
    name="minimal",
    source_weights={"skills": 0.70, "constraints": 0.30, "failures": 0.0, "trajectories": 0.0, "facts": 0.0},
    top_k=3,
    minimal=True,
)

_BALANCED_STRATEGY = RetrievalStrategy(name="balanced")


class RetrievalStrategySelector:
    def select(self, query: RetrievalQuery) -> RetrievalStrategy:
        if query.last_error:
            return _FAILURE_STRATEGY
        if query.current_step <= 2:
            return _EXPLORATION_STRATEGY
        if query.current_step > 10:
            return _MINIMAL_STRATEGY
        return _BALANCED_STRATEGY
