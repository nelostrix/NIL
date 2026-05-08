"""
SkillRegistry — manages extracted skills.

Skills are reusable patterns the agent has learned from past
experience. They're stored in memory and injected into the
system prompt so the agent can use them without re-deriving.

Phase 2 feature — stub for now.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SkillStatus(str, Enum):
    """Lifecycle status of a skill."""

    CANDIDATE = "candidate"  # Just extracted, not yet validated
    ACTIVE = "active"  # Validated and in use
    DEPRECATED = "deprecated"  # Was active, no longer effective
    SUPERSEDED = "superseded"  # Replaced by a better skill


class SkillOrigin(str, Enum):
    """How the skill was created."""

    PATTERN_EXTRACTION = "pattern_extraction"  # From trace patterns
    GRADIENT_INJECTION = "gradient_injection"  # From textual gradient engine
    MANUAL = "manual"  # User-defined
    COMPRESSED = "compressed"  # From chain compression


@dataclass
class TriggerCondition:
    """
    When a skill should activate.

    Triggers can be based on:
      - Task keywords: words in the task description
      - Error patterns: specific errors that appeared
      - Tool state: which tools have already been called
      - Domain signals: any domain-specific conditions
    """

    keywords: list[str] = field(default_factory=list)
    error_patterns: list[str] = field(default_factory=list)
    after_tools: list[str] = field(default_factory=list)
    not_after_tools: list[str] = field(default_factory=list)
    min_step: int = 0
    description: str = ""

    def matches(
        self,
        task: str = "",
        errors_seen: list[str] = None,
        tools_called: list[str] = None,
        current_step: int = 0,
    ) -> float:
        """Score how well current context matches this trigger."""
        if not any(
            [self.keywords, self.error_patterns, self.after_tools, self.description]
        ):
            return 0.5

        scores = []
        errors_seen = errors_seen or []
        tools_called = tools_called or []

        if self.keywords:
            task_lower = task.lower()
            matched = sum(1 for kw in self.keywords if kw.lower() in task_lower)
            scores.append(matched / len(self.keywords) if self.keywords else 0)

        if self.error_patterns:
            errors_str = " ".join(errors_seen).lower()
            matched = sum(1 for ep in self.error_patterns if ep.lower() in errors_str)
            scores.append(matched / len(self.error_patterns))

        if self.after_tools:
            tools_set = set(tools_called)
            matched = sum(1 for t in self.after_tools if t in tools_set)
            scores.append(matched / len(self.after_tools))

        if self.not_after_tools:
            tools_set = set(tools_called)
            blocked = any(t in tools_set for t in self.not_after_tools)
            if blocked:
                return 0.0

        if self.min_step > 0 and current_step < self.min_step:
            return 0.0

        return sum(scores) / len(scores) if scores else 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords": self.keywords,
            "error_patterns": self.error_patterns,
            "after_tools": self.after_tools,
            "not_after_tools": self.not_after_tools,
            "min_step": self.min_step,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TriggerCondition:
        return cls(
            keywords=d.get("keywords", []),
            error_patterns=d.get("error_patterns", []),
            after_tools=d.get("after_tools", []),
            not_after_tools=d.get("not_after_tools", []),
            min_step=d.get("min_step", 0),
            description=d.get("description", ""),
        )


@dataclass
class Skill:
    """
    A learned, compressed unit of agent competence.

    This is what gets injected into the agent's system prompt.
    """

    name: str
    description: str = ""
    trigger: TriggerCondition = field(default_factory=TriggerCondition)
    action_sequence: list[str] = field(default_factory=list)
    compressed_instruction: str = ""
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source_runs: list[str] = field(default_factory=list)
    usage_count: int = 0
    success_when_used: int = 0
    failure_when_used: int = 0
    avg_cost: float = 0.0
    avg_duration: float = 0.0
    status: SkillStatus = SkillStatus.CANDIDATE
    origin: SkillOrigin = SkillOrigin.PATTERN_EXTRACTION
    version: int = 1
    parent_skill: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    @property
    def usage_success_rate(self) -> float:
        total = self.success_when_used + self.failure_when_used
        return self.success_when_used / total if total > 0 else 0.0

    @property
    def is_effective(self) -> bool:
        total = self.success_when_used + self.failure_when_used
        return total >= 3 and self.usage_success_rate > 0.7

    @property
    def chain_length(self) -> int:
        return len(self.action_sequence)

    @property
    def compression_ratio(self) -> float:
        if not self.compressed_instruction:
            return 1.0
        return max(1.0, self.chain_length)

    def to_prompt_fragment(self) -> str:
        """Convert to a string for injection into the system prompt."""
        parts = [f"SKILL: {self.name}"]

        if self.trigger.description:
            parts.append(f"  WHEN: {self.trigger.description}")
        elif self.trigger.keywords:
            parts.append(f"  WHEN: Task involves {', '.join(self.trigger.keywords)}")

        if self.preconditions:
            parts.append(f"  REQUIRES: {'; '.join(self.preconditions)}")

        if self.compressed_instruction:
            parts.append(f"  DO: {self.compressed_instruction}")
        elif self.description:
            parts.append(f"  DO: {self.description}")

        if self.action_sequence:
            parts.append(f"  STEPS: {' → '.join(self.action_sequence)}")

        if self.postconditions:
            parts.append(f"  EXPECT: {'; '.join(self.postconditions)}")

        if self.confidence > 0:
            parts.append(f"  CONFIDENCE: {self.confidence:.0%} ({self.usage_count} uses)")

        return "\n".join(parts)

    def to_compact_prompt(self) -> str:
        """Shorter version for when prompt space is limited."""
        trigger = self.trigger.description or ""
        action = (
            self.compressed_instruction
            or self.description
            or " → ".join(self.action_sequence)
        )
        conf = f" [{self.confidence:.0%}]" if self.confidence > 0 else ""
        if trigger:
            return f"• {self.name}: When {trigger} → {action}{conf}"
        return f"• {self.name}: {action}{conf}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger.to_dict(),
            "action_sequence": self.action_sequence,
            "compressed_instruction": self.compressed_instruction,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "confidence": self.confidence,
            "source_runs": self.source_runs,
            "usage_count": self.usage_count,
            "success_when_used": self.success_when_used,
            "failure_when_used": self.failure_when_used,
            "avg_cost": self.avg_cost,
            "avg_duration": self.avg_duration,
            "status": self.status.value,
            "origin": self.origin.value,
            "version": self.version,
            "parent_skill": self.parent_skill,
            "created_at": self.created_at,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Skill:
        trigger = d.get("trigger", {})
        if isinstance(trigger, dict):
            trigger = TriggerCondition.from_dict(trigger)
        elif isinstance(trigger, str):
            trigger = TriggerCondition(description=trigger)
        else:
            trigger = TriggerCondition()

        return cls(
            name=d["name"],
            description=d.get("description", ""),
            trigger=trigger,
            action_sequence=d.get("action_sequence", []),
            compressed_instruction=d.get("compressed_instruction", ""),
            preconditions=d.get("preconditions", []),
            postconditions=d.get("postconditions", []),
            confidence=d.get("confidence", 0.0),
            source_runs=d.get("source_runs", []),
            usage_count=d.get("usage_count", 0),
            success_when_used=d.get("success_when_used", 0),
            failure_when_used=d.get("failure_when_used", 0),
            avg_cost=d.get("avg_cost", 0.0),
            avg_duration=d.get("avg_duration", 0.0),
            status=SkillStatus(d.get("status", "candidate")),
            origin=SkillOrigin(d.get("origin", "pattern_extraction")),
            version=d.get("version", 1),
            parent_skill=d.get("parent_skill"),
            created_at=d.get("created_at", time.time()),
            tags=d.get("tags", []),
        )