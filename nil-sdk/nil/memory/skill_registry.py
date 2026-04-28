"""
SkillRegistry — manages extracted skills.

Skills are reusable patterns the agent has learned from past
experience. They're stored in memory and injected into the
system prompt so the agent can use them without re-deriving.

Phase 2 feature — stub for now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Skill:
    """
    A learned skill extracted from successful traces.

    Attributes:
        name: Short identifier (e.g., "check_db_pool_first").
        description: What this skill does in natural language.
        trigger: When to use it (condition or pattern match).
        action_sequence: The tool-call sequence this skill encodes.
        confidence: How reliably this skill leads to success (0-1).
        source_runs: Which run IDs this skill was extracted from.
    """

    name: str
    description: str = ""
    trigger: str = ""
    action_sequence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source_runs: list[str] = field(default_factory=list)
    usage_count: int = 0

    def to_prompt_fragment(self) -> str:
        """Convert to a string that can be injected into a system prompt."""
        parts = [f"Skill: {self.name}"]
        if self.description:
            parts.append(f"  When: {self.trigger or 'applicable'}")
            parts.append(f"  Do: {self.description}")
        if self.action_sequence:
            parts.append(f"  Steps: {' → '.join(self.action_sequence)}")
        return "\n".join(parts)
