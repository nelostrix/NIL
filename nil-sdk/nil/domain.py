"""
Domain — defines the problem space an agent operates in.

A Domain tells Nil:
  - What tools the agent can use
  - What it's trying to accomplish (objectives)
  - How to tell if a run was successful (success_signal)

Example:
    domain = Domain(
        name="incident-response",
        tools=[pagerduty_tool, logs_tool, deploy_tool],
        objectives=["resolve production incidents under 5 minutes"],
        success_signal=lambda run: run.resolved and run.duration < 300
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nil.tools.base import  Tool


@dataclass
class Domain:
    """
    A problem domain that an agent specializes in.

    Attributes:
        name: Human-readable identifier (used as namespace in memory store).
        tools: List of Tool instances the agent can invoke.
        objectives: Plain-language descriptions of what the agent should achieve.
        success_signal: A callable that receives a RunOutcome and returns True
            if the run should be considered successful. Used by the Evolution
            Engine to score strategies.
        metadata: Optional dict for any extra domain-specific config
            (e.g., allowed APIs, rate limits, cost budgets).
    """

    name: str
    tools: list[Tool] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)
    success_signal: Optional[Callable[..., bool]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Domain name cannot be empty")
        # Normalize name for filesystem/db usage
        self.name = self.name.strip().lower().replace(" ", "-")

    @property
    def tool_name(self) -> list[str]:
        """Return the names of all registered tools."""
        return [t.name for t in self.tools]

    def evaluate(self, outcome: Any) -> bool:
        """
        Evaluate whether a run outcome counts as success.
        Returns True if no success_signal is defined (optimistic default).
        """
        if self.success_signal is None:
            return True
        return self.success_signal(outcome)
