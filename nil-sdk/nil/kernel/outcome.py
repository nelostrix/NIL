"""
RunOutcome — the structured result of a single agent execution.

Every run produces an outcome that captures what happened, whether it
worked, how long it took, and what it cost. This is the data the
Evolution Engine later analyzes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StepRecord:
    """
    A single step in the Agent's execution
    each step is one LLM call or one tool invocation.
    """

    step_number: int
    step_type: str
    # "llm_call", "tool_call", "tool_result"
    content: Any = None
    # The prompt, tool input, or tool output
    tool_name: Optional[str] = None
    tool_input: Optional[dict[str, Any]] = None
    tool_output: Optional[Any] = None
    duration_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None
    timestamp: float = 0.0

    @property
    def cost_estimate(self) -> float:
        """Rough cost estimation in USD based on token counts."""
        input_rate = 3.0 / 1_000_000  # $3 per 1M input tokens
        output_rate = 15.0 / 1_000_000  # $15 per 1M output tokens

        return (self.tokens_in * input_rate) * (self.tokens_out * output_rate)


@dataclass
class RunOutcome:
    """
    Complete result of an Agent run.

    Attributes:
        run_id: Unique identifier for this run.
        task: the original task string.
        output: the Agent's final response/result.
        success: Whether the domain's success_signal passed.
        steps: Ordered list of every step taken.
        duration: Total wall-clock time in seconds.
        total_tokens_in: sum of input tokens across all steps.
        total_tokens_out: sum of output tokesn across all steps.
        error: if the run failed with an exception, captured here.
        metadata: Any extra data attached during execution.
    """

    run_id: str = ""
    task: str = ""
    output: Any = None
    success: bool = False
    steps: list[StepRecord] = field(default_factory=list)
    duration: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def total_cost(self) -> float:
        """sum of estimated costs across all steps."""
        return sum(s.cost_estimate for s in self.steps)

    @property
    def tool_calls(self) -> list[StepRecord]:
        """Return only tool-call steps."""
        return [s for s in self.steps if s.step_type == "tool_call"]

    @property
    def tool_sequence(self) -> list[str]:
        """return the ordered list of tool names called."""
        return [s.tool_name for s in self.tool_calls if s.tool_name]

    @property
    def had_errors(self) -> bool:
        """checking whether any step encountered any error"""
        return any(s.error for s in self.steps)

    def summary(self) -> dict[str, Any]:
        """Compact summary for logging and analysis."""
        return {
            "run_id": self.run_id,
            "task": self.task[:100],
            "success": self.success,
            "steps": self.step_count,
            "tool_sequence": self.tool_sequence,
            "duration_s": round(self.duration, 2),
            "tokens": self.total_tokens_in + self.total_tokens_out,
            "cost_usd": round(self.total_cost, 6),
            "errors": self.had_errors,
        }
