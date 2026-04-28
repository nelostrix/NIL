"""
Tracer — structured experience capture for agent runs.

The Tracer records every decision, tool call, and outcome during a run.
This is NOT logging — it's structured data designed for machine
consumption by the Evolution Engine.

Every run gets its own Tracer instance. After execution, the tracer
produces a Trace object that gets persisted to the memory store.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.kernel.outcome import RunOutcome, StepRecord


@dataclass
class Trace:
    """
    A complete execution trace for one agent run.

    This is the primary data structure the Evolution Engine
    analyzes to extract skills and improve strategies.
    """

    run_id: str
    domain: str
    task: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    outcome_success: bool = False
    outcome_output: Any = None
    total_duration: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost: float = 0.0
    tool_sequence: list[str] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """serialize trace to a directory for storage."""
        return {
            "run_id": self.run_id,
            "domain": self.domain,
            "task": self.task,
            "steps": [
                {
                    "step_number": s.step_number,
                    "step_type": s.step_type,
                    "tool_name": s.tool_name,
                    "tool_input": s.tool_input,
                    "tool_output": str(s.tool_output)[:500] if s.tool_output else None,
                    "duration_ms": s.duration_ms,
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    "error": s.error,
                    "timestamp": s.timestamp,
                }
                for s in self.steps
            ],
            "outcome_success": self.outcome_success,
            "outcome_output": str(self.outcome_output)[:1000] if self.outcome_output else None,
            "total_duration": self.total_duration,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_cost": self.total_cost,
            "tool_sequence": self.tool_sequence,
            "error": self.error,
            "created_at": self.created_at,
            "matadata": self.metadata,
        }


class Tracer:
    """Records execution steps during an agent run.

    Usage:
        tracer = Tracer(run_id="abc-123", domain="debugging")
        tracer.record_llm_call(prompt="...", response="...", tokens_in=100, tokens_out=50)
        tracer.record_tool_call(name="search", input={...}, output="...", duration_ms=120)
        trace = tracer.finalize(outcome)"""

    def __init__(self, run_id: str, domain: str) -> None:
        self.run_id = run_id
        self.domain = domain
        self._steps: list[StepRecord] = []
        self._start_time = time.time()
        self._step_counter = 0

    def record_llm_call(
        self,
        prompt: Any = None,
        response: Any = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> StepRecord:
        """record an LLM API CALL."""
        self._step_counter += 1
        step = StepRecord(
            step_number=self._step_counter,
            step_type="llm_call",
            content={"prompt": str(prompt)[:500], "response": str(response)[:500]},
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            error=error,
            timestamp=time.time(),
        )
        self._steps.append(step)
        return step

    def record_tool_call(
        self,
        name: str,
        tool_input: Optional[dict[str, Any]] = None,
        tool_output: Any = None,
        duration_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> StepRecord:
        """Record a tool invocation."""
        self._step_counter += 1
        step = StepRecord(
            step_number=self._step_counter,
            step_type="tool_call",
            tool_name=name,
            tool_input=tool_input,
            tool_output=tool_output,
            duration_ms=duration_ms,
            error=error,
            timestamp=time.time(),
        )
        self._steps.append(step)
        return step

    def record_event(
        self,
        event_type: str,
        content: Any = None,
        error: Optional[str] = None,
    ) -> StepRecord:
        """record a genertic event (e.g. skill invocation, cache hit)."""
        self._step_counter += 1
        step = StepRecord(
            step_number=self._step_counter,
            step_type=event_type,
            content=content,
            error=error,
            timestamp=time.time(),
        )
        self._steps.append(step)
        return step

    def finalize(self, outcome: RunOutcome) -> Trace:
        """
        Produce the final Trace from the recorded steps + outcome.
        """
        total_duration = time.time() - self._start_time
        total_in = sum(s.tokens_in for s in self._steps)
        total_out = sum(s.tokens_out for s in self._steps)

        trace = Trace(
            run_id=self.run_id,
            domain=self.domain,
            task=outcome.task,
            steps=self._steps,
            outcome_success=outcome.success,
            outcome_output=outcome.output,
            total_duration=total_duration,
            total_tokens_in=total_in,
            total_tokens_out=total_out,
            total_cost=outcome.total_cost,
            tool_sequence=outcome.tool_sequence,
            error=outcome.error,
            metadata=outcome.metadata,
        )
        return trace

    @property
    def step_count(self) -> int:
        return len(self._steps)
