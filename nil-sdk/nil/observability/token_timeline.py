from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from nil.observability.events import EventType, ObservabilityEvent


@dataclass
class TokenStep:
    step_number: int = 0
    step_type: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "step_type": self.step_type,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost": self.cost,
            "description": self.description,
        }


@dataclass
class TokenReport:
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost: float = 0.0
    steps: list[TokenStep] = field(default_factory=list)
    breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_cost": self.total_cost,
            "steps": [s.to_dict() for s in self.steps],
            "breakdown": dict(self.breakdown),
        }


class TokenTimeline:
    INPUT_COST_PER_TOKEN = 3e-6
    OUTPUT_COST_PER_TOKEN = 15e-6

    def compute(self, events: list[ObservabilityEvent]) -> TokenReport:
        steps: list[TokenStep] = []
        total_in = 0
        total_out = 0

        for event in events:
            if event.event_type == EventType.LLM_CALL_COMPLETED:
                tokens_in = event.payload.get("tokens_in", 0)
                tokens_out = event.payload.get("tokens_out", 0)
                step = event.payload.get("step_number", 0)
                cost = self._estimate_cost(tokens_in, tokens_out)

                steps.append(TokenStep(
                    step_number=step,
                    step_type="llm",
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost=cost,
                    description=event.payload.get("description", ""),
                ))
                total_in += tokens_in
                total_out += tokens_out

            elif event.event_type == EventType.MEMORY_RETRIEVAL:
                context_tokens = event.payload.get("context_tokens", 0)
                steps.append(TokenStep(
                    step_number=event.payload.get("step_number", 0),
                    step_type="retrieval",
                    tokens_in=context_tokens,
                    tokens_out=0,
                    cost=0,
                    description=f"Retrieved: {event.payload.get('source', 'unknown')}",
                ))
                total_in += context_tokens

            elif event.event_type == EventType.DELTA_APPLIED:
                steps.append(TokenStep(
                    step_number=event.payload.get("step_number", 999),
                    step_type="evolution",
                    tokens_in=event.payload.get("tokens_in", 0),
                    tokens_out=event.payload.get("tokens_out", 0),
                    cost=0,
                    description="Evolution pipeline",
                ))

        return TokenReport(
            total_tokens_in=total_in,
            total_tokens_out=total_out,
            total_cost=self._estimate_cost(total_in, total_out),
            steps=sorted(steps, key=lambda s: s.step_number),
            breakdown={
                "llm_reasoning": sum(s.tokens_in + s.tokens_out for s in steps if s.step_type == "llm"),
                "retrieved_context": sum(s.tokens_in for s in steps if s.step_type == "retrieval"),
                "evolution": sum(s.tokens_in + s.tokens_out for s in steps if s.step_type == "evolution"),
            },
        )

    def _estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        return round(
            tokens_in * self.INPUT_COST_PER_TOKEN + tokens_out * self.OUTPUT_COST_PER_TOKEN,
            6,
        )

    def render_text(self, events: list[ObservabilityEvent]) -> str:
        report = self.compute(events)
        lines = [
            f"Token Timeline",
            f"Total tokens: {report.total_tokens_in:,} in / {report.total_tokens_out:,} out",
            f"Total cost: ${report.total_cost:.4f}",
            "",
            f"  {'Step':<6} {'Type':<12} {'In':>8} {'Out':>8} {'Cost':>10}",
            f"  {'-'*6} {'-'*12} {'-'*8} {'-'*8} {'-'*10}",
        ]
        for s in report.steps:
            lines.append(
                f"  {s.step_number:<6} {s.step_type:<12} {s.tokens_in:>8,} {s.tokens_out:>8,} ${s.cost:<8.4f}"
            )
        return "\n".join(lines)
