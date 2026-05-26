from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nil.tools.base import Tool
from nil.tools.embeddings import _text_to_embedding


@dataclass
class CompositeTool:
    name: str = ""
    tools: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    trigger: Optional[dict] = None
    avg_cost: float = 0.0
    avg_duration: float = 0.0
    success_rate: float = 0.0

    def to_tool(self) -> Tool:
        desc = f"Composite: {' → '.join(self.tools)}. Success rate: {self.success_rate:.0%}"
        return Tool(
            name=self.name,
            description=desc,
            parameters={},
            fn=self._execute,
        )

    def _execute(self, **kwargs: Any) -> Any:
        pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tools": list(self.tools),
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "trigger": self.trigger,
            "avg_cost": self.avg_cost,
            "avg_duration": self.avg_duration,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompositeTool:
        return cls(
            name=d.get("name", ""),
            tools=list(d.get("tools", [])),
            preconditions=list(d.get("preconditions", [])),
            postconditions=list(d.get("postconditions", [])),
            trigger=d.get("trigger"),
            avg_cost=d.get("avg_cost", 0.0),
            avg_duration=d.get("avg_duration", 0.0),
            success_rate=d.get("success_rate", 0.0),
        )


@dataclass
class _FrequentPattern:
    sequence: list[str]
    success_rate: float
    occurrences: int
    avg_cost: float


@dataclass
class _CompositionBenefit:
    cost_savings: float = 0.0
    step_savings: float = 0.0


class ToolCompositionSynthesizer:
    def __init__(self):
        self.tool_costs: dict[str, float] = {}
        self.tool_durations: dict[str, float] = {}

    def discover_compositions(
        self,
        traces: list[Any],
        min_occurrences: int = 5,
        min_success_rate: float = 0.75,
    ) -> list[CompositeTool]:
        tool_sequences = [t.tool_sequence for t in traces if hasattr(t, 'tool_sequence')]
        patterns = self._mine_frequent_sequences(tool_sequences, min_support=min_occurrences)

        candidates = []
        for seq_str, seq_list, occ, sr, cost in patterns:
            if sr < min_success_rate:
                continue
            benefit = self._estimate_composition_benefit(seq_list)
            if benefit.cost_savings <= 0 and benefit.step_savings <= 0:
                continue
            composite = CompositeTool(
                name=self._generate_composite_name(seq_list),
                tools=seq_list,
                success_rate=sr,
                avg_cost=cost,
                avg_duration=self._estimate_duration(seq_list),
            )
            candidates.append(composite)
        return candidates

    def synthesize_composite_tool(
        self,
        sequence: list[str],
        trace_success_rate: float = 1.0,
    ) -> CompositeTool:
        return CompositeTool(
            name=self._generate_composite_name(sequence),
            tools=sequence,
            preconditions=[f"{sequence[0]} is available"],
            postconditions=[f"{sequence[-1]} completed"],
            trigger={"tools": sequence[:2]},
            avg_cost=sum(self.tool_costs.get(t, 1.0) for t in sequence),
            avg_duration=sum(self.tool_durations.get(t, 1.0) for t in sequence),
            success_rate=trace_success_rate,
        )

    def _mine_frequent_sequences(
        self,
        tool_sequences: list[list[str]],
        min_support: int,
    ) -> list[tuple[str, list[str], int, float, float]]:
        seq_counts: dict[str, dict] = {}
        for seq in tool_sequences:
            for length in range(2, min(len(seq) + 1, 6)):
                for i in range(len(seq) - length + 1):
                    sub = seq[i:i + length]
                    key = "→".join(sub)
                    if key not in seq_counts:
                        seq_counts[key] = {"seq": sub, "count": 0, "success": 0, "cost": 0.0}
                    seq_counts[key]["count"] += 1

        results = []
        for key, info in seq_counts.items():
            if info["count"] >= min_support:
                sr = info.get("success", info["count"]) / info["count"]
                results.append((key, info["seq"], info["count"], sr, info["cost"] / info["count"]))
        return sorted(results, key=lambda x: -x[2])

    def _estimate_composition_benefit(self, sequence: list[str]) -> _CompositionBenefit:
        total_cost = sum(self.tool_costs.get(t, 1.0) for t in sequence)
        return _CompositionBenefit(
            cost_savings=total_cost * 0.3,
            step_savings=len(sequence) - 1,
        )

    def _estimate_duration(self, sequence: list[str]) -> float:
        return sum(self.tool_durations.get(t, 1.0) for t in sequence)

    def _generate_composite_name(self, sequence: list[str]) -> str:
        if len(sequence) <= 2:
            return "_".join(sequence)
        return f"{sequence[0]}_thru_{sequence[-1]}"

    def _generate_name(self, sequence: list[str]) -> str:
        return self._generate_composite_name(sequence)

    def _infer_trigger_conditions(
        self, sequence: list[str], traces: list[Any]
    ) -> dict:
        return {"tools": sequence[:2]}

    def _compose_preconditions(self, sequence: list[str], traces: list[Any]) -> list[str]:
        return [f"{sequence[0]} is available"]

    def _compose_postconditions(self, sequence: list[str], traces: list[Any]) -> list[str]:
        return [f"{sequence[-1]} completed"]

    def _sequence_success_rate(self, sequence: list[str], traces: list[Any]) -> float:
        total = 0
        success = 0
        for t in traces:
            seq = getattr(t, 'tool_sequence', [])
            if len(seq) >= len(sequence):
                for i in range(len(seq) - len(sequence) + 1):
                    if seq[i:i + len(sequence)] == sequence:
                        total += 1
                        if getattr(t, 'success', False):
                            success += 1
                        break
        return success / max(total, 1)

    def _derive_composite_affordance(
        self, sequence: list[str], traces: list[Any]
    ) -> dict:
        return {"tools": sequence, "transitions": []}
