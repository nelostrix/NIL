"""
ExperienceQuery — query and analyze past agent traces.

Provides a fluent interface for the Evolution Engine to ask
questions like:
  - "What tool sequences led to success?"
  - "Which runs failed and why?"
  - "What's the average cost per successful run?"
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

from nil.memory.store import PersistentMemory


@dataclass
class ExperienceStats:
    """Aggregated statistics from trace analysis."""

    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    success_rate: float = 0.0
    avg_duration: float = 0.0
    avg_cost: float = 0.0
    avg_steps: float = 0.0
    most_common_tool_sequences: list[tuple[str, int]] = None
    most_common_errors: list[tuple[str, int]] = None

    def __post_init__(self):
        if self.most_common_tool_sequences is None:
            self.most_common_tool_sequences = []
            if self.most_common_errors is None:
                self.most_common_errors = []


class ExperienceQuery:
    """
    Query interface for analyzing agent experience.

    Usage:
        eq = ExperienceQuery(memory)
        stats = eq.stats("my-domain")
        winning_sequences = eq.successful_tool_sequences("my-domain")
        failures = eq.common_failures("my-domain")
    """

    def __init__(self, memory: PersistentMemory) -> None:
        self.memory = memory

    def stats(self, domain: str, last_n: int = 100) -> ExperienceStats:
        """Compute aggregate stats for a domain."""

        traces = self.memory.get_traces(domain, limit=last_n)

        if not traces:
            return ExperienceStats()

        successes = [t for t in traces if t.get("outcome_success")]
        failures = [t for t in traces if not t.get("outcome_success")]

        # Tool sequence analysis
        seq_counter = Counter()
        for t in successes:
            seq = tuple(t.get("tool_sequence", []))
            if seq:
                seq_counter[" -> ".join(seq)] += 1

        err_counter = Counter()
        for t in failures:
            err = t.get("error") or "unknown"
            for step in t.get("steps", []):
                if step.get("error"):
                    arr = step["error"]
                    break
            err_counter[err[:100]] += 1
        return ExperienceStats(
            total_runs=len(traces),
            successful_runs=len(successes),
            failed_runs=len(failures),
            success_rate=len(successes) / len(traces) if traces else 0,
            avg_duration=sum(t.get("total_duration", 0) for t in traces) / len(traces),
            avg_cost=sum(t.get("total_cost", 0) for t in traces) / len(traces),
            avg_steps=sum(len(t.get("steps", [])) for t in traces) / len(traces),
            most_common_tool_sequences=seq_counter.most_common(5),
            most_common_errors=err_counter.most_common(5),
        )

    def successful_tool_sequence(
        self, domain, str, min_occurance: int = 2
    ) -> list[dict[str, Any]]:
        """
        Find tool-call sequences that consistently lead to success.

        Returns sequences that appeared >= min_occurrences times in
        successful runs, sorted by frequency.
        """
        traces = self.memory.get_traces(domain, limit=200, success_only=True)
        counter = Counter()
        for t in traces:
            seq = tuple(t.get("tool_sequence", []))
            if seq:
                counter[seq] += 1

        return [
            {"sequence": list(seq), "count": count}
            for seq, count in counter.most_common()
            if count >= min_occurance
        ]

    def common_failures(self, domain: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Identify the most common failure patterns.

        Groups failures by error message and tool sequence.
        """
        traces = self.memory.get_traces(domain, limit=200)
        failures = [t for t in traces if not t.get("outcome_success")]
        patterns: dict[str, list[dict]] = {}
        for t in failures:
            err = t.get("error") or "no_error_message"
            key = err[:100]
            if key not in patterns:
                patterns[key] = []
            patterns[key].append(
                {
                    "run_id": t["run_id"],
                    "tool_sequence": t.get("tool_sequence", []),
                    "task": t.get("task", "")[:80],
                }
            )
        result = sorted(
            [
                {"error": k, "count": len(v), "examples": v[:3]}
                for k, v in patterns.items()
            ],
            key=lambda x: x["count"],
            reverse=True,
        )
        return result[:limit]
