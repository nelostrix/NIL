"""
Analyzer — finds patterns in execution traces.

Phase 2 module. Reads traces from memory and identifies:
  - Recurring tool sequences in successful runs
  - Common failure points
  - Performance trends over time
  - Redundant steps that waste tokens
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from nil.memory.store import PersistentMemory

logger = logging.getLogger("nil.evolution.analyzer")


class TraceAnalyzer:
    """
    Analyzes accumulated traces to find actionable patterns.

    This is the first stage of the evolution pipeline:
    raw traces → patterns → candidate skills.
    """

    def __init__(self, memory: PersistentMemory) -> None:
        self.memory = memory

    def find_winning_patterns(
        self, domain: str, min_frequency: int = 3
    ) -> list[dict[str, Any]]:
        """
        Identify tool-call sequences that consistently appear in
        successful runs but not in failures.

        Returns patterns sorted by frequency.
        """
        successes = self.memory.get_traces(domain, limit=200, success_only=True)
        all_traces = self.memory.get_traces(domain, limit=200)
        failures = [t for t in all_traces if not t.get("outcome_success")]

        # Count sequences in successes
        success_seqs = Counter()
        for t in successes:
            seq = tuple(t.get("tool_sequence", []))
            if seq:
                success_seqs[seq] += 1

        # Count sequences in failures
        failure_seqs = Counter()
        for t in failures:
            seq = tuple(t.get("tool_sequence", []))
            if seq:
                failure_seqs[seq] += 1

        # Find sequences that appear frequently in successes
        # but rarely in failures
        patterns = []
        for seq, count in success_seqs.most_common():
            if count >= min_frequency:
                fail_count = failure_seqs.get(seq, 0)
                success_rate = count / (count + fail_count)
                patterns.append(
                    {
                        "sequence": list(seq),
                        "success_count": count,
                        "failure_count": fail_count,
                        "success_rate": round(success_rate, 3),
                    }
                )

        return sorted(patterns, key=lambda p: p["success_rate"], reverse=True)

    def find_redundant_steps(self, domain: str) -> list[dict[str, Any]]:
        """
        Identify tool calls that appear in traces but don't
        contribute to success.

        A step is considered potentially redundant if:
        - It appears in both successful and failed runs at similar rates
        - Removing it from the sequence doesn't change the success pattern
        """
        traces = self.memory.get_traces(domain, limit=200)
        # Count per-tool success vs failure contribution
        tool_outcomes: dict[str, dict[str, int]] = {}

        for t in traces:
            success = t.get("outcome_success", False)
            for step in t.get("steps", []):
                name = step.get("tool_name")
                if not name:
                    continue
                if name not in tool_outcomes:
                    tool_outcomes[name] = {"success": 0, "failure": 0, "errors": 0}
                if success:
                    tool_outcomes[name]["success"] += 1
                else:
                    tool_outcomes[name]["failure"] += 1
                if step.get("error"):
                    tool_outcomes[name]["errors"] += 1

        redundant = []
        for name, counts in tool_outcomes.items():
            total = counts["success"] + counts["failure"]
            if total < 5:
                continue
            success_rate = counts["success"] / total
            # Tool is potentially redundant if it doesn't correlate
            # with success (rate close to overall baseline)
            if 0.3 < success_rate < 0.7:
                redundant.append(
                    {
                        "tool": name,
                        "total_uses": total,
                        "success_rate": round(success_rate, 3),
                        "error_rate": round(counts["errors"] / total, 3),
                    }
                )

        return redundant

    def detect_performance_trend(self, domain: str, window: int = 20) -> dict[str, Any]:
        """
        Compare recent performance against historical baseline.

        Returns a dict indicating whether the agent is improving,
        stable, or degrading.
        """
        traces = self.memory.get_traces(domain, limit=window * 2)

        if len(traces) < window:
            return {"status": "insufficient_data", "runs": len(traces)}

        recent = traces[:window]
        older = traces[window:]

        recent_rate = sum(1 for t in recent if t.get("outcome_success")) / len(recent)
        older_rate = (
            sum(1 for t in older if t.get("outcome_success")) / len(older)
            if older
            else 0
        )

        diff = recent_rate - older_rate

        if diff > 0.1:
            status = "improving"
        elif diff < -0.1:
            status = "degrading"
        else:
            status = "stable"

        return {
            "status": status,
            "recent_success_rate": round(recent_rate, 3),
            "historical_success_rate": round(older_rate, 3),
            "delta": round(diff, 3),
            "recent_window": len(recent),
            "historical_window": len(older),
        }
