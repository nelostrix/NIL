"""
DriftDetector — monitors for concept drift and performance degradation.

Detects when strategies that previously worked start failing,
which indicates the domain has changed (new APIs, new data patterns,
changed requirements).

Triggers re-evolution when drift is detected.
"""

from __future__ import annotations

import logging
from typing import Any

from nil.evolution.analyzer import TraceAnalyzer
from nil.memory.store import PersistentMemory

logger = logging.getLogger("nil.evolution.drift_detector")


class DriftDetector:
    """
    Monitors agent performance for signs of concept drift.

    Concept drift occurs when:
        - Success rate drops significantly vs historical baseline
        - Previously reliable tool sequences start failing
        - Error patterns change (new types of errors appear)
    """

    def __init__(
        self,
        memory: PersistentMemory,
        alert_threshold: float = 0.15,
        window_size: int = 20,
    ) -> None:
        self.memory = memory
        self.analyzer = TraceAnalyzer(memory)
        self.alert_threshold = alert_threshold
        self.window_size = window_size

    def check(self, domain: str) -> dict[str, Any]:
        """
        Run a drift check for a domain.

        Returns:
            {
                "drift_detected": bool,
                "severity": "none" | "mild" | "significant" | "critical",
                "details": {...},
                "recommendation": str,
            }
        """
        trend = self.analyzer.detect_performance_trend(domain, window=self.window_size)

        if trend.get("status") == "insufficient_data":
            return {
                "drift_detected": False,
                "severity": "none",
                "details": trend,
                "recommendation": "Not enough data yet. Keep running.",
            }

        delta = trend.get("delta", 0)

        if delta < -0.3:
            severity = "critical"
            recommendation = (
                "Critical performance drop detected. "
                "Immediate re-evolution recommended. "
                "Consider rolling back to a previous strategy."
            )
        elif delta < -self.alert_threshold:
            severity = "significant"
            recommendation = (
                "Significant performance degradation. "
                "Evolution should be triggered to adapt."
            )
        elif delta < -0.05:
            severity = "mild"
            recommendation = (
                "Slight performance dip. Monitor closely. "
                "May self-correct or may indicate early drift."
            )
        else:
            severity = "none"
            recommendation = "Performance is stable or improving."

        return {
            "drift_detected": severity in ("significant", "critical"),
            "severity": severity,
            "details": trend,
            "recommendation": recommendation,
        }
