from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from nil.cognition.index import ScoredResult


@dataclass
class _FeedbackSignal:
    source: str
    outcome: str
    relevance_score: float
    context: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "outcome": self.outcome,
            "relevance_score": self.relevance_score,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _FeedbackSignal:
        return cls(
            source=d["source"],
            outcome=d["outcome"],
            relevance_score=d["relevance_score"],
            context=d["context"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
        )


class RetrievalFeedback:
    def __init__(self, max_signals: int = 1000):
        self.signals: list[_FeedbackSignal] = []
        self.max_signals = max_signals

    def record_success(
        self,
        query: str,
        selected_results: list[ScoredResult],
        outcome: str,
    ) -> None:
        avg_score = (
            sum(r.score for r in selected_results) / max(len(selected_results), 1)
        )
        signal = _FeedbackSignal(
            source=query,
            outcome=outcome,
            relevance_score=min(1.0, avg_score * 1.2),
            context=f"selected {len(selected_results)} items, avg score {avg_score:.2f}",
        )
        self._register(signal)

    def record_failure(
        self,
        query: str,
        selected_results: list[ScoredResult],
        error: str,
    ) -> None:
        avg_score = (
            sum(r.score for r in selected_results) / max(len(selected_results), 1)
        )
        signal = _FeedbackSignal(
            source=query,
            outcome="failure",
            relevance_score=max(0.0, avg_score * 0.5),
            context=f"error: {error}",
        )
        self._register(signal)

    def record_feedback(
        self,
        source: str,
        outcome: str,
        relevance_score: float,
        context: str,
    ) -> None:
        signal = _FeedbackSignal(
            source=source,
            outcome=outcome,
            relevance_score=relevance_score,
            context=context,
        )
        self._register(signal)

    def _register(self, signal: _FeedbackSignal) -> None:
        self.signals.append(signal)
        if len(self.signals) > self.max_signals:
            self.signals.pop(0)

    def recent(self, n: int = 10) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.signals[-n:]]

    def positive_signals(self, min_score: float = 0.5) -> list[_FeedbackSignal]:
        return [
            s for s in self.signals
            if s.relevance_score >= min_score and s.outcome == "success"
        ]

    def negative_signals(self, max_score: float = 0.3) -> list[_FeedbackSignal]:
        return [
            s for s in self.signals
            if s.relevance_score <= max_score or s.outcome == "failure"
        ]

    def clear(self) -> None:
        self.signals.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "signals": [s.to_dict() for s in self.signals],
            "max_signals": self.max_signals,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RetrievalFeedback:
        fb = cls(max_signals=d.get("max_signals", 1000))
        for s in d.get("signals", []):
            fb.signals.append(_FeedbackSignal.from_dict(s))
        return fb
