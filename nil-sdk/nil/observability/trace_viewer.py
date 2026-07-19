from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from nil.observability.events import ObservabilityEvent, ObservabilityStore


@dataclass
class TimelineRow:
    span_name: str = ""
    depth: int = 0
    start_offset: float = 0.0
    duration_pct: float = 0.0
    duration_s: float = 0.0
    status: str = ""
    reasoning: str = ""
    tokens_used: int = 0
    cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_name": self.span_name,
            "depth": self.depth,
            "start_offset": self.start_offset,
            "duration_pct": self.duration_pct,
            "duration_s": self.duration_s,
            "status": self.status,
            "reasoning": self.reasoning,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
        }


@dataclass
class TimelineView:
    run_id: str = ""
    total_duration: float = 0.0
    total_cost: float = 0.0
    rows: list[TimelineRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_duration": self.total_duration,
            "total_cost": self.total_cost,
            "rows": [r.to_dict() for r in self.rows],
        }


@dataclass
class _Span:
    name: str
    depth: int
    start: float
    end: float
    status: str
    reasoning: str
    tokens: int
    cost: float
    children: list[_Span] = field(default_factory=list)


class TraceViewer:
    def __init__(self, store: ObservabilityStore):
        self.store = store

    def render_timeline(self, run_id: str) -> TimelineView:
        events = self.store.get_events(run_id=run_id)
        if not events:
            return TimelineView(run_id=run_id)

        spans = self._build_spans(events)
        start_time = min(e.timestamp for e in events)
        max_time = max(e.timestamp for e in events)
        total_duration = max(max_time - start_time, 0.001)

        rows = []
        for span in spans:
            row = TimelineRow(
                span_name=span.name,
                depth=span.depth,
                start_offset=(span.start - start_time) / total_duration * 100,
                duration_pct=(span.end - span.start) / total_duration * 100,
                duration_s=round(span.end - span.start, 2),
                status=span.status,
                reasoning=span.reasoning,
                tokens_used=span.tokens,
                cost=span.cost,
            )
            rows.append(row)

            for child in span.children:
                child_row = TimelineRow(
                    span_name=child.name,
                    depth=child.depth,
                    start_offset=(child.start - start_time) / total_duration * 100,
                    duration_pct=(child.end - child.start) / total_duration * 100,
                    duration_s=round(child.end - child.start, 2),
                    status=child.status,
                    reasoning=child.reasoning,
                    tokens_used=child.tokens,
                    cost=child.cost,
                )
                rows.append(child_row)

        return TimelineView(
            run_id=run_id,
            total_duration=round(total_duration, 2),
            total_cost=sum(r.cost for r in rows),
            rows=rows,
        )

    def _build_spans(self, events: list[ObservabilityEvent]) -> list[_Span]:
        unique_spans: dict[str, dict] = {}
        for e in events:
            span_key = e.span_id or e.event_type.value
            if span_key not in unique_spans:
                unique_spans[span_key] = {
                    "name": e.event_type.value,
                    "start": e.timestamp,
                    "end": e.timestamp,
                    "status": "completed",
                    "reasoning": "",
                    "tokens": 0,
                    "cost": 0.0,
                    "depth": 0,
                    "count": 0,
                }
            info = unique_spans[span_key]
            info["start"] = min(info["start"], e.timestamp)
            info["end"] = max(info["end"], e.timestamp)
            info["count"] += 1
            if "error" in str(e.payload):
                info["status"] = "failed"
            info["tokens"] += e.payload.get("tokens_in", 0) + e.payload.get(
                "tokens_out", 0
            )
            info["cost"] += e.payload.get("cost", 0.0)
            if "reasoning" in e.payload:
                info["reasoning"] = str(e.payload.get("reasoning", ""))[:200]

        spans = []
        for key, info in sorted(unique_spans.items(), key=lambda x: x[1]["start"]):
            spans.append(
                _Span(
                    name=info["name"],
                    depth=info["depth"],
                    start=info["start"],
                    end=info["end"],
                    status=info["status"],
                    reasoning=info["reasoning"],
                    tokens=info["tokens"],
                    cost=info["cost"],
                )
            )
        return spans
