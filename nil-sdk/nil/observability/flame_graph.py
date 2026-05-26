from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from nil.observability.events import ObservabilityEvent, ObservabilityStore


@dataclass
class FlameNode:
    name: str = ""
    value: int = 0
    children: list[FlameNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "children": [c.to_dict() for c in self.children],
            "metadata": dict(self.metadata),
        }


@dataclass
class FlameGraph:
    run_id: str = ""
    root: Optional[FlameNode] = None
    total_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "root": self.root.to_dict() if self.root else None,
            "total_ms": self.total_ms,
            "metadata": dict(self.metadata),
        }


@dataclass
class _SpanData:
    name: str
    start: float
    end: float
    duration_s: float
    duration_ms: int
    status: str
    cost: float
    tokens: int
    children: list[_SpanData] = field(default_factory=list)


class FlameGraphGenerator:
    def __init__(self, store: ObservabilityStore):
        self.store = store

    def generate(self, run_id: str) -> FlameGraph:
        events = self.store.get_events(run_id=run_id)
        spans = self._build_span_tree(events)

        if not spans:
            return FlameGraph(run_id=run_id)

        total_duration = max(
            (max(s.end for s in spans) - min(s.start for s in spans)) if spans else 0,
            0.001,
        )

        root = FlameNode(
            name=f"Run: {run_id[:16]}",
            value=int(total_duration * 1000),
            children=[],
        )

        for span in spans:
            root.children.append(self._span_to_node(span))

        total_cost = sum(s.cost for s in spans)
        total_llm = sum(1 for s in spans if "llm" in s.name.lower())
        total_tools = sum(1 for s in spans if "tool" in s.name.lower())

        return FlameGraph(
            run_id=run_id,
            root=root,
            total_ms=total_duration * 1000,
            metadata={
                "total_cost": round(total_cost, 4),
                "n_spans": len(spans),
                "n_llm_calls": total_llm,
                "n_tool_calls": total_tools,
            },
        )

    def _build_span_tree(self, events: list[ObservabilityEvent]) -> list[_SpanData]:
        span_map: dict[str, dict] = {}
        for e in events:
            key = e.span_id or e.event_type.value
            if key not in span_map:
                span_map[key] = {
                    "name": e.event_type.value,
                    "start": e.timestamp,
                    "end": e.timestamp,
                    "status": "completed",
                    "cost": 0.0,
                    "tokens": 0,
                    "children": [],
                }
            info = span_map[key]
            info["start"] = min(info["start"], e.timestamp)
            info["end"] = max(info["end"], e.timestamp)
            if "error" in str(e.payload):
                info["status"] = "failed"
            info["cost"] += e.payload.get("cost", 0.0)
            info["tokens"] += e.payload.get("tokens_in", 0) + e.payload.get("tokens_out", 0)

        result = []
        for key, info in sorted(span_map.items(), key=lambda x: x[1]["start"]):
            dur = max(info["end"] - info["start"], 0.001)
            result.append(_SpanData(
                name=info["name"],
                start=info["start"],
                end=info["end"],
                duration_s=round(dur, 2),
                duration_ms=int(dur * 1000),
                status=info["status"],
                cost=info["cost"],
                tokens=info["tokens"],
            ))
        return result

    def _span_to_node(self, span: _SpanData) -> FlameNode:
        node = FlameNode(
            name=f"{span.name} ({span.duration_s:.2f}s)",
            value=span.duration_ms,
            metadata={
                "status": span.status,
                "cost": span.cost,
                "tokens": span.tokens,
            },
        )
        for child in getattr(span, "children", []):
            node.children.append(self._span_to_node(child))
        return node

    def render_text(self, run_id: str) -> str:
        graph = self.generate(run_id)
        lines = [f"Flame Graph - Run: {run_id}", f"Total: {graph.total_ms / 1000:.1f}s", ""]
        if graph.root and graph.root.children:
            for node in graph.root.children:
                pct = node.value / graph.total_ms * 100 if graph.total_ms > 0 else 0
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                lines.append(f"  {bar} {node.name} ({pct:.0f}%)")
        return "\n".join(lines)
