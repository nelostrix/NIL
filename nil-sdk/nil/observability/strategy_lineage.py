from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LineageNode:
    version: int = 0
    mutation_type: str = "unknown"
    score: float = 0.0
    parent_version: Optional[int] = None
    is_active: bool = False
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mutation_type": self.mutation_type,
            "score": self.score,
            "parent_version": self.parent_version,
            "is_active": self.is_active,
            "timestamp": self.timestamp,
        }


@dataclass
class LineageGraph:
    nodes: list[LineageNode] = field(default_factory=list)

    def add_node(self, node: LineageNode) -> None:
        self.nodes.append(node)

    def get_node(self, version: int) -> Optional[LineageNode]:
        for n in self.nodes:
            if n.version == version:
                return n
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [n.to_dict() for n in self.nodes]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LineageGraph:
        graph = cls()
        graph.nodes = [LineageNode(**n) for n in d.get("nodes", [])]
        return graph


class StrategyLineage:
    def __init__(self):
        self._strategies: list[dict[str, Any]] = []

    def set_strategies(self, strategies: list[dict[str, Any]]) -> None:
        self._strategies = list(strategies)

    def build(self, domain: str) -> LineageGraph:
        versions = self._strategies
        graph = LineageGraph()
        for v in versions:
            node = LineageNode(
                version=v.get("version", 0),
                mutation_type=v.get("mutation_type", "unknown"),
                score=v.get("score", 0),
                parent_version=v.get("parent_version"),
                is_active=v.get("is_active", False),
                timestamp=str(v.get("created_at", "")),
            )
            graph.add_node(node)
        return graph

    def render_text(self, domain: str) -> str:
        graph = self.build(domain)
        if not graph.nodes:
            return f"Strategy Lineage - Domain: {domain}\n  No version history."
        lines = [f"Strategy Lineage - Domain: {domain}", ""]
        active = [n for n in graph.nodes if n.is_active]
        for node in sorted(graph.nodes, key=lambda n: n.version):
            marker = " ✓" if node.is_active else ""
            parent = f" parent=v{node.parent_version}" if node.parent_version else " (initial)"
            lines.append(f"  v{node.version}: {node.mutation_type}{parent}{marker}")
        if active:
            lines.append(f"\n  Current: v{active[0].version}")
        return "\n".join(lines)
