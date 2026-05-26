from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ReasoningNode:
    run_id: str = ""
    step_number: int = 0
    prompt_preview: str = ""
    response_preview: str = ""
    options: list[str] = field(default_factory=list)
    chosen: str = ""
    rationale: str = ""
    influences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_number": self.step_number,
            "prompt_preview": self.prompt_preview[:200],
            "response_preview": self.response_preview[:200],
            "options": list(self.options),
            "chosen": self.chosen,
            "rationale": self.rationale,
            "influences": list(self.influences),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReasoningNode:
        return cls(
            run_id=d.get("run_id", ""),
            step_number=d.get("step_number", 0),
            prompt_preview=d.get("prompt_preview", ""),
            response_preview=d.get("response_preview", ""),
            options=list(d.get("options", [])),
            chosen=d.get("chosen", ""),
            rationale=d.get("rationale", ""),
            influences=list(d.get("influences", [])),
        )


class ReasoningTree:
    def __init__(self):
        self._nodes: list[ReasoningNode] = []

    def capture(
        self,
        run_id: str,
        step_number: int,
        prompt: str,
        response: str,
        options: Optional[list[str]] = None,
        chosen: str = "",
        rationale: str = "",
        influences: Optional[list[str]] = None,
    ) -> ReasoningNode:
        node = ReasoningNode(
            run_id=run_id,
            step_number=step_number,
            prompt_preview=prompt[:200],
            response_preview=response[:200],
            options=options or [],
            chosen=chosen,
            rationale=rationale,
            influences=influences or [],
        )
        self._nodes.append(node)
        return node

    def get_nodes(self, run_id: str) -> list[ReasoningNode]:
        return [n for n in self._nodes if n.run_id == run_id]

    def render_text(self, run_id: str) -> str:
        nodes = self.get_nodes(run_id)
        if not nodes:
            return f"No reasoning data for run {run_id}"
        lines = [f"Reasoning Tree - Run: {run_id}", ""]
        for node in nodes:
            lines.append(f"Step {node.step_number}:")
            lines.append(f"  Chosen: {node.chosen}")
            if node.rationale:
                lines.append(f"  Rationale: {node.rationale[:200]}")
            if node.options:
                lines.append(f"  Options considered: {', '.join(node.options[:5])}")
            if node.influences:
                lines.append(f"  Influences: {', '.join(node.influences[:3])}")
            lines.append("")
        return "\n".join(lines)

    def clear(self) -> None:
        self._nodes.clear()
