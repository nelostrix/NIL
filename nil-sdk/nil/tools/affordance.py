from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from nil.tools.base import Tool
from nil.tools.embeddings import ToolEmbedding, _cosine_sim


@dataclass
class ToolAffordance:
    tool_name: str = ""
    enabled_transitions: list[tuple[str, str, float]] = field(default_factory=list)
    associated_goals: list[tuple[str, float]] = field(default_factory=list)
    preconditions: list[tuple[str, float]] = field(default_factory=list)
    postconditions: list[tuple[str, float]] = field(default_factory=list)
    substitutes: list[tuple[str, float]] = field(default_factory=list)
    complements: list[tuple[str, float]] = field(default_factory=list)
    failure_contexts: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "enabled_transitions": list(self.enabled_transitions),
            "associated_goals": list(self.associated_goals),
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "substitutes": list(self.substitutes),
            "complements": list(self.complements),
            "failure_contexts": list(self.failure_contexts),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolAffordance:
        return cls(
            tool_name=d.get("tool_name", ""),
            enabled_transitions=[(a, b, c) for a, b, c in d.get("enabled_transitions", [])],
            associated_goals=[(a, b) for a, b in d.get("associated_goals", [])],
            preconditions=[(a, b) for a, b in d.get("preconditions", [])],
            postconditions=[(a, b) for a, b in d.get("postconditions", [])],
            substitutes=[(a, b) for a, b in d.get("substitutes", [])],
            complements=[(a, b) for a, b in d.get("complements", [])],
            failure_contexts=[(a, b) for a, b in d.get("failure_contexts", [])],
        )


@dataclass
class _TraceStub:
    tool_sequence: list[str] = field(default_factory=list)
    success: bool = True
    error: str = ""
    state_before: dict = field(default_factory=dict)
    state_after: dict = field(default_factory=dict)
    cost: float = 0.0
    task: str = ""


class AffordanceLearner:
    def __init__(self):
        self.all_tools: list[Tool] = []
        self._tool_embeddings: dict[str, ToolEmbedding] = {}
        self._traces: list[_TraceStub] = []

    def register_tool(self, tool: Tool, embedding: Optional[ToolEmbedding] = None) -> None:
        self.all_tools.append(tool)
        if embedding:
            self._tool_embeddings[tool.name] = embedding

    def _get_traces(self) -> list[_TraceStub]:
        return self._traces

    def learn_affordances(self, tool: Tool) -> ToolAffordance:
        traces = self._get_traces()
        tool_traces = [t for t in traces if tool.name in t.tool_sequence]

        transitions = self._extract_state_transitions(tool, tool_traces)
        goals = self._extract_associated_goals(tool, tool_traces)
        preconditions = self._extract_preconditions(tool, tool_traces)
        postconditions = self._extract_postconditions(tool, tool_traces)
        substitutes = self._discover_substitutes(tool, traces)
        complements = self._discover_complements(tool, traces)
        failure_contexts = self._extract_failure_contexts(tool, tool_traces)

        return ToolAffordance(
            tool_name=tool.name,
            enabled_transitions=transitions,
            associated_goals=goals,
            preconditions=preconditions,
            postconditions=postconditions,
            substitutes=substitutes,
            complements=complements,
            failure_contexts=failure_contexts,
        )

    def _extract_state_transitions(
        self, tool: Tool, traces: list[_TraceStub]
    ) -> list[tuple[str, str, float]]:
        transitions: dict[tuple[str, str], int] = {}
        total = 0
        for t in traces:
            before = str(t.state_before.get("status", "unknown"))
            after = str(t.state_after.get("status", "unknown"))
            key = (before, after)
            transitions[key] = transitions.get(key, 0) + 1
            total += 1
        if total == 0:
            return []
        return [(k[0], k[1], v / total) for k, v in sorted(transitions.items(), key=lambda x: -x[1])]

    def _extract_associated_goals(self, tool: Tool, traces: list[_TraceStub]) -> list[tuple[str, float]]:
        goals: dict[str, int] = {}
        total = 0
        for t in traces:
            if t.task:
                goals[t.task] = goals.get(t.task, 0) + 1
                total += 1
        if total == 0:
            return []
        return [(k, v / total) for k, v in sorted(goals.items(), key=lambda x: -x[1])]

    def _extract_preconditions(self, tool: Tool, traces: list[_TraceStub]) -> list[tuple[str, float]]:
        state_keys: dict[str, int] = {}
        total = 0
        for t in traces:
            if t.success:
                for k, v in t.state_before.items():
                    key = f"{k}={v}"
                    state_keys[key] = state_keys.get(key, 0) + 1
                total += 1
        if total == 0:
            return []
        return [(k, v / total) for k, v in sorted(state_keys.items(), key=lambda x: -x[1])[:5]]

    def _extract_postconditions(self, tool: Tool, traces: list[_TraceStub]) -> list[tuple[str, float]]:
        state_keys: dict[str, int] = {}
        total = 0
        for t in traces:
            if t.success:
                for k, v in t.state_after.items():
                    key = f"{k}={v}"
                    state_keys[key] = state_keys.get(key, 0) + 1
                total += 1
        if total == 0:
            return []
        return [(k, v / total) for k, v in sorted(state_keys.items(), key=lambda x: -x[1])[:5]]

    def _discover_substitutes(self, tool: Tool, traces: list[_TraceStub]) -> list[tuple[str, float]]:
        candidates = []
        for other in self.all_tools:
            if other.name == tool.name:
                continue
            context_sim = self._context_similarity(tool, other, traces)
            outcome_sim = self._outcome_similarity(tool, other, traces)
            transition_sim = self._transition_similarity(tool, other, traces)
            similarity = 0.4 * context_sim + 0.3 * outcome_sim + 0.3 * transition_sim
            if similarity > 0.6:
                candidates.append((other.name, similarity))
        return sorted(candidates, key=lambda x: x[1], reverse=True)

    def _discover_complements(self, tool: Tool, traces: list[_TraceStub]) -> list[tuple[str, float]]:
        complements = []
        for other in self.all_tools:
            if other.name == tool.name:
                continue
            lift = self._cooccurrence_lift(tool, other, traces)
            if lift > 1.5:
                complements.append((other.name, lift))
        return sorted(complements, key=lambda x: x[1], reverse=True)

    def _context_similarity(self, a: Tool, b: Tool, traces: list[_TraceStub]) -> float:
        emb_a = self._tool_embeddings.get(a.name)
        emb_b = self._tool_embeddings.get(b.name)
        if emb_a and emb_b:
            return emb_a.similarity(emb_b)
        return 0.0

    def _outcome_similarity(self, a: Tool, b: Tool, traces: list[_TraceStub]) -> float:
        a_traces = [t for t in traces if a.name in t.tool_sequence]
        b_traces = [t for t in traces if b.name in t.tool_sequence]
        if not a_traces or not b_traces:
            return 0.0
        a_sr = sum(1 for t in a_traces if t.success) / len(a_traces)
        b_sr = sum(1 for t in b_traces if t.success) / len(b_traces)
        return 1.0 - abs(a_sr - b_sr)

    def _transition_similarity(self, a: Tool, b: Tool, traces: list[_TraceStub]) -> float:
        a_traces = [t for t in traces if a.name in t.tool_sequence]
        b_traces = [t for t in traces if b.name in t.tool_sequence]
        a_trans = set()
        for t in a_traces:
            before = str(t.state_before.get("status", "unknown"))
            after = str(t.state_after.get("status", "unknown"))
            a_trans.add((before, after))
        b_trans = set()
        for t in b_traces:
            before = str(t.state_before.get("status", "unknown"))
            after = str(t.state_after.get("status", "unknown"))
            b_trans.add((before, after))
        if not a_trans and not b_trans:
            return 1.0
        if not a_trans or not b_trans:
            return 0.0
        intersection = a_trans & b_trans
        return len(intersection) / max(len(a_trans | b_trans), 1)

    def _cooccurrence_lift(self, a: Tool, b: Tool, traces: list[_TraceStub]) -> float:
        total = len(traces) or 1
        a_count = sum(1 for t in traces if a.name in t.tool_sequence)
        b_count = sum(1 for t in traces if b.name in t.tool_sequence)
        ab_count = sum(1 for t in traces if a.name in t.tool_sequence and b.name in t.tool_sequence)
        p_b = b_count / total
        p_b_given_a = ab_count / max(a_count, 1)
        if p_b == 0:
            return 0.0
        return p_b_given_a / p_b

    def _extract_failure_contexts(self, tool: Tool, traces: list[_TraceStub]) -> list[tuple[str, float]]:
        failures = [t for t in traces if not t.success and tool.name in t.tool_sequence]
        if not failures:
            return []
        errors: dict[str, int] = {}
        for t in failures:
            err = t.error or "unknown"
            errors[err] = errors.get(err, 0) + 1
        total = len(failures)
        return [(k, v / total) for k, v in sorted(errors.items(), key=lambda x: -x[1])]


class ToolAffordanceGraph:
    def __init__(self):
        self._nodes: dict[str, ToolAffordance] = {}
        self._edges: list[dict] = []

    def add_tool(self, affordance: ToolAffordance) -> None:
        self._nodes[affordance.tool_name] = affordance
        for sub_name, weight in affordance.substitutes:
            self._edges.append({
                "source": affordance.tool_name,
                "target": sub_name,
                "type": "substitutes_for",
                "weight": weight,
            })
        for comp_name, weight in affordance.complements:
            self._edges.append({
                "source": affordance.tool_name,
                "target": comp_name,
                "type": "complements",
                "weight": weight,
            })

    def substitutes(self, tool_name: str) -> list[tuple[str, float]]:
        return [
            (e["target"], e["weight"])
            for e in self._edges
            if e["source"] == tool_name and e["type"] == "substitutes_for"
        ]

    def downstream_tools(self, tool_name: str) -> list[tuple[str, float]]:
        return [
            (e["target"], e["weight"])
            for e in self._edges
            if e["source"] == tool_name and e["type"] == "complements"
        ]

    def tools_for_goal(self, goal: str, top_k: int = 5) -> list[tuple[str, float]]:
        scored = []
        for name, aff in self._nodes.items():
            for g, relevance in aff.associated_goals:
                if goal.lower() in g.lower():
                    scored.append((name, relevance))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {k: v.to_dict() for k, v in self._nodes.items()},
            "edges": list(self._edges),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolAffordanceGraph:
        g = cls()
        for name, aff_dict in d.get("nodes", {}).items():
            g._nodes[name] = ToolAffordance.from_dict(aff_dict)
        g._edges = list(d.get("edges", []))
        return g
