from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.tools.base import Tool
from nil.tools.embeddings import ToolEmbedding, _text_to_embedding, _cosine_sim
from nil.tools.affordance import ToolAffordance


@dataclass
class ScoredTool:
    tool_name: str = ""
    score: float = 0.0
    similarity: float = 0.0
    affordance_score: float = 0.0
    exploration_bonus: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "score": self.score,
            "similarity": self.similarity,
            "affordance_score": self.affordance_score,
            "exploration_bonus": self.exploration_bonus,
        }


class LatentToolSelector:
    def __init__(self):
        self.tool_index: dict[str, ToolEmbedding] = {}
        self.affordances: dict[str, ToolAffordance] = {}
        self._visit_counts: dict[str, int] = {}

    def add_tool_embedding(self, embedding: ToolEmbedding) -> None:
        self.tool_index[embedding.tool_name] = embedding

    def add_affordance(self, affordance: ToolAffordance) -> None:
        self.affordances[affordance.tool_name] = affordance

    def select_tools(
        self,
        context_text: str,
        current_tool_state: Optional[list[str]] = None,
        top_k: int = 3,
    ) -> list[ScoredTool]:
        query_emb = self._encode_context(context_text, current_tool_state or [])

        candidates = []
        for tool_name, emb in self.tool_index.items():
            sim = _cosine_sim(query_emb, emb.full_embedding)
            candidates.append((tool_name, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        gated = []
        for tool_name, similarity in candidates[:top_k * 2]:
            affordance = self.affordances.get(tool_name)
            if affordance and not self._preconditions_met(affordance, context_text):
                continue
            gated.append((tool_name, similarity))

        ranked = []
        for tool_name, similarity in gated[:top_k]:
            affordance_score = self._compute_affordance_match(tool_name, context_text)
            exploration_bonus = self._get_exploration_bonus(tool_name)
            total_score = 0.5 * similarity + 0.3 * affordance_score + 0.2 * exploration_bonus
            ranked.append(ScoredTool(
                tool_name=tool_name,
                score=total_score,
                similarity=similarity,
                affordance_score=affordance_score,
                exploration_bonus=exploration_bonus,
            ))

        ranked.sort(key=lambda x: x.score, reverse=True)
        for st in ranked[:top_k]:
            self._visit_counts[st.tool_name] = self._visit_counts.get(st.tool_name, 0) + 1
        return ranked[:top_k]

    def record_selection(self, tool_name: str) -> None:
        self._visit_counts[tool_name] = self._visit_counts.get(tool_name, 0) + 1

    def _encode_context(self, text: str, tool_state: list[str]) -> list[float]:
        parts = [text] + tool_state
        combined = " ".join(parts)
        return _text_to_embedding(combined)

    def _preconditions_met(self, affordance: ToolAffordance, context: str) -> bool:
        for precondition, confidence in affordance.preconditions:
            if confidence > 0.8 and precondition.split("=")[0] not in context:
                return False
        return True

    def _compute_affordance_match(self, tool_name: str, context: str) -> float:
        affordance = self.affordances.get(tool_name)
        if not affordance:
            return 0.0
        goals = affordance.associated_goals
        if not goals:
            return 0.0
        max_match = 0.0
        for goal, relevance in goals:
            if goal.lower() in context.lower():
                max_match = max(max_match, relevance)
        return max_match

    def _get_exploration_bonus(self, tool_name: str) -> float:
        visits = self._visit_counts.get(tool_name, 0)
        return 1.0 / (1.0 + math.sqrt(visits))


class LatentAwarePromptBuilder:
    def __init__(self, selector: Optional[LatentToolSelector] = None):
        self.selector = selector or LatentToolSelector()
        self.tool_manager: dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        self.tool_manager[tool.name] = tool

    def build_tool_section(
        self,
        context_text: str,
        top_k: int = 5,
    ) -> str:
        selected = self.selector.select_tools(context_text, top_k=top_k)

        lines = ["Available tools (priority order):"]
        for st in selected:
            tool = self.tool_manager.get(st.tool_name)
            if tool:
                desc = tool.description[:100]
                lines.append(f"  [{st.score:.2f}] {tool.name}: {desc}")
            else:
                lines.append(f"  [{st.score:.2f}] {st.tool_name}")

        return "\n".join(lines) if len(lines) > 1 else "No tools available."
