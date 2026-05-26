from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from nil.cognition.index import ScoredResult


@dataclass
class ContextAssembler:
    max_tokens: int = 2048
    source_budget: dict[str, float] = field(default_factory=lambda: {
        "skills": 0.30,
        "trajectories": 0.25,
        "failures": 0.20,
        "constraints": 0.15,
        "facts": 0.10,
    })

    def assemble(self, results: list[ScoredResult]) -> str:
        grouped: dict[str, list[ScoredResult]] = {}
        for r in results:
            src = r.source
            if src not in grouped:
                grouped[src] = []
            grouped[src].append(r)

        for src in grouped:
            grouped[src].sort(key=lambda x: x.score, reverse=True)

        sections = ["RELEVANT PAST EXPERIENCE (retrieved for this task):"]
        tokens_used = len("".join(sections))

        for source_name in ["skills", "trajectories", "failures", "constraints", "facts"]:
            items = grouped.get(source_name, [])
            if not items:
                continue

            budget_ratio = self.source_budget.get(source_name, 0.1)
            source_token_budget = int(self.max_tokens * budget_ratio)
            source_tokens = 0

            section_lines = [f"\n\xad\xad {source_name.title()} \xad\xad"]
            for item in items:
                line = self._format_item(item)
                line_tokens = len(line.split())
                if source_tokens + line_tokens > source_token_budget:
                    line = self._compact_format(item)
                    line_tokens = len(line.split())
                    if source_tokens + line_tokens > source_token_budget:
                        break
                section_lines.append(line)
                source_tokens += line_tokens

            if len(section_lines) > 1:
                sections.extend(section_lines)
                tokens_used += source_tokens

        return "\n".join(sections)

    def _format_item(self, item: ScoredResult) -> str:
        content = item.item.content or ""
        kw = ", ".join(item.item.keywords[:3]) if item.item.keywords else ""
        score = item.score
        parts = [f"  \u2022 {content[:200]}"]
        if kw:
            parts.append(f"    [{kw}]")
        parts.append(f"    (confidence: {score:.2f})")
        return "\n".join(parts)

    def _compact_format(self, item: ScoredResult) -> str:
        short = (item.item.content or "")[:80]
        return f"  \u2022 {short} (score: {item.score:.2f})"
