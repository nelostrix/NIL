from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class _HtmlSection:
    title: str = ""
    content: str = ""


class MemoryBrowser:
    def __init__(self):
        self.skills: list[dict[str, Any]] = []
        self.strategies: list[dict[str, Any]] = []
        self.facts: list[dict[str, Any]] = []

    def set_skills(self, skills: list[dict[str, Any]]) -> None:
        self.skills = list(skills)

    def set_strategies(self, strategies: list[dict[str, Any]]) -> None:
        self.strategies = list(strategies)

    def set_facts(self, facts: list[dict[str, Any]]) -> None:
        self.facts = list(facts)

    def render_text(self, domain: str) -> str:
        sections = [
            self._build_skills_section(domain),
            self._build_strategy_section(domain),
            self._build_facts_section(domain),
        ]
        return "\n\n".join(sections)

    def _build_skills_section(self, domain: str) -> str:
        if not self.skills:
            return f"Skills (0) - Domain: {domain}\n  No skills stored."
        active = [s for s in self.skills if s.get("status") != "deprecated"]
        deprecated = [s for s in self.skills if s.get("status") == "deprecated"]
        lines = [f"Skills ({len(active)} active, {len(deprecated)} deprecated) - Domain: {domain}"]
        for s in active:
            name = s.get("name", "unknown")
            conf = s.get("confidence", 0)
            uses = s.get("usage_count", 0)
            status = s.get("status", "active")
            marker = "✓" if status == "active" else "⚠"
            lines.append(f"  {marker} {name:<40} {conf:.0%}  {uses:>4} uses  {status}")
        for s in deprecated:
            name = s.get("name", "unknown")
            conf = s.get("confidence", 0)
            uses = s.get("usage_count", 0)
            lines.append(f"  ✗ {name:<40} {conf:.0%}  {uses:>4} uses  deprecated")
        return "\n".join(lines)

    def _build_strategy_section(self, domain: str) -> str:
        if not self.strategies:
            return f"Strategy - Domain: {domain}\n  No strategy versions."
        current = max(self.strategies, key=lambda s: s.get("version", 0))
        lines = [
            f"Strategy (version {current.get('version', '?')}, active) - Domain: {domain}",
            "Version history:",
        ]
        for s in reversed(self.strategies[-10:]):
            v = s.get("version", "?")
            ts = s.get("created_at", "")
            marker = "← current" if s.get("is_active") else ""
            lines.append(f"  v{v} ({ts}) {marker}")
        return "\n".join(lines)

    def _build_facts_section(self, domain: str) -> str:
        if not self.facts:
            return f"Facts - Domain: {domain}\n  No facts stored."
        lines = [f"Facts ({len(self.facts)}) - Domain: {domain}"]
        for f in self.facts[:20]:
            content = f.get("content", "")[:80]
            conf = f.get("confidence", 0)
            lines.append(f"  \u2022 {content} (conf: {conf:.2f})")
        if len(self.facts) > 20:
            lines.append(f"  ... and {len(self.facts) - 20} more")
        return "\n".join(lines)

    def render_html(self, domain: str) -> str:
        sections = []
        sections.append(f"<h2>Memory Browser - {domain}</h2>")

        skills_html = self._build_skills_section(domain).replace("\n", "<br>")
        sections.append(f"<div class='section'><pre>{skills_html}</pre></div>")

        strategy_html = self._build_strategy_section(domain).replace("\n", "<br>")
        sections.append(f"<div class='section'><pre>{strategy_html}</pre></div>")

        facts_html = self._build_facts_section(domain).replace("\n", "<br>")
        sections.append(f"<div class='section'><pre>{facts_html}</pre></div>")

        return f"<html><body>{''.join(sections)}</body></html>"
