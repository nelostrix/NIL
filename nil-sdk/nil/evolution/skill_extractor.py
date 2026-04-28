"""
SkillExtractor — extracts reusable skills from successful traces.

Phase 2 core module. Takes patterns found by the Analyzer and
converts them into named, storable Skills that get injected
into future agent runs.

A skill is essentially: "When you see X, do Y" — learned from
the agent's own experience.
"""

from __future__ import annotations

import logging
from typing import Any

from nil.evolution.analyzer import TraceAnalyzer
from nil.memory.skill_registry import Skill
from nil.memory.store import PersistentMemory

logger = logging.getLogger("nil.evolution.skill_extractor")


class SkillExtractor:
    """
    Converts trace patterns into reusable skills.

    Pipeline:
        1. Analyzer finds winning patterns
        2. SkillExtractor names + describes them
        3. Skills get stored in memory
        4. Future runs inject skills into the system prompt
    """

    def __init__(self, memory: PersistentMemory) -> None:
        self.memory = memory
        self.analyzer = TraceAnalyzer(memory)

    def extract(
        self,
        domain: str,
        min_frequency: int = 3,
        min_confidence: float = 0.7,
    ) -> list[Skill]:
        """
        Extract skills from accumulated traces.

        Only extracts skills that meet both frequency and
        confidence thresholds.
        """
        patterns = self.analyzer.find_winning_patterns(
            domain, min_frequency=min_frequency
        )

        extracted = []
        for pattern in patterns:
            if pattern["success_rate"] < min_confidence:
                continue

            skill = self._pattern_to_skill(domain, pattern)
            if skill:
                # Store in memory
                self.memory.store_skill(
                    domain=domain,
                    name=skill.name,
                    description=skill.description,
                    pattern={
                        "sequence": pattern["sequence"],
                        "success_rate": pattern["success_rate"],
                    },
                    confidence=pattern["success_rate"],
                )
                extracted.append(skill)
                logger.info(
                    "Extracted skill '%s' (confidence=%.2f) for domain '%s'",
                    skill.name,
                    pattern["success_rate"],
                    domain,
                )

        return extracted

    def _pattern_to_skill(self, domain: str, pattern: dict[str, Any]) -> Skill | None:
        """
        Convert a raw pattern into a named Skill.

        Uses the tool sequence to generate a descriptive name.
        """
        sequence = pattern.get("sequence", [])
        if not sequence:
            return None

        # Generate a name from the sequence
        name = self._generate_skill_name(sequence)

        # Generate description
        description = (
            f"When applicable, use this sequence: {' → '.join(sequence)}. "
            f"This pattern succeeded {pattern['success_count']} times "
            f"with a {pattern['success_rate']:.0%} success rate."
        )

        return Skill(
            name=name,
            description=description,
            action_sequence=sequence,
            confidence=pattern["success_rate"],
        )

    def _generate_skill_name(self, sequence: list[str]) -> str:
        """
        Generate a readable skill name from a tool sequence.

        Examples:
            ["search", "analyze", "fix"] → "search_then_analyze_then_fix"
            ["check_logs", "check_logs", "deploy"] → "check_logs_twice_then_deploy"
        """
        # Collapse consecutive duplicates
        collapsed = []
        for tool_name in sequence:
            if collapsed and collapsed[-1][0] == tool_name:
                collapsed[-1] = (tool_name, collapsed[-1][1] + 1)
            else:
                collapsed.append((tool_name, 1))

        parts = []
        for name, count in collapsed:
            clean = name.replace("-", "_").replace(".", "_")
            if count > 1:
                parts.append(f"{clean}_x{count}")
            else:
                parts.append(clean)

        return "_then_".join(parts)
