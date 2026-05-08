"""
Skill Compression Engine — converts raw trace patterns into
compressed, callable skills.

This is NOT just "name a tool sequence." This module:

1. CHAIN COMPRESSION
   - Multi-step sequences become single callable instructions
   - "search_logs → analyze_output → identify_root_cause → apply_fix"
     becomes: "Run the incident_diagnosis skill"
   - The agent invokes one concept instead of re-deriving 4 steps

2. TRIGGER INFERENCE
   - Analyzes WHAT tasks/errors/contexts preceded successful use
   - Builds trigger conditions: "Activate when task mentions 'timeout'
     AND logs tool hasn't been called yet"

   - Uses counterfactual data from the analyzer

3. PRECONDITION / POSTCONDITION EXTRACTION
   - What must be true before this skill applies?
   - What should be true after it runs?
   - Inferred from the trace context around the pattern

4. COST/DURATION ESTIMATION
   - Average cost and duration when this pattern runs
   - Used by the optimization layer for budgeting

5. SKILL LIFECYCLE MANAGEMENT
   - candidate → active → deprecated/superseded
   - Skills that stop working get deprecated
   - Better skills supersede older ones
   - Version tracking for skill evolution

6. SKILL MERGING & DEDUPLICATION
   - Two skills with overlapping sequences get merged
   - Subsequences of a larger skill are absorbed
   - Prevents skill proliferation

7. PROMPT GENERATION
   - Skills produce optimized prompt fragments
   - Compact mode for constrained context windows
   - Full mode with triggers, preconditions, steps, confidence
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.evolution.analyzer import (
    AnalysisReport,
    CausalPattern,
    TraceAnalyzer,
)
from nil.memory.skill_registry import (
    Skill,
    SkillOrigin,
    SkillStatus,
    TriggerCondition,
)
from nil.memory.store import PersistentMemory

logger = logging.getLogger("nil.evolution.skill_extractor")


# ---------------------------------------------------------------------------
# Extraction Result
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Result of a skill extraction cycle."""

    domain: str
    extracted: list[Skill] = field(default_factory=list)
    merged: list[Skill] = field(default_factory=list)
    deprecated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    total_patterns_analyzed: int = 0
    total_skills_before: int = 0
    total_skills_after: int = 0

    @property
    def new_skill_count(self) -> int:
        return len(self.extracted)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "new_skills": self.new_skill_count,
            "merged": len(self.merged),
            "deprecated": len(self.deprecated),
            "unchanged": len(self.unchanged),
            "patterns_analyzed": self.total_patterns_analyzed,
            "skills_before": self.total_skills_before,
            "skills_after": self.total_skills_after,
        }


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------


class SkillExtractor:
    """
    Skill Compression Engine — extracts, compresses, and manages skills.

    Usage:
        extractor = SkillExtractor(memory)
        result = extractor.extract("my-domain")

        for skill in result.extracted:
            print(skill.to_prompt_fragment())

    The extraction pipeline:
        1. Run causal analysis to find significant patterns
        2. Filter by confidence + frequency thresholds
        3. Infer trigger conditions from trace context
        4. Compress multi-step sequences into instructions
        5. Extract preconditions and postconditions
        6. Calculate cost/duration estimates
        7. Deduplicate and merge with existing skills
        8. Manage skill lifecycle (deprecate ineffective skills)
        9. Store everything to memory
    """

    def __init__(
        self,
        memory: PersistentMemory,
        analyzer: Optional[TraceAnalyzer] = None,
        min_frequency: int = 3,
        min_confidence: float = 0.7,
        max_skills_per_domain: int = 20,
    ) -> None:
        self.memory = memory
        self.analyzer = analyzer or TraceAnalyzer(memory)
        self.min_frequency = min_frequency
        self.min_confidence = min_confidence
        self.max_skills_per_domain = max_skills_per_domain

    # ------------------------------------------------------------------
    # Main extraction pipeline
    # ------------------------------------------------------------------

    def extract(
        self,
        domain: str,
        min_frequency: int = None,
        min_confidence: float = None,
        max_traces: int = 500,
    ) -> ExtractionResult:
        """
        Run the full skill extraction pipeline.

        Returns an ExtractionResult with new, merged, deprecated,
        and unchanged skills.
        """
        freq = min_frequency if min_frequency is not None else self.min_frequency
        conf = min_confidence if min_confidence is not None else self.min_confidence

        existing_skills = self._load_existing_skills(domain)
        result = ExtractionResult(
            domain=domain,
            total_skills_before=len(existing_skills),
        )

        # Step 1: Run causal analysis
        report = self.analyzer.full_analysis(domain, max_traces=max_traces)
        patterns = report.causal_patterns
        result.total_patterns_analyzed = len(patterns)

        if not patterns:
            result.total_skills_after = len(existing_skills)
            result.unchanged = [s.name for s in existing_skills]
            return result

        # Step 2: Get traces for context analysis
        traces = self.memory.get_traces(domain, limit=max_traces)

        # Step 3: Extract new skills from significant patterns
        candidates: list[Skill] = []
        for pattern in patterns:
            if not pattern.is_significant:
                continue
            if pattern.success_rate < conf:
                continue
            if pattern.total < freq:
                continue

            skill = self._pattern_to_skill(pattern, traces, report)
            if skill:
                candidates.append(skill)

        # Step 4: Deduplicate candidates
        deduplicated = self._deduplicate_skills(candidates)

        # Step 5: Merge with existing skills
        new_skills, merged_skills, unchanged = self._merge_with_existing(
            deduplicated, existing_skills
        )

        # Step 6: Deprecate ineffective existing skills
        deprecated = self._deprecate_ineffective(existing_skills, traces)

        # Step 7: Enforce max skills limit
        all_active = [
            s
            for s in (
                new_skills
                + merged_skills
                + [s for s in existing_skills if s.name in unchanged and s.name not in deprecated]
            )
            if s.status != SkillStatus.DEPRECATED
        ]

        if len(all_active) > self.max_skills_per_domain:
            # Keep top N by confidence
            all_active.sort(key=lambda s: s.confidence, reverse=True)
            kept = all_active[: self.max_skills_per_domain]
            kept_names = {s.name for s in kept}
            for s in all_active[self.max_skills_per_domain :]:
                if s.name not in {sk.name for sk in new_skills}:
                    deprecated.append(s.name)

        # Step 8: Store to memory
        for skill in new_skills + merged_skills:
            self._store_skill(domain, skill)

        for name in deprecated:
            self._mark_deprecated(domain, name)

        result.extracted = new_skills
        result.merged = merged_skills
        result.deprecated = deprecated
        result.unchanged = list(unchanged)
        result.total_skills_after = (
            len(new_skills) + len(merged_skills) + len(unchanged) - len(deprecated)
        )

        logger.info(
            "Extraction for '%s': %d new, %d merged, %d deprecated, %d unchanged",
            domain,
            len(new_skills),
            len(merged_skills),
            len(deprecated),
            len(unchanged),
        )

        return result

    # ------------------------------------------------------------------
    # Pattern → Skill conversion
    # ------------------------------------------------------------------

    def _pattern_to_skill(
        self,
        pattern: CausalPattern,
        traces: list[dict],
        report: AnalysisReport,
    ) -> Optional[Skill]:
        """Convert a CausalPattern into a full Skill."""
        sequence = pattern.sequence
        if not sequence:
            return None

        # Generate name
        name = self._generate_name(sequence)

        # Infer trigger conditions
        trigger = self._infer_trigger(pattern, traces)

        # Generate compressed instruction
        compressed = self._compress_chain(sequence, pattern)

        # Extract pre/post conditions
        preconditions = self._extract_preconditions(pattern, traces)
        postconditions = self._extract_postconditions(pattern, traces)

        # Generate description
        description = self._generate_description(sequence, pattern, trigger)

        # Find cost/duration from source traces
        source_traces = [t for t in traces if t.get("run_id") in set(pattern.source_run_ids)]
        avg_cost = (
            sum(t.get("total_cost", 0) for t in source_traces) / len(source_traces)
            if source_traces
            else pattern.avg_cost
        )
        avg_duration = (
            sum(t.get("total_duration", 0) for t in source_traces) / len(source_traces)
            if source_traces
            else pattern.avg_duration
        )

        skill = Skill(
            name=name,
            description=description,
            trigger=trigger,
            action_sequence=sequence,
            compressed_instruction=compressed,
            preconditions=preconditions,
            postconditions=postconditions,
            confidence=pattern.success_rate,
            source_runs=pattern.source_run_ids[:10],
            avg_cost=round(avg_cost, 6),
            avg_duration=round(avg_duration, 4),
            status=SkillStatus.CANDIDATE,
            origin=SkillOrigin.PATTERN_EXTRACTION,
            tags=self._infer_tags(sequence, pattern),
        )

        return skill

    # ------------------------------------------------------------------
    # Name generation
    # ------------------------------------------------------------------

    def _generate_name(self, sequence: list[str]) -> str:
        """
        Generate a readable, unique skill name from a tool sequence.

        Examples:
            ["search", "analyze", "fix"] → "search_then_analyze_then_fix"
            ["check_logs", "check_logs", "deploy"] → "check_logs_x2_then_deploy"
            ["diagnose"] → "diagnose_skill"
        """
        collapsed = []
        for tool in sequence:
            if collapsed and collapsed[-1][0] == tool:
                collapsed[-1] = (tool, collapsed[-1][1] + 1)
            else:
                collapsed.append((tool, 1))

        parts = []
        for tool_name, count in collapsed:
            clean = tool_name.replace("-", "_").replace(".", "_").replace(" ", "_")
            if count > 1:
                parts.append(f"{clean}_x{count}")
            else:
                parts.append(clean)

        name = "_then_".join(parts)

        # Single-tool skills get a suffix
        if len(sequence) == 1:
            name = f"{name}_skill"

        return name

    # ------------------------------------------------------------------
    # Trigger inference
    # ------------------------------------------------------------------

    def _infer_trigger(self, pattern: CausalPattern, traces: list[dict]) -> TriggerCondition:
        """
        Infer when this skill should activate by analyzing the
        tasks and contexts where it was successfully used.
        """
        # Collect tasks from source runs
        source_ids = set(pattern.source_run_ids)
        source_traces = [t for t in traces if t.get("run_id") in source_ids]
        successful_sources = [t for t in source_traces if t.get("outcome_success")]

        # Extract keywords from successful tasks
        keywords = self._extract_keywords([t.get("task", "") for t in successful_sources])

        # Extract error patterns that preceded this sequence
        error_patterns = []
        for t in successful_sources:
            for step in t.get("steps", []):
                if step.get("error") and step.get("step_number", 99) <= 2:
                    err = step["error"][:50]
                    if err not in error_patterns:
                        error_patterns.append(err)

        # Determine tool prerequisites
        after_tools = []
        sequence_set = set(pattern.sequence)
        for t in successful_sources:
            seq = t.get("tool_sequence", [])
            first_idx = None
            for i, tool in enumerate(seq):
                if tool == pattern.sequence[0]:
                    first_idx = i
                    break
            if first_idx and first_idx > 0:
                preceding = seq[first_idx - 1]
                if preceding not in sequence_set:
                    after_tools.append(preceding)

        # Build description
        desc_parts = []
        if keywords:
            desc_parts.append(f"task involves {', '.join(keywords[:3])}")
        if error_patterns:
            desc_parts.append(f"errors like '{error_patterns[0]}'")
        description = " and ".join(desc_parts) if desc_parts else ""

        return TriggerCondition(
            keywords=keywords[:5],
            error_patterns=error_patterns[:3],
            after_tools=list(set(after_tools))[:3],
            description=description,
        )

    def _extract_keywords(self, tasks: list[str]) -> list[str]:
        """Extract common meaningful keywords from task descriptions."""
        if not tasks:
            return []

        # Tokenize and count
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "each",
            "every",
            "all",
            "any",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "about",
            "above",
            "also",
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
            "what",
            "which",
            "who",
            "whom",
            "how",
            "when",
            "where",
            "why",
            "if",
            "then",
            "else",
            "up",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "once",
        }

        word_counts: Counter = Counter()
        for task in tasks:
            words = task.lower().split()
            for w in words:
                cleaned = w.strip(".,!?;:'\"()[]{}").strip()
                if len(cleaned) > 2 and cleaned not in stop_words:
                    word_counts[cleaned] += 1

        # Keep words that appear in >50% of tasks
        threshold = max(1, len(tasks) * 0.5)
        common = [word for word, count in word_counts.most_common(10) if count >= threshold]

        return common

    # ------------------------------------------------------------------
    # Chain compression
    # ------------------------------------------------------------------

    def _compress_chain(self, sequence: list[str], pattern: CausalPattern) -> str:
        """
        Compress a multi-step tool sequence into a single instruction.

        ["search_logs", "analyze_output", "apply_fix"]
        → "Search logs for the issue, analyze the output to find
           root cause, then apply the appropriate fix"
        """
        if len(sequence) == 1:
            return f"Use {sequence[0]} to address the issue"

        # Build human-readable instruction
        steps = []
        for i, tool in enumerate(sequence):
            readable = tool.replace("_", " ").replace("-", " ")
            if i == 0:
                steps.append(f"First, {readable}")
            elif i == len(sequence) - 1:
                steps.append(f"finally {readable}")
            else:
                steps.append(f"then {readable}")

        instruction = ", ".join(steps) + "."

        # Add confidence note
        rate = pattern.success_rate
        if rate >= 0.9:
            instruction += f" This sequence is highly reliable ({rate:.0%} success rate)."
        elif rate >= 0.7:
            instruction += f" This sequence works well ({rate:.0%} success rate)."

        return instruction

    # ------------------------------------------------------------------
    # Pre/post condition extraction
    # ------------------------------------------------------------------

    def _extract_preconditions(self, pattern: CausalPattern, traces: list[dict]) -> list[str]:
        """Infer what must be true before this skill applies."""
        preconditions = []

        # If the pattern always starts after certain tools
        source_ids = set(pattern.source_run_ids)
        source_traces = [t for t in traces if t.get("run_id") in source_ids]

        preceding_tools = Counter()
        for t in source_traces:
            seq = t.get("tool_sequence", [])
            first_tool = pattern.sequence[0]
            for i, tool in enumerate(seq):
                if tool == first_tool and i > 0:
                    preceding_tools[seq[i - 1]] += 1

        for tool, count in preceding_tools.most_common(2):
            if count >= len(source_traces) * 0.7:
                preconditions.append(f"'{tool}' should have been called first")

        return preconditions

    def _extract_postconditions(self, pattern: CausalPattern, traces: list[dict]) -> list[str]:
        """Infer what should be true after this skill runs."""
        postconditions = []

        if pattern.success_rate >= 0.9:
            postconditions.append("Issue should be resolved")
        elif pattern.success_rate >= 0.7:
            postconditions.append("Issue should be partially or fully resolved")

        # Check if there's a consistent tool called after this pattern
        source_ids = set(pattern.source_run_ids)
        source_traces = [t for t in traces if t.get("run_id") in source_ids]
        last_tool = pattern.sequence[-1]

        following_tools = Counter()
        for t in source_traces:
            seq = t.get("tool_sequence", [])
            for i, tool in enumerate(seq):
                if tool == last_tool and i < len(seq) - 1:
                    following_tools[seq[i + 1]] += 1

        for tool, count in following_tools.most_common(1):
            if count >= len(source_traces) * 0.5:
                postconditions.append(f"Consider following up with '{tool}'")

        return postconditions

    # ------------------------------------------------------------------
    # Description generation
    # ------------------------------------------------------------------

    def _generate_description(
        self,
        sequence: list[str],
        pattern: CausalPattern,
        trigger: TriggerCondition,
    ) -> str:
        """Generate a human-readable description of the skill."""
        seq_str = " → ".join(sequence)
        parts = [f"Execute the sequence: {seq_str}."]

        if trigger.description:
            parts.append(f"Use when {trigger.description}.")

        parts.append(
            f"Succeeded {pattern.success_count} times "
            f"({pattern.success_rate:.0%} rate, p={pattern.p_value:.4f})."
        )

        if pattern.temporal_trend == "rising":
            parts.append("This pattern is becoming more effective recently.")
        elif pattern.temporal_trend == "falling":
            parts.append("Note: effectiveness may be declining.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Tag inference
    # ------------------------------------------------------------------

    def _infer_tags(self, sequence: list[str], pattern: CausalPattern) -> list[str]:
        """Infer categorical tags for the skill."""
        tags = []

        # Length-based tags
        if len(sequence) == 1:
            tags.append("single-step")
        elif len(sequence) <= 3:
            tags.append("short-chain")
        else:
            tags.append("long-chain")

        # Confidence-based tags
        if pattern.success_rate >= 0.95:
            tags.append("high-reliability")
        elif pattern.success_rate >= 0.8:
            tags.append("reliable")

        # Trend tags
        if pattern.temporal_trend == "rising":
            tags.append("trending-up")
        elif pattern.temporal_trend == "falling":
            tags.append("trending-down")

        # Significance
        if pattern.is_significant:
            tags.append("statistically-significant")

        return tags

    # ------------------------------------------------------------------
    # Deduplication & merging
    # ------------------------------------------------------------------

    def _deduplicate_skills(self, candidates: list[Skill]) -> list[Skill]:
        """
        Remove duplicate and subsumed skills.

        If skill A's sequence is a subsequence of skill B's,
        and both have similar confidence, keep only B.
        """
        if len(candidates) <= 1:
            return candidates

        # Sort by sequence length descending (prefer longer patterns)
        sorted_skills = sorted(candidates, key=lambda s: len(s.action_sequence), reverse=True)

        kept: list[Skill] = []
        for skill in sorted_skills:
            seq = skill.action_sequence
            is_subsumed = False

            for existing in kept:
                if self._is_subsequence(seq, existing.action_sequence):
                    # seq is a subsequence of existing — skip it
                    # unless it has significantly higher confidence
                    if skill.confidence <= existing.confidence + 0.1:
                        is_subsumed = True
                        break

            if not is_subsumed:
                # Also check if an existing shorter skill should be replaced
                to_remove = []
                for i, existing in enumerate(kept):
                    if self._is_subsequence(existing.action_sequence, seq):
                        if existing.confidence <= skill.confidence + 0.1:
                            to_remove.append(i)

                for idx in reversed(to_remove):
                    kept.pop(idx)

                kept.append(skill)

        return kept

    def _merge_with_existing(
        self,
        new_skills: list[Skill],
        existing_skills: list[Skill],
    ) -> tuple[list[Skill], list[Skill], set[str]]:
        """
        Merge new skills with existing ones.

        Returns:
            (truly_new, merged_updates, unchanged_names)
        """
        existing_by_name = {s.name: s for s in existing_skills}
        existing_by_seq = {tuple(s.action_sequence): s for s in existing_skills}

        truly_new = []
        merged = []
        seen_existing = set()

        for skill in new_skills:
            seq_key = tuple(skill.action_sequence)

            if skill.name in existing_by_name:
                # Same name — update existing
                old = existing_by_name[skill.name]
                updated = self._merge_skill(old, skill)
                merged.append(updated)
                seen_existing.add(skill.name)

            elif seq_key in existing_by_seq:
                # Same sequence, different name — update existing
                old = existing_by_seq[seq_key]
                updated = self._merge_skill(old, skill)
                merged.append(updated)
                seen_existing.add(old.name)

            else:
                truly_new.append(skill)

        unchanged = {s.name for s in existing_skills if s.name not in seen_existing}

        return truly_new, merged, unchanged

    def _merge_skill(self, old: Skill, new: Skill) -> Skill:
        """Merge a new skill version into an existing one."""
        return Skill(
            name=old.name,
            description=new.description or old.description,
            trigger=new.trigger if new.trigger.keywords else old.trigger,
            action_sequence=new.action_sequence or old.action_sequence,
            compressed_instruction=new.compressed_instruction or old.compressed_instruction,
            preconditions=new.preconditions or old.preconditions,
            postconditions=new.postconditions or old.postconditions,
            confidence=max(old.confidence, new.confidence),
            source_runs=list(set(old.source_runs + new.source_runs))[:20],
            usage_count=old.usage_count,
            success_when_used=old.success_when_used,
            failure_when_used=old.failure_when_used,
            avg_cost=new.avg_cost if new.avg_cost > 0 else old.avg_cost,
            avg_duration=new.avg_duration if new.avg_duration > 0 else old.avg_duration,
            status=SkillStatus.ACTIVE
            if old.status == SkillStatus.ACTIVE
            else SkillStatus.CANDIDATE,
            origin=old.origin,
            version=old.version + 1,
            parent_skill=old.name,
            tags=list(set(old.tags + new.tags)),
        )

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def _deprecate_ineffective(
        self,
        existing_skills: list[Skill],
        traces: list[dict],
    ) -> list[str]:
        """
        Find skills that are no longer effective and mark them deprecated.

        A skill is deprecated if:
        - It has been used 5+ times with <40% success rate
        - Its pattern no longer appears in recent successful traces
        """
        deprecated = []
        recent_traces = sorted(traces, key=lambda t: t.get("created_at", 0), reverse=True)[:50]
        recent_success_seqs = [
            tuple(t.get("tool_sequence", [])) for t in recent_traces if t.get("outcome_success")
        ]

        for skill in existing_skills:
            if skill.status == SkillStatus.DEPRECATED:
                continue

            # Check usage-based deprecation
            total_use = skill.success_when_used + skill.failure_when_used
            if total_use >= 5 and skill.usage_success_rate < 0.4:
                deprecated.append(skill.name)
                logger.info(
                    "Deprecating skill '%s': low success rate (%.0f%%)",
                    skill.name,
                    skill.usage_success_rate * 100,
                )
                continue

            # Check if pattern still appears in recent successes
            if total_use == 0 and skill.status == SkillStatus.ACTIVE:
                seq_tuple = tuple(skill.action_sequence)
                if seq_tuple and recent_success_seqs:
                    found = any(
                        self._is_subsequence(list(seq_tuple), list(rss))
                        for rss in recent_success_seqs
                    )
                    if not found and len(recent_traces) >= 20:
                        deprecated.append(skill.name)
                        logger.info(
                            "Deprecating skill '%s': pattern absent from recent successes",
                            skill.name,
                        )

        return deprecated

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def _store_skill(self, domain: str, skill: Skill) -> None:
        """Store a skill to persistent memory."""
        self.memory.store_skill(
            domain=domain,
            name=skill.name,
            description=skill.to_prompt_fragment(),
            pattern=skill.to_dict(),
            source_runs=skill.source_runs,
            confidence=skill.confidence,
        )

    def _mark_deprecated(self, domain: str, name: str) -> None:
        """Mark a skill as deprecated in memory."""
        existing = self.memory.get_skill(domain, name)
        if existing:
            pattern = existing.get("pattern")
            if pattern and isinstance(pattern, str):
                import json

                try:
                    pattern = json.loads(pattern)
                except (json.JSONDecodeError, TypeError):
                    pattern = {}
            elif not isinstance(pattern, dict):
                pattern = {}
            pattern["status"] = "deprecated"
            self.memory.store_skill(
                domain=domain,
                name=name,
                description=existing.get("description", ""),
                pattern=pattern,
                confidence=0.0,  # Drop confidence to sort to bottom
            )

    def _load_existing_skills(self, domain: str) -> list[Skill]:
        """Load all existing skills from memory as Skill objects."""
        import json

        names = self.memory.list_skills(domain)
        skills = []
        for name in names:
            raw = self.memory.get_skill(domain, name)
            if not raw:
                continue
            pattern = raw.get("pattern")
            if isinstance(pattern, str):
                try:
                    pattern = json.loads(pattern)
                except (json.JSONDecodeError, TypeError):
                    pattern = {}
            if isinstance(pattern, dict) and "name" in pattern:
                skills.append(Skill.from_dict(pattern))
            else:
                # Legacy skill without full data
                skills.append(
                    Skill(
                        name=name,
                        description=raw.get("description", ""),
                        confidence=raw.get("confidence", 0.0),
                        action_sequence=pattern.get("sequence", [])
                        if isinstance(pattern, dict)
                        else [],
                    )
                )
        return skills

    # ------------------------------------------------------------------
    # Utility: generate prompt for all active skills
    # ------------------------------------------------------------------

    def generate_skill_prompt(self, domain: str, compact: bool = False) -> str:
        """
        Generate the full skill injection block for the system prompt.

        Args:
            compact: If True, use short format for constrained context.
        """
        skills = self._load_existing_skills(domain)
        active = [s for s in skills if s.status != SkillStatus.DEPRECATED and s.confidence > 0]

        if not active:
            return ""

        active.sort(key=lambda s: s.confidence, reverse=True)

        lines = ["LEARNED SKILLS (from past experience):"]
        for skill in active:
            if compact:
                lines.append(skill.to_compact_prompt())
            else:
                lines.append(skill.to_prompt_fragment())
                lines.append("")  # blank line between skills

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_subsequence(short: list[str], long: list[str]) -> bool:
        """Check if `short` is a contiguous subsequence of `long`."""
        if not short:
            return True
        if len(short) > len(long):
            return False
        for i in range(len(long) - len(short) + 1):
            if long[i : i + len(short)] == short:
                return True
        return False
