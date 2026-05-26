from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.memory.episodic import EpisodicMemory, Episode, RetentionTier
from nil.memory.semantic import SemanticMemory, Fact, FactType
from nil.memory.procedural import ProceduralMemory, Procedure
from nil.memory.meta_memory import MetaMemory

logger = logging.getLogger("nil.memory.consolidation")


@dataclass
class ConsolidationReport:
    episodes_compressed: int = 0
    episodes_archived: int = 0
    episodes_forgotten: int = 0
    facts_extracted: int = 0
    facts_confirmed: int = 0
    facts_decayed: int = 0
    skills_refined: int = 0
    contradictions_found: int = 0
    gaps_identified: int = 0
    duration_ms: float = 0.0
    details: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "compressed": self.episodes_compressed,
            "archived": self.episodes_archived,
            "forgotten": self.episodes_forgotten,
            "facts_extracted": self.facts_extracted,
            "facts_confirmed": self.facts_confirmed,
            "facts_decayed": self.facts_decayed,
            "skills_refined": self.skills_refined,
            "contradictions": self.contradictions_found,
            "gaps": self.gaps_identified,
            "duration_ms": self.duration_ms,
        }


class ConsolidationPipeline:
    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
        meta: MetaMemory,
        config: Optional[dict] = None,
    ):
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural
        self.meta = meta
        self.config = config or {}

    def run(self, domain: str) -> ConsolidationReport:
        start = time.time()
        report = ConsolidationReport()

        report.episodes_compressed = self._compress_old_episodes(domain)
        report.episodes_archived = self._archive_episodes(domain)
        report.episodes_forgotten = self._apply_retention(domain)
        report.facts_extracted = self._extract_facts_from_new_episodes(domain)
        report.facts_confirmed = self._confirm_existing_facts(domain)
        report.facts_decayed = self.semantic.decay_confidence(
            domain, decay_rate=self.config.get("decay_rate", 0.05)
        )
        report.skills_refined = self._refine_skills_from_facts(domain)
        report.contradictions_found = len(self.meta.detect_contradictions(domain))
        report.gaps_identified = self._identify_knowledge_gaps(domain)

        report.duration_ms = (time.time() - start) * 1000
        logger.info(
            "Consolidation for %s: %d facts extracted, %d compressed, %d forgotten (%.0fms)",
            domain, report.facts_extracted, report.episodes_compressed,
            report.episodes_forgotten, report.duration_ms,
        )
        return report

    def _compress_old_episodes(self, domain: str) -> int:
        recent = self.episodic.search_episodes(
            domain=domain, tier=RetentionTier.RECENT, limit=200
        )
        config = self.config.get("retention", {})
        max_recent = config.get("max_recent", 100)
        if len(recent) <= max_recent:
            return 0

        count = 0
        to_demote = recent[max_recent:]
        for ep in to_demote:
            ep.retention_tier = RetentionTier.SUMMARIZED
            ep.summary = self._generate_summary(ep)
            ep.steps = self._compress_steps(ep.steps)
            self.episodic.store_episode(ep)
            count += 1
        return count

    def _archive_episodes(self, domain: str) -> int:
        summarized = self.episodic.search_episodes(
            domain=domain, tier=RetentionTier.SUMMARIZED, limit=1000
        )
        config = self.config.get("retention", {})
        max_summarized = config.get("max_summarized", 500)
        if len(summarized) <= max_summarized:
            return 0

        count = 0
        to_archive = summarized[max_summarized:]
        for ep in to_archive:
            ep.retention_tier = RetentionTier.COMPRESSED
            ep.steps = []
            ep.metadata["compressed_at"] = time.time()
            self.episodic.store_episode(ep)
            count += 1
        return count

    def _apply_retention(self, domain: str) -> int:
        config = self.config.get("retention", {})
        return self.episodic.apply_retention(domain, config)

    def _extract_facts_from_new_episodes(self, domain: str) -> int:
        recent = self.episodic.search_episodes(
            domain=domain, tier=RetentionTier.RECENT, limit=50
        )
        count = 0
        for ep in recent:
            extracted = self._extract_facts_from_episode(ep)
            for fact in extracted:
                self.semantic.observe_fact(fact, source_episode_id=ep.episode_id)
                count += 1
        return count

    def _extract_facts_from_episode(self, episode: Episode) -> list[Fact]:
        facts = []
        tool_seqs = self._extract_tool_sequences(episode.steps)
        for seq in tool_seqs:
            if len(seq) >= 2:
                for i in range(len(seq) - 1):
                    fact = Fact(
                        fact_id="",
                        subject=f"tool:{seq[i]}",
                        predicate="precedes",
                        object=f"tool:{seq[i + 1]}",
                        confidence=0.6 if episode.outcome.get("success") else 0.3,
                        fact_type=FactType.RELATIONSHIP,
                        domain=episode.domain,
                    )
                    facts.append(fact)

        if episode.outcome.get("error"):
            err = str(episode.outcome["error"])
            error_fact = Fact(
                fact_id="",
                subject=f"error:{err[:50]}",
                predicate="causes",
                object="failure",
                confidence=0.7,
                fact_type=FactType.HEURISTIC,
                domain=episode.domain,
            )
            facts.append(error_fact)

        importance_tag = f"importance:{episode.importance:.1f}"
        fact = Fact(
            fact_id="",
            subject=f"task:{episode.task[:30]}",
            predicate="has_importance",
            object=importance_tag,
            confidence=0.5,
            fact_type=FactType.HEURISTIC,
            domain=episode.domain,
        )
        facts.append(fact)

        return facts

    def _extract_tool_sequences(self, steps: list[dict]) -> list[list[str]]:
        sequences = []
        current = []
        for step in steps:
            if isinstance(step, dict) and step.get("step_type") == "tool_call":
                name = step.get("tool_name", "")
                if name:
                    current.append(name)
            else:
                if current:
                    sequences.append(current)
                    current = []
        if current:
            sequences.append(current)
        return sequences

    def _confirm_existing_facts(self, domain: str) -> int:
        recent = self.episodic.search_episodes(
            domain=domain, tier=RetentionTier.RECENT, limit=50
        )
        if not recent:
            return 0

        facts = self.semantic.query_facts(domain=domain, limit=1000)
        confirmed = 0
        for episode in recent:
            tool_names = {
                s.get("tool_name", "") for s in episode.steps
                if isinstance(s, dict) and s.get("step_type") == "tool_call"
            }
            for fact in facts:
                subject_tool = fact.subject.replace("tool:", "")
                object_tool = fact.object.replace("tool:", "")
                if subject_tool in tool_names and object_tool in tool_names:
                    self.semantic.observe_fact(fact, source_episode_id=episode.episode_id)
                    confirmed += 1
        return confirmed

    def _refine_skills_from_facts(self, domain: str) -> int:
        high_conf_facts = self.semantic.query_facts(
            domain=domain, min_confidence=0.6, limit=100
        )
        if len(high_conf_facts) < 3:
            return 0

        tool_precedence = {}
        for f in high_conf_facts:
            if f.fact_type == FactType.RELATIONSHIP:
                subj = f.subject.replace("tool:", "")
                obj = f.object.replace("tool:", "")
                if subj and obj:
                    if subj not in tool_precedence:
                        tool_precedence[subj] = set()
                    tool_precedence[subj].add(obj)

        refined = 0
        for tool, following in tool_precedence.items():
            if len(following) >= 2:
                proc = Procedure(
                    procedure_id=f"auto_{tool}_{int(time.time())}",
                    domain=domain,
                    name=f"after_{tool}_do_{'_'.join(list(following)[:3])}",
                    steps=[
                        {"tool": t, "type": "tool_call"}
                        for t in [tool] + list(following)
                    ],
                    trigger_conditions=self._build_trigger(tool),
                    preconditions=[f"{tool} available"],
                    postconditions=[f"{t} completed" for t in following],
                    success_rate=0.5,
                )
                self.procedural.store_procedure(proc)
                refined += 1

        return refined

    def _build_trigger(self, tool_name: str) -> Any:
        from nil.memory.skill_registry import TriggerCondition
        return TriggerCondition(
            after_tools=[tool_name],
            description=f"After calling {tool_name}",
        )

    def _identify_knowledge_gaps(self, domain: str) -> int:
        facts = self.semantic.query_facts(domain=domain, limit=5000)
        subjects = set(f.subject for f in facts)
        objects = set(f.object for f in facts)

        tools_in_subjects = {
            s.replace("tool:", "") for s in subjects if s.startswith("tool:")
        }
        tools_in_objects = {
            o.replace("tool:", "") for o in objects if o.startswith("tool:")
        }
        all_tools = tools_in_subjects | tools_in_objects

        gaps_found = 0
        for tool in all_tools:
            has_precedes = any(
                f.subject == f"tool:{tool}" and f.predicate == "precedes"
                for f in facts
            )
            has_preceded_by = any(
                f.object == f"tool:{tool}" and f.predicate == "precedes"
                for f in facts
            )

            if not has_precedes and not has_preceded_by:
                self.meta.identify_gap(
                    area=f"tool:{tool}_relationships",
                    evidence=f"No known relationships for tool {tool}",
                    domain=domain,
                )
                gaps_found += 1

        return gaps_found

    def _generate_summary(self, episode: Episode) -> str:
        steps = episode.steps
        tool_calls = [
            s.get("tool_name", "") for s in steps
            if isinstance(s, dict) and s.get("step_type") == "tool_call"
        ]
        outcome = "success" if episode.outcome.get("success") else "failure"
        seq = " → ".join(tool_calls[:5])
        if not seq:
            seq = "(no tools)"
        return f"[{outcome}] {episode.task[:60]}: {seq}"

    def _compress_steps(self, steps: list[dict]) -> list[dict]:
        compressed = []
        for step in steps:
            if isinstance(step, dict):
                s = dict(step)
                if "tool_output" in s and isinstance(s["tool_output"], str):
                    s["tool_output"] = s["tool_output"][:200]
                if "content" in s and isinstance(s.get("content"), dict):
                    if "prompt" in s["content"]:
                        s["content"] = {"prompt": str(s["content"]["prompt"])[:100]}
                compressed.append(s)
            else:
                compressed.append(step)
        return compressed
