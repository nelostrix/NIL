"""
Domain World Model — the agent's internal map of its environment.

After N traces, this module automatically constructs a structured
representation of the domain:

1. ENTITY DISCOVERY
   - What tools exist and what they do (inferred from usage)
   - What types of tasks appear (clustered by keywords)
   - What error types exist and how they relate to tools

2. TOOL CAPABILITY MODEL
   - Per-tool success rates, costs, durations
   - What inputs each tool typically receives
   - What outputs each tool typically produces
   - Which tools are complementary vs redundant

3. TASK TAXONOMY
   - Clusters tasks by similarity
   - Maps task types to winning tool sequences
   - Identifies which task types the agent handles well vs poorly

4. STATE TRANSITION GRAPH
   - Valid action sequences (what can follow what)
   - Forbidden transitions (dead ends from the analyzer)
   - Expected outcomes per transition
   - Builds on the Markov model from the analyzer but adds
     semantic meaning

5. CONSTRAINT DISCOVERY
   - "Tool X only works after tool Y"
   - "Error Z always occurs when tool W is used on task type T"
   - "Success requires at least N steps"
   - Inferred automatically from trace patterns

6. DOMAIN STATISTICS
   - Overall health metrics
   - Cost and duration distributions
   - Improvement trajectory over time

This is what NeoCognition raised $40M to build. We do it as
an SDK feature using only the agent's own execution traces —
no external knowledge, no manual annotation.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.evolution.analyzer import AnalysisReport, TraceAnalyzer
from nil.memory.store import PersistentMemory

logger = logging.getLogger("nil.evolution.world_model")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ToolProfile:
    """Everything the world model knows about a single tool."""

    name: str
    total_invocations: int = 0
    invocations_in_success: int = 0
    invocations_in_failure: int = 0
    success_rate: float = 0.0
    avg_duration_ms: float = 0.0
    avg_cost_per_use: float = 0.0
    common_inputs: list[str] = field(default_factory=list)
    common_outputs: list[str] = field(default_factory=list)
    common_errors: list[str] = field(default_factory=list)
    error_rate: float = 0.0
    typical_position: float = 0.0  # avg step number when called
    complementary_tools: list[str] = field(default_factory=list)
    redundant_with: list[str] = field(default_factory=list)
    causal_direction: str = "neutral"  # from analyzer

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_invocations": self.total_invocations,
            "invocations_in_success": self.invocations_in_success,
            "invocations_in_failure": self.invocations_in_failure,
            "success_rate": self.success_rate,
            "avg_duration_ms": self.avg_duration_ms,
            "avg_cost_per_use": self.avg_cost_per_use,
            "common_errors": self.common_errors,
            "error_rate": self.error_rate,
            "typical_position": self.typical_position,
            "complementary_tools": self.complementary_tools,
            "redundant_with": self.redundant_with,
            "causal_direction": self.causal_direction,
        }


@dataclass
class TaskCluster:
    """A group of similar tasks."""

    cluster_id: str
    keywords: list[str] = field(default_factory=list)
    task_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    winning_sequences: list[list[str]] = field(default_factory=list)
    avg_cost: float = 0.0
    avg_duration: float = 0.0
    example_tasks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "keywords": self.keywords,
            "task_count": self.task_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "winning_sequences": self.winning_sequences,
            "avg_cost": self.avg_cost,
            "avg_duration": self.avg_duration,
            "example_tasks": self.example_tasks[:5],
        }


@dataclass
class Constraint:
    """A discovered constraint about the domain."""

    constraint_id: str
    constraint_type: str  # "prerequisite", "forbidden", "required_minimum", "conditional_error"
    description: str
    confidence: float = 0.0
    evidence_count: int = 0
    involved_tools: list[str] = field(default_factory=list)
    involved_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type,
            "description": self.description,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "involved_tools": self.involved_tools,
            "involved_errors": self.involved_errors,
        }


@dataclass
class StateTransition:
    """A transition in the state graph with semantic meaning."""

    from_state: str
    to_state: str
    count: int = 0
    success_rate: float = 0.0
    avg_cost: float = 0.0
    is_recommended: bool = False
    is_forbidden: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_state,
            "to": self.to_state,
            "count": self.count,
            "success_rate": self.success_rate,
            "is_recommended": self.is_recommended,
            "is_forbidden": self.is_forbidden,
        }


@dataclass
class DomainHealth:
    """Overall health metrics for the domain."""

    total_runs: int = 0
    success_rate: float = 0.0
    avg_cost: float = 0.0
    avg_duration: float = 0.0
    cost_p50: float = 0.0
    cost_p95: float = 0.0
    duration_p50: float = 0.0
    duration_p95: float = 0.0
    trend: str = "stable"
    trend_slope: float = 0.0
    unique_tools: int = 0
    unique_task_types: int = 0
    active_skills: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "success_rate": self.success_rate,
            "avg_cost": self.avg_cost,
            "avg_duration": self.avg_duration,
            "cost_p50": self.cost_p50,
            "cost_p95": self.cost_p95,
            "duration_p50": self.duration_p50,
            "duration_p95": self.duration_p95,
            "trend": self.trend,
            "trend_slope": self.trend_slope,
            "unique_tools": self.unique_tools,
            "unique_task_types": self.unique_task_types,
            "active_skills": self.active_skills,
        }


@dataclass
class WorldModel:
    """
    Complete world model for a domain.

    This is the agent's internal representation of its operating
    environment — constructed entirely from execution traces.
    """

    domain: str
    tool_profiles: list[ToolProfile] = field(default_factory=list)
    task_clusters: list[TaskCluster] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    transitions: list[StateTransition] = field(default_factory=list)
    health: DomainHealth = field(default_factory=DomainHealth)
    generated_at: float = field(default_factory=time.time)
    trace_count_at_generation: int = 0

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tool_profiles]

    @property
    def positive_tools(self) -> list[ToolProfile]:
        return [t for t in self.tool_profiles if t.causal_direction == "positive"]

    @property
    def negative_tools(self) -> list[ToolProfile]:
        return [t for t in self.tool_profiles if t.causal_direction == "negative"]

    @property
    def recommended_transitions(self) -> list[StateTransition]:
        return [t for t in self.transitions if t.is_recommended]

    @property
    def forbidden_transitions(self) -> list[StateTransition]:
        return [t for t in self.transitions if t.is_forbidden]

    @property
    def strong_constraints(self) -> list[Constraint]:
        return [c for c in self.constraints if c.confidence >= 0.8]

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "tool_profiles": [t.to_dict() for t in self.tool_profiles],
            "task_clusters": [c.to_dict() for c in self.task_clusters],
            "constraints": [c.to_dict() for c in self.constraints],
            "transitions": [t.to_dict() for t in self.transitions],
            "health": self.health.to_dict(),
            "generated_at": self.generated_at,
            "trace_count": self.trace_count_at_generation,
        }

    def to_prompt_injection(self) -> str:
        """
        Generate a system prompt section describing the domain model.

        This is injected into the agent's system prompt so it
        understands its environment without re-learning.
        """
        lines = ["DOMAIN KNOWLEDGE (learned from past experience):"]

        # Tool info
        if self.tool_profiles:
            lines.append("")
            lines.append("Tool capabilities:")
            for tp in sorted(self.tool_profiles, key=lambda t: t.success_rate, reverse=True):
                status = (
                    "✓ effective"
                    if tp.causal_direction == "positive"
                    else ("✗ avoid" if tp.causal_direction == "negative" else "~ neutral")
                )
                lines.append(
                    f"  • {tp.name}: {status}, "
                    f"success rate {tp.success_rate:.0%}, "
                    f"typically used at step {tp.typical_position:.0f}"
                )
                if tp.common_errors:
                    lines.append(f"    Watch for: {', '.join(tp.common_errors[:2])}")

        # Constraints
        strong = self.strong_constraints
        if strong:
            lines.append("")
            lines.append("Domain rules:")
            for c in strong[:5]:
                lines.append(f"  • {c.description}")

        # Recommended sequences
        rec = self.recommended_transitions
        if rec:
            lines.append("")
            lines.append("Recommended tool sequences:")
            for t in rec[:5]:
                lines.append(
                    f"  • After {t.from_state} → use {t.to_state} ({t.success_rate:.0%} success)"
                )

        # Forbidden transitions
        forbidden = self.forbidden_transitions
        if forbidden:
            lines.append("")
            lines.append("Avoid these sequences:")
            for t in forbidden[:5]:
                lines.append(f"  • After {t.from_state} → do NOT use {t.to_state}")

        # Task clusters
        if self.task_clusters:
            lines.append("")
            lines.append("Known task types:")
            for tc in self.task_clusters[:5]:
                kw = ", ".join(tc.keywords[:3])
                if tc.winning_sequences:
                    seq = " → ".join(tc.winning_sequences[0])
                    lines.append(
                        f"  • Tasks about [{kw}]: best approach is {seq} ({tc.success_rate:.0%})"
                    )
                else:
                    lines.append(f"  • Tasks about [{kw}]: {tc.success_rate:.0%} success rate")

        return "\n".join(lines)

    def to_compact_injection(self) -> str:
        """Short version for constrained context windows."""
        lines = ["DOMAIN RULES:"]

        for c in self.strong_constraints[:3]:
            lines.append(f"  • {c.description}")

        for t in self.forbidden_transitions[:3]:
            lines.append(f"  • Never: {t.from_state} → {t.to_state}")

        for t in self.recommended_transitions[:3]:
            lines.append(f"  • Prefer: {t.from_state} → {t.to_state}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# World Model Builder
# ---------------------------------------------------------------------------


class WorldModelBuilder:
    """
    Builds a WorldModel from accumulated execution traces.

    Usage:
        builder = WorldModelBuilder(memory)
        model = builder.build("my-domain")
        print(model.to_prompt_injection())
    """

    def __init__(
        self,
        memory: PersistentMemory,
        analyzer: Optional[TraceAnalyzer] = None,
        min_traces: int = 10,
    ) -> None:
        self.memory = memory
        self.analyzer = analyzer or TraceAnalyzer(memory)
        self.min_traces = min_traces

    def build(self, domain: str, max_traces: int = 500) -> WorldModel:
        """Build or rebuild the world model for a domain."""
        traces = self.memory.get_traces(domain, limit=max_traces)

        if len(traces) < self.min_traces:
            logger.info(
                "Domain '%s' has %d traces (need %d) — building minimal model",
                domain,
                len(traces),
                self.min_traces,
            )

        model = WorldModel(
            domain=domain,
            trace_count_at_generation=len(traces),
        )

        if not traces:
            return model

        # Run analyzer for causal data
        report = self.analyzer.full_analysis(domain, max_traces=max_traces)

        # Build each component
        model.tool_profiles = self._build_tool_profiles(traces, report)
        model.task_clusters = self._build_task_clusters(traces)
        model.constraints = self._discover_constraints(traces, report)
        model.transitions = self._build_transitions(report)
        model.health = self._compute_health(traces, report, domain)

        logger.info(
            "World model for '%s': %d tools, %d task clusters, %d constraints, %d transitions",
            domain,
            len(model.tool_profiles),
            len(model.task_clusters),
            len(model.constraints),
            len(model.transitions),
        )

        return model

    # ------------------------------------------------------------------
    # Tool profiles
    # ------------------------------------------------------------------

    def _build_tool_profiles(self, traces: list[dict], report: AnalysisReport) -> list[ToolProfile]:
        """Build detailed profiles for each tool from traces."""
        tool_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "total": 0,
                "in_success": 0,
                "in_failure": 0,
                "durations": [],
                "costs": [],
                "errors": [],
                "positions": [],
                "co_tools_success": Counter(),
                "co_tools_failure": Counter(),
            }
        )

        for t in traces:
            success = t.get("outcome_success", False)
            tools_in_trace = set()
            cost_per_step = t.get("total_cost", 0) / max(len(t.get("steps", [])), 1)

            for step in t.get("steps", []):
                name = step.get("tool_name")
                if not name:
                    continue

                tools_in_trace.add(name)
                d = tool_data[name]
                d["total"] += 1

                if success:
                    d["in_success"] += 1
                else:
                    d["in_failure"] += 1

                d["durations"].append(step.get("duration_ms", 0))
                d["costs"].append(cost_per_step)
                d["positions"].append(step.get("step_number", 0))

                if step.get("error"):
                    err = step["error"][:60]
                    if err not in d["errors"]:
                        d["errors"].append(err)

            # Track co-occurrence
            for name in tools_in_trace:
                others = tools_in_trace - {name}
                for other in others:
                    if success:
                        tool_data[name]["co_tools_success"][other] += 1
                    else:
                        tool_data[name]["co_tools_failure"][other] += 1

        # Build attribution lookup from report
        attr_map = {a.tool_name: a for a in report.step_attributions}

        profiles = []
        for name, d in tool_data.items():
            total = d["total"]
            if total == 0:
                continue

            s_rate = d["in_success"] / total
            err_rate = len(d["errors"]) / total if d["errors"] else 0

            # Find complementary tools (appear together in successes)
            complementary = [
                tool for tool, count in d["co_tools_success"].most_common(5) if count >= 3
            ]

            # Find redundant tools (appear together but don't help)
            redundant = []
            for tool, s_count in d["co_tools_success"].most_common(10):
                f_count = d["co_tools_failure"].get(tool, 0)
                if s_count > 0 and f_count > 0:
                    co_rate = s_count / (s_count + f_count)
                    if abs(co_rate - s_rate) < 0.05 and (s_count + f_count) >= 3:
                        redundant.append(tool)

            # Get causal direction from analyzer
            attr = attr_map.get(name)
            direction = attr.causal_direction if attr else "neutral"

            profiles.append(
                ToolProfile(
                    name=name,
                    total_invocations=total,
                    invocations_in_success=d["in_success"],
                    invocations_in_failure=d["in_failure"],
                    success_rate=round(s_rate, 4),
                    avg_duration_ms=round(sum(d["durations"]) / len(d["durations"]), 2)
                    if d["durations"]
                    else 0,
                    avg_cost_per_use=round(sum(d["costs"]) / len(d["costs"]), 6)
                    if d["costs"]
                    else 0,
                    common_errors=d["errors"][:5],
                    error_rate=round(err_rate, 4),
                    typical_position=round(sum(d["positions"]) / len(d["positions"]), 1)
                    if d["positions"]
                    else 0,
                    complementary_tools=complementary[:5],
                    redundant_with=redundant[:3],
                    causal_direction=direction,
                )
            )

        profiles.sort(key=lambda p: p.total_invocations, reverse=True)
        return profiles

    # ------------------------------------------------------------------
    # Task clustering
    # ------------------------------------------------------------------

    def _build_task_clusters(self, traces: list[dict], max_clusters: int = 10) -> list[TaskCluster]:
        """
        Cluster tasks by keyword similarity.

        Uses a simple keyword-based clustering: extract top keywords
        from each task, group tasks that share keywords.
        """
        if not traces:
            return []

        # Extract keywords per task
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "and",
            "or",
            "not",
            "it",
            "this",
            "that",
            "my",
            "your",
            "do",
            "does",
            "did",
            "be",
            "been",
            "have",
            "has",
            "had",
            "will",
            "can",
            "should",
        }

        task_keywords: list[tuple[dict, set[str]]] = []
        for t in traces:
            task = t.get("task", "").lower()
            words = set()
            for w in task.split():
                cleaned = w.strip(".,!?;:'\"()[]{}").strip()
                if len(cleaned) > 2 and cleaned not in stop_words:
                    words.add(cleaned)
            task_keywords.append((t, words))

        # Count keyword frequency
        kw_freq = Counter()
        for _, kws in task_keywords:
            kw_freq.update(kws)

        # Top keywords become cluster anchors
        top_keywords = [kw for kw, _ in kw_freq.most_common(max_clusters * 2)]

        # Assign each task to the cluster of its most frequent keyword
        cluster_map: dict[str, list[dict]] = defaultdict(list)
        assigned = set()
        for trace, kws in task_keywords:
            best_kw = None
            best_freq = 0
            for kw in kws:
                if kw in top_keywords:
                    freq = kw_freq[kw]
                    if freq > best_freq:
                        best_freq = freq
                        best_kw = kw
            if best_kw:
                cluster_map[best_kw].append(trace)
                assigned.add(trace["run_id"])

        # Unassigned go to "other"
        unassigned = [t for t, _ in task_keywords if t["run_id"] not in assigned]
        if unassigned:
            cluster_map["other"] = unassigned

        clusters = []
        for kw, cluster_traces in cluster_map.items():
            if len(cluster_traces) < 2:
                continue

            successes = [t for t in cluster_traces if t.get("outcome_success")]
            failures = [t for t in cluster_traces if not t.get("outcome_success")]
            s_rate = len(successes) / len(cluster_traces)

            # Find winning sequence for this cluster
            seq_counter = Counter()
            for t in successes:
                seq = tuple(t.get("tool_sequence", []))
                if seq:
                    seq_counter[seq] += 1

            winning = [list(seq) for seq, _ in seq_counter.most_common(3)]

            # Find related keywords
            related_kws = Counter()
            for t in cluster_traces:
                task = t.get("task", "").lower()
                for w in task.split():
                    cleaned = w.strip(".,!?;:'\"()[]{}").strip()
                    if len(cleaned) > 2 and cleaned not in stop_words and cleaned != kw:
                        related_kws[cleaned] += 1

            all_keywords = [kw] + [k for k, _ in related_kws.most_common(4)]

            clusters.append(
                TaskCluster(
                    cluster_id=f"tc-{kw}",
                    keywords=all_keywords,
                    task_count=len(cluster_traces),
                    success_count=len(successes),
                    failure_count=len(failures),
                    success_rate=round(s_rate, 4),
                    winning_sequences=winning,
                    avg_cost=round(
                        sum(t.get("total_cost", 0) for t in cluster_traces) / len(cluster_traces), 6
                    ),
                    avg_duration=round(
                        sum(t.get("total_duration", 0) for t in cluster_traces)
                        / len(cluster_traces),
                        4,
                    ),
                    example_tasks=[t.get("task", "")[:80] for t in cluster_traces[:5]],
                )
            )

        clusters.sort(key=lambda c: c.task_count, reverse=True)
        return clusters[:max_clusters]

    # ------------------------------------------------------------------
    # Constraint discovery
    # ------------------------------------------------------------------

    def _discover_constraints(self, traces: list[dict], report: AnalysisReport) -> list[Constraint]:
        """
        Automatically discover domain constraints from trace patterns.
        """
        constraints = []
        idx = 0
        successes = [t for t in traces if t.get("outcome_success")]
        failures = [t for t in traces if not t.get("outcome_success")]

        if not traces:
            return []

        # 1. Prerequisite constraints: tool X only succeeds after tool Y
        tool_pairs: dict[tuple, dict[str, int]] = defaultdict(
            lambda: {"with_prereq": 0, "without_prereq": 0, "total": 0}
        )
        for t in traces:
            seq = t.get("tool_sequence", [])
            success = t.get("outcome_success", False)
            for i in range(1, len(seq)):
                pair = (seq[i - 1], seq[i])
                tool_pairs[pair]["total"] += 1
                if success:
                    tool_pairs[pair]["with_prereq"] += 1
                else:
                    tool_pairs[pair]["without_prereq"] += 1

        for (prereq, tool), counts in tool_pairs.items():
            if prereq in ("__START__", "__END__"):
                continue
            total = counts["total"]
            if total < 5:
                continue
            s_rate = counts["with_prereq"] / total
            if s_rate > 0.85:
                idx += 1
                constraints.append(
                    Constraint(
                        constraint_id=f"c-{idx}",
                        constraint_type="prerequisite",
                        description=f"'{tool}' works best when preceded by '{prereq}' ({s_rate:.0%} success)",
                        confidence=round(s_rate, 2),
                        evidence_count=total,
                        involved_tools=[prereq, tool],
                    )
                )

        # 2. Forbidden transitions from dead ends
        for de in report.dead_end_transitions:
            idx += 1
            constraints.append(
                Constraint(
                    constraint_id=f"c-{idx}",
                    constraint_type="forbidden",
                    description=f"Never use '{de.to_tool}' after '{de.from_tool}' (0% success in {de.total_count} cases)",
                    confidence=1.0 if de.total_count >= 5 else 0.8,
                    evidence_count=de.total_count,
                    involved_tools=[de.from_tool, de.to_tool],
                )
            )

        # 3. Required minimum steps
        if successes:
            success_lengths = [len(t.get("tool_sequence", [])) for t in successes]
            min_length = min(success_lengths) if success_lengths else 0
            if min_length >= 2:
                short_failures = sum(
                    1 for t in failures if len(t.get("tool_sequence", [])) < min_length
                )
                if short_failures >= 3:
                    idx += 1
                    constraints.append(
                        Constraint(
                            constraint_id=f"c-{idx}",
                            constraint_type="required_minimum",
                            description=f"Successful resolution requires at least {min_length} tool calls",
                            confidence=0.7,
                            evidence_count=short_failures,
                        )
                    )

        # 4. Conditional error constraints
        tool_error_map: dict[str, Counter] = defaultdict(Counter)
        for t in failures:
            for step in t.get("steps", []):
                name = step.get("tool_name")
                err = step.get("error")
                if name and err:
                    tool_error_map[name][err[:60]] += 1

        for tool_name, errors in tool_error_map.items():
            for err, count in errors.most_common(3):
                if count >= 3:
                    idx += 1
                    constraints.append(
                        Constraint(
                            constraint_id=f"c-{idx}",
                            constraint_type="conditional_error",
                            description=f"'{tool_name}' frequently produces error: '{err}'",
                            confidence=min(0.9, count / 10),
                            evidence_count=count,
                            involved_tools=[tool_name],
                            involved_errors=[err],
                        )
                    )

        # 5. First-tool constraint
        if successes:
            first_tools = Counter()
            for t in successes:
                seq = t.get("tool_sequence", [])
                if seq:
                    first_tools[seq[0]] += 1

            if first_tools:
                best_first, best_count = first_tools.most_common(1)[0]
                ratio = best_count / len(successes)
                if ratio > 0.7 and best_count >= 5:
                    idx += 1
                    constraints.append(
                        Constraint(
                            constraint_id=f"c-{idx}",
                            constraint_type="prerequisite",
                            description=f"Successful runs typically start with '{best_first}' ({ratio:.0%} of successes)",
                            confidence=round(ratio, 2),
                            evidence_count=best_count,
                            involved_tools=[best_first],
                        )
                    )

        constraints.sort(key=lambda c: c.confidence, reverse=True)
        return constraints

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _build_transitions(self, report: AnalysisReport) -> list[StateTransition]:
        """
        Build semantic state transitions from the analyzer's
        transition matrix.
        """
        transitions = []
        for edge in report.transitions:
            is_recommended = edge.success_rate >= 0.8 and edge.total_count >= 5
            is_forbidden = edge.is_dead_end

            transitions.append(
                StateTransition(
                    from_state=edge.from_tool,
                    to_state=edge.to_tool,
                    count=edge.total_count,
                    success_rate=edge.success_rate,
                    is_recommended=is_recommended,
                    is_forbidden=is_forbidden,
                )
            )

        transitions.sort(key=lambda t: t.count, reverse=True)
        return transitions

    # ------------------------------------------------------------------
    # Health metrics
    # ------------------------------------------------------------------

    def _compute_health(
        self,
        traces: list[dict],
        report: AnalysisReport,
        domain: str,
    ) -> DomainHealth:
        """Compute overall domain health metrics."""
        if not traces:
            return DomainHealth()

        successes = sum(1 for t in traces if t.get("outcome_success"))
        costs = sorted(t.get("total_cost", 0) for t in traces)
        durations = sorted(t.get("total_duration", 0) for t in traces)

        def percentile(sorted_list: list[float], p: float) -> float:
            if not sorted_list:
                return 0.0
            k = (len(sorted_list) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_list[int(k)]
            return sorted_list[f] * (c - k) + sorted_list[c] * (k - f)

        # Count unique tools
        all_tools = set()
        for t in traces:
            for step in t.get("steps", []):
                name = step.get("tool_name")
                if name:
                    all_tools.add(name)

        # Get trend from report
        trend_data = report.performance_trend
        trend_status = trend_data.get("status", "stable")
        trend_slope = trend_data.get("slope", 0.0)

        # Count skills
        skills = self.memory.list_skills(domain)

        return DomainHealth(
            total_runs=len(traces),
            success_rate=round(successes / len(traces), 4),
            avg_cost=round(sum(costs) / len(costs), 6),
            avg_duration=round(sum(durations) / len(durations), 4),
            cost_p50=round(percentile(costs, 0.5), 6),
            cost_p95=round(percentile(costs, 0.95), 6),
            duration_p50=round(percentile(durations, 0.5), 4),
            duration_p95=round(percentile(durations, 0.95), 4),
            trend=trend_status,
            trend_slope=trend_slope,
            unique_tools=len(all_tools),
            unique_task_types=len(set(t.get("task", "")[:30] for t in traces)),
            active_skills=len(skills),
        )
