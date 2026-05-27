from __future__ import annotations

import logging
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.memory.concept import AbstractionLevel, Concept, ConceptGraph, ConceptType
from nil.memory.store import PersistentMemory

logger = logging.getLogger("nil.memory.abstraction")


@dataclass
class TraceCluster:
    label: str
    trace_ids: list[str] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    tool_sequences: list[list[str]] = field(default_factory=list)
    errors: Counter = field(default_factory=Counter)
    success_rate: float = 0.0
    common_tools: Counter = field(default_factory=Counter)
    avg_steps: float = 0.0

    @property
    def size(self) -> int:
        return len(self.traces)

    @property
    def dominant_errors(self) -> list[tuple[str, int]]:
        return self.errors.most_common(5)

    @property
    def dominant_tools(self) -> list[tuple[str, int]]:
        return self.common_tools.most_common(10)


@dataclass
class CausalPattern:
    antecedent: str
    consequent: str
    pattern_type: ConceptType
    confidence: float = 0.0
    support: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    cluster_label: str = ""

    def to_concept(self, domain: str, trace_ids: list[str]) -> Concept:
        return Concept(
            concept_id=str(uuid.uuid4()),
            domain=domain,
            concept_type=self.pattern_type,
            abstraction_level=AbstractionLevel.COMPRESSED,
            description=f"{self.antecedent} -> {self.consequent}",
            pattern_data={
                "antecedent": self.antecedent,
                "consequent": self.consequent,
                "pattern_type": self.pattern_type.value,
                "cluster_label": self.cluster_label,
                "context": self.context,
            },
            confidence=self.confidence,
            support_count=self.support,
            source_traces=trace_ids,
            tags=[self.pattern_type.value, self.cluster_label],
        )


@dataclass
class SymbolicConcept:
    description: str
    concept_type: ConceptType
    confidence: float = 0.0
    support: int = 0
    pattern_data: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source_traces: list[str] = field(default_factory=list)

    def to_concept(self, domain: str) -> Concept:
        return Concept(
            concept_id=str(uuid.uuid4()),
            domain=domain,
            concept_type=self.concept_type,
            abstraction_level=AbstractionLevel.SYMBOLIC,
            description=self.description,
            pattern_data=self.pattern_data,
            confidence=self.confidence,
            support_count=self.support,
            source_traces=self.source_traces,
            tags=self.tags,
        )


@dataclass
class InducedPolicy:
    description: str
    action: str
    condition: str = ""
    confidence: float = 0.0
    support: int = 0
    source_concepts: list[str] = field(default_factory=list)
    pattern_data: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source_traces: list[str] = field(default_factory=list)

    def to_concept(self, domain: str) -> Concept:
        return Concept(
            concept_id=str(uuid.uuid4()),
            domain=domain,
            concept_type=ConceptType.POLICY,
            abstraction_level=AbstractionLevel.POLICY,
            description=self.description,
            pattern_data={
                "action": self.action,
                "condition": self.condition,
                "source_concepts": self.source_concepts,
                **self.pattern_data,
            },
            confidence=self.confidence,
            support_count=self.support,
            source_traces=self.source_traces,
            tags=self.tags + ["policy"],
        )


@dataclass
class AbstractionReport:
    clusters_formed: int = 0
    causal_patterns: int = 0
    symbolic_concepts: int = 0
    policies_induced: int = 0
    total_concepts: int = 0
    traces_processed: int = 0
    duration_ms: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "clusters": self.clusters_formed,
            "causal_patterns": self.causal_patterns,
            "symbolic_concepts": self.symbolic_concepts,
            "policies": self.policies_induced,
            "total_concepts": self.total_concepts,
            "traces": self.traces_processed,
            "duration_ms": round(self.duration_ms, 1),
        }


class AbstractionEngine:
    def __init__(
        self,
        memory: PersistentMemory,
        min_cluster_size: int = 2,
        min_concept_confidence: float = 0.3,
        max_concepts_per_pass: int = 20,
    ):
        self.memory = memory
        self.min_cluster_size = min_cluster_size
        self.min_concept_confidence = min_concept_confidence
        self.max_concepts_per_pass = max_concepts_per_pass

    def run(self, domain: str, limit: int = 200) -> AbstractionReport:
        start = time.time()
        report = AbstractionReport()

        traces = self.memory.get_traces(domain, limit=limit)
        if not traces:
            logger.info("No traces to abstract for domain '%s'", domain)
            return report

        report.traces_processed = len(traces)

        clusters = self._cluster_traces(traces)
        report.clusters_formed = len(clusters)

        causal_patterns = self._compress_causal(clusters, domain, traces)
        report.causal_patterns = len(causal_patterns)

        symbolic_concepts = self._summarize_symbolic(causal_patterns, clusters, domain)
        report.symbolic_concepts = len(symbolic_concepts)

        all_concepts = causal_patterns + symbolic_concepts
        policies = self._induce_policies(all_concepts, domain)
        report.policies_induced = len(policies)

        final_concepts = self._persist_concepts(
            all_concepts + policies, domain
        )
        report.total_concepts = len(final_concepts)
        report.duration_ms = (time.time() - start) * 1000

        logger.info(
            "Abstraction for '%s': %d traces → %d clusters → "
            "%d causal → %d symbolic → %d policies (%d total, %.0fms)",
            domain, report.traces_processed, report.clusters_formed,
            report.causal_patterns, report.symbolic_concepts,
            report.policies_induced, report.total_concepts, report.duration_ms,
        )

        return report

    def _cluster_traces(self, traces: list[dict]) -> list[TraceCluster]:
        if len(traces) < self.min_cluster_size:
            return []

        clusters: dict[str, TraceCluster] = {}

        success_traces = [t for t in traces if t.get("outcome_success")]
        fail_traces = [t for t in traces if not t.get("outcome_success")]

        if success_traces:
            clusters["successful"] = self._build_cluster(
                "successful", success_traces
            )

        if fail_traces:
            clusters["failed"] = self._build_cluster(
                "failed", fail_traces
            )

        tool_clusters = self._cluster_by_tool_sequence(traces)
        for label, cluster_traces in tool_clusters.items():
            key = f"tool:{label}"
            if key not in clusters:
                clusters[key] = self._build_cluster(label, cluster_traces)

        error_clusters = self._cluster_by_error_pattern(fail_traces)
        for label, cluster_traces in error_clusters.items():
            key = f"error:{label}"
            if key not in clusters:
                clusters[key] = self._build_cluster(label, cluster_traces)

        return [
            c for c in clusters.values()
            if c.size >= self.min_cluster_size
        ]

    def _build_cluster(
        self, label: str, traces: list[dict]
    ) -> TraceCluster:
        trace_ids = [t.get("run_id", str(uuid.uuid4())) for t in traces]
        tool_seqs: list[list[str]] = []
        errors: Counter = Counter()
        all_tools: Counter = Counter()
        total_steps = 0

        for t in traces:
            seq = t.get("tool_sequence", [])
            if seq:
                tool_seqs.append(seq)
                for tool in seq:
                    all_tools[tool] += 1

            err = t.get("error")
            if err:
                key = err[:80]
                errors[key] += 1

            steps = t.get("steps", [])
            if isinstance(steps, list):
                total_steps += len(steps)

        successes = sum(1 for t in traces if t.get("outcome_success"))
        total = len(traces)

        return TraceCluster(
            label=label,
            trace_ids=trace_ids,
            traces=traces,
            tool_sequences=tool_seqs,
            errors=errors,
            success_rate=successes / total if total > 0 else 0.0,
            common_tools=all_tools,
            avg_steps=total_steps / total if total > 0 else 0.0,
        )

    def _cluster_by_tool_sequence(
        self, traces: list[dict]
    ) -> dict[str, list[dict]]:
        tool_seq_map: dict[str, list[dict]] = {}
        for t in traces:
            seq = t.get("tool_sequence", [])
            if seq:
                key = tuple(seq[:3])
                label = "_".join(seq[:3])[:60]
                if label not in tool_seq_map:
                    tool_seq_map[label] = []
                tool_seq_map[label].append(t)
        return {
            k: v for k, v in tool_seq_map.items()
            if len(v) >= self.min_cluster_size
        }

    def _cluster_by_error_pattern(
        self, fail_traces: list[dict]
    ) -> dict[str, list[dict]]:
        error_map: dict[str, list[dict]] = {}
        for t in fail_traces:
            err = t.get("error", "unknown")
            label = err[:40]
            if label not in error_map:
                error_map[label] = []
            error_map[label].append(t)
        return {
            k: v for k, v in error_map.items()
            if len(v) >= self.min_cluster_size
        }

    def _compress_causal(
        self,
        clusters: list[TraceCluster],
        domain: str,
        all_traces: list[dict],
    ) -> list[Concept]:
        patterns: list[Concept] = []

        for cluster in clusters:
            if cluster.size < 2:
                continue

            patterns.extend(self._find_tool_ordering_patterns(cluster, domain))
            patterns.extend(self._find_error_causal_patterns(cluster, domain))

        # Cross-cluster conditional analysis
        success_traces = [t for t in all_traces if t.get("outcome_success")]
        fail_traces = [t for t in all_traces if not t.get("outcome_success")]
        if len(success_traces) >= 2 and len(fail_traces) >= 1:
            patterns.extend(
                self._find_conditional_patterns(success_traces, fail_traces, domain)
            )

        patterns.sort(key=lambda c: c.confidence, reverse=True)
        return patterns[:self.max_concepts_per_pass]

    def _find_tool_ordering_patterns(
        self, cluster: TraceCluster, domain: str
    ) -> list[Concept]:
        concepts: list[Concept] = []
        seqs = cluster.tool_sequences

        if len(seqs) < 2:
            return concepts

        pair_counts: dict[tuple[str, str], int] = Counter()
        for seq in seqs:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += 1

        total = len(seqs)
        for (first, second), count in pair_counts.most_common(5):
            if first == second:
                continue
            ratio = min(1.0, count / total)
            concepts.append(Concept(
                concept_id=str(uuid.uuid4()),
                domain=domain,
                concept_type=ConceptType.ORDERING,
                abstraction_level=AbstractionLevel.COMPRESSED,
                description=f"\"{first}\" precedes \"{second}\" in {ratio:.0%} of traces",
                pattern_data={
                    "first_tool": first,
                    "second_tool": second,
                    "frequency": round(ratio, 3),
                    "cluster": cluster.label,
                },
                confidence=ratio,
                support_count=count,
                source_traces=cluster.trace_ids,
                tags=["ordering", cluster.label],
            ))

        return concepts

    def _find_error_causal_patterns(
        self, cluster: TraceCluster, domain: str
    ) -> list[Concept]:
        concepts: list[Concept] = []
        if cluster.success_rate > 0.3:
            return concepts

        for error, count in cluster.dominant_errors:
            ratio = count / cluster.size if cluster.size > 0 else 0
            if ratio < 0.2:
                continue

            common_before: Counter = Counter()
            for t in cluster.traces:
                err = t.get("error", "")
                if err and err[:80] == error[:80]:
                    seq = t.get("tool_sequence", [])
                    if seq:
                        common_before[seq[-1] if len(seq) > 0 else "none"] += 1

            before_tool = common_before.most_common(1)
            antecedent = f"after {before_tool[0][0]}" if before_tool else "unknown trigger"

            concepts.append(Concept(
                concept_id=str(uuid.uuid4()),
                domain=domain,
                concept_type=ConceptType.CAUSAL,
                abstraction_level=AbstractionLevel.COMPRESSED,
                description=f"{antecedent} causes \"{error[:60]}\" ({ratio:.0%} of failures)",
                pattern_data={
                    "error": error[:80],
                    "frequency_in_cluster": round(ratio, 3),
                    "antecedent": antecedent,
                    "cluster": cluster.label,
                },
                confidence=ratio * 0.8,
                support_count=count,
                source_traces=cluster.trace_ids,
                tags=["causal", "error", cluster.label],
            ))

        return concepts

    def _find_conditional_patterns(
        self,
        success_traces: list[dict],
        fail_traces: list[dict],
        domain: str,
    ) -> list[Concept]:
        concepts: list[Concept] = []

        success_tools: Counter = Counter()
        for t in success_traces:
            seq = t.get("tool_sequence", [])
            for tool in set(seq):
                success_tools[tool] += 1

        fail_tools: Counter = Counter()
        for t in fail_traces:
            seq = t.get("tool_sequence", [])
            for tool in set(seq):
                fail_tools[tool] += 1

        total_success = len(success_traces)
        total_fail = len(fail_traces)
        all_trace_ids = [t.get("run_id", "") for t in success_traces + fail_traces]

        for tool in success_tools:
            if tool in fail_tools:
                continue
            sr_ratio = success_tools[tool] / total_success
            if sr_ratio > 0.3:
                concepts.append(Concept(
                    concept_id=str(uuid.uuid4()),
                    domain=domain,
                    concept_type=ConceptType.CONDITIONAL,
                    abstraction_level=AbstractionLevel.COMPRESSED,
                    description=f"Using \"{tool}\" correlates with success ({sr_ratio:.0%} of successful runs, 0% of failures)",
                    pattern_data={
                        "tool": tool,
                        "success_rate_when_used": round(sr_ratio, 3),
                        "fail_rate_when_used": 0.0,
                    },
                    confidence=sr_ratio * 0.7,
                    support_count=success_tools[tool],
                    source_traces=all_trace_ids,
                    tags=["conditional"],
                ))

        # Check for tools used more in success than failure
        for tool in success_tools:
            if tool not in fail_tools:
                continue
            sr = success_tools[tool] / total_success
            fr = fail_tools[tool] / total_fail if total_fail > 0 else 1.0
            diff = sr - fr
            if diff > 0.3 and sr > 0.4:
                concepts.append(Concept(
                    concept_id=str(uuid.uuid4()),
                    domain=domain,
                    concept_type=ConceptType.CONDITIONAL,
                    abstraction_level=AbstractionLevel.COMPRESSED,
                    description=f"Using \"{tool}\" improves success rate ({sr:.0%} success vs {fr:.0%} failure)",
                    pattern_data={
                        "tool": tool,
                        "success_rate_when_used": round(sr, 3),
                        "fail_rate_when_used": round(fr, 3),
                        "improvement": round(diff, 3),
                    },
                    confidence=diff * 0.7,
                    support_count=success_tools[tool],
                    source_traces=all_trace_ids,
                    tags=["conditional"],
                ))

        return concepts

        if cluster.label in ("successful", "failed"):
            success_seqs = [
                t.get("tool_sequence", [])
                for t in cluster.traces if t.get("outcome_success")
            ]
            fail_seqs = [
                t.get("tool_sequence", [])
                for t in cluster.traces if not t.get("outcome_success")
            ]

            if len(success_seqs) >= 2 and len(fail_seqs) >= 2:
                success_tools = set()
                for seq in success_seqs:
                    success_tools.update(seq[:2])
                fail_tools = set()
                for seq in fail_seqs:
                    fail_tools.update(seq[:2])

                diff = success_tools - fail_tools
                if diff:
                    tool = list(diff)[0]
                    sr_success = sum(1 for seq in success_seqs if tool in seq)
                    sr_ratio = sr_success / len(success_seqs) if success_seqs else 0
                    if sr_ratio > 0.4:
                        concepts.append(Concept(
                            concept_id=str(uuid.uuid4()),
                            domain=domain,
                            concept_type=ConceptType.CONDITIONAL,
                            abstraction_level=AbstractionLevel.COMPRESSED,
                            description=f"Using \"{tool}\" correlates with success ({sr_ratio:.0%} of successful runs)",
                            pattern_data={
                                "tool": tool,
                                "success_rate_when_used": round(sr_ratio, 3),
                                "cluster": cluster.label,
                            },
                            confidence=sr_ratio * 0.7,
                            support_count=sr_success,
                            source_traces=cluster.trace_ids,
                            tags=["conditional", cluster.label],
                        ))

        return concepts

    def _summarize_symbolic(
        self,
        causal_concepts: list[Concept],
        clusters: list[TraceCluster],
        domain: str,
    ) -> list[Concept]:
        symbolic: list[Concept] = []
        seen_descriptions: set[str] = set()

        for cc in causal_concepts:
            if cc.confidence < self.min_concept_confidence:
                continue

            symbolic_desc = self._generate_symbolic_description(cc)
            if symbolic_desc and symbolic_desc not in seen_descriptions:
                seen_descriptions.add(symbolic_desc)
                symbolic.append(Concept(
                    concept_id=str(uuid.uuid4()),
                    domain=domain,
                    concept_type=cc.concept_type,
                    abstraction_level=AbstractionLevel.SYMBOLIC,
                    description=symbolic_desc,
                    pattern_data=cc.pattern_data,
                    confidence=cc.confidence * 0.9,
                    support_count=cc.support_count,
                    source_traces=cc.source_traces,
                    tags=cc.tags + ["symbolic"],
                ))

        for cluster in clusters:
            if cluster.success_rate > 0.8:
                desc = self._generate_high_success_description(cluster)
                if desc and desc not in seen_descriptions:
                    seen_descriptions.add(desc)
                    symbolic.append(Concept(
                        concept_id=str(uuid.uuid4()),
                        domain=domain,
                        concept_type=ConceptType.PATTERN,
                        abstraction_level=AbstractionLevel.SYMBOLIC,
                        description=desc,
                        pattern_data={
                            "cluster": cluster.label,
                            "success_rate": cluster.success_rate,
                            "top_tools": [t for t, _ in cluster.dominant_tools[:5]],
                            "avg_steps": cluster.avg_steps,
                        },
                        confidence=cluster.success_rate * 0.85,
                        support_count=cluster.size,
                        source_traces=cluster.trace_ids,
                        tags=["pattern", "high_success", cluster.label],
                    ))

            if cluster.errors and cluster.success_rate < 0.4:
                desc = self._generate_failure_description(cluster)
                if desc and desc not in seen_descriptions:
                    seen_descriptions.add(desc)
                    error_tools = [t for t, _ in cluster.dominant_tools[:3]]
                    symbolic.append(Concept(
                        concept_id=str(uuid.uuid4()),
                        domain=domain,
                        concept_type=ConceptType.PATTERN,
                        abstraction_level=AbstractionLevel.SYMBOLIC,
                        description=desc,
                        pattern_data={
                            "cluster": cluster.label,
                            "success_rate": cluster.success_rate,
                            "common_errors": [e for e, _ in cluster.dominant_errors[:3]],
                            "common_tools": error_tools,
                            "avg_steps": cluster.avg_steps,
                        },
                        confidence=(1.0 - cluster.success_rate) * 0.75,
                        support_count=cluster.size,
                        source_traces=cluster.trace_ids,
                        tags=["pattern", "high_failure", cluster.label],
                    ))

        return symbolic

    def _generate_symbolic_description(self, cc: Concept) -> str:
        pd = cc.pattern_data

        if cc.concept_type == ConceptType.ORDERING:
            first = pd.get("first_tool", "?")
            second = pd.get("second_tool", "?")
            freq = pd.get("frequency", 0)
            if freq > 0.7:
                return f"\"{first}\" works best before \"{second}\""
            return f"\"{first}\" usually precedes \"{second}\""

        if cc.concept_type == ConceptType.CAUSAL:
            ante = pd.get("antecedent", "?")
            err = pd.get("error", "?")
            return f"{ante} causes \"{err}\""

        if cc.concept_type == ConceptType.CONDITIONAL:
            tool = pd.get("tool", "?")
            rate = pd.get("success_rate_when_used", 0)
            fail_rate = pd.get("fail_rate_when_used", None)
            if fail_rate is not None and pd.get("improvement", 0) > 0.3:
                return f"Using \"{tool}\" improves accuracy ({rate:.0%} success vs {fail_rate:.0%} failure rate)"
            if rate > 0.7:
                return f"\"{tool}\" improves accuracy on uncertain tasks"
            return f"\"{tool}\" correlates with success ({rate:.0%})"

        return cc.description

    def _generate_high_success_description(self, cluster: TraceCluster) -> str:
        top_tools = [t for t, _ in cluster.dominant_tools[:3]]
        if not top_tools:
            return ""

        if len(top_tools) >= 2:
            return (
                f"Using {', '.join(top_tools[:-1])} and {top_tools[-1]} "
                f"achieves {cluster.success_rate:.0%} success rate "
                f"(avg {cluster.avg_steps:.1f} steps)"
            )
        return (
            f"Using \"{top_tools[0]}\" achieves "
            f"{cluster.success_rate:.0%} success rate"
        )

    def _generate_failure_description(self, cluster: TraceCluster) -> str:
        errs = [e for e, _ in cluster.dominant_errors[:2]]
        if not errs:
            return ""
        tools = [t for t, _ in cluster.dominant_tools[:2]]
        tool_part = f" after {', '.join(tools)}" if tools else ""
        return (
            f"Errors like \"{errs[0][:50]}\" occur{tool_part} "
            f"({cluster.size} failures observed)"
        )

    def _induce_policies(
        self,
        concepts: list[Concept],
        domain: str,
    ) -> list[Concept]:
        policies: list[Concept] = []
        high_conf = [c for c in concepts if c.confidence >= 0.5]

        ordering = [c for c in high_conf if c.concept_type == ConceptType.ORDERING]
        if ordering:
            top = max(ordering, key=lambda c: c.confidence)
            pd = top.pattern_data
            first = pd.get("first_tool", "")
            second = pd.get("second_tool", "")
            if first and second:
                policies.append(Concept(
                    concept_id=str(uuid.uuid4()),
                    domain=domain,
                    concept_type=ConceptType.POLICY,
                    abstraction_level=AbstractionLevel.POLICY,
                    description=f"ALWAYS run \"{first}\" before \"{second}\"",
                    pattern_data={
                        "action": f"call {first} before {second}",
                        "condition": f"when both {first} and {second} are needed",
                        "source_concepts": [top.concept_id],
                    },
                    confidence=top.confidence * 0.95,
                    support_count=top.support_count,
                    source_traces=top.source_traces,
                    tags=["policy", "ordering", "always"],
                ))

        conditionals = [c for c in high_conf if c.concept_type == ConceptType.CONDITIONAL]
        for cond in conditionals:
            tool = cond.pattern_data.get("tool", "")
            rate = cond.pattern_data.get("success_rate_when_used", 0)
            if tool and rate > 0.5:
                policies.append(Concept(
                    concept_id=str(uuid.uuid4()),
                    domain=domain,
                    concept_type=ConceptType.POLICY,
                    abstraction_level=AbstractionLevel.POLICY,
                    description=f"PREFER using \"{tool}\" when uncertain",
                    pattern_data={
                        "action": f"use {tool}",
                        "condition": "high uncertainty or complex task",
                        "source_concepts": [cond.concept_id],
                    },
                    confidence=cond.confidence * 0.9,
                    support_count=cond.support_count,
                    source_traces=cond.source_traces,
                    tags=["policy", "conditional", "prefer"],
                ))

        high_failure = [
            c for c in high_conf
            if c.concept_type == ConceptType.CAUSAL
        ]
        for fail in high_failure[:2]:
            err = fail.pattern_data.get("error", "")
            ante = fail.pattern_data.get("antecedent", "")
            if err:
                policies.append(Concept(
                    concept_id=str(uuid.uuid4()),
                    domain=domain,
                    concept_type=ConceptType.POLICY,
                    abstraction_level=AbstractionLevel.POLICY,
                    description=f"AVOID \"{err[:60]}\" by checking {ante}",
                    pattern_data={
                        "action": f"avoid trigger: {ante}",
                        "condition": f"when {ante} is detected",
                        "source_concepts": [fail.concept_id],
                    },
                    confidence=fail.confidence * 0.85,
                    support_count=fail.support_count,
                    source_traces=fail.source_traces,
                    tags=["policy", "avoid", "causal"],
                ))

        return policies

    def _persist_concepts(
        self,
        concepts: list[Concept],
        domain: str,
    ) -> list[Concept]:
        stored: list[Concept] = []
        seen: set[str] = set()

        for c in concepts:
            if c.description in seen:
                continue
            seen.add(c.description)

            if c.confidence < self.min_concept_confidence:
                continue

            try:
                self.memory.store_concept(c)
                stored.append(c)
            except Exception as e:
                logger.warning("Failed to persist concept: %s", e)

        return stored[:self.max_concepts_per_pass]
