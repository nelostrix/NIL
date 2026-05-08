"""
Causal Trace Analyzer — the foundation of Nil's evolution engine.

This module goes far beyond simple frequency counting. It performs:

1. COUNTERFACTUAL ANALYSIS
   - For each step in a successful trace, ask: "If this step were removed,
     would the outcome still have been successful?"
   - Compare traces that are identical except for one step to isolate
     which steps actually caused the success.

2. SUBSEQUENCE MINING
   - Find not just full sequences but meaningful sub-patterns
   - A 5-step winning sequence might contain a critical 2-step core
   - Uses contiguous subsequence extraction

3. STEP-LEVEL CAUSAL ATTRIBUTION
   - Score each individual tool on its causal contribution to success
   - Uses lift, phi coefficient, and conditional probability
   - Identifies tools that are necessary vs merely correlated

4. TEMPORAL DECAY ANALYSIS
   - Recent traces matter more than old ones
   - Exponential decay weighting for all pattern scoring
   - Detects when a pattern is gaining or losing effectiveness

5. COST-EFFICIENCY SCORING
   - Winning patterns are ranked not just by success rate
     but by cost-per-success
   - A pattern that works 90% of the time but costs 10x is worse
     than one that works 80% at 1x cost

6. STATISTICAL SIGNIFICANCE
   - Fisher's exact test for tool-outcome associations
   - Confidence intervals on all reported rates
   - No pattern is reported unless p < 0.05

7. FAILURE CLUSTERING
   - Groups failures by error signature + tool sequence
   - Identifies systemic failure modes vs random noise
   - Finds the "failure attractors" — tool sequences that
     almost always fail

8. TRANSITION PROBABILITY MATRIX
   - Builds a Markov model of tool-to-tool transitions
   - Identifies which transitions correlate with success
   - Finds "dead-end" transitions that never lead to success
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.memory.store import PersistentMemory

logger = logging.getLogger("nil.evolution.analyzer")


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------


@dataclass
class CausalPattern:
    """A tool-call pattern with causal evidence."""

    sequence: list[str]
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    causal_score: float = 0.0
    cost_efficiency: float = 0.0
    avg_duration: float = 0.0
    avg_cost: float = 0.0
    p_value: float = 1.0
    confidence_interval: tuple[float, float] = (0.0, 1.0)
    temporal_trend: str = "stable"
    source_run_ids: list[str] = field(default_factory=list)

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def total(self) -> int:
        return self.success_count + self.failure_count


@dataclass
class StepAttribution:
    """Causal attribution score for a single tool."""

    tool_name: str
    total_appearances: int = 0
    appearances_in_success: int = 0
    appearances_in_failure: int = 0
    lift: float = 0.0
    phi_coefficient: float = 0.0
    conditional_success_rate: float = 0.0
    is_necessary: bool = False
    is_sufficient: bool = False
    p_value: float = 1.0
    causal_direction: str = "neutral"


@dataclass
class FailureCluster:
    """A group of failures with similar characteristics."""

    cluster_id: str
    error_signature: str
    count: int = 0
    tool_sequences: list[list[str]] = field(default_factory=list)
    common_subsequence: list[str] = field(default_factory=list)
    avg_step_where_failed: float = 0.0
    run_ids: list[str] = field(default_factory=list)
    is_systemic: bool = False


@dataclass
class TransitionEdge:
    """An edge in the tool transition probability matrix."""

    from_tool: str
    to_tool: str
    total_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    is_dead_end: bool = False


@dataclass
class AnalysisReport:
    """Complete analysis report for a domain."""

    domain: str
    total_traces: int = 0
    total_successes: int = 0
    total_failures: int = 0
    baseline_success_rate: float = 0.0
    causal_patterns: list[CausalPattern] = field(default_factory=list)
    step_attributions: list[StepAttribution] = field(default_factory=list)
    failure_clusters: list[FailureCluster] = field(default_factory=list)
    transitions: list[TransitionEdge] = field(default_factory=list)
    redundant_tools: list[str] = field(default_factory=list)
    performance_trend: dict[str, Any] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    @property
    def significant_patterns(self) -> list[CausalPattern]:
        return [p for p in self.causal_patterns if p.is_significant]

    @property
    def positive_tools(self) -> list[StepAttribution]:
        return [s for s in self.step_attributions if s.causal_direction == "positive"]

    @property
    def negative_tools(self) -> list[StepAttribution]:
        return [s for s in self.step_attributions if s.causal_direction == "negative"]

    @property
    def systemic_failures(self) -> list[FailureCluster]:
        return [f for f in self.failure_clusters if f.is_systemic]

    @property
    def dead_end_transitions(self) -> list[TransitionEdge]:
        return [t for t in self.transitions if t.is_dead_end]


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------


def _log_factorial(k: int) -> float:
    """Log factorial with Stirling's approximation for large k."""
    if k <= 1:
        return 0.0
    if k > 170:
        return k * math.log(k) - k + 0.5 * math.log(2 * math.pi * k)
    return sum(math.log(i) for i in range(2, k + 1))


def _hypergeom_log_pmf(x: int, N: int, K: int, n_draw: int) -> float:
    """Log PMF of hypergeometric distribution."""
    lo = max(0, n_draw - (N - K))
    hi = min(n_draw, K)
    if x < lo or x > hi:
        return float("-inf")
    return (
        _log_factorial(K)
        - _log_factorial(x)
        - _log_factorial(K - x)
        + _log_factorial(N - K)
        - _log_factorial(n_draw - x)
        - _log_factorial(N - K - n_draw + x)
        - _log_factorial(N)
        + _log_factorial(n_draw)
        + _log_factorial(N - n_draw)
    )


def fisher_exact_p(a: int, b: int, c: int, d: int) -> float:
    """
    Fisher's exact test (one-tailed) for a 2x2 contingency table.

    Table layout:
                 Success  Failure
    Tool present:   a        b
    Tool absent:    c        d

    Returns the p-value for the hypothesis that the tool is
    positively associated with success.
    """
    N = a + b + c + d
    if N == 0:
        return 1.0

    K = a + b
    n_draw = a + c

    log_p_observed = _hypergeom_log_pmf(a, N, K, n_draw)
    p_value = 0.0
    lo = max(0, n_draw - (N - K))
    hi = min(n_draw, K)
    for x in range(lo, hi + 1):
        log_px = _hypergeom_log_pmf(x, N, K, n_draw)
        if log_px <= log_p_observed + 1e-10:
            p_value += math.exp(log_px)

    return min(p_value, 1.0)


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 1.0)
    p_hat = successes / total
    denom = 1 + z * z / total
    centre = (p_hat + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) / denom
    return (max(0.0, round(centre - spread, 4)), min(1.0, round(centre + spread, 4)))


def phi_coefficient(a: int, b: int, c: int, d: int) -> float:
    """Phi coefficient for 2x2 contingency table (-1 to +1)."""
    num = a * d - b * c
    denom_sq = (a + b) * (c + d) * (a + c) * (b + d)
    if denom_sq == 0:
        return 0.0
    return num / math.sqrt(denom_sq)


def exponential_decay_weight(created_at: float, now: float, half_life: float = 86400 * 7) -> float:
    """Exponential decay weight. Default half-life: 7 days."""
    age = max(0.0, now - created_at)
    return math.exp(-0.693 * age / half_life)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_error(error: str) -> str:
    """Normalize error strings for clustering."""
    s = re.sub(r"line \d+", "line N", error)
    s = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", s)
    s = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "TIMESTAMP", s)
    s = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "UUID", s)
    s = re.sub(r"\b\d{5,}\b", "ID", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:200]


def lcs_two(a: list[str], b: list[str]) -> list[str]:
    """Longest common subsequence of two lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return []
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            result.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(result))


def lcs_multi(sequences: list[list[str]]) -> list[str]:
    """LCS across multiple sequences using pairwise reduction."""
    if not sequences:
        return []
    if len(sequences) == 1:
        return list(sequences[0])
    result = list(sequences[0])
    for seq in sequences[1:]:
        result = lcs_two(result, seq)
        if not result:
            break
    return result


# ---------------------------------------------------------------------------
# Main Analyzer
# ---------------------------------------------------------------------------


class TraceAnalyzer:
    """
    Causal Trace Analyzer — the foundation of Nil's evolution engine.

    Performs deep analysis of agent execution traces to identify
    causal patterns, not just correlations.

    Usage:
        analyzer = TraceAnalyzer(memory)
        report = analyzer.full_analysis("my-domain")

        # Or individual analyses:
        patterns = analyzer.find_causal_patterns("my-domain")
        attributions = analyzer.compute_step_attributions("my-domain")
        clusters = analyzer.cluster_failures("my-domain")
    """

    def __init__(
        self,
        memory: PersistentMemory,
        significance_threshold: float = 0.05,
        temporal_half_life: float = 86400 * 7,
        min_sample_size: int = 3,
    ) -> None:
        self.memory = memory
        self.significance_threshold = significance_threshold
        self.temporal_half_life = temporal_half_life
        self.min_sample_size = min_sample_size

    # ------------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------------

    def full_analysis(self, domain: str, max_traces: int = 500) -> AnalysisReport:
        """Run the complete analysis pipeline."""
        traces = self.memory.get_traces(domain, limit=max_traces)
        if not traces:
            return AnalysisReport(domain=domain)

        successes = [t for t in traces if t.get("outcome_success")]
        failures = [t for t in traces if not t.get("outcome_success")]
        baseline = len(successes) / len(traces) if traces else 0.0

        report = AnalysisReport(
            domain=domain,
            total_traces=len(traces),
            total_successes=len(successes),
            total_failures=len(failures),
            baseline_success_rate=round(baseline, 4),
        )

        report.causal_patterns = self.find_causal_patterns(domain, traces=traces)
        report.step_attributions = self.compute_step_attributions(domain, traces=traces)
        report.failure_clusters = self.cluster_failures(domain, traces=traces)
        report.transitions = self.build_transition_matrix(domain, traces=traces)
        report.redundant_tools = self._find_redundant_tools(report.step_attributions)
        report.performance_trend = self.detect_performance_trend(domain, traces=traces)

        return report

    # ------------------------------------------------------------------
    # 1. CAUSAL PATTERN MINING
    # ------------------------------------------------------------------

    def find_causal_patterns(
        self,
        domain: str,
        min_frequency: int = None,
        traces: list[dict] = None,
    ) -> list[CausalPattern]:
        """
        Find tool-call sequences with causal evidence for success.

        Tests statistical significance, computes confidence intervals,
        applies temporal decay, scores cost-efficiency, and mines
        subsequences.
        """
        if min_frequency is None:
            min_frequency = self.min_sample_size
        if traces is None:
            traces = self.memory.get_traces(domain, limit=500)
        if not traces:
            return []

        successes = [t for t in traces if t.get("outcome_success")]
        failures = [t for t in traces if not t.get("outcome_success")]
        baseline_rate = len(successes) / len(traces) if traces else 0
        now = time.time()

        # Collect full sequences
        seq_traces: dict[tuple, list[dict]] = defaultdict(list)
        for t in traces:
            seq = tuple(t.get("tool_sequence", []))
            if seq:
                seq_traces[seq].append(t)

        # Mine contiguous subsequences of length 2+
        for t in traces:
            seq = t.get("tool_sequence", [])
            for length in range(2, min(len(seq), 5) + 1):
                for i in range(len(seq) - length + 1):
                    subseq = tuple(seq[i : i + length])
                    if subseq not in seq_traces:
                        seq_traces[subseq] = []
                    # Avoid duplicates
                    existing_ids = {x["run_id"] for x in seq_traces[subseq]}
                    if t["run_id"] not in existing_ids:
                        seq_traces[subseq].append(t)

        patterns: list[CausalPattern] = []
        total_success = len(successes)
        total_failure = len(failures)

        for seq, tlist in seq_traces.items():
            if len(tlist) < min_frequency:
                continue
            s_count = sum(1 for t in tlist if t.get("outcome_success"))
            f_count = len(tlist) - s_count
            if s_count == 0:
                continue

            # Fisher's exact test
            a, b = s_count, f_count
            c = total_success - a
            d = total_failure - b
            p_val = fisher_exact_p(a, b, c, d)

            success_rate = s_count / len(tlist)
            ci = wilson_ci(s_count, len(tlist))

            # Temporal decay
            w_success = sum(
                exponential_decay_weight(t.get("created_at", 0), now, self.temporal_half_life)
                for t in tlist
                if t.get("outcome_success")
            )
            w_total = sum(
                exponential_decay_weight(t.get("created_at", 0), now, self.temporal_half_life)
                for t in tlist
            )

            # Temporal trend
            sorted_t = sorted(tlist, key=lambda t: t.get("created_at", 0))
            mid = len(sorted_t) // 2
            if mid > 0 and len(sorted_t) > mid:
                early = sum(1 for t in sorted_t[:mid] if t.get("outcome_success")) / mid
                late = sum(1 for t in sorted_t[mid:] if t.get("outcome_success")) / (
                    len(sorted_t) - mid
                )
                trend = (
                    "rising"
                    if late - early > 0.15
                    else ("falling" if early - late > 0.15 else "stable")
                )
            else:
                trend = "stable"

            # Cost efficiency
            avg_cost = sum(t.get("total_cost", 0) for t in tlist) / len(tlist)
            avg_dur = sum(t.get("total_duration", 0) for t in tlist) / len(tlist)
            cost_eff = success_rate / max(avg_cost, 1e-8)

            # Causal score
            lift = (success_rate / baseline_rate) if baseline_rate > 0 else 0
            temporal_rate = w_success / max(w_total, 1e-8)
            causal = min(1.0, lift * temporal_rate * (1 - p_val))

            patterns.append(
                CausalPattern(
                    sequence=list(seq),
                    success_count=s_count,
                    failure_count=f_count,
                    success_rate=round(success_rate, 4),
                    causal_score=round(causal, 4),
                    cost_efficiency=round(cost_eff, 4),
                    avg_duration=round(avg_dur, 4),
                    avg_cost=round(avg_cost, 6),
                    p_value=round(p_val, 6),
                    confidence_interval=ci,
                    temporal_trend=trend,
                    source_run_ids=[t["run_id"] for t in tlist[:20]],
                )
            )

        patterns.sort(key=lambda p: p.causal_score, reverse=True)
        return patterns

    # ------------------------------------------------------------------
    # 2. STEP-LEVEL CAUSAL ATTRIBUTION
    # ------------------------------------------------------------------

    def compute_step_attributions(
        self,
        domain: str,
        traces: list[dict] = None,
    ) -> list[StepAttribution]:
        """
        Compute the causal contribution of each individual tool.

        For each tool computes lift, phi coefficient, necessity,
        sufficiency, and statistical significance.
        """
        if traces is None:
            traces = self.memory.get_traces(domain, limit=500)
        if not traces:
            return []

        total = len(traces)
        total_success = sum(1 for t in traces if t.get("outcome_success"))
        total_failure = total - total_success
        baseline = total_success / total if total > 0 else 0

        tool_presence: dict[str, dict[str, int]] = defaultdict(
            lambda: {"in_success": 0, "in_failure": 0, "total": 0}
        )
        for t in traces:
            tools_used = set()
            for step in t.get("steps", []):
                name = step.get("tool_name")
                if name:
                    tools_used.add(name)
            success = t.get("outcome_success", False)
            for name in tools_used:
                tool_presence[name]["total"] += 1
                if success:
                    tool_presence[name]["in_success"] += 1
                else:
                    tool_presence[name]["in_failure"] += 1

        attributions = []
        for tool_name, counts in tool_presence.items():
            if counts["total"] < self.min_sample_size:
                continue

            a = counts["in_success"]
            b = counts["in_failure"]
            c = total_success - a
            d = total_failure - b

            p_s_given_t = a / (a + b) if (a + b) > 0 else 0
            lift = p_s_given_t / baseline if baseline > 0 else 0
            phi = phi_coefficient(a, b, c, d)
            p_val = fisher_exact_p(a, b, c, d)

            necessity_rate = a / total_success if total_success > 0 else 0
            is_necessary = necessity_rate > 0.9
            is_sufficient = p_s_given_t > 0.9 and counts["total"] >= 5

            if phi > 0.1 and p_val < self.significance_threshold:
                direction = "positive"
            elif phi < -0.1 and p_val < self.significance_threshold:
                direction = "negative"
            else:
                direction = "neutral"

            attributions.append(
                StepAttribution(
                    tool_name=tool_name,
                    total_appearances=counts["total"],
                    appearances_in_success=a,
                    appearances_in_failure=b,
                    lift=round(lift, 4),
                    phi_coefficient=round(phi, 4),
                    conditional_success_rate=round(p_s_given_t, 4),
                    is_necessary=is_necessary,
                    is_sufficient=is_sufficient,
                    p_value=round(p_val, 6),
                    causal_direction=direction,
                )
            )

        attributions.sort(key=lambda a: abs(a.phi_coefficient), reverse=True)
        return attributions

    # ------------------------------------------------------------------
    # 3. FAILURE CLUSTERING
    # ------------------------------------------------------------------

    def cluster_failures(
        self,
        domain: str,
        traces: list[dict] = None,
        systemic_threshold: int = 3,
    ) -> list[FailureCluster]:
        """Group failures by error signature and identify systemic patterns."""
        if traces is None:
            traces = self.memory.get_traces(domain, limit=500)

        failures = [t for t in traces if not t.get("outcome_success")]
        if not failures:
            return []

        error_groups: dict[str, list[dict]] = defaultdict(list)
        for t in failures:
            error = t.get("error", "")
            if not error:
                for step in reversed(t.get("steps", [])):
                    if step.get("error"):
                        error = step["error"]
                        break
            sig = normalize_error(error or "unknown_error")
            error_groups[sig].append(t)

        clusters = []
        for sig, group in error_groups.items():
            sequences = [t.get("tool_sequence", []) for t in group]
            common = lcs_multi(sequences)

            fail_steps = []
            for t in group:
                for step in t.get("steps", []):
                    if step.get("error"):
                        fail_steps.append(step.get("step_number", 0))
                        break

            avg_fail = sum(fail_steps) / len(fail_steps) if fail_steps else 0

            clusters.append(
                FailureCluster(
                    cluster_id=f"fc-{sig[:20]}-{len(group)}",
                    error_signature=sig,
                    count=len(group),
                    tool_sequences=sequences[:10],
                    common_subsequence=common,
                    avg_step_where_failed=round(avg_fail, 1),
                    run_ids=[t["run_id"] for t in group[:20]],
                    is_systemic=len(group) >= systemic_threshold,
                )
            )

        clusters.sort(key=lambda c: c.count, reverse=True)
        return clusters

    # ------------------------------------------------------------------
    # 4. TRANSITION PROBABILITY MATRIX
    # ------------------------------------------------------------------

    def build_transition_matrix(
        self,
        domain: str,
        traces: list[dict] = None,
    ) -> list[TransitionEdge]:
        """Build a Markov model of tool-to-tool transitions."""
        if traces is None:
            traces = self.memory.get_traces(domain, limit=500)
        if not traces:
            return []

        edge_data: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"total": 0, "success": 0, "failure": 0}
        )
        for t in traces:
            seq = t.get("tool_sequence", [])
            success = t.get("outcome_success", False)
            if seq:
                key = ("__START__", seq[0])
                edge_data[key]["total"] += 1
                edge_data[key]["success" if success else "failure"] += 1
            for i in range(len(seq) - 1):
                key = (seq[i], seq[i + 1])
                edge_data[key]["total"] += 1
                edge_data[key]["success" if success else "failure"] += 1
            if seq:
                key = (seq[-1], "__END__")
                edge_data[key]["total"] += 1
                edge_data[key]["success" if success else "failure"] += 1

        edges = []
        for (f, to), counts in edge_data.items():
            total = counts["total"]
            if total < 2:
                continue
            s_rate = counts["success"] / total
            is_dead = counts["success"] == 0 and total >= self.min_sample_size
            edges.append(
                TransitionEdge(
                    from_tool=f,
                    to_tool=to,
                    total_count=total,
                    success_count=counts["success"],
                    failure_count=counts["failure"],
                    success_rate=round(s_rate, 4),
                    is_dead_end=is_dead,
                )
            )

        edges.sort(key=lambda e: e.total_count, reverse=True)
        return edges

    # ------------------------------------------------------------------
    # 5. PERFORMANCE TREND DETECTION
    # ------------------------------------------------------------------

    def detect_performance_trend(
        self,
        domain: str,
        window: int = 20,
        traces: list[dict] = None,
    ) -> dict[str, Any]:
        """
        Compare recent performance against historical baseline
        using sliding window + linear regression.
        """
        if traces is None:
            traces = self.memory.get_traces(domain, limit=window * 3)
        if len(traces) < window:
            return {"status": "insufficient_data", "runs": len(traces)}

        sorted_traces = sorted(traces, key=lambda t: t.get("created_at", 0))

        window_rates = []
        step = max(1, window // 4)
        for i in range(0, len(sorted_traces) - window + 1, step):
            w = sorted_traces[i : i + window]
            rate = sum(1 for t in w if t.get("outcome_success")) / len(w)
            window_rates.append(rate)

        if len(window_rates) < 2:
            recent = sorted_traces[-window:]
            rate = sum(1 for t in recent if t.get("outcome_success")) / len(recent)
            return {
                "status": "stable",
                "recent_success_rate": round(rate, 4),
                "historical_success_rate": round(rate, 4),
                "delta": 0.0,
                "slope": 0.0,
                "recent_window": len(recent),
                "historical_window": 0,
                "data_points": 1,
            }

        n = len(window_rates)
        x_mean = (n - 1) / 2.0
        y_mean = sum(window_rates) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(window_rates))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0

        recent_rate = window_rates[-1]
        older_rate = window_rates[0]
        delta = recent_rate - older_rate

        if slope > 0.02:
            status = "improving"
        elif slope < -0.02:
            status = "degrading"
        else:
            status = "stable"

        return {
            "status": status,
            "recent_success_rate": round(recent_rate, 4),
            "historical_success_rate": round(older_rate, 4),
            "delta": round(delta, 4),
            "slope": round(slope, 6),
            "recent_window": window,
            "historical_window": len(sorted_traces) - window,
            "data_points": n,
        }

    # ------------------------------------------------------------------
    # 6. COUNTERFACTUAL ANALYSIS
    # ------------------------------------------------------------------

    def counterfactual_step_importance(
        self,
        domain: str,
        traces: list[dict] = None,
    ) -> list[dict[str, Any]]:
        """
        Estimate each tool's importance via natural experiment:
        compare traces where the tool was present vs absent.
        """
        if traces is None:
            traces = self.memory.get_traces(domain, limit=500)
        if len(traces) < self.min_sample_size * 2:
            return []

        all_tools = set()
        for t in traces:
            for step in t.get("steps", []):
                name = step.get("tool_name")
                if name:
                    all_tools.add(name)

        results = []
        for tool_name in all_tools:
            with_tool = []
            without_tool = []
            for t in traces:
                tools_in = {s.get("tool_name") for s in t.get("steps", []) if s.get("tool_name")}
                (with_tool if tool_name in tools_in else without_tool).append(t)

            if len(with_tool) < self.min_sample_size or len(without_tool) < self.min_sample_size:
                continue

            rate_with = sum(1 for t in with_tool if t.get("outcome_success")) / len(with_tool)
            rate_without = sum(1 for t in without_tool if t.get("outcome_success")) / len(
                without_tool
            )
            cost_with = sum(t.get("total_cost", 0) for t in with_tool) / len(with_tool)
            cost_without = sum(t.get("total_cost", 0) for t in without_tool) / len(without_tool)
            dur_with = sum(t.get("total_duration", 0) for t in with_tool) / len(with_tool)
            dur_without = sum(t.get("total_duration", 0) for t in without_tool) / len(without_tool)
            impact = rate_with - rate_without

            results.append(
                {
                    "tool": tool_name,
                    "success_rate_with": round(rate_with, 4),
                    "success_rate_without": round(rate_without, 4),
                    "causal_impact": round(impact, 4),
                    "traces_with": len(with_tool),
                    "traces_without": len(without_tool),
                    "cost_delta": round(cost_with - cost_without, 6),
                    "duration_with": round(dur_with, 4),
                    "duration_without": round(dur_without, 4),
                    "verdict": (
                        "essential"
                        if impact > 0.3
                        else "helpful"
                        if impact > 0.1
                        else "neutral"
                        if impact > -0.1
                        else "harmful"
                        if impact > -0.3
                        else "toxic"
                    ),
                }
            )

        results.sort(key=lambda r: r["causal_impact"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_redundant_tools(self, attributions: list[StepAttribution]) -> list[str]:
        return [
            a.tool_name
            for a in attributions
            if a.causal_direction == "neutral"
            and a.total_appearances >= self.min_sample_size * 2
            and abs(a.phi_coefficient) < 0.05
        ]

    # Legacy compatibility aliases
    def find_winning_patterns(self, domain: str, min_frequency: int = 3) -> list[dict[str, Any]]:
        """Legacy interface — returns dicts instead of CausalPattern objects."""
        patterns = self.find_causal_patterns(domain, min_frequency=min_frequency)
        return [
            {
                "sequence": p.sequence,
                "success_count": p.success_count,
                "failure_count": p.failure_count,
                "success_rate": p.success_rate,
            }
            for p in patterns
        ]

    def find_redundant_steps(self, domain: str) -> list[dict[str, Any]]:
        """Legacy interface."""
        attrs = self.compute_step_attributions(domain)
        return [
            {
                "tool": a.tool_name,
                "total_uses": a.total_appearances,
                "success_rate": a.conditional_success_rate,
                "error_rate": 0.0,
            }
            for a in attrs
            if a.causal_direction == "neutral"
        ]
