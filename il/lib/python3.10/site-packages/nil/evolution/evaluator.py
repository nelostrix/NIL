"""
Replay Evaluator — tests candidate strategies before deployment.

Before any mutation goes live, the evaluator answers:
"If the agent had used this new strategy on the last N tasks,
would it have performed better or worse?"

1. REPLAY SCORING
   - Takes a candidate strategy and historical traces
   - Scores the candidate by checking: would this strategy's
     recommendations have matched what actually succeeded?
   - Compares against the baseline strategy

2. STRATEGY COMPATIBILITY SCORING
   - Does the candidate's preferred tool sequence match
     successful historical sequences?
   - Does it avoid known failure patterns?
   - Does it align with the world model's constraints?

3. STATISTICAL SIGNIFICANCE
   - Paired sign test on per-trace scores
   - Requires p < 0.05 before recommending promotion
   - Reports confidence intervals on improvement

4. MULTI-DIMENSIONAL SCORING
   - Success alignment (would it have chosen winning patterns?)
   - Failure avoidance (does it avoid known failure patterns?)
   - Cost impact (would it have been cheaper?)
   - Constraint compliance (does it respect discovered rules?)

5. A/B COMPARISON
   - Side-by-side comparison of two strategies
   - Per-dimension breakdown
   - Clear promote/reject/inconclusive recommendation

6. ROLLBACK SAFETY
   - Evaluates rollback risk: if promotion fails, how bad?
   - Scores strategy reversibility
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.evolution.analyzer import AnalysisReport, TraceAnalyzer
from nil.evolution.world_model import WorldModel, WorldModelBuilder
from nil.memory.store import PersistentMemory

logger = logging.getLogger("nil.evolution.evaluator")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TraceScore:
    """Score for a single trace under a strategy."""

    run_id: str
    success_alignment: float = 0.0  # Did strategy match winning pattern?
    failure_avoidance: float = 0.0  # Does strategy avoid failure patterns?
    cost_score: float = 0.0  # Cost efficiency score
    constraint_compliance: float = 0.0  # Does strategy respect constraints?
    composite: float = 0.0  # Weighted combination

    def compute_composite(
        self,
        w_success: float = 0.4,
        w_failure: float = 0.3,
        w_cost: float = 0.15,
        w_constraint: float = 0.15,
    ) -> float:
        self.composite = (
            self.success_alignment * w_success
            + self.failure_avoidance * w_failure
            + self.cost_score * w_cost
            + self.constraint_compliance * w_constraint
        )
        return self.composite


@dataclass
class EvaluationResult:
    """Complete result of a strategy evaluation."""

    domain: str
    candidate_id: str = ""
    candidate_score: float = 0.0
    baseline_score: float = 0.0
    improvement: float = 0.0
    improvement_pct: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    recommendation: str = "inconclusive"  # "promote", "reject", "inconclusive"
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    traces_evaluated: int = 0
    candidate_wins: int = 0
    baseline_wins: int = 0
    ties: int = 0
    per_dimension: dict[str, dict[str, float]] = field(default_factory=dict)
    rollback_risk: str = "low"  # "low", "medium", "high"
    details: dict[str, Any] = field(default_factory=dict)
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "candidate_id": self.candidate_id,
            "candidate_score": round(self.candidate_score, 4),
            "baseline_score": round(self.baseline_score, 4),
            "improvement": round(self.improvement, 4),
            "improvement_pct": round(self.improvement_pct, 2),
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "recommendation": self.recommendation,
            "confidence_interval": self.confidence_interval,
            "traces_evaluated": self.traces_evaluated,
            "candidate_wins": self.candidate_wins,
            "baseline_wins": self.baseline_wins,
            "ties": self.ties,
            "per_dimension": self.per_dimension,
            "rollback_risk": self.rollback_risk,
        }


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------


def _sign_test_p(wins: int, losses: int) -> float:
    """
    Two-sided sign test p-value.

    Tests whether candidate wins significantly more than losses
    using the binomial distribution.
    """
    n = wins + losses
    if n == 0:
        return 1.0

    # Under H0, probability of each outcome is 0.5
    # P-value = 2 * P(X >= max(wins, losses)) for two-sided
    k = max(wins, losses)

    # Sum binomial probabilities for k..n
    p = 0.0
    for i in range(k, n + 1):
        log_prob = _log_comb(n, i) + i * math.log(0.5) + (n - i) * math.log(0.5)
        p += math.exp(log_prob)

    return min(1.0, 2 * p)  # Two-sided


def _log_comb(n: int, k: int) -> float:
    """Log of binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return float("-inf")
    if k == 0 or k == n:
        return 0.0
    # Use lgamma for numerical stability
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, round(centre - spread, 4)), min(1.0, round(centre + spread, 4)))


# ---------------------------------------------------------------------------
# Main Evaluator
# ---------------------------------------------------------------------------


class ReplayEvaluator:
    """
    Replay Evaluator — tests strategies against historical data.

    Usage:
        evaluator = ReplayEvaluator(memory)

        # Evaluate a candidate strategy
        result = evaluator.evaluate(
            domain="my-domain",
            candidate={"reasoning_hints": ["Always check logs first"]},
        )

        if result.recommendation == "promote":
            # Safe to deploy
            ...

        # Compare two strategies head-to-head
        result = evaluator.compare(
            domain="my-domain",
            strategy_a=old_strategy,
            strategy_b=new_strategy,
        )
    """

    def __init__(
        self,
        memory: PersistentMemory,
        analyzer: Optional[TraceAnalyzer] = None,
        significance_threshold: float = 0.05,
        min_traces_for_evaluation: int = 10,
    ) -> None:
        self.memory = memory
        self.analyzer = analyzer or TraceAnalyzer(memory)
        self.significance_threshold = significance_threshold
        self.min_traces = min_traces_for_evaluation

    def evaluate(
        self,
        domain: str,
        candidate: dict[str, Any],
        baseline: Optional[dict[str, Any]] = None,
        max_traces: int = 200,
        candidate_id: str = "",
    ) -> EvaluationResult:
        """
        Evaluate a candidate strategy against baseline using replay.

        If no baseline is provided, uses the current active strategy
        (or empty dict if none exists).
        """
        if baseline is None:
            baseline = self.memory.get_active_strategy(domain) or {}

        traces = self.memory.get_traces(domain, limit=max_traces)

        if len(traces) < self.min_traces:
            return EvaluationResult(
                domain=domain,
                candidate_id=candidate_id,
                recommendation="inconclusive",
                traces_evaluated=len(traces),
                details={"reason": f"Need at least {self.min_traces} traces, have {len(traces)}"},
            )

        # Build world model for constraint checking
        builder = WorldModelBuilder(self.memory, analyzer=self.analyzer, min_traces=1)
        world = builder.build(domain, max_traces=max_traces)

        # Score each trace under both strategies
        candidate_scores = []
        baseline_scores = []

        for trace in traces:
            c_score = self._score_trace(trace, candidate, world)
            b_score = self._score_trace(trace, baseline, world)
            candidate_scores.append(c_score)
            baseline_scores.append(b_score)

        # Aggregate scores
        c_avg = sum(s.composite for s in candidate_scores) / len(candidate_scores)
        b_avg = sum(s.composite for s in baseline_scores) / len(baseline_scores)
        improvement = c_avg - b_avg

        # Count wins/losses/ties
        wins, losses, ties = 0, 0, 0
        for cs, bs in zip(candidate_scores, baseline_scores):
            diff = cs.composite - bs.composite
            if diff > 0.01:
                wins += 1
            elif diff < -0.01:
                losses += 1
            else:
                ties += 1

        # Statistical significance
        p_value = _sign_test_p(wins, losses)
        is_sig = p_value < self.significance_threshold

        # Confidence interval on improvement
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5
        ci = _wilson_ci(wins, wins + losses)

        # Recommendation
        if is_sig and improvement > 0:
            recommendation = "promote"
        elif is_sig and improvement < 0:
            recommendation = "reject"
        else:
            recommendation = "inconclusive"

        # Per-dimension breakdown
        dims = self._per_dimension_breakdown(candidate_scores, baseline_scores)

        # Rollback risk
        rollback_risk = self._assess_rollback_risk(improvement, is_sig, traces)

        # Improvement percentage
        imp_pct = (improvement / max(abs(b_avg), 1e-8)) * 100

        return EvaluationResult(
            domain=domain,
            candidate_id=candidate_id,
            candidate_score=round(c_avg, 4),
            baseline_score=round(b_avg, 4),
            improvement=round(improvement, 4),
            improvement_pct=round(imp_pct, 2),
            p_value=round(p_value, 6),
            is_significant=is_sig,
            recommendation=recommendation,
            confidence_interval=ci,
            traces_evaluated=len(traces),
            candidate_wins=wins,
            baseline_wins=losses,
            ties=ties,
            per_dimension=dims,
            rollback_risk=rollback_risk,
            details={
                "candidate_keys": list(candidate.keys()),
                "baseline_keys": list(baseline.keys()),
                "world_model_constraints": len(world.constraints),
            },
        )

    def compare(
        self,
        domain: str,
        strategy_a: dict[str, Any],
        strategy_b: dict[str, Any],
        max_traces: int = 200,
    ) -> EvaluationResult:
        """
        Head-to-head comparison of two strategies.

        strategy_b is treated as the candidate, strategy_a as baseline.
        """
        return self.evaluate(
            domain=domain,
            candidate=strategy_b,
            baseline=strategy_a,
            max_traces=max_traces,
            candidate_id="strategy_b",
        )

    def evaluate_deltas(
        self,
        domain: str,
        deltas: list[dict[str, Any]],
        max_traces: int = 200,
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple strategy deltas independently.

        Returns a list of results, one per delta, sorted best-first.
        """
        baseline = self.memory.get_active_strategy(domain) or {}
        results = []

        for i, delta in enumerate(deltas):
            # Merge delta into baseline
            merged = dict(baseline)
            changes = delta.get("changes", delta)
            for key, val in changes.items():
                if isinstance(val, list) and isinstance(merged.get(key), list):
                    merged[key] = merged[key] + [v for v in val if v not in merged[key]]
                else:
                    merged[key] = val

            result = self.evaluate(
                domain=domain,
                candidate=merged,
                baseline=baseline,
                max_traces=max_traces,
                candidate_id=delta.get("delta_id", f"delta-{i}"),
            )
            results.append(result)

        results.sort(key=lambda r: r.improvement, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Trace scoring
    # ------------------------------------------------------------------

    def _score_trace(
        self,
        trace: dict[str, Any],
        strategy: dict[str, Any],
        world: WorldModel,
    ) -> TraceScore:
        """
        Score a single trace against a strategy.

        Answers: "How well does this strategy align with what
        happened in this trace?"
        """
        score = TraceScore(run_id=trace.get("run_id", ""))
        success = trace.get("outcome_success", False)
        tool_seq = trace.get("tool_sequence", [])

        # 1. Success alignment
        score.success_alignment = self._score_success_alignment(tool_seq, success, strategy, world)

        # 2. Failure avoidance
        score.failure_avoidance = self._score_failure_avoidance(tool_seq, success, strategy, world)

        # 3. Cost score
        score.cost_score = self._score_cost(trace, strategy)

        # 4. Constraint compliance
        score.constraint_compliance = self._score_constraints(tool_seq, strategy, world)

        score.compute_composite()
        return score

    def _score_success_alignment(
        self,
        tool_seq: list[str],
        success: bool,
        strategy: dict[str, Any],
        world: WorldModel,
    ) -> float:
        """
        Would this strategy have recommended the winning pattern?

        High score if:
        - Strategy's preferred tools match what succeeded
        - Strategy's reasoning hints align with successful approaches
        """
        if not success:
            return 0.5  # Neutral for failures

        score = 0.5  # Start neutral

        # Check preferred_first_tool
        preferred_first = strategy.get("preferred_first_tool")
        if preferred_first and tool_seq:
            if tool_seq[0] == preferred_first:
                score += 0.2
            else:
                score -= 0.1

        # Check skill_injections match the sequence
        injections = strategy.get("skill_injections", [])
        for inj in injections:
            # If any tool in sequence is mentioned in the injection
            inj_lower = inj.lower()
            matches = sum(1 for t in tool_seq if t.lower() in inj_lower)
            if matches > 0:
                score += 0.1 * min(matches, 3)

        # Check reasoning_hints relevance
        hints = strategy.get("reasoning_hints", [])
        for hint in hints:
            hint_lower = hint.lower()
            if any(t.lower() in hint_lower for t in tool_seq):
                score += 0.05

        return max(0.0, min(1.0, score))

    def _score_failure_avoidance(
        self,
        tool_seq: list[str],
        success: bool,
        strategy: dict[str, Any],
        world: WorldModel,
    ) -> float:
        """
        Does this strategy help avoid failure patterns?

        High score if:
        - Strategy's avoidance rules would have prevented this failure
        - Strategy warns about errors that occurred
        """
        if success:
            return 0.8  # Success = no failure to avoid

        score = 0.5

        # Check avoidance rules
        avoidance_rules = strategy.get("avoidance_rules", [])
        for rule in avoidance_rules:
            rule_lower = rule.lower()
            for tool in tool_seq:
                if tool.lower() in rule_lower:
                    score += 0.15  # Strategy would have warned
                    break

        # Check error handling hints
        error_hints = strategy.get("error_handling_hints", [])
        if error_hints:
            score += 0.1  # At least has error handling

        # Check against forbidden transitions
        for t in world.forbidden_transitions:
            for i in range(len(tool_seq) - 1):
                if tool_seq[i] == t.from_state and tool_seq[i + 1] == t.to_state:
                    # This trace hit a forbidden transition
                    # Score higher if strategy warns about it
                    for rule in avoidance_rules:
                        if t.to_state.lower() in rule.lower():
                            score += 0.2
                            break

        return max(0.0, min(1.0, score))

    def _score_cost(self, trace: dict[str, Any], strategy: dict[str, Any]) -> float:
        """Score based on cost efficiency."""
        cost = trace.get("total_cost", 0)
        if cost <= 0:
            return 0.5

        # If strategy has cost optimization hints, give bonus
        hints = strategy.get("reasoning_hints", [])
        has_efficiency = any(
            "concise" in h.lower() or "efficient" in h.lower() or "cost" in h.lower() for h in hints
        )

        # Lower cost = higher score (normalized)
        # Use log scale for cost scoring
        cost_score = max(0.0, 1.0 - math.log1p(cost * 100) / 5.0)

        if has_efficiency:
            cost_score = min(1.0, cost_score + 0.1)

        return max(0.0, min(1.0, cost_score))

    def _score_constraints(
        self,
        tool_seq: list[str],
        strategy: dict[str, Any],
        world: WorldModel,
    ) -> float:
        """Score how well the strategy respects domain constraints."""
        if not world.constraints:
            return 0.5

        total_constraints = 0
        satisfied = 0

        for c in world.strong_constraints:
            total_constraints += 1
            if c.constraint_type == "forbidden":
                # Check if strategy has avoidance for this
                avoidance = strategy.get("avoidance_rules", [])
                for tool in c.involved_tools:
                    for rule in avoidance:
                        if tool.lower() in rule.lower():
                            satisfied += 1
                            break
                    else:
                        continue
                    break

            elif c.constraint_type == "prerequisite":
                # Check if strategy's hints mention prerequisite
                hints = strategy.get("reasoning_hints", []) + strategy.get("skill_injections", [])
                for tool in c.involved_tools:
                    for hint in hints:
                        if tool.lower() in hint.lower():
                            satisfied += 1
                            break
                    else:
                        continue
                    break

        if total_constraints == 0:
            return 0.5

        return satisfied / total_constraints

    # ------------------------------------------------------------------
    # Per-dimension breakdown
    # ------------------------------------------------------------------

    def _per_dimension_breakdown(
        self,
        candidate_scores: list[TraceScore],
        baseline_scores: list[TraceScore],
    ) -> dict[str, dict[str, float]]:
        """Compute per-dimension averages for both strategies."""
        n = len(candidate_scores)
        if n == 0:
            return {}

        dims = {}
        for dim_name, getter in [
            ("success_alignment", lambda s: s.success_alignment),
            ("failure_avoidance", lambda s: s.failure_avoidance),
            ("cost_score", lambda s: s.cost_score),
            ("constraint_compliance", lambda s: s.constraint_compliance),
        ]:
            c_avg = sum(getter(s) for s in candidate_scores) / n
            b_avg = sum(getter(s) for s in baseline_scores) / n
            dims[dim_name] = {
                "candidate": round(c_avg, 4),
                "baseline": round(b_avg, 4),
                "delta": round(c_avg - b_avg, 4),
            }

        return dims

    # ------------------------------------------------------------------
    # Rollback risk assessment
    # ------------------------------------------------------------------

    def _assess_rollback_risk(
        self,
        improvement: float,
        is_significant: bool,
        traces: list[dict],
    ) -> str:
        """
        Assess how risky it is to promote this strategy.

        Low risk: significant improvement, large sample
        Medium risk: marginal improvement or small sample
        High risk: non-significant or negative improvement
        """
        if not is_significant:
            return "high"

        if improvement > 0.1 and len(traces) >= 50:
            return "low"
        elif improvement > 0.05 and len(traces) >= 20:
            return "medium"
        elif improvement > 0:
            return "medium"
        else:
            return "high"

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------

    def evaluate_legacy(
        self,
        domain: str,
        candidate: dict[str, Any],
        baseline: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Legacy interface returning a plain dict."""
        result = self.evaluate(domain, candidate, baseline)
        return result.to_dict()
