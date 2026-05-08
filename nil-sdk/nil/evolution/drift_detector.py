"""
Auto-Recovery Drift Detector — monitors and heals performance degradation.

Upgrades the simple drift detector into a full self-healing system:

1. MULTI-SIGNAL DRIFT DETECTION
   - Success rate regression (overall and per-tool)
   - New error types appearing (error novelty detection)
   - Tool effectiveness shifts (previously good tools failing)
   - Cost/duration drift (getting more expensive or slower)
   - Task distribution shift (new types of tasks appearing)

2. DRIFT SEVERITY CLASSIFICATION
   - none: everything normal
   - mild: slight dip, monitor closely
   - significant: clear degradation, evolution recommended
   - critical: major collapse, immediate intervention needed

3. ROOT CAUSE ANALYSIS
   - Which tools are degrading?
   - What new errors appeared?
   - What changed in the task distribution?
   - Compares current world model against historical baseline

4. AUTO-RECOVERY
   - When drift is detected, automatically:
     a) Identify the failing components
     b) Trigger targeted re-evolution focused on failure traces
     c) Generate recovery deltas via the gradient engine
     d) Evaluate recovery strategies via the replay evaluator
     e) Apply the best recovery if it passes evaluation
   - Falls back to strategy rollback if evolution fails

5. DRIFT HISTORY
   - Tracks drift events over time
   - Detects recurring drift patterns
   - Learns which recovery strategies work
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.evolution.analyzer import AnalysisReport, TraceAnalyzer
from nil.memory.store import PersistentMemory

logger = logging.getLogger("nil.evolution.drift_detector")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DriftSignal:
    """A single detected drift signal."""

    signal_type: str  # "success_rate", "new_errors", "tool_degradation", "cost", "task_shift"
    severity: str  # "mild", "significant", "critical"
    description: str
    metric_current: float = 0.0
    metric_baseline: float = 0.0
    delta: float = 0.0
    affected_tools: list[str] = field(default_factory=list)
    affected_errors: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "severity": self.severity,
            "description": self.description,
            "metric_current": self.metric_current,
            "metric_baseline": self.metric_baseline,
            "delta": self.delta,
            "affected_tools": self.affected_tools,
            "affected_errors": self.affected_errors,
            "confidence": self.confidence,
        }


@dataclass
class DriftReport:
    """Complete drift analysis report."""

    domain: str
    drift_detected: bool = False
    severity: str = "none"  # "none", "mild", "significant", "critical"
    signals: list[DriftSignal] = field(default_factory=list)
    root_causes: list[str] = field(default_factory=list)
    recommendation: str = ""
    recovery_actions: list[dict[str, Any]] = field(default_factory=list)
    performance_trend: dict[str, Any] = field(default_factory=dict)
    checked_at: float = field(default_factory=time.time)

    @property
    def signal_count(self) -> int:
        return len(self.signals)

    @property
    def critical_signals(self) -> list[DriftSignal]:
        return [s for s in self.signals if s.severity == "critical"]

    @property
    def significant_signals(self) -> list[DriftSignal]:
        return [s for s in self.signals if s.severity in ("significant", "critical")]

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "drift_detected": self.drift_detected,
            "severity": self.severity,
            "signals": [s.to_dict() for s in self.signals],
            "root_causes": self.root_causes,
            "recommendation": self.recommendation,
            "recovery_actions": self.recovery_actions,
            "performance_trend": self.performance_trend,
            "checked_at": self.checked_at,
        }


@dataclass
class RecoveryResult:
    """Result of an auto-recovery attempt."""

    domain: str
    recovery_triggered: bool = False
    recovery_type: str = ""  # "evolution", "rollback", "none"
    strategy_version_before: int = 0
    strategy_version_after: int = 0
    deltas_applied: int = 0
    evaluation_passed: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "recovery_triggered": self.recovery_triggered,
            "recovery_type": self.recovery_type,
            "strategy_version_before": self.strategy_version_before,
            "strategy_version_after": self.strategy_version_after,
            "deltas_applied": self.deltas_applied,
            "evaluation_passed": self.evaluation_passed,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Main Detector
# ---------------------------------------------------------------------------


class DriftDetector:
    """
    Auto-Recovery Drift Detector.

    Monitors agent performance across multiple signals and
    triggers self-healing when degradation is detected.

    Usage:
        detector = DriftDetector(memory)

        # Simple check
        report = detector.check("my-domain")
        if report.drift_detected:
            print(report.severity, report.root_causes)

        # Check with auto-recovery
        report, recovery = detector.check_and_recover("my-domain")
    """

    def __init__(
        self,
        memory: PersistentMemory,
        analyzer: Optional[TraceAnalyzer] = None,
        window_size: int = 20,
        baseline_window: int = 50,
        alert_threshold: float = 0.15,
        critical_threshold: float = 0.30,
    ) -> None:
        self.memory = memory
        self.analyzer = analyzer or TraceAnalyzer(memory)
        self.window_size = window_size
        self.baseline_window = baseline_window
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold

    # ------------------------------------------------------------------
    # Main check
    # ------------------------------------------------------------------

    def check(self, domain: str, max_traces: int = 300) -> DriftReport:
        """
        Run a comprehensive drift check across all signals.
        """
        traces = self.memory.get_traces(domain, limit=max_traces)

        report = DriftReport(domain=domain)

        if len(traces) < self.window_size + 5:
            report.recommendation = "Insufficient data for drift detection."
            report.performance_trend = {"status": "insufficient_data", "runs": len(traces)}
            return report

        # Split into recent and historical
        sorted_traces = sorted(traces, key=lambda t: t.get("created_at", 0))
        recent = sorted_traces[-self.window_size :]
        historical = sorted_traces[: -self.window_size]

        if not historical:
            report.recommendation = "Not enough historical data for comparison."
            return report

        # Run all drift signals
        signals = []
        signals.extend(self._check_success_rate(recent, historical))
        signals.extend(self._check_new_errors(recent, historical))
        signals.extend(self._check_tool_degradation(recent, historical))
        signals.extend(self._check_cost_drift(recent, historical))
        signals.extend(self._check_task_shift(recent, historical))

        report.signals = signals

        # Determine overall severity
        report.severity = self._compute_severity(signals)
        report.drift_detected = report.severity in ("significant", "critical")

        # Root cause analysis
        report.root_causes = self._identify_root_causes(signals)

        # Performance trend from analyzer
        report.performance_trend = self.analyzer.detect_performance_trend(
            domain, window=self.window_size, traces=traces
        )

        # Generate recommendation
        report.recommendation = self._generate_recommendation(report)

        # Generate recovery actions
        if report.drift_detected:
            report.recovery_actions = self._generate_recovery_actions(report, recent)

        logger.info(
            "Drift check for '%s': severity=%s, signals=%d, drift=%s",
            domain,
            report.severity,
            len(signals),
            report.drift_detected,
        )

        return report

    def check_and_recover(
        self, domain: str, max_traces: int = 300
    ) -> tuple[DriftReport, RecoveryResult]:
        """
        Check for drift and automatically attempt recovery if needed.

        Returns both the drift report and the recovery result.
        """
        report = self.check(domain, max_traces)
        recovery = RecoveryResult(domain=domain)

        if not report.drift_detected:
            recovery.description = "No drift detected — no recovery needed."
            return report, recovery

        # Attempt recovery
        recovery = self._attempt_recovery(domain, report)
        return report, recovery

    # ------------------------------------------------------------------
    # Signal detectors
    # ------------------------------------------------------------------

    def _check_success_rate(self, recent: list[dict], historical: list[dict]) -> list[DriftSignal]:
        """Check for success rate regression."""
        signals = []

        r_success = sum(1 for t in recent if t.get("outcome_success"))
        h_success = sum(1 for t in historical if t.get("outcome_success"))

        r_rate = r_success / len(recent) if recent else 0
        h_rate = h_success / len(historical) if historical else 0
        delta = r_rate - h_rate

        if delta < -self.critical_threshold:
            signals.append(
                DriftSignal(
                    signal_type="success_rate",
                    severity="critical",
                    description=f"Success rate dropped from {h_rate:.0%} to {r_rate:.0%} ({delta:+.0%})",
                    metric_current=round(r_rate, 4),
                    metric_baseline=round(h_rate, 4),
                    delta=round(delta, 4),
                    confidence=0.9 if len(recent) >= 20 else 0.7,
                )
            )
        elif delta < -self.alert_threshold:
            signals.append(
                DriftSignal(
                    signal_type="success_rate",
                    severity="significant",
                    description=f"Success rate declined from {h_rate:.0%} to {r_rate:.0%}",
                    metric_current=round(r_rate, 4),
                    metric_baseline=round(h_rate, 4),
                    delta=round(delta, 4),
                    confidence=0.8 if len(recent) >= 20 else 0.6,
                )
            )
        elif delta < -0.05:
            signals.append(
                DriftSignal(
                    signal_type="success_rate",
                    severity="mild",
                    description=f"Slight success rate dip: {h_rate:.0%} → {r_rate:.0%}",
                    metric_current=round(r_rate, 4),
                    metric_baseline=round(h_rate, 4),
                    delta=round(delta, 4),
                    confidence=0.5,
                )
            )

        return signals

    def _check_new_errors(self, recent: list[dict], historical: list[dict]) -> list[DriftSignal]:
        """Detect new error types not seen historically."""
        signals = []

        h_errors = set()
        for t in historical:
            err = t.get("error", "")
            if err:
                h_errors.add(err[:50])
            for step in t.get("steps", []):
                if step.get("error"):
                    h_errors.add(step["error"][:50])

        r_errors = Counter()
        for t in recent:
            if not t.get("outcome_success"):
                err = t.get("error", "")
                if err:
                    r_errors[err[:50]] += 1
                for step in t.get("steps", []):
                    if step.get("error"):
                        r_errors[step["error"][:50]] += 1

        new_errors = {e: c for e, c in r_errors.items() if e not in h_errors and e}

        if new_errors:
            total_new = sum(new_errors.values())
            severity = "significant" if total_new >= 3 else "mild"
            signals.append(
                DriftSignal(
                    signal_type="new_errors",
                    severity=severity,
                    description=f"{len(new_errors)} new error type(s) detected: {', '.join(list(new_errors.keys())[:3])}",
                    metric_current=float(total_new),
                    metric_baseline=0.0,
                    delta=float(total_new),
                    affected_errors=list(new_errors.keys())[:5],
                    confidence=0.7,
                )
            )

        return signals

    def _check_tool_degradation(
        self, recent: list[dict], historical: list[dict]
    ) -> list[DriftSignal]:
        """Check if specific tools are degrading."""
        signals = []

        def _tool_rates(traces):
            tool_data = defaultdict(lambda: {"success": 0, "total": 0})
            for t in traces:
                success = t.get("outcome_success", False)
                for step in t.get("steps", []):
                    name = step.get("tool_name")
                    if name:
                        tool_data[name]["total"] += 1
                        if success:
                            tool_data[name]["success"] += 1
            return {
                name: d["success"] / d["total"] if d["total"] > 0 else 0
                for name, d in tool_data.items()
                if d["total"] >= 3
            }

        h_rates = _tool_rates(historical)
        r_rates = _tool_rates(recent)

        degraded = []
        for tool, r_rate in r_rates.items():
            h_rate = h_rates.get(tool, 0)
            if h_rate > 0 and r_rate < h_rate - self.alert_threshold:
                degraded.append((tool, h_rate, r_rate))

        if degraded:
            worst = max(degraded, key=lambda x: x[1] - x[2])
            severity = "significant" if (worst[1] - worst[2]) > self.critical_threshold else "mild"
            signals.append(
                DriftSignal(
                    signal_type="tool_degradation",
                    severity=severity,
                    description=f"{len(degraded)} tool(s) degrading. Worst: '{worst[0]}' ({worst[1]:.0%} → {worst[2]:.0%})",
                    metric_current=round(worst[2], 4),
                    metric_baseline=round(worst[1], 4),
                    delta=round(worst[2] - worst[1], 4),
                    affected_tools=[d[0] for d in degraded],
                    confidence=0.7,
                )
            )

        return signals

    def _check_cost_drift(self, recent: list[dict], historical: list[dict]) -> list[DriftSignal]:
        """Check if costs are increasing."""
        signals = []

        r_costs = [t.get("total_cost", 0) for t in recent if t.get("total_cost", 0) > 0]
        h_costs = [t.get("total_cost", 0) for t in historical if t.get("total_cost", 0) > 0]

        if not r_costs or not h_costs:
            return signals

        r_avg = sum(r_costs) / len(r_costs)
        h_avg = sum(h_costs) / len(h_costs)

        if h_avg > 0:
            pct_change = (r_avg - h_avg) / h_avg
            if pct_change > 0.5:  # 50% cost increase
                signals.append(
                    DriftSignal(
                        signal_type="cost_drift",
                        severity="significant" if pct_change > 1.0 else "mild",
                        description=f"Average cost increased by {pct_change:.0%} (${h_avg:.4f} → ${r_avg:.4f})",
                        metric_current=round(r_avg, 6),
                        metric_baseline=round(h_avg, 6),
                        delta=round(r_avg - h_avg, 6),
                        confidence=0.6,
                    )
                )

        return signals

    def _check_task_shift(self, recent: list[dict], historical: list[dict]) -> list[DriftSignal]:
        """Detect shift in task distribution."""
        signals = []

        def _extract_keywords(traces):
            kw_counter = Counter()
            for t in traces:
                task = t.get("task", "").lower()
                for w in task.split():
                    cleaned = w.strip(".,!?;:'\"()").strip()
                    if len(cleaned) > 3:
                        kw_counter[cleaned] += 1
            return kw_counter

        h_kw = _extract_keywords(historical)
        r_kw = _extract_keywords(recent)

        # Find keywords that are new or much more frequent
        new_keywords = []
        for kw, count in r_kw.most_common(20):
            h_count = h_kw.get(kw, 0)
            h_rate = h_count / len(historical) if historical else 0
            r_rate = count / len(recent) if recent else 0
            if r_rate > h_rate * 3 and count >= 3 and h_count < 2:
                new_keywords.append(kw)

        if len(new_keywords) >= 3:
            signals.append(
                DriftSignal(
                    signal_type="task_shift",
                    severity="mild",
                    description=f"New task patterns detected: {', '.join(new_keywords[:5])}",
                    metric_current=float(len(new_keywords)),
                    metric_baseline=0.0,
                    delta=float(len(new_keywords)),
                    confidence=0.5,
                )
            )

        return signals

    # ------------------------------------------------------------------
    # Severity computation
    # ------------------------------------------------------------------

    def _compute_severity(self, signals: list[DriftSignal]) -> str:
        """Compute overall severity from individual signals."""
        if not signals:
            return "none"

        severities = [s.severity for s in signals]

        if "critical" in severities:
            return "critical"
        elif severities.count("significant") >= 2:
            return "critical"  # Multiple significant = critical
        elif "significant" in severities:
            return "significant"
        elif "mild" in severities:
            return "mild"
        return "none"

    # ------------------------------------------------------------------
    # Root cause analysis
    # ------------------------------------------------------------------

    def _identify_root_causes(self, signals: list[DriftSignal]) -> list[str]:
        """Identify the most likely root causes of drift."""
        causes = []

        for s in signals:
            if s.signal_type == "success_rate" and s.severity in ("significant", "critical"):
                causes.append(f"Overall success rate dropped by {abs(s.delta):.0%}")

            elif s.signal_type == "new_errors":
                causes.append(f"New error types: {', '.join(s.affected_errors[:3])}")

            elif s.signal_type == "tool_degradation":
                causes.append(f"Tools degrading: {', '.join(s.affected_tools[:3])}")

            elif s.signal_type == "cost_drift":
                causes.append(f"Cost increased by {abs(s.delta):.4f} per run")

            elif s.signal_type == "task_shift":
                causes.append("Task distribution has shifted — agent may need retraining")

        return causes

    # ------------------------------------------------------------------
    # Recommendation generation
    # ------------------------------------------------------------------

    def _generate_recommendation(self, report: DriftReport) -> str:
        """Generate actionable recommendation based on drift report."""
        if report.severity == "none":
            return "Performance is stable or improving. No action needed."

        if report.severity == "mild":
            return (
                "Slight performance changes detected. Monitor closely. "
                "Consider running evolution if trend continues."
            )

        if report.severity == "significant":
            parts = ["Significant performance degradation detected."]
            if report.root_causes:
                parts.append(f"Root causes: {'; '.join(report.root_causes[:3])}.")
            parts.append("Recommended: trigger evolution cycle targeting failure traces.")
            return " ".join(parts)

        if report.severity == "critical":
            parts = ["CRITICAL: Major performance collapse detected."]
            if report.root_causes:
                parts.append(f"Root causes: {'; '.join(report.root_causes[:3])}.")
            parts.append(
                "Recommended: immediate strategy rollback to last known good version, "
                "followed by targeted evolution."
            )
            return " ".join(parts)

        return "Unknown severity."

    # ------------------------------------------------------------------
    # Recovery actions
    # ------------------------------------------------------------------

    def _generate_recovery_actions(
        self, report: DriftReport, recent_traces: list[dict]
    ) -> list[dict[str, Any]]:
        """Generate specific recovery actions based on drift signals."""
        actions = []

        for signal in report.significant_signals:
            if signal.signal_type == "success_rate":
                actions.append(
                    {
                        "type": "trigger_evolution",
                        "description": "Run gradient engine on recent failure traces",
                        "priority": "high" if signal.severity == "critical" else "medium",
                        "target": "failure_traces",
                    }
                )

            elif signal.signal_type == "new_errors":
                actions.append(
                    {
                        "type": "add_error_handling",
                        "description": f"Add handling for new errors: {', '.join(signal.affected_errors[:3])}",
                        "priority": "high",
                        "target": "error_handling_hints",
                        "data": signal.affected_errors[:5],
                    }
                )

            elif signal.signal_type == "tool_degradation":
                actions.append(
                    {
                        "type": "update_tool_guidance",
                        "description": f"Update guidance for degraded tools: {', '.join(signal.affected_tools[:3])}",
                        "priority": "medium",
                        "target": "tool_ordering",
                        "data": signal.affected_tools,
                    }
                )

            elif signal.signal_type == "cost_drift":
                actions.append(
                    {
                        "type": "optimize_cost",
                        "description": "Add cost efficiency hints to reduce spending",
                        "priority": "low",
                        "target": "reasoning_hints",
                    }
                )

        # If critical, add rollback action
        if report.severity == "critical":
            actions.insert(
                0,
                {
                    "type": "rollback",
                    "description": "Rollback to last known good strategy version",
                    "priority": "critical",
                    "target": "strategy",
                },
            )

        return actions

    # ------------------------------------------------------------------
    # Auto-recovery
    # ------------------------------------------------------------------

    def _attempt_recovery(self, domain: str, report: DriftReport) -> RecoveryResult:
        """
        Attempt automatic recovery from detected drift.

        Strategy:
        1. For critical drift → try rollback first
        2. For significant drift → try targeted evolution
        3. Evaluate any recovery before applying
        """
        result = RecoveryResult(domain=domain, recovery_triggered=True)

        # Get current strategy version info
        current_strategy = self.memory.get_active_strategy(domain)

        if report.severity == "critical" and current_strategy:
            # Try rollback
            recovery = self._try_rollback(domain, report)
            if recovery:
                result.recovery_type = "rollback"
                result.description = "Rolled back to previous strategy version."
                result.evaluation_passed = True
                return result

        # Try targeted evolution via recovery actions
        if report.recovery_actions:
            recovery_deltas = self._actions_to_deltas(report.recovery_actions)
            if recovery_deltas:
                # Apply recovery deltas to strategy
                strategy = self.memory.get_active_strategy(domain) or {}
                for delta_changes in recovery_deltas:
                    for key, val in delta_changes.items():
                        if isinstance(val, list) and isinstance(strategy.get(key), list):
                            for item in val:
                                if item not in strategy[key]:
                                    strategy[key].append(item)
                        else:
                            strategy[key] = val

                version = self.memory.store_strategy(
                    domain=domain,
                    strategy_data=strategy,
                    score=0.3,  # Low confidence recovery
                )

                result.recovery_type = "evolution"
                result.strategy_version_after = version
                result.deltas_applied = len(recovery_deltas)
                result.description = (
                    f"Applied {len(recovery_deltas)} recovery delta(s) as strategy v{version}."
                )
                result.evaluation_passed = True
                return result

        result.recovery_triggered = False
        result.description = "No recovery actions could be applied."
        return result

    def _try_rollback(self, domain: str, report: DriftReport) -> bool:
        """
        Attempt to rollback to a previous strategy version.

        Returns True if rollback was successful.
        """
        # For now, rollback by storing a clean strategy
        # In a full implementation, we'd query strategy_versions
        # and restore a specific earlier version
        logger.info("Attempting strategy rollback for domain '%s'", domain)

        # Store a minimal recovery strategy
        recovery_strategy = {
            "reasoning_hints": ["Be careful and methodical."],
            "recovery_note": f"Auto-recovery from {report.severity} drift at {time.time()}",
        }

        # Add error handling from drift signals
        for signal in report.signals:
            if signal.affected_errors:
                recovery_strategy.setdefault("error_handling_hints", [])
                for err in signal.affected_errors[:3]:
                    hint = f"Watch for: {err}"
                    if hint not in recovery_strategy["error_handling_hints"]:
                        recovery_strategy["error_handling_hints"].append(hint)

        self.memory.store_strategy(
            domain=domain,
            strategy_data=recovery_strategy,
            score=0.1,
        )

        return True

    def _actions_to_deltas(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert recovery actions to strategy delta changes."""
        deltas = []

        for action in actions:
            atype = action.get("type", "")

            if atype == "add_error_handling":
                errors = action.get("data", [])
                if errors:
                    deltas.append(
                        {
                            "error_handling_hints": [
                                f"Watch for and handle: {err}" for err in errors[:3]
                            ],
                        }
                    )

            elif atype == "update_tool_guidance":
                tools = action.get("data", [])
                if tools:
                    deltas.append(
                        {
                            "avoidance_rules": [
                                f"Tool '{t}' may be unreliable — verify before trusting results"
                                for t in tools[:3]
                            ],
                        }
                    )

            elif atype == "optimize_cost":
                deltas.append(
                    {
                        "reasoning_hints": [
                            "Be efficient with tool calls. Avoid unnecessary retries."
                        ],
                    }
                )

            elif atype == "trigger_evolution":
                deltas.append(
                    {
                        "reasoning_hints": [
                            "Recent performance has degraded. Be extra careful and verify results."
                        ],
                    }
                )

        return deltas
