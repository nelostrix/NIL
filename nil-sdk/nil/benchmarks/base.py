from __future__ import annotations

import abc
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol


class Agent(Protocol):
    def run(self, task: str) -> Outcome:
        ...


@dataclass
class Outcome:
    success: bool = False
    steps: list[dict] = field(default_factory=list)
    total_cost: float = 0.0
    duration: float = 0.0
    error: str = ""
    evolution_duration: float = 0.0
    evolution_triggered: bool = False
    step_scores: list[float] = field(default_factory=list)
    had_errors: bool = False
    error_steps: list[dict] = field(default_factory=list)

    def steps_after(self, step: dict) -> list[dict]:
        idx = self.steps.index(step) if step in self.steps else -1
        if idx >= 0:
            return self.steps[idx + 1:]
        return []

    @property
    def tool_sequence(self) -> list[str]:
        return [s.get("tool_name", "") for s in self.steps if s.get("tool_name")]


@dataclass
class ForgettingEvent:
    run: int = 0
    drop: float = 0.0
    duration: int = 0


@dataclass
class LearningReport:
    insufficient_data: bool = False
    initial_accuracy: float = 0.0
    final_accuracy: float = 0.0
    learning_velocity: float = 0.0
    runs_to_plateau: int = 0
    sample_efficiency: float = 0.0
    plateau_accuracy: float = 0.0
    forgetting_events: list[ForgettingEvent] = field(default_factory=list)
    total_runs: int = 0
    composite: float = 0.0

    def __post_init__(self):
        if not self.insufficient_data:
            self.composite = self.plateau_accuracy


@dataclass
class HorizonResult:
    task_length: int = 0
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    error_recovery_rate: float = 0.0
    completion_rate: float = 0.0
    composite: float = 0.0

    def __post_init__(self):
        self.composite = (
            self.mean_accuracy * 0.4
            + self.error_recovery_rate * 0.3
            + self.completion_rate * 0.3
        )


@dataclass
class TestResult:
    completion_rate: float = 0.0
    tool_switch_latency: float = 0.0
    unnecessary_retries: int = 0
    pre_shift_accuracy: float = 0.0
    shift_drop: float = 0.0
    recovery_runs: int = 0
    post_shift_accuracy: float = 0.0
    score: float = 0.0

    def __post_init__(self):
        self.score = self.completion_rate


@dataclass
class RecoveryReport:
    overall_recovery_score: float = 0.0
    per_test: dict[str, TestResult] = field(default_factory=dict)

    @property
    def composite(self) -> float:
        if not self.per_test:
            return 0.0
        return sum(t.score for t in self.per_test.values()) / len(self.per_test)


@dataclass
class ForgettingPoint:
    delay: int = 0
    accuracy: float = 0.0


@dataclass
class ForgettingCurve:
    task: str = ""
    points: list[ForgettingPoint] = field(default_factory=list)
    half_life: float = 0.0
    retention_at_100: float = 0.0
    composite: float = 0.0

    def __post_init__(self):
        if self.points:
            self.composite = self.points[-1].accuracy if len(self.points) > 1 else self.points[0].accuracy


@dataclass
class InterferenceResult:
    accuracy_a_before: float = 0.0
    accuracy_a_after: float = 0.0
    interference_drop: float = 0.0
    interference_ratio: float = 0.0
    composite: float = 0.0

    def __post_init__(self):
        ratio = self.accuracy_a_after / max(self.accuracy_a_before, 1e-6)
        self.composite = min(1.0, ratio)


@dataclass
class TransferResult:
    domain: str = ""
    similarity: float = 0.0
    zero_shot_accuracy: float = 0.0
    few_shot_accuracy: dict[int, float] = field(default_factory=dict)
    transfer_efficiency: float = 0.0
    composite: float = 0.0

    def __post_init__(self):
        self.composite = self.zero_shot_accuracy * 0.3 + (
            self.few_shot_accuracy.get(10, 0.0) * 0.7 if self.few_shot_accuracy else 0.0
        )


@dataclass
class TransferReport:
    source_domain: str = ""
    transfer_results: dict[str, TransferResult] = field(default_factory=dict)
    overall_transfer_score: float = 0.0
    composite: float = 0.0

    def __post_init__(self):
        if self.transfer_results:
            self.composite = sum(
                r.composite for r in self.transfer_results.values()
            ) / len(self.transfer_results)


@dataclass
class SystemReport:
    cost_per_run: float = 0.0
    cost_p50: float = 0.0
    cost_p95: float = 0.0
    latency_per_run: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    memory_growth_per_run: float = 0.0
    total_memory_after: int = 0
    avg_evolution_time: float = 0.0
    evolution_overhead_pct: float = 0.0
    retrieval_recall: float = 0.0
    retrieval_latency: float = 0.0
    composite: float = 0.0

    def __post_init__(self):
        cost_eff = 1.0 / (1.0 + self.cost_per_run)
        lat_eff = 1.0 / (1.0 + self.latency_per_run)
        mem_eff = 1.0 / (1.0 + self.memory_growth_per_run)
        evo_eff = max(0.0, 1.0 - self.evolution_overhead_pct / 100.0)
        self.composite = (cost_eff + lat_eff + mem_eff + evo_eff) / 4.0


@dataclass
class BenchmarkReport:
    agent_name: str = ""
    learning_curves: dict[str, LearningReport] = field(default_factory=dict)
    horizon_results: dict[int, HorizonResult] = field(default_factory=dict)
    recovery_results: RecoveryReport = field(default_factory=RecoveryReport)
    memory_results: dict[str, ForgettingCurve | InterferenceResult] = field(default_factory=dict)
    transfer_results: TransferReport = field(default_factory=TransferReport)
    system_metrics: SystemReport = field(default_factory=SystemReport)
    composite: CompositeScore = field(default_factory=lambda: CompositeScore())
    total_duration: float = 0.0
    generated_at: float = field(default_factory=time.time)

    def _pass_mark(self, v: float) -> str:
        if v >= 0.7:
            return "PASS"
        if v >= 0.4:
            return "WARN"
        return "FAIL"

    def summary(self) -> str:
        lines = [f"Benchmark Report — {self.agent_name}", "=" * 60]

        if self.learning_curves:
            lines.append(f"\nLearning Curves ({len(self.learning_curves)} benchmarks):")
            for name, lr in self.learning_curves.items():
                status = "insufficient data" if lr.insufficient_data else f"plateau={lr.plateau_accuracy:.2f}, velocity={lr.learning_velocity:.4f}"
                lines.append(f"  {name}: initial={lr.initial_accuracy:.2f}, {status}")

        if self.horizon_results:
            lines.append("\nLong-Horizon Tasks:")
            for length, hr in sorted(self.horizon_results.items()):
                c = hr.composite
                lines.append(f"  {length:>4}-step: accuracy={hr.mean_accuracy:.2f} recovery={hr.error_recovery_rate:.2f} completion={hr.completion_rate:.2f}  [{self._pass_mark(c)}]")

        r = self.recovery_results
        lines.append(f"\nRecovery: score={r.overall_recovery_score:.3f}  [{self._pass_mark(r.composite)}]")

        mem_entries = []
        for k, v in self.memory_results.items():
            if hasattr(v, "composite"):
                mem_entries.append(f"{k}={v.composite:.2f}")
        if mem_entries:
            lines.append(f"Memory: {', '.join(mem_entries)}")

        t = self.transfer_results
        lines.append(f"Transfer: score={t.composite:.3f}  [{self._pass_mark(t.composite)}]")

        s = self.system_metrics
        lines.append(f"System: cost={s.cost_per_run:.6f} lat={s.latency_per_run:.2f}s growth={s.memory_growth_per_run:.1f} evol={s.evolution_overhead_pct:.1f}%  [{self._pass_mark(s.composite)}]")

        lines.append(f"\nComposite Score Breakdown:")
        c = self.composite
        lines.append(f"  Learning  (25%): {c.learning_score:.4f}")
        lines.append(f"  Horizon   (20%): {c.horizon_score:.4f}")
        lines.append(f"  Recovery  (20%): {c.recovery_score:.4f}")
        lines.append(f"  Memory    (15%): {c.memory_score:.4f}")
        lines.append(f"  Transfer  (10%): {c.transfer_score:.4f}")
        lines.append(f"  System    (10%): {c.system_score:.4f}")
        lines.append(f"  TOTAL: {c.total:.4f}  [{self._pass_mark(c.total)}]")

        lines.append(f"\nDuration: {self.total_duration:.1f}s")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "composite": self.composite.to_dict(),
            "learning_curves": {k: v.__dict__ for k, v in self.learning_curves.items()},
            "horizon_results": {str(k): v.__dict__ for k, v in self.horizon_results.items()},
            "recovery_results": self.recovery_results.__dict__,
            "system_metrics": self.system_metrics.__dict__,
            "total_duration": self.total_duration,
        }


@dataclass
class CompositeScore:
    learning_score: float = 0.0
    horizon_score: float = 0.0
    recovery_score: float = 0.0
    memory_score: float = 0.0
    transfer_score: float = 0.0
    system_score: float = 0.0
    total: float = 0.0

    def __post_init__(self):
        self.total = (
            self.learning_score * 0.25
            + self.horizon_score * 0.20
            + self.recovery_score * 0.20
            + self.memory_score * 0.15
            + self.transfer_score * 0.10
            + self.system_score * 0.10
        )

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in self.__dict__.items() if k != "total"} | {
            "total": round(self.total, 4)
        }


def percentile(sorted_list: list[float], p: float) -> float:
    if not sorted_list:
        return 0.0
    k = (len(sorted_list) - 1) * p
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return sorted_list[int(k)]
    return sorted_list[f] * (c - k) + sorted_list[c] * (k - f)


def std(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
    return math.sqrt(variance)
