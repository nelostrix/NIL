from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.benchmarks.base import (
    Agent,
    BenchmarkReport,
    CompositeScore,
)
from nil.benchmarks.horizon import LongHorizonBenchmark
from nil.benchmarks.learning import LearningCurveBenchmarks
from nil.benchmarks.memory import MemoryRetentionBenchmark
from nil.benchmarks.recovery import RecoveryBenchmark
from nil.benchmarks.system import SystemBenchmark
from nil.benchmarks.transfer import TransferBenchmark


class NilBenchmarkSuite:
    def __init__(self):
        self.learning = LearningCurveBenchmarks()
        self.horizon = LongHorizonBenchmark()
        self.recovery = RecoveryBenchmark()
        self.memory = MemoryRetentionBenchmark()
        self.transfer = TransferBenchmark()
        self.system = SystemBenchmark()

    def run(
        self,
        agent: Agent,
        output_dir: str = "./benchmark_results",
    ) -> BenchmarkReport:
        start = time.time()
        report = BenchmarkReport(agent_name=str(agent))

        report.learning_curves = self.learning.run(agent)
        report.horizon_results = self.horizon.run(agent)
        report.recovery_results = self.recovery.run(agent)

        memory_results: dict = {}
        learn_fn = lambda: "Learn pattern: respond to error X with action Y"
        test_fn = lambda: "Recall: what action for error X?"
        memory_results["forgetting_curve"] = self.memory.forgetting_curve(
            agent, learn_fn, test_fn, delays=[1, 5, 10]
        )

        pattern_a = lambda: "Pattern A: use tool_alpha for issue type 1"
        pattern_b = lambda: "Pattern B: use tool_beta for issue type 1 (conflicting)"
        test_a = lambda: "Test: what tool for issue type 1?"
        memory_results["interference"] = self.memory.interference_test(
            agent, pattern_a, pattern_b, test_a
        )
        report.memory_results = memory_results

        report.transfer_results = self.transfer.run(agent)
        report.system_metrics = self.system.run(agent, n_runs=100)
        report.composite = self._compute_composite(report)

        report.total_duration = time.time() - start
        self._save_report(report, output_dir)

        return report

    def _compute_composite(self, report: BenchmarkReport) -> CompositeScore:
        lc_values = [lr.composite for lr in report.learning_curves.values()]
        learning_score = sum(lc_values) / max(len(lc_values), 1)

        hr_values = [hr.composite for hr in report.horizon_results.values()]
        horizon_score = sum(hr_values) / max(len(hr_values), 1)

        recovery_score = report.recovery_results.composite

        mem_values = [
            getattr(r, "composite", 0.0) for r in report.memory_results.values()
        ]
        memory_score = sum(mem_values) / max(len(mem_values), 1)

        transfer_score = report.transfer_results.composite
        system_score = report.system_metrics.composite

        return CompositeScore(
            learning_score=learning_score,
            horizon_score=horizon_score,
            recovery_score=recovery_score,
            memory_score=memory_score,
            transfer_score=transfer_score,
            system_score=system_score,
        )

    def _save_report(self, report: BenchmarkReport, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "benchmark_report.json")
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
