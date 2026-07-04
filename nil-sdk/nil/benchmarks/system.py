from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nil.benchmarks.base import (
    Agent,
    Outcome,
    SystemReport,
    percentile,
)


class SystemBenchmark:
    def __init__(self, seed: int = 42, n_runs: int = 100):
        self.rng = random.Random(seed)
        self.n_runs = n_runs

    def run(self, agent: Agent, n_runs: Optional[int] = None) -> SystemReport:
        n_runs = n_runs or self.n_runs
        tasks = [self._task() for _ in range(n_runs)]
        runs = [agent.run(t) for t in tasks]

        costs = [r.total_cost for r in runs]
        latencies = [r.duration for r in runs]

        memory_before = self._measure_memory_size(agent)
        for _ in range(n_runs):
            agent.run(self._task())
        memory_after = self._measure_memory_size(agent)
        memory_growth = (memory_after - memory_before) / max(n_runs, 1)

        evolution_times = [
            r.evolution_duration for r in runs if r.evolution_triggered
        ]

        retrieval_stats = self._measure_retrieval_quality(agent)

        return SystemReport(
            cost_per_run=sum(costs) / max(len(costs), 1),
            cost_p50=percentile(sorted(costs), 0.5),
            cost_p95=percentile(sorted(costs), 0.95),
            latency_per_run=sum(latencies) / max(len(latencies), 1),
            latency_p50=percentile(sorted(latencies), 0.5),
            latency_p95=percentile(sorted(latencies), 0.95),
            memory_growth_per_run=memory_growth,
            total_memory_after=memory_after,
            avg_evolution_time=sum(evolution_times) / max(len(evolution_times), 1) if evolution_times else 0,
            evolution_overhead_pct=(
                sum(evolution_times) / max(sum(latencies), 1) * 100
                if sum(latencies) > 0 else 0
            ),
            retrieval_recall=retrieval_stats.get("recall@5", 0.0),
            retrieval_latency=retrieval_stats.get("avg_latency_ms", 0.0),
        )

    def _task(self) -> str:
        tasks = [
            "Investigate error in production system",
            "Debug database connection timeout",
            "Resolve memory leak in worker",
            "Fix authentication failure",
            "Deploy service update",
            "Monitor system health",
            "Process data pipeline",
            "Validate configuration",
        ]
        return self.rng.choice(tasks)

    def _measure_memory_size(self, agent: Agent) -> int:
        return 0

    def _measure_retrieval_quality(self, agent: Agent) -> dict[str, float]:
        return {"recall@5": 0.0, "avg_latency_ms": 0.0}
