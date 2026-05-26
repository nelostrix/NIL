from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nil.benchmarks.base import (
    Agent,
    Outcome,
    TransferReport,
    TransferResult,
)


class TransferBenchmark:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def run(self, agent: Agent, source_domain: str = "incident_response") -> TransferReport:
        for _ in range(20):
            agent.run(self._random_task(source_domain))

        results = {}
        for target_domain, similarity in [
            ("same_new_task", 1.0),
            ("similar_domain", 0.6),
            ("different_domain", 0.2),
        ]:
            zero_shot = self._measure_accuracy(agent, target_domain, n_shots=0)

            few_shot_results = {}
            for n in [1, 3, 5, 10]:
                for shot in range(n):
                    agent.run(self._random_task(target_domain))
                accuracy = self._measure_accuracy(agent, target_domain, n=10)
                few_shot_results[n] = accuracy

            results[target_domain] = TransferResult(
                domain=target_domain,
                similarity=similarity,
                zero_shot_accuracy=zero_shot,
                few_shot_accuracy=few_shot_results,
                transfer_efficiency=self._transfer_efficiency(source_domain, target_domain, zero_shot),
            )

        return TransferReport(
            source_domain=source_domain,
            transfer_results=results,
            overall_transfer_score=self._composite_transfer(results),
        )

    def _random_task(self, domain: str) -> str:
        task_types = {
            "incident_response": [
                "Investigate API error in production",
                "Debug database connection timeout",
                "Resolve memory leak in worker process",
                "Fix authentication failure in login service",
            ],
            "same_new_task": [
                "Handle new type of incident: certificate expiry",
                "Investigate DNS resolution failure",
                "Debug SSL handshake timeout",
                "Resolve rate limiting configuration",
            ],
            "similar_domain": [
                "Monitor system health dashboard",
                "Set up alerting for CPU usage",
                "Configure log aggregation pipeline",
                "Deploy monitoring agent to cluster",
            ],
            "different_domain": [
                "Process raw data through ETL pipeline",
                "Validate data quality in warehouse",
                "Debug data transformation failure",
                "Optimize query performance",
            ],
        }
        tasks = task_types.get(domain, task_types["incident_response"])
        return self.rng.choice(tasks)

    def _measure_accuracy(self, agent: Agent, domain: str, n_shots: int = 0, n: int = 10) -> float:
        if n_shots > 0:
            for _ in range(n_shots):
                agent.run(self._random_task(domain))
        outcomes = [agent.run(self._random_task(domain)) for _ in range(n)]
        return sum(o.success for o in outcomes) / max(len(outcomes), 1)

    def _transfer_efficiency(self, source: str, target: str, zero_shot: float) -> float:
        return zero_shot

    def _composite_transfer(self, results: dict[str, TransferResult]) -> float:
        if not results:
            return 0.0
        similarity_weighted = sum(
            r.composite * r.similarity for r in results.values()
        )
        total_weight = sum(r.similarity for r in results.values())
        return similarity_weighted / total_weight if total_weight > 0 else 0.0
