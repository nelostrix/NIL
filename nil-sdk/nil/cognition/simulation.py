from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("nil.cognition.simulation")


@dataclass
class SimulationCandidate:
    label: str
    plan: Any
    reward: float = 0.0
    failure_probability: float = 0.0
    tool_cost: float = 0.0
    uncertainty_score: float = 0.0
    rollout_steps: list[str] = field(default_factory=list)

    def score(self) -> float:
        effective_reward = self.reward * (1.0 - self.failure_probability)
        penalty = max(0.01, self.tool_cost + self.uncertainty_score)
        return effective_reward / penalty

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "reward": round(self.reward, 3),
            "failure_probability": round(self.failure_probability, 3),
            "tool_cost": round(self.tool_cost, 3),
            "uncertainty": round(self.uncertainty_score, 3),
            "score": round(self.score(), 3),
            "steps": len(self.rollout_steps),
        }


def _parse_plan_steps(plan: Any) -> list[str]:
    if isinstance(plan, list):
        return [str(s) for s in plan]
    if isinstance(plan, str):
        return [s.strip() for s in plan.strip().split("\n") if s.strip()]
    if isinstance(plan, dict):
        steps = plan.get("steps") or plan.get("subtasks") or []
        return [str(s) for s in steps]
    return [str(plan)]


class SimulationEngine:
    def __init__(
        self,
        reward_noise: float = 0.1,
        cost_base: float = 0.05,
        random_seed: Optional[int] = None,
    ):
        self.reward_noise = reward_noise
        self.cost_base = cost_base
        self._rng = random.Random(random_seed)

    def simulate(
        self,
        ctx: Any,
        plan: Any = None,
        n_candidates: int = 3,
    ) -> SimulationCandidate:
        if plan is None:
            plan = getattr(ctx, "plan", None)
        if plan is None:
            plan = []

        candidates = self._generate_candidates(plan, n_candidates)
        scored: list[SimulationCandidate] = []
        for c in candidates:
            self._run_rollout(c, ctx)
            c.reward = self._estimate_reward(c, ctx)
            c.failure_probability = self._estimate_failure_probability(c, ctx)
            c.tool_cost = self._estimate_tool_costs(c, ctx)
            c.uncertainty_score = self._estimate_uncertainty(c, ctx)
            scored.append(c)

        scored.sort(key=lambda c: c.score(), reverse=True)

        if scored:
            best = scored[0]
            logger.info(
                "Simulation: %d candidates, best='%s' score=%.3f reward=%.3f fail=%.3f cost=%.3f",
                len(scored), best.label, best.score(), best.reward,
                best.failure_probability, best.tool_cost,
            )
            return best

        return SimulationCandidate(label="fallback", plan=plan)

    def _generate_candidates(
        self,
        plan: Any,
        n: int,
    ) -> list[SimulationCandidate]:
        steps = _parse_plan_steps(plan)
        candidates: list[SimulationCandidate] = []

        candidates.append(SimulationCandidate(
            label="original",
            plan=plan,
            rollout_steps=list(steps),
        ))

        if n <= 1:
            return candidates

        step_variations = {
            "parallel": ("parallel", "Execute in parallel: " + ", ".join(steps)),
            "conservative": ("conservative", steps + ["Validate each step before proceeding"]),
            "aggressive": ("aggressive", steps[:max(1, len(steps) // 2)]),
            "delegated": ("delegated", ["Delegate: " + s for s in steps]),
            "cached": ("cached", steps + ["Check cache for prior result"]),
        }

        labels = list(step_variations.keys())
        self._rng.shuffle(labels)

        for label in labels[: n - 1]:
            name, variant_steps = step_variations[label]
            candidates.append(SimulationCandidate(
                label=name,
                plan=variant_steps,
                rollout_steps=list(variant_steps) if isinstance(variant_steps, list) else [str(variant_steps)],
            ))

        return candidates[:n]

    def _run_rollout(
        self,
        candidate: SimulationCandidate,
        ctx: Any,
    ) -> None:
        pass

    def _estimate_reward(
        self,
        candidate: SimulationCandidate,
        ctx: Any,
    ) -> float:
        task = getattr(ctx, "task", "")
        task_complexity = min(1.0, len(task.split()) / 30)
        step_count = max(1, len(candidate.rollout_steps))

        base_reward = 1.0 - task_complexity * 0.3
        step_bonus = min(0.3, step_count * 0.05)
        noise = self._rng.uniform(-self.reward_noise, self.reward_noise)

        return max(0.0, min(1.0, base_reward + step_bonus + noise))

    def _estimate_failure_probability(
        self,
        candidate: SimulationCandidate,
        ctx: Any,
    ) -> float:
        step_count = len(candidate.rollout_steps)
        errors = getattr(ctx, "errors", [])
        base_failure = 0.05 + (step_count * 0.03)
        if errors:
            base_failure += 0.1 * min(3, len(errors))
        noise = self._rng.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, base_failure + noise))

    def _estimate_tool_costs(
        self,
        candidate: SimulationCandidate,
        ctx: Any,
    ) -> float:
        step_count = len(candidate.rollout_steps)
        base = self.cost_base * step_count
        retrieval_depth = getattr(ctx, "retrieval_depth", 5)
        return round(base * (1.0 + retrieval_depth * 0.02), 4)

    def _estimate_uncertainty(
        self,
        candidate: SimulationCandidate,
        ctx: Any,
    ) -> float:
        step_count = len(candidate.rollout_steps)
        uncertainty = getattr(ctx, "uncertainty", None)
        if uncertainty and hasattr(uncertainty, "overall_confidence"):
            confidence = uncertainty.overall_confidence()
            return max(0.0, 1.0 - confidence) + step_count * 0.02
        return 0.3 + step_count * 0.02
