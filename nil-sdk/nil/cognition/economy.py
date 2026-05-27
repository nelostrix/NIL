"""
Cognitive Economy — efficient intelligence through cost-aware cognition.

The system reasons about:
  - Token costs per tool, model, and phase
  - Planning vs execution cost tradeoffs
  - Model tier selection (small vs large)
  - Retrieval depth cost curves
  - Expected value of cognition decisions

This ensures profitability by allocating compute where it matters most.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from nil.cognition.loop import Phase, PhaseContext
from nil.cognition.modes import CognitiveMode, ModeProfile, MODE_PROFILES, Uncertainty

logger = logging.getLogger("nil.cognition.economy")


# ---------------------------------------------------------------------------
# Phase 7a — Cost Model
# ---------------------------------------------------------------------------


@dataclass
class ModelCostProfile:
    model_id: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    speed_quality_factor: float = 1.0

    def estimate_call_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            self.input_cost_per_1k * input_tokens / 1000
            + self.output_cost_per_1k * output_tokens / 1000
        )


COST_TIERS: dict[str, ModelCostProfile] = {
    "claude-sonnet-4-20250514": ModelCostProfile(
        model_id="claude-sonnet-4-20250514",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        speed_quality_factor=1.0,
    ),
    "claude-3.5-sonnet-20241022": ModelCostProfile(
        model_id="claude-3.5-sonnet-20241022",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        speed_quality_factor=0.9,
    ),
    "claude-3-haiku-20240307": ModelCostProfile(
        model_id="claude-3-haiku-20240307",
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.00125,
        speed_quality_factor=0.4,
    ),
}

TOOL_COST_ESTIMATES: dict[str, float] = {
    "search": 0.001,
    "read": 0.0005,
    "write": 0.0005,
    "list": 0.0003,
    "execute": 0.002,
}


@dataclass
class PhaseCost:
    phase: str
    model_cost: float = 0.0
    token_cost: float = 0.0
    tool_cost: float = 0.0
    duration_ms: float = 0.0
    total_cost: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


class CostModel:
    def __init__(self):
        self._phase_costs: dict[str, list[PhaseCost]] = defaultdict(list)
        self._tool_costs: dict[str, list[float]] = defaultdict(list)
        self._model_usage: dict[str, int] = defaultdict(int)
        self._total_spend: float = 0.0

    def record_phase(
        self,
        phase: str,
        model: str = "claude-sonnet-4-20250514",
        tokens_in: int = 0,
        tokens_out: int = 0,
        tool_calls: int = 0,
        duration_ms: float = 0.0,
    ) -> PhaseCost:
        profile = COST_TIERS.get(model, COST_TIERS["claude-sonnet-4-20250514"])
        token_cost = profile.estimate_call_cost(tokens_in, tokens_out)
        tool_cost = tool_calls * 0.0005

        pc = PhaseCost(
            phase=phase,
            model_cost=token_cost * 0.7,
            token_cost=token_cost,
            tool_cost=tool_cost,
            duration_ms=duration_ms,
            total_cost=token_cost + tool_cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        self._phase_costs[phase].append(pc)
        self._model_usage[model] += 1
        self._total_spend += pc.total_cost
        return pc

    def record_tool_cost(self, tool_name: str, cost: float) -> None:
        self._tool_costs[tool_name].append(cost)
        self._total_spend += cost

    def average_phase_cost(self, phase: str) -> float:
        costs = self._phase_costs.get(phase, [])
        if not costs:
            return TOOL_COST_ESTIMATES.get(phase, 0.001)
        return sum(c.total_cost for c in costs) / len(costs)

    def average_tool_cost(self, tool_name: str) -> float:
        costs = self._tool_costs.get(tool_name, [])
        if not costs:
            return TOOL_COST_ESTIMATES.get(tool_name, 0.001)
        return sum(costs) / len(costs)

    def total_spend(self) -> float:
        return self._total_spend

    def phase_cost_breakdown(self) -> dict[str, float]:
        return {p: sum(c.total_cost for c in cs) for p, cs in self._phase_costs.items()}

    def model_usage(self) -> dict[str, int]:
        return dict(self._model_usage)

    def reset(self) -> None:
        self._phase_costs.clear()
        self._tool_costs.clear()
        self._model_usage.clear()
        self._total_spend = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_spend": round(self._total_spend, 6),
            "phase_costs": self.phase_cost_breakdown(),
            "model_usage": self.model_usage(),
            "tool_cost_count": len(self._tool_costs),
        }


# ---------------------------------------------------------------------------
# Phase 7b — Cognitive Budget
# ---------------------------------------------------------------------------


@dataclass
class CognitiveBudget:
    max_total_cost: float = 0.10
    max_cost_per_phase: dict[str, float] = field(default_factory=dict)
    max_tokens_per_run: int = 100000
    max_steps: int = 25
    preferred_model: str = "claude-3-haiku-20240307"
    expensive_model: str = "claude-sonnet-4-20250514"
    budget_remaining: float = 0.10

    @classmethod
    def default(cls) -> CognitiveBudget:
        return cls(
            max_total_cost=0.10,
            max_cost_per_phase={
                "observe": 0.001,
                "retrieve": 0.005,
                "reason": 0.02,
                "plan": 0.02,
                "simulate": 0.02,
                "act": 0.05,
                "evaluate": 0.005,
                "reflect": 0.01,
                "learn": 0.01,
                "adapt": 0.005,
            },
            max_tokens_per_run=100000,
            max_steps=25,
            preferred_model="claude-3-haiku-20240307",
            expensive_model="claude-sonnet-4-20250514",
            budget_remaining=0.10,
        )

    def allocate_for_mode(self, mode: CognitiveMode) -> CognitiveBudget:
        profile = MODE_PROFILES.get(mode, MODE_PROFILES[CognitiveMode.DELIBERATIVE])
        alloc = profile.compute_allocation
        return CognitiveBudget(
            max_total_cost=self.max_total_cost * alloc,
            max_cost_per_phase={
                k: v * alloc for k, v in self.max_cost_per_phase.items()
            },
            max_tokens_per_run=int(self.max_tokens_per_run * alloc),
            max_steps=int(self.max_steps * alloc),
            preferred_model=(
                self.preferred_model
                if alloc <= 1.0
                else self.expensive_model
            ),
            expensive_model=self.expensive_model,
            budget_remaining=self.max_total_cost * alloc,
        )


# ---------------------------------------------------------------------------
# Phase 7c — Expected Value Calculator
# ---------------------------------------------------------------------------


@dataclass
class ExpectedValue:
    action: str
    expected_benefit: float
    expected_cost: float
    net_value: float
    confidence: float
    recommendation: str

    @property
    def is_worthwhile(self) -> bool:
        return self.net_value > 0 and self.confidence > 0.3


class ExpectedValueCalculator:
    def __init__(self, cost_model: CostModel):
        self.cost_model = cost_model

    def evaluate_phase(self, phase: Phase, ctx: PhaseContext) -> ExpectedValue:
        phase_name = phase.value
        avg_cost = self.cost_model.average_phase_cost(phase_name)
        uncertainty = getattr(ctx.uncertainty, phase_name, Uncertainty())

        benefit = self._estimate_phase_benefit(phase, ctx)
        cost = max(avg_cost, 0.001)

        net = benefit - cost
        confidence = 1.0 - uncertainty.uncertainty

        if net > 0.001 and confidence > 0.4:
            recommendation = "execute"
        elif benefit > cost * 2:
            recommendation = "execute_with_budget"
        elif confidence < 0.2:
            recommendation = "defer"
        else:
            recommendation = "skip"

        return ExpectedValue(
            action=f"phase:{phase_name}",
            expected_benefit=round(benefit, 6),
            expected_cost=round(cost, 6),
            net_value=round(net, 6),
            confidence=round(confidence, 3),
            recommendation=recommendation,
        )

    def evaluate_model_switch(
        self,
        current_model: str,
        target_model: str,
        task_complexity: float,
    ) -> ExpectedValue:
        current_profile = COST_TIERS.get(
            current_model, COST_TIERS["claude-sonnet-4-20250514"]
        )
        target_profile = COST_TIERS.get(
            target_model, COST_TIERS["claude-3-haiku-20240307"]
        )

        cost_savings = (
            current_profile.input_cost_per_1k - target_profile.input_cost_per_1k
        ) * 2

        quality_loss = target_profile.speed_quality_factor / max(
            current_profile.speed_quality_factor, 0.01
        )
        risk = (1.0 - quality_loss) * task_complexity

        net_value = cost_savings - risk * 0.01

        if net_value > 0 and risk < 0.3:
            recommendation = "switch_down"
        elif risk > 0.5:
            recommendation = "stay_up"
        else:
            recommendation = "maintain"

        return ExpectedValue(
            action=f"model:{current_model}->{target_model}",
            expected_benefit=round(cost_savings, 6),
            expected_cost=round(risk * 0.01, 6),
            net_value=round(net_value, 6),
            confidence=round(1.0 - risk, 3),
            recommendation=recommendation,
        )

    def evaluate_simulation(self, ctx: PhaseContext) -> ExpectedValue:
        sim_cost = self.cost_model.average_phase_cost("simulate")
        uncertainty = ctx.uncertainty.plan

        failure_risk = uncertainty.risk
        simulation_value = failure_risk * 0.1

        net = simulation_value - sim_cost

        if net > 0:
            recommendation = "simulate"
        elif uncertainty.uncertainty > 0.6:
            recommendation = "simulate_shallow"
        else:
            recommendation = "skip_simulation"

        return ExpectedValue(
            action="simulate",
            expected_benefit=round(simulation_value, 6),
            expected_cost=round(sim_cost, 6),
            net_value=round(net, 6),
            confidence=round(1.0 - uncertainty.uncertainty, 3),
            recommendation=recommendation,
        )

    def _estimate_phase_benefit(self, phase: Phase, ctx: PhaseContext) -> float:
        benefits = {
            Phase.OBSERVE: 0.002,
            Phase.RETRIEVE: 0.005,
            Phase.REASON: 0.01,
            Phase.PLAN: 0.015,
            Phase.SIMULATE: 0.01,
            Phase.ACT: 0.05,
            Phase.EVALUATE: 0.003,
            Phase.REFLECT: 0.008,
            Phase.LEARN: 0.01,
            Phase.ADAPT: 0.005,
        }

        base = benefits.get(phase, 0.001)

        profile = MODE_PROFILES.get(ctx.mode, MODE_PROFILES[CognitiveMode.DELIBERATIVE])

        if phase == Phase.SIMULATE and profile.simulation_depth <= 0:
            return 0.0
        if phase == Phase.REFLECT and not profile.should_reflect:
            return 0.0
        if phase == Phase.PLAN and profile.planning_depth <= 0:
            return 0.0

        uncertainty_val = getattr(ctx.uncertainty, phase.value, Uncertainty())
        uncertainty_modifier = 1.0 + uncertainty_val.uncertainty * 0.5

        return base * uncertainty_modifier


# ---------------------------------------------------------------------------
# Phase 7d — Cognitive Economy (Orchestrator)
# ---------------------------------------------------------------------------


@dataclass
class EconomyDecision:
    budget: CognitiveBudget = field(default_factory=CognitiveBudget.default)
    phase_decisions: list[ExpectedValue] = field(default_factory=list)
    model_decision: Optional[ExpectedValue] = None
    simulation_decision: Optional[ExpectedValue] = None
    total_estimated_cost: float = 0.0
    total_estimated_value: float = 0.0
    recommendations: list[str] = field(default_factory=list)


class CognitiveEconomy:
    def __init__(self, cost_model: Optional[CostModel] = None):
        self.cost_model = cost_model or CostModel()
        self.ev_calculator = ExpectedValueCalculator(self.cost_model)
        self._decision_history: list[dict[str, Any]] = []

    def plan_run(
        self,
        ctx: PhaseContext,
        budget: Optional[CognitiveBudget] = None,
    ) -> EconomyDecision:
        decision = EconomyDecision()
        decision.budget = (budget or CognitiveBudget.default()).allocate_for_mode(ctx.mode)

        for phase in Phase:
            ev = self.ev_calculator.evaluate_phase(phase, ctx)
            decision.phase_decisions.append(ev)
            decision.total_estimated_cost += ev.expected_cost
            decision.total_estimated_value += ev.expected_benefit

            if ev.recommendation == "skip":
                decision.recommendations.append(f"skip_{phase.value}")
            elif ev.recommendation == "defer":
                decision.recommendations.append(f"defer_{phase.value}")

        current_model = ctx.state.get("model", "claude-sonnet-4-20250514")
        task_complexity = len(ctx.task.split()) / 50 if ctx.task else 0.5

        model_ev = self.ev_calculator.evaluate_model_switch(
            current_model, decision.budget.preferred_model, task_complexity
        )
        decision.model_decision = model_ev
        decision.recommendations.append(model_ev.recommendation)

        sim_ev = self.ev_calculator.evaluate_simulation(ctx)
        decision.simulation_decision = sim_ev
        if sim_ev.recommendation != "simulate":
            decision.recommendations.append(sim_ev.recommendation)

        self._decision_history.append(
            {
                "mode": ctx.mode.value,
                "estimated_cost": round(decision.total_estimated_cost, 6),
                "recommendations": decision.recommendations,
            }
        )

        return decision

    def should_skip_phase(
        self, phase: Phase, ctx: PhaseContext
    ) -> bool:
        ev = self.ev_calculator.evaluate_phase(phase, ctx)
        if ev.recommendation == "skip":
            logger.info(
                "Economy: skipping phase %s (cost=%.4f, benefit=%.4f, net=%.4f)",
                phase.value, ev.expected_cost, ev.expected_benefit, ev.net_value,
            )
            return True
        if ev.recommendation == "defer":
            logger.info(
                "Economy: deferring phase %s (low confidence %.2f)",
                phase.value, ev.confidence,
            )
            return True
        return False

    def suggest_model(
        self,
        current_model: str,
        task: str,
        uncertainty: Uncertainty,
    ) -> str:
        task_complexity = len(task.split()) / 50 if task else 0.5

        if uncertainty.is_risky or uncertainty.is_novel:
            return COST_TIERS.get(current_model, list(COST_TIERS.values())[0]).model_id

        if task_complexity < 0.3 and not uncertainty.is_risky:
            return "claude-3-haiku-20240307"

        if task_complexity > 0.7 or uncertainty.uncertainty > 0.5:
            return "claude-sonnet-4-20250514"

        return current_model

    def estimated_run_cost(self, ctx: PhaseContext) -> float:
        total = 0.0
        for phase in Phase:
            total += self.cost_model.average_phase_cost(phase.value)
        return total

    def record_run(self, ctx: PhaseContext) -> None:
        for phase_name, duration in ctx.timing.items():
            self.cost_model.record_phase(
                phase=phase_name,
                tokens_in=ctx.state.get(f"{phase_name}_tokens_in", 0),
                tokens_out=ctx.state.get(f"{phase_name}_tokens_out", 0),
                duration_ms=duration * 1000,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_model": self.cost_model.to_dict(),
            "decisions": self._decision_history[-10:] if self._decision_history else [],
        }
