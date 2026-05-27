from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional, TYPE_CHECKING

from nil.cognition.modes import CognitiveMode, ModeProfile, MODE_PROFILES, Uncertainty
from nil.memory.store import PersistentMemory

if TYPE_CHECKING:
    from nil.cognition.loop import Phase, PhaseContext

logger = logging.getLogger("nil.cognition.policy")

PhaseFn = Callable[["PhaseContext"], None]


def _compute_task_complexity(task: str) -> float:
    words = task.split()
    if len(words) < 5:
        return 0.2
    if len(words) < 15:
        return 0.5
    if len(words) < 30:
        return 0.7
    return 0.9


def _compute_urgency(task: str) -> float:
    urgent_keywords = [
        "urgent", "critical", "asap", "immediately", "emergency",
        "broken", "down", "crash", "fire", "blocker", "time-sensitive",
    ]
    task_lower = task.lower()
    matches = sum(1 for kw in urgent_keywords if kw in task_lower)
    return min(1.0, matches * 0.25)


def _has_exploration_signal(task: str) -> bool:
    explore_keywords = [
        "explore", "investigate", "find", "discover", "research",
        "unknown", "new", "novel", "unfamiliar", "learn",
    ]
    task_lower = task.lower()
    return any(kw in task_lower for kw in explore_keywords)


def _has_complex_dependency(task: str) -> bool:
    dep_keywords = [
        "coordinate", "parallel", "simultaneous", "multiple",
        "delegate", "assign", "team", "together", "collaborate",
    ]
    task_lower = task.lower()
    return sum(1 for kw in dep_keywords if kw in task_lower) >= 2


class CognitivePolicyEngine:
    def __init__(
        self,
        memory: Optional[PersistentMemory] = None,
        default_mode: CognitiveMode = CognitiveMode.DELIBERATIVE,
    ):
        self.memory = memory
        self.default_mode = default_mode
        self._mode_history: list[dict[str, Any]] = []

    def select_mode(self, ctx: Any) -> CognitiveMode:
        task = getattr(ctx, "task", "")
        state = getattr(ctx, "state", {})
        uncertainty = getattr(ctx, "uncertainty", None) or {}
        error_count = len(getattr(ctx, "errors", []))
        retry_count = state.get("retry_count", 0)
        mode_override = state.get("mode_override")

        if mode_override:
            return mode_override

        error_rate = self._compute_error_rate(ctx)
        success_rate = 1.0 - error_rate

        if error_count > 0 and retry_count >= 2:
            return CognitiveMode.RECOVERY

        if error_rate > 0.4:
            return CognitiveMode.RECOVERY

        if _has_complex_dependency(task):
            return CognitiveMode.MULTI_AGENT

        task_complexity = _compute_task_complexity(task)
        urgency = _compute_urgency(task)

        meta_uncertainty = uncertainty.get("meta", Uncertainty())
        if meta_uncertainty.is_novel or _has_exploration_signal(task):
            return CognitiveMode.EXPLORATORY

        if task_complexity > 0.7 and urgency < 0.5:
            return CognitiveMode.PLANNING

        if meta_uncertainty.is_certain and task_complexity < 0.3:
            return CognitiveMode.REACTIVE

        if success_rate < 0.6 and state.get("trace_count", 0) >= 10:
            return CognitiveMode.REFLECTIVE

        pattern_count = state.get("significant_patterns", 0)
        if pattern_count >= 5:
            return CognitiveMode.COMPILATION

        if urgency > 0.7:
            return CognitiveMode.REACTIVE

        return self.default_mode

    def build_pipeline(
        self,
        ctx: Any,
        all_phases: list[PhaseFn],
    ) -> list[PhaseFn]:
        from nil.cognition.loop import Phase

        mode = getattr(ctx, "mode", self.default_mode)
        profile = MODE_PROFILES.get(mode, MODE_PROFILES[self.default_mode])
        errors = getattr(ctx, "errors", [])
        pipeline: list[PhaseFn] = []
        phase_map: dict[Phase, PhaseFn] = {fn.__phase__: fn for fn in all_phases}

        pipeline.append(phase_map[Phase.OBSERVE])
        pipeline.append(phase_map[Phase.RETRIEVE])

        if profile.planning_depth > 0:
            pipeline.append(phase_map[Phase.REASON])
            pipeline.append(phase_map[Phase.PLAN])

        if self.should_simulate(ctx):
            pipeline.append(phase_map[Phase.SIMULATE])

        pipeline.append(phase_map[Phase.ACT])
        pipeline.append(phase_map[Phase.EVALUATE])

        if profile.should_reflect:
            pipeline.append(phase_map[Phase.REFLECT])

        if mode in (CognitiveMode.REFLECTIVE, CognitiveMode.COMPILATION):
            pipeline.append(phase_map[Phase.LEARN])
            pipeline.append(phase_map[Phase.ADAPT])

        if profile.should_compile:
            pipeline.append(phase_map[Phase.LEARN])
            pipeline.append(phase_map[Phase.ADAPT])

        if errors:
            pipeline.append(phase_map[Phase.REFLECT])

        return pipeline

    def allocate_budget(self, ctx: Any) -> dict[str, Any]:
        mode = getattr(ctx, "mode", self.default_mode)
        profile = MODE_PROFILES.get(mode, MODE_PROFILES[self.default_mode])
        task = getattr(ctx, "task", "")
        task_complexity = _compute_task_complexity(task)

        base_steps = max(5, int(10 * profile.compute_allocation))
        max_steps = int(base_steps * (1.0 + task_complexity))

        return {
            "max_steps": max_steps,
            "max_retries": profile.max_retries,
            "max_cost": 0.05 * profile.compute_allocation,
            "retrieval_depth": profile.retrieval_depth,
            "planning_depth": profile.planning_depth,
            "exploration_bonus": profile.exploration_bonus,
            "compute_allocation": profile.compute_allocation,
        }

    def should_reflect(self, ctx: Any) -> bool:
        mode = getattr(ctx, "mode", self.default_mode)
        profile = MODE_PROFILES.get(mode)
        if profile is None:
            return True
        return profile.should_reflect

    def should_spawn_agents(self, ctx: Any) -> bool:
        mode = getattr(ctx, "mode", self.default_mode)
        profile = MODE_PROFILES.get(mode)
        if profile is None:
            return False
        return profile.spawn_agents

    def should_simulate(self, ctx: Any) -> bool:
        mode = getattr(ctx, "mode", self.default_mode)
        profile = MODE_PROFILES.get(mode)
        if profile is None:
            return False
        return profile.simulation_depth > 0

    def should_plan(self, ctx: Any) -> bool:
        mode = getattr(ctx, "mode", self.default_mode)
        profile = MODE_PROFILES.get(mode)
        if profile is None:
            return True
        return profile.planning_depth > 0

    def should_explore(self, ctx: Any) -> bool:
        mode = getattr(ctx, "mode", self.default_mode)
        profile = MODE_PROFILES.get(mode)
        if profile is None:
            return False
        return profile.exploration_bonus > 0

    def should_retry(self, ctx: Any) -> bool:
        mode = getattr(ctx, "mode", self.default_mode)
        profile = MODE_PROFILES.get(mode)
        if profile is None:
            return True
        retry_count = getattr(ctx, "retry_count", 0)
        return retry_count < profile.max_retries

    def should_compile(self, ctx: Any) -> bool:
        mode = getattr(ctx, "mode", self.default_mode)
        profile = MODE_PROFILES.get(mode)
        if profile is None:
            return False
        return profile.should_compile

    def record_mode_decision(
        self,
        mode: CognitiveMode,
        task: str,
        signals: dict[str, Any],
    ) -> None:
        self._mode_history.append({
            "mode": mode.value,
            "task": task[:80],
            "signals": signals,
        })

    def _compute_error_rate(self, ctx: Any) -> float:
        if self.memory is None:
            return 0.0
        trace_count = getattr(ctx, "trace_count", None)
        if trace_count is None:
            trace_count = self.memory.get_trace_count(
                getattr(ctx, "domain", "")
            )
        if trace_count < 5:
            return 0.0
        success_rate = self.memory.get_success_rate(
            getattr(ctx, "domain", ""), last_n=20
        )
        return 1.0 - success_rate


def build_default_policy(memory: Optional[PersistentMemory] = None) -> CognitivePolicyEngine:
    return CognitivePolicyEngine(memory=memory)
