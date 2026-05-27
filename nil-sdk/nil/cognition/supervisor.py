from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nil.cognition.adapters import (
    AgentSpawner,
    ModelSwitcher,
    ReasoningDeeper,
    RetrievalAlterer,
)
from nil.cognition.loop import CognitiveMode, Phase, PhaseContext
from nil.cognition.modes import MODE_PROFILES, Uncertainty

logger = logging.getLogger("nil.cognition.supervisor")


@dataclass
class SupervisionReport:
    phases_supervised: int = 0
    triggers_fired: int = 0
    actions_taken: list[str] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)
    replans: int = 0
    model_switches: int = 0
    agents_spawned: int = 0
    retrieval_alters: int = 0
    reasoning_deepened: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "phases_supervised": self.phases_supervised,
            "triggers_fired": self.triggers_fired,
            "actions": len(self.actions_taken),
            "escalations": len(self.escalations),
            "replans": self.replans,
            "model_switches": self.model_switches,
            "agents_spawned": self.agents_spawned,
            "retrieval_alters": self.retrieval_alters,
            "reasoning_deepened": self.reasoning_deepened,
        }


TriggerFn = Callable[[PhaseContext], Optional[str]]


class LiveSupervisor:
    SIGNAL_DECAY_SECONDS = 30.0

    def __init__(
        self,
        model_switcher: Optional[ModelSwitcher] = None,
        agent_spawner: Optional[AgentSpawner] = None,
        retrieval_alterer: Optional[RetrievalAlterer] = None,
        reasoning_deeper: Optional[ReasoningDeeper] = None,
        min_phase_duration_ms: float = 50.0,
        cooldown_phases: int = 2,
        enable_auto_adapt: bool = True,
    ):
        self.model_switcher = model_switcher or ModelSwitcher()
        self.agent_spawner = agent_spawner or AgentSpawner()
        self.retrieval_alterer = retrieval_alterer or RetrievalAlterer()
        self.reasoning_deeper = reasoning_deeper or ReasoningDeeper()
        self.min_phase_duration_ms = min_phase_duration_ms
        self.cooldown_phases = cooldown_phases
        self.enable_auto_adapt = enable_auto_adapt

        self._phase_history: list[dict[str, Any]] = []
        self._last_action_phase: dict[str, int] = defaultdict(lambda: -self.cooldown_phases)
        self._report = SupervisionReport()
        self._baseline_timing: dict[str, float] = {}

    @property
    def report(self) -> SupervisionReport:
        return self._report

    def reset(self) -> None:
        self._phase_history.clear()
        self._last_action_phase.clear()
        self._report = SupervisionReport()
        self._baseline_timing.clear()

    def supervise_pre_phase(self, ctx: PhaseContext) -> None:
        if not self.enable_auto_adapt:
            return

        triggers: list[tuple[str, TriggerFn]] = [
            ("pre_uncertainty_spike", self._check_pre_uncertainty_spike),
            ("pre_fallback_pattern", self._check_pre_fallback_pattern),
            ("pre_novel_task_spawn", self._check_pre_novel_task_spawn),
        ]

        for trigger_name, check_fn in triggers:
            if self._on_cooldown(trigger_name, ctx):
                continue
            action = check_fn(ctx)
            if action:
                self._fire(trigger_name, action, ctx)

    def supervise_post_phase(self, ctx: PhaseContext) -> None:
        if not self.enable_auto_adapt:
            self._record_phase(ctx)
            return

        self._record_phase(ctx)

        triggers: list[tuple[str, TriggerFn]] = [
            ("uncertainty_escalate", self._check_uncertainty_escalate),
            ("error_escalate", self._check_error_escalate),
            ("timing_anomaly", self._check_timing_anomaly),
            ("confidence_drop", self._check_confidence_drop),
            ("novelty_explore", self._check_novelty_explore),
            ("retrieval_weak", self._check_retrieval_weak),
            ("repetitive_failure_replan", self._check_repetitive_failure_replan),
        ]

        for trigger_name, check_fn in triggers:
            if self._on_cooldown(trigger_name, ctx):
                continue
            action = check_fn(ctx)
            if action:
                self._fire(trigger_name, action, ctx)

    def _fire(self, trigger_name: str, action: str, ctx: PhaseContext) -> None:
        phase_idx = len(self._phase_history)
        self._last_action_phase[trigger_name] = phase_idx
        self._report.triggers_fired += 1
        self._report.actions_taken.append(f"{trigger_name}:{action}")
        logger.info(
            "Trigger '%s' fired during phase %s -> %s",
            trigger_name, ctx.phase.value, action,
        )

    def _on_cooldown(self, trigger_name: str, ctx: PhaseContext) -> bool:
        phase_idx = len(self._phase_history)
        last = self._last_action_phase.get(trigger_name, -self.cooldown_phases)
        return (phase_idx - last) < self.cooldown_phases

    def _record_phase(self, ctx: PhaseContext) -> None:
        elapsed = ctx.timing.get(ctx.phase.value, 0.0)
        entry = {
            "phase": ctx.phase.value,
            "elapsed": elapsed,
            "errors": list(ctx.errors),
            "mode": ctx.mode.value,
            "confidence": ctx.uncertainty.overall_confidence(),
        }
        self._phase_history.append(entry)
        self._report.phases_supervised += 1

        if ctx.phase.value not in self._baseline_timing:
            self._baseline_timing[ctx.phase.value] = elapsed
        else:
            alpha = 0.3
            prev = self._baseline_timing[ctx.phase.value]
            self._baseline_timing[ctx.phase.value] = prev * (1 - alpha) + elapsed * alpha

    # --- Pre-phase triggers ---

    def _check_pre_uncertainty_spike(self, ctx: PhaseContext) -> Optional[str]:
        if not self._phase_history:
            return None
        last = self._phase_history[-1]
        if last.get("confidence", 1.0) < 0.3:
            mode_up = self._suggest_escalation(ctx.mode)
            if mode_up and mode_up != ctx.mode:
                ctx.escalate_mode(mode_up)
                self._report.escalations.append(f"{ctx.mode.value}->{mode_up.value}")
                return f"escalated_to_{mode_up.value}_due_to_low_confidence"
        return None

    def _check_pre_fallback_pattern(self, ctx: PhaseContext) -> Optional[str]:
        recent = self._phase_history[-3:] if len(self._phase_history) >= 3 else self._phase_history
        if len(recent) >= 2:
            errors_in_recent = sum(1 for p in recent if p.get("errors"))
            if errors_in_recent >= 2:
                if self.reasoning_deeper:
                    self.reasoning_deeper.deepen_planning(ctx)
                    self._report.reasoning_deepened += 1
                ctx.replan()
                self._report.replans += 1
                return "deepened_reasoning_and_replanned"
        return None

    def _check_pre_novel_task_spawn(self, ctx: PhaseContext) -> Optional[str]:
        obs_uncertainty = ctx.uncertainty.observe
        if obs_uncertainty.is_novel and ctx.mode in (
            CognitiveMode.DELIBERATIVE, CognitiveMode.PLANNING
        ):
            if self.agent_spawner:
                self.agent_spawner.spawn(ctx, sub_task=f"explore:{ctx.task[:60]}")
                self._report.agents_spawned += 1
                return "spawned_exploration_agent"
        return None

    # --- Post-phase triggers ---

    def _check_uncertainty_escalate(self, ctx: PhaseContext) -> Optional[str]:
        phase_uncertainty = getattr(ctx.uncertainty, ctx.phase.value, None)
        if phase_uncertainty and phase_uncertainty.uncertainty > 0.7:
            mode_up = self._suggest_escalation(ctx.mode)
            if mode_up and mode_up != ctx.mode:
                ctx.escalate_mode(mode_up)
                self._report.escalations.append(f"{ctx.mode.value}->{mode_up.value}")
                return f"escalated_to_{mode_up.value}"
        return None

    def _check_error_escalate(self, ctx: PhaseContext) -> Optional[str]:
        if ctx.errors:
            last_error = ctx.errors[-1] if ctx.errors else ""
            current_mode = ctx.mode
            if current_mode != CognitiveMode.RECOVERY:
                ctx.escalate_mode(CognitiveMode.RECOVERY)
                self._report.escalations.append(f"{current_mode.value}->recovery")
                return "escalated_to_recovery"
            if self.model_switcher and self.model_switcher.can_switch_up():
                self.model_switcher.switch_up(ctx)
                self._report.model_switches += 1
                return "switched_model_up"
        return None

    def _check_timing_anomaly(self, ctx: PhaseContext) -> Optional[str]:
        elapsed = ctx.timing.get(ctx.phase.value, 0.0)
        if elapsed < self.min_phase_duration_ms / 1000.0:
            return None

        baseline = self._baseline_timing.get(ctx.phase.value, elapsed)
        if baseline > 0 and elapsed > baseline * 3.0 and elapsed > 1.0:
            if self.model_switcher and self.model_switcher.can_switch_down():
                self.model_switcher.switch_down(ctx)
                self._report.model_switches += 1
                return "switched_model_down_due_to_slowdown"
        return None

    def _check_confidence_drop(self, ctx: PhaseContext) -> Optional[str]:
        current_conf = ctx.uncertainty.overall_confidence()
        if len(self._phase_history) >= 2:
            prev_conf = self._phase_history[-2].get("confidence", current_conf)
            if prev_conf - current_conf > 0.4:
                ctx.replan()
                self._report.replans += 1
                return "replanned_due_to_confidence_drop"
        return None

    def _check_novelty_explore(self, ctx: PhaseContext) -> Optional[str]:
        phase_uncertainty = getattr(ctx.uncertainty, ctx.phase.value, None)
        if phase_uncertainty and phase_uncertainty.is_novel:
            if self.retrieval_alterer:
                self.retrieval_alterer.deepen(ctx)
                self._report.retrieval_alters += 1
                return "deepened_retrieval_due_to_novelty"
        return None

    def _check_retrieval_weak(self, ctx: PhaseContext) -> Optional[str]:
        if ctx.phase == Phase.RETRIEVE:
            retrieve_uncertainty = ctx.uncertainty.retrieve
            if retrieve_uncertainty.confidence < 0.3:
                if self.retrieval_alterer:
                    self.retrieval_alterer.deepen(ctx, amount=5)
                    self._report.retrieval_alters += 1
                    self.retrieval_alterer.switch_strategy(ctx, "broad")
                    return "deepened_and_broadened_retrieval"
        return None

    def _check_repetitive_failure_replan(self, ctx: PhaseContext) -> Optional[str]:
        recent = self._phase_history[-3:] if len(self._phase_history) >= 3 else self._phase_history
        errors = [p for p in recent if p.get("errors")]
        if len(errors) >= 2:
            ctx.replan()
            self._report.replans += 1
            if self.reasoning_deeper:
                self.reasoning_deeper.deepen_planning(ctx)
                self._report.reasoning_deepened += 1
            return "replanned_and_deepened_due_to_repetitive_errors"
        return None

    @staticmethod
    def _suggest_escalation(current: CognitiveMode) -> Optional[CognitiveMode]:
        escalation_path = {
            CognitiveMode.REACTIVE: CognitiveMode.DELIBERATIVE,
            CognitiveMode.DELIBERATIVE: CognitiveMode.PLANNING,
            CognitiveMode.PLANNING: CognitiveMode.MULTI_AGENT,
            CognitiveMode.EXPLORATORY: CognitiveMode.PLANNING,
            CognitiveMode.REFLECTIVE: CognitiveMode.COMPILATION,
            CognitiveMode.RECOVERY: CognitiveMode.DELIBERATIVE,
            CognitiveMode.MULTI_AGENT: CognitiveMode.COMPILATION,
            CognitiveMode.COMPILATION: None,
        }
        return escalation_path.get(current)
