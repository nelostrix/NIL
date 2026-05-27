from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING

from nil.cognition.modes import CognitiveMode, ModeProfile, MODE_PROFILES, Uncertainty
from nil.cognition.policy import CognitivePolicyEngine, build_default_policy
from nil.cognition.simulation import SimulationCandidate, SimulationEngine

if TYPE_CHECKING:
    from nil.cognition.supervisor import LiveSupervisor

logger = logging.getLogger("nil.cognition.loop")


class Phase(Enum):
    OBSERVE = "observe"
    RETRIEVE = "retrieve"
    REASON = "reason"
    PLAN = "plan"
    SIMULATE = "simulate"
    ACT = "act"
    EVALUATE = "evaluate"
    REFLECT = "reflect"
    LEARN = "learn"
    ADAPT = "adapt"


@dataclass
class PhaseUncertainty:
    observe: Uncertainty = field(default_factory=Uncertainty)
    retrieve: Uncertainty = field(default_factory=Uncertainty)
    reason: Uncertainty = field(default_factory=Uncertainty)
    plan: Uncertainty = field(default_factory=Uncertainty)
    simulate: Uncertainty = field(default_factory=Uncertainty)
    act: Uncertainty = field(default_factory=Uncertainty)
    evaluate: Uncertainty = field(default_factory=Uncertainty)
    reflect: Uncertainty = field(default_factory=Uncertainty)
    learn: Uncertainty = field(default_factory=Uncertainty)
    adapt: Uncertainty = field(default_factory=Uncertainty)
    meta: Uncertainty = field(default_factory=Uncertainty)

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {k: v.to_dict() for k, v in self.__dict__.items()}

    def overall_confidence(self) -> float:
        scores = [
            u.confidence * (1.0 - u.uncertainty)
            for u in self.__dict__.values()
            if isinstance(u, Uncertainty)
        ]
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class PhaseContext:
    phase: Phase
    domain: str
    task: str
    mode: CognitiveMode = CognitiveMode.DELIBERATIVE
    context: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    uncertainty: PhaseUncertainty = field(default_factory=PhaseUncertainty)
    plan: Optional[Any] = None
    simulation: Optional[SimulationCandidate] = None
    action_result: Optional[Any] = None
    evaluation: Optional[Any] = None
    reflection: Optional[Any] = None
    learning: Optional[Any] = None
    adaptation: Optional[Any] = None
    errors: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    retry_count: int = 0
    max_retries: int = 1

    _loop: Optional[Any] = None
    _pending_phases: list[Phase] = field(default_factory=list)

    def record_time(self, phase_name: str, duration: float) -> None:
        self.timing[phase_name] = duration

    def request_phase(self, phase: Phase) -> None:
        self._pending_phases.insert(0, phase)

    def retry(self) -> None:
        self.retry_count += 1
        if self.retry_count <= self.max_retries:
            self._pending_phases.insert(0, self.phase)
            logger.info(
                "Retrying phase %s (attempt %d/%d)",
                self.phase.value, self.retry_count, self.max_retries,
            )

    def replan(self) -> None:
        logger.info("Replanning triggered from phase %s", self.phase.value)
        self._pending_phases.insert(0, Phase.PLAN)

    def escalate_mode(self, mode: CognitiveMode) -> None:
        logger.info(
            "Mode escalated: %s -> %s",
            self.mode.value, mode.value,
        )
        self.mode = mode
        self.state["mode_override"] = mode


@dataclass
class LoopResult:
    success: bool = False
    output: Any = None
    run_id: str = ""
    mode: CognitiveMode = CognitiveMode.DELIBERATIVE
    phases_completed: list[str] = field(default_factory=list)
    phase_timing: dict[str, float] = field(default_factory=dict)
    total_duration: float = 0.0
    errors: list[str] = field(default_factory=list)
    had_evolution: bool = False
    skills_added: int = 0
    strategy_updated: bool = False
    context: Optional[PhaseContext] = None
    uncertainty: PhaseUncertainty = field(default_factory=PhaseUncertainty)
    retries_used: int = 0
    replans_used: int = 0
    mode_escalations: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode.value,
            "success": self.success,
            "phases": self.phases_completed,
            "total_duration_s": round(self.total_duration, 3),
            "errors": self.errors,
            "had_evolution": self.had_evolution,
            "skills_added": self.skills_added,
            "strategy_updated": self.strategy_updated,
            "retries": self.retries_used,
            "replans": self.replans_used,
            "mode_escalations": len(self.mode_escalations),
            "confidence": round(self.uncertainty.overall_confidence(), 3),
        }


class CognitiveLoopPlugin:
    def get_id(self) -> str:
        raise NotImplementedError

    def pre_phase(self, ctx: PhaseContext) -> None:
        pass

    def post_phase(self, ctx: PhaseContext) -> None:
        pass


PhaseFn = Callable[[PhaseContext], None]


class CognitiveLoop:
    MAX_RECURSION_DEPTH = 10

    def __init__(
        self,
        policy: Optional[CognitivePolicyEngine] = None,
        simulation_engine: Optional[SimulationEngine] = None,
        supervisor: Optional[LiveSupervisor] = None,
    ):
        self._plugins: dict[str, CognitiveLoopPlugin] = {}
        self._run_count = 0
        self._policy: CognitivePolicyEngine = policy or build_default_policy()
        self._simulation_engine: SimulationEngine = simulation_engine or SimulationEngine()
        self._supervisor: Optional[LiveSupervisor] = supervisor

    def register(self, plugin: CognitiveLoopPlugin) -> None:
        pid = plugin.get_id()
        self._plugins[pid] = plugin
        logger.debug("CognitiveLoop plugin registered: %s", pid)

    def unregister(self, plugin_id: str) -> None:
        self._plugins.pop(plugin_id, None)

    @property
    def policy(self) -> CognitivePolicyEngine:
        return self._policy

    @policy.setter
    def policy(self, p: CognitivePolicyEngine) -> None:
        self._policy = p

    def run(
        self,
        domain: str,
        task: str,
        context: Optional[dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> LoopResult:
        self._run_count += 1
        rid = run_id or f"{domain}-loop-{self._run_count}-{int(time.time())}"

        mode = self._policy.select_mode(self._make_signal_envelope(domain, task))
        profile = MODE_PROFILES.get(mode, MODE_PROFILES[CognitiveMode.DELIBERATIVE])

        ctx = PhaseContext(
            phase=Phase.OBSERVE,
            domain=domain,
            task=task,
            mode=mode,
            context=context or {},
            state={
                "run_id": rid,
                "trace_count": 0,
                "significant_patterns": 0,
                "retry_count": 0,
                "phase_index": 0,
            },
            max_retries=profile.max_retries,
        )
        ctx._loop = self

        result = LoopResult(
            run_id=rid,
            mode=mode,
            uncertainty=ctx.uncertainty,
        )
        start_total = time.time()

        self._policy.record_mode_decision(mode, task, {
            "domain": domain,
            "task_length": len(task),
        })

        if self._supervisor:
            self._supervisor.reset()

        recursion_depth = 0
        pending_retries = 0
        pending_replans = 0
        pipeline = self._build_pipeline(ctx)

        result.phases_completed = self._execute_pipeline(
            ctx, result, pipeline, recursion_depth,
            pending_retries, pending_replans,
        )

        result.total_duration = time.time() - start_total
        result.phase_timing = dict(ctx.timing)
        result.output = ctx.action_result
        result.success = not ctx.errors
        result.context = ctx
        result.uncertainty = ctx.uncertainty
        result.retries_used = ctx.retry_count
        result.replans_used = pending_replans

        if isinstance(ctx.state.get("mode_override"), CognitiveMode):
            result.mode_escalations.append(ctx.state["mode_override"].value)

        if ctx.evaluation is not None and isinstance(ctx.evaluation, dict):
            result.success = ctx.evaluation.get("success", result.success)

        if ctx.adaptation is not None and isinstance(ctx.adaptation, dict):
            result.had_evolution = ctx.adaptation.get("had_evolution", False)
            result.strategy_updated = ctx.adaptation.get("strategy_updated", False)
            result.skills_added = ctx.adaptation.get("skills_added", 0)

        return result

    def _execute_pipeline(
        self,
        ctx: PhaseContext,
        result: LoopResult,
        initial_pipeline: list[PhaseFn],
        recursion_depth: int,
        pending_retries: int,
        pending_replans: int,
    ) -> list[str]:
        phases_completed: list[str] = []
        pipeline = list(initial_pipeline)

        while pipeline:
            phase_fn = pipeline.pop(0)
            ctx.phase = phase_fn.__phase__

            if ctx._pending_phases:
                while ctx._pending_phases:
                    next_phase = ctx._pending_phases.pop(0)
                    if next_phase == ctx.phase:
                        pending_retries += 1
                        recursion_depth += 1
                    elif next_phase == Phase.PLAN:
                        pending_replans += 1
                        recursion_depth += 1

                    if recursion_depth > self.MAX_RECURSION_DEPTH:
                        logger.warning(
                            "Max recursion depth (%d) reached", self.MAX_RECURSION_DEPTH
                        )
                        ctx.errors.append("Max recursion depth exceeded")
                        phases_completed.append(f"recursion:limit")
                        return phases_completed

                    for candidate in pipeline:
                        if candidate.__phase__ == next_phase:
                            pipeline.insert(0, candidate)
                            break
                    else:
                        for candidate in initial_pipeline:
                            if candidate.__phase__ == next_phase:
                                pipeline.insert(0, candidate)
                                break

            phase_start = time.perf_counter()

            if self._supervisor:
                self._supervisor.supervise_pre_phase(ctx)

            try:
                self._run_phase_hooks("pre", ctx)
                phase_fn(ctx)
                self._run_phase_hooks("post", ctx)
            except Exception as e:
                logger.error("Phase %s failed: %s", ctx.phase.value, e)
                ctx.errors.append(f"{ctx.phase.value}: {e}")
                phases_completed.append(f"{ctx.phase.value}:error")
                ctx.record_time(ctx.phase.value, time.perf_counter() - phase_start)
                result.errors.append(str(e))

                should_retry = self._policy.should_retry(ctx)
                if should_retry and ctx.retry_count < ctx.max_retries:
                    ctx.retry()
                    continue
                break

            elapsed = time.perf_counter() - phase_start
            ctx.record_time(ctx.phase.value, elapsed)
            phases_completed.append(ctx.phase.value)
            logger.debug("Phase %s done in %.3fs", ctx.phase.value, elapsed)

            if self._supervisor:
                self._supervisor.supervise_post_phase(ctx)

        return phases_completed

    def _build_pipeline(self, ctx: PhaseContext) -> list[PhaseFn]:
        all_phase_fns = [
            self._observe,
            self._retrieve,
            self._reason,
            self._plan,
            self._simulate,
            self._act,
            self._evaluate,
            self._reflect,
            self._learn,
            self._adapt,
        ]
        return self._policy.build_pipeline(ctx, all_phase_fns)

    def _run_phase_hooks(self, hook: str, ctx: PhaseContext) -> None:
        for pid, plugin in self._plugins.items():
            try:
                if hook == "pre":
                    plugin.pre_phase(ctx)
                else:
                    plugin.post_phase(ctx)
            except Exception as e:
                logger.warning("Plugin %s %s-phase error: %s", pid, hook, e)

    def _make_signal_envelope(self, domain: str, task: str) -> Any:
        class SignalEnvelope:
            pass
        env = SignalEnvelope()
        env.domain = domain
        env.task = task
        env.errors = []
        env.state = {}
        env.uncertainty = {}
        return env

    # ---- Phase Implementations ----

    def _simulate(self, ctx: PhaseContext) -> None:
        n_candidates = MODE_PROFILES.get(ctx.mode, MODE_PROFILES[CognitiveMode.DELIBERATIVE]).simulation_depth
        if n_candidates <= 0:
            logger.debug("Simulation skipped (depth=0 for mode %s)", ctx.mode.value)
            return

        best = self._simulation_engine.simulate(
            ctx=ctx,
            plan=ctx.plan,
            n_candidates=n_candidates,
        )
        ctx.simulation = best
        ctx.plan = best.plan

        ctx.uncertainty.simulate = Uncertainty(
            confidence=1.0 - best.failure_probability,
            uncertainty=best.uncertainty_score,
            risk=best.failure_probability,
            novelty=best.uncertainty_score,
            reliability=1.0 - best.failure_probability,
        )

        logger.info(
            "Simulation complete: candidate='%s' score=%.3f reward=%.3f fail=%.3f cost=%.3f",
            best.label, best.score(), best.reward,
            best.failure_probability, best.tool_cost,
        )

    def _observe(self, ctx: PhaseContext) -> None:
        pass

    def _retrieve(self, ctx: PhaseContext) -> None:
        pass

    def _reason(self, ctx: PhaseContext) -> None:
        pass

    def _plan(self, ctx: PhaseContext) -> None:
        pass

    def _act(self, ctx: PhaseContext) -> None:
        pass

    def _evaluate(self, ctx: PhaseContext) -> None:
        pass

    def _reflect(self, ctx: PhaseContext) -> None:
        pass

    def _learn(self, ctx: PhaseContext) -> None:
        pass

    def _adapt(self, ctx: PhaseContext) -> None:
        pass


CognitiveLoop._observe.__phase__ = Phase.OBSERVE
CognitiveLoop._retrieve.__phase__ = Phase.RETRIEVE
CognitiveLoop._reason.__phase__ = Phase.REASON
CognitiveLoop._plan.__phase__ = Phase.PLAN
CognitiveLoop._simulate.__phase__ = Phase.SIMULATE
CognitiveLoop._act.__phase__ = Phase.ACT
CognitiveLoop._evaluate.__phase__ = Phase.EVALUATE
CognitiveLoop._reflect.__phase__ = Phase.REFLECT
CognitiveLoop._learn.__phase__ = Phase.LEARN
CognitiveLoop._adapt.__phase__ = Phase.ADAPT
