from __future__ import annotations

import logging
from typing import Any, Optional

from nil.cognition.loop import CognitiveLoop, CognitiveLoopPlugin, Phase, PhaseContext
from nil.cognition.modes import Uncertainty
from nil.cognition.runner import RACEngine
from nil.evolution.pipeline import EvolutionPipeline
from nil.kernel.runner import KernelRunner
from nil.kernel.tracer import Tracer
from nil.memory.abstraction import AbstractionEngine
from nil.memory.store import PersistentMemory
from nil.rl.agent import RLEnabledAgent

logger = logging.getLogger("nil.cognition.plugins")


class MemoryPlugin(CognitiveLoopPlugin):
    def __init__(
        self,
        memory: PersistentMemory,
        abstraction_engine: Optional[AbstractionEngine] = None,
    ) -> None:
        self.memory = memory
        self._abstraction_engine = abstraction_engine

    def get_id(self) -> str:
        return "memory"

    def pre_phase(self, ctx: PhaseContext) -> None:
        if ctx.phase == Phase.OBSERVE:
            tc = self.memory.get_trace_count(ctx.domain)
            sr = self.memory.get_success_rate(ctx.domain)
            ctx.memory["trace_count"] = tc
            ctx.memory["success_rate"] = sr
            ctx.memory["strategy"] = self.memory.get_active_strategy(ctx.domain)
            ctx.memory["skills"] = self.memory.list_skills(ctx.domain)
            ctx.state["trace_count"] = tc

            ctx.uncertainty.observe = Uncertainty(
                confidence=min(1.0, tc / 50),
                uncertainty=max(0.0, 1.0 - tc / 50),
                risk=1.0 - sr if tc > 0 else 0.5,
                novelty=1.0 if tc < 5 else 0.0,
                reliability=sr if tc > 0 else 0.0,
            )

        if ctx.phase == Phase.SIMULATE:
            sr = ctx.memory.get("success_rate", 0.5)
            tc = ctx.memory.get("trace_count", 0)
            ctx.state["simulation_memory_confidence"] = sr
            ctx.state["simulation_trace_count"] = tc

    def post_phase(self, ctx: PhaseContext) -> None:
        if ctx.phase == Phase.EVALUATE and ctx.action_result is not None:
            tracer: Optional[Tracer] = ctx.state.get("_tracer")
            if tracer:
                trace = tracer.finalize(ctx.action_result)
                self.memory.store_trace(trace)

        if ctx.phase == Phase.ADAPT and ctx.adaptation is not None:
            if isinstance(ctx.adaptation, dict):
                new_strategy = ctx.adaptation.get("strategy")
                if new_strategy:
                    self.memory.store_strategy(ctx.domain, new_strategy)

        if ctx.phase == Phase.LEARN and self._abstraction_engine is not None:
            try:
                report = self._abstraction_engine.run(ctx.domain, limit=200)
                ctx.state["abstraction_report"] = report.summary()
                if report.total_concepts > 0:
                    logger.info(
                        "Abstraction: %d concepts for '%s' "
                        "(%d clusters, %d causal, %d symbolic, %d policies)",
                        report.total_concepts, ctx.domain,
                        report.clusters_formed, report.causal_patterns,
                        report.symbolic_concepts, report.policies_induced,
                    )
            except Exception as e:
                logger.warning("Abstraction error: %s", e)


class RACPlugin(CognitiveLoopPlugin):
    def __init__(self, engine: Optional[RACEngine] = None) -> None:
        self.engine = engine or RACEngine()

    def get_id(self) -> str:
        return "rac"

    def pre_phase(self, ctx: PhaseContext) -> None:
        if ctx.phase == Phase.RETRIEVE:
            result = self.engine.run(ctx.task)
            ctx.memory["retrieved_context"] = result.context
            ctx.memory["retrieved_results"] = result.results

            n_results = len(result.results)
            ctx.uncertainty.retrieve = Uncertainty(
                confidence=min(1.0, n_results / 5),
                uncertainty=max(0.0, 1.0 - n_results / 5),
                risk=0.3 if n_results == 0 else 0.1,
                novelty=0.5 if n_results < 3 else 0.1,
                reliability=min(1.0, n_results / 8),
            )

        if ctx.phase == Phase.SIMULATE:
            retrieved = ctx.memory.get("retrieved_results", [])
            ctx.state["simulation_retrieved_count"] = len(retrieved)

        if ctx.phase == Phase.REFLECT and ctx.evaluation is not None:
            success = True
            if isinstance(ctx.evaluation, dict):
                success = ctx.evaluation.get("success", True)
            error = None
            if not success:
                error = ctx.evaluation.get("error", "unknown")
            self.engine.run_with_feedback(
                ctx.task, "success" if success else "failure", error
            )

    def post_phase(self, ctx: PhaseContext) -> None:
        if ctx.phase == Phase.LEARN and ctx.learning is not None:
            for skill_content in ctx.learning.get("new_skills", []):
                self.engine.add_skill(skill_content)
            for failure in ctx.learning.get("new_failures", []):
                self.engine.add_failure(failure)


class KernelPlugin(CognitiveLoopPlugin):
    def __init__(self, runner: KernelRunner) -> None:
        self.runner = runner

    def get_id(self) -> str:
        return "kernel"

    def pre_phase(self, ctx: PhaseContext) -> None:
        if ctx.phase == Phase.ACT:
            tracer = Tracer(run_id=ctx.state["run_id"], domain=ctx.domain)
            ctx.state["_tracer"] = tracer

            combined_context = dict(ctx.context)
            if ctx.memory.get("retrieved_context"):
                combined_context["retrieved"] = ctx.memory["retrieved_context"]
            if ctx.memory.get("skills"):
                combined_context["skills"] = ctx.memory["skills"]

            outcome = self.runner.execute(
                task=ctx.task,
                context=combined_context,
                tracer=tracer,
            )
            ctx.action_result = outcome

            had_error = getattr(outcome, "error", None) is not None
            step_count = getattr(outcome, "step_count", 0)
            ctx.uncertainty.act = Uncertainty(
                confidence=0.3 if had_error else 0.7,
                uncertainty=0.8 if had_error else 0.3,
                risk=0.9 if had_error else 0.2,
                novelty=0.2,
                reliability=0.2 if had_error else 0.7,
            )

    def post_phase(self, ctx: PhaseContext) -> None:
        pass


class EvolutionPlugin(CognitiveLoopPlugin):
    def __init__(self, pipeline: EvolutionPipeline) -> None:
        self.pipeline = pipeline

    def get_id(self) -> str:
        return "evolution"

    def pre_phase(self, ctx: PhaseContext) -> None:
        if ctx.phase == Phase.REFLECT and ctx.action_result is not None:
            analysis = self.pipeline.analyzer.full_analysis(ctx.domain)
            ctx.reflection = analysis

            n_patterns = len(getattr(analysis, "significant_patterns", []))
            ctx.state["significant_patterns"] = n_patterns
            ctx.uncertainty.reflect = Uncertainty(
                confidence=min(1.0, n_patterns / 5),
                uncertainty=max(0.0, 1.0 - n_patterns / 3),
                risk=1.0 - getattr(analysis, "baseline_success_rate", 0.5),
                novelty=0.3 if n_patterns > 0 else 0.0,
                reliability=getattr(analysis, "baseline_success_rate", 0.5),
            )

        if ctx.phase == Phase.LEARN:
            result = self.pipeline.run(ctx.domain)
            ctx.learning = {
                "evolution_result": result,
                "new_skills": [],
                "new_failures": [],
                "deltas_applied": result.deltas_applied,
            }
            if result.extraction:
                ctx.learning["new_skills"] = [
                    s.to_prompt_fragment() for s in result.extraction.extracted
                ]
            if result.analysis and result.analysis.failure_clusters:
                for fc in result.analysis.failure_clusters:
                    if fc.is_systemic:
                        ctx.learning["new_failures"].append(
                            f"Systemic failure: {fc.error_signature}"
                        )

            ctx.uncertainty.learn = Uncertainty(
                confidence=min(1.0, result.skills_extracted / 3),
                uncertainty=0.3 if result.had_impact else 0.7,
                risk=0.1 if result.drift_detected else 0.0,
                novelty=0.5 if result.skills_extracted > 0 else 0.0,
                reliability=0.8 if result.had_impact else 0.3,
            )

        if ctx.phase == Phase.ADAPT:
            evo_result = ctx.learning.get("evolution_result") if ctx.learning else None
            adaptation_data: dict[str, Any] = {}
            if evo_result and evo_result.had_impact:
                deltas = (
                    evo_result.gradient_batch.deltas
                    if evo_result.gradient_batch
                    else []
                )
                new_strategy = self.pipeline.gradient_engine.apply_deltas(
                    ctx.domain, deltas
                )
                adaptation_data["strategy"] = new_strategy
                adaptation_data["strategy_version"] = evo_result.strategy_version
                adaptation_data["had_evolution"] = True
                adaptation_data["strategy_updated"] = True
                adaptation_data["skills_added"] = evo_result.skills_extracted

            drift_report = self.pipeline.drift_detector.check(ctx.domain)
            if drift_report.drift_detected:
                logger.warning(
                    "Drift detected: %s — %s",
                    drift_report.severity,
                    "; ".join(drift_report.root_causes),
                )
                adaptation_data["drift_detected"] = True
                adaptation_data["drift_severity"] = drift_report.severity

            ctx.adaptation = adaptation_data if adaptation_data else None

            ctx.uncertainty.adapt = Uncertainty(
                confidence=0.7 if ctx.adaptation else 0.0,
                uncertainty=0.3 if ctx.adaptation else 0.9,
                risk=0.5 if ctx.adaptation and ctx.adaptation.get("drift_detected") else 0.1,
                novelty=0.6 if ctx.adaptation else 0.0,
                reliability=0.6 if ctx.adaptation else 0.0,
            )

    def post_phase(self, ctx: PhaseContext) -> None:
        if ctx.phase == Phase.REFLECT and ctx.reflection is not None:
            if hasattr(ctx.reflection, "significant_patterns"):
                logger.info(
                    "Reflection found %d significant patterns",
                    len(ctx.reflection.significant_patterns),
                )
        if ctx.phase == Phase.LEARN and ctx.learning is not None:
            evo_result = ctx.learning.get("evolution_result")
            if evo_result and evo_result.had_impact:
                logger.info(
                    "Evolution impact: %d skills, %d deltas applied",
                    evo_result.skills_extracted,
                    evo_result.deltas_applied,
                )


class RLPlugin(CognitiveLoopPlugin):
    def __init__(self, rl_agent: RLEnabledAgent) -> None:
        self.rl_agent = rl_agent

    def get_id(self) -> str:
        return "rl"

    def pre_phase(self, ctx: PhaseContext) -> None:
        if ctx.phase == Phase.SIMULATE:
            ctx.state["rl_policy_certainty"] = getattr(
                self.rl_agent, "_policy_certainty", 0.5
            )

        if ctx.phase == Phase.EVALUATE:
            self.rl_agent.on_step_start(ctx.state)

    def post_phase(self, ctx: PhaseContext) -> None:
        if ctx.phase == Phase.EVALUATE and ctx.action_result is not None:
            trace = {}
            if hasattr(ctx.action_result, "to_dict"):
                trace = ctx.action_result.to_dict()
            self.rl_agent.finish_trajectory(trace)

            has_error = bool(trace.get("error"))
            ctx.uncertainty.evaluate = Uncertainty(
                confidence=0.8 if trace.get("outcome_success") else 0.2,
                uncertainty=0.2 if trace.get("outcome_success") else 0.8,
                risk=0.9 if has_error else 0.1,
                novelty=0.1,
                reliability=0.9 if trace.get("outcome_success") else 0.1,
            )

        if ctx.phase == Phase.LEARN and ctx.learning is not None:
            ctx.learning["rl_policy_updated"] = self.rl_agent._step_count > 0


def wire_cognitive_loop(
    loop: CognitiveLoop,
    memory: PersistentMemory,
    kernel_runner: KernelRunner,
    evolution_pipeline: EvolutionPipeline,
    rac_engine: Optional[RACEngine] = None,
    rl_agent: Optional[RLEnabledAgent] = None,
    abstraction_engine: Optional[AbstractionEngine] = None,
) -> None:
    loop.register(MemoryPlugin(memory, abstraction_engine=abstraction_engine))
    loop.register(RACPlugin(rac_engine or RACEngine()))
    loop.register(KernelPlugin(kernel_runner))
    loop.register(EvolutionPlugin(evolution_pipeline))
    if rl_agent:
        loop.register(RLPlugin(rl_agent))
