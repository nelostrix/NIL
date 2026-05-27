from __future__ import annotations

import logging
import time
from typing import Any, Optional

from nil.cognition.loop import CognitiveLoop, LoopResult, Phase
from nil.cognition.plugins import EvolutionPlugin, KernelPlugin, MemoryPlugin, RACPlugin, RLPlugin, wire_cognitive_loop
from nil.cognition.policy import CognitivePolicyEngine, build_default_policy
from nil.cognition.runner import RACEngine
from nil.config import EvolutionConfig, NilConfig
from nil.domain import Domain
from nil.evolution.gradient_engine import GradientProvider
from nil.evolution.pipeline import EvolutionPipeline, PipelineResult
from nil.kernel.outcome import RunOutcome
from nil.kernel.runner import KernelRunner
from nil.memory.store import PersistentMemory
from nil.rl.agent import RLConfig, RLEnabledAgent

logger = logging.getLogger("nil")


class Agent:
    def __init__(
        self,
        domain: Domain,
        base_model: str = "claude-sonnet-4-20250514",
        memory: Optional[PersistentMemory] = None,
        evolution: Optional[EvolutionConfig] = None,
        config: Optional[NilConfig] = None,
        provider: Optional[str] = None,
        gradient_provider: Optional[GradientProvider] = None,
        rl_config: Optional[RLConfig] = None,
        enable_rl: bool = False,
        enable_multi_agent: bool = False,
    ) -> None:
        self.domain = domain
        self.base_model = base_model
        self.config = config or NilConfig()
        self.evolution_cfg = evolution or EvolutionConfig(enabled=False)
        self.memory = memory or PersistentMemory()
        self.provider = provider

        self._run_count: int = 0
        self._evolution_history: list[dict[str, Any]] = []

        self._runner = KernelRunner(
            domain=self.domain,
            model=self.base_model,
            config=self.config,
            memory=self.memory,
            provider=self.provider,
        )

        self._pipeline: Optional[EvolutionPipeline] = None
        self._gradient_provider = gradient_provider
        self._rac_engine: Optional[RACEngine] = None
        self._rl_agent: Optional[RLEnabledAgent] = None

        if enable_rl:
            self._rl_agent = RLEnabledAgent(domain=self.domain.name)

        self._cognitive_loop: Optional[CognitiveLoop] = None
        self._cognitive_policy: CognitivePolicyEngine = build_default_policy()
        self._mode_history: list[dict[str, Any]] = []

        logging.basicConfig(level=getattr(logging, self.config.log_level))
        logger.info(
            "Agent initialized for domain='%s' with model='%s'",
            self.domain.name,
            self.base_model,
        )

    def _get_pipeline(self) -> EvolutionPipeline:
        if self._pipeline is None:
            self._pipeline = EvolutionPipeline(
                memory=self.memory,
                gradient_provider=self._gradient_provider,
                auto_apply=not self.evolution_cfg.human_approval,
                min_traces_for_evolution=max(10, self.evolution_cfg.evolve_interval),
            )
        return self._pipeline

    def _get_loop(self) -> CognitiveLoop:
        if self._cognitive_loop is None:
            loop = CognitiveLoop(policy=self._cognitive_policy)
            wire_cognitive_loop(
                loop=loop,
                memory=self.memory,
                kernel_runner=self._runner,
                evolution_pipeline=self._get_pipeline(),
                rac_engine=self._rac_engine,
                rl_agent=self._rl_agent,
            )
            self._cognitive_loop = loop
        return self._cognitive_loop

    def run(self, task: str, context: Optional[dict[str, Any]] = None) -> RunOutcome:
        self._run_count += 1
        run_id = f"{self.domain.name}-run-{self._run_count}-{int(time.time())}"

        logger.info("Starting run %s: %s", run_id, task[:80])

        loop = self._get_loop()
        result: LoopResult = loop.run(
            domain=self.domain.name,
            task=task,
            context=context or {},
            run_id=run_id,
        )

        outcome = RunOutcome(
            run_id=run_id,
            task=task,
            output=result.output,
            success=result.success,
            duration=result.total_duration,
        )

        outcome.success = self.domain.evaluate(outcome)

        if result.mode_escalations:
            self._mode_history.append({
                "run_id": run_id,
                "mode": result.mode.value,
                "escalations": result.mode_escalations,
                "confidence": result.uncertainty.overall_confidence(),
                "phases": list(result.phases_completed),
            })

        if self.evolution_cfg.enabled:
            self._maybe_evolve()

        return outcome

    def _maybe_evolve(self) -> None:
        if self.evolution_cfg.strategy == "continuous":
            self._evolve()
        elif self.evolution_cfg.strategy == "periodic":
            if self._run_count % self.evolution_cfg.evolve_interval == 0:
                self._evolve()

    def _evolve(self) -> None:
        logger.info(
            "Evolution triggered after %d runs (strategy=%s)",
            self._run_count,
            self.evolution_cfg.strategy,
        )

        pipeline = self._get_pipeline()
        result = pipeline.run(self.domain.name)

        self._evolution_history.append(result.summary())

        if result.had_impact:
            logger.info(
                "Evolution impact: %d skills extracted, %d deltas applied, promoted=%s, drift=%s",
                result.skills_extracted,
                result.deltas_applied,
                result.strategy_promoted,
                result.drift_detected,
            )
            self._refresh_runner()

        if result.error:
            logger.warning("Evolution error: %s", result.error)

    def evolve(self) -> PipelineResult:
        logger.info("Manual evolution triggered")
        pipeline = self._get_pipeline()
        result = pipeline.run(self.domain.name)
        self._evolution_history.append(result.summary())

        if result.had_impact:
            self._refresh_runner()

        return result

    def _refresh_runner(self) -> None:
        self._runner = KernelRunner(
            domain=self.domain,
            model=self.base_model,
            config=self.config,
            memory=self.memory,
            provider=self.provider,
        )
        self._cognitive_loop = None
        self._cognitive_policy = build_default_policy()

    def learned_skills(self) -> list[str]:
        return self.memory.list_skills(self.domain.name)

    def get_skill_prompt(self, compact: bool = False) -> str:
        pipeline = self._get_pipeline()
        return pipeline.get_skill_prompt(self.domain.name, compact=compact)

    def get_world_model(self) -> Any:
        pipeline = self._get_pipeline()
        return pipeline.get_world_model(self.domain.name)

    def get_strategy(self) -> Optional[dict[str, Any]]:
        return self.memory.get_active_strategy(self.domain.name)

    def evolution_history(self) -> list[dict[str, Any]]:
        return list(self._evolution_history)

    def check_drift(self) -> Any:
        pipeline = self._get_pipeline()
        return pipeline.drift_detector.check(self.domain.name)

    def success_rate(self, last_n: int = 50) -> float:
        return self.memory.get_success_rate(self.domain.name, last_n=last_n)

    def stats(self) -> dict[str, Any]:
        trace_count = self.memory.get_trace_count(self.domain.name)
        return {
            "domain": self.domain.name,
            "model": self.base_model,
            "run_count": self._run_count,
            "traces_in_db": trace_count,
            "success_rate": self.success_rate(),
            "skills": self.learned_skills(),
            "strategy": self.get_strategy(),
            "evolution_runs": len(self._evolution_history),
            "evolution_enabled": self.evolution_cfg.enabled,
            "evolution_strategy": self.evolution_cfg.strategy,
            "cognitive_modes": list({m["mode"] for m in self._mode_history}),
            "mode_escalations": sum(len(m["escalations"]) for m in self._mode_history),
        }

    @property
    def run_count(self) -> int:
        return self._run_count

    def __repr__(self) -> str:
        return (
            f"Agent(domain='{self.domain.name}', model='{self.base_model}', "
            f"runs={self._run_count}, skills={len(self.learned_skills())})"
        )
