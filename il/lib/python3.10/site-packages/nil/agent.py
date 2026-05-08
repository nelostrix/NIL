"""
Agent — the primary interface to Nil.

Users create an Agent with a Domain, a model, and optional evolution config.
Calling agent.run(task) executes the task, records the trace, and (if
evolution is enabled) triggers self-improvement via the evolution pipeline.

The Agent connects:
    - Kernel (execution) → runs tasks via LLM + tools
    - Memory (persistence) → stores traces, skills, strategies
    - Evolution Pipeline → analyzes, critiques, extracts, evaluates, heals

Example:
    agent = Agent(
        domain=domain,
        base_model="claude-sonnet-4-20250514",
        memory=PersistentMemory("./store"),
        evolution=EvolutionConfig(strategy="continuous"),
    )
    result = agent.run("Fix the broken API endpoint")
    print(result.success, result.output)
    print(agent.learned_skills())
    print(agent.evolution_history())
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from nil.config import EvolutionConfig, NilConfig
from nil.domain import Domain
from nil.evolution.gradient_engine import GradientProvider
from nil.evolution.pipeline import EvolutionPipeline, PipelineResult
from nil.kernel.outcome import RunOutcome
from nil.kernel.runner import KernelRunner
from nil.kernel.tracer import Tracer
from nil.memory.store import PersistentMemory

logger = logging.getLogger("nil")


class Agent:
    """
    A self-evolving AI agent bound to a specific domain.

    The Agent is the orchestrator — it connects the Kernel (execution),
    Memory (persistence), and Evolution (self-improvement) systems.

    When evolution is enabled, the agent automatically:
        - Extracts skills from successful patterns
        - Critiques failures and proposes improvements
        - Tests improvements before deploying them
        - Monitors for performance drift and self-heals
        - Injects learned knowledge into its own system prompt
    """

    def __init__(
        self,
        domain: Domain,
        base_model: str = "claude-sonnet-4-20250514",
        memory: Optional[PersistentMemory] = None,
        evolution: Optional[EvolutionConfig] = None,
        config: Optional[NilConfig] = None,
        provider: Optional[str] = None,
        gradient_provider: Optional[GradientProvider] = None,
    ) -> None:
        self.domain = domain
        self.base_model = base_model
        self.config = config or NilConfig()
        self.evolution = evolution or EvolutionConfig(enabled=False)
        self.memory = memory or PersistentMemory()
        self.provider = provider

        # Internal state
        self._run_count: int = 0
        self._evolution_history: list[dict[str, Any]] = []

        # Kernel runner
        self._runner = KernelRunner(
            domain=self.domain,
            model=self.base_model,
            config=self.config,
            memory=self.memory,
            provider=self.provider,
        )

        # Evolution pipeline (lazy init on first evolve)
        self._pipeline: Optional[EvolutionPipeline] = None
        self._gradient_provider = gradient_provider

        logging.basicConfig(level=getattr(logging, self.config.log_level))
        logger.info(
            "Agent initialized for domain='%s' with model='%s'",
            self.domain.name,
            self.base_model,
        )

    # ------------------------------------------------------------------
    # Core: run a task
    # ------------------------------------------------------------------

    def run(self, task: str, context: Optional[dict[str, Any]] = None) -> RunOutcome:
        """
        Execute a task within the agent's domain.

        Args:
            task: Natural language description of what to do.
            context: Optional dict of additional context.

        Returns:
            RunOutcome with the result, trace, timing, and success flag.
        """
        self._run_count += 1
        run_id = f"{self.domain.name}-run-{self._run_count}-{int(time.time())}"

        logger.info("Starting run %s: %s", run_id, task[:80])

        # Create a tracer for this run
        tracer = Tracer(run_id=run_id, domain=self.domain.name)

        # Execute via the kernel
        outcome = self._runner.execute(
            task=task,
            context=context or {},
            tracer=tracer,
        )

        # Evaluate success using the domain's signal
        outcome.success = self.domain.evaluate(outcome)

        # Persist the trace
        trace = tracer.finalize(outcome)
        self.memory.store_trace(trace)

        logger.info(
            "Run %s completed: success=%s, duration=%.2fs, steps=%d",
            run_id,
            outcome.success,
            outcome.duration,
            outcome.step_count,
        )

        # Check if evolution should trigger
        if self.evolution.enabled:
            self._maybe_evolve()

        return outcome

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def _get_pipeline(self) -> EvolutionPipeline:
        """Lazily initialize the evolution pipeline."""
        if self._pipeline is None:
            self._pipeline = EvolutionPipeline(
                memory=self.memory,
                gradient_provider=self._gradient_provider,
                auto_apply=not self.evolution.human_approval,
                min_traces_for_evolution=max(10, self.evolution.evolve_interval),
            )
        return self._pipeline

    def _maybe_evolve(self) -> None:
        """Check evolution strategy and trigger if conditions are met."""
        if self.evolution.strategy == "continuous":
            self._evolve()
        elif self.evolution.strategy == "periodic":
            if self._run_count % self.evolution.evolve_interval == 0:
                self._evolve()
        # "manual" strategy: user calls agent.evolve() directly

    def _evolve(self) -> None:
        """
        Trigger the full evolution pipeline.

        Runs all 6 stages: analyze → extract skills → compute gradients
        → evaluate → build world model → check drift.
        """
        logger.info(
            "Evolution triggered after %d runs (strategy=%s)",
            self._run_count,
            self.evolution.strategy,
        )

        pipeline = self._get_pipeline()
        result = pipeline.run(self.domain.name)

        # Record in history
        self._evolution_history.append(result.summary())

        if result.had_impact:
            logger.info(
                "Evolution impact: %d skills extracted, %d deltas applied, promoted=%s, drift=%s",
                result.skills_extracted,
                result.deltas_applied,
                result.strategy_promoted,
                result.drift_detected,
            )

            # Rebuild the runner's system prompt with new knowledge
            self._refresh_runner()

        if result.error:
            logger.warning("Evolution error: %s", result.error)

    def evolve(self) -> PipelineResult:
        """
        Manually trigger evolution and return the result.

        Use with strategy='manual' or anytime you want to force
        an evolution cycle.
        """
        logger.info("Manual evolution triggered")
        pipeline = self._get_pipeline()
        result = pipeline.run(self.domain.name)
        self._evolution_history.append(result.summary())

        if result.had_impact:
            self._refresh_runner()

        return result

    def _refresh_runner(self) -> None:
        """
        Rebuild the kernel runner to pick up new skills,
        strategies, and world model knowledge.

        This is called after evolution changes the strategy.
        """
        self._runner = KernelRunner(
            domain=self.domain,
            model=self.base_model,
            config=self.config,
            memory=self.memory,
            provider=self.provider,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def learned_skills(self) -> list[str]:
        """Return the names of skills this agent has extracted."""
        return self.memory.list_skills(self.domain.name)

    def get_skill_prompt(self, compact: bool = False) -> str:
        """
        Get the skill injection block that's currently being
        added to the system prompt.
        """
        pipeline = self._get_pipeline()
        return pipeline.get_skill_prompt(self.domain.name, compact=compact)

    def get_world_model(self) -> Any:
        """Build and return the current world model."""
        pipeline = self._get_pipeline()
        return pipeline.get_world_model(self.domain.name)

    def get_strategy(self) -> Optional[dict[str, Any]]:
        """Get the current active strategy."""
        return self.memory.get_active_strategy(self.domain.name)

    def evolution_history(self) -> list[dict[str, Any]]:
        """Return the history of all evolution runs."""
        return list(self._evolution_history)

    def check_drift(self) -> Any:
        """Run a drift check without triggering full evolution."""
        pipeline = self._get_pipeline()
        return pipeline.drift_detector.check(self.domain.name)

    def success_rate(self, last_n: int = 50) -> float:
        """Get the agent's success rate over the last N runs."""
        return self.memory.get_success_rate(self.domain.name, last_n=last_n)

    def stats(self) -> dict[str, Any]:
        """Get a summary of the agent's current state."""
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
            "evolution_enabled": self.evolution.enabled,
            "evolution_strategy": self.evolution.strategy,
        }

    @property
    def run_count(self) -> int:
        """Total number of runs executed by this agent."""
        return self._run_count

    def __repr__(self) -> str:
        return (
            f"Agent(domain='{self.domain.name}', model='{self.base_model}', "
            f"runs={self._run_count}, skills={len(self.learned_skills())})"
        )
