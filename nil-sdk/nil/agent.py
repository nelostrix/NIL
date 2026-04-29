"""
Agent — the primary interface to Nil.

Users create an Agent with a Domain, a model, and optional evolution config.
Calling agent.run(task) executes the task, records the trace, and (if
evolution is enabled) triggers self-improvement.

Example:
    agent = Agent(
        domain=domain,
        base_model="claude-sonnet-4-20250514",
        memory=PersistentMemory("./store"),
        evolution=EvolutionConfig(strategy="continuous"),
    )
    result = agent.run("Fix the broken API endpoint")
    print(result.success, result.output)
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional

from nil.config import EvolutionConfig, NilConfig
from nil.domain import Domain
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
    """

    def __init__(
        self,
        domain: Domain,
        base_model: str = "claude-sonnet-4-20260526",
        memory: Optional[PersistentMemory] = None,
        evolution: Optional[EvolutionConfig] = None,
        config: Optional[NilConfig] = None,
        provider: Optional[str] = None,
    ) -> None:
        self.domain = domain
        self.base_model = base_model
        self.config = config or NilConfig()
        self.evolution = evolution or EvolutionConfig(enabled=False)
        self.memory = memory or PersistentMemory()
        self.provider = provider

        # Internal state
        self._run_count: int = 0
        self._runner = KernelRunner(
            domain=self.domain,
            model=self.base_model,
            config=self.config,
            memory=self.memory,
            provider=self.provider,
        )
        logging.basicConfig(level=getattr(logging, self.config.log_level, logging.INFO))
        logger.info(
            "Agent initialized for domain='%s'with model='%s'",
            self.domain.name,
            self.base_model,
        )

    def run(self, task: str, context: Optional[dict[str, Any]] = None) -> RunOutcome:
        """
        Execute a task within the agent's domain.

        Args:
            task: Natural language description of what to do.
            context: Optional dict of additional context (e.g., environment
                variables, prior outputs, user preferences).

        Returns:
            RunOutcome with the result, trace, timing, and success flag.
        """
        self._run_count += 1
        run_id = f"{self.domain.name}-run-{self._run_count}-{int(time.time())}"

        logger.info("starting run %s: %s", run_id, task[:80])

        # Create a tracer for this run
        tracer = Tracer(run_id=run_id, domain=self.domain.name)

        # Execute via the kernel
        outcome = self._runner.execute(
            task=task,
            context=context or {},
            tracer=tracer,
        )

        # persit the trace
        trace = tracer.finalize(outcome)
        self.memory.store_trace(trace)

        logger.info(
            "Run %s completed: success=%s, duration=%.2fs, steps=%d",
            run_id,
            outcome.success,
            outcome.duration,
            outcome.step_count,
        )

        # check if evolution should trigger
        if self.evolution.enabled:
            self._maybe_evolve()
        return outcome

    def _maybe_evolve(self) -> None:
        """Check evolution strategy and trigger if conditions are met."""
        if self.evolution.strategy == "continuous":
            self._evolve()
        elif self.evolution.strategy == "periodic":
            if self._run_count % self.evolution.evolve_internal == 0:
                self._evolve()
        # manual strategy: user calls agent.evolve()directly

    def _evolve(self) -> None:
        """
        Trigger the evolution engine.

        Phase 2-3 implementation. Currently logs a placeholder.
        """
        logger.info(
            "Evolution triggered after %d runs (strategy=%s)",
            self._run_count,
            self.evolution.strategy,
        )
        # TODO: Phase 2 — skill extraction from traces
        # TODO: Phase 3 — strategy mutation + evaluation

    def evolve(self) -> None:
        """Manually trigger evolution. Use with strategy='manual'."""

        self._evolve()

    def learned_skills(self) -> list[str]:
        """
        Return the list of skills this agent has extracted.

        Phase 2 feature. Returns empty list until skill extraction
        is implemented.
        """
        return self.memory.list_skills(self.domain.name)

    @property
    def run_count(self) -> int:
        """total number of runs executed by this agent."""
        return self._run_count

    def __repr__(self) -> str:
        return (
            f"Agent(domain='{self.domain.name}', model='{self.base_model}', runs={self._run_count})"
        )
