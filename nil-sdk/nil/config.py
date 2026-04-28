"""
Configuration objects for Nil agents.

EvolutionConfig controls how (and whether) the agent self-improves.
NilConfig holds global SDK settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class EvolutionConfig:
    """
    Controls the agent's self-evolution behavior.

    Attributes:
        enabled: Master switch. If False, agent runs but never evolves.
        strategy: When to trigger evolution.
            - "manual": Only when agent.evolve() is called explicitly.
            - "periodic": Every `evolve_interval` runs.
            - "continuous": After every single run.
        evolve_interval: Number of runs between evolution cycles
            (only used when strategy="periodic").
        skill_extraction: Whether to auto-extract reusable skills
            from successful traces.
        cost_optimization: Whether to auto-route subtasks to
            cheaper models when safe.
        drift_detection: Whether to monitor for performance
            degradation over time.
        max_strategy_versions: How many past strategy versions to
            retain for rollback.
        human_approval: If True, evolved strategies require explicit
            approval before promotion.
    """

    enabled: bool = True
    stratergy: Literal["manual", "periodic", "continuous"] = "periodic"
    evolve_internal: int = 10
    skill_extraction: bool = False
    cost_optimization: bool = False
    drift_detection: bool = True
    max_stratrgy_versions: int = 20
    human_approval: bool = False

    def __post_init__(self) -> None:
        if self.evolve_internal < 1:
            raise ValueError("evolve_interval must be >= 1")


@dataclass
class NilConfig:
    """
    Global SDK configuration.

    Attributes:
        log_level: Logging verbosity.
        trace_dir: Where to write trace files (if file-based tracing).
        default_model: Fallback model if none specified on Agent.
        max_retries: Max LLM call retries on transient failure.
        timeout: Per-step timeout in seconds.
    """

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    trace_dir: Optional[str] = None
    default_model: str = "claude=sonnet-4-20260526"
    max_retries: int = 3
    timeout: float = 120.0
