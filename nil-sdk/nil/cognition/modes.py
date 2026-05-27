from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CognitiveMode(Enum):
    REACTIVE = "reactive"
    DELIBERATIVE = "deliberative"
    PLANNING = "planning"
    EXPLORATORY = "exploratory"
    REFLECTIVE = "reflective"
    RECOVERY = "recovery"
    MULTI_AGENT = "multi_agent"
    COMPILATION = "compilation"


@dataclass
class ModeProfile:
    mode: CognitiveMode
    planning_depth: int = 1
    simulation_depth: int = 0
    compute_allocation: float = 1.0
    retrieval_depth: int = 5
    spawn_agents: bool = False
    should_reflect: bool = True
    should_compile: bool = False
    max_retries: int = 0
    exploration_bonus: float = 0.0
    description: str = ""


MODE_PROFILES: dict[CognitiveMode, ModeProfile] = {
    CognitiveMode.REACTIVE: ModeProfile(
        mode=CognitiveMode.REACTIVE,
        planning_depth=0,
        simulation_depth=0,
        compute_allocation=0.3,
        retrieval_depth=3,
        spawn_agents=False,
        should_reflect=False,
        should_compile=False,
        max_retries=0,
        exploration_bonus=0.0,
        description="Quick response with minimal reasoning and no planning",
    ),
    CognitiveMode.DELIBERATIVE: ModeProfile(
        mode=CognitiveMode.DELIBERATIVE,
        planning_depth=1,
        simulation_depth=1,
        compute_allocation=1.0,
        retrieval_depth=5,
        spawn_agents=False,
        should_reflect=True,
        should_compile=False,
        max_retries=1,
        exploration_bonus=0.0,
        description="Full reasoning chain with retrieval and reflection",
    ),
    CognitiveMode.PLANNING: ModeProfile(
        mode=CognitiveMode.PLANNING,
        planning_depth=3,
        simulation_depth=3,
        compute_allocation=1.5,
        retrieval_depth=8,
        spawn_agents=False,
        should_reflect=True,
        should_compile=False,
        max_retries=2,
        exploration_bonus=0.0,
        description="Deep multi-step planning with task decomposition",
    ),
    CognitiveMode.EXPLORATORY: ModeProfile(
        mode=CognitiveMode.EXPLORATORY,
        planning_depth=1,
        simulation_depth=2,
        compute_allocation=1.2,
        retrieval_depth=10,
        spawn_agents=False,
        should_reflect=True,
        should_compile=False,
        max_retries=1,
        exploration_bonus=0.3,
        description="High retrieval depth with exploration bonus for novel tasks",
    ),
    CognitiveMode.REFLECTIVE: ModeProfile(
        mode=CognitiveMode.REFLECTIVE,
        planning_depth=0,
        simulation_depth=1,
        compute_allocation=0.8,
        retrieval_depth=8,
        spawn_agents=False,
        should_reflect=True,
        should_compile=True,
        max_retries=0,
        exploration_bonus=0.0,
        description="Heavy analysis mode focused on extracting patterns and compiling skills",
    ),
    CognitiveMode.RECOVERY: ModeProfile(
        mode=CognitiveMode.RECOVERY,
        planning_depth=1,
        simulation_depth=1,
        compute_allocation=1.5,
        retrieval_depth=7,
        spawn_agents=False,
        should_reflect=True,
        should_compile=False,
        max_retries=3,
        exploration_bonus=0.0,
        description="Conservative error recovery mode with extra retries and cautious execution",
    ),
    CognitiveMode.MULTI_AGENT: ModeProfile(
        mode=CognitiveMode.MULTI_AGENT,
        planning_depth=2,
        simulation_depth=2,
        compute_allocation=2.0,
        retrieval_depth=8,
        spawn_agents=True,
        should_reflect=True,
        should_compile=True,
        max_retries=1,
        exploration_bonus=0.0,
        description="Multi-agent coordination with sub-agent spawning and parallel execution",
    ),
    CognitiveMode.COMPILATION: ModeProfile(
        mode=CognitiveMode.COMPILATION,
        planning_depth=0,
        simulation_depth=1,
        compute_allocation=2.0,
        retrieval_depth=10,
        spawn_agents=False,
        should_reflect=True,
        should_compile=True,
        max_retries=0,
        exploration_bonus=0.0,
        description="Workflow compilation mode that extracts and stores reusable patterns",
    ),
}


@dataclass
class Uncertainty:
    confidence: float = 0.0
    uncertainty: float = 1.0
    risk: float = 0.0
    novelty: float = 0.0
    reliability: float = 1.0

    @property
    def is_certain(self) -> bool:
        return self.confidence > 0.8 and self.uncertainty < 0.2

    @property
    def is_risky(self) -> bool:
        return self.risk > 0.6

    @property
    def is_novel(self) -> bool:
        return self.novelty > 0.6

    def to_dict(self) -> dict[str, float]:
        return {
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "risk": round(self.risk, 3),
            "novelty": round(self.novelty, 3),
            "reliability": round(self.reliability, 3),
        }
