from __future__ import annotations

import logging
from typing import Any, Optional

from nil.cognition.loop import CognitiveMode, Phase, PhaseContext

logger = logging.getLogger("nil.cognition.adapters")


class ModelSwitcher:
    def __init__(self, available_models: Optional[list[str]] = None):
        self.available_models = available_models or [
            "claude-sonnet-4-20250514",
            "claude-3.5-sonnet-20241022",
            "claude-3-haiku-20240307",
        ]
        self._current_idx = 0

    def get_current_model(self) -> str:
        return self.available_models[self._current_idx]

    def switch_up(self, ctx: PhaseContext) -> str:
        if self._current_idx < len(self.available_models) - 1:
            self._current_idx += 1
        model = self.available_models[self._current_idx]
        ctx.state["model"] = model
        ctx.state["model_switched"] = True
        logger.info("Model switched UP to %s (phase %s)", model, ctx.phase.value)
        return model

    def switch_down(self, ctx: PhaseContext) -> str:
        if self._current_idx > 0:
            self._current_idx -= 1
        model = self.available_models[self._current_idx]
        ctx.state["model"] = model
        ctx.state["model_switched"] = True
        logger.info("Model switched DOWN to %s (phase %s)", model, ctx.phase.value)
        return model

    def can_switch_up(self) -> bool:
        return self._current_idx < len(self.available_models) - 1

    def can_switch_down(self) -> bool:
        return self._current_idx > 0


class AgentSpawner:
    def __init__(self, max_agents: int = 3):
        self.max_agents = max_agents

    def spawn(self, ctx: PhaseContext, sub_task: str = "") -> Optional[dict[str, Any]]:
        current_agents = ctx.state.get("spawned_agents", [])
        if len(current_agents) >= self.max_agents:
            logger.warning("Max agents (%d) reached, cannot spawn", self.max_agents)
            return None

        agent_id = f"agent_{len(current_agents) + 1}"
        agent_info = {
            "id": agent_id,
            "task": sub_task or ctx.task,
            "phase": ctx.phase.value,
            "status": "spawned",
        }
        current_agents.append(agent_info)
        ctx.state["spawned_agents"] = current_agents
        ctx.state["agents_spawned"] = True
        logger.info("Spawned agent '%s' for sub-task: %s", agent_id, sub_task[:60])
        return agent_info

    def spawn_from_plan(
        self, ctx: PhaseContext, plan_steps: list[str]
    ) -> list[dict[str, Any]]:
        spawned: list[dict[str, Any]] = []
        for i, step in enumerate(plan_steps):
            if len(spawned) >= self.max_agents:
                break
            agent = self.spawn(ctx, sub_task=step)
            if agent:
                spawned.append(agent)
        return spawned


class RetrievalAlterer:
    def __init__(self, default_depth: int = 5):
        self.default_depth = default_depth

    def deepen(self, ctx: PhaseContext, amount: int = 3) -> int:
        current = ctx.state.get("retrieval_depth", self.default_depth)
        new_depth = min(current + amount, 20)
        ctx.state["retrieval_depth"] = new_depth
        ctx.state["retrieval_altered"] = True
        logger.info(
            "Retrieval depth increased: %d -> %d", current, new_depth
        )
        return new_depth

    def narrow(self, ctx: PhaseContext, amount: int = 2) -> int:
        current = ctx.state.get("retrieval_depth", self.default_depth)
        new_depth = max(current - amount, 1)
        ctx.state["retrieval_depth"] = new_depth
        ctx.state["retrieval_altered"] = True
        logger.info(
            "Retrieval depth decreased: %d -> %d", current, new_depth
        )
        return new_depth

    def switch_strategy(
        self, ctx: PhaseContext, strategy: str
    ) -> None:
        ctx.state["retrieval_strategy"] = strategy
        ctx.state["retrieval_altered"] = True
        logger.info("Retrieval strategy switched to '%s'", strategy)


class ReasoningDeeper:
    def __init__(self, default_planning_depth: int = 1, default_simulation_depth: int = 1):
        self.default_planning_depth = default_planning_depth
        self.default_simulation_depth = default_simulation_depth

    def deepen_planning(self, ctx: PhaseContext, amount: int = 1) -> int:
        current = ctx.state.get("planning_depth", self.default_planning_depth)
        new_depth = current + amount
        ctx.state["planning_depth"] = new_depth
        ctx.state["reasoning_deepened"] = True

        if ctx.phase != Phase.PLAN and ctx.phase != Phase.SIMULATE:
            ctx.request_phase(Phase.PLAN)
            ctx.request_phase(Phase.REASON)

        logger.info("Planning depth increased: %d -> %d", current, new_depth)
        return new_depth

    def deepen_simulation(self, ctx: PhaseContext, amount: int = 1) -> int:
        current = ctx.state.get("simulation_depth", self.default_simulation_depth)
        new_depth = current + amount
        ctx.state["simulation_depth"] = new_depth
        ctx.state["reasoning_deepened"] = True

        if ctx.phase != Phase.SIMULATE:
            ctx.request_phase(Phase.SIMULATE)

        logger.info("Simulation depth increased: %d -> %d", current, new_depth)
        return new_depth
