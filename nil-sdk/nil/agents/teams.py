from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.agents.blackboard import Blackboard
from nil.agents.orchestrator import OrchestratorAgent
from nil.agents.roles import AgentRole, AgentSpec


@dataclass
class TeamResult:
    success: bool = False
    output: str = ""
    cost: float = 0.0
    subtask_results: dict[str, Any] = field(default_factory=dict)


class AgentTeam:
    def __init__(
        self,
        leader: Optional[AgentSpec] = None,
        members: dict[str, AgentSpec] = None,
        coordination: str = "hierarchical",
        name: str = "",
    ):
        self.leader = leader or AgentSpec(role=AgentRole.ORCHESTRATOR, name="team-leader")
        self.members = members or {}
        self.coordination = coordination
        self.name = name or f"team-{uuid.uuid4().hex[:6]}"
        self.subteams: list[AgentTeam] = []
        self.parent: Optional[AgentTeam] = None
        self.blackboard = Blackboard()

    def add_subteam(self, team: AgentTeam) -> None:
        self.subteams.append(team)
        team.parent = self

    def run(self, task: str) -> TeamResult:
        if self.coordination == "hierarchical":
            return self._run_hierarchical(task)
        elif self.coordination == "swarm":
            return self._run_swarm(task)
        elif self.coordination == "democratic":
            return self._run_democratic(task)
        return self._run_hierarchical(task)

    def _run_hierarchical(self, task: str) -> TeamResult:
        from nil.agents.pool import AgentPool

        pool = AgentPool()
        pool.register_spec(self.leader)
        for member in self.members.values():
            pool.register_spec(member)

        sub_results = {}
        for subteam in self.subteams:
            sub_results[subteam.name] = subteam.run(task)

        orchestrator = OrchestratorAgent(
            domain=self.name,
            agent_pool=pool,
            blackboard=self.blackboard,
        )
        outcome = orchestrator.run(task)

        all_success = outcome.success and all(
            r.success for r in sub_results.values()
        )

        return TeamResult(
            success=all_success,
            output=outcome.final_output,
            cost=outcome.total_cost,
            subtask_results={"main": outcome, **sub_results},
        )

    def _run_swarm(self, task: str) -> TeamResult:
        from nil.agents.pool import AgentPool

        pool = AgentPool()
        for member in self.members.values():
            pool.register_spec(member)

        results = {}
        for role_key, spec in self.members.items():
            agent = pool.spawn(spec.role)
            subtask = _SwarmTask(description=f"{task} [{role_key}]")
            result = agent.run(subtask, self.blackboard)
            results[role_key] = result
            pool.release(agent)

        output = "; ".join(f"{k}: {v.output}" for k, v in results.items())
        return TeamResult(
            success=all(r.success for r in results.values()),
            output=output,
            cost=sum(r.cost for r in results.values()),
            subtask_results=results,
        )

    def _run_democratic(self, task: str) -> TeamResult:
        from nil.agents.consensus import ConsensusMechanism
        from nil.agents.pool import AgentPool

        pool = AgentPool()
        for member in self.members.values():
            pool.register_spec(member)

        proposals = {}
        for role_key, spec in self.members.items():
            agent = pool.spawn(spec.role)
            subtask = _SwarmTask(description=f"Analyze and propose for: {task} [{role_key}]")
            result = agent.run(subtask, self.blackboard)
            proposals[role_key] = result
            pool.release(agent)

        output = "; ".join(f"{k}: {v.output}" for k, v in proposals.items())
        return TeamResult(
            success=all(r.success for r in proposals.values()),
            output=output,
            cost=sum(r.cost for r in proposals.values()),
            subtask_results=proposals,
        )


@dataclass
class _SwarmTask:
    description: str = ""
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:8]
