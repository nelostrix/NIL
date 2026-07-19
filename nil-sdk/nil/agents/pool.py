from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from nil.agents.roles import AgentResult, AgentRole, AgentSpec


class AgentPool:
    def __init__(self):
        self.persistent_agents: dict[str, object] = {}
        self.pooled_agents: dict[str, list[_PooledAgent]] = defaultdict(list)
        self.agent_specs: dict[str, AgentSpec] = {}
        self._agent_count: dict[str, int] = defaultdict(int)

    def register_spec(self, spec: AgentSpec) -> None:
        self.agent_specs[spec.role.value] = spec

    def get_spec(self, role: AgentRole) -> Optional[AgentSpec]:
        return self.agent_specs.get(role.value)

<<<<<<< HEAD
    def spawn(self, role: AgentRole ) -> _PooledAgent:
=======
    def spawn(self, role: AgentRole) -> _PooledAgent:
>>>>>>> f0750a2c14fede477692762ed0c0ffda3ec4e090
        spec = self.get_spec(role)
        if not spec:
            raise ValueError(f"No spec registered for role {role.value}")
        self._agent_count[role.value] += 1
        return _PooledAgent(
            name=f"{spec.name}-{self._agent_count[role.value]}",
            role=role,
            spec=spec,
        )

    def acquire(self, role: AgentRole) -> _PooledAgent:
        pool = self.pooled_agents.get(role.value, [])
        if pool:
            return pool.pop()
        if role.value not in self.agent_specs:
            raise ValueError(f"No spec registered for role {role.value}") 
        return self.spawn(role)

    def release(self, agent: _PooledAgent) -> None:
        self.pooled_agents[agent.role.value].append(agent)

    def list_agents(self) -> list[dict]:
        agents = []
        for role_key, spec in self.agent_specs.items():
            pool_size = len(self.pooled_agents.get(role_key, []))
            total = self._agent_count.get(role_key, 0)
            agents.append(
                {
                    "role": role_key,
                    "spec_name": spec.name,
                    "pool_size": pool_size,
                    "total_spawned": total,
                }
            )
        return agents


@dataclass
class _PooledAgent:
    name: str = ""
    role: AgentRole = AgentRole.INVESTIGATOR
    spec: Optional[AgentSpec] = None
    delegation_allowed: bool = False
    tools: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.spec:
            self.delegation_allowed = self.spec.delegation_allowed
            self.tools = list(self.spec.tools)

    def run(self, subtask: object, blackboard: object = None) -> AgentResult:
        task_desc = getattr(subtask, "description", str(subtask))
        return AgentResult(
            subtask_id=getattr(subtask, "id", "") or uuid.uuid4().hex[:8],
            success=True,
            output=f"[{self.name}] executed: {task_desc[:80]}",
            cost=0.001,
            duration=0.5,
        )
