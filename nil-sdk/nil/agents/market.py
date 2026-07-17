from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from nil.agents.blackboard import Blackboard
from nil.agents.pool import _PooledAgent


@dataclass
class Bid:
    agent_name: str = ""
    total_cost: float = 0.0
    capability_score: float = 0.0
    load_factor: float = 0.0
    past_success_rate: float = 1.0


@dataclass
class Subtask:
    id: str = ""
    description: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    max_budget: float = 0.1

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:8]


class MarketCoordinator:
    def __init__(self, blackboard: Optional[Blackboard] = None):
        self.blackboard = blackboard or Blackboard()

    def assign(self, subtask: Subtask, agents: list[_PooledAgent]) -> _PooledAgent:
        bids: list[tuple[_PooledAgent, Bid]] = []
        for agent in agents:
            bid = self._compute_bid(agent, subtask)
            bids.append((agent, bid))

        bids.sort(key=lambda x: x[1].total_cost)

        if not bids:
            raise ValueError("No agents available to assign subtask")

        winner = bids[0][0]
        self.blackboard.write(
            f"bid_{subtask.id}",
            {
                "winner": winner.name,
                "bid": bids[0][1].total_cost,
                "all_bids": [(a.name, b.total_cost) for a, b in bids],
            },
        )
        return winner

    def _compute_bid(self, agent: _PooledAgent, subtask: Subtask) -> Bid:
        capability_score = self._capability_match(agent, subtask)
        load_factor = 0.5
        past_success_rate = 1.0

        total_cost = (1.0 / max(capability_score, 0.01)) * 0.1 + load_factor * 0.05
        return Bid(
            agent_name=agent.name,
            total_cost=round(total_cost, 4),
            capability_score=capability_score,
            load_factor=load_factor,
            past_success_rate=past_success_rate,
        )

    def _capability_match(self, agent: _PooledAgent, subtask: Subtask) -> float:
        if not subtask.required_capabilities or not agent.tools:
            return 0.5
        matches = sum(1 for cap in subtask.required_capabilities if cap in agent.tools)
        return matches / max(len(subtask.required_capabilities), 1)
