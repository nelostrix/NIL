"""Multi-agent coordination layer — agent roles, blackboard, orchestration, consensus, and team management."""

from nil.agents.blackboard import Blackboard
from nil.agents.consensus import ConsensusMechanism
from nil.agents.framework import MultiAgentNil
from nil.agents.messages import AgentMessage, MessageBus, MessageType
from nil.agents.orchestrator import OrchestratorAgent
from nil.agents.teams import AgentTeam
from nil.agents.pool import AgentPool
from nil.agents.roles import AgentRole, AgentSpec
from nil.agents.market import MarketCoordinator

__all__ = [
    "AgentRole",
    "AgentPool",
    "AgentSpec",
    "Blackboard",
    "AgentMessage",
    "OrchestratorAgent",
    "MessageType",
    "MessageBus",
    "ConsensusMechanism",
    "MarketCoordinator",
    "AgentTeam",
    "MultiAgentNil",
]
