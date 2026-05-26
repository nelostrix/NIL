"""Graph execution engine — DAG-based task decomposition and parallel execution."""

from nil.kernel.graph.critic import Critic, CritiqueVerdict
from nil.kernel.graph.graph_types import (
    AgentSpec,
    DependencyEdge,
    EdgeType,
    ExecutionGraph,
    ExecutionNode,
    GraphMetadata,
    GraphOutcome,
    NodeResult,
    NodeStatus,
    NodeType,
    SpeculativeNode,
    SubagentNode,
)
from nil.kernel.graph.planner import Planner, PlanningStrategy
from nil.kernel.graph.runner import GraphAgentRunner
from nil.kernel.graph.scheduler import GraphScheduler
from nil.kernel.graph.templates import GraphTemplate, TemplateLibrary

__all__ = [
    "ExecutionGraph",
    "ExecutionNode",
    "DependencyEdge",
    "AgentSpec",
    "GraphMetadata",
    "GraphOutcome",
    "NodeResult",
    "SpeculativeNode",
    "SubagentNode",
    "NodeType",
    "NodeStatus",
    "EdgeType",
    "Planner",
    "PlanningStrategy",
    "GraphScheduler",
    "Critic",
    "CritiqueVerdict",
    "GraphAgentRunner",
    "GraphTemplate",
    "TemplateLibrary",
]
