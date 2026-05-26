"""
Kernel — the agent execution engine.

The kernel handles the actual run loop: sending prompts to the LLM,
executing tool calls, recording every step in the tracer, and
producing a RunOutcome.
"""

from nil.kernel.outcome import RunOutcome
from nil.kernel.runner import KernelRunner
from nil.kernel.tool_manager import ToolManager
from nil.kernel.tracer import Tracer

# Graph execution engine
from nil.kernel.graph import (
    AgentSpec,
    Critic,
    CritiqueVerdict,
    DependencyEdge,
    EdgeType,
    ExecutionGraph,
    ExecutionNode,
    GraphAgentRunner,
    GraphOutcome,
    GraphTemplate,
    GraphMetadata,
    NodeResult,
    NodeStatus,
    NodeType,
    Planner,
    PlanningStrategy,
    SpeculativeNode,
    SubagentNode,
    TemplateLibrary,
)

__all__ = [
    "KernelRunner",
    "Tracer",
    "RunOutcome",
    "ToolManager",
    # Graph execution
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
