"""Runtime Observability — structured events, trace viewer, flame graphs, token timelines, reasoning trees, memory browser, evolution diffs, strategy lineage, and live debugging."""

from nil.observability.events import (
    EventType,
    ObservabilityEvent,
    EventRecorder,
    ObservabilityStore,
)
from nil.observability.trace_viewer import TraceViewer, TimelineView, TimelineRow
from nil.observability.flame_graph import FlameGraphGenerator, FlameGraph, FlameNode
from nil.observability.token_timeline import TokenTimeline, TokenReport, TokenStep
from nil.observability.reasoning_tree import ReasoningTree, ReasoningNode
from nil.observability.memory_browser import MemoryBrowser
from nil.observability.evolution_diff import EvolutionDiffViewer, EvolutionDiff
from nil.observability.strategy_lineage import StrategyLineage, LineageGraph, LineageNode
from nil.observability.debugger import LiveDebugger, DebugCommand, Breakpoint, AgentSnapshot

__all__ = [
    "EventType", "ObservabilityEvent", "EventRecorder", "ObservabilityStore",
    "TraceViewer", "TimelineView", "TimelineRow",
    "FlameGraphGenerator", "FlameGraph", "FlameNode",
    "TokenTimeline", "TokenReport", "TokenStep",
    "ReasoningTree", "ReasoningNode",
    "MemoryBrowser",
    "EvolutionDiffViewer", "EvolutionDiff",
    "StrategyLineage", "LineageGraph", "LineageNode",
    "LiveDebugger", "DebugCommand", "Breakpoint", "AgentSnapshot",
]
