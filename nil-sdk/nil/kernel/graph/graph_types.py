from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class NodeType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    SUBGRAPH = "subgraph"
    MERGE = "merge"
    CONDITIONAL = "conditional"
    SPECULATIVE = "speculative"
    CRITIC = "critic"
    INPUT = "input"
    OUTPUT = "output"


class NodeStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLING_BACK = "rolling_back"


class EdgeType(str, Enum):
    DATA_FLOW = "data_flow"
    CONTROL_FLOW = "control_flow"
    CONDITIONAL = "conditional"


@dataclass
class AgentSpec:
    model: str = "claude-sonnet-4-20250514"
    provider: str = ""
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)
    max_tokens: int = 2048
    temperature: float = 0.0
    cost_budget: float = 0.05

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "cost_budget": self.cost_budget,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AgentSpec:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DependencyEdge:
    source: str
    target: str
    edge_type: EdgeType = EdgeType.DATA_FLOW
    condition: Optional[str] = None
    data_map: Optional[dict[str, str]] = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "condition": self.condition,
            "data_map": self.data_map,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DependencyEdge:
        return cls(
            source=d["source"],
            target=d["target"],
            edge_type=EdgeType(d.get("edge_type", "data_flow")),
            condition=d.get("condition"),
            data_map=d.get("data_map"),
        )


@dataclass
class NodeResult:
    node_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "success": self.success,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost": self.cost,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionNode:
    node_id: str
    node_type: NodeType = NodeType.LLM_CALL
    label: str = ""
    description: str = ""
    agent_spec: AgentSpec = field(default_factory=AgentSpec)
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    max_retries: int = 3
    timeout_ms: int = 30000
    fallback_node: Optional[str] = None
    rollback_node: Optional[str] = None
    task: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    status: NodeStatus = NodeStatus.PENDING
    result: Optional[NodeResult] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "description": self.description,
            "agent_spec": self.agent_spec.to_dict(),
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "max_retries": self.max_retries,
            "timeout_ms": self.timeout_ms,
            "fallback_node": self.fallback_node,
            "rollback_node": self.rollback_node,
            "task": self.task,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "status": self.status.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExecutionNode:
        node = cls(
            node_id=d["node_id"],
            node_type=NodeType(d.get("node_type", "llm_call")),
            label=d.get("label", ""),
            description=d.get("description", ""),
            agent_spec=AgentSpec.from_dict(d.get("agent_spec", {})),
            input_schema=d.get("input_schema", {}),
            output_schema=d.get("output_schema", {}),
            max_retries=d.get("max_retries", 3),
            timeout_ms=d.get("timeout_ms", 30000),
            fallback_node=d.get("fallback_node"),
            rollback_node=d.get("rollback_node"),
            task=d.get("task", ""),
            tool_name=d.get("tool_name", ""),
            tool_args=d.get("tool_args", {}),
            metadata=d.get("metadata", {}),
        )
        node.status = NodeStatus(d.get("status", "pending"))
        return node


@dataclass
class SubagentNode(ExecutionNode):
    sub_domain: str = ""
    sub_task: str = ""
    inherit_context: bool = True
    max_depth: int = 3

    def __post_init__(self):
        self.node_type = NodeType.SUBGRAPH


@dataclass
class SpeculativeNode(ExecutionNode):
    hypotheses: list[ExecutionNode] = field(default_factory=list)
    merge_strategy: str = "first_success"
    max_branches: int = 3

    def __post_init__(self):
        self.node_type = NodeType.SPECULATIVE


@dataclass
class GraphMetadata:
    domain: str = ""
    task: str = ""
    created_at: float = field(default_factory=time.time)
    total_cost_budget: float = 2.0
    max_concurrency: int = 5
    max_depth: int = 5
    execution_plan: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "task": self.task,
            "created_at": self.created_at,
            "total_cost_budget": self.total_cost_budget,
            "max_concurrency": self.max_concurrency,
            "max_depth": self.max_depth,
            "execution_plan": self.execution_plan,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GraphMetadata:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GraphOutcome:
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost: float = 0.0
    node_results: dict[str, NodeResult] = field(default_factory=dict)
    node_count: int = 0
    nodes_completed: int = 0
    nodes_failed: int = 0
    nodes_skipped: int = 0
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def success(output: Any, node_results: dict[str, NodeResult] = None, **kwargs) -> GraphOutcome:
        results = node_results or {}
        completed = sum(1 for r in results.values() if r.success)
        failed = sum(1 for r in results.values() if not r.success)
        total_tokens_in = sum(r.tokens_in for r in results.values())
        total_tokens_out = sum(r.tokens_out for r in results.values())
        total_cost = sum(r.cost for r in results.values())
        return GraphOutcome(
            success=True,
            output=output,
            node_results=results,
            node_count=len(results),
            nodes_completed=completed,
            nodes_failed=failed,
            total_tokens_in=total_tokens_in,
            total_tokens_out=total_tokens_out,
            total_cost=total_cost,
            **kwargs,
        )

    @staticmethod
    def failure(error: str, node_results: dict[str, NodeResult] = None, **kwargs) -> GraphOutcome:
        results = node_results or {}
        return GraphOutcome(
            success=False,
            error=error,
            node_results=results,
            node_count=len(results),
            **kwargs,
        )

    def log_summary(self) -> dict:
        return {
            "success": self.success,
            "node_count": self.node_count,
            "completed": self.nodes_completed,
            "failed": self.nodes_failed,
            "skipped": self.nodes_skipped,
            "duration_ms": round(self.duration_ms, 1),
            "tokens": self.total_tokens_in + self.total_tokens_out,
            "cost": round(self.total_cost, 6),
            "error": self.error,
        }


class ExecutionGraph:
    def __init__(
        self,
        nodes: dict[str, ExecutionNode] = None,
        edges: list[DependencyEdge] = None,
        metadata: Optional[GraphMetadata] = None,
    ):
        self.nodes: dict[str, ExecutionNode] = nodes or {}
        self.edges: list[DependencyEdge] = edges or []
        self.metadata: GraphMetadata = metadata or GraphMetadata()
        self._outputs: dict[str, Any] = {}

    def add_node(self, node: ExecutionNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: DependencyEdge) -> None:
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[ExecutionNode]:
        return self.nodes.get(node_id)

    def get_dependents(self, node_id: str) -> list[str]:
        return [e.target for e in self.edges if e.source == node_id]

    def get_dependencies(self, node_id: str) -> list[str]:
        return [e.source for e in self.edges if e.target == node_id]

    def get_upstream_output(self, node_id: str, field: str = "") -> Any:
        deps = self.get_dependencies(node_id)
        for dep_id in deps:
            if dep_id in self._outputs:
                output = self._outputs[dep_id]
                if field:
                    if isinstance(output, dict):
                        return output.get(field)
                else:
                    return output
        return None

    def set_node_output(self, node_id: str, output: Any) -> None:
        self._outputs[node_id] = output

    def get_node_output(self, node_id: str) -> Any:
        return self._outputs.get(node_id)

    def mark_completed(self, node_id: str) -> None:
        node = self.nodes.get(node_id)
        if node:
            node.status = NodeStatus.COMPLETED

    def mark_failed(self, node_id: str, error: str) -> None:
        node = self.nodes.get(node_id)
        if node:
            node.status = NodeStatus.FAILED
            node.result = NodeResult(node_id=node_id, success=False, error=error)

    def activate_fallback(self, failed_node_id: str, fallback_id: str) -> None:
        self.mark_failed(failed_node_id, "Activating fallback")
        fallback = self.nodes.get(fallback_id)
        if fallback:
            fallback.status = NodeStatus.READY

    def activate_rollback(self, failed_node_id: str, rollback_id: str) -> None:
        failed = self.nodes.get(failed_node_id)
        if failed:
            failed.status = NodeStatus.ROLLING_BACK
        rollback = self.nodes.get(rollback_id)
        if rollback:
            rollback.status = NodeStatus.READY

    def find_ready_nodes(self, max_concurrency: int = 5) -> list[ExecutionNode]:
        ready: list[ExecutionNode] = []
        for node_id, node in self.nodes.items():
            if node.status != NodeStatus.PENDING:
                continue
            deps = self.get_dependencies(node_id)
            all_done = all(
                self.nodes[d].status == NodeStatus.COMPLETED for d in deps
            )
            if all_done:
                condition_met = True
                for edge in self.edges:
                    if edge.target == node_id and edge.edge_type == EdgeType.CONDITIONAL:
                        source_output = self._outputs.get(edge.source)
                        if edge.condition and source_output is not None:
                            if isinstance(source_output, dict):
                                val = source_output.get(edge.condition)
                                if val is None or val is False or val == "fail":
                                    condition_met = False
                            elif str(source_output).lower() != edge.condition.lower():
                                condition_met = False
                if condition_met:
                    node.status = NodeStatus.READY
                    ready.append(node)
                    if len(ready) >= max_concurrency:
                        break
        return ready

    def all_completed(self) -> bool:
        return all(
            n.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED, NodeStatus.FAILED)
            for n in self.nodes.values()
        )

    def any_failed(self) -> bool:
        return any(n.status == NodeStatus.FAILED for n in self.nodes.values())

    def collect_outputs(self) -> dict[str, Any]:
        return dict(self._outputs)

    def to_dict(self) -> dict:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExecutionGraph:
        graph = cls(
            nodes={nid: ExecutionNode.from_dict(nd) for nid, nd in d.get("nodes", {}).items()},
            edges=[DependencyEdge.from_dict(e) for e in d.get("edges", [])],
            metadata=GraphMetadata.from_dict(d.get("metadata", {})),
        )
        return graph

    def clone(self) -> ExecutionGraph:
        return ExecutionGraph.from_dict(self.to_dict())
