from __future__ import annotations

import logging
import time
from typing import Any, Optional

from nil.config import NilConfig
from nil.domain import Domain
from nil.kernel.graph.critic import Critic, CritiqueVerdict, ErrorFreeValidator, SchemaValidator
from nil.kernel.graph.graph_types import (
    AgentSpec,
    ExecutionGraph,
    ExecutionNode,
    GraphOutcome,
    NodeResult,
    NodeStatus,
    NodeType,
)
from nil.kernel.graph.planner import Planner, PlanningStrategy
from nil.kernel.graph.scheduler import GraphScheduler
from nil.kernel.graph.templates import TemplateLibrary
from nil.kernel.outcome import RunOutcome, StepRecord
from nil.kernel.tool_manager import ToolManager
from nil.kernel.tracer import Tracer
from nil.memory.store import PersistentMemory

logger = logging.getLogger("nil.kernel.graph.runner")


class GraphAgentRunner:
    def __init__(
        self,
        domain: Domain,
        model: str,
        config: NilConfig,
        memory: PersistentMemory,
        provider: Optional[str] = None,
        max_concurrency: int = 5,
    ):
        self.domain = domain
        self.model = model
        self.config = config
        self.memory = memory
        self.tool_manager = ToolManager(domain.tools)
        self._provider = self._init_provider(provider)
        self.template_library = TemplateLibrary()

        self.scheduler = GraphScheduler(
            max_concurrency=max_concurrency,
            max_cost_per_run=config.metadata.get("cost_budget", 2.0) if hasattr(config, "metadata") else 2.0,
        )
        self.planner = Planner(llm_provider=self._provider)
        self.critic = Critic()
        self.critic.add_validator(SchemaValidator())
        self.critic.add_validator(ErrorFreeValidator())

        self._register_default_executors()

    def _init_provider(self, provider_name: Optional[str] = None) -> Any:
        name = provider_name or self._detect_provider()
        if name == "anthropic":
            from nil.providers.anthropic import AnthropicProvider
            return AnthropicProvider(model=self.model, config=self.config)
        elif name == "openai":
            from nil.providers.openai import OpenAIProvider
            return OpenAIProvider(model=self.model, config=self.config)
        elif name == "gemini":
            from nil.providers.gemini import GeminiProvider
            return GeminiProvider(model=self.model, config=self.config)
        else:
            from nil.providers.stub import StubProvider
            return StubProvider(model=self.model, config=self.config)

    def _detect_provider(self) -> str:
        import os
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            return "openai"
        elif os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        return "stub"

    def _register_default_executors(self) -> None:
        from nil.kernel.graph.scheduler import NodeExecutor

        def llm_executor(node: ExecutionNode, graph: ExecutionGraph) -> NodeResult:
            task = node.task
            upstream = graph.get_upstream_output(node.node_id)
            if upstream:
                if isinstance(upstream, dict):
                    context_str = "; ".join(f"{k}: {v}" for k, v in upstream.items() if not k.startswith("_"))
                    task = f"{task}\n\nContext from upstream: {context_str}"
                else:
                    task = f"{task}\n\nContext: {upstream}"

            messages = [{"role": "user", "content": task}]
            system = node.agent_spec.system_prompt or (
                f"You are a specialized agent for {node.label}. "
                f"{node.description}"
            )

            try:
                tool_schemas = None
                tool_names = node.agent_spec.tools or self.tool_manager.names
                if tool_names:
                    tool_schemas = self.tool_manager.to_schemas()

                response = self._provider.chat(
                    system=system,
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                )

                content = response.get("content", "")
                tokens_in = response.get("tokens_in", 0)
                tokens_out = response.get("tokens_out", 0)

                return NodeResult(
                    node_id=node.node_id,
                    success=True,
                    output=content,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost=0.0,
                )
            except Exception as e:
                return NodeResult(
                    node_id=node.node_id,
                    success=False,
                    error=str(e),
                )

        def tool_executor(node: ExecutionNode, graph: ExecutionGraph) -> NodeResult:
            tool_name = node.tool_name
            tool_args = dict(node.tool_args)

            upstream = graph.get_upstream_output(node.node_id)
            if upstream and isinstance(upstream, dict):
                for k, v in upstream.items():
                    if k not in tool_args:
                        tool_args[k] = v

            if not tool_name and self.tool_manager.names:
                tool_name = self.tool_manager.names[0]

            if not tool_name:
                return NodeResult(
                    node_id=node.node_id,
                    success=False,
                    error="No tool specified for TOOL_CALL node",
                )

            try:
                result = self.tool_manager.execute(tool_name, tool_args)
                if result["error"]:
                    return NodeResult(
                        node_id=node.node_id,
                        success=False,
                        error=result["error"],
                        duration_ms=result["duration_ms"],
                        metadata={"tool_name": tool_name, "tool_args": tool_args},
                    )
                return NodeResult(
                    node_id=node.node_id,
                    success=True,
                    output=result["output"],
                    duration_ms=result["duration_ms"],
                    metadata={"tool_name": tool_name, "tool_args": tool_args},
                )
            except Exception as e:
                return NodeResult(
                    node_id=node.node_id,
                    success=False,
                    error=str(e),
                )

        def critic_executor(node: ExecutionNode, graph: ExecutionGraph) -> NodeResult:
            dep_ids = graph.get_dependencies(node.node_id)
            outputs = {did: graph.get_node_output(did) for did in dep_ids}
            result = self.critic.evaluate(node, outputs)
            return NodeResult(
                node_id=node.node_id,
                success=result.passed,
                output={"verdict": result.verdict.value, "issues": [i.to_dict() for i in result.issues]},
                metadata={"verdict": result.verdict.value, "issue_count": len(result.issues)},
            )

        self.scheduler.register_executor(NodeType.LLM_CALL, llm_executor)
        self.scheduler.register_executor(NodeType.TOOL_CALL, tool_executor)
        self.scheduler.register_executor(NodeType.CRITIC, critic_executor)

    def build_system_prompt(self) -> str:
        parts = [
            f"You are an AI agent specialized in the domain: {self.domain.name}.",
            "",
            "Your objectives:",
        ]
        for obj in self.domain.objectives:
            parts.append(f"  - {obj}")
        skills = self.memory.list_skills(self.domain.name)
        if skills:
            parts.append("")
            parts.append("Learned skills from past experience:")
            for skill in skills:
                parts.append(f"  - {skill}")
        parts.append("")
        parts.append(
            "Available tools: " + ", ".join(self.tool_manager.names)
            if self.tool_manager.names else "No tools available."
        )
        return "\n".join(parts)

    def run(
        self,
        task: str,
        context: dict[str, Any] = None,
        strategy: Optional[PlanningStrategy] = None,
        tracer: Optional[Tracer] = None,
    ) -> tuple[GraphOutcome, RunOutcome]:
        context = context or {}
        tracer = tracer or Tracer(run_id=f"graph-{int(time.time())}", domain=self.domain.name)

        templates = self.template_library.find_for_task(task, self.domain.name)
        if templates:
            logger.info("Found %d matching templates for task, using '%s'",
                        len(templates), templates[0].name)
            graph = templates[0].instantiate(task, self.domain.name, self.tool_manager.names, context)
        else:
            graph = self.planner.plan(
                task=task,
                domain_name=self.domain.name,
                tools=self.tool_manager.names,
                strategy=strategy,
                context=context,
            )

        graph_outcome = self.scheduler.execute(graph, context=context)

        outcome = self._to_run_outcome(graph, graph_outcome, task, tracer)
        return graph_outcome, outcome

    def plan(
        self,
        task: str,
        context: dict[str, Any] = None,
        strategy: Optional[PlanningStrategy] = None,
    ) -> ExecutionGraph:
        context = context or {}
        templates = self.template_library.find_for_task(task, self.domain.name)
        if templates:
            return templates[0].instantiate(task, self.domain.name, self.tool_manager.names, context)
        return self.planner.plan(
            task=task,
            domain_name=self.domain.name,
            tools=self.tool_manager.names,
            strategy=strategy,
            context=context,
        )

    def _to_run_outcome(
        self, graph: ExecutionGraph, graph_outcome: GraphOutcome, task: str, tracer: Tracer
    ) -> RunOutcome:
        outcome = RunOutcome(
            run_id=tracer.run_id,
            task=task,
            success=graph_outcome.success,
            output=graph_outcome.output,
            duration=graph_outcome.duration_ms / 1000.0,
            total_tokens_in=graph_outcome.total_tokens_in,
            total_tokens_out=graph_outcome.total_tokens_out,
        )
        outcome.metadata["graph"] = {
            "node_count": graph_outcome.node_count,
            "completed": graph_outcome.nodes_completed,
            "failed": graph_outcome.nodes_failed,
            "skipped": graph_outcome.nodes_skipped,
        }

        for node_id, node_result in graph_outcome.node_results.items():
            node = graph.get_node(node_id)
            step_type = f"graph_{node.node_type.value}" if node else "graph_node"
            tracer.record_event(
                event_type=step_type,
                content={
                    "node_id": node_id,
                    "success": node_result.success,
                    "output": str(node_result.output)[:200] if node_result.output else None,
                    "error": node_result.error,
                    "duration_ms": node_result.duration_ms,
                },
            )

        if not graph_outcome.success:
            outcome.error = graph_outcome.error

        return outcome
