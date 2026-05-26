from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from nil.kernel.graph.graph_types import (
    AgentSpec,
    DependencyEdge,
    EdgeType,
    ExecutionGraph,
    ExecutionNode,
    GraphMetadata,
    NodeType,
)

logger = logging.getLogger("nil.kernel.graph.planner")


class PlanningStrategy(str, Enum):
    MONOLITHIC = "monolithic"
    SEQUENTIAL = "sequential"
    PARALLEL_FAN_OUT = "parallel_fan_out"
    EXPLORE_EXPLOIT = "explore_exploit"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    CONTINGENCY = "contingency"


@dataclass
class PlanStep:
    label: str
    description: str
    node_type: NodeType = NodeType.LLM_CALL
    depends_on: list[str] = field(default_factory=list)
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    task: str = ""
    agent_spec: Optional[AgentSpec] = None
    fallback_node: Optional[str] = None
    rollback_node: Optional[str] = None
    output_schema: dict = field(default_factory=dict)


class Planner:
    def __init__(self, llm_provider=None):
        self._llm = llm_provider
        self._templates: dict[str, list[PlanStep]] = {}

    def register_template(self, name: str, steps: list[PlanStep]) -> None:
        self._templates[name] = steps

    def plan(
        self,
        task: str,
        domain_name: str = "",
        tools: list[str] = None,
        strategy: Optional[PlanningStrategy] = None,
        context: dict[str, Any] = None,
    ) -> ExecutionGraph:
        inferred = self._infer_strategy(task, tools or [])
        strategy = strategy or inferred
        context = context or {}

        method = getattr(self, f"_plan_{strategy.value}", self._plan_monolithic)
        return method(task, domain_name, tools or [], context)

    def _infer_strategy(self, task: str, tools: list[str]) -> PlanningStrategy:
        task_lower = task.lower()
        if not tools:
            return PlanningStrategy.MONOLITHIC
        if any(w in task_lower for w in ["investigate", "diagnose", "find why", "root cause"]):
            return PlanningStrategy.HYPOTHESIS_TESTING
        if any(w in task_lower for w in ["deploy", "migrate", "rollback", "release"]):
            return PlanningStrategy.CONTINGENCY
        if any(w in task_lower for w in ["and", "also", "then", "compare"]):
            return PlanningStrategy.PARALLEL_FAN_OUT
        if len(tools) >= 3:
            return PlanningStrategy.EXPLORE_EXPLOIT
        return PlanningStrategy.SEQUENTIAL

    def _plan_monolithic(
        self, task: str, domain: str, tools: list[str], context: dict
    ) -> ExecutionGraph:
        graph = ExecutionGraph(metadata=GraphMetadata(
            domain=domain, task=task, execution_plan="monolithic",
        ))
        input_node = ExecutionNode(
            node_id=_nid("input"), node_type=NodeType.INPUT, task=task,
            label="Input", description="Task input",
        )
        work_node = ExecutionNode(
            node_id=_nid("work"), node_type=NodeType.LLM_CALL, task=task,
            label="Execute", description="Execute the task in a single step",
            agent_spec=AgentSpec(tools=tools),
        )
        output_node = ExecutionNode(
            node_id=_nid("output"), node_type=NodeType.OUTPUT,
            label="Output", description="Collect final output",
        )
        graph.add_node(input_node)
        graph.add_node(work_node)
        graph.add_node(output_node)
        graph.add_edge(DependencyEdge(source=input_node.node_id, target=work_node.node_id))
        graph.add_edge(DependencyEdge(source=work_node.node_id, target=output_node.node_id))
        return graph

    def _plan_sequential(
        self, task: str, domain: str, tools: list[str], context: dict
    ) -> ExecutionGraph:
        graph = ExecutionGraph(metadata=GraphMetadata(
            domain=domain, task=task, execution_plan="sequential",
        ))
        input_node = ExecutionNode(
            node_id=_nid("input"), node_type=NodeType.INPUT, task=task,
            label="Input",
        )
        graph.add_node(input_node)
        prev_id = input_node.node_id

        for i, tool in enumerate(tools):
            step_node = ExecutionNode(
                node_id=_nid(f"step_{i}"),
                node_type=NodeType.TOOL_CALL,
                task=f"Step {i+1}: {tool}",
                label=f"{tool}",
                description=f"Execute tool: {tool}",
                tool_name=tool,
                agent_spec=AgentSpec(tools=[tool]),
            )
            graph.add_node(step_node)
            graph.add_edge(DependencyEdge(source=prev_id, target=step_node.node_id))
            prev_id = step_node.node_id

        output_node = ExecutionNode(
            node_id=_nid("output"), node_type=NodeType.OUTPUT, label="Output",
        )
        graph.add_node(output_node)
        graph.add_edge(DependencyEdge(source=prev_id, target=output_node.node_id))
        return graph

    def _plan_parallel_fan_out(
        self, task: str, domain: str, tools: list[str], context: dict
    ) -> ExecutionGraph:
        graph = ExecutionGraph(metadata=GraphMetadata(
            domain=domain, task=task, execution_plan="parallel_fan_out",
        ))
        input_node = ExecutionNode(
            node_id=_nid("input"), node_type=NodeType.INPUT, task=task, label="Input",
        )
        graph.add_node(input_node)

        fan_out_nodes = []
        for i, tool in enumerate(tools):
            work_node = ExecutionNode(
                node_id=_nid(f"parallel_{i}"),
                node_type=NodeType.TOOL_CALL,
                task=f"Parallel: {tool}",
                label=f"{tool}",
                description=f"Execute tool in parallel: {tool}",
                tool_name=tool,
                agent_spec=AgentSpec(tools=[tool]),
            )
            graph.add_node(work_node)
            graph.add_edge(DependencyEdge(source=input_node.node_id, target=work_node.node_id))
            fan_out_nodes.append(work_node.node_id)

        merge_node = ExecutionNode(
            node_id=_nid("merge"), node_type=NodeType.MERGE,
            label="Merge", description="Merge parallel results",
        )
        graph.add_node(merge_node)
        for nid in fan_out_nodes:
            graph.add_edge(DependencyEdge(source=nid, target=merge_node.node_id))

        output_node = ExecutionNode(
            node_id=_nid("output"), node_type=NodeType.OUTPUT, label="Output",
        )
        graph.add_node(output_node)
        graph.add_edge(DependencyEdge(source=merge_node.node_id, target=output_node.node_id))
        return graph

    def _plan_explore_exploit(
        self, task: str, domain: str, tools: list[str], context: dict
    ) -> ExecutionGraph:
        graph = ExecutionGraph(metadata=GraphMetadata(
            domain=domain, task=task, execution_plan="explore_exploit",
        ))
        input_node = ExecutionNode(
            node_id=_nid("input"), node_type=NodeType.INPUT, task=task, label="Input",
        )
        graph.add_node(input_node)

        explore_nodes = []
        for i, tool in enumerate(tools[:3]):
            explore = ExecutionNode(
                node_id=_nid(f"explore_{i}"),
                node_type=NodeType.TOOL_CALL,
                task=f"Investigate using {tool}",
                label=f"Explore:{tool}",
                description=f"Gather data using {tool}",
                tool_name=tool,
            )
            graph.add_node(explore)
            graph.add_edge(DependencyEdge(source=input_node.node_id, target=explore.node_id))
            explore_nodes.append(explore.node_id)

        merge_node = ExecutionNode(
            node_id=_nid("merge"), node_type=NodeType.MERGE,
            label="Analyze", description="Correlate investigation results",
        )
        graph.add_node(merge_node)
        for nid in explore_nodes:
            graph.add_edge(DependencyEdge(source=nid, target=merge_node.node_id))

        critic_node = ExecutionNode(
            node_id=_nid("critic"), node_type=NodeType.CRITIC,
            label="Critique", description="Validate findings",
            agent_spec=AgentSpec(tools=[]),
        )
        graph.add_node(critic_node)
        graph.add_edge(DependencyEdge(source=merge_node.node_id, target=critic_node.node_id))

        act_node = ExecutionNode(
            node_id=_nid("act"),
            node_type=NodeType.LLM_CALL,
            task=f"Execute fix: {task}",
            label="Act", description="Execute based on findings",
            agent_spec=AgentSpec(tools=tools),
        )
        graph.add_node(act_node)
        graph.add_edge(DependencyEdge(
            source=critic_node.node_id, target=act_node.node_id,
            edge_type=EdgeType.CONDITIONAL, condition="pass",
        ))
        graph.add_edge(DependencyEdge(
            source=critic_node.node_id, target=act_node.node_id,
            edge_type=EdgeType.CONDITIONAL, condition="needs_revision",
        ))

        output_node = ExecutionNode(
            node_id=_nid("output"), node_type=NodeType.OUTPUT, label="Output",
        )
        graph.add_node(output_node)
        graph.add_edge(DependencyEdge(source=act_node.node_id, target=output_node.node_id))
        return graph

    def _plan_hypothesis_testing(
        self, task: str, domain: str, tools: list[str], context: dict
    ) -> ExecutionGraph:
        graph = ExecutionGraph(metadata=GraphMetadata(
            domain=domain, task=task, execution_plan="hypothesis_testing",
        ))
        input_node = ExecutionNode(
            node_id=_nid("input"), node_type=NodeType.INPUT, task=task, label="Input",
        )
        graph.add_node(input_node)

        hypotheses = []
        hyp_descriptions = context.get("hypotheses", [
            "Check database for errors or slow queries",
            "Check application logs for exceptions",
            "Check infrastructure metrics (CPU, memory, network)",
        ])
        for i, hyp in enumerate(hyp_descriptions[:3]):
            hyp_node = ExecutionNode(
                node_id=_nid(f"hyp_{i}"),
                node_type=NodeType.LLM_CALL,
                task=hyp,
                label=f"Hypothesis {i+1}",
                description=hyp,
                agent_spec=AgentSpec(
                    tools=tools[:2] if tools else [],
                    temperature=0.3,
                ),
            )
            graph.add_node(hyp_node)
            graph.add_edge(DependencyEdge(source=input_node.node_id, target=hyp_node.node_id))
            hypotheses.append(hyp_node.node_id)

        merge_node = ExecutionNode(
            node_id=_nid("merge"), node_type=NodeType.MERGE,
            label="Correlate", description="Merge hypothesis results",
        )
        graph.add_node(merge_node)
        for nid in hypotheses:
            graph.add_edge(DependencyEdge(source=nid, target=merge_node.node_id))

        conclusion_node = ExecutionNode(
            node_id=_nid("conclusion"),
            node_type=NodeType.LLM_CALL,
            task=f"Analyze all findings and determine root cause for: {task}",
            label="Root Cause Analysis",
            description="Determine which hypothesis is correct",
            agent_spec=AgentSpec(tools=[], temperature=0.1),
        )
        graph.add_node(conclusion_node)
        graph.add_edge(DependencyEdge(source=merge_node.node_id, target=conclusion_node.node_id))

        output_node = ExecutionNode(
            node_id=_nid("output"), node_type=NodeType.OUTPUT, label="Output",
        )
        graph.add_node(output_node)
        graph.add_edge(DependencyEdge(source=conclusion_node.node_id, target=output_node.node_id))
        return graph

    def _plan_contingency(
        self, task: str, domain: str, tools: list[str], context: dict
    ) -> ExecutionGraph:
        graph = ExecutionGraph(metadata=GraphMetadata(
            domain=domain, task=task, execution_plan="contingency",
        ))
        input_node = ExecutionNode(
            node_id=_nid("input"), node_type=NodeType.INPUT, task=task, label="Input",
        )
        graph.add_node(input_node)

        precheck = ExecutionNode(
            node_id=_nid("precheck"),
            node_type=NodeType.TOOL_CALL,
            task="Pre-flight check",
            label="Pre-check",
            description="Verify prerequisites before main action",
            tool_name=tools[0] if tools else "",
            agent_spec=AgentSpec(tools=[tools[0]] if tools else []),
        )
        graph.add_node(precheck)
        graph.add_edge(DependencyEdge(source=input_node.node_id, target=precheck.node_id))

        primary = ExecutionNode(
            node_id=_nid("primary"),
            node_type=NodeType.LLM_CALL,
            task=f"Execute primary action: {task}",
            label="Primary Action",
            description="Execute the main operation",
            agent_spec=AgentSpec(tools=tools),
        )
        graph.add_node(primary)
        graph.add_edge(DependencyEdge(source=precheck.node_id, target=primary.node_id))

        verify = ExecutionNode(
            node_id=_nid("verify"),
            node_type=NodeType.TOOL_CALL,
            task="Verify success",
            label="Verify",
            description="Confirm the action succeeded",
            tool_name=tools[-1] if len(tools) > 1 else "",
        )
        graph.add_node(verify)
        graph.add_edge(DependencyEdge(source=primary.node_id, target=verify.node_id))

        rollback = ExecutionNode(
            node_id=_nid("rollback"),
            node_type=NodeType.LLM_CALL,
            task=f"Rollback: Undo the changes from: {task}",
            label="Rollback",
            description="Rollback if primary action fails",
            agent_spec=AgentSpec(tools=tools),
        )
        graph.add_node(rollback)
        primary.fallback_node = rollback.node_id
        primary.rollback_node = rollback.node_id

        output_node = ExecutionNode(
            node_id=_nid("output"), node_type=NodeType.OUTPUT, label="Output",
        )
        graph.add_node(output_node)
        graph.add_edge(DependencyEdge(source=verify.node_id, target=output_node.node_id))
        graph.add_edge(DependencyEdge(source=rollback.node_id, target=output_node.node_id))
        return graph

    def plan_from_template(
        self, template_name: str, task: str, domain: str, tools: list[str], context: dict = None
    ) -> Optional[ExecutionGraph]:
        steps = self._templates.get(template_name)
        if not steps:
            return None
        context = context or {}
        graph = ExecutionGraph(metadata=GraphMetadata(
            domain=domain, task=task, execution_plan=f"template:{template_name}",
        ))
        input_node = ExecutionNode(
            node_id=_nid("input"), node_type=NodeType.INPUT, task=task, label="Input",
        )
        graph.add_node(input_node)
        prev_id = input_node.node_id
        created: dict[str, str] = {}

        for step in steps:
            node_id = _nid(step.label)
            agent = step.agent_spec or AgentSpec(tools=tools)
            node = ExecutionNode(
                node_id=node_id,
                node_type=step.node_type,
                label=step.label,
                description=step.description,
                task=step.task or task,
                tool_name=step.tool_name,
                tool_args=step.tool_args,
                agent_spec=agent,
                output_schema=step.output_schema,
            )
            if step.fallback_node and step.fallback_node in created:
                node.fallback_node = created[step.fallback_node]
            if step.rollback_node and step.rollback_node in created:
                node.rollback_node = created[step.rollback_node]
            graph.add_node(node)
            created[step.label] = node_id

            if step.depends_on:
                for dep_label in step.depends_on:
                    if dep_label in created:
                        graph.add_edge(DependencyEdge(source=created[dep_label], target=node_id))
            else:
                graph.add_edge(DependencyEdge(source=prev_id, target=node_id))
            prev_id = node_id

        output_node = ExecutionNode(
            node_id=_nid("output"), node_type=NodeType.OUTPUT, label="Output",
        )
        graph.add_node(output_node)
        graph.add_edge(DependencyEdge(source=prev_id, target=output_node.node_id))
        return graph

    def plan_with_llm(
        self, task: str, domain: str, tools: list[str], context: dict = None
    ) -> ExecutionGraph:
        if not self._llm:
            return self.plan(task, domain, tools, context=context)
        context = context or {}
        strategy_prompt = (
            f"Given the task '{task}' with tools {tools}, decompose it into a DAG. "
            "Respond with a JSON object having 'strategy' and 'steps' array. "
            "Valid strategies: monolithic, sequential, parallel_fan_out, explore_exploit, hypothesis_testing, contingency."
        )
        try:
            response = self._llm.chat(system="You are a planning assistant.", messages=[
                {"role": "user", "content": strategy_prompt},
            ])
            content = response.get("content", "")
            if "sequential" in content:
                return self._plan_sequential(task, domain, tools, context)
            elif "parallel" in content:
                return self._plan_parallel_fan_out(task, domain, tools, context)
            elif "hypothesis" in content:
                return self._plan_hypothesis_testing(task, domain, tools, context)
            elif "contingency" in content:
                return self._plan_contingency(task, domain, tools, context)
            elif "explore" in content:
                return self._plan_explore_exploit(task, domain, tools, context)
        except Exception as e:
            logger.warning("LLM planning failed, falling back: %s", e)
        return self._plan_monolithic(task, domain, tools, context)


def _nid(prefix: str = "") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}" if prefix else uuid.uuid4().hex[:12]
