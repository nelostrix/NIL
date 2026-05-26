from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.kernel.graph.graph_types import (
    AgentSpec,
    DependencyEdge,
    ExecutionGraph,
    ExecutionNode,
    GraphMetadata,
    NodeType,
)
from nil.kernel.graph.planner import PlanStep

logger = logging.getLogger("nil.kernel.graph.templates")

BUILTIN_TEMPLATES: dict[str, list[PlanStep]] = {
    "investigate_and_fix": [
        PlanStep(
            label="gather_data",
            description="Gather initial data using investigation tools",
            node_type=NodeType.TOOL_CALL,
        ),
        PlanStep(
            label="analyze",
            description="Analyze gathered data to determine root cause",
            node_type=NodeType.LLM_CALL,
            depends_on=["gather_data"],
        ),
        PlanStep(
            label="apply_fix",
            description="Apply the identified fix",
            node_type=NodeType.LLM_CALL,
            depends_on=["analyze"],
        ),
        PlanStep(
            label="verify",
            description="Verify the fix resolved the issue",
            node_type=NodeType.TOOL_CALL,
            depends_on=["apply_fix"],
        ),
    ],
    "root_cause_analysis": [
        PlanStep(
            label="hypothesis_1",
            description="Investigate hypothesis 1",
            node_type=NodeType.LLM_CALL,
        ),
        PlanStep(
            label="hypothesis_2",
            description="Investigate hypothesis 2",
            node_type=NodeType.LLM_CALL,
        ),
        PlanStep(
            label="hypothesis_3",
            description="Investigate hypothesis 3",
            node_type=NodeType.LLM_CALL,
        ),
        PlanStep(
            label="correlate",
            description="Correlate all hypothesis results",
            node_type=NodeType.MERGE,
            depends_on=["hypothesis_1", "hypothesis_2", "hypothesis_3"],
        ),
        PlanStep(
            label="conclusion",
            description="Determine root cause from evidence",
            node_type=NodeType.LLM_CALL,
            depends_on=["correlate"],
        ),
    ],
    "safe_deploy": [
        PlanStep(
            label="pre_check",
            description="Run pre-deployment checks",
            node_type=NodeType.TOOL_CALL,
        ),
        PlanStep(
            label="deploy",
            description="Execute deployment",
            node_type=NodeType.LLM_CALL,
            depends_on=["pre_check"],
        ),
        PlanStep(
            label="verify",
            description="Verify deployment succeeded",
            node_type=NodeType.TOOL_CALL,
            depends_on=["deploy"],
        ),
        PlanStep(
            label="rollback",
            description="Rollback if deployment fails",
            node_type=NodeType.LLM_CALL,
            depends_on=["deploy"],
        ),
    ],
    "incident_triage": [
        PlanStep(
            label="check_logs",
            description="Search application logs for errors",
            node_type=NodeType.TOOL_CALL,
        ),
        PlanStep(
            label="check_metrics",
            description="Check system metrics",
            node_type=NodeType.TOOL_CALL,
        ),
        PlanStep(
            label="check_alerts",
            description="Review active alerts",
            node_type=NodeType.TOOL_CALL,
        ),
        PlanStep(
            label="assess_severity",
            description="Assess incident severity from all data",
            node_type=NodeType.LLM_CALL,
            depends_on=["check_logs", "check_metrics", "check_alerts"],
        ),
        PlanStep(
            label="route",
            description="Route to appropriate response team",
            node_type=NodeType.TOOL_CALL,
            depends_on=["assess_severity"],
        ),
    ],
    "data_migration": [
        PlanStep(
            label="backup",
            description="Create backup before migration",
            node_type=NodeType.TOOL_CALL,
        ),
        PlanStep(
            label="migrate",
            description="Execute data migration",
            node_type=NodeType.LLM_CALL,
            depends_on=["backup"],
        ),
        PlanStep(
            label="validate",
            description="Validate migrated data",
            node_type=NodeType.TOOL_CALL,
            depends_on=["migrate"],
        ),
        PlanStep(
            label="verify_backup",
            description="Verify backup integrity",
            node_type=NodeType.TOOL_CALL,
            depends_on=["migrate"],
        ),
    ],
}


@dataclass
class GraphTemplate:
    template_id: str = ""
    name: str = ""
    domain: str = ""
    description: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0
    avg_cost: float = 0.0
    avg_duration_ms: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0

    def instantiate(
        self, task: str, domain: str, tools: list[str], context: dict = None
    ) -> ExecutionGraph:
        context = context or {}
        graph = ExecutionGraph(metadata=GraphMetadata(
            domain=domain, task=task, execution_plan=f"template:{self.name}",
        ))
        input_node = ExecutionNode(
            node_id=_tid("input"), node_type=NodeType.INPUT, task=task, label="Input",
        )
        graph.add_node(input_node)
        created: dict[str, str] = {"Input": input_node.node_id}

        for step in self.steps:
            nid = _tid(step.label)
            agent = step.agent_spec or AgentSpec(tools=tools)
            node = ExecutionNode(
                node_id=nid,
                node_type=step.node_type,
                label=step.label,
                description=step.description,
                task=step.task or task,
                tool_name=step.tool_name,
                tool_args=step.tool_args,
                agent_spec=agent,
                output_schema=step.output_schema,
            )
            graph.add_node(node)
            created[step.label] = nid

            if step.depends_on:
                for dep in step.depends_on:
                    if dep in created:
                        graph.add_edge(DependencyEdge(source=created[dep], target=nid))
            else:
                graph.add_edge(DependencyEdge(
                    source=created.get("Input", list(created.values())[0]), target=nid,
                ))

        output_node = ExecutionNode(
            node_id=_tid("output"), node_type=NodeType.OUTPUT, label="Output",
        )
        graph.add_node(output_node)
        last_label = self.steps[-1].label if self.steps else "Input"
        if last_label in created:
            graph.add_edge(DependencyEdge(
                source=created[last_label], target=output_node.node_id,
            ))
        return graph

    def record_usage(self, success: bool, cost: float, duration_ms: float) -> None:
        self.usage_count += 1
        if success:
            self.success_count += 1
        old_total = self.avg_cost * (self.usage_count - 1)
        self.avg_cost = (old_total + cost) / max(self.usage_count, 1)
        old_dur = self.avg_duration_ms * (self.usage_count - 1)
        self.avg_duration_ms = (old_dur + duration_ms) / max(self.usage_count, 1)

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "avg_cost": self.avg_cost,
            "avg_duration_ms": self.avg_duration_ms,
            "success_rate": self.success_rate,
            "created_at": self.created_at,
        }


class TemplateLibrary:
    def __init__(self):
        self._templates: dict[str, GraphTemplate] = {}
        self._load_builtins()

    def _load_builtins(self) -> None:
        for name, steps in BUILTIN_TEMPLATES.items():
            template = GraphTemplate(
                template_id=str(uuid.uuid4()),
                name=name,
                description=_DESCRIPTIONS.get(name, ""),
                steps=steps,
            )
            self._templates[name] = template

    def get(self, name: str) -> Optional[GraphTemplate]:
        return self._templates.get(name)

    def register(self, template: GraphTemplate) -> None:
        self._templates[template.name] = template

    def find_for_task(self, task: str, domain: str = "") -> list[GraphTemplate]:
        task_lower = task.lower()
        scored = []
        for t in self._templates.values():
            score = 0.0
            if t.domain and t.domain == domain:
                score += 0.3
            keywords = {
                "investigate_and_fix": ["fix", "error", "bug", "issue", "broken", "fail"],
                "root_cause_analysis": ["why", "root cause", "analyze", "investigate", "diagnose"],
                "safe_deploy": ["deploy", "release", "rollout", "publish"],
                "incident_triage": ["incident", "alert", "pager", "emergency", "critical"],
                "data_migration": ["migrate", "migration", "move data", "transfer"],
            }.get(t.name, [])
            matches = sum(1 for kw in keywords if kw in task_lower)
            score += matches * 0.15
            if score > 0:
                scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:5]]

    def list_templates(self) -> list[GraphTemplate]:
        return list(self._templates.values())

    def to_dict(self) -> dict:
        return {
            name: t.to_dict() for name, t in self._templates.items()
        }


_DESCRIPTIONS: dict[str, str] = {
    "investigate_and_fix": "Investigate an issue and apply a fix",
    "root_cause_analysis": "Parallel hypothesis testing to find root cause",
    "safe_deploy": "Deploy with pre-checks and rollback capability",
    "incident_triage": "Triage a production incident with parallel data gathering",
    "data_migration": "Safely migrate data with backup and validation",
}


def _tid(prefix: str = "") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}" if prefix else uuid.uuid4().hex[:12]
