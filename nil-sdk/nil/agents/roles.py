from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


class AgentRole(enum.Enum):
    ORCHESTRATOR = "orchestrator"
    INVESTIGATOR = "investigator"
    ANALYST = "analyst"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    CRITIC = "critic"
    NEGOTIATOR = "negotiator"
    MONITOR = "monitor"
    MEMORY_MANAGER = "memory_manager"


_DEFAULT_PROMPTS = {
    AgentRole.ORCHESTRATOR: (
        "You are an orchestrator. Decompose tasks, assign them to specialists, "
        "manage dependencies, handle failures, and synthesize results."
    ),
    AgentRole.INVESTIGATOR: (
        "You are an investigator. Use search, query, and data-gathering tools "
        "to collect information efficiently."
    ),
    AgentRole.ANALYST: (
        "You are an analyst. Examine collected data, identify root causes, "
        "and produce thorough diagnoses."
    ),
    AgentRole.EXECUTOR: (
        "You are an executor. Carry out remediation actions precisely as "
        "specified with deterministic accuracy."
    ),
    AgentRole.VERIFIER: (
        "You are a verifier. Check all work for correctness, completeness, "
        "and safety before sign-off."
    ),
    AgentRole.CRITIC: (
        "You are a critic. Find flaws in plans and outputs, suggest "
        "improvements, and challenge assumptions."
    ),
    AgentRole.NEGOTIATOR: (
        "You are a negotiator. Resolve disagreements between agents by "
        "finding common ground and compromise solutions."
    ),
    AgentRole.MONITOR: (
        "You are a monitor. Watch for drift, anomalies, and performance "
        "degradation across the system."
    ),
    AgentRole.MEMORY_MANAGER: (
        "You are a memory manager. Consolidate, summarize, index, and "
        "prune information for long-term retention."
    ),
}

_DEFAULT_TOOLS: dict[AgentRole, list[str]] = {
    AgentRole.INVESTIGATOR: [
        "search_logs", "query_metrics", "check_db", "read_file",
    ],
    AgentRole.ANALYST: [
        "analyze_trace", "query_db",
    ],
    AgentRole.EXECUTOR: [
        "restart_service", "deploy", "scale_up", "drain_connections",
    ],
    AgentRole.VERIFIER: [
        "check_endpoint", "run_test", "validate_config",
    ],
}


@dataclass
class AgentSpec:
    role: AgentRole
    name: str = ""
    model: str = "claude-sonnet-4-20250514"
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 2048
    max_concurrent_tasks: int = 1
    cost_budget_per_run: float = 0.10
    timeout_seconds: int = 60
    delegation_allowed: bool = False

    def __post_init__(self):
        if not self.name:
            self.name = self.role.value
        if not self.system_prompt:
            self.system_prompt = _DEFAULT_PROMPTS.get(self.role, "")
        if not self.tools:
            self.tools = list(_DEFAULT_TOOLS.get(self.role, []))


@dataclass
class AgentResult:
    subtask_id: str = ""
    success: bool = False
    output: str = ""
    cost: float = 0.0
    duration: float = 0.0
    error: str = ""
    metadata: dict = field(default_factory=dict)

    @classmethod
    def failure(cls, subtask_id: str, error: str) -> AgentResult:
        return cls(subtask_id=subtask_id, success=False, error=error)


@dataclass
class MultiAgentOutcome:
    task: str = ""
    final_output: str = ""
    contributions: dict[str, AgentResult] = field(default_factory=dict)
    agent_timeline: list[dict] = field(default_factory=list)
    total_cost: float = 0.0
    total_duration: float = 0.0
    success: bool = False
