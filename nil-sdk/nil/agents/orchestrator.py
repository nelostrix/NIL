from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.agents.blackboard import Blackboard
from nil.agents.pool import AgentPool, _PooledAgent
from nil.agents.roles import AgentResult, AgentRole, MultiAgentOutcome


@dataclass
class Subtask:
    id: str = ""
    description: str = ""
    assigned_role: AgentRole = AgentRole.INVESTIGATOR
    depends_on: list[str] = field(default_factory=list)
    timeout: int = 60
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:8]


@dataclass
class Phase:
    id: str = ""
    subtasks: list[Subtask] = field(default_factory=list)
    parallel: bool = True

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:8]


@dataclass
class ExecutionPlan:
    phases: list[Phase] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_phase(self, phase: Phase) -> None:
        self.phases.append(phase)


class OrchestratorAgent:
    def __init__(
        self,
        domain: str = "",
        agent_pool: Optional[AgentPool] = None,
        blackboard: Optional[Blackboard] = None,
        name: str = "orchestrator",
    ):
        self.domain = domain
        self.agent_pool = agent_pool or AgentPool()
        self.blackboard = blackboard or Blackboard()
        self.name = name
        self._timeline: list[dict] = []
        self._phase_index = 0

    def run(self, task: str, context: dict = None) -> MultiAgentOutcome:
        context = context or {}
        start = time.time()
        self._timeline = []
        self._phase_index = 0

        plan = self._decompose(task)
        self.blackboard.write("task.description", task, writer=self.name)
        self.blackboard.write("task.status", "running", writer=self.name)
        self._log("decompose", f"plan has {len(plan.phases)} phases")

        results: dict[str, AgentResult] = {}
        for phase in plan.phases:
            phase_results = self._execute_phase(phase)
            results.update(phase_results)

            if not self._verify_phase(phase, phase_results):
                phase_results = self._recover_phase(phase, phase_results)
                results.update(phase_results)

        final = self._synthesize(results)
        duration = time.time() - start

        self.blackboard.write("task.status", "completed", writer=self.name)

        return MultiAgentOutcome(
            task=task,
            final_output=final,
            contributions=results,
            agent_timeline=list(self._timeline),
            total_cost=sum(r.cost for r in results.values()),
            total_duration=duration,
            success=all(r.success for r in results.values()),
        )

    def _decompose(self, task: str) -> ExecutionPlan:
        plan = ExecutionPlan()
        investigation = Phase(id="phase_1", subtasks=[
            Subtask(description=f"Gather data about: {task[:100]}", assigned_role=AgentRole.INVESTIGATOR),
        ])
        plan.add_phase(investigation)

        analysis = Phase(id="phase_2", subtasks=[
            Subtask(description=f"Analyze findings for: {task[:100]}", assigned_role=AgentRole.ANALYST, depends_on=["phase_1"]),
        ])
        plan.add_phase(analysis)

        execution = Phase(id="phase_3", subtasks=[
            Subtask(description=f"Execute remediation for: {task[:100]}", assigned_role=AgentRole.EXECUTOR, depends_on=["phase_2"]),
        ])
        plan.add_phase(execution)

        verification = Phase(id="phase_4", subtasks=[
            Subtask(description=f"Verify results for: {task[:100]}", assigned_role=AgentRole.VERIFIER, depends_on=["phase_3"]),
        ])
        plan.add_phase(verification)
        return plan

    def _execute_phase(self, phase: Phase) -> dict[str, AgentResult]:
        results: dict[str, AgentResult] = {}
        for subtask in phase.subtasks:
            agent = self.agent_pool.acquire(subtask.assigned_role)
            result = agent.run(subtask, self.blackboard)
            self.blackboard.write(
                f"subtask.{subtask.id}",
                {"result": result.output, "success": result.success},
                writer=agent.name,
            )
            self._log("execute", f"{agent.name}: {subtask.description[:60]}")
            self.agent_pool.release(agent)
            results[subtask.id] = result
        return results

    def _verify_phase(self, phase: Phase, results: dict[str, AgentResult]) -> bool:
        return all(r.success for r in results.values())

    def _recover_phase(self, phase: Phase, results: dict[str, AgentResult]) -> dict[str, AgentResult]:
        recovered: dict[str, AgentResult] = {}
        for subtask in phase.subtasks:
            if subtask.id in results and not results[subtask.id].success:
                agent = self.agent_pool.acquire(subtask.assigned_role)
                retry_result = agent.run(subtask, self.blackboard)
                self._log("recover", f"retry {subtask.description[:40]}: {retry_result.success}")
                self.agent_pool.release(agent)
                recovered[f"{subtask.id}_retry"] = retry_result
        return recovered

    def _synthesize(self, results: dict[str, AgentResult]) -> str:
        successes = sum(1 for r in results.values() if r.success)
        total = len(results)
        return f"Completed {successes}/{total} subtasks successfully."

    def _log(self, event: str, detail: str) -> None:
        self._timeline.append({
            "event": event,
            "detail": detail,
            "timestamp": time.time(),
            "phase": self._phase_index,
        })
