# Multi-Agent Coordination Layer

> **Status: Implemented** — The full multi-agent architecture exists in `nil/agents/`. `OrchestratorAgent`, `Blackboard`, `ConsensusMechanism` (majority vote, structured debate, confidence-weighted consensus), `MarketCoordinator` (bidding-based task allocation), `AgentPool` (persistent/on-demand/pooled lifecycle), `AgentTeam` (hierarchical team nesting), `MessageBus` (pub/sub, request/reply), and `MultiAgentNil` (unified entry point) are all implemented.

---

## Background

Nil originally operated a single agent per domain. One agent executed the entire task — planning, tool selection, execution, verification — through one LLM loop. This hit fundamental limits: a single agent cannot scale its cognition across parallel workstreams, cannot specialize, cannot debate, cannot self-correct through independent verification.

The multi-agent architecture enables a *society of agents* to collaborate on tasks through delegation, negotiation, consensus, and specialized roles.

---

## Before Multi-Agent

```
Single Agent Architecture:

  Agent.run(task):
    1. LLM thinks (decides next action)
    2. LLM calls tool
    3. LLM evaluates result
    4. Repeat until done

  Limitations:
    • No parallelism (one thought stream at a time)
    • No specialization (same LLM does everything)
    • No verification gap (LLM evaluates its own work)
    • No delegation (one agent does all subtasks)
    • No debate (no alternative viewpoints)
    • No consensus (no cross-checking)
    • Blackboard: ∅ (no shared reasoning space)
```

---

## Architecture

```
                         ┌─────────────────────┐
                         │   ORCHESTRATOR      │
                         │   (Planner Agent)   │
                         │                     │
                         │  Decomposes task,   │
                         │  assigns work,      │
                         │  manages lifecycle  │
                         └──────┬──────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         v                      v                      v
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ SPECIALIST   │     │ SPECIALIST   │     │ SPECIALIST   │
  │ AGENT A      │     │ AGENT B      │     │ AGENT C      │
  │ (Investigate)│     │ (Analyze)    │     │ (Execute)    │
  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
         │                     │                    │
         └─────────────────────┴────────────────────┘
                               │
                               v
                      ┌──────────────────┐
                      │   VERIFIER       │
                      │   (Critic Agent) │
                      │                  │
                      │  Cross-checks    │
                      │  all outputs     │
                      └────────┬─────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    v                     v
            ┌──────────────┐    ┌──────────────────┐
            │ CONSENSUS    │    │ NEGOTIATOR       │
            │ (Agree?      │    │ (Resolve         │
            │  Vote)       │    │  Conflicts)      │
            └──────────────┘    └──────────────────┘
                               │
                               v
                      ┌──────────────────┐
                      │    BLACKBOARD    │
                      │   (Shared State) │
                      └──────────────────┘
```

---

## 1. Agent Roles

Each agent in the system has a specialized role:

```python
class AgentRole(Enum):
    ORCHESTRATOR   = "orchestrator"    # Plans, decomposes, assigns
    INVESTIGATOR   = "investigator"    # Gathers data, explores
    ANALYST        = "analyst"         # Reasons about data, diagnoses
    EXECUTOR       = "executor"        # Carries out actions
    VERIFIER       = "verifier"        # Validates outputs
    CRITIC         = "critic"          # Finds flaws, suggests improvements
    NEGOTIATOR     = "negotiator"      # Resolves inter-agent conflicts
    MONITOR        = "monitor"         # Watches for drift, anomalies
    MEMORY_MANAGER = "memory_manager"  # Consolidates, forgets, indexes
```

### Agent Specification

```python
@dataclass
class AgentSpec:
    role: AgentRole
    name: str
    model: str                           # Can differ per agent
    system_prompt: str                   # Role-specific prompt
    tools: list[str]                     # Subset of tools this agent can use
    temperature: float = 0.0
    max_tokens: int = 2048
    max_concurrent_tasks: int = 1
    cost_budget_per_run: float = 0.10
    timeout_seconds: int = 60
    delegation_allowed: bool = False     # Can this agent spawn sub-agents?
```

### Example Agent Configuration

```python
agents = {
    "orchestrator": AgentSpec(
        role=AgentRole.ORCHESTRATOR,
        model="claude-sonnet-4-20250514",
        system_prompt="You are an orchestrator. Decompose tasks, assign them to specialists, and synthesize results.",
        tools=[],                 # No direct tool access — assigns to others
        temperature=0.2,
        delegation_allowed=True,
    ),
    "investigator": AgentSpec(
        role=AgentRole.INVESTIGATOR,
        model="claude-haiku-3-20250301",   # Cheap, fast
        system_prompt="You are an investigator. Use search, query, and data-gathering tools to collect information.",
        tools=["search_logs", "query_metrics", "check_db", "read_file"],
        temperature=0.3,                   # Slight randomness for exploration
    ),
    "analyst": AgentSpec(
        role=AgentRole.ANALYST,
        model="claude-sonnet-4-20250514",  # Strong reasoning
        system_prompt="You are an analyst. Examine collected data, identify root causes, and produce diagnoses.",
        tools=["analyze_trace", "query_db"],
        temperature=0.0,
    ),
    "executor": AgentSpec(
        role=AgentRole.EXECUTOR,
        model="claude-haiku-3-20250301",   # Cheap for routine actions
        system_prompt="You are an executor. Carry out remediation actions precisely as specified.",
        tools=["restart_service", "deploy", "scale_up", "drain_connections"],
        temperature=0.0,                   # Deterministic
    ),
    "verifier": AgentSpec(
        role=AgentRole.VERIFIER,
        model="claude-opus-4-20250514",    # Best model for catching errors
        system_prompt="You are a verifier. Check all work for correctness, completeness, and safety before sign-off.",
        tools=["check_endpoint", "run_test", "validate_config"],
        temperature=0.0,
    ),
}
```

---

## 2. Orchestration

The orchestrator manages the lifecycle of a task across multiple agents:

```python
class OrchestratorAgent:
    """
    Plans and manages multi-agent task execution.

    For each task:
      1. Decompose into subtasks
      2. Assign subtasks to specialist agents
      3. Manage dependencies between subtasks
      4. Handle agent failures (retry, reassign)
      5. Synthesize results
    """

    def run(self, task: str, context: dict) -> MultiAgentOutcome:
        # Phase 1: Decompose
        plan = self._decompose(task)

        # Phase 2: Assign and execute
        results = {}
        for phase in plan.phases:
            # Parallel execution within a phase
            phase_results = self._execute_phase(phase)
            results.update(phase_results)

            # Verification gate
            if not self._verify_phase(phase, phase_results):
                # Trigger negotiation/recovery
                phase_results = self._recover_phase(phase, phase_results)

        # Phase 3: Synthesize
        final = self._synthesize(results)

        return MultiAgentOutcome(
            task=task,
            final_output=final,
            contributions=results,
            agent_timeline=self._timeline,
            total_cost=sum(r.cost for r in results.values()),
        )

    def _decompose(self, task: str) -> ExecutionPlan:
        """
        Decompose task into a DAG of subtasks, each assigned to a role.

        Returns a plan with phases (parallel groups) and dependencies.
        """
        # LLM call: analyze task, identify subtasks, assign roles
        plan = self.llm.chat(
            system=self._decomposition_prompt,
            messages=[{"role": "user", "content": task}],
        )
        return self._parse_plan(plan)

    def _execute_phase(self, phase: Phase) -> dict[str, AgentResult]:
        """
        Execute all subtasks in a phase in parallel.

        Each subtask is run by its assigned agent role.
        Results are collected and written to the blackboard.
        """
        futures = []
        for subtask in phase.subtasks:
            agent = self.agent_pool.get(subtask.assigned_role)
            future = self.executor.submit(agent.run, subtask, self.blackboard)
            futures.append((subtask.id, future))

        results = {}
        for subtask_id, future in futures:
            try:
                result = future.result(timeout=subtask.timeout)
                results[subtask_id] = result
                self.blackboard.write(subtask_id, result)
            except TimeoutError:
                results[subtask_id] = AgentResult.failure(
                    subtask_id, "timeout"
                )

        return results
```

### Task Decomposition Example

```
Task: "Resolve production incident #4421 — API latency spike on /v2/users"

Plan:
  Phase 1 (parallel):
    [Investigator A] Search logs for error patterns in /v2/users
    [Investigator B] Query metrics dashboard for latency/cpu/memory
    [Investigator C] Check recent deploy history

  Phase 2 (depends on Phase 1):
    [Analyst] Correlate findings → root cause diagnosis

  Phase 3 (depends on Phase 2, parallel):
    [Executor A] Apply remediation
    [Executor B] Prepare rollback as contingency

  Phase 4 (depends on Phase 3):
    [Verifier] Confirm fix, validate no regression

  Phase 5:
    [Orchestrator] Synthesize final report
```

---

## 3. Blackboard Memory

The blackboard is a shared memory space all agents read from and write to. It replaces the need for agents to pass messages directly.

```python
class Blackboard:
    """
    Shared memory for multi-agent coordination.

    All agents read from and write to the blackboard.
    This replaces direct agent-to-agent messaging.
    """

    def __init__(self):
        self.store: dict[str, BlackboardEntry] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    def write(self, key: str, value: Any, metadata: dict = None):
        """Write to the blackboard. Overwrites existing."""
        self.store[key] = BlackboardEntry(
            key=key,
            value=value,
            metadata=metadata or {},
            timestamp=time.time(),
            writer=current_agent_name(),
        )

    def append(self, key: str, value: Any):
        """Append to a list entry (for accumulating results)."""
        if key not in self.store:
            self.store[key] = BlackboardEntry(
                key=key, value=[], metadata={}, timestamp=time.time()
            )
        self.store[key].value.append(value)
        self.store[key].timestamp = time.time()

    def read(self, key: str, default=None) -> Any:
        """Read from the blackboard."""
        entry = self.store.get(key)
        return entry.value if entry else default

    def search(self, query: str) -> list[BlackboardEntry]:
        """Search blackboard entries by key pattern or metadata."""
        return [
            entry for entry in self.store.values()
            if query.lower() in entry.key.lower()
            or any(query.lower() in str(v).lower() for v in entry.metadata.values())
        ]

    def namespace(self, prefix: str) -> dict[str, BlackboardEntry]:
        """Get all entries with a given prefix."""
        return {
            k: v for k, v in self.store.items() if k.startswith(prefix)
        }

    def subscribe(self, pattern: str, callback: Callable):
        """Subscribe to changes matching a pattern."""
        # Enables agents to react to new information
        ...


# Blackboard layout for a typical incident:
blackboard_sections = {
    "task": {
        "description": "API latency spike on /v2/users",
        "status": "investigating",
        "priority": "high",
    },
    "investigation": {
        "logs_summary": "...",             # Written by Investigator A
        "metrics_summary": "...",          # Written by Investigator B
        "deploy_history": "...",           # Written by Investigator C
    },
    "analysis": {
        "root_cause": "connection_pool_exhaustion",  # Written by Analyst
        "confidence": 0.92,
        "evidence": [...],
    },
    "action": {
        "plan": "drain → restart → verify",
        "execution_status": "pending",
        "rollback_ready": True,
    },
    "verification": {
        "result": "pass",
        "checks_run": ["endpoint_200", "latency_p99", "error_rate"],
    },
}
```

---

## 4. Negotiation & Consensus

When agents disagree, the negotiator resolves conflicts:

```python
class ConsensusMechanism:
    """
    Methods for reaching agreement among multiple agents.
    """

    # ── Voting ──

    def majority_vote(
        self,
        proposals: list[AgentProposal],
        voters: list[AgentSpec],
    ) -> ConsensusResult:
        """
        Each voter agent evaluates proposals and votes.

        Used for: which root cause is correct, which action to take.

        Returns: the proposal with majority support, or "inconclusive"
        """
        votes = {p.id: 0 for p in proposals}
        for voter in voters:
            evaluation = voter.evaluate(proposals)
            votes[evaluation.chosen_id] += 1

        winner = max(votes, key=votes.get)
        total = sum(votes.values())

        return ConsensusResult(
            winner=winner,
            vote_share=votes[winner] / total,
            confidence=self._compute_confidence(votes, voters),
        )

    # ── Structured Debate ──

    def debate(
        self,
        thesis: str,
        antithesis: str,
        rounds: int = 3,
    ) -> DebateOutcome:
        """
        Two agents debate opposing positions.

        Each round:
          1. Proponent argues for thesis
          2. Opponent argues against
          3. Judge (orchestrator) scores arguments

        Used for: high-stakes decisions where consensus is critical.
        """
        for round_num in range(rounds):
            pro_argument = self.proponent.argue(thesis, antithesis, round_num)
            con_argument = self.opponent.argue(antithesis, thesis, round_num)
            score = self.judge.score(pro_argument, con_argument)
            self._log_debate_round(round_num, pro_argument, con_argument, score)

        return DebateOutcome(
            winner="proponent" if score > 0.5 else "opponent",
            confidence=abs(score - 0.5) * 2,
            transcript=self._debate_log,
        )

    # ── Confidence-Weighted Consensus ──

    def confidence_weighted_consensus(
        self,
        opinions: list[AgentOpinion],
    ) -> ConsensusResult:
        """
        Agents express opinions with confidence scores.

        Higher-confidence agents have more weight.

        Used for: continuous decisions where agents have
        varying expertise levels.
        """
        total_weight = sum(o.confidence for o in opinions)
        weighted_sum = sum(
            o.value * o.confidence for o in opinions
        )
        average = weighted_sum / total_weight

        # Bootstrap confidence interval
        variance = sum(
            o.confidence * (o.value - average) ** 2 for o in opinions
        ) / total_weight

        return ConsensusResult(
            value=average,
            confidence=1.0 / (1.0 + variance),
            n_agents=len(opinions),
        )
```

---

## 5. Agent Pool & Lifecycle

```python
class AgentPool:
    """
    Manages a pool of agent instances.

    Agents can be:
      - Persistent (always running, e.g., monitor)
      - On-demand (spawned per task, destroyed after)
      - Pooled (limited instances, recycled)
    """

    def __init__(self):
        self.persistent_agents: dict[str, Agent] = {}
        self.pooled_agents: dict[str, list[Agent]] = {}
        self.agent_specs: dict[str, AgentSpec] = {}

    def register_spec(self, spec: AgentSpec):
        self.agent_specs[spec.role.value] = spec

    def spawn(self, role: AgentRole, task_context: dict = None) -> Agent:
        """Spawn a new agent instance for a role."""
        spec = self.agent_specs[role.value]
        agent = Agent(
            domain=self.domain,
            base_model=spec.model,
            config=AgentConfig(
                system_prompt=spec.system_prompt,
                max_tokens=spec.max_tokens,
                temperature=spec.temperature,
            ),
        )
        agent.role = role
        agent.delegation_allowed = spec.delegation_allowed
        agent.tools = [self.tool_manager.get(t) for t in spec.tools]
        return agent

    def acquire(self, role: AgentRole) -> Agent:
        """Get an agent from the pool (or spawn if none available)."""
        pool = self.pooled_agents.get(role.value, [])
        if pool:
            return pool.pop()
        return self.spawn(role)

    def release(self, agent: Agent):
        """Return agent to pool."""
        role = agent.role.value
        if role not in self.pooled_agents:
            self.pooled_agents[role] = []
        self.pooled_agents[role].append(agent)
```

---

## 6. Hierarchical Agent Teams

Sub-agents can themselves have sub-agents, creating recursive team structures:

```python
class AgentTeam:
    """
    A team of agents working on a common goal.

    Teams can be nested — a team leader is an agent that
    manages sub-teams.
    """

    def __init__(
        self,
        leader: AgentSpec,
        members: dict[str, AgentSpec],
        coordination: str = "hierarchical",   # or "swarm", "democratic"
    ):
        self.leader = leader
        self.members = members
        self.coordination = coordination
        self.subteams: list[AgentTeam] = []
        self.blackboard = Blackboard()

    def add_subteam(self, team: 'AgentTeam'):
        """Nest a sub-team (hierarchical decomposition)."""
        self.subteams.append(team)
        team.parent = self

    def run(self, task: str) -> TeamResult:
        if self.coordination == "hierarchical":
            return self._run_hierarchical(task)
        elif self.coordination == "swarm":
            return self._run_swarm(task)
        elif self.coordination == "democratic":
            return self._run_democratic(task)


# Example hierarchy:
root_team = AgentTeam(
    leader=orchestrator_spec,
    members={
        "investigator": investigator_spec,
        "analyst": analyst_spec,
        "executor": executor_spec,
    },
)

# Incident response sub-team (nested)
incident_team = AgentTeam(
    leader=incident_commander_spec,
    members={
        "triage": triage_spec,
        "diagnostics": diagnostics_spec,
        "remediation": remediation_spec,
    },
)

root_team.add_subteam(incident_team)
```

---

## 7. Coordination Strategies

### Hierarchical (Top-Down)

```
Orchestrator assigns subtasks → specialists execute → verifier checks
Best for: well-understood domains with clear role boundaries
```

### Democratic (Consensus-Driven)

```
All agents are peers. Decisions made by vote.
Best for: ambiguous situations requiring diverse perspectives
```

### Swarm (Emergent)

```
Agents pick up tasks from a shared queue, no central coordinator.
Best for: large-scale parallel work (e.g., monitoring many services)
```

### Market-Based

```
Agents bid on subtasks based on capability + current load.
Orchestrator awards tasks to lowest-cost bidder.
Best for: cost-optimized execution across heterogeneous agents
```

```python
class MarketCoordinator:
    """
    Agents bid on subtasks. Lowest bid wins.

    Each agent bids based on:
      - Capability match (how well-suited they are)
      - Current load (how busy they are)
      - Past success rate on similar tasks
      - Cost (cheapest capable agent wins)
    """

    def assign(self, subtask: Subtask, agents: list[Agent]) -> Agent:
        bids = []
        for agent in agents:
            bid = agent.compute_bid(subtask)
            bids.append((agent, bid))

        # Lowest combined cost wins
        bids.sort(key=lambda x: x[1].total_cost)
        winner = bids[0][0]

        self.blackboard.write(f"bid_{subtask.id}", {
            "winner": winner.name,
            "bid": bids[0][1].total_cost,
            "all_bids": [(a.name, b.total_cost) for a, b in bids],
        })

        return winner
```

---

## 8. Communication Protocol

```python
@dataclass
class AgentMessage:
    """Message between agents (routed via blackboard or direct)."""

    source: str               # agent name
    target: str               # agent name, or "broadcast"
    message_type: MessageType # REQUEST | RESPONSE | INFORM | DELEGATE | VOTE
    content: Any
    thread_id: str            # conversation thread
    priority: int = 0
    deadline: float = None    # response deadline
    correlation_id: str = None # for matching requests to responses


class MessageBus:
    """
    Async message bus for inter-agent communication.

    Supports:
      - Point-to-point (agent → agent)
      - Broadcast (agent → all)
      - Publish/subscribe (agent → topic → subscribers)
      - Request/reply (with correlation)
    """

    def __init__(self):
        self.queues: dict[str, asyncio.Queue] = {}
        self.topics: dict[str, list[Callable]] = {}

    async def send(self, message: AgentMessage):
        """Send message to a specific agent."""
        if message.target == "broadcast":
            await self._broadcast(message)
        else:
            queue = self.queues.get(message.target)
            if queue:
                await queue.put(message)

    async def request(self, target: str, content: Any, timeout: float = 30) -> Any:
        """Send a request and wait for response."""
        correlation_id = str(uuid.uuid4())
        message = AgentMessage(
            source=self.agent_name,
            target=target,
            message_type=MessageType.REQUEST,
            content=content,
            correlation_id=correlation_id,
        )
        response_queue = asyncio.Queue()
        self._pending_requests[correlation_id] = response_queue
        await self.send(message)

        try:
            response = await asyncio.wait_for(
                response_queue.get(), timeout=timeout
            )
            return response.content
        except asyncio.TimeoutError:
            return AgentResponse.error(f"Timeout waiting for {target}")
```

---

## 9. Integration with Nil

```python
class MultiAgentNil:
    """
    Multi-agent version of the Nil framework.

    Usage:
        society = MultiAgentNil(domain=domain)
        society.register_agent_spec("investigator", investigator_spec)
        society.register_agent_spec("analyst", analyst_spec)
        society.register_agent_spec("executor", executor_spec)
        society.register_agent_spec("verifier", verifier_spec)

        result = society.run("Resolve API latency spike")
    """

    def __init__(self, domain: Domain):
        self.domain = domain
        self.agent_pool = AgentPool()
        self.blackboard = Blackboard()
        self.message_bus = MessageBus()
        self.orchestrator = OrchestratorAgent(
            domain=domain,
            agent_pool=self.agent_pool,
            blackboard=self.blackboard,
        )

    def register_agent_spec(self, spec: AgentSpec):
        self.agent_pool.register_spec(spec)

    def run(self, task: str) -> MultiAgentOutcome:
        return self.orchestrator.run(task, self.blackboard)
```

---

## Implementation Status

| Phase | What | Status |
|---|---|---|
| 1 | Agent role definitions | **Done** — `AgentRole` enum (9 roles), `AgentSpec` dataclass, per-role prompts and tool subsets |
| 2 | Blackboard | **Done** — shared key-value store with namespacing, search, subscription |
| 3 | Orchestrator | **Done** — task decomposition → phase execution → result synthesis |
| 4 | Parallel agent execution | **Done** — async agent pool, concurrent phase execution, timeout handling |
| 5 | Verifier agent | **Done** — cross-check outputs, independent validation gate |
| 6 | Message bus | **Done** — async pub/sub, request/reply, broadcast |
| 7 | Voting & consensus | **Done** — majority vote, confidence-weighted consensus |
| 8 | Debate mechanism | **Done** — structured multi-round debate with judge scoring |
| 9 | Market-based assignment | **Done** — agents bid on subtasks, lowest cost wins |
| 10 | Hierarchical teams | **Done** — nested agent teams, recursive decomposition |

---

## Key Differentiator

**Single agent:**
One mind, one perspective, one bottleneck. The agent cannot challenge its own assumptions, cannot parallelize, cannot specialize. It does everything — and therefore nothing optimally.

**Multi-agent society:**
A system of specialized minds. Investigators explore while analysts reason. Executors act while verifiers check. The orchestrator sees the whole picture while specialists dive deep. When they disagree, debate surfaces the best argument. When they agree, confidence is high.
