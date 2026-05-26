# Graph Execution Engine

**Beyond Linear Agent Loops**

The current Nil kernel runs a single-threaded loop: LLM → tool → LLM → tool → ... → final answer. This is fundamentally sequential. Every step waits for the previous step. There is no parallelism, no planning, no delegation, no speculative execution.

This document proposes replacing the linear loop with a DAG (Directed Acyclic Graph) execution engine where tasks are decomposed into a graph of nodes, executed with dependency-aware parallelism, and orchestrated by a planner/critic architecture.

---

## Current State

```
Linear Agent Loop:

  ┌─────────┐     ┌─────────┐     ┌─────────┐          ┌─────────┐
  │  LLM    │────→│  Tool   │────→│  LLM    │──...──→  │  Final  │
  │  think  │     │ execute │     │  think  │          │  Answer │
  └─────────┘     └─────────┘     └─────────┘          └─────────┘

  Limitations:
    • One thing at a time
    • No planning horizon (LLM decides next step with only current context)
    • No speculative execution (can't explore branches in parallel)
    • No specialization (same LLM does planning, execution, verification)
    • No sub-delegation (all tools called from one context)
    • No rollback (if a branch fails, entire run fails)
    • No structured error handling (failure = stop or retry, no alternative paths)
```

---

## Proposed Architecture

```
                     ┌─────────────┐
                     │   PLANNER   │
                     │  (decompose │
                     │   task)     │
                     └──────┬──────┘
                            │
                    DAG Specification
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              v             v             v
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Worker A │  │ Worker B │  │ Worker C │  ← Parallel branches
        │ (investi-│  │ (check   │  │ (prepare │
        │  gate)   │  │  alerts) │  │  fix)    │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │             │             │
             └─────────────┼─────────────┘
                           │
                           v
                    ┌──────────────┐
                    │   MERGE      │
                    │  (aggregate  │
                    │   results)   │
                    └──────┬───────┘
                           │
                           v
                    ┌──────────────┐
                    │   CRITIC     │
                    │  (validate,  │
                    │   critique)  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              v            v            v
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Pass     │  │ Needs    │  │ Fail     │
        │ → Output │  │ Revision │  │ → Fallback│
        └──────────┘  └────┬─────┘  └────┬─────┘
                           │             │
                           v             v
                    ┌──────────┐   ┌──────────┐
                    │  Revise  │   │ Rollback │
                    │  & retry │   │ & retry  │
                    └──────────┘   └──────────┘
```

---

## Core Components

### 1. Planner — Task Decomposition

The Planner replaces the LLM's first "think" step with a structured decomposition phase. Instead of producing one next action, it produces a full or partial execution graph.

```python
class Planner:
    def plan(self, task: str, domain: Domain, context: Context) -> ExecutionGraph:
        """
        Decompose a task into an execution DAG.

        Returns:
            ExecutionGraph: nodes (work units) + edges (dependencies)
        """
```

**Planning strategies:**

| Strategy | When | Output |
|---|---|---|
| **Monolithic** | Simple tasks | Single node, no decomposition |
| **Sequential** | Linear dependency chain | Ordered nodes |
| **Parallel fan-out** | Independent sub-tasks | Fan-out → merge |
| **Explore-exploit** | Investigation + fix | Parallel investigation → merge → sequential fix |
| **Hypothesis testing** | Unknown root cause | Multiple parallel investigation branches |
| **Contingency** | High-risk operations | Primary path + rollback path |

```
Example decomposition for "API latency spike on /v2/users":

  Task: Investigate and resolve API latency spike on /v2/users

  Plan:
    ┌──────────────────────────────────────────────────────┐
    │  STAGE 1: INVESTIGATE (parallel)                     │
    │                                                       │
    │  ┌─────────────────┐  ┌──────────────────────────┐    │
    │  │ Node: check_logs│  │ Node: check_metrics     │    │
    │  │ Tool: search    │  │ Tools: query_dashboard  │    │
    │  │ Description:    │  │ Description:            │    │
    │  │ Search for error│  │ Check CPU/memory/latency│    │
    │  │ logs in last 30m│  │ metrics last 30 minutes │    │
    │  └────────┬────────┘  └─────────────┬────────────┘    │
    │           │                         │                  │
    │           └─────────────────────────┘                  │
    │                           │                            │
    │                           v                            │
    │              ┌──────────────────────────┐              │
    │              │ Node: correlate          │              │
    │              │ LLM: analyze logs+metrics│              │
    │              │ to identify root cause   │              │
    │              └─────────────┬────────────┘              │
    │                            │                            │
    ├────────────────────────────┼────────────────────────────┤
    │  STAGE 2: ACT (sequential, with rollback)               │
    │                            │                            │
    │              ┌──────────────────────────┐              │
    │              │ Node: deploy_fix        │              │
    │              │ Tool: deploy, restart   │              │
    │              │ Description: Apply fix  │              │
    │              │ Rollback: Node: revert  │              │
    │              └─────────────┬────────────┘              │
    │                            │                            │
    │                            v                            │
    │              ┌──────────────────────────┐              │
    │              │ Node: verify             │              │
    │              │ Tool: check_endpoint     │              │
    │              │ Description: Confirm fix │              │
    │              └──────────────────────────┘              │
    └────────────────────────────────────────────────────────┘
```

### 2. Execution Graph — The DAG

```python
@dataclass
class ExecutionGraph:
    nodes: dict[str, ExecutionNode]
    edges: list[DependencyEdge]
    metadata: GraphMetadata

@dataclass
class ExecutionNode:
    node_id: str
    node_type: NodeType          # LLM_CALL | TOOL_CALL | SUBGRAPH | MERGE | CONDITIONAL
    agent_spec: AgentSpec        # which LLM, which provider, which config
    input_schema: dict           # what inputs this node expects
    output_schema: dict          # what outputs this node produces
    max_retries: int = 3
    timeout_ms: int = 30000
    fallback_node: Optional[str] = None   # on failure, route here
    rollback_node: Optional[str] = None   # on failure, undo actions here

@dataclass
class DependencyEdge:
    source: str
    target: str
    edge_type: EdgeType          # DATA_FLOW | CONTROL_FLOW | CONDITIONAL
    condition: Optional[str] = None   # "success", "failure", custom predicate
    data_map: Optional[dict] = None   # maps output fields → input fields
```

### 3. Workers — Specialized Executors

Workers are not the same LLM with the same prompt. Each worker can have its own model, config, tools, and system prompt:

```python
@dataclass
class AgentSpec:
    model: str                         # "claude-opus-4-20250514" or "claude-haiku-3-20250301"
    provider: str                      # "anthropic", "openai", "gemini"
    system_prompt: str                 # specialized for this worker's role
    tools: list[str]                   # subset of tools this worker can use
    max_tokens: int = 2048
    temperature: float = 0.0           # lower for execution, higher for exploration
    cost_budget: float = 0.05          # max cost for this worker
```

**Example worker roles in the same run:**

| Worker | Model | Temp | Tools | Role |
|---|---|---|---|---|
| Investigator | Claude Haiku | 0.3 | search_logs, query_metrics | Fast exploration, gather data |
| Diagnostic | Claude Sonnet | 0.1 | search_logs, query_db, analyze_trace | Root cause analysis |
| Executor | Claude Haiku | 0.0 | deploy, restart, scale | Carry out the fix |
| Verifier | Claude Sonnet | 0.0 | check_endpoint, run_test | Confirm fix worked |
| Critic | Claude Opus | 0.2 | (analysis only, no tools) | Review and validate output |

### 4. Critic — Validation & Error Handling

The Critic is a separate node (not the planner) that validates outputs before they're consumed downstream. It answers:

- Is this result complete and correct?
- Does it satisfy the preconditions of downstream nodes?
- Are there any contradictions with other branch results?
- Should we proceed, retry, or take an alternative path?

```python
class CriticNode(ExecutionNode):
    def evaluate(self, node_output: Any, context: Context) -> CritiqueVerdict:
        # verdict: PASS | NEEDS_REVISION | FAIL
        # if FAIL → activate fallback_node
        # if NEEDS_REVISION → retry with critique injected as feedback
```

```
Critique example:

  Verdict: NEEDS_REVISION
  Issues:
    - "The fix restarts the service but does not drain connections first.
      This will cause a brief second outage (see failure run #42).
      Add a drain step before restart."
  Severity: MEDIUM
  Suggested action: Revise deploy_fix node to include drain step
```

### 5. Subagent Spawning

Any node can spawn a subagent — a fully independent agent run with its own task, domain, and memory namespace. This enables hierarchical decomposition:

```python
@dataclass
class SubagentNode(ExecutionNode):
    node_type: NodeType = NodeType.SUBGRAPH
    sub_domain: str                    # domain for the sub-agent
    sub_task: str                      # task for the sub-agent
    sub_memory: str                    # memory namespace (separate from parent)
    inherit_context: bool = True       # pass parent context
    max_depth: int = 3                 # prevent infinite recursion
    timeout_ms: int = 60000
```

```
Example subagent hierarchy:

  Main Agent: "Resolve production incident #4421"
    │
    ├── Subagent: "Investigate API latency on /v2/users"
    │     │
    │     ├── Subagent: "Check database replica lag"
    │     │     └── Queries DB, reports status
    │     │
    │     └── Subagent: "Analyze recent deploy for regressions"
    │           └── Checks deploy history, git log
    │
    ├── Subagent: "Prepare rollback plan"
    │     └── Identifies safe rollback sequence
    │
    └── Subagent: "Draft post-mortem document"
          └── Captures timeline, root cause, action items
```

---

## Execution Engine

### Scheduler — Dependency-Aware Parallelism

```python
class GraphScheduler:
    """
    Executes a DAG with dependency tracking and parallelism.

    At each tick:
      1. Find all nodes whose dependencies are satisfied (ready set)
      2. Launch ready nodes in parallel (respecting concurrency limits)
      3. Monitor execution, collect results
      4. On node completion, update dependency graph
      5. On node failure, activate fallback/rollback if configured
      6. Repeat until all nodes complete or unrecoverable failure
    """

    max_concurrency: int = 5           # max parallel LLM/tool calls
    max_cost_per_run: float = 2.0      # total cost limit
    timeout_per_node: int = 30000      # ms

    def execute(self, graph: ExecutionGraph) -> GraphOutcome:
        ready = self._find_ready_nodes(graph)
        while ready or self._in_flight:
            # Launch up to max_concurrency
            batch = ready[:self.max_concurrency - len(self._in_flight)]
            for node in batch:
                future = self._launch_node(node)
                self._in_flight.append(future)

            # Wait for next completion
            done = self._wait_for_any(self._in_flight)
            self._in_flight.remove(done)
            result = done.result()

            # Handle outcome
            if result.success:
                graph.set_node_output(node.node_id, result.output)
                graph.mark_completed(node.node_id)
            else:
                if node.fallback_node:
                    graph.activate_fallback(node.node_id, node.fallback_node)
                elif node.rollback_node:
                    graph.activate_rollback(node.node_id, node.rollback_node)
                else:
                    return GraphOutcome.failure(node, result.error)

            # Recompute ready set
            ready = self._find_ready_nodes(graph)

        return GraphOutcome.success(graph.collect_outputs())
```

### Execution Flow Visualization

```
Time ─────────────────────────────────────────────────────►

  │  Planner: decompose task
  │     ↓
  │  ┌─────────────────────────────────────────────────────┐
  │  │ Node A (investigate)   ████████████████░░░░░░░░░░  │
  │  │ Node B (check alerts)  ██████████░░░░░░░░░░░░░░░░  │  ← parallel
  │  │ Node C (prep rollback) ██████████████████████░░░░  │
  │  └─────────────────────────────────────────────────────┘
  │                          ↓
  │  ┌─────────────────────────────────────────────────────┐
  │  │ Merge: correlate A+B+C  ██████████████░░░░░░░░░░░  │
  │  └─────────────────────────────────────────────────────┘
  │                          ↓
  │  ┌─────────────────────────────────────────────────────┐
  │  │ Critic: validate findings  ████████░░░░░░░░░░░░░░  │
  │  └─────────────────────────────────────────────────────┘
  │                          ↓ (PASS)
  │  ┌─────────────────────────────────────────────────────┐
  │  │ Node D: deploy fix  ██████████████████░░░░░░░░░░░  │
  │  └─────────────────────────────────────────────────────┘
  │                          ↓
  │  ┌─────────────────────────────────────────────────────┐
  │  │ Node E: verify        ████████░░░░░░░░░░░░░░░░░░░  │
  │  └─────────────────────────────────────────────────────┘
  │                          ↓
  │  Critic: final validation → Output
```

---

## Speculative Execution

When the planner generates multiple plausible hypotheses, speculative execution explores them in parallel rather than sequentially:

```
Without speculation (current):
  1. "Is it a DB issue?" → check DB → no
  2. "Is it a code issue?" → check code → no
  3. "Is it a config issue?" → check config → yes → fix
  Latency: sum of all three checks

With speculation:
  1. ┌─ Check DB ──┐  (parallel)
  2. ├─ Check code ┤
  3. └─ Check cfg ─┘
  4. Merge: config issue found → fix
  Latency: max of three checks, not sum
```

```python
@dataclass
class SpeculativeNode(ExecutionNode):
    node_type: NodeType = NodeType.SPECULATIVE
    hypotheses: list[ExecutionNode]     # parallel branches to explore
    confirmation_node: ExecutionNode     # determines which hypothesis is correct
    timeout_ms: int = 15000             # kill speculative branches after timeout
    max_branches: int = 3               # limit parallelism
```

---

## Comparison: Linear vs. Graph

| Dimension | Linear Loop | Graph Execution |
|---|---|---|
| **Parallelism** | None | Dependency-aware, configurable concurrency |
| **Planning** | Implicit (LLM decides step-by-step) | Explicit (Planner produces DAG upfront) |
| **Error handling** | Retry or fail | Fallback paths, rollback paths, conditional routing |
| **Model usage** | Same model for everything | Specialized models per worker (cheap for simple, expensive for critical) |
| **Cost control** | Hard (no insight into future steps) | Easy (entire graph known upfront, budget per node) |
| **Observability** | Sequential log | Full DAG trace with timing per node |
| **Rollback** | Manual or impossible | Built into graph structure |
| **Sub-delegation** | None | Subagent spawning with isolated context |
| **Speculative exec** | Impossible | Parallel hypothesis testing |
| **Reusability** | None (each run is ad-hoc) | Graph templates saved as "strategies" |
| **Cognitive load on LLM** | High (must plan + execute + verify) | Low (each worker has focused role) |

---

## Graph Templates (Reusable Strategies)

Successful execution graphs are saved as templates — analogous to procedures in the current skill system, but far more powerful:

```python
@dataclass
class GraphTemplate:
    name: str
    domain: str
    description: str
    graph: ExecutionGraph
    trigger_conditions: TriggerCondition
    usage_count: int
    success_rate: float
    avg_cost: float

    def instantiate(self, task: str, context: Context) -> ExecutionGraph:
        # Deep-copy the graph template
        # Bind parameters (task-specific tool args, thresholds, etc.)
        # Return a ready-to-execute graph
```

**Examples of learned graph templates:**

| Template | Trigger | Graph Shape |
|---|---|---|
| `investigate_and_fix` | Error/failure task | Fan-out investigate → merge → fix → verify |
| `root_cause_analysis` | Unknown issue | Speculative hypothesis branches → confirm → report |
| `safe_deploy` | Deployment task | Pre-check → deploy → verify → rollback-on-fail |
| `data_migration` | Schema change task | Backup → migrate → validate → verify-backup |
| `incident_triage` | Production alert | Parallel data gathering → severity assess → route |

---

## Implementation Roadmap

| Phase | What | How |
|---|---|---|
| 1 | DAG data model | `ExecutionGraph`, `ExecutionNode`, `DependencyEdge` — the core types |
| 2 | Sequential graph executor | Execute linear DAGs (one node after another) — same as current but structured |
| 3 | Parallel executor | Dependency-aware scheduler with concurrency limits |
| 4 | Planner module | LLM-based task decomposition into DAGs. Start with templates, add freeform |
| 5 | Critic module | Post-execution validation node. Retry with feedback on NEEDS_REVISION |
| 6 | Fallback/rollback | Conditional edges, alternative paths, rollback node execution |
| 7 | Subagent spawning | Nested execution graphs with isolated memory namespaces |
| 8 | Speculative execution | Parallel hypothesis branches, confirmation merge |
| 9 | Graph templates | Save successful graphs, retrieve by similarity, instantiate |
| 10 | Full integration | Planner → Scheduler → Workers → Critic. Replaces linear loop entirely |

---

## Key Differentiator

**Linear loop:**
The agent reacts step-by-step, never looking more than one action ahead. It's a greedy algorithm in cognitive form. Every run is ad-hoc, every mistake is learned only after it's made.

**Graph execution:**
The agent plans before it acts. It explores multiple hypotheses in parallel. It delegates specialized work to specialized sub-agents. It validates before proceeding. It rolls back cleanly when wrong. It reuses successful graph structures across runs.

The difference is between **reactive cognition** and **deliberate cognition** — the same leap that separates reflex from reasoned action in biological intelligence.
