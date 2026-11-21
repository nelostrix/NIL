# Nil SDK — User Guide

## What This System Is

Nil is a **self-evolving AI agent framework**. You give it tools and tasks, it runs them via an LLM, remembers every run in a local SQLite database, and automatically improves its strategy over time — extracting skills, detecting drift, and optimizing its approach — all without you touching any configuration.

```
You define:  tools + domain + tasks
Nil does:   run → remember → analyze → evolve → improve
```

---

## Quick Start (Zero Setup)

```python
import nil
j
@nil.tool
def search_logs(query: str) -> str:
    """Search application logs for errors."""
    return f"Found entries matching: {query}"

@nil.tool
def apply_fix(action: str) -> str:
    """Apply a remediation action."""
    return f"Applied: {action}"

domain = nil.Domain(
    name="incident-response",
    tools=[search_logs, apply_fix],
    objectives=["Diagnose and resolve production incidents"],
)

agent = nil.Agent(
    domain=domain,
    evolution=nil.https://meet.google.com/bjc-zgfn-dpxEvolutionConfig(enabled=True, strategy="continuous"),
)

result = agent.run("Database timeout — find and fix")
print(f"Success: {result.success}")
print(f"Tools used: {result.tool_sequence}")
print(f"Duration: {result.duration:.1f}s")
```

Run this. With no API keys it falls back to `StubProvider` (offline deterministic). Point it at Ollama or any OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
python your_script.py
```

---

## What Users See and Experience

### 1. The Agent Runs Tasks

Every call to `agent.run(task)` produces a `RunOutcome`:

```python
result = agent.run("Investigate API latency spike")

result.success       # True/False
result.output        # The agent's final response
result.steps         # Every step taken (LLM calls + tool calls)
result.tool_sequence # e.g. ["search_logs", "check_status", "apply_fix", "verify"]
result.duration      # Wall-clock seconds
result.total_cost    # Estimated cost in USD
result.had_errors    # Whether any step errored
```

### 2. The Agent Remembers

All runs are stored in a local SQLite database (`./agent_store/nil.db` by default). You can inspect what it's learned:

```python
agent.run_count          # Total runs executed
agent.success_rate()     # Success rate over last N runs
agent.learned_skills()   # Skills the agent has extracted
agent.get_strategy()     # Current active strategy dict
agent.evolution_history()# List of past evolution cycles
agent.stats()            # Full stats dictionary
agent.check_drift()      # Check if performance is degrading
```

### 3. The Agent Evolves (Self-Improvement)

When evolution is enabled, the agent periodically analyzes its own traces:

- **Analyze** — Finds causal patterns: which steps lead to success/failure, statistical significance tests, Markov transition matrices, performance trends
- **Extract Skills** — Compresses effective multi-step patterns into reusable skills with trigger conditions
- **Compute Gradients** — LLM-based textual critique of failed traces → strategy improvement deltas
- **Evaluate** — Replay-tests candidate strategies against historical data before deployment
- **Build World Model** — Learns tool profiles, task clusters, constraints, state transitions
- **Check Drift** — Detects regression, new error types, cost inflation, task shifts; auto-recovers by rolling back or targeting evolution

The agent tells you what changed after each evolution:

```
Evolution #1: skills_extracted=2, deltas_applied=3, drift=no, duration=1.2s
  → Learned skill: "Always search_logs before apply_fix" (confidence 87%)
  → Strategy delta: add reasoning hint "check logs first"
```

### 4. Strategy Updates Affect Future Runs

After evolution, the agent rebuilds its internal runner with the updated strategy. The next `agent.run()` will use the improved system prompt, tool guidance, and reasoning hints. You see the effect in gradually improving success rates and more efficient tool sequences.

---

## What's Inside the SDK

### nil/agent.py — Top-level orchestrator

```python
agent = nil.Agent(
    domain=domain,
    base_model="llama3.2",
    memory=nil.PersistentMemory("./my_store"),
    provider="openai",                    # auto-detected if not set
    evolution=nil.EvolutionConfig(...),    # controls how/when to evolve
    enable_rl=False,                       # reinforcement learning signals
    enable_multi_agent=False,              # multi-agent coordination
    enable_tool_evolution=False,           # auto-create new tools from workflows
    enable_curriculum=False,               # adaptive task difficulty
    enable_economy=False,                  # cost-aware budget tracking
)
```

### nil/tools/ — Tool System

- **`@nil.tool` decorator** — Auto-generates JSON Schema from Python type hints. Two modes:

```python
# Decorator style (recommended)
@nil.tool
def my_tool(param1: str, param2: int = 0) -> str:
    """Description becomes the tool description."""
    return f"Result: {param1}, {param2}"
```

- **Tool Embeddings** (`embeddings.py`) — Continuous vector representations from name, description, parameters; refined through usage patterns
- **Tool Affordance** (`affordance.py`) — Emergent tool properties: pre/postconditions, substitutes, complements, failure contexts
- **Tool Composition** (`composition.py`) — Frequent sequence mining → composite tools
- **Tool Selection** (`selection.py`) — Latent selection via embedding similarity + precondition gating + exploration bonus
- **Tool Evolution** (`evolver.py`) — `ToolEvolver` discovers repeated workflows from traces and synthesizes new composite tools

### nil/kernel/ — Execution Engine

- **KernelRunner** (`runner.py`) — Agentic loop: build prompt → LLM call → tool execution → repeat (up to 25 steps). Model-agnostic, auto-detects provider from env vars
- **Graph Execution** (`graph/`) — DAG-based execution replacing the linear loop:
  - `Planner` — Decomposes tasks into execution DAGs (6 planning strategies)
  - `GraphScheduler` — Dependency-aware parallel execution
  - `GraphAgentRunner` — Full DAG execution end-to-end
  - `Critic` — Validates outputs with PASS/NEEDS_REVISION/FAIL verdicts
  - `TemplateLibrary` — Reusable DAG templates (`investigate_and_fix`, etc.)
- **ToolManager** (`tool_manager.py`) — Registers tools, converts to LLM schemas, dispatches with timing
- **Tracer** (`tracer.py`) — Structured experience capture per run

### nil/memory/ — Storage

- **PersistentMemory** (`store.py`) — Core SQLite-backed store. Tables: `traces`, `skills`, `strategy_versions`, `concepts`. CRUD for all.
- **HierarchicalMemory** (`hierarchical.py`) — 7-tier memory system:
  - **Episodic** — Raw experiences with retention lifecycle (working → short → long → archival)
  - **Semantic** — Facts, entity graphs, typed relationships
  - **Procedural** — Skills, policies, strategies with refinement tracking
  - **Working** — Sliding window with importance-based retention
  - **World State** — External state snapshots with delta tracking
  - **Social/User** — User profiles, preferences, corrections
  - **Meta-Memory** — Knowledge gaps, contradictions, self-awareness
- **Consolidation** (`consolidation.py`) — Cross-tier consolidation pipeline
- **Concept Learning** (`concept.py`, `abstraction.py`) — Concept graphs and abstraction engine
- **Experience Query** (`experience.py`) — Query interface over historical traces
- **Skill Registry** (`skill_registry.py`) — Skill lifecycle management
- **Strategy Versioning** (`strategy_version.py`) — Version history with rollback

### nil/evolution/ — Self-Improvement

- **Pipeline** (`pipeline.py`) — Orchestrates 6 stages
- **Analyzer** (`analyzer.py`) — 8 causal analysis techniques (counterfactual importance, subsequence mining, Fisher's exact test, Markov transitions, regression trends, etc.)
- **Skill Extractor** (`skill_extractor.py`) — Pattern → compressed skill with triggers, pre/postconditions, dedup, merge, deprecate
- **Gradient Engine** (`gradient_engine.py`) — LLM-based textual gradient descent; critiques failed traces → strategy deltas
- **Evaluator** (`evaluator.py`) — Replay-test candidates against historical traces; paired sign test for statistical significance
- **World Model** (`world_model.py`) — Tool profiles, task clusters, constraints, state transitions, health metrics
- **Drift Detector** (`drift_detector.py`) — Multi-signal monitoring: regression, new errors, tool degradation, cost inflation, task shift; auto-recovery

### nil/providers/ — LLM Adapters

Auto-detected from env vars. No import needed.

| Provider | Env Variable | Extra |
|---|---|---|
| **Anthropic** | `ANTHROPIC_API_KEY` | `pip install anthropic` |
| **OpenAI** | `OPENAI_API_KEY` (+ `OPENAI_BASE_URL` for Ollama/vLLM) | `pip install openai` |
| **Gemini** | `GEMINI_API_KEY` | `pip install google-genai` |
| **Stub** (fallback) | — | No deps. Offline, deterministic. |

### nil/cognition/ — Extended Execution

- **CognitiveLoop** (`loop.py`) — Phase-based execution: Plan → Execute → Reflect → Adapt → Learn
- **CognitiveMode** (`modes.py`) — Mode profiles: explore, exploit, investigate, recover
- **Plugins** (`plugins.py`) — Pluggable stage handlers (Evolution, Kernel, Memory, RAC, RL)
- **Policy Engine** (`policy.py`) — Mode selection, escalation, phase management
- **Retrieval-Augmented Cognition (RAC)** (`runner.py`) — Dynamic retrieval from skills, failures, trajectories, constraints, facts. Hybrid sparse+dense retrieval with cross-encoder re-ranking
- **CognitiveEconomy** (`economy.py`) — Token/cost/step budgets per run. EV-based action selection
- **SimulationEngine** (`simulation.py`) — Simulates candidate actions before execution
- **LiveSupervisor** (`supervisor.py`) — Online supervision with warnings and recommendations

### nil/rl/ — Reinforcement Learning Signals

- Dense per-step reward model (progress, cost, error, novelty, efficiency)
- Reward shaping via potential-based functions
- Inverse RL from expert demonstrations
- Trajectory scoring and Elo-based pairwise ranking
- Direct Preference Optimization (DPO)
- Agent action-weight adaptation, terminal-aware return/advantage helpers and
  action-frequency behavioral cloning. These agent helpers are not neural PPO,
  CQL or full RLHF/DPO implementations. Environment-training integrations are
  documented separately in `nil help rl` and the live TODO.md checklist.
- Bandit exploration (UCB1, Thompson Sampling, epsilon-greedy)
- Monte Carlo Tree Search (MCTS) for high-stakes decisions
- Full integration via `RLEnabledAgent` with replay buffer

### nil/agents/ — Multi-Agent Coordination

- Agent roles: orchestrator, investigator, analyst, executor, verifier, critic, negotiator, monitor, memory_manager
- Orchestrator decomposes tasks → DAG of subtasks → parallel execution → synthesis
- Blackboard shared memory with namespacing, search, subscription
- Message bus: point-to-point, broadcast, pub/sub, request/reply
- Consensus: majority vote, structured debate, confidence-weighted
- Market-based task allocation (agents bid on subtasks)

### nil/models/ — Environment Models (POMDP)

- Formal POMDP framework: StateSpace, TransitionModel, ObservationModel, RewardModel
- State space discovery from traces (clustering, dimensionality reduction)
- Belief tracking via Bayes filter
- Model-based planning: value iteration, MCTS, heuristic policy
- Online/incremental model learning from every run
- Counterfactual query: "What if I had done X instead?"

### nil/training/ — Curriculum Learning

- **CurriculumEngine** — Analyzes historical weaknesses and generates targeted practice tasks
- **WeaknessAnalyzer** — Identifies skill gaps, frequent failure modes, underutilized tools
- **PracticeEngine** — Generates synthetic tasks for identified weaknesses
- **SyntheticTaskGenerator** — Creates varied practice scenarios from domain definition

### nil/compilation/ — Skill Compilation

- Compilation levels: text → few-shot → constrained generation → DSL → native code → decision tree
- IR with optimization passes: dead-step elimination, tool fusion, constant folding, specialization
- Code generation: Python functions, constrained grammar, DSL scripts
- Hybrid executor: mix compiled (deterministic) and LLM-guided steps
- Macro system: promote high-usage skills to single-call macros

### nil/benchmarks/ — System Benchmarks

- `NilBenchmarkSuite` — Runs all benchmarks and produces a composite score
- **Learning**: tool selection, sequential reasoning, context dependence
- **Horizon**: 10/25/50/100-step tasks with step accuracy and error recovery
- **Recovery**: tool failure, degradation, distribution shift, delayed rewards
- **Memory**: forgetting curves, interference testing
- **Transfer**: zero-shot and few-shot (1/3/5/10) to similar/different domains
- **System**: cost, latency, memory growth, evolution overhead

### nil/observability/ — Runtime Observability

- Structured event store (23 event types)
- Trace viewer with timeline, span grouping, timing, costs
- Flame graphs for hot spot identification
- Token/cost breakdown per step
- Reasoning tree: options, evidence, chosen action, rationale
- Memory browser: inspect skills, strategies, facts, world model
- Evolution diff viewer: before/after strategy comparison
- Strategy lineage: version history tree with branching, rollback, merge
- Live debugger: pause/resume/step/inspect/edit_prompt/set_breakpoint/snapshot/rollback

### nil/optimization/ — Cost & Performance Optimization

- `quantize`: supported CPU dynamic INT8 conversion.
- `export_onnx`: export and validate a compatible model.
- `benchmark`, `profile`: explicit CPU inference measurements.

---

## Ready-to-Run Scripts

| Script | What it does | Run it |
|---|---|---|
| `run_real.py` | 10 incident-response tasks with periodic evolution | `python run_real.py` |
| `run_evolution.py` | Injects synthetic traces → triggers evolution pipeline | `python run_evolution.py` |
| `run_benchmarks.py` | Full system benchmark suite with PASS/FAIL scoring | `python run_benchmarks.py --quick` |
| `examples/robotics_assembly.py` | Dual-arm robotics workcell simulation with 11 tools | `python examples/robotics_assembly.py` |

---

## Typical User Workflow

```
1. Define tools (@nil.tool) with clear descriptions + type hints
2. Define Domain(name, tools, objectives, success_signal)
3. Create Agent(domain, evolution=EvolutionConfig(enabled=True))
4. Run tasks → agent.run(task)
5. Inspect results → result.success, result.tool_sequence, agent.stats()
6. Watch it evolve → agent.evolution_history(), agent.learned_skills()
7. Benchmark → python run_benchmarks.py --quick
8. Iterate → Add tools, adjust objectives, tune evolution config
```

The agent works with zero config (StubProvider), gets better with a local model (Ollama), and is production-ready with Anthropic/OpenAI.

---

## Common Questions

**Q: Do I need API keys?**  
No. Without any, the StubProvider runs fully offline. Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY` for real LLMs.

**Q: Where is data stored?**  
Local SQLite file. Default is `./agent_store/nil.db`. Pass a custom path to `PersistentMemory("./path")`.

**Q: How many runs before evolution kicks in?**  
Evolution needs at least 10 traces. With `continuous` strategy, it runs after every `agent.run()`. With `periodic`, every N runs.

**Q: Can I see what the agent learned?**  
Yes — `agent.learned_skills()`, `agent.get_strategy()`, `agent.evolution_history()`, `agent.stats()`.

**Q: How do I reset?**  
Delete the store directory: `rm -rf ./agent_store`. Or use a new path.

**Q: Can I run with Docker?**  
```bash
docker build -t nil-benchmark .
docker run --rm nil-benchmark python run_benchmarks.py --quick
```
