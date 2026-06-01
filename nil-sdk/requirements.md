# Intelligence Layer — Overall Requirements

## Overview

The **Intelligence Layer** is a Python framework for building **self-evolving AI agents** that improve across runs. Agents remember what worked, extract reusable skills, detect when strategies stop working, and optimize their own performance — all autonomously, with SQLite-backed persistence and zero infrastructure.

The project consists of:

1. **nil-sdk** (`nil/`) — Core framework: kernel execution, persistent memory, evolution pipeline, LLM providers, tools system, cost optimization
2. **Extended Modules** — Retrieval-Augmented Cognition (RAC), Reinforcement Learning (RL), Multi-Agent Coordination, Environment Models (POMDP), Long-Horizon Benchmarking, Skill Compilation, Tool Embeddings & Affordance, Runtime Observability
3. **Meta-Learning Experiments** (`experiments/`) — MAML-based meta-learning with LiquidNN for few-shot adaptation
4. **NIL-STACC Bridge** — Connects stacc robots to a NIL self-evolving agent (in separate `nil-stacc/` repo)

---

## 1. nil-sdk Core Framework

### 1.1 Agent (`nil/agent.py`)

- **Agent** class — top-level orchestrator connecting Kernel, Memory, and Evolution
- `agent.run(task)` executes a task via KernelRunner, records trace in PersistentMemory, triggers evolution if enabled
- Evolution strategies: `"continuous"` (every run), `"periodic"` (every N runs), `"manual"` (on demand)
- Introspection API: `learned_skills()`, `success_rate()`, `stats()`, `evolution_history()`, `check_drift()`, `get_world_model()`, `get_strategy()`, `get_skill_prompt()`
- Auto-rebuilds runner after evolution changes strategy

### 1.2 Domain (`nil/domain.py`)

- **Domain** dataclass — defines problem space: name, tool list, objectives, success_signal callback
- Domain name normalized to lowercase, hyphenated for filesystem/db usage
- `domain.evaluate(outcome)` — evaluates run success; optimistic default (True) if no signal defined

### 1.3 Configuration (`nil/config.py`)

- **EvolutionConfig**: enabled, strategy (manual/periodic/continuous), evolve_interval, skill_extraction, cost_optimization, drift_detection, max_strategy_versions, human_approval
- **NilConfig**: log_level, trace_dir, default_model, max_retries, timeout

### 1.4 Kernel — Execution Engine (`nil/kernel/`)

- **KernelRunner** (`runner.py`) — Agentic loop: build system prompt → LLM call → tool execution → repeat up to 25 steps. Model-agnostic; auto-detects provider from env vars
- **Tracer** (`tracer.py`) — Structured experience capture per run. Records LLM calls, tool invocations, events into a `Trace` dataclass for Evolution Engine consumption
- **ToolManager** (`tool_manager.py`) — Registers tools, converts to LLM-compatible function-calling schemas, dispatches with timing + error capture
- **RunOutcome** + **StepRecord** (`outcome.py`) — Structured run result: output, success flag, step list, token counts, cost estimation ($3/M input, $15/M output tokens), tool sequence

### 1.5 Memory — Persistent Storage (`nil/memory/`)

- **PersistentMemory** (`store.py`) — SQLite-backed store with tables:
  - `traces` — run_id, domain, task, trace_data (JSON), success, duration, tokens, cost, tool_sequence, error
  - `skills` — domain, name, description, pattern (JSON), source_runs, confidence, usage_count
  - `strategy_versions` — domain, version, strategy_data (JSON), score, is_active
  - CRUD: store/get/list traces, skills, strategies; success rate calculation
- **HierarchicalMemory** (`hierarchical.py`) — Multi-tier memory: episodic, semantic, procedural, working, world state, social/user, meta-memory (design phase)

### 1.6 Evolution — Self-Improvement (`nil/evolution/`)

- **EvolutionPipeline** (`pipeline.py`) — Orchestrates 6 stages:
  1. **Analyze** — Causal analysis: counterfactual step importance, contiguous subsequence mining, Fisher's exact test, Wilson CI, temporal decay weighting, Markov transition matrices, regression trend detection
  2. **Extract Skills** — Compress patterns into reusable skills with trigger conditions, pre/postconditions; dedup, merge, deprecate
  3. **Compute Gradients** — Textual gradient descent: LLM-based critique of failed traces → strategy deltas (system prompt edits, tool ordering, reasoning hints)
  4. **Evaluate** — Replay-test candidates against historical traces; paired sign test for statistical significance; promote/reject/inconclusive
  5. **Build World Model** — Tool profiles, task clusters, domain constraints, state transition graphs, health metrics
  6. **Check Drift** — Multi-signal: success rate regression, new errors, tool degradation, cost inflation, task shift; auto-recovery (rollback on critical, targeted evolution on significant)
- **PipelineResult** — Tracks stages run, skills extracted/deprecated, deltas applied, strategy promoted, drift detected

### 1.7 Tools System (`nil/tools/`)

- **Tool** dataclass + `@tool` decorator (`base.py`) — Auto-generates JSON Schema from Python type hints; two usage modes: class-based or decorator-based
- **Registry** (`registry.py`) — Global in-process tool registry for cross-domain discovery
- **Tool Embeddings** (`embeddings.py`) — Continuous vector space representations from name, description, parameters; refined through usage, co-occurrence, outcome vectors
- **Affordance** (`affordance.py`) — Learns emergent tool properties: enabled transitions, associated goals, pre/postconditions, substitutes, complements, failure contexts; typed relationship graph
- **Composition** (`composition.py`) — Frequent sequence mining; synthesizes composite tools with aggregated metadata
- **Selection** (`selection.py`) — Latent tool selection via embedding similarity; precondition gating; affordance match + exploration bonus ranking

### 1.8 LLM Providers (`nil/providers/`)

- Provider adapters with `chat()` method; auto-detected from env vars:
  - **AnthropicProvider** — Anthropic Messages API (ANTHROPIC_API_KEY)
  - **OpenAIProvider** — Chat Completions API; also compatible with Ollama/vLLM via OPENAI_BASE_URL
  - **GeminiProvider** — Google Gemini API (GEMINI_API_KEY)
  - **StubProvider** — Deterministic fake LLM for testing; no API key needed

### 1.9 Optimization (`nil/optimization/`)

- **ModelRouter** — Route subtasks to cheapest adequate model (stub)
- **ChainCompressor** — Collapse deterministic multi-step chains (stub)
- **CacheManager** — In-memory LRU cache (SHA256 keyed on tool_name+args; max 1000 entries)
- **BudgetController** — Per-run ($1 default) and per-day ($50 default) cost limits

---

## 2. Extended Modules

### 2.1 Retrieval-Augmented Cognition — RAC (`nil/cognition/`)

- Dynamic retrieval replacing static skill injection
- Multi-source retrieval: skills, failures, trajectories, constraints, facts
- Hybrid sparse + dense retrieval with cross-encoder re-ranking
- Context assembler with per-source token budget allocation
- Strategy selection by task type (debug/fix, investigate, routine, recovery)
- Relevance scoring: semantic match + recency + specificity + confidence + diversity
- Feedback loop: boost/penalty retrieval scores from outcome

### 2.2 Reinforcement Learning Signals (`nil/rl/`)

- Dense per-step reward model (progress, cost, error, novelty, efficiency)
- Reward shaping via potential-based functions
- Inverse RL from expert demonstrations
- Trajectory scoring and Elo-based pairwise ranking
- Direct Preference Optimization (DPO)
- Offline RL (Conservative Q-Learning), Online RL (PPO), Behavioral Cloning
- Advantage estimation (GAE) and value function learning
- Bandit exploration (UCB1, Thompson Sampling, epsilon-greedy)
- Monte Carlo Tree Search (MCTS) for high-stakes decisions
- Full integration via RLEnabledAgent with replay buffer

### 2.3 Multi-Agent Coordination (`nil/agents/`)

- Agent role system: orchestrator, investigator, analyst, executor, verifier, critic, negotiator, monitor, memory_manager
- Per-agent model/config/prompt/temperature/cost budget
- Orchestrator: task decomposition → DAG of subtasks → parallel execution → synthesis
- Blackboard shared memory with namespacing, search, subscription
- Message bus: point-to-point, broadcast, pub/sub, request/reply
- Consensus: majority vote, structured debate, confidence-weighted consensus
- Agent pool with persistent/on-demand/pooled lifecycle
- Market-based task allocation (agents bid on subtasks)
- Hierarchical agent teams with nested decomposition

### 2.4 Environment Models — POMDP (`nil/models/`)

- Formal POMDP framework: StateSpace, TransitionModel, ObservationModel, RewardModel
- State space discovery from traces (clustering, dimensionality reduction, factorization)
- Belief tracking via Bayes filter (prediction + update steps)
- Model-based planning: value iteration, MCTS, heuristic policy
- Online/incremental model learning from every run
- Counterfactual query: "What if I had done X instead?"
- Full ModelAwareAgent integration with belief state

### 2.5 Long-Horizon Benchmarking (`nil/benchmarks/`)

- Learning curve benchmarks: cold start, learning velocity, plateau detection, sample efficiency, forgetting events
- Long-horizon tasks: 10/25/50/100-step sequences with step-level scoring and error recovery
- Recovery & robustness: tool failure, degradation, distribution shift, delayed rewards, adversarial inputs
- Memory retention: forgetting curves, interference testing, delayed recall
- Transfer & adaptation: zero-shot and few-shot (1/3/5/10) across domain similarity levels
- System benchmarks: cost, latency, memory growth, evolution overhead, retrieval quality
- Unified BenchmarkRunner with composite scoring and visualization

### 2.6 Skill Compilation (`nil/compilation/`)

- Compilation levels: text → few-shot → constrained generation → DSL → native code → decision tree
- SkillParser: extract tool sequences and control flow from textual skills
- Intermediate Representation (IR) with optimization passes: dead-step elimination, tool fusion, constant folding, specialization
- Code generation: Python functions, constrained grammar, DSL scripts
- Hybrid executor: mix compiled (deterministic) and LLM-guided steps
- Macro system: promote high-usage skills to single-call macros
- CompilationLearner: decide when/what level to compile based on determinism + confidence

### 2.7 Tool Embeddings & Affordance (`nil/tools/`)

- Tool embeddings from name, description, parameters; refined from usage patterns
- Affordance learner: state transitions, goals, pre/postconditions, substitutes, complements, failure contexts
- Compositional synthesis: frequent sequence mining → composite tools with benefit evaluation
- Latent tool selection: encode context → nearest embeddings → precondition gating → ranked activation
- ToolAffordanceGraph: NetworkX graph with typed edges for reasoning queries

### 2.8 Runtime Observability (`nil/observability/`)

- Structured event store: 23 event types across run lifecycle, LLM calls, tool calls, memory, retrieval, evolution
- Trace viewer: timeline with span grouping, timing, costs, expandable reasoning
- Flame graphs: span-based timing visualization, hot spot identification
- Token timeline: per-step token/cost breakdown with allocation analysis
- Reasoning tree: capture LLM options, evidence, chosen action, rationale, influences
- Memory browser: inspect skills, strategies, facts, world model in text and HTML
- Evolution diff viewer: per-cycle breakdown with before/after strategy comparison
- Strategy lineage: version history tree with branching, rollback, merge visualization
- Live debugger: pause/resume/step/inspect/edit_prompt/set_breakpoint/snapshot/rollback

---

## 3. Meta-Learning Experiments (`experiments/`)

- **Model**: LiquidNN (4→32→2, 5 layers) — liquid time-constant networks for continuous-time dynamics
- **Algorithm**: MAML (Model-Agnostic Meta-Learning) — inner_lr=0.01, 5 inner steps, outer_lr=0.001, 1000 epochs, 5 tasks per batch
- **Task**: 2D reaching (workspace_size=1.0, max 50 steps per episode); 10 support + 5 query episodes per task
- **Training**: Meta-trains across randomized reach tasks; MSE loss; checkpoint every 100 epochs
- **Evaluation**: Pre-adaptation episodes → 10 demos for few-shot adaptation → post-adaptation episodes → box plot comparison
- **Logging**: TensorBoard + Weights & Biases

---

## 4. NIL-STACC Bridge (`nil-stacc/nil_stacc/bridge.py`)

- **`bridge_tools(robot)`** — Auto-generates NIL tools from a stacc Robot:
  - `get_joint_state()` — joint positions and velocities
  - `move_joints(angles)` — set joint positions (comma-separated radians)
  - `set_torques(torques)` — apply joint torques
  - `stop_robot()` — emergency stop (zero all torques)
  - `get_robot_info()` — robot structure (links, joints, DOF)
  - Per-sensor `read_{name}()` — read each sensor with JSON clean output
  - `open_gripper()` / `close_gripper(force)` — end-effector control
  - `plan_path(sx, sy, gx, gy)` — A* collision-free path planning (OccupancyGrid 200×200, 0.05m resolution)
  - `smooth_move(target_angles, duration)` — MinJerk trajectory generation
  - `detect_hardware()` — hardware platform detection via Board.detect()
  - `parse_command(text)` — NLP command parsing via CommandParser
- **`RobotAgent`** — Self-evolving robot agent with NIL + stacc; persistent memory, continuous evolution
- **Factory functions**: `warehouse_agent()`, `inspection_agent()`, `research_agent()` with domain-specific objectives

---

## 5. Runtime Requirements

- Python >= 3.10
- Zero infrastructure: all storage local (SQLite)
- Model-agnostic: Anthropic, OpenAI, Gemini, or any OpenAI-compatible endpoint (Ollama, vLLM)
- Optional dependencies: `anthropic`, `openai` provider packages
- Dev dependencies: `pytest>=8.0`, `pytest-cov`, `ruff`
- Package name: `nil-sdk` v0.1.0

---

## 6. Design Documents (`docs/`)

| Document | Topic |
|---|---|
| `CODEBASE.md` | Complete codebase map with every file |
| `ENVIRONMENT_MODELS.md` | POMDP formalization: state space, transitions, rewards, belief tracking |
| `GRAPH_EXECUTION.md` | DAG-based execution replacing linear agent loop |
| `HIERARCHICAL_MEMORY.md` | 7-tier memory: episodic, semantic, procedural, working, world, social, meta |
| `LONG_HORIZON_BENCHMARKING.md` | Learning curves, forgetting, transfer, recovery, system benchmarks |
| `MULTI_AGENT_COORDINATION.md` | Society of agents: roles, blackboard, consensus, market, teams |
| `REINFORCEMENT_LEARNING.md` | RL signals: reward models, trajectory scoring, DPO, CQL, PPO, MCTS |
| `RETRIEVAL_AUGMENTED_COGNITION.md` | Dynamic context retrieval replacing static skill injection |
| `RUNTIME_OBSERVABILITY.md` | Structured events, flame graphs, token timelines, reasoning trees, debugger |
| `SKILL_COMPILATION.md` | Compilation pipeline: parse → IR → optimize → codegen → macro |
| `TOOL_EMBEDDINGS_AND_AFFORDANCE.md` | Tool embeddings, affordance learning, composition, latent selection |

---

## 7. Public API

Five primary exports from `nil`:

| Symbol | Source | Purpose |
|---|---|---|
| `nil.tool` | `nil/tools/base.py` | Decorator to convert function into Tool with auto-generated JSON Schema |
| `nil.Domain` | `nil/domain.py` | Binds tools, objectives, success signal |
| `nil.PersistentMemory` | `nil/memory/store.py` | SQLite-backed persistent storage |
| `nil.Agent` | `nil/agent.py` | Primary orchestrator — Kernel + Memory + Evolution |
| `nil.EvolutionConfig` | `nil/config.py` | Evolution settings (strategy, interval, approval gates) |

---

## 8. License

Apache 2.0 — Built by **Nelo Robotics Pvt Ltd**
