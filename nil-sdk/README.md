# NIL SDK 0.2

For model-development installation, APIs, runnable examples and limitations, see
[the model-development guide](../docs/MODEL_DEVELOPMENT.md) and
[verification report](../SDK_IMPLEMENTATION_REPORT.md) in the repository root.

# NIL SDK

**Self-evolving AI agents that improve across runs.**

NIL is a Python framework for building agents that remember what worked, extract reusable skills, detect when strategies stop working, and optimize their own performance — all autonomously, with SQLite-backed persistence and zero infrastructure.

## Installation

```bash
pip install nil-sdk
```

Core SDK has **zero external dependencies** (stdlib only). Install provider packages as needed:

```bash
pip install anthropic    # Claude
pip install openai       # GPT / any OpenAI-compatible API
pip install google-genai # Gemini
```

## Quick Start

```python
from nil import Agent, Domain, tool, PersistentMemory, EvolutionConfig

@tool
def search_docs(query: str) -> str:
    """Search the documentation for a query."""
    return f"Results for: {query}"

@tool
def run_sql(sql: str) -> str:
    """Execute a SQL query and return results."""
    return f"Executed: {sql}"

domain = Domain(
    name="data-analysis",
    tools=[search_docs, run_sql],
    objectives=["Answer questions from the company database"],
    success_signal=lambda run: run.success,
)

agent = Agent(
    domain=domain,
    evolution=EvolutionConfig(strategy="continuous"),
    memory=PersistentMemory("./my_store"),
)

result = agent.run("What were total sales last quarter?")
print(result.output)
print(f"Success: {result.success}")

# Agent improves over time
agent.run("Break down sales by region")
agent.run("Compare Q1 vs Q2 revenue")
print(f"Success rate: {agent.success_rate():.0%}")
print(f"Learned skills: {agent.learned_skills()}")
```

## Architecture

```
nil/
├── agent.py                 # Top-level orchestrator
├── domain.py                # Problem space definition
├── config.py                # EvolutionConfig, NilConfig
│
├── kernel/                  # Execution engine
│   ├── runner.py            # Agentic loop (LLM → tools → repeat)
│   ├── outcome.py           # RunOutcome, StepRecord
│   ├── tracer.py            # Structured experience capture
│   ├── tool_manager.py      # Tool registration + dispatch
│   └── graph/               # DAG-based parallel execution
│       ├── graph_types.py   # ExecutionGraph, ExecutionNode, SubagentNode
│       ├── planner.py       # Task → DAG decomposition
│       ├── scheduler.py     # Dependency-aware scheduling
│       ├── runner.py        # End-to-end graph execution
│       ├── critic.py        # Output quality critique
│       └── templates.py     # Reusable graph templates
│
├── memory/                  # Persistent storage (SQLite)
│   ├── store.py             # PersistentMemory — traces, skills, strategies
│   ├── hierarchical.py      # 7-tier memory system
│   ├── episodic.py          # Episodes with retention tiers
│   ├── semantic.py          # Facts + entity graph
│   ├── procedural.py        # Procedures with refinement tracking
│   ├── working.py           # Content blocks with goals
│   ├── world_state.py       # Snapshots with delta tracking
│   ├── social_memory.py     # User profiles + corrections
│   ├── meta_memory.py       # Knowledge gaps + contradictions
│   ├── consolidation.py     # Cross-tier consolidation pipeline
│   ├── concept.py           # Concept graphs + abstraction
│   ├── abstraction.py       # Trace clustering engine
│   ├── experience.py        # Memory query interface
│   ├── skill_registry.py    # Skill metadata management
│   └── strategy_version.py  # Strategy history + rollback
│
├── evolution/               # Self-improvement pipeline
│   ├── pipeline.py          # 7-stage evolution orchestrator
│   ├── analyzer.py          # Causal analysis of traces
│   ├── skill_extractor.py   # Pattern → reusable skill extraction
│   ├── gradient_engine.py   # Textual gradient descent (LLM critique)
│   ├── evaluator.py         # Replay-test + statistical significance
│   ├── world_model.py       # Tool profiles, task clusters, transitions
│   ├── drift_detector.py    # Multi-signal performance degradation
│   └── strategy_mutator.py  # Strategy edit + rollback
│
├── cognition/               # Cognitive loop + RAC + economy
│   ├── loop.py              # 10-phase cognitive pipeline
│   ├── modes.py             # explore/exploit/investigate/recover modes
│   ├── policy.py            # Mode selection + escalation
│   ├── plugins.py           # Pluggable stage handlers
│   ├── runner.py            # RACEngine (retrieval-augmented cognition)
│   ├── retrieval.py         # Hybrid sparse + dense retriever
│   ├── sources.py           # Multi-source retrieval
│   ├── scoring.py           # Relevance scoring
│   ├── assembler.py         # Context assembly with token budgets
│   ├── index.py             # Retrieval index
│   ├── query.py             # Retrieval strategy selection
│   ├── feedback.py          # Outcome-based score adjustment
│   ├── simulation.py        # Candidate action simulation
│   ├── supervisor.py        # Online supervision
│   ├── adapters.py          # Cognition-triggered system modifications
│   └── economy.py           # Budget-aware execution (EV-based)
│
├── rl/                      # Reinforcement learning signals
│   ├── agent.py             # RLEnabledAgent + RLConfig
│   ├── rewards.py           # Dense, shaped, inverse RL reward models
│   ├── trajectory.py        # Trajectory scoring + Elo rating
│   ├── policy.py            # Behavioral cloning
│   ├── value.py             # Value functions + GAE advantage estimation
│   ├── exploration.py       # UCB1, Thompson Sampling, ε-greedy
│   ├── replay.py            # Experience replay buffer
│   └── preference.py        # DPO + trajectory ranking
│
├── models/                  # POMDP environment models
│   ├── state_space.py       # StateSpace + StateSpaceLearner
│   ├── transitions.py       # TransitionModel (Markov dynamics)
│   ├── observations.py      # ObservationModel
│   ├── rewards.py           # RewardModel + RewardLearner
│   ├── belief.py            # BeliefTracker (Bayes filter)
│   ├── planners.py          # Value iteration, MCTS, heuristic policy
│   ├── framework.py         # EnvironmentModel + ModelAwareAgent
│   └── learning.py          # Online model learning from traces
│
├── agents/                  # Multi-agent coordination
│   ├── roles.py             # 9 agent roles (orchestrator → monitor)
│   ├── orchestrator.py      # Task decomposition + DAG dispatch
│   ├── blackboard.py        # Shared memory with namespacing
│   ├── messages.py          # Point-to-point, broadcast, pub/sub
│   ├── consensus.py         # Vote, debate, confidence-weighted
│   ├── pool.py              # Persistent/on-demand agent lifecycle
│   ├── market.py            # Market-based task allocation (bidding)
│   ├── teams.py             # Hierarchical agent teams
│   └── framework.py         # MultiAgentNil coordinator
│
├── tools/                   # Tool system
│   ├── base.py              # @tool decorator + JSON Schema generation
│   ├── registry.py          # Global in-process tool registry
│   ├── embeddings.py        # Continuous vector representations
│   ├── affordance.py        # Emergent tool properties + graph
│   ├── composition.py       # Frequent sequence mining → composites
│   ├── selection.py         # Latent tool selection + precondition gating
│   └── evolver.py           # Workflow detection → tool synthesis
│
├── compilation/             # Skill compilation
│   └── pipeline.py          # Parse → IR → optimize → codegen → execute → macros
│
├── benchmarks/              # Long-horizon benchmarking
│   ├── learning.py          # Learning curves, velocity, plateau, transfer
│   ├── horizon.py           # 10/25/50/100-step task sequences
│   ├── recovery.py          # Failure, degradation, distribution shift
│   ├── memory.py            # Forgetting curves, interference, recall
│   ├── transfer.py          # Zero/few-shot cross-domain transfer
│   ├── system.py            # Cost, latency, memory growth
│   └── runner.py            # NilBenchmarkSuite + composite scoring
│
├── observability/           # Runtime observability
│   ├── events.py            # 23 event types across run lifecycle
│   ├── trace_viewer.py      # Timeline with span grouping
│   ├── flame_graph.py       # Span-based timing visualization
│   ├── token_timeline.py    # Per-step token/cost breakdown
│   ├── reasoning_tree.py    # LLM decision capture + rationale
│   ├── memory_browser.py    # Inspect skills, strategies, facts
│   ├── evolution_diff.py    # Before/after strategy comparison
│   ├── strategy_lineage.py  # Version history tree
│   └── debugger.py          # Pause/resume/step/breakpoint/snapshot/rollback
│
├── training/                # Curriculum learning
│   └── curriculum.py        # Weakness detection + synthetic practice
│
├── optimization/            # Cost/speed tuning
│   ├── model_router.py      # Route to cheapest adequate model (stub)
│   ├── chain_compressor.py  # Collapse deterministic chains (stub)
│   ├── cache_manager.py     # In-memory LRU cache (SHA256, 1000 entries)
│   └── budget_controller.py # Per-run ($1) + per-day ($50) limits
│
└── providers/               # LLM API adapters
    ├── anthropic.py         # Anthropic Messages API
    ├── openai.py            # OpenAI / Ollama / vLLM
    ├── gemini.py            # Google Gemini
    └── stub.py              # Deterministic fake for testing
```

## Public API

```python
from nil import (
    tool,            # Decorator: function → Tool with auto-generated JSON Schema
    Domain,          # Binds tools, objectives, success signal
    PersistentMemory, # SQLite-backed persistent storage
    HierarchicalMemory, # 7-tier hierarchical memory system
    Agent,           # Primary orchestrator — Kernel + Memory + Evolution + Cognition + RL
    EvolutionConfig, # Evolution settings (strategy, interval, approval gates)
)
```

## Key Features

| Feature | Description |
|---|---|
| **Self-Evolution** | 7-stage pipeline: analyze → extract skills → compute gradients → evaluate → build world model → check drift → evolve tools |
| **Cognitive Loop** | 10-phase pipeline: observe → retrieve → reason → plan → simulate → act → evaluate → reflect → learn → adapt |
| **Persistent Memory** | SQLite-backed 7-tier memory: episodic, semantic, procedural, working, world state, social, meta |
| **Retrieval-Augmented Cognition** | Dynamic context retrieval replacing static skill injection |
| **Reinforcement Learning** | Dense rewards, trajectory scoring, DPO, behavioral cloning, value functions, bandit exploration |
| **Multi-Agent** | 9 roles, blackboard shared memory, consensus mechanisms, market-based allocation, hierarchical teams |
| **POMDP Models** | Formal state tracking, belief updates, MCTS planning, online model learning |
| **Tool Evolution** | Workflow detection → composite tool synthesis |
| **Skill Compilation** | Parse → IR → optimize → codegen → hybrid execution → macros |
| **Long-Horizon Benchmarks** | Learning curves, forgetting, transfer, recovery, system benchmarks |
| **Observability** | Flame graphs, token timelines, reasoning trees, memory browser, live debugger |
| **Curriculum Learning** | Weakness detection → synthetic practice tasks |

## Examples

```bash
# Basic agent
python examples/basic_agent.py

# Robotics assembly cell (full domain-specific application)
python examples/robotics_assembly.py
```

## Scripts

```bash
python scripts/verify_system.py    # System health check
python scripts/run_benchmarks.py   # Run benchmark suite
python scripts/run_evolution.py    # Manual evolution cycle
python scripts/run_real.py         # Run with real LLM provider
```

## Configuration

```python
from nil import EvolutionConfig

# Conservative: evolve every 10 runs
config = EvolutionConfig(strategy="periodic", evolve_interval=10)

# Aggressive: evolve after every run
config = EvolutionConfig(strategy="continuous")

# Manual: only when you call agent.evolve()
config = EvolutionConfig(strategy="manual")

# With approval gates
config = EvolutionConfig(human_approval=True, max_strategy_versions=20)
```

## Testing

```bash
pytest tests/ -v
```

## Requirements

- Python >= 3.10
- Zero infrastructure: all storage local (SQLite)
- Model-agnostic: Anthropic, OpenAI, Gemini, or any OpenAI-compatible endpoint
- See [requirements.md](requirements.md) for full specification

## License

Apache 2.0 — Built by **Nelo Robotics Pvt Ltd**
