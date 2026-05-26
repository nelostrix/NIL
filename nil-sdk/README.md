# Nil — Self-Evolving Domain Agent Framework

**AI agents that get better every time they run.**

Nil is a Python SDK that gives AI agents persistent memory and self-improvement. Unlike frameworks where agents start from zero every session, Nil agents remember what worked, extract reusable skills, detect when strategies stop working, and optimize their own performance — all autonomously.

## Quick Start

```bash
pip install nil-sdk
```

```python
import nil

@nil.tool
def search_logs(query: str) -> str:
    """Search application logs."""
    return search_my_logs(query)

@nil.tool
def apply_fix(action: str) -> str:
    """Apply a remediation action."""
    return execute_fix(action)

domain = nil.Domain(
    name="incident-response",
    tools=[search_logs, apply_fix],
    objectives=["resolve production incidents autonomously"],
    success_signal=lambda run: run.success
)

agent = nil.Agent(
    domain=domain,
    base_model="claude-sonnet-4-20250514",
    memory=nil.PersistentMemory("./agent_store"),
    evolution=nil.EvolutionConfig(strategy="continuous"),
)

# Every run makes the agent smarter
result = agent.run("API latency spike on /v2/users")
```

## Why Nil?

Every AI agent framework today produces **goldfish agents** — they start from zero every time, repeat the same mistakes, and can't learn from experience.

Nil solves this with three systems:

1. **Agent Kernel** — Executes tasks while recording structured experience traces (not just logs — machine-readable execution data).

2. **Evolution Engine** — Analyzes traces to extract reusable skills, detect performance drift, and autonomously improve strategies.

3. **Optimization Layer** — Routes subtasks to cheaper models when safe, compresses redundant chains, and manages cost budgets.

## Key Features

- **Persistent learning** — agents remember across sessions via SQLite
- **Skill extraction** — successful patterns become reusable skills with trigger conditions, preconditions, and postconditions
- **Causal analysis** — Fisher's exact test, Wilson CI, temporal decay weighting, counterfactual analysis to identify what actually caused success or failure
- **Textual gradient descent** — inspired by TextGrad: critique → gradient → delta → apply, with both heuristic and LLM-based providers
- **Drift detection** — multi-signal monitoring (success rate, new errors, tool degradation, cost, task shift) with severity classification and auto-recovery
- **World model building** — entity discovery, tool profiling, task clustering, constraint discovery, state transition graphs
- **Model agnostic** — works with Claude (Anthropic), GPT (OpenAI), Gemini (Google), or any LLM via OpenAI-compatible endpoints (Ollama, vLLM)
- **Zero infrastructure** — runs on any laptop, stores everything locally
- **Cost tracking** — per-run token and cost accounting

## Project Structure

```
nil/
├── __init__.py           # Public API: tool, Domain, Agent, PersistentMemory, EvolutionConfig
├── agent.py              # Main Agent class — orchestrates Kernel + Memory + Evolution
├── domain.py             # Domain definitions — binds tools, objectives, success signals
├── config.py             # EvolutionConfig, NilConfig
├── kernel/               # Execution engine
│   ├── runner.py         # Agent loop (LLM + tool calls, up to 25 steps)
│   ├── tracer.py         # Structured experience capture (LLM calls, tool calls, events)
│   ├── tool_manager.py   # Tool registration, validation, dispatch
│   └── outcome.py        # RunOutcome, StepRecord — result + timing + costs
├── memory/               # Persistent storage
│   ├── store.py          # SQLite-backed memory — traces, skills, strategy versions
│   ├── experience.py     # ExperienceQuery, ExperienceStats — aggregation, patterns
│   ├── skill_registry.py # Skill data model with full lifecycle (candidate → active → deprecated)
│   └── strategy_version.py # Strategy version dataclass
├── evolution/            # Self-improvement pipeline
│   ├── pipeline.py       # Orchestrates all 6 stages (analyze → extract → gradient → evaluate → model → drift)
│   ├── analyzer.py       # Causal analysis: Fisher's test, Wilson CI, subsequence mining, transition matrices
│   ├── skill_extractor.py # Skill extraction: chain compression, dedup, trigger inference, merge
│   ├── gradient_engine.py # Textual gradient descent: critique → gradient → delta → apply
│   ├── evaluator.py      # Replay evaluator, A/B comparison, multi-dimensional scoring, rollback risk
│   ├── drift_detector.py # Multi-signal drift detection, severity, root cause, auto-recovery
│   ├── world_model.py    # Entity discovery, tool profiles, task clusters, state transition graphs
│   └── strategy_mutator.py # Strategy mutation (placeholder — not yet implemented)
├── optimization/         # Cost optimization (stubs — under development)
│   ├── model_router.py   # Smart model routing (always returns default)
│   ├── chain_compressor.py # Chain compression (no-op)
│   ├── cache_manager.py  # In-memory LRU response cache
│   └── budget_controller.py # Per-run and per-day cost limit enforcement
├── providers/            # LLM adapters
│   ├── anthropic.py      # Claude (Anthropic API)
│   ├── openai.py         # OpenAI + any OpenAI-compatible endpoint (Ollama, vLLM)
│   ├── gemini.py         # Google Gemini
│   └── stub.py           # Deterministic fake LLM for testing (no API key needed)
└── tools/                # Tool system
    ├── base.py           # Tool dataclass + @tool decorator (auto-generates JSON Schema)
    └── registry.py       # Global in-process tool registry
```

## Code Understanding

### How the systems connect

```
Agent.run(task)
  │
  ├─ KernelRunner.execute(task, tracer)
  │    ├─ Builds system prompt (includes active skills + strategy)
  │    ├─ LLM call → tool calls → LLM call → ... (up to 25 steps)
  │    └─ Returns RunOutcome (success, steps, timing, costs)
  │
  ├─ Tracer.finalize(outcome) → Trace (all LLM calls, tool calls, events)
  │
  ├─ Domain.evaluate(outcome) → bool (success signal)
  │
  ├─ PersistentMemory.store_trace(trace) → persists to SQLite
  │
  └─ EvolutionPipeline.run (if enabled)
       ├─ TraceAnalyzer: statistical + causal analysis
       ├─ SkillExtractor: extract + dedup + merge skills
       ├─ TextualGradientEngine: critique failures → generate deltas
       ├─ ReplayEvaluator: score + A/B test the deltas
       ├─ WorldModelBuilder: entity/tool/constraint discovery
       └─ DriftDetector: check performance degradation
            └─ On impact: Agent._refresh_runner() → picks up new skills
```

### Public API

Only 5 symbols are exported from the `nil` top-level package:

| Symbol | Type | Source | Purpose |
|--------|------|--------|---------|
| `nil.tool` | decorator | `nil/tools/base.py` | Convert a Python function into a tool with auto-generated JSON Schema |
| `nil.Domain` | class | `nil/domain.py` | Binds tools, objectives, and a success signal into a named domain |
| `nil.PersistentMemory` | class | `nil/memory/store.py` | SQLite-backed storage for traces, skills, and strategy versions |
| `nil.Agent` | class | `nil/agent.py` | Primary orchestrator — connects Kernel, Memory, Evolution |
| `nil.EvolutionConfig` | class | `nil/config.py` | Evolution settings (strategy, interval, human approval gates) |

Sub-package internals (`nil.kernel`, `nil.evolution.pipeline`, `nil.providers`, etc.) can be imported by their full module path when you need direct access.

### Agent evolution strategies

- **`"continuous"`** — evolves after every run
- **`"periodic"`** — evolves every N runs (configurable via `evolve_interval`)
- **`"manual"`** — you call `agent.evolve()` explicitly

### Introspection

```python
agent.learned_skills()       # list of skill names
agent.get_skill_prompt()     # skill injection block for system prompts
agent.get_world_model()      # entity/tool/constraint model
agent.get_strategy()         # active strategy
agent.evolution_history()    # list of evolution run summaries
agent.check_drift()          # drift check without full evolution
agent.success_rate()         # success rate over last N runs
agent.stats()                # full agent state summary
```

### Running without an API key

Nil ships a `StubProvider` that works without any API key — it calls the first available tool on the first invocation and returns a canned response on subsequent calls. Useful for testing and development.

```python
agent = nil.Agent(domain=domain)
# When no ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY is set,
# the kernel auto-selects the StubProvider.
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```

## Built by

**Nelo Robotics Pvt Ltd** — Building the intelligence layer for autonomous systems.

## License

Apache 2.0
