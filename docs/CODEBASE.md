# Intelligence Layer — Codebase Map

## Overview

**Nil** (`nil-sdk`) is a Python framework for building **self-evolving AI agents** that improve across runs. Agents remember what worked, extract reusable skills, detect when strategies stop working, and optimize their own performance — all autonomously, with SQLite-backed persistence and zero infrastructure.

---

## Top-Level Files

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata: `nil-sdk` v0.1.0, Python >=3.10, optional deps (anthropic, openai), dev deps (pytest, ruff) |
| `setup.py` | Directory scaffolding script (generates the project tree, not used at runtime) |
| `README.md` | Project documentation: quickstart, feature overview, architecture diagram, roadmap |

---

## `nil/` — Public API

| File | Purpose |
|---|---|
| `__init__.py` | Exports the public surface: `tool`, `Domain`, `PersistentMemory`, `Agent`, `EvolutionConfig` |
| `agent.py` | **Agent** — top-level orchestrator. `agent.run(task)` executes a task via the Kernel, records a trace in Memory, and (if enabled) triggers the Evolution Pipeline. Provides introspection: `learned_skills()`, `success_rate()`, `stats()`, `evolution_history()`, `check_drift()` |
| `domain.py` | **Domain** — defines the problem space: name, tool list, objectives, and `success_signal` callback to evaluate whether a run succeeded |
| `config.py` | **EvolutionConfig** (enabled, strategy: manual/periodic/continuous, evolve_interval, skill_extraction, drift_detection, human_approval) + **NilConfig** (log_level, max_retries, timeout) |

---

## `nil/kernel/` — Execution Engine

| File | Purpose |
|---|---|
| `runner.py` | **KernelRunner** — the agentic loop. Builds a system prompt (domain context + learned skills), calls the LLM, executes tool calls, feeds results back, repeats until a final answer or max steps (25). Model-agnostic; auto-detects provider from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY` |
| `tracer.py` | **Tracer** + **Trace** — structured experience capture. Records every LLM call, tool invocation, and event into a machine-readable `Trace` object designed for consumption by the Evolution Engine |
| `tool_manager.py` | **ToolManager** — registers tools, converts them to LLM-compatible function-calling schemas, dispatches invocations with timing and error capture |
| `outcome.py` | **RunOutcome** + **StepRecord** — the structured result of a run: output, success flag, step list, token counts, cost estimation ($3/M input tokens, $15/M output tokens), tool sequence |

---

## `nil/memory/` — Persistent Storage

| File | Purpose |
|---|---|
| `store.py` | **PersistentMemory** — SQLite-backed long-term store. Tables: `traces` (run_id, domain, task, trace_data JSON, success, tokens, cost, tool_sequence, error), `skills` (domain, name, description, pattern JSON, confidence, usage_count), `strategy_versions` (domain, version, strategy_data JSON, score, is_active). Methods: store/get/list traces, skills, strategies, success rates |
| `experience.py` | **ExperienceQuery** — query/analysis interface over historical traces. `stats()` aggregates success rate, avg cost, avg steps, most common tool sequences, most common errors. `successful_tool_sequences()` finds winning patterns. `common_failures()` groups failures by error signature |
| `skill_registry.py` | **Skill** + **TriggerCondition** — data model for extracted skills. Lifecycle: candidate → active → deprecated → superseded. Trigger conditions based on keywords, error patterns, tool prerequisites, step position. Generates prompt fragments (`to_prompt_fragment()`, `to_compact_prompt()`) for system prompt injection |
| `strategy_version.py` | **StrategyVersion** — versioned snapshot of an agent's strategy (system prompt modifications, skill selections, model routing rules, tool preferences). Enables rollback |

---

## `nil/evolution/` — Self-Improvement Engine

| File | Purpose |
|---|---|
| `pipeline.py` | **EvolutionPipeline** — orchestrator that runs all 6 stages in sequence. Decides what to skip based on data availability (e.g., needs 10+ traces, world model rebuilds every 50 traces). Returns `PipelineResult` with summary of what changed |
| `analyzer.py` | **TraceAnalyzer** — deep causal analysis. 8 techniques: counterfactual step importance, contiguous subsequence mining, step-level causal attribution (lift, phi coefficient, Fisher's exact test, p<0.05, Wilson confidence intervals), exponential temporal decay weighting, cost-efficiency scoring, failure clustering (LCS-based), Markov transition probability matrix, linear regression performance trend detection |
| `skill_extractor.py` | **SkillExtractor** — converts causal patterns into compressed, reusable skills. Pipeline: run analysis → filter by confidence/frequency → infer triggers from task keywords → compress multi-step sequences into instructions → extract pre/postconditions → deduplicate → merge with existing skills → deprecate ineffective ones → enforce max skills limit → store to memory |
| `gradient_engine.py` | **TextualGradientEngine** — core IP. Inspired by TextGrad: treats agent optimization as automatic differentiation via text. A `GradientProvider` (StubGradientProvider for testing, AnthropicGradientProvider for real LLM critiques) takes failed traces, produces critiques, and proposes strategy improvements. Critiques become TextualGradients, which become StrategyDeltas (concrete changes to system prompt, tool ordering, reasoning hints, etc.) |
| `evaluator.py` | **ReplayEvaluator** — tests candidate strategies against historical data before deployment. Scores each trace on success alignment, failure avoidance, cost efficiency, and constraint compliance. Uses paired sign test for statistical significance. Returns promote/reject/inconclusive with confidence intervals and rollback risk assessment |
| `drift_detector.py` | **DriftDetector** — multi-signal performance monitoring. Detects: success rate regression, new error types, tool degradation, cost inflation, task distribution shift. Severity levels: none → mild → significant → critical. Auto-recovery: rollback on critical, targeted evolution on significant. Generates recovery actions (add error handling, update tool guidance, optimize cost) |
| `world_model.py` | **WorldModelBuilder** — constructs a structured domain model from traces. Components: tool profiles (success rates, costs, complementary/redundant tools, typical position), task clusters (keyword-based), domain constraints (prerequisites, forbidden transitions, required minimum steps, conditional errors), state transition graph, health metrics. Generates prompt injection blocks for agent self-awareness |

---

## `nil/optimization/` — Cost & Performance (Phase 4 Stubs)

| File | Purpose |
|---|---|
| `model_router.py` | **ModelRouter** — routes subtasks to the cheapest adequate model (e.g., simple classification → Haiku, complex reasoning → Opus). Currently a stub |
| `chain_compressor.py` | **ChainCompressor** — identifies deterministic multi-step chains and collapses them into single cached steps. Currently a stub |
| `cache_manager.py` | **CacheManager** — in-memory hash-based cache for deterministic tool results. Uses SHA256 keyed on (tool_name, arguments). Max 1000 entries with LRU eviction |
| `budget_controller.py` | **BudgetController** — enforces cost limits per run ($1 default) and per day ($50 default). Prevents runaway API spend |

---

## `nil/providers/` — LLM Adapters

| File | Purpose |
|---|---|
| `anthropic.py` | **AnthropicProvider** — wraps Anthropic Messages API. Translates internal message format ↔ Anthropic tool_use/tool_result content blocks. Requires `ANTHROPIC_API_KEY` |
| `openai.py` | **OpenAIProvider** — wraps OpenAI Chat Completions API. Also compatible with any OpenAI-API server (Ollama, vLLM) via `OPENAI_BASE_URL`. Requires `OPENAI_API_KEY` |
| `gemini.py` | **GeminiProvider** — wraps Google Gemini API via `google-genai` SDK. Converts to FunctionCall/FunctionResponse parts. Requires `GEMINI_API_KEY` |
| `stub.py` | **StubProvider** — deterministic fake LLM for testing without API keys. First call invokes the first available tool with stub arguments; subsequent calls return a success message. Resets call count per run |

---

## `nil/tools/` — Tool System

| File | Purpose |
|---|---|
| `base.py` | **Tool** dataclass + `@tool` decorator. Two usage modes: class-based (`class MyTool(Tool): invoke(self, args)`) or decorator-based (`@tool def my_fn(param: str) -> ...`). Auto-generates JSON Schema parameter definitions from Python type hints |
| `registry.py` | Global in-process tool registry. `register_tool()`, `get_tool()`, `list_tools()` for cross-domain tool discovery |

---

## `experiments/` — Meta-Learning (separate from nil-sdk)

| File | Purpose |
|---|---|
| `configs/meta_reach.yaml` | Experiment config: LiquidNN model (4→32→2, 5 layers), MAML meta-learning (inner_lr=0.01, 5 inner steps, outer_lr=0.001, 1000 epochs), reach task settings |
| `testings/train_meta_learner.py` | Meta-trains a LiquidNN using MAML (Model-Agnostic Meta-Learning) on randomized reach tasks. Collects support/query episodes per task, computes meta-loss via MSE, saves checkpoints and final model |
| `testings/evaluate_train_meta_learner.py` | Evaluates trained meta-learner: runs pre-adaptation episodes, collects 10 demos for few-shot adaptation, runs post-adaptation episodes, plots box plot comparison |
| `runs/meta_reach_v1/` | Checkpoints and final trained model output |

---

## Other Directories

| Path | Contents |
|---|---|
| `test_store/` | Pre-existing SQLite database (`nil.db`) from test executions |
| `nil_store/`, `real_store/`, `evolution_test_store/` | Additional SQLite store directories created during development/testing |
