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
| `requirements.md` | Comprehensive dependency overview covering all modules, providers, experiments, and dev tooling |
| `requirements.txt` | Pip-installable requirements file with pinned extras for LLM providers, experiments, and development |

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
| `embeddings.py` | **ToolEmbedding** + **ToolEmbeddingGenerator** — continuous vector space representations of tools. Constructed from name, description, and parameter embeddings; refined through usage, co-occurrence, and outcome vectors. Cosine similarity for tool comparison |
| `affordance.py` | **ToolAffordance**, **AffordanceLearner**, **ToolAffordanceGraph** — learns emergent tool properties from execution traces: enabled state transitions, associated goals, pre/postconditions, substitutes, complements, and failure contexts. Graph with typed edges for tool relationship queries |
| `composition.py` | **CompositeTool** + **ToolCompositionSynthesizer** — discovers frequent tool sequences via pattern mining, evaluates composition benefits, and registers synthesized composite tools with aggregated cost/duration/success-rate metadata |
| `selection.py` | **LatentToolSelector**, **LatentAwarePromptBuilder**, **ScoredTool** — selects tools via embedding similarity rather than text generation. Encodes context, finds nearest tools in embedding space, gates by precondition satisfaction, ranks with affordance match + exploration bonus |

---

## `nil/models/` — POMDP Environment Models

| File | Purpose |
|---|---|
| `state_space.py` | **EnvironmentState**, **StateSpace**, **StateSpaceLearner** — structured representation of agent/environment states. Discovers states from traces (by error, tool used, domain keywords), builds transition counts between states |
| `transitions.py` | **TransitionModel** — learns P(s' \| s, a) from trace data. Records outcome distributions per action, computes expected outcomes, uncertainty estimates, known actions, and merges models |
| `observations.py` | **ObservationModel** — learns P(o \| s) observation probabilities. Records observations per state, computes most likely state, diagnostic value of observations |
| `rewards.py` | **RewardModel** — discounted return computation, reward-per-trace learning from cost/success/novelty signals |
| `belief.py` | **BeliefTracker** — Bayesian belief updates over state space. Maintains P(s \| history), incorporates evidence and task priors, returns top-N most likely states |
| `planners.py` | **ValuePlanner** (value iteration), **MCTSPlanner** (Monte Carlo tree search with UCB1), **HeuristicPolicy** (rule-based policy). All produce action recommendations from belief states |
| `learning.py` | **OnlineModelLearner** — incremental learning from traces. Updates transition, observation, reward models per trace or batch |
| `framework.py` | **EnvironmentModel** (unified POMDP) + **ModelAwareAgent** — full cycle: ingest traces, maintain belief, suggest actions, replan automatically after N traces |

---

## `nil/benchmarks/` — Long-Horizon Benchmarking

| File | Purpose |
|---|---|
| `base.py` | **BenchmarkSuite**, **BenchmarkTask**, **BenchmarkResult** — framework for defining and running long-horizon evaluations. Task templates with success criteria, cost/time budgets, metrics tracking |
| `learning.py` | **LearningCurve**, **LearningAnalyzer** — tracks performance across sequential attempts. Computes convergence rate, plateau detection, learning efficiency from exponentially-weighted moving averages |
| `horizon.py` | **HorizonBenchmark** — tests agent performance over sustained multi-turn interactions. Evaluates consistency across extended episodes with fatigue and drift detection |
| `recovery.py` | **RecoveryBenchmark** — injects controlled failures to test the agent's ability to detect, diagnose, and recover from errors. Measures recovery time, success rate, and adaptation |
| `memory.py` | **MemoryBenchmark** — evaluates retention and recall quality across increasing memory loads. Tests working memory, long-term knowledge retention, and interference resilience |
| `transfer.py` | **TransferBenchmark** — measures how well skills and strategies transfer across related domains. Tests generalization, adaptation speed, and negative transfer detection |
| `system.py` | **SystematicVariationBenchmark** — stress-tests agents under systematic perturbations: tool failures, delayed responses, ambiguous tasks, conflicting objectives |
| `runner.py` | **BenchmarkRunner** — orchestrates benchmark execution. Registers benchmark suites, schedules trials, collects results, generates comprehensive reports with pass/fail/duration/cost |

---

## `nil/agents/` — Multi-Agent Coordination

| File | Purpose |
|---|---|
| `roles.py` | **AgentRole** + **RoleRegistry** — defines agent capabilities, responsibilities, and interaction patterns. Supports coordinator, executor, critic, observer, explorer roles with delegation rules |
| `blackboard.py` | **Blackboard**, **BlackboardEntry**, **BlackboardSubscription** — shared memory space where agents post findings, hypotheses, and pending tasks. Subscription-based notification on key updates |
| `messages.py` | **AgentMessage**, **MessageBus**, **MessagePriority** — typed inter-agent communication with priority levels, routing, and delivery guarantees. Structured message format with references and threading |
| `pool.py` | **AgentPool** — manages a collection of agent instances. Load-balancing, lifecycle management, capacity-based routing, and health monitoring |
| `orchestrator.py` | **Orchestrator** — top-level coordinator that decomposes tasks, assigns sub-tasks, monitors progress, and handles agent failures. Runs the full plan → assign → monitor → consolidate → adapt loop |
| `consensus.py` | **ConsensusEngine** — voting-based decision making across agents. Supports majority, supermajority, unanimous, and weighted voting strategies with confidence scoring |
| `market.py` | **MarketAllocator** — economic task allocation where agents bid on sub-tasks. Auction-based with budget constraints, bid scoring, and dynamic pricing |
| `teams.py` | **AgentTeam** + **TeamRegistry** — persistent agent teams with role composition, performance tracking, and dynamic restructuring on underperformance |
| `framework.py` | **MultiAgentFramework** — unified entry point integrating all coordination components. Full workflow: plan → form team → assign tasks → execute → communicate → reach consensus → consolidate |

---

## `nil/rl/` — Reinforcement Learning Signals

| File | Purpose |
|---|---|
| `rewards.py` | **DenseRewardModel**, **ShapedRewardModel**, **InverseRLRewardModel** — step-level reward computation from progress, cost penalties, error penalties, and exploration bonuses. Potential-based shaping and IRL weight learning from demonstrations |
| `trajectory.py` | **TrajectoryScorer**, **EloTrajectoryRating** — scores full trajectories on cost-efficiency, success rate, and step economy. Elo-based pairwise comparison with leaderboard ranking |
| `preference.py` | **DirectPreferenceOptimizer** — learns from pairwise trajectory preferences. Bradley-Terry model for preference probability, updates action logits from comparison data |
| `policy.py` | **Policy**, **BehavioralCloning** — action selection policy with log-probability computation, weighted updates, and BC from demonstration traces |
| `value.py` | **ValueFunction**, **AdvantageEstimator** — state-value prediction with MSE-based updates. GAE and Monte Carlo advantage estimation for policy gradient methods |
| `exploration.py` | **ExplorationScheduler** — UCB1-based exploration with visit counting, uncertainty bonuses, and action sampling. Balances exploration vs exploitation per action |
| `replay.py` | **ReplayBuffer** — fixed-capacity experience replay with FIFO eviction, uniform sampling, and latest-N retrieval |
| `agent.py` | **RLEnabledAgent** — integrates RL into the agent loop. Selects actions via policy, records step outcomes, computes rewards, stores in replay buffer, finishes trajectories |

---

## `nil/cognition/` — Retrieval-Augmented Cognition (RAC)

| File | Purpose |
|---|---|
| `query.py` | **RetrievalQuery**, **RetrievalStrategy**, **RetrievalStrategySelector** — encodes the current task context (task, tool state, errors, step, domain) into a query. Selects retrieval strategy by context: failure_recovery (on error), exploration (early steps), minimal (late steps), balanced (default) |
| `index.py` | **StoredItem**, **ScoredResult**, **RetrievalIndex** — persistent storage for items with dense embeddings (cosine similarity), sparse keyword index, and constraint-based retrieval. Supports upsert, delete, and multi-modal search |
| `retrieval.py` | **HybridRetriever** — fuses dense, sparse, and structure-based retrieval into combined relevance scores. Re-ranks by keyword overlap with the query |
| `scoring.py` | **RelevanceScorer** — multi-factor relevance scoring: semantic match, recency (exponential decay), specificity (keyword overlap), confidence, and diversity penalty to avoid redundant results |
| `sources.py` | **MultiSourceRetriever** — indexes and retrieves from five source stores: skills, failures, trajectories, constraints, facts. Per-source scoring with failure-event boosting and tool-context matching |
| `assembler.py` | **ContextAssembler** — assembles scored results into a formatted context string with per-source token budgets. Falls back to compact format when budget is exceeded |
| `feedback.py` | **RetrievalFeedback** — records success/failure feedback on retrieval quality. Tracks positive/negative signals, supports to_dict/from_dict serialization |
| `runner.py` | **RACEngine** — unified entry point. Maintains both index and source stores, runs search with strategy selection, assembles context, supports run_with_feedback for reinforcement learning |

---

## `nil/observability/` — Runtime Observability

| File | Purpose |
|---|---|
| `events.py` | **EventType**, **ObservabilityEvent**, **EventRecorder**, **ObservabilityStore** — structured event system with 23 event types spanning run lifecycle, LLM calls, tool calls, memory operations, retrieval, and evolution. Span stack for flame graph grouping |
| `trace_viewer.py` | **TraceViewer**, **TimelineView**, **TimelineRow** — renders execution timelines from stored events. Groups by span ID, computes timing, supports tree layout |
| `flame_graph.py` | **FlameGraphGenerator**, **FlameGraph**, **FlameNode** — generates flame graph data from span-timed events. Identifies hot spots, computes aggregated duration/cost/token metadata |
| `token_timeline.py` | **TokenTimeline**, **TokenReport**, **TokenStep** — tracks token consumption per step. Estimates cost ($3/M in, $15/M out), breaks down by category (LLM, retrieval, evolution) |
| `reasoning_tree.py` | **ReasoningTree**, **ReasoningNode** — captures LLM reasoning per step: options considered, chosen action, rationale, skill/memory influences. Text rendering for inspection |
| `memory_browser.py` | **MemoryBrowser** — inspects agent memory: skills (active/deprecated with stats), strategy versions (with history), facts. Generates text and HTML views |
| `evolution_diff.py` | **EvolutionDiffViewer**, **EvolutionDiff** — per-evolution-cycle breakdown. Shows trigger, duration, traces analyzed, stage results (analysis, extraction, gradients, evaluation, drift), and strategy diff |
| `strategy_lineage.py` | **StrategyLineage**, **LineageGraph**, **LineageNode** — version history tree with mutation types, parent tracking, rollback detection, and active-version marking |
| `debugger.py` | **LiveDebugger**, **DebugCommand**, **Breakpoint**, **AgentSnapshot** — interactive debugger with pause/resume, step, breakpoints (step, tool, event), snapshots, rollback, and command execution |

---

## `nil/compilation/` — Skill Compilation

| File | Purpose |
|---|---|
| `pipeline.py` | **SkillParser** (extracts tool sequences and control flow from skill descriptions), **IntermediateRepresentation** (structured IR with linear/conditional detection and compilation eligibility), **SkillOptimizer** (dead-step elimination, tool fusion, specialization), **SkillCodeGenerator** (emits Python, DSL, or constrained grammar), **CompiledSkillExecutor** (compile + cache + execute), **CompilationLearner** (measures determinism, decides compilation level), **HybridSkillExecutor** (mixes compiled and LLM steps), **SkillCompilationEngine** (full pipeline: should_compile → parse → optimize → codegen → promote to macro), **SkillMacro** (callable macro with usage tracking and avg duration), **MacroRegistry** (promote high-usage skills to macros), **CompilationResult** (compilation outcome with level, compiled object, and IR) |

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
| `docs/` | Project documentation: codebase map (`CODEBASE.md`), design docs |
| `test_store/` | Pre-existing SQLite database (`nil.db`) from test executions |
| `nil_store/`, `real_store/`, `evolution_test_store/` | Additional SQLite store directories created during development/testing |
