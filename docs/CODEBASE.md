# Intelligence Layer — Codebase Map

## Overview

**Nil** (`nil-sdk`) is a Python framework for building **self-evolving AI agents** that improve across runs. Agents remember what worked, extract reusable skills, detect when strategies stop working, and optimize their own performance — all autonomously, with SQLite-backed persistence and zero infrastructure.

---

## Top-Level Files

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata: `nil-sdk` v0.1.0, Python >=3.10, optional deps (anthropic, openai), dev deps (pytest, ruff) |
| `nil-sdk/pyproject.toml` | Package metadata: `nil-sdk` v0.1.0, Python >=3.10, optional deps, dev deps |
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
| `graph/` | **DAG Execution Engine** — `planner.py` (6 planning strategies), `scheduler.py` (dependency-aware parallelism), `runner.py` (graph executor), `critic.py` (validation with PASS/NEEDS_REVISION/FAIL), `graph_types.py` (ExecutionNode, DependencyEdge), `templates.py` (reusable DAG templates like `investigate_and_fix`) |

---

## `nil/memory/` — Persistent Storage (SQLite + In-Memory Tiers)

| File | Purpose |
|---|---|
| `store.py` | **PersistentMemory** — core SQLite-backed long-term store. Tables: `traces` (run_id, domain, task, trace_data JSON, success, tokens, cost, tool_sequence, error), `skills`, `strategy_versions`. Methods: store/get/list traces, skills, strategies, success rates, `update_trace_success()` for post-hoc success correction |
| `experience.py` | **ExperienceQuery** — query/analysis interface over historical traces. `stats()` aggregates success rate, avg cost, avg steps, tool sequences, errors |
| `skill_registry.py` | **Skill** + **TriggerCondition** — data model with lifecycle: candidate → active → deprecated → superseded. Generates prompt fragments for system prompt injection |
| `strategy_version.py` | **StrategyVersion** — versioned strategy snapshots with rollback support |
| `hierarchical.py` | **HierarchicalMemory** — 7-tier memory coordinator (episodic, semantic, procedural, working, world_state, social, meta) |
| `episodic.py` | **EpisodicMemory** — raw traces with retention lifecycle: RECENT → SUMMARIZED → COMPRESSED → ARCHIVED → FORGOTTEN |
| `semantic.py` | **SemanticMemory** — facts, rules, concepts, relationships with confidence tracking and decay |
| `procedural.py` | **ProceduralMemory** — skills, policies, strategies with refinement from failures |
| `working.py` | **WorkingMemory** — sliding window with importance-based retention, attention mechanism, scratchpad |
| `world_state.py` | **WorldStateMemory** — persistent external state snapshots, diff-logging, consistency checking |
| `social_memory.py` | **SocialMemory** — user profiles, preferences, interaction history, trust calibration |
| `meta_memory.py` | **MetaMemory** — confidence calibration, knowledge gaps, contradictions, epistemic curiosity |
| `concept.py` | **Concept**, **ConceptType**, **AbstractionLevel**, **ConceptGraph** — building blocks for learned concepts |
| `abstraction.py` | **AbstractionEngine** — concept abstraction pipeline: cluster → causal → symbolic → policies |
| `consolidation.py` | **ConsolidationPipeline** — cross-tier memory consolidation (per-run, per-N-runs, on-demand) |

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
| `strategy_mutator.py` | **StrategyMutator** — mutation operations for strategy evolution (placeholder — under development) |

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

## `nil/training/` — Autonomous Curriculum (Phase 6)

| File | Purpose |
|---|---|
| `__init__.py` | Exports `CurriculumEngine`, `WeaknessAnalyzer`, `SyntheticTaskGenerator`, `PracticeEngine`, `BenchmarkSuite` |
| `curriculum.py` | **CurriculumEngine** — orchestrates self-directed capability growth. Pipeline: `WeaknessAnalyzer` → identify low-success tools, weak task types, and missing transitions from trace data → `SyntheticTaskGenerator` creates targeted training tasks → `PracticeEngine` runs the agent on exercises and tracks improvement → `BenchmarkSuite` measures before/after performance. All modules share a common `_trace_success()` helper for dict/object trace compatibility |

## `nil/tools/` — Tool System

| File | Purpose |
|---|---|
| `base.py` | **Tool** dataclass + `@tool` decorator. Two usage modes: class-based (`class MyTool(Tool): invoke(self, args)`) or decorator-based (`@tool def my_fn(param: str) -> ...`). Auto-generates JSON Schema parameter definitions from Python type hints |
| `registry.py` | Global in-process tool registry. `register_tool()`, `get_tool()`, `list_tools()` for cross-domain tool discovery |
| `embeddings.py` | **ToolEmbedding** + **ToolEmbeddingGenerator** — continuous vector space representations of tools. Constructed from name, description, and parameter embeddings; refined through usage, co-occurrence, and outcome vectors. Cosine similarity for tool comparison |
| `affordance.py` | **ToolAffordance**, **AffordanceLearner**, **ToolAffordanceGraph** — learns emergent tool properties from execution traces: enabled state transitions, associated goals, pre/postconditions, substitutes, complements, and failure contexts. Graph with typed edges for tool relationship queries |
| `composition.py` | **CompositeTool** + **ToolCompositionSynthesizer** — discovers frequent tool sequences via pattern mining, evaluates composition benefits, and registers synthesized composite tools with aggregated cost/duration/success-rate metadata |
| `selection.py` | **LatentToolSelector**, **LatentAwarePromptBuilder**, **ScoredTool** — selects tools via embedding similarity rather than text generation. Encodes context, finds nearest tools in embedding space, gates by precondition satisfaction, ranks with affordance match + exploration bonus |
| `evolver.py` | **ToolEvolver** — Phase 5 Tool Evolution. Pipeline: `WorkflowDetector` mines traces for repeated tool sequences (configurable min_occurrences, min_success_rate, sequence length bounds), `WorkflowCompiler` converts detected workflows into `Tool` objects with auto-generated parameter schemas derived from observed parameter variations, and `ToolEvolver` orchestrates detect → filter (by fingerprint, avoiding duplicates) → compile → register. Evolved tools are named `evolved_<prefix>_<hash>` and execute the full workflow sequence by dispatching to underlying registered tools. `EvolutionResult` reports workflows detected, tools evolved, and duration |

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

## `nil/cognition/` — Retrieval-Augmented Cognition + Executive Cognition + Cognitive Economy

| File | Purpose |
|---|---|
| `query.py` | **RetrievalQuery**, **RetrievalStrategy**, **RetrievalStrategySelector** — encodes task context into a query with strategy selection per context |
| `index.py` | **RetrievalIndex** — persistent storage with dense embeddings (cosine similarity), sparse keyword index, constraint-based retrieval |
| `retrieval.py` | **HybridRetriever** — fuses dense, sparse, and structure-based retrieval with cross-encoder re-ranking |
| `scoring.py` | **RelevanceScorer** — multi-factor scoring: semantic match, recency decay, specificity, confidence, diversity penalty |
| `sources.py` | **MultiSourceRetriever** — indexes and retrieves from skills, failures, trajectories, constraints, facts stores |
| `assembler.py` | **ContextAssembler** — assembles scored results into formatted context with per-source token budgets |
| `feedback.py` | **RetrievalFeedback** — records success/failure feedback on retrieval quality |
| `runner.py` | **RACEngine** — unified entry point, runs search with strategy selection, assembles context |
| `loop.py` | **CognitiveLoop** — 10-phase Unified Cognitive Loop (observe→retrieve→reason→plan→simulate→act→evaluate→reflect→learn→adapt). Dynamic pipeline building, recursive cognition, per-phase uncertainty, plugin hooks, supervisor integration |
| `modes.py` | **CognitiveMode** enum (8 modes), **ModeProfile**, **MODE_PROFILES**, **Uncertainty** dataclass |
| `policy.py` | **CognitivePolicyEngine** — heuristic mode selection, dynamic pipeline building, budget allocation |
| `simulation.py` | **SimulationEngine** — candidate plan generation, world-model rollouts, reward/risk estimation |
| `supervisor.py` | **LiveSupervisor** — mid-task adaptation with 10 triggers, per-trigger cooldown, mode escalation |
| `adapters.py` | **ModelSwitcher**, **AgentSpawner**, **RetrievalAlterer**, **ReasoningDeeper** — run-time cognitive adaptations |
| `economy.py` | **CognitiveEconomy** — cost-aware cognition, model tiering, phase gating, EV calculation |
| `plugins.py` | **MemoryPlugin**, **RACPlugin**, **KernelPlugin**, **EvolutionPlugin**, **RLPlugin** — cognitive loop plugins. `wire_cognitive_loop()` wires all dependencies |

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

## `nil-sdk/tests/` — Test Suite

20 pytest test files covering all subsystems (kernel, memory, evolution, cognition, agents, benchmarks, RL, compilation, affordance, hierarchical memory, graph execution, observability, world model, end-to-end). Run via `pytest tests/` from the `nil-sdk/` directory.

## Other Directories

| Path | Contents |
|---|---|
| `docs/` | Project documentation: codebase map (`CODEBASE.md`), design docs |
| `test_store/`, `nil_store/`, `real_store/`, `evolution_test_store/` | SQLite store directories created during development/testing |
