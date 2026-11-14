# Complete AI Development SDK & IDE Audit

Audit date: 2026-11-14. Scope: NIL at `ed7f987ced82bbb59c8373c1d9bd3d9dd7cd8563` and adjacent NELO Studio IDE at `e517c6e02cd1e8bf59fccaa8c947143cae67537c`. Reports reside in NIL because that is the writable workspace. NELO Studio **is the IDE**; NIL is an agent SDK/backend, not the complete IDE. Existing audit documents were not treated as test evidence. No existing source code was modified.

Evidence is local source inspection and explicitly listed safe probes. No external API requests, dependency downloads, model downloads, cloud jobs, deployments or hardware access were performed. Optional installed runtimes and release bundles may differ from these checkouts; their functionality is not assumed.

## Readiness assessment

NELO Studio is an implemented IDE with substantial backend integration code. The combined system is an alpha engineering platform, not a verified production AI development SDK. It contains real provider adapters, agent orchestration, SQLite memory and native worker boundaries, alongside missing runtime packages, disconnected governance paths and explicitly simulated legacy services. The root NIL daemon is a real execution bridge in source; its claim of real robotics depends on the external STACC service and was not exercised. No production-readiness claim is supportable with the current evidence.

## Scoring methodology

The 79 rows in AI_FEATURE_MATRIX.md are the scored scope; each row has equal weight within its area. Area weights total 100. Credit: Implemented and verified = 1.00; Implemented but untested = 0.50; Partially implemented = 0.25; Broken, UI-only / placeholder and Not implemented = 0.00. Not applicable would be excluded from the denominator; none of the requested areas is excluded here. A partial score reflects useful implementation fragments, not production readiness. Broken is used for a concrete defect or required component missing in this checkout; an uninstalled optional library alone does not prove code is broken.

Weighted completeness = sum(area weight × mean row credit) = **25.71/100**. Verified-only evidence floor = sum(area weight × verified-row fraction) = **4.04/100**. These are audit judgments over this fixed scope, not model accuracy, test coverage or measured reliability. Untested credit is explicitly provisional. Verification is narrow: a passing helper test never verifies its complete product feature. P0 defects block readiness regardless of score.

| Area | Weight | Completeness within area | Verified-only within area |
|---|---:|---:|---:|
| AI IDE | 12% | 34.4% | 0.0% |
| Data Engineering | 10% | 14.3% | 0.0% |
| Machine Learning | 13% | 21.4% | 0.0% |
| Generative AI | 13% | 33.3% | 11.1% |
| AI Agents | 13% | 47.5% | 20.0% |
| Multimodal AI | 6% | 10.7% | 0.0% |
| Optimization | 7% | 10.7% | 0.0% |
| MLOps | 10% | 20.8% | 0.0% |
| Security | 10% | 21.9% | 0.0% |
| Robotics AI | 6% | 20.0% | 0.0% |

## Technology and repository reconstruction

NIL is Python >=3.10, setuptools, SQLite, synchronous provider SDK calls, stdlib HTTPServer and optional NumPy robotics. NELO Studio is Next.js 16.4.5, React 19, TypeScript 5.7.3, Monaco, xterm, Tauri 2/Rust and native C++ workers. The network worker uses pinned Burn 0.22.1 flex CPU backend and Rerun 0.42.0. Historical Electron/browser implementations and studio/packages SDKs coexist with the current Tauri/Next app. A directory or installed package is not counted as a working integration.

## Critical and material findings

### P0-01: Plan mode can execute action tools

Priority: **P0**. Evidence: `../NELO-Studio/app/api/nil/route.ts:61`; `nil_server.py:107`.

The IDE builds read-only plan instructions, but first posts to the daemon without mode or policy. The daemon tools include joint motion, IK with motion, and gain tuning. A fallback-only prompt is not an execution boundary. Static control-flow defect; no motion attempted. Enforce immutable server-side read-only capability sets in plan mode.

### P0-02: Physical safety contract overclaims guarantees

Priority: **P0**. Evidence: `nil-sdk/nil/robotics/safety_gatekeeper.py:15`.

Only position and velocity are constrained. Torque and obstacle thresholds are stored but unused. NaN inputs pass comparison checks; mismatched/extra joints and nonpositive dt lack validation. Velocity interpolation from an out-of-limits current pose can remain outside limits. Do not treat this as a certified barrier or collision/torque supervisor; enforce finite dimensions, validated state and independent control-layer checks.

### P0-03: NIL action tools lack shared authorization and validation

Priority: **P0**. Evidence: `nil-sdk/nil/kernel/tool_manager.py:63`; `nil_server.py:87`.

NIL executes Python functions in its own process. T1 confirms wrong-type values reach the callable; T2 confirms a schema mismatch on typed lists. The root daemon does not call Studio authorizeAction or PhysicalSafetyGatekeeper. Existing Studio approvals do not establish universal enforcement. Require schema validation plus per-tool effects and safety policy at dispatch.

### P0-04: Legacy bridge exposes fabricated operational success

Priority: **P0**. Evidence: `nil-sdk/nil/server.py:27`; `nil-sdk/nil/server.py:149`.

Keyword replies fabricate simulation, gain-tuning and skill results; T4 confirms fake successful simulation response with no backend. Wildcard CORS and 0.0.0.0 bind lack authentication. This is a separate executable service; main IDE route targets root daemon instead. Remove from production or explicitly isolate as demo.

### P0-05: IK failures become actionable zero poses

Priority: **P0**. Evidence: `nil-sdk/nil/robotics/stacc_bridge.py:55`.

Unavailable STACC and broad exceptions return numeric zeros. Comment claims explicit degradation, but return type carries no status. Require a typed failure and block downstream actuation. This audit does not assert this helper is on the active desktop action path.

### P1-01: Public SDK install metadata cannot satisfy public import

Priority: **P1**. Evidence: `pyproject.toml:1`; `nil-sdk/nil/__init__.py:1`; `nil-sdk/nil/robotics/dmp.py:1`.

Public import eagerly imports robotics and NumPy, but neither pyproject declares NumPy. Clean copied checkout import fails ModuleNotFoundError: numpy. Stdlib-only requirement comment is false for public import. Make robotics lazy/optional or declare core dependency; test a fresh wheel.

### P1-02: Shared mutable defaults and inert API options

Priority: **P1**. Evidence: `nil-sdk/nil/agent.py:35`.

Default PersistentMemory is constructed while importing agent.py and reused across agents; EvolutionConfig default is also shared. enable_multi_agent and rl_config are accepted but not consumed; enable_rl constructs RLEnabledAgent without supplied rl_config. This creates misleading public API and lifecycle coupling. Use None defaults, explicit ownership and supported option tests.

### P1-03: Provider identity and request context are inconsistent

Priority: **P1**. Evidence: `../NELO-Studio/app/api/nil/route.ts:292`; `nil_server.py:161`.

Cloud fallback uses llama3.2 but does not update activeProviderUsed/activeModelUsed. Daemon request drops history, image, mode and system instructions; codeContent is sent by IDE but not consumed into daemon context. Keys are cache-key strings, and provider clients read process-global environment. Inject immutable clients per session, normalize context and disclose all fallback transitions.

### P1-04: Transport limits are advertised without common enforcement

Priority: **P1**. Evidence: `nil-sdk/nil/config.py:62`; `nil-sdk/nil/providers/openai.py:112`; `nil-sdk/nil/kernel/graph/scheduler.py:29`.

Config timeout/max_retries are not passed into OpenAI client/call; vendor defaults may exist but are not these settings. Scheduler stores timeout fields but waits on completed futures and does not enforce hard per-node termination. Root HTTPServer is single-threaded; long inference blocks health and other requests. Use cancellable worker processes and tested deadlines.

### P1-05: Bridge telemetry is split from SDK learning storage

Priority: **P1**. Evidence: `nil_server.py:119`; `nil_server.py:297`; `nil-sdk/nil/memory/store.py:36`.

PersistentMemory expects directory, but daemon passes a .db-looking path, producing nil_agent_memory.db/nil.db. UI nil_skills table is distinct from SDK skills; no synchronization insert found. EVOLUTION_CYCLES starts at 1 without persisted evidence. Expose a single read model from canonical storage and persist counters.

### P1-06: Embodied domain tools are plain functions

Priority: **P1**. Evidence: `nil-sdk/nil/robotics/domains.py:54`; `nil-sdk/nil/kernel/tool_manager.py:37`.

AutonomousManipulationDomain passes undecorated functions into Domain, while ToolManager.register requires Tool.name. Agent construction with that domain is structurally incompatible. execute_safe_joint_motion only returns computed targets; it does not send motion. Decorate tools and make simulated vs applied status explicit.

### P2-01: Training/notebook/data/serving runtime source absent

Priority: **P2**. Evidence: `../NELO-Studio/lib/training/runs.ts:52`; `../NELO-Studio/lib/notebook/kernels.ts:45`; `../NELO-Studio/lib/serving/server.ts:87`.

Default NELO-Studio/runtimes/python is absent and ignored by Git. No nelo_train implementation found in the inspected sibling STACC Python search either. Both artifact and ONNX tests fail at import. Optional runtime environment overrides could supply it externally; no reproducible source/bootstrap was verified here. Restore a versioned distributable package and reference it explicitly.

### P2-02: Serving rollback sacrifices healthy service before validation

Priority: **P2**. Evidence: `../NELO-Studio/lib/serving/server.ts:81`.

Existing process is stopped before candidate starts and passes health. Failure does not automatically restore previous model; PID-only persistence can misidentify recycled processes. Add candidate port, readiness, atomic routing and stable worker identity.

### P2-03: Reproducibility and artifact lifecycle incomplete

Priority: **P2**. Evidence: `requirements.txt:1`; `nil-sdk/Dockerfile:1`; `../NELO-Studio/lib/data/versions.ts:58`.

Python requirements are unpinned; root sklearn distribution name should be scikit-learn. Two same-name pyprojects diverge. Dockerfile CMD run_benchmarks.py does not match scripts/run_benchmarks.py in its build context. Dataset hash records are metadata, not reproducible immutable snapshots. Consolidate packaging and artifact/version contracts.

### P3-01: Embeddings, multimodal and optimization claims exceed implementation

Priority: **P3**. Evidence: `nil-sdk/nil/tools/embeddings.py:23`; `nil-sdk/nil/open_models/nelo_vision_mini.py:1`; `../NELO-Studio/studio/packages/compute-engine/nelo_compute/runtime.py:86`.

Hash vectors lack semantic similarity guarantees; vision module syntax invalid; legacy compute fabricates throughput/detections and registry writes tiny text placeholders as installed weights. Legacy CLI/control-plane imports mean this is reachable code, not merely an unused comment. Main IDE local inference instead calls Ollama; distinguish these paths in product labels.

### P3-02: Advanced learning modules are not proven production integrations

Priority: **P3**. Evidence: `nil-sdk/nil/agents/orchestrator.py:1`; `nil-sdk/nil/rl/policy.py:1`; `nil-sdk/nil/evolution/pipeline.py:1`.

Multi-agent, RL, curricula and strategy evolution are concrete modules, but do not establish model fine-tuning, benchmark improvement or safe autonomous robotics. Run held-out task evaluation and enforce promotion/rollback evidence before exposing guarantees.

### P4-01: Optimization modules retain explicit placeholders

Priority: **P4**. Evidence: `nil-sdk/nil/optimization/model_router.py:28`; `nil-sdk/nil/optimization/chain_compressor.py:22`.

Routing ignores task/complexity and compressor returns empty candidates. Other budget/cache helpers may work independently but are not proof of end-to-end optimization. Remove unsupported labels or integrate only against measurable baselines.

## Coverage and interpretation

All ten requested areas and individual capability classifications are in AI_FEATURE_MATRIX.md. Architecture boundaries are in AI_ARCHITECTURE_REVIEW.md. The missing-feature inventory preserves distinctions between unavailable runtime, partial adapter, placeholder and absent integration. Tests are documented with exact outcomes in AI_TEST_REPORT.md; existing tests and checked-in benchmark JSON were not counted as passing. No liveness, model accuracy, GPU speed, safe hardware operation or cloud provider correctness was demonstrated by this audit.
