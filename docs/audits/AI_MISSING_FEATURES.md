# Missing, Broken and Incomplete AI Capabilities

Audit date: 2026-11-14. Scope: NIL at `ed7f987ced82bbb59c8373c1d9bd3d9dd7cd8563` and adjacent NELO Studio IDE at `e517c6e02cd1e8bf59fccaa8c947143cae67537c`. Reports reside in NIL because that is the writable workspace. NELO Studio **is the IDE**; NIL is an agent SDK/backend, not the complete IDE. Existing audit documents were not treated as test evidence. No existing source code was modified.

Evidence is local source inspection and explicitly listed safe probes. No external API requests, dependency downloads, model downloads, cloud jobs, deployments or hardware access were performed. Optional installed runtimes and release bundles may differ from these checkouts; their functionality is not assumed.

## Concrete defects and misleading interfaces

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

## Capability gaps by area

### AI IDE

- **LSP integration — Partially implemented**. Native bridge and initialization exist; packaging helpers pass 3 tests, actual server sessions untested. Evidence: `../NELO-Studio/lib/lspClient.ts:26`; `../NELO-Studio/lib/tauriLsp.ts:1`. Next step: Validate bundled servers and native IPC on each OS.
- **Notebook execution — Broken**. Spawns nelo_train.kernel; default runtimes/python absent. Evidence: `../NELO-Studio/lib/notebook/kernels.ts:41`. Next step: Ship versioned Jupyter bridge and kernelspec tests.
- **Visual network workflows — Partially implemented**. Real graph/export and Rust worker code; executable not built or run. Evidence: `../NELO-Studio/lib/network/workerClient.ts:50`; `../NELO-Studio/lib/network/export.ts:77`. Next step: Build CPU worker and compare exported model outputs.
- **Extensions / MCP tools — Partially implemented**. Manifest capabilities currently limited to agent.tools; not a general VS Code extension host. Evidence: `../NELO-Studio/lib/plugins/registry.ts:21`. Next step: Enforce capabilities at transport and process boundaries.

### Data Engineering

- **Local table profiling / validation — Broken**. Calls tables helper through runtimeDir; helper runtime absent. Evidence: `../NELO-Studio/app/api/data/table/route.ts:25`. Next step: Restore helper package with schema, leakage and malformed-data fixtures.
- **Preprocessing pipelines — Not implemented**. No owned executable transform DAG found in inspected data paths. Evidence: `../NELO-Studio/components/training/DataHub.tsx:1`. Next step: Integrate Arrow/DuckDB transforms with manifests.
- **Annotation — Not implemented**. Media browsing is present; no labeling schema, editor persistence or review workflow found. Evidence: `../NELO-Studio/components/training/DataHub.tsx:1`. Next step: Integrate established labeling backend.
- **Dataset version records — Partially implemented**. Hashes and JSON metadata; no immutable snapshot, retrieval or restore. Evidence: `../NELO-Studio/lib/data/versions.ts:58`. Next step: Use DVC/lakeFS or content-addressed snapshots.
- **Data lineage — Partially implemented**. Joins dataset hash events to runs; producer runtime missing. Evidence: `../NELO-Studio/lib/data/versions.ts:1`. Next step: Record dataset reads automatically and validate lineage end to end.
- **Streaming ingestion — Not implemented**. Paginated remote reads are not streaming ingestion; no offsets/checkpoints found. Evidence: `../NELO-Studio/app/api/data/hf/route.ts:1`. Next step: Use framework streaming datasets with resumable offsets.

### Machine Learning

- **PyTorch training launch — Partially implemented**. Real process launch but nelo_train runner absent in default checkout. Evidence: `../NELO-Studio/lib/training/runs.ts:230`. Next step: Ship runner and tiny CPU optimizer-step test.
- **TensorFlow integration — Partially implemented**. Framework scanner and generic artifact intent; no verified TensorFlow training integration. Evidence: `../NELO-Studio/lib/mlFrameworks.ts:38`. Next step: Add framework callback adapter and SavedModel roundtrip.
- **Scikit-learn integration — Not implemented**. sklearn requirement is not integration; no estimator lifecycle adapter found. Evidence: `requirements.txt:1`. Next step: Use scikit-learn package and fit/predict/artifact adapter.
- **Checkpointing and resumption — Broken**. Artifact tests fail importing nelo_train; optimizer and RNG resume not verified. Evidence: `../NELO-Studio/lib/training/runs.ts:8`; `../NELO-Studio/tests/python/test_training_artifacts.py:1`. Next step: Package checkpoint API and compare resumed vs uninterrupted training.
- **Distributed execution — Partially implemented**. torchrun and cluster submission code exist; no multi-process launch verified. Evidence: `../NELO-Studio/lib/training/runs.ts:257`; `../NELO-Studio/lib/compute/cluster.ts:1`. Next step: Test two CPU ranks, rank-zero events, cancellation and failure propagation.
- **Experiment tracking / sweeps — Partially implemented**. Parsers, sweeps and MLflow interchange exist; runner/event production unverified. Evidence: `../NELO-Studio/lib/training/events.ts:1`; `../NELO-Studio/lib/training/sweeps.ts:1`; `../NELO-Studio/lib/training/mlflow.ts:1`. Next step: Restore runner and test metadata/metric/event persistence.

### Generative AI

- **Local inference — Partially implemented**. Actual external Ollama-compatible transport; no local model process tested. Evidence: `../NELO-Studio/app/api/nil/route.ts:2`; `nil-sdk/nil/providers/openai.py:6`. Next step: Probe model readiness and forbid silent identity changes.
- **Model discovery / management — Partially implemented**. Discovery and management paths; legacy registry fakes installs. Evidence: `../NELO-Studio/lib/modelCatalog.ts:1`; `../NELO-Studio/app/api/compute/models/route.ts:1`. Next step: Unify catalog, runtime state and digest-verified installation.
- **Fine-tuning / LoRA — Not implemented**. Strategy evolution changes agent artifacts; no foundation-model parameter fine-tuning pipeline found. Evidence: `nil-sdk/nil/evolution/pipeline.py:1`. Next step: Integrate Transformers/PEFT after reproducible training exists.
- **Semantic embeddings — UI-only / placeholder**. SHA-256 vectors are not semantic model embeddings. Evidence: `nil-sdk/nil/tools/embeddings.py:23`. Next step: Use explicit embedding provider, dimensions and version metadata.
- **RAG — Partially implemented**. Sparse/dense index and retrieval injection exist; no verified document ingestion-to-grounded-answer path. Evidence: `nil-sdk/nil/cognition/index.py:108`; `nil-sdk/nil/cognition/plugins.py:87`. Next step: Add chunking, provenance, embedding backend and retrieval evaluation.
- **Structured generation — Partially implemented**. Tool schemas and critic checks; generic type schema defect; no strict provider response schema contract. Evidence: `nil-sdk/nil/tools/base.py:63`; `nil-sdk/nil/kernel/graph/critic.py:150`. Next step: Use JSON Schema/Pydantic validation with constrained outputs.

### AI Agents

- **Tool argument validation — Broken**. T1 accepts wrong integer type; T2 encodes List[float] as string. Evidence: `nil-sdk/nil/tools/base.py:93`; `nil-sdk/nil/tools/base.py:12`. Next step: Generate nested schemas using type introspection and validate before dispatch.
- **Multi-agent systems — Partially implemented**. Separate orchestration modules; Agent enable_multi_agent parameter not consumed. Evidence: `nil-sdk/nil/agents/orchestrator.py:1`; `nil-sdk/nil/agent.py:46`. Next step: Wire explicit orchestrator or remove unsupported flag.
- **Sandbox / permission enforcement — Partially implemented**. Studio has approval and resource policies; NIL directly invokes tools in-process. Evidence: `../NELO-Studio/lib/agent/approvals.ts:1`; `../NELO-Studio/lib/agent/taskExecutionPolicy.ts:1`; `nil-sdk/nil/kernel/tool_manager.py:79`. Next step: Apply one policy engine to every action entry point.
- **Evaluation / evolution — Partially implemented**. Evaluation and strategy modules; default Domain.evaluate returns True without success signal. Evidence: `nil-sdk/nil/evolution/evaluator.py:1`; `nil-sdk/nil/domain.py:59`. Next step: Require evidence-based evaluators and held-out promotion tests.

### Multimodal AI

- **Image input to cloud models — Partially implemented**. Gemini fallback image branch and CLI staging; daemon-first branch omits image. Evidence: `../NELO-Studio/app/api/nil/route.ts:185`; `../NELO-Studio/app/api/cli-bridge/route.ts:126`. Next step: Normalize image parts through provider interface.
- **Local vision model — Broken**. AST parse fails at line 8: incomplete import; no model implementation. Evidence: `nil-sdk/nil/open_models/nelo_vision_mini.py:1`. Next step: Replace with optional tested inference adapter.
- **Speech recognition / synthesis — UI-only / placeholder**. Legacy compute module is simulated; no verified transcription/synthesis path in main IDE. Evidence: `../NELO-Studio/studio/packages/compute-engine/nelo_compute/runtime.py:1`. Next step: Integrate explicit speech backend and fixtures.
- **Video inference — Not implemented**. No executable frame-to-model streaming pipeline found. Evidence: `../NELO-Studio/app/api/data/media/route.ts:1`. Next step: Use timestamped frame adapter and bounded queue.
- **OCR — Not implemented**. No OCR pipeline found in inspected runtime/API paths. Evidence: `nil-sdk/nil/open_models/nelo_vision_mini.py:1`. Next step: Add optional OCR backend with text/box contract.
- **Streaming multimodal inference — Not implemented**. Main chat request path uses non-streaming provider requests; media transport alone is insufficient. Evidence: `../NELO-Studio/app/api/nil/route.ts:1`. Next step: Add cancellation, timestamps, backpressure and stream protocol.

### Optimization

- **ONNX inference — Broken**. Provider probe exists but nelo_train.serve missing; test fails at import. Evidence: `../NELO-Studio/lib/serving/server.ts:23`; `../NELO-Studio/tests/python/test_onnx_serving.py:1`. Next step: Ship loader and test tiny ONNX model on CPU.
- **Quantization — UI-only / placeholder**. Catalog labels are not executable quantization; no calibrated export pipeline found. Evidence: `../NELO-Studio/studio/packages/compute-engine/nelo_compute/model_registry.py:19`. Next step: Integrate ONNX Runtime or framework quantizers.
- **Pruning — Not implemented**. No parameter pruning pipeline found. Evidence: `nil-sdk/nil/optimization/model_router.py:1`. Next step: Add only with accuracy and latency acceptance thresholds.
- **GPU / accelerators — Partially implemented**. Container GPU flags and ONNX provider detection; visual worker is flex CPU. No accelerator verified. Evidence: `../NELO-Studio/lib/serving/server.ts:23`; `../NELO-Studio/lib/training/runs.ts:46`; `../NELO-Studio/crates/nelo-ml-worker/src/main.rs:35`. Next step: Publish capability matrix based on actual runtime probes.
- **Dynamic batching — Not implemented**. Concurrent HTTP benchmark clients do not implement server batching. Evidence: `../NELO-Studio/lib/serving/server.ts:125`. Next step: Use established serving scheduler where justified.
- **Agent cost routing / chain compression — UI-only / placeholder**. Explicit stub route and empty analysis; cache/budget helpers do not establish integration. Evidence: `nil-sdk/nil/optimization/model_router.py:28`; `nil-sdk/nil/optimization/chain_compressor.py:22`. Next step: Wire measurable routing policy or remove advertised feature.

### MLOps

- **Policy/model version registry — Partially implemented**. Local policy versions and evidence gates; legacy model registry has fake installs. Evidence: `../NELO-Studio/lib/training/policies.ts:1`; `../NELO-Studio/app/api/training/policies/route.ts:1`. Next step: Use immutable artifacts and MLflow registry adapter.
- **CI/CD — Partially implemented**. Build/check and packaging paths exist; no passing release pipeline reproduced. Evidence: `../NELO-Studio/package.json:39`; `../NELO-Studio/packaging/README.md:1`. Next step: Require wheel/import, IDE build, contract and CPU lifecycle checks.
- **Deployment / serving — Broken**. Process/health logic exists but default Python serving module absent. Evidence: `../NELO-Studio/lib/serving/server.ts:75`. Next step: Restore versioned serving runtime and isolated model loader.
- **Observability — Partially implemented**. Local training and agent telemetry modules; no unified production metrics/traces integration verified. Evidence: `../NELO-Studio/lib/training/events.ts:1`; `nil-sdk/nil/observability/events.py:1`. Next step: Add OpenTelemetry and redacted structured logs.
- **Rollback — Partially implemented**. Serving stops previous process before candidate health; strategy history is separate. Evidence: `../NELO-Studio/lib/serving/server.ts:81`; `nil-sdk/nil/memory/strategy_version.py:1`. Next step: Health-gate replacement and restore last healthy model automatically.
- **Drift detection — Partially implemented**. Agent trace performance drift code; not deployed model feature/data drift monitoring. Evidence: `nil-sdk/nil/evolution/drift_detector.py:1`. Next step: Separate online model drift and agent strategy regression.

### Security

- **Local API access boundary — Partially implemented**. Host/Origin guard and loopback binding present; local clients trusted, not per-request authentication. Evidence: `../NELO-Studio/proxy.ts:18`; `nil_server.py:386`. Next step: Use daemon capability token and explicit local trust model.
- **Legacy daemon access boundary — Broken**. 0.0.0.0, wildcard CORS and no auth; fabricated state exposed. Evidence: `nil-sdk/nil/server.py:149`. Next step: Retire or bind authenticated demo-only service.
- **Plan-mode side-effect prevention — Broken**. Plan instruction applies to fallback prompt; daemon receives no mode and may execute robot tools first. Evidence: `../NELO-Studio/app/api/nil/route.ts:61`; `nil_server.py:107`. Next step: Enforce read-only tool policy server-side before model invocation.
- **Prompt injection defenses — Partially implemented**. Some approval gates; retrieved/tool/user context lacks common trust and effect policy across NIL. Evidence: `../NELO-Studio/lib/agent/tools.ts:146`; `nil-sdk/nil/kernel/runner.py:154`. Next step: Use typed untrusted context and deny-by-default capabilities.
- **Audit logs — Partially implemented**. Local JSONL/SQLite logs; no immutability, retention or tamper evidence verified. Evidence: `../NELO-Studio/lib/agent/approvals.ts:80`; `nil-sdk/nil/memory/store.py:56`. Next step: Redact secrets, correlate requests and add integrity/retention controls.
- **Supply-chain integrity — Partially implemented**. 3 packaging tests pass including checksum plumbing/path traversal; Python deps unpinned and NumPy undeclared. Evidence: `../NELO-Studio/scripts/package-native-language-servers.py:1`; `pyproject.toml:1`. Next step: Lock dependencies, verify runtime archives, inventory licenses/SBOM.
- **Model provenance — Partially implemented**. Hashing intent and policy checks; artifact tests blocked; legacy fake model installation. Evidence: `../NELO-Studio/lib/training/policies.ts:1`; `../NELO-Studio/tests/python/test_training_artifacts.py:1`. Next step: Bind source/dataset/base-model hashes and approvals to artifact digest.

### Robotics AI

- **ROS 2 integration — Partially implemented**. Environment/compatibility and deployment integration code; no ROS graph exercised. Evidence: `../NELO-Studio/lib/domain/ros2/environment.ts:1`; `../NELO-Studio/app/api/robotics/deployment/route.ts:1`. Next step: Test with isolated ROS 2 simulator and supported distro matrix.
- **Sensor interfaces — Partially implemented**. Sensor buffer and telemetry mapping; no sensor driver or live acquisition verified. Evidence: `nil-sdk/nil/robotics/sensorium.py:54`; `../NELO-Studio/lib/domain/nil/bridge.ts:28`. Next step: Add timestamp/frame/unit contracts and stale-data rejection.
- **Edge inference — Partially implemented**. Policy artifacts and external bridge; no target-device inference verified. Evidence: `../NELO-Studio/lib/training/policies.ts:1`; `nil-sdk/nil/robotics/stacc_bridge.py:1`. Next step: Validate model/runtime compatibility in emulated target before deployment.
- **Perception — UI-only / placeholder**. World-object lookup and fixed legacy detection are not real vision perception. Evidence: `nil-sdk/nil/robotics/domains.py:30`; `../NELO-Studio/studio/packages/compute-engine/nelo_compute/runtime.py:86`. Next step: Integrate calibrated camera/model backend with recorded-data tests.
- **Simulation integration — Partially implemented**. Actual HTTP calls to external STACC; no daemon or physics simulation started. Evidence: `nil_server.py:53`; `../NELO-Studio/lib/domain/nil/bridge.ts:90`. Next step: Test simulated action-to-observation roundtrip with safe limits.
- **Hardware deployment — Partially implemented**. Approval/compatibility gates exist; physical deployment intentionally untested. Evidence: `../NELO-Studio/lib/domain/nil/governance.ts:6`; `../NELO-Studio/app/api/robotics/deployment/route.ts:1`. Next step: Use signed evidence-bound approvals and hardware-in-loop acceptance.
- **Safety filter — Broken**. Position/velocity clamp only; max_tau and d_safe unused; no finite/dimension/dt validation. Evidence: `nil-sdk/nil/robotics/safety_gatekeeper.py:30`. Next step: Fail closed on invalid input; collision/torque checks belong in control boundary.
- **IK bridge failure reporting — Broken**. Missing SDK or exceptions return zero joint vector without typed failure. Evidence: `nil-sdk/nil/robotics/stacc_bridge.py:55`. Next step: Return explicit unavailable/solve failure, never actionable fallback pose.
- **Safe autonomy command boundary — Partially implemented**. Validated Studio route exists, root daemon bypasses it via legacy endpoints; backend safeguards not established here. Evidence: `../NELO-Studio/lib/domain/nil/governance.ts:96`; `nil_server.py:87`. Next step: Unify tool dispatch through authoritative safety/permission service.

## Dead, inert and disconnected code distinctions

`Agent.enable_multi_agent` and `rl_config` are inert arguments, not implemented switches. The router/compressor are explicit placeholders. The legacy nil.server is independently runnable and fabricated; it is not the endpoint targeted by the inspected primary IDE route. Legacy nelo_compute is imported by studio/bin/nelo-studio and control-plane code, so it cannot be called globally dead. Graph execution, RL and observability utilities have callers/tests but no verified end-to-end current-IDE integration. An exhaustive reachability proof across dynamic plugin entry points was not performed. No source was deleted or changed.
