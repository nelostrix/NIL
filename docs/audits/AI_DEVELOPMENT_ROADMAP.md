# AI Development Roadmap

Audit date: 2026-11-14. Scope: NIL at `ed7f987ced82bbb59c8373c1d9bd3d9dd7cd8563` and adjacent NELO Studio IDE at `e517c6e02cd1e8bf59fccaa8c947143cae67537c`. Reports reside in NIL because that is the writable workspace. NELO Studio **is the IDE**; NIL is an agent SDK/backend, not the complete IDE. Existing audit documents were not treated as test evidence. No existing source code was modified.

Evidence is local source inspection and explicitly listed safe probes. No external API requests, dependency downloads, model downloads, cloud jobs, deployments or hardware access were performed. Optional installed runtimes and release bundles may differ from these checkouts; their functionality is not assumed.

## Delivery rules

Estimates are engineering effort ranges for a prepared team, not calendar promises; add design/review and platform variability. S ≤2 days, M 3–7 days, L 1–3 weeks, XL ≥3 weeks. Cross-cutting tasks may exceed the nominal band as shown. Do not add these estimates mechanically: dependencies, parallel work and existing external runtime availability change effort. No task authorizes cloud spend, model downloads, deployment or hardware access. All proposed APIs are recommendations, not claims of existing code.

P0 security/safety tasks block any production rollout. P1 stabilizes core AI/IDE behavior. P2 makes training/evaluation/deployment reproducible. P3 adds advanced agents, multimodal and robotics AI. P4 is optional experimentation and optimization. Restore missing runtime packages before treating existing UI workflows as operational. Keep proven frameworks and serializers; build thin integration layers rather than new training engines.

## R01 — P0: Enforce plan/read-only policy on every action route

- Implementation approach: Move mode/effect policy into daemon and dispatcher; reject mutating tools in plan mode, including fallback actions; bind policy to session rather than prompt.
- Dependencies: None.
- Acceptance criteria: A plan request cannot call motion, IK-with-motion, gain changes, training start or write tools; normal approved actions remain possible.
- Estimated complexity: M (3–5 engineer-days).
- Validation tests: Mock provider asks for each forbidden tool; assert zero backend calls through IDE and direct daemon entry points.

## R02 — P0: Repair physical command validation and failure semantics

- Implementation approach: Validate finite vectors, expected DOF, dt/state limits; remove zero-pose error fallback; enforce torque/collision checks at authoritative STACC boundary and separate simulation/hardware modes.
- Dependencies: R01; external STACC safety contract.
- Acceptance criteria: Invalid/stale/NaN/oversized commands fail closed; IK failure cannot become motion; no unimplemented safety guarantees.
- Estimated complexity: L (2–4 engineer-weeks).
- Validation tests: Property/boundary tests; recorded collision/torque cases in simulator; no physical execution needed.

## R03 — P0: Unify effect authorization and local daemon security

- Implementation approach: Route NIL tools through typed action policy; add daemon capability token and request size/schema limits; isolate legacy demo server; redact traces and keys.
- Dependencies: R01.
- Acceptance criteria: All action entry points enforce same policy; unrelated browser/local unauthorized client is denied; demo cannot report production success.
- Estimated complexity: L (1–2 weeks).
- Validation tests: Route coverage matrix, denied action probes, malformed/oversized HTTP bodies, origin/token tests and secret canaries.

## R04 — P1: Make packaging and public SDK reproducible

- Implementation approach: Consolidate pyprojects; lazy robotics extras or declare NumPy; replace shared defaults; support documented Python versions and correct Docker script command.
- Dependencies: None.
- Acceptance criteria: Fresh wheel imports core without optional extras; two default agents own distinct memory; Docker command exists; requirements use scikit-learn.
- Estimated complexity: M (3–5 days).
- Validation tests: Build/install wheel offline from prepared cache; public import/close tests; independent agent state test; container smoke in CI.

## R05 — P1: Fix tool schema and runtime argument contracts

- Implementation approach: Use JSON Schema/Pydantic and typing-origin support for lists, optional, nested objects; reject invalid arguments/unknown fields consistently.
- Dependencies: R04.
- Acceptance criteria: List[float] becomes array/number; wrong-type and unrecognized arguments fail before callable entry.
- Estimated complexity: M (3–5 days).
- Validation tests: Golden schema fixtures and invalid argument cases for robot, plugin and provider tools.

## R06 — P1: Unify provider and conversation contracts

- Implementation approach: Introduce typed provider Protocol/capability descriptors; immutable client configuration; carry history/images/mode/context consistently; report actual fallback identity and errors.
- Dependencies: R04,R05.
- Acceptance criteria: No secret process-global mutation; correct selected provider/model and explicit fallback disclosure; no stub execution in production.
- Estimated complexity: L (1–2 weeks).
- Validation tests: Mock all three providers and Ollama tool turns; multi-session key isolation; image/history preservation; fallback identity tests.

## R07 — P1: Add cancellable supervised execution

- Implementation approach: Enforce configured timeouts/retries/budgets through provider and tool workers; bound concurrency and request bodies; retain responsive health endpoint.
- Dependencies: R03,R06.
- Acceptance criteria: Timed-out operations cannot keep effectful work running; cancelled work is classified; health stays responsive.
- Estimated complexity: L (1–2 weeks).
- Validation tests: Hung callable/provider fault injection, worker kill/restart, retry idempotency and concurrent health checks.

## R08 — P1: Unify SDK/IDE memory and observable state

- Implementation approach: Canonical read API for traces/skills/strategies, explicit directory-vs-file storage, schema migrations, persisted counters and session ownership.
- Dependencies: R04,R06.
- Acceptance criteria: IDE skill counts equal actual SDK learned artifacts; no seeded/fabricated metrics; independent sessions do not leak memory.
- Estimated complexity: M (4–7 days).
- Validation tests: Learn/read/reopen integration, migration and corrupted-store fixtures; compare UI/API counts.

## R09 — P1: Validate IDE development tools end to end

- Implementation approach: Exercise Monaco workspace, native LSP, DAP, PTY, uv and supported plugin APIs against packaged binaries; keep explicit availability states.
- Dependencies: R04; prepared Node/Tauri toolchain.
- Acceptance criteria: Edit/save, diagnostics, breakpoint/step, terminal resize/cancel and environment selection work on supported OS; unsupported tools show unavailable.
- Estimated complexity: L (2–3 weeks).
- Validation tests: Desktop automation and process lifecycle tests on Linux/macOS/Windows; realpath/symlink confinement fixtures.

## R10 — P2: Restore training/runtime distribution

- Implementation approach: Publish nelo_train, notebook kernel and table helpers as versioned source/wheels; bootstrap pinned runtime explicitly rather than relying on ignored local folder.
- Dependencies: R04.
- Acceptance criteria: Current artifact/ONNX tests import; clean checkout can resolve installed runtime; notebook and tables have documented optional dependencies.
- Estimated complexity: L (1–2 weeks).
- Validation tests: Fresh environment artifact, kernel execute/interrupt, table profile and serving helper imports.

## R11 — P2: Integrate framework lifecycle adapters

- Implementation approach: Wrap PyTorch, TensorFlow and sklearn for metrics/artifacts/checkpoints; use native serializers, optimizer/RNG state and actual tiny CPU model training.
- Dependencies: R08,R10.
- Acceptance criteria: Each framework produces reproducible artifact and metrics; resume preserves training semantics; visual export produces valid model code.
- Estimated complexity: L (2–3 weeks).
- Validation tests: One tiny fit/forward/backward roundtrip per framework; resumed/uninterrupted comparison; Burn/PyTorch shape/output equivalence.

## R12 — P2: Complete datasets and lineage

- Implementation approach: Integrate durable HF/local import, Arrow/DuckDB validation/preprocessing and DVC or equivalent immutable versions; add explicit annotation backend and streaming checkpoints.
- Dependencies: R08,R10.
- Acceptance criteria: Dataset bytes can be restored by digest; transforms, labels and runs link to exact versions; interrupted stream resumes without duplication.
- Estimated complexity: XL (3–5 weeks).
- Validation tests: Malformed/schema/leakage fixtures, deterministic transform test, snapshot restore, label export and streaming restart tests.

## R13 — P2: Validate distributed/remote jobs and experiments

- Implementation approach: Use torchrun for local ranks and established cluster/container scheduler; retain rank-zero metrics, environment manifests and MLflow interchange.
- Dependencies: R07,R10,R11.
- Acceptance criteria: Two CPU ranks complete; rank failure cancels peers; remote credentials scoped; sweeps and metrics survive restart.
- Estimated complexity: L (2–3 weeks).
- Validation tests: CPU gloo two-rank test, mocked scheduler lifecycle, MLflow export/import/push with fixtures; no paid jobs.

## R14 — P2: Implement trustworthy registry and serving rollback

- Implementation approach: Use immutable model artifacts and MLflow registry adapter; remove fake installer state; start candidate isolated serving backend before atomically switching routing.
- Dependencies: R03,R10,R11.
- Acceptance criteria: No text placeholder is treated as weights; digest verified; candidate failure leaves old model healthy; approvals bind to artifact.
- Estimated complexity: L (2–3 weeks).
- Validation tests: Tiny CPU ONNX/Torch serving, corrupted model rejection, readiness failure/rollback, worker identity and concurrent requests.

## R15 — P2: Add release evidence and production observability

- Implementation approach: Introduce CI gates for core wheel, IDE build, runtime artifacts, contracts and CPU lifecycle; OpenTelemetry correlation, redaction, retention and model drift separately from agent drift.
- Dependencies: R07,R08,R11,R14.
- Acceptance criteria: Release provenance and dependency locks available; one trace spans UI/provider/tool; canary secrets absent; drift alert tied to reproducible data.
- Estimated complexity: L (2–3 weeks).
- Validation tests: End-to-end trace assertion, dependency/license inventory, redaction tests, known shifted dataset and rollback rehearsal.

## R16 — P3: Make RAG and structured generation meaningful

- Implementation approach: Replace hash embeddings with explicit embedding adapter; chunk/provenance store and retrieval evaluation; enforce output schemas; use established retrieval library where useful.
- Dependencies: R05,R06,R08,R12.
- Acceptance criteria: Semantic retrieval beats lexical baseline on held-out tasks; citations resolve to stored chunks; invalid structured output rejected.
- Estimated complexity: L (1–2 weeks).
- Validation tests: Small offline corpus relevance/groundedness tests, dimension/version mismatch, injection-in-document and JSON validation cases.

## R17 — P3: Validate multi-agent and learning integrations

- Implementation approach: Wire or remove inert enable_multi_agent/rl_config flags; use explicit orchestrator and held-out strategy evaluations with approval/promotion/rollback.
- Dependencies: R03,R07,R08,R15.
- Acceptance criteria: Flags change documented behavior; no unsupported neural fine-tuning claim; regression cannot auto-promote.
- Estimated complexity: L (2–3 weeks).
- Validation tests: Mocked multi-agent coordination, budget/deadlock/cancel tests, RL config propagation and negative promotion fixtures.

## R18 — P3: Implement real multimodal adapters

- Implementation approach: Replace broken vision module; add optional established vision, OCR and speech engines plus timestamped video/stream adapters; distinguish media viewing from inference.
- Dependencies: R06,R07,R10.
- Acceptance criteria: Known image/audio fixtures produce model outputs; model lifecycle is real; stream has bounded memory, ordering and cancellation.
- Estimated complexity: XL (3–5 weeks).
- Validation tests: Tiny local fixtures with prepared weights, speech/OCR expected outputs, frame backpressure and missing-model tests.

## R19 — P3: Complete robotics simulation and edge contracts

- Implementation approach: Decorate embodied tools; connect sensor/world models to timestamped observations; enforce units/frames; run recorded perception and simulated action roundtrips before evidence-bound HIL/hardware.
- Dependencies: R02,R03,R05,R14,R18; supported STACC/ROS 2 environment.
- Acceptance criteria: Simulation action returns measured result; stale sensors refuse autonomy; hardware requires artifact-specific verified acceptance and arming.
- Estimated complexity: XL (4–6 weeks).
- Validation tests: Mocked robot transport, recorded sensor/perception cases, isolated ROS 2 graph and simulator acceptance; hardware test separately authorized.

## R20 — P4: Add measured optimization and optional fine-tuning

- Implementation approach: Integrate ORT quantization and supported accelerators, established batching server and Transformers/PEFT where justified; remove empty routing/compression promises.
- Dependencies: R11,R14,R15,R16.
- Acceptance criteria: Accuracy/latency/memory gates exist; provider claims match actual hardware; optional training stays within declared budget.
- Estimated complexity: XL (3–6 weeks).
- Validation tests: Tiny CPU quantization/numerical comparison, routing baseline, batch fairness and small fine-tuning smoke; GPU acceptance separately provisioned.

## Exit gates

1. P0 closure: read-only mode demonstrably cannot mutate, command validation fails closed, zero-pose failures and fabricated production status are eliminated.
2. Developer alpha: clean SDK import, selected provider identity, typed tools and IDE editing/debugging/terminal paths verified in supported environments.
3. AI SDK beta: one complete tiny CPU data→train→resume→registry→serve→rollback lifecycle from clean checkout, with lineage and traces.
4. Production candidate: release CI and cross-platform evidence, resource/isolation failure tests, artifact integrity and operational rollback drill pass. Hardware readiness remains a separate evidence gate.

The 79-row feature matrix is the acceptance backlog; R09 covers IDE tools, R12 data, R11/R13 ML, R06/R16/R20 generative AI, R17 agents, R18 multimodal, R20 optimization, R14/R15 MLOps, R01–R08/R15 security and R19 robotics. Prioritize unsupported capability disclosure even before implementation so users can trust availability and telemetry.
