# NIL AI framework master TODO

Updated: 2026-11-24. This is the active backlog for the 32-area capability map.
It includes extensions to existing functionality as well as new work. An unchecked
area is not a claim that NIL has no implementation there. Audit existing modules,
contracts and evidence before adding another implementation or dependency.

[PHASES.md](PHASES.md) retains the concrete implementation/target acceptance gates
for the existing phases. The earlier checklist and dated reports are preserved in
the historical section below; their old open boxes are not the current master status.

## Completion contract and architecture

- [x] Publish a versioned supported-workflow matrix: API, input/output schema,
  native/integration status, backend versions, OS/device, limits and evidence.
- [x] Define the stable SDK release boundary against that matrix, including the
  intentionally unsupported combinations. Do not infer worldwide AI coverage from
  constructor counts, adapters or the length of this backlog.
- [x] Label each declared capability as implemented, validated, experimental,
  integration-only, target-blocked or unsupported; distinguish these in CLI/help.
- [x] Define independent-engine milestones separately from unified SDK coverage.
  Optional backend APIs do not imply NIL-owned numerical execution.

Evidence for these bounded foundation deliverables: [workflow inventory](../WORKFLOWS.md),
[SDK contracts/release criteria](../SDK_CONTRACTS.md), `nil workflows --json`,
`nil help coverage` and `nil help contracts`. Stable release criteria are defined;
NIL is still alpha. The inventory has 30 scoped records, not 32 completed areas.

Permanent constraints: base installation adds only nil-sdk; imports start no model
loads/downloads, compilation, processes or services. Backends, accelerators, model
weights and specialist modules are explicit opt-ins. Keep shallow public modules,
shared contracts and private implementation helpers. Each public module/model
needs offline help.md. Preserve measured low-resource budgets and existing APIs.

For every task below, close the checkbox only after publishing the supported API,
runnable example/help, meaningful correctness/learning tests, persistence/recovery
checks where relevant, resource measurements, exact tested versions/targets and
known failures. Record commands and raw evidence. A skip is not target acceptance;
compilation or an exposed backend symbol alone does not establish workflow coverage.

## Existing validated scope to preserve

- [x] NIL-only offline wheel installation, passive imports and bundled help:
  [installation contract](../INSTALLATION.md).
- [x] Bounded nonempty rank0–8 CPU float64 tensors/autograd, ND SGD/Adam,
  COO/CSR and explicit Linux C matmul/VJP: [native scope](../NATIVE_ENGINE.md).
- [x] Documented optional classical/specialized workflows and persistence:
  [classical guide](../CLASSICAL_ML.md), [specialized limits](../CLASSICAL_SPECIALIZED.md).
- [x] Scoped CPU neural, offline/online RL integrations, masked parallel collection,
  centralized-value AWR and declared simulator continuation: [DL/RL](../NEURAL_RL.md).
- [x] Pinned134.5M pretrained CPU inference, LoRA/adapter reload and INT8 weight
  storage within actual2GiB/no-swap limits: [pretrained evidence](../PRETRAINED_CPU.md).
- [x] Base Linux/macOS/Windows Python3.10–3.12 and declared minimum/current backend
  profiles: [compatibility evidence](../BACKEND_COMPATIBILITY.md).
- [x] Current-host one-CPU/1GiB/no-swap small workloads; physical old-PC acceptance
  remains open: [hardware evidence](../HARDWARE_ACCEPTANCE.md).
- [x] Managed single-tenant CPU TLS service reference with bounded execution and
  lifecycle checks; public deployment remains open: [deployment scope](../DEPLOYMENT.md).

These checks certify their documented subsets, not the full categories below.

## Execution order for the expanded roadmap

| Wave | Priority and area IDs | Exit gate |
| --- | --- | --- |
| A | Stable SDK/data/evaluation/release contracts: 1, 6, 11, 12, 26, 28, 30–32 | Published coverage matrix, stable contracts, reproducible installed examples and compatibility policy |
| B | Independent execution/training/scale: 2–5, 9, 10, 24, 25 | Measured CPU improvements and actual acceptance for each declared accelerator/distributed target |
| C | Broader common AI workflows: 7, 8, 13–20 | Real datasets/environments, held-out evaluation, artifacts and declared resource/quality limits |
| D | Production serving and trust boundaries: 27, 29 | Acceptance on a chosen deployment with authorization, failure recovery and workload SLO evidence |
| E | Specialist extensions: 21–23 and optional domains | Independently scoped plugins with validated user workloads; no mandatory core dependencies |

Current user priority: implement framework foundations first, then image workflows.
The first foundation and classification deliverables are complete; this does not
close every Wave A gate or all computer-vision tasks.

Within each wave, inventory and repair existing paths before expansion. Hardware
or deployment gates can remain blocked while unrelated software work proceeds.
New specialist rows are optional extension scope, not prerequisites for every SDK
release. Revisit priorities using actual supported user workloads.

## Capability backlog — all 32 areas

### 1. Consistent public API — Wave A

- [x] Add backend-neutral structural dataset/predictor/result protocols and
  schema-1 metadata for native, classical, neural and online/offline/centralized
  RL outcomes. Accept neural TrainingResult directly in predict/evaluate and
  verify mode restoration, errors and real outcome families.
- [x] Ship passive workflow discovery, metadata-only availability, CLI and offline
  help; verify NIL-only packaging and no backend/process/network work during discovery.

- [ ] Audit model/data/result/config/error contracts across fit/train/predict,
  evaluate, persistence and serving; document intentional task-specific differences.
- [ ] Stabilize callbacks, cancellation, progress and capability reporting; validate
  compatible behavior across native execution and declared optional backends.
- [ ] Define extension/registration interfaces, type hints and compatibility tests
  so adding a model does not require edits throughout the framework.

### 2. Numerical foundation — Wave B

- [ ] Expand beyond the bounded float64 subset: declared float32/float16/bfloat16,
  integer/bool semantics, conversions, promotion, overflow and numerical guarantees.
- [ ] Specify/test zero-sized tensors, advanced indexing, concatenation/stacking,
  additional reductions and linear algebra; retain existing scalar/vector behavior.
- [ ] Expand sparse formats/operators only with explicit storage, coalescing,
  gradient and densification budgets; document owned-storage/view rules.

### 3. Automatic differentiation — Wave B

- [ ] Define a supported custom differentiable-operator interface and independent
  gradient-checking utilities rather than depending on private graph internals.
- [ ] Add declared higher-order gradients, vector/Jacobian products and Hessian
  workflows with finite-difference/reference tests and memory limits.
- [ ] Validate accumulation, retained/reused graphs and mutation behavior across
  every newly supported operator/dtype/device combination.

### 4. Efficient execution — Wave B

- [ ] Expand optimized CPU coverage beyond explicit matmul/VJP: elementwise work,
  reductions, convolution, normalization and attention where benchmarks justify it.
- [ ] Define graph capture/compilation, fusion, reusable plans and memory planning
  with explicit fallbacks, cache ownership and cold/warm compilation budgets.
- [ ] Track import latency, end-to-end wrapper overhead, throughput and peak memory;
  publish wins and regressions across small and representative larger workloads.

### 5. Hardware support — Wave B

- [ ] Define and implement selected accelerator backends/device transfers,
  synchronization, allocation ownership and unsupported-operation errors.
- [ ] Accept real CUDA/other chosen devices, precision combinations and process-tree
  cleanup; validate hard GPU containment separately from allocator preferences.
- [ ] Run actual older-PC/OS/ISA tests and publish usable workload limits; current
  resource-constrained modern-host results remain a separate evidence class.

### 6. Data system — Wave A

- [ ] Audit typed datasets, schemas, preprocessing, transforms and provenance;
  publish modality-specific data contracts and reusable validation rules.
- [ ] Extend memory-bounded streaming, caching, batching, shuffling and parallel
  loading to declared out-of-RAM datasets with corruption/backpressure tests.
- [ ] Broaden exact iterator/sampler/offset resume and schema migrations; explicitly
  scope prefetch, external broker and model-state transaction boundaries.

### 7. Classical machine learning — Wave C

- [ ] Inventory actual end-to-end support for linear/generalized linear, tree/
  ensemble/boosting, kernel, neighbor and probabilistic estimator families.
- [ ] Audit clustering, mixtures, dimensionality/manifold reduction, feature
  selection and anomaly workflows against the shared SDK contract.
- [ ] Close matrix gaps in weighting, sparse/multioutput data, preprocessing,
  streaming and persistence; label integrations separately from native algorithms.

### 8. Specialized machine learning — Wave C

- [ ] Extend forecasting to declared multivariate, hierarchical, irregular-time,
  state-space/ARIMA and covariate cases with leakage-safe backtests.
- [ ] Expand collaborative/neural recommendation and ranking beyond current scoped
  baselines; validate retrieval, cold starts and feedback assumptions.
- [ ] Scope advanced survival, spatial, imbalanced and online-learning extensions;
  publish assumptions and real-data validation for each added method.

### 9. Deep-learning building blocks — Wave B

- [ ] Map backend-provided versus NIL-owned dense/convolutional/recurrent/attention,
  embedding, normalization, graph and pooling layers.
- [ ] Expand independent layers, losses, initializers, optimizers, schedules and
  regularization with numerical gradients, serialization and measured kernels.
- [ ] Test supported combinations of shapes, masks, dtypes, stateful layers and
  custom modules; do not advertise an alias as independent implementation.

### 10. Training engine — Wave B

- [ ] Audit custom objectives, accumulation, clipping, mixed precision, callbacks,
  early stopping and reproducibility across all declared task families.
- [ ] Broaden resumable execution and recovery for additional data-loader,
  optimizer/scheduler and asynchronous job configurations.
- [ ] Validate real accelerator training, failure/cancellation and resource budgets;
  retain exact definitions of epoch/update/rollout continuation boundaries.

### 11. Model selection — Wave A

- [ ] Publish a common leakage-safe pipeline/search contract for grouped, temporal,
  nested and other declared validation strategies.
- [ ] Extend budgeted Bayesian/adaptive search, pruning, multi-objective selection
  and distributed tuning beyond the current supported sampler/host combinations.
- [ ] Verify deterministic recovery, fit accounting, failed-trial handling and
  preprocessing isolation under actual concurrent workers.

### 12. Evaluation and uncertainty — Wave A

- [ ] Inventory metrics for each supported task and add independent reference,
  weighting, threshold, aggregation and degenerate-input tests.
- [ ] Extend calibration, uncertainty, conformal and model-explanation workflows
  with documented statistical assumptions and leakage guards.
- [ ] Define robustness, drift, subgroup and multi-seed comparison reports;
  distinguish measured coverage from unsupported conditional/causal claims.

### 13. Language-model workflows — Wave C

- [ ] Expand pinned pretrained/tokenizer coverage to declared generation,
  embeddings, classification, extraction and summarization architectures.
- [ ] Broaden fine-tuning/adapters/preference optimization, long contexts and
  generation controls with matching checkpoint/tokenizer/processor recovery.
- [ ] Accept larger actual checkpoints on provisioned hardware, recording data,
  quality, memory, latency and numerical failures; preserve opt-in downloads.

### 14. Retrieval and RAG — Wave C

- [ ] Audit existing memory/retrieval components and define ingestion/chunking,
  embedding index, hybrid search and reranking interfaces.
- [ ] Add provenance, incremental updates, access-aware retrieval and index/schema
  migration with bounded storage and restart/recovery behavior.
- [ ] Publish retrieval and grounded-answer evaluation on declared datasets;
  expose index/backend requirements as explicit integrations.

### 15. Computer vision — Wave C

- [x] Add the local classification subset: bounded lazy image decoding, EXIF/RGB/
  alpha/resize semantics, shared train/test classes, CNN training, held-out metrics,
  JSON preprocessing/label manifest and checkpoint inference. Validate file failures,
  exact reload and real 1000/300 handwritten-digit split (89.33% held-out accuracy).
  [Image guide and measured limits](../VISION.md).

- [ ] Define end-to-end classification, detection, segmentation, tracking, OCR and
  image-retrieval contracts beyond existing construction/training examples.
- [ ] Add image transforms/augmentation, labels, pretrained weights/processors and
  task metrics with consistent artifact packaging.
- [ ] Validate representative real datasets, export and serving paths; scope image
  generation under the generative workflows rather than duplicate engines.

### 16. Speech and audio — Wave C

- [ ] Define waveform/sample-rate/channel, spectrogram, feature and alignment
  contracts; add bounded preprocessing and augmentation.
- [ ] Add declared recognition, synthesis, speaker and classification workflows
  with local/pinned models and task-specific metrics.
- [ ] Validate streaming/chunk boundaries, latency, resume and export on actual
  audio data and selected devices.

### 17. Video and multimodal AI — Wave C

- [ ] Define video sampling/timestamp, multimodal processor and aligned-label
  contracts with bounded decoding and data loading.
- [ ] Add declared image/text/audio/video input-output workflows and test absent,
  corrupt, variable-length and missing-modality behavior.
- [ ] Validate training, inference and processor/model artifact compatibility
  on representative tasks and documented hardware.

### 18. Generative modeling — Wave C

- [ ] Inventory autoregressive support and scope optional diffusion, autoencoder,
  adversarial and other generative-model families according to actual demand.
- [ ] Standardize conditioning, reproducible sampling, checkpoint/resume and
  model/processor packaging across declared families.
- [ ] Add meaningful quality/diversity, numerical and resource evaluation;
  distinguish functional generation from useful model quality.

### 19. Reinforcement learning — Wave C

- [ ] Audit the declared discrete/continuous, vector, recurrent/masked, replay and
  offline algorithm matrix before adding another overlapping implementation.
- [ ] Extend beyond discrete CQL/fixed offline centralized AWR where required:
  continuous offline RL, online centralized multi-agent and variable populations.
- [ ] Broaden exact simulator continuation only for explicit environment/wrapper/
  algorithm contracts and test real learning, recovery and multiple seeds.

### 20. Agents and planning — Wave C

- [ ] Audit existing tools, memory, agent state and planning/search components;
  align them with shared execution, configuration and artifact contracts.
- [ ] Standardize structured outputs, permissions, bounded retries/tool budgets,
  cancellation and deterministic trace/replay where supported.
- [ ] Add task-level evaluations and recovery tests with real tool/environment
  boundaries; distinguish stored traces from demonstrated task competence.

### 21. Graph learning — Wave E

- [ ] Define graph data, sparse storage, sampling and batching contracts before
  adding node/edge/graph prediction and link-prediction workflows.
- [ ] Add selected message-passing, heterogeneous-graph and embedding integrations
  with independent parity, leakage-safe splits and documented memory limits.
- [ ] Validate actual graph-learning datasets, artifacts and target backends.

### 22. Probabilistic and causal modeling — Wave E

- [ ] Inventory existing uncertainty/probabilistic utilities and scope Bayesian
  inference, probabilistic programs and causal graph extensions separately.
- [ ] Add selected treatment-effect/counterfactual workflows with explicit
  identification assumptions, diagnostics and appropriate reference tests.
- [ ] Publish numerical/statistical validation and failure limits rather than
  presenting predictive accuracy or attribution as causal evidence.

### 23. Other learning regimes — Wave E

- [ ] Scope self-supervised, semi-supervised, active, transfer and few-shot
  workflows using existing training/data abstractions where possible.
- [ ] Audit existing continual/evolution/memory work; add forgetting, transfer,
  stream-order and recovery evaluation for supported continual-learning methods.
- [ ] Add selective-labeling/data-efficiency comparisons on held-out workloads;
  report limitations and independently scoped optional dependencies.

### 24. Distributed and large-model systems — Wave B

- [ ] Accept real multi-GPU/multi-host data, tensor and pipeline parallel execution
  for explicitly chosen strategies and network topologies.
- [ ] Extend distributed checkpoints, sharding/offload and worker failure recovery
  beyond the current accepted CPU/same-world contracts.
- [ ] Publish measured communication, memory, scaling and resume behavior;
  GPU tests, driver/runtime versions and rank failures require actual targets.

### 25. Compression and inference optimization — Wave B

- [ ] Expand the model/device/fidelity matrix for existing dynamic and weight-only
  INT8; preserve failed configurations and caller ownership tests.
- [ ] Add selected static quantization/calibration, QAT, lower-bit formats,
  pruning and distillation only against declared model/backend contracts.
- [ ] Measure accuracy, serialized size, resident/peak memory and actual latency
  independently; storage reduction does not certify a compute speedup.

### 26. Model interoperability — Wave A

- [ ] Audit artifact schemas, safe/trusted serialization, processor/tokenizer
  packaging, architecture manifests and version compatibility.
- [ ] Expand tested import/export and ONNX cases, including dynamic shapes,
  recurrent/cache state and unsupported graph/operator diagnostics.
- [ ] Add format/version migration, integrity, partial-publication and roundtrip
  checks without silent data, dtype or model-semantic changes.

### 27. Serving — Wave D

- [ ] Preserve the validated managed single-tenant reference and define the chosen
  public/multi-tenant serving target before broadening its acceptance claim.
- [ ] Scope batch/online/streaming inference, batching, warmup, concurrency,
  deadlines, version routing and rollback with resource/model boundaries.
- [ ] Validate authorization/revocation, quotas, monitoring, recovery and actual
  workload reliability on the deployment host; local TLS tests are separate.

### 28. Experiment lifecycle — Wave A

- [ ] Audit runs, metrics, artifacts, lineage, registries and tracking integrations
  against a stable shared schema and clear local/remote ownership.
- [ ] Add reproducible environment manifests, comparison and model/data promotion
  with explicit retention, migration and failure-recovery policies.
- [ ] Test restart/reconciliation across the supported training, search and serving
  workflows without enabling unsolicited remote tracking or background services.

### 29. Security and privacy — Wave D

- [ ] Audit credential handling, artifact integrity, executable/trusted-code
  boundaries and access policies for the actual supported workflows.
- [ ] Add declared tenant isolation, authorization/revocation and audit behavior
  with deployment-specific tests and explicit process/device resource policies.
- [ ] Scope optional differential privacy/federated learning as separate modules;
  validate their threat models and numerical/privacy guarantees before advertising.

### 30. Developer experience — Wave A

- [ ] Audit all public APIs against offline help.md, runnable installed examples,
  reference documentation, typing and actionable error messages.
- [ ] Stabilize extension interfaces, diagnostics and migration guides with a
  shallow public module tree and shared utilities instead of duplicate wrappers.
- [ ] Add onboarding checks for representative low-resource, backend and device
  configurations; preserve deterministic opt-ins and readable capability reports.

### 31. Release engineering — Wave A

- [ ] Publish API stability/versioning/deprecation and supported-Python/backend
  policies; define the release candidate acceptance checklist.
- [ ] Extend reproducible build, source/wheel parity, install/upgrade/uninstall,
  dependency and platform-matrix checks for declared release targets.
- [ ] Publish versioned artifacts and evidence with rollback/migration guidance;
  verification-branch CI artifacts are not a permanent package-index release.

### 32. Validation infrastructure — Wave A

- [ ] Map each advertised workflow to numerical references, real learning tests,
  data provenance, multiple seeds and persistence/fault/resource acceptance.
- [ ] Expand reproducible performance regressions and actual hardware checks;
  keep CPU constraints, physical older machines and GPUs as distinct targets.
- [ ] Maintain a release dashboard of passing/failed/skipped/blocked combinations,
  with exact commands, versions and raw reports; never turn missing targets green.

## Optional specialist extension backlog — Wave E

- [ ] Scientific ML and differential-equation workflows: define solver/gradient,
  stability, physical constraints and reference-validation contracts.
- [ ] Robotics/control extensions: audit current modules; select actual simulators,
  control loops and hardware interfaces with bounded execution and safety ownership.
- [ ] Symbolic reasoning, rules, constraint solving and neural-symbolic interfaces:
  select concrete user tasks and formal correctness/reference checks.
- [ ] Evolutionary/gradient-free/multi-objective optimization: audit current
  evolutionary/search utilities and fill justified gaps without duplicate engines.
- [ ] Domain-specific/federated/privacy extensions: declare data, evaluation,
  governance and optional dependency boundaries per supported application.

## Current external acceptance blockers

- [ ] Provision actual supported GPU targets for operators, precision, distributed
  training/sharding and device-specific aggregate memory containment.
- [ ] Obtain an identified physical older PC; run the same published workloads
  and record CPU/ISA/RAM/OS provenance and backend-wrapper overhead.
- [ ] Select larger model/architecture/data/resource targets; the accepted134.5M
  CPU checkpoint does not certify billion-scale, long-context or all quantizers.
- [ ] Select and provision a public/multi-tenant deployment target and its
  authorization/revocation policy and workload reliability budgets.

Prior hardware availability was explicitly limited to this modern CPU host;
current low-resource tests and the accepted pretrained CPU case remain valid.
No new release, feature, hardware result or implementation is claimed by this
roadmap update. The first next task is Wave A's coverage/API audit and release scope.

<details>
<summary>Earlier checklist and dated validation records (historical)</summary>

The records below are preserved from the preceding implementation batches. Their
checkboxes and target availability describe those batches; use the master backlog
above and the current PHASES.md gates for present planning and status.

## Validated foundation (existing work)

- [x] Dependency-free base installation, offline help and lazy backend loading.
  Gate: clean offline wheel installation adds only nil-sdk; build verifier passes.
- [x] Classical preprocessing, weighted single/multilabel/multioutput evaluation,
  fold-local validation, resampling, bounded grid/random/TPE search, nested CV,
  conformal intervals/sets, supervised streaming checkpoints and JSONL snapshots.
  Gate: native extended/full suite on sklearn 1.4.2 and 1.9.1; CPU XGBoost 3.4.1
  pipeline roundtrips. Evidence: docs/CLASSICAL_ML.md and SDK_IMPLEMENTATION_REPORT.md.
- [x] Existing core/neural baseline. Gate: 876 passed, 12 optional skips in the
  torch environment; full classical environment 978 passed, four optional skips.
  These counts describe the preceding batch, not unimplemented features below.

## 1. Shared execution safety and recovery — Linux core validated

- [x] POSIX owned-worker execution with automatic wall timeout, cancellation,
  bounded logs, thread settings and durable terminal status.
  Gate: real process success/error/timeout/cancel tests; child cleanup and bounded
  output tests; source and installed-wheel execution checks.
- [x] Linux address-space caps with explicit platform capability reporting.
  Gate: a real allocation fails within the cap; parent remains usable; no claim
  of aggregate RSS/GPU memory containment.
- [x] Recover dead-worker records without signalling unrelated/reused PIDs and
  reconcile registered Run status. Gate: interrupted/finished outcome recovery,
  native classical and DL worker/checkpoint tests.
- [x] Independent watchdog after client death. Gate: actual client-kill tests
  with a detached stdlib supervisor, deadline enforcement, descendant cleanup,
  failed Run reconciliation and successful result recovery after client SIGKILL.
- [x] Last-valid-checkpoint recovery during interrupted DL publication.
  Gate: actual writer-process cancellation, commit fault injection and reload.
- [ ] Windows/macOS containment acceptance and aggregate process/GPU memory quotas.
  Gate: platform-specific process-tree and resource tests on actual target hosts.
- [ ] Representative scale/old-PC benchmarks with latency, peak RSS and workload
  definitions. Gate: reproducible benchmark report on documented hardware.

## 2. Deep learning correctness and training coverage

- [x] Audit/test gradient accumulation, uneven batches, cancellation and exact
  epoch-boundary optimizer/scheduler/RNG resume before expanding training.
- [x] AMP/autocast implementation with saved GradScaler state, unscale-before-clip
  and finite gradient guards; CPU bf16 training and exact epoch resume validated.
- [ ] CUDA fp16/bf16 acceptance on actual GPU hardware, including overflow/scale
  adaptation and resume parity. Gate: GPU tests, not CPU stand-ins.
- [x] Robust resumable checkpoint publication, environment/schema checks and
  rejection before state mutation for incompatible resumes.
- [x] Standard supervised DL task coverage: regression, multiclass, binary and
  multilabel; documented custom segmentation/sequence/multitask step contracts.
  Gate: native loss/metric parity, shape/masking tests and tiny actual training runs.
- [x] CNN, recurrent and Transformer construction/examples using optional native
  torch modules. Gate: forward/backward, convergence sanity and checkpoint roundtrips.
- [x] DataLoader/sampler resume contracts, iterable/streaming datasets and bounded
  worker/prefetch/pinned-memory controls. Gate: actual two-worker RNG parity,
  loader/sampler generator and epoch restoration, exact bounded JSONL offsets,
  source/contract rejection before mutation and explicit refusal of unsafe
  persistent-worker/iterable-prefetch cases. See training/data help.md for scope.
- [ ] DDP CPU/GPU training, rank-safe callbacks/checkpointing and worker failure
  recovery. Gate: real multi-process parity and interruption tests.
- [x] Scoped CPU DDP: two actual Gloo ranks, uneven global-batch parity, observer
  callbacks, exact epoch resume and previous checkpoint recovery on rank failure.
  GPU DDP acceptance remains part of the unchecked parent gate.
- [ ] FSDP/large-model sharding and activation checkpointing. Gate: measured
  memory behavior and distributed save/load on supported hardware.
- [x] Scoped CPU FSDP: actual two-rank partitioning, serial parity, native
  local model/optimizer/RNG shards, exact same-world resume and publication/rank
  fault recovery. GPU and large-model acceptance remains unchecked.
- [x] CPU Sequential activation checkpointing: saved-tensor byte reduction,
  exact gradient parity and training/checkpoint reload. FSDP stays open.
- [x] Generative/LoRA, quantization and export compatibility matrix. Gate: actual
  tiny-model evaluation, ONNX/runtime parity and documented hardware limits.

## 3. Reinforcement learning coverage and correctness

- [x] Audit existing RL: agent action weights/reward signals must not be described
  as full environment-training PPO/CQL implementations.
- [x] Correct terminal/truncation-aware returns and GAE, bounded replay, replay-local
  RNG ownership and serialization. Gate: hand-calculated numeric and boundary tests.
- [x] Gymnasium environment training/evaluation API with explicit episode/step
  budgets, seeding, action/observation checks and separate evaluation environments.
- [x] Verified optional backend integrations for PPO/A2C/DQN/SAC/TD3/DDPG and HER;
  distinguish integrations from native implementations. Gate: real discrete and
  continuous environment learning, save/load, replay/normalization resume tests.
- [x] Offline RL and neural imitation learning (not action-frequency tables).
  Gate: dataset schema/mask tests and reproducible held-out learning checks.
- [x] Vectorized/recurrent/masked/multi-agent RL contracts. Gate: episode boundary,
  recurrent-state reset, action-mask and actual multi-agent environment tests.
- [x] RL checkpoint/environment-state/replay/RNG continuation and cancellation.
  Gate: deterministic supported-environment resume and explicit unsupported cases.
- [x] Multi-seed learning benchmarks with confidence intervals and real baselines.
  Gate: publish measured results; a smoke test is not a learning-quality claim.

## 4. Classical and specialized workflows — scoped acceptance

- [x] Optional LightGBM/CatBoost adapters and native CPU pipeline acceptance.
- [x] Forecasting: lag/horizon models, date features and rolling-origin backtests.
- [x] Ranking: query-group validation and held-out NDCG/MAP metrics.
- [x] Recommendation: sparse baselines/retrieval, bounded top-k and cold starts.
- [x] Survival: event/time schema, censoring-aware training and held-out metrics.
- [x] Bounded SHAP with transformed feature names and sparse/dense safeguards.
- [x] Typed dataset snapshots, schema evolution and transactional stream offsets.
- [x] Same-host distributed random-search quotas, stale-worker recovery and deterministic sampler continuation.
- [x] Advanced uncertainty: known-ratio weighted, simultaneous multioutput, exchangeable-block and fixed-group methods,
  each with its own assumptions and statistical acceptance.

Each item here needs native functional/parity tests, failure tests, persistence
where applicable, help.md, import isolation and an installed-wheel example.

Acceptance scope and statistical/resource limits: docs/CLASSICAL_SPECIALIZED.md.
Current source suite: 1026 passed, eight unrelated optional skips. Final installed
specialized suite outside the source tree: 30 passed, zero skips. The complete
installed --full example passes actual LightGBM/CatBoost pipeline reloads, ranking,
survival and the generic workflows. Clean offline base installation adds only
nil-sdk; 186 Python sources match the wheel and 101 help guides are bundled.
These close the nine stated workflow milestones, not every algorithm or
advanced variant in each field. Native engines remain explicit integrations.

## 5. Independent engines and production acceptance

- [x] Define the supported NIL-native numerical subset and benchmark target;
  implement/test it independently before claiming engine-free training.
- [x] Expand native autodiff/tensors/operators/optimizers with numerical gradient
  checks, sparse/dtype/device behavior and published compatibility coverage.
- [ ] Production serving, concurrency/backpressure, observability and compatibility
  matrices. Gate: load/fault/security tests and deployment documentation.
- [x] Local service building block: isolated trusted CPU worker, bounded threads/
  admission, hard inference deadlines, crash restart, authenticated metrics and
  actual concurrent HTTP/fault/auth tests. External production acceptance stays open.
- [ ] Complete hardware/OS/Python/backend matrix and reproducible release checks.

## Completion evidence

After each completed item record the implementation, exact test command/results,
platform/backend versions and remaining limits here before ticking its box.

### Earlier full validation (before specialized ML batch) — 2026-11-21

- Full CPU neural/RL environment: **924 passed, 14 optional/hardware skips**.
- Full classical environment: **990 passed, seven optional skips**.
- Focused current DL matrix: **24 passed, one real CUDA skip**.
- Native control integration: **12 passed**; pure RL helpers: **51 passed**.
- Commands: `PYTHONPATH=nil-sdk OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python
  -m pytest nil-sdk/tests -q` in each existing environment; focused files named
  below. Metadata parity, Python syntax and whitespace checks pass.
- `python scripts/verify_neural.py --rl --cuda` exits **2** on this CPU-only
  host. CUDA/other-platform/distributed/quality gates stay unchecked.
- These are separate backend environments, not a summed universal coverage score.

### Installed-wheel validation — 2026-11-21

The offline build verifier installs only nil-sdk in a fresh environment, resolves
help/lazy APIs and executes a real stdlib worker. The wheel is also installed in
both development environments without adding dependencies. Extracted CNN, LSTM,
Transformer and PPO examples all pass from /tmp with PYTHONPATH unset, including
checkpoint/policy reloads. A fresh installed-wheel owned RL job trains, saves and
returns a policy bundle that reloads successfully. Bundle hashes, 86 help guides,
metadata parity, Python syntax and whitespace checks pass. The checklist and
NEURAL_RL guide ship in the review bundle. This validates these paths, not the
remaining unchecked GPU/distributed/independent-engine items.

### Execution core — 2026-11-21

Implemented nil.execution and registered-run reconciliation. Native process
acceptance in the classical environment: eight passed, one torch-dependent test
skipped. Execution + help/catalog checks in the torch environment: 20 passed,
two sklearn-dependent skips. Tests use actual timeout/cancelled processes,
descendant cleanup, a 192 MiB virtual-address cap rejecting a 512 MiB allocation,
bounded logs/payloads, PID-identity recovery, sklearn fitting and torch training
with checkpoint reload. Commands: `PYTHONPATH=nil-sdk python -m pytest
nil-sdk/tests/test_execution.py -q` in each existing environment. The bundle build
also passes a clean NIL-only install and a real installed-wheel stdlib worker.
Linux/Python 3.12 tested. At this earlier checkpoint independent supervision
was still open; it is completed in the later batch below. Other-platform/GPU
quotas remain open; virtual address space is not aggregate RSS.

### DL correctness, CPU AMP and checkpoint recovery — 2026-11-21

`PYTHONPATH=nil-sdk python -m pytest nil-sdk/tests/test_dl_correctness.py
nil-sdk/tests/test_development_sdk.py nil-sdk/tests/test_quick_development.py -q`
passes 41 tests, skips six optional checks (one actual CUDA check and existing
optional backend checks), with torch 2.15.0+cpu / Python 3.12 on Linux. Validated
native mean/sum loss parity across uneven microbatches, weighted/ignored CE,
binary/multilabel learning and metric parity, convolutional segmentation masks,
exact bf16 dropout resume, incompatible-resume rejection before state mutation,
and actual process cancellation during checkpoint publication followed by recovery
of the last committed generation. File-commit fault injection and explicit
one-generation rollback also pass. AMP CUDA code exists but hardware acceptance
is deliberately unchecked. Checkpoint model structure/environment checks are
additive; legacy valid version-1 files remain loadable. Arbitrary custom step,
loader, callback, cross-device and distributed continuation are not implied.

### RL numeric correctness and neural control integration — 2026-11-21

Pure RL helper/legacy acceptance: 51 passed with `PYTHONPATH=nil-sdk python -m
pytest nil-sdk/tests/test_rl.py nil-sdk/tests/test_rl_correctness.py -q`.
This catches analytic termination/truncation bootstrap behavior, no advantage
leakage across episodes, replay ordering/overflow/local RNG restoration and the
value-gradient update using old rather than already-updated hidden weights.
Agent updates now use ordered rollouts and honest advantage_weights naming.

Native control: `PYTHONPATH=nil-sdk OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python
-m pytest nil-sdk/tests/test_rl_control.py -q` passes 12 tests on SB3 2.9.2,
Gymnasium 1.4.0, torch 2.15.0+cpu, NumPy 2.5.4, Python 3.12/Linux. All six
algorithms actually update parameters, save/load and continue counters; off-policy
replay and VecNormalize statistics roundtrip. PPO/DQN learn a deterministic
discrete reward preference and SAC improves a tiny continuous-action reward.
HER trains and restores on a goal environment. Tests also cover cancellation,
collection rounding, replay-array preflight, tampering, evaluation RNG/mode
restoration, bounded evaluation, an owned RL worker returning a validated policy
bundle, and rejection of training/evaluation env reuse. Root nil.predict/evaluate
also dispatch to native RL without unrelated backend loading.
These are functional regression cases, not broad multi-seed quality benchmarks.
Arbitrary simulator-state continuation, offline/recurrent/masked/multi-agent RL
and GPU validation remain unchecked.

### DL architecture and token-weighted objective checks — 2026-11-21

Expanded `test_dl_correctness.py`: 24 passed, one actual CUDA skip on CPU. Added
CNN/LSTM/Transformer forward/backward/learning and exact state-dict roundtrips
through examples/deep_workflows.py, plus causal-LM gradient parity across unequal
unmasked shifted-token counts. Fixed generator-valued MLP hidden widths being
consumed during validation. Exact momentum/StepLR/plateau-scheduler continuation,
custom two-head objective parity, environment rejection, optimizer ownership and
parameter-order rejection also pass. Tiny constant-input learning checks prove the path
executes, not architectural quality or benchmark performance.


## 2026-11-21 batch evidence and unavailable acceptance targets

Of the next five requested items, detached supervision and the documented
loader/streaming contract are implemented and validated. Actual client SIGKILL
acceptance runs in separate OS processes. Loader tests run real Torch workers,
not mocked batches. Full environment runs: classical 992 passed/8 optional skips;
Torch/RL 931 passed/14 optional skips, followed by final loader/execution checks
17 passed/1 optional sklearn skip. Neural/RL preflight acceptance: 130 passed,
7 explicit optional/platform skips. The rebuilt bundle passes clean offline installation with normal dependency
resolution, adding only nil-sdk. All 172 Python sources match the wheel and all
bundle manifest hashes pass. Installed-wheel loader/execution checks outside
the source tree: 17 passed/1 optional sklearn skip; installed JSONL resume
example passes exact history/weight parity.

The other three gates remain open: this Linux Ryzen 3 7320U host has no CUDA
GPU and is not a ten-year-old PC. Its cgroup v2 memory controller exists, but
the session hierarchy is root-owned without delegated write access; no tested
aggregate memory quota is available. Windows/macOS targets are absent. Current
host benchmark raw samples/report are in docs/benchmarks/2026-11-21-*.json/.md;
these are evidence of measured workloads, not certification of older hardware.


## Classical/specialized acceptance — 2026-11-21

Completed the nine scoped items in section 4. New modules: boosting, forecasting,
ranking, recommendation, survival and datasets; expanded inspection, selection
and uncertainty. The catalog exposes 54 constructors in ten shallow groups with
101 help guides. Root/nested project metadata match and base dependencies remain
empty. Both existing development installations use the rebuilt regular wheel;
no new backend was installed into the secondary Torch environment.

Source full acceptance: 1026 passed/8 optional skips. Specialized preflight with
actual LightGBM/CatBoost/lifelines: 166 passed/one unrelated neural skip at that
checkpoint, followed by the expanded full suite. Final installed tests: 30 passed,
zero skips, with PYTHONPATH removed and cwd outside the source tree. The installed
specialized_workflows.py --full example passes every included workflow. The
clean offline resolver adds only nil-sdk and verifies passive imports, missing
backend messages, offline help and a real stdlib execution job. Native versions
and known advanced limits are recorded in docs/CLASSICAL_SPECIALIZED.md.


Final minimum-backend matrix: 67 passed/12 explicit optional-backend skips on
sklearn 1.4.2 / NumPy 1.26.4. Full installed specialized acceptance uses actual
LightGBM/CatBoost/lifelines in the current environment and passes 30/30 tests.
All nine section-4 boxes are checked; 14 other NIL-wide items remain open.


## CPU completion batch — 2026-11-22

Seven of the preceding 14 broad open items now meet their scoped gates. Seven
remain open: other-OS/aggregate quotas, representative old-PC scale acceptance,
CUDA AMP, GPU DDP, FSDP sharding, external production-serving acceptance and the
complete OS/hardware/Python/backend matrix. CPU DDP, activation checkpointing
and local serving have separately checked validated subsets.

- Native subset: independent CPU float64 dense reference Tensor/autodiff,
  Linear/Sequential/activations, MSE/stable BCE, SGD momentum/decay, plain JSON
  model/optimizer state. Finite-difference and Torch parity checks; unsupported
  sparse/dtype/device inputs reject explicitly. Five fresh Python -S children
  meet the published benchmark target without importing numerical backends.
- Offline RL: actual held-out discrete/continuous neural imitation and masked
  fitted-Q baseline, terminal/truncation tests, exact optimizer/target/RNG resume.
- Structured RL: native recurrent/masked policy train/save/load/evaluate,
  vector terminal observations and independent multi-agent neural learning.
  Masked multi-agent combination and centralized critics remain unsupported.
- Simulator state: exact supported CPU DQN/ordinary replay with explicit raw
  state protocol and standard file-free Monitor; cancellation at committed
  collection boundaries; unsafe/unsupported cases refused. Identical learn-call
  schedules required, not arbitrary simulator resume.
- Learning-quality gate: five PPO seeds, actual random-policy baseline and
  bootstrap CI over seed means. Tiny bandit only: 1.000 vs 0.532, paired 95% CI
  [0.444,0.492]. Raw measurements in docs/benchmarks/2026-11-22-rl-quality.json.
- CPU generative/export matrix: local GPT2 held-out perplexity/reload, actual
  LoRA update/merge/safetensors/generation, dynamic INT8 Linear, MLP/CNN/LSTM
  ONNX runtime parity. No downloads, GPU or universal operator support claim.

Full-suite runs in this batch: classical **1034 passed, 15 optional skips**;
neural/RL **967 passed, 13 optional/hardware skips**. Final numerical guards add
explicit matmul-output and normalized offline-storage budget tests: **14 passed**
in the focused native/offline suite after those changes. Installed acceptance
from outside the source tree passes **26 tests**, with one source-bootstrap test
intentionally deselected because Python -S cannot resolve an installed package.
Its dependency-free training/JSON path is separately exercised by the clean-wheel
verifier and installed example. The installed --neural --ddp example passes
independent training/persistence, neural imitation/root inference and actual
two-rank CPU training. Backend versions, compatibility cells, exact
commands, serving deployment contract and remaining limitations are documented in
[docs/CPU_COMPLETION.md](../CPU_COMPLETION.md). Base dependencies remain empty;
optional rl-contrib/generative/export backends were installed only explicitly
into the existing neural development environment.


Distribution: 194 Python sources match the wheel, 103 bundled help.md guides,
manifest hashes pass and fresh offline dependency resolution adds only nil-sdk.
Both existing development environments use the regular rebuilt wheel, with no
automatic backend additions. The wheel is approximately 476 KiB.

Commands: `PYTHONPATH=nil-sdk OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m
pytest nil-sdk/tests -q` separately in each backend environment (HF_HUB_OFFLINE=1
and TRANSFORMERS_OFFLINE=1 for neural). Installed checks use `env -u PYTHONPATH
python -I -m pytest` on extracted test files from a directory outside the source
tree. `python scripts/build_sdk_bundle.py --output /home/us/.nelo/nil-cleanup-artifacts`
rebuilds and verifies the dependency-free distribution. Exact new test files:
test_native_engine.py, test_distributed_training.py, test_neural_compatibility.py,
test_offline_neural_rl.py, test_rl_structured.py, test_rl_snapshot.py and
test_isolated_serving.py. Upstream quantization/export deprecations and tracing
warnings remain documented compatibility limits, not suppressed failures.


Final installed numerical acceptance after the normalized-storage budget fix:
**13 passed, one source-bootstrap deselection**. The installed --neural example
passes again. No additional dependency was installed during either wheel update.


## Ordered follow-on: phases 1–2 — 2026-11-22

The nine-phase follow-on is [PHASES.md](PHASES.md). Seven broad parent acceptance
gates above remain open. Phase 1 software adds opt-in Linux summed process-group
RSS enforcement with typed resource_limited status, peak/limit evidence, Run
reconciliation and cleanup that survives client exit. It is a sampled guard,
not a hard memory ceiling. Default jobs do not scan procfs.

The separate cgroup-v2 aggregate-memory implementation requires explicit
delegation, enforces memory.max/swap=0/group OOM, attaches before input
deserialization and uses cgroup.kill. Local missing-delegation preflight fails
closed before user code. Real kernel OOM/new-session cleanup tests remain open
and skipped without an explicitly delegated target; this host's required-kernel
acceptance report exits 2. Windows/macOS/GPU memory acceptance is also still open.

Phase 2 software defines small/medium workloads and small-profile latency/RSS
budgets, bounded fresh-process samples and structured failure reports. Small
classical/neural profiles pass declared budgets; medium baselines run successfully
without enforced thresholds. Three samples per case record raw timing and peak
RSS, actual host/backend versions and NIL/native Ridge wrapper fit overhead.
Every report keeps old_pc_acceptance=false; no older target has been tested.

Focused initial execution/resource/runner/help acceptance: 24 passed/three skips.
Full classical source suite after registered-Run resource-limit acceptance:
**1045 passed, 17 optional/target skips**. Neural and installed acceptance follows
below. Evidence and reproduction commands:
[docs/benchmarks/2026-11-22-phases-1-2.md](../benchmarks/2026-11-22-phases-1-2.md).

Full CPU neural/RL source suite: **979 passed, 15 optional/target skips**. Both
full environments pass with actual resource-limit/Run tests; skips include the
two intentionally unaccepted real-cgroup tests. No GPU or delegated-kernel pass
is inferred from these runs.


Final installed acceptance for phases 1–2: **20 passed/three explicit skips**
(two unconfigured delegated-kernel tests, one unrelated Torch test absent in the
classical environment). The installed stdlib execution runner passes all five
requested checks outside the source tree. Full source suites remain classical
1045/17 skips and neural 979/15 skips. Clean offline dependency resolution adds
only nil-sdk and executes the new RSS path. Source/wheel parity covers 195 Python
files and 103 help guides; both development environments use the rebuilt wheel
without adding dependencies. All bundle hashes pass. The approximately 478 KiB
wheel, phase list, runners, budget files and measured reports are bundled.


## Sharding and target acceptance implementation — 2026-11-22

Implemented optional fixed-world CPU FSDP through nil.distributed.train(strategy=
'fsdp'), bounded model/data preflight and explicit trusted native-shard resume.
Rank-zero export/callbacks gather full parameters temporarily. Actual local shard
counts, serial/global-batch parity, exact momentum/RNG resume, rank failure and
manifest-publication failure recovery pass. Hash tampering and data/world
changes reject before training; hung DDP ranks are terminated within a deadline.

Focused sharding/DDP/acceptance-runner suite: **12 passed**. Full CPU neural/RL
source suite: **985 passed/20 skips** (five new actual-CUDA tests are unrun here).
GPU runner records real device/driver/runtime requirements and rejects this host
with exit 2; no GPU pass is claimed. Nine-cell base platform/Python and manual
trusted self-hosted GPU CI definitions are written, not remotely executed.
At that earlier checkpoint Windows containment was unsupported; its CI scope
was base-package installation/refusal. The later Windows implementation batch
below replaces that scope with required native containment tests.

Scope, commands and limits: docs/benchmarks/2026-11-22-sharding-gates.md.
PHASES.md distinguishes the now-validated CPU FSDP software from open GPU,
large-model, other-OS and production target acceptance.


Full classical source acceptance: **1046 passed, 19 optional/target skips**.
Full neural/RL source acceptance: **985 passed, 20 optional/target skips**.
Final bounded-input/help/runner checks validate complex-input rejection and
the public FSDP documentation after the full-suite runs.


Installed initial acceptance outside the source tree: **12 passed**, and the
installed --fsdp example passes actual two-rank training and checkpoint export.
Final recovery handling also permits exporting an already-completed shard epoch
without another optimizer update; the resumable manifest stays at its original
location and the full export is written to the requested new output directory.
The rebuilt distribution is revalidated for that final path below.


Final rebuilt-wheel recovery acceptance outside the source tree: **3 passed**,
including zero-update export of a completed epoch with a valid resumable manifest
reference. Initial installed sharding/DDP/runner suite passed 12 tests; the FSDP
example also passed. Both development environments use the final regular wheel
without adding dependencies. Clean offline installation adds only nil-sdk;
196 Python sources match and 103 help guides remain bundled. Installed offline
help does not load Torch/NumPy/sklearn. CI YAML parses, but no remote jobs were
executed. Hardware/OS/large-model acceptance remains open as listed above.


## Windows owned process-tree implementation — 2026-11-22

Implemented native Windows Job Object execution with standard-library ctypes and
CPython _winapi, preserving a NIL-only base installation. Workers are created
suspended, assigned before resume and run with an explicit stdio handle list.
No unowned fallback or breakaway is permitted. A detached supervisor retains the
noninheritable kill-on-close job handle; cleanup terminates the owned tree and
waits for zero active processes on success/error/timeout/cancel. Process creation
times support read-only identity-safe recovery. Transient Windows reader-sharing
errors get a bounded atomic-journal retry preserving the last committed record.

Portable launch/failure and journal tests plus Linux execution/resource/help:
**31 passed, three explicit skips**. Final classical full suite: **1056 passed,
29 optional/target skips**. Full neural/RL run before the final bounded journal
retry: **992 passed, 29 optional/target skips**; targeted checks cover final changes.
Required Windows acceptance correctly refuses this Linux host with exit 2.
Ten actual Windows kernel tests are present and remain unrun; the three Windows
Python CI cells now run them and the required Windows stdlib runner. No remote
workflow has been dispatched. This closes the Windows implementation task, not
Windows platform acceptance. Details: docs/WINDOWS_EXECUTION.md.

Remaining implementation: production deployment; broader independent numerical
engine; advanced RL/model coverage; device-specific GPU quotas. Remaining target
acceptance: Windows/macOS and release/backend matrix, delegated Linux cgroups,
old-PC performance, actual CUDA/two-GPU DDP/FSDP and large-model memory behavior.


Final Windows launch review uses creation-time PROC_THREAD_ATTRIBUTE_JOB_LIST
plus suspended startup and membership verification. This closes the documented
create-then-assign orphan window. Windows 10+/Server 2016+ is required; native
acceptance remains pending. No fallback to older unowned process creation exists.


Final creation-time launch contracts/help/runner checks: **20 passed**. Installed
Linux execution/resource/portable-Windows contracts outside the source tree:
**26 passed, three explicit skips**. Installed stdlib execution acceptance passes
including successful/timeout child cleanup and the RSS guard. Fresh offline
installation adds only nil-sdk. Both development environments use the rebuilt
regular wheel; 197 Python sources match and 103 help guides are bundled.
Native Windows acceptance is still unrun; ten real-kernel cases and three Windows
CI cells require a target host. The Linux --require-windows report fails with
exit 2, so portable test success is not mistaken for Windows certification.

## Managed observability, installed profiles and native multiclass — 2026-11-23

- [x] Add opt-in atomic service metrics snapshots, final drained status and private
  runtime directory in the systemd template; no monitoring dependency or new thread.
- [x] Validate descriptor-based regular/private token files, reject FIFO/final
  symlink on POSIX, validate bounded printable distinct capabilities before model
  loading, refuse output/input path collisions and sanitize CLI failures.
- [x] Authenticated metrics and real signal-lifecycle acceptance: 19 focused tests
  passed. Counts advance and drained connection/inflight counts are zero; sample
  bodies, credentials and configuration contents are absent from observable output.
- [x] Add installed backend verifier and four minimum/current Linux/Python3.12 CI
  cells. All four passed at c03b20f: classical 67/69 tests plus two examples each;
  Torch2.6/2.15 CPU 52 tests plus three architecture examples each; zero skips.
  Report: ../benchmarks/2026-11-23-backend-matrix.json. Workflow45115748766.
- [x] Index all nine passing base-platform build artifacts, digests and expiry at
  c03b20f: ../benchmarks/2026-11-23-platform-artifacts.json. Workflow45115748776.
  Verification artifacts are retained by CI; no stable tag/release was published.
- [x] Add independent-engine row-wise softmax/log-softmax, stable categorical loss,
  transactional Adam, typed JSON optimizer dispatch and ownership/order checks.
  Existing SGD/MSE/BCE defaults remain. Native finite-difference, Torchfloat64,
  real learning, exact continuation and no-site-package checks: 18 passed.
- [x] Benchmark five fresh `python -S` native multiclass processes: 90x2, 3 classes,
  100 Adam steps; final CE0.00696, accuracy1.0, median fit0.101s, median RSS17.65MiB.
  Current Ryzen3 7320U host only; ../benchmarks/2026-11-23-native-multiclass.json.
- [x] Full neural/RL development-environment regression after implementation:
  1026 passed, 30 explicit optional/platform/device skips in102.03s.

Commands: `PYTHONPATH=nil-sdk python -m pytest nil-sdk/tests -q` in the neural
backend env; focused tests test_service_credentials/test_tls_gateway/test_tls_inference
and test_native_engine/test_native_multiclass. `python scripts/accept_backends.py
--profile classical|torch --output PATH` requires a regular installed wheel and
explicit backends; copies tests/examples, never SDK source, to isolated scratch.
`PYTHONPATH=nil-sdk python scripts/benchmark_native.py --workload multiclass
--output report.json` records the independent workload without optional imports.

Remaining: actual GPU quotas/AMP/distributed/sharded target tests, identified old-PC
acceptance, ND/sparse/compiled/device numerical engine, advanced RL/model learning
coverage, other optional-backend platform combinations and public-ingress review.
The managed Linux reference and declared backend profiles passed; this does not
certify all possible deployments, models or optional dependency combinations.

Final native installed acceptance at 7fc1943: hosted backend run45116372373 passes
all four cells; both Torch2.6/2.15 profiles now pass70 tests and all three installed
examples, zero skips. Base run45116372289 passes all nine OS/Python cells including
NIL-only multiclass training. Local final wheel passes70 Torch/native/service and
67 minimum-classical tests plus examples outside the checkout. All199 packaged
Python sources match the workspace; ZIP manifest hashes verify. Base offline
resolver installs nil-sdk alone. The local verifier additionally rejects editable
metadata and checks that the imported package belongs to the installed distribution.

## Requested low-resource hardware checks — 2026-11-23

User confirmed separate GPU and older-PC targets are unavailable and requested
low-resource checks here. Actual inventory: Ryzen3 7320U/7GiB, Radeon610M,
CPU-only Torch2.15, no CUDA/HIP runtime, no configured GPU runners.

- [x] Add a kernel-verified low-resource gate using installed isolated workers;
  fail before workloads if memory/swap/CPU/task policy is missing or too loose.
- [x] Check numerical outcomes as well as time/RSS: native MSE<1e-6, Ridge RMSE<.05,
  and two finite decreasing neural losses. Fast incorrect results cannot pass.
- [x] Run core, classical and neural small profiles, three fresh samples per case,
  each in a new real CPUQuota100%/MemoryMax1GiB/no-swap/TasksMax64 user unit.
  All pass; final kernel peaks30.17/121.12/239.35MiB; zero OOM kills, zero swap use.
- [x] Portable policy/functional/refusal regressions:21 passed in1.25s in the
  classical environment. Full raw samples and exact reproduction are in
  ../HARDWARE_ACCEPTANCE.md and ../benchmarks/2026-11-23-low-resource-*.json.
- [x] Record actual GPU preflight failure with adapter/runtime diagnostics; no
  unrun CUDA tests or CPU checks are reported as GPU acceptance.
- [ ] Actual older-PC and CUDA/two-GPU acceptance: hardware remains unavailable;
  current-host throttling does not certify old ISA/cache/storage or GPU behavior.

Commands used systemd-run --user --wait --pipe --collect with the above limits,
explicit existing backend interpreters and scripts/accept_low_resource.py.
No permanent service, global controller change, driver or backend installation.
The selected fallback is completed; actual target gates remain open in PHASES.md.

Final harness retains one self-contained report per profile and removes temporary
child outputs. Repeated actual-unit acceptance passes all three profiles again;
final portable checks:21 passed in3.64s in the Torch environment. Raw reports
contain kernel policy/counters and all benchmark samples without duplicate log
files in the repository. Earlier diagnostics remain under workspace artifacts.

## Phase8 CPU engine and Phase9 RL extensions — 2026-11-23

- [x] Preserve the original rank2 API and model archives while adding explicit
  nonempty rank0–8 shapes/strides, owned reshape/permutation/slicing, trailing
  broadcasting, axis reductions, batched/vector matmul and shape-preserving grads.
- [x] Extend stable losses and transactional SGD/Adam to scalar/vector/ND leaves;
  add bounded plain-JSON tensor persistence and exact optimizer-state continuation.
- [x] Implement explicit COO/CSR, duplicate/coalescing gradients, sparse×dense,
  bounded densification, empty sparse matrices, storage/coordinate guards and JSON.
- [x] Build NIL-owned C matmul/VJP only on explicit request, with local compiler/
  header preflight, active ABI/binary digest checks, exclusive publication and no
  numerical library dependency. Core install/import never builds/loads a binary.
- [x] Measure five fresh processes/backend:96x64 @64x48 forward+backward, numerical
  parity and RSS; observed7.29x for this workload only. Raw report:
  ../benchmarks/2026-11-23-native-cpu-kernels.json. Not universal/old-PC performance.
- [x] Add masked parallel collection, per-agent spaces/terminal masks, authoritative
  terminal observations, recurrent lifecycle and RNG/mode restoration on failures.
- [x] Add discrete CQL-style log-sum-exp conservative Q updates, recorded-action/
  OOD contracts, masks and exact same-data/config checkpoint continuation.
- [x] Add separately scoped offline centralized-value AWR with joint critic/local
  discrete actors, explicit aligned rows/team returns, bounds and exact update resume.
- [x] Extend supported exact protocol-environment continuation to CPU PPO/A2C
  completed rollout/update boundaries. No recurrent/continuous/wrapper guarantee.
- [x] Run three-seed real learning gates for discrete FrozenLake CQL, two-CartPole
  offline centralized AWR, continuous MountainCar imitation, contextual masked PPO
  and recurrent CartPole PPO. All paired bootstrap lower bounds exceed zero.
  Source/replay/scaling/algorithm limits: ../NEURAL_RL.md and raw
  ../benchmarks/2026-11-23-rl-coverage.json. No remote data/model download.

Validation: full neural/RL development suite **1098 passed,30 explicit skips** in
106.66s. Final focused engine/RL checks after ABI/terminal guards:
**86 passed** in4.86s, including actual C compilation and independent numerical/
Torch parity. Earlier full run predates the final two added guard regressions;
the final focused run covers them. No skipped case counts as target acceptance.
Commands use PYTHONPATH=nil-sdk and the explicit neural backend interpreter:
`python -m pytest nil-sdk/tests -q`; focused native_* plus RL contract tests;
`python scripts/benchmark_rl_coverage.py --output report.json` (cases selectable).
Use `nil.native.build_cpu_kernel(new_directory)` then
`python scripts/benchmark_native_kernels.py --kernel PATH --output report.json`.

Remaining target work: actual GPU kernels/quotas/AMP/distributed/sharding, physical
older-PC behavior, larger pretrained/LoRA/quantization/export workload coverage,
public-ingress deployment/security/workload acceptance, and untested optional
backend/platform combinations. ND/CPU algebra and the declared RL extensions are
implemented; zero-sized tensors, universal acceleration, arbitrary MARL/simulator
continuation and broad unseen-task quality are not implied by these checkmarks.

Additional declared CPU model/data gate: bundled digits MLP held-out accuracy0.9667,
dynamic INT8 and ONNX parity checks; separate-document local causal training,
LoRA frozen-base validation and merged local reload. All pass in5.87s on the
current host with no downloads. Raw source/data hashes, versions and limits:
../benchmarks/2026-11-23-model-coverage.json and ../NEURAL_RL.md.
Large downloaded pretrained architectures and realistic large-model/device memory
remain unrun; these small cases do not close that target gate.

Hosted verification at aa3bfb8: all nine base-platform cells pass (run45143679690),
all ten expanded backend cells pass (run45143679440), full SDK CI passes
(run45143679678). Backend scope: minimum/current classical on Linux/macOS/Windows
Python3.12, plus minimum/current CPU Torch and RL on Linux/Python3.12. This is
acceptance of declared profiles; other extras/platform/precision combinations
remain separate gates. Final scalar/vector implicit-operand regression also
passes66 native engine tests after this hosted snapshot.

Final hosted snapshot761ca07 passes base45145136514 (9cells),
backend45145136494 (10cells) and full SDK45145136525. Final offline bundle build
verifies NIL-only dependency resolution; all203 Python sources match its wheel,
C source is bundled without native binaries, and ZIP hashes verify. Language
corpus SHA256s still match the benchmark's exact train/held-out files. The current
ND core/native profile passes a fresh actual one-CPU/1GiB/no-swap/64-task unit
again, peak32.6MiB. Runtime code was published only to the approved verification
branch; no main-branch merge or permanent service/release deployment occurred.

## Pinned external pretrained CPU follow-on — 2026-11-23

- [x] Resolve and explicitly prepare SmolLM2-135M safetensors at immutable revision
  93efa2f097d58c2a74874c7e644dbc9b0cee75a2; verify271,971,074 downloaded bytes and
  file SHA256s outside the repository/distribution. No new backend package installed.
- [x] Fix model/tokenizer revision/cache/auth forwarding and strictly boolean
  download opt-in. Default remains offline with remote model code disabled.
- [x] Remove redundant second model clone from dynamic quantization; preserve
  caller mode/state/ownership and add explicit subtree exclusions/per-channel Linear.
- [x] Record dynamic-activation INT8 fidelity failure without relaxing +0.5 NLL.
- [x] Implement explicit CPU Linear INT8 per-channel weight storage with FP32
  per-operation dequantization; no persistent FP32 cache or speedup/QAT claim.
- [x] Real134,515,008-parameter inference, deterministic generation,230,400-parameter
  LoRA/eight updates, unchanged frozen weights and adapter-only reload pass.
- [x] Weight-only INT8 passes: NLL+0.147897, state220,317,635 bytes versus538,060,032
  FP32 parameter bytes. Dynamic activation per-channel variant remains failed.
- [x] All three stages pass in actual CPUQuota100%/MemoryMax2GiB/no-swap/TasksMax128
  unit: peak1,719,111,680 bytes, zero OOM kills and zero swap, wall41.47s.
- [x] Add bounded explicit checkpoint preparation, digest verification, parent-owned
  scratch cleanup and fail-closed numerical/policy reports; no install/import downloads.

Contract and previous neural compatibility tests:14 passed in4.24s. Full source,
installed-wheel and hosted validation are recorded below when complete. Raw report
(including failed dynamic variant), exact data limits and primary source attribution:
../PRETRAINED_CPU.md and ../benchmarks/2026-11-23-pretrained-135m.json.
Eight short evaluation windows are functional/numerical checks, not a general
language benchmark. Billion/GPU/other architecture/public deployment/physical old-PC
acceptance remains target-dependent and unchecked. Weight artifacts remain only
under workspace artifacts/nil/models; the SDK bundles code/help/reports, not models.

Pretrained follow-on installed acceptance: same three stages pass from regular
site-packages NIL under actual2GiB/no-swap/one-CPU limits, kernel peak
1,705,091,072 bytes, zero OOM kills, wall44.61s. The current raw report records
that installed run and retains the failed dynamic configuration. Installed
Torch/native/service/preparation profile:131 passed, zero skips, three examples.
Full neural/RL source suite:1110 passed/30 explicit skips in119.76s; final
preparation/loading/quantization guards:13 passed. The full run predates the three
new preparation regressions and final complex-dtype rejection; focused checks
and hosted tests cover the final guard paths. No mandatory dependency changed.

Final pretrained snapshot93d8937 passes base45151221151 (9cells),
backend45151221178 (10cells) and full SDK45151221134. All204 Python sources match
the regular wheel; ZIP hashes and updated documentation links verify. Fresh
base installation still adds nil-sdk alone, with no model or compiled payload.
The actual pretrained snapshot stays outside the distribution. Final runtime
updates were published only on the already approved verification branch.

</details>
