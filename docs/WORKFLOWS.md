# Supported workflow inventory

Matrix 2026.11.24.3; schema 1; SDK 0.2.1 (alpha).

This is a bounded inventory, not a completeness score for the 32 roadmap areas.
Recorded validation does not verify the current machine. Package detection does
not verify binary compatibility, prerequisites or hardware. See [SDK contracts](SDK_CONTRACTS.md).

Inspect offline with `nil workflows`, `nil workflows native.tensor --json`,
or `nil.coverage.matrix()`. All APIs below resolve without optional backend imports.

| Workflow | Implementation | Validation | Optional distributions |
|---|---|---|---|
| sdk.extensions | native | validated | none |
| sdk.offline_help | native | validated | none |
| native.tensor | native | validated | none |
| native.sparse | native | validated | none |
| native.compiled_cpu | native | validated | none |
| classical.supervised | integration | validated | scikit-learn |
| classical.validation | integration | validated | scikit-learn |
| classical.unsupervised | integration | validated | scikit-learn |
| specialized.forecasting | integration | validated | scikit-learn |
| specialized.ranking | integration | validated | lightgbm, scikit-learn |
| specialized.recommendation | integration | validated | scikit-learn |
| specialized.survival | integration | validated | lifelines, scikit-learn |
| data.snapshots | native | validated | none |
| neural.supervised | integration | validated | torch, numpy |
| neural.distributed_cpu | integration | validated | torch, numpy |
| rl.control | integration | validated | torch, gymnasium, stable-baselines3 |
| rl.offline | integration | validated | torch, numpy |
| rl.parallel | integration | validated | torch, numpy, gymnasium |
| generative.pretrained_cpu | integration | validated | torch, transformers, peft |
| optimization.cpu_quantization | hybrid | validated | torch |
| execution.owned_workers | native | validated | none |
| execution.linux_memory | native | validated | none |
| serving.managed_cpu | hybrid | validated | torch |
| gpu.mixed_precision | integration | target_blocked | torch |
| gpu.distributed | integration | target_blocked | torch |
| gpu.native_engine | unimplemented | target_blocked | none |
| hardware.old_pc | native | target_blocked | none |
| serving.public | unimplemented | target_blocked | none |
| vision.classification | integration | validated | torch, Pillow, numpy |
| sdk.result_contract | native | validated | none |
| rag.lexical | native | validated | none |
| agents.tool_plans | native | validated | none |
| probability.causal | hybrid | validated | numpy, scikit-learn |
| learning.regimes | hybrid | validated | numpy, scikit-learn, torch |
| audio.classification | integration | validated | numpy |
| video.multimodal | hybrid | validated | numpy, torch, Pillow |
| graphs.classification | hybrid | validated | torch |

## sdk.extensions

Lazy model registration

Areas: 1, 30. Offline help: `nil help catalog`.

APIs: `nil.register_model`, `nil.model`, `nil.list_models`

Input: Identifier, callable constructor, group and optional local help path

Output: Native constructor object selected explicitly

Limits: Factories execute only at explicit construction; registration is process-local; names cannot replace built-ins

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux", "Darwin", "Windows"], "python": ["3.10", "3.11", "3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [test_catalog.py](../nil-sdk/tests/test_catalog.py), [SHALLOW_DESIGN.md](../docs/SHALLOW_DESIGN.md)

## sdk.offline_help

Offline module/model documentation

Areas: 30. Offline help: `nil help catalog`.

APIs: `nil.help`, `nil.documentation.topics`

Input: Known topic identifier

Output: Bundled Markdown string or sorted topic list

Limits: No backend import, remote fetch or documentation server

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux", "Darwin", "Windows"], "python": ["3.10", "3.11", "3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [test_catalog.py](../nil-sdk/tests/test_catalog.py), [MODULE_REFERENCE.md](../docs/MODULE_REFERENCE.md)

## native.tensor

Independent dense tensors and autograd

Areas: 2, 3, 9. Offline help: `nil help native`.

APIs: `nil.native.Tensor`, `nil.native.fit`, `nil.native.SGD`, `nil.native.Adam`, `nil.native.save_tensor`, `nil.native.load_tensor`

Input: Finite built-in numbers, nonempty rectangular lists, optional explicit shape; native model and targets for fit

Output: Rank0–8 float64 CPU Tensor, matching-shape leaf gradients; fit mapping with model/history/steps

Limits: 100,000 elements per tensor; no zero-size dimensions or GPU operators; legacy scalar/vector normalization retained

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: backend-ubuntu-latest-torch-current, backend-ubuntu-latest-torch-minimum

Evidence: [NATIVE_ENGINE.md](../docs/NATIVE_ENGINE.md), [test_native_nd.py](../nil-sdk/tests/test_native_nd.py), [test_native_multiclass.py](../nil-sdk/tests/test_native_multiclass.py)

## native.sparse

Independent COO/CSR algebra

Areas: 2, 3. Offline help: `nil help native`.

APIs: `nil.native.SparseCOO`, `nil.native.SparseCSR`, `nil.native.save_sparse`, `nil.native.load_sparse`

Input: Explicit rank2 coordinates/row pointers, finite values and positive logical shape

Output: Sparse container or bounded dense result; gradients reach supplied values/dense operands

Limits: 100,000 stored entries/dense result; no implicit densification or sparse×sparse

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: backend-ubuntu-latest-torch-current, backend-ubuntu-latest-torch-minimum

Evidence: [NATIVE_ENGINE.md](../docs/NATIVE_ENGINE.md), [test_native_sparse.py](../nil-sdk/tests/test_native_sparse.py)

## native.compiled_cpu

Explicit compiled matmul and gradients

Areas: 4. Offline help: `nil help native`.

APIs: `nil.native.build_cpu_kernel`, `nil.native.load_cpu_kernel`

Input: Fresh output directory, trusted ABI-matching kernel, Tensor.matmul(kernel=...)

Output: Locally built kernel or float64 matmul result with VJPs

Limits: Only explicit opt-in; Linux/CPython target; no automatic compilation or general operator acceleration

Prerequisites: Linux C compiler; Matching CPython development headers

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: backend-ubuntu-latest-torch-current, backend-ubuntu-latest-torch-minimum

Evidence: [NATIVE_ENGINE.md](../docs/NATIVE_ENGINE.md), [2026-11-23-native-cpu-kernels.json](../docs/benchmarks/2026-11-23-native-cpu-kernels.json)

## classical.supervised

Classical fit/predict/evaluate/persistence

Areas: 7, 26. Offline help: `nil help classical`.

APIs: `nil.fit`, `nil.predict`, `nil.evaluate`, `nil.save`, `nil.load`

Input: Estimator, row-aligned features/targets, optional nonnegative weights and held-out validation

Output: ClassicalResult with model/run/metrics/task/warnings/status; prediction arrays or versioned trusted bundle

Limits: Backend estimator parameters remain explicit; trust required for executable artifact loading

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux", "Darwin", "Windows"], "python": ["3.12"]}]`

Recorded profiles: backend-macos-latest-classical-current, backend-macos-latest-classical-minimum, backend-windows-latest-classical-minimum, backend-ubuntu-latest-classical-minimum, backend-ubuntu-latest-classical-current, backend-windows-latest-classical-current

Evidence: [CLASSICAL_ML.md](../docs/CLASSICAL_ML.md), [test_classical_workflow.py](../nil-sdk/tests/test_classical_workflow.py)

## classical.validation

Pipelines, leakage-safe CV and search

Areas: 6, 11, 12. Offline help: `nil help validation`.

APIs: `nil.pipeline`, `nil.split`, `nil.cross_validate`, `nil.tune`

Input: Unfitted pipeline, features/targets, explicit split/CV/search parameters and fit budgets

Output: Split/CV/search results preserving held-out boundaries

Limits: Declared group/temporal/nested contracts; adaptive distributed search is separately scoped

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux", "Darwin", "Windows"], "python": ["3.12"]}]`

Recorded profiles: backend-macos-latest-classical-current, backend-macos-latest-classical-minimum, backend-windows-latest-classical-minimum, backend-ubuntu-latest-classical-minimum, backend-ubuntu-latest-classical-current, backend-windows-latest-classical-current

Evidence: [CLASSICAL_ML.md](../docs/CLASSICAL_ML.md), [test_classical_validation.py](../nil-sdk/tests/test_classical_validation.py)

## classical.unsupervised

Unsupervised catalog workflows

Areas: 7. Offline help: `nil help classical`.

APIs: `nil.model`, `nil.fit`, `nil.transform`

Input: Selected estimator and finite compatible feature representation

Output: Fitted ClassicalResult, labels/projections or task metrics

Limits: Supported paths depend on estimator predict/transform methods; catalog aliases are not independent implementations

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: backend-macos-latest-classical-current, backend-macos-latest-classical-minimum, backend-windows-latest-classical-minimum, backend-ubuntu-latest-classical-minimum, backend-ubuntu-latest-classical-current, backend-windows-latest-classical-current

Evidence: [CLASSICAL_ML.md](../docs/CLASSICAL_ML.md), [test_classical_models.py](../nil-sdk/tests/test_classical_models.py)

## specialized.forecasting

Fixed-horizon forecasting

Areas: 8. Offline help: `nil help forecasting`.

APIs: `nil.forecasting.fit`

Input: Ordered univariate series, lags/horizon and fixed-cadence calendar options

Output: Scoped forecaster and rolling-origin reports

Limits: Not a general ARIMA, multivariate, irregular or hierarchical suite

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [CLASSICAL_SPECIALIZED.md](../docs/CLASSICAL_SPECIALIZED.md), [test_specialized_ml.py](../nil-sdk/tests/test_specialized_ml.py)

## specialized.ranking

Query-group ranking

Areas: 8. Offline help: `nil help ranking`.

APIs: `nil.ranking.fit`, `nil.ranking.evaluate`

Input: Features, relevance grades and explicit query groups

Output: Ranker and per-query ranking metrics

Limits: No unbiased click-feedback/causal-ranking guarantee

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [CLASSICAL_SPECIALIZED.md](../docs/CLASSICAL_SPECIALIZED.md), [test_specialized_ml.py](../nil-sdk/tests/test_specialized_ml.py)

## specialized.recommendation

Sparse recommendation baseline

Areas: 8. Offline help: `nil help recommendation`.

APIs: `nil.recommendation.fit`, `nil.recommendation.evaluate`

Input: User/item interactions with explicit candidate/co-occurrence budgets

Output: Item-cosine/popularity recommendations and held-out ranking metrics

Limits: No neural/ALS or vector-database claim

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [CLASSICAL_SPECIALIZED.md](../docs/CLASSICAL_SPECIALIZED.md), [test_specialized_ml.py](../nil-sdk/tests/test_specialized_ml.py)

## specialized.survival

Right-censored survival integration

Areas: 8. Offline help: `nil help survival`.

APIs: `nil.survival.fit`, `nil.survival.evaluate`

Input: Features, durations and event indicators

Output: Cox/Kaplan–Meier outputs and scoped concordance

Limits: No competing risks, time-varying or clinical validity claim

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [CLASSICAL_SPECIALIZED.md](../docs/CLASSICAL_SPECIALIZED.md), [test_specialized_ml.py](../nil-sdk/tests/test_specialized_ml.py)

## data.snapshots

Bounded immutable JSONL data

Areas: 6, 28. Offline help: `nil help data`.

APIs: `nil.data.snapshot`, `nil.data.load_snapshot`

Input: JSON-compatible records and explicit byte/record budgets

Output: Immutable JSONL snapshot and manifest; fresh Records on load

Limits: Local data only; no Arrow dtype or external broker transaction claim

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [CLASSICAL_SPECIALIZED.md](../docs/CLASSICAL_SPECIALIZED.md), [test_typed_datasets.py](../nil-sdk/tests/test_typed_datasets.py)

## neural.supervised

Torch model training and evaluation

Areas: 9, 10. Offline help: `nil help training`.

APIs: `nil.train`, `nil.predict`, `nil.evaluate`, `nil.training.checkpoint.save`, `nil.training.checkpoint.load`

Input: Torch module and dataset/loader of inputs/targets or declared custom task batches

Output: TrainingResult(model/run/history/checkpoint/status); task metrics/predictions

Limits: Exact resume only for declared loader/config/epoch contracts; generic artifact save remains classical-specific

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: backend-ubuntu-latest-torch-current, backend-ubuntu-latest-torch-minimum

Evidence: [NEURAL_RL.md](../docs/NEURAL_RL.md), [test_dl_correctness.py](../nil-sdk/tests/test_dl_correctness.py)

## neural.distributed_cpu

CPU DDP/FSDP training and shards

Areas: 24. Offline help: `nil help distributed`.

APIs: `nil.distributed.train`

Input: CPU module, tensors and explicit fixed-world distributed options

Output: Native exported model plus checkpoint/shard diagnostics

Limits: Two-rank CPU contracts; exact trusted shard resume has version/world/data restrictions; no multi-host/GPU acceptance

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: backend-ubuntu-latest-torch-current, backend-ubuntu-latest-torch-minimum

Evidence: [NEURAL_RL.md](../docs/NEURAL_RL.md), [test_fsdp_training.py](../nil-sdk/tests/test_fsdp_training.py)

## rl.control

Native environment-training integrations

Areas: 19. Offline help: `nil help rl`.

APIs: `nil.rl.create`, `nil.rl.train`, `nil.rl.evaluate`, `nil.rl.save`, `nil.rl.load`

Input: Supported algorithm and compatible environment; explicit timesteps/evaluation environment

Output: RLTrainingResult(model/run/requested_steps/actual_steps/status); trusted policy bundles

Limits: Simulator continuation limited to declared CPU DQN/PPO/A2C protocol environments

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: backend-ubuntu-latest-rl-minimum, backend-ubuntu-latest-rl-current

Evidence: [NEURAL_RL.md](../docs/NEURAL_RL.md), [test_rl_snapshot.py](../nil-sdk/tests/test_rl_snapshot.py)

## rl.offline

Offline imitation and discrete CQL

Areas: 19. Offline help: `nil help rl`.

APIs: `nil.rl.offline_fit`

Input: Bounded explicit transition/action arrays, masks and matching config

Output: OfflineResult with model/history/checkpoint/status

Limits: Discrete CQL is not continuous CQL/IQL; no arbitrary OOD quality guarantee

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: backend-ubuntu-latest-rl-minimum, backend-ubuntu-latest-rl-current

Evidence: [NEURAL_RL.md](../docs/NEURAL_RL.md), [2026-11-23-rl-coverage.json](../docs/benchmarks/2026-11-23-rl-coverage.json)

## rl.parallel

Masked parallel collection and centralized offline AWR

Areas: 19. Offline help: `nil help rl`.

APIs: `nil.rl.collect_multiagent`, `nil.rl.centralized_fit`

Input: Explicit parallel environment/policies or aligned agent datasets and team returns

Output: Per-agent arrays or CentralizedResult(policies/critic/history/checkpoint/status)

Limits: Fixed-cohort offline AWR is not online MAPPO; truncation requires authoritative terminal metadata

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: backend-ubuntu-latest-rl-minimum, backend-ubuntu-latest-rl-current

Evidence: [NEURAL_RL.md](../docs/NEURAL_RL.md), [2026-11-23-rl-coverage.json](../docs/benchmarks/2026-11-23-rl-coverage.json)

## generative.pretrained_cpu

Pinned pretrained CPU inference and LoRA

Areas: 13. Offline help: `nil help generative`.

APIs: `nil.generative.load_causal_lm`, `nil.generative.generate`, `nil.generative.lora`

Input: Explicit pinned local safetensors snapshot/tokenizer; token windows and adapter config

Output: Native model/tokenizer, generated text, NLL reports and adapter artifacts

Limits: Declared134.5M SmolLM2 target; no download at install/import; no billion/GPU/all-architecture acceptance

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [PRETRAINED_CPU.md](../docs/PRETRAINED_CPU.md), [2026-11-23-pretrained-135m.json](../docs/benchmarks/2026-11-23-pretrained-135m.json)

## optimization.cpu_quantization

CPU quantization and INT8 weight storage

Areas: 25. Offline help: `nil help optimization`.

APIs: `nil.optimization.quantize`

Input: Compatible CPU model; explicit method, exclusions and numerical tolerance

Output: Owned quantized model preserving caller state

Limits: Weight-only mode uses FP32 compute; dynamic activation INT8 failed the declared SmolLM2 tolerance; no automatic speedup

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: backend-ubuntu-latest-torch-current, backend-ubuntu-latest-torch-minimum

Evidence: [PRETRAINED_CPU.md](../docs/PRETRAINED_CPU.md), [2026-11-23-model-coverage.json](../docs/benchmarks/2026-11-23-model-coverage.json)

## execution.owned_workers

Owned process-tree execution

Areas: 1, 29. Offline help: `nil help execution`.

APIs: `nil.execution.run`, `nil.execution.submit`

Input: Explicit trusted worker target, data and resource/deadline limits

Output: Result or typed execution error with durable status/recovery record

Limits: Trusted-code ownership, not hostile-code sandbox; Windows requires creation-time Job Object support

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux", "Darwin", "Windows"], "python": ["3.10", "3.11", "3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [WINDOWS_EXECUTION.md](../docs/WINDOWS_EXECUTION.md), [BACKEND_COMPATIBILITY.md](../docs/BACKEND_COMPATIBILITY.md)

## execution.linux_memory

Linux kernel memory containment

Areas: 5, 29. Offline help: `nil help execution`.

APIs: `nil.execution.Limits`, `nil.execution.run`

Input: Explicit delegated cgroup-v2 subtree and job limits

Output: Kernel-contained job plus memory/OOM evidence

Limits: Delegation required; no global controller mutation or hard GPU quota claim

Prerequisites: Owned delegated Linux cgroup-v2 subtree

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [DEPLOYMENT.md](../docs/DEPLOYMENT.md), [2026-11-22-cgroup-kernel.json](../docs/benchmarks/2026-11-22-cgroup-kernel.json)

## serving.managed_cpu

Managed single-tenant TLS inference reference

Areas: 27, 29. Offline help: `nil help serving`.

APIs: `nil.serving.create_gateway`, `nil.serving.client`, `nil.serving.create_isolated_server`

Input: Trusted CPU model, explicit credentials/TLS and bounded request budgets

Output: Isolated inference, authenticated gateway/client and aggregate metrics

Limits: Gateway/client themselves use stdlib only; reference is not audited public/multi-tenant ingress

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: backend-ubuntu-latest-torch-current, backend-ubuntu-latest-torch-minimum

Evidence: [DEPLOYMENT.md](../docs/DEPLOYMENT.md), [test_tls_inference.py](../nil-sdk/tests/test_tls_inference.py)

## gpu.mixed_precision

Real CUDA AMP acceptance

Areas: 5, 10. Offline help: `nil help training`.

APIs: `nil.train`

Input: Suitable explicitly provisioned target

Output: Acceptance report only after real target execution

Limits: Actual CUDA FP16/BF16/overflow/resume acceptance not run here

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cuda", "os": [], "python": []}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [PHASES.md](../docs/project/PHASES.md)

## gpu.distributed

Real GPU distributed/sharded acceptance

Areas: 24. Offline help: `nil help distributed`.

APIs: `nil.distributed.train`

Input: Suitable explicitly provisioned target

Output: Acceptance report only after real target execution

Limits: Requires two supported GPUs/NCCL; CPU passes do not certify this target

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cuda", "os": [], "python": []}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [PHASES.md](../docs/project/PHASES.md)

## gpu.native_engine

Independent GPU operators and hard VRAM containment

Areas: 4, 5. Offline help: `nil help native`.

APIs: No accepted public workflow.

Input: Suitable explicitly provisioned target

Output: Acceptance report only after real target execution

Limits: Device-specific kernels/aggregate VRAM quotas are not implemented/validated

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cuda", "os": [], "python": []}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [PHASES.md](../docs/project/PHASES.md)

## hardware.old_pc

Physical older-PC acceptance

Areas: 5, 32. Offline help: `nil help coverage`.

APIs: No accepted public workflow.

Input: Suitable explicitly provisioned target

Output: Acceptance report only after real target execution

Limits: Constrained modern-host benchmarks do not certify old ISA/cache/storage

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "target-specific", "os": [], "python": []}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [PHASES.md](../docs/project/PHASES.md)

## serving.public

Public or multi-tenant deployment acceptance

Areas: 27, 29. Offline help: `nil help serving`.

APIs: No accepted public workflow.

Input: Suitable explicitly provisioned target

Output: Acceptance report only after real target execution

Limits: Actual deployment host, authorization/revocation and workload SLO acceptance remain open

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "target-specific", "os": [], "python": []}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [PHASES.md](../docs/project/PHASES.md)

## vision.classification

Bounded local image classification and checkpoint inference

Areas: 6, 9, 12, 15, 26. Offline help: `nil help vision`.

APIs: `nil.vision.ImageTransform`, `nil.vision.image_folders`, `nil.vision.classify`

Input: Local root/class/file images; shared class order; EXIF/RGB/white-alpha/bilinear/CHW float32 preprocessing with byte/pixel/inventory limits.

Output: Lazy map dataset, neural TrainingResult, held-out loss/accuracy, JSON preprocessing/label manifest, checkpoint and streamed labels/probabilities.

Limits: 1000 training/300 held-out real digits; 89.3% accuracy, not natural-image accuracy. Default 16MiB encoded/4M source pixels per image; no decoded cache. Detection/segmentation/OCR/tracking, pretrained backbones, export and serving acceptance are open.

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: local-vision-cpu-current

Evidence: [VISION.md](../docs/VISION.md), [2026-11-24-vision-digits.json](../docs/benchmarks/2026-11-24-vision-digits.json), [test_vision.py](../nil-sdk/tests/test_vision.py)

## sdk.result_contract

Backend-neutral result metadata and neural result dispatch

Areas: 1, 30. Offline help: `nil help contracts`.

APIs: `nil.contracts.result_info`, `nil.contracts.MapDataset`, `nil.contracts.Predictor`, `nil.contracts.SingleModelResult`

Input: Supported native, classical, neural, online/offline/centralized RL training outcomes; numerical backends only to construct those outcomes.

Output: Schema-1 ResultInfo metadata (family, status, counts, paths, steps), without weights; prediction/evaluation also accept neural TrainingResult.

Limits: Structural protocols do not validate backend types or tensor shapes. Family-specific data, callbacks, cancellation and persistence remain explicit; alpha contract.

Prerequisites: No additional declared prerequisites.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [SDK_CONTRACTS.md](../docs/SDK_CONTRACTS.md), [test_framework_contracts.py](../nil-sdk/tests/test_framework_contracts.py), [test_neural_result_contract.py](../nil-sdk/tests/test_neural_result_contract.py), [test_classical_result_contract.py](../nil-sdk/tests/test_classical_result_contract.py), [test_rl_control.py](../nil-sdk/tests/test_rl_control.py), [test_rl_centralized.py](../nil-sdk/tests/test_rl_centralized.py), [test_offline_neural_rl.py](../nil-sdk/tests/test_offline_neural_rl.py)

## rag.lexical

Cited lexical retrieval and explicit-generator RAG

Areas: 14. Offline help: `nil help rag`.

APIs: `nil.rag.Index`, `nil.rag.evaluate`, `nil.rag.load_encoder`, `nil.rag.semantic_index`, `nil.rag.evaluate_answers`

Input: Unicode document text with source IDs, bounded chunking, questions and optional explicit generator callback.

Output: Ranked provenance/citations, extractive or callback answers, abstention, retrieval metrics and immutable index reload.

Limits: Dense/hybrid retrieval and explicit reranking are bounded CPU additions; entailment, database-scale storage and multi-writer ingestion remain open. See ADVANCED_WORKFLOWS.md for the new local CPU scope; earlier v1 hosted results do not certify these additions.

Prerequisites: Advanced local CPU additions have separate tests/evidence; optional model backends remain explicit.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [SEVEN_WORKFLOWS.md](../docs/SEVEN_WORKFLOWS.md), [2026-11-24-seven-workflows.json](../docs/benchmarks/2026-11-24-seven-workflows.json), [test_seven_core.py](../nil-sdk/tests/test_seven_core.py), [ADVANCED_WORKFLOWS.md](../docs/ADVANCED_WORKFLOWS.md), [2026-11-24-advanced-workflows.json](../docs/benchmarks/2026-11-24-advanced-workflows.json)

## agents.tool_plans

Approved tool plans, owned deadlines and durable prefix recovery

Areas: 20. Offline help: `nil help planning`.

APIs: `nil.planning.run`, `nil.planning.shortest_path`, `nil.planning.agent`, `nil.planning.reconcile`

Input: Ordered tool plan, existing Tool schemas, explicit allowed names/versions and journal path.

Output: Committed plain-JSON trace, bounded owned calls, cancellation/deadlines, explicit safe-prefix resume and shortest paths.

Limits: Bounded caller-supplied policy loop; per-call deadlines and explicit verified-result reconciliation. Exclusive caller ownership required; no general autonomous reasoning or automatic external-effect recovery. See ADVANCED_WORKFLOWS.md for the new local CPU scope; earlier v1 hosted results do not certify these additions.

Prerequisites: Advanced local CPU additions have separate tests/evidence; optional model backends remain explicit.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [SEVEN_WORKFLOWS.md](../docs/SEVEN_WORKFLOWS.md), [2026-11-24-seven-workflows.json](../docs/benchmarks/2026-11-24-seven-workflows.json), [test_seven_core.py](../nil-sdk/tests/test_seven_core.py), [ADVANCED_WORKFLOWS.md](../docs/ADVANCED_WORKFLOWS.md), [2026-11-24-advanced-workflows.json](../docs/benchmarks/2026-11-24-advanced-workflows.json)

## probability.causal

Conjugate posteriors and cross-fitted binary-treatment AIPW

Areas: 22. Offline help: `nil help probabilistic`.

APIs: `nil.probabilistic.BetaBernoulli`, `nil.probabilistic.BayesianLinear`, `nil.probabilistic.causal_ate`, `nil.probabilistic.metropolis`, `nil.probabilistic.load_metropolis`, `nil.probabilistic.instrumental_ate`

Input: Binary outcomes for Beta; Gaussian linear data; or bounded features/binary treatment/continuous outcome with identification assumptions.

Output: Posterior predictive statistics and JSON reload, or causal ATE/asymptotic interval with explicit assumptions and fold assignment.

Limits: Random-walk MCMC/basic diagnostics and linear IV assumptions only; no HMC/NUTS, full probabilistic programming or causal discovery. See ADVANCED_WORKFLOWS.md for the new local CPU scope; earlier v1 hosted results do not certify these additions.

Prerequisites: Advanced local CPU additions have separate tests/evidence; optional model backends remain explicit.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [SEVEN_WORKFLOWS.md](../docs/SEVEN_WORKFLOWS.md), [2026-11-24-seven-workflows.json](../docs/benchmarks/2026-11-24-seven-workflows.json), [test_seven_classical.py](../nil-sdk/tests/test_seven_classical.py), [ADVANCED_WORKFLOWS.md](../docs/ADVANCED_WORKFLOWS.md), [2026-11-24-advanced-workflows.json](../docs/benchmarks/2026-11-24-advanced-workflows.json)

## learning.regimes

Active/semi/self-supervised, prototype and bounded replay workflows

Areas: 23. Offline help: `nil help learning`.

APIs: `nil.learning.query_uncertain`, `nil.learning.self_training`, `nil.learning.prototypes`, `nil.learning.contrastive_loss`, `nil.learning.ReplayMemory`, `nil.learning.adapt`, `nil.learning.continual_metrics`

Input: Explicit labeled/unlabeled pools, paired views, support/query embeddings or bounded JSON stream records.

Output: Budgeted query indices, fitted native self-training model, differentiable NT-Xent, few-shot predictions and exact replay continuation.

Limits: Functional MAML inner loop and measured matrix summaries, not an end-to-end continual/meta-learning engine. See ADVANCED_WORKFLOWS.md for the new local CPU scope; earlier v1 hosted results do not certify these additions.

Prerequisites: Advanced local CPU additions have separate tests/evidence; optional model backends remain explicit.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [SEVEN_WORKFLOWS.md](../docs/SEVEN_WORKFLOWS.md), [2026-11-24-seven-workflows.json](../docs/benchmarks/2026-11-24-seven-workflows.json), [test_seven_classical.py](../nil-sdk/tests/test_seven_classical.py), [ADVANCED_WORKFLOWS.md](../docs/ADVANCED_WORKFLOWS.md), [2026-11-24-advanced-workflows.json](../docs/benchmarks/2026-11-24-advanced-workflows.json)

## audio.classification

PCM WAV and audio classification frontend

Areas: 16. Offline help: `nil help audio`.

APIs: `nil.audio.load_wav`, `nil.audio.resample`, `nil.audio.segment`, `nil.audio.features`, `nil.audio.load_asr`, `nil.audio.load_tts`, `nil.audio.mono`, `nil.audio.save_wav`, `nil.audio.word_error_rate`

Input: Bounded local uncompressed integer PCM WAV, explicit sample rate/channel/duration and FFT configuration.

Output: Channel-first waveform, explicit linear resampling, aligned segments and streaming-summary spectral features for classifiers.

Limits: Explicit local native ASR/VITS models require Transformers/Torch; polyphase resampling requires SciPy. Streaming, fine-tuning and multilingual quality acceptance remain open. See ADVANCED_WORKFLOWS.md for the new local CPU scope; earlier v1 hosted results do not certify these additions.

Prerequisites: Advanced local CPU additions have separate tests/evidence; optional model backends remain explicit.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [SEVEN_WORKFLOWS.md](../docs/SEVEN_WORKFLOWS.md), [2026-11-24-seven-workflows.json](../docs/benchmarks/2026-11-24-seven-workflows.json), [test_seven_media.py](../nil-sdk/tests/test_seven_media.py), [ADVANCED_WORKFLOWS.md](../docs/ADVANCED_WORKFLOWS.md), [2026-11-24-advanced-workflows.json](../docs/benchmarks/2026-11-24-advanced-workflows.json)

## video.multimodal

Timed visual clips, owned video decoding and fixed-schema fusion

Areas: 17. Offline help: `nil help multimodal`.

APIs: `nil.multimodal.load_clip`, `nil.multimodal.load_video`, `nil.multimodal.aligned_audio`, `nil.multimodal.fuse_features`, `nil.multimodal.load_encoder`

Input: Local GIF/TIFF or explicit FFmpeg video; declared frames/timelines/audio coverage and modality feature dimensions.

Output: Bounded RGB clips/timestamps, aligned audio segments and fused fixed-order vectors/presence masks with checkpointed classifier examples.

Limits: Explicit local native CLIP requires Transformers; original PTS is optional and refuses unavailable duration/timestamps. Semantic video-language training and real-world quality acceptance remain open. See ADVANCED_WORKFLOWS.md for the new local CPU scope; earlier v1 hosted results do not certify these additions.

Prerequisites: Explicit FFmpeg/ffprobe installation for load_video only; Advanced local CPU additions have separate tests/evidence; optional model backends remain explicit.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [SEVEN_WORKFLOWS.md](../docs/SEVEN_WORKFLOWS.md), [2026-11-24-seven-workflows.json](../docs/benchmarks/2026-11-24-seven-workflows.json), [test_seven_media.py](../nil-sdk/tests/test_seven_media.py), [ADVANCED_WORKFLOWS.md](../docs/ADVANCED_WORKFLOWS.md), [2026-11-24-advanced-workflows.json](../docs/benchmarks/2026-11-24-advanced-workflows.json)

## graphs.classification

Homogeneous GCN node and graph classification

Areas: 21. Offline help: `nil help graphs`.

APIs: `nil.graphs.gcn`, `nil.graphs.propagate`, `nil.graphs.batch`, `nil.graphs.pool`, `nil.graphs.save`, `nil.graphs.load`, `nil.graphs.weighted_gcn`, `nil.graphs.weighted_propagate`, `nil.graphs.negative_edges`, `nil.graphs.split_links`, `nil.graphs.link_logits`, `nil.graphs.sample_neighbors`

Input: Finite float32/64 homogeneous node features, int64 endpoints, caller-owned labels and explicit train/test masks.

Output: Sparse normalized propagation, disjoint batches, differentiable node/graph logits, pooled features, JSON graphs and neural checkpoints.

Limits: Weighted/directed graphs, bounded CPU neighbor sampling and leakage-aware link splits added. Heterogeneous/temporal graphs and distributed loading remain open. See ADVANCED_WORKFLOWS.md for the new local CPU scope; earlier v1 hosted results do not certify these additions.

Prerequisites: Advanced local CPU additions have separate tests/evidence; optional model backends remain explicit.

Targets: `[{"device": "cpu", "os": ["Linux"], "python": ["3.12"]}]`

Recorded profiles: See workflow-specific evidence; no backend-matrix acceptance claimed.

Evidence: [SEVEN_WORKFLOWS.md](../docs/SEVEN_WORKFLOWS.md), [2026-11-24-seven-workflows.json](../docs/benchmarks/2026-11-24-seven-workflows.json), [test_seven_neural.py](../nil-sdk/tests/test_seven_neural.py), [ADVANCED_WORKFLOWS.md](../docs/ADVANCED_WORKFLOWS.md), [2026-11-24-advanced-workflows.json](../docs/benchmarks/2026-11-24-advanced-workflows.json)

## Recorded backend profiles

These profiles describe historical verification environments. Native workflows
need no numerical backend even when their reference tests ran in a Torch environment.

- **backend-macos-latest-classical-current**: Python 3.12.10; macOS-26.7.2-arm64-arm-64bit; nil-sdk=0.2.1, scikit-learn=1.9.1, numpy=2.5.5, scipy=1.18.1, pandas=3.1.1, pytest=9.2.0. [Evidence](../docs/benchmarks/2026-11-23-phase8-9-backends.json).
- **backend-macos-latest-classical-minimum**: Python 3.12.10; macOS-26.7.2-arm64-arm-64bit; nil-sdk=0.2.1, scikit-learn=1.4.2, numpy=1.26.4, scipy=1.13.1, pandas=2.3.3, pytest=9.2.0. [Evidence](../docs/benchmarks/2026-11-23-phase8-9-backends.json).
- **backend-ubuntu-latest-torch-current**: Python 3.12.16; Linux-7.0.0-1020-azure-x86_64-with-glibc2.43; nil-sdk=0.2.1, torch=2.15.0+cpu, numpy=2.5.5, pytest=9.2.0. [Evidence](../docs/benchmarks/2026-11-23-phase8-9-backends.json).
- **backend-windows-latest-classical-minimum**: Python 3.12.10; Windows-2025Server-10.0.26100-SP0; nil-sdk=0.2.1, scikit-learn=1.4.2, numpy=1.26.4, scipy=1.13.1, pandas=2.3.3, pytest=9.2.0. [Evidence](../docs/benchmarks/2026-11-23-phase8-9-backends.json).
- **backend-ubuntu-latest-torch-minimum**: Python 3.12.16; Linux-7.0.0-1016-aws-x86_64-with-glibc2.43; nil-sdk=0.2.1, torch=2.6.0+cpu, numpy=1.26.4, pytest=9.2.0. [Evidence](../docs/benchmarks/2026-11-23-phase8-9-backends.json).
- **backend-ubuntu-latest-rl-minimum**: Python 3.12.16; Linux-7.0.0-1020-azure-x86_64-with-glibc2.43; nil-sdk=0.2.1, torch=2.6.0+cpu, numpy=1.26.4, gymnasium=1.0.0, stable-baselines3=2.4.0, sb3-contrib=2.4.0, pytest=9.2.0. [Evidence](../docs/benchmarks/2026-11-23-phase8-9-backends.json).
- **backend-ubuntu-latest-classical-minimum**: Python 3.12.16; Linux-7.0.0-1020-azure-x86_64-with-glibc2.43; nil-sdk=0.2.1, scikit-learn=1.4.2, numpy=1.26.4, scipy=1.13.1, pandas=2.3.3, pytest=9.2.0. [Evidence](../docs/benchmarks/2026-11-23-phase8-9-backends.json).
- **backend-ubuntu-latest-rl-current**: Python 3.12.16; Linux-7.0.0-1020-azure-x86_64-with-glibc2.43; nil-sdk=0.2.1, torch=2.15.0+cpu, numpy=2.5.5, gymnasium=1.4.0, stable-baselines3=2.9.2, sb3-contrib=2.9.2, pytest=9.2.0. [Evidence](../docs/benchmarks/2026-11-23-phase8-9-backends.json).
- **backend-ubuntu-latest-classical-current**: Python 3.12.16; Linux-7.0.0-1016-aws-x86_64-with-glibc2.43; nil-sdk=0.2.1, scikit-learn=1.9.1, numpy=2.5.5, scipy=1.18.1, pandas=3.1.1, pytest=9.2.0. [Evidence](../docs/benchmarks/2026-11-23-phase8-9-backends.json).
- **backend-windows-latest-classical-current**: Python 3.12.10; Windows-2025Server-10.0.26100-SP0; nil-sdk=0.2.1, scikit-learn=1.9.1, numpy=2.5.5, scipy=1.18.1, pandas=3.1.1, pytest=9.2.0. [Evidence](../docs/benchmarks/2026-11-23-phase8-9-backends.json).
- **local-vision-cpu-current**: Python 3.12.16; Linux-7.3.5-arch1-1-x86_64-with-glibc2.44; nil-sdk=0.2.1, torch=2.15.0+cpu, numpy=2.5.4, Pillow=13.0.0. [Evidence](../docs/benchmarks/2026-11-24-vision-digits.json).
