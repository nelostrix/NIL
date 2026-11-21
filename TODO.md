# NIL completion checklist

This is the live implementation and validation checklist. Work proceeds in the
numbered order below. An item is checked only after its stated acceptance gate
passes; optional-backend adapters are labelled as integrations, not NIL-native
numerical engines. Hardware/platform checks remain open until actually run.

Every public module needs its own help.md. Every batch must retain a NIL-only
base wheel installation, lazy imports, honest capability reports and a tested
distribution artifact. No blanket claim of complete AI/DL/RL coverage is made.

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
- [ ] FSDP/large-model sharding and activation checkpointing. Gate: measured
  memory behavior and distributed save/load on supported hardware.
- [ ] Generative/LoRA, quantization and export compatibility matrix. Gate: actual
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
- [ ] Offline RL and neural imitation learning (not action-frequency tables).
  Gate: dataset schema/mask tests and reproducible held-out learning checks.
- [ ] Vectorized/recurrent/masked/multi-agent RL contracts. Gate: episode boundary,
  recurrent-state reset, action-mask and actual multi-agent environment tests.
- [ ] RL checkpoint/environment-state/replay/RNG continuation and cancellation.
  Gate: deterministic supported-environment resume and explicit unsupported cases.
- [ ] Multi-seed learning benchmarks with confidence intervals and real baselines.
  Gate: publish measured results; a smoke test is not a learning-quality claim.

## 4. Classical and specialized workflows — scoped acceptance

- [ ] Optional LightGBM/CatBoost adapters and native CPU pipeline acceptance.
- [ ] Forecasting: lag/horizon models, date features and rolling-origin backtests.
- [ ] Ranking: query-group validation and held-out NDCG/MAP metrics.
- [ ] Recommendation: sparse baselines/retrieval, bounded top-k and cold starts.
- [ ] Survival: event/time schema, censoring-aware training and held-out metrics.
- [ ] Bounded SHAP with transformed feature names and sparse/dense safeguards.
- [ ] Typed dataset snapshots, schema evolution and transactional stream offsets.
- [ ] Same-host distributed random-search quotas, stale-worker recovery and deterministic sampler continuation.
- [ ] Advanced uncertainty: known-ratio weighted, simultaneous multioutput, exchangeable-block and fixed-group methods,
  each with its own assumptions and statistical acceptance.

Each item here needs native functional/parity tests, failure tests, persistence
where applicable, help.md, import isolation and an installed-wheel example.

## 5. Independent engines and production acceptance

- [ ] Define the supported NIL-native numerical subset and benchmark target;
  implement/test it independently before claiming engine-free training.
- [ ] Expand native autodiff/tensors/operators/optimizers with numerical gradient
  checks, sparse/dtype/device behavior and published compatibility coverage.
- [ ] Production serving, concurrency/backpressure, observability and compatibility
  matrices. Gate: load/fault/security tests and deployment documentation.
- [ ] Complete hardware/OS/Python/backend matrix and reproducible release checks.

## Completion evidence

After each completed item record the implementation, exact test command/results,
platform/backend versions and remaining limits here before ticking its box.

### Latest full validation — 2026-11-21

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
Linux/Python 3.12 tested. Client-independent supervision and other platform/GPU
quotas are still unchecked; virtual address space is not aggregate RSS.

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
