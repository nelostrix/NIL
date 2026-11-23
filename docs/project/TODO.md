# NIL completion checklist

Ordered follow-on work: [PHASES.md](PHASES.md). Phase 1 is active; target-specific
acceptance remains separate from implementation and local test completion.

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
