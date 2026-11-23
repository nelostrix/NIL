# NIL remaining development phases

This is the ordered follow-on to TODO.md. A checked implementation task means its
stated software tests passed. Hardware, OS and deployment acceptance are separate
checkboxes: writing a runner or encountering a skip does not pass them. Preserve
NIL-only installation, passive imports, shallow public modules and per-module
help. Existing optional integrations remain available while native work expands.

## Phase 1 — Execution containment and resource budgets (active)

- [x] Add optional Linux process-group RSS monitoring with typed resource-limit
  failure, durable peak/limit evidence and owned descendant cleanup.
- [x] Add explicit cgroup-v2 request configuration and fail-closed preflight;
  invalid-root and actual missing-delegation refusal checks pass. Worker
  attachment and kernel containment passed the delegated-unit acceptance below.
- [x] Test normal completion, limit breach, cancellation, descendant cleanup,
  missing permissions and supervisor survival after client exit.
- [x] Publish a dependency-free platform acceptance runner with required-capability
  preflight; unsupported targets must fail acceptance rather than silently skip.
- [x] Validate worker attachment, kernel memory/swap/group-OOM limits and
  escaped-session descendant cleanup in an owned delegated transient user unit.
  All eight real kernel tests pass; recorded peak reaches the 128 MiB job cap.
- [x] Implement Windows Job Object process-tree ownership, creation-time assignment,
  handle cleanup and portable launch/failure contracts.
- [x] Run actual Windows kernel acceptance on Python 3.10, 3.11 and 3.12;
  installed-wheel acceptance and native Job Object tests passed in hosted CI.
- [x] Run actual macOS timeout/cancel/process-tree acceptance on Python 3.10–3.12.
- [ ] Specify and validate device-specific aggregate GPU memory containment;
  allocator preferences alone must not be advertised as hard GPU quotas.

Exit: publish software and real-platform evidence separately. Full cross-platform
and hard aggregate-memory acceptance remains open until the target checks pass.

## Phase 2 — Performance and older-computer acceptance (local software validated)

- [x] Define small/medium workload sizes and explicit small-profile import,
  latency and peak-RSS budgets; preserve raw samples, host details and backend versions.
- [x] Extend the fresh-process benchmark runner with bounded time, per-sample
  failures, distribution summaries and machine-readable budget decisions.
- [x] Run core/native, classical and neural profiles on the current host.
- [ ] Run the same workload/budgets on an identified ten-year-old PC; publish raw
  samples and compare NIL wrapper overhead against the corresponding backend.

Exit: current-host evidence can pass independently; old-PC certification cannot.

## Phase 3 — Actual CUDA mixed-precision training (runner built; GPU gate open)

- [x] Build a required-device acceptance runner and GPU tests; missing hardware
  refusal is tested on this CPU-only host.
- [ ] Accept actual GPU/driver/Torch reports and positive CUDA test results.
- [ ] Exercise FP16/BF16 learning, overflow/GradScaler adaptation, clipping after
  unscale, accumulation and exact supported checkpoint continuation on GPU.
- [ ] Publish measured memory/latency and unsupported dtype/device combinations.

## Phase 4 — GPU distributed training (target tests built; GPU gate open)

- [x] Add explicit two-GPU DDP/FSDP acceptance cases and per-worker memory
  diagnostics. CPU rank fault/deadline cleanup tests pass; GPU cases remain unrun.
- [ ] Run at least two real NCCL ranks with serial/global-batch parity, uneven
  batches, rank-zero callbacks and exact supported epoch resume.
- [ ] Test failed/hung ranks, timeout, group cleanup and last-valid checkpoint.
- [ ] Publish single-host/device restrictions; expand multi-host only with tests.

## Phase 5 — Large-model sharding

- [x] Implement explicit optional FSDP construction with bounded model/data policy.
  Real two-rank CPU FULL_SHARD, local partition counts and serial parity pass.
- [x] Implement sharded model/optimizer checkpoint publication and recovery.
  CPU exact resume, rank error, publication exception, tamper and data/world
  rejection pass; native partition metadata requires explicit trusted resume.
- [x] Test CPU FP32 distributed save/load and same-world/version contracts.
- [ ] Accept GPU sharded save/load, supported precision and rank failure recovery.
- [ ] Measure actual GPU peak memory with/without sharding and checkpointing.

## Phase 6 — Production service acceptance

- [x] Define a managed single-tenant Linux/CPU target and warm inference budgets
  (four clients, zero failures, observed p95 <=250 ms for the declared affine model).
- [x] Add explicit service lifecycle/systemd examples, TLS/mTLS gateway, protected
  credentials and bounded HTTPS client retries. No additional base dependency.
- [x] Test the managed reference: stalled TLS/body deadlines, framing, native
  worker crash/restart, signal shutdown and 30-second load under real 1 GiB,
  zero-swap, one-CPU and 128-task kernel limits. 3,141 requests, zero failures,
  observed p95 about 91 ms. Public Internet ingress and arbitrary-model SLOs remain open.
- [x] Validate managed-reference observability, secret handling and declared Linux
  CPU deployment matrix. Authenticated counters, periodic/final snapshots,
  descriptor-validated credentials and sanitized CLI failures pass. Public ingress
  and other service targets remain separate acceptance.

## Phase 7 — Release compatibility matrix

- [x] Define isolated Python 3.10–3.12 base-wheel CI checks on Linux/macOS/Windows.
  Fresh base-wheel installation and execution passed in all nine remote cells.
- [x] Run and publish all nine platform/Python matrix cells.
- [x] Add tested minimum/current classical and Torch CPU profiles and installed
  examples on Linux/Python3.12. All four hosted cells pass; other extras remain open.
- [x] Define manual-only GPU acceptance on explicitly provisioned trusted runners.
  GPU job execution remains pending; no workflow was dispatched by this session.
- [x] Publish verification wheel/bundle artifacts and evidence for all nine passing
  cells; indexed SHA-256 digests and CI retention are recorded. Stable release/tag
  publication is separate; failed first-run evidence is retained.

## Phase 8 — Expand the independent numerical engine

- [x] Define benchmarked stages beyond the existing CPU float64 reference engine:
  N-dimensional shape/stride semantics, dtype policy and parameter/state contracts.
- [x] Add selected dense operators and optimizers with independent numerical
  gradients, persistence and reference-backend parity before expanding further.
- [ ] Add sparse representations/operators and explicit unsupported combinations.
- [ ] Add compiled/vectorized CPU kernels before claiming large-array performance.
- [ ] Add device/kernel management and GPU operators only with real-device tests.
- [x] Map which workflows can run using NIL alone; do not present adapters as
  independent engines or force optional dependencies into base installation.

Phase8 scope: row-wise stable softmax/log-softmax, categorical cross entropy and
Adam with exact model/optimizer continuation. Finite differences and Torch float64
parity pass. ND shape/stride, sparse, compiled CPU and GPU stages are defined in
[the engine contract](../NATIVE_ENGINE.md), not implemented by these checkmarks.

## Phase 9 — Broader RL and model compatibility

- [ ] Implement/test masked parallel multi-agent collection including terminal
  masks, heterogeneous spaces and recurrent state lifecycle.
- [ ] Add a separately scoped centralized-critic multi-agent learning algorithm
  with real environments and multi-seed baseline comparisons.
- [ ] Extend exact simulator-state continuation beyond the existing CPU DQN
  contract only for explicitly supported env/wrapper/algorithm combinations.
- [ ] Add offline CQL/IQL-style algorithms if required by the declared coverage
  target, with dataset/OOD-action contracts and real held-out learning evidence.
- [ ] Broaden RL learning benchmarks beyond the tiny bandit to discrete,
  continuous, recurrent/masked and multi-agent tasks with seed-level intervals.
- [ ] Broaden pretrained-model/LoRA/quantization/export cases and measure realistic
  datasets/large models on documented supported hardware.

Every phase must update TODO.md with exact commands, outcomes, versions and known
limits before its acceptance checkbox can be closed. Do not infer a percentage
of global library coverage from a constructor count or these phase checkboxes.


## Local batch evidence — 2026-11-22

Phase 1 Linux software and Phase 2 portable software are implemented. Actual
nondelegated-kernel, other-OS, GPU and older-target acceptance remains open.
Initial resource/runners/help acceptance: 24 passed/three explicit skips (two
real-kernel tests without delegation, one unrelated missing Torch test). Later
full-suite and installed-wheel validation is recorded in TODO.md. Raw local
execution and small/medium classical/neural reports plus commands are in
[docs/benchmarks/2026-11-22-phases-1-2.md](../benchmarks/2026-11-22-phases-1-2.md).

Phase 3 is the next ordered development target. This host cannot run its actual
CUDA acceptance; the existing verifier must reject --cuda here. No phase is
closed merely because its runner exists.


Final current batch acceptance: full classical **1045 passed/17 skips**; full
neural/RL **979 passed/15 skips**. Installed execution/resource/runner checks
outside the source tree: **20 passed/three explicit skips**. Installed stdlib
acceptance runner passes actual completion/deadline/cancel/child/RSS checks.
The clean offline base resolver adds only nil-sdk and runs the new RSS path.
195 Python sources match the tested wheel; all 103 help guides and this phase
list ship in the bundle. Other-target acceptance remains unchanged.


## Sharding and target-gate implementation — 2026-11-22

Focused CPU sharding/DDP/runner acceptance: **12 passed**. Includes actual two-rank
FSDP serial parity/exact resume, rank and publication faults, hash/data/world
preflight, DDP hung-worker deadline and caller immutability. New GPU cases are
executable target tests, not GPU passes. The current-host GPU report must fail
with exit 2. CI definitions are unexecuted configurations, not matrix acceptance.
Limits and commands: [docs/benchmarks/2026-11-22-sharding-gates.md](../benchmarks/2026-11-22-sharding-gates.md).


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


## Windows containment implementation — 2026-11-22

The Windows backend is implemented using ctypes and CPython _winapi only; no
third-party package is added. Suspended launch/assignment and inherited-handle
policy are documented in docs/WINDOWS_EXECUTION.md. Required Windows acceptance
now replaces the earlier unsupported-feature refusal check in CI. Windows kernel
tests are present, but this Linux host cannot run them. Keep that gate open.


Final creation-time launch contracts/help/runner checks: **20 passed**. Installed
Linux execution/resource/portable-Windows contracts outside the source tree:
**26 passed, three explicit skips**. Installed stdlib execution acceptance passes
including successful/timeout child cleanup and the RSS guard. Fresh offline
installation adds only nil-sdk. Both development environments use the rebuilt
regular wheel; 197 Python sources match and 103 help guides are bundled.
Native Windows acceptance is still unrun; ten real-kernel cases and three Windows
CI cells require a target host. The Linux --require-windows report fails with
exit 2, so portable test success is not mistaken for Windows certification.


## Approved CI and workspace cleanup — 2026-11-23

Published verification branch codex/verify-nil-platforms-20261122 with user approval.
Initial Linux base cells and Torch2.6 CPU development workflow passed. Windows
legacy-code-page CLI output, outdated preinstalled build tooling and macOS
zombie-group signalling exposed failures; targeted fixes are under a second CI run.
No Windows/macOS acceptance is checked before its required tests actually pass.

Workspace validation logs/scratch exports now live under ../artifacts from the
workspace root; audit reports are under docs/audits and planning/checklists under
docs/project. The project root retains README, source/packaging entry points and
standard community files. Old generated outputs were moved with a recovery map,
not deleted. README examples execute successfully in their declared backend envs.

## Hosted platform acceptance — 2026-11-23

[Platform run 45113943448](https://github.com/nelostrix/NIL/actions/runs/45113943448)
passed all nine Linux/macOS/Windows × Python 3.10–3.12 cells at commit c663114.
Each cell builds a fresh NIL-only wheel and runs actual owned-worker acceptance;
Windows also runs native Job Object and launcher tests against the installed wheel.
[CPU SDK run 45113943452](https://github.com/nelostrix/NIL/actions/runs/45113943452)
also passed. These results cover the base SDK and declared CPU workflows, not GPU
acceptance or every optional backend/platform combination.

## Native engine and installed backend acceptance — 2026-11-23

[Backend run45116372373](https://github.com/nelostrix/NIL/actions/runs/45116372373)
passed all four minimum/current profiles at 7fc1943. Each Torch CPU cell passed
70 tests, zero skips, and three architecture examples; classical cells passed
67/69 tests, zero skips, and two examples each. This includes the native Adam and
multiclass additions. [Base run45116372289](https://github.com/nelostrix/NIL/actions/runs/45116372289)
also passed all nine target cells with dependency-free multiclass training in the
fresh-wheel verifier. Local full neural/RL source regression: 1026 passed/30
explicit skips. Local final regular-wheel checks: 70 Torch and 67 minimum-classical
tests plus installed examples, all with zero skips. No base dependency was added.
