# NIL 0.2 model-development implementation

Current acceptance update (2026-11-23): managed CPU TLS serving now has protected
credential reads and explicit periodic/final aggregate metrics. Installed
minimum/current classical and Torch CPU profiles pass; base Linux/macOS/Windows
Python3.10–3.12 acceptance passes. The independent float64 CPU engine adds stable
multiclass loss, softmax and transactional Adam with JSON continuation.
See [the live phases](PHASES.md), [backend profiles](../BACKEND_COMPATIBILITY.md)
and [native scope](../NATIVE_ENGINE.md) for current evidence and open gates.
The dated sections below are historical records, not current completeness claims.


Date: 2026-11-14. This change implements a usable PyTorch-native SDK release;
it does not claim that every capability in the earlier ten-area audit is complete.
Those audit reports remain the pre-change baseline.

## Delivered code

- Lazy public SDK imports, so importing core NIL no longer requires NumPy or
  constructs a shared default agent memory store. Existing agent tests retained.
- Native tensor/nn/optim/autograd access, MLP and small image-classifier builders.
- Local JSON/JSONL/CSV records, transforms, validation, reproducible splits,
  file fingerprints, TensorDataset and DataLoader helpers.
- Trainer with arbitrary scalar-loss steps, native optimizer/scheduler access,
  CPU/explicit device selection, gradient accumulation/clipping, validation,
  nonfinite checks, epoch checkpoints/resume and cooperative cancellation.
- State-dict checkpoints with restricted deserialization, integrity manifest,
  optimizer/scheduler/Torch/Python RNG restoration and resume-configuration checks.
- Real local byte-level BPE tokenizer and causal language-model construction,
  full-parameter training, optional PEFT LoRA, explicit local model loading,
  safetensors save/reload and generation. No automatic pretrained downloads.
- Actual CPU dynamic INT8 conversion for supported layers, measured CPU inference
  timing, ONNX graph export/validation and ONNX Runtime inference.
- Local run metadata/JSONL events, transactional version allocation and copied
  artifacts with SHA-256 verification; restore into caller-created architectures.
- Explicit optional MLflow run export adapter (not live-tested).
- Native sklearn fit/evaluation adapter, offline robotics policy construction and
  imitation-learning demonstrations.
- Token-authenticated loopback development inference service with bounded bodies
  and browser-origin rejection. No deployment started outside its temporary test.
- Nested tool schemas and argument validation before callable execution, decorated
  embodied-domain tools, finite/DOF/dt/current-state checks on joint filtering,
  and explicit IK failure instead of silently returning zero poses.
- CLI diagnostics/demo, two runnable model-development examples, optional extras,
  identical package metadata for both existing build roots, a wheel/docs/examples
  bundle builder and CPU acceptance CI configuration.

## Verification

Final installed-wheel acceptance: **839 tests passed**, including all 817 existing
suite tests and 22 new model-development tests; two non-failing warnings
(existing TestResult collection and PEFT Conv1D orientation adjustment).
The final test run completed in 13.84 seconds. The traceback/module paths confirm
imports from `/tmp/nil-dev-env/lib/python3.12/site-packages/nil`, not PYTHONPATH
source overrides.

The built wheel installed with `--no-deps` into a separate clean core environment;
core import, records and diagnostics worked with PyTorch/NumPy absent. The installed
CLI demo also completed and reduced regression MSE. All 154 SDK Python source
files parsed, package project metadata matched across both build roots,
`git diff --check` passed, and E/F/I lint checks passed on new model-development
modules, examples, tests and bundle builder.

Both the wheel and the wheel/docs/examples ZIP built successfully. The ZIP's
manifest was verified against every archived file. Dependencies/model weights
are not bundled. Setuptools emitted existing license-metadata deprecation
warnings; they did not prevent building. The added GitHub Actions workflow was
not run on a hosted CI service during this session.

Commands used during implementation:

```sh
PYTHONPATH=nil-sdk HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/nil-dev-env/bin/python -m pytest nil-sdk/tests -q
PYTHONPATH=nil-sdk /tmp/nil-dev-env/bin/python -m nil demo --output /tmp/nil-acceptance/demo
PYTHONPATH=nil-sdk /tmp/nil-dev-env/bin/python nil-sdk/examples/model_development.py \
  --output /tmp/nil-acceptance/models --onnx
PYTHONPATH=nil-sdk HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/nil-dev-env/bin/python nil-sdk/examples/local_language_model.py \
  --output /tmp/nil-acceptance/language --lora
/tmp/nil-dev-env/bin/python -m build --wheel --no-isolation --outdir /tmp/nil-built-wheels .
```

The serving test needed an approved execution outside the sandbox to create its
loopback socket. It uses an ephemeral port, checks denied/authorized requests,
then shuts down and joins the server thread. There was no physical hardware,
paid inference, model download, cloud job or external model deployment.
Dependency libraries were installed only in an isolated temporary environment.

Observed regression CLI MSE: 2.9632753 before training, 0.000527258 afterward.
The longer model-development example measured MSE 0.548397 before,
0.00000908383 after, and 0.0000674886 for its INT8 model. ONNX predictions ran.
These are tiny synthetic acceptance results, not generalization or speed claims.
The LoRA example's batch-mean loss decreased from 5.70117 to 5.65329 over four
short epochs and the saved/reloaded model generated text. This establishes the
execution path, not useful language-model quality.

The dedicated tests additionally verify exact CPU checkpoint/resume equivalence
with shuffle and dropout, actual quantized module replacement, ONNX/native output
agreement, full-parameter LM updates, finite-loss failure records, cancellation,
artifact corruption rejection, class/image metrics, sklearn regression, offline
imitation learning, nested argument rejection before side effects, IK failure,
empty datasets, and core import without optional dependencies.

Tested environment: Linux, Python 3.12.16, PyTorch 2.6.0+cpu, NumPy 2.5.4,
Transformers 4.57.6, PEFT 0.21.3, scikit-learn 1.9.1, ONNX 1.23.2,
ONNX Runtime 1.32.0 and pytest 9.2.0. Other OS/Python/GPU combinations are
not validated by this run. The broad extras ranges are compatibility intent,
not an assertion that every dependency combination was tested.

## Explicit scope and remaining work

This is the model-development core, not a complete replacement for PyTorch or a
completed universal AI/robotics platform. Native PyTorch/custom-step support lets
developers build other architectures, but specialized recipes still need their
own implementation and acceptance tests.

Not delivered here: TensorFlow-native trainer, distributed/AMP/FSDP/DeepSpeed
trainer, QLoRA/INT4/GPTQ/AWQ, RLHF/DPO trainer, production serving scheduler,
immutable dataset snapshots, annotation service, video/speech/OCR pipelines,
remote registry/object storage, hardware deployment acceptance or a new NELO
Studio UI adapter. MLflow live export, GPU acceleration and cross-platform
packaging were not tested. Python dependencies, drivers and model weights are
not bundled in the distribution ZIP.

Checkpoint recovery is at epoch boundaries. Custom loader/augmentation/NumPy RNG
state is not automatically captured. Cancellation is cooperative. The trainer
executes trusted native Python code with its process privileges. Local hashes
check integrity, not authenticity. The development server is CPU-only and
single-process; use an established serving backend for public/production use.

PhysicalSafetyGatekeeper enforces only joint position/velocity bounds. Torque,
collision, watchdog, emergency-stop and robot-specific constraints require an
independent control boundary. No autonomous hardware safety guarantee is made.

Earlier audit issues outside these new model-development paths still require
work, including IDE plan-mode propagation, root-daemon governance, split agent/UI
storage, fabricated legacy server/compute paths and advanced agent integration.
The default Agent now uses fresh memory/config objects while preserving implicit
continuous evolution and explicit evolution=None disabling behavior. Tool arguments
are now rejected rather than silently filtering unknown fields.

## How to use the result

See [the model-development guide](../MODEL_DEVELOPMENT.md). Build the ZIP with
`python scripts/build_sdk_bundle.py`. The package is an alpha development SDK;
its public modules expose real operations with explicit optional dependencies,
not claims of an installed or running backend based on UI labels.

## 0.2.1 quick additions

Added four bounded local capabilities: EarlyStopping callback with explicit
best-weight snapshots; sequential reproducible grid search with a preflight trial
limit and persisted failures/results; shifted/masked token-weighted causal-LM
perplexity; and CPU model footprint/latency profiling on a clone. Ten new offline
tests passed in the focused source test run. The language-model example now also
prints actual perplexity after save/reload. No new dependencies, model downloads,
cloud jobs or hardware operations were needed.

Installed-wheel regression verification: **849 tests passed** (839 previous plus
10 new), with the same two non-failing warnings, in 14.54 seconds. Tests imported
NIL 0.2.1 from the temporary environment's site-packages, without PYTHONPATH
source overrides. E/F/I lint and git diff whitespace checks passed. The updated
local LoRA example completed, reloaded its model and evaluated 344 next-token
targets with NLL 5.63621 and perplexity 280.397. This tiny synthetic example tests
the execution path, not useful language-model quality. The uniform-logit fixture
checks perplexity exactly against vocabulary size, including masked/left/right
padding cases and different batch sizes.

Early-stopping state resets per run; best.pt stores model weights and latest.pt
remains the training-resume checkpoint. Grid search is sequential and limited to
TrainingConfig fields, uses final-epoch metric selection and requires callers to
supply held-out validation data. Profiling measures a CPU clone's serialized
state size and inference latency; it does not measure GPU or peak RAM. These
features do not change the unfinished distributed-training or advanced LLM
quantization scope above.

## 2026-11-20 — Shallow catalog, modular imports and offline help

Added a plain registry with 17 native model constructors in four groups, explicit lazy custom-model registration, and an offline help reader/CLI. Every catalog model and public module topic has its own help.md: 48 files, included in package data. Backend construction is direct; no additional per-prediction wrapper, automatic dependency installation or plugin scanning is introduced. Training and robotics exports now resolve individually instead of importing their optional application subsystems eagerly.

Removed six unused/obsolete modules: model_router, chain_compressor, strategy_mutator, cache_manager, budget_controller and the canned nil.server bridge. The actual evolution pipeline, agents, neural/model workflows and real root daemon remain. The legacy Studio bundle bridge reports its removed NIL backend unavailable instead of generating a successful response. Simplified default requirements to the dependency-free core, removed obsolete verification checks, fixed the verification script filename/path, and replaced the duplicate entry-point feature claims with the current model-oriented guide. Historical audits retain their original evidence.

Verification in the available Python 3.12 CPU environment: before changes 843 passed, five skipped, one local-serving failure; after changes 855 passed, six skipped, the same local-serving failure. The failure is sandbox socket creation denied with PermissionError, not a new test regression. Catalog/help focused checks pass (12 passed, one sklearn test skipped). Two legacy bridge tests pass without sockets. A wheel-only Python -S acceptance confirms 17 model names, 48 offline help files, lazy Trainer/policy access, no framework imports and no files created. Packaging verifies every ZIP manifest hash, all wheel Python syntax, package metadata parity and absence of the six removed modules. Scikit-learn and XGBoost execution require their optional backends and were not validated in this environment; real PyTorch model training passed. The available environment lacks a wheel package, so packaging uses a read-only cached wheel distribution on PYTHONPATH rather than modifying that environment.

This remains a scoped alpha SDK: classical/native fit and neural training/evaluation/results are not yet one universal contract. See docs/SHALLOW_DESIGN.md and docs/MODULE_REFERENCE.md. Native engine computation still depends on model, data and hardware; shallow module organization alone does not guarantee execution speed.

## 2026-11-20 — Classical workflow extension (native acceptance pending)

Implemented the requested sequence: fitted numeric/categorical preprocessing and feature-selection pipelines; ClassicalResult with held-out metrics, warnings and compatibility dictionary access; complete trusted pipeline save/load with schema/version/integrity metadata; 28 additional constructor adapters (45 total); random/stratified/group/time-aware holdouts and fold-local cloned CV; count-bounded grid/random search with failure records and final refit; native/coefficient/permutation inspection, group-aware calibration and count-bounded learning curves. Top-level fit/pipeline/split/cross_validate/tune/save/load/transform functions keep the public structure shallow. Neural train/checkpoint behavior is retained, and evaluate/predict dispatch between classical and neural paths. Transform-only estimators use transform rather than fabricated predictive metrics.

Default CV/search uses one worker, slices each fold's features/weights once, and clones preprocessing inside folds. Search budgets are checked before sampling parameter distributions or fitting; counts include the final refit. Fit-count budgets are not hard runtime/memory limits. Native estimator constraints, optional backend dependencies and explicit serialization trust remain visible. The boosting extra now includes scikit-learn for the native wrapper/metric dependency. The core remains dependency-free and has no new automatic service or backend import.

Expanded per-model/module help to 80 guides, added the classical workflow guide, and added a runnable mixed-data/CV/tuning/inspection/save-reload example. Acceptance files cover leakage, missing/unseen categories, sample weighting, sparse and named data, native adapters, exact metric comparisons, artifact roundtrips/corruption, fold/group isolation, reproducibility, fit preflight limits, failed trials, calibration, importances, learning curves, and convergence reporting. Artifact rejection tests run without any model backend. scripts/verify_classical.py preflights a real sklearn installation before running acceptance; --full additionally requires pandas and XGBoost.

Available-environment verification: 863 passed, 10 skipped, one preexisting local-serving failure (sandbox denies socket creation). Focused artifact/core/help/neural checks: 29 passed, two optional checks skipped. Python syntax and project metadata parity pass. Core-only Python -S resolves all new APIs, lists 45 models and 80 help topics, loads no compute backend and creates no state. The sandbox lacks scikit-learn, and installing the isolated classical environment failed because PyPI DNS/network access is blocked. Accordingly the four native classical acceptance modules are skipped here: the new classical implementation is not runtime-verified. No mock estimator outputs are substituted for native acceptance. An interpreter with sklearn is still required to run scripts/verify_classical.py (or --full) and resolve any backend-specific failures. See docs/CLASSICAL_ML.md for exact contracts and limits.

## 2026-11-21 — Enforced NIL-only base installation

Made the zero-dependency base requirement explicit in both project metadata files.
Added scripts/verify_core_install.py and made bundle builds run it before creating
an artifact. The verifier checks the wheel metadata/payload, creates an isolated
temporary environment, inventories packages, installs offline with normal pip
dependency resolution, and requires exactly NIL to be added. It checks imports,
lazy APIs, offline help, CLI commands and missing-backend behavior from the
installed wheel. The smoke check rejects network/process activity during API
imports. The bundle includes the verifier and docs/INSTALLATION.md.

Verification: the real clean-environment wheel installation passed and added
only nil-sdk 0.2.1. Focused packaging, catalog, documentation and artifact checks:
28 passed, one native sklearn check skipped. Missing backends now explicitly
state that NIL does not install or download dependencies automatically. Optional
extras remain explicit requests for external packages; current training adapters
still need those engines. This change does not implement native replacements or
the remaining classical-ML roadmap. Retrying the isolated sklearn development
installation still failed at PyPI DNS resolution; native classical acceptance
remains pending, rather than being reported as complete.

## 2026-11-21 — Installed scikit-learn and mirrored its public API

After network access was enabled, installed scikit-learn 1.9.1 and its numerical
dependencies in /home/us/.nelo/nil-classical-env. A separate compatibility
environment uses scikit-learn 1.4.2, NumPy 1.26.4 and SciPy 1.13.1. These are
development environments; neither changes the base wheel's dependency policy.

Added nil.sklearn with normal public/nested imports, root configuration/clone
helpers and version reporting. A narrowly scoped finder is installed only on
explicit mirror access. Namespace modules proxy backend exports; estimator
classes/functions retain native identity, signatures, clone behavior, pickle
paths and execution. Private underscore modules are excluded. Experimental
enable-import gates and native optional requirements remain upstream behavior.
This exposes the installed scikit-learn public API beyond NIL's curated catalog,
without creating duplicate numerical implementations or per-prediction wrappers.
Added a help.md guide (81 bundled guides), an executable text-classification
example, compatibility tests and current documentation.

Real native acceptance: 118 passed, three skipped with scikit-learn 1.9.1;
116 passed, three skipped with 1.4.2 (module counts vary by version). The skips
are the two unavailable XGBoost adapters and the separate neural adapter. Tests
cover all installed public module exports, nested imports, original class and
function identity, native serialization, text pipeline/search plus NIL complete
pipeline persistence, incremental training and multioutput regression. The
previous classical workflow checks also pass against both engines.

The existing core/neural environment runs 873 passed, 11 optional-dependency
skips. An attempted combined suite in the classical-only environment cannot
collect two preexisting neural test files because they import torch directly;
neural verification uses the existing torch environment instead. There is no
remaining socket failure with the updated permissions. Both runnable classical
and text examples pass using the installed NIL wheel from outside the repository;
the synthetic classical pipeline's prediction roundtrip difference is zero.
Offline clean-environment wheel acceptance still adds exactly nil-sdk alone.

This is an optional backend API mirror, not a source vendoring effort or native
engine replacement. Universal dependency-free training, specialized capabilities
absent from scikit-learn, XGBoost execution and old-PC performance are not implied.

## 2026-11-21 — Classical workflow gaps and failure handling

Added multioutput/multilabel metrics and correct default folds/scorers; modern
cluster-tag detection; a fix for mirror star imports containing blocked private
exports; and multioutput class descriptions. Optional neural tests now skip at
collection when torch is absent, allowing a classical-only full-suite run.

Added training-time DataFrame column inference, ordinal/target encoding, text and
polynomial helpers. Ordinal input normalization handles missing-only batches;
target encoding explicitly distinguishes regression and configures CV without
deprecated random_state arguments on sklearn 1.9. Added immutable bounded JSONL
snapshots with checksum manifests, non-overwrite publication and independent
reloads. They preserve JSON records, not typed DataFrame schemas.

New shallow modules: imbalance (native fold-local samplers and vectorized
weighted threshold selection); streaming (supervised partial_fit, frozen
first-batch preprocessing, full checkpoints and a failed-batch guard); uncertainty
(finite-sample split-conformal intervals and LAC sets); selection (optional
Optuna TPE, persistent total-trial budgets, input/fold/config/version isolation).
TPE uses two random startup trials before adaptive proposals. Added nested CV,
same-fold candidate comparison and bounded numeric PDP/ICE. Normalized the
older PDP grid alias so reports are compatible across tested engine versions.
Artifact manifests now track NIL and optional estimator backends where used.

No new mandatory dependency: tabular/imbalance/search are explicit extras and
all imports stay lazy. Added four help guides (85 total), a runnable extended
workflow, an --extended native acceptance preflight, and CLASSICAL_GAPS.md with
the remaining work and concrete acceptance requirements. The distribution bundle
includes the coverage ledger, guides, examples and clean-install verifier.

Verification: full classical environment 978 passed, four optional skips; existing
neural/core environment 876 passed, 12 optional skips. Native extended/full acceptance
137 passed/one skipped on sklearn 1.9.1 with imbalanced-learn 0.14.2; 135
passed/one skipped on sklearn 1.4.2 with imbalanced-learn 0.12.4. Both use
Optuna 5.1.0, with real persistent resume, bounded trial extension, changed-data
isolation and all-failed search checks. The only remaining native adapter skip
is the separately tested neural adapter. New coverage includes weighted
threshold parity against native scorer grids, conformal quantiles/sets,
checkpoint/pipeline persistence, frozen coordinates, failed-batch rejection,
multioutput metrics, inference leakage, snapshot corruption/bounds, PDP/ICE,
nested fit preflight, comparison and native import behavior. The extended
synthetic example completes with a saved/reloaded conformal model, 150 snapshot
rows and two resumed streaming batches. Its tiny test coverage/accuracy is not
a performance or useful-model-quality guarantee.

Installed the optional xgboost-cpu 3.4.1 variant (5.5 MiB download) in both
classical environments and added an explicit boosting-cpu extra. Native
classifier/regressor checks and full-pipeline metrics/prediction roundtrips pass
on both sklearn versions. Manifest and doctor metadata recognize the CPU
distribution. Base-wheel dependencies remain empty; the CPU engine is not
bundled or automatically installed. GPU XGBoost execution is unverified.

Hard wall-time/RAM containment, distributed quotas/recovery, SHAP, typed dataset
schema evolution, LightGBM/CatBoost acceptance and specialized forecasting,
survival, ranking and recommendation workflows remain explicit gaps. Training
still uses optional external numerical engines; no universal NIL-only engine
or old-PC speed guarantee is claimed.


## 2026-11-21 — Live ML/DL/RL checklist and validated execution/training batches

Added TODO.md as the persistent completion checklist, covering common execution,
classical ML, deep learning, reinforcement learning, independent engines and
production/hardware acceptance. Each checked scope includes an acceptance gate
and concrete evidence. CUDA, distributed/large-model, offline/recurrent/masked/
multi-agent RL and remaining specialized workflows are deliberately not marked
complete. docs/NEURAL_RL.md describes the implemented contracts and limitations.

Implemented opt-in POSIX execution workers with automatic independently supervised timeout,
cancellation/group cleanup, bounded logs/serialized payloads, thread environment
settings, durable outcome/status, Linux virtual address-space limits and safe
dead-worker recovery. Run records created in workers are reconciled on failure.
Classical and actual DL worker/checkpoint tests pass, including a real memory
allocation rejected by the cap. This is not a security sandbox, aggregate RSS/
GPU quota; a detached client-independent supervisor is now implemented and tested.

DL fixes/features: mean losses accumulate using actual loss denominators across
uneven microbatches, weighted/ignored class labels and masked shifted LM tokens;
sum/custom objective contracts are explicit. Added binary/multilabel tasks and
metrics, ignored-label segmentation evaluation, CPU bf16 autocast and CUDA fp16
GradScaler code with saved state/unscale-before-clip/unsafe-step rejection. CPU
bf16 dropout resume is exact. Incompatible config/precision/environment/model/
optimizer/scheduler states are rejected before restore. Optimizer ownership,
subset gradients and parameter names/order are checked. Standard and plateau
scheduler/momentum continuation is exact in native tests.

Checkpoint publication now retains a committed immutable generation plus one
previous generation and a normal .pt alias. Explicit recover repairs interrupted
publication or rolls back one generation. Tests actually terminate a writer
between alias publication and commit, inject commit failures, then reload the
last valid state. Single-writer access and extra retained disk usage are required.
CNN/LSTM/Transformer examples execute actual backward/learning and exact state
roundtrips. Fixed generator-valued hidden widths being consumed during validation.

RL audit removed misleading PPO/CQL claims for legacy action-weight agent
heuristics. GAE/returns now respect terminal/truncation bootstrap boundaries and
ordered rollouts. Replay uses bounded O(1) insertion, chronological access, local
RNG and contents/cursor/RNG serialization. Fixed the agent value-gradient update
using already-updated hidden weights and clear completed trajectory buffers.

Added optional native SB3 PPO/A2C/DQN/SAC/TD3/DDPG/HER integration through nil.rl
with bounded replay-array preflight, CPU-friendly defaults, real run records,
step collection accounting, cancellation, separate bounded Gymnasium evaluation,
normalization/RNG/mode isolation and trusted versioned policy/replay/normalization/
RNG bundles. Root predict/evaluate dispatch to native RL models/results. An owned
RL submit helper constructs/trains/saves inside its worker and returns plain
bundle metadata instead of trying to pickle a live SB3 algorithm.

Native control acceptance: 12 passed on SB3 2.9.2, Gymnasium 1.4.0, torch
2.15.0+cpu, NumPy 2.5.4 / Python 3.12 Linux. All six algorithms update parameters
and roundtrip; PPO/DQN learn a tiny discrete preference and SAC improves a tiny
continuous reward. HER/replay/normalization continuation, environment isolation,
step limits, cancellation, tampering and owned-worker bundle loading are tested.
Pure helper acceptance: 51 passed. Focused DL matrix: 24 passed, one actual CUDA
skip. These tiny tests establish execution/correctness paths, not general policy
quality or old-PC performance. Full regressions and installed-wheel evidence are
recorded in TODO.md. CUDA preflight exits 2 on this CPU-only machine; no GPU
acceptance is claimed.

The base dependency list stays empty. RL engines are explicit optional extras;
SB3 was added only to the existing CPU torch development environment while keeping
torch and NumPy pinned. No backend downloads, worker processes or services start
on import. Bundle builds include the live checklist, updated help, DL/RL guide
and runnable examples, and exercise an installed-wheel worker in a clean NIL-only
environment. Broader production/distributed/independent-engine scope remains open.

Final checklist batch acceptance: full classical environment 990 passed/seven
optional skips; full neural/RL environment 924 passed/14 optional-hardware skips.
Clean NIL-only installation, real installed-wheel stdlib execution, extracted
CNN/LSTM/Transformer/PPO examples and an installed owned RL worker/bundle reload
pass outside the source checkout. Lazy checkpoint-module resolution is also
available in a fresh core-only process for recovery without importing torch.


### 2026-11-21: supervision, bounded data continuation and measurements

A detached stdlib supervisor now owns execution deadlines, group cleanup, bounded
logs, durable status and Run reconciliation independently of the submitting
client. Actual SIGKILL tests verify deadline and descendant cleanup, as well as
successful-result recovery after client death. Reaping waits until owned group
cleanup on waitid-capable hosts to avoid signalling a reused leader PID.
Supervisor/machine death itself is outside the availability guarantee.

DataLoader controls default to zero workers, with bounded workers/prefetch and
explicit optional pinning. Checkpoints capture native loader/sampler generators
and validated dataset/sampler state, call native sampler epoch hooks and reject
changed contracts before model/optimizer/RNG restore. JSONL streaming uses a
verified immutable source and byte cursor for bounded single-worker chunks.
Persistent workers and iterable worker prefetch are refused to avoid untracked
RNG/offset loss; this is a scoped epoch/chunk resume contract, not universal
exactly-once ingestion. See the training/data help guides and loader_resume.py.

Validation: full classical 992 passed/8 optional skips; full Torch/RL 931 passed/
14 optional skips; final loader/execution focused suite 17 passed/1 optional
sklearn skip; neural/RL preflight 130 passed/7 optional/platform skips. Actual
CUDA/Windows/macOS/old-PC gates remain open. Measured current-host core import,
100k×24 Ridge fit/predict and tiny CPU MLP training are documented under
`docs/benchmarks/2026-11-21-*`; these cannot certify other hardware.


Final installed acceptance: the rebuilt 0.2.1 wheel installs offline through
normal dependency resolution with only nil-sdk added. Manifest hashes and all
172 Python source files match the artifact. Both local development environments
now use that regular wheel. From outside the source tree with PYTHONPATH removed,
17 loader/execution tests pass (one unavailable sklearn skip in the Torch env),
and the installed JSONL example verifies exact weights/history and the next
unconsumed record. Independent supervision is exercised from the installed
package by actual client SIGKILL tests. CUDA preflight exits 2 explicitly on
this CPU-only host; it is not recorded as a GPU pass.


### Classical/specialized completion batch — 2026-11-21

Implemented the nine scoped milestones in TODO section 4: optional native CPU
LightGBM/CatBoost pipelines; direct lag/horizon/calendar forecasting and rolling
backtests; query-safe ranking/metrics; bounded sparse item-cosine/popularity
recommendation; right-censored native Cox/KM workflows; bounded interventional
permutation Shapley explanations; logical typed snapshots/migrations and atomic
SQLite ingestion offsets; global same-host random-search attempts with leases and
deterministic continuation; and simultaneous multioutput/fixed-group/known-ratio/
exchangeable-block regression uncertainty. Native backend versions, survival
assumptions, distributed-random versus adaptive-TPE scope and unsupported advanced
methods are explicit in docs/CLASSICAL_SPECIALIZED.md and each module's help.md.

The shallow catalog now has 54 constructors across ten groups and 101 offline
help guides. No base dependency or implicit installation was introduced. Source
acceptance: full current classical suite 1026 passed/8 unrelated optional skips;
minimum sklearn 1.4.2 specialized/catalog acceptance 67 passed/12 missing optional
backend skips. Actual native CPU versions: LightGBM 4.7.0, CatBoost 1.2.10,
lifelines 0.30.0, sklearn 1.9.1, NumPy 2.5.5, pandas 3.1.1 and SciPy 1.18.1.
The full source specialized example passes. New process tests use independent
OS workers/connections, global attempt limits, actual killed-writer rollback and
lease/token rejection. These checks do not imply all specialized algorithms,
all-host performance, GPU support or end-to-end exactly-once model training.


Final installed acceptance: **30 specialized tests passed, zero skips** outside
the source tree with PYTHONPATH removed, and the full extracted example passes
all included workflows. The offline normal resolver installs only nil-sdk.
All 186 Python source files match the wheel, all manifest hashes verify, and
101 offline help guides ship. Both existing development environments now use
the rebuilt regular wheel; the secondary environment received NIL alone and
passes catalog/schema/help/import isolation checks. No universal classical,
GPU, other-host or old-PC completion claim is implied by these scoped checks.


## CPU completion — 2026-11-22

Implemented dependency-free reference `nil.native`, scoped single-host CPU DDP,
Sequential activation recomputation, offline neural imitation/fitted-Q, explicit
structured RL workflows and CPU DQN simulator-state continuation/cancellation.
Added bounded local inference process serving with hard deadlines/restart and
authenticated metrics. CPU generative/LoRA/INT8/ONNX tests exercise real installed
backends and local tiny models without model downloads. Five-seed learning and
fresh-process native numerical benchmark raw reports are bundled.

The numerical engine is deliberately a small dense float64 CPU reference, not a
large vectorized/GPU engine. Optional Torch/SB3/Transformers/PEFT/etc remain explicit
integrations. Classical source acceptance: 1034 passed/15 optional skips. Seven
previous broad checklist gates close with scope recorded in TODO; GPU/other-OS,
old-PC, FSDP, aggregate quotas, public production-serving and comprehensive release
matrix acceptance remain open. Exact coverage/commands/limits: docs/CPU_COMPLETION.md.


Full neural/RL run: 967 passed/13 optional or hardware skips. Final native/offline
numerical-budget checks: 14 passed after adding output/allocation guards. Installed
acceptance: 26 passed/one source-bootstrap deselection; installed completion example
passes native persistence, neural inference and two-rank DDP. Clean offline base
installation adds only nil-sdk; wheel/source parity covers 194 Python files and
103 help guides. The approximately 476 KiB wheel is reinstalled in both existing
development environments without resolving optional packages.


Final installed numerical acceptance after the normalized-storage budget fix:
**13 passed, one source-bootstrap deselection**. The installed --neural example
passes again. No additional dependency was installed during either wheel update.


## Ordered phases 1–2 — 2026-11-22

PHASES.md defines nine ordered follow-on phases including all remaining broad
acceptance gates plus independent-engine and advanced-RL scope. Implemented
opt-in Linux group RSS monitoring and typed durable resource-limit failures,
registered Run reconciliation and detached cleanup. Added explicit delegated
cgroup-v2 memory/disabled-swap/group-OOM containment code; valid kernel behavior
remains unaccepted without a delegated test host. Required-capability preflight
reports failure, rather than a skipped pass.

Fresh-process benchmark runner now has small/medium workload sizes, timeouts,
raw failed-sample records, explicit small-profile memory/wall budgets and measured
Ridge wrapper overhead. All four current-host small/medium classical/neural
profiles execute successfully; small budgets pass, older-PC acceptance stays
false. Source acceptance: classical 1045 passed/17 skips; neural 979 passed/15
skips. Real Windows/macOS/GPU/kernel/older-PC targets remain open. Detailed
measurements and commands: docs/benchmarks/2026-11-22-phases-1-2.md.


Final installed acceptance for phases 1–2: **20 passed/three explicit skips**
(two unconfigured delegated-kernel tests, one unrelated Torch test absent in the
classical environment). The installed stdlib execution runner passes all five
requested checks outside the source tree. Full source suites remain classical
1045/17 skips and neural 979/15 skips. Clean offline dependency resolution adds
only nil-sdk and executes the new RSS path. Source/wheel parity covers 195 Python
files and 103 help guides; both development environments use the rebuilt wheel
without adding dependencies. All bundle hashes pass. The approximately 478 KiB
wheel, phase list, runners, budget files and measured reports are bundled.


## CPU FSDP and target infrastructure — 2026-11-22

Implemented opt-in same-world FP32 FSDP in nil.distributed, parameter/data limits,
trusted native rank-local model/optimizer/RNG checkpoints with manifest-last
publication and last-valid-epoch recovery. Actual CPU ranks validate local
partitioning, serial SGD parity, exact resume, callback/rank faults, publication
exception recovery and caller immutability. DDP hangs terminate within a bounded
wall deadline. No GPU or meta-device/large-host-memory acceptance is inferred.

Added required-hardware GPU test/report runner and executable AMP/DDP/FSDP cases;
actual CPU-only preflight fails with exit 2. Added unexecuted nine-cell base
platform/Python and manual-only trusted GPU CI configurations. Windows execution
remains unsupported and checked for explicit refusal rather than advertised as
contained. Native FSDP1 local state APIs have documented deprecation/version/trust
limits. Full source tests: neural 985 passed/20 skips; classical 1046 passed/19
skips; focused sharding/DDP/runner 12 passed. Commands and exact scope:
docs/benchmarks/2026-11-22-sharding-gates.md.


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


## Windows containment — 2026-11-22

Implemented suspended worker creation, Job Object assignment before resume,
restricted inherited stdio handles, detached supervision and handle-owned tree
termination. Windows kill-on-close ownership, process-time identity recovery and
bounded atomic sharing-lock retry use only CPython/ctypes. No Windows memory/GPU
quota is implied. Portable/Linux focus: 31 passed/three skips; final classical
full suite: 1056 passed/29 skips. Prior full neural: 992 passed/29 skips. Ten
Windows native tests are written and wired into CI but cannot execute on this
Linux host. Required Windows preflight fails rather than passing mocks or skips.
The implementation is complete for review; native platform acceptance stays open.
See docs/WINDOWS_EXECUTION.md for mechanism, commands, target scope and limits.


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

## Phase8–9 implementation update — 2026-11-23

The independent float64 CPU engine now has explicit ND semantics, COO/CSR algebra
and an optional NIL-owned C matmul/VJP kernel. Base packaging remains pure Python
plus help/C source, with no mandatory numerical dependency or install-time compiler.
RL adds masked parallel collection, discrete CQL, fixed-cohort offline centralized
AWR and supported CPU PPO/A2C simulator snapshots. Five three-seed task-specific
learning benchmarks pass; exact data/protocol limits are in NEURAL_RL.md.
Full neural/RL source regression:1098 passed/30 skips; final focused guard checks:
86 passed. See the live phases/TODO for installed/matrix acceptance and remaining
real-device, old-machine, larger-model and deployment gates. These are bounded
implemented contracts, not a claim of replacing all numerical kernels/algorithms.

## Framework boundary and image classification — 2026-11-24

Implemented versioned workflow discovery (`nil.coverage`, `nil workflows`),
structural contracts and result metadata (`nil.contracts`), and neural
TrainingResult prediction/evaluation dispatch. Offline metadata remains passive.
The master TODO checks these scoped deliverables without claiming completion of
all 32 areas.

Added optional `nil.vision`: bounded lazy decoding, explicit EXIF/RGB/alpha/resize
semantics, shared class mapping, CNN training/evaluation, JSON preprocessing/label
manifest, checkpoint reload and streamed inference. A real digit dataset with
1,000 training and 300 held-out images reached 89.33% accuracy in 29.93 seconds,
with 333,979,648-byte peak child RSS and exact prediction reload. Detection,
segmentation, OCR, tracking, retrieval and visual export/serving remain open.

Installed profiles passed 153 Torch, 36 RL, 70 current-classical and 68 minimum-
classical tests, with zero skips and every example passing. The full current-
classical source suite passed 1,154 tests with 50 optional skips. Hosted full SDK
CI passed 1,253 tests with 46 optional skips. Offline installation still adds only
NIL with the resolver enabled. The 545,103-byte wheel matches 207 Python sources
and 107 help guides, with no compiled binaries or model weights.

All nine Linux/macOS/Windows Python 3.10–3.12 base jobs and all ten declared
minimum/current backend jobs passed at commit `318e453`. Actual GPU, older-PC and
public deployment acceptance remains unchanged. See the [framework evidence](../benchmarks/2026-11-24-framework.json),
[image evidence](../benchmarks/2026-11-24-vision-digits.json),
[workflow inventory](../WORKFLOWS.md) and [contracts](../SDK_CONTRACTS.md).


## Compact-research readiness — 2026-11-24

Added an existing-capability audit and a 78-task research plan for efficient training,
compact intelligence and all seven previously requested incomplete workflow areas.
The first four preparation/readiness tasks are checked, with real evidence for
the fourth. The proposed 1M/10M study and 10–20B/100B–1T comparisons remain open.

The local CPU example uses shared attention and a custom frozen-logit distillation
loss. Three seeds passed updates/loss-reduction/reload checks. Ten focused tests
passed shared-gradient parity, independent KL reference, frozen teacher behavior,
uneven accumulation, exact resume and changed-depth refusal before weight loading.
Distillation did not improve on ordinary shared-model accuracy on this tiny task;
that negative outcome is retained, with no intelligence-equivalence claim.

Installed-wheel Torch acceptance passed 163 tests with zero skips and all five
examples. Offline resolver-enabled base installation still adds NIL only. See
[readiness audit](../RESEARCH_READINESS.md), [ordered tasks](RESEARCH_TODO.md)
and [evidence](../benchmarks/2026-11-24-compact-readiness.json). The existing scoped
FSDP, feature-type adapters, training measurement and hardware/resource gaps are
explicitly documented; no additional GPU or multi-host acceptance was performed.


## Seven AI workflow areas — v1, 2026-11-24

Implemented nil.rag, nil.planning, nil.probabilistic, nil.learning, nil.audio,
nil.multimodal and nil.graphs. Each has offline help and runnable end-to-end
examples. The [scope guide](../SEVEN_WORKFLOWS.md) records exact contracts and
advanced work left open; the [37-workflow inventory](../WORKFLOWS.md) exposes the
new subset. The master 32-area backlog remains intact.

Local installed-wheel profiles passed 189 Torch, 88 classical-current and 86
classical-minimum tests with zero skips and all examples passing. Final hosted
SDK CI passed 1,294 tests/46 optional skips; serial local neural suite passed 1,171
with 32 skips. An interrupted concurrent local run's four execution/TLS failures
are retained separately; serial recovery checks passed 22. CI explicitly installs
FFmpeg for codec acceptance; NIL itself still installs no dependencies by default.

Real Iris self-training reached 93.33% held-out accuracy; real karate-club node
classification reached 93.75% (transductive, 2 labeled training/32 held-out nodes).
Recorded speech/whistle acceptance covers decoding/features only. Synthetic
audio/video and representation tests are not ASR/TTS or general multimodal quality.
Active entropy selection reduced Iris accuracy; no improvement claim is made.

The wheel matches 216 Python sources and 114 help guides, with no compiled binary,
model weights or media fixtures. A 6M-character/6667-chunk BM25 indexing check took
about 0.275 seconds on the current host after moving provenance hashing outside
the chunk loop. See [complete dated evidence](../benchmarks/2026-11-24-seven-workflows.json).

Final verification at `c8dfba5`: all nine base-platform and ten installed-backend
jobs passed. Hosted SDK: 1294 passed, 46 optional-target skips.
Exact versions, failures, safe-refusal limits and raw-report references are in the
[dated evidence](../benchmarks/2026-11-24-seven-workflows.json).

Advanced seven-area additions and still-open requirements: [advanced workflow phase](../ADVANCED_WORKFLOWS.md). New CPU checks are separate from the earlier v1 hosted matrix.
