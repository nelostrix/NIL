# NIL 0.2 model-development implementation

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

See [the model-development guide](docs/MODEL_DEVELOPMENT.md). Build the ZIP with
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
