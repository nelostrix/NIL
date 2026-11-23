# NIL AI development SDK

[![CI](https://github.com/nelostrix/NIL/actions/workflows/sdk.yml/badge.svg)](https://github.com/nelostrix/NIL/actions/workflows/sdk.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

NIL provides a shallow, modular Python interface for classical machine learning,
deep learning, reinforcement learning and model deployment workflows.

**Installing the base package installs only `nil-sdk`.** It has no mandatory
runtime dependencies. Numerical backends load when requested and are installed
through explicit extras. Every public module and catalog model has offline help.

NIL is an independent open-source project and is not affiliated with or endorsed
by OpenAI. OpenAI-compatible provider support is an optional integration.

## Install

From the repository root, using Python 3.10 or newer:

```sh
python -m pip install .
python -m nil models
python -m nil help --list
python -m nil doctor
```

The dependency-free numerical subset can train small dense CPU models directly:

```python
import nil

model = nil.native.Linear(1, 1, seed=7)
result = nil.native.fit(
    model, [[0.], [1.], [2.]], [[1.], [3.], [5.]], epochs=300, lr=.1
)
print(model(nil.native.Tensor([[3.]])).tolist())  # Approximately [[7.0]]
```

`nil.native` also supports stable multiclass loss, softmax and Adam with JSON
checkpoint continuation. See the [independent engine scope](docs/NATIVE_ENGINE.md).
It now includes nonempty multidimensional tensors, explicit COO/CSR sparse
algebra and an optional locally compiled NIL-owned CPU matrix kernel.
It is a bounded float64 CPU reference engine. Larger classical,
neural and specialized workflows use explicitly installed backends:

| Install | Enables |
| --- | --- |
| `python -m pip install '.[classical]'` | scikit-learn constructors, pipelines and validation |
| `python -m pip install '.[boosting-cpu]'` | CPU XGBoost integration |
| `python -m pip install '.[lightgbm,catboost,survival]'` | Specialized boosting and survival integrations |
| `python -m pip install '.[torch]'` | Torch training, evaluation and distributed workflows |
| `python -m pip install '.[generative]'` | Local causal models, LoRA and perplexity |
| `python -m pip install '.[rl]'` | Gymnasium/SB3 environment training |
| `python -m pip install '.[rl-contrib]'` | Recurrent and masked SB3 policies |
| `python -m pip install '.[onnx]'` | ONNX export and runtime inference |

Choose Torch wheels for your hardware explicitly. Selecting an extra requests its
additional packages; NIL does not install backends automatically during execution.
See the [installation contract](docs/INSTALLATION.md).

## Classical machine learning

After installing the classical extra:

```python
import nil

X = [[float(i)] for i in range(30)]
y = [2 * i + 1 for i in range(30)]
data = nil.split(X, y, seed=7)
model = nil.pipeline(
    nil.model("ridge_regression"), preprocess=nil.preprocessing.numeric()
)
result = nil.fit(model, *data.train, validation=data.test)
print(result.metrics)
print(nil.predict(result, [[31.0]]))
```

The catalog covers regression, classification, clustering, dimensionality
reduction, anomaly detection and feature selection. Shared workflows include
preprocessing, fold-isolated validation, bounded search, streaming, uncertainty,
inspection and trusted artifact persistence. Specialized modules cover scoped
forecasting, ranking, recommendation, survival, boosting and dataset ledgers.

The optional API mirror also exposes public backend APIs:

```python
from nil.sklearn.linear_model import LogisticRegression
from nil.sklearn.model_selection import GridSearchCV
```

These remain scikit-learn objects. Read the [classical guide](docs/CLASSICAL_ML.md)
and [specialized scope](docs/CLASSICAL_SPECIALIZED.md) for supported variants and
statistical limits.

## Deep learning and reinforcement learning

With the Torch extra installed:

```python
import nil
import torch

x = torch.linspace(-1, 1, 32).reshape(-1, 1)
data = nil.data.tensors(x, 2 * x + 1)
model = nil.nn.mlp(1, 1, hidden=(16,))
result = nil.train(model, data, epochs=20, device="cpu")
print(nil.evaluate(result.model, data, device_name="cpu"))
```

Neural workflows include custom objectives, accumulation, CPU BF16, checkpoint
recovery, CNN/recurrent/Transformer construction, local language models and LoRA.
`nil.distributed.train(..., strategy='fsdp')` adds CPU-tested, fixed-world FP32
sharding with model/optimizer/RNG checkpoints and explicit trusted shard resume.

`nil.rl` provides optional environment-training integrations, neural behavior
cloning/fitted-Q baselines, recurrent/masked policies and independent multi-agent
workflows. Exact simulator continuation supports a declared CPU DQN contract.
Additional scoped CPU workflows include discrete CQL, offline centralized-value
AWR, masked parallel collection and exact PPO/A2C protocol-environment continuation.
See [DL/RL workflows](docs/NEURAL_RL.md) and [CPU acceptance](docs/CPU_COMPLETION.md).

## Owned execution and serving

- `nil.execution`: explicit workers, cancellation, deadlines, bounded logs,
  read-only recovery and Linux memory controls. Real delegated-cgroup tests pass.
- Windows Job Object ownership is implemented with creation-time assignment and
  cleanup. Base-wheel and owned-execution checks passed on Linux, Windows and
  macOS with Python 3.10–3.12 in [platform CI](https://github.com/nelostrix/NIL/actions/runs/45113943448).
- `nil.serving`: isolated CPU inference, TLS/mTLS gateway, verified HTTPS client
  with bounded retries, and an explicit service lifecycle. Gateway/client code
  uses the standard library; Torch inference is an optional backend.
- Managed Linux serving has passed local TLS, fault and resource-policy checks.
  Public Internet deployment and arbitrary-model SLOs remain separate gates.

Read [Windows execution](docs/WINDOWS_EXECUTION.md) and
[managed deployment](docs/DEPLOYMENT.md) before selecting these workflows.

## Project layout and help

```text
nil-sdk/nil/        SDK modules and bundled per-module help
nil-sdk/tests/      Functional, numerical and platform tests
nil-sdk/examples/   Runnable workflows
scripts/           Build and acceptance tools
docs/              Guides, audits, phase checklists and measured evidence
deploy/            Explicit service configuration examples
benchmarks/        Workload budget definitions
```

```sh
python -m nil help linear_regression
python -m nil help distributed
python -m nil help serving
```

Register an extension with `nil.register_model(..., help_path='module/help.md')`.
Its factory runs only when selected. Existing agent/provider/robotics modules
remain opt-in. `nil.models` retains its environment-model meaning; use
`nil.model()` for catalog constructors.

## Validation and remaining work

NIL is alpha software. Its independent engine is a small dense CPU subset;
optional integrations do not make it an independent replacement for every AI
library. GPU, older-PC, full platform/backend and public-production acceptance
must be demonstrated on their actual targets.

Small core, classical and neural workloads passed under actual one-CPU quota,
1 GiB memory and zero-swap limits on the current host. See the
[hardware evidence and reproduction steps](docs/HARDWARE_ACCEPTANCE.md).

The [ordered phase checklist](docs/project/PHASES.md) and
[live TODO](docs/project/TODO.md) distinguish implemented code, passing tests,
unsupported combinations and unrun hardware gates. Detailed historical evidence
is in the [implementation report](docs/project/SDK_IMPLEMENTATION_REPORT.md).
The [installed backend profiles](docs/BACKEND_COMPATIBILITY.md) document separate
minimum/current classical and Torch CPU checks, including runnable examples.

For development, install the explicit dev extra and run the relevant backend tests:

```sh
python -m pip install '.[dev]'
PYTHONPATH=nil-sdk python -m pytest nil-sdk/tests -q
python scripts/build_sdk_bundle.py --output dist
```

The bundle builder verifies a fresh offline installation adds only NIL. A skipped
optional test is not acceptance for that feature. GPU tests require real devices;
Windows acceptance requires Windows rather than portable mocks.
