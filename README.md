# NIL AI development SDK

NIL provides model-development helpers and optional agent tools. Its public layout stays shallow; compute backends load only when needed.

The [live completion checklist](TODO.md) tracks classical ML, DL and RL work,
with explicit testing and validation required before each item is checked.

```text
NIL
├── regression         linear/robust/SGD, neighbors, SVM, trees, ensembles
├── classification     logistic/SGD, neighbors, SVM, Bayes, ensembles
├── unsupervised       clustering, mixtures, dimensionality reduction
├── anomaly detection  Isolation Forest, One-Class SVM
├── feature selection  variance, statistical, model-based
└── neural networks    MLP and custom PyTorch modules
```

`nil models` prints the exact model names. `nil.list_models()` returns them in Python. Listing models does not import PyTorch, scikit-learn, XGBoost, or agent subsystems. A listed adapter still requires its optional backend; listing is not an installation or hardware-readiness claim.

## Install only what you use

The base wheel installs **only `nil-sdk`**, with zero runtime dependencies and no
automatic backend downloads. Use the wheel without extras for a NIL-only
environment. The [installation contract](docs/INSTALLATION.md) explains the
clean-environment release check and the current computation boundary: training
through external engines still requires those engines. Selecting an extra below
explicitly requests additional packages.

From either supported project root:

```sh
python -m pip install .                   # dependency-free core
python -m pip install '.[classical]'       # scikit-learn model constructors
python -m pip install '.[boosting-cpu]'    # optional CPU XGBoost (compatible wheels)
python -m pip install '.[torch]'           # neural training
python -m pip install '.[rl]'              # optional native neural control
python -m nil doctor
python -m nil models
python -m nil help linear_regression
python -m nil help --list
```

Select a suitable PyTorch wheel for your hardware separately. Transformers/LoRA, ONNX, and provider integrations have their own extras. The requirements files install only the core; they do not install unrelated experiment/tracking packages.

With the classical backend installed, access scikit-learn's public APIs directly
through NIL, including APIs beyond the curated catalog:

```python
from nil.sklearn.linear_model import LogisticRegression
from nil.sklearn.feature_extraction.text import TfidfVectorizer
from nil.sklearn.model_selection import GridSearchCV
```

These are native backend objects, preserving their parameters, performance and
serialization. See `nil help sklearn`. This is an optional API mirror; it does
not package a separate numerical engine inside NIL.

From an extracted distribution bundle, install its wheel instead of the source project, for example `python -m pip install './wheels/nil_sdk-0.2.1-py3-none-any.whl[classical]'`.

## Classical pipeline workflow

```python
import nil

X = [[float(i)] for i in range(30)]
y = [2 * i + 1 for i in range(30)]
data = nil.split(X, y, seed=7)
model = nil.pipeline(nil.model("ridge_regression"), preprocess=nil.preprocessing.numeric())
result = nil.fit(model, *data.train, validation=data.test)
print(result.metrics)
result.save("./models/ridge-v1")
loaded = nil.load("./models/ridge-v1", trusted=True)
print(nil.predict(loaded, [[31.0]]))
```

Use `logistic_regression` for classification. Mixed numeric/categorical preprocessing, sample weights, feature selection, stratified/group/time-aware splits, cross-validation, bounded grid/random/Bayesian tuning, nested CV, multioutput evaluation, inspection, calibration and learning curves have explicit APIs. See [the classical workflow guide](docs/CLASSICAL_ML.md) for contracts, budgets, persistence trust requirements and current verification limits. The catalog has 54 constructors; listing does not establish installed-backend readiness.

Add a custom model with `nil.register_model("my_model", factory, group="regression", help_path="my_model/help.md")`. Its factory runs only when selected. Every catalog model and public module has a separate [help.md reference](docs/MODULE_REFERENCE.md): `nil help linear_regression`, `nil help validation`, or `print(nil.help("preprocessing"))` work offline without loading a compute backend.

See [DL/RL workflows and validation](docs/NEURAL_RL.md) for CPU bf16, loss normalization, recoverable checkpoints, native control algorithms and the remaining distributed/GPU/offline-RL gates.

## Neural training

```python
import nil
import torch

nil.runtime.seed(7)
x = torch.linspace(-1, 1, 32).reshape(-1, 1)
data = nil.data.tensors(x, 2 * x + 1)
model = nil.model("mlp", input_dim=1, output_dim=1, hidden=(16,))
result = nil.train(model, data, epochs=20, device="cpu")
print(nil.evaluate(result.model, data, device_name="cpu"))
```

## Public surface

| API | Current responsibility |
|---|---|
| `nil.model`, `nil.list_models`, `nil.catalog` | A small model registry, six groups, direct backend constructors |
| `nil.sklearn` | Public scikit-learn APIs, including nested modules, through an optional backend |
| `nil.data` | Records, immutable JSONL snapshots, validation, splits and tensor datasets/loaders |
| `nil.fit`, `nil.pipeline`, `nil.preprocessing` | Classical fitting/results, learned preprocessing and selection |
| `nil.train`, `nil.Trainer`, `nil.training` | Neural/custom-step training, checkpoints, callbacks, bounded search |
| `nil.split`, `nil.cross_validate`, `nil.tune`, `nil.validation` | Fold-isolated classical validation and bounded selection |
| `nil.inspection` | Importances, calibration, learning curves, PDP/ICE and model comparisons |
| `nil.execution` | Opt-in POSIX workers, cancellation and Linux address-space caps |
| `nil.rl` | Optional neural PPO/A2C/DQN/SAC/TD3/DDPG/HER control and separate agent helpers |
| `nil.streaming` | Incremental supervised training, frozen preprocessing and checkpoints |
| `nil.uncertainty` | Split-conformal intervals/sets and held-out coverage |
| `nil.imbalance`, `nil.selection` | Fold-local resampling, threshold selection and resumable Bayesian search |
| `nil.save`, `nil.load`, `nil.persistence` | Complete trusted classical pipeline artifacts |
| `nil.classical` | Classical fit results, run records, warnings and held-out metrics |
| `nil.evaluate`, `nil.predict`, `nil.transform` | Evaluation/prediction dispatch and native feature transformation |
| `nil.lifecycle`, `nil.Registry` | Local run records and versioned checkpoint artifacts |
| `nil.optimization`, `nil.inference`, `nil.serving` | CPU optimization/ONNX, inference, local development serving |
| `nil.generative` | Optional local language models, LoRA and perplexity |
| `nil.Agent`, agent and robotics modules | Optional existing applications; loaded when requested |

Classical and neural engines remain optional. Classical pipeline/result operations are shared through NIL; gradient training still uses `nil.train`, and neural checkpoints retain their own format. The SDK remains alpha. Classical workflows and the API mirror have passed real-backend acceptance with scikit-learn 1.4.2 and 1.9.1. CPU XGBoost pipeline/metric/save-reload checks also pass; GPU execution is unverified. Distributed training, production serving and universal workflow coverage are not claimed. Native computation still depends on model, data and hardware.

See [model development](docs/MODEL_DEVELOPMENT.md), [shallow design and cleanup](docs/SHALLOW_DESIGN.md), and the [agent reference](docs/USERGUIDE.md). `nil.models` retains its legacy POMDP/environment-model meaning; use `nil.model()` to construct trainable models.

The [remaining work and next fixes](docs/CLASSICAL_GAPS.md) distinguishes implemented workflows from specialized engines, broader resource isolation, SHAP and native-engine work still to do. Each new module has a bundled help.md. The base install remains NIL only; tabular, imbalance and search extras are explicit opt-ins.

## Verification

```sh
PYTHONPATH=nil-sdk python -m pytest nil-sdk/tests -q
```

Run this from the repository root with your chosen optional dependencies. Optional backend tests may skip when dependencies are absent. The dependency-free catalog and import-isolation tests run without them.

Classical and specialized workflows now include forecasting, ranking, sparse recommendation, survival, optional LightGBM/CatBoost, bounded Shapley explanations, typed transactional datasets, distributed random search and additional conformal contracts. See [supported scope and acceptance](docs/CLASSICAL_SPECIALIZED.md).
