# NIL: shallow public design

The public model catalog has two levels: group and model. A plain registry maps a name directly to a native backend constructor. It adds no wrapper to model fitting or prediction, starts no worker, and does not select hardware or download packages automatically.

```text
NIL
├── regression         linear/robust/SGD, neighbors, SVM, trees, ensembles
├── classification     logistic/SGD, neighbors, SVM, Bayes, ensembles
├── unsupervised       clustering, mixtures, dimensionality reduction
├── anomaly_detection  Isolation Forest, One-Class SVM
├── feature_selection  variance, statistical, model-based
└── neural_networks    MLP and custom PyTorch modules
```

Logistic regression belongs to classification. The tree is a way to find models, not a chain of execution stages. `gradient_boosting_*` selects scikit-learn's histogram gradient boosting; `xgboost_*` explicitly selects XGBoost.

```python
import nil

print(nil.list_models("regression"))
print(nil.catalog.tree())
model = nil.model("linear_regression", fit_intercept=True)
```

`python -m nil models` prints the tree without loading any compute backend. Listing a model describes an available constructor adapter, not an installed or hardware-verified backend. Backend parameters and errors remain native; there is no automatic expensive tuning, background service, or installation. Choose the `classical`, `boosting`, or `torch` extra explicitly.

## Add a model without changing the core

```python
import nil

def build_forecaster(**parameters):
    from my_forecasting_package import Forecaster
    return Forecaster(**parameters)

nil.register_model("my_forecaster", build_forecaster, group="time_series", help_path="forecaster/help.md")
model = nil.model("my_forecaster", window=24)
```

Registration records the factory without calling it. The optional package loads only when that model is constructed. A new group adds one row to the same shallow tree. Registration is explicit, process-local, and cannot replace an existing name; there is no automatic plugin scanning or startup service. Import your registration module in each process that needs the extension. The extension supplies its dependency handling and its native model behavior; registration alone does not implement a new trainer or serializer.

## Per-module help

Each of the 54 catalog models and 47 public module topics has a dedicated `_help/<name>/help.md` file bundled in the wheel. Each file contains usage, parameters, outputs, requirements and limitations. Use `nil help linear_regression`, `nil help training`, or `nil help --list`; Python callers can use `print(nil.help("linear_regression"))`. Reading help loads no backend and starts no service. Custom models can attach their own help.md through `help_path` at registration.

## Module boundaries

- `catalog` owns discovery and construction, with no dependency on training or agents.
- `data` owns records and dataset preparation.
- `training`, `evaluation`, and `inference` own their execution paths.
- `lifecycle` owns run metadata and artifacts.
- Native backends and agent/robotics applications remain optional consumers of these modules.

New features should own one module and a focused acceptance test. Share data/contracts where needed, keep optional imports inside their execution path, and avoid cross-importing application subsystems into basic model operations.

## What stays

| Surface | Why it stays |
|---|---|
| `catalog`, `data`, `nn`, `runtime` | Find/build models, prepare data, and select explicitly installed compute |
| `training`, `evaluation`, `inference` | Real neural training and inference, custom objectives, callbacks and search |
| `lifecycle`, checkpoints and registry | Record experiments and recover model state |
| `classical`, `preprocessing`, `persistence` | Fitted classical pipelines, metrics, run records and trusted artifacts |
| `validation`, `inspection` | Fold isolation, bounded tuning, calibration and diagnostics |
| `generative`, optimization, ONNX, local serving | Existing optional model-development workflows |
| Agents, cognition, memory, tools, RL helpers, robotics, observability | Working optional applications with existing callers and regression coverage |
| Stub providers and synthetic tests | Explicit test fixtures that keep offline verification reproducible |

Training exports and robotics helpers now resolve individually. Resolving `nil.Trainer` does not import agent curriculum or memory. Resolving `nil.robotics.policy` does not import robotics domains, NumPy, or physical-world tooling. Existing public names remain available when requested.

This is an import and API simplification, not a new model execution engine. A large dataset or neural network still requires native compute and memory. Classical models support `nil.fit`, `nil.evaluate`, `nil.predict`, `nil.save` and `nil.load` with complete fitted pipelines; neural modules use `nil.train` and their existing checkpoints. Shared neural/classical artifact formats and universal backend coverage remain separate work. See [the classical guide](CLASSICAL_ML.md). `nil.models` retains its legacy environment/POMDP-model meaning.

## Removed

- `nil.optimization.model_router`: unused placeholder returning its default model.
- `nil.optimization.chain_compressor`: unused placeholder returning no candidates.
- `nil.evolution.strategy_mutator`: unused skeleton; the active evolution pipeline remains.
- `nil.optimization.cache_manager` and `budget_controller`: standalone helpers with no production callers; these were not integrated agent optimizations.
- `nil.server`: obsolete canned bridge reporting invented traces, skills and successful robot operations. Use the existing real `nil_server.py` daemon or the SDK. The legacy Studio bundle bridge now returns explicit HTTP 503 for its retired NIL endpoints rather than fabricating success.
- Outdated verification checks for the removed helpers, and bulk experiment/provider/tracking packages from default requirements files.
- Duplicate marketing-heavy entry documentation. Model development is the primary entry point; detailed agent docs remain optional references.

The former deep imports above intentionally stop working. Historical audits retain their original evidence and are not rewritten as current capability claims. Projects, model artifacts, memory databases, and working agent implementations are preserved.

## Verification — 2026-11-20

Before cleanup, this environment ran 843 existing tests successfully, skipped five optional-backend tests, and failed one local-serving test because sandbox sockets are blocked. The new catalog tests cover shallow groups, correct logistic classification, unknown-name errors, dependency-free listing, lazy training/robotics imports, and real PyTorch training. Classical fitting checks skip when scikit-learn is absent; XGBoost execution requires its own acceptance environment.

Single-process import probes before/after showed `nil.Trainer` loading nine/six NIL modules and `nil.robotics.policy` loading 21/four. Single timing samples improved from 66 to 39 ms and 221 to 30 ms respectively, but these are local observations rather than portable performance guarantees. No claim of 98% library coverage, universal training support, or constant model execution latency is made.

## Classical extension verification

The catalog has 54 models across ten shallow groups, with 101 help files. On
2026-11-21 network access became available and real classical acceptance passed
with scikit-learn 1.4.2 and 1.9.1. `nil.sklearn` also provides the installed
backend's public API beyond that catalog. One namespace module and one private
import helper support nested imports, rather than generating a source file for
every upstream algorithm. The import finder is installed only when the mirror
is explicitly requested; model operations use original native objects. The base
wheel remains dependency-free. See CLASSICAL_ML.md for the verification boundary.

Classical and specialized workflows now include forecasting, ranking, sparse recommendation, survival, optional LightGBM/CatBoost, bounded Shapley explanations, typed transactional datasets, distributed random search and additional conformal contracts. See [supported scope and acceptance](CLASSICAL_SPECIALIZED.md).
