# Classical ML through NIL

NIL exposes native model constructors and a shallow workflow: `model` → `pipeline` → `split` → `fit` → `evaluate` → `save/load`. Optional preprocessing, validation and inspection modules own their respective operations. No model engine loads when listing models or reading help.

## Install and verify

From the source root, install `python -m pip install '.[classical,dev]'`. The boosting extra also includes scikit-learn because XGBoost's sklearn-style estimators and NIL metrics need it. pandas is optional for named DataFrame inputs; arrays/lists and sparse matrices do not require pandas.

## Scikit-learn API mirror

Use any public scikit-learn module through `nil.sklearn`, beyond the curated
54-model catalog. For example:

```python
from nil.sklearn.ensemble import ExtraTreesClassifier
from nil.sklearn.gaussian_process import GaussianProcessRegressor
from nil.sklearn.feature_extraction.text import TfidfVectorizer
from nil.sklearn.model_selection import GridSearchCV
from nil.sklearn.multioutput import MultiOutputClassifier
from nil.sklearn.metrics import classification_report
```

The installed backend determines available APIs. Models/functions retain native
identity, signatures and serialization, with no per-fit or per-prediction wrapper.
`nil.sklearn` loads only namespace support; importing its backend APIs loads
scikit-learn. The narrowly scoped import finder is installed only when this
namespace is explicitly requested. Private underscore modules are excluded.
Experimental APIs preserve their upstream enable-import requirements.

This includes preprocessing/encoding, text features, feature engineering,
multioutput wrappers, incremental estimators, native search, partial dependence
and other APIs that scikit-learn provides. Their native constraints still apply.
NIL's classification evaluator supports multilabel and multioutput targets,
including aggregate/per-output metrics. Native searches do not acquire
NIL's fit-count budgets automatically. Some plotting and data APIs need further
optional dependencies, and dataset fetchers download data when called.
This is not a native engine replacement, nor does it add capabilities absent
from scikit-learn. See `nil help sklearn` and the upstream
[API reference](https://scikit-learn.org/stable/api/index.html).

```python
import nil

X = [[float(i), float(i % 3)] for i in range(60)]
y = [2 * i + 1 for i in range(60)]
data = nil.split(X, y, seed=7)

model = nil.pipeline(
    nil.model("ridge_regression", alpha=0.1),
    preprocess=nil.preprocessing.numeric(),
)
result = nil.fit(model, *data.train, validation=data.test)
print(result.metrics)
print(result.predict([[61., 1.]]))

result.save("./models/ridge-v1")
loaded = nil.load("./models/ridge-v1", trusted=True)
print(nil.evaluate(loaded, *data.test))
```

The saved bundle contains the complete fitted pipeline, feature names/count when available, native configuration/state, dependency versions and a SHA-256 manifest. Existing bundles are never overwritten. The manifest is committed last; an interrupted save has no completed manifest. Joblib can execute Python when loaded, so `trusted=True` is mandatory. Integrity hashes do not authenticate the source. Exact dependency versions and the Python major/minor version are checked by default. Neural models continue to use their existing checkpoint APIs.

## 1. Preprocessing, fitting and evaluation

- `numeric(impute="median", scale=True)` imputes and scales without centering sparse matrices. `scale` can also name `standard`, `robust`, `minmax`, or `maxabs`; minmax requires dense data.
- `categorical(sparse=True)` imputes and one-hot encodes. Unknown prediction-time categories are ignored rather than refitted.
- `tabular(numeric_columns=..., categorical_columns=...)` accepts integer positions or DataFrame column names, rejects duplicate selections and drops unselected columns. Missing values use numpy.nan; normalize other sentinels explicitly.
- `pipeline(model, preprocess=..., select=...)` keeps learned preprocessing and feature selection inside each training fold. Choose dense output explicitly for estimators that require it; NIL does not silently densify large matrices.
- `select_features(k=..., task="classification"/"regression")` supplies task-appropriate supervised feature scoring. Other selectors are in the catalog.
- `fit` supports aligned sample weights, separate held-out validation weights, native fit arguments, run records and captured fit warnings. Unsupported weight arguments raise instead of being discarded.

By default sample weights reach the final estimator; preprocessing statistics are fitted with the transformers' native unweighted behavior. Estimator weighting does not imply weighted imputation/scaling.

`ClassicalResult` exposes `model`, `run`, `metrics`, `task`, `warnings`, `status`, `predict`, `predict_proba`, `evaluate`, and `save`. Legacy `result["model"]`/`result["metrics"]` access remains valid. Search results expose the same main methods plus search details. Bare estimators remain native objects, so custom code and native APIs stay usable.

Regression metrics are MAE, MSE, RMSE and R². Single-output classification includes accuracy, balanced accuracy, Matthews correlation, macro/weighted precision/recall/F1, confusion matrices and per-class metrics. Probability-capable models additionally receive log loss and applicable ROC-AUC/average-precision/Brier metrics. Undefined AUC is omitted; one-sample R² is None. Multilabel/multioutput classification reports subset accuracy, Hamming loss and per-output metrics, plus micro/macro/weighted/sample F1 where appropriate. Per-output probability lists receive per-output log loss; arbitrary chain probability shapes are not silently treated as class-probability matrices. Targets with classes absent from fitting are rejected explicitly when the model exposes the corresponding class vocabularies.

`nil.evaluate(model_or_result, X, y)` and `nil.predict(model_or_result, X)` support classical models without loading torch. Existing neural evaluation/prediction still works, including the older positional neural evaluation form. Configure classical devices through native estimator parameters, not neural device options.

Feature transformers such as PCA/selectors also support `nil.fit` and bundle persistence; use `nil.transform(result_or_model, X)` or `result.transform(X)` for their output. They have no default predictive metric, so evaluate a downstream supervised pipeline or choose an explicit CV scorer. Clustering metrics reject sample weights they cannot apply; unlabeled anomaly fractions honor supplied weights.

## 2. Common model families

The catalog now includes 54 constructors, including the preexisting neural/boosting entries:

- Regression: linear, ridge, lasso, elastic net, SGD, Huber/RANSAC, nearest neighbors, kernel/linear SVR, trees, forests, bagging, histogram boosting, XGBoost, voting and stacking.
- Classification: logistic, SGD, nearest neighbors, kernel/linear SVC, Gaussian/multinomial/Bernoulli Naive Bayes, trees, forests, bagging, histogram boosting, XGBoost, voting, stacking and calibrated classifiers.
- Unsupervised: k-means, DBSCAN, agglomerative clustering, Gaussian mixtures, PCA and truncated SVD.
- Anomalies: Isolation Forest and One-Class SVM.
- Feature selection: variance threshold, SelectKBest and SelectFromModel.

Each constructor has a dedicated help.md with native parameter names and constraints. Multinomial NB needs nonnegative features. SGD/linear SVM often benefit from scaling. `class_weight="balanced"` is available where supported by the selected estimator. Voting/stacking require explicitly named base estimators; keep preprocessing inside each base pipeline when internal cross-validation is involved.

For clustering, `nil.evaluate(..., task="clustering")` reports cluster/noise counts, bounded-sample silhouette when defined, and optional adjusted-Rand/NMI against reference labels. Silhouette uses the fitted pipeline's transformed feature space. For transductive DBSCAN/agglomerative clustering, supply `cluster_labels` explicitly and align them with the evaluated observations; these estimators cannot generally predict unseen samples. Anomaly labels retain native -1/+1 semantics; unlabeled evaluation reports outlier fraction rather than inventing accuracy.

## 3. Splits and cross-validation

```python
cv = nil.cross_validate(
    model, *data.train, cv=3, seed=7,
    scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
)
print(cv.summary)
```

Splits support random, stratified, group-disjoint and chronological data. Chronological inputs must already be sorted; a gap can exclude observations between training/test. Group test_size refers to groups, not an exact sample fraction. CV adds stratified-group folds and accepts native splitters or integer-index fold iterables. Supplied groups may not cross train/test boundaries, including custom folds. Invalid indices/overlap are rejected before fitting.

CV clones the whole estimator/pipeline per fold and passes corresponding sample weights to both fitting and scoring. The supplied estimator remains unfitted/unmodified. Fit warnings, fit/score times, fold sizes and scores are returned. Default scoring is accuracy for classifiers and negative RMSE for regressors. Explicit unsupervised/custom scorers are supported. Native loss scorer signs are retained; fold summaries are means/std of fold scores, not pooled predictions. `n_jobs=1` is the default; parallel execution is explicit.

## 4. Bounded tuning

```python
search = nil.tune(
    model, *data.train,
    {"model__alpha": [0.01, 0.1, 1.0]},
    method="grid", max_trials=3, cv=3, max_fits=10, seed=7,
)
print(search.best_params, search.best_score)
print(search.evaluate(*data.test))
```

Grid/random search uses the same fold isolation, scoring and weight rules. Pipeline parameters use the stable `model__`, `preprocess__`, or `select__` namespaces. A single selection metric is required and maximized; negative-loss scorers remain negative. Fit limits include the final full-training-data refit and are checked before fitting or sampling random parameter distributions. Failed parameter trials are recorded; all failed trials raise `SearchFailed` with details instead of returning a fake best model. Search choices/summary are logged to the final run; full trial/fold details remain in the returned result.

Keep the final test split out of tuning. Nested cross-validation, resumable distributed searches and hard native-process timeouts are not provided. Budgeting fit counts does not bound the cost of an individual native fit. Seeds do not override an estimator's own random_state; set both explicitly when needed.

## 5. Inspection, calibration and diagnostics

```python
importance = nil.inspection.feature_importance(
    result, *data.test, method="permutation", repeats=5, seed=7,
)
curve = nil.inspection.learning_curve(
    model, *data.train, train_sizes=(0.5, 1.0), cv=3, max_fits=6,
)
print(nil.inspection.describe(result))
```

Native importances and coefficient reports use transformed feature names where available; permutation importance evaluates the full pipeline against held-out raw features. Coefficients depend on scaling and importance is not causality. Learning curves return training/test scores and warnings under an explicit total fit budget. Built-in chronological folds preserve training prefixes; chronological custom fold iterables require `shuffle=False`.

`nil.inspection.calibrate(classifier_pipeline, X, y, cv=3, groups=...)` fits preprocessing inside each base calibration fold. Calibration returns a ClassicalResult and can accept weights when the base estimator supports them. Native sklearn versions may issue a pipeline-weight warning even when namespaced base-model fit parameters forward weights; captured warnings are exposed rather than hidden. Convergence warnings from fitting/CV/refitting remain inspectable.

## Additional workflows

- `nil.preprocessing.auto()` infers supported DataFrame column roles at training
  fit; `infer_columns` previews roles. Ordinal/target encoders, text vectorizers
  and polynomial features have explicit helpers. Choose `task='regression'` for
  continuous target encoding; keep encoders inside pipelines and respect the
  documented group/time restrictions on their inner folds.
- `nil.data.snapshot` and `load_snapshot` provide immutable JSONL datasets with
  checksum/count/byte validation and fresh independent records on reload.
- `nil.imbalance` supplies explicit fold-local SMOTE/random resampling and fast
  held-out weighted threshold selection. Sample weights cannot be automatically
  aligned after resampling, so NIL rejects that combination.
- `nil.streaming.create` adds frozen preprocessing, partial_fit, complete
  checkpoints and failure guards. Input replay/offset management remains explicit.
- `nil.uncertainty.calibrate` creates finite-sample split-conformal regression
  intervals or LAC prediction sets without refitting; separate calibration and
  final test data are required. Refitting invalidates calibration.
- `nil.tune(method='bayesian', storage=...)` provides bounded Optuna TPE search
  with restart and input/config/version isolation. `nested_validate` estimates
  performance without selecting parameters on outer test folds.
- `nil.inspection.partial_dependence` supplies bounded numeric PDP/ICE;
  `compare` ranks candidate clones on identical CV folds.

Each has a bundled help guide. Run `python scripts/verify_classical.py --extended`
to require the preparation/imbalance/search dependencies rather than skip them,
and `python nil-sdk/examples/classical_extensions.py` for a runnable workflow.
These helpers do not provide hard RAM/time limits or complete specialized-task
engines. See [the coverage ledger and next fixes](CLASSICAL_GAPS.md).

## Current verification

The runnable `nil-sdk/examples/classical_workflow.py --output ./nil_runs` example exercises mixed/missing/categorical preprocessing, fitting, held-out metrics, CV, tuning, importances, learning curves and a saved-pipeline prediction roundtrip using only NIL's public API.

Scikit-learn 1.9.1 is installed in the isolated development environment. The
classical acceptance suite now includes CPU XGBoost 3.4.1. It checks
pipeline leakage, mixed/missing/unseen categories, sparse/named inputs, weighted
metrics, artifacts, native adapters, splits, CV, bounded search, calibration,
importances and learning curves. The expanded mirror suite also checks public
module exports, nested imports, native identity/serialization, text search and
NIL persistence, multioutput regression, incremental learning and experimental
enablement. No mocked estimator results substitute for native acceptance.
Including the mirror and workflow extensions, native acceptance passes with
scikit-learn 1.9.1 (137 passed, one skipped) and 1.4.2 (135 passed, one skipped;
the export counts differ by version). The full classical environment passes
978 tests with four optional-dependency skips; the separate core/neural environment
passes 876 tests with 12 optional-dependency skips. Runnable workflow examples
also pass using an installed NIL wheel; saved-pipeline predictions roundtrip
exactly on the synthetic classical example. These are correctness checks,
not useful-model-quality or old-hardware performance guarantees.

```sh
python scripts/verify_classical.py
# Require auto preparation, resampling and persistent Bayesian tuning:
python scripts/verify_classical.py --extended
# Require named DataFrame and XGBoost checks as well:
python scripts/verify_classical.py --full
```

The verification entry point fails immediately when scikit-learn is unavailable, rather than reporting a skip-only native acceptance success. Run it from the source repository; full verification additionally requires pandas and an XGBoost backend (boosting or boosting-cpu extra).

Core discovery/help and neural regression checks can run in their own environments.
This extension does not claim complete classical-ML coverage or production/hardware acceptance.

Classical and specialized workflows now include forecasting, ranking, sparse recommendation, survival, optional LightGBM/CatBoost, bounded Shapley explanations, typed transactional datasets, distributed random search and additional conformal contracts. See [supported scope and acceptance](CLASSICAL_SPECIALIZED.md).
