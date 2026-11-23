# inspection

Inspect coefficients/importances, calibrate safely, and measure bounded learning curves.

## Usage

```python
import nil

X = [[float(i), float(i % 3)] for i in range(30)]
y = [2 * i + 1 for i in range(30)]
model = nil.pipeline(nil.model("linear_regression"), preprocess=nil.preprocessing.numeric())
result = nil.fit(model, X[:20], y[:20])
print(nil.inspection.describe(result))
print(nil.inspection.feature_importance(result, X[20:], y[20:], method="permutation"))
curve = nil.inspection.learning_curve(model, X, y, cv=3, train_sizes=(0.5, 1.), max_fits=6)
print(curve["train_sizes"])
```

## Parameters

feature_importance method=auto/native/coefficients/permutation, repeats, scoring and held-out data. calibrate accepts a complete classifier pipeline, CV/groups and optional weights. learning_curve takes sizes, CV/groups/gap, scorer and max_fits.
`partial_dependence(model, data, columns=[0], kind='average'/'individual'/'both',
grid_resolution=30, max_samples=200, max_evaluations=100000)` computes PDP/ICE
on raw input columns using the complete fitted model. `compare({name: model},
features, targets, cv=5, scoring=None, max_fits=100)` compares candidates on
identical fold plans under a shared fit budget.

## Output

Plain dictionaries of named importances, diagnostics and learning-curve points; calibration returns a ClassicalResult.
PDP/ICE returns numeric grids/effects and sampled row indices; no plotting
package is required. Comparison returns ranking, CV scores, summaries and times.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Importance is not causality. Coefficients depend on feature scaling; transformed names can differ from raw columns. Native importance is not a held-out estimate. Use permutation data that was not used in fitting. Put preprocessing inside each stacking/calibration base pipeline. Chronological custom fold iterables require shuffle=False for learning curves. Native calibration may record a pipeline weight-support warning even when namespaced fit parameters forward base-model weights.
PDP supports one/two input columns and uses brute evaluation. Its evaluation
bound counts sampled input rows times the maximum grid size, not inner trees,
bytes or wall time. Correlated features can make interventions unrealistic.
Model comparisons rank CV results; final test data must remain separate.


## Bounded interventional Shapley explanations

`shap_values(model,X,background,permutations=8,max_rows=32,max_background=32,
max_features=64,max_evaluations=100000,max_dense_bytes=32000000,
class_index=None,seed=0)` averages complete random feature permutations against
the supplied empirical background. Regression/default output is `predict`; pass
class_index for one predict_proba column. The result contains values, base_values,
predictions, transformed feature_names, Monte Carlo standard_errors (None for one
permutation), an evaluation-row budget and JSON-compatible to_dict().

For a fitted preprocessing pipeline, explanations describe its transformed
features and final estimator. Names come from get_feature_names_out when
available. DataFrame columns and background order must match the fitted schema. Numeric finite dense/sparse inputs are checked before sparse-to-dense
conversion. Row/feature/evaluation limits reject rather than truncate. Tensor
size depends on the explicitly chosen budgets; arbitrary preprocessing/model
code can itself allocate memory outside NIL's checks. Background rows are not
automatically sampled: choose a representative background explicitly.

This is a model-agnostic permutation approximation to interventional Shapley
values, not a TreeSHAP engine, conditional/correlation-aware attribution or a
causal explanation. It does not install/import the shap package. Additivity is
checked against deterministic scalar predictions; linear models have analytic
parity tests. Interactions need enough permutations, and estimated standard
errors are not universal confidence guarantees. See the primary method context:
https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html
