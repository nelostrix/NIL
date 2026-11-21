# validation

Split, cross-validate and tune classical pipelines with explicit fit budgets.

## Usage

```python
import nil

X = [[float(i)] for i in range(30)]
y = [2 * i + 1 for i in range(30)]
split = nil.split(X, y, seed=7)
model = nil.pipeline(nil.model("ridge_regression"), preprocess=nil.preprocessing.numeric())
cv = nil.cross_validate(model, *split.train, cv=3)
search = nil.tune(model, *split.train, {"model__alpha": [0.01, 0.1, 1.]},
                  method="grid", max_trials=3, cv=3, max_fits=10)
print(search.best_params, search.evaluate(*split.test))
```

## Parameters

split methods: random, stratified, group, time. CV also supports stratified_group, custom splitters and integer-index fold iterables. groups and sample_weight remain aligned. tune takes method, max_trials, max_fits, cv, one scorer and seed. n_jobs=1 by default.
`tune(method='bayesian', storage=..., study_name=...)` adds optional resumable TPE
search. `nested_validate(..., outer_cv=5, inner_cv=3, max_trials=10, max_fits=200)`
tunes inside each outer training fold and reports untouched outer test scores.
Default multilabel/multioutput folds use KFold rather than invalid single-label
stratification. Multioutput multiclass defaults to subset-accuracy scoring.

## Output

DataSplit with indices/train/test pairs; CVResult with fold scores/times/warnings; SearchResult with model, best_params, best_score, trials and fit-result methods.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Time splits assume already sorted input and allow a gap. Group test_size is a fraction of groups. Loss scorers retain sklearn negative-score conventions and tuning maximizes the scorer. Fit budgets limit counts, not native runtime or memory. Weight support depends on the estimator/scorer. Use nil.imbalance pipelines for explicit fold-local resampling. Nested validation reports estimates; it does not choose a final model from outer test scores. Bayesian persistence/quotas follow the selection help guide.
