# selection

Bounded optional Bayesian TPE tuning with persistent restart support.

## Usage

```python
import nil

X = [[float(i)] for i in range(30)]
y = [2 * i + 1 for i in range(30)]
result = nil.tune(nil.model("ridge_regression"), X, y,
    {"alpha": {"type": "float", "low": 0.01, "high": 10., "log": True}},
    method="bayesian", cv=3, max_trials=3, max_fits=10,
    storage="sqlite:///nil-search.db", study_name="ridge")
print(result.best_params, result.study_name)
```

## Parameters

Categorical spaces use nonempty lists of JSON scalars. Numeric specs use
`type='int'/'float'`, low, high, optional log or positive step. Log bounds must
be positive and cannot use step. max_trials counts total persisted attempts,
including failures and unfinished trials; max_fits includes final refit.
Supply storage to persist and resume. `nil.selection.bayesian` is also callable.

## Output

SearchResult with model, best_params, best_score, trials and effective study_name.
The effective name appends a fingerprint of inputs, groups, weights, fold plan,
estimator, scoring, space, seed and NIL/numerical/backend versions. Changed inputs
use a separate study, never cached scores from a different problem.

## Requirements

`nil-sdk[search]`. Bayesian studies use Optuna TPE; the first two startup trials
are random. Single-metric native scores are maximized, including negative losses.

## Limitations

One invocation owns a study's trial budget. Persistent storage supports restart;
simultaneous workers do not have an atomic global quota in this wrapper.
Unfinished trials consume budget and are not silently re-run. Exact sampler RNG
continuation after restart is not guaranteed. Data hashing scans inputs once;
large databases/history can grow on disk. Fit budgets are counts, not wall-time
or RAM caps. No distributed worker launcher, pruning, remote storage setup or
safe loading of untrusted storage is provided. Use native Optuna for advanced
multiworker orchestration with explicit resource management.


## Distributed random search

Distributed random search needs only the classical backend; Optuna is required
only by the separate Bayesian TPE API.

`distributed_search(estimator,X,y,space,storage='study.sqlite',max_trials=10,
cv=3,scoring=None,max_fits=100,seed=0)` binds a shared local SQLite ledger to the
exact data, estimator, folds, weights, metric and backend versions. Each worker
opens that same definition and calls `search.run(estimator,X,y,
cv_plan=search.cv_plan,scoring=search.scoring,worker='unique-name')`.
Use a context manager or close the ledger. Worker inputs are fingerprinted again
before claiming work. Completed/failed attempts persist; all processes share one
total trial count. Index-seeded parameters are identical after restart without
relying on process-local RNG. `search.best()` selects a completed trial; final
`nil.fit` is explicit and adds an additional fit outside the CV budget.

Leases renew while a worker fits. Expired/dead-worker attempts become failed and
consume their slots, so unknown partial work is never automatically retried past
the reserved total fit budget. Stale tokens cannot commit results. A replacement
worker continues remaining pending candidates. Worker death may lose one result,
not duplicate the attempt. A worker process itself is not killed by lease expiry;
use nil.execution to bound its wall time/memory when required. Reliable local
SQLite/filesystem and reasonably synchronized host clocks are required. This is
same-host multi-process random search, not a multi-host scheduler. Adaptive TPE
remains the separate single-owner bayesian API; no distributed TPE continuation
is claimed. No automatic best-model refit or hidden background worker starts.
