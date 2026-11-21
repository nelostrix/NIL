# Classical and specialized ML: supported completion scope

The nine classical/specialized checklist milestones now have implementations,
help, failure contracts and acceptance tests. This does not mean NIL implements
every algorithm in these fields or replaces its optional numerical backends.
The base wheel has no dependencies; no backend is auto-installed or launched.

| Checklist area | Supported API | Accepted scope |
|---|---|---|
| LightGBM/CatBoost | nil.model / nil.boosting | Native CPU classifier/regressor pipelines, one-thread defaults, trusted complete persistence |
| Forecasting | nil.forecasting | Direct univariate lag/horizon regression, fixed-cadence calendar features, cloned rolling-origin backtests and gaps |
| Ranking | nil.ranking | Stable query regrouping, query-held-out splits, native LGBMRanker, per-query NDCG/MAP |
| Recommendation | nil.recommendation | Sparse item-cosine/popularity, excluded seen items, bounded top-k/candidate work, cold starts, held-out precision/recall/NDCG |
| Survival | nil.survival | Native lifelines Cox PH/Kaplan–Meier, right-censor validation, risk/survival curves and held-out Harrell concordance |
| SHAP-style explanation | nil.inspection.shap_values | Bounded numeric interventional permutation Shapley approximation, transformed feature names, explicit background and additivity |
| Typed data/offsets | nil.datasets | Scalar logical schemas, bounded immutable JSONL, compatible migrations, SQLite transactional records+offsets and idempotent batches |
| Distributed tuning | nil.selection.distributed_search | Same-host multi-process random search, global CV-attempt budget, leases, stale-token refusal and index-seeded continuation |
| Advanced uncertainty | nil.uncertainty | Simultaneous multioutput, fixed-group Mondrian, known-ratio covariate shift and exchangeable-block regression intervals |

## Unified surface and persistence

`nil.catalog.tree()` lists forecasting, ranking, recommendation and survival as
shallow groups alongside the existing classical/neural groups. Constructors are
`lag_forecaster`, `query_ranker`, `item_cosine`, `cox_ph`, `kaplan_meier`, plus
LightGBM/CatBoost classifier/regressor names. Their matching help.md files and
module help ship offline. Ordinary classification/regression use nil.fit; the
specialized module fit functions express their different input contracts.

Fitted specialized estimators and calibrated wrappers use nil.save/load with the
existing explicit trust/hash/environment checks. New wrappers retain their
required optional backend versions. Explanations are data, not fitted models:
use explanation.to_dict() for JSON-compatible export. Dataset snapshots and
search ledgers have their own documented persistence protocols.

## Statistical and resource boundaries

- Forecasts are fixed-horizon direct models, not a full ARIMA/state-space,
  multivariate, irregular-time or hierarchical forecasting suite. Rolling tests
  inspect only each training prefix; future calendar values are known covariates.
- Ranking uses relevance grades and prespecified queries, not unbiased click
  feedback or causal ranking. Stable score ties retain input order. Query-level
  splitting is required before preprocessing/training.
- Recommendation is a sparse retrieval baseline, not neural collaborative
  filtering/ALS or a vector database. The preflight co-occurrence budget is
  conservative. External user/item ID vocabularies remain caller-owned.
- Survival supports right censoring under native lifelines assumptions, not
  competing risks/time-varying covariates or IPCW Brier scores. A concordance
  smoke/parity check is not evidence of clinical or causal validity.
- Permutation explanations are model-agnostic approximations in transformed
  numeric coordinates, not TreeSHAP or causal/correlation-conditional attribution.
  Budgets reject instead of silently truncating; arbitrary user preprocessing
  and native model code can allocate outside those checks.
- Logical schemas are not Arrow/Pandas physical dtype preservation. Local sink
  records and offsets commit together; external broker acknowledgements and
  model weights are not automatically part of that transaction.
- Distributed search reserves top-level CV fit attempts. Expired attempts consume
  their slots and are not silently retried; a replacement continues pending
  candidates. Sampler continuation is deterministic random search, not distributed
  adaptive TPE. Final refitting is explicit and adds an extra fit. Estimator-
  internal fits/threads and hard time/RSS are separate resource contracts.
- Conformal methods require frozen independent models/calibration. Multioutput
  coverage is joint across coordinates; group coverage requires fixed groups
  and within-group exchangeability; weighted coverage needs known correct
  target/source ratios and unchanged Y|X. Whole-block exchangeability is required
  for block coverage. Arbitrary temporal drift or pointwise conditional coverage
  is not covered. Weighted intervals can legitimately be infinite.

## Reproduce acceptance

Use a separately selected development environment with explicit optional extras:

```sh
python scripts/verify_classical.py --full --extended --specialized
PYTHONPATH=nil-sdk python -m pytest nil-sdk/tests -q
PYTHONPATH=nil-sdk python nil-sdk/examples/specialized_workflows.py --full
python scripts/build_sdk_bundle.py --output dist
```

The specialized preflight requires actual LightGBM, CatBoost and lifelines before
running; missing backends are not accepted as skip-only validation. The default
example omits those optional integrations; --full requires them. Source tests
also use actual independent OS processes for shared-budget tuning and killed
writers/workers for transactional/lease recovery. No CUDA/distributed-machine or
old-PC performance acceptance is inferred from CPU functional tests.

Primary implementation references:
[LightGBM ranker](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html),
[CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier),
[lifelines Cox PH](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html),
[permutation Shapley method](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html),
[weighted conformal prediction](https://arxiv.org/abs/1904.06019).

## Recorded source acceptance — 2026-11-21

Full classical development environment: **1021 passed, eight optional skips**.
Python 3.12.16, sklearn 1.9.1, NumPy 2.5.5, SciPy 1.18.1, pandas 3.1.1,
LightGBM 4.7.0, CatBoost 1.2.10 and lifelines 0.30.0. The complete --full example
passes all optional pipeline roundtrips, query fitting and censored held-out
survival evaluation. The catalog exposes 54 constructors in ten shallow groups;
101 offline help guides cover every public module and catalog constructor.

Tests include actual separate worker processes sharing the search ledger and
actual killed SQLite writers/leased workers. On Linux the process acceptance uses a separate preloaded forkserver, so
workers share read-only numerical-library pages without forking the threaded
parent. Each worker opens its own SQLite connection. Other platforms use spawn. Other hosts still require their own platform
acceptance. Eight optional skips in the full
classical environment are not counted as completed DL/RL/hardware capabilities.

Minimum classical matrix (sklearn 1.4.2 / NumPy 1.26.4): **63 passed, 11 explicit
optional skips** for specialized/catalog checks. LightGBM/CatBoost/lifelines are
not installed in that environment; their native acceptance comes from the full
current environment above, not from skipped tests.
