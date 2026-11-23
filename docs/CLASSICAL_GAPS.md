# Classical ML coverage and remaining boundaries

The nine planned classical/specialized milestones have scoped implementations.
The base wheel installs only NIL; numerical backends remain explicit optional
extras. Public sklearn APIs remain available through nil.sklearn. See
[CLASSICAL_SPECIALIZED.md](CLASSICAL_SPECIALIZED.md) for contracts and acceptance.

| Area | Implemented | Beyond the accepted scope |
|---|---|---|
| Preparation | Fold-local numeric/categorical pipelines, target/ordinal encoding, text/polynomial features | Automatic domain-role inference and arbitrary complex dtype support |
| Classification | Weighted single/multilabel/multioutput fitting, metrics, safe CV and native sklearn APIs | Every third-party estimator/version is not independently certified |
| Model engines | Classical catalog, CPU XGBoost, native LightGBM/CatBoost classifier/regressor pipelines | GPU backend acceptance and an independent NIL numerical replacement |
| Forecasting | Direct univariate lag/horizon model, fixed-cadence calendar features and rolling-origin gap backtests | ARIMA/state-space, irregular/multivariate/probabilistic/hierarchical forecasting |
| Ranking | Query-safe training/splitting, native LGBMRanker, per-query NDCG/MAP | Counterfactual click learning and arbitrary ranking backend contracts |
| Recommendation | Bounded sparse item-cosine/popularity, cold starts, top-k and held-out metrics | Neural/ALS, ANN services and external ID vocabulary management |
| Survival | Native Cox PH/Kaplan–Meier, right-censor validation, curves and held-out concordance | Competing risks, time-varying models, IPCW/time-dependent Brier metrics |
| Inspection | Native/permutation importance, PDP/ICE and bounded interventional permutation Shapley | TreeSHAP, causal/conditional attribution and unrestricted dense conversions |
| Data | Typed scalar JSONL snapshots, compatible schema migration, atomic SQLite records+offsets | Arrow/Pandas physical dtype fidelity and transactions spanning model weights/external brokers |
| Streaming | partial_fit, frozen preprocessing, model checkpoints, bounded JSONL cursor resume and a durable record sink | Automatic atomic coupling of external ingestion and arbitrary training state |
| Selection | Grid/random/TPE/nested CV, same-host distributed random-search ledger, global attempts, leases and deterministic continuation | Distributed adaptive TPE, multi-host launching and hard estimator-internal resource quotas |
| Uncertainty | Ordinary split conformal, simultaneous multioutput, fixed groups, known-ratio weighting and exchangeable blocks | Arbitrary pointwise conditional coverage, drift-safe temporal guarantees, adaptive/censored uncertainty |
| Runtime | Detached supervised execution, Linux per-process address-space caps and safe recovery | Aggregate RSS/GPU quotas, other target-host acceptance and old-PC certification |

## Verification

```sh
python scripts/verify_classical.py --full --extended --specialized
PYTHONPATH=nil-sdk python -m pytest nil-sdk/tests -q
PYTHONPATH=nil-sdk python nil-sdk/examples/specialized_workflows.py --full
python scripts/build_sdk_bundle.py --output dist
```

The specialized preflight requires actual optional LightGBM/CatBoost/lifelines;
missing integrations cannot count as a successful skip-only acceptance. Tests
on the minimum sklearn environment cover the available generic workflows;
uninstalled specialized backends are explicitly skipped there. Current CPU
native integrations are verified separately in the full development environment.
No one workload/smoke test establishes universal speed or model quality.
