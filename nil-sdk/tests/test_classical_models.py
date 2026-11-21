"""Every catalog adapter constructs the advertised native estimator and fits small data."""
import pytest

pytest.importorskip("sklearn")
import numpy as np
from sklearn.datasets import make_classification, make_regression

import nil
from nil.catalog import _MODELS


@pytest.mark.parametrize("name", [name for name, entry in _MODELS.items() if entry[0] not in {"forecasting","ranking","recommendation","survival"}])
def test_catalog_model_fits_native_data(name):
    group, module, constructor, extra = _MODELS[name]
    if extra == "torch":
        pytest.skip("Neural acceptance belongs to its own suite")
    if extra == "boosting":
        pytest.importorskip("xgboost")
    if extra in {"lightgbm", "catboost"}:
        pytest.importorskip(extra)
    classification = group in {"classification", "feature_selection"}
    if classification:
        X, y = make_classification(n_samples=48, n_features=4, n_redundant=0, random_state=7)
    else:
        X, y = make_regression(n_samples=48, n_features=4, noise=0.1, random_state=7)
    options = {}
    if name.startswith(("voting_", "stacking_")):
        base = "logistic_regression" if classification else "linear_regression"
        options["estimators"] = [("first", nil.model(base)), ("second", nil.model(base))]
        if name.startswith("stacking_"):
            options["cv"] = 2
    if name == "calibrated_classifier":
        options.update(estimator=nil.model("logistic_regression"), cv=2)
    if name == "select_from_model":
        options["estimator"] = nil.model("logistic_regression")
    if name == "select_k_best":
        options["k"] = 2
    if name in {"random_forest_classifier", "random_forest_regressor", "isolation_forest"}:
        options.update(n_estimators=5, n_jobs=1, random_state=7)
    if extra == "boosting":
        options.update(n_estimators=5, max_depth=2, n_jobs=1, random_state=7)
    if name == "multinomial_nb":
        X = np.abs(X)
    estimator = nil.model(name, **options)
    assert type(estimator).__name__ == constructor
    estimator.fit(X, y if group in {"regression", "classification", "feature_selection"} else None)
    if hasattr(estimator, "predict"):
        predicted = estimator.predict(X[:3])
        assert len(predicted) == 3 and np.isfinite(predicted).all()
    elif hasattr(estimator, "transform"):
        assert estimator.transform(X[:3]).shape[0] == 3
    else:
        assert hasattr(estimator, "labels_")
