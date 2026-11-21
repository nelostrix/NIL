"""Held-out inspection, calibration and learning curves use real native estimators."""
import pytest

pytest.importorskip("sklearn")
import numpy as np
from sklearn.datasets import make_classification, make_regression

import nil


def test_permutation_importance_and_coefficients_describe_the_right_features(tmp_path):
    X, y = make_regression(n_samples=80, n_features=4, n_informative=1, shuffle=False, random_state=2)
    model = nil.pipeline(nil.model("linear_regression"), preprocess=nil.preprocessing.numeric())
    fitted = nil.fit(model, X[:60], y[:60], output_dir=tmp_path)
    native = nil.inspection.feature_importance(fitted, method="coefficients")
    assert len(native["features"]) == 4
    assert np.argmax(native["values"]) == 0
    permuted = nil.inspection.feature_importance(fitted, X[60:], y[60:], method="permutation", repeats=3, seed=1)
    assert np.argmax(permuted["values"]) == 0
    assert nil.inspection.describe(fitted)["feature_count"] == 4


def test_calibration_refits_entire_base_pipeline_inside_folds(tmp_path):
    X, y = make_classification(n_samples=90, n_features=4, n_redundant=0, random_state=4)
    base = nil.pipeline(nil.model("linear_svc", random_state=2), preprocess=nil.preprocessing.numeric())
    result = nil.inspection.calibrate(base, X, y, cv=3, output_dir=tmp_path)
    probabilities = result.predict_proba(X)
    assert probabilities.shape == (90, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
    assert "log_loss" in result.evaluate(X, y)
    assert not hasattr(base.named_steps["preprocess"].named_steps["impute"], "statistics_")


def test_learning_curve_respects_fit_budget_and_returns_train_and_held_out_scores():
    X, y = make_regression(n_samples=60, n_features=3, random_state=3)
    model = nil.pipeline(nil.model("ridge_regression"), preprocess=nil.preprocessing.numeric())
    report = nil.inspection.learning_curve(model, X, y, train_sizes=(0.5, 1.), cv=3, max_fits=6)
    assert report["fit_count"] == 6
    assert report["train_sizes"] == [20, 40]
    assert len(report["points"][0]["test_scores"]["neg_root_mean_squared_error"]) == 3
    assert report["points"][0]["train_scores"]
    with pytest.raises(ValueError, match="max_fits"):
        nil.inspection.learning_curve(model, X, y, train_sizes=(0.5, 1.), cv=3, max_fits=5)


def test_convergence_warning_is_returned_instead_of_disappearing(tmp_path):
    X, y = make_classification(n_samples=60, n_features=8, n_redundant=0, random_state=8)
    result = nil.fit(nil.model("logistic_regression", max_iter=1, tol=1e-12), X, y, output_dir=tmp_path)
    assert any(item["category"] == "ConvergenceWarning" for item in result.warnings)
    assert nil.inspection.describe(result)["warnings"] == result.warnings
