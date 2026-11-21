"""Fold isolation, reproducibility, explicit fit budgets and search failures."""
import pytest

pytest.importorskip("sklearn")
import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

import nil


def problem():
    random = np.random.RandomState(7)
    X = random.normal(size=(60, 3))
    return X, 2 * X[:, 0] - X[:, 1] + random.normal(scale=0.1, size=60)


def test_stratified_group_and_chronological_splits_are_reproducible():
    X = np.arange(120).reshape(60, 2)
    y = np.tile([0, 0, 0, 0, 1], 12)
    first = nil.split(X, y, method="stratified", seed=11)
    second = nil.split(X, y, method="stratified", seed=11)
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    assert sum(first.y_test) == 2 or sum(first.y_test) == 3
    groups = np.repeat(np.arange(12), 5)
    grouped = nil.split(X, y, method="group", groups=groups)
    assert not set(groups[grouped.train_indices]) & set(groups[grouped.test_indices])
    chronological = nil.split(X, y, method="time", test_size=10, gap=3)
    assert chronological.train_indices[-1] + 4 == chronological.test_indices[0]
    with pytest.raises(ValueError, match="groups"):
        nil.split(X, y, groups=groups)


def test_cv_clones_preprocessing_and_matches_native_fold_fit():
    X, y = problem()
    pipeline = nil.pipeline(nil.model("ridge_regression", alpha=0.5), preprocess=nil.preprocessing.numeric())
    splitter = nil.validation.folds(method="random", n_splits=3, seed=2)
    plan = list(splitter.split(X, y))
    result = nil.cross_validate(pipeline, X, y, cv=plan, scoring="neg_mean_squared_error")
    expected = []
    for train, test in plan:
        trained = clone(pipeline).fit(X[train], y[train])
        expected.append(-np.mean((trained.predict(X[test]) - y[test]) ** 2))
    np.testing.assert_allclose(result.scores["neg_mean_squared_error"], expected)
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline)


def test_group_cv_and_weighted_scores_reject_leakage_and_invalid_weights():
    X, y = problem()
    groups = np.repeat(np.arange(12), 5)
    weights = np.linspace(1, 3, len(y))
    model = nil.pipeline(nil.model("linear_regression"), preprocess=nil.preprocessing.numeric())
    result = nil.cross_validate(model, X, y, groups=groups, sample_weight=weights, cv=3,
                                scoring={"mae": "neg_mean_absolute_error", "r2": "r2"})
    assert len(result.folds) == 3 and set(result.summary) == {"mae", "r2"}
    with pytest.raises(ValueError, match="group"):
        nil.cross_validate(model, X, y, groups=groups, cv=[(np.arange(0, 30, 2), np.arange(1, 30, 2))])
    with pytest.raises(ValueError, match="overlap"):
        nil.cross_validate(model, X, y, cv=[([0, 1, 2], [2, 3])])
    with pytest.raises(ValueError, match="max_fits"):
        nil.cross_validate(model, X, y, cv=5, max_fits=2)


def test_bounded_grid_and_random_search_repeat_and_roundtrip(tmp_path):
    X, y = problem()
    pipeline = nil.pipeline(nil.model("ridge_regression"), preprocess=nil.preprocessing.numeric())
    space = {"model__alpha": [0.01, 0.1, 1.0]}
    first = nil.tune(pipeline, X, y, space, method="grid", max_trials=3, cv=3,
                     seed=3, max_fits=10, output_dir=tmp_path / "first")
    second = nil.tune(pipeline, X, y, space, method="grid", max_trials=3, cv=3,
                      seed=3, max_fits=10, output_dir=tmp_path / "second")
    assert first.best_params == second.best_params
    assert first.best_score == pytest.approx(second.best_score)
    np.testing.assert_allclose(first.predict(X), second.predict(X))
    path = first.save(tmp_path / "best")
    np.testing.assert_allclose(nil.load(path, trusted=True).predict(X), first.predict(X))
    random = nil.tune(pipeline, X, y, space, max_trials=2, cv=2, seed=3, output_dir=tmp_path / "random")
    assert len(random.trials) == 2
    with pytest.raises(ValueError, match="max_trials"):
        nil.tune(pipeline, X, y, space, method="grid", max_trials=2)
    with pytest.raises(ValueError, match="max_fits"):
        nil.tune(pipeline, X, y, space, method="grid", max_trials=3, max_fits=9)


def test_search_records_failures_and_does_not_claim_an_all_failed_search_succeeded(tmp_path):
    X, y = problem()
    result = nil.tune(nil.model("ridge_regression"), X, y, {"alpha": [-1., 1.]},
                      method="grid", max_trials=2, cv=2, output_dir=tmp_path)
    assert result.best_params == {"alpha": 1.}
    assert [trial["status"] for trial in result.trials] == ["failed", "completed"]
    with pytest.raises(nil.validation.SearchFailed) as failure:
        nil.tune(nil.model("ridge_regression"), X, y, {"alpha": [-1.]}, method="grid", max_trials=1, cv=2)
    assert failure.value.trials[0]["status"] == "failed"


def test_search_fit_budget_is_checked_before_sampling_distributions():
    class Distribution:
        def rvs(self, **options):
            raise AssertionError("Budget must be checked before sampling")

    X, y = problem()
    with pytest.raises(ValueError, match="max_fits"):
        nil.tune(nil.model("ridge_regression"), X, y, {"alpha": Distribution()},
                 max_trials=1000, cv=3, max_fits=10)


def test_unavailable_probability_method_is_a_failed_trial_not_a_fake_best_model(tmp_path):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=40, n_features=4, n_redundant=0, random_state=7)
    result = nil.tune(nil.model("svc", random_state=7), X, y,
                      {"probability": [False, True]}, method="grid", max_trials=2,
                      cv=2, scoring="neg_log_loss", output_dir=tmp_path)
    assert result.best_params == {"probability": True}
    assert [trial["status"] for trial in result.trials] == ["failed", "completed"]
