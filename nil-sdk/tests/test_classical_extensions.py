"""Acceptance of missing workflow pieces using actual optional engines."""

import pytest

pytest.importorskip("sklearn")
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification, make_multilabel_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.metrics import hamming_loss, f1_score

import nil


def test_multioutput_evaluation_and_default_cv(tmp_path):
    from sklearn.multioutput import MultiOutputClassifier
    X, y = make_classification(n_samples=90, n_features=4, random_state=3)
    targets = np.column_stack((y, np.where(X[:, 0] > 0, 2, 3)))
    model = MultiOutputClassifier(nil.model("logistic_regression"))
    result = nil.fit(model, X[:60], targets[:60], validation=(X[60:], targets[60:]), output_dir=tmp_path)
    predicted = model.predict(X[60:])
    assert result.metrics["subset_accuracy"] == np.mean(np.all(targets[60:] == predicted, axis=1))
    assert result.metrics["hamming_loss"] == np.mean(targets[60:] != predicted)
    assert len(nil.inspection.describe(result)["classes"]) == 2
    cv = nil.cross_validate(clone(model), X, targets, cv=3)
    assert "subset_accuracy" in cv.scores
    assert len(cv.folds) == 3


def test_multilabel_metrics_match_native(tmp_path):
    X, y = make_multilabel_classification(n_samples=90, n_features=5, n_classes=3, random_state=3)
    result = nil.fit(nil.model("random_forest_classifier", n_estimators=5, random_state=0), X[:60], y[:60], output_dir=tmp_path)
    weight = np.linspace(1, 2, 30)
    scores = result.evaluate(X[60:], y[60:], sample_weight=weight)
    predicted = result.predict(X[60:])
    assert scores["f1_micro"] == pytest.approx(f1_score(y[60:], predicted, average="micro", sample_weight=weight))
    assert scores["hamming_loss"] == pytest.approx(hamming_loss(y[60:], predicted, sample_weight=weight))
    assert len(nil.cross_validate(clone(result.model), X, y, cv=3).folds) == 3


def test_modern_cluster_tags_and_mirror_star_import(tmp_path):
    from nil.sklearn.cluster import MiniBatchKMeans
    namespace = {}
    exec("from nil.sklearn.utils import *", namespace)
    assert "check_array" in namespace
    X = np.random.RandomState(0).normal(size=(40, 2))
    model = MiniBatchKMeans(n_clusters=2, random_state=0, n_init=1)
    result = nil.fit(model, X, output_dir=tmp_path)
    assert result.task == "clustering"
    assert result.evaluate(X)["clusters"] == 2


def test_auto_columns_are_train_fitted_and_complete_pipeline_roundtrips(tmp_path):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"amount": [1., 2., np.nan, 4., 5., 6.],
                          "kind": ["a", "b", "a", "b", "a", "b"]})
    assert nil.preprocessing.infer_columns(frame) == {"numeric": ["amount"], "categorical": ["kind"]}
    model = nil.pipeline(nil.model("logistic_regression"), preprocess=nil.preprocessing.auto())
    trained = nil.fit(model, frame, [0, 1, 0, 1, 0, 1], output_dir=tmp_path / "runs")
    test = pd.DataFrame({"amount": [1000.], "kind": ["unseen"]})
    imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[0] == 4
    trained.predict(test)
    assert imputer.statistics_[0] == 4
    path = trained.save(tmp_path / "auto")
    np.testing.assert_array_equal(nil.load(path, trusted=True).predict(test), trained.predict(test))
    with pytest.raises(ValueError, match="unsupported dtype"):
        nil.preprocessing.infer_columns(pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3)}))


def test_encoders_text_and_polynomial_features():
    ordinal = nil.preprocessing.ordinal().fit(np.array([["a"], ["b"], [np.nan]], dtype=object))
    assert ordinal.transform([["unseen"]])[0, 0] == -1
    assert ordinal.transform([[np.nan]])[0, 0] == -2
    X = np.array([[f"unique-{i}"] for i in range(30)])
    y = np.arange(30, dtype=float)
    encoder = nil.preprocessing.target_encoder(task="regression", cv=3, seed=0)
    crossfit = encoder.fit_transform(X, y)
    assert not np.allclose(crossfit, encoder.transform(X))
    assert nil.preprocessing.text().fit_transform(["good day", "bad day"]).shape == (2, 3)
    assert nil.preprocessing.text("hashing", n_features=16).transform(["good day"]).shape == (1, 16)
    assert nil.preprocessing.polynomial().fit_transform([[1, 2]]).shape == (1, 5)


def test_streaming_checkpoint_exact_resume_frozen_preprocessing_and_failure_guard(tmp_path):
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = (X[:, 0] > 20).astype(int)
    online = nil.streaming.create(nil.model("sgd_classifier", loss="log_loss", random_state=7),
                                  preprocess=nil.preprocessing.scaler(), classes=[0, 1])
    online.partial_fit(X[:20], y[:20])
    mean = online.preprocess_.mean_.copy()
    path = online.checkpoint(tmp_path / "checkpoint")
    with pytest.raises(ValueError, match="trusted=True"):
        nil.streaming.resume(path)
    resumed = nil.streaming.resume(path, trusted=True)
    online.partial_fit(X[20:], y[20:])
    resumed.partial_fit(X[20:], y[20:])
    assert resumed.batch_count_ == 2
    np.testing.assert_array_equal(online.predict(X), resumed.predict(X))
    np.testing.assert_array_equal(mean, online.preprocess_.mean_)
    with pytest.raises(ValueError):
        online.partial_fit([[50]], [99])
    with pytest.raises(RuntimeError, match="restore"):
        online.partial_fit([[51]], [1])
    with pytest.raises(NotFittedError):
        online.checkpoint(tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    with pytest.raises(ValueError, match="classes"):
        nil.streaming.create(nil.model("sgd_classifier"))


def test_streaming_regression_works_through_nil_fit_and_eval(tmp_path):
    online = nil.streaming.create(nil.model("sgd_regressor", random_state=0))
    result = nil.fit(online, [[0], [1], [2]], [0, 1, 2], output_dir=tmp_path)
    assert result.task == "regression"
    assert np.isfinite(result.evaluate([[3]], [3])["mae"])


def test_conformal_finite_sample_quantile_and_regression_persistence(tmp_path):
    from nil.uncertainty import _quantile
    assert _quantile(np.arange(1, 10), .2) == 8
    with pytest.raises(ValueError, match="too small"):
        _quantile([1, 2], .1)
    model = nil.model("linear_regression").fit([[0], [1], [2]], [0, 2, 4])
    calibration = np.arange(9).reshape(-1, 1) + 10
    y = 2 * calibration[:, 0] + np.arange(1, 10)
    conformal = nil.uncertainty.calibrate(model, calibration, y, alpha=.2)
    np.testing.assert_allclose(conformal.interval([[30]]), [[52, 68]])
    assert nil.uncertainty.evaluate(conformal, [[30]], [60])["coverage"] == 1
    path = nil.save(conformal, tmp_path / "conformal")
    np.testing.assert_array_equal(nil.load(path, trusted=True).interval([[30]]), conformal.interval([[30]]))
    conformal.fit([[0], [1]], [0, 1])
    with pytest.raises(NotFittedError):
        conformal.interval([[2]])


def test_conformal_prediction_sets_and_threshold_persist(tmp_path):
    X, y = make_classification(n_samples=120, n_features=4, random_state=3)
    model = nil.model("logistic_regression").fit(X[:60], y[:60])
    conformal = nil.uncertainty.calibrate(model, X[60:90], y[60:90], alpha=.2)
    probabilities = model.predict_proba(X[90:])
    np.testing.assert_array_equal(conformal.prediction_sets(X[90:]), 1 - probabilities <= conformal.cutoff_)
    assert 0 <= nil.uncertainty.evaluate(conformal, X[90:], y[90:])["coverage"] <= 1
    path = nil.save(conformal, tmp_path / "prediction-sets")
    np.testing.assert_array_equal(nil.load(path, trusted=True).prediction_sets(X), conformal.prediction_sets(X))
    tuned = nil.imbalance.threshold(model, X[60:90], y[60:90], candidates=21)
    np.testing.assert_array_equal(tuned.predict(X[90:]), model.classes_[(probabilities[:, 1] >= tuned.threshold).astype(int)])
    path = nil.save(tuned, tmp_path / "threshold")
    np.testing.assert_array_equal(nil.load(path, trusted=True).predict(X), tuned.predict(X))
    assert nil.evaluate(tuned, X[90:], y[90:])["count"] == 30


def test_resampling_happens_only_during_fit_and_rejects_misaligned_weights(tmp_path):
    pytest.importorskip("imblearn")
    X, y = make_classification(n_samples=80, n_features=4, weights=[.8, .2], random_state=3)
    model = nil.imbalance.pipeline(nil.model("logistic_regression"), resample=nil.imbalance.sampler("over"))
    result = nil.fit(model, X[:60], y[:60], output_dir=tmp_path)
    assert len(result.predict(X[60:])) == 20
    assert len(model.named_steps["resample"].sample_indices_) > 60
    path = result.save(tmp_path / "resampling-model")
    np.testing.assert_array_equal(nil.load(path, trusted=True).predict(X), result.predict(X))
    assert len(nil.cross_validate(clone(model), X, y, cv=3).folds) == 3
    with pytest.raises(ValueError, match="resampling"):
        nil.fit(clone(model), X, y, sample_weight=np.ones(80), output_dir=tmp_path)


@pytest.mark.parametrize("metric", ["f1", "accuracy", "balanced_accuracy"])
def test_vectorized_threshold_matches_weighted_native_grid_oracle(metric):
    from sklearn import metrics
    X, y = make_classification(n_samples=80, n_features=4, random_state=3)
    model = nil.model("logistic_regression").fit(X[:40], y[:40])
    weight = np.linspace(1, 3, 40)
    selected = nil.imbalance.threshold(model, X[40:], y[40:], metric=metric,
                                       candidates=31, sample_weight=weight)
    score = {"f1": metrics.f1_score, "accuracy": metrics.accuracy_score,
             "balanced_accuracy": metrics.balanced_accuracy_score}[metric]
    probability = model.predict_proba(X[40:])[:, 1]
    expected = max(score(y[40:], probability >= value, sample_weight=weight)
                   for value in np.linspace(0, 1, 31))
    assert selected.selection_score_ == pytest.approx(expected)


def test_pdp_ice_compare_and_nested_budget(tmp_path):
    X, y = make_regression(n_samples=45, n_features=3, random_state=3)
    model = nil.model("ridge_regression").fit(X, y)
    report = nil.inspection.partial_dependence(model, X, columns=[0], kind="both", grid_resolution=5, max_samples=10)
    assert np.asarray(report["individual"]).shape == (1, 10, 5)
    assert report["sample_count"] == 10
    with pytest.raises(ValueError, match="max_evaluations"):
        nil.inspection.partial_dependence(model, X, columns=[0, 1], grid_resolution=100, max_evaluations=10)
    comparison = nil.inspection.compare({"ridge": nil.model("ridge_regression"), "linear": nil.model("linear_regression")}, X, y, cv=3)
    assert comparison["fit_count"] == 6
    with pytest.raises(ValueError, match="max_fits"):
        nil.validation.nested_validate(clone(model), X, y, {"alpha": [0.1, 1.]}, outer_cv=3,
            inner_cv=2, max_trials=2, max_fits=14, output_dir=tmp_path)
    nested = nil.validation.nested_validate(clone(model), X, y, {"alpha": [0.1, 1.]}, method="grid",
        outer_cv=3, inner_cv=2, max_trials=2, max_fits=15, output_dir=tmp_path)
    assert len(nested["folds"]) == 3 and nested["fit_budget"] == 15


def test_bayesian_resume_total_budget_and_data_isolation(tmp_path):
    pytest.importorskip("optuna")
    X, y = make_regression(n_samples=45, n_features=3, random_state=3)
    storage = "sqlite:///" + str(tmp_path / "study.db")
    options = dict(method="bayesian", cv=2, max_trials=2, max_fits=5,
                   storage=storage, study_name="ridge", output_dir=tmp_path / "runs", seed=3)
    space = {"alpha": {"type": "float", "low": .01, "high": 10., "log": True}}
    model = nil.model("ridge_regression")
    first = nil.tune(model, X, y, space, **options)
    resumed = nil.tune(model, X, y, space, **options)
    assert resumed.study_name == first.study_name
    assert len(resumed.trials) == 2
    assert resumed.best_params == first.best_params
    extended = nil.tune(model, X, y, space, **{**options, "max_trials": 3, "max_fits": 7})
    assert extended.study_name == first.study_name and len(extended.trials) == 3
    changed = nil.tune(model, X, y + 1, space, **options)
    assert changed.study_name != first.study_name
    with pytest.raises(ValueError, match="max_fits"):
        nil.tune(model, X, y, space, **{**options, "max_fits": 4})
    with pytest.raises(ValueError, match="positive bounds"):
        nil.tune(model, X, y, {"alpha": {"type": "float", "low": 0, "high": 1, "log": True}}, **options)
    with pytest.raises(nil.validation.SearchFailed) as failed:
        nil.tune(model, X, y, {"alpha": [-1.]}, **options)
    assert all(item["status"] == "failed" for item in failed.value.trials)


@pytest.mark.parametrize("task", ["classifier", "regressor"])
def test_xgboost_cpu_native_metrics_and_complete_pipeline_roundtrip(tmp_path, task):
    pytest.importorskip("xgboost")
    import json
    from importlib import metadata
    if task == "classifier":
        X, y = make_classification(n_samples=60, n_features=4, random_state=3)
    else:
        X, y = make_regression(n_samples=60, n_features=4, random_state=3)
    model = nil.pipeline(nil.model("xgboost_" + task, n_estimators=5, max_depth=2,
                                    n_jobs=1, device="cpu", random_state=3),
                          preprocess=nil.preprocessing.numeric())
    result = nil.fit(model, X[:40], y[:40], validation=(X[40:], y[40:]), output_dir=tmp_path / "runs")
    assert result.metrics["count"] == 20
    path = result.save(tmp_path / "xgb")
    np.testing.assert_array_equal(nil.load(path, trusted=True).predict(X), result.predict(X))
    manifest = json.loads((path / "manifest.json").read_text())
    installed = [name for name in ("xgboost", "xgboost-cpu") if name in manifest["dependencies"]]
    assert installed
    assert all(manifest["dependencies"][name] == metadata.version(name) for name in installed)
