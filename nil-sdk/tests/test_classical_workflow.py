"""Acceptance of native pipelines, held-out metrics and trusted artifact roundtrips."""
import json

import pytest

pytest.importorskip("sklearn")
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

import nil


def test_preprocessing_fits_only_training_data_and_preserves_unknown_categories(tmp_path):
    X = np.array([[1.0, "a"], [3.0, "b"], [np.nan, "a"], [5.0, "b"]], dtype=object)
    y = [0, 1, 0, 1]
    prep = nil.preprocessing.tabular(numeric_columns=[0], categorical_columns=[1])
    model = nil.pipeline(nil.model("logistic_regression", random_state=7), preprocess=prep)
    result = nil.fit(model, X, y, output_dir=tmp_path / "runs")
    before = prep.named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
    predictions = result.predict(np.array([[10000.0, "unseen"], [np.nan, "a"]], dtype=object))
    np.testing.assert_array_equal(before, [3.0])
    np.testing.assert_array_equal(prep.named_transformers_["numeric"].named_steps["impute"].statistics_, before)
    assert len(predictions) == 2
    assert result["model"] is result.model
    assert result.status == "completed"


def test_classification_metrics_and_weights_match_native_metrics(tmp_path):
    X = [[-3], [-2], [-1], [1], [2], [3]]
    y = np.array(["no", "no", "yes", "no", "yes", "yes"])
    weights = np.array([1., 1., 8., 1., 1., 1.])
    result = nil.fit(nil.model("logistic_regression"), X, y,
                     sample_weight=weights, output_dir=tmp_path)
    scores = nil.evaluate(result, X, y, sample_weight=weights)
    predicted = nil.predict(result, X)
    assert scores["accuracy"] == pytest.approx(accuracy_score(y, predicted, sample_weight=weights))
    assert scores["roc_auc"] == pytest.approx(roc_auc_score(y == "yes", result.predict_proba(X)[:, 1], sample_weight=weights))
    assert np.asarray(scores["confusion_matrix"]).shape == (2, 2)
    assert 0 <= scores["f1_macro"] <= 1
    assert "roc_auc" not in nil.evaluate(result, X[:2], y[:2])


def test_regression_result_metrics_and_single_sample_r2(tmp_path):
    model = nil.pipeline(nil.model("linear_regression"), preprocess=nil.preprocessing.numeric())
    result = nil.fit(model, [[0.], [1.], [2.]], [1., 3., 5.],
                     validation=([[3.], [4.]], [7., 9.]), output_dir=tmp_path)
    assert result.metrics["mse"] < 1e-20
    assert nil.evaluate(result, [[3.]], [7.])["r2"] is None
    assert mean_squared_error([7., 9.], result.predict([[3.], [4.]])) == pytest.approx(result.metrics["mse"])


def test_complete_pipeline_bundle_reproduces_predictions_and_rejects_corruption(tmp_path):
    model = nil.pipeline(nil.model("linear_regression"), preprocess=nil.preprocessing.numeric())
    result = nil.fit(model, [[0.], [1.], [2.]], [1., 3., 5.], output_dir=tmp_path / "runs")
    destination = result.save(tmp_path / "model")
    with pytest.raises(ValueError, match="trusted=True"):
        nil.load(destination)
    loaded = nil.load(destination, trusted=True)
    np.testing.assert_allclose(loaded.predict([[3.]]), result.predict([[3.]]))
    with pytest.raises(FileExistsError):
        result.save(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["feature_count"] == 1
    with (destination / "model.joblib").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        nil.load(destination, trusted=True)


def test_sparse_input_stays_sparse(tmp_path):
    from scipy.sparse import csr_matrix, issparse
    X = csr_matrix([[0., 1.], [1., 0.], [0., 2.], [2., 0.]])
    model = nil.pipeline(nil.model("logistic_regression"), preprocess=nil.preprocessing.numeric())
    result = nil.fit(model, X, [0, 1, 0, 1], output_dir=tmp_path / "sparse")
    assert issparse(model.named_steps["preprocess"].transform(X))
    assert len(result.predict(X)) == 4


def test_named_schema_roundtrips(tmp_path):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"value": [1., 2., 3., 4.], "kind": ["a", "b", "a", "b"]})
    pipeline = nil.pipeline(nil.model("logistic_regression"), preprocess=nil.preprocessing.tabular(
        numeric_columns=["value"], categorical_columns=["kind"]))
    trained = nil.fit(pipeline, frame, [0, 1, 0, 1], output_dir=tmp_path / "named")
    path = trained.save(tmp_path / "named-model")
    assert json.loads((path / "manifest.json").read_text())["feature_names"] == ["value", "kind"]
    np.testing.assert_array_equal(nil.load(path, trusted=True).predict(frame), trained.predict(frame))


def test_invalid_weights_and_overlapping_columns_are_rejected(tmp_path):
    for weight in ([1, -1], [0, 0], [float("nan"), 1], [1]):
        with pytest.raises(ValueError, match="sample_weight"):
            nil.fit(nil.model("linear_regression"), [[0], [1]], [0, 1], sample_weight=weight, output_dir=tmp_path)
    with pytest.raises(ValueError, match="only once"):
        nil.preprocessing.tabular(numeric_columns=[0], categorical_columns=[0])


def test_feature_transformers_fit_save_load_and_transform_without_fake_predictive_metrics(tmp_path):
    X = np.array([[0., 1., 0.], [1., 0., 1.], [2., 1., 2.], [3., 2., 4.]])
    result = nil.fit(nil.model("pca", n_components=2), X, output_dir=tmp_path / "runs")
    assert result.task == "transformation"
    transformed = nil.transform(result, X)
    assert transformed.shape == (4, 2)
    path = result.save(tmp_path / "pca")
    np.testing.assert_allclose(nil.transform(nil.load(path, trusted=True), X), transformed)
    with pytest.raises(TypeError, match="nil.transform"):
        nil.predict(result, X)
    with pytest.raises(ValueError, match="predictive metrics"):
        nil.evaluate(result, X)
