"""Public catalog and import isolation, including native model acceptance when installed."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

import nil


def test_catalog_is_shallow_and_logistic_is_classification():
    groups = nil.catalog.tree()
    assert set(groups) == {"regression", "classification", "unsupervised", "neural_networks", "anomaly_detection", "feature_selection", "forecasting", "ranking", "recommendation", "survival"}
    assert "linear_regression" in groups["regression"]
    assert "logistic_regression" in groups["classification"]
    assert "logistic_regression" not in groups["regression"]
    assert all(isinstance(name, str) for names in groups.values() for name in names)
    assert nil.list_models("regression") == groups["regression"]
    assert len(nil.list_models()) == len(set(nil.list_models()))
    groups["regression"].clear()
    assert "linear_regression" in nil.list_models()


def test_unknown_names_fail_before_loading_backends():
    with pytest.raises(ValueError, match="nil.list_models"):
        nil.model("imaginary_model")
    with pytest.raises(ValueError, match="Unknown model group"):
        nil.list_models("imaginary_group")


def test_catalog_training_and_policy_imports_need_no_backends_or_agent_system(tmp_path):
    source = str(Path(__file__).resolve().parents[1])
    script = """
import nil, sys, pathlib
assert nil.training.checkpoint.recover
assert nil.list_models()
assert nil.catalog.tree()
assert nil.Trainer
assert nil.training.EarlyStopping
assert nil.robotics.policy
assert nil.robotics.demonstrations
assert nil.pipeline and nil.fit and nil.save and nil.load
assert nil.cross_validate and nil.tune
assert nil.preprocessing and nil.validation and nil.inspection
assert nil.streaming and nil.uncertainty and nil.imbalance and nil.selection
assert nil.execution
assert nil.boosting and nil.forecasting and nil.ranking and nil.recommendation and nil.survival and nil.datasets
assert nil.rl.algorithms()
import threading
assert not any(t.name.startswith('nil-') for t in threading.enumerate())
assert nil.sklearn
import importlib
mirror = importlib.import_module('nil.sklearn')
importlib.reload(mirror)
assert sum(bool(getattr(finder, '_nil_sklearn_mirror', False)) for finder in sys.meta_path) == 1
try:
    nil.fit(object(), [[1]], [1])
except TypeError as error:
    assert 'nil.train' in str(error)
else:
    raise AssertionError('wrong model family must produce a helpful error')
for prefix in ('torch', 'numpy', 'sklearn', 'xgboost', 'lightgbm', 'catboost', 'lifelines', 'optuna', 'imblearn', 'stable_baselines3', 'gymnasium', 'nil.cognition', 'nil.memory',
               'nil.training.curriculum', 'nil.robotics.domains'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
assert not list(pathlib.Path('.').iterdir())
try:
    nil.model('linear_regression')
except ImportError as error:
    assert 'nil-sdk[classical]' in str(error)
else:
    raise AssertionError('missing backend should explain the extra to install')
"""
    subprocess.run([sys.executable, "-S", "-c", script], cwd=tmp_path,
                   env={**os.environ, "PYTHONPATH": source}, check=True)


def test_models_cli_lists_names_without_dependencies(tmp_path):
    source = str(Path(__file__).resolve().parents[1])
    completed = subprocess.run([sys.executable, "-S", "-m", "nil", "models"],
                               cwd=tmp_path, env={**os.environ, "PYTHONPATH": source},
                               text=True, capture_output=True, check=True)
    assert "linear_regression" in completed.stdout
    assert "logistic_regression" in completed.stdout
    assert "neural_networks" in completed.stdout


def test_extension_is_lazy_discoverable_and_receives_parameters(monkeypatch):
    monkeypatch.setattr(nil.catalog, "_EXTENSIONS", {})
    calls = []

    class CustomModel:
        def __init__(self, width):
            calls.append(width)
            self.width = width

    nil.register_model("my_model", CustomModel, group="time_series")
    assert not calls
    assert nil.list_models("time_series") == ["my_model"]
    assert nil.catalog.tree()["time_series"] == ["my_model"]
    model = nil.model("my_model", width=16)
    assert isinstance(model, CustomModel)
    assert model.width == 16 and calls == [16]


def test_extensions_cannot_replace_models_or_register_invalid_factories(monkeypatch):
    monkeypatch.setattr(nil.catalog, "_EXTENSIONS", {})
    with pytest.raises(ValueError, match="already registered"):
        nil.register_model("linear_regression", dict, group="regression")
    with pytest.raises(TypeError, match="callable"):
        nil.register_model("bad_factory", None, group="regression")
    with pytest.raises(ValueError, match="identifier"):
        nil.register_model("bad.name", dict, group="regression")
    nil.register_model("custom", dict, group="regression")
    with pytest.raises(ValueError, match="already registered"):
        nil.register_model("custom", list, group="regression")


def test_native_linear_and_logistic_models_fit_and_predict():
    linear_model = pytest.importorskip("sklearn.linear_model")
    regression = nil.model("linear_regression", fit_intercept=False)
    assert isinstance(regression, linear_model.LinearRegression)
    regression.fit([[0.0], [1.0], [2.0]], [0.0, 2.0, 4.0])
    assert regression.predict([[3.0]])[0] == pytest.approx(6.0)
    classifier = nil.model("logistic_regression", random_state=7)
    classifier.fit([[-2.0], [-1.0], [1.0], [2.0]], [0, 0, 1, 1])
    assert classifier.predict([[-3.0], [3.0]]).tolist() == [0, 1]


def test_catalog_neural_model_trains_with_existing_api(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    nil.runtime.seed(7)
    model = nil.model("mlp", input_dim=1, output_dim=1, hidden=())
    assert isinstance(model, torch.nn.Module)
    x = torch.linspace(-1, 1, 32).reshape(-1, 1)
    data = nil.data.tensors(x, 2 * x + 1)
    before = nil.evaluate(model, data, device_name="cpu")["mse"]
    result = nil.train(model, data, epochs=20, learning_rate=0.1,
                       device="cpu", output_dir=str(tmp_path))
    assert result.status == "completed"
    assert nil.evaluate(result.model, data, device_name="cpu")["mse"] < before
