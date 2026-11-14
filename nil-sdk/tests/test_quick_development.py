"""Offline acceptance tests for the small model-development additions."""

import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

import nil
from nil.training import EarlyStopping, grid_search
from nil.training.checkpoint import load

torch.set_num_threads(1)


def test_early_stop_saves_best_weights_and_completes(tmp_path):
    data = nil.data.tensors(torch.ones(8, 1), torch.zeros(8, 1))
    model = nil.nn.mlp(1, 1, hidden=())
    stopping = EarlyStopping(patience=2)
    values = iter([1.0, 2.0, 3.0, 4.0])
    snapshots = []

    def measured_fixture(trainer, run, metrics):
        metrics["val_loss"] = next(values)
        snapshots.append(copy.deepcopy(trainer.model.state_dict()))

    trainer = nil.Trainer(
        model,
        config=nil.TrainingConfig(epochs=10, device="cpu", output_dir=tmp_path),
        callbacks=[measured_fixture, stopping],
    )
    result = trainer.fit(data, validation=data)
    assert result.status == "completed"
    assert len(result.history) == 3
    assert result.run.metadata["reason"] == "early_stopping"
    assert result.checkpoint.is_file() and stopping.best_checkpoint.is_file()
    restored = nil.nn.mlp(1, 1, hidden=())
    load(stopping.best_checkpoint, restored)
    for key, value in restored.state_dict().items():
        assert torch.equal(value, snapshots[0][key])
    assert stopping.best_value == 1.0


def test_early_stopping_modes_reset_and_missing_metric(tmp_path):
    from nil.lifecycle import Run

    data = nil.data.tensors(torch.ones(2, 1), torch.ones(2, 1))
    trainer = nil.Trainer(
        nil.nn.mlp(1, 1, hidden=()),
        config=nil.TrainingConfig(epochs=1, device="cpu", output_dir=tmp_path),
    )
    stopping = EarlyStopping(patience=1, monitor="score", mode="max", min_delta=0.1)
    first = Run(tmp_path)
    stopping(trainer, first, {"score": 0.5, "epoch": 1})
    stopping(trainer, first, {"score": 0.55, "epoch": 2})
    assert trainer._finish_reason == "early_stopping"
    trainer.fit(data)  # Finish requests reset for a new fit.
    assert trainer._finish_reason is None
    second = Run(tmp_path)
    stopping(trainer, second, {"score": 0.1, "epoch": 1})
    assert stopping.best_value == 0.1 and stopping.bad_epochs == 0
    with pytest.raises(ValueError, match="absent"):
        stopping(trainer, second, {"epoch": 2})


def test_grid_search_trains_and_restores_selected_model(tmp_path):
    x = torch.linspace(-1, 1, 16).reshape(-1, 1)
    data = nil.data.tensors(x, 2 * x + 1)
    config = nil.TrainingConfig(epochs=3, batch_size=4, device="cpu", output_dir=tmp_path, seed=9)
    result = grid_search(
        lambda: nil.nn.mlp(1, 1, hidden=()),
        data,
        data,
        {"learning_rate": [0.001, 0.1]},
        config=config,
    )
    assert len(result.trials) == 2
    best = min(result.trials, key=lambda trial: trial["value"])
    assert result.best_config == best["config"]
    assert result.run.metadata["best_trial"] == best["trial"]
    model = result.load_best(nil.nn.mlp(1, 1, hidden=()))
    actual = nil.evaluate(model, data, device_name="cpu")["mse"]
    assert actual == pytest.approx(best["value"], rel=1e-5)
    assert config.learning_rate == 0.001  # Caller configuration is not mutated.
    persisted = json.loads((result.run.path / "search.json").read_text())
    assert persisted["best"]["checkpoint"] == str(result.best_checkpoint)


def test_grid_search_is_reproducible(tmp_path):
    data = nil.data.tensors(torch.ones(4, 1), torch.zeros(4, 1))
    config = nil.TrainingConfig(epochs=1, device="cpu", output_dir=tmp_path)

    def factory():
        return nil.nn.mlp(1, 1, hidden=())

    first = grid_search(factory, data, data, {"learning_rate": [0.01, 0.02]}, config=config)
    second = grid_search(factory, data, data, {"learning_rate": [0.01, 0.02]}, config=config)
    assert [t["value"] for t in first.trials] == [t["value"] for t in second.trials]


def test_search_rejects_excessive_or_invalid_grid_before_model_creation(tmp_path):
    calls = []

    def factory():
        calls.append(1)
        return nil.nn.Linear(1, 1)

    config = nil.TrainingConfig(output_dir=tmp_path)
    for grid, maximum in [
        ({"epochs": [1, 2, 3]}, 2),
        ({"learning_rate": [-1]}, 2),
        ({"output_dir": ["elsewhere"]}, 2),
        ({"epochs": []}, 2),
    ]:
        with pytest.raises(ValueError):
            grid_search(factory, [], [], grid, config=config, max_trials=maximum)
    assert calls == [] and not list(tmp_path.iterdir())


def test_search_records_failed_trials(tmp_path):
    def broken():
        raise RuntimeError("fixture factory failure")

    config = nil.TrainingConfig(epochs=1, output_dir=tmp_path)
    with pytest.raises(ValueError, match="All search trials failed"):
        grid_search(broken, [], [], {"learning_rate": [0.1, 0.2]}, config=config)
    report = json.loads(next(tmp_path.glob("*/search.json")).read_text())
    assert len(report["trials"]) == 2
    assert all(t["status"] == "failed" for t in report["trials"])
    assert json.loads(next(tmp_path.glob("*/run.json")).read_text())["status"] == "failed"


class UniformLM(torch.nn.Module):
    def __init__(self, vocabulary=5):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.0))
        self.vocabulary = vocabulary

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, self.vocabulary) + self.offset)


def test_perplexity_counts_shifted_unmasked_tokens_and_is_batch_invariant():
    dataset = nil.data.Records(
        [
            {
                "input_ids": torch.tensor([1, 2, 3, 0]),
                "attention_mask": torch.tensor([1, 1, 1, 0]),
                "labels": torch.tensor([1, 2, 3, -100]),
            },
            {
                "input_ids": torch.tensor([0, 0, 1, 2]),
                "attention_mask": torch.tensor([0, 0, 1, 1]),
                "labels": torch.tensor([-100, -100, 1, 2]),
            },
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": torch.ones(4, dtype=torch.long),
                "labels": torch.tensor([1, -100, 3, 4]),
            },
        ]
    )
    model = UniformLM()
    for batch_size in [1, 2, 3]:
        result = nil.generative.perplexity(model, dataset, batch_size=batch_size, device_name="cpu")
        assert result["tokens"] == 5
        assert result["nll"] == pytest.approx(math.log(5), rel=1e-6)
        assert result["perplexity"] == pytest.approx(5, rel=1e-6)
        assert model.training


def test_perplexity_rejects_empty_targets_and_restores_mode_on_error():
    model = UniformLM()
    dataset = nil.data.Records(
        [{"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([-100, -100])}]
    )
    with pytest.raises(ValueError, match="No unmasked"):
        nil.generative.perplexity(model, dataset, device_name="cpu")
    assert model.training
    with pytest.raises(ValueError, match="require"):
        nil.generative.perplexity(
            model, nil.data.Records([{"input_ids": torch.tensor([1, 2])}]), device_name="cpu"
        )
    assert model.training


def test_perplexity_with_real_local_transformer():
    pytest.importorskip("transformers")
    model = nil.generative.causal_lm(8, context_length=8, width=8, layers=1, heads=2)
    data = nil.data.Records(
        [{"input_ids": torch.tensor([1, 2, 3, 4]), "labels": torch.tensor([1, 2, 3, 4])}]
    )
    result = nil.generative.perplexity(model, data, device_name="cpu")
    assert result["tokens"] == 3 and result["perplexity"] > 1


def test_profile_measures_footprint_and_leaves_model_unchanged():
    model = nil.nn.mlp(4, 2, hidden=(16,))
    before = copy.deepcopy(model.state_dict())
    report = nil.optimization.profile(model, torch.zeros(2, 4), iterations=2, warmup=0)
    assert report["parameters"] == sum(p.numel() for p in model.parameters())
    assert report["parameter_bytes"] == report["parameters"] * 4
    assert report["serialized_state_bytes"] > report["parameter_bytes"]
    assert report["latency"]["mean_ms"] > 0 and model.training
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    quantized = nil.optimization.quantize(model)
    assert (
        nil.optimization.profile(quantized, torch.zeros(2, 4), iterations=1, warmup=0)[
            "serialized_state_bytes"
        ]
        > 0
    )
