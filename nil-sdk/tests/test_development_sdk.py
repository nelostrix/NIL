"""Offline acceptance tests of actual model development (no pretrained weights)."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import nil

torch.set_num_threads(1)


def regression():
    nil.runtime.seed(11)
    x = torch.linspace(-1, 1, 48).reshape(-1, 1)
    return nil.nn.mlp(1, 1, hidden=()), nil.data.tensors(x, 2 * x + 1)


def test_core_import_does_not_import_backends_or_create_store(tmp_path):
    source = str(Path(__file__).resolve().parents[1])
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import nil,sys,pathlib; assert 'torch' not in sys.modules; "
            "assert 'numpy' not in sys.modules; assert not list(pathlib.Path('.').iterdir()); "
            "assert nil.data.Records([1])[0]==1",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": source},
        check=True,
    )


def test_records_validation_split_and_fingerprint(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps({"x": i}) for i in range(10)))
    data = nil.data.open(p).validate(lambda x: isinstance(x["x"], int))
    a, b = data.split(seed=8)
    assert len(a) + len(b) == 10
    assert {r["x"] for r in a}.isdisjoint(r["x"] for r in b)
    assert [r for r in a] == [r for r in data.split(seed=8)[0]]
    assert len(nil.data.fingerprint(p)) == 64
    with pytest.raises(ValueError, match="record 0"):
        data.validate(lambda x: False)


def test_real_training_evaluation_checkpoint_and_registry(tmp_path):
    model, data = regression()
    before = nil.evaluate(model, data, device_name="cpu")["mse"]
    result = nil.train(
        model,
        data,
        epochs=35,
        batch_size=12,
        learning_rate=0.05,
        output_dir=str(tmp_path),
        device="cpu",
        validation=data,
    )
    assert result.status == "completed"
    assert nil.evaluate(model, data, device_name="cpu")["mse"] < before * 0.01
    assert result.history[-1]["val_loss"] < result.history[0]["val_loss"]
    clone = nil.nn.mlp(1, 1, hidden=())
    from nil.training import checkpoint

    state = checkpoint.load(result.checkpoint, clone)
    assert state["epoch"] == 35
    x = torch.tensor([[0.2]])
    assert torch.allclose(nil.predict(model, x), nil.predict(clone, x))
    registry = nil.Registry(tmp_path / "registry")
    first = registry.register("regressor", result.checkpoint, {"run": result.run.id})
    second = registry.register("regressor", result.checkpoint)
    assert first["version"] == 1 and second["version"] == 2
    assert registry.resolve("regressor", 1)["sha256"] == first["sha256"]
    registered = nil.nn.mlp(1, 1, hidden=())
    registry.load("regressor", registered, version=1)
    assert torch.allclose(nil.predict(model, x), nil.predict(registered, x))
    Path(first["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        registry.resolve("regressor", 1)
    result.checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        checkpoint.load(result.checkpoint, clone)


def test_resume_matches_uninterrupted_with_shuffle_and_dropout(tmp_path):
    nil.runtime.seed(2)
    x = torch.randn(30, 3)
    data = nil.data.tensors(x, x.sum(1, keepdim=True))
    original = nil.nn.mlp(3, 1, hidden=(8,), dropout=0.2)
    full, interrupted = copy.deepcopy(original), copy.deepcopy(original)
    config = dict(batch_size=7, learning_rate=0.01, output_dir=str(tmp_path), device="cpu", seed=9)
    complete = nil.train(full, data, epochs=5, **config)
    partial = nil.train(interrupted, data, epochs=2, **config)
    resumed = nil.train(interrupted, data, epochs=5, resume=partial.checkpoint, **config)
    assert len(resumed.history) == 5
    for a, b in zip(full.parameters(), interrupted.parameters()):
        assert torch.equal(a, b)
    assert complete.history == resumed.history


def test_accumulation_flushes_final_partial_window(tmp_path):
    model = nil.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    data = nil.data.tensors(torch.ones(3, 1), torch.ones(3, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = nil.Trainer(
        model,
        optimizer=optimizer,
        config=nil.TrainingConfig(
            epochs=1,
            batch_size=1,
            accumulation_steps=2,
            shuffle=False,
            device="cpu",
            output_dir=str(tmp_path),
        ),
    )
    trainer.fit(data)
    assert model.weight.item() == pytest.approx(0.36)


def test_failure_and_cancel_are_persisted(tmp_path):
    model, data = regression()

    def invalid(model, batch):
        return model(batch[0]).mean() * float("nan")

    trainer = nil.Trainer(
        model,
        task=invalid,
        config=nil.TrainingConfig(epochs=1, output_dir=str(tmp_path), device="cpu"),
    )
    with pytest.raises(ValueError, match="Non-finite"):
        trainer.fit(data)
    statuses = [json.loads(p.read_text())["status"] for p in tmp_path.glob("*/run.json")]
    assert statuses == ["failed"]
    trainer = nil.Trainer(
        model,
        config=nil.TrainingConfig(epochs=3, output_dir=str(tmp_path), device="cpu"),
        callbacks=[lambda trainer, run, metrics: trainer.cancel()],
    )
    result = trainer.fit(data)
    assert result.status == "cancelled" and result.checkpoint.exists()
    assert len(result.history) == 1


def test_custom_step_and_native_optimizer(tmp_path):
    model, data = regression()

    def step(model, batch):
        return torch.nn.functional.l1_loss(model(batch[0]), batch[1])

    trainer = nil.Trainer(
        model,
        task=step,
        optimizer=torch.optim.SGD(model.parameters(), 0.1),
        config=nil.TrainingConfig(epochs=3, device="cpu", output_dir=str(tmp_path)),
    )
    result = trainer.fit(data)
    assert result.history[-1]["loss"] < result.history[0]["loss"]


def test_classification_and_vision(tmp_path):
    from nil.open_models.nelo_vision_mini import build

    model = build(2, channels=1, width=4)
    data = nil.data.tensors(torch.randn(8, 1, 8, 8), torch.arange(8) % 2)
    result = nil.train(
        model, data, task="classification", epochs=2, device="cpu", output_dir=str(tmp_path)
    )
    metrics = nil.evaluate(result.model, data, task="classification", device_name="cpu")
    assert metrics["count"] == 8 and 0 <= metrics["accuracy"] <= 1
    assert result.model.training  # Evaluation restores training state.


def test_real_dynamic_quantization(tmp_path):
    model, data = regression()
    result = nil.train(model, data, epochs=3, device="cpu", output_dir=str(tmp_path))
    quantized = nil.optimization.quantize(result.model)
    assert any(isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in quantized.modules())
    x = data.tensors[0]
    assert torch.max(torch.abs(nil.predict(model, x) - nil.predict(quantized, x))) < 0.05
    report = nil.optimization.benchmark(quantized, x, iterations=3, warmup=1)
    assert report["mean_ms"] > 0
    assert isinstance(model[0], torch.nn.Linear)


def test_real_onnx_roundtrip(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    model, data = regression()
    path = nil.optimization.export_onnx(
        model,
        data.tensors[0][:2],
        tmp_path / "model.onnx",
        input_names=["features"],
        output_names=["prediction"],
        dynamic_axes={"features": {0: "batch"}, "prediction": {0: "batch"}},
    )
    runtime = nil.inference.OnnxModel(path)
    x = data.tensors[0][:5]
    assert torch.allclose(
        torch.from_numpy(runtime.predict(x.numpy())[0]), nil.predict(model, x), atol=1e-5
    )
    with pytest.raises(ValueError, match="unavailable"):
        nil.inference.OnnxModel(path, ["MadeUpExecutionProvider"])


def test_llm_from_scratch_lora_save_reload_and_generate(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    texts = ["robot moves safely", "robot stops safely", "robot reads sensors"] * 4
    tokenizer = nil.generative.tokenizer_from_texts(texts, vocab_size=280)
    dataset = nil.generative.text_dataset(texts, tokenizer, max_length=16)
    model = nil.generative.causal_lm(len(tokenizer), context_length=32, width=16, layers=1, heads=2)
    adapted = nil.generative.lora(model, rank=2, alpha=4, target_modules=["c_attn"])
    counts = nil.nn.parameters(adapted)
    assert 0 < counts["trainable"] < counts["total"]
    initial = {k: v.detach().clone() for k, v in adapted.named_parameters() if v.requires_grad}
    result = nil.train(
        adapted,
        dataset,
        task="causal_lm",
        epochs=2,
        batch_size=4,
        device="cpu",
        output_dir=str(tmp_path),
    )
    assert any(
        not torch.equal(initial[k], v) for k, v in adapted.named_parameters() if v.requires_grad
    )
    # Merge only this tiny local adapter to obtain a standalone full model artifact.
    merged = result.model.merge_and_unload()
    destination = tmp_path / "local-llm"
    nil.generative.save(merged, tokenizer, destination)
    loaded, loaded_tokenizer = nil.generative.load_causal_lm(destination)
    text = nil.generative.generate(
        loaded,
        loaded_tokenizer,
        "robot",
        max_new_tokens=3,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    assert isinstance(text, str) and text.startswith("robot")
    assert all(record["labels"][record["attention_mask"] == 0].eq(-100).all() for record in dataset)


def test_classical_adapter(tmp_path):
    sklearn = pytest.importorskip("sklearn.linear_model")
    result = nil.classical.fit(
        sklearn.LinearRegression(),
        [[0], [1], [2]],
        [1, 3, 5],
        validation=([[3]], [7]),
        task="regression",
        output_dir=str(tmp_path),
    )
    assert result["metrics"]["mse"] < 1e-10


def test_imitation_policy_and_fail_closed_safety(tmp_path):
    from nil.robotics import PhysicalSafetyGatekeeper, demonstrations, policy

    model = policy(2, 1, hidden=(4,))
    data = demonstrations(torch.randn(12, 2), torch.zeros(12, 1))
    assert (
        nil.train(
            model, data, task="imitation_learning", epochs=2, device="cpu", output_dir=str(tmp_path)
        ).status
        == "completed"
    )
    safety = PhysicalSafetyGatekeeper(joint_limits_rad=[(-1, 1)])
    for current, target, dt in [
        ([0], [float("nan")], 0.01),
        ([2], [0], 0.01),
        ([0], [1], 0),
        ([0, 0], [1, 1], 0.01),
    ]:
        with pytest.raises(ValueError):
            safety.sanitize_joint_commands(current, target, dt)


def test_cli_doctor(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "nil", "doctor"], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout)["packages"]["torch"]["installed"]


def test_nested_tool_contract_rejects_bad_input_before_effects():
    from typing import List, Literal, Optional

    from nil.kernel.tool_manager import ToolManager

    calls = []

    @nil.tool
    def joints(
        q: List[float], mode: Literal["simulate", "plan"] = "plan", note: Optional[str] = None
    ):
        calls.append(q)
        return q

    manager = ToolManager([joints])
    schema = joints.to_schema()["function"]["parameters"]
    assert schema["properties"]["q"] == {"type": "array", "items": {"type": "number"}}
    for arguments in [
        {"q": "bad"},
        {"q": [float("nan")]},
        {"q": [True]},
        {"q": [1], "mode": "hardware"},
        {"q": [1], "extra": 3},
        {},
    ]:
        assert manager.execute("joints", arguments)["error"]
    assert calls == []
    assert manager.execute("joints", {"q": [1, 0.5]})["output"] == [1, 0.5]


def test_registry_collision_does_not_delete_existing_files(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"data")
    registry = nil.Registry(tmp_path / "registry")
    existing = registry.root / "artifacts" / "model" / "1"
    existing.mkdir(parents=True)
    (existing / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        registry.register("model", source)
    assert (existing / "keep").read_text() == "keep"


def test_local_serving_auth_and_prediction():
    import threading
    import urllib.error
    import urllib.request

    model = torch.nn.Linear(1, 1)
    token = "local-test-capability-token"
    server = nil.serving.create_server(model, token=token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/health")
        assert error.value.code == 401
        request = urllib.request.Request(
            base + "/predict",
            data=json.dumps({"inputs": [[1.0]]}).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            actual = json.load(response)["outputs"]
        assert actual == model(torch.ones(1, 1)).detach().tolist()
        request.add_header("Origin", "https://example.invalid")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_training_rejects_implicit_target_broadcasting(tmp_path):
    model = nil.nn.mlp(2, 1, hidden=())
    data = nil.data.tensors(torch.zeros(3, 2), torch.zeros(3))
    with pytest.raises(ValueError, match="broadcasting"):
        nil.train(model, data, epochs=1, device="cpu", output_dir=str(tmp_path))


def test_full_parameter_language_model_training(tmp_path):
    pytest.importorskip("transformers")
    texts = ["robot observes", "robot stops"] * 4
    tokenizer = nil.generative.tokenizer_from_texts(texts, vocab_size=270)
    dataset = nil.generative.text_dataset(texts, tokenizer, max_length=12)
    model = nil.generative.causal_lm(len(tokenizer), context_length=16, width=16, layers=1, heads=2)
    initial = {k: v.detach().clone() for k, v in model.named_parameters()}
    result = nil.train(
        model,
        dataset,
        task="causal_lm",
        epochs=3,
        batch_size=4,
        learning_rate=0.01,
        device="cpu",
        output_dir=tmp_path,
    )
    assert result.history[-1]["loss"] < result.history[0]["loss"]
    assert any(not torch.equal(initial[k], v) for k, v in model.named_parameters())


def test_unavailable_ik_never_returns_an_actionable_pose():
    from nil.robotics import STACCRoboticsBridge

    bridge = STACCRoboticsBridge(stacc_path="/nonexistent/nil-test-stacc")
    bridge._stacc_available = False
    with pytest.raises(RuntimeError, match="unavailable"):
        bridge.solve_spatial_ik((0.1, 0.2, 0.3))


def test_empty_data_fails_without_success_status(tmp_path):
    model = nil.nn.mlp(1, 1, hidden=())
    data = nil.data.tensors(torch.empty(0, 1), torch.empty(0, 1))
    with pytest.raises(ValueError):
        nil.train(model, data, epochs=1, device="cpu", output_dir=tmp_path)
    assert json.loads(next(tmp_path.glob("*/run.json")).read_text())["status"] == "failed"


def test_agent_defaults_are_fresh_and_preserve_evolution_semantics(tmp_path):
    first = nil.Agent(
        nil.Domain("one"), provider="stub", memory=nil.PersistentMemory(tmp_path / "one")
    )
    second = nil.Agent(
        nil.Domain("two"), provider="stub", memory=nil.PersistentMemory(tmp_path / "two")
    )
    disabled = nil.Agent(
        nil.Domain("three"),
        provider="stub",
        evolution=None,
        memory=nil.PersistentMemory(tmp_path / "three"),
    )
    try:
        assert first.evolution_cfg.strategy == "continuous"
        assert first.evolution_cfg is not second.evolution_cfg
        assert first.memory is not second.memory
        assert disabled.evolution_cfg.enabled is False
    finally:
        first.close()
        second.close()
        disabled.close()
