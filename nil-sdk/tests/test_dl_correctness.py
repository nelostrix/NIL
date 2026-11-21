"""Native DL gradient/metric parity and rejection before resume-state mutation."""

import copy
import random

import pytest

torch = pytest.importorskip('torch')
torch.set_num_threads(1)

import nil
from nil.training import checkpoint
from nil.training.tasks import Supervised


def trainer(model, tmp_path, *, task='regression', **options):
    return nil.Trainer(model, task=task, optimizer=torch.optim.SGD(model.parameters(), lr=.01),
        config=nil.TrainingConfig(epochs=1, batch_size=2, accumulation_steps=3, shuffle=False,
                                  device='cpu', output_dir=tmp_path, **options))


@pytest.mark.parametrize('reduction', ['mean', 'sum'])
def test_uneven_microbatches_match_one_native_batch(reduction, tmp_path):
    torch.manual_seed(1)
    model = torch.nn.Linear(1, 1)
    reference = copy.deepcopy(model)
    X = torch.tensor([[1.], [2.], [3.], [5.], [8.]])
    y = X * 2 + 1
    loss_fn = torch.nn.MSELoss(reduction=reduction)
    loss = loss_fn(reference(X), y)
    expected_loss = loss.item()
    loss.backward()
    torch.optim.SGD(reference.parameters(), lr=.01).step()
    trained = trainer(model, tmp_path, task=Supervised(torch.nn.MSELoss(reduction=reduction))).fit(nil.data.tensors(X, y), validation=nil.data.tensors(X, y))
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert trained.history[0]['loss'] == pytest.approx(expected_loss, rel=1e-6)
    assert trained.history[0]['val_loss'] == pytest.approx(loss_fn(reference(X), y).item(), rel=1e-6)


def test_ignored_weighted_class_labels_match_native_accumulation(tmp_path):
    torch.manual_seed(1)
    model = torch.nn.Linear(2, 2)
    reference = copy.deepcopy(model)
    X = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10
    y = torch.tensor([0, 1, -100, 0, 1])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 4.]))
    loss = criterion(reference(X), y)
    expected_loss = loss.item()
    loss.backward()
    torch.optim.SGD(reference.parameters(), lr=.01).step()
    result = trainer(model, tmp_path, task=Supervised(criterion)).fit(nil.data.tensors(X, y))
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert result.history[0]['loss'] == pytest.approx(expected_loss, rel=1e-6)


def test_resume_config_rejected_before_model_optimizer_or_rng_mutation(tmp_path):
    data = nil.data.tensors(torch.ones(4, 1), torch.ones(4, 1))
    source = nil.train(torch.nn.Linear(1, 1), data, epochs=1, batch_size=2,
                       device='cpu', output_dir=tmp_path / 'source')
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    before = copy.deepcopy(model.state_dict())
    random.seed(17)
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state().clone()
    engine = nil.Trainer(model, optimizer=optimizer, config=nil.TrainingConfig(
        epochs=2, batch_size=3, device='cpu', output_dir=tmp_path / 'rejected'))
    with pytest.raises(ValueError, match='matching batch_size'):
        engine.fit(data, resume=source.checkpoint)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert optimizer.state_dict()['state'] == {}
    assert random.getstate() == python_rng
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)


def test_shape_and_optimizer_class_rejected_before_any_weights_change(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = checkpoint.save(tmp_path / 'model.pt', model, optimizer)
    wrong_shape = torch.nn.Linear(2, 1)
    before = copy.deepcopy(wrong_shape.state_dict())
    with pytest.raises(ValueError, match='shape/dtype|structure'):
        checkpoint.load(path, wrong_shape)
    for key in before:
        torch.testing.assert_close(wrong_shape.state_dict()[key], before[key], rtol=0, atol=0)
    other = torch.nn.Linear(1, 1)
    before = copy.deepcopy(other.state_dict())
    with pytest.raises(ValueError, match='optimizer class'):
        checkpoint.load(path, other, torch.optim.SGD(other.parameters(), lr=.1))
    for key in before:
        torch.testing.assert_close(other.state_dict()[key], before[key], rtol=0, atol=0)


def test_boolean_training_counts_are_rejected():
    for options in ({'epochs': True}, {'batch_size': True}, {'accumulation_steps': True}, {'learning_rate': True}):
        with pytest.raises(ValueError):
            nil.TrainingConfig(**options)
    with pytest.raises(TypeError, match='torch.nn.Module'):
        nil.Trainer(object())


def test_bf16_training_and_epoch_resume_match_uninterrupted(tmp_path):
    torch.manual_seed(3)
    model = nil.nn.mlp(2, 1, hidden=(4,), dropout=.1)
    full, partial = copy.deepcopy(model), copy.deepcopy(model)
    X = torch.randn(15, 2)
    data = nil.data.tensors(X, X.sum(dim=1, keepdim=True))
    options = dict(batch_size=4, accumulation_steps=2, precision='bf16', seed=7,
                   device='cpu', output_dir=tmp_path, learning_rate=.03)
    complete = nil.train(full, data, epochs=4, **options)
    first = nil.train(partial, data, epochs=2, **options)
    resumed = nil.train(partial, data, epochs=4, resume=first.checkpoint, **options)
    assert complete.history == resumed.history
    for actual, expected in zip(full.parameters(), partial.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    payload = checkpoint.read(resumed.checkpoint)
    assert payload['state']['config']['precision'] == 'bf16'
    assert payload['state']['amp_scaler'] == {}
    with pytest.raises(ValueError, match='matching precision'):
        nil.train(copy.deepcopy(model), data, epochs=5, resume=first.checkpoint,
                   **{**options, 'precision': 'fp32'})
    with pytest.raises(ValueError, match='requires CUDA'):
        nil.Trainer(copy.deepcopy(model), config=nil.TrainingConfig(device='cpu', precision='fp16'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual CUDA fp16 acceptance requires GPU')
def test_cuda_fp16_scaler_checkpoint_and_resume(tmp_path):
    model = torch.nn.Linear(2, 1)
    data = nil.data.tensors(torch.ones(12, 2), torch.ones(12, 1))
    options = dict(batch_size=4, accumulation_steps=2, precision='fp16', seed=7,
                   device='cuda', output_dir=tmp_path, clip_grad_norm=1.)
    first = nil.train(model, data, epochs=1, **options)
    assert checkpoint.read(first.checkpoint)['state']['amp_scaler']['scale'] > 0
    resumed = nil.train(model, data, epochs=2, resume=first.checkpoint, **options)
    assert resumed.status == 'completed'
    assert all(torch.isfinite(value).all() for value in model.parameters())


@pytest.mark.parametrize('task', ['binary_classification', 'multilabel_classification'])
def test_bce_metrics_and_real_training(task, tmp_path):
    logits = torch.tensor([[-2., 1.], [3., -4.], [-1., 2.]])
    targets = torch.tensor([[0., 1.], [1., 0.], [1., 0.]])
    if task == 'binary_classification':
        logits, targets = logits[:, :1], targets[:, :1]
    metrics = nil.evaluate(torch.nn.Identity(), nil.data.tensors(logits, targets), task=task,
                           batch_size=2, device_name='cpu')
    assert metrics['loss'] == pytest.approx(torch.nn.functional.binary_cross_entropy_with_logits(logits, targets).item())
    assert metrics['accuracy'] == pytest.approx(((logits >= 0) == targets.bool()).float().mean().item())
    if task == 'multilabel_classification':
        assert metrics['subset_accuracy'] == pytest.approx(2 / 3)
    model = torch.nn.Linear(logits.shape[1], targets.shape[1])
    data = nil.data.tensors(logits, targets)
    before = nil.evaluate(model, data, task=task, device_name='cpu')['loss']
    nil.train(model, data, task=task, epochs=12, batch_size=2, device='cpu',
               learning_rate=.1, output_dir=tmp_path)
    assert nil.evaluate(model, data, task=task, device_name='cpu')['loss'] < before


def test_segmentation_classification_ignores_masked_labels_in_loss_and_accuracy(tmp_path):
    torch.manual_seed(3)
    model = nil.nn.Conv2d(1, 2, kernel_size=1)
    X = torch.randn(3, 1, 2, 2)
    y = (X[:, 0] > 0).long()
    y[0, 0, 0] = -100
    data = nil.data.tensors(X, y)
    result = nil.train(model, data, task='classification', epochs=1, batch_size=2,
                        device='cpu', output_dir=tmp_path)
    metrics = nil.evaluate(result.model, data, task='classification', batch_size=2, device_name='cpu')
    prediction = model(X)
    assert metrics['count'] == 11
    assert metrics['loss'] == pytest.approx(torch.nn.functional.cross_entropy(prediction, y).item())


def test_checkpoint_failed_commit_and_interrupted_publication_recover_last_valid(tmp_path, monkeypatch):
    import os
    path = tmp_path / 'latest.pt'
    model = torch.nn.Linear(1, 1)
    checkpoint.save(path, model, epoch=1)
    original = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.weight.add_(10)
    real_commit = checkpoint.atomic_json

    def failed_commit(*args, **kwargs):
        raise OSError('simulated disk commit failure')

    monkeypatch.setattr(checkpoint, 'atomic_json', failed_commit)
    with pytest.raises(OSError, match='commit failure'):
        checkpoint.save(path, model, epoch=2)
    monkeypatch.setattr(checkpoint, 'atomic_json', real_commit)
    restored = torch.nn.Linear(1, 1)
    assert checkpoint.load(path, restored)['epoch'] == 1
    for key, value in original.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    # A process death after replacing the alias but before manifest commit leaves
    # the committed generation intact. Normal load refuses; explicit recovery fixes it.
    interrupted = tmp_path / 'uncommitted.pt'
    interrupted.write_bytes(b'partial uncommitted payload')
    os.replace(interrupted, path)
    with pytest.raises(ValueError, match='integrity'):
        checkpoint.load(path, restored)
    checkpoint.recover(path)
    assert checkpoint.load(path, restored)['epoch'] == 1


def test_checkpoint_retains_one_previous_generation_and_explicit_rollback(tmp_path):
    path = tmp_path / 'latest.pt'
    model = torch.nn.Linear(1, 1)
    for epoch in range(1, 5):
        with torch.no_grad():
            model.weight.fill_(epoch)
        checkpoint.save(path, model, epoch=epoch)
    assert len(list(tmp_path.glob('.latest.pt.generation-*'))) == 2
    checkpoint.recover(path, previous=True)
    restored = torch.nn.Linear(1, 1)
    assert checkpoint.load(path, restored)['epoch'] == 3
    assert restored.weight.item() == 3


@pytest.mark.skipif(__import__('os').name != 'posix', reason='Actual owned-worker interruption')
def test_killed_checkpoint_writer_recovers_committed_generation(tmp_path):
    import sys
    import time
    script = """
import sys, time, torch
from pathlib import Path
from nil.training import checkpoint
path, marker = Path(sys.argv[1]), Path(sys.argv[2])
model = torch.nn.Linear(1, 1)
with torch.no_grad(): model.weight.fill_(1)
checkpoint.save(path, model, epoch=1)
with torch.no_grad(): model.weight.fill_(2)
def pause(*args, **kwargs):
    marker.write_text('ready')
    time.sleep(30)
checkpoint.atomic_json = pause
checkpoint.save(path, model, epoch=2)
"""
    path, marker = tmp_path / 'latest.pt', tmp_path / 'ready'
    job = nil.execution.submit('subprocess:run', [sys.executable, '-c', script, str(path), str(marker)],
                                limits=nil.execution.Limits(timeout=20), job_root=tmp_path / 'jobs', check=True)
    deadline = time.monotonic() + 15
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(.05)
    assert marker.exists(), job.status()
    job.cancel()
    with pytest.raises(nil.execution.ExecutionCancelled):
        job.result()
    checkpoint.recover(path)
    model = torch.nn.Linear(1, 1)
    assert checkpoint.load(path, model)['epoch'] == 1
    assert model.weight.item() == 1


def test_causal_lm_accumulation_uses_unmasked_shifted_token_count(tmp_path):
    class TinyCausal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 3)
            self.head = torch.nn.Linear(3, 4)

        def forward(self, input_ids, labels):
            logits = self.head(self.embedding(input_ids))
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 4), labels[:, 1:].reshape(-1))
            return {'loss': loss, 'logits': logits}

    torch.manual_seed(3)
    model = TinyCausal()
    reference = copy.deepcopy(model)
    tokens = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]])
    labels = tokens.clone()
    labels[1, 3:] = -100
    labels[2, 2:] = -100
    loss = reference(tokens, labels)['loss']
    expected_loss = loss.item()
    loss.backward()
    torch.optim.SGD(reference.parameters(), lr=.01).step()
    data = nil.data.Records([{'input_ids': row, 'labels': target} for row, target in zip(tokens, labels)])
    engine = nil.Trainer(model, task='causal_lm', optimizer=torch.optim.SGD(model.parameters(), lr=.01),
        config=nil.TrainingConfig(epochs=1, batch_size=1, accumulation_steps=3,
                                  shuffle=False, device='cpu', output_dir=tmp_path))
    result = engine.fit(data)
    assert result.history[0]['loss'] == pytest.approx(expected_loss, rel=1e-6)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('architecture', ['cnn', 'lstm', 'transformer'])
def test_native_architecture_backward_learning_and_checkpoint(tmp_path, architecture):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'examples/deep_workflows.py'
    spec = importlib.util.spec_from_file_location('deep_workflows', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.exercise(architecture, tmp_path)
    assert result['after_mse'] < result['before_mse']


def test_mlp_generator_widths_are_not_consumed_during_validation():
    model = nil.nn.mlp(2, 1, hidden=(value for value in [3, 2]))
    layers = [layer for layer in model if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [(2, 3), (3, 2), (2, 1)]
    with pytest.raises(ValueError, match='positive integers'):
        nil.nn.mlp(True, 1)


@pytest.mark.parametrize('scheduler_kind', ['step', 'plateau'])
def test_optimizer_scheduler_resume_is_exact(scheduler_kind, tmp_path):
    torch.manual_seed(3)
    initial = torch.nn.Linear(1, 1)
    X = torch.arange(12, dtype=torch.float32).reshape(-1, 1) / 10
    data = nil.data.tensors(X, 2 * X + 1)

    def engine(model, epochs):
        optimizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=.9)
        scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.7)
                     if scheduler_kind == 'step' else
                     torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=.5))
        return nil.Trainer(model, optimizer=optimizer, scheduler=scheduler,
            config=nil.TrainingConfig(epochs=epochs, batch_size=3, accumulation_steps=2,
                learning_rate=.05, seed=7, device='cpu', output_dir=tmp_path))

    full = engine(copy.deepcopy(initial), 4)
    complete = full.fit(data, validation=data)
    first = engine(copy.deepcopy(initial), 2).fit(data, validation=data)
    resumed_engine = engine(copy.deepcopy(initial), 4)
    resumed = resumed_engine.fit(data, validation=data, resume=first.checkpoint)
    assert complete.history == resumed.history
    assert full.scheduler.state_dict() == resumed_engine.scheduler.state_dict()
    assert full.optimizer.state_dict()['param_groups'] == resumed_engine.optimizer.state_dict()['param_groups']
    for actual, expected in zip(full.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for index, state in full.optimizer.state_dict()['state'].items():
        torch.testing.assert_close(state['momentum_buffer'], resumed_engine.optimizer.state_dict()['state'][index]['momentum_buffer'], rtol=0, atol=0)


def test_environment_guard_rejects_before_loading_weights(tmp_path):
    import json
    from nil.lifecycle import sha256
    path = checkpoint.save(tmp_path / 'model.pt', torch.nn.Linear(1, 1))
    payload = checkpoint.read(path)
    payload['environment']['torch'] = 'unsupported-version'
    torch.save(payload, path)
    metadata = json.loads((tmp_path / 'model.pt.json').read_text())
    metadata['sha256'] = sha256(path)
    (tmp_path / 'model.pt.json').write_text(json.dumps(metadata))
    model = torch.nn.Linear(1, 1)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match='environment mismatch'):
        checkpoint.load(path, model)
    for key in before:
        torch.testing.assert_close(model.state_dict()[key], before[key], rtol=0, atol=0)
    assert checkpoint.read(path, allow_version_mismatch=True)['environment']['torch'] == 'unsupported-version'


def test_optimizer_ownership_scheduler_binding_and_subset_gradients(tmp_path):
    first, other = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    foreign = torch.optim.SGD(other.parameters(), lr=.1)
    with pytest.raises(ValueError, match='members of this model'):
        nil.Trainer(first, optimizer=foreign, config=nil.TrainingConfig(device='cpu'))
    owned = torch.optim.SGD(first.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(foreign, step_size=1)
    with pytest.raises(ValueError, match='bound'):
        nil.Trainer(first, optimizer=owned, scheduler=scheduler, config=nil.TrainingConfig(device='cpu'))
    subset = torch.optim.SGD([first.bias], lr=.1)
    before = first.weight.detach().clone()
    engine = nil.Trainer(first, optimizer=subset, config=nil.TrainingConfig(
        epochs=2, batch_size=2, device='cpu', clip_grad_norm=1., output_dir=tmp_path))
    engine.fit(nil.data.tensors(torch.ones(4, 1), torch.ones(4, 1)))
    torch.testing.assert_close(first.weight, before, rtol=0, atol=0)
    assert first.weight.grad is None


def test_optimizer_parameter_reordering_rejected_before_restore(tmp_path):
    model = torch.nn.Linear(1, 1)
    path = checkpoint.save(tmp_path / 'ordered.pt', model, torch.optim.SGD([model.weight, model.bias], lr=.1))
    restored = torch.nn.Linear(1, 1)
    before = copy.deepcopy(restored.state_dict())
    reversed_optimizer = torch.optim.SGD([restored.bias, restored.weight], lr=.1)
    with pytest.raises(ValueError, match='names/order'):
        checkpoint.load(path, restored, reversed_optimizer)
    for key in before:
        torch.testing.assert_close(restored.state_dict()[key], before[key], rtol=0, atol=0)


def test_custom_multitask_step_matches_native_two_head_gradient(tmp_path):
    torch.manual_seed(3)
    model = torch.nn.ModuleDict({'regression': torch.nn.Linear(1, 1), 'binary': torch.nn.Linear(1, 1)})
    reference = copy.deepcopy(model)
    X = torch.arange(-2, 3, dtype=torch.float32).reshape(-1, 1)
    regression, binary = 2 * X, (X > 0).float()

    def objective(module, batch):
        inputs, continuous, labels = batch
        return (torch.nn.functional.mse_loss(module['regression'](inputs), continuous) +
                torch.nn.functional.binary_cross_entropy_with_logits(module['binary'](inputs), labels))

    objective.normalization_count = lambda batch: len(batch[0])
    loss = objective(reference, (X, regression, binary))
    expected_loss = loss.item()
    loss.backward()
    torch.optim.SGD(reference.parameters(), lr=.01).step()
    result = trainer(model, tmp_path, task=objective).fit(nil.data.tensors(X, regression, binary))
    assert result.history[0]['loss'] == pytest.approx(expected_loss, rel=1e-6)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
