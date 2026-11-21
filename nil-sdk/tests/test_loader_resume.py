"""Real loader and bounded streaming continuation, no simulated worker batches."""
import copy
import json
import random
import pytest

torch = pytest.importorskip('torch')
torch.set_num_threads(1)
import nil


class Augmented(torch.utils.data.Dataset):
    def __len__(self): return 12
    def __getitem__(self, index):
        x = torch.tensor([index / 20 + random.random() / 100])
        return x, x * 2


def initial():
    torch.manual_seed(7)
    return torch.nn.Linear(1, 1)


def options(tmp_path, **extra):
    return dict(batch_size=2, device='cpu', output_dir=tmp_path, seed=11, **extra)


def compare(left, right):
    assert left.history == right.history
    for actual, expected in zip(left.model.parameters(), right.model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_two_worker_epoch_resume_with_random_augmentation(tmp_path):
    model = initial()
    opts = options(tmp_path, num_workers=2, prefetch_factor=1)
    full = nil.train(copy.deepcopy(model), Augmented(), epochs=3, **opts)
    first = nil.train(copy.deepcopy(model), Augmented(), epochs=1, **opts)
    resumed = nil.train(copy.deepcopy(model), Augmented(), epochs=3, resume=first.checkpoint, **opts)
    compare(full, resumed)


def test_external_sampler_generator_and_loader_generator_resume(tmp_path):
    model = initial()
    data = nil.data.tensors(torch.arange(12).float().reshape(-1, 1) / 20, torch.ones(12, 1))
    def loader():
        sampler = torch.utils.data.RandomSampler(data, generator=torch.Generator().manual_seed(31))
        return torch.utils.data.DataLoader(data, batch_size=2, sampler=sampler, generator=torch.Generator().manual_seed(41))
    full = nil.train(copy.deepcopy(model), loader(), epochs=3, **options(tmp_path))
    first = nil.train(copy.deepcopy(model), loader(), epochs=1, **options(tmp_path))
    resumed = nil.train(copy.deepcopy(model), loader(), epochs=3, resume=first.checkpoint, **options(tmp_path))
    compare(full, resumed)


def transform(record):
    return torch.tensor([record['x'] / 20]), torch.tensor([record['x'] / 10])


def test_bounded_jsonl_continuation_consumes_exactly_trained_records(tmp_path):
    source = tmp_path / 'data.jsonl'
    lines = [json.dumps({'x': index}) + '\n' for index in range(24)]
    source.write_text(''.join(lines))
    model = initial()
    opts = options(tmp_path, shuffle=False, max_batches_per_epoch=2)
    full_stream = nil.data.jsonl_stream(source, transform)
    full = nil.train(copy.deepcopy(model), full_stream, epochs=3, **opts)
    first_stream = nil.data.jsonl_stream(source, transform)
    first = nil.train(copy.deepcopy(model), first_stream, epochs=1, **opts)
    assert first_stream.offset == len(''.join(lines[:4]).encode())
    resumed_stream = nil.data.jsonl_stream(source, transform)
    resumed = nil.train(copy.deepcopy(model), resumed_stream, epochs=3, resume=first.checkpoint, **opts)
    compare(full, resumed)
    assert resumed_stream.offset == full_stream.offset == len(''.join(lines[:12]).encode())
    assert next(iter(resumed_stream))[0].item() == pytest.approx(12 / 20)
    with pytest.raises(ValueError, match='record boundary'):
        resumed_stream.load_state_dict({**resumed_stream.state_dict(), 'offset': 2})
    source.write_text(''.join(lines).replace('0}', '9}', 1))
    with pytest.raises(ValueError, match='changed'):
        resumed_stream.state_dict()


def test_changed_loader_contract_refused_before_weights_or_rng_mutation(tmp_path):
    model = initial()
    data = nil.data.tensors(torch.ones(4, 1), torch.ones(4, 1))
    first = nil.train(copy.deepcopy(model), torch.utils.data.DataLoader(data, batch_size=2), epochs=1, **options(tmp_path))
    target = copy.deepcopy(model)
    before = copy.deepcopy(target.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='contract'):
        nil.train(target, torch.utils.data.DataLoader(data, batch_size=1), epochs=2, resume=first.checkpoint, **options(tmp_path))
    for key, tensor in target.state_dict().items(): torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_unsafe_stream_prefetch_and_unbounded_iterations_rejected(tmp_path):
    source = tmp_path / 'data.jsonl'; source.write_text('{"x": 1}\n')
    for extra, message in [({}, 'max_batches'), ({'max_batches_per_epoch': 1, 'num_workers': 1}, 'num_workers=0'), ({'max_batches_per_epoch': 1, 'shuffle': True}, 'shuffle=False')]:
        opts = options(tmp_path, **extra)
        opts.setdefault('shuffle', False)
        with pytest.raises(ValueError, match=message):
            nil.train(initial(), nil.data.jsonl_stream(source, transform), epochs=1, **opts)
    for config in [dict(num_workers=9), dict(prefetch_factor=9), dict(max_batches_per_epoch=0), dict(persistent_workers=True)]:
        with pytest.raises(ValueError): nil.TrainingConfig(**config)


def test_changed_stream_source_is_rejected_before_model_or_rng_restore(tmp_path):
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps({'x': i}) + '\n' for i in range(12)))
    opts = options(tmp_path, shuffle=False, max_batches_per_epoch=1)
    saved = nil.train(initial(), nil.data.jsonl_stream(source, transform), epochs=1, **opts)
    source.write_text(source.read_text().replace('0}', '9}', 1))
    model = initial()
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='same immutable source'):
        nil.train(model, nil.data.jsonl_stream(source, transform), epochs=2, resume=saved.checkpoint, **opts)
    for name, value in model.state_dict().items(): torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_native_distributed_sampler_epoch_is_set_and_seed_contract_checked(tmp_path):
    data = nil.data.tensors(torch.arange(12).float().reshape(-1, 1) / 20, torch.ones(12, 1))
    def loader(seed=31):
        sampler = torch.utils.data.DistributedSampler(data, num_replicas=1, rank=0, seed=seed)
        return torch.utils.data.DataLoader(data, batch_size=2, sampler=sampler, generator=torch.Generator().manual_seed(41))
    full_loader = loader()
    full = nil.train(initial(), full_loader, epochs=3, **options(tmp_path))
    assert full_loader.sampler.epoch == 2
    first = nil.train(initial(), loader(), epochs=1, **options(tmp_path))
    compare(full, nil.train(initial(), loader(), epochs=3, resume=first.checkpoint, **options(tmp_path)))
    with pytest.raises(ValueError, match='contract'):
        nil.train(initial(), loader(32), epochs=3, resume=first.checkpoint, **options(tmp_path))
