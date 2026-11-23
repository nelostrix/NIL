"""Installed NIL example: bounded JSONL chunks and exact epoch-boundary resume."""
import copy
import json
from pathlib import Path
import tempfile

import torch
import nil


def pair(row):
    return torch.tensor([row['x'] / 20]), torch.tensor([row['x'] / 10])


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    initial = torch.nn.Linear(1, 1)
    with tempfile.TemporaryDirectory(prefix='nil-loader-example-') as root:
        root = Path(root)
        source = root / 'train.jsonl'
        source.write_text(''.join(json.dumps({'x': i}) + '\n' for i in range(24)))
        options = dict(batch_size=2, shuffle=False, max_batches_per_epoch=2,
                       num_workers=0, device='cpu', output_dir=root / 'runs')
        full = nil.train(copy.deepcopy(initial), nil.data.jsonl_stream(source, pair), epochs=3, **options)
        first = nil.train(copy.deepcopy(initial), nil.data.jsonl_stream(source, pair), epochs=1, **options)
        stream = nil.data.jsonl_stream(source, pair)
        resumed = nil.train(copy.deepcopy(initial), stream, epochs=3, resume=first.checkpoint, **options)
        assert full.history == resumed.history
        for actual, expected in zip(full.model.parameters(), resumed.model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print('PASS: identical history and weights; next record:', next(iter(stream)))


if __name__ == '__main__': main()
