"""Composable local datasets; native PyTorch datasets/loaders are also accepted."""

import builtins
import csv
import hashlib
import json
import random
from pathlib import Path

from nil.runtime import require


class Records:
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def map(self, transform):
        return Records(transform(item) for item in self.records)

    def filter(self, predicate):
        return Records(item for item in self.records if predicate(item))

    def validate(self, validator):
        for index, item in enumerate(self.records):
            if not validator(item):
                raise ValueError(f"Dataset validation failed at record {index}")
        return self

    def split(self, fraction=0.8, seed=0):
        if not 0 < fraction < 1:
            raise ValueError("fraction must be between zero and one")
        if len(self) < 2:
            raise ValueError("Need at least two records to split")
        indices = list(range(len(self)))
        random.Random(seed).shuffle(indices)
        cut = max(1, min(len(self) - 1, int(len(self) * fraction)))
        return (
            Records(self.records[i] for i in indices[:cut]),
            Records(self.records[i] for i in indices[cut:]),
        )

    def loader(self, batch_size=32, shuffle=False, **kwargs):
        return loader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)


def open(path):
    """Read local JSON arrays, JSONL or CSV. Never implicitly download data."""
    path = Path(path)
    with builtins.open(path, encoding="utf-8", newline="") as stream:
        if path.suffix == ".jsonl":
            return Records(json.loads(line) for line in stream if line.strip())
        if path.suffix == ".csv":
            return Records(csv.DictReader(stream))
        if path.suffix == ".json":
            value = json.load(stream)
            if not isinstance(value, list):
                raise ValueError("JSON dataset must contain an array")
            return Records(value)
    raise ValueError("Supported dataset extensions: .json, .jsonl, .csv")


def tensors(*values):
    torch = require("torch", "torch")
    if not values:
        raise ValueError("Provide at least one tensor")
    return torch.utils.data.TensorDataset(*(torch.as_tensor(v) for v in values))


def loader(dataset, batch_size=32, shuffle=False, **kwargs):
    return require("torch", "torch").utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
    )


def fingerprint(path):
    """Streaming SHA-256 content identifier for a single local file."""
    digest = hashlib.sha256()
    with builtins.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
