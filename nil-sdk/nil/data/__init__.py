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


def snapshot(records, path, *, max_records=1000000, max_bytes=100000000):
    """Publish a new immutable JSONL dataset snapshot; never replace a prior one."""
    import itertools
    import os
    import shutil
    import tempfile
    from nil._classical_common import positive_int

    positive_int(max_records, "max_records")
    positive_int(max_bytes, "max_bytes")
    destination = Path(path).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".nil-data-", dir=destination.parent))
    try:
        count, size = 0, 0
        with builtins.open(temporary / "records.jsonl", "wb") as stream:
            for index, record in enumerate(itertools.islice(records, max_records + 1)):
                if index == max_records:
                    raise ValueError("Snapshot exceeds max_records")
                encoded = (json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                size += len(encoded)
                if size > max_bytes:
                    raise ValueError("Snapshot exceeds max_bytes")
                stream.write(encoded)
                count += 1
        if not count:
            raise ValueError("A snapshot needs at least one record")
        manifest = {"schema_version": 1, "format": "nil-jsonl", "record_count": count,
                    "bytes": size, "sha256": fingerprint(temporary / "records.jsonl")}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        destination.mkdir()
        try:
            os.replace(temporary / "records.jsonl", destination / "records.jsonl")
            os.replace(temporary / "manifest.json", destination / "manifest.json")
        except BaseException:
            (destination / "records.jsonl").unlink(missing_ok=True)
            destination.rmdir()
            raise
        return destination
    finally:
        shutil.rmtree(temporary)


def load_snapshot(path, *, max_records=1000000, max_bytes=100000000):
    """Validate an immutable JSON snapshot and return a fresh Records instance."""
    from nil._classical_common import positive_int
    positive_int(max_records, "max_records")
    positive_int(max_bytes, "max_bytes")
    root = Path(path)
    artifact, metadata = root / "records.jsonl", root / "manifest.json"
    if root.is_symlink() or artifact.is_symlink() or metadata.is_symlink():
        raise ValueError("Dataset snapshots must not use symlinks")
    if metadata.stat().st_size > 65536:
        raise ValueError("Snapshot manifest is too large")
    manifest = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1 or manifest.get("format") != "nil-jsonl" or type(manifest.get("record_count")) is not int or type(manifest.get("bytes")) is not int:
        raise ValueError("Unsupported dataset snapshot manifest")
    if not 0 < manifest["record_count"] <= max_records or not 0 < artifact.stat().st_size <= max_bytes:
        raise ValueError("Snapshot exceeds loading limits or is empty")
    if artifact.stat().st_size != manifest["bytes"] or fingerprint(artifact) != manifest.get("sha256"):
        raise ValueError("Snapshot integrity check failed")
    with builtins.open(artifact, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    if len(records) != manifest["record_count"]:
        raise ValueError("Snapshot record count differs from manifest")
    return Records(records)


def jsonl_stream(path, transform=None):
    """Single-worker JSONL iterator with a validated byte cursor, no materialization.

    Transform identity/behavior must stay the same across resume. Use saved
    immutable data; source checksums are verified at checkpoint/recovery.
    """
    from nil.data._stream import JsonlStream
    return JsonlStream(path, transform)
