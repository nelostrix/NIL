# data

Load and validate records, split reproducibly, and prepare native tensor datasets.

## Usage

```python
import nil

records = nil.data.Records([{"x": i} for i in range(10)])
train, valid = records.split(fraction=0.8, seed=7)
print(len(train), len(valid))
```

## Parameters

`Records(items)`: iterable. `split(fraction, seed)`: training fraction and seed. `open(path)`: JSON, JSONL or CSV.
`snapshot(records, path, max_records=1000000, max_bytes=100000000)` saves a new
immutable JSONL dataset with a versioned checksum manifest. `load_snapshot(path,
max_records=..., max_bytes=...)` checks bounds/integrity and returns fresh Records.

## Output

Records collections; tensor helpers return native PyTorch datasets/loaders.

## Requirements

Core SDK for records. The torch extra is needed for tensors/loaders.

## Limitations

Splits are basic random splits, not stratified or time-aware. Keep validation/test data out of training transformations.
Snapshot records must be JSON-compatible, with missing values normalized to
None and no NaN/infinity. Existing snapshots are never overwritten. Immutability
means NIL does not modify the saved dataset; filesystem owners can still tamper
with files, which checksums detect unless the manifest is also changed. Hashes
do not authenticate a source. Loading returns independent mutable Python
records. Snapshots preserve JSON types, not DataFrame dtype/schema metadata.
Limits bound serialized size/count, not process peak RAM.


## Bounded JSONL training stream (optional Torch)

```python
import torch
import nil

def pair(row):
    return torch.tensor([row['x']], dtype=torch.float32), torch.tensor([row['y']], dtype=torch.float32)

stream = nil.data.jsonl_stream('train.jsonl', pair)
model = torch.nn.Linear(1, 1)
result = nil.train(model, stream, epochs=2, batch_size=16, shuffle=False,
                   num_workers=0, max_batches_per_epoch=100, device='cpu')
# A new stream over identical immutable bytes can resume the saved cursor.
continued = nil.train(model, nil.data.jsonl_stream('train.jsonl', pair),
    epochs=3, batch_size=16, shuffle=False, num_workers=0,
    max_batches_per_epoch=100, device='cpu', resume=result.checkpoint)
```

`jsonl_stream` reads one record at a time with an 8 MiB record limit and a byte
cursor. Blank lines are skipped. EOF does not rewind; an exhausted training
chunk fails clearly. Resume validates the SHA-256, length, cursor range and record
boundary. Only `num_workers=0` is supported: worker prefetch would otherwise
consume records without a training acknowledgement. Keep transform behavior and
its external state/RNG identical; arbitrary transforms cannot be authenticated
or reconstructed. Hash verification scans the file but does not materialize it.
Use completed epoch/chunk checkpoints; a failed batch does not commit an offset.
