# distributed

Explicit single-host supervised DDP training. No worker starts at import.

## Usage

```python
# Run as a Python file, protected for multiprocessing spawn.
import nil
import torch
if __name__ == '__main__':
    model = torch.nn.Linear(2, 1)
    result = nil.distributed.train(model, torch.randn(11, 2), torch.randn(11, 1),
        epochs=3, batch_size=4, world_size=2, output_dir='./ddp-run')
    print(result.history, result.checkpoint)
```

## Parameters

`train(model, features, targets, epochs, batch_size, world_size, lr, momentum,
seed, task, output_dir, resume, callback, timeout, device)` uses global batches
split without duplication. Tasks: regression, classification, binary_classification.
`resume` is a trusted NIL epoch checkpoint; `epochs` is the final epoch budget.
Callbacks are spawn-pickleable top-level rank-zero observers, not model mutators.

## Output

`DistributedResult(model, history, checkpoint, world_size)`. Successful completion
copies final weights to the supplied model; failures leave it unchanged.

## Requirements

Optional torch/NumPy backend, OS process support. CPU Gloo tested with two ranks.

## Limitations

FP32 SGD only, finite in-memory tensors, maximum eight ranks. No dropout,
BatchNorm or mutable forward buffers. Last committed epoch is the recovery unit;
failing ranks abort the group and require an explicit resume. CPU uneven-batch
serial parity, rank-zero callbacks and fault recovery are tested. CUDA/NCCL
implementation needs actual GPU acceptance. Default strategy='ddp'; optional FSDP is documented below. Multi-host is unsupported.

## Fixed-world FSDP sharding

`train(..., strategy='fsdp', world_size=2)` explicitly wraps the model with native
FSDP1 FULL_SHARD and an optimizer constructed after wrapping. Actual CPU Gloo
training, local parameter sharding, global-batch parity, epoch resume and
publication-fault recovery are tested. `max_parameters` (default 100 million)
and `max_dataset_bytes` (default 256 MiB normalized data) reject oversized inputs
before worker launch. FP32, fully trainable untied parameters only; existing
no-buffer/no-dropout restrictions apply. GPU/NCCL acceptance remains open.

`result.checkpoint` names `shards/latest.json`: an atomic manifest published only
after all rank-local model/optimizer/RNG files are durable. Retains two committed
epoch directories; interrupted pending data is ignored/cleaned after failed
workers stop. Resume requires identical model/data/options/world size and exact
Python/Torch versions. Native ShardedTensor metadata contains executable pickle:
`resume=..., trusted_resume=True` is mandatory for artifacts you trust. Hashes
validate integrity; they do not establish authorship. Rank files are not universal
exports and cannot be resharded to a different world size by this API.

`result.export_checkpoint` contains a full restricted state-dict checkpoint after
successful completion, and the caller model receives those weights. Failures
leave the caller unchanged and retain the last committed shard manifest.
Rank-zero callbacks are observers invoked with temporarily gathered parameters.
One writer per output directory. Final export temporarily gathers full parameters;
this reference path still constructs a complete model per rank before wrapping.
It is not meta-device loading of models larger than host RAM.

`result.diagnostics` reports per-rank local parameter element counts and, for CUDA,
worker peak allocated/reserved bytes through epoch collection. Those figures are
measurements, not aggregate device-memory quotas. `scripts/accept_gpu.py` requires
real CUDA/NCCL hardware and refuses skipped/failed required tests. The FSDP1 local
state-dict API emits upstream deprecation warnings; exact-version resumes are
intentional. GPU large-model memory/speed and mixed-precision FSDP remain unaccepted.

Resume with `epochs` equal to the committed epoch exports the existing state
without an extra optimizer step. In that case `checkpoint` keeps pointing to
the original shard manifest and `export_checkpoint` names the new full export.
