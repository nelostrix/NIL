# training

Train neural models with custom losses, callbacks, checkpoints and optional bounded search.

## Usage

```python
import nil
import torch

x = torch.linspace(-1, 1, 32).reshape(-1, 1)
data = nil.data.tensors(x, 2 * x + 1)
model = nil.model("mlp", input_dim=1, output_dim=1, hidden=())
result = nil.train(model, data, epochs=20, device="cpu")
print(result.status, result.history[-1])
```

## Parameters

`model`, `data`, `task`, `epochs`, `learning_rate`, `device`, `output_dir`. Advanced users can pass a native optimizer and custom step through Trainer.
Tasks include regression, multiclass classification, binary_classification,
multilabel_classification, imitation_learning and causal_lm. Binary/multilabel
models emit logits and use matching-shape BCE targets. Integer multiclass masks
use -100 for ignored labels. Native convolutional segmentation works with the
classification task; arbitrary sequence/multitask objectives use custom steps.
Standard mean losses accumulate by their actual native loss denominator, including
short batches and weighted/ignored class labels. Sum losses are not averaged.
Custom steps default to equal microbatch weighting; define normalization_count
and an optional reduction='sum' when your objective needs different aggregation.
`precision='fp32'` is the default; CPU/CUDA bf16 and CUDA fp16 use autocast.
fp16 uses GradScaler, unscales before clipping, skips unsafe updates and saves
scaler state. Use hardware-specific acceptance before relying on CUDA precision.

## Output

TrainingResult containing model, run, history, checkpoint and status.

## Requirements

nil-sdk[torch].

## Limitations

Single-process training; no integrated DDP/FSDP. CPU bf16 is tested; CUDA tests
require actual GPU hardware and remain unverified here. bf16 can be slower on
older CPUs. Resume is at epoch boundaries and rejects incompatible config,
precision, model structure/shapes/dtypes and optimizer/scheduler classes before
loading weights. Loader descriptors, external loader/sampler generators and explicit dataset/sampler
state are captured in checkpoints. Custom step state and callbacks own their
additional resume contracts.
Custom optimizers must own unique parameters from this model, with schedulers
bound to that optimizer. Build a custom optimizer after moving the model to its
training device. Optimizer parameter names/order are checked during resume;
subset optimizers normalize/clip their own gradients while all model gradients
are cleared between updates. Standard checkpoints keep the current and one
previous immutable generation; the usual .pt alias remains loadable. On an
interrupted publication, call nil.training.checkpoint.recover(path) before load.
recover(path, previous=True) explicitly rolls back one committed generation.
Single-writer access is required. Retaining a prior generation adds disk usage;
hard links avoid an extra payload copy where supported. Checksums detect changes,
not source authenticity; restricted torch loading is not an untrusted-code sandbox.
Native execution time/memory depend on workload; use nil.execution for opted-in
POSIX timeouts, cancellation and Linux per-process virtual address-space limits.


## Bounded loaders and streams

`num_workers=0`, `prefetch_factor=2`, `pin_memory=False` preserve the lightweight
path. Worker count and prefetch factor are each bounded to eight; pinned bytes
still depend on tensor size, so this is not a hard memory quota. Map datasets
use an epoch-seeded loader and seed each worker's Python/NumPy RNG. Native
external loaders preserve loader/sampler generator states and call sampler
`set_epoch` when supplied. Keep dataset contents and transforms immutable;
matching type/length is not a checksum of arbitrary map datasets. Custom sampler
constructor semantics must remain identical, alongside their explicit state.

Stateful objects implement `state_dict`, nonmutating `validate_state_dict` and
`load_state_dict`. Validation runs before model/optimizer/RNG restore. These
methods must agree and preserve unrelated RNG; arbitrary custom code is trusted.
Stateful data/samplers use a single worker so a prefetched cursor cannot advance
past trained records. Persistent workers are rejected because live worker RNG
cannot be restored at this boundary. External collate/worker transforms and
third-party RNG remain caller-owned; matching numerical results require the
same functions and state. Ordinary in-process Python/Torch RNG is checkpointed.

Use `nil.data.jsonl_stream(path, transform)` with `shuffle=False`, `num_workers=0`
and explicit `max_batches_per_epoch` to consume bounded chunks without loading
the whole file. Each epoch continues at the next record; it does not rewind at
EOF. A fresh stream instance resumes from the checkpoint's validated byte offset.
The iterator never fetches a surplus batch to test the budget. Source SHA-256
is checked before iteration/checkpoint/restore; use a local immutable file.
The supported boundary is a completed bounded chunk/epoch, not an interrupted
batch or arbitrary live simulator/worker state. Empty/exhausted streams fail.

## Compact-model research experiments

The bundled `compact_research.py` example uses an ordinary custom Torch module
with shared attention blocks and an optional custom temperature-scaled offline
logit-distillation objective. The Trainer's normal accumulation and checkpoint
contracts apply; encode reuse depth in architecture identity because tied-depth
changes can leave parameter shapes unchanged. Teacher logits are frozen dataset
entries and are not teacher model checkpoints. Learned halting, token/process
objectives and shared-model distributed acceptance are not implemented by that
example. See docs/RESEARCH_READINESS.md and docs/project/RESEARCH_TODO.md.
