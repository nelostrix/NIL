# Sharded training and target acceptance infrastructure — 2026-11-22

NIL now provides explicit `nil.distributed.train(..., strategy='fsdp')` in the
existing shallow public module. It is an optional Torch FSDP1 integration, not a
NIL-native sharding engine. Base installation still has no dependencies and import
starts no workers or GPU contexts.

## Actual CPU acceptance

Two real Gloo ranks exercise FULL_SHARD, unequal rank minibatches, serial global
SGD/momentum parity, actual local parameter partitioning and exact epoch resume.
Model/optimizer/RNG shards publish through a manifest committed after all rank
files are durable. Rank failure and a real rank-zero publication exception retain
the preceding committed epoch and leave the caller unchanged. Recovery trains
from that epoch. Hash tampering and mismatched data/world/model contracts are
rejected. Single-host FP32, finite bounded data and fully trainable untied models;
no mutable buffers, dropout or batch normalization.

Native LOCAL_STATE_DICT includes executable ShardedTensor metadata. Explicit
`trusted_resume=True` is required before deserialization; exact Python/Torch and
world size are mandatory. The rank files are not cross-world exports. A full,
restricted model checkpoint is exported only after successful training. Final
export and callbacks temporarily gather complete parameters. A full model is
constructed before each worker wraps it, so this is not memory-free meta loading
or certification for models larger than host RAM. Keep one writer per output root.

The tested implementation uses documented FSDP1 local state-dict interfaces,
which emit upstream deprecation warnings. Newer distributed checkpoint APIs are
not claimed as an accepted interchange format. See the
[official FSDP documentation](https://docs.pytorch.org/docs/2.15/fsdp.html).

## GPU acceptance runner (actual target remains unavailable)

```sh
# Requires an explicitly provisioned Torch/CUDA/pytest interpreter.
python scripts/accept_gpu.py --mode amp --precision fp16 --output amp.json
python scripts/accept_gpu.py --mode amp --precision bf16 --output bf16.json
python scripts/accept_gpu.py --mode ddp --output ddp.json
python scripts/accept_gpu.py --mode fsdp --output fsdp.json
python scripts/accept_gpu.py --mode all --output all.json
```

AMP requires one real CUDA device. DDP/FSDP require two visible devices and NCCL.
Requested BF16 must be supported. Reports include Python/Torch/runtime, actual
device model/compute capability/memory and required nvidia-smi driver metadata.
Unavailable required capabilities return 2 with a failed JSON report; tests are
not silently downgraded. A test failure, skip or missing execution returns 1.

The executable GPU cases cover learning, accumulation, clipping after unscale,
forced FP16 overflow and skipped optimizer update, scaler checkpoint continuation,
serial/global-rank parity, rank-zero fault recovery and caller state preservation.
Worker memory metrics are reported by worker processes, not inferred from the
parent. GPU test code exists but no GPU positive result is recorded on this
CPU-only host. Broad large-model savings, FSDP mixed precision, device quotas and
multi-host support remain open.

## Platform and GPU CI definitions

`.github/workflows/sdk-platforms.yml` defines nine isolated base-wheel cells:
Python 3.10/3.11/3.12 across Linux/macOS/Windows. Fresh offline resolution must add
only NIL. POSIX cells run actual owned-worker acceptance. Windows currently checks
that unsupported process ownership fails explicitly; it does not claim Windows
containment support. `.github/workflows/sdk-gpu.yml` is manual-only on a trusted
self-hosted Linux x64 `nil-cuda` runner. Its GPU backends/pytest must be provisioned
explicitly; the job never substitutes CPU acceptance or installs a different Torch.

These workflow definitions have not been remotely executed by this session.
Cross-platform cells, Windows implementation, actual GPU tests and publication of
per-cell evidence remain open. Writing CI configuration does not close them.


Full classical source acceptance: **1046 passed, 19 optional/target skips**.
Full neural/RL source acceptance: **985 passed, 20 optional/target skips**.
Final bounded-input/help/runner checks validate complex-input rejection and
the public FSDP documentation after the full-suite runs.


Installed initial acceptance outside the source tree: **12 passed**, and the
installed --fsdp example passes actual two-rank training and checkpoint export.
Final recovery handling also permits exporting an already-completed shard epoch
without another optimizer update; the resumable manifest stays at its original
location and the full export is written to the requested new output directory.
The rebuilt distribution is revalidated for that final path below.


Final rebuilt-wheel recovery acceptance outside the source tree: **3 passed**,
including zero-update export of a completed epoch with a valid resumable manifest
reference. Initial installed sharding/DDP/runner suite passed 12 tests; the FSDP
example also passed. Both development environments use the final regular wheel
without adding dependencies. Clean offline installation adds only nil-sdk;
196 Python sources match and 103 help guides remain bundled. Installed offline
help does not load Torch/NumPy/sklearn. CI YAML parses, but no remote jobs were
executed. Hardware/OS/large-model acceptance remains open as listed above.
