# Independent numerical engine stages

Only the first stage below is implemented. This is a development contract for
expansion, not a claim that arbitrary tensors or accelerated kernels already work.
The default installation stays NIL-only; imports start no execution or installation.

| Stage | Shape/storage and numeric contract | Acceptance gate |
| --- | --- | --- |
| Current dense reference | Nonempty rank2, row-major immutable tuples, scalar/vector inputs normalize to 1x1/1xN, float64 CPU only; 100,000 elements/tensor | Finite differences, Torch float64 parity, real regression/classification learning and exact JSON optimizer continuation |
| General dimensions (planned) | Explicit shapes, rank0 scalars, contiguous row-major storage and strides; preserve legacy rank2 input behavior through an explicit migration contract | Index/reshape/transpose/reduction/broadcast property tests, bounds/size preflight and independent gradient checks |
| Sparse (planned) | Explicit COO/CSR values, duplicate/coalescing and sparse-gradient contracts; refuse implicit densification | Dense reference parity, empty/duplicate/bounds cases and measured memory limits |
| Compiled CPU (planned) | Optional vectorized kernels with explicit layout/dtype fallbacks; no silent package downloads | Representative workload speed/RSS measurements against reference and optional backends |
| Devices (planned) | Explicit device ownership, transfers, kernel synchronization and measured aggregate resource policy | Actual device tests, error cleanup and numerical/performance evidence |

For the proposed general-dimension stage, C-order strides are products of trailing
sizes; indices must be within every dimension. Start with owned contiguous storage:
reshape/transpose return copies until alias/version safety has independent tests.
Define zero-sized dimension/reduction behavior before accepting it. Broadcasting
aligns trailing dimensions and rejects incompatible sizes before allocation. Never
infer a dense allocation from a sparse input. No user-supplied arbitrary strides
or device pointers in that stage. These decisions require implementation and tests
before an API is advertised; current Tensor still accepts only rank2 semantics.

Current parameter contracts: optimizers own unique differentiable leaves, graphs
capture parent versions and reject stale mutations, model/optimizer JSON formats
are versioned and bounded. Saved optimizer parameter order must equal model order.
SGD and Adam checkpoint continuation is tested. Adam validates all parameter/moment
updates before committing any; its state loader rejects bad shapes/configuration
without mutation. Adam uses coupled L2 decay, not decoupled AdamW. There is no
AMSGrad, mixed precision or arbitrary optimizer-state import.

Row-wise stable softmax/log-softmax and categorical cross entropy support integer
labels and mean reduction. Binary BCE-logit and MSE behavior remains available.
Class weights, ignored labels, target gradients and smoothing require separate
contracts. General ND tensors, convolution and sparse/device execution remain open.

## Measured current stage

Five fresh Python processes ran with `-S` and no NumPy/Torch/sklearn import on the
identified development host. The 90-row, two-feature, three-class linear model ran
100 Adam steps, achieved accuracy1.0 and final cross entropy about0.00696.
Median fit time was about0.101s, median peak process RSS about17.65MiB.
These are tiny reference-workload results on a Ryzen3 7320U, not old-PC acceptance
or a claim of competitive large-array performance. Raw samples are in
[the benchmark report](benchmarks/2026-11-23-native-multiclass.json).

```sh
PYTHONPATH=nil-sdk python scripts/benchmark_native.py --workload multiclass --output report.json
PYTHONPATH=nil-sdk python -m pytest nil-sdk/tests/test_native_engine.py nil-sdk/tests/test_native_multiclass.py -q
```

Native-only workflows now include small dense regression, binary/multiclass
classification, shallow MLP training, autograd and JSON persistence with SGD/Adam.
General classical estimators, large/deep models, ONNX, RL integrations and GPU
workflows still use their explicit optional backends. Adapters do not make those
workflows independent engines.
