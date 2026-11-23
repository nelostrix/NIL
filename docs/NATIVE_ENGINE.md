# Independent numerical engine stages

The dependency-free CPU engine now supports general nonempty tensors, explicit
sparse matrices and an optional NIL-owned compiled matmul kernel. These are
bounded float64 reference workflows; they are not a complete accelerated DL engine.
No optional package, compiler invocation or native library load occurs on import.

| Stage | Implemented contract | Validation |
| --- | --- | --- |
| Dense CPU | Rank0–8, explicit shapes, C-order strides, slicing, owned reshape/permute/transpose, trailing broadcasting, axis reductions, batched/vector matmul | Finite differences, Torch float64 forward/backward, old API compatibility |
| Losses/optimizers | Stable last-axis softmax/log-softmax/cross entropy, shape-matched BCE/MSE, SGD/Adam over scalar/vector/ND leaves | Real learning, transactional updates, exact same-config optimizer continuation |
| Sparse CPU | Explicit rank2 COO/CSR, duplicates/coalescing, sparse×dense, values/dense gradients, bounded JSON persistence | Dense/Torch parity, finite differences, empty matrices, bounds and huge logical shapes without allocation |
| Compiled CPU | Explicit local Linux C kernel for matmul and its VJP, including broadcast batches; no external numerical runtime | Actual compilation, concurrent calls, float64 parity, dimensions/nonfinite/overflow guards and measured workload |
| GPU | Not accepted on available hardware | Actual devices, kernels, quotas and measured target tests remain open |

## Shapes and compatibility

`Tensor(2)` still has shape(1,1); `Tensor([1,2])` still has shape(1,2).
Use `Tensor(2, shape=())` for a rank0 scalar or `Tensor([1,2], shape=(2,))`
for a vector. Nested rank3+ arrays infer their actual shape; explicit shapes can
also reshape flat built-in input. Every dimension must be positive. Empty
selections, zero-sized dimensions, ragged data, unsupported dtypes/devices and
more than100,000 elements per tensor are rejected before result allocation.

```python
from nil import native as n
x = n.Tensor(list(range(24)), shape=(2, 3, 4), requires_grad=True)
y = x.permute(2, 0, 1)[:, 1, ::-1].reshape(3, 4)
loss = (y ** 2).mean()
loss.backward()
print(x.shape, x.strides, x.grad)
```

Shapes/strides describe owned contiguous row-major storage. Reshape, permutation
and slicing produce copies connected by autograd; there are no mutable views or
user-supplied strides. Integer/slice/ellipsis indexing is supported; fancy indexing
and inserted axes are not. Reductions accept an integer or tuple of axes and
keepdims. Default full reduction on rank2 retains legacy(1,1); explicit axis
reductions and full reductions on other ranks follow ordinary rank semantics.
A one-element differentiable tensor can call backward(). Negative axes are valid.

Matmul implements vector dot, matrix-vector/vector-matrix and broadcast matrix
batches. Linear supports arbitrary leading batch axes. Tensor gradients preserve
the original shape. Stale leaf mutation is rejected before backward changes any
gradient; assigning into differentiable intermediates is rejected immediately.

SGD/Adam accept unique differentiable leaves of any supported rank. All updates
and moment buffers are validated before commit. State loaders reject mismatched
configuration, count, shape or nonfinite moments without mutation. Adam has bias
correction and coupled L2 decay; AdamW/AMSGrad are not implemented. JSON tensor
archives preserve shape/dtype/device/requires_grad, not graphs or accumulated
gradients: `save_tensor(tensor,path)` and `load_tensor(path)`.
Existing model/optimizer archives and model save/load remain supported.

## Sparse matrices

```python
from nil import native as n
values = n.Tensor([2., 3., 4.], shape=(3,), requires_grad=True)
sparse = n.SparseCOO([(0, 1), (0, 1), (1, 0)], values, (2, 3))
csr = sparse.to_csr()  # canonical order, duplicate coordinates summed
result = csr @ n.Tensor([[1.], [2.], [3.]])
result.sum().backward()  # gradients also reach the original duplicate values
print(result.tolist(), values.grad)
```

COO stores coordinate pairs; CSR stores indptr and column indices. At most100,000
stored entries, and at most100,000 CSR pointer entries. Logical dimensions may be
large, up to2**31-1, without allocating their product. Explicit to_dense() and
matmul outputs remain subject to the dense100,000-element budget. Duplicates
are retained until coalescing and their gradients sum correctly; zero-valued
coalesced entries stay present. Empty sparse matrices use values=None and support
zero output/gradient matmul. Sparse×sparse, sparse optimizers, other ranks and
implicit densification are unsupported. `save_sparse`/`load_sparse` store bounded,
versioned plain JSON and reload constant values; no executable deserialization.

## Optional compiled CPU kernel

```python
from nil import native as n
path = n.build_cpu_kernel('/absolute/new-build-directory')
kernel = n.load_cpu_kernel(path)  # explicit trusted native code
result = n.Tensor([[1., 2.]]).matmul(n.Tensor([[3.], [4.]]), kernel=kernel)
print(result.item())  # 11.0
```

Builds require an existing Linux C compiler and matching Python development
headers. Nothing is downloaded, registered globally or compiled at install time.
The ordinary wheel contains only Python/help/C source, with no bundled binary or
external numerical runtime. Output files live only in the requested directory;
existing binaries are not overwritten. A build manifest records source/binary
hashes, ABI and compiler command. Loading verifies the active ABI and binary hash.
Rebuild per Python ABI and target platform. The builder supplies no -march=native;
operator-supplied CC flags remain explicit build configuration.

Python remains the default. Pass kernel= explicitly for matmul; forward and both
matrix VJPs use it. Arithmetic, activations, sparse operations and reductions
remain Python. C/Python reductions may differ at float64 rounding level; acceptance
uses declared tolerances, not cross-backend bitwise equality. Exact optimizer
continuation requires the same supported execution configuration.

## Measured current stage

The [C-kernel report](benchmarks/2026-11-23-native-cpu-kernels.json) records five
fresh no-site-package processes per backend,96x64 @64x48 forward plus backward,
independent parity checks, RSS and compiler metadata. The measured speedup was
about7.29x on the current host for that workload only. No NumPy/Torch/sklearn was
imported. This does not claim universal large-array performance or old-PC/GPU
acceptance. Smaller matrices may not benefit after conversion/graph overhead.

The earlier [multiclass report](benchmarks/2026-11-23-native-multiclass.json)
records the preceding rank2 engine; its timings are historical. The new ND engine
also converges on that90x2 reference workload without numerical backends.

Native-only scope: bounded dense regression/classification/MLPs, ND autograd,
SGD/Adam, sparse×dense algebra and JSON persistence. Broader classical estimators,
large/deep models, ONNX, environment RL and GPU workflows still use explicit
optional backends. See module help for usage and the live phase checklist for
remaining device, large-model and coverage gates.
