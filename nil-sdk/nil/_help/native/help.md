# native

A dependency-free CPU/float64 reference engine for small dense models. Importing
and training here needs only Python, including `python -S`.

## Usage

```python
import nil
model = nil.native.Linear(1, 1, seed=0)
result = nil.native.fit(model, [[0.], [1.], [2.]], [[1.], [3.], [5.]], epochs=300, lr=.1)
print(model(nil.native.Tensor([[3.]])).tolist())
# nil.native.save(model, 'model.json')  # New path; plain JSON, no pickle.
```

## Parameters

`Tensor(data, requires_grad=False, dtype='float64', device='cpu')` accepts scalar,
vector (one row), or rectangular built-in arrays; explicit ND shapes are described below. `Linear(inputs, outputs)`;
`Sequential(Linear(...), Tanh(), Linear(...))`; `fit(..., loss='mse')` or
`loss='bce_logits'`. Direct loss/backward and SGD support momentum and
weight decay. `zero_grad()` before each independent update.

For multiclass learning, `Tensor.softmax()` and `Tensor.log_softmax()` operate
across each row. `cross_entropy(logits, [class_index, ...])` computes mean loss
from integer class labels, with stable log-sum-exp for extreme logits. Labels
are constant; weights, ignored labels and label smoothing are not supported.

```python
model = nil.native.Linear(2, 3, seed=7)
result = nil.native.fit(model, [[1, 0], [0, 1], [-1, -1]], [0, 1, 2],
                        epochs=100, lr=.05, loss='cross_entropy', optimizer='adam')
probabilities = model(nil.native.Tensor([[1, 0]])).softmax().tolist()
```

`Adam(parameters, lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=0)`
uses bias correction and coupled L2 decay, matching standard Adam (not AdamW).
All parameters must have finite gradients. Steps validate all updates before
mutating parameters or moments. `state_dict()` returns independent buffers;
`load_state_dict()` rejects mismatched configuration/shapes transactionally.
`save(model, path, optimizer=optimizer)` preserves exact supported continuation
for both SGD and Adam. Optimizer parameters must match the model in order.

## Output

Tensor values/gradients, fit result dictionary, or immutable JSON model/optimizer
archive. `load(path)` returns `(model, optimizer)`; the optimizer is None when not saved.

## Requirements

Python standard library only. Optional Torch is used solely for independent
parity tests, never engine execution.

## Limitations

Reference subset, bounded at 100,000 elements per Tensor, 16 layers and 10,000
fit epochs. CPU float64 only; implicit sparse conversion, other dtypes/devices and ragged/nonfinite
values are rejected. Explicit COO/CSR is supported below. Supports row/column/scalar broadcasting, matmul, arithmetic,
ReLU/tanh/sigmoid/exp/log, reductions, MSE, stable BCE/categorical losses and Adam.
No convolution, GPU, mixed precision or unbounded large tensor engine.
Pure Python prioritizes inspectability and zero backend dependencies; numerical
backends are faster for large arrays. Stale graphs reject mutation before backward.

## Multidimensional and sparse CPU tensors

The dense engine supports nonempty rank0–8 tensors. Preserve legacy scalar/vector
normalization by default; pass shape=() or shape=(N,) for explicit scalar/vector
semantics. `Tensor(flat_data, shape=(B,T,F))`, strides, reshape(*shape),
permute(*axes), transpose(a,b), integer/slice/ellipsis indexing, sum/mean(axis=...,
keepdims=...), last-axis softmax/log_softmax and broadcast batched/vector matmul
are available. Reshape/permutation/slices own their storage and retain gradients.
Empty dimensions, fancy indexing, mutable views and arbitrary strides are refused.
Scalar/vector/ND SGD and Adam state continuation is supported. save_tensor and
load_tensor preserve tensor values/shape and requires_grad, not its graph/gradients.

`SparseCOO([(row,col),...], values, (rows,cols))` and
`SparseCSR(indptr, columns, values, shape)` provide explicit rank2 sparse storage.
Values can be a differentiable vector Tensor. Duplicates are summed by coalesce;
to_csr/to_coo, explicit to_dense, sparse@dense, save_sparse/load_sparse are supported.
No implicit densification. Dense results and stored arrays are bounded at100,000
entries; huge logical sparse shapes do not allocate their product. Empty sparse
matrices are valid; their values property is None. Sparse×sparse is unsupported.

## Explicit CPU compilation

`build_cpu_kernel(new_directory)` builds NIL's own float64 matrix kernel using an
existing Linux C compiler and matching Python headers. It downloads nothing and
never runs during import/install. `load_cpu_kernel(path)` checks its ABI/hash and
loads explicit trusted code. `a.matmul(b, kernel=kernel)` uses it for forward and
matrix gradients. Python remains the default; other operators remain Python.
Rebuild per ABI/target; compiled and reference reductions need numerical tolerance.
See docs/NATIVE_ENGINE.md for supported shapes, examples and measured limits.
