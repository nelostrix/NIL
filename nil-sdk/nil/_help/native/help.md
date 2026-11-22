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
vector (one row), or rectangular 2D built-in numbers. `Linear(inputs, outputs)`;
`Sequential(Linear(...), Tanh(), Linear(...))`; `fit(..., loss='mse')` or
`loss='bce_logits'`. Direct loss/backward and SGD support momentum and
weight decay. `zero_grad()` before each independent update.

## Output

Tensor values/gradients, fit result dictionary, or immutable JSON model/optimizer
archive. `load(path)` returns `(model, optimizer)`; the optimizer is None when not saved.

## Requirements

Python standard library only. Optional Torch is used solely for independent
parity tests, never engine execution.

## Limitations

Reference subset, bounded at 100,000 elements per Tensor, 16 layers and 10,000
fit epochs. CPU float64 only; sparse data, other dtypes/devices and ragged/nonfinite
values are rejected. Supports row/column/scalar broadcasting, matmul, arithmetic,
ReLU/tanh/sigmoid/exp/log, reductions, MSE and stable BCE logits. No convolution,
GPU, sparse kernel, mixed precision or general-purpose large tensor engine.
Pure Python prioritizes inspectability and zero backend dependencies; numerical
backends are faster for large arrays. Stale graphs reject mutation before backward.
