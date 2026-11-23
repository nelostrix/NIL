# mlp

Construct a native PyTorch multilayer perceptron for custom neural training.

## Usage

```python
import nil
import torch

nil.runtime.seed(7)
x = torch.linspace(-1, 1, 32).reshape(-1, 1)
data = nil.data.tensors(x, 2 * x + 1)
model = nil.model("mlp", input_dim=1, output_dim=1, hidden=(16,))
result = nil.train(model, data, epochs=20, device="cpu")
print(nil.evaluate(result.model, data, device_name="cpu"))
```

## Parameters

`input_dim`, `output_dim`: positive feature dimensions. `hidden`: hidden widths. `activation`: torch.nn activation name. `dropout`: probability in [0, 1).

## Output

A native `torch.nn.Sequential`; `nil.train` returns a TrainingResult with model, run, history, checkpoint and status.

## Requirements

Install `nil-sdk[torch]` or the source project with `pip install ".[torch]"`.

## Limitations

Output shape and loss must match the task. The example uses regression; classification requires class logits and classification loss. Compute and memory depend on the architecture and data.
