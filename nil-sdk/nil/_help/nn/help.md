# nn

Construct neural layers and modules using native PyTorch objects.

## Usage

```python
import nil

model = nil.nn.mlp(4, 1, hidden=(16,))
print(nil.nn.parameters(model))
```

## Parameters

MLP dimensions, hidden widths, activation and dropout. Other names delegate to torch.nn.

## Output

Native neural modules and parameter counts.

## Requirements

nil-sdk[torch].

## Limitations

Does not install drivers or select a suitable architecture automatically.
CNN/LSTM/Transformer examples and actual CPU training/roundtrip checks live in
examples/deep_workflows.py. Standard torch.nn modules remain directly accessible;
the examples do not introduce independent numerical implementations. MLP hidden
width iterables are materialized once so generators produce all requested layers.
