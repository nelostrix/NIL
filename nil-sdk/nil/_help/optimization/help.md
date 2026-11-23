# optimization

Explicitly quantize, export or measure a compatible neural model.

## Usage

```python
import nil
import torch

model = nil.nn.mlp(1, 1, hidden=(16,))
quantized = nil.optimization.quantize(model)
print(quantized(torch.zeros(1, 1)))
```

## Parameters

Model and example inputs as required by quantize, export_onnx, benchmark or profile.

## Output

A separate quantized model, exported file or measured CPU statistics.

## Requirements

torch; ONNX export additionally needs the onnx extra.

## Limitations

CPU dynamic INT8 is limited to supported native layers. No general INT4/QLoRA quantizer or automatic speedup guarantee.

## Activation checkpointing

`nil.optimization.activation_checkpoint(sequential, segments=2)` wraps a nonempty
native `torch.nn.Sequential`, using nonreentrant activation recomputation during
training and direct evaluation otherwise. RNG state is preserved. Modules with
buffers are rejected. Actual saved-activation byte reduction, gradient parity and
training/checkpoint reload are tested on CPU. It does not implement FSDP sharding.
