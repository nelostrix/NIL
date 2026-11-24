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

## Selective quantization and INT8 weight storage

`quantize(model, exclude=('module.path',), weight_scheme='per_channel')` supports
explicit module/subtree exclusion. Unknown paths fail before copying the model.
Per-channel weights apply to Linear in dynamic mode; recurrent layers retain
native dynamic configuration. The caller's mode, weights and module ownership
are preserved. Quantization works on one private copy, avoiding a redundant
full-model clone before conversion.

`quantize(model, method='weight_only_int8', exclude=('lm_head',))` uses NIL-owned
CPU Linear modules with signed INT8 weight buffers, per-output-channel FP32
scales and FP32 computation. Each operation dequantizes its own current weight;
no persistent FP32 weight cache is created. This reduces weight storage and
is not an INT8 activation kernel, QAT, INT4 or a speed guarantee. It requires
CPU float32 inputs. Only standard Linear is converted in this mode. State dicts
can be restored into the same converted architecture. Excluded modules stay
floating-point, which can preserve sensitive/tied output heads.

The pinned SmolLM2-135M check passed weight-only storage with a +0.148 NLL delta
under the declared +0.5 limit. Dynamic-activation INT8 failed that model's fidelity
limit, including the tested per-channel/head-excluded variant, and is not counted
as accepted there. Existing small-model dynamic INT8 checks remain separate.
See docs/PRETRAINED_CPU.md for exact source, memory and numerical limits.
