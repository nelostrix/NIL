# inference

Run classical/neural prediction, classical feature transformation, or explicit ONNX inference.

## Usage

```python
import nil
import torch

model = nil.nn.mlp(1, 1, hidden=())
print(nil.predict(model, torch.tensor([[0.5]]), device_name="cpu"))
```

## Parameters

`model`, `inputs`, `device_name`; ONNX uses OnnxModel(path, providers=...).
For native RL models/results use nil.predict(model, observation,
deterministic=True); its return remains the native (action, recurrent_state).
Extra classical options pass through to native predict. Neural multi-input
arguments remain an input tuple/dictionary.

## Output

Native tensor predictions; OnnxModel returns native runtime outputs.

## Requirements

classical for native sklearn prediction/transforms; torch for neural prediction; the onnx extra for ONNX inference.

## Limitations

Inputs must match the model signature. nil.predict(model_or_result, X) delegates directly to classical prediction without importing torch.

## Feature transformers

Use `nil.transform(model_or_result, inputs)` for fitted PCA, selectors and other feature transformers. Output remains native (including sparse matrices); unsupported predict/transform operations fail explicitly.
