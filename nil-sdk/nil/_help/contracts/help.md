# Framework contracts

## Usage

```python
import nil
from nil.contracts import MapDataset, Predictor, SingleModelResult, result_info
# result = nil.train(model, dataset, device='cpu', output_dir='runs')
# print(result_info(result).to_dict())
# prediction = nil.predict(result, inputs, device_name='cpu')
```

## Parameters

The protocols document structural interfaces without importing backends.
MapDataset provides len and integer indexing; Predictor exposes predict;
SingleModelResult exposes model and status. Runtime protocol checks inspect
attributes only; they do not validate shapes or backend compatibility.

## Output

result_info supports NIL classical, neural, online/offline/centralized RL and
native fit outcomes. It returns family, lifecycle status, model count, optional
history length, metric names, paths and step counts. to_dict contains metadata,
never model weights or live backend objects. Unsupported outcomes raise TypeError.

## Limitations

Classical fit accepts feature/target arrays; neural train accepts map datasets or
loaders; RL train accepts environments. Shared prediction unwraps single-model
classical, neural and online RL outcomes; native and centralized policies keep
explicit APIs. Neural prediction preserves training mode but may move devices.
Persistence remains family-specific: nil.save/load for classical; neural
training.checkpoint; RL save/load; native save_tensor/load_tensor and model JSON.
See docs/SDK_CONTRACTS.md for the release boundary and callback/error semantics.

## Requirements

Metadata/protocols use the standard library; constructing numerical outcomes requires their explicit backends.
