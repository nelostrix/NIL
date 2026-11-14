"""Inference helpers that preserve native model state and framework access."""

from nil.runtime import device, move, require


def predict(model, inputs, device_name="auto"):
    torch = require("torch", "torch")
    target = device(device_name)
    previous = model.training
    model.to(target).eval()
    try:
        with torch.inference_mode():
            inputs = move(inputs, target)
            if isinstance(inputs, dict):
                return model(**inputs)
            if isinstance(inputs, tuple):
                return model(*inputs)
            return model(inputs)
    finally:
        model.train(previous)


class OnnxModel:
    def __init__(self, path, providers=None):
        ort = require("onnxruntime", "onnx")
        ort.disable_telemetry_events()
        providers = providers or ["CPUExecutionProvider"]
        unavailable = set(providers) - set(ort.get_available_providers())
        if unavailable:
            raise ValueError(f"ONNX execution providers unavailable: {sorted(unavailable)}")
        self.session = ort.InferenceSession(str(path), providers=providers)

    def predict(self, inputs):
        numpy = require("numpy", "onnx")
        names = [item.name for item in self.session.get_inputs()]
        if not isinstance(inputs, dict):
            if len(names) != 1:
                raise ValueError("Multi-input ONNX models require a dictionary")
            inputs = {names[0]: inputs}
        if set(inputs) != set(names):
            raise ValueError(f"Expected ONNX input names {names}")
        return self.session.run(None, {k: numpy.asarray(v) for k, v in inputs.items()})
