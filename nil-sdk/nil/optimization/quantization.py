"""Real CPU dynamic INT8 quantization and ONNX export, with explicit limits."""

import copy
import time
from pathlib import Path

from nil.runtime import require


def quantize(model, method="dynamic_int8"):
    """Return a separate CPU eval model; supports PyTorch Linear/LSTM/GRU.

    This is not INT4/GPTQ/AWQ, QAT or a general transformer quantizer. Advanced
    backends can consume the same native model without conversion wrappers.
    """
    torch = require("torch", "torch")
    if method != "dynamic_int8":
        raise ValueError("Supported method: dynamic_int8 (CPU Linear/LSTM/GRU)")
    kinds = {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}
    if not any(isinstance(module, tuple(kinds)) for module in model.modules()):
        raise ValueError("Model has no supported dynamic-quantization layers")
    if not torch.backends.quantized.supported_engines or (
        torch.backends.quantized.engine == "none"
    ):
        raise RuntimeError("This PyTorch build has no selected quantization engine")
    native = copy.deepcopy(model).cpu().eval()
    # PyTorch does not replace a quantizable root module, so wrap just that case.
    if isinstance(native, tuple(kinds)):
        native = torch.nn.Sequential(native)
    return torch.ao.quantization.quantize_dynamic(native, kinds, dtype=torch.qint8, inplace=False)


def export_onnx(
    model,
    example_inputs,
    path,
    input_names=None,
    output_names=None,
    dynamic_axes=None,
    opset_version=17,
):
    torch = require("torch", "torch")
    onnx = require("onnx", "onnx")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Export a clone: do not change the user's device/training mode.
    native = copy.deepcopy(model).cpu().eval()
    from nil.runtime import move

    torch.onnx.export(
        native,
        move(example_inputs, "cpu"),
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    return path


def benchmark(model, example_inputs, iterations=50, warmup=5):
    """Measure CPU inference, never extrapolate throughput from token counts."""
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup nonnegative")
    from nil.inference import predict

    times = []
    for index in range(warmup + iterations):
        start = time.perf_counter()
        predict(model, example_inputs, device_name="cpu")
        elapsed = (time.perf_counter() - start) * 1000
        if index >= warmup:
            times.append(elapsed)
    times.sort()
    return {
        "device": "cpu",
        "iterations": iterations,
        "mean_ms": sum(times) / len(times),
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[min(len(times) - 1, int(len(times) * 0.95))],
    }
