"""Real CPU dynamic INT8 quantization and ONNX export, with explicit limits."""

import copy
import time
from pathlib import Path

from nil.runtime import require


def quantize(model, method="dynamic_int8", *, exclude=(), weight_scheme=None):
    """Return a separate CPU eval model; supports PyTorch Linear/LSTM/GRU.

    This is not INT4/GPTQ/AWQ, QAT or a general transformer quantizer. Advanced
    backends can consume the same native model without conversion wrappers.
    """
    torch = require("torch", "torch")
    if method not in ('dynamic_int8','weight_only_int8'):
        raise ValueError('Choose dynamic_int8 or weight_only_int8')
    if weight_scheme is None:weight_scheme='per_channel' if method=='weight_only_int8' else 'per_tensor'
    if weight_scheme not in ('per_tensor','per_channel'):
        raise ValueError('weight_scheme must be per_tensor or per_channel')
    if method=='weight_only_int8' and weight_scheme!='per_channel':
        raise ValueError('weight_only_int8 requires per_channel weights')
    if not isinstance(exclude,(tuple,list)) or any(not isinstance(name,str) for name in exclude):
        raise ValueError('exclude must contain explicit module paths')
    modules=dict(model.named_modules(remove_duplicate=False))
    if set(exclude)-set(modules):raise ValueError('Unknown excluded module path')
    blocked={name for name in modules if any(not root or name==root or name.startswith(root+'.') for root in exclude)}
    kinds = {torch.nn.Linear} if method=='weight_only_int8' else {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}
    if not any(type(module) in kinds and name not in blocked for name,module in modules.items()):
        raise ValueError("Model has no supported dynamic-quantization layers")
    if method=='weight_only_int8' and any(value.device.type!='cpu' or value.is_complex() or (value.is_floating_point() and value.dtype!=torch.float32)
            for value in list(model.parameters())+list(model.buffers())):
        raise ValueError('weight_only_int8 requires an explicit CPU float32 model')
    if method=='dynamic_int8' and (not torch.backends.quantized.supported_engines or (
        torch.backends.quantized.engine == "none"
    )):
        raise RuntimeError("This PyTorch build has no selected quantization engine")
    native = copy.deepcopy(model).cpu().eval()
    # PyTorch does not replace a quantizable root module, so wrap just that case.
    if isinstance(native, tuple(kinds)):
        native = torch.nn.Sequential(native)
        blocked={'0'+('.'+name if name else '') for name in blocked}
    if method=='weight_only_int8':
        from nil.optimization._weight_only import convert
        return convert(native,blocked)
    specification={kind:torch.ao.quantization.default_dynamic_qconfig for kind in kinds}
    if weight_scheme=='per_channel':specification[torch.nn.Linear]=torch.ao.quantization.per_channel_dynamic_qconfig
    specification.update({name:None for name in blocked})
    for name,module in native.named_modules(remove_duplicate=False):
        if name in blocked:module.qconfig=None
        elif type(module) in kinds:module.qconfig=specification[type(module)]
    # `native` is already our private copy. A second deepcopy here can add a
    # full model's worth of peak RAM before any layers become INT8.
    return torch.ao.quantization.quantize_dynamic(native, specification, dtype=torch.qint8, inplace=True)


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
