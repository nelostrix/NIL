"""Measured CPU model footprint and latency; no estimated accelerator performance."""

import copy
import io

from nil.optimization.quantization import benchmark
from nil.runtime import require


def profile(model, example_inputs, iterations=20, warmup=3):
    """Profile a CPU clone without mutating the caller's device or training mode.

    Serialized state size includes metadata/packing; parameter bytes omit packed
    quantized weights, so serialized_state_bytes is the comparison for INT8.
    This does not measure peak RAM, FLOPs or training throughput.
    """
    torch = require("torch", "torch")
    native = copy.deepcopy(model).cpu().eval()
    stream = io.BytesIO()
    torch.save(native.state_dict(), stream)
    parameters = list(native.parameters())
    buffers = list(native.buffers())
    return {
        "parameters": sum(p.numel() for p in parameters),
        "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
        "buffer_bytes": sum(p.numel() * p.element_size() for p in buffers),
        "serialized_state_bytes": stream.tell(),
        "latency": benchmark(native, example_inputs, iterations=iterations, warmup=warmup),
    }
