"""Explicit model quantization, export, benchmarking and CPU profiling."""

from nil.optimization.profiling import profile
from nil.optimization.quantization import benchmark, export_onnx, quantize

__all__ = ["quantize", "export_onnx", "benchmark", "profile"]


def activation_checkpoint(sequence,*,segments=2):
    """Explicit non-reentrant checkpointing for native Sequential tensor-input models."""
    from nil.runtime import require
    from nil._classical_common import positive_int
    torch=require('torch','torch'); positive_int(segments,'segments')
    if not isinstance(sequence,torch.nn.Sequential) or not len(sequence) or segments>len(sequence):
        raise ValueError('Use a nonempty torch.nn.Sequential and at most one segment per layer')
    if list(sequence.buffers()): raise ValueError('Stateful forward buffers cannot be recomputed safely')
    from nil.optimization._checkpointing import ActivationCheckpoint
    return ActivationCheckpoint(sequence,segments)


__all__.append('activation_checkpoint')
