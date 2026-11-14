"""
Optimization — cost, speed, and accuracy tuning.

Phase 4 modules. Automatically routes subtasks to the cheapest
model that can handle them, compresses multi-step chains into
fewer steps, and manages caching.
"""

from nil.optimization.profiling import profile
from nil.optimization.quantization import benchmark, export_onnx, quantize

__all__ = ["quantize", "export_onnx", "benchmark", "profile"]
