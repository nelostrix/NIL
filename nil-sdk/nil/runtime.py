"""Optional backend discovery and explicit device selection."""

import platform
import random
from importlib import import_module, metadata, util


class MissingDependencyError(ImportError):
    """An optional backend has not been installed."""


def require(module: str, extra: str):
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == module or module.startswith(f"{exc.name}."):
            raise MissingDependencyError(
                f"{module} is required for this operation. Install nil-sdk[{extra}] "
                "in your selected Python environment."
            ) from exc
        raise


def capabilities() -> dict:
    """Discover packages without importing frameworks or claiming tested hardware."""
    packages = {
        "torch": "torch",
        "numpy": "numpy",
        "transformers": "transformers",
        "peft": "peft",
        "sklearn": "scikit-learn",
        "tensorflow": "tensorflow",
        "onnx": "onnx",
        "onnxruntime": "onnxruntime",
        "mlflow": "mlflow",
    }
    result = {}
    for module, distribution in packages.items():
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            version = None
        result[module] = {"installed": util.find_spec(module) is not None, "version": version}
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": result,
        "hardware_verified": False,
    }


def device(name="auto"):
    torch = require("torch", "torch")
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    selected = torch.device(name)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; choose cpu or install a compatible runtime."
        )
    if selected.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but unavailable.")
    return selected


def seed(value: int):
    random.seed(value)
    torch = require("torch", "torch")
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def move(value, target):
    """Move nested tensors without changing non-tensor user metadata."""
    torch = require("torch", "torch")
    if isinstance(value, torch.Tensor):
        return value.to(target)
    if isinstance(value, dict):
        return {k: move(v, target) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move(v, target) for v in value)
    if isinstance(value, list):
        return [move(v, target) for v in value]
    return value
