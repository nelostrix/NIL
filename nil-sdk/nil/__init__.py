"""NIL model development and agent SDK, with optional compute backends."""
from importlib import import_module

__version__ = "0.2.1"
_EXPORTS = {
    "tool": ("nil.tools.base", "tool"),
    "Domain": ("nil.domain", "Domain"),
    "PersistentMemory": ("nil.memory.store", "PersistentMemory"),
    "HierarchicalMemory": ("nil.memory.hierarchical", "HierarchicalMemory"),
    "Agent": ("nil.agent", "Agent"),
    "EvolutionConfig": ("nil.config", "EvolutionConfig"),
    "Trainer": ("nil.training.trainer", "Trainer"),
    "TrainingConfig": ("nil.training.trainer", "TrainingConfig"),
    "train": ("nil.training.trainer", "train"),
    "evaluate": ("nil.evaluation", "evaluate"),
    "predict": ("nil.inference", "predict"),
    "Run": ("nil.lifecycle", "Run"),
    "Registry": ("nil.lifecycle", "Registry"),
}
for _name in (
    "EmbodiedSensorium", "SensorFrame", "PhysicalWorldModel", "PhysicalObject",
    "AffordanceQuery", "DynamicMovementPrimitive", "EmbodiedSkillMemory",
    "EmbodiedSkill", "PhysicalSafetyGatekeeper", "STACCRoboticsBridge",
    "AutonomousManipulationDomain",
):
    _EXPORTS[_name] = ("nil.robotics", _name)
_MODULES = {"data", "nn", "training", "evaluation", "inference", "optimization",
            "generative", "lifecycle", "runtime", "models", "robotics", "classical", "serving"}
__all__ = list(_EXPORTS) + sorted(_MODULES)


def __getattr__(name):
    if name in {"tensor", "Tensor", "optim", "autograd", "no_grad", "inference_mode"}:
        from nil.runtime import require
        value = getattr(require("torch", "torch"), name)
    elif name in _EXPORTS:
        module, symbol = _EXPORTS[name]
        value = getattr(import_module(module), symbol)
    elif name in _MODULES:
        value = import_module(f"nil.{name}")
    else:
        raise AttributeError(f"module 'nil' has no attribute {name!r}")
    globals()[name] = value
    return value
