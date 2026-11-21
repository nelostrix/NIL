"""Lazy robotics APIs: load each feature only when it is requested."""

from importlib import import_module

_EXPORTS = {
    'EmbodiedSensorium': ('nil.robotics.sensorium', 'EmbodiedSensorium'),
    'SensorFrame': ('nil.robotics.sensorium', 'SensorFrame'),
    'PhysicalWorldModel': ('nil.robotics.world_model', 'PhysicalWorldModel'),
    'PhysicalObject': ('nil.robotics.world_model', 'PhysicalObject'),
    'AffordanceQuery': ('nil.robotics.affordance', 'AffordanceQuery'),
    'DynamicMovementPrimitive': ('nil.robotics.dmp', 'DynamicMovementPrimitive'),
    'EmbodiedSkillMemory': ('nil.robotics.skill_memory', 'EmbodiedSkillMemory'),
    'EmbodiedSkill': ('nil.robotics.skill_memory', 'EmbodiedSkill'),
    'PhysicalSafetyGatekeeper': ('nil.robotics.safety_gatekeeper', 'PhysicalSafetyGatekeeper'),
    'STACCRoboticsBridge': ('nil.robotics.stacc_bridge', 'STACCRoboticsBridge'),
    'AutonomousManipulationDomain': ('nil.robotics.domains', 'AutonomousManipulationDomain'),
    'policy': ('nil.robotics.learning', 'policy'),
    'demonstrations': ('nil.robotics.learning', 'demonstrations'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value
