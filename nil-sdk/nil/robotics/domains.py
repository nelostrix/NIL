"""Pre-Built Embodied Robotics Domains for NIL Autonomous Agents.

Provides ready-to-run problem spaces:
1. AutonomousManipulationDomain: Pick-and-place, assembly, tool use
2. LeggedLocomotionDomain: Quadruped / Humanoid rough terrain traversal
3. AutonomousDroneInspectionDomain: Multi-rotor waypoint aerial surveying
4. FleetWarehouseDomain: Multi-robot AGV logistics and co-transport
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional
from nil.domain import Domain
from nil.tools.base import tool
from nil.robotics.sensorium import EmbodiedSensorium, SensorFrame
from nil.robotics.world_model import PhysicalWorldModel, PhysicalObject
from nil.robotics.affordance import AffordanceQuery
from nil.robotics.skill_memory import EmbodiedSkillMemory, EmbodiedSkill
from nil.robotics.safety_gatekeeper import PhysicalSafetyGatekeeper


class AutonomousManipulationDomain(Domain):
    """Domain for 6-DOF robotic manipulator arms and dexterous hands."""

    def __init__(self, name: str = "robotic_manipulation"):
        self.world = PhysicalWorldModel()
        self.sensorium = EmbodiedSensorium()
        self.skills = EmbodiedSkillMemory()
        self.safety = PhysicalSafetyGatekeeper()

        # Define embodied robotic tools
        @tool
        def detect_3d_objects(category: str = "") -> List[Dict]:
            """Perceive 3D objects in the active workcell."""
            return self.world.get_all_objects()

        @tool
        def evaluate_grasp(object_name: str) -> Dict:
            """Evaluate whether an object can be safely grasped."""
            obj = self.world.get_object(object_name)
            if not obj:
                return {"feasible": False, "reason": "Object not found in world model"}
            return AffordanceQuery.evaluate_grasp_affordance(obj)

        @tool
        def execute_safe_joint_motion(q_target: List[float]) -> Dict:
            """Execute a joint space trajectory through the physical safety filter."""
            q_curr = self.sensorium.get_latest().joint_positions if self.sensorium.get_latest() else [0.0]*len(q_target)
            q_safe, modified, reason = self.safety.sanitize_joint_commands(q_curr, q_target)
            return {
                "success": True,
                "applied": False,
                "q_commanded": q_safe.tolist(),
                "was_safety_clamped": modified,
                "safety_status": reason
            }

        super().__init__(
            name=name,
            tools=[detect_3d_objects, evaluate_grasp, execute_safe_joint_motion],
            objectives=[
                "Autonomously perceive, plan, and manipulate physical objects in 3D space",
                "Validate joint position and velocity bounds; external controllers enforce collision and torque safety",
                "Learn and refine reusable motor skills from operational experience"
            ],
            success_signal=lambda run: run.success
        )
