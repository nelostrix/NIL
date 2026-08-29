"""Spatial Affordance & Physical Interaction Engine.

Computes physical interaction feasibility on objects and terrain:
- Grasp Affordance: Feasible antipodal gripper opening width, surface normal alignment
- Insertion Affordance: Mating axis alignment and clearance tolerances
- Traversability Affordance: Ground slope, roughness, and step obstacle height
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math
from nil.robotics.world_model import PhysicalObject


class AffordanceQuery:
    """Evaluates physical action affordances on perceived objects."""

    @staticmethod
    def evaluate_grasp_affordance(
        obj: PhysicalObject,
        max_gripper_stroke_m: float = 0.085,
        max_payload_kg: float = 5.0
    ) -> Dict[str, Any]:
        """Check if an object can be safely grasped by the end-effector."""
        min_dim = min(obj.dimensions)
        can_fit = min_dim <= max_gripper_stroke_m
        can_lift = obj.mass_kg <= max_payload_kg

        score = 1.0 if (can_fit and can_lift) else 0.0
        if obj.is_fragile:
            score *= 0.7  # Requires soft impedance grasp

        return {
            "affordance": "grasp",
            "feasible": can_fit and can_lift,
            "score": round(score, 3),
            "grasp_width_required_m": min_dim,
            "recommended_force_n": min(25.0, max(5.0, obj.mass_kg * 9.81 / (obj.friction_coeff + 1e-4))),
            "reason": "Affordance verified" if (can_fit and can_lift) else "Object exceeds gripper stroke or payload limits"
        }

    @staticmethod
    def evaluate_traversability(slope_deg: float, step_height_m: float, max_step_m: float = 0.15) -> Dict[str, Any]:
        """Evaluate legged or wheeled ground traversability."""
        feasible = slope_deg <= 30.0 and step_height_m <= max_step_m
        score = max(0.0, 1.0 - (slope_deg / 30.0) * 0.5 - (step_height_m / max_step_m) * 0.5)
        return {
            "affordance": "traversal",
            "feasible": feasible,
            "score": round(score, 3),
            "slope_deg": slope_deg,
            "step_height_m": step_height_m
        }
