"""3D Physical World Model & Object-Centric Semantic Memory.

Maintains spatial representation of the physical environment:
- 3D Object Entities: Position, Orientation, Bounding Dimensions, Mass, Friction, Fragility
- Spatial Relationships: "on_top_of", "inside", "adjacent_to", "held_by"
- Dynamic State Changes & Tracking over Time
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import time


class PhysicalObject:
    """A recognized 3D physical object in the robot's workspace."""

    def __init__(
        self,
        name: str,
        category: str = "generic",
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        dimensions: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        mass_kg: float = 0.5,
        friction_coeff: float = 0.5,
        is_fragile: bool = False,
        is_movable: bool = True
    ):
        self.name = str(name)
        self.category = str(category)
        self.position = list(position)
        self.orientation = list(orientation_rpy)
        self.dimensions = list(dimensions)
        self.mass_kg = float(mass_kg)
        self.friction_coeff = float(friction_coeff)
        self.is_fragile = bool(is_fragile)
        self.is_movable = bool(is_movable)
        self.last_seen = time.time()

    def distance_to(self, target_pos: Tuple[float, float, float]) -> float:
        """Euclidean distance in meters."""
        return math.sqrt(
            (self.position[0] - target_pos[0]) ** 2 +
            (self.position[1] - target_pos[1]) ** 2 +
            (self.position[2] - target_pos[2]) ** 2
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "position": self.position,
            "orientation": self.orientation,
            "dimensions": self.dimensions,
            "mass_kg": self.mass_kg,
            "friction_coeff": self.friction_coeff,
            "is_fragile": self.is_fragile,
            "is_movable": self.is_movable,
            "last_seen": self.last_seen
        }


class PhysicalWorldModel:
    """3D Semantic World Model & Spatial Reasoning Engine."""

    def __init__(self):
        self.objects: Dict[str, PhysicalObject] = {}
        self.robot_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.held_object: Optional[str] = None

    def upsert_object(self, obj: PhysicalObject) -> None:
        """Add or update an object entity in the 3D world model."""
        self.objects[obj.name] = obj

    def get_object(self, name: str) -> Optional[PhysicalObject]:
        """Look up an object by name."""
        return self.objects.get(name)

    def find_nearest_object(self, query_pos: Tuple[float, float, float], category: Optional[str] = None) -> Optional[PhysicalObject]:
        """Find the closest object matching the given category."""
        best_obj = None
        min_dist = float("inf")

        for obj in self.objects.values():
            if category and obj.category.lower() != category.lower():
                continue
            dist = obj.distance_to(query_pos)
            if dist < min_dist:
                min_dist = dist
                best_obj = obj

        return best_obj

    def get_all_objects(self) -> List[Dict[str, Any]]:
        """Return serialized list of all recognized objects."""
        return [o.to_dict() for o in self.objects.values()]
