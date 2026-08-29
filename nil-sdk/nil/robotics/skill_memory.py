"""Embodied Skill Memory & Motor Policy Store.

Connects NIL's 7-tier persistent memory to continuous motor policies:
- DMP Skills: Parameterized movement trajectories
- Impedance Skills: Target Cartesian poses with compliance matrices Kp, Kd
- Skill Self-Evolution: Automatic tuning of compliance and trajectory profiles based on outcome rewards
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from nil.robotics.dmp import DynamicMovementPrimitive


class EmbodiedSkill:
    """A persistent, reusable physical robotic skill."""

    def __init__(
        self,
        name: str,
        category: str = "manipulation",
        dmp: Optional[DynamicMovementPrimitive] = None,
        impedance_stiffness: Optional[List[float]] = None,
        max_contact_force_n: float = 20.0,
        confidence: float = 0.80,
        success_count: int = 0,
        failure_count: int = 0
    ):
        self.name = str(name)
        self.category = str(category)
        self.dmp = dmp
        self.stiffness = list(impedance_stiffness or [200.0, 200.0, 200.0])
        self.max_force = float(max_contact_force_n)
        self.confidence = float(confidence)
        self.successes = int(success_count)
        self.failures = int(failure_count)

    def record_execution(self, success: bool) -> None:
        """Update skill confidence based on physical execution outcome."""
        if success:
            self.successes += 1
            self.confidence = min(0.99, self.confidence + 0.04)
        else:
            self.failures += 1
            self.confidence = max(0.20, self.confidence - 0.10)
            # Automatic compliance adaptation: soften stiffness if impact occurred
            self.stiffness = [max(50.0, k * 0.9) for k in self.stiffness]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "stiffness": self.stiffness,
            "max_force": self.max_force,
            "confidence": round(self.confidence, 3),
            "successes": self.successes,
            "failures": self.failures
        }


class EmbodiedSkillMemory:
    """Robotics Skill Library & Autonomous Policy Store."""

    def __init__(self):
        self.skills: Dict[str, EmbodiedSkill] = {}

    def register_skill(self, skill: EmbodiedSkill) -> None:
        """Register a new physical skill."""
        self.skills[skill.name] = skill

    def get_skill(self, name: str) -> Optional[EmbodiedSkill]:
        """Look up skill by name."""
        return self.skills.get(name)

    def get_top_skills(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve skills sorted by confidence."""
        res = [s for s in self.skills.values() if not category or s.category == category]
        res.sort(key=lambda s: s.confidence, reverse=True)
        return [s.to_dict() for s in res]
