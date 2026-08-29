"""Physical Safety Gatekeeper & Real-Time Control Barrier Filter.

Enforces physical robot safety invariants before hardware/actuator execution:
- Position & Joint Velocity Limits: q_min <= q <= q_max, |q_dot| <= q_dot_max
- Torque Limits: |tau| <= tau_max
- Control Barrier Distance Constraints: Distance to obstacles >= d_safe
- Self-Correction from Physical Failures: Attributes root causes to memory
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class PhysicalSafetyGatekeeper:
    """Safety Supervisor & Control Filter for Embodied AI Agents."""

    def __init__(
        self,
        joint_limits_rad: Optional[List[Tuple[float, float]]] = None,
        max_joint_velocities_rad_s: float = 3.14,
        max_torque_nm: float = 50.0,
        min_obstacle_distance_m: float = 0.05
    ):
        self.joint_limits = list(joint_limits_rad or [(-3.14, 3.14)] * 6)
        self.max_vel = float(max_joint_velocities_rad_s)
        self.max_tau = float(max_torque_nm)
        self.d_safe = float(min_obstacle_distance_m)

    def sanitize_joint_commands(
        self,
        q_current: np.ndarray,
        q_target: np.ndarray,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, bool, str]:
        """Verify and clamp target joint angles to obey kinematic velocity and position bounds."""
        q_curr = np.asarray(q_current, dtype=np.float64)
        q_targ = np.asarray(q_target, dtype=np.float64)

        # 1. Clamp to position limits
        q_safe = q_targ.copy()
        is_modified = False
        reason = "Safe"

        for i, (q_min, q_max) in enumerate(self.joint_limits[:len(q_targ)]):
            if q_safe[i] < q_min:
                q_safe[i] = q_min
                is_modified = True
                reason = f"Joint {i+1} clamped to lower limit {q_min:.2f} rad"
            elif q_safe[i] > q_max:
                q_safe[i] = q_max
                is_modified = True
                reason = f"Joint {i+1} clamped to upper limit {q_max:.2f} rad"

        # 2. Limit velocity step
        delta_q = q_safe - q_curr
        max_step = self.max_vel * float(dt)
        step_magnitudes = np.abs(delta_q)

        if np.any(step_magnitudes > max_step):
            scale = max_step / np.max(step_magnitudes)
            q_safe = q_curr + delta_q * scale
            is_modified = True
            reason = "Velocity rate-limited to avoid dynamic over-acceleration"

        return q_safe, is_modified, reason
