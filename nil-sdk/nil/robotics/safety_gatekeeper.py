"""Joint position/velocity filter, not a collision or torque safety controller.

max_tau and d_safe are configuration metadata for external control systems.
They are not enforced by sanitize_joint_commands. No hardware is accessed here.
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
        limits = np.asarray(self.joint_limits, dtype=np.float64)
        if limits.ndim != 2 or limits.shape[1] != 2 or not np.isfinite(limits).all():
            raise ValueError("Joint limits must be finite (lower, upper) pairs")
        if not len(limits) or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("Joint lower limits must be below upper limits")
        if not np.isfinite([self.max_vel, self.max_tau, self.d_safe]).all() or min(
            self.max_vel, self.max_tau, self.d_safe
        ) <= 0:
            raise ValueError("Velocity, torque and distance settings must be finite and positive")

    def sanitize_joint_commands(
        self,
        q_current: np.ndarray,
        q_target: np.ndarray,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, bool, str]:
        """Verify and clamp target joint angles to obey kinematic velocity and position bounds."""
        q_curr = np.asarray(q_current, dtype=np.float64)
        q_targ = np.asarray(q_target, dtype=np.float64)
        if q_curr.ndim != 1 or q_targ.shape != q_curr.shape or len(q_curr) != len(self.joint_limits):
            raise ValueError("Commands must match the configured joint count")
        if not np.isfinite(q_curr).all() or not np.isfinite(q_targ).all():
            raise ValueError("Joint values must be finite")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        limits = np.asarray(self.joint_limits)
        if np.any(q_curr < limits[:, 0]) or np.any(q_curr > limits[:, 1]):
            raise ValueError("Current joint state is outside limits; recovery requires external control")

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
