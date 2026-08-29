"""Dynamic Movement Primitives (DMPs) for Continuous Motor Skills."""

from __future__ import annotations
from typing import List, Optional, Tuple, Union
import math
import numpy as np


class DynamicMovementPrimitive:
    """Multi-DOF Dynamic Movement Primitive (DMP)."""

    def __init__(
        self,
        num_dof: int = 3,
        num_basis: int = 25,
        K: float = 150.0,
        D: Optional[float] = None,
        alpha_s: float = 4.0
    ):
        self.dim = int(num_dof)
        self.num_basis = int(num_basis)
        self.K = float(K)
        self.D = float(D) if D is not None else 2.0 * math.sqrt(self.K)
        self.alpha_s = float(alpha_s)

        self.c = np.exp(-self.alpha_s * np.linspace(0, 1, self.num_basis))
        self.h = np.ones(self.num_basis) * (1.0 / (np.diff(self.c)[0] ** 2 if len(self.c) > 1 else 1.0))
        self.w = np.zeros((self.dim, self.num_basis), dtype=np.float64)

    def fit_demonstration(self, trajectory: np.ndarray, tau: float = 1.0) -> None:
        """Learn nonlinear forcing weights w from a demonstrated trajectory (N x dim)."""
        traj = np.asarray(trajectory, dtype=np.float64)
        N = len(traj)
        if N < 2:
            return

        x0 = traj[0]
        g = traj[-1]
        dt = float(tau) / (N - 1)
        t = np.linspace(0, 1, N)
        s_vec = np.exp(-self.alpha_s * t)

        v = np.gradient(traj, dt, axis=0)
        v_dot = np.gradient(v, dt, axis=0)

        for d in range(self.dim):
            denom = (g[d] - x0[d]) if abs(g[d] - x0[d]) > 1e-4 else 1.0
            f_target = (v_dot[:, d] - self.K * (g[d] - traj[:, d]) + self.D * v[:, d]) / denom

            Psi = np.zeros((N, self.num_basis))
            for i, s in enumerate(s_vec):
                psi_i = np.exp(-self.h * ((s - self.c) ** 2))
                Psi[i] = (psi_i * s) / (np.sum(psi_i) + 1e-12)

            self.w[d] = np.linalg.pinv(Psi) @ f_target

    def generate_rollout(
        self,
        x0: np.ndarray,
        g: np.ndarray,
        tau: float = 1.0,
        dt: float = 0.01
    ) -> np.ndarray:
        """Roll out dynamic movement primitive to target goal g from x0."""
        x0 = np.asarray(x0, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)
        num_steps = max(2, int(tau / dt))

        trajectory = np.zeros((num_steps, self.dim), dtype=np.float64)
        x = x0.copy()
        v = np.zeros(self.dim, dtype=np.float64)
        s = 1.0

        for step in range(num_steps):
            trajectory[step] = x

            psi = np.exp(-self.h * ((s - self.c) ** 2))
            psi_sum = np.sum(psi) + 1e-12

            for d in range(self.dim):
                f = (np.dot(psi, self.w[d]) * s) / psi_sum
                scaling = (g[d] - x0[d])
                v_dot = (self.K * (g[d] - x[d]) - self.D * v[d] + scaling * f) / tau
                v[d] += v_dot * dt
                x[d] += (v[d] / tau) * dt

            s += (-self.alpha_s * s / tau) * dt

        return trajectory
