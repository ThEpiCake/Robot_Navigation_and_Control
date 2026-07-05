"""Controllers for the 6-DOF arm (§5.1 of brief).

Implements:
  ComputedTorquePID   — model-based controller (computed-torque + PID)
  quintic_trajectory  — quintic-polynomial desired trajectory
  pure_feedforward    — FF-only run (comparison baseline)
  pure_pid            — joint-space PID without model (comparison baseline)
"""
from __future__ import annotations

from typing import Callable
import numpy as np

from .urdf_params import JOINT_EFFORT


# ── Quintic polynomial trajectory generator ───────────────────────────────────

def quintic_trajectory(q0: np.ndarray, qf: np.ndarray,
                       T: float
                       ) -> tuple[Callable, Callable, Callable]:
    """Minimum-jerk quintic polynomial from q0 to qf in time T.

    Zero initial and final velocities AND accelerations.
    Returns (q_d(t), qdot_d(t), qddot_d(t)) — each returns (6,).
    """
    # Coefficients for s(t) = a3·t³ + a4·t⁴ + a5·t⁵  normalised to T
    # s(T)=1, s'(T)=0, s''(T)=0 → a3=10/T³, a4=-15/T⁴, a5=6/T⁵
    a3 = 10.0 / T**3
    a4 = -15.0 / T**4
    a5 =   6.0 / T**5
    dq = qf - q0

    def q_d(t: float) -> np.ndarray:
        t_ = float(np.clip(t, 0.0, T))
        s = a3*t_**3 + a4*t_**4 + a5*t_**5
        return q0 + s * dq

    def qdot_d(t: float) -> np.ndarray:
        t_ = float(np.clip(t, 0.0, T))
        sd = 3*a3*t_**2 + 4*a4*t_**3 + 5*a5*t_**4
        return sd * dq

    def qddot_d(t: float) -> np.ndarray:
        t_ = float(np.clip(t, 0.0, T))
        sdd = 6*a3*t_ + 12*a4*t_**2 + 20*a5*t_**3
        return sdd * dq

    return q_d, qdot_d, qddot_d


# ── Computed-Torque Control with PID (§5.1) ───────────────────────────────────

class ComputedTorquePID:
    """Model-based Computed-Torque controller with PID outer loop (§5.1).

    Control law:
        τ = M(q)·(q̈_d + Kd·ė + Kp·e + Ki·∫e) + C(q,q̇)·q̇ + G(q)

    With perfect model, error dynamics become: ë + Kd·ė + Kp·e + Ki·∫e = 0.
    Choose Kp = ωₙ², Kd = 2ζωₙ for critical damping.
    """

    def __init__(self,
                 Kp: np.ndarray, Kd: np.ndarray, Ki: np.ndarray,
                 M_func: Callable, C_func: Callable, G_func: Callable,
                 integral_limit: float = 10.0):
        """
        Args:
            Kp, Kd, Ki    : (6,) gain arrays (diagonal matrices)
            integral_limit: anti-windup clamp on |∫e| per joint
        """
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.M_func = M_func
        self.C_func = C_func
        self.G_func = G_func
        self.integral_limit = integral_limit
        self._int_e = np.zeros(6)   # integral of position error

    def reset(self):
        self._int_e = np.zeros(6)

    def compute_tau(self, t: float,
                    q: np.ndarray, qdot: np.ndarray,
                    q_d: np.ndarray, qdot_d: np.ndarray, qddot_d: np.ndarray,
                    dt: float) -> np.ndarray:
        """Compute τ for one time step.

        Args:
            q, qdot       : measured state
            q_d, qdot_d, qddot_d : desired trajectory at time t
            dt            : time step for integral update
        """
        e    = q_d   - q       # position error
        edot = qdot_d - qdot   # velocity error

        # Anti-windup: integrate only when not saturated
        self._int_e = np.clip(self._int_e + e * dt,
                               -self.integral_limit, self.integral_limit)

        # PID corrective acceleration
        v = qddot_d + self.Kd * edot + self.Kp * e + self.Ki * self._int_e

        M = self.M_func(q)
        C = self.C_func(q, qdot)
        G = self.G_func(q)

        tau = M @ v + C @ qdot + G
        # Clamp to actuator limits
        tau = np.clip(tau, -JOINT_EFFORT, JOINT_EFFORT)
        return tau

    def as_tau_func(self, q_d_func: Callable, qdot_d_func: Callable,
                    qddot_d_func: Callable, dt: float) -> Callable:
        """Wrap into a tau_func(t, q, qdot) callable for the integrator."""
        def tau_func(t, q, qdot):
            return self.compute_tau(
                t, q, qdot,
                q_d_func(t), qdot_d_func(t), qddot_d_func(t), dt)
        return tau_func


# ── Pure feedforward (comparison baseline) ────────────────────────────────────

def pure_feedforward(M_func: Callable, C_func: Callable, G_func: Callable,
                     q_d_func: Callable, qdot_d_func: Callable,
                     qddot_d_func: Callable) -> Callable:
    """Feedforward-only: τ = M(q)·q̈_d + C(q,q̇)·q̇_d + G(q).

    Uses ACTUAL q,q̇ for model but desired trajectory for derivatives.
    Demonstrates why feedback is needed when model mismatch / noise exists.
    """
    def tau_func(t, q, qdot):
        M = M_func(q)
        C = C_func(q, qdot)
        G = G_func(q)
        qdot_d = qdot_d_func(t)
        qddot_d = qddot_d_func(t)
        return M @ qddot_d + C @ qdot_d + G
    return tau_func


# ── Pure joint-space PID (without model — comparison) ─────────────────────────

class PurePID:
    """Joint-space PID WITHOUT model: τ = Kp·e + Ki·∫e + Kd·ė  (comparison)."""

    def __init__(self, Kp: np.ndarray, Kd: np.ndarray, Ki: np.ndarray,
                 integral_limit: float = 10.0):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.integral_limit = integral_limit
        self._int_e = np.zeros(6)

    def reset(self):
        self._int_e = np.zeros(6)

    def compute_tau(self, t, q, qdot, q_d, qdot_d, dt):
        e    = q_d   - q
        edot = qdot_d - qdot
        self._int_e = np.clip(self._int_e + e * dt,
                               -self.integral_limit, self.integral_limit)
        tau = self.Kp * e + self.Ki * self._int_e + self.Kd * edot
        return np.clip(tau, -JOINT_EFFORT, JOINT_EFFORT)

    def as_tau_func(self, q_d_func, qdot_d_func, dt):
        def tau_func(t, q, qdot):
            return self.compute_tau(t, q, qdot, q_d_func(t), qdot_d_func(t), dt)
        return tau_func


# ── Zero torque ───────────────────────────────────────────────────────────────

def zero_torque(t: float, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
    """τ = 0 — free response / gravity-only simulation."""
    return np.zeros(6)
