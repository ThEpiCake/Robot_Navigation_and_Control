"""Numerical integrator for the 6-DOF arm dynamics (§4 of brief).

Implements a fixed-step RK4 integrator written from scratch,
plus scipy RK45 as a cross-check (§4).

State vector: x = [q (6,), q̇ (6,)]  — 12-dimensional.
"""
from __future__ import annotations

from typing import Callable
import numpy as np

from .urdf_params import JOINT_LOWER, JOINT_UPPER, JOINT_IS_REVOLUTE, G_CONST


# ── EOM right-hand side ───────────────────────────────────────────────────────

def eom_rhs(q: np.ndarray, qdot: np.ndarray,
            tau: np.ndarray,
            M_func: Callable, C_func: Callable, G_func: Callable
            ) -> np.ndarray:
    """State-space RHS: q̈ = M⁻¹(τ − C q̇ − G)  (§4).

    Returns q̈ as a (6,) array.
    """
    M = M_func(q)
    C = C_func(q, qdot)
    G = G_func(q)
    rhs = tau - C @ qdot - G
    return np.linalg.solve(M, rhs)


# ── Fixed-step RK4 (§4 — "implement it ourselves") ───────────────────────────

def rk4_step(q: np.ndarray, qdot: np.ndarray, tau: np.ndarray,
             dt: float,
             M_func: Callable, C_func: Callable, G_func: Callable
             ) -> tuple[np.ndarray, np.ndarray]:
    """Single RK4 step.  Returns (q_new, qdot_new)."""
    def f(q_, qd_):
        return qd_, eom_rhs(q_, qd_, tau, M_func, C_func, G_func)

    k1q, k1qd = f(q,               qdot)
    k2q, k2qd = f(q + dt/2 * k1q,  qdot + dt/2 * k1qd)
    k3q, k3qd = f(q + dt/2 * k2q,  qdot + dt/2 * k2qd)
    k4q, k4qd = f(q + dt   * k3q,  qdot + dt   * k3qd)

    q_new    = q    + (dt / 6) * (k1q  + 2*k2q  + 2*k3q  + k4q)
    qdot_new = qdot + (dt / 6) * (k1qd + 2*k2qd + 2*k3qd + k4qd)
    return q_new, qdot_new


def _clamp_joints(q: np.ndarray, qdot: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Enforce prismatic joint limits; zero velocity at the boundary."""
    q_c = q.copy()
    qd_c = qdot.copy()
    for i in range(6):
        if not JOINT_IS_REVOLUTE[i]:
            if q_c[i] < JOINT_LOWER[i]:
                q_c[i] = JOINT_LOWER[i]
                if qd_c[i] < 0:
                    qd_c[i] = 0.0
            elif q_c[i] > JOINT_UPPER[i]:
                q_c[i] = JOINT_UPPER[i]
                if qd_c[i] > 0:
                    qd_c[i] = 0.0
    return q_c, qd_c


def simulate(q0: np.ndarray, qdot0: np.ndarray,
             tau_func: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
             t_span: tuple[float, float],
             dt: float,
             M_func: Callable, C_func: Callable, G_func: Callable,
             enforce_limits: bool = True,
             ) -> dict:
    """Full RK4 simulation from t_span[0] to t_span[1].

    Args:
        q0, qdot0       : initial conditions (6,)
        tau_func(t,q,qd): callable returning (6,) torque/force
        t_span          : (t0, tf)
        dt              : fixed step size [s]
        enforce_limits  : clamp prismatic joints at URDF limits

    Returns dict with keys:
        't'     : (N,) time array
        'q'     : (N,6) joint positions
        'qdot'  : (N,6) joint velocities
        'tau'   : (N,6) applied torques
    """
    t0, tf = t_span
    steps = int(np.ceil((tf - t0) / dt))
    t_arr    = np.zeros(steps + 1)
    q_arr    = np.zeros((steps + 1, 6))
    qdot_arr = np.zeros((steps + 1, 6))
    tau_arr  = np.zeros((steps + 1, 6))

    q, qdot = q0.copy(), qdot0.copy()
    t = t0
    t_arr[0]    = t
    q_arr[0]    = q
    qdot_arr[0] = qdot

    for k in range(steps):
        tau = tau_func(t, q, qdot)
        tau_arr[k] = tau
        q, qdot = rk4_step(q, qdot, tau, dt, M_func, C_func, G_func)
        if enforce_limits:
            q, qdot = _clamp_joints(q, qdot)
        t += dt
        t_arr[k + 1]    = t
        q_arr[k + 1]    = q
        qdot_arr[k + 1] = qdot

    tau_arr[-1] = tau_func(t, q, qdot)
    return {"t": t_arr, "q": q_arr, "qdot": qdot_arr, "tau": tau_arr}


def simulate_scipy(q0: np.ndarray, qdot0: np.ndarray,
                   tau_func: Callable,
                   t_span: tuple[float, float],
                   dt_eval: float,
                   M_func: Callable, C_func: Callable, G_func: Callable,
                   ) -> dict:
    """Cross-check simulation using scipy RK45  (§4).

    Same interface as simulate() for easy comparison.
    """
    from scipy.integrate import solve_ivp

    def f_ivp(t, x):
        q_   = x[:6]
        qd_  = x[6:]
        tau  = tau_func(t, q_, qd_)
        qdd_ = eom_rhs(q_, qd_, tau, M_func, C_func, G_func)
        return np.concatenate([qd_, qdd_])

    x0 = np.concatenate([q0, qdot0])
    t_eval = np.arange(t_span[0], t_span[1] + dt_eval, dt_eval)
    sol = solve_ivp(f_ivp, t_span, x0, method="RK45", t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)

    return {
        "t":    sol.t,
        "q":    sol.y[:6].T,
        "qdot": sol.y[6:].T,
        "tau":  np.array([tau_func(t, sol.y[:6, k], sol.y[6:, k])
                          for k, t in enumerate(sol.t)]),
    }


# ── Energy computation ────────────────────────────────────────────────────────

def kinetic_energy(q: np.ndarray, qdot: np.ndarray, M_func: Callable) -> float:
    """T = ½ q̇ᵀ M(q) q̇"""
    M = M_func(q)
    return float(0.5 * qdot @ M @ qdot)


def total_energy(q: np.ndarray, qdot: np.ndarray,
                 M_func: Callable, V_func: Callable) -> float:
    """E = T + V — should be ≈ constant when τ = 0 and no friction."""
    return kinetic_energy(q, qdot, M_func) + V_func(q)


def compute_energy_history(result: dict,
                            M_func: Callable, V_func: Callable) -> np.ndarray:
    """Return (N,) array of total energy E(t) = T(t) + V(t)."""
    N = len(result["t"])
    E = np.zeros(N)
    for k in range(N):
        E[k] = total_energy(result["q"][k], result["qdot"][k], M_func, V_func)
    return E
