"""Lagrangian dynamics for the 6-DOF R-P-P-R-R-R arm (§3 of brief).

Implements Euler–Lagrange in three stages:
  1. _derive_symbolic()  — builds M(q), C(q,q̇), G(q), V(q) symbolically with SymPy.
  2. _lambdify_all()     — converts them to fast NumPy callables via lambdify+CSE.
  3. load_dynamics()     — loads from cache (.dynamics_cache.pkl) or re-derives.

Equations of motion:  M(q) q̈ + C(q,q̇) q̇ + G(q) = τ   (§3.6)

Run this file directly to force re-derivation and export derivation.md:
    python -m my_robot_control.dynamics
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np

_CACHE_FILE = Path(__file__).resolve().parent / ".dynamics_cache.pkl"
_DERIV_MD   = Path.cwd() / "derivation.md"   # written next to pid_trajectory.csv

# ── Public interface ──────────────────────────────────────────────────────────

def load_dynamics(force_rederive: bool = False,
                  verbose: bool = True
                  ) -> tuple[Callable, Callable, Callable, Callable]:
    """Return (M_func, C_func, G_func, V_func) as NumPy callables.

    Signatures:
      M_func(q)         → (6,6) mass matrix
      C_func(q, qdot)   → (6,6) Coriolis/centrifugal matrix
      G_func(q)         → (6,)  gravity vector
      V_func(q)         → float potential energy
    """
    if not force_rederive and _CACHE_FILE.exists():
        if verbose:
            print("[dynamics] Loading cached symbolic expressions …")
        with open(_CACHE_FILE, "rb") as fh:
            M_sym, C_sym, G_sym, V_sym, q_syms, qd_syms = pickle.load(fh)
        if verbose:
            print("[dynamics] Lambdifying …")
        return _lambdify_all(M_sym, C_sym, G_sym, V_sym, q_syms, qd_syms)

    if verbose:
        print("[dynamics] Deriving M, C, G symbolically with SymPy — please wait …")
    t0 = time.time()
    M_sym, C_sym, G_sym, V_sym, q_syms, qd_syms = _derive_symbolic(verbose=verbose)
    if verbose:
        print(f"[dynamics] Derivation done in {time.time()-t0:.1f} s. Caching …")

    with open(_CACHE_FILE, "wb") as fh:
        pickle.dump((M_sym, C_sym, G_sym, V_sym, q_syms, qd_syms), fh,
                    protocol=pickle.HIGHEST_PROTOCOL)

    _export_derivation_md(M_sym, G_sym, V_sym, q_syms, qd_syms)

    return _lambdify_all(M_sym, C_sym, G_sym, V_sym, q_syms, qd_syms)


# ── Symbolic derivation ───────────────────────────────────────────────────────

def _derive_symbolic(verbose: bool = True):
    """Euler–Lagrange derivation (§3.1 – §3.5 of brief).

    Returns (M_sym, C_sym, G_sym, V_sym, q_syms, qd_syms).
    All are SymPy objects.
    """
    import sympy as sp
    from .urdf_params import (
        A1_Z, A2_Z, A3_Z, A4_Z, A5_Z, A6_Z, A7_Z, A8_Z,
        G_CONST, LINKS,
        WRIST_Z1_LEFT_ORIGIN, WRIST_Z1_RIGHT_ORIGIN,
        LEFT_FINGER_ORIGIN, RIGHT_FINGER_ORIGIN,
        GRIPPER_ORIGIN,
    )

    # ── Convert physical constants to exact SymPy Rationals ──────────────────
    # Eliminates float noise (1e-20 residuals) so trigsimp can collapse
    # sin²+cos²=1 identities cleanly, yielding compact M(q) expressions.
    def _r(x):
        return sp.nsimplify(float(x), rational=True, tolerance=1e-9)
    def _r3(arr):
        return [_r(x) for x in arr]

    A1_Z = _r(A1_Z); A2_Z = _r(A2_Z); A3_Z = _r(A3_Z)
    A4_Z = _r(A4_Z); A5_Z = _r(A5_Z); A6_Z = _r(A6_Z)
    A7_Z = _r(A7_Z); G_CONST = _r(G_CONST)
    WRIST_Z1_LEFT_ORIGIN  = _r3(WRIST_Z1_LEFT_ORIGIN)
    WRIST_Z1_RIGHT_ORIGIN = _r3(WRIST_Z1_RIGHT_ORIGIN)
    LEFT_FINGER_ORIGIN    = _r3(LEFT_FINGER_ORIGIN)
    RIGHT_FINGER_ORIGIN   = _r3(RIGHT_FINGER_ORIGIN)
    GRIPPER_ORIGIN        = _r3(GRIPPER_ORIGIN)

    # ── Symbolic variables ────────────────────────────────────────────────────
    q1, q2, q3, q4, q5, q6 = sp.symbols("q1 q2 q3 q4 q5 q6", real=True)
    qd1, qd2, qd3, qd4, qd5, qd6 = sp.symbols(
        "qd1 qd2 qd3 qd4 qd5 qd6", real=True)
    q_syms  = sp.Matrix([q1, q2, q3, q4, q5, q6])
    qd_syms = sp.Matrix([qd1, qd2, qd3, qd4, qd5, qd6])

    a = sp.sqrt(2) / 2   # sin(45°) = cos(45°) = √2/2

    # ── Symbolic transform helpers ────────────────────────────────────────────
    def Rz_s(th):
        c, s = sp.cos(th), sp.sin(th)
        return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def Ry_s(th):
        c, s = sp.cos(th), sp.sin(th)
        return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def Tz_s(z):
        """4×4 pure-Z translation."""
        T = sp.eye(4)
        T[2, 3] = z
        return T

    def T_s(R, p):
        """4×4 from (3×3) R and (3,1) p."""
        T = sp.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    # ── Build FK transform chain (matches kinematics.hpp / §3.1) ─────────────
    A1 = T_s(Rz_s(q1), sp.Matrix([0, 0, A1_Z]))
    A2 = Tz_s(A2_Z + q2)
    A3 = sp.Matrix([[-a, 0,  a, 0],
                    [ 0, 1,  0, 0],
                    [-a, 0, -a, A3_Z],
                    [ 0, 0,  0, 1]])
    A4 = Tz_s(A4_Z + q3)
    A5 = T_s(Rz_s(q4), sp.Matrix([0, 0, A5_Z]))
    A6 = T_s(Ry_s(q5), sp.Matrix([0, 0, A6_Z]))
    A7 = T_s(Rz_s(q6), sp.Matrix([0, 0, A7_Z]))

    T01 = A1
    T02 = T01 * A2
    T03 = T02 * A3   # after 135° tilt
    T04 = T03 * A4   # after q3 prismatic  (arm2 inner top)
    T05 = T04 * A5   # after q4 revolute   (wrist_z1)
    T06 = T05 * A6   # after q5 revolute   (wrist_y)
    T07 = T06 * A7   # after q6 revolute   (wrist_z2)

    # Rotation matrices of key frames
    R01 = T01[:3, :3]   # = Rz(q1)
    R03 = T03[:3, :3]   # = Rz(q1)·Ry(135°)
    R05 = T05[:3, :3]   # = Rz(q1)·Ry(135°)·Rz(q4)
    R06 = T06[:3, :3]   # + Ry(q5)
    R07 = T07[:3, :3]   # + Rz(q6)

    # ── Joint axes in the world frame (for angular Jacobians §3.2) ────────────
    z0 = sp.Matrix([0, 0, 1])                         # q1 axis (always world-Z)
    z4 = R03 * z0                                      # q4 axis = z-col of R03
    y5 = R05 * sp.Matrix([0, 1, 0])                   # q5 axis = y-col of R05
    z6 = R06 * z0                                      # q6 axis = z-col of R06

    # ── Helper: build (p_com, R_link, I_body_diag) for each link ─────────────
    def p_col(T4x4):
        return T4x4[:3, 3]

    def R_col(T4x4):
        return T4x4[:3, :3]

    def _add_com(T_link_origin, com_in_link, R_link):
        """COM position in world = T_link_origin[:3,3] + R_link * com."""
        return p_col(T_link_origin) + R_link * sp.Matrix(com_in_link.tolist())

    # Additional fixed transforms needed for side/distal links
    T_arm1_outer_origin = T01 * Tz_s(sp.Rational(3, 25))    # 0.12 m
    T_arm1_inner_origin = T01 * Tz_s(A2_Z + q2)             # arm1_inner origin  (= T02)
    T_tilt_origin       = T01 * Tz_s(A2_Z + q2 + sp.Rational(3, 10))  # +0.30 m
    T_135_origin        = T03                                # tilt_135_link origin (= T03)
    T_arm2_outer_origin = T03 * Tz_s(sp.Rational(2, 25))    # 0.08 m
    T_arm2_inner_origin = T04                                # arm2_inner origin (= T04)
    T_wz1_origin        = T05                                # wrist_z1_base origin (= T05)
    T_wy_origin         = T06                                # wrist_y origin  (= T06)
    T_wz2_origin        = T07                                # wrist_z2 origin (= T07)
    # gripper_base is at z=0.05 past wrist_z2
    T_grip_origin       = T07 * Tz_s(sp.Rational(1, 20))    # 0.05 m

    # wrist side links: origin = T_wz1 * Trans(offset)
    def _Toff(T_parent, offset):
        """Translate in parent frame by 3-vector offset."""
        T = sp.eye(4)
        T[:3, 3] = sp.Matrix(list(offset))
        return T_parent * T

    T_wz1l_origin = _Toff(T_wz1_origin, WRIST_Z1_LEFT_ORIGIN)
    T_wz1r_origin = _Toff(T_wz1_origin, WRIST_Z1_RIGHT_ORIGIN)
    T_lf_origin   = _Toff(T_grip_origin, LEFT_FINGER_ORIGIN)
    T_rf_origin   = _Toff(T_grip_origin, RIGHT_FINGER_ORIGIN)

    # link_table: (T_link_origin, R_link, com_in_link, mass, I_diag, name)
    lp = {lk.name: lk for lk in LINKS}
    link_table = [
        ("base_motor_link",    T01,               R01, lp["base_motor_link"].com,    lp["base_motor_link"].mass,    lp["base_motor_link"].inertia),
        ("arm1_outer_link",    T_arm1_outer_origin, R01, lp["arm1_outer_link"].com,  lp["arm1_outer_link"].mass,    lp["arm1_outer_link"].inertia),
        ("arm1_inner_link",    T_arm1_inner_origin, R01, lp["arm1_inner_link"].com,  lp["arm1_inner_link"].mass,    lp["arm1_inner_link"].inertia),
        ("tilt_link",          T_tilt_origin,       R01, lp["tilt_link"].com,        lp["tilt_link"].mass,          lp["tilt_link"].inertia),
        ("tilt_135_link",      T_135_origin,        R03, lp["tilt_135_link"].com,    lp["tilt_135_link"].mass,      lp["tilt_135_link"].inertia),
        ("arm2_outer_link",    T_arm2_outer_origin, R03, lp["arm2_outer_link"].com,  lp["arm2_outer_link"].mass,    lp["arm2_outer_link"].inertia),
        ("arm2_inner_link",    T_arm2_inner_origin, R03, lp["arm2_inner_link"].com,  lp["arm2_inner_link"].mass,    lp["arm2_inner_link"].inertia),
        ("wrist_z1_base_link", T_wz1_origin,        R05, lp["wrist_z1_base_link"].com, lp["wrist_z1_base_link"].mass, lp["wrist_z1_base_link"].inertia),
        ("wrist_z1_left_link", T_wz1l_origin,       R05, lp["wrist_z1_left_link"].com, lp["wrist_z1_left_link"].mass, lp["wrist_z1_left_link"].inertia),
        ("wrist_z1_right_link",T_wz1r_origin,       R05, lp["wrist_z1_right_link"].com,lp["wrist_z1_right_link"].mass,lp["wrist_z1_right_link"].inertia),
        ("wrist_y_link",       T_wy_origin,         R06, lp["wrist_y_link"].com,     lp["wrist_y_link"].mass,       lp["wrist_y_link"].inertia),
        ("wrist_z2_link",      T_wz2_origin,        R07, lp["wrist_z2_link"].com,    lp["wrist_z2_link"].mass,      lp["wrist_z2_link"].inertia),
        ("gripper_base_link",  T_grip_origin,       R07, lp["gripper_base_link"].com,lp["gripper_base_link"].mass,  lp["gripper_base_link"].inertia),
        ("left_finger_link",   T_lf_origin,         R07, lp["left_finger_link"].com, lp["left_finger_link"].mass,   lp["left_finger_link"].inertia),
        ("right_finger_link",  T_rf_origin,         R07, lp["right_finger_link"].com,lp["right_finger_link"].mass,  lp["right_finger_link"].inertia),
    ]

    # ── Upstream revolute joints for each link (for Jw) ───────────────────────
    # True = revolute joint k is upstream of this link
    # Columns: [q1_up, q2_up(prismatic→0), q3_up(prismatic→0), q4_up, q5_up, q6_up]
    _upstream = {
        "base_motor_link":     [True,  False, False, False, False, False],
        "arm1_outer_link":     [True,  False, False, False, False, False],
        "arm1_inner_link":     [True,  False, False, False, False, False],
        "tilt_link":           [True,  False, False, False, False, False],
        "tilt_135_link":       [True,  False, False, False, False, False],
        "arm2_outer_link":     [True,  False, False, False, False, False],
        "arm2_inner_link":     [True,  False, False, False, False, False],
        "wrist_z1_base_link":  [True,  False, False, True,  False, False],
        "wrist_z1_left_link":  [True,  False, False, True,  False, False],
        "wrist_z1_right_link": [True,  False, False, True,  False, False],
        "wrist_y_link":        [True,  False, False, True,  True,  False],
        "wrist_z2_link":       [True,  False, False, True,  True,  True ],
        "gripper_base_link":   [True,  False, False, True,  True,  True ],
        "left_finger_link":    [True,  False, False, True,  True,  True ],
        "right_finger_link":   [True,  False, False, True,  True,  True ],
    }
    _axes = [z0, None, None, z4, y5, z6]   # None = prismatic (zero contribution)

    # ── Build M(q) and V(q) ───────────────────────────────────────────────────
    M = sp.zeros(6, 6)
    V_expr = sp.Integer(0)

    n_links = len(link_table)
    for idx, (name, T_orig, R_lnk, com_np, mass, I_np_diag) in enumerate(link_table):
        if verbose:
            print(f"  [{idx+1}/{n_links}] {name} …", flush=True)

        # Convert link data to exact Rationals (eliminates float noise in M)
        mass_r   = _r(mass)
        com_r    = _r3(com_np)
        I_r_diag = _r3(I_np_diag)

        # COM position in world frame (§3.1)
        p_c = p_col(T_orig) + R_lnk * sp.Matrix(com_r)

        # Linear COM Jacobian Jv (3×6), §3.2
        Jv = p_c.jacobian(q_syms)

        # Angular Jacobian Jw (3×6), §3.2
        Jw = sp.zeros(3, 6)
        up = _upstream[name]
        for k, (is_up, ax) in enumerate(zip(up, _axes)):
            if is_up and ax is not None:
                Jw[:, k] = ax

        # Inertia tensor rotated into world frame: R I_body R^T  (§3.3)
        I_body = sp.diag(*I_r_diag)
        I_world = R_lnk * I_body * R_lnk.T

        # Mass matrix contribution: m·Jv^T·Jv + Jw^T·I_world·Jw  (§3.3)
        M += mass_r * Jv.T * Jv + Jw.T * I_world * Jw

        # Potential energy contribution: m·g·z_c  (§3.4)
        V_expr += mass_r * G_CONST * p_c[2]

    # ── Gravity vector G(q) = ∂V/∂q  (§3.4) ─────────────────────────────────
    if verbose:
        print("  Computing G(q) = dV/dq …", flush=True)
    G_sym = sp.Matrix([sp.diff(V_expr, qi) for qi in q_syms])

    # ── Coriolis/centrifugal C(q,q̇) via Christoffel symbols  (§3.5) ──────────
    if verbose:
        print("  Computing C(q,q̇) via Christoffel symbols …", flush=True)
    n = 6
    C_sym = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            cij = sp.Integer(0)
            for k in range(n):
                cijk = sp.Rational(1, 2) * (
                    sp.diff(M[i, j], q_syms[k]) +
                    sp.diff(M[i, k], q_syms[j]) -
                    sp.diff(M[j, k], q_syms[i])
                )
                cij += cijk * qd_syms[k]
            C_sym[i, j] = cij

    V_sym = V_expr
    return M, C_sym, G_sym, V_sym, q_syms, qd_syms


# ── Lambdification ────────────────────────────────────────────────────────────

def _lambdify_all(M_sym, C_sym, G_sym, V_sym, q_syms, qd_syms):
    import sympy as sp

    args_q  = list(q_syms)
    args_qd = list(qd_syms)

    # Use common-subexpression elimination for speed
    M_func  = sp.lambdify(args_q,         M_sym,  modules="numpy", cse=True)
    G_func  = sp.lambdify(args_q,         G_sym,  modules="numpy", cse=True)
    V_func  = sp.lambdify(args_q,         V_sym,  modules="numpy", cse=True)
    C_func  = sp.lambdify(args_q + args_qd, C_sym, modules="numpy", cse=True)

    def M_wrap(q):
        return np.asarray(M_func(*q), dtype=float)

    def C_wrap(q, qdot):
        return np.asarray(C_func(*q, *qdot), dtype=float)

    def G_wrap(q):
        return np.asarray(G_func(*q), dtype=float).ravel()

    def V_wrap(q):
        return float(V_func(*q))

    return M_wrap, C_wrap, G_wrap, V_wrap


# ── derivation.md export ──────────────────────────────────────────────────────

def _export_derivation_md(M_sym, G_sym, V_sym, q_syms, qd_syms):
    """Write symbolic EOM to derivation.md for the written report (§5 item 3)."""
    import sympy as sp

    lines = [
        "# Derived Equations of Motion — 6-DOF R-P-P-R-R-R Arm",
        "",
        "Derived with SymPy using Euler–Lagrange (§3 of brief).",
        "Joint order: **[q1=θ1, q2=d1, q3=d2, q4=θ4, q5=θ5, q6=θ6]**",
        "",
        "## Potential Energy  V(q)",
        "",
        "```",
        f"V = {sp.simplify(V_sym)}",
        "```",
        "",
        "## Gravity Vector  G(q) = ∂V/∂q",
        "",
        "```",
    ]
    for i, gi in enumerate(G_sym):
        lines.append(f"G[{i+1}] = {gi}")
    lines += [
        "```",
        "",
        "## Mass Matrix  M(q)  (6×6 symmetric, positive-definite)",
        "",
        "```",
    ]
    for i in range(6):
        for j in range(i, 6):
            entry = sp.trigsimp(M_sym[i, j])
            if entry != 0:
                lines.append(f"M[{i+1},{j+1}] = M[{j+1},{i+1}] = {entry}")
    lines += [
        "```",
        "",
        "## Equations of Motion",
        "",
        "    M(q) q̈ + C(q,q̇) q̇ + G(q) = τ",
        "",
        "Substituting τ = M(q)·(q̈_d + Kd·ė + Kp·e + Ki·∫e) + C(q,q̇)·q̇ + G(q)",
        "yields decoupled linear error dynamics: ë + Kd·ė + Kp·e + Ki·∫e = 0 (§5.1)",
        "",
        "*(Full C(q,q̇) not printed here — see dynamics.py:_derive_symbolic for detail)*",
        "",
        "## Total Mechanical Energy  E = T + V",
        "",
        "    T = ½ q̇ᵀ M(q) q̇",
        "    V = Σᵢ mᵢ · g · zᵢ(q)",
        "    E(t) ≈ const  when τ = 0  (energy-conservation check for §9.3)",
    ]
    _DERIV_MD.write_text("\n".join(lines))
    print(f"[dynamics] derivation.md written → {_DERIV_MD}")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_dynamics(force_rederive=True, verbose=True)
    print("[dynamics] Done. derivation.md and cache written.")
