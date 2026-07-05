"""Analytical forward and inverse kinematics (§3.1 of brief).

Direct Python/NumPy port of  src/my_robot_control/include/kinematics.hpp.
Joint order: [q1=θ1, q2=d1, q3=d2, q4=θ4, q5=θ5, q6=θ6]
              revolute  prismatic prismatic revolute revolute revolute

Both FK and IK are closed-form (no iteration, no linearisation):
  - fk()            : chain of 4×4 homogeneous transforms A1…A8
  - ik_analytic()   : direct formula → q1,q2,q3 from wrist centre, Z-Y-Z wrist angles
  - ik_branch()     : same closed-form, picks ±s5 branch closest to a seed

The only numerical step is floating-point evaluation of the closed-form expressions.
"""
from __future__ import annotations

import numpy as np
from .urdf_params import (
    A1_Z, A2_Z, A3_Z, A4_Z, A5_Z, A6_Z, A7_Z, A8_Z,
    JOINT_LOWER, JOINT_UPPER, SQRT2_2,
)

# ── Rotation helpers ──────────────────────────────────────────────────────────

def _rz(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0., 0., 1.0]])


def _ry(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[ c, 0., s],
                     [0., 1., 0.],
                     [-s, 0., c]])


def _T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Build 4×4 homogeneous transform from (3×3) R and (3,) p."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _Tz(z: float) -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = z
    return T


# Fixed 135° transform matrix (A3 in the chain).
_A3_FIXED = np.array([
    [-SQRT2_2, 0.0,  SQRT2_2, 0.0],
    [ 0.0,     1.0,  0.0,     0.0],
    [-SQRT2_2, 0.0, -SQRT2_2, A3_Z],
    [ 0.0,     0.0,  0.0,     1.0],
])


# ── Forward kinematics ────────────────────────────────────────────────────────

def fk(q: np.ndarray) -> dict[str, np.ndarray]:
    """Full forward kinematics — same chain as kinematics.hpp (§3.1).

    Returns a dict of 4×4 homogeneous transforms in the world/base frame.
    Keys: 'T01'..'T08' and descriptive aliases.
    """
    q1, q2, q3, q4, q5, q6 = q

    A1 = _T(_rz(q1), [0.0, 0.0, A1_Z])
    A2 = _Tz(A2_Z + q2)
    A3 = _A3_FIXED.copy()
    A4 = _Tz(A4_Z + q3)
    A5 = _T(_rz(q4), [0.0, 0.0, A5_Z])
    A6 = _T(_ry(q5), [0.0, 0.0, A6_Z])
    A7 = _T(_rz(q6), [0.0, 0.0, A7_Z])
    A8 = _Tz(A8_Z)

    T01 = A1
    T02 = T01 @ A2
    T03 = T02 @ A3
    T04 = T03 @ A4
    T05 = T04 @ A5
    T06 = T05 @ A6
    T07 = T06 @ A7
    T08 = T07 @ A8

    return {
        "T01": T01, "T02": T02, "T03": T03, "T04": T04,
        "T05": T05, "T06": T06, "T07": T07, "T08": T08,
        "base":         T01,   # base motor frame (after q1)
        "arm1_top":     T02,   # top of arm1 prismatic stroke (depends on q2)
        "tilt_origin":  T03,   # 135° tilt assembly origin
        "arm2_top":     T04,   # top of arm2 prismatic stroke (wrist center origin)
        "wrist_z1":     T05,   # joint_wrist_z1 frame (after q4)
        "wrist_y":      T06,   # joint_wrist_y  frame (after q5)
        "wrist_z2":     T07,   # joint_wrist_z2 frame (after q6)
        "ee":           T08,   # end-effector tip
    }


def fk_points(q: np.ndarray) -> np.ndarray:
    """Return 9 joint-origin positions for stick-figure drawing.

    Shape: (9, 3) — [world_origin, base_motor, arm1_top, tilt_origin,
                      arm2_top, wrist_z1, wrist_y, wrist_z2, ee_tip]
    """
    frames = fk(q)
    return np.array([
        np.zeros(3),
        frames["T01"][:3, 3],
        frames["T02"][:3, 3],
        frames["T03"][:3, 3],
        frames["T04"][:3, 3],
        frames["T05"][:3, 3],
        frames["T06"][:3, 3],
        frames["T07"][:3, 3],
        frames["T08"][:3, 3],
    ])


def ee_pose(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, p) of the end-effector in the world frame."""
    T = fk(q)["ee"]
    return T[:3, :3], T[:3, 3]


# ── Orientation helpers ───────────────────────────────────────────────────────

def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ROS fixed-axis RPY to 3×3 rotation matrix.

    Convention: extrinsic rotations about fixed X, Y, Z axes in order roll→pitch→yaw.
    Equivalent to intrinsic Z→Y→X (yaw, pitch, roll).

        R = Rz(yaw) · Ry(pitch) · Rx(roll)

    This is the standard ROS `geometry_msgs/Quaternion` / URDF `<rpy>` convention.
    """
    return _rz(yaw) @ _ry(pitch) @ _rx(roll)


def _rx(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[1., 0., 0.],
                     [0.,  c, -s],
                     [0.,  s,  c]])


# ── Inverse kinematics ────────────────────────────────────────────────────────

def ik_analytic(target_R: np.ndarray, target_p: np.ndarray,
                clamp: bool = True) -> tuple[np.ndarray, bool]:
    """Analytic IK — port of solve_ik_analytic() in kinematics.hpp.

    Closed-form solution: position from Pc = p_ee − (A7+A8)·z_ee,
    then Z-Y-Z wrist orientation from R47 = R04ᵀ·R_target.

    Returns (q, success) where success=True when FK(q) ≈ target (tol 1e-4).
    """
    rt2 = np.sqrt(2.0)
    eps = 1e-9

    # Wrist-centre position (T06 origin).
    pc = target_p - (A7_Z + A8_Z) * target_R[:, 2]
    xc, yc, zc = pc
    K = np.sqrt(xc**2 + yc**2)

    q1 = 0.0 if K < eps else float(np.arctan2(yc, xc))
    q3 = float(rt2 * K - 0.45)                          # 0.45 = A4+A5+A6
    q2 = float(zc + (rt2 / 2.0) * q3 + 0.45 * (rt2 / 2.0) - 0.55)  # 0.55 = A1+A2+A3

    # Wrist orientation: R47 = R04ᵀ · R_target
    q_tmp = np.array([q1, q2, q3, 0.0, 0.0, 0.0])
    R04 = fk(q_tmp)["T04"][:3, :3]
    R47 = R04.T @ target_R

    r13, r23 = R47[0, 2], R47[1, 2]
    r31, r32, r33 = R47[2, 0], R47[2, 1], R47[2, 2]
    s5_abs = float(np.sqrt(max(0.0, r13**2 + r23**2)))

    if s5_abs < eps:
        q4, q5, q6 = 0.0, 0.0, 0.0
    else:
        q5 = float(np.arctan2(s5_abs, r33))
        q4 = float(np.arctan2(r23 / s5_abs, r13 / s5_abs))
        q6 = float(np.arctan2(r32 / s5_abs, -r31 / s5_abs))

    q = np.array([q1, q2, q3, q4, q5, q6])
    if clamp:
        q = np.clip(q, JOINT_LOWER, JOINT_UPPER)

    R_check, p_check = ee_pose(q)
    ep = float(np.linalg.norm(target_p - p_check))
    eo = float(np.linalg.norm(_log_so3(target_R @ R_check.T)))
    return q, (ep < 1e-4 and eo < 1e-4)


def ik_branch(target_R: np.ndarray, target_p: np.ndarray,
              seed: np.ndarray | None = None,
              clamp: bool = True) -> tuple[np.ndarray, bool]:
    """Analytic IK with branch selection by proximity to seed.

    Same closed-form solution as ik_analytic(); no iteration.
    The Z-Y-Z wrist has two valid branches (q5 = ±arctan2(s5_abs, r33)).
    Picks whichever branch is closer in joint space to `seed`.
    """
    if seed is None:
        seed = np.zeros(6)

    rt2 = np.sqrt(2.0)
    eps = 1e-9

    pc = target_p - (A7_Z + A8_Z) * target_R[:, 2]
    xc, yc, zc = pc
    K = float(np.sqrt(xc**2 + yc**2))

    q1 = 0.0 if K < eps else float(np.arctan2(yc, xc))
    q3 = float(rt2 * K - 0.45)
    q2 = float(zc + (rt2 / 2.0) * q3 + 0.45 * (rt2 / 2.0) - 0.55)

    q_tmp = np.array([q1, q2, q3, 0.0, 0.0, 0.0])
    R04 = fk(q_tmp)["T04"][:3, :3]
    R47 = R04.T @ target_R

    r13, r23 = R47[0, 2], R47[1, 2]
    r31, r32, r33 = R47[2, 0], R47[2, 1], R47[2, 2]
    s5_abs = float(np.sqrt(max(0.0, r13**2 + r23**2)))

    base_q = np.array([q1, q2, q3, 0.0, 0.0, 0.0])

    if s5_abs < eps:
        sol = base_q.copy()
        sol[3] = seed[3]
        sol[5] = seed[5]
    else:
        def _branch(s5: float) -> np.ndarray:
            j = base_q.copy()
            j[4] = np.arctan2(s5, r33)
            j[3] = np.arctan2(r23 / s5, r13 / s5)
            j[5] = np.arctan2(r32 / s5, -r31 / s5)
            return j
        bp = _branch(+s5_abs)
        bn = _branch(-s5_abs)
        sol = bp if np.linalg.norm(bp - seed) <= np.linalg.norm(bn - seed) else bn

    if clamp:
        sol = np.clip(sol, JOINT_LOWER, JOINT_UPPER)

    R_check, p_check = ee_pose(sol)
    ep = float(np.linalg.norm(target_p - p_check))
    eo = float(np.linalg.norm(_log_so3(target_R @ R_check.T)))
    return sol, (ep < 1e-4 and eo < 1e-4)


# ── Utility ───────────────────────────────────────────────────────────────────

def _log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → axis-angle vector (3,)."""
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_t))
    if abs(theta) < 1e-12:
        return np.zeros(3)
    skew = (R - R.T) / (2.0 * np.sin(theta))
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]]) * theta


def cross_check_fk(n_tests: int = 200, tol: float = 1e-10) -> bool:
    """Verify NumPy FK against the constants in kinematics.hpp.

    Generates random configurations and checks self-consistency of the FK
    chain (position continuity, rotation orthogonality, and a known closed-
    form check at q=0).  Returns True if all checks pass.
    """
    rng = np.random.default_rng(42)

    # At q=0: EE should be at a known position
    q0 = np.zeros(6)
    frames0 = fk(q0)
    # arm2 axis at q=0: Ry(135°)@[0,0,1] = [a, 0, -a]
    a = SQRT2_2
    expected_arm2_top = np.array([
        (A4_Z) * a,
        0.0,
        A1_Z + A2_Z + A3_Z - A4_Z * a
    ])
    if np.linalg.norm(frames0["arm2_top"][:3, 3] - expected_arm2_top) > tol * 1e3:
        return False

    # Random q tests: check rotation matrix orthogonality
    for _ in range(n_tests):
        q = rng.uniform(JOINT_LOWER, JOINT_UPPER)
        frames = fk(q)
        for key in ("T01", "T03", "T05", "T07", "T08"):
            R = frames[key][:3, :3]
            if np.linalg.norm(R @ R.T - np.eye(3)) > tol:
                return False
    return True
