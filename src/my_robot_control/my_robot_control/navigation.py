"""Navigation functions for collision-free motion (Lecture 10).

Implements the potential-field framework taught in the course:

    U(q) = U_att(q) + U_rep(q)

    U_att = ½·c·‖q − q_goal‖²              (attractive — pulls toward goal)
    U_rep = ½·η·(1/ρ − 1/ρ₀)²  if ρ≤ρ₀  (repulsive — pushes from obstacles)
             0                   if ρ>ρ₀

Control addition to Computed-Torque+PID (Lecture 10 / navigation functions):
    τ_nav = −∇_q U_rep(q)  = −J(q)ᵀ · ∇_p U_rep(p(q))

The attractive part is already handled by the PID (Kp·e ≡ ∇U_att).

Obstacle model (AABB, world frame) + obstacle_inflation:
  Every box is inflated by `obstacle_inflation` metres in all directions before
  distance/gradient computation.  This accounts for the physical radius of the
  arm links (~0.035 m cylinder radius) so that repulsion activates before the
  arm surface — not only its centre-line — makes contact with the shelf.

Collision proxy points:
  The arm is represented by 8 named points along the kinematic chain.  Points
  are chosen to cover every link that has physical volume near the shelf:

    arm2_top     (T04 origin)         — tip of tilted prismatic section
    wrist_z1     (T05 origin)         — wrist base; long arm above this
    wrist_y      (T06 = wrist centre) — spherical-wrist symmetry axis
    wrist_z2     (T07 origin)         — inner wrist; passes divider height
                                         during rise with q3=0
    gripper_base                      — 0.05 m beyond T07 along arm axis;
                                         passes divider height at x≈0.389 m
    left_finger                       — y+0.02, z+0.03 in gripper frame
    right_finger                      — y−0.02, z+0.03 in gripper frame
    EE tip       (T08)                — furthest point; 0.431 m from base axis
                                         when q3=0 (barely outside shelf front)

  Frame assumptions: all positions are expressed in the WORLD frame (frame 0).
  Jacobians are numerical (7 FK evaluations per column, 6 joints → 42 FK calls
  per step); they are correct for any arm configuration.

Bookshelf obstacle layout (world frame, model pose at x=0.63, y=-0.10):
  Shelf 1 floor    z=0.08–0.10   x=0.44–0.64
  Divider plank    z=0.23–0.25   x=0.44–0.64   ← CRITICAL for A→B path
  Top board        z=0.43–0.45   x=0.44–0.64
  Side panels      y=-0.41..-0.39 and y=0.19..0.21   x=0.44–0.64
  Back panel       x=0.62–0.64   z=0.00–0.45
"""
from __future__ import annotations

import numpy as np
from .kinematics import fk_points, fk
from .urdf_params import GRIPPER_ORIGIN, LEFT_FINGER_ORIGIN, RIGHT_FINGER_ORIGIN


# ── Bookshelf obstacle definitions (world frame) ──────────────────────────────

_DIVIDER_MIN = np.array([0.44, -0.40, 0.23])
_DIVIDER_MAX = np.array([0.64,  0.20, 0.25])


def critical_obstacles(inflation: float = 0.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ONLY the divider plank — the one obstacle blocking A→B.

    Args:
        inflation: expand the box by this many metres in every direction.
                   Use the arm link radius (~0.035 m) to account for physical
                   link volume; repulsion then activates before surface contact.
    """
    margin = np.full(3, inflation)
    return [(_DIVIDER_MIN - margin, _DIVIDER_MAX + margin)]


def shelf_obstacles(inflation: float = 0.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return all bookshelf AABB obstacles, each inflated by `inflation` m."""
    margin = np.full(3, inflation)
    raw = [
        # Divider plank (most critical)
        (_DIVIDER_MIN,               _DIVIDER_MAX),
        # Shelf 1 bottom board
        (np.array([0.44, -0.40, 0.08]), np.array([0.64,  0.20, 0.10])),
        # Top board
        (np.array([0.44, -0.40, 0.43]), np.array([0.64,  0.20, 0.45])),
        # Back panel
        (np.array([0.62, -0.40, 0.00]), np.array([0.64,  0.20, 0.45])),
        # Left side panel
        (np.array([0.44, -0.41, 0.00]), np.array([0.64, -0.39, 0.45])),
        # Right side panel
        (np.array([0.44,  0.19, 0.00]), np.array([0.64,  0.21, 0.45])),
    ]
    return [(lo - margin, hi + margin) for lo, hi in raw]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def dist_point_to_box(p: np.ndarray,
                      box_min: np.ndarray,
                      box_max: np.ndarray) -> float:
    """Minimum Euclidean distance from point p to AABB box.

    Returns 0 if p is inside or on the box surface.
    """
    dx = max(box_min[0] - p[0], 0.0, p[0] - box_max[0])
    dy = max(box_min[1] - p[1], 0.0, p[1] - box_max[1])
    dz = max(box_min[2] - p[2], 0.0, p[2] - box_max[2])
    return float(np.sqrt(dx*dx + dy*dy + dz*dz))


def dist_segment_to_box(p1: np.ndarray,
                        p2: np.ndarray,
                        box_min: np.ndarray,
                        box_max: np.ndarray,
                        resolution: float = 0.005) -> float:
    """Minimum distance from line segment to AABB box using adaptive sampling."""
    seg_length = float(np.linalg.norm(p2 - p1))

    if seg_length == 0:
        n_samples = 1
    else:
        n_samples = max(2, int(np.ceil(seg_length / resolution)))

    min_d = float('inf')
    for t in np.linspace(0.0, 1.0, n_samples):
        d = dist_point_to_box(p1 + t * (p2 - p1), box_min, box_max)
        if d < min_d:
            min_d = d
            if min_d == 0.0:
                break
    return min_d


def grad_dist_to_box(p: np.ndarray,
                     box_min: np.ndarray,
                     box_max: np.ndarray) -> np.ndarray:
    """Unit vector from closest box point toward p (= ∂ρ/∂p).

    Returns zero vector if p is inside the box (gradient undefined there;
    repulsion is clamped downstream so this case is handled safely).
    """
    closest = np.clip(p, box_min, box_max)
    diff    = p - closest
    dist    = float(np.linalg.norm(diff))
    if dist < 1e-9:
        return np.zeros(3)
    return diff / dist


# ── Potential field functions ─────────────────────────────────────────────────

def grad_U_rep_world(p: np.ndarray,
                     box_min: np.ndarray,
                     box_max: np.ndarray,
                     rho0: float,
                     eta: float = 1.0) -> np.ndarray:
    """Gradient of U_rep at point p w.r.t. world position (3-vector).

        ∇_p U_rep = −η·(1/ρ − 1/ρ₀)·(1/ρ²)·∇_p ρ
    """
    rho = dist_point_to_box(p, box_min, box_max)
    if rho >= rho0:
        return np.zeros(3)
    rho      = max(rho, 1e-9)
    grad_rho = grad_dist_to_box(p, box_min, box_max)
    coeff    = -eta * (1.0/rho - 1.0/rho0) / (rho**2)
    return coeff * grad_rho


# ── Collision proxy: 8 points covering all arm volumes near the shelf ─────────

def _arm_check_points(q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (world_position, 3×6_jacobian) for each collision proxy point.

    Points are ordered proximal → distal along the kinematic chain.
    All positions are in the world frame (frame 0).

    Each Jacobian J maps joint velocities → point velocity:  ṗ = J·q̇
    Used to project world-space repulsion forces to joint torques:
        τ_rep = −J^T · ∇_p U_rep(p)
    """
    frames = fk(q)
    pts: list[tuple[np.ndarray, np.ndarray]] = []

    # ── 1. arm2_top (T04 origin) ─────────────────────────────────────────────
    # Top of the tilted prismatic section (arm2_inner endpoint).
    # Stays deep inside the arm when q3=0 (x≈0.08 m), but used for coverage.
    p_arm2 = frames["arm2_top"][:3, 3]
    J_arm2 = _numeric_jacobian(q, lambda qq: fk(qq)["arm2_top"][:3, 3])
    pts.append((p_arm2, J_arm2))

    # ── 2. wrist_z1 (T05 origin) ─────────────────────────────────────────────
    # Base of the spherical wrist assembly.
    p_wz1 = frames["wrist_z1"][:3, 3]
    J_wz1 = _numeric_jacobian(q, lambda qq: fk(qq)["wrist_z1"][:3, 3])
    pts.append((p_wz1, J_wz1))

    # ── 3. wrist_y (T06 = wrist centre) ──────────────────────────────────────
    # Intersection of the spherical wrist axes; key point for orientation.
    p_wy = frames["wrist_y"][:3, 3]
    J_wy = jacobian_wrist(q)
    pts.append((p_wy, J_wy))

    # ── 4. wrist_z2 (T07 origin) ─────────────────────────────────────────────
    # Inner wrist link.  At q3=0, q2≈0.05 this point is at z≈0.246 m —
    # i.e. INSIDE the divider height band (0.23–0.25 m) even though x≈0.354 m
    # is outside the nominal shelf boundary (x=0.44 m).  With obstacle
    # inflation the proximity is detected and repulsion activates.
    p_wz2 = frames["wrist_z2"][:3, 3]
    J_wz2 = _numeric_jacobian(q, lambda qq: fk(qq)["wrist_z2"][:3, 3])
    pts.append((p_wz2, J_wz2))

    # ── 5. gripper_base ───────────────────────────────────────────────────────
    # 0.05 m beyond wrist_z2 along the arm axis (= joint_gripper_fixed offset).
    # At q3=0, q2≈0.08 m the gripper base is at z≈0.241 m (divider height)
    # and x≈0.389 m.  This is the point most likely to contact the divider in
    # Gazebo when the arm rises through the divider zone.
    R07     = frames["wrist_z2"][:3, :3]
    p07     = frames["wrist_z2"][:3, 3]
    p_grip  = p07 + R07 @ GRIPPER_ORIGIN      # GRIPPER_ORIGIN = [0,0,0.05] m
    J_grip  = _numeric_jacobian(
        q, lambda qq: fk(qq)["wrist_z2"][:3, 3] + fk(qq)["wrist_z2"][:3, :3] @ GRIPPER_ORIGIN)
    pts.append((p_grip, J_grip))

    # ── 6. left_finger ────────────────────────────────────────────────────────
    # LEFT_FINGER_ORIGIN = [0, +0.02, 0.03] in gripper frame.
    # The finger extends +y and +z_gripper from the gripper base.
    p_lfing = p_grip + R07 @ LEFT_FINGER_ORIGIN
    J_lfing = _numeric_jacobian(
        q, lambda qq: (fk(qq)["wrist_z2"][:3, 3]
                       + fk(qq)["wrist_z2"][:3, :3] @ GRIPPER_ORIGIN
                       + fk(qq)["wrist_z2"][:3, :3] @ LEFT_FINGER_ORIGIN))
    pts.append((p_lfing, J_lfing))

    # ── 7. right_finger ───────────────────────────────────────────────────────
    # RIGHT_FINGER_ORIGIN = [0, −0.02, 0.03] in gripper frame.
    p_rfing = p_grip + R07 @ RIGHT_FINGER_ORIGIN
    J_rfing = _numeric_jacobian(
        q, lambda qq: (fk(qq)["wrist_z2"][:3, 3]
                       + fk(qq)["wrist_z2"][:3, :3] @ GRIPPER_ORIGIN
                       + fk(qq)["wrist_z2"][:3, :3] @ RIGHT_FINGER_ORIGIN))
    pts.append((p_rfing, J_rfing))

    # ── 8. EE tip (T08) ───────────────────────────────────────────────────────
    # Furthest distal point.  At q3=0, x≈0.431 m (1.1 cm inside the nominal
    # shelf x boundary with 4 cm obstacle inflation).
    p_ee = fk_points(q)[-1]
    J_ee = jacobian_ee(q)
    pts.append((p_ee, J_ee))

    return pts


# ── Numeric Jacobians ─────────────────────────────────────────────────────────

def _numeric_jacobian(q: np.ndarray,
                      point_fn,
                      eps: float = 1e-6) -> np.ndarray:
    """Numeric 3×6 Jacobian for any world-space point function q → R³."""
    p0 = point_fn(q)
    J  = np.zeros((3, 6))
    for j in range(6):
        qp     = q.copy(); qp[j] += eps
        J[:, j] = (point_fn(qp) - p0) / eps
    return J


def jacobian_ee(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numeric 3×6 Jacobian of EE tip position."""
    p0 = fk_points(q)[-1]
    J  = np.zeros((3, 6))
    for j in range(6):
        qp     = q.copy(); qp[j] += eps
        J[:, j] = (fk_points(qp)[-1] - p0) / eps
    return J


def jacobian_wrist(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numeric 3×6 Jacobian of wrist-centre (T06) position."""
    p0 = fk(q)["wrist_y"][:3, 3]
    J  = np.zeros((3, 6))
    for j in range(6):
        qp     = q.copy(); qp[j] += eps
        J[:, j] = (fk(qp)["wrist_y"][:3, 3] - p0) / eps
    return J


# ── Repulsive torques ─────────────────────────────────────────────────────────

def tau_repulsive(q: np.ndarray,
                  obstacles: list[tuple[np.ndarray, np.ndarray]],
                  rho0: float = 0.025,
                  eta:  float = 5.0) -> np.ndarray:
    """Total repulsive joint torques from all obstacles.

    Checks all 8 arm proxy points (see _arm_check_points).
    Obstacles must already include any desired inflation before this call.

    Returns 6-vector τ_rep to be ADDED to the CTC control torque:
        τ = τ_ctc + τ_rep
        τ_rep = −∇_q U_rep = −J(q)^T · ∇_p U_rep(p(q))

    Clamped to ±50 N·m to prevent overflow and avoid overpowering PID.
    """
    check_pts = _arm_check_points(q)
    tau = np.zeros(6)

    for box_min, box_max in obstacles:
        for p, J in check_pts:
            g    = grad_U_rep_world(p, box_min, box_max, rho0, eta)
            tau -= J.T @ g

    # Clamp to ±50 N·m: enough to steer the arm away from the obstacle, but
    # well below the CTC controller's authority (~400 N·m max effort), so the
    # PID can still track the desired trajectory without large steady-state error.
    return np.clip(tau, -50.0, 50.0)


# ── Clearance diagnostics ─────────────────────────────────────────────────────

def min_clearance(q: np.ndarray,
                  obstacles: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Minimum distance from any arm proxy point to any obstacle.

    Uses the same 8-point set as tau_repulsive for consistency.
    Obstacles should already be inflated if inflation-based safety is needed.
    """
    check_pts = _arm_check_points(q)
    dists = [
        dist_point_to_box(p, lo, hi)
        for lo, hi in obstacles
        for p, _ in check_pts
    ]
    return float(min(dists))


def arm_capsule_clearance(q: np.ndarray,
                          obstacles: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Minimum SEGMENT distance from any arm link to any obstacle.

    Implements the correct C-obstacle model from Lecture 2:
        C_obs = {q | robot_body(q) ∩ obstacle ≠ ∅}

    The arm is modelled as a union of CAPSULES (line segments with radius r).
    The C-obstacle check for a capsule with segment S and radius r against
    obstacle O is: dist(S, O) ≤ r, i.e., dist(S, O) is the segment clearance.
    The caller subtracts arm_surface_radius from this value to get the
    capsule-surface-to-obstacle gap.

    Segments (proximal → distal, covering all arm link cylinders):
      tilt_origin→arm2_top  (T03→T04): tilt+arm2_outer section
      arm2_top→wrist_z1     (T04→T05): CRITICAL — 0.30 m tilted cylinder
      wrist_z1→wrist_y      (T05→T06): wrist rotation
      wrist_y→wrist_z2      (T06→T07): inner wrist
      wrist_z2→gripper_base (T07→grip): gripper mount
      gripper_base→EE       (grip→T08): tool segment
      gripper_base→l_finger (grip→lf):  left finger
      gripper_base→r_finger (grip→rf):  right finger

    Returns the SEGMENT minimum distance (subtract arm radius for capsule gap).
    """
    frames = fk(q)
    T03    = frames['tilt_origin'][:3, 3]
    T04    = frames['arm2_top'][:3, 3]
    T05    = frames['wrist_z1'][:3, 3]
    T06    = frames['wrist_y'][:3, 3]
    T07    = frames['wrist_z2'][:3, 3]
    R07    = frames['wrist_z2'][:3, :3]
    p_grip = T07 + R07 @ GRIPPER_ORIGIN
    p_lf   = p_grip + R07 @ LEFT_FINGER_ORIGIN
    p_rf   = p_grip + R07 @ RIGHT_FINGER_ORIGIN
    p_ee   = fk_points(q)[-1]

    segments = [
        (T03,    T04),     # tilt → arm2_top
        (T04,    T05),     # arm2_inner (CRITICAL: 0.30 m tilted link)
        (T05,    T06),     # wrist_z1
        (T06,    T07),     # wrist_y
        (T07,    p_grip),  # wrist_z2 → gripper
        (p_grip, p_ee),    # gripper → EE (tool)
        (p_grip, p_lf),    # left finger
        (p_grip, p_rf),    # right finger
    ]

    min_d = float('inf')
    for lo, hi in obstacles:
        for p1, p2 in segments:
            d = dist_segment_to_box(p1, p2, lo, hi)
            if d < min_d:
                min_d = d
            if min_d == 0.0:
                return 0.0
    return min_d


def check_collision(q: np.ndarray,
                    obstacles: list[tuple[np.ndarray, np.ndarray]],
                    margin: float = 0.005) -> bool:
    """True if any arm proxy point is within `margin` m of any obstacle."""
    return min_clearance(q, obstacles) < margin


def clearance_report(t_arr: np.ndarray,
                     q_arr: np.ndarray,
                     obstacles: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Compute per-timestep clearance and return a diagnostics dict.

    Returns:
        {
          "clearances": np.ndarray (N,),   # clearance at each time step
          "min_clearance": float,           # global minimum
          "min_time": float,                # time of minimum
          "min_idx": int,                   # index of minimum
        }
    """
    clr = np.array([arm_capsule_clearance(q_arr[k], obstacles) for k in range(len(t_arr))])
    idx = int(np.argmin(clr))
    return {
        "clearances":   clr,
        "min_clearance": float(clr[idx]),
        "min_time":      float(t_arr[idx]),
        "min_idx":       idx,
    }
