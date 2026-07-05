"""Physical parameters extracted directly from the robot URDF.

Source files:
  src/my_robot_description/urdf/robot_arm.urdf.xacro   (arm_length=0.3, arm_radius=0.035)
  src/my_robot_description/urdf/common_properties.xacro (inertia macros)

All values are exact (no approximations).  This module is the single
source of truth for every physical constant used in kinematics, dynamics,
integrator, and controller modules.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# ── Geometric constants (§3 of brief, matches kinematics.hpp) ────────────────
#  Transform chain A1…A8; see include/kinematics.hpp for the C++ reference.
A1_Z: float = 0.02    # joint_base_yaw height above ground  [m]
A2_Z: float = 0.15    # arm1 fixed offset (outer 0.12 + prismatic-origin 0.03)
A3_Z: float = 0.38    # vertical distance to 135° bend  (tilt 0.30 + tilt_link 0.08)
A4_Z: float = 0.11    # in tilted frame: bend to arm2 prismatic origin (0.08+0.03)
A5_Z: float = 0.30    # arm2 inner → wrist_z1 (= arm_length)
A6_Z: float = 0.04    # wrist_z1 → wrist_y
A7_Z: float = 0.05    # wrist_y  → wrist_z2
A8_Z: float = 0.11    # gripper extension (gripper_fixed 0.05 + ee_tip 0.06)

TILT_DEG: float = 135.0          # fixed tilt angle [degrees]
TILT_RAD: float = 3.0 * np.pi / 4  # 135° in radians  (= π·(3/4))
SQRT2_2:  float = np.sqrt(2.0) / 2.0   # sin(45°) = cos(45°)

G_CONST: float = 9.81   # gravitational acceleration [m/s²]

# ── Joint limits ─────────────────────────────────────────────────────────────
JOINT_NAMES = [
    "joint_base_yaw",       # q1 revolute  about Z
    "joint_arm1_prismatic", # q2 prismatic along Z
    "joint_arm2_prismatic", # q3 prismatic along tilted Z
    "joint_wrist_z1",       # q4 revolute  about tilted Z
    "joint_wrist_y",        # q5 revolute  about tilted Y
    "joint_wrist_z2",       # q6 revolute  about tilted Z
]

JOINT_LOWER = np.array([-np.pi,   0.0,   0.0,   -np.pi,    -np.pi / 2, -np.pi])
JOINT_UPPER = np.array([ np.pi,   0.255, 0.255,  np.pi,     np.pi / 2,  np.pi])
JOINT_EFFORT = np.array([120.0,   400.0, 400.0, 80.0,      80.0,       60.0])  # N·m / N
JOINT_IS_REVOLUTE = np.array([True, False, False, True, True, True])

# ── URDF raw dimensions ───────────────────────────────────────────────────────
_L = 0.3    # arm_length
_R = 0.035  # arm_radius


def _box_inertia(m: float, x: float, y: float, z: float) -> np.ndarray:
    """Diagonal inertia [Ixx,Iyy,Izz] for a solid box, axes through COM."""
    return np.array([(m / 12) * (y**2 + z**2),
                     (m / 12) * (x**2 + z**2),
                     (m / 12) * (x**2 + y**2)])


def _cyl_inertia(m: float, r: float, h: float) -> np.ndarray:
    """Diagonal inertia for a solid cylinder (axis = z)."""
    t = (m / 12) * (3 * r**2 + h**2)
    return np.array([t, t, (m / 2) * r**2])


def _cyl_tube_inertia(m: float, r_out: float, r_in: float, h: float) -> np.ndarray:
    """Diagonal inertia for a hollow cylinder (axis = z)."""
    t = (m / 12) * (3 * (r_out**2 + r_in**2) + h**2)
    return np.array([t, t, (m / 2) * (r_out**2 + r_in**2)])


# ── Per-link physical data ────────────────────────────────────────────────────
@dataclass
class LinkParams:
    """Physical properties of one link as extracted from the URDF."""
    name: str
    mass: float           # [kg]
    com: np.ndarray       # COM in the link's own frame  (3,) [m]
    inertia: np.ndarray   # diagonal inertia in link frame [Ixx,Iyy,Izz] [kg·m²]


# Link list in FK traversal order (base_link is fixed → excluded from dynamics).
LINKS: list[LinkParams] = [
    # ── arm-1 vertical chain ─────────────────────────────────────────────────
    LinkParams("base_motor_link", 1.0,
               np.array([0.0, 0.0, _L * 0.2]),              # 0.06 m
               _cyl_inertia(1.0, _R * (12 / 7), _L * 0.4)),

    LinkParams("arm1_outer_link", 1.0,
               np.array([0.0, 0.0, _L * 0.5]),              # 0.15 m
               _cyl_tube_inertia(1.0, _R, _R * (5 / 7), _L)),

    LinkParams("arm1_inner_link", 1.0,
               np.array([0.0, 0.0, _L * 0.5]),
               _cyl_tube_inertia(1.0, _R * (5 / 7), _R * (4 / 7), _L)),

    # ── 135° tilt assembly ───────────────────────────────────────────────────
    LinkParams("tilt_link", 1.0,
               np.array([0.0, 0.0, _L * 2 / 15]),           # 0.04 m
               _cyl_inertia(1.0, _R * (8 / 7), _L * 4 / 15)),

    LinkParams("tilt_135_link", 1.0,
               np.array([0.0, 0.0, _L * 2 / 15]),
               _cyl_inertia(1.0, _R * (8 / 7), _L * 4 / 15)),

    # ── arm-2 tilted chain ───────────────────────────────────────────────────
    LinkParams("arm2_outer_link", 1.0,
               np.array([0.0, 0.0, _L * 0.5]),
               _cyl_tube_inertia(1.0, _R, _R * (5 / 7), _L)),

    LinkParams("arm2_inner_link", 1.0,
               np.array([0.0, 0.0, _L * 0.5]),
               _cyl_tube_inertia(1.0, _R * (5 / 7), _R * (4 / 7), _L)),

    # ── wrist assembly ───────────────────────────────────────────────────────
    LinkParams("wrist_z1_base_link", 0.5,
               np.array([0.0, 0.0, 0.005]),
               _box_inertia(0.5, 0.05, 0.05, 0.01)),

    LinkParams("wrist_z1_left_link", 0.5,
               np.array([0.0, 0.0, 0.025]),
               _box_inertia(0.5, 0.05, 0.01, 0.05)),

    LinkParams("wrist_z1_right_link", 0.5,
               np.array([0.0, 0.0, 0.025]),
               _box_inertia(0.5, 0.05, 0.01, 0.05)),

    LinkParams("wrist_y_link", 0.5,
               np.array([0.0, 0.0, 0.020]),
               _box_inertia(0.5, 0.02, 0.03, 0.06)),

    LinkParams("wrist_z2_link", 1.0,
               np.array([0.0, 0.0, 0.025]),
               _cyl_inertia(1.0, 0.018, 0.05)),

    # ── gripper + fingers ────────────────────────────────────────────────────
    LinkParams("gripper_base_link", 0.5,
               np.array([0.0, 0.0, 0.030]),
               _box_inertia(0.5, 0.03, 0.03, 0.06)),

    LinkParams("left_finger_link", 0.3,
               np.array([0.0, 0.0, 0.030]),
               _box_inertia(0.3, 0.01, 0.01, 0.06)),

    LinkParams("right_finger_link", 0.3,
               np.array([0.0, 0.0, 0.030]),
               _box_inertia(0.3, 0.01, 0.01, 0.06)),
]

TOTAL_MASS: float = sum(lp.mass for lp in LINKS)   # ≈ 11.6 kg

# ── Origins of side-link joints (in parent link's frame) ─────────────────────
# Used when computing COM transforms for offset links.
WRIST_Z1_LEFT_ORIGIN  = np.array([0.0,  0.02,  0.01])  # in wrist_z1_base frame
WRIST_Z1_RIGHT_ORIGIN = np.array([0.0, -0.02,  0.01])
LEFT_FINGER_ORIGIN    = np.array([0.0,  0.02,  0.03])  # in gripper_base frame
RIGHT_FINGER_ORIGIN   = np.array([0.0, -0.02,  0.03])
GRIPPER_ORIGIN        = np.array([0.0,  0.0,   0.05])  # wrist_z2 → gripper_base
EE_TIP_ORIGIN         = np.array([0.0,  0.0,   0.06])  # gripper_base → ee_tip
