"""Computed-Torque + PID with C-space planning + Navigation Functions.

Algorithm (matches Lecture 9+10 of the course):

  Lagrangian dynamics (Lecture 9):
      M(q)q̈ + V(q,q̇) + G(q) = τ

  Control law (Lecture 10 — navigation functions):
      τ = τ_ctc  +  τ_rep
      τ_ctc = M(q)·(q̈_d + Kd·ė + Kp·e + Ki·∫e) + V(q,q̇)·q̇ + G(q)
      τ_rep = −J(q)ᵀ · ∇_p U_rep(p_ee(q))   ← repulsion from shelf obstacles

  Path planning (Lecture 4 roadmap principle):
      Build a collision-free path in C-space (sampling roadmap / bi-RRT style)
      between A and B. No manual `pre_B` waypoint is required.

  Each segment: quintic polynomial q_d(t) with zero endpoint velocities/accelerations.

Run:
    python3 -m my_robot_control.simulate_pid
or: ros2 run my_robot_control simulate_pid

Outputs in results/ (under cwd):
    pid_trajectory.csv   — joint trajectory for Gazebo playback
    joint_states_pid.png, tracking_error_pid.png, ee_path_pid.png
    stick_snapshots_pid.png, animation_pid.gif
    clearance_pid.png    — minimum clearance to shelf obstacles over time
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from ament_index_python.packages import get_package_share_directory as _share

from .dynamics    import load_dynamics
from .integrator  import simulate
from .kinematics  import fk_points, ee_pose, ik_analytic, ik_branch, rpy_to_rotation_matrix
from .controllers import (ComputedTorquePID, quintic_trajectory)
from .navigation  import (tau_repulsive, critical_obstacles, shelf_obstacles,
                           min_clearance, arm_capsule_clearance, clearance_report)
from .plotting    import (plot_joint_states, plot_velocities,
                           plot_tracking_error, plot_ee_path,
                           plot_stick_snapshots, animate_stick_figure)
from .urdf_params import JOINT_LOWER, JOINT_UPPER

_GCFG = Path(_share("my_robot_control")) / "config" / "pid_gains.yaml"
_RES  = Path.cwd() / "results"


def _resolve_config_path() -> Path:
    """Return the path to sim.yaml — single source of truth.

    Search order (first found wins):
      1. Live source tree  <workspace>/src/my_robot_control/config/sim.yaml
         → preferred for interactive tuning; edits take effect immediately
           without requiring `colcon build`.
      2. Installed share   <workspace>/install/.../share/.../config/sim.yaml
         → used when running via `ros2 run` or after a clean install.

    This prevents the "shadow parameters" bug where edits to src/ are silently
    ignored because the code reads a stale copy from the install directory.
    """
    cwd = Path.cwd()
    for candidate_root in [cwd, *cwd.parents]:
        src_cfg = candidate_root / "src" / "my_robot_control" / "config" / "sim.yaml"
        if src_cfg.exists():
            return src_cfg
    # Fall back to the installed share directory.
    return Path(_share("my_robot_control")) / "config" / "sim.yaml"


def _require(section: dict, key: str, section_name: str):
    """Read a REQUIRED parameter from a config section dict.

    Raises a descriptive KeyError if the key is absent, rather than silently
    falling back to a hardcoded default that may no longer match the intent in
    sim.yaml.  All safety-critical parameters (collision radii, planner limits,
    APF gains) must use this function so that a missing key is immediately
    visible at startup rather than causing subtle mis-behaviour.

    Args:
        section:      dict loaded from the relevant YAML section.
        key:          parameter name within that section.
        section_name: human-readable section label for the error message.

    Returns the raw value (caller must cast to float/int/str as needed).
    """
    if key not in section:
        raise KeyError(
            f"\n[CONFIG ERROR] Required parameter '{section_name}.{key}' is "
            f"missing from sim.yaml.\n"
            f"Add it to the [{section_name}] section and run 'colcon build' "
            f"(or restart if loading from src/).\n"
            f"See config/sim.yaml for all available parameters."
        )
    return section[key]


def main():
    _RES.mkdir(parents=True, exist_ok=True)

    # ── Locate and load sim.yaml (single source of truth) ────────────────────
    # Prefer live src/ so edits take effect without colcon rebuild.
    cfg_path = _resolve_config_path()

    # ── Load config ───────────────────────────────────────────────────────────
    # All safety-critical parameters use _require() to crash loudly on a missing
    # key rather than silently applying a stale hardcoded default.
    # Optional/convenience parameters still use .get() with a documented default.
    cfg  = yaml.safe_load(cfg_path.read_text())
    gcfg = yaml.safe_load(_GCFG.read_text())

    print("=" * 60)
    print("simulate_pid.py — CTC+PID + C-space planner + Navigation")
    print("       Shelf A→B demo: automatic collision-free planning")
    print("=" * 60)
    print(f"  config: {cfg_path}")

    sim_cfg      = cfg["simulation"]
    # dt: RK4 integration step — keep small (≤0.002) for numerical stability.
    dt           = float(sim_cfg["dt"])

    # ── Navigation / APF parameters (from sim.yaml [navigation] section) ─────
    # These control the repulsive potential field τ_rep = -J^T ∇U_rep added to
    # the CTC controller.  All are REQUIRED so a stale default cannot silently
    # change collision behaviour between tuning sessions.
    nav_cfg    = cfg.get("navigation", {})
    # rho0: activation radius [m] — repulsion starts when any proxy point is
    #       within rho0 of an obstacle surface.
    rho0       = float(_require(nav_cfg, "rho0",                   "navigation"))
    # eta: repulsion gain — higher values increase the push-away force.
    #      Set to 0.0 to disable APF and rely purely on the biRRT path.
    eta        = float(_require(nav_cfg, "eta",                    "navigation"))
    # obstacle_inflation: expand obstacle AABBs [m] before APF gradient calc.
    inflation  = float(_require(nav_cfg, "obstacle_inflation",     "navigation"))
    # min_clearance_threshold: post-run safety floor [m] (capsule model).
    clr_thresh = float(_require(nav_cfg, "min_clearance_threshold","navigation"))
    # repulsion_scope: 'critical' (divider only) or 'shelf' (all panels).
    rep_scope  = str(  _require(nav_cfg, "repulsion_scope",        "navigation")).strip().lower()

    # ── Planner parameters (from sim.yaml [planning] section) ─────────────────
    planning_cfg = cfg.get("planning", {})
    # enabled: if False, use pose_waypoints directly (no biRRT).
    planner_on   = bool(planning_cfg.get("enabled", True))
    # obstacle_inflation for planning (may differ from APF inflation).
    plan_infl    = float(planning_cfg.get("obstacle_inflation", inflation))
    # arm_surface_radius: physical arm cylinder radius [m] for C-obstacle check.
    #   arm_capsule_clearance(q, obs) > arm_surface_radius + surface_margin → Q_free
    plan_arm_rad = float(_require(planning_cfg, "arm_surface_radius", "planning"))
    # swept_samples: number of joint-space samples per segment in swept check.
    swept_samples = int(_require(planning_cfg, "swept_samples", "planning"))

    gains   = gcfg["computed_torque_pid"]
    Kp      = np.array(gains["Kp"], dtype=float)
    Kd      = np.array(gains["Kd"], dtype=float)
    Ki      = np.array(gains["Ki"], dtype=float)
    int_lim = float(gains["integral_limit"])

    # ── Build obstacle sets ───────────────────────────────────────────────────
    # all_obstacles: used for planning collision checks and post-run safety report.
    # repulsion_obstacles: used for APF τ_rep during tracking.
    all_obstacles = shelf_obstacles(inflation=plan_infl)
    if rep_scope == "shelf":
        repulsion_obstacles = shelf_obstacles(inflation=inflation)
    else:
        rep_scope = "critical"
        repulsion_obstacles = critical_obstacles(inflation=inflation)

    print(f"  planner.enabled      = {planner_on}")
    print(f"  planner.arm_radius   = {plan_arm_rad:.3f} m")
    print(f"  repulsion.scope      = {rep_scope}")
    print(f"  navigation.eta       = {eta}  (0.0 = APF disabled)")
    print(f"  navigation.rho0      = {rho0:.3f} m")

    # ── Resolve waypoints ──────────────────────────────────────────────────────
    if planner_on:
        if "pose_waypoints" in cfg:
            pose_wps = cfg["pose_waypoints"]
            if len(pose_wps) < 2:
                raise ValueError("planning.enabled=true requires at least 2 pose_waypoints (A,B)")
            print("\n[0] Resolving START/GOAL poses via IK …")
            print("    Convention: ROS fixed-axis RPY  →  R = Rz(yaw)·Ry(pitch)·Rx(roll)")
            resolved_sg = _resolve_pose_waypoints([pose_wps[0], pose_wps[-1]])
        elif "waypoints" in cfg:
            waypoints = cfg["waypoints"]
            if len(waypoints) < 2:
                raise ValueError("planning.enabled=true requires at least 2 waypoints (A,B)")
            print("\n[0] Using START/GOAL joint waypoints …")
            resolved_sg = [
                {"name": waypoints[0]["name"], "q": list(map(float, waypoints[0]["q"])),
                 "duration": float(waypoints[0]["duration"])},
                {"name": waypoints[-1]["name"], "q": list(map(float, waypoints[-1]["q"])),
                 "duration": float(waypoints[-1]["duration"])},
            ]
        else:
            raise KeyError("sim.yaml must contain 'pose_waypoints' or 'waypoints'")

        print("\n[1] Planning collision-free path in C-space …")
        raw_waypoints = _plan_cspace_waypoints(
            resolved_sg[0], resolved_sg[-1], all_obstacles, planning_cfg
        )
    else:
        if "pose_waypoints" in cfg:
            print("\n[0] Resolving pose_waypoints via IK …")
            print("    Convention: ROS fixed-axis RPY  →  R = Rz(yaw)·Ry(pitch)·Rx(roll)")
            raw_waypoints = _resolve_pose_waypoints(cfg["pose_waypoints"])
        elif "waypoints" in cfg:
            print("\n[0] Using joint-space waypoints (legacy mode)")
            raw_waypoints = cfg["waypoints"]
        else:
            raise KeyError("sim.yaml must contain 'pose_waypoints' or 'waypoints'")

    # ── Swept-path pre-run collision check (catches bad designs early) ──────────
    print("\n[2] Pre-run swept-path collision check …")
    swept = _swept_path_collision_check(raw_waypoints, all_obstacles,
                                        n_samples=swept_samples,
                                        arm_surface_radius=plan_arm_rad)
    for sr in swept["segments"]:
        flag = "✓" if sr["min_surface_gap"] > 0 else "⚠ SURFACE CONTACT"
        print(f"  {sr['segment']:30s}  centre={sr['min_centre_clr']*1000:.0f}mm  "
              f"surface_gap={sr['min_surface_gap']*1000:.0f}mm  s={sr['worst_s']:.2f}  {flag}")
    if not swept["ok"]:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"SWEPT-PATH CHECK FAILED — arm surface contacts shelf in planned path!\n"
            f"  global min centre clearance: {swept['global_min_centre']*1000:.0f} mm\n"
            f"  arm cylinder radius:          35 mm\n"
            f"  → surface gap:               {swept['global_min_surface']*1000:.0f} mm\n"
            f"Adjust waypoints or segment timings before running dynamics.\n"
            f"{'='*60}"
        )
    print(f"  ✓ Swept-path OK  "
          f"(min surface gap = {swept['global_min_surface']*1000:.0f} mm)")

    # ── Load Lagrangian dynamics ──────────────────────────────────────────────
    print("\n[3] Loading Lagrangian dynamics (M, C, G) …")
    M_func, C_func, G_func, V_func = load_dynamics()

    # ── Build multi-segment trajectory ────────────────────────────────────────
    print("\n[4] Building multi-segment trajectory …")
    q_segs = [np.array(wp["q"], dtype=float) for wp in raw_waypoints]
    T_segs = [float(wp["duration"])           for wp in raw_waypoints[:-1]]

    # Print resolved waypoints with clearances (vs full shelf obstacles)
    print("\nResolved waypoints:")
    for wp in raw_waypoints:
        q  = np.array(wp["q"], dtype=float)
        ee = fk_points(q)[-1]
        cl = min_clearance(q, all_obstacles)
        print(f"  {wp['name']:20s} q={np.round(q,4)}"
              f"  EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})"
              f"  clr={cl:.3f}m")

    # Concatenate quintic segments: A→R, R→Rise, Rise→B
    all_t, all_q, all_qd, all_qdd = [], [], [], []
    t_offset = 0.0

    for seg_i in range(len(T_segs)):
        q_from = q_segs[seg_i]
        q_to   = q_segs[seg_i + 1]
        T_seg  = T_segs[seg_i]

        q_d_f, qd_d_f, qdd_d_f = quintic_trajectory(q_from, q_to, T_seg)

        t_local = np.arange(0.0, T_seg + dt, dt)
        for t in t_local:
            all_t.append(t_offset + t)
            all_q.append(q_d_f(t))
            all_qd.append(qd_d_f(t))
            all_qdd.append(qdd_d_f(t))

        t_offset += T_seg

    # Hold final position
    T_hold = float(raw_waypoints[-1]["duration"])
    t_local = np.arange(0.0, T_hold + dt, dt)
    q_final = q_segs[-1]
    for t in t_local:
        all_t.append(t_offset + t)
        all_q.append(q_final.copy())
        all_qd.append(np.zeros(6))
        all_qdd.append(np.zeros(6))

    all_t    = np.array(all_t)
    all_q    = np.array(all_q)
    all_qd   = np.array(all_qd)
    all_qdd  = np.array(all_qdd)

    t_total = float(all_t[-1])
    result_desired = {"t": all_t, "q": all_q, "qdot": all_qd, "tau": np.zeros_like(all_q)}

    # Segment boundary times (for plot markers)
    seg_times = list(np.cumsum(T_segs))

    print(f"  Segments: {len(T_segs)}, total duration: {t_total:.1f}s, steps: {len(all_t)}")

    # ── Build tau function: CTC+PID + navigation repulsion ───────────────────
    ctrl = ComputedTorquePID(Kp, Kd, Ki, M_func, C_func, G_func, int_lim)

    # Index-based lookup for desired trajectory
    t_arr_ref = all_t
    q_arr_ref = all_q
    qd_arr_ref = all_qd
    qdd_arr_ref = all_qdd

    def _lookup(t_query):
        idx = int(np.clip(np.searchsorted(t_arr_ref, t_query), 0, len(t_arr_ref)-1))
        return q_arr_ref[idx], qd_arr_ref[idx], qdd_arr_ref[idx]

    def tau_nav(t, q, qdot):
        q_d, qd_d, qdd_d = _lookup(t)
        tau = ctrl.compute_tau(t, q, qdot, q_d, qd_d, qdd_d, dt)
        tau += tau_repulsive(q, repulsion_obstacles, rho0=rho0, eta=eta)
        return tau

    # ── Run simulation ────────────────────────────────────────────────────────
    q0    = q_segs[0].copy()
    qdot0 = np.zeros(6)

    print("\n[5] Running CTC+PID + navigation simulation …")
    result = simulate(q0, qdot0, tau_nav, (0.0, t_total), dt,
                      M_func, C_func, G_func, enforce_limits=True)
    print("  Done.")

    # ── Post-run safety check (required) ────────────────────────────────────
    print("\n  Running post-run safety check …")
    # clearance_report now uses arm_capsule_clearance internally for accuracy
    crpt = clearance_report(result["t"], result["q"], all_obstacles)
    min_cl  = crpt["min_clearance"]
    min_t   = crpt["min_time"]
    min_idx = crpt["min_idx"]

    # Identify which segment the minimum occurs in
    seg_name = "hold"
    seg_labels = [wp["name"] for wp in raw_waypoints]
    for si, T_seg in enumerate(seg_times):
        if min_t <= T_seg:
            seg_name = f"{seg_labels[si]}→{seg_labels[si+1]}"
            break

    print(f"  obstacle_inflation   = {plan_infl:.3f} m")
    print(f"  rho0                 = {rho0:.3f} m")
    print(f"  min_clearance_threshold = {clr_thresh:.3f} m")
    print(f"  Min clearance (8-pt proxy, inflated obstacles): {min_cl:.4f} m")
    print(f"  Worst at t={min_t:.3f}s  segment='{seg_name}'")

    if min_cl < clr_thresh:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"SAFETY CHECK FAILED — arm too close to shelf!\n"
            f"  min_clearance      = {min_cl*1000:.1f} mm  "
            f"(threshold = {clr_thresh*1000:.0f} mm)\n"
            f"  Worst at t = {min_t:.3f}s  segment = '{seg_name}'\n"
            f"  Arm proxy point at index {min_idx} of trajectory.\n"
            f"Actions: increase obstacle_inflation, raise rho0/eta, or\n"
            f"         adjust intermediate waypoints for more clearance.\n"
            f"{'='*60}"
        )
    else:
        print(f"  ✓  Safety check PASSED  "
              f"(clearance {min_cl*100:.1f} cm ≥ threshold {clr_thresh*100:.1f} cm)")

    # Final tracking error
    q_fin     = result["q"][-1]
    err_final = float(np.linalg.norm(q_segs[-1] - q_fin))
    pB        = fk_points(q_segs[-1])[-1]
    pF        = fk_points(q_fin)[-1]
    print(f"\n  Final joint error   ‖q_f − q_B‖ = {err_final:.5f}")
    print(f"  Final EE position: ({pF[0]:.4f}, {pF[1]:.4f}, {pF[2]:.4f}) m")
    print(f"  EE error ‖pF − pB‖ = {np.linalg.norm(pF - pB):.5f} m")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[6] Saving plots …")
    _trim(result_desired, len(result["t"]))
    plot_joint_states(result, result_desired, tag="pid")
    plot_velocities(result, result_desired, tag="pid", segment_times=seg_times)
    plot_tracking_error(result, result_desired, tag="pid", segment_times=seg_times)
    plot_ee_path(result, tag="pid")
    plot_stick_snapshots(result, n_snaps=5, tag="pid")

    try:
        out = animate_stick_figure(result, tag="pid", fps=25)
        print(f"  {out.name} saved")
    except Exception as exc:
        print(f"  (animation skipped: {exc})")

    # Clearance plot (using the report already computed)
    _plot_clearance(result["t"], crpt["clearances"],
                    clr_thresh=clr_thresh, inflation=plan_infl)

    # ── CSV export for Gazebo playback ────────────────────────────────────────
    csv_path = _RES / "pid_trajectory.csv"
    _save_csv(result, csv_path)
    print(f"\nResults saved to: {_RES}/")


# ── Pose-waypoint resolution ──────────────────────────────────────────────────

def _resolve_pose_waypoints(pose_wps: list[dict]) -> list[dict]:
    """Resolve pose_waypoints to joint configs.  Each item may have either:

      - position + rpy  → IK (ROS fixed-axis RPY: R = Rz(yaw)·Ry(pitch)·Rx(roll))
      - q               → joint-space config used directly (no IK called)

    Joint-space items are labelled clearly in the log and bypass IK entirely.
    This keeps backward compatibility with legacy configs.

    IK branch continuity: each IK call uses the previous solution as seed.
    Raises RuntimeError with diagnostics if IK fails for any pose waypoint.
    """
    seed = np.zeros(6)
    resolved = []

    for wp in pose_wps:
        name = wp["name"]
        dur  = float(wp["duration"])

        # ── Joint-space waypoint: use q directly ──────────────────────────────
        if "q" in wp:
            q = np.array(wp["q"], dtype=float)
            ee = fk_points(q)[-1]
            print(f"  {name:20s} → q={np.round(q,4)}  [joint-space]  "
                  f"EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})")
            seed = q
            resolved.append({"name": name, "q": q.tolist(), "duration": dur})
            continue

        # ── Pose waypoint: resolve via IK ─────────────────────────────────────
        pos  = np.array(wp["position"], dtype=float)
        rpy  = np.array(wp["rpy"],      dtype=float)

        R = rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2])

        # Try analytic first (handles singular wrist: R47=I → q4=q5=q6=0)
        q, ok = ik_analytic(R, pos, clamp=True)
        if not ok:
            q, ok = ik_branch(R, pos, seed=seed, clamp=True)

        if not ok:
            R_fk, p_fk = ee_pose(q)
            raise RuntimeError(
                f"IK failed for pose_waypoint '{name}':\n"
                f"  target position = {np.round(pos, 4)}\n"
                f"  target RPY      = {np.round(rpy, 4)}\n"
                f"  FK pos error    = {float(np.linalg.norm(pos-p_fk)):.2e} m\n"
                f"  resolved q      = {np.round(q, 4)}\n"
                f"Check that the pose is within the arm's reachable workspace."
            )

        print(f"  {name:20s} → q={np.round(q,4)}  "
              f"pos_err={float(np.linalg.norm(pos-ee_pose(q)[1])):.2e}")
        seed = q
        resolved.append({"name": name, "q": q.tolist(), "duration": dur})

    return resolved


def _cspace_distance(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Distance in normalized joint space (joint limits map to [0,1])."""
    span = JOINT_UPPER - JOINT_LOWER
    return float(np.linalg.norm((q_b - q_a) / span))


def _state_is_valid(q: np.ndarray,
                    obstacles: list[tuple[np.ndarray, np.ndarray]],
                    arm_surface_radius: float,
                    surface_margin: float = 0.0) -> bool:
    """Collision-free test for one configuration.

    Implements the C-obstacle check from Lecture 2 (Minkowski-sum / C-space):
        q ∈ Q_free  ⟺  arm_capsule_clearance(q, W_obs) > arm_r + margin

    arm_capsule_clearance uses segment distances (capsule model), giving an
    accurate Minkowski-sum approximation: each arm link is a cylinder of
    radius arm_surface_radius.  The check fails whenever the SURFACE of any
    arm cylinder is within `surface_margin` metres of any obstacle boundary.
    """
    return arm_capsule_clearance(q, obstacles) > (arm_surface_radius + surface_margin)


def _edge_is_valid(q_a: np.ndarray,
                   q_b: np.ndarray,
                   obstacles: list[tuple[np.ndarray, np.ndarray]],
                   arm_surface_radius: float,
                   edge_resolution_norm: float,
                   surface_margin: float = 0.0) -> bool:
    """Collision test for a joint-space segment via interpolation sampling."""
    d = _cspace_distance(q_a, q_b)
    n = max(2, int(np.ceil(d / max(edge_resolution_norm, 1e-4))))
    for s in np.linspace(0.0, 1.0, n):
        q = q_a + s * (q_b - q_a)
        if not _state_is_valid(q, obstacles, arm_surface_radius, surface_margin):
            return False
    return True


def _nearest_index(nodes: list[np.ndarray], q_target: np.ndarray) -> int:
    d = [_cspace_distance(q, q_target) for q in nodes]
    return int(np.argmin(d))


def _steer(q_from: np.ndarray, q_to: np.ndarray, step_norm: float) -> np.ndarray:
    """Move from q_from toward q_to by at most `step_norm` in normalized space."""
    span = JOINT_UPPER - JOINT_LOWER
    delta_n = (q_to - q_from) / span
    d = float(np.linalg.norm(delta_n))
    if d <= step_norm:
        q_new = q_to.copy()
    else:
        q_new = q_from + (step_norm / d) * (q_to - q_from)
    return np.clip(q_new, JOINT_LOWER, JOINT_UPPER)


def _extend_tree(nodes: list[np.ndarray],
                 parents: list[int],
                 q_target: np.ndarray,
                 obstacles: list[tuple[np.ndarray, np.ndarray]],
                 arm_surface_radius: float,
                 step_norm: float,
                 edge_resolution_norm: float,
                 surface_margin: float) -> tuple[str, int | None, np.ndarray | None]:
    """Single RRT extend step; returns (status, new_idx, new_q)."""
    idx_near = _nearest_index(nodes, q_target)
    q_near = nodes[idx_near]
    q_new = _steer(q_near, q_target, step_norm)

    if np.allclose(q_new, q_near):
        return "trapped", None, None

    if not _edge_is_valid(q_near, q_new, obstacles, arm_surface_radius,
                          edge_resolution_norm, surface_margin):
        return "trapped", None, None

    nodes.append(q_new)
    parents.append(idx_near)
    idx_new = len(nodes) - 1
    if _cspace_distance(q_new, q_target) <= step_norm:
        return "reached", idx_new, q_new
    return "advanced", idx_new, q_new


def _connect_tree(nodes: list[np.ndarray],
                  parents: list[int],
                  q_target: np.ndarray,
                  obstacles: list[tuple[np.ndarray, np.ndarray]],
                  arm_surface_radius: float,
                  step_norm: float,
                  edge_resolution_norm: float,
                  surface_margin: float) -> tuple[str, int | None, np.ndarray | None]:
    """Repeatedly extend a tree toward q_target until trapped/reached."""
    status = "advanced"
    idx_new = None
    q_new = None
    while status == "advanced":
        status, idx_new, q_new = _extend_tree(
            nodes, parents, q_target, obstacles, arm_surface_radius,
            step_norm, edge_resolution_norm, surface_margin
        )
    return status, idx_new, q_new


def _trace_path(nodes: list[np.ndarray], parents: list[int], idx: int) -> list[np.ndarray]:
    """Return root→idx path for one tree."""
    path = []
    k = idx
    while k != -1:
        path.append(nodes[k])
        k = parents[k]
    return list(reversed(path))


def _shortcut_path(path_q: list[np.ndarray],
                   obstacles: list[tuple[np.ndarray, np.ndarray]],
                   arm_surface_radius: float,
                   edge_resolution_norm: float,
                   surface_margin: float,
                   trials: int,
                   rng: np.random.Generator) -> list[np.ndarray]:
    """Path shortening with random shortcut attempts."""
    if len(path_q) <= 2:
        return path_q

    path = [q.copy() for q in path_q]
    for _ in range(max(0, trials)):
        if len(path) <= 2:
            break
        i = int(rng.integers(0, len(path) - 2))
        j = int(rng.integers(i + 2, len(path)))
        if _edge_is_valid(path[i], path[j], obstacles, arm_surface_radius,
                          edge_resolution_norm, surface_margin):
            path = path[:i + 1] + path[j:]
    return path


def _plan_birrt_path(q_start: np.ndarray,
                     q_goal: np.ndarray,
                     obstacles: list[tuple[np.ndarray, np.ndarray]],
                     arm_surface_radius: float,
                     edge_resolution_norm: float,
                     step_norm: float,
                     goal_bias: float,
                     max_iters: int,
                     surface_margin: float,
                     rng: np.random.Generator) -> list[np.ndarray]:
    """Bidirectional RRT path planner in C-space."""
    if not _state_is_valid(q_start, obstacles, arm_surface_radius, surface_margin):
        raise RuntimeError("Planner start configuration is in collision (or below margin).")
    if not _state_is_valid(q_goal, obstacles, arm_surface_radius, surface_margin):
        raise RuntimeError("Planner goal configuration is in collision (or below margin).")

    if _edge_is_valid(q_start, q_goal, obstacles, arm_surface_radius,
                      edge_resolution_norm, surface_margin):
        return [q_start.copy(), q_goal.copy()]

    # Tree A rooted at start, tree B rooted at goal.
    nodes_a, parents_a = [q_start.copy()], [-1]
    nodes_b, parents_b = [q_goal.copy()], [-1]

    for _ in range(max_iters):
        if rng.random() < goal_bias:
            q_rand = q_goal.copy()
        else:
            q_rand = rng.uniform(JOINT_LOWER, JOINT_UPPER)

        status_a, idx_a, q_new_a = _extend_tree(
            nodes_a, parents_a, q_rand, obstacles, arm_surface_radius,
            step_norm, edge_resolution_norm, surface_margin
        )
        if status_a == "trapped":
            continue

        status_b, idx_b, _ = _connect_tree(
            nodes_b, parents_b, q_new_a, obstacles, arm_surface_radius,
            step_norm, edge_resolution_norm, surface_margin
        )
        if status_b == "reached" and idx_a is not None and idx_b is not None:
            path_a = _trace_path(nodes_a, parents_a, idx_a)      # start → meet
            path_b = _trace_path(nodes_b, parents_b, idx_b)      # goal  → meet
            path = path_a + list(reversed(path_b))[1:]           # start → goal
            return path

    raise RuntimeError(
        "C-space planner failed to connect start and goal. "
        "Try increasing planning.max_iters/samples or reducing safety margin."
    )


def _plan_cspace_waypoints(start_wp: dict,
                           goal_wp: dict,
                           obstacles: list[tuple[np.ndarray, np.ndarray]],
                           planning_cfg: dict) -> list[dict]:
    """Plan a collision-free joint-space path from start to goal using biRRT.

    All safety-critical parameters are read with _require() so a missing key
    raises an immediate, descriptive error instead of silently using a stale
    hardcoded default.  Every parameter name maps 1-to-1 to a key in the
    sim.yaml [planning] section — the YAML is the single source of truth.

    Parameters from sim.yaml [planning]:
      arm_surface_radius  — physical arm cylinder radius [m]; defines C-obstacle
                            boundary via Minkowski sum (Lecture 2).
      surface_margin      — extra clearance [m] beyond arm_surface_radius;
                            total rejection threshold = arm_r + margin.
      max_iters           — biRRT iteration budget.
      step_norm           — max RRT extension step in normalised joint space.
      edge_resolution_norm — collision-check spacing along edges.
      goal_bias           — probability of sampling the goal directly.
      shortcut_trials     — random path-shortening attempts after planning.
      rng_seed            — (optional) reproducibility seed; default 7.
    """
    q_start = np.array(start_wp["q"], dtype=float)
    q_goal  = np.array(goal_wp["q"],  dtype=float)

    # rng_seed is optional — randomness is fine without a fixed seed.
    rng_seed = int(planning_cfg.get("rng_seed", 7))
    rng      = np.random.default_rng(rng_seed)

    # All safety-critical parameters are REQUIRED in sim.yaml.
    arm_surface_radius = float(_require(planning_cfg, "arm_surface_radius",   "planning"))
    edge_res           = float(_require(planning_cfg, "edge_resolution_norm", "planning"))
    step_norm          = float(_require(planning_cfg, "step_norm",            "planning"))
    goal_bias          = float(_require(planning_cfg, "goal_bias",            "planning"))
    max_iters          = int(  _require(planning_cfg, "max_iters",            "planning"))
    surface_margin     = float(_require(planning_cfg, "surface_margin",       "planning"))
    shortcut_trials    = int(  _require(planning_cfg, "shortcut_trials",      "planning"))

    print(f"  planner type         = biRRT")
    print(f"  planner max_iters    = {max_iters}")
    print(f"  planner step_norm    = {step_norm:.3f}")
    print(f"  planner edge_res     = {edge_res:.3f}")
    print(f"  planner goal_bias    = {goal_bias:.2f}")
    print(f"  planner arm_r+margin = {arm_surface_radius:.3f}+{surface_margin:.3f}"
          f" = {(arm_surface_radius+surface_margin)*1000:.0f}mm threshold")

    path_q = _plan_birrt_path(
        q_start=q_start, q_goal=q_goal, obstacles=obstacles,
        arm_surface_radius=arm_surface_radius,
        edge_resolution_norm=edge_res,
        step_norm=step_norm,
        goal_bias=goal_bias,
        max_iters=max_iters,
        surface_margin=surface_margin,
        rng=rng,
    )
    path_q = _shortcut_path(path_q, obstacles, arm_surface_radius,
                            edge_res, surface_margin, shortcut_trials, rng)

    # Convert path nodes to runtime waypoints with automatic segment timing.
    # These are less safety-critical so .get() with documented defaults is OK.
    min_seg      = float(planning_cfg.get("min_segment_duration", 0.50))
    max_seg      = float(planning_cfg.get("max_segment_duration", 2.00))
    sec_per_norm = float(planning_cfg.get("seconds_per_unit",     3.5))
    hold_duration = float(goal_wp["duration"])

    planned_wps: list[dict] = []
    for i, q in enumerate(path_q):
        if i == 0:
            name = start_wp["name"]
        elif i == len(path_q) - 1:
            name = goal_wp["name"]
        else:
            name = f"plan_{i:02d}"

        if i < len(path_q) - 1:
            seg_d = _cspace_distance(path_q[i], path_q[i + 1])
            dur = float(np.clip(sec_per_norm * seg_d, min_seg, max_seg))
        else:
            dur = hold_duration

        planned_wps.append({"name": name, "q": q.tolist(), "duration": dur})

    # Path diagnostics
    path_len = sum(_cspace_distance(path_q[i], path_q[i + 1]) for i in range(len(path_q) - 1))
    print(f"  planned nodes        = {len(path_q)}")
    print(f"  planned cspace len   = {path_len:.3f} (normalized)")
    print(f"  planned motion time  = {sum(w['duration'] for w in planned_wps[:-1]):.2f} s")

    return planned_wps


def _swept_path_collision_check(
        waypoints: list[dict],
        obstacles: list[tuple[np.ndarray, np.ndarray]],
        n_samples: int = 80,
        arm_surface_radius: float = 0.035,
) -> dict:
    """Sample each segment in joint space and report minimum clearance.

    This is a PRE-RUN check (before the expensive dynamics simulation) that
    catches trajectory designs that would hit the shelf even with perfect tracking.
    It samples joint-space linear interpolation, which is what the quintic
    trajectory approximates for short segments.

    Args:
        waypoints: resolved joint-space waypoints
        obstacles: AABB list (as in navigation.critical_obstacles)
        n_samples: samples per segment
        arm_surface_radius: arm cylinder radius added to the reported physical gap

    Returns dict with:
        ok: bool — True if all physical gaps > 0
        segments: list of per-segment worst-clearance records
        global_min_centre: float — minimum centre-line clearance
        global_min_surface: float — minimum surface clearance (centre - radius)
    """
    # Use capsule model for accurate swept-path check (Lecture 2 C-obstacle mapping).
    # arm_capsule_clearance returns SEGMENT distances; subtract arm_surface_radius
    # to get the physical capsule-surface-to-obstacle gap.

    def seg_worst(q_from, q_to):
        worst_c, worst_s = 999.0, 0.0
        for s in np.linspace(0, 1, n_samples):
            q = q_from + s * (q_to - q_from)
            c = arm_capsule_clearance(q, obstacles)
            if c < worst_c:
                worst_c = c
                worst_s = float(s)
        return worst_c, worst_s

    seg_results = []
    q_arr = [np.array(wp["q"], dtype=float) for wp in waypoints]
    names = [wp["name"] for wp in waypoints]
    global_min = 999.0

    for i in range(len(q_arr) - 1):
        wc, ws = seg_worst(q_arr[i], q_arr[i + 1])
        surface_gap = wc - arm_surface_radius
        if wc < global_min:
            global_min = wc
        seg_results.append({
            "segment":       f"{names[i]}→{names[i+1]}",
            "min_centre_clr": wc,
            "min_surface_gap": surface_gap,
            "worst_s":        ws,
        })

    return {
        "ok":                global_min > arm_surface_radius,
        "segments":          seg_results,
        "global_min_centre": global_min,
        "global_min_surface": global_min - arm_surface_radius,
    }


# ── General helpers ───────────────────────────────────────────────────────────

def _trim(rd: dict, n: int):
    for k in ("t", "q", "qdot", "tau"):
        rd[k] = rd[k][:n]


def _plot_clearance(t: np.ndarray, clearances: np.ndarray,
                    clr_thresh: float = 0.01, inflation: float = 0.035):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path.cwd() / "results" / "clearance_pid.png"

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, clearances * 100, "r-", lw=1.5, label="min clearance (8-pt proxy)")
    ax.axhline(clr_thresh * 100, color="orange", ls="--", lw=1.5,
               label=f"safety threshold ({clr_thresh*100:.0f} mm)")
    ax.axhline(0.0, color="darkred", ls=":", lw=0.8, label="contact")
    ax.fill_between(t, 0, clearances * 100, alpha=0.15, color="red")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Clearance [cm]")
    ax.set_title(
        f"Min arm-to-obstacle clearance  |  inflation={inflation*100:.0f} mm  "
        f"|  8-proxy-point model  |  min={clearances.min()*100:.1f} cm")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  clearance_pid.png saved  (min={clearances.min()*100:.1f} cm)")


def _save_csv(result: dict, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"q{i+1}" for i in range(6)] +
                   [f"qd{i+1}" for i in range(6)] +
                   [f"tau{i+1}" for i in range(6)])
        for k in range(len(result["t"])):
            w.writerow([result["t"][k]] +
                       list(result["q"][k]) +
                       list(result["qdot"][k]) +
                       list(result["tau"][k]))


if __name__ == "__main__":
    main()
