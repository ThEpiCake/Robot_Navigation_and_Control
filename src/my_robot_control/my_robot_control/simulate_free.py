"""Zero-input (τ=0) free-response simulation (§5 item 6 / §9.3).

Run:
    python -m my_robot_control.simulate_free
or (after colcon build):
    ros2 run my_robot_control simulate_free

Outputs in results/:
  joint_states_free.png   — q_i(t), q̇_i(t)
  ee_path_free.png        — end-effector 3D path + (x,y,z)(t)
  energy_free.png         — T+V(t) conservation check
  stick_snapshots_free.png
  animation_free.mp4
  free_response.csv       — full trajectory for Gazebo playback
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from .dynamics    import load_dynamics
from .integrator  import simulate, compute_energy_history
from .controllers import zero_torque
from .kinematics  import fk_points, cross_check_fk, ee_pose, ik_analytic
from .plotting   import (plot_joint_states, plot_ee_path, plot_energy,
                          plot_stick_snapshots, animate_stick_figure)

from ament_index_python.packages import get_package_share_directory as _share
_CFG = Path(_share("my_robot_control")) / "config" / "sim.yaml"
_RES = Path.cwd() / "results"


def main():
    # ── Load config ──────────────────────────────────────────────────────────
    cfg = yaml.safe_load(_CFG.read_text())
    sim_cfg = cfg["simulation"]
    dt     = float(sim_cfg["dt"])
    t_free = float(sim_cfg["t_free"])
    q0, q0_source = _resolve_q_start(cfg)
    qdot0  = np.zeros(6)

    print("=" * 60)
    print("simulate_free.py — τ=0 free-response simulation")
    print(f"  q_start = {q0}  ({q0_source})")
    print(f"  t_free  = {t_free} s,  dt = {dt} s")
    print("=" * 60)

    # ── Cross-check FK ───────────────────────────────────────────────────────
    print("Running FK cross-check …")
    assert cross_check_fk(), "FK cross-check FAILED — check kinematics.py"
    print("FK cross-check PASSED.\n")

    # ── Load dynamics ────────────────────────────────────────────────────────
    M_func, C_func, G_func, V_func = load_dynamics()

    # ── Initial EE pose report ───────────────────────────────────────────────
    pts0 = fk_points(q0)
    print(f"EE position at q_start: {pts0[-1]}")
    print()

    # ── Free simulation ──────────────────────────────────────────────────────
    print(f"Integrating {int(t_free/dt)} RK4 steps …")
    result = simulate(q0, qdot0, zero_torque, (0.0, t_free), dt,
                      M_func, C_func, G_func, enforce_limits=True)
    print("Done.\n")

    # Physical sanity check
    dq1 = abs(result["q"][-1, 0] - q0[0])
    print(f"Δq1 (base yaw) = {dq1:.4f} rad  — should be ≈ 0 (no gravitational torque on Z-revolute)")
    dq2 = result["q"][-1, 1] - q0[1]
    print(f"Δq2 (arm1 pris) = {dq2:.4f} m  — should be negative (gravity pulls it down)")
    print()

    # ── Energy conservation check ────────────────────────────────────────────
    print("Computing energy history …")
    E = compute_energy_history(result, M_func, V_func)
    E0 = E[0]
    drift = abs(E[-1] - E0)
    pct   = 100 * drift / abs(E0) if abs(E0) > 1e-12 else 0
    print(f"  E(0)   = {E0:.4f} J")
    print(f"  E(end) = {E[-1]:.4f} J")
    print(f"  drift  = {drift:.2e} J  ({pct:.2f}%)")
    if pct > 1.0:
        hit_lower = result["q"][-1, 1] <= 0.001 or result["q"][-1, 2] <= 0.001
        if hit_lower:
            print("  NOTE: drift expected — prismatic joint hit lower limit (hard stop absorbs KE)")
        else:
            print("  WARNING: energy drift > 1% — consider reducing dt")
    print()

    # ── Plots ────────────────────────────────────────────────────────────────
    print("Saving plots …")
    plot_joint_states(result, tag="free")
    plot_ee_path(result, tag="free")
    plot_energy(result["t"], E, tag="free")
    plot_stick_snapshots(result, n_snaps=4, tag="free")
    out = animate_stick_figure(result, tag="free", fps=25)
    print(f"  {out.name} saved")

    # ── CSV export for Gazebo playback ───────────────────────────────────────
    csv_path = _RES / "free_response.csv"
    _save_csv(result, csv_path)
    print(f"\nResults saved to:  {_RES}/")


def _save_csv(result: dict, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"q{i+1}" for i in range(6)] +
                   [f"qd{i+1}" for i in range(6)])
        for k in range(len(result["t"])):
            w.writerow([result["t"][k]] +
                       list(result["q"][k]) +
                       list(result["qdot"][k]))


def _resolve_q_start(cfg: dict) -> tuple[np.ndarray, str]:
    """Resolve q_start robustly from config.

    Priority:
      1) explicit `q_start` field
      2) first entry in `waypoints`
      3) IK from `ee_A.position` with natural EE orientation at q=0
    """
    if "q_start" in cfg:
        return np.array(cfg["q_start"], dtype=float), "from sim.yaml:q_start"

    waypoints = cfg.get("waypoints", [])
    if waypoints and "q" in waypoints[0]:
        return np.array(waypoints[0]["q"], dtype=float), "from sim.yaml:waypoints[0]"

    ee_A = cfg.get("ee_A", {})
    if "position" in ee_A:
        pA = np.array(ee_A["position"], dtype=float)
        R_natural, _ = ee_pose(np.zeros(6))
        qA, ok = ik_analytic(R_natural, pA)
        if not ok:
            raise RuntimeError(
                "IK failed for ee_A while resolving q_start. "
                "Set `q_start` explicitly in config/sim.yaml.")
        return qA, "from sim.yaml:ee_A via IK"

    raise KeyError(
        "No q_start source found in config/sim.yaml. "
        "Provide `q_start`, or `waypoints`, or `ee_A.position`.")


if __name__ == "__main__":
    main()
