"""Visualisation utilities (§5, §9.3 of brief).

All plot functions accept a simulation result dict
(keys: 't', 'q', 'qdot', 'tau') as returned by integrator.simulate().

Every function saves its output under results/ and optionally displays it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    pass   # 3D plots will fall back gracefully if mpl_toolkits is broken

from .kinematics import fk_points

def _res() -> Path:
    """Return results dir, creating it under cwd if needed."""
    d = Path.cwd() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d

_JOINT_LABELS = ["q1 (yaw, rad)", "q2 (arm1 pris, m)", "q3 (arm2 pris, m)",
                 "q4 (wrist-Z, rad)", "q5 (wrist-Y, rad)", "q6 (wrist-Z2, rad)"]


# ── Joint state plots ─────────────────────────────────────────────────────────

def plot_joint_states(result: dict,
                      result_desired: dict | None = None,
                      tag: str = "free",
                      show: bool = False) -> Path:
    """Plot q_i(t) and q̇_i(t) — item 6 / 9 of §5."""
    t  = result["t"]
    q  = result["q"]
    qd = result["qdot"]

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    fig.suptitle(f"Joint states ({tag})", fontsize=13)

    for i in range(6):
        ax_q  = axes[i // 2, (i % 2) * 2]
        ax_qd = axes[i // 2, (i % 2) * 2 + 1]

        ax_q.plot(t, q[:, i], "b", lw=1.2, label="actual")
        ax_q.set_ylabel(_JOINT_LABELS[i])
        ax_q.set_xlabel("t [s]")
        ax_q.grid(True, alpha=0.3)

        if result_desired is not None:
            qd_arr = result_desired["q"]
            ax_q.plot(t, qd_arr[:, i], "r--", lw=1.0, label="desired")
            ax_q.legend(fontsize=7)

        ax_qd.plot(t, qd[:, i], "g", lw=1.2)
        ax_qd.set_ylabel(f"q̇{i+1}")
        ax_qd.set_xlabel("t [s]")
        ax_qd.grid(True, alpha=0.3)

    fig.tight_layout()
    out = _res() / f"joint_states_{tag}.png"
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out


def plot_velocities(result: dict,
                    result_desired: dict | None = None,
                    tag: str = "pid",
                    segment_times: list[float] | None = None,
                    show: bool = False) -> Path:
    """Dedicated q̇_i(t) velocity plot — course requirement (§5 items 6/9)."""
    t  = result["t"]
    qd = result["qdot"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(f"Joint velocities q̇_i(t) ({tag})", fontsize=13)

    for i in range(6):
        ax = axes[i // 3, i % 3]
        ax.plot(t, qd[:, i], "g", lw=1.5, label="actual q̇")
        if result_desired is not None and "qdot" in result_desired:
            ax.plot(t, result_desired["qdot"][:, i], "r--", lw=1.0, label="desired q̇")
        if segment_times:
            for st in segment_times:
                ax.axvline(st, color="gray", ls=":", lw=0.8)
        ax.set_ylabel(_JOINT_LABELS[i])
        ax.set_xlabel("t [s]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = _res() / f"velocities_{tag}.png"
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out


def plot_tracking_error(result: dict, result_desired: dict,
                        tag: str = "pid",
                        segment_times: list[float] | None = None,
                        show: bool = False) -> Path:
    """Plot per-joint tracking error e_i(t) and τ_i(t) — item 9 of §5."""
    t   = result["t"]
    e   = result_desired["q"] - result["q"]
    tau = result["tau"]

    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    fig.suptitle(f"Tracking error & control effort ({tag})", fontsize=13)
    for i in range(6):
        axes[0, i].plot(t, e[:, i], "r", lw=1.2)
        axes[0, i].set_title(f"e{i+1}  [max={np.abs(e[:,i]).max():.3f}]")
        axes[0, i].set_xlabel("t [s]")
        axes[0, i].grid(True, alpha=0.3)
        if segment_times:
            for st in segment_times:
                axes[0, i].axvline(st, color="gray", ls=":", lw=0.8)

        axes[1, i].plot(t, tau[:, i], "m", lw=1.2)
        axes[1, i].set_title(f"τ{i+1}")
        axes[1, i].set_xlabel("t [s]")
        axes[1, i].grid(True, alpha=0.3)
        if segment_times:
            for st in segment_times:
                axes[1, i].axvline(st, color="gray", ls=":", lw=0.8)

    fig.tight_layout()
    out = _res() / f"tracking_error_{tag}.png"
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out


# ── End-effector path ─────────────────────────────────────────────────────────

def plot_ee_path(result: dict, tag: str = "free", show: bool = False) -> Path:
    """3D end-effector path (x,y,z)(t) — item 6/9 of §5."""
    q = result["q"]
    t = result["t"]
    xyz = np.array([fk_points(q[k])[-1] for k in range(len(t))])

    fig = plt.figure(figsize=(14, 5))
    # 3D trajectory (graceful fallback if mpl_toolkits 3d is broken)
    try:
        ax3 = fig.add_subplot(1, 2, 1, projection="3d")
        ax3.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "b", lw=1.2)
        ax3.scatter(*xyz[0], c="g", s=60, label="start")
        ax3.scatter(*xyz[-1], c="r", s=60, label="end")
        ax3.set_xlabel("X [m]"); ax3.set_ylabel("Y [m]"); ax3.set_zlabel("Z [m]")
        ax3.set_title(f"EE path 3D ({tag})")
        ax3.legend()
    except Exception:
        ax3 = fig.add_subplot(1, 2, 1)
        ax3.plot(xyz[:, 0], xyz[:, 2], "b", lw=1.2)
        ax3.scatter(xyz[0, 0], xyz[0, 2], c="g", s=60, label="start (X-Z)")
        ax3.scatter(xyz[-1, 0], xyz[-1, 2], c="r", s=60, label="end (X-Z)")
        ax3.set_xlabel("X [m]"); ax3.set_ylabel("Z [m]")
        ax3.set_title(f"EE path X-Z ({tag})")
        ax3.legend(); ax3.grid(True, alpha=0.3)

    # Time plots
    ax2 = fig.add_subplot(1, 2, 2)
    for i, lbl in enumerate(["X", "Y", "Z"]):
        ax2.plot(t, xyz[:, i], label=lbl)
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("position [m]")
    ax2.set_title(f"EE (x,y,z)(t) ({tag})")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = _res() / f"ee_path_{tag}.png"
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out


# ── Energy conservation ───────────────────────────────────────────────────────

def plot_energy(t: np.ndarray, E: np.ndarray,
                tag: str = "free", show: bool = False) -> Path:
    """Plot T+V(t) — energy-conservation check (§4 / §9.3)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, E, "k", lw=1.5)
    drift = E[-1] - E[0]
    ax.set_title(f"Total mechanical energy ({tag}) — drift = {drift:.4g} J")
    ax.set_xlabel("t [s]"); ax.set_ylabel("E = T+V  [J]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _res() / f"energy_{tag}.png"
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out


# ── 3-D stick figure ──────────────────────────────────────────────────────────

def _draw_stick(ax, q: np.ndarray, color: str = "b",
                label: str | None = None, alpha: float = 1.0):
    """Draw one stick-figure onto 3D axes."""
    pts = fk_points(q)   # (9, 3)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ax.plot(x, y, z, f"{color}o-", lw=2, ms=5, alpha=alpha, label=label)


def plot_stick_snapshots(result: dict, n_snaps: int = 4,
                         tag: str = "free", show: bool = False) -> Path:
    """Stick-figure snapshots at t=0, t=mid, t=end (§5 / §9.3)."""
    q = result["q"]
    t = result["t"]
    idx = np.linspace(0, len(t) - 1, n_snaps, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_snaps))

    fig = plt.figure(figsize=(12, 5))
    ax2 = fig.add_subplot(1, 2, 2)
    try:
        ax3 = fig.add_subplot(1, 2, 1, projection="3d")
        use_3d = True
    except Exception:
        ax3 = fig.add_subplot(1, 2, 1)
        use_3d = False

    for ci, ki in zip(colors, idx):
        pts = fk_points(q[ki])
        if use_3d:
            _draw_stick(ax3, q[ki], color="k", label=f"t={t[ki]:.2f}s", alpha=0.5)
            ax3.scatter(*pts[-1], s=30, color=ci)
        else:
            ax3.plot(pts[:, 0], pts[:, 2], "ko-", lw=1.5, alpha=0.5,
                     label=f"t={t[ki]:.2f}s")
            ax3.scatter(pts[-1, 0], pts[-1, 2], s=30, color=ci)
        ax2.plot(pts[:, 0], pts[:, 2], "o-", color=ci,
                 lw=1.5, label=f"t={t[ki]:.2f}s")

    if use_3d:
        ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
        ax3.set_title(f"Stick figure 3D ({tag})")
    else:
        ax3.set_xlabel("X [m]"); ax3.set_ylabel("Z [m]")
        ax3.set_title(f"Stick figure X-Z ({tag})")
        ax3.grid(True, alpha=0.3)
    ax2.set_xlabel("X [m]"); ax2.set_ylabel("Z [m]")
    ax2.set_title(f"Stick figure X-Z ({tag})")
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)

    fig.tight_layout()
    out = _res() / f"stick_snapshots_{tag}.png"
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out


def animate_stick_figure(result: dict, tag: str = "free",
                         fps: int = 25, show: bool = False) -> Path:
    """Save MP4 animation of stick-figure motion."""
    q = result["q"]
    t = result["t"]

    # Downsample to target fps
    dt = t[1] - t[0]
    step = max(1, int(round(1.0 / (fps * dt))))
    idx  = np.arange(0, len(t), step)

    all_pts = np.array([fk_points(q[k]) for k in idx])  # (M,9,3)
    x_all, y_all, z_all = all_pts[:, :, 0], all_pts[:, :, 1], all_pts[:, :, 2]

    pad = 0.05
    xlim = (x_all.min() - pad, x_all.max() + pad)
    ylim = (y_all.min() - pad, y_all.max() + pad)
    zlim = (z_all.min() - pad, z_all.max() + pad)

    fig = plt.figure(figsize=(7, 6))
    try:
        ax = fig.add_subplot(111, projection="3d")
        use_3d = True
    except Exception:
        ax = fig.add_subplot(111)
        use_3d = False

    if use_3d:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        line, = ax.plot([], [], [], "bo-", lw=2, ms=5)
        time_txt = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    else:
        ax.set_xlim(*xlim); ax.set_ylim(*zlim)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Z [m]")
        ax.grid(True, alpha=0.3)
        line, = ax.plot([], [], "bo-", lw=2, ms=5)
        time_txt = ax.text(0.05, 0.95, "", transform=ax.transAxes)
    ax.set_title(f"Arm animation ({tag})")

    def init():
        if use_3d:
            line.set_data([], [])
            line.set_3d_properties([])
        else:
            line.set_data([], [])
        return line, time_txt

    def update(frame):
        pts = all_pts[frame]
        if use_3d:
            line.set_data(pts[:, 0], pts[:, 1])
            line.set_3d_properties(pts[:, 2])
        else:
            line.set_data(pts[:, 0], pts[:, 2])
        time_txt.set_text(f"t = {t[idx[frame]]:.2f} s")
        return line, time_txt

    ani = animation.FuncAnimation(fig, update, frames=len(idx),
                                   init_func=init, blit=True,
                                   interval=int(1000 / fps))
    try:
        out = _res() / f"animation_{tag}.mp4"
        ani.save(str(out), writer="ffmpeg", fps=fps, dpi=100)
    except Exception:
        out = _res() / f"animation_{tag}.gif"
        ani.save(str(out), writer="pillow", fps=fps)
    if show:
        plt.show()
    plt.close(fig)
    return out
