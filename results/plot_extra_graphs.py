#!/usr/bin/env python3
"""Additional 3D and analysis graphs for Part 2 report."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
CSV = OUT / "drone_flight_data.csv"
df = pd.read_csv(CSV)

STATE_COLORS = {
    "TAKEOFF_CLIMB":      "#4CAF50",
    "TAKEOFF_SETTLE":     "#81C784",
    "TAKEOFF_LOOKAROUND": "#A5D6A7",
    "SCAN":               "#2196F3",
    "TARGET_INSPECT":     "#FF9800",
    "RETURN":             "#9C27B0",
    "LANDING":            "#E91E63",
    "DONE":               "#607D8B",
}
DEFAULT_COLOR = "#607D8B"

def sc(s):
    return STATE_COLORS.get(s, DEFAULT_COLOR)

# ── 1. 3D Trajectory (N, E, Alt) ──────────────────────────────────────────────
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

for state in df["mission_state"].unique():
    seg = df[df["mission_state"] == state]
    ax.plot(seg["east_m"], seg["north_m"], seg["alt_m"],
            '.', markersize=2, color=sc(state), label=state, alpha=0.75)

# Start/end markers
ax.scatter(df["east_m"].iloc[0], df["north_m"].iloc[0], df["alt_m"].iloc[0],
           c='green', s=120, marker='o', zorder=10, label='Start')
ax.scatter(df["east_m"].iloc[-1], df["north_m"].iloc[-1], df["alt_m"].iloc[-1],
           c='red', s=120, marker='s', zorder=10, label='End (Landing)')

# Zone boundary lines
for east_line in [7.5, 13.5]:
    for north_val in [-8.5, 8.5]:
        ax.plot([east_line, east_line], [north_val, north_val], [0, 8],
                color='gray', alpha=0.3, linewidth=1)
    ax.plot([east_line, east_line], [-8.5, 8.5], [0, 0], color='gray', alpha=0.3, linewidth=1)

ax.set_xlabel("East [m]", fontsize=11, labelpad=8)
ax.set_ylabel("North [m]", fontsize=11, labelpad=8)
ax.set_zlabel("Altitude [m AGL]", fontsize=11, labelpad=8)
ax.set_title("3D Task Space — Drone Flight Trajectory", fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=8, markerscale=3)
ax.view_init(elev=25, azim=-60)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "drone_3d_trajectory.png", dpi=150)
plt.close(fig)
print("Saved drone_3d_trajectory.png")

# ── 2. Coverage % vs Time (with state shading) ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))

# Shade mission states
prev_state = None
t_start = None
for i, row in df.iterrows():
    if row["mission_state"] != prev_state:
        if prev_state is not None:
            ax.axvspan(t_start, row["time_s"], alpha=0.12, color=sc(prev_state))
        prev_state = row["mission_state"]
        t_start = row["time_s"]
if prev_state is not None:
    ax.axvspan(t_start, df["time_s"].iloc[-1], alpha=0.12, color=sc(prev_state))

ax.plot(df["time_s"], df["coverage_pct"], color="#1565C0", linewidth=2, label="Coverage %")
ax.set_xlabel("Time [s]", fontsize=12)
ax.set_ylabel("Coverage [%]", fontsize=12)
ax.set_title("Mission Coverage vs Time (by State)", fontsize=13, fontweight='bold')
ax.set_ylim(0, 80)
ax.axhline(69.5, color='red', linestyle='--', alpha=0.6, label="Final: 69.5%")

# Legend patches for states
patches = [mpatches.Patch(color=sc(s), label=s, alpha=0.7)
           for s in df["mission_state"].unique()]
patches.append(plt.Line2D([0], [0], color='red', linestyle='--', label='Final 69.5%'))
ax.legend(handles=patches + [plt.Line2D([0],[0], color='#1565C0', lw=2, label='Coverage %')],
          fontsize=8, loc='upper left', ncol=2)

ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "drone_coverage_timeline.png", dpi=150)
plt.close(fig)
print("Saved drone_coverage_timeline.png")

# ── 3. State Machine Gantt Chart ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))

state_order = ["TAKEOFF_CLIMB","TAKEOFF_SETTLE","TAKEOFF_LOOKAROUND",
               "SCAN","TARGET_INSPECT","RETURN","LANDING","DONE"]

segments = []
prev = df.iloc[0]
cur_state = prev["mission_state"]
t_start = prev["time_s"]
for _, row in df.iterrows():
    if row["mission_state"] != cur_state:
        segments.append((cur_state, t_start, row["time_s"]))
        cur_state = row["mission_state"]
        t_start = row["time_s"]
segments.append((cur_state, t_start, df["time_s"].iloc[-1]))

for (state, ts, te) in segments:
    y = state_order.index(state) if state in state_order else len(state_order)
    ax.barh(y, te - ts, left=ts, height=0.6, color=sc(state), alpha=0.8)

ax.set_yticks(range(len(state_order)))
ax.set_yticklabels(state_order, fontsize=10)
ax.set_xlabel("Time [s]", fontsize=12)
ax.set_title("Mission State Machine — Timeline (Gantt)", fontsize=13, fontweight='bold')
ax.set_xlim(0, df["time_s"].max())
ax.grid(True, axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "drone_state_gantt.png", dpi=150)
plt.close(fig)
print("Saved drone_state_gantt.png")

# ── 4. 3D Altitude Heatmap (exploration density) ──────────────────────────────
scan_df = df[df["mission_state"] == "SCAN"].copy()

fig = plt.figure(figsize=(12, 5))

# Left: 2D density heatmap (N-E)
ax1 = fig.add_subplot(121)
h, xe, ye = np.histogram2d(scan_df["east_m"], scan_df["north_m"],
                            bins=40, range=[[-1,28],[-7.5,7.5]])
im = ax1.imshow(h.T, origin='lower', aspect='auto',
                extent=[-1,28,-7.5,7.5],
                cmap='hot_r', interpolation='bilinear')
plt.colorbar(im, ax=ax1, label='Visit count')
ax1.set_xlabel("East [m]", fontsize=11)
ax1.set_ylabel("North [m]", fontsize=11)
ax1.set_title("Exploration Density — XY Plane", fontsize=12, fontweight='bold')
# Room boundaries
for x_line in [7.5, 13.5]:
    ax1.axvline(x_line, color='cyan', linestyle='--', alpha=0.7, linewidth=1.5)
ax1.text(3.5, 6.8, 'Room A', color='cyan', fontsize=9, ha='center')
ax1.text(10.5, 6.8, 'Corridor', color='cyan', fontsize=9, ha='center')
ax1.text(21, 6.8, 'Room B', color='cyan', fontsize=9, ha='center')

# Right: Altitude distribution (histogram by state)
ax2 = fig.add_subplot(122)
scan_layers = scan_df["alt_m"].values
bins = np.linspace(0.5, 8.5, 17)
ax2.hist(scan_layers, bins=bins, color='#2196F3', alpha=0.75, edgecolor='white', label='SCAN')
inspect_df = df[df["mission_state"] == "TARGET_INSPECT"]
if len(inspect_df):
    ax2.hist(inspect_df["alt_m"].values, bins=bins, color='#FF9800',
             alpha=0.7, edgecolor='white', label='TARGET_INSPECT')
ax2.set_xlabel("Altitude [m AGL]", fontsize=11)
ax2.set_ylabel("Sample count", fontsize=11)
ax2.set_title("Scan Layer Distribution", fontsize=12, fontweight='bold')
for layer_z in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
    ax2.axvline(layer_z, color='gray', linestyle=':', alpha=0.5)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "drone_exploration_density.png", dpi=150)
plt.close(fig)
print("Saved drone_exploration_density.png")

# ── 5. Distance to Goal over Time + VFH heading combined ──────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

scan_only = df[df["mission_state"].isin(["SCAN", "TARGET_INSPECT"])]

ax1.plot(df["time_s"], df["dist_to_goal_m"], color="#E91E63",
         linewidth=1.2, alpha=0.8, label="Distance to goal [m]")
ax1.set_ylabel("Distance to Goal [m]", fontsize=11)
ax1.set_title("Navigation Performance — Goal Distance & VFH Heading", fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 35)

valid_hdg = df.dropna(subset=["vfh_heading_deg"])
ax2.plot(valid_hdg["time_s"], valid_hdg["vfh_heading_deg"],
         color="#2196F3", linewidth=1.2, alpha=0.85, label="VFH heading [deg]")
ax2.axhline(0, color="gray", linestyle="--", alpha=0.4)
ax2.set_ylabel("VFH Heading [deg]", fontsize=11)
ax2.set_xlabel("Time [s]", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-200, 380)

fig.tight_layout()
fig.savefig(OUT / "drone_nav_performance.png", dpi=150)
plt.close(fig)
print("Saved drone_nav_performance.png")

print("\nAll extra graphs generated.")
