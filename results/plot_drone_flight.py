#!/usr/bin/env python3
"""Generate flight graphs from drone_flight_data.csv."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
CSV = OUT / "drone_flight_data.csv"
df = pd.read_csv(CSV)

STATE_COLORS = {
    "TAKEOFF_LOOKAROUND": "#4CAF50",
    "SCAN": "#2196F3",
    "RETURN": "#FF9800",
    "LANDING": "#9C27B0",
}
DEFAULT_COLOR = "#607D8B"

def state_color(s):
    return STATE_COLORS.get(s, DEFAULT_COLOR)

# ── 1. Task Space — XY Trajectory ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
states = df["mission_state"].unique()
for state in states:
    seg = df[df["mission_state"] == state]
    c = state_color(state)
    ax.plot(seg["east_m"], seg["north_m"], ".", markersize=3, color=c, label=state)

ax.plot(df["east_m"].iloc[0], df["north_m"].iloc[0], "go", markersize=10,
        label="Start", zorder=5)
ax.plot(df["east_m"].iloc[-1], df["north_m"].iloc[-1], "rs", markersize=10,
        label="End", zorder=5)
ax.set_xlabel("East [m]", fontsize=12)
ax.set_ylabel("North [m]", fontsize=12)
ax.set_title("Task Space — Drone XY Trajectory", fontsize=13)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT / "drone_task_xy.png", dpi=150)
plt.close(fig)
print("Saved drone_task_xy.png")

# ── 2. Task Space — Altitude vs Time ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
colors = [state_color(s) for s in df["mission_state"]]
for i in range(len(df) - 1):
    ax.plot(df["time_s"].iloc[i:i+2], df["alt_m"].iloc[i:i+2],
            color=colors[i], linewidth=1.8)

patches = [mpatches.Patch(color=state_color(s), label=s) for s in states]
ax.legend(handles=patches, fontsize=9, loc="upper left")
ax.set_xlabel("Time [s]", fontsize=12)
ax.set_ylabel("Altitude [m AGL]", fontsize=12)
ax.set_title("Task Space — Drone Altitude over Time", fontsize=13)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Layer 1 (1.0 m)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "drone_task_altitude.png", dpi=150)
plt.close(fig)
print("Saved drone_task_altitude.png")

# ── 3. Config Space — VFH Heading vs Time ────────────────────────────────────
scan_df = df[df["mission_state"] == "SCAN"].copy()
valid_hdg = scan_df.dropna(subset=["vfh_heading_deg"])

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(valid_hdg["time_s"], valid_hdg["vfh_heading_deg"],
        color="#2196F3", linewidth=1.5, label="VFH Heading")
ax.set_xlabel("Time [s]", fontsize=12)
ax.set_ylabel("VFH Heading [deg]", fontsize=12)
ax.set_title("Configuration Space — VFH Heading during SCAN", fontsize=13)
ax.set_ylim(-190, 370)
ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "drone_config_heading.png", dpi=150)
plt.close(fig)
print("Saved drone_config_heading.png")

# ── 4. Config Space — Blocked Sectors vs Time ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

ax1.fill_between(df["time_s"], df["n_blocked_raw"],
                 alpha=0.6, color="#FF5722", label="Raw blocked sectors")
ax1.fill_between(df["time_s"], df["n_blocked_dilated"],
                 alpha=0.4, color="#FF9800", label="After dilation (±3 sectors)")
ax1.set_ylabel("Number of blocked sectors", fontsize=11)
ax1.set_title("Configuration Space — VFH Blocked Sectors over Time", fontsize=13)
ax1.legend(fontsize=9)
ax1.set_ylim(0, 75)
ax1.axhline(72, color="red", linestyle=":", alpha=0.5, label="Maximum (72)")
ax1.grid(True, alpha=0.3)

ax2.fill_between(df["time_s"], df["blocked_fraction"] * 100,
                 alpha=0.7, color="#9C27B0", label="Blocked fraction")
ax2.set_ylabel("Blocked fraction [%]", fontsize=11)
ax2.set_xlabel("Time [s]", fontsize=12)
ax2.set_ylim(0, 105)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

for ax in (ax1, ax2):
    for state in ["TAKEOFF_LOOKAROUND", "SCAN"]:
        seg = df[df["mission_state"] == state]
        if len(seg) > 0:
            ax.axvspan(seg["time_s"].iloc[0], seg["time_s"].iloc[-1],
                      alpha=0.05, color=state_color(state))

fig.tight_layout()
fig.savefig(OUT / "drone_config_sectors.png", dpi=150)
plt.close(fig)
print("Saved drone_config_sectors.png")

print("\nAll graphs generated successfully.")
