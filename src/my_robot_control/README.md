# my_robot_control — Part 1 Dynamics, PID, and Shelf Demo

BGU "Robots Motion Planning and Control" mini-project (Part 1).

Robot model: 6-DOF `R-P-P-R-R-R` arm with a Z-Y-Z wrist.

## Scope

- Dynamics are derived with the Lagrangian method and simulated with a custom RK4 integrator.
- Closed-loop control uses Computed-Torque + PID.
- Shelf motion demo plans a collision-free path in C-space from A to B (automatic waypoints).
- A local repulsive navigation layer (`tau_rep = -J^T grad(U_rep)`) is added during tracking.

## Package Highlights

- `my_robot_control/dynamics.py`: symbolic `M(q), C(q,qdot), G(q)` with SymPy + lambdify cache.
- `my_robot_control/integrator.py`: RK4 simulation and energy utilities.
- `my_robot_control/controllers.py`: Computed-Torque PID and trajectory generator.
- `my_robot_control/navigation.py`: potential-field repulsion for shelf divider.
- `my_robot_control/simulate_free.py`: zero-input free response (`tau=0`).
- `my_robot_control/simulate_pid.py`: shelf A->B demo with automatic C-space planning + repulsion.
- `my_robot_control/gazebo_control_node.py`: replay CSV trajectory in Gazebo.

## Build

```bash
cd ~/Master_HW
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run

1) Optional one-time dynamics derivation:

```bash
python3 -m my_robot_control.dynamics
```

2) Free response:

```bash
python3 -m my_robot_control.simulate_free
# or: ros2 run my_robot_control simulate_free
```

3) PID + navigation shelf demo:

```bash
python3 -m my_robot_control.simulate_pid
# or: ros2 run my_robot_control simulate_pid
```

4) One-command full demo (recommended):

```bash
ros2 launch my_robot_control playback.launch.py
```

This single launch does:
- start Gazebo bringup,
- generate `results/pid_trajectory.csv` (via `simulate_pid`),
- then start playback automatically.

Useful arguments:

```bash
# replay an existing CSV without re-running simulation
ros2 launch my_robot_control playback.launch.py generate_trajectory:=false

# if Gazebo is already running, do not launch/spawn again (prevents duplicates)
ros2 launch my_robot_control playback.launch.py \
    launch_gazebo:=false generate_trajectory:=false

# if you keep Gazebo open, set launch_gazebo:=false to avoid a second bringup

# optional cleanup of stale renamed robot entities (_0, _1, ...)
ros2 launch my_robot_control playback.launch.py cleanup_renamed_duplicates:=true

# custom workspace/results location
ros2 launch my_robot_control playback.launch.py \
    workspace_dir:=/home/thepicake/Master_HW
```

5) Manual Gazebo playback (equivalent split into 2 terminals):

```bash
# Terminal 1
ros2 launch my_robot_bringup my_robot_gazebo.launch.xml

# Terminal 2
ros2 run my_robot_control gazebo_control --ros-args \
    -p csv_file:=/home/thepicake/Master_HW/results/pid_trajectory.csv
```

Legacy wrapper (playback only):

```bash
ros2 launch my_robot_bringup my_robot_full_demo.launch.xml
```

## Configuration

Main config: `config/sim.yaml`.  Two input modes are supported:

### Mode A — Pose waypoints (preferred)

Specify task-space poses; IK converts each pose → joint config internally.

```yaml
pose_waypoints:
  - name: A_shelf1
    position: [0.467, 0.0, 0.133]       # x, y, z [m] in world frame
    rpy:      [3.1416, 0.7854, 3.1416]  # roll, pitch, yaw [rad]
    duration: 1.5                        # segment duration [s]
  - name: B_shelf2
    position: [0.467, 0.0, 0.283]
    rpy:      [3.1416, 0.7854, 3.1416]
    duration: 1.0
```

**Orientation convention**: ROS fixed-axis RPY → `R = Rz(yaw) · Ry(pitch) · Rx(roll)`

The natural arm EE orientation (wrist at zero: q4=q5=q6=0) is
`rpy = [π, π/4, π]` ≈ `[3.1416, 0.7854, 3.1416]`.

**Internal pipeline:**
```
pose_waypoints
   └─ rpy_to_rotation_matrix(r,p,y)        → R
   └─ IK(start), IK(goal)                  → q_start, q_goal
   └─ C-space planner (biRRT)              → [q0, q1, ..., qN]
   └─ quintic_trajectory(q_i, q_{i+1}, T)  → q_d(t)
   └─ ComputedTorquePID + τ_rep             → τ(t)
   └─ RK4 integrator                        → q(t)
```

### Mode B — Joint-space waypoints (legacy)

```yaml
waypoints:
  - name: A
    q: [0.0, 0.05, 0.05, 0.0, 0.0, 0.0]
    duration: 1.5
```

**Precedence**: `pose_waypoints` takes priority; `waypoints` is used only if `pose_waypoints` is absent.

### Other config keys

| Key | Purpose |
|---|---|
| `q_start` | Free-response simulation start (`simulate_free.py`) |
| `planning.enabled` | If `true`, only start+goal are taken and intermediate waypoints are planned automatically in C-space. |
| `planning.arm_surface_radius` | Effective arm radius used for collision checks (clearance -> physical surface gap). |
| `planning.step_norm`, `planning.edge_resolution_norm`, `planning.max_iters` | biRRT resolution and search budget in normalized joint space. |
| `planning.shortcut_trials` | Number of random path-shortening attempts after planning. |
| `planning.seconds_per_unit` | Converts C-space segment length to segment duration. |
| `navigation.obstacle_inflation` | Inflation used by repulsive potential field. |
| `navigation.rho0` | Repulsion activation distance from obstacles. |
| `navigation.eta` | Repulsive potential gain. Torques are clamped at ±50 N·m so `eta` saturates quickly; keep < 10. |
| `navigation.min_clearance_threshold` | Post-run safety floor [m] on full shelf obstacles. |

### Collision proxy model (8 points)

`navigation.py` checks 8 points along the arm kinematic chain — not just the EE:

```
arm2_top → wrist_z1 → wrist_y → wrist_z2 → gripper_base → left_finger → right_finger → EE
```

This covers all link volumes near the shelf.  Each point has its own Jacobian so
`τ_rep = −Σ Jᵢ^T · ∇U_rep(pᵢ)` is computed correctly.

### Collision handling strategy

The run now uses **automatic C-space planning** from `A` to `B`:

1. Resolve only start/goal poses with IK.
2. Plan a collision-free joint path with a sampling-based planner (biRRT).
3. Shortcut/smooth the found path.
4. Convert planned nodes to quintic trajectory segments.
5. Run a swept-path pre-check and a post-run safety check against shelf obstacles.

This removes manual `pre_B` tuning and keeps obstacle avoidance in the planner itself.

Controller gains: `config/pid_gains.yaml`

## Outputs

Generated in workspace `results/`:

| File | Contents |
|---|---|
| `joint_states_pid.png` | q_i(t) desired vs actual |
| `velocities_pid.png` | q̇_i(t) per joint |
| `tracking_error_pid.png` | e_i(t) and τ_i(t) with segment markers |
| `clearance_pid.png` | Min distance arm → shelf obstacles over time |
| `ee_path_pid.png` | EE X-Z path A→B |
| `stick_snapshots_pid.png` | Arm posture snapshots |
| `animation_pid.gif` | Animated stick figure |
| `pid_trajectory.csv` | Joint trajectory for Gazebo playback |
| `derivation.md` | Symbolic EOM (M, C, G, V) |

## Notes

- `simulate_free.py` resolves `q_start` from: `q_start` field → first waypoint → IK of `ee_A`.
- Repulsion (`τ_rep`) is clamped to ±50 N·m to avoid overpowering PID tracking.
- IK branch continuity: each waypoint seeds the IK solver from the previous solution.
