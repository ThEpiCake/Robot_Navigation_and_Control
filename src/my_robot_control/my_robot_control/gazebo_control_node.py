"""ROS 2 node: replays computed joint trajectory in Gazebo.

Architecture (proper ROS 2 pub/sub):
  Subscriber: /joint_states  (sensor_msgs/JointState)
              → reads actual joint positions from Gazebo simulation
  Publisher:  /arm_position_controller/commands  (std_msgs/Float64MultiArray)
              → sends desired position setpoints to ForwardCommandController

The node reads a CSV trajectory (computed offline by simulate_pid.py using
Lagrangian dynamics + Computed-Torque PID) and publishes it at the correct
wall-clock rate.  Live joint state feedback is logged for comparison.

Usage:
    ros2 run my_robot_control gazebo_control \
        --ros-args -p csv_file:=results/pid_trajectory.csv
    # or via launch file:
    ros2 launch my_robot_control playback.launch.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
try:
    from controller_manager_msgs.srv import ListControllers
except Exception:  # pragma: no cover - optional runtime dependency
    ListControllers = None


# Joint order expected by arm_position_controller (from ros2_controllers.yaml)
_CONTROLLER_JOINTS = [
    "joint_base_yaw",
    "joint_arm1_prismatic",
    "joint_arm2_prismatic",
    "joint_wrist_z1",
    "joint_wrist_y",
    "joint_wrist_z2",
]


class ArmControlNode(Node):
    """Publishes trajectory setpoints; subscribes to actual joint states."""

    def __init__(self):
        super().__init__("arm_control_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("csv_file",     "results/pid_trajectory.csv")
        self.declare_parameter("loop",         False)
        self.declare_parameter("speed_factor", 1.0)
        self.declare_parameter("wait_for_subscriber",   True)
        self.declare_parameter("preposition_enabled",   True)
        self.declare_parameter("preposition_duration",  2.0)
        self.declare_parameter("preposition_timeout",   8.0)
        self.declare_parameter("preposition_tolerance", 0.05)
        self.declare_parameter("require_controller_active", True)
        self.declare_parameter("require_preposition_success", False)
        self.declare_parameter("ready_timeout",         20.0)

        csv_path     = Path(self.get_parameter("csv_file").value)
        self._loop   = self.get_parameter("loop").value
        self._speed  = float(self.get_parameter("speed_factor").value)
        self._wait_sub = bool(self.get_parameter("wait_for_subscriber").value)
        self._preposition = bool(self.get_parameter("preposition_enabled").value)
        self._preposition_duration = max(
            0.0, float(self.get_parameter("preposition_duration").value))
        self._preposition_timeout = max(
            self._preposition_duration, float(self.get_parameter("preposition_timeout").value))
        self._preposition_tol = max(
            0.0, float(self.get_parameter("preposition_tolerance").value))
        self._require_ctrl_active = bool(
            self.get_parameter("require_controller_active").value)
        self._require_preposition_success = bool(
            self.get_parameter("require_preposition_success").value)
        self._ready_timeout = max(0.0, float(self.get_parameter("ready_timeout").value))

        # ── Load trajectory from CSV ──────────────────────────────────────────
        if not csv_path.exists():
            self.get_logger().error(f"CSV not found: {csv_path}")
            raise FileNotFoundError(str(csv_path))

        self._trajectory = self._load_csv(csv_path)
        self._idx        = 0
        self._t_start    = None          # set on first publish
        self._ready_t0   = time.monotonic()
        self._actual_q   = None          # latest joint states from Gazebo
        self._js_names   = set()         # latest joint names seen
        self._phase      = "wait_ready"  # wait_ready -> preposition -> replay
        self._phase_t0   = None
        self._q_pre_from = None
        self._q0         = self._trajectory[0][1]
        self._cm_ready_cached = (not self._require_ctrl_active) or (ListControllers is None)
        self._cm_warned_missing = False
        self._cm_last_query_t = 0.0
        self._cm_query_period = 0.5
        self._cm_future = None
        self._cm_cli = None
        if self._require_ctrl_active and ListControllers is not None:
            self._cm_cli = self.create_client(ListControllers, "/controller_manager/list_controllers")

        n_pts = len(self._trajectory)
        t_end = self._trajectory[-1][0]
        self.get_logger().info(
            f"Loaded {n_pts} trajectory points  "
            f"(duration {t_end:.2f}s, speed×{self._speed})")

        # ── Publisher: position commands → ForwardCommandController ──────────
        self._cmd_pub = self.create_publisher(
            Float64MultiArray,
            "/arm_position_controller/commands",
            10)

        # ── Subscriber: actual joint states ← Gazebo ─────────────────────────
        self._js_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            10)

        # ── Timer: publish at 100 Hz ──────────────────────────────────────────
        self._timer = self.create_timer(0.01, self._control_loop)

        self.get_logger().info("ArmControlNode ready — waiting for /joint_states …")

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState):
        """Store actual joint positions (reordered to controller joint order)."""
        name_to_pos = dict(zip(msg.name, msg.position))
        self._js_names = set(msg.name)
        self._actual_q = [
            name_to_pos.get(j, 0.0) for j in _CONTROLLER_JOINTS
        ]

    def _ready_for_replay(self) -> bool:
        """True when inputs/outputs needed for stable playback are ready."""
        # Need real joint state with all expected joints.
        if self._actual_q is None:
            return False
        if not all(j in self._js_names for j in _CONTROLLER_JOINTS):
            return False
        # Optional: wait until controller subscribes to commands topic.
        if self._wait_sub and self._cmd_pub.get_subscription_count() < 1:
            return False
        if not self._cm_ready_cached:
            return False
        return True

    def _poll_controller_state(self, now: float):
        """Update cached readiness of controllers through controller_manager."""
        if self._cm_ready_cached:
            return
        if self._cm_cli is None:
            if not self._cm_warned_missing:
                self.get_logger().warn(
                    "controller_manager_msgs not available; skipping active-controller check.")
                self._cm_warned_missing = True
            self._cm_ready_cached = True
            return
        if not self._cm_cli.service_is_ready():
            return

        # Dispatch periodic async query.
        if self._cm_future is None and (now - self._cm_last_query_t) >= self._cm_query_period:
            req = ListControllers.Request()
            self._cm_future = self._cm_cli.call_async(req)
            self._cm_last_query_t = now
            return

        # Consume response when available.
        if self._cm_future is not None and self._cm_future.done():
            ok = False
            try:
                resp = self._cm_future.result()
                state = {c.name: c.state for c in resp.controller}
                ok = (state.get("joint_state_broadcaster") == "active" and
                      state.get("arm_position_controller") == "active")
            except Exception:
                ok = False
            self._cm_ready_cached = ok
            self._cm_future = None

    def _control_loop(self):
        """Publish trajectory setpoints with robust startup sequencing."""
        now = time.monotonic()

        # Phase 1: wait until controller + joint states are really ready.
        if self._phase == "wait_ready":
            self._poll_controller_state(now)
            if self._ready_for_replay():
                self._q_pre_from = list(self._actual_q)
                if self._preposition and self._preposition_duration > 1e-3:
                    self._phase = "preposition"
                    self._phase_t0 = now
                    self.get_logger().info(
                        f"Ready. Pre-positioning to first waypoint over "
                        f"{self._preposition_duration:.2f}s.")
                    return
                self._phase = "replay"
                self._phase_t0 = now
                self._t_start = now
                self.get_logger().info("Ready. Starting trajectory replay.")
                return

            if now - self._ready_t0 > self._ready_timeout:
                self.get_logger().warn(
                    "Startup readiness timeout; starting replay anyway.")
                self._phase = "replay"
                self._phase_t0 = now
                self._t_start = now
                return
            return

        # Phase 2: smooth move from actual pose to first trajectory point.
        if self._phase == "preposition":
            if self._q_pre_from is None:
                self._q_pre_from = self._actual_q if self._actual_q is not None else self._q0
            alpha = (now - self._phase_t0) / self._preposition_duration
            alpha = max(0.0, min(1.0, alpha))
            q_cmd = [
                (1.0 - alpha) * self._q_pre_from[i] + alpha * self._q0[i]
                for i in range(6)
            ]
            msg = Float64MultiArray()
            msg.data = q_cmd
            self._cmd_pub.publish(msg)

            reached = False
            if self._actual_q is not None:
                pre_err = [abs(self._q0[i] - self._actual_q[i]) for i in range(6)]
                reached = max(pre_err) <= self._preposition_tol

            timed_out = (now - self._phase_t0) >= self._preposition_timeout

            if reached or alpha >= 1.0 or timed_out:
                if timed_out and not reached and self._require_preposition_success:
                    self.get_logger().error(
                        "Pre-position failed to reach tolerance. Aborting replay.")
                    self._timer.cancel()
                    return
                self._phase = "replay"
                self._phase_t0 = now
                self._t_start = now
                self._idx = 0
                if reached:
                    self.get_logger().info("Pre-position reached tolerance. Starting trajectory replay.")
                elif timed_out:
                    self.get_logger().warn("Pre-position timeout. Starting trajectory replay anyway.")
                else:
                    self.get_logger().info("Pre-position complete. Starting trajectory replay.")
            return

        # Phase 3: replay trajectory with time base anchored at replay start.
        if self._t_start is None:
            self._t_start = now

        elapsed = (now - self._t_start) * self._speed

        # Advance trajectory index to match elapsed time
        while (self._idx < len(self._trajectory) - 1 and
               self._trajectory[self._idx + 1][0] <= elapsed):
            self._idx += 1

        t_traj, q_des = self._trajectory[self._idx]

        # Publish desired position
        msg = Float64MultiArray()
        msg.data = q_des
        self._cmd_pub.publish(msg)

        # Log tracking error when actual states are available
        if self._actual_q is not None:
            err = [abs(q_des[i] - self._actual_q[i]) for i in range(6)]
            max_err = max(err)
            if max_err > 0.05:               # log only when tracking error is large
                self.get_logger().warn(
                    f"t={t_traj:.2f}s  max tracking error = {max_err:.4f}")

        # End of trajectory
        if self._idx >= len(self._trajectory) - 1:
            if self._loop:
                self.get_logger().info("Trajectory complete — looping.")
                self._idx    = 0
                self._t_start = time.monotonic()
            else:
                self.get_logger().info("Trajectory complete. Holding final pose.")
                self._timer.cancel()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_csv(path: Path) -> list[tuple[float, list[float]]]:
        """Load (t, [q1..q6]) from CSV written by simulate_pid.py."""
        trajectory = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t  = float(row["t"])
                qs = [float(row[f"q{i+1}"]) for i in range(6)]
                trajectory.append((t, qs))
        return trajectory


def main(args=None):
    rclpy.init(args=args)
    node = ArmControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
