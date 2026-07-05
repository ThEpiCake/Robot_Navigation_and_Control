"""One-command demo launch: generate trajectory + Gazebo playback."""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _guess_workspace_dir() -> str:
    """Best-effort workspace root detection.

    Works for both source and installed launch paths by searching upwards for a
    directory that contains both `src/` and `install/`.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src").is_dir() and (parent / "install").is_dir():
            return str(parent)
    return str(Path.cwd())


def _resolve_bringup_launch_path(workspace_dir: str) -> tuple[AnyLaunchDescriptionSource, str]:
    """Prefer local workspace bringup launch to avoid overlay package mismatches."""
    local = Path(workspace_dir) / "src" / "my_robot_bringup" / "launch" / "my_robot_gazebo.launch.xml"
    if local.exists():
        return AnyLaunchDescriptionSource(str(local)), str(local)
    # Fallback to package share resolution.
    pkg_path = PathJoinSubstitution([
        FindPackageShare("my_robot_bringup"),
        "launch", "my_robot_gazebo.launch.xml",
    ])
    return AnyLaunchDescriptionSource(pkg_path), "package://my_robot_bringup/launch/my_robot_gazebo.launch.xml"


def generate_launch_description():
    workspace_default = _guess_workspace_dir()
    bringup_source, bringup_label = _resolve_bringup_launch_path(workspace_default)

    workspace_arg = DeclareLaunchArgument(
        "workspace_dir",
        default_value=workspace_default,
        description="Workspace root (used as CWD for simulation output files)",
    )

    results_dir_arg = DeclareLaunchArgument(
        "results_dir",
        default_value=PathJoinSubstitution([
            LaunchConfiguration("workspace_dir"), "results"
        ]),
        description="Directory for simulation outputs (CSV/plots)",
    )

    source_py_path = PathJoinSubstitution([
        LaunchConfiguration("workspace_dir"), "src", "my_robot_control",
    ])
    extend_pythonpath = SetEnvironmentVariable(
        name="PYTHONPATH",
        value=[
            source_py_path, ":",
            EnvironmentVariable("PYTHONPATH", default_value=""),
        ],
    )

    generate_arg = DeclareLaunchArgument(
        "generate_trajectory",
        default_value="true",
        description="If true: run simulate_pid before playback",
    )

    csv_file_arg = DeclareLaunchArgument(
        "csv_file",
        default_value=PathJoinSubstitution([
            LaunchConfiguration("results_dir"), "pid_trajectory.csv"
        ]),
        description="Path to joint trajectory CSV",
    )

    loop_arg = DeclareLaunchArgument(
        "loop", default_value="false",
        description="Loop playback indefinitely",
    )

    speed_arg = DeclareLaunchArgument(
        "speed_factor", default_value="1.0",
        description="Playback speed multiplier",
    )

    preposition_arg = DeclareLaunchArgument(
        "preposition_enabled", default_value="true",
        description="Align robot to first trajectory point before timed replay",
    )

    preposition_duration_arg = DeclareLaunchArgument(
        "preposition_duration", default_value="2.0",
        description="Seconds for smooth move to first waypoint before replay",
    )

    preposition_timeout_arg = DeclareLaunchArgument(
        "preposition_timeout", default_value="8.0",
        description="Max seconds allowed for pre-position convergence",
    )

    preposition_tolerance_arg = DeclareLaunchArgument(
        "preposition_tolerance", default_value="0.05",
        description="Max joint error tolerance (rad/m) before replay starts",
    )

    require_controller_active_arg = DeclareLaunchArgument(
        "require_controller_active", default_value="true",
        description="Wait for active arm/joint-state controllers before replay",
    )

    require_preposition_success_arg = DeclareLaunchArgument(
        "require_preposition_success", default_value="false",
        description="If true, abort replay when pre-position does not converge",
    )

    startup_delay_arg = DeclareLaunchArgument(
        "startup_delay", default_value="3.0",
        description="Seconds to wait before starting playback node",
    )
    startup_delay_no_generate_arg = DeclareLaunchArgument(
        "startup_delay_no_generate", default_value="10.0",
        description="Seconds to wait before playback when generate_trajectory=false",
    )

    launch_gazebo_arg = DeclareLaunchArgument(
        "launch_gazebo", default_value="true",
        description="If true: launch Gazebo bringup from this launch file",
    )

    spawn_robot_arg = DeclareLaunchArgument(
        "spawn_robot", default_value="true",
        description="If true: spawn robot entity in Gazebo (disable if already spawned)",
    )

    # Reuse existing Gazebo bringup.
    bringup_info = LogInfo(msg=f"[playback.launch] Gazebo bringup source: {bringup_label}")
    gazebo_launch = IncludeLaunchDescription(
        bringup_source,
        condition=IfCondition(LaunchConfiguration("launch_gazebo")),
        launch_arguments={
            "spawn_robot": LaunchConfiguration("spawn_robot"),
        }.items(),
    )

    generate_traj = ExecuteProcess(
        cmd=["python3", "-u", "-m", "my_robot_control.simulate_pid"],
        cwd=LaunchConfiguration("workspace_dir"),
        output="screen",
        emulate_tty=True,
        additional_env={
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib",
        },
        condition=IfCondition(LaunchConfiguration("generate_trajectory")),
    )

    playback_node_after_gen = Node(
        package="my_robot_control",
        executable="gazebo_control",
        name="arm_control_node",
        output="screen",
        parameters=[{
            "csv_file": LaunchConfiguration("csv_file"),
            "loop": LaunchConfiguration("loop"),
            "speed_factor": LaunchConfiguration("speed_factor"),
            "preposition_enabled": LaunchConfiguration("preposition_enabled"),
            "preposition_duration": LaunchConfiguration("preposition_duration"),
            "preposition_timeout": LaunchConfiguration("preposition_timeout"),
            "preposition_tolerance": LaunchConfiguration("preposition_tolerance"),
            "require_controller_active": LaunchConfiguration("require_controller_active"),
            "require_preposition_success": LaunchConfiguration("require_preposition_success"),
        }],
    )

    playback_node_direct = TimerAction(
        period=LaunchConfiguration("startup_delay_no_generate"),
        actions=[Node(
            package="my_robot_control",
            executable="gazebo_control",
            name="arm_control_node",
            output="screen",
            parameters=[{
                "csv_file": LaunchConfiguration("csv_file"),
                "loop": LaunchConfiguration("loop"),
                "speed_factor": LaunchConfiguration("speed_factor"),
                "preposition_enabled": LaunchConfiguration("preposition_enabled"),
                "preposition_duration": LaunchConfiguration("preposition_duration"),
                "preposition_timeout": LaunchConfiguration("preposition_timeout"),
                "preposition_tolerance": LaunchConfiguration("preposition_tolerance"),
                "require_controller_active": LaunchConfiguration("require_controller_active"),
                "require_preposition_success": LaunchConfiguration("require_preposition_success"),
            }],
        )],
        condition=UnlessCondition(LaunchConfiguration("generate_trajectory")),
    )

    start_playback_after_gen = RegisterEventHandler(
        OnProcessExit(
            target_action=generate_traj,
            on_exit=[
                # Allow Gazebo/controllers a short stabilization before playback.
                TimerAction(period=LaunchConfiguration("startup_delay"), actions=[playback_node_after_gen]),
            ],
        ),
        condition=IfCondition(LaunchConfiguration("generate_trajectory")),
    )

    return LaunchDescription([
        workspace_arg,
        results_dir_arg,
        extend_pythonpath,
        generate_arg,
        csv_file_arg,
        loop_arg,
        speed_arg,
        preposition_arg,
        preposition_duration_arg,
        preposition_timeout_arg,
        preposition_tolerance_arg,
        require_controller_active_arg,
        require_preposition_success_arg,
        startup_delay_arg,
        startup_delay_no_generate_arg,
        launch_gazebo_arg,
        spawn_robot_arg,
        bringup_info,
        gazebo_launch,
        generate_traj,
        playback_node_direct,
        start_playback_after_gen,
    ])
