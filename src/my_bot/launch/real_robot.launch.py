# Brings up the real robot: robot_state_publisher, the controller manager
# talking to the ESP32 over serial, the two controllers, and the YDLidar.
#
# Teleop is deliberately NOT launched here -- teleop_twist_keyboard needs its
# own terminal to read keystrokes. Run it separately:
#
#   ros2 run teleop_twist_keyboard teleop_twist_keyboard \
#     --ros-args -r /cmd_vel:=/diff_cont/cmd_vel_unstamped
#
# PORTS: the lidar and the ESP32 base controller are both USB serial adapters,
# and whichever enumerates first becomes /dev/ttyUSB0, so raw ttyUSB* numbers
# are a coin flip across reboots. Both are addressed by the stable names that
# scripts/setup_udev.sh installs (/dev/ydlidar here, /dev/esp32 in
# description/ros2_control.xacro). Run that script once before `make real`.
# Override per run if you need a raw port:
#
#   ros2 launch my_bot real_robot.launch.py lidar_port:=/dev/ttyUSB1

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_path = get_package_share_directory('my_bot')
    xacro_file = os.path.join(pkg_path, 'description', 'robot.urdf.xacro')
    controller_params = os.path.join(pkg_path, 'config', 'my_controllers.yaml')
    lidar_params = os.path.join(pkg_path, 'config', 'ydlidar.yaml')

    lidar_port = LaunchConfiguration('lidar_port')
    lidar_port_arg = DeclareLaunchArgument(
        'lidar_port',
        default_value='/dev/ydlidar',
        description='Serial device for the YDLidar. Defaults to the udev '
                    'symlink from scripts/setup_udev.sh so it survives '
                    're-enumeration.',
    )

    # ydlidar_ros2_driver is NOT an apt package: it is built from source
    # against the YDLidar SDK, and on a fresh board it is not there yet. An
    # absent executable takes the WHOLE launch down, so the base -- which is
    # what Day 2 needs -- would never come up. Gate it instead:
    #
    #   make real USE_LIDAR=false
    #
    # Drive and odometry work with this false; /scan does not, so SLAM and
    # Nav2 need it back on. Default stays true so the normal path is unchanged.
    use_lidar = LaunchConfiguration('use_lidar')
    use_lidar_arg = DeclareLaunchArgument(
        'use_lidar',
        default_value='true',
        description='Start the YDLidar driver. Set false before '
                    'ydlidar_ros2_driver is built, to bring up drive and '
                    'odometry alone.',
    )

    # sim_mode:=false selects the DiffDriveSerial plugin in ros2_control.xacro.
    robot_description = ParameterValue(
        Command([
            'xacro ', xacro_file,
            ' use_ros2_control:=true',
            ' sim_mode:=false',
        ]),
        value_type=str,
    )

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_path, 'launch', 'rsp.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'false',
            'use_ros2_control': 'true',
        }.items(),
    )

    # On real hardware there is no gz_ros2_control to host the controller
    # manager, so we run ros2_control_node ourselves.
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            {'robot_description': robot_description},
            controller_params,
        ],
        output='screen',
    )

    joint_broad_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_broad'],
    )

    diff_drive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_cont'],
    )

    # The interface sleeps ~2 s on configure waiting out the ESP32's
    # DTR-triggered reset, so give the manager time before spawning.
    delayed_spawners = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[
                TimerAction(
                    period=5.0,
                    actions=[joint_broad_spawner, diff_drive_spawner],
                )
            ],
        )
    )

    # Publishes /scan with BEST_EFFORT reliability. It is a LifecycleNode, but
    # the driver self-configures and self-activates on startup, so no external
    # lifecycle transitions are needed here.
    #
    # Upstream's ydlidar_launch.py also starts a static_transform_publisher for
    # base_link -> laser_frame. We deliberately do NOT: laser_joint now lives in
    # description/lidar.xacro, so robot_state_publisher owns that transform. Two
    # publishers on the same TF edge make the frame jitter between the two
    # poses, which looks like a lidar mounting problem rather than a TF one.
    lidar = LifecycleNode(
        package='ydlidar_ros2_driver',
        executable='ydlidar_ros2_driver_node',
        name='ydlidar_ros2_driver_node',
        namespace='/',
        parameters=[lidar_params, {'port': lidar_port}],
        output='screen',
        emulate_tty=True,
        condition=IfCondition(use_lidar),
    )

    return LaunchDescription([
        lidar_port_arg,
        use_lidar_arg,
        rsp,
        controller_manager,
        delayed_spawners,
        lidar,
    ])
