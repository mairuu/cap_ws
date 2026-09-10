# Nav2 + twist_mux: the layer that actually drives the robot.
#
# slam_toolbox only answers "where am I and what does the room look like" -- it
# publishes map -> odom and /map and nothing else. This launch adds the part
# that turns a goal into wheel velocities: global planner, local planner
# (DWB), the two costmaps, recovery behaviours, and the behaviour tree.
#
# Bring the robot and the mapper up first, in their own terminals:
#
#   real robot:  make real
#                make slam
#                make nav
#
#   simulation:  make sim
#                make slam SIM_TIME=true
#                make nav  SIM_TIME=true
#
# Then either click "2D Goal Pose" in RViz, or start `make explore` to have the
# robot pick its own goals.
#
# ---------------------------------------------------------------------------
# CMD_VEL CHAIN. Nav2 does not publish where diff_cont listens, so twist_mux
# closes the gap and arbitrates at the same time:
#
#   controller_server -> /cmd_vel_nav -> velocity_smoother -> /cmd_vel --+
#                                                                        +-> twist_mux
#   teleop_twist_keyboard ------------------> /cmd_vel_teleop -----------+       |
#                                                                                v
#                                                        /diff_cont/cmd_vel_unstamped
#
# (The first two hops are nav2_bringup's own internal remaps, not ours.)
# Teleop outranks Nav2, so grabbing the keyboard overrides an autonomous run --
# see config/twist_mux.yaml. That is the only e-stop this robot has.
# ---------------------------------------------------------------------------

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_path = get_package_share_directory('my_bot')
    nav2_bringup_path = get_package_share_directory('nav2_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use the /clock topic instead of wall time. true only when '
                    'running against launch_sim.launch.py. Must match the value '
                    'given to slam.launch.py, or TF lookups silently fail.',
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_path, 'config', 'nav2_params.yaml'),
        description='Full path to the Nav2 parameters file.',
    )

    # navigation_launch.py is the SLAM-compatible half of nav2_bringup: planner,
    # controller, costmaps, behaviours, BT navigator, lifecycle manager.
    #
    # Deliberately NOT bringup_launch.py or localization_launch.py -- those add
    # amcl and map_server, and both would fight slam_toolbox for the map -> odom
    # transform. Two publishers on that edge make the robot's pose jump.
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_path, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            # The params file already carries autostart-friendly settings; the
            # lifecycle manager still needs to be told to bring the stack up.
            'autostart': 'true',
        }.items(),
    )

    # cmd_vel_out is twist_mux's own output topic name; remap it onto the topic
    # diff_cont actually subscribes to (use_stamped_vel:false in
    # config/my_controllers.yaml selects the _unstamped one).
    twist_mux = Node(
        package='twist_mux',
        executable='twist_mux',
        name='twist_mux',
        output='screen',
        parameters=[
            os.path.join(pkg_path, 'config', 'twist_mux.yaml'),
            {'use_sim_time': use_sim_time},
        ],
        remappings=[('cmd_vel_out', '/diff_cont/cmd_vel_unstamped')],
    )

    return LaunchDescription([
        use_sim_time_arg,
        params_file_arg,
        twist_mux,
        nav2,
    ])
