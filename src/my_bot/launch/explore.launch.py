# Frontier exploration: the robot picks its own goals.
#
# This is the TOP layer and it drives nothing itself -- it sends NavigateToPose
# goals to Nav2. Everything below it must already be running:
#
#   real robot:  make real           (base + lidar)
#                make slam           (map + map->odom)
#                make nav            (Nav2 + twist_mux)
#                make explore        <- this
#
#   simulation:  make sim
#                make slam    SIM_TIME=true
#                make nav     SIM_TIME=true
#                make explore SIM_TIME=true
#
# Sanity check before blaming this node: can you click "2D Goal Pose" in RViz
# and have the robot drive there? If not, the problem is in Nav2, not here.
#
# We do NOT include explore_lite's own explore.launch.py. That launch hardcodes
# its package's params.yaml (tuned for a TurtleBot3, min_frontier_size 0.75 --
# wider than a doorway for this robot) and, worse, defaults use_sim_time to
# TRUE. On the real robot that means the node's clock never advances, every TF
# lookup fails, and it silently never sends a goal. Ours defaults to false and
# points at config/explore_params.yaml.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_path = get_package_share_directory('my_bot')

    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use the /clock topic instead of wall time. Must match the '
                    'value given to slam.launch.py and navigation.launch.py.',
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_path, 'config', 'explore_params.yaml'),
        description='Full path to the explore_lite parameters file.',
    )

    # /tf and /tf_static are remapped to relative names so the node works if it
    # is ever namespaced; harmless in the default empty namespace.
    explore = Node(
        package='explore_lite',
        executable='explore',
        name='explore_node',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    return LaunchDescription([
        use_sim_time_arg,
        params_file_arg,
        explore,
    ])
