# Online asynchronous SLAM with slam_toolbox.
#
# "Online" = consumes a live /scan (not a bag); "async" = the scan callback
# returns immediately instead of blocking until the scan is matched, so a slow
# match drops scans rather than stalling the pipeline. That is the right trade
# on the Jetson.
#
# This launches ONLY the mapper. Bring the robot up first, in another terminal:
#
#   real robot:  ros2 launch my_bot real_robot.launch.py
#                ros2 launch my_bot slam.launch.py
#
#   simulation:  ros2 launch my_bot launch_sim.launch.py
#                ros2 launch my_bot slam.launch.py use_sim_time:=true
#
# use_sim_time defaults to FALSE here (unlike upstream's launch, which defaults
# to true). Getting it wrong is the classic failure: with sim time enabled and
# no /clock publisher the node's clock never advances, so every TF lookup fails
# and no map is ever produced -- with no error that names the cause.
#
# Then drive with teleop and watch the map in RViz (Map display on /map, fixed
# frame "map"). Save when done:
#
#   ros2 run nav2_map_server map_saver_cli -f ~/my_map
#
# or call slam_toolbox's own /slam_toolbox/save_map service, which also lets
# /slam_toolbox/serialize_map write a .posegraph you can resume from later.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_path = get_package_share_directory('my_bot')

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use the /clock topic instead of wall time. true only when '
                    'running against launch_sim.launch.py.',
    )

    slam_params_file = LaunchConfiguration('slam_params_file')
    slam_params_file_arg = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(
            pkg_path, 'config', 'mapper_params_online_async.yaml'
        ),
        description='Full path to the slam_toolbox parameters file.',
    )

    # The node name must stay "slam_toolbox" -- it is the top-level key in the
    # params file, and a mismatch means every parameter is silently ignored and
    # the node runs on upstream defaults instead.
    slam_toolbox = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=[
            slam_params_file,
            {'use_sim_time': use_sim_time},
        ],
        output='screen',
    )

    return LaunchDescription([
        use_sim_time_arg,
        slam_params_file_arg,
        slam_toolbox,
    ])
