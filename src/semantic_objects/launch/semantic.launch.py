# Camera + lidar semantic fusion -- Day 6.
#
# Starts two things:
#   1. semantic_objects_node: /scan + /detections -> /semantic_landmarks,
#      /semantic_markers, and the clear_landmarks service.
#   2. image_transport republish: /image (raw, from cam2image) -> /image/compressed
#      for the browser bridge's camera pane. cam2image uses a plain rclcpp
#      publisher, so no transport plugin ever attaches to it and /image/compressed
#      does not otherwise exist. (STATE.md, 11 Sep.)
#
# Needs, in other terminals: make real (TF, odom, /scan), make slam (map frame),
# make yolo (/detections and /image). Without make real the node still starts
# (it waits up to tf.static_timeout for the URDF frames, then gives up loudly),
# and with it but without odom the motion gate stays CLOSED and says so.
#
# The camera intrinsics are read from my_bot's installed calibration file, so
# there is exactly one copy of fx/fy/cx/cy on this robot. The sensor geometry
# comes from TF (decision D-10). robot_params.yaml carries everything else.
#
#   ros2 launch semantic_objects semantic.launch.py
#   ros2 launch semantic_objects semantic.launch.py persist_path:=~/maps/landmarks.json
#   ros2 launch semantic_objects semantic.launch.py republish:=false

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("semantic_objects")
    my_bot = get_package_share_directory("my_bot")

    params_file = LaunchConfiguration("params_file")
    calibration_file = LaunchConfiguration("calibration_file")
    persist_path = LaunchConfiguration("persist_path")

    node = Node(
        package="semantic_objects",
        executable="semantic_objects_node",
        name="semantic_objects",   # must match the top-level key in robot_params.yaml
        output="screen",
        parameters=[
            params_file,
            {
                "camera.calibration_file": calibration_file,
                "landmark.persist_path": persist_path,
            },
        ],
    )

    republish = Node(
        package="image_transport",
        executable="republish",
        name="image_republish",
        arguments=["raw", "compressed"],
        remappings=[
            ("in", "/image"),
            ("out/compressed", "/image/compressed"),
        ],
        condition=IfCondition(LaunchConfiguration("republish")),
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(pkg, "config", "robot_params.yaml"),
            description="semantic_objects parameters (dotted keys)"),
        DeclareLaunchArgument(
            "calibration_file",
            default_value=os.path.join(my_bot, "config", "c615_640x480.yaml"),
            description="camera_info YAML; the only copy of the intrinsics"),
        DeclareLaunchArgument(
            "persist_path",
            default_value=os.path.expanduser("~/maps/landmarks.json"),
            description="JSON written on the publish timer; '' disables"),
        DeclareLaunchArgument(
            "republish", default_value="true",
            description="also run image_transport republish for /image/compressed"),
        node,
        republish,
    ])
