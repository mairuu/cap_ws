# YOLO detection + tracking off the C615, publishing vision_msgs on /detections.
#
# Day 5, decision D-11 option B: our own node (scripts/yolo_detector.py) with
# ultralytics' built-in tracker, NOT yolo_ros. The recovered launch file for
# the yolo_ros stack is kept in recoverable/mount/my_bot/launch/yolo.launch.py
# for reference; what survives from it here is the venv trick below.
#
# THE VENV TRICK. torch/ultralytics live in a hand-built uv venv (default
# ~/yolo/venv, rebuilt by cap_ws/yolo/setup_yolo_venv.sh). Its interpreter is
# /usr/bin/python3.10, the same one the ROS nodes run under, so the node does
# NOT run the venv's python and there is no `activate`: this launch prepends
# the venv's site-packages to PYTHONPATH and that is the whole mechanism.
# rclpy, cv_bridge and vision_msgs come from the system; torch and ultralytics
# come from the venv because they are found first.
#
# NEVER let anything run `uv sync` against that venv. The recovered uv.lock
# pins generic PyPI torch 2.13 / torchvision 0.28 and omits tensorrt; syncing
# it rips out the JetPack CUDA wheels. Upstream yolo_bringup's launch does
# exactly that on every start, which is why this file exists instead.
#
# THE CAMERA. cam2image (ros-humble-image-tools) on /image, 640x480 @ 15 Hz,
# RELIABLE. Focus is LOCKED before it starts, at the same FOCUS the intrinsics
# were calibrated at (51, see records/calibration.md): the C615 is varifocal
# and autofocus moves fx. Same rule as `make camera`. use_camera:=false if a
# `make camera` is already up -- two processes cannot hold /dev/video0.
#
#   ros2 launch my_bot yolo.launch.py
#   ros2 launch my_bot yolo.launch.py model:=/home/mic-711/yolo/yolov8n.pt imgsz:=480
#   ros2 launch my_bot yolo.launch.py use_camera:=false      # camera already up
#   ros2 launch my_bot yolo.launch.py device:=cpu            # demo-day fallback, ~5 Hz
#
# Then:  ros2 run my_bot detection_report.py --seconds 60   (the Day 5 gate)

import glob
import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            OpaqueFunction, RegisterEventHandler, Shutdown)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

HOME = os.path.expanduser("~")
DEFAULT_VENV = os.path.join(HOME, "yolo", "venv")
DEFAULT_MODEL = os.path.join(HOME, "yolo", "yolo26n.pt")

# Native mode of the C615 and the size the intrinsics were calibrated at.
# cam2image DEFAULTS TO 320x240 -- these are not decoration.
CAM_W, CAM_H = 640, 480


def _venv_pythonpath(venv: str) -> str:
    """Venv site-packages first, so its torch/ultralytics shadow the system's."""
    site = glob.glob(os.path.join(venv, "lib", "python*", "site-packages"))
    if not site:
        raise RuntimeError(
            f"No site-packages under {venv}. The YOLO venv is missing; rebuild "
            f"it with cap_ws/yolo/setup_yolo_venv.sh (NOT uv sync).")
    return ":".join(site + [os.environ.get("PYTHONPATH", "")]).strip(":")


def _setup(context):
    venv = LaunchConfiguration("venv").perform(context)
    env = {"PYTHONPATH": _venv_pythonpath(venv),
           # ultralytics "AutoUpdate" pip-installs missing packages into
           # whatever python it finds. Never silently at demo time.
           "YOLO_OFFLINE": "1"}

    cam_dev = LaunchConfiguration("camera_device").perform(context)
    focus = LaunchConfiguration("focus").perform(context)
    use_camera = LaunchConfiguration("use_camera").perform(context).lower() == "true"

    detector = Node(
        package="my_bot",
        executable="yolo_detector.py",
        name="yolo_detector",
        output="screen",
        additional_env=env,
        parameters=[{
            "model": LaunchConfiguration("model"),
            "device": LaunchConfiguration("device"),
            "imgsz": ParameterValue(LaunchConfiguration("imgsz"), value_type=int),
            "conf": ParameterValue(LaunchConfiguration("conf"), value_type=float),
            "image_topic": LaunchConfiguration("image_topic"),
            "image_reliability": "reliable",
            "publish_debug": ParameterValue(LaunchConfiguration("debug"), value_type=bool),
        }],
    )

    if not use_camera:
        return [detector]

    # Lock focus first, then start the camera once that has exited. Two v4l2
    # calls: setting focus_absolute while AF is still on is silently undone.
    if focus == "auto":
        focus_cmd = ["v4l2-ctl", "-d", cam_dev, "-c", "focus_automatic_continuous=1"]
    else:
        focus_cmd = ["bash", "-c",
                     f"v4l2-ctl -d {cam_dev} -c focus_automatic_continuous=0 && "
                     f"v4l2-ctl -d {cam_dev} -c focus_absolute={focus} && "
                     f"echo 'focus locked at {focus}'"]
    focus_lock = ExecuteProcess(cmd=focus_cmd, output="screen", name="focus_lock")

    # cam2image aborts if it cannot open the device -- usually a previous run
    # still holding /dev/video0 -- so take the whole launch down with it rather
    # than leave the detector up and silently receiving nothing.
    camera = Node(
        package="image_tools",
        executable="cam2image",
        name="cam2image",
        output="screen",
        parameters=[{
            "device_id": ParameterValue(LaunchConfiguration("camera_device_id"), value_type=int),
            "width": CAM_W,
            "height": CAM_H,
            "frequency": ParameterValue(LaunchConfiguration("camera_fps"), value_type=float),
            "reliability": "reliable",
            "frame_id": "camera_link",
        }],
        arguments=["--ros-args", "--log-level", "cam2image:=warn"],
        on_exit=Shutdown(reason="cam2image exited -- is another process holding the camera?"),
    )
    camera_after_focus = RegisterEventHandler(
        OnProcessExit(target_action=focus_lock, on_exit=[camera]))

    return [focus_lock, camera_after_focus, detector]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("model", default_value=DEFAULT_MODEL,
                              description="Path to a .pt or a .engine built on THIS board"),
        DeclareLaunchArgument("device", default_value="cuda:0",
                              description="cuda:0, or cpu as the demo-day fallback"),
        DeclareLaunchArgument("imgsz", default_value="640",
                              description="Inference size (letterboxed from 640x480)"),
        DeclareLaunchArgument("conf", default_value="0.5",
                              description="Minimum confidence to publish"),
        DeclareLaunchArgument("debug", default_value="true",
                              description="Publish the annotated image on /detections/image"),
        DeclareLaunchArgument("venv", default_value=DEFAULT_VENV,
                              description="Hand-built venv whose site-packages go on PYTHONPATH"),
        DeclareLaunchArgument("image_topic", default_value="/image",
                              description="cam2image publishes /image"),
        DeclareLaunchArgument("use_camera", default_value="true",
                              description="Start cam2image; false if `make camera` is already up"),
        DeclareLaunchArgument("camera_device", default_value="/dev/video0",
                              description="v4l2 node for the focus lock"),
        DeclareLaunchArgument("camera_device_id", default_value="0",
                              description="cam2image device index; the C615 is /dev/video0"),
        DeclareLaunchArgument("camera_fps", default_value="15.0"),
        DeclareLaunchArgument("focus", default_value="51",
                              description="focus_absolute; MUST match the calibration. 'auto' re-enables AF"),
        OpaqueFunction(function=_setup),
    ])
