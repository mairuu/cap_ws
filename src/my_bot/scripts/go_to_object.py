#!/usr/bin/env python3
"""Navigate to the nearest landmark of a named class. Day 7's stretch goal.

WHAT THIS IS. "Go to the nearest chair." It reads /semantic_landmarks, picks the
nearest landmark of the requested class TO THE ROBOT, computes a standoff pose
`standoff` metres in front of it facing it, and sends that to Nav2's
NavigateToPose. That is the whole feature; everything below is the reasoning.

NOT A ROS ACTION, DELIBERATELY. The design note calls this an action server, and
a custom .action would be the textbook interface -- but interface generation
needs a rosidl/CMake package, and `semantic_objects` is ament_python while this
is a stretch goal that may be cut on the day. A new interface package would be
build infrastructure bought for a feature that might not ship. So the command
surface is two plain topics and no new message types:

    ros2 topic pub --once /go_to_object std_msgs/String "{data: chair}"
    ros2 topic echo /go_to_object/result

Consequence to state honestly if this ships: there is no goal id, no cancel, and
no feedback stream -- a second command while one is running pre-empts the first
by cancelling the underlying NavigateToPose goal. That is enough for a demo and
is not enough for a robot that takes orders from anything but a person.

WHY NEAREST-TO-ROBOT, NOT NEAREST-TO-ANYTHING. LandmarkStore._nearest_of_class
already exists (landmark_store.py) but it answers a different question: it finds
the landmark nearest an *observation*, for data association. This needs the one
nearest the *robot*, which needs the robot's pose, which is a TF lookup. Same
shape, different input -- a template, not a drop-in. Hence this node subscribes
to the published JSON rather than importing the store at all, which also means
it needs no access to semantic_objects' internals.

THE STANDOFF POSE. Nav2 will refuse a goal inside an obstacle, and a landmark
sits ON the object, so driving to the landmark itself asks the planner to drive
into a chair. The pose is therefore `standoff` metres from the object along the
line from the object back toward the robot -- the side the robot can already
see, which is the side most likely to be free -- and yawed to face the object so
the camera ends up pointing at the thing it was asked to go to.

    make real / slam / nav / yolo / semantic, then:
    ros2 run my_bot go_to_object.py
    ros2 topic pub --once /go_to_object std_msgs/String "{data: chair}"
"""

import json
import math
import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String


class GoToObject(Node):

    def __init__(self):
        super().__init__("go_to_object")

        self.declare_parameter("standoff", 0.8)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("landmarks_topic", "/semantic_landmarks")

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self._standoff = float(p("standoff"))
        self._map_frame = p("map_frame")
        self._base_frame = p("base_frame")

        self._landmarks = []
        self._goal_handle = None

        # /semantic_landmarks is TRANSIENT_LOCAL, so match it or this node sees
        # nothing until the next 2 Hz publish -- and a durability mismatch does
        # not warn, it simply never connects.
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.create_subscription(String, p("landmarks_topic"), self._on_landmarks, latched)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self, spin_thread=True)

        self._result_pub = self.create_publisher(String, "/go_to_object/result", 10)
        self.create_subscription(String, "/go_to_object", self._on_command, 10)

        self._nav = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self.get_logger().info(
            f"go_to_object ready; standoff {self._standoff:.2f} m, "
            f"frames {self._map_frame} -> {self._base_frame}. "
            f"Command:  ros2 topic pub --once /go_to_object std_msgs/String \"{{data: chair}}\"")

    # ------------------------------------------------------------------

    def _on_landmarks(self, msg):
        try:
            self._landmarks = json.loads(msg.data)["landmarks"]
        except (ValueError, KeyError, TypeError) as e:
            self.get_logger().warn(f"bad landmark payload: {e}", throttle_duration_sec=5.0)

    def _say(self, text, ok=False):
        (self.get_logger().info if ok else self.get_logger().warn)(text)
        self._result_pub.publish(String(data=text))

    def _robot_xy(self):
        """Robot position in the map frame, or None."""
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception as e:  # noqa: BLE001 -- tf2 raises several unrelated types
            self.get_logger().warn(f"no {self._map_frame} -> {self._base_frame}: {e}")
            return None
        t = tf.transform.translation
        return t.x, t.y

    def _on_command(self, msg):
        want = msg.data.strip()
        if not want:
            self._say("empty command; send a class name, e.g. 'chair'")
            return

        if not self._landmarks:
            self._say("no landmarks published yet -- is make semantic running, and has "
                      "anything been fused? Check its 'fused n/m' line.")
            return

        mine = [lm for lm in self._landmarks if lm.get("class_label") == want]
        if not mine:
            seen = sorted({lm.get("class_label", "?") for lm in self._landmarks})
            self._say(f"no landmark of class {want!r}. Known classes: {', '.join(seen) or 'none'}")
            return

        rxy = self._robot_xy()
        if rxy is None:
            self._say("cannot locate the robot; is slam_toolbox up?")
            return
        rx, ry = rxy

        target = min(mine, key=lambda lm: math.hypot(lm["x"] - rx, lm["y"] - ry))
        ox, oy = target["x"], target["y"]

        # Standoff on the robot's side of the object, facing it.
        dx, dy = rx - ox, ry - oy
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            # Robot is effectively on top of the landmark; no meaningful
            # direction to back off along. Refuse rather than pick one at random.
            self._say(f"already at the {want} ({dist:.2f} m); no standoff direction")
            return
        sx = ox + dx / dist * self._standoff
        sy = oy + dy / dist * self._standoff
        yaw = math.atan2(oy - sy, ox - sx)

        if not self._nav.wait_for_server(timeout_sec=5.0):
            self._say("NavigateToPose action server not available -- is make nav running?")
            return

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self._map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = sx
        goal.pose.pose.position.y = sy
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self._say(f"going to {want} at ({ox:+.2f}, {oy:+.2f}), {dist:.2f} m away; "
                  f"standoff ({sx:+.2f}, {sy:+.2f}) facing {math.degrees(yaw):+.0f} deg",
                  ok=True)

        if self._goal_handle is not None:
            # No goal ids on a topic interface, so a new command pre-empts.
            self._nav.async_cancel_goal_async(self._goal_handle)
            self._goal_handle = None

        self._nav.send_goal_async(goal).add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self._say("Nav2 rejected the goal -- is the standoff pose inside an obstacle "
                      "or outside the map?")
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        self._goal_handle = None
        status = future.result().status
        # 4 == STATUS_SUCCEEDED in action_msgs/msg/GoalStatus
        if status == 4:
            self._say("arrived", ok=True)
        else:
            self._say(f"navigation ended without success (status {status}); "
                      f"check the bt_navigator log")


def main():
    rclpy.init()
    node = GoToObject()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
