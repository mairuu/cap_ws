#!/usr/bin/env python3
"""Hard-clamp the human teleop channel, so `q` cannot quietly ruin a map.

WHY THIS EXISTS. teleop_twist_keyboard's `q` multiplies its speed by 1.1 and
`z` divides it, PERMANENTLY -- there is no auto-reset -- and the current value
is echoed only in the teleop terminal, which nobody is looking at while they
watch RViz. `make teleop-nav` starts at SPEED=0.10, but a default is not a
limit: one stray keypress and the rest of the run is faster, with no indication
anywhere the driver is looking.

That is not cosmetic. The X3 Pro sweeps 360 deg in ~86 ms and slam_toolbox does
not deskew, so every scan is sheared along the path by (speed x sweep time):
1.0 cm at Nav2's 0.055 m/s, 0.9 cm at 0.10, but 8.7 cm at teleop's stock 0.5.
Sheared scans enter the pose graph and STAY there -- optimisation can move a
scan's pose, it cannot un-shear the scan -- so a few seconds of fast driving
leaves a doubled wall that slowing down will not undo. That is exactly how the
10 Sep gate run was lost.

WHERE IT SITS.

    teleop_twist_keyboard --> /cmd_vel_teleop_raw --> THIS --> /cmd_vel_teleop
                                                                     |
                                          twist_mux (priority 100) <--+

It is launched from navigation.launch.py alongside twist_mux, because those two
are the same safety chain and must live and die together.

FAIL-SAFE, AND WHY THAT CLAIM IS SAFE TO MAKE. This node sits in the e-stop
path, so "what if it dies" is the question that matters. navigation.launch.py
gives it on_exit=Shutdown: if it exits, the whole navigation stack goes down
with it -- twist_mux, controller_server, velocity_smoother, all of it. Nothing
is left publishing to /diff_cont/cmd_vel_unstamped, and diff_drive_controller
halts the wheels when commands stop arriving (cmd_vel_timeout). The failure
mode is a stopped robot, not a runaway one, and never a live Nav2 with no
override.

ZERO ALWAYS PASSES THROUGH UNCLAMPED. Clamping is by magnitude, so a stop
command is never scaled, delayed or dropped. `k` stops the robot at any limit.

IT IS A CLAMP, NOT A SCALER. Each component is limited independently rather
than the vector being scaled, matching how DWB and velocity_smoother express
their own limits in nav2_params.yaml.

    ros2 run my_bot teleop_speed_guard.py
    ros2 run my_bot teleop_speed_guard.py --ros-args -p max_linear:=0.3
"""

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Twist

# Matches the Makefile's SPEED/TURN defaults for teleop-nav. The guard is the
# limit; the teleop default is only a convenience so the first keypress is
# already correct.
DEFAULT_MAX_LINEAR = 0.10
DEFAULT_MAX_ANGULAR = 0.50

IN_TOPIC = '/cmd_vel_teleop_raw'
OUT_TOPIC = '/cmd_vel_teleop'


def clamp(v, lim):
    """Limit magnitude, preserve sign. Exact zero stays exactly zero."""
    if v > lim:
        return lim
    if v < -lim:
        return -lim
    return v


class SpeedGuard(Node):

    def __init__(self):
        super().__init__('teleop_speed_guard')
        self.max_linear = self.declare_parameter(
            'max_linear', DEFAULT_MAX_LINEAR).value
        self.max_angular = self.declare_parameter(
            'max_angular', DEFAULT_MAX_ANGULAR).value

        self.pub = self.create_publisher(Twist, OUT_TOPIC, 10)
        self.create_subscription(Twist, IN_TOPIC, self.on_cmd, 10)

        # Rate-limit the "I clamped something" warning: at 30 Hz of held-down
        # key a per-message log would bury every other line in the terminal.
        self.clamped = 0
        self.last_warn = 0.0

        self.get_logger().info(
            'teleop speed guard up: %s -> %s, limits %.3f m/s / %.3f rad/s. '
            'Zero always passes. Raising the limit needs a relaunch, not a '
            'keypress.' % (IN_TOPIC, OUT_TOPIC, self.max_linear,
                           self.max_angular))

    def on_cmd(self, msg):
        out = Twist()
        out.linear.x = clamp(msg.linear.x, self.max_linear)
        out.linear.y = clamp(msg.linear.y, self.max_linear)
        out.angular.z = clamp(msg.angular.z, self.max_angular)

        hit = (out.linear.x != msg.linear.x or
               out.linear.y != msg.linear.y or
               out.angular.z != msg.angular.z)
        if hit:
            self.clamped += 1
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self.last_warn > 2.0:
                self.last_warn = now
                self.get_logger().warn(
                    'CLAMPED: asked %.3f m/s %.3f rad/s, sending %.3f / %.3f. '
                    'Something pressed `q` -- press `z` to bring teleop back '
                    'down, or the robot will keep asking for more than it is '
                    'allowed. (%d messages clamped so far.)'
                    % (msg.linear.x, msg.angular.z,
                       out.linear.x, out.angular.z, self.clamped))

        self.pub.publish(out)


def main():
    rclpy.init()
    node = SpeedGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        if node.clamped:
            node.get_logger().warn(
                'guard clamped %d messages this run -- the teleop speed was '
                'raised above the limit at some point.' % node.clamped)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
