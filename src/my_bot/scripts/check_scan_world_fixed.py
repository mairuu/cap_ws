#!/usr/bin/env python3
"""Check that /scan is world-fixed: turn the robot, and the scan must stay put.

The failure this catches is a MIRRORED scan (config/ydlidar.yaml `inverted`).
A wrong-handed scan does not look mirrored -- once TF places it in odom it
counter-rotates at TWICE the robot's yaw rate, because a feature belonging at
bearing b is emitted at -(b - yaw) and lands at 2*yaw - b. So the giveaway is
the best-aligning rotation:

    ~0 deg           -> correct, scan is world-fixed
    ~ -1x the turn   -> scan is rigidly following the robot (a TF problem)
    ~ -2x the turn   -> scan is mirrored (flip `inverted` in ydlidar.yaml)

Needs real_robot.launch.py (or launch_sim.launch.py) up. THIS DRIVES THE ROBOT:
it turns ~90 degrees in place, so give it clear space.

    ros2 run my_bot check_scan_world_fixed.py
"""

import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

TURN_RATE = 0.5           # rad/s
TURN_SECS = 3.14          # ~90 degrees
CELL = 0.10               # m, raster size for the overlap metric
ODOM_FRAME = 'odom'
CMD_TOPIC = '/diff_cont/cmd_vel_unstamped'

# The reference raster is dilated by one cell before matching. Two scans taken
# at different headings sample a wall at different points -- at 9 m, adjacent
# rays are ~16 cm apart while cells are 10 cm -- so an exact cell match tops
# out around 25% even when the scan is perfectly world-fixed. One cell of slop
# turns that into a number worth thresholding.
PASS_OVERLAP = 0.60

# Guard against a degenerate scan (few or no valid returns), where the overlap
# curve is flat and its peak lands at 0 for no good reason. A real alignment is
# a distinct peak, not a plateau.
PASS_PEAK_RATIO = 1.5
MIN_POINTS = 40


def yaw_of(rot):
    return math.atan2(2 * (rot.w * rot.z + rot.x * rot.y),
                      1 - 2 * (rot.y * rot.y + rot.z * rot.z))


class Checker(Node):
    def __init__(self):
        super().__init__('check_scan_world_fixed')
        self.scan = None
        self.buf = Buffer()
        self.listener = TransformListener(self.buf, self)
        self.pub = self.create_publisher(Twist, CMD_TOPIC, 10)
        self.create_subscription(
            LaserScan, '/scan', self._on_scan, qos_profile_sensor_data)

    def _on_scan(self, msg):
        self.scan = msg

    def settle(self, secs):
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def cloud(self):
        """Latest scan as (points in odom, robot xy, robot yaw)."""
        msg = self.scan
        stamp = rclpy.time.Time(seconds=msg.header.stamp.sec,
                                nanoseconds=msg.header.stamp.nanosec)
        tf = self.buf.lookup_transform(ODOM_FRAME, msg.header.frame_id, stamp)
        th = yaw_of(tf.transform.rotation)
        ox = tf.transform.translation.x
        oy = tf.transform.translation.y

        pts = []
        for i, r in enumerate(msg.ranges):
            # The driver reports invalid returns as 0.0, not inf
            # (invalid_range_is_inf: false), so range_min does the rejecting.
            if not (msg.range_min < r < msg.range_max):
                continue
            a = msg.angle_min + i * msg.angle_increment
            x, y = r * math.cos(a), r * math.sin(a)
            pts.append((ox + x * math.cos(th) - y * math.sin(th),
                        oy + x * math.sin(th) + y * math.cos(th)))
        return pts, (ox, oy), th

    def turn(self, rate, secs):
        tw = Twist()
        tw.angular.z = rate
        end = time.time() + secs
        while time.time() < end:
            self.pub.publish(tw)
            rclpy.spin_once(self, timeout_sec=0.05)
        for _ in range(30):
            self.pub.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    rclpy.init()
    node = Checker()

    end = time.time() + 15
    while node.scan is None and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.scan is None:
        print('FAIL: no /scan received -- is the lidar up?')
        return 1
    node.settle(3.0)

    before, _, yaw0 = node.cloud()
    node.turn(TURN_RATE, TURN_SECS)
    node.settle(2.0)
    after, centre, yaw1 = node.cloud()

    turned = math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))

    if len(before) < MIN_POINTS or len(after) < MIN_POINTS:
        print('FAIL: too few valid returns (%d / %d) to judge -- point the '
              'robot at something with structure in range.'
              % (len(before), len(after)))
        return 1

    grid = set()
    for x, y in before:
        gx, gy = round(x / CELL), round(y / CELL)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                grid.add((gx + dx, gy + dy))
    cx, cy = centre

    def overlap(deg):
        t = math.radians(deg)
        c, s = math.cos(t), math.sin(t)
        got = set()
        for x, y in after:
            dx, dy = x - cx, y - cy
            got.add((round((cx + dx * c - dy * s) / CELL),
                     round((cy + dx * s + dy * c) / CELL)))
        return len(grid & got) / max(1, len(got))

    sweep = {d: overlap(d) for d in range(-180, 181, 2)}
    here = sweep[0]
    best = max(sweep, key=sweep.get)
    typical = sorted(sweep.values())[len(sweep) // 2]
    ratio = here / max(typical, 1e-6)

    print('robot turned            : %+.1f deg' % turned)
    print('scan overlap as-is      : %.1f%%' % (100 * here))
    print('best-aligning rotation  : %+d deg (%.1f%%)' % (best, 100 * sweep[best]))
    print('peak ratio vs typical   : %.1fx' % ratio)

    if here >= PASS_OVERLAP and ratio >= PASS_PEAK_RATIO:
        print('\nPASS: scan is world-fixed.')
        return 0

    print('\nFAIL: scan moves with the robot.')
    if abs(best - (-2 * turned)) < 25:
        print('Best alignment is ~ -2x the turn: the scan is MIRRORED. '
              'Flip `inverted` in config/ydlidar.yaml.')
    elif abs(best - (-turned)) < 25:
        print('Best alignment is ~ -1x the turn: the scan is rigidly following '
              'the robot. Check odom -> base_link and the laser_joint TF.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
