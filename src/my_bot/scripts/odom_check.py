#!/usr/bin/env python3
"""Watch odometry while you move the robot BY HAND, and report what it thinks.

The Day 2 gate asks for two sanity checks: push the robot forward a metre and
odom's x should rise about a metre; turn it 90 degrees and yaw should change
about pi/2. Doing that by eye off `ros2 topic echo` means reading a firehose of
quaternions and subtracting them in your head. This subtracts them for you.

IT COMMANDS NOTHING. No publisher, no motion -- you move the robot, this only
listens. That is deliberate: an encoder-sign fault is exactly what this test
exists to catch, and the last thing you want while catching it is a PID with an
opinion. Safe to run with the robot on the floor.

WHAT IT CANNOT SETTLE. This is a SANITY check, not a calibration. It tells you
the sign is right and the scale is roughly right. It does NOT give you
wheel_radius or wheel_separation -- a closed loop steering on odom hides its own
error, which is the whole point of calibrate_straight.py and calibrate_spin.py.
Day 2 wants "roughly"; Day 3 makes it true. Do not copy numbers from here into
records/calibration.md as calibration.

READING IT. Push the robot forward one metre:

  dx should be about +1.0 with dy and dyaw near zero.
  NEGATIVE dx means the robot thinks it reversed: both encoders are inverted,
    or motors_reversed is wrong on the ROS side. Fix it in ROS, not by rewiring.
  dx near zero while |dyaw| grows means the two wheels disagree in sign -- one
    encoder is backwards. That is the PID runaway condition; see Day 1 section
    5.7 before driving it under power.
  |dx| far from 1.0 (say under 0.8 or over 1.25) points at wheel_radius or
    enc_counts_per_rev, and is Day 3's problem, not a blocker today.

Turn the robot 90 degrees on the spot: |dyaw| about 90, dx and dy small.
A dyaw of roughly HALF what you turned, or roughly double, is a
wheel_separation scale error and again belongs to Day 3.

Needs real_robot.launch.py up. Ctrl-C to stop, or --seconds to bound it.

    ros2 run my_bot odom_check.py                 # runs until Ctrl-C
    ros2 run my_bot odom_check.py --seconds 30
"""

import argparse
import math
import sys

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

# diff_drive_controller publishes here, NOT on /odom. Echoing /odom shows
# nothing and looks exactly like a dead controller.
ODOM_TOPIC = '/diff_cont/odom'


def yaw_of(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class OdomCheck(Node):
    def __init__(self, seconds):
        super().__init__('odom_check')
        self.first = None
        self.last = None
        self.count = 0
        self.unwrapped_yaw = 0.0
        self.prev_yaw = None
        self.create_subscription(Odometry, ODOM_TOPIC, self.cb, 10)
        self.create_timer(0.5, self.report)
        if seconds:
            self.create_timer(float(seconds), self.stop)
        self.done = False
        print(f'\n  listening on {ODOM_TOPIC} -- move the robot by hand, Ctrl-C to finish\n')

    def stop(self):
        self.done = True

    def cb(self, msg):
        p = msg.pose.pose.position
        y = yaw_of(msg.pose.pose.orientation)
        # Unwrap so a turn through +-pi does not read as a jump of 2pi.
        if self.prev_yaw is not None:
            d = y - self.prev_yaw
            while d > math.pi:
                d -= 2 * math.pi
            while d < -math.pi:
                d += 2 * math.pi
            self.unwrapped_yaw += d
        else:
            self.unwrapped_yaw = 0.0
        self.prev_yaw = y
        if self.first is None:
            self.first = (p.x, p.y)
        self.last = (p.x, p.y)
        self.count += 1

    def line(self):
        if self.first is None:
            return '  waiting for odometry...'
        dx = self.last[0] - self.first[0]
        dy = self.last[1] - self.first[1]
        dist = math.hypot(dx, dy)
        return (f'  dx={dx:+.3f} m   dy={dy:+.3f} m   '
                f'straight-line={dist:.3f} m   dyaw={math.degrees(self.unwrapped_yaw):+.1f} deg'
                f'   ({self.count} msgs)')

    def report(self):
        print(self.line(), end='\r', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=None,
                    help='stop automatically after this long')
    args = ap.parse_args()

    rclpy.init()
    node = OdomCheck(args.seconds)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    print('\n')
    if node.first is None:
        print(f'  NOTHING RECEIVED on {ODOM_TOPIC}.')
        print('  diff_cont did not spawn, or is not active.')
        print('  Check: ros2 control list_controllers\n')
        rc = 1
    else:
        print('  final:')
        print(node.line())
        print('\n  Sanity only. Day 3 calibrates this properly with'
              '\n  calibrate_straight.py and calibrate_spin.py.\n')
        rc = 0
    node.destroy_node()
    rclpy.shutdown()
    return rc


if __name__ == '__main__':
    sys.exit(main())
