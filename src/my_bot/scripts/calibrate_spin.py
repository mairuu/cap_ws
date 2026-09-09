#!/usr/bin/env python3
"""Spin in place N turns and report what odometry THINKS it rotated.

This calibrates wheel_separation, which is to heading what wheel_radius is to
distance. Odom yaw is (right arc - left arc) / wheel_separation, so a wrong
separation scales every heading the robot ever reports -- and unlike a distance
error, slam_toolbox cannot quietly absorb it: a yaw prior that drifts during a
turn is exactly what makes an already-mapped wall get drawn a second time,
rotated off the first.

Measuring the axle with a ruler does NOT settle this. The number that matters
is the effective track width at the contact patches, which depends on tyre
crown, camber and how much the tyres scrub during a turn. It is routinely
several percent away from the ruler value, and this test is the only way to see
that.

METHOD. Stick tape on the floor along the robot's centreline, or line the robot
up against a wall or floor seam. Run this. It stops when ODOM says exactly N
full turns; the robot will not be back where it started, and that residual
angle is the whole measurement. Read it off with a protractor, or measure the
sideways offset of a point a known distance out from the centre of rotation and
let calibrate_correct.py do the trigonometry.

Do it BOTH ways. A separation error is symmetric: clockwise and counter-
clockwise must give the same size of error, opposite in sign. If they differ,
something asymmetric is in play (a dragging wheel, a stiff caster) and the
separation number you would fit here would just be papering over it.

Needs real_robot.launch.py up. THIS SPINS THE ROBOT IN PLACE many times: give
it clear space, and keep the cables clear. Ctrl-C stops it.

    ros2 run my_bot calibrate_spin.py --turns 10
    ros2 run my_bot calibrate_spin.py --turns 10 --cw
"""

import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

CMD_TOPIC = '/diff_cont/cmd_vel_unstamped'
ODOM_TOPIC = '/diff_cont/odom'
RATE_HZ = 20.0

# Slow on purpose. This test measures the geometry, so anything that makes the
# tyres scrub or slip is contamination, not signal -- and scrub is precisely
# what rises with yaw rate. It also keeps the run honest against the deskew
# problem: see the max_vel_theta note in config/nav2_params.yaml.
DEFAULT_SPEED = 0.5
RAMP_SECS = 1.5
BRAKE_ANGLE = 0.6          # rad remaining at which to start easing off
MIN_SPEED = 0.10
COAST_SECS = 2.0
TIMEOUT_FACTOR = 3.0


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SpinRun(Node):

    def __init__(self, turns, speed, ccw):
        super().__init__('calibrate_spin')
        self.target = turns * 2.0 * math.pi
        self.speed = speed
        self.sign = 1.0 if ccw else -1.0
        self.odom = None
        self.total = 0.0          # unwrapped, signed
        self.max_drift = 0.0
        self.pub = self.create_publisher(Twist, CMD_TOPIC, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self._odom_cb, 10)

    def _odom_cb(self, msg):
        self.odom = msg

    def _yaw(self):
        return yaw_of(self.odom.pose.pose.orientation)

    def _xy(self):
        p = self.odom.pose.pose.position
        return p.x, p.y

    def spin_until(self, pred, timeout):
        end = time.time() + timeout
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if pred():
                return True
        return False

    def stop(self):
        for _ in range(10):
            self.pub.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.02)

    def run(self):
        if not self.spin_until(lambda: self.odom is not None, 10.0):
            print('ERROR: no %s in 10 s. Is real_robot.launch.py up, and did '
                  'diff_cont spawn?' % ODOM_TOPIC)
            return 1

        prev = self._yaw()
        x0, y0 = self._xy()
        print('spinning %.2f turns (%.0f deg) %s at %.2f rad/s ...'
              % (self.target / (2 * math.pi), math.degrees(self.target),
                 'CCW (left)' if self.sign > 0 else 'CW (right)', self.speed))

        timeout = TIMEOUT_FACTOR * self.target / self.speed + 10.0
        t_start = time.time()
        period = 1.0 / RATE_HZ

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=period)
            y = self._yaw()
            # Unwrap: accumulate the shortest step each tick. Sampling at 20 Hz
            # against a 0.5 rad/s spin gives 0.025 rad steps, nowhere near the
            # pi that would make this ambiguous.
            self.total += wrap(y - prev)
            prev = y

            done = self.sign * self.total
            x, y_ = self._xy()
            self.max_drift = max(self.max_drift, math.hypot(x - x0, y_ - y0))
            if done >= self.target:
                break
            elapsed = time.time() - t_start
            if elapsed > timeout:
                self.stop()
                print('\nERROR: timed out at %.1f deg after %.0f s.'
                      % (math.degrees(done), elapsed))
                return 1

            v = self.speed
            if elapsed < RAMP_SECS:
                v *= max(elapsed / RAMP_SECS, 0.2)
            remaining = self.target - done
            if remaining < BRAKE_ANGLE:
                v *= max(remaining / BRAKE_ANGLE, 0.0)
            cmd = Twist()
            cmd.angular.z = self.sign * max(v, MIN_SPEED)
            self.pub.publish(cmd)
            print('  %.1f / %.0f deg' % (math.degrees(done),
                                         math.degrees(self.target)),
                  end='\r', flush=True)

        self.stop()
        end = time.time() + COAST_SECS
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            self.pub.publish(Twist())
            y = self._yaw()
            self.total += wrap(y - prev)
            prev = y

        done = self.sign * self.total
        turns = done / (2 * math.pi)
        print()
        print('=' * 64)
        print('ODOM SAYS')
        print('  rotated              %.2f deg  (%.4f turns)'
              % (math.degrees(done), turns))
        print('  overshoot past target %+.2f deg  (coast after the stop)'
              % math.degrees(done - self.target))
        print('  centre wandered      %.3f m   (should be small; a big number'
              % self.max_drift)
        print('                                means it is not spinning in')
        print('                                place, so the test is invalid)')
        print()
        print('NOW MEASURE THE ROBOT:')
        print('  How far past (or short of) its starting heading did it')
        print('  actually finish? Sign it + for OVER-rotated in the direction')
        print('  it was spinning.')
        print()
        print('  Then, with e = that angle in degrees:')
        print('    ros2 run my_bot calibrate_correct.py \\')
        print('        --spin-odom-deg %.2f --spin-error-deg <e>'
              % math.degrees(done))
        print()
        print('  Easiest way to read e off the floor: mark a point on the')
        print('  robot a known distance d out from the centre of rotation,')
        print('  and measure how far sideways it sits from where it started.')
        print('  Pass --spin-lever d --spin-offset <metres> instead.')
        print('=' * 64)
        return 0


def main():
    ap = argparse.ArgumentParser(
        description='Spin-in-place calibration for wheel_separation.')
    ap.add_argument('--turns', type=float, default=10.0,
                    help='full turns to make. More turns amplify the error '
                         'against a fixed protractor accuracy, but also '
                         'accumulate scrub; 10 is a good compromise.')
    ap.add_argument('--speed', type=float, default=DEFAULT_SPEED,
                    help='yaw rate in rad/s')
    ap.add_argument('--cw', action='store_true',
                    help='spin clockwise instead of counter-clockwise. Run '
                         'both directions and compare.')
    args = ap.parse_args()

    rclpy.init()
    node = SpinRun(args.turns, args.speed, not args.cw)
    try:
        rc = node.run()
    except KeyboardInterrupt:
        node.stop()
        print('\ninterrupted, stopped')
        rc = 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(rc)


if __name__ == '__main__':
    main()
