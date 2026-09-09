#!/usr/bin/env python3
"""Drive straight and report what odometry THINKS it travelled, so you can
compare it against a tape measure.

This calibrates wheel_radius. Odometry distance is (wheel angle) x
(wheel_radius), and diff_drive_controller takes wheel_radius from
config/my_controllers.yaml -- NOT from the URDF. So if odom over-reports, the
configured radius is too big, and the correction is a pure ratio:

    corrected_radius = wheel_radius * (tape distance / odom distance)

Measuring the wheel with calipers does not settle this. A rubber tyre under the
robot's weight rolls on a radius smaller than its free radius, and any error in
enc_counts_per_rev_* (description/ros2_control.xacro) lands in exactly the same
place. This test folds all of it into the one number that actually scales the
map.

CLOSED LOOP, AND WHAT IT DOES NOT FIX. By default a cross-track controller
holds the robot on the line it started along, steering on ODOM yaw and ODOM
lateral offset. Read that carefully: the only heading reference available is
the one the encoders provide, so the loop drives odom's ESTIMATE of the path
straight. If enc_counts_per_rev_left and _right are mis-split, odom's yaw is
biased, and holding odom straight makes the robot physically curve.

That is not a flaw in the test, it is the sharpest measurement in it:

    odom lateral ~0, floor lateral ~0     -> encoder split is right
    odom lateral ~0, floor lateral large  -> encoder split is wrong, by
                                             about (2 * floor_lateral /
                                             distance) radians of yaw bias

so measure the floor offset at the end as well as the distance. Open loop
(--open-loop) steers nothing and lets the robot go where the wheels take it;
that is the better run for judging raw asymmetry, and the worse one for
measuring distance, because a curved path makes the tape chord shorter than the
distance the wheels actually rolled.

METHOD. Snap a chalk line, or sight along a wall, and park the robot on it.
Mark the floor at the robot's starting point: sight straight down through the
lidar puck's centre, which sits over base_link x=-0.034. Run this. Mark the
same feature again at the end. Measure BOTH the distance between marks and how
far the end mark sits off the chalk line.

Needs real_robot.launch.py up. THIS DRIVES THE ROBOT FORWARD several metres:
clear the space in front of it first. Ctrl-C stops it.

    ros2 run my_bot calibrate_straight.py --distance 3.0
    ros2 run my_bot calibrate_straight.py --distance 3.0 --open-loop
"""

import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState

CMD_TOPIC = '/diff_cont/cmd_vel_unstamped'
ODOM_TOPIC = '/diff_cont/odom'
RATE_HZ = 20.0

# Ramp in and out. The ESP32 gets commands unclamped (see twist_mux.yaml), and
# a step to full speed makes the wheels slip -- slip is encoder counts with no
# ground travel, which is precisely the error this test is trying to measure.
RAMP_SECS = 1.0
# Start easing off this far out, so the coast at the end is short and the robot
# stops near the target rather than well past it.
BRAKE_DIST = 0.30
MIN_SPEED = 0.04
# How long to keep integrating odom after commanding zero, to capture coast.
COAST_SECS = 2.0
# Give up rather than drive forever if odom is dead or the wheels are stalled.
TIMEOUT_FACTOR = 4.0

# --- cross-track controller -------------------------------------------------
# Steering is a cascade: lateral offset asks for a heading, heading asks for a
# yaw rate. Gains are deliberately soft. This runs at 0.1 m/s over a few metres
# and only ever has centimetres to correct, so a stiff loop would weave -- and
# weaving inflates the path length, which corrupts the distance measurement
# this whole script exists to take.
K_CROSS = 1.2          # rad of heading demand per metre of lateral offset
MAX_CROSS_HEADING = 0.30   # rad, cap on that demand (~17 deg)
KP_YAW = 1.5           # rad/s per rad of heading error
KI_YAW = 0.25          # rad/s per rad-second, trims steady drift
MAX_YAW_INTEGRAL = 0.40    # rad-s, anti-windup clamp
MAX_OMEGA = 0.40       # rad/s, hard cap on commanded turn rate


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class StraightRun(Node):

    def __init__(self, target, speed, closed_loop):
        super().__init__('calibrate_straight')
        self.target = target
        self.speed = speed
        self.closed_loop = closed_loop
        self.odom = None
        self.joints = None
        self.yaw_integral = 0.0
        self.max_cross = 0.0
        self.path_len = 0.0
        self.pub = self.create_publisher(Twist, CMD_TOPIC, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self._odom_cb, 10)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)

    def _odom_cb(self, msg):
        self.odom = msg

    def _joint_cb(self, msg):
        self.joints = msg

    def _pose(self):
        p = self.odom.pose.pose
        return p.position.x, p.position.y, yaw_of(p.orientation)

    def _wheels(self):
        """Wheel angles in radians, (left, right), or None if unavailable."""
        if self.joints is None:
            return None
        try:
            li = self.joints.name.index('left_wheel_joint')
            ri = self.joints.name.index('right_wheel_joint')
        except ValueError:
            return None
        if len(self.joints.position) <= max(li, ri):
            return None
        return self.joints.position[li], self.joints.position[ri]

    def _frame(self, x0, y0, yaw0):
        """Displacement resolved into the start heading: (along, lateral)."""
        x, y, _ = self._pose()
        dx, dy = x - x0, y - y0
        along = dx * math.cos(yaw0) + dy * math.sin(yaw0)
        lateral = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
        return along, lateral

    def _steer(self, cross, yaw_err, dt):
        """Cascade lateral offset -> heading demand -> yaw rate."""
        # Positive cross means we are LEFT of the line, so demand a heading to
        # the right to come back. yaw_err is measured against that demand.
        demand = max(-MAX_CROSS_HEADING,
                     min(MAX_CROSS_HEADING, -K_CROSS * cross))
        err = wrap(demand - yaw_err)
        self.yaw_integral = max(-MAX_YAW_INTEGRAL,
                                min(MAX_YAW_INTEGRAL,
                                    self.yaw_integral + err * dt))
        omega = KP_YAW * err + KI_YAW * self.yaw_integral
        return max(-MAX_OMEGA, min(MAX_OMEGA, omega))

    def spin_until(self, pred, timeout):
        end = time.time() + timeout
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if pred():
                return True
        return False

    def stop(self):
        """Command zero repeatedly. One message can be missed; this cannot."""
        for _ in range(10):
            self.pub.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.02)

    def run(self):
        if not self.spin_until(lambda: self.odom is not None, 10.0):
            print('ERROR: no %s in 10 s. Is real_robot.launch.py up, and did '
                  'diff_cont spawn?' % ODOM_TOPIC)
            return 1

        x0, y0, yaw0 = self._pose()
        w0 = self._wheels()
        if w0 is None:
            print('WARNING: no usable /joint_states; wheel angles unavailable.')

        print('start   odom x=%.4f y=%.4f yaw=%.2f deg' %
              (x0, y0, math.degrees(yaw0)))
        print('driving %.2f m at %.2f m/s, steering %s' %
              (self.target, self.speed,
               'CLOSED LOOP on odom cross-track' if self.closed_loop
               else 'OPEN LOOP (no correction)'))

        timeout = TIMEOUT_FACTOR * self.target / self.speed + RAMP_SECS + 5.0
        t_start = time.time()
        t_prev = t_start
        prev_x, prev_y, _ = self._pose()
        dist = 0.0
        period = 1.0 / RATE_HZ

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=period)
            now = time.time()
            dt = max(now - t_prev, 1e-3)
            t_prev = now

            x, y, yaw = self._pose()
            self.path_len += math.hypot(x - prev_x, y - prev_y)
            prev_x, prev_y = x, y

            dist = math.hypot(x - x0, y - y0)
            along, cross = self._frame(x0, y0, yaw0)
            self.max_cross = max(self.max_cross, abs(cross))

            # Stop on distance ALONG the start heading, not the chord. If the
            # robot has curved, the chord understates how far down the line it
            # has gone, and stopping on it would overshoot the tape mark.
            if along >= self.target:
                break
            elapsed = now - t_start
            if elapsed > timeout:
                self.stop()
                print('\nERROR: timed out at %.3f m after %.0f s. Wheels not '
                      'turning, or odom not updating.' % (along, elapsed))
                return 1

            v = self.speed
            if elapsed < RAMP_SECS:
                v *= max(elapsed / RAMP_SECS, 0.2)
            remaining = self.target - along
            if remaining < BRAKE_DIST:
                v *= max(remaining / BRAKE_DIST, 0.0)

            cmd = Twist()
            cmd.linear.x = max(v, MIN_SPEED)
            if self.closed_loop:
                cmd.angular.z = self._steer(cross, wrap(yaw - yaw0), dt)
            self.pub.publish(cmd)
            print('  along %.3f m   cross %+.3f m   omega %+.3f rad/s   ' %
                  (along, cross, cmd.angular.z), end='\r', flush=True)

        self.stop()
        print('\n  commanded stop, settling ...')

        end = time.time() + COAST_SECS
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            self.pub.publish(Twist())

        x1, y1, yaw1 = self._pose()
        w1 = self._wheels()
        dist = math.hypot(x1 - x0, y1 - y0)
        along, lateral = self._frame(x0, y0, yaw0)
        dyaw = math.degrees(wrap(yaw1 - yaw0))

        print()
        print('=' * 64)
        print('ODOM SAYS')
        print('  straight-line chord  %.4f m   <- compare against the tape' %
              dist)
        print('  along start heading  %.4f m' % along)
        print('  path length walked   %.4f m   (chord + %.1f mm of weave)' %
              (self.path_len, 1000.0 * (self.path_len - dist)))
        print('  lateral drift        %+.4f m   (+ = drifted LEFT)' % lateral)
        print('  worst lateral        %.4f m' % self.max_cross)
        print('  yaw drift            %+.2f deg  (+ = turned LEFT)' % dyaw)
        if w0 is not None and w1 is not None:
            dl, dr = w1[0] - w0[0], w1[1] - w0[1]
            print('  left wheel           %+.3f rad  (%.3f rev)' %
                  (dl, dl / (2 * math.pi)))
            print('  right wheel          %+.3f rad  (%.3f rev)' %
                  (dr, dr / (2 * math.pi)))
            mean = 0.5 * (abs(dl) + abs(dr))
            if mean > 1e-6:
                print('  L/R imbalance        %+.2f %%' %
                      (100.0 * (abs(dl) - abs(dr)) / mean))
                print('  implied radius       %.5f m  (odom dist / wheel angle)'
                      % (dist / mean))
        print()
        print('NOW MEASURE THE FLOOR:')
        print('  1. distance between the marks ->')
        print('       corrected wheel_radius = 0.034 * (tape_m / %.4f)' % dist)
        print('       goes in config/my_controllers.yaml AND '
              'description/robot_core.xacro')
        if self.closed_loop:
            print('  2. how far the end mark sits OFF the chalk line ->')
            print('       odom thinks it is %+.4f m. If the floor disagrees,'
                  % lateral)
            print('       the encoder split is wrong, not the geometry.')
        print('=' * 64)
        return 0


def main():
    ap = argparse.ArgumentParser(
        description='Straight-line odometry calibration run.')
    ap.add_argument('--distance', type=float, default=3.0,
                    help='metres to drive, measured along the start heading')
    ap.add_argument('--speed', type=float, default=0.10,
                    help='cruise speed in m/s')
    ap.add_argument('--open-loop', action='store_true',
                    help='steer nothing; let the wheels take it where they '
                         'will. Better for judging raw asymmetry, worse for '
                         'measuring distance.')
    args = ap.parse_args()

    rclpy.init()
    node = StraightRun(args.distance, args.speed, not args.open_loop)
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
