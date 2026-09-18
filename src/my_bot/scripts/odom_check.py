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

COMPARING WHEEL ODOMETRY AGAINST THE EKF (D-25). `--compare` watches
/odometry/filtered alongside /diff_cont/odom and prints both, so ONE hand-push
scores both estimators against the same physical metre. Without it this script
sees only raw wheel odometry, which the EKF does not touch -- so the default
invocation cannot tell you anything about the fusion at all.

What the comparison can and cannot show. A hand-push turns the wheels, so the
encoders are honest and both estimates should agree closely; that agreement is
the point -- it says the EKF is not corrupting a working estimate, and that the
gyro's sign and scale match the wheels. It is NOT the case the EKF exists for.
That case is wheel SLIP, where the wheels lie and the gyro does not, and a hand
push cannot produce it. Lift one wheel and spin it to see them disagree on
purpose.

Watch for: dyaw agreeing to a few degrees over a 90 degree turn (a gyro scale
or axis error shows up here as a systematic fraction, not as noise), and the
EKF's dx tracking the wheels' (the EKF fuses only vx and vyaw, so a large dx
disagreement means the twist covariances in my_controllers.yaml are wrong).
A dyaw of OPPOSITE SIGN between the two is the yaw-axis sign error that
imu.xacro's rpy exists to prevent -- stop and re-run imu_check.py --axes.

Needs real_robot.launch.py up. Ctrl-C to stop, or --seconds to bound it.

    ros2 run my_bot odom_check.py                   # runs until Ctrl-C
    ros2 run my_bot odom_check.py --seconds 30
    ros2 run my_bot odom_check.py --compare         # wheels vs EKF, D-25
"""

import argparse
import math
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

# diff_drive_controller publishes here, NOT on /odom. Echoing /odom shows
# nothing and looks exactly like a dead controller.
ODOM_TOPIC = '/diff_cont/odom'

# robot_localization's fused output, present only when the stack was started
# with use_ekf:=true. This is the topic the EKF actually affects.
EKF_TOPIC = '/odometry/filtered'


def yaw_of(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class Track:
    """One topic's start-to-now delta. Kept separate per topic so --compare
    scores both estimators against the same physical movement."""

    def __init__(self, label):
        self.label = label
        self.first = None
        self.last = None
        self.count = 0
        self.unwrapped_yaw = 0.0
        self.prev_yaw = None

    def add(self, msg):
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

    def deltas(self):
        if self.first is None:
            return None
        dx = self.last[0] - self.first[0]
        dy = self.last[1] - self.first[1]
        return dx, dy, math.hypot(dx, dy), math.degrees(self.unwrapped_yaw)

    def line(self, width=0):
        d = self.deltas()
        tag = f'{self.label:>{width}}  ' if width else ''
        if d is None:
            return f'  {tag}waiting...'
        dx, dy, dist, dyaw = d
        return (f'  {tag}dx={dx:+.3f} m   dy={dy:+.3f} m   '
                f'straight-line={dist:.3f} m   dyaw={dyaw:+.1f} deg'
                f'   ({self.count} msgs)')


class OdomCheck(Node):
    def __init__(self, seconds, compare=False):
        super().__init__('odom_check')
        self.wheels = Track('wheels')
        self.ekf = Track('EKF') if compare else None
        self.compare = compare
        self.create_subscription(Odometry, ODOM_TOPIC, self.cb, 10)
        if compare:
            self.create_subscription(Odometry, EKF_TOPIC, self.cb_ekf, 10)
        # On a terminal, redraw one line in place. When stdout is redirected to a
        # file there is no cursor to return, so \r just concatenates every update
        # into one unreadable line -- print periodically instead, and only when
        # something actually moved.
        self.tty = sys.stdout.isatty() and not compare
        self.create_timer(0.5 if self.tty else 2.0, self.report)
        self.last_printed = None
        if seconds:
            self.create_timer(float(seconds), self.stop)
        self.done = False
        self.started = time.monotonic()
        topics = ODOM_TOPIC + (f' and {EKF_TOPIC}' if compare else '')
        print(f'\n  listening on {topics}'
              f'\n  MOVE THE ROBOT BY HAND -- nothing here commands motion. Ctrl-C to finish\n',
              flush=True)

    def stop(self):
        self.done = True

    def cb(self, msg):
        self.wheels.add(msg)

    def cb_ekf(self, msg):
        self.ekf.add(msg)

    def line(self):
        if not self.compare:
            d = self.wheels.deltas()
            return '  waiting for odometry...' if d is None else self.wheels.line()
        return self.wheels.line(7) + '\n' + self.ekf.line(7)

    def report(self):
        if self.tty:
            print(self.line(), end='\r', flush=True)
            return
        d = self.wheels.deltas()
        if d is None:
            return
        # Redirected: emit only when the reading has moved enough to be worth a
        # line, so a long quiet window does not bury the interesting part.
        #
        # The EKF's own deltas are part of that test, not just the wheels'.
        # They have to be: the interesting case is precisely the one where the
        # two DISAGREE, and gating on the wheels alone hides it. First run of
        # this mode, 18 Sep, did exactly that -- the EKF turned through 86 deg
        # while the wheels sat at 0.6, and because the wheels had not moved,
        # 68 seconds of it printed nothing at all.
        now = (round(d[0], 3), round(d[1], 3), round(d[3], 1))
        if self.ekf is not None:
            e = self.ekf.deltas()
            if e is not None:
                now = now + (round(e[0], 3), round(e[1], 3), round(e[3], 1))
        thresholds = (0.01, 0.01, 1.0) * (len(now) // 3)
        if self.last_printed is None or len(now) != len(self.last_printed) or any(
                abs(a - b) >= t for a, b, t in
                zip(now, self.last_printed, thresholds)):
            print(f'  [{time.monotonic() - self.started:5.1f}s]{self.line()}', flush=True)
            self.last_printed = now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=None,
                    help='stop automatically after this long')
    ap.add_argument('--compare', action='store_true',
                    help=f'also watch {EKF_TOPIC} and print both, so one '
                         f'hand-push scores wheel odometry against the EKF '
                         f'(D-25). Needs use_ekf:=true.')
    args = ap.parse_args()

    rclpy.init()
    node = OdomCheck(args.seconds, compare=args.compare)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    print('\n')
    if node.wheels.deltas() is None:
        print(f'  NOTHING RECEIVED on {ODOM_TOPIC}.')
        print('  diff_cont did not spawn, or is not active.')
        print('  Check: ros2 control list_controllers\n')
        rc = 1
    else:
        print('  final:')
        print(node.line())
        rc = 0
        if node.compare:
            e = node.ekf.deltas()
            if e is None:
                print(f'\n  NOTHING RECEIVED on {EKF_TOPIC} -- the stack was not'
                      '\n  started with use_ekf:=true, so this is wheel odometry only.\n')
                rc = 1
            else:
                w = node.wheels.deltas()
                print(f'\n  agreement   d(straight-line)={e[2] - w[2]:+.3f} m'
                      f'   d(dyaw)={e[3] - w[3]:+.1f} deg')
                if w[3] * e[3] < 0 and min(abs(w[3]), abs(e[3])) > 5.0:
                    print('  ** OPPOSITE YAW SIGNS. This is the gyro yaw-axis sign'
                          '\n     error. Stop and re-run imu_check.py with its axes'
                          '\n     flag; do not drive until imu.xacro agrees.')
                    rc = 1
        print('\n  Sanity only. Day 3 calibrates this properly with'
              '\n  calibrate_straight.py and calibrate_spin.py.'
              '\n  A hand-push cannot produce wheel SLIP, which is the case the'
              '\n  EKF exists for -- see D-25.\n')
    node.destroy_node()
    rclpy.shutdown()
    return rc


if __name__ == '__main__':
    sys.exit(main())
