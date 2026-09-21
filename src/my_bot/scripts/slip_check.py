#!/usr/bin/env python3
"""Measure what wheel slip does to the heading estimate. This is D-23's test.

D-23 recorded, as an accepted limitation, that when the robot wedges on an
obstacle and spins, the wheels slip and odometry reports rotation that did not
happen. It also recorded that the fix it could not have was an IMU. D-25 fitted
one. THIS SCRIPT IS THE MEASUREMENT THAT TURNS THAT FROM AN ARGUMENT INTO A
NUMBER, and it is the figure the report should carry.

IT COMMANDS NOTHING. You wedge the robot and you drive it -- with `make
teleop-nav`, which is the e-stop. Run it twice, once each way:

    make real                      # wheels only
    ros2 run my_bot slip_check.py --seconds 60

    make real USE_EKF=true         # gyro fused
    ros2 run my_bot slip_check.py --seconds 60

WHAT A SLIP LOOKS LIKE, AND HOW THIS FINDS IT

A slipping wheel turns while the robot does not. So the wheels report an
angular velocity the gyro does not confirm, and the gap between them IS the
slip -- there is no other signal that separates "the robot turned" from "the
wheels turned". This watches |wheel omega - gyro omega| and calls anything
above --threshold, sustained past --min-duration, a slip window.

Inside each window it integrates that gap. That integral is PHANTOM YAW: the
rotation the wheels invented. Then it asks the question that matters --

    did the published estimate follow the wheels, or the gyro?

by watching odom -> base_link yaw across the same window. Without the EKF that
edge is diff_cont's and phantom yaw goes straight into it. With the EKF it is
robot_localization's, weighted 100:1 toward the gyro, and it should not.

WHY THE +/-10 DEGREE NUMBER IS PRINTED

slam_toolbox absorbs odometry error into map -> odom, so the robot's pose in
the MAP frame can survive a slip that the odom frame never recovers from. But
D-18 narrowed the matcher's angular search window to +/-10 degrees
(coarse_search_angle_offset 0.175 rad) on the explicit premise that odometry is
trustworthy. Phantom yaw larger than that window is outside what the matcher
can search, so it cannot find the truth and will either fail the match or lock
onto a wrong one. D-23 predicted 0.35 s of full-speed slip exhausts it. Each
window is scored against that budget.

READING THE RESULT

  phantom yaw large, odom yaw followed it        -> wheels-only. The failure
                                                    D-23 describes, measured.
  phantom yaw large, odom yaw stayed near zero   -> the EKF rejected it. This
                                                    is the result worth
                                                    reporting.
  phantom yaw over the +/-10 deg budget          -> quote it. It is the
                                                    difference between "the
                                                    matcher recovers" and "the
                                                    matcher cannot".
  no windows detected                            -> the robot did not actually
                                                    slip. Wedge it harder, or
                                                    lower --threshold.
  correction moved but odom did not              -> slam is doing the work the
                                                    EKF was supposed to. Worth
                                                    saying so.

CAVEAT THIS CANNOT ESCAPE. The gyro is the reference here, so this measures
the estimate against the gyro rather than against ground truth. That is sound
for slip specifically -- a MEMS gyro does not care what the wheels are doing --
but it means a gyro scale error shows up as apparent phantom yaw. odom_check.py
--compare bounds that at a few percent over 90 degrees.
"""

import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import TransformStamped  # noqa: F401  (documents the shape)
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf2_ros import Buffer, TransformListener

WHEEL_ODOM_TOPIC = '/diff_cont/odom'
IMU_TOPIC = '/imu_broad/imu'
EKF_TOPIC = '/odometry/filtered'

# D-18's coarse_search_angle_offset, in degrees. Phantom yaw past this is
# outside what the scan matcher can search for.
MATCHER_WINDOW_DEG = 10.0


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def unwrap(prev, now):
    d = now - prev
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


class SlipCheck(Node):
    def __init__(self, args):
        super().__init__('slip_check')
        self.args = args
        self.wheel_w = None
        self.gyro_w = None
        self.have_ekf = False

        self.buf = Buffer()
        self.listener = TransformListener(self.buf, self)

        self.create_subscription(Odometry, WHEEL_ODOM_TOPIC, self.cb_wheel, 20)
        self.create_subscription(Imu, IMU_TOPIC, self.cb_imu, 20)
        self.create_subscription(Odometry, EKF_TOPIC, self.cb_ekf, 20)

        self.samples = []       # (t, wheel_w, gyro_w, odom_yaw, corr_yaw, map_yaw)
        self.yaw_state = {}     # frame pair -> (prev_raw, unwrapped)
        self.started = time.monotonic()
        self.done = False
        self.warned = set()

        self.create_timer(1.0 / args.rate, self.sample)
        if args.seconds:
            self.create_timer(float(args.seconds), self.stop)

    def stop(self):
        self.done = True

    def cb_wheel(self, m):
        self.wheel_w = m.twist.twist.angular.z

    def cb_imu(self, m):
        self.gyro_w = m.angular_velocity.z

    def cb_ekf(self, _m):
        self.have_ekf = True

    def tf_yaw(self, parent, child):
        """Unwrapped yaw of parent->child, or None if TF has no answer yet."""
        try:
            t = self.buf.lookup_transform(parent, child, rclpy.time.Time())
        except Exception as e:
            key = f'{parent}->{child}'
            if key not in self.warned:
                self.warned.add(key)
                print(f'  [tf] {key} not available yet ({type(e).__name__})',
                      file=sys.stderr)
            return None
        raw = yaw_of(t.transform.rotation)
        key = f'{parent}->{child}'
        prev = self.yaw_state.get(key)
        if prev is None:
            self.yaw_state[key] = (raw, 0.0)
            return 0.0
        prev_raw, acc = prev
        acc += unwrap(prev_raw, raw)
        self.yaw_state[key] = (raw, acc)
        return acc

    def sample(self):
        if self.wheel_w is None:
            return
        t = time.monotonic() - self.started
        self.samples.append((
            t,
            self.wheel_w,
            self.gyro_w,
            self.tf_yaw('odom', 'base_link'),
            self.tf_yaw('map', 'odom'),
            self.tf_yaw('map', 'base_footprint'),
        ))


def find_windows(rows, threshold, min_duration):
    """Contiguous spans where |wheel - gyro| exceeds threshold."""
    windows = []
    start = None
    for i, r in enumerate(rows):
        gap = abs(r[1] - r[2]) if r[2] is not None else 0.0
        if gap >= threshold:
            if start is None:
                start = i
        elif start is not None:
            if rows[i - 1][0] - rows[start][0] >= min_duration:
                windows.append((start, i - 1))
            start = None
    if start is not None and rows[-1][0] - rows[start][0] >= min_duration:
        windows.append((start, len(rows) - 1))
    return windows


def span(rows, a, b, idx):
    """Change in column idx across rows[a..b], skipping absent values."""
    vals = [r[idx] for r in rows[a:b + 1] if r[idx] is not None]
    return (vals[-1] - vals[0]) if len(vals) >= 2 else None


def fmt(v, unit='deg'):
    return '  n/a  ' if v is None else f'{math.degrees(v):+7.2f} {unit}'


def main():
    ap = argparse.ArgumentParser(
        description='Measure phantom yaw from wheel slip and whether the '
                    'heading estimate rejected it. Commands nothing.')
    ap.add_argument('--seconds', type=float, default=60.0,
                    help='how long to watch (default 60)')
    ap.add_argument('--rate', type=float, default=20.0,
                    help='sample rate, Hz (default 20)')
    ap.add_argument('--threshold', type=float, default=0.15,
                    help='rad/s of wheel-vs-gyro disagreement that counts as '
                         'slip (default 0.15; diff_cont ceiling is 0.5)')
    ap.add_argument('--min-duration', type=float, default=0.2,
                    help='seconds a window must last to count (default 0.2)')
    args = ap.parse_args()

    rclpy.init()
    node = SlipCheck(args)
    print(f'\n  SLIP CHECK (D-23). Watching for {args.seconds:.0f} s.')
    print('  NOTHING HERE COMMANDS MOTION. Wedge the robot and drive it with')
    print('  `make teleop-nav`, which is the e-stop.\n')
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    rows = node.samples
    print()
    if len(rows) < 10:
        print(f'  NOTHING RECEIVED on {WHEEL_ODOM_TOPIC}. Is the stack up?')
        print('  Check: ros2 control list_controllers\n')
        node.destroy_node()
        rclpy.shutdown()
        return 1

    estimator = 'EKF (robot_localization)' if node.have_ekf else 'wheel odometry only'
    have_gyro = any(r[2] is not None for r in rows)
    print(f'  estimator publishing odom -> base_link : {estimator}')
    print(f'  samples {len(rows)} over {rows[-1][0]:.1f} s')
    if not have_gyro:
        print(f'\n  NO {IMU_TOPIC}. Without the gyro there is no reference to')
        print('  separate "the robot turned" from "the wheels turned", so slip')
        print('  cannot be detected at all. Start the stack with USE_IMU=true.\n')
        node.destroy_node()
        rclpy.shutdown()
        return 1

    windows = find_windows(rows, args.threshold, args.min_duration)
    print(f'  slip windows (|wheel - gyro| >= {args.threshold} rad/s for '
          f'>= {args.min_duration} s): {len(windows)}')

    if not windows:
        peak = max(abs(r[1] - r[2]) for r in rows if r[2] is not None)
        print(f'\n  The robot never slipped. Peak disagreement was {peak:.3f} rad/s,')
        print(f'  under the {args.threshold} threshold. Wedge it harder against a')
        print('  fixed obstacle and drive into it, or lower --threshold.\n')
        node.destroy_node()
        rclpy.shutdown()
        return 1

    budget = math.radians(MATCHER_WINDOW_DEG)
    worst = 0.0
    for n, (a, b) in enumerate(windows, 1):
        dt = rows[b][0] - rows[a][0]
        phantom = 0.0
        for i in range(a, b):
            step = rows[i + 1][0] - rows[i][0]
            phantom += (rows[i][1] - rows[i][2]) * step
        worst = max(worst, abs(phantom))
        d_odom = span(rows, a, b, 3)
        d_corr = span(rows, a, b, 4)
        d_map = span(rows, a, b, 5)
        peak = max(abs(rows[i][1] - rows[i][2]) for i in range(a, b + 1))

        print(f'\n  window {n}: t = {rows[a][0]:.1f} to {rows[b][0]:.1f} s  ({dt:.1f} s)')
        print(f'    phantom yaw  (integral of wheel - gyro) {fmt(phantom)}'
              f'   <- the rotation the wheels invented')
        print(f'    odom -> base_link yaw moved             {fmt(d_odom)}')
        print(f'    map  -> odom  correction absorbed       {fmt(d_corr)}')
        print(f'    map  -> base_footprint yaw moved        {fmt(d_map)}')
        print(f'    peak |wheel - gyro|                      {peak:.3f} rad/s')

        if d_odom is not None and abs(phantom) > 1e-6:
            followed = d_odom / phantom
            if followed > 0.6:
                verdict = 'FOLLOWED THE WHEELS -- phantom yaw entered the estimate'
            elif abs(followed) < 0.25:
                verdict = 'REJECTED IT -- the estimate followed the gyro'
            else:
                verdict = 'partially absorbed it'
            print(f'    -> the estimate {verdict}')
            print(f'       (odom moved {followed * 100:.0f} % of the phantom yaw)')

        ratio = abs(phantom) / budget
        note = ('INSIDE the matcher window' if ratio <= 1.0
                else 'BEYOND the matcher window -- slam_toolbox cannot search this far')
        print(f'    vs D-18 +/-{MATCHER_WINDOW_DEG:.0f} deg search window: '
              f'{ratio:.2f}x  ({note})')

    print(f'\n  worst phantom yaw this run: {math.degrees(worst):.2f} deg'
          f'  ({worst / budget:.2f}x the +/-{MATCHER_WINDOW_DEG:.0f} deg window)')
    print('\n  Run this again the other way (with/without USE_EKF=true) and put')
    print('  both in records/calibration.md. The pair IS the D-23 result.\n')

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
