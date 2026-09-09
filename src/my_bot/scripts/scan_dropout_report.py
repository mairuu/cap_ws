#!/usr/bin/env python3
"""Measure how many of the X2's rays come back empty, and which bearings are blind.

The YDLidar X2 does not return a range for every ray. Indoors, on the bench,
roughly HALF of them come back as 0.0 -- that figure is recorded in
config/ydlidar.yaml, and this script is what re-derives it on a new board.

WHY 0.0 AND NOT inf. ydlidar.yaml sets `invalid_range_is_inf: false`, so a
dropout arrives as 0.0 rather than +inf. 0.0 is BELOW range_min, which means a
consumer that only filters inf/nan -- the obvious way to write it -- reads every
dropout as an obstacle zero metres away, dead centre on the robot. Nav2 and
slam_toolbox both check range_min so they are safe; our own semantic fusion code
is the thing to watch. This script counts the two kinds separately so you can
see which convention the driver is actually using today.

WHAT IT IS FOR. The dropout fraction sets `detection.min_returns` on Day 6. If
half the rays are empty, a narrow bounding box at 3 m may be backed by only two
or three live returns, and `min_returns: 3` will reject objects you can plainly
see. That may still be the right trade -- but make it knowingly, with this
number in hand.

THE PER-SECTOR BREAKDOWN MATTERS MORE THAN THE AVERAGE. Dropout is rarely
uniform: dark surfaces, glass, and whatever the robot's own chassis clips all
show up as one bad bearing band rather than as noise spread evenly. A 50%
average made of "front is fine, left is blind" behaves nothing like a 50%
average that is uniform, and only one of them breaks the fusion. Knowing which
bearings are blind before Day 6 saves you blaming the camera.

BEARINGS ARE IN THE LASER FRAME, REP-103: 0 deg is straight ahead, +90 is left,
+/-180 is behind. That already accounts for `reversion` and `inverted` in
ydlidar.yaml, because the SDK applies both before publishing. If the sectors
this prints do not match the room -- you are facing a wall and "AHEAD" is the
emptiest sector -- suspect those two flags, not this script, and read their
comments in ydlidar.yaml before touching either.

IT COMMANDS NOTHING and needs no TF. Only the lidar driver has to be running:

    make real                       # or the driver alone, see below
    ros2 run my_bot scan_dropout_report.py
    ros2 run my_bot scan_dropout_report.py --scans 200 --sectors 24

Stand the robot still in an ordinary room. Walking around it while this runs
measures your legs, not the lidar.
"""

import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

# /scan is published BEST_EFFORT (sensor data QoS). A default RELIABLE
# subscription is QoS-incompatible and receives NOTHING, silently, with no
# error on either side -- `ros2 topic hz /scan` fails exactly this way and looks
# like a dead lidar. qos_profile_sensor_data below is not optional.
QOS = qos_profile_sensor_data

DEFAULT_SCANS = 100
DEFAULT_SECTORS = 12          # 30-degree bands
DISCOVERY_TIMEOUT = 10.0      # s to wait for the first scan before giving up


def sector_label(centre_deg):
    """Name a bearing the way you would say it out loud, for the report."""
    a = (centre_deg + 180.0) % 360.0 - 180.0
    if -22.5 <= a < 22.5:
        return 'AHEAD'
    if 22.5 <= a < 67.5:
        return 'front-left'
    if 67.5 <= a < 112.5:
        return 'LEFT'
    if 112.5 <= a < 157.5:
        return 'rear-left'
    if -67.5 <= a < -22.5:
        return 'front-right'
    if -112.5 <= a < -67.5:
        return 'RIGHT'
    if -157.5 <= a < -112.5:
        return 'rear-right'
    return 'BEHIND'


class DropoutReport(Node):
    def __init__(self, want_scans, n_sectors):
        super().__init__('scan_dropout_report')
        self.want = want_scans
        self.n_sectors = n_sectors

        self.seen = 0
        self.rays_total = 0
        self.zero = 0             # exactly 0.0 -- the X2's dropout marker
        self.below_min = 0        # >0 but under range_min
        self.non_finite = 0       # inf/nan, i.e. invalid_range_is_inf took effect
        self.above_max = 0
        self.valid = 0

        self.sector_rays = [0] * n_sectors
        self.sector_bad = [0] * n_sectors

        self.per_scan_bad = []    # dropout fraction of each individual scan
        self.first_stamp = None
        self.last_stamp = None
        self.first_wall = None
        self.last_wall = None
        self.config = None

        self.sub = self.create_subscription(
            LaserScan, '/scan', self.on_scan, QOS)

    def on_scan(self, msg):
        now = time.monotonic()
        if self.first_wall is None:
            self.first_wall = now
        self.last_wall = now

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.first_stamp is None:
            self.first_stamp = stamp
            self.config = {
                'frame_id': msg.header.frame_id,
                'range_min': msg.range_min,
                'range_max': msg.range_max,
                'angle_min': msg.angle_min,
                'angle_max': msg.angle_max,
                'angle_increment': msg.angle_increment,
                'scan_time': msg.scan_time,
                'rays': len(msg.ranges),
            }
        self.last_stamp = stamp

        bad_here = 0
        for i, r in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            # Sector index from bearing, wrapped into [0, 2pi).
            frac = ((angle + math.pi) % (2 * math.pi)) / (2 * math.pi)
            s = min(int(frac * self.n_sectors), self.n_sectors - 1)
            self.sector_rays[s] += 1
            self.rays_total += 1

            if not math.isfinite(r):
                self.non_finite += 1
                bad = True
            elif r == 0.0:
                self.zero += 1
                bad = True
            elif r < msg.range_min:
                self.below_min += 1
                bad = True
            elif r > msg.range_max:
                self.above_max += 1
                bad = True
            else:
                self.valid += 1
                bad = False

            if bad:
                self.sector_bad[s] += 1
                bad_here += 1

        if msg.ranges:
            self.per_scan_bad.append(bad_here / len(msg.ranges))
        self.seen += 1

    @property
    def done(self):
        return self.seen >= self.want

    def report(self):
        if self.seen == 0 or self.rays_total == 0:
            print('No scans received. Is the driver up, and is /scan '
                  'BEST_EFFORT? See the QOS note at the top of this file.')
            return 1

        c = self.config
        bad = self.rays_total - self.valid
        frac = bad / self.rays_total

        # Rate two ways. Header stamps come from the lidar itself; wall time is
        # what this process observed. They disagree when the driver is dropping
        # scans on the floor, which is worth seeing.
        span_stamp = (self.last_stamp - self.first_stamp) if self.seen > 1 else 0.0
        span_wall = (self.last_wall - self.first_wall) if self.seen > 1 else 0.0
        hz_stamp = (self.seen - 1) / span_stamp if span_stamp > 0 else float('nan')
        hz_wall = (self.seen - 1) / span_wall if span_wall > 0 else float('nan')

        print()
        print('=' * 68)
        print(f'  /scan dropout report -- {self.seen} scans, frame '
              f'{c["frame_id"]!r}')
        print('=' * 68)
        print(f'  rays per scan        {c["rays"]}')
        print(f'  angle range          {math.degrees(c["angle_min"]):+.1f} to '
              f'{math.degrees(c["angle_max"]):+.1f} deg, '
              f'{math.degrees(c["angle_increment"]):.3f} deg/ray')
        print(f'  range window         {c["range_min"]:.2f} .. '
              f'{c["range_max"]:.2f} m')
        print(f'  rate                 {hz_stamp:.2f} Hz by header stamp, '
              f'{hz_wall:.2f} Hz by wall clock')
        print()
        print(f'  DROPOUT FRACTION     {frac * 100:.1f} %   '
              f'({bad} of {self.rays_total} rays)')
        if self.per_scan_bad:
            lo = min(self.per_scan_bad) * 100
            hi = max(self.per_scan_bad) * 100
            print(f'  per-scan spread      {lo:.1f} % .. {hi:.1f} %')
        print(f'  live returns/scan    {self.valid / self.seen:.0f}')
        print()
        print('  how the bad rays are marked:')
        print(f'    exactly 0.0        {self.zero}   '
              '(the X2 dropout marker; BELOW range_min)')
        print(f'    inf / nan          {self.non_finite}   '
              '(invalid_range_is_inf took effect)')
        print(f'    under range_min    {self.below_min}')
        print(f'    over range_max     {self.above_max}')
        if self.zero and self.non_finite:
            print('    NOTE: both conventions present in one run -- check '
                  'invalid_range_is_inf.')
        print()

        width = 360.0 / self.n_sectors
        print(f'  per-sector dropout ({self.n_sectors} sectors of '
              f'{width:.0f} deg, 0 deg = straight ahead):')
        rows = []
        for s in range(self.n_sectors):
            centre = -180.0 + (s + 0.5) * width
            rays = self.sector_rays[s]
            f = self.sector_bad[s] / rays if rays else float('nan')
            rows.append((f, centre, rays))
            bar = '#' * int(round(f * 40)) if rays else ''
            print(f'    {centre:+7.1f} deg  {sector_label(centre):<11} '
                  f'{f * 100:5.1f} %  {bar}')
        worst = max((r for r in rows if r[2]), default=None)
        best = min((r for r in rows if r[2]), default=None)
        print()
        if worst:
            print(f'  worst sector         {worst[1]:+.1f} deg '
                  f'({sector_label(worst[1])}) at {worst[0] * 100:.1f} %')
        if best:
            print(f'  best sector          {best[1]:+.1f} deg '
                  f'({sector_label(best[1])}) at {best[0] * 100:.1f} %')
        print()

        # Day 6 is the reason this number is being measured, so say what it
        # implies there rather than leaving it as a percentage to interpret.
        live = 1.0 - frac
        print('  what this means for Day 6 (detection.min_returns):')
        for dist in (1.0, 2.0, 3.0):
            # Rays subtended by a 0.3 m wide object at `dist`, times the
            # fraction that actually come back.
            span = 2 * math.atan2(0.15, dist)
            rays = span / c['angle_increment'] if c['angle_increment'] else 0
            print(f'    a 0.3 m object at {dist:.0f} m subtends '
                  f'{rays:.0f} rays -> about {rays * live:.1f} live returns')
        print('    Set min_returns below the 3 m figure or distant objects '
              'are silently dropped.')
        print()
        print('  Record this in records/calibration.md with today\'s date '
              'and the room.')
        print('=' * 68)
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--scans', type=int, default=DEFAULT_SCANS,
                    help=f'scans to accumulate (default {DEFAULT_SCANS})')
    ap.add_argument('--sectors', type=int, default=DEFAULT_SECTORS,
                    help=f'bearing bands to split the report into '
                         f'(default {DEFAULT_SECTORS})')
    args, ros_args = ap.parse_known_args()
    if args.sectors < 1:
        ap.error('--sectors must be at least 1')

    rclpy.init(args=sys.argv)
    node = DropoutReport(args.scans, args.sectors)
    print(f'Waiting for /scan ... collecting {args.scans} scans. '
          'Keep the robot still.')

    deadline = time.monotonic() + DISCOVERY_TIMEOUT
    rc = 1
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
            if node.seen == 0 and time.monotonic() > deadline:
                print(f'\nNo /scan in {DISCOVERY_TIMEOUT:.0f} s. The driver is '
                      'not running, or the topic is not /scan.')
                break
            if node.seen and node.seen % 20 == 0:
                print(f'  {node.seen}/{args.scans}', end='\r', flush=True)
        rc = node.report()
    except KeyboardInterrupt:
        print()
        rc = node.report()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return rc


if __name__ == '__main__':
    sys.exit(main())
