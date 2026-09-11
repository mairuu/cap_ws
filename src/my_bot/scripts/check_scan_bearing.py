#!/usr/bin/env python3
"""Point at the robot and check the scan agrees. Settles `reversion` and `inverted`.

WHAT THIS IS FOR. config/ydlidar.yaml sets two orientation flags, both `true`:

    reversion: true    the puck's 0 deg reference faces the robot's REAR, so the
                       SDK adds pi to every bearing to undo it
    inverted:  true    the X3 Pro numbers its rays CLOCKWISE; REP-103 wants
                       counter-clockwise, so the SDK applies angle = 2pi - angle

Both values were recovered from the previous board. They describe how the lidar
is BOLTED ON, not how the sensor behaves, so neither transfers automatically to
a rebuilt robot. `true` is only correct while the puck is mounted the same way
round. Nothing in the message says which way that is.

WHY THE EXISTING TOOLS DO NOT CATCH IT. check_scan_world_fixed.py rotates the
robot and checks the scan stays put in the odom frame. That catches `inverted`,
because a mirrored scan counter-rotates at twice the yaw rate. It CANNOT catch
`reversion`: a scan rotated by pi is still world-fixed under rotation. The
symptom of a wrong `reversion` is that driving FORWARD smears the map, because
the error is a reflection through the lidar centre and that point moves with the
robot. By the time you see that smear you are debugging SLAM, which is the wrong
place to be looking.

THE TEST. Put something the lidar can see -- a box, a chair leg, your own shin
-- about half a metre from the robot in a KNOWN direction, and read the bearing
this prints. Then move it to another side. Each of the four flag combinations
gives a different, unmistakable answer:

    object placed   correct   reversion    inverted    both
                              wrong        wrong       wrong
    ------------------------------------------------------------
    IN FRONT          0 deg    180 deg       0 deg     180 deg
    to its LEFT     +90 deg    -90 deg     -90 deg     +90 deg
    to its RIGHT    -90 deg    +90 deg     +90 deg     -90 deg
    BEHIND          180 deg      0 deg     180 deg       0 deg

FRONT/BEHIND alone distinguishes `reversion`. LEFT/RIGHT alone distinguishes
`inverted`. Do both and you have settled the pair, in about thirty seconds,
with the robot stationary and nothing commanded.

BEARINGS ARE REP-103, in the laser frame: 0 deg straight ahead, +90 to the
robot's LEFT, +/-180 behind. The driver has already applied both flags by the
time the message reaches us, so what this prints is what slam_toolbox, Nav2 and
the semantic layer all see.

IF IT IS WRONG, FIX IT IN config/ydlidar.yaml. Do NOT yaw laser_joint in
description/lidar.xacro to compensate -- that file is shared with the simulator
and would break the sim scan instead. Neither flag shows up in simulation at
all: Gazebo's gpu_lidar sits at laser_frame with identity rotation and counts
counter-clockwise natively, so these are real-robot-only failures.

IT COMMANDS NOTHING and needs no TF. Only the lidar driver has to be running:

    make real
    ros2 run my_bot check_scan_bearing.py
    ros2 run my_bot check_scan_bearing.py --sectors 12 --max-range 3.0

Ctrl-C to stop.
"""

import argparse
import math
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

# /scan is BEST_EFFORT. A default (RELIABLE) subscription never matches it and
# never says why -- it simply receives nothing, which looks exactly like a dead
# lidar. This is not optional. See troubleshooting/symptom-index.md.
QOS = qos_profile_sensor_data

# A dropout arrives as 0.0, which is BELOW range_min, not as inf --
# invalid_range_is_inf is false in ydlidar.yaml. Filtering only inf/nan would
# read every dropout as an obstacle zero metres away, dead centre on the robot.
MIN_VALID = 1e-3


def sector_label(deg):
    """REP-103 bearing -> a word, so the output cannot be misread as clockwise."""
    a = abs(deg)
    if a <= 15.0:
        return 'AHEAD'
    if a >= 165.0:
        return 'BEHIND'
    side = 'LEFT' if deg > 0 else 'RIGHT'
    if a <= 75.0:
        return 'front-' + side.lower()
    if a >= 105.0:
        return 'rear-' + side.lower()
    return side


class BearingWatch(Node):

    def __init__(self, n_sectors, max_range, hz):
        super().__init__('check_scan_bearing')
        self.n_sectors = n_sectors
        self.max_range = max_range
        self.every = max(1, int(round(11.5 / max(hz, 0.1))))
        self.count = 0
        self.seen = False
        self.create_subscription(LaserScan, '/scan', self.on_scan, QOS)

        print(__doc__.split('THE TEST.')[1].split('BEARINGS ARE')[0].strip())
        print()
        print('=' * 70)
        print('  waiting for /scan ...')

    def on_scan(self, msg):
        self.count += 1
        if not self.seen:
            self.seen = True
            print('  got it. %d rays, %.3f deg/ray, range %.2f..%.2f m'
                  % (len(msg.ranges), math.degrees(msg.angle_increment),
                     msg.range_min, msg.range_max))
            print('=' * 70)
        if self.count % self.every:
            return

        # Nearest valid return, and the minimum range in each sector.
        best_r = float('inf')
        best_a = 0.0
        sect = [float('inf')] * self.n_sectors
        for i, r in enumerate(msg.ranges):
            if r < MIN_VALID or r > self.max_range:
                continue
            if not math.isfinite(r):
                continue
            a = msg.angle_min + i * msg.angle_increment
            # Wrap to (-pi, pi] so the printed bearing matches REP-103 rather
            # than the driver's 0..2pi indexing.
            a = math.atan2(math.sin(a), math.cos(a))
            if r < best_r:
                best_r, best_a = r, a
            frac = (math.degrees(a) + 180.0) / 360.0
            s = min(int(frac * self.n_sectors), self.n_sectors - 1)
            if r < sect[s]:
                sect[s] = r

        if not math.isfinite(best_r):
            print('  nothing within %.1f m -- move the object closer, or raise '
                  '--max-range' % self.max_range)
            return

        deg = math.degrees(best_a)
        print()
        print('  NEAREST  %.3f m  at  %+7.1f deg   %s'
              % (best_r, deg, sector_label(deg)))

        width = 360.0 / self.n_sectors
        cells = []
        for s in range(self.n_sectors):
            centre = -180.0 + (s + 0.5) * width
            r = sect[s]
            cells.append('%+5.0f:%s' % (centre, '  --  ' if not math.isfinite(r)
                                        else '%.2fm' % r))
        for i in range(0, len(cells), 4):
            print('    ' + '   '.join(cells[i:i + 4]))


def main():
    ap = argparse.ArgumentParser(
        description='Live nearest-return bearing, to verify reversion/inverted.')
    ap.add_argument('--sectors', type=int, default=8,
                    help='sectors in the per-bearing minimum-range row '
                         '(default 8, i.e. 45 deg each)')
    ap.add_argument('--max-range', type=float, default=2.0,
                    help='ignore returns beyond this, so the room does not '
                         'outvote the object you are holding (default 2.0 m)')
    ap.add_argument('--hz', type=float, default=2.0,
                    help='print rate (default 2.0)')
    args = ap.parse_args()

    rclpy.init()
    node = BearingWatch(args.sectors, args.max_range, args.hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nstopped')
    except ExternalShutdownException:
        # SIGTERM, e.g. from `timeout` or a launch shutdown. Exiting on a
        # traceback here would look like the script had failed.
        print('\nstopped (external shutdown)')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
