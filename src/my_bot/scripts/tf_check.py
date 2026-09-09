#!/usr/bin/env python3
"""Assert the TF chain the robot actually depends on, and print each edge's age.

Catches the silent TF gap that makes SLAM, Nav2 or the semantic layer look
broken when the real fault is a node that never started. Run this BEFORE
blaming anything else.

WHAT A GAP LOOKS LIKE. Nothing errors. RViz shows a robot, topics publish, the
logs are clean -- there is simply no transform between two frames, and every
consumer that needs it silently drops its work. `view_frames` renders a PDF you
then have to open and read; this prints the answer.

WHAT THIS CANNOT SETTLE. That a transform RESOLVES says nothing about whether
it is CORRECT. A camera bolted on backwards resolves. laser_joint yawed 180
degrees resolves. An assumed sign on camera_offset_y resolves. This tool proves
the tree is connected and fresh, which is a precondition for the geometry being
right and no kind of evidence that it is.

Two edges are worth understanding rather than just seeing green:

  map -> odom      published by slam_toolbox. MISSING until SLAM runs, which is
                   expected before Day 3 -- hence --no-map. If it is present but
                   pinned at exactly identity, SLAM is running and failing to
                   match scans, which is a WORSE state than missing and this
                   tool flags it.
  odom -> base_link  published by diff_cont. If this is missing the controller
                   did not spawn; check `ros2 control list_controllers`.

The static edges (base_link -> laser_frame, -> camera_link, -> camera_optical_link)
come from robot_state_publisher off the URDF. If one is missing, rsp is down or
the xacro did not include that file.

    ros2 run my_bot tf_check.py              # expects map -> odom too
    ros2 run my_bot tf_check.py --no-map     # before SLAM exists (Day 2)
"""

import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener

SETTLE_S = 3.0
# Anything older than this is stale rather than merely late: diff_cont publishes
# odom TF at 50 Hz and rsp republishes static edges on latch.
STALE_S = 1.0


def quat_to_rpy(q):
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-map', action='store_true',
                    help='skip map -> odom; use before slam_toolbox exists')
    args = ap.parse_args()

    edges = []
    if not args.no_map:
        edges.append(('map', 'odom'))
    edges += [
        ('odom', 'base_link'),
        ('base_link', 'base_footprint'),
        ('base_link', 'laser_frame'),
        ('base_link', 'camera_link'),
        ('base_link', 'camera_optical_link'),
        ('base_link', 'left_wheel'),
        ('base_link', 'right_wheel'),
    ]

    rclpy.init()
    node = Node('tf_check')
    buf = Buffer()
    TransformListener(buf, node)

    # A listener needs time to fill. Without this every edge reports missing on
    # a perfectly healthy robot, which is a very convincing false alarm.
    end = node.get_clock().now() + Duration(seconds=SETTLE_S)
    while rclpy.ok() and node.get_clock().now() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    now = node.get_clock().now()
    width = max(len(f'{p} -> {c}') for p, c in edges)
    failures = []
    warnings = []

    print()
    for parent, child in edges:
        label = f'{parent} -> {child}'.ljust(width)
        try:
            tf = buf.lookup_transform(parent, child, rclpy.time.Time())
        except Exception as exc:
            print(f'  {label}  MISSING   {type(exc).__name__}')
            failures.append(f'{parent} -> {child}')
            continue

        stamp = rclpy.time.Time.from_msg(tf.header.stamp)
        # A static transform is latched with an old stamp on purpose; treat a
        # zero stamp as "static", not as infinitely stale.
        age = 0.0 if stamp.nanoseconds == 0 else (now - stamp).nanoseconds / 1e9
        t = tf.transform.translation
        _, _, yaw = quat_to_rpy(tf.transform.rotation)
        flag = ''
        if age > STALE_S:
            flag = f'  STALE ({age:.1f}s)'
            warnings.append(f'{parent} -> {child} is {age:.1f}s old')
        print(f'  {label}  ok  xyz=({t.x:+.3f}, {t.y:+.3f}, {t.z:+.3f})  '
              f'yaw={math.degrees(yaw):+7.2f}deg  age={age:.2f}s{flag}')

        if (parent, child) == ('map', 'odom'):
            if abs(t.x) < 1e-9 and abs(t.y) < 1e-9 and abs(yaw) < 1e-9:
                warnings.append(
                    'map -> odom is EXACTLY identity. slam_toolbox is running '
                    'and matching nothing -- see the lidar entries in '
                    'troubleshooting/symptom-index.md, not this tool.')

    print()
    for w in warnings:
        print(f'  WARNING: {w}')
    if failures:
        print(f'\n  FAILED: {len(failures)} edge(s) missing: {", ".join(failures)}')
        print('  A missing edge means a node did not start, not that geometry is wrong.')
    else:
        print('  All edges resolve.')
    print('\n  Resolving is NOT the same as being correct. This proves the tree'
          '\n  is connected; it proves nothing about the numbers in it.\n')

    node.destroy_node()
    rclpy.shutdown()
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
