#!/usr/bin/env python3
"""Find what makes the robot rubber-band in RViz: jerk forward, then snap back.

Run this WHILE DRIVING, slowly, in a straight line. It watches the three things
that produce that exact shape and prints which one is actually happening.

    ros2 run my_bot check_pose_stability.py --seconds 30

WHY NOT `ros2 run tf2_ros tf_monitor`. Two reasons, both fatal:
  1. The executable is `tf2_monitor` in ROS 2, not `tf_monitor`.
  2. Its authority column CANNOT find a duplicate broadcaster here. ROS 1 read
     the publisher from the message's connection header; DDS has no equivalent,
     so Humble's listener hardcodes the authority -- libtf2_ros.so literally
     contains the string "Authority undetectable". Every edge reports the same
     made-up authority no matter how many nodes publish it.
     This script finds duplicates the way that actually works in ROS 2:
     the DDS publisher census on /tf, plus per-edge rate and stamp regression.

THE THREE SIGNATURES, and which section reports each:

  [1] TWO BROADCASTERS on one edge. Section "TF PUBLISHERS" names anyone
      publishing /tf beyond the expected three, and "PER-EDGE" shows the edge
      at roughly DOUBLE its expected rate with stamps that step backwards as
      the two publishers interleave. This is the only one of the three that is
      a true rubber band: the pose alternates between two answers every frame.

  [2] STALE OR JITTERING TIMESTAMPS. Section "MESSAGE TIMING". A scan stamped
      older than slam_toolbox's transform_timeout (0.2 s) is dropped outright;
      one stamped inconsistently gets matched against the wrong odom pose, and
      the correction that follows yanks the robot back.

  [3] SCAN MATCHER FIGHTING ODOM. Section "map -> odom CORRECTION" integrates
      how hard slam_toolbox is pulling. Healthy is a few mm of drift per metre
      driven plus the occasional loop closure. A correction that grows steadily
      against travel, or that reverses sign every keyframe, is the matcher
      rejecting where odom says the robot is.

CLOCK SKEW, if RViz is on the laptop. This runs on the Jetson and sees the
Jetson's clock only. Two machines whose clocks differ by more than a TF buffer
produce a rubber band that EXISTS ONLY IN RVIZ -- the robot maps fine. Check
that separately; --check-peer prints the offset.

WHAT THIS CANNOT SETTLE. Motion shear. Driving too fast smears each scan along
the path and doubles walls, and that is a MAP defect with a steady pose, not a
pose that snaps. If sections 1-3 come back clean, the map is smearing and the
pose is not jumping, you are driving too fast -- see the symptom index.
"""

import argparse
import math
import subprocess
import sys
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# Exactly these three nodes may publish /tf on the real robot. Anything else is
# the bug. robot_state_publisher owns the wheel joints (the fixed links go on
# /tf_static), diff_cont owns odom -> base_link, slam_toolbox owns map -> odom.
# Verified against the live graph 11 Sep: diff_drive_controller publishes /tf
# under its OWN node name, not the controller_manager's, even though it runs
# inside ros2_control_node.
EXPECTED_TF_PUBLISHERS = {
    'robot_state_publisher',
    'diff_cont',
    'slam_toolbox',
}

# slam_toolbox's transform_timeout in mapper_params_online_async.yaml. A scan
# older than this on arrival is not late, it is discarded.
TRANSFORM_TIMEOUT_S = 0.2


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Watcher(Node):

    def __init__(self, odom_topic, scan_topic):
        super().__init__('check_pose_stability')

        # /tf is RELIABLE VOLATILE, depth 100 -- match it or lose messages and
        # invent a rate problem that is not there.
        tf_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=100,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(TFMessage, '/tf', self.on_tf, tf_qos)
        self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        self.create_subscription(LaserScan, scan_topic, self.on_scan, sensor_qos)

        # per edge "parent->child"
        self.edge_count = defaultdict(int)
        self.edge_last_stamp = {}
        self.edge_regressions = defaultdict(int)   # stamp went backwards
        self.edge_dup_stamp = defaultdict(int)     # same stamp, different value
        self.edge_last_xy = {}
        self.edge_first_t = {}
        self.edge_last_t = {}

        # map -> odom correction history
        self.mo_samples = []        # (wall_t, x, y, yaw)
        self.mo_jumps = []          # (wall_t, d_trans, d_yaw)

        # message timing: list of (wall_recv, stamp)
        self.odom_timing = []
        self.scan_timing = []
        self.odom_backwards = 0
        self.scan_backwards = 0
        self.odom_last_stamp = None
        self.scan_last_stamp = None
        self.odom_travel = 0.0
        self.odom_last_xy = None

        self.scan_sweep = None      # scan_time * (ranges-1), the sweep duration

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_tf(self, msg):
        now = self._now()
        for t in msg.transforms:
            edge = f'{t.header.frame_id.lstrip("/")}->{t.child_frame_id.lstrip("/")}'
            stamp = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
            x, y = t.transform.translation.x, t.transform.translation.y
            yaw = yaw_of(t.transform.rotation)

            prev = self.edge_last_stamp.get(edge)
            if prev is not None:
                if stamp < prev:
                    # Two broadcasters interleaving is the usual cause. One
                    # publisher cannot stamp backwards.
                    self.edge_regressions[edge] += 1
                elif stamp == prev:
                    px, py = self.edge_last_xy.get(edge, (x, y))
                    if abs(px - x) > 1e-9 or abs(py - y) > 1e-9:
                        # Same instant, two different answers. Conclusive.
                        self.edge_dup_stamp[edge] += 1

            self.edge_last_stamp[edge] = stamp
            self.edge_last_xy[edge] = (x, y)
            self.edge_count[edge] += 1
            self.edge_first_t.setdefault(edge, now)
            self.edge_last_t[edge] = now

            if edge == 'map->odom':
                if self.mo_samples:
                    _, px, py, pyaw = self.mo_samples[-1]
                    d = math.hypot(x - px, y - py)
                    dyaw = abs(math.atan2(math.sin(yaw - pyaw),
                                          math.cos(yaw - pyaw)))
                    if d > 1e-4 or dyaw > 1e-4:
                        self.mo_jumps.append((now, d, dyaw))
                self.mo_samples.append((now, x, y, yaw))

    def on_odom(self, msg):
        now = self._now()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.odom_last_stamp is not None and stamp < self.odom_last_stamp:
            self.odom_backwards += 1
        self.odom_last_stamp = stamp
        self.odom_timing.append((now, stamp))

        p = msg.pose.pose.position
        if self.odom_last_xy is not None:
            self.odom_travel += math.hypot(p.x - self.odom_last_xy[0],
                                           p.y - self.odom_last_xy[1])
        self.odom_last_xy = (p.x, p.y)

    def on_scan(self, msg):
        now = self._now()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.scan_last_stamp is not None and stamp < self.scan_last_stamp:
            self.scan_backwards += 1
        self.scan_last_stamp = stamp
        self.scan_timing.append((now, stamp))
        if self.scan_sweep is None and msg.scan_time > 0.0:
            # LaserScan.scan_time is already the FULL sweep period in seconds.
            # time_increment is the per-ray figure. Multiplying the two together
            # is a factor-of-350 error (caught 11 Sep: it reported a 30 s sweep).
            self.scan_sweep = msg.scan_time


def stats(values):
    if not values:
        return (0.0, 0.0, 0.0)
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return (mean, math.sqrt(var), max(values))


def tf_publisher_census(node):
    """Who is publishing /tf, by DDS endpoint. This is what tf2_monitor cannot do."""
    out = []
    for topic in ('/tf', '/tf_static'):
        infos = node.get_publishers_info_by_topic(topic)
        names = [f'{i.node_namespace.rstrip("/")}/{i.node_name}'.lstrip('/') or i.node_name
                 for i in infos]
        out.append((topic, names))
    return out


def peer_clock_offset(host):
    """Round-trip the peer's clock. Anything past ~50 ms will show in RViz."""
    t0 = time.time()
    try:
        r = subprocess.run(
            ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
             host, 'date +%s.%N'],
            capture_output=True, text=True, timeout=15)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return None, f'unreachable ({e.__class__.__name__})'
    t1 = time.time()
    if r.returncode != 0:
        return None, f'unreachable ({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "rc=%d" % r.returncode})'
    try:
        peer = float(r.stdout.strip())
    except ValueError:
        return None, 'peer did not return a timestamp'
    # Peer clock read somewhere inside [t0, t1]; compare against the midpoint.
    return peer - (t0 + t1) / 2.0, None


def main():
    ap = argparse.ArgumentParser(
        description='Diagnose RViz rubber-banding: TF conflicts, stamp jitter, '
                    'and slam_toolbox fighting odometry.')
    ap.add_argument('--seconds', type=float, default=30.0,
                    help='Observation window. DRIVE during it. Default 30.')
    ap.add_argument('--odom-topic', default='/diff_cont/odom')
    ap.add_argument('--scan-topic', default='/scan')
    ap.add_argument('--check-peer', metavar='USER@HOST',
                    help='Also measure clock offset to the RViz laptop, '
                         'e.g. ju@172.20.10.5')
    args = ap.parse_args()

    rclpy.init()
    node = Watcher(args.odom_topic, args.scan_topic)

    print(f'Watching for {args.seconds:.0f} s. DRIVE THE ROBOT NOW, slowly, '
          f'in a straight line.\n', flush=True)

    deadline = time.time() + args.seconds
    while rclpy.ok() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    findings = []

    # ---------------------------------------------------------------- [1] TF
    print('=' * 68)
    print('TF PUBLISHERS  (duplicate broadcaster check)')
    print('=' * 68)
    for topic, names in tf_publisher_census(node):
        print(f'  {topic}: {len(names)} publisher(s)')
        seen = defaultdict(int)
        for n in sorted(names):
            seen[n] += 1
        for n, c in sorted(seen.items()):
            base = n.split('/')[-1]
            mark = ''
            if c > 1:
                # A node may hold several DDS writers on one topic and send on
                # only one of them -- slam_toolbox does exactly this (measured
                # 11 Sep: 2 endpoints, map -> odom still at exactly 50 Hz, which
                # is 1/transform_publish_period, so one writer is idle). Endpoint
                # count alone proves nothing. The per-edge section below is what
                # decides, and `pgrep` settles whether two stacks are up.
                mark = f'   <== {c} endpoints (see PER-EDGE before concluding)'
            elif topic == '/tf' and base not in EXPECTED_TF_PUBLISHERS:
                mark = '   <== UNEXPECTED'
                findings.append(
                    f'{base} publishes /tf and should not. Only '
                    f'{", ".join(sorted(EXPECTED_TF_PUBLISHERS))} may.')
            print(f'      {n}{mark}')
    print()

    print('=' * 68)
    print('PER-EDGE  (rate, and evidence of two sources on one edge)')
    print('=' * 68)
    print(f'  {"edge":<34}{"count":>7}{"Hz":>8}{"back":>7}{"dup":>6}')
    expected_hz = {'odom->base_link': 30.0, 'map->odom': 50.0}
    for edge in sorted(node.edge_count):
        c = node.edge_count[edge]
        span = node.edge_last_t[edge] - node.edge_first_t[edge]
        hz = (c - 1) / span if span > 0 and c > 1 else 0.0
        back = node.edge_regressions[edge]
        dup = node.edge_dup_stamp[edge]
        flag = ''
        exp = expected_hz.get(edge)
        if exp and hz > exp * 1.6:
            flag = f'  <== ~2x the {exp:.0f} Hz one publisher gives'
            findings.append(
                f'{edge} arrives at {hz:.1f} Hz against {exp:.0f} Hz expected '
                f'from a single publisher. Two nodes are publishing it.')
        if dup:
            flag = '  <== SAME STAMP, DIFFERENT POSE'
            findings.append(
                f'{edge}: {dup} transforms share a stamp but disagree on '
                f'position. That is conclusively two broadcasters.')
        elif back and not flag:
            flag = '  <== stamps step backwards'
        print(f'  {edge:<34}{c:>7}{hz:>8.1f}{back:>7}{dup:>6}{flag}')
    if not node.edge_count:
        print('  NOTHING ON /tf. Is the stack up? Is ROS_DOMAIN_ID 42?')
        findings.append('No /tf traffic at all.')
    print()

    # ------------------------------------------------------------ [2] timing
    print('=' * 68)
    print('MESSAGE TIMING  (stamp age and jitter, against this node\'s clock)')
    print('=' * 68)
    for label, timing, backwards, vs_timeout in (
            (args.odom_topic, node.odom_timing, node.odom_backwards, False),
            (args.scan_topic, node.scan_timing, node.scan_backwards, True)):
        if not timing:
            print(f'  {label}: NO MESSAGES')
            findings.append(f'{label} published nothing during the window.')
            continue
        ages = sorted(recv - stamp for recv, stamp in timing)
        mean_age, sd_age, max_age = stats(ages)
        p50 = ages[len(ages) // 2]
        p99 = ages[min(len(ages) - 1, int(len(ages) * 0.99))]
        span = timing[-1][0] - timing[0][0]
        hz = (len(timing) - 1) / span if span > 0 else 0.0
        print(f'  {label}')
        print(f'      {len(timing)} msgs at {hz:.1f} Hz')
        print(f'      stamp age on arrival: mean {mean_age*1e3:7.1f} ms  '
              f'sd {sd_age*1e3:5.1f} ms')
        print(f'      p50 {p50*1e3:6.1f} ms   p99 {p99*1e3:6.1f} ms   '
              f'max {max_age*1e3:6.1f} ms')
        print(f'      stamps going backwards: {backwards}')
        # Judge on p99, not max. One outlier is a scheduling hiccup; a p99 past
        # the timeout means scans are being discarded routinely.
        if vs_timeout and p99 > TRANSFORM_TIMEOUT_S:
            print(f'      <== p99 EXCEEDS transform_timeout '
                  f'({TRANSFORM_TIMEOUT_S} s); slam_toolbox DROPS these')
            findings.append(
                f'{label} is {p99*1e3:.0f} ms old at the 99th percentile, past '
                f'the {TRANSFORM_TIMEOUT_S*1e3:.0f} ms transform_timeout. Those '
                f'scans are discarded routinely, so the matcher works from a '
                f'gappy history and corrects hard when it does match.')
        elif vs_timeout and max_age > TRANSFORM_TIMEOUT_S:
            print(f'      (max is past transform_timeout but p99 is not: '
                  f'occasional hiccup, not a systemic drop)')
        if backwards:
            findings.append(
                f'{label}: {backwards} messages stamped earlier than their '
                f'predecessor. Either two publishers, or the clock stepped.')
        # Jitter against the MEDIAN, and only when the spread is big in
        # absolute terms. A 4 ms median with a couple of 300 ms outliers has a
        # huge relative sd and is not a problem.
        if len(timing) > 20 and (p99 - p50) > 0.050:
            findings.append(
                f'{label} spreads {(p99-p50)*1e3:.0f} ms between its median '
                f'({p50*1e3:.0f} ms) and p99 ({p99*1e3:.0f} ms). The matcher '
                f'pairs scans with odom poses from the wrong instant.')
    if node.scan_sweep:
        print(f'  /scan sweep duration: {node.scan_sweep*1e3:.0f} ms '
              f'(shear = sweep x speed; 0.5 m/s gives '
              f'{node.scan_sweep*0.5*100:.1f} cm per scan, 0.10 m/s gives '
              f'{node.scan_sweep*0.10*100:.1f} cm)')
    print()

    # --------------------------------------------------------- [3] map->odom
    print('=' * 68)
    print('map -> odom CORRECTION  (is the scan matcher fighting odom?)')
    print('=' * 68)
    if len(node.mo_samples) < 2:
        print('  No map -> odom seen. slam_toolbox is not running, or not '
              'matching at all.')
        findings.append('slam_toolbox published no map -> odom.')
    else:
        _, x0, y0, yaw0 = node.mo_samples[0]
        _, x1, y1, yaw1 = node.mo_samples[-1]
        net = math.hypot(x1 - x0, y1 - y0)
        net_yaw = math.degrees(abs(math.atan2(math.sin(yaw1 - yaw0),
                                              math.cos(yaw1 - yaw0))))
        total = sum(d for _, d, _ in node.mo_jumps)
        big = [(t, d, a) for t, d, a in node.mo_jumps if d > 0.02]
        print(f'  odom travelled       : {node.odom_travel:.3f} m')
        print(f'  net correction       : {net*100:.1f} cm, {net_yaw:.2f} deg')
        print(f'  total path of the correction: {total*100:.1f} cm '
              f'over {len(node.mo_jumps)} updates')
        print(f'  corrections > 2 cm   : {len(big)}')
        if big:
            print(f'      largest: {max(d for _, d, _ in big)*100:.1f} cm')
        # A correction that wanders far more than it ends up from where it
        # started is oscillating, not drifting -- that is the snap-back.
        if net > 1e-3 and total > 4.0 * net and len(node.mo_jumps) > 10:
            print('  <== The correction travels far more than it nets. '
                  'It is OSCILLATING, not drifting.')
            findings.append(
                f'map -> odom moved {total*100:.0f} cm in total but ended only '
                f'{net*100:.0f} cm from where it started. slam_toolbox is '
                f'pushing the robot back and forth -- that is the snap-back '
                f'you see, and it means the matcher disagrees with odom every '
                f'keyframe.')
        if node.odom_travel > 0.3 and net / node.odom_travel > 0.10:
            findings.append(
                f'map -> odom corrected {net*100:.0f} cm over '
                f'{node.odom_travel:.2f} m driven ({net/node.odom_travel*100:.0f}%). '
                f'Odometry scale or the scan geometry is wrong, not merely noisy.')
        if len(big) > 3:
            findings.append(
                f'{len(big)} corrections over 2 cm in {args.seconds:.0f} s. '
                f'Healthy mapping corrects at the millimetre level between '
                f'loop closures.')
    print()

    # ------------------------------------------------------------ clock skew
    if args.check_peer:
        print('=' * 68)
        print('PEER CLOCK  (rubber-banding that exists only in RViz)')
        print('=' * 68)
        off, err = peer_clock_offset(args.check_peer)
        if err:
            print(f'  {args.check_peer}: {err}')
        else:
            print(f'  {args.check_peer} clock is {off*1e3:+.0f} ms from this board')
            if abs(off) > 0.05:
                print('  <== TOO FAR. RViz interpolates TF against ITS OWN clock.')
                findings.append(
                    f'The RViz laptop\'s clock is {off*1e3:+.0f} ms off this '
                    f'board. RViz resolves every transform against its own '
                    f'clock, so the robot model and the map are drawn at '
                    f'different instants -- a rubber band that is purely a '
                    f'display artefact. Sync both machines before believing '
                    f'anything you see on screen.')
        print()

    # ---------------------------------------------------------------- verdict
    print('=' * 68)
    print('VERDICT')
    print('=' * 68)
    if findings:
        for i, f in enumerate(findings, 1):
            print(f'  {i}. {f}')
    else:
        print('  No TF conflict, no stamp problem, no matcher oscillation.')
        print('  If the MAP still smears while the POSE holds steady, this is')
        print('  motion shear and not a rubber band: you are driving too fast.')
        print('  See troubleshooting/symptom-index.md, "The map smears when')
        print('  driving forward".')
    print()

    node.destroy_node()
    rclpy.shutdown()
    return 1 if findings else 0


if __name__ == '__main__':
    sys.exit(main())
