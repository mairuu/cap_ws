#!/usr/bin/env python3
"""Score SLAM's reported pose against tape-measured ground truth. Objective 1.

WHAT THIS IS FOR. The report's objective 1 sets a number -- positional error
not worse than 10 cm over a test area of at least 5x5 m -- and the evaluation
table names the method: mark reference points whose true coordinates you know,
put the robot on them, and compare the pose the system reports with the truth.
This script is the "compare" half. It reads `map -> base_footprint` off TF,
averages it while the robot sits still, and writes one line per visit to a
session file. `--summary` turns those lines into the two numbers the report
needs.

    ABSOLUTE ERROR   distance from the reported pose to the tape truth for
                     that mark. Needs --truth. This is the objective's number.
    REPEATABILITY    how far apart repeated visits to the SAME mark land.
                     Needs no tape at all, and it is the honest measure of
                     what SLAM contributes, because it contains no tape error
                     and no parking error of the kind a single visit hides.
    SCALE            for any two marks whose truths are both known, the
                     distance the system reports between them against the
                     distance you measured. A pure scale error (wheel radius,
                     ticks per rev) shows here and nowhere else.

WHERE THE MAP FRAME'S ORIGIN IS, AND WHY IT DECIDES YOUR TAPE WORK.
`slam_toolbox` puts the map origin at the robot's pose when it started, with
+x along the robot's heading and +y to its LEFT (REP-103). So every --truth
you give must be measured from THAT spot in THAT direction -- not from a room
corner, unless the robot started on one with a known heading.

The practical recipe, which needs one tape and a floor marker:

  1. Tape a cross on the floor. Park the robot on it, note the heading it
     faces (aim it down a wall; that makes step 4 easy).
  2. `make real`, then `make slam`. The origin is now that cross.
  3. `slam_accuracy_check.py mark HOME --truth 0 0` -- this should read
     ~(0,0) and is the check that TF is sane before you spend an hour.
  4. Tape crosses at your other marks and measure each one from HOME: x along
     the heading of step 1, y to the left of it. Right of it is NEGATIVE y.
  5. Drive the loop. Park on each mark, run the script with that mark's name
     and truth. Park on HOME again at the end of every lap.
  6. Repeat the lap at least three times, then `--summary`.

WHAT THE NUMBER INCLUDES, and say this in the report. Parking the robot on a
cross by hand is worth a couple of centimetres on its own, and the tape is
worth a centimetre or two more. Both are inside the measurement, so the
reported error is an UPPER BOUND on SLAM's own error, never an under-estimate.
That is the right direction to be wrong in for a pass/fail criterion, and it
is why REPEATABILITY is reported alongside: it drops the tape out entirely.

RUN IT WITH THE ROBOT STATIONARY. It averages over --settle seconds and will
refuse the sample if the pose moves more than --still-tol during the window,
because a pose captured mid-roll measures your reflexes, not the mapper.

    ros2 run my_bot slam_accuracy_check.py mark HOME --truth 0 0
    ros2 run my_bot slam_accuracy_check.py mark CORNER_A --truth 4.20 -2.65
    ros2 run my_bot slam_accuracy_check.py --summary

Each visit appends to --session (default ~/maps/slam_accuracy.jsonl). Start a
fresh file for a fresh mapping session: the marks are only comparable within
one run of `make slam`, because the map frame is re-created every time.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

import rclpy
from rclpy.node import Node

import tf2_ros


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def circular_mean(angles):
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return math.atan2(s, c)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class PoseSampler(Node):
    def __init__(self, map_frame, base_frame):
        super().__init__("slam_accuracy_check")
        self.map_frame = map_frame
        self.base_frame = base_frame
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)

    def lookup(self):
        """Latest map -> base transform, or None if it is not there yet."""
        try:
            tf = self.buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t = tf.transform.translation
        return t.x, t.y, yaw_of(tf.transform.rotation)


def sample(node, seconds, still_tol):
    """Average the pose over `seconds`. Returns (x, y, yaw, spread) or None."""
    deadline = time.monotonic() + seconds
    xs, ys, yaws = [], [], []
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        p = node.lookup()
        if p is not None:
            xs.append(p[0])
            ys.append(p[1])
            yaws.append(p[2])
    if len(xs) < 5:
        return None
    x, y = statistics.median(xs), statistics.median(ys)
    yaw = circular_mean(yaws)
    spread = max(math.hypot(a - x, b - y) for a, b in zip(xs, ys))
    if spread > still_tol:
        print(f"!! the pose moved {spread*100:.1f} cm during the window "
              f"(tolerance {still_tol*100:.0f} cm).")
        print("   Either the robot is not stationary, or the matcher is "
              "correcting hard. Sample NOT recorded.")
        return None
    return x, y, yaw, spread


def wait_for_tf(node, map_frame, base_frame, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.lookup() is not None:
            return True
    print(f"no {map_frame} -> {base_frame} transform after {timeout:.0f} s.")
    print("Is `make slam` running? `make real` alone publishes odom, not map.")
    return False


def do_mark(args):
    rclpy.init()
    node = PoseSampler(args.map_frame, args.base_frame)
    try:
        if not wait_for_tf(node, args.map_frame, args.base_frame, args.timeout):
            return 1
        print(f"sampling {args.settle:.0f} s -- hold still ...")
        got = sample(node, args.settle, args.still_tol)
        if got is None:
            return 1
        x, y, yaw, spread = got
    finally:
        node.destroy_node()
        rclpy.shutdown()

    row = {
        "mark": args.mark,
        "t": time.time(),
        "x": x, "y": y, "yaw_deg": math.degrees(yaw),
        "window_spread_m": spread,
    }
    print(f"\n{args.mark}: reported ({x:+.3f}, {y:+.3f}) m, "
          f"yaw {math.degrees(yaw):+.1f} deg   "
          f"[window spread {spread*100:.1f} cm]")

    if args.truth is not None:
        tx, ty = args.truth
        err = math.hypot(x - tx, y - ty)
        row.update(truth_x=tx, truth_y=ty, error_m=err)
        verdict = "PASS" if err <= args.error_max else "FAIL"
        print(f"{'':>{len(args.mark)}}  truth    ({tx:+.3f}, {ty:+.3f}) m")
        print(f"{'':>{len(args.mark)}}  error     {err*100:6.1f} cm  "
              f"(dx {(x-tx)*100:+.1f}, dy {(y-ty)*100:+.1f})  "
              f"-> {verdict} against {args.error_max*100:.0f} cm")

    with open(os.path.expanduser(args.session), "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"\nappended to {args.session}")
    print("more visits to the same mark give repeatability; "
          "`--summary` when the laps are done.")
    return 0


def do_summary(args):
    path = os.path.expanduser(args.session)
    if not os.path.exists(path):
        print(f"no session file at {path}. Record visits with `mark` first.")
        return 1
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        print(f"{path} is empty.")
        return 1

    marks = {}
    for r in rows:
        marks.setdefault(r["mark"], []).append(r)

    print(f"=== SLAM accuracy, {len(rows)} visit(s) to "
          f"{len(marks)} mark(s) ===")
    print(f"    session {path}\n")

    print(f"{'mark':<12}{'n':>3}  {'mean reported':>22}  "
          f"{'abs error':>11}  {'repeatability':>14}")
    worst_err = worst_rep = 0.0
    any_truth = False
    for name, visits in marks.items():
        xs = [v["x"] for v in visits]
        ys = [v["y"] for v in visits]
        cx, cy = statistics.fmean(xs), statistics.fmean(ys)
        rep = max((math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)),
                  default=0.0)
        worst_rep = max(worst_rep, rep)
        errs = [v["error_m"] for v in visits if "error_m" in v]
        if errs:
            any_truth = True
            worst_err = max(worst_err, max(errs))
            ecol = f"{statistics.fmean(errs)*100:6.1f} cm"
        else:
            ecol = "   no truth"
        repcol = f"{rep*100:6.1f} cm" if len(visits) > 1 else "  (1 visit)"
        print(f"{name:<12}{len(visits):>3}  ({cx:+7.3f}, {cy:+7.3f}) m  "
              f"{ecol:>11}  {repcol:>14}")

    truths = {n: (v[0]["truth_x"], v[0]["truth_y"])
              for n, v in marks.items() if "truth_x" in v[0]}
    if len(truths) >= 2:
        print(f"\n{'pair':<26}{'reported':>10}{'tape':>10}{'error':>10}")
        names = sorted(truths)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ax = statistics.fmean([v["x"] for v in marks[a]])
                ay = statistics.fmean([v["y"] for v in marks[a]])
                bx = statistics.fmean([v["x"] for v in marks[b]])
                by = statistics.fmean([v["y"] for v in marks[b]])
                d_rep = math.hypot(bx - ax, by - ay)
                d_tape = math.hypot(truths[b][0] - truths[a][0],
                                    truths[b][1] - truths[a][1])
                print(f"{a + ' - ' + b:<26}{d_rep:7.3f} m{d_tape:8.3f} m"
                      f"{(d_rep - d_tape)*100:+8.1f} cm")
        print("  A consistent sign across every pair is a SCALE error "
              "(wheel radius / ticks per rev),")
        print("  not a mapping error. Mixed signs are matcher noise.")

    print()
    if any_truth:
        v = "PASS" if worst_err <= args.error_max else "FAIL"
        print(f"ABSOLUTE   worst {worst_err*100:.1f} cm  -> {v} against "
              f"{args.error_max*100:.0f} cm")
    else:
        print("ABSOLUTE   not measured -- no visit carried --truth")
    if worst_rep > 0:
        print(f"REPEATABLE worst spread {worst_rep*100:.1f} cm across "
              "repeat visits to one mark")
    print("\nQuote BOTH in the report. Absolute error contains your tape and "
          "your parking;\nrepeatability contains neither, and is what SLAM "
          "itself is worth.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mark", nargs="?", help="literally the word `mark`, then --")
    ap.add_argument("name", nargs="?", help="name of the reference point, e.g. HOME")
    ap.add_argument("--truth", nargs=2, type=float, metavar=("X", "Y"),
                    help="tape-measured truth in the MAP frame (metres). "
                         "x along the robot's start heading, y to its LEFT")
    ap.add_argument("--settle", type=float, default=5.0,
                    help="seconds to average the pose over (default 5)")
    ap.add_argument("--still-tol", type=float, default=0.03,
                    help="reject the sample if the pose moves more than this "
                         "during the window, metres (default 0.03)")
    ap.add_argument("--error-max", type=float, default=0.10,
                    help="objective 1's criterion, metres (default 0.10)")
    ap.add_argument("--map-frame", default="map")
    ap.add_argument("--base-frame", default="base_footprint")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--session", default="~/maps/slam_accuracy.jsonl")
    ap.add_argument("--summary", action="store_true",
                    help="read the session file back and print the table")
    args = ap.parse_args()

    if args.summary:
        return do_summary(args)

    # `mark NAME` -- the first positional is the verb, the second the name.
    if args.mark != "mark" or not args.name:
        ap.error("usage: slam_accuracy_check.py mark NAME [--truth X Y]  "
                 "|  slam_accuracy_check.py --summary")
    args.mark = args.name
    return do_mark(args)


if __name__ == "__main__":
    sys.exit(main())
