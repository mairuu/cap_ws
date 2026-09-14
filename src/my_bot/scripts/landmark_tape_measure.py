#!/usr/bin/env python3
"""Score /semantic_landmarks for one class against a tape-measured truth.

WHAT THIS IS FOR. The design note's validation protocol (section 08) and the
Day 7 tape-measure gate: put one object of a known class at a position you
have measured with a tape, run the stack, and read off

    absolute error   distance from the published landmark to the truth.
                     Tests the GEOMETRY CHAIN: intrinsics, the mirror fix,
                     the range origin, TF at the right stamp, camera side.
    spread           how far the published position wanders over the run
                     (and, on Day 7, across passes from four directions).
                     Tests the FUSION: EMA, association, motion gate.
    duplicates       how many landmarks of that class exist within
                     --dup-radius of the truth. One real object must give
                     exactly one. Two means association split it (P5 broken,
                     or a TF-at-latest error while turning, P2).

They fail for different reasons, so all three are reported. Targets from
the design note: error < 0.25 m, spread < 0.15 m, duplicates 1.0.

THE TRUTH. --truth X Y is in the MAP frame. For the Day 6 stationary bench
check the robot has not moved since `make slam` started, so map == odom ==
base_link to within the matcher's noise and you can measure X (forward) and
Y (LEFT, REP-103) from the robot's drive-axle centre with a tape. Put the
object 20-25 deg OFF the centreline: an object dead ahead cannot reveal a
mirrored window or a wrong camera side. Try both sides. For Day 7 measure
against two walls and read the robot's start pose off the same walls.

LISTEN ONLY. Needs make real, make slam, make yolo, make semantic.

    ros2 run my_bot landmark_tape_measure.py chair --truth 1.80 0.75
    ros2 run my_bot landmark_tape_measure.py chair --truth 1.80 0.75 --seconds 60
"""

import argparse
import json
import math
import statistics
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class Listener(Node):

    def __init__(self, topic):
        super().__init__("landmark_tape_measure")
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.lock = threading.Lock()
        self.msgs = 0
        self.latest = []          # landmarks in the last message
        self.history = {}         # id -> list of (t, x, y)
        self.create_subscription(String, topic, self._cb, qos)

    def _cb(self, msg):
        try:
            lms = json.loads(msg.data)["landmarks"]
        except (ValueError, KeyError, TypeError) as e:
            self.get_logger().warn(f"bad payload: {e}", throttle_duration_sec=5.0)
            return
        t = time.monotonic()
        with self.lock:
            self.msgs += 1
            self.latest = lms
            for lm in lms:
                self.history.setdefault(lm["id"], []).append((t, lm["x"], lm["y"], lm))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("class_label", help="COCO class name as YOLO publishes it, e.g. chair")
    ap.add_argument("--truth", nargs=2, type=float, metavar=("X", "Y"), required=True,
                    help="tape-measured position in the map frame (Y positive LEFT)")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--topic", default="/semantic_landmarks")
    ap.add_argument("--dup-radius", type=float, default=1.0,
                    help="landmarks of the class within this of the truth count as the same object")
    ap.add_argument("--error-max", type=float, default=0.25)
    ap.add_argument("--spread-max", type=float, default=0.15)
    args = ap.parse_args()
    tx, ty = args.truth

    rclpy.init()
    node = Listener(args.topic)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print(f"listening on {args.topic} for {args.seconds:.0f} s; truth {args.class_label!r} "
          f"at ({tx:.2f}, {ty:.2f})")
    t0 = time.monotonic()
    last = t0
    try:
        while time.monotonic() - t0 < args.seconds and rclpy.ok():
            time.sleep(0.5)
            if time.monotonic() - last >= 5.0:
                last = time.monotonic()
                with node.lock:
                    mine = [lm for lm in node.latest if lm["class_label"] == args.class_label]
                    n_all = len(node.latest)
                if mine:
                    s = "  ".join(f"({lm['x']:+.2f},{lm['y']:+.2f}) x{lm['seen_count']} "
                                 f"err {math.hypot(lm['x']-tx, lm['y']-ty):.2f}m" for lm in mine)
                else:
                    s = "none of that class yet"
                print(f"  {time.monotonic()-t0:4.0f} s  {n_all} landmark(s) published; "
                      f"{args.class_label}: {s}", flush=True)
    except KeyboardInterrupt:
        pass

    with node.lock:
        msgs = node.msgs
        latest = list(node.latest)
        history = {k: list(v) for k, v in node.history.items()}

    print()
    print(f"=== {args.class_label!r} vs truth ({tx:.2f}, {ty:.2f}) over {time.monotonic()-t0:.0f} s ===")
    if msgs == 0:
        print("FAIL  no /semantic_landmarks messages. Is make semantic running?")
        rclpy.shutdown()
        sys.exit(1)

    mine = [lm for lm in latest if lm["class_label"] == args.class_label]
    near = [lm for lm in mine if math.hypot(lm["x"] - tx, lm["y"] - ty) <= args.dup_radius]
    print(f"messages  {msgs}; landmarks now {len(latest)}, of class {len(mine)}, "
          f"within {args.dup_radius:.1f} m of truth {len(near)}")

    if not mine:
        print("FAIL  no landmark of that class was published. Read the semantic node's "
              "'fused n/m' line: gate closed (no odom)? rejected (min_returns / max_spread)? "
              "tf miss? Or the detector never saw the object (check /detections/image).")
        rclpy.shutdown()
        sys.exit(1)

    best = min(mine, key=lambda lm: math.hypot(lm["x"] - tx, lm["y"] - ty))
    err = math.hypot(best["x"] - tx, best["y"] - ty)
    dx, dy = best["x"] - tx, best["y"] - ty
    bearing_truth = math.degrees(math.atan2(ty, tx))
    bearing_pub = math.degrees(math.atan2(best["y"], best["x"]))
    range_truth = math.hypot(tx, ty)
    range_pub = math.hypot(best["x"], best["y"])
    print(f"nearest   ({best['x']:+.3f}, {best['y']:+.3f})  seen x{best['seen_count']}  "
          f"conf {best['confidence']:.2f}  id {best['id'][:8]}")
    print(f"error     {err:.3f} m   (dx {dx:+.3f}, dy {dy:+.3f})")
    print(f"          range  published {range_pub:.3f} vs truth {range_truth:.3f} "
          f"({(range_pub-range_truth)*100:+.1f} cm)")
    print(f"          bearing published {bearing_pub:+.2f} vs truth {bearing_truth:+.2f} deg "
          f"({bearing_pub-bearing_truth:+.2f} deg)  -- a sign flip here is the camera side "
          f"or the mirror; a constant offset is cx or camera yaw")

    xs_ys = [(x, y) for (_, x, y, _) in history.get(best["id"], [])]
    if len(xs_ys) >= 2:
        cxm = statistics.mean(x for x, _ in xs_ys)
        cym = statistics.mean(y for _, y in xs_ys)
        spread = max(math.hypot(x - cxm, y - cym) for x, y in xs_ys) * 2
        drift = math.hypot(xs_ys[-1][0] - xs_ys[0][0], xs_ys[-1][1] - xs_ys[0][1])
        print(f"spread    {spread:.3f} m peak-to-peak over {len(xs_ys)} publishes; "
              f"first->last drift {drift:.3f} m")
    else:
        spread = 0.0
        print("spread    n/a (one publish)")

    others = [lm for lm in near if lm["id"] != best["id"]]
    for lm in others:
        print(f"duplicate ({lm['x']:+.2f},{lm['y']:+.2f}) x{lm['seen_count']} id {lm['id'][:8]}")

    print()
    print("=== verdicts ===")
    ok_err = err <= args.error_max
    ok_spread = spread <= args.spread_max
    ok_dup = len(near) == 1
    print(f"[{'PASS' if ok_err else 'FAIL'}] absolute error {err:.3f} m <= {args.error_max}")
    print(f"[{'PASS' if ok_spread else 'FAIL'}] spread {spread:.3f} m <= {args.spread_max}")
    print(f"[{'PASS' if ok_dup else 'FAIL'}] exactly one landmark within {args.dup_radius:.1f} m "
          f"of truth (found {len(near)})")
    rclpy.shutdown()
    sys.exit(0 if (ok_err and ok_spread and ok_dup) else 1)


if __name__ == "__main__":
    main()
