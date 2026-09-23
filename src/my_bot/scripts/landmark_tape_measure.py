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

THE DAY 7 FOUR-PASS PROTOCOL. The design note's section 08 asks for spread
ACROSS four passes from four directions, which is not the same number as the
spread within a single run. Within-run spread is fusion jitter while the robot
sits still; across-pass spread is how far the four estimates disagree with each
other, and only that one exposes a bias that depends on viewing angle. Run one
pass per direction, then summarise:

    ros2 run my_bot landmark_tape_measure.py chair --truth 1.80 0.75 --pass-label front
    ros2 run my_bot landmark_tape_measure.py chair --truth 1.80 0.75 --pass-label right
    ros2 run my_bot landmark_tape_measure.py chair --truth 1.80 0.75 --pass-label back
    ros2 run my_bot landmark_tape_measure.py chair --truth 1.80 0.75 --pass-label left
    ros2 run my_bot landmark_tape_measure.py chair --truth 1.80 0.75 --summary

Each pass appends a line to --session (default ~/maps/tape_session.jsonl);
--summary reads them back and prints the table Day 7 wants. Delete the session
file before starting a fresh set, or move the chair and the summary will warn
that it holds more than one truth position.
"""

import argparse
import json
import math
import os
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


def record_pass(args, row):
    """Append one pass to the session file. One JSON object per line."""
    path = os.path.expanduser(args.session)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\nrecorded pass {row['pass_label']!r} -> {path}")
    print(f"when all four are in:  ros2 run my_bot landmark_tape_measure.py "
          f"{args.class_label} --truth {row['truth_x']:.2f} {row['truth_y']:.2f} --summary")


def summarise(args):
    """The Day 7 table: per-pass rows plus spread ACROSS passes.

    This is the number the design note section 08 actually asks for, and it is
    NOT the same as the within-run spread each pass prints. Within-run spread is
    fusion jitter while the robot sits in one place; across-pass spread is how
    far the four estimates disagree with each other, which is what exposes a
    viewing-angle-dependent bias. Reporting one in place of the other would hide
    exactly the error the four-direction protocol exists to find.
    """
    path = os.path.expanduser(args.session)
    try:
        rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    except FileNotFoundError:
        print(f"no session file at {path}. Run each pass with --pass-label first.")
        return 1
    rows = [r for r in rows if r.get("truth_x") is not None]
    if not rows:
        print(f"{path} has no passes in it.")
        return 1

    tx, ty = rows[-1]["truth_x"], rows[-1]["truth_y"]
    mixed = {(r["truth_x"], r["truth_y"]) for r in rows}
    if len(mixed) > 1:
        print(f"WARNING  {len(mixed)} different truth positions in this session file. "
              f"Reporting against the most recent ({tx:.2f}, {ty:.2f}); delete the file "
              f"and re-run the passes if the chair moved.\n")

    print(f"=== four-pass summary, {len(rows)} pass(es), truth ({tx:.2f}, {ty:.2f}) ===\n")
    print(f"| {'Pass':<10} | {'Published x':>11} | {'Published y':>11} | {'Error':>7} | {'Dups':>4} |")
    print(f"|{'-'*12}|{'-'*13}|{'-'*13}|{'-'*9}|{'-'*6}|")
    for r in rows:
        print(f"| {r['pass_label']:<10} | {r['x']:>11.3f} | {r['y']:>11.3f} | "
              f"{r['error']:>6.3f}m | {r['duplicates']:>4} |")

    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    cx, cy = statistics.mean(xs), statistics.mean(ys)
    across = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)) * 2
    mean_err = statistics.mean(r["error"] for r in rows)
    worst_err = max(r["error"] for r in rows)
    max_dups = max(r["duplicates"] for r in rows)

    print(f"\nmean error        {mean_err:.3f} m     (worst pass {worst_err:.3f} m)")
    print(f"spread ACROSS {len(rows)} passes  {across:.3f} m peak-to-peak about "
          f"({cx:+.3f}, {cy:+.3f})")
    print(f"duplicates        {max_dups} (worst pass)")

    if len(rows) < 4:
        print(f"\nNOTE only {len(rows)} of 4 passes recorded -- the protocol wants front, "
              f"right, back and left.")

    print("\n=== verdicts ===")
    ok_err = worst_err <= args.error_max
    ok_spread = across <= args.spread_max
    ok_dup = max_dups == 1
    print(f"[{'PASS' if ok_err else 'FAIL'}] worst-pass error {worst_err:.3f} m <= {args.error_max}")
    print(f"[{'PASS' if ok_spread else 'FAIL'}] across-pass spread {across:.3f} m <= {args.spread_max}")
    print(f"[{'PASS' if ok_dup else 'FAIL'}] duplicates per true object == 1 (worst {max_dups})")
    return 0 if (ok_err and ok_spread and ok_dup) else 1


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
    # --- Day 7 four-pass protocol (design note section 08) ---
    ap.add_argument("--pass-label", metavar="NAME",
                    help="name this run as one PASS (e.g. front/right/back/left) and "
                         "append its result to --session")
    ap.add_argument("--session", metavar="FILE",
                    default=os.path.expanduser("~/maps/tape_session.jsonl"),
                    help="JSONL file accumulating one line per pass "
                         "(default: ~/maps/tape_session.jsonl)")
    ap.add_argument("--summary", action="store_true",
                    help="do not listen; read --session and report the across-pass "
                         "numbers the Day 7 table wants, then exit")
    args = ap.parse_args()

    if args.summary:
        sys.exit(summarise(args))
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

    # WITHIN-RUN spread, over EVERY id that sat near the truth -- not just
    # best["id"]. Keying on the nearest id alone silently drops the earlier
    # history when association splits the object mid-run, which is exactly what
    # a four-direction pass stresses, and understates the number being reported.
    xs_ys = []
    for lm_id, hist in history.items():
        if not hist:
            continue
        _, hx, hy, _ = hist[-1]
        if math.hypot(hx - tx, hy - ty) <= args.dup_radius:
            xs_ys.extend((x, y) for (_, x, y, _) in hist)
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

    if args.pass_label:
        record_pass(args, dict(pass_label=args.pass_label, x=best["x"], y=best["y"],
                               truth_x=tx, truth_y=ty, error=err,
                               within_run_spread=spread, duplicates=len(near),
                               seen_count=best["seen_count"], id=best["id"],
                               # every publish near the truth, for the report's
                               # scatter figure (plot_objectives.py object)
                               samples=[[round(x, 4), round(y, 4)]
                                        for x, y in xs_ys[-400:]]))
    rclpy.shutdown()
    sys.exit(0 if (ok_err and ok_spread and ok_dup) else 1)


if __name__ == "__main__":
    main()
