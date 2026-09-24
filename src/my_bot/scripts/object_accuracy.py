#!/usr/bin/env python3
"""Score a mapped object against a tape measured FROM THE ROBOT. Objective 4.

WHAT THIS IS FOR. The report's objective 4 sets a number -- the position of an
object on the map must not differ from its real position by more than 50 cm.
`landmark_tape_measure.py` scores that against map-frame truth, which means the
map origin has to have been surveyed first: HOME taped, axes laid out, the whole
objective-1 setup. That couples two objectives that have nothing to do with each
other, and it means objective 4 cannot be re-run on its own.

THIS SCRIPT NEEDS NO SURVEY. Park the robot wherever it can see the object, pull
a tape twice, and run one line. The measurement is entirely local:

    the system's answer      landmark (map) -> robot frame, via map -> base_footprint
    your answer              two tape pulls, robot -> object
    the error                the distance between those two points

Both are expressed in the robot's own frame at the moment of the sample, so the
map origin, its orientation, and any drift in it all cancel. What is left is
what objective 4 actually asks about: does the fusion put the object where the
object is.

HOW TO PULL THE TAPE. `base_link` sits at the centre of the wheel axle (see
description/robot_core.xacro) and `base_footprint` is that point dropped to the
floor, so the tape starts at THE MIDPOINT OF THE DRIVE AXLE, not the nose.

    --fwd    along the robot's centreline, from the axle midpoint to the point
             on the centreline nearest the object. Positive is forwards.
    --left   from that point out to the object, perpendicular to the centreline.
             Positive is to the robot's LEFT, negative to its right.

Sight the centreline with two marks on the robot, the way the axis is laid out
for objective 1 -- eyeballing the chassis is worth several degrees, and at 1.6 m
three degrees is 8 cm, which is most of a sensible error budget.

    ros2 run my_bot object_accuracy.py chair --fwd 1.50 --left -0.50 \\
        --pass-label front

FOUR VIEWPOINTS, AND WHAT THE SPREAD MEANS. Run it once per side, backing the
robot off to a similar distance each time. Whether the passes are INDEPENDENT
depends on what you do between them:

  * clear the landmarks between passes (the UI's clear button, or the bridge's
    POST /clear) and each pass is its own estimate from one viewing direction.
    The spread across passes is then a real viewpoint-dependence number.
  * leave them, and the landmark is refined by the running average as each new
    view arrives. The spread across passes then measures CONVERGENCE, not
    independence, and is usually smaller. That is fine to report -- but say
    which one you did, because they answer different questions.

`--summary` prints both the per-pass errors and the spread, and says which
interpretation applies based on whether the landmark id changed between passes.

NOT MEASURED HERE. Whether the object is in the right place in the WORLD -- that
needs objective 1's survey and is a different claim. This script scores the
object relative to the robot, which is the part the fusion pipeline is
responsible for.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

DEFAULT_SESSION = "~/maps/object_accuracy.jsonl"
LANDMARK_TOPIC = "/semantic_landmarks"


# ---------------------------------------------------------------------------
# geometry -- pure, so it can be exercised without ROS
# ---------------------------------------------------------------------------

def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def to_robot_frame(lx, ly, rx, ry, ryaw):
    """Map-frame point -> (forward, left) in the robot's frame."""
    dx, dy = lx - rx, ly - ry
    c, s = math.cos(ryaw), math.sin(ryaw)
    return dx * c + dy * s, -dx * s + dy * c


def pick_landmark(landmarks, cls, pose, fwd, left, dup_radius):
    """The landmark of `cls` closest to where the tape says the object is.

    Returns (chosen, n_within_dup_radius, candidates_of_class). `chosen` carries
    the robot-frame position and the error, so the caller does no geometry.
    """
    rx, ry, ryaw = pose
    cands = []
    for lm in landmarks:
        if lm.get("class_label") != cls:
            continue
        f, l = to_robot_frame(lm["x"], lm["y"], rx, ry, ryaw)
        cands.append(dict(lm, fwd=f, left=l,
                          error=math.hypot(f - fwd, l - left)))
    if not cands:
        return None, 0, []
    cands.sort(key=lambda c: c["error"])
    near = sum(1 for c in cands if c["error"] <= dup_radius)
    return cands[0], near, cands


# ---------------------------------------------------------------------------
# one pass
# ---------------------------------------------------------------------------

def do_measure(args):
    import rclpy
    import tf2_ros
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)

    rclpy.init()
    node = Node("object_accuracy")
    latest = {"json": None}
    node.create_subscription(String, args.topic,
                             lambda m: latest.__setitem__("json", m.data), qos)
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)

    def pose_now():
        try:
            tf = buf.lookup_transform(args.map_frame, args.base_frame,
                                      rclpy.time.Time())
        except Exception:
            return None
        t = tf.transform.translation
        return t.x, t.y, yaw_of(tf.transform.rotation)

    # wait for both inputs
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if latest["json"] is not None and pose_now() is not None:
            break
    if latest["json"] is None:
        print(f"!! nothing on {args.topic} after {args.timeout:.0f} s -- is "
              f"`make semantic` running?", file=sys.stderr)
        return 1
    if pose_now() is None:
        print(f"!! no {args.map_frame} -> {args.base_frame} transform after "
              f"{args.timeout:.0f} s -- is `make slam` running?", file=sys.stderr)
        return 1

    print(f"sampling {args.seconds:.0f} s -- KEEP THE ROBOT STILL")
    samples, ids, start = [], set(), time.monotonic()
    n_near_last, cands_last = 0, []
    while time.monotonic() - start < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.05)
        pose = pose_now()
        if pose is None or latest["json"] is None:
            continue
        try:
            lms = json.loads(latest["json"]).get("landmarks", [])
        except (ValueError, AttributeError):
            continue
        chosen, n_near, cands = pick_landmark(lms, args.cls, pose,
                                              args.fwd, args.left,
                                              args.dup_radius)
        if chosen is None:
            continue
        samples.append(chosen)
        ids.add(chosen["id"])
        n_near_last, cands_last = n_near, cands

    rclpy.shutdown()

    if not samples:
        print(f"\n!! no landmark of class {args.cls!r} was published during the "
              f"window.", file=sys.stderr)
        print("   Check the class name, drive closer, or check that the object "
              "is on the lidar's", file=sys.stderr)
        print("   scan plane -- an object above it is rejected by design.",
              file=sys.stderr)
        return 1

    errs = [s["error"] for s in samples]
    fwds = [s["fwd"] for s in samples]
    lefts = [s["left"] for s in samples]
    mx, my = statistics.fmean([s["x"] for s in samples]), \
        statistics.fmean([s["y"] for s in samples])
    err = statistics.fmean(errs)
    spread = max(math.hypot(f - statistics.fmean(fwds), l - statistics.fmean(lefts))
                 for f, l in zip(fwds, lefts))

    print(f"\n=== {args.cls} · pass {args.pass_label!r} ===")
    print(f"  tape says          fwd {args.fwd:+.3f}  left {args.left:+.3f} m")
    print(f"  system says        fwd {statistics.fmean(fwds):+.3f}  "
          f"left {statistics.fmean(lefts):+.3f} m")
    print(f"  ERROR              {err*100:.1f} cm      "
          f"(criterion {args.error_max*100:.0f} cm)")
    print(f"  spread in window   {spread*100:.1f} cm over {len(samples)} samples")
    print(f"  landmark map pos   ({mx:+.3f}, {my:+.3f})   id {sorted(ids)[0][:8]}")
    if len(ids) > 1:
        print(f"  ! the chosen landmark changed id {len(ids)} times during the "
              f"window -- the store may be creating duplicates")
    if n_near_last > 1:
        print(f"  ! {n_near_last} landmarks of this class within "
              f"{args.dup_radius:.2f} m of the taped point (duplicates)")
    if len(cands_last) > 1:
        others = ", ".join(f"{c['error']*100:.0f}cm" for c in cands_last[1:4])
        print(f"    other {args.cls} landmarks, by distance from the tape: {others}")

    verdict = "PASS" if err <= args.error_max else "FAIL"
    print(f"\n  {verdict}")

    row = {"kind": "object_accuracy", "class": args.cls,
           "pass_label": args.pass_label,
           "tape_fwd": args.fwd, "tape_left": args.left,
           "sys_fwd": round(statistics.fmean(fwds), 4),
           "sys_left": round(statistics.fmean(lefts), 4),
           "error_m": round(err, 4), "worst_m": round(max(errs), 4),
           "spread_m": round(spread, 4), "samples": len(samples),
           "map_x": round(mx, 4), "map_y": round(my, 4),
           "landmark_ids": sorted(ids),
           "duplicates_within_radius": n_near_last,
           "error_max": args.error_max, "verdict": verdict,
           "stamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path = os.path.expanduser(args.session)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"  appended to {path}")
    return 0 if verdict == "PASS" else 1


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def summarise(rows, error_max):
    """Pure. rows = the session rows for one class. Returns printable lines."""
    out = [f"=== objective 4 - {rows[0]['class']} - {len(rows)} pass(es) ===", ""]
    out.append(f"{'pass':<10}{'tape (fwd,left)':>22}{'system (fwd,left)':>22}"
               f"{'error':>10}{'spread':>10}")
    out.append("-" * 74)
    for r in rows:
        tape = f"({r['tape_fwd']:+.2f}, {r['tape_left']:+.2f})"
        sysp = f"({r['sys_fwd']:+.2f}, {r['sys_left']:+.2f})"
        out.append(f"{r['pass_label']:<10}{tape:>22}{sysp:>22}"
                   f"{r['error_m'] * 100:>8.1f}cm{r['spread_m'] * 100:>8.1f}cm")

    errs = [r["error_m"] for r in rows]
    out.append("-" * 74)
    out.append(f"mean error   {statistics.fmean(errs) * 100:6.1f} cm")
    out.append(f"worst error  {max(errs) * 100:6.1f} cm   "
               f"(criterion {error_max * 100:.0f} cm)")

    if len(rows) > 1:
        xs = [r["map_x"] for r in rows]
        ys = [r["map_y"] for r in rows]
        cx, cy = statistics.fmean(xs), statistics.fmean(ys)
        across = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys))
        out.append(f"across-pass spread of the mapped position   {across * 100:.1f} cm")
        ids = {tuple(r["landmark_ids"]) for r in rows}
        if len(ids) == 1:
            out.append("  Same landmark id in every pass: the passes REFINED one estimate,")
            out.append("  so this spread is CONVERGENCE, not viewpoint independence.")
            out.append("  Clear the landmarks between passes to get the independent number.")
        else:
            out.append("  The landmark id changed between passes: each pass is an")
            out.append("  INDEPENDENT estimate, so this spread is the viewpoint dependence.")

    dups = max(r.get("duplicates_within_radius", 1) for r in rows)
    if dups > 1:
        out.append(f"  ! up to {dups} landmarks of this class sat within the "
                   f"duplicate radius")

    out.append("")
    out.append("PASS" if max(errs) <= error_max else "FAIL")
    return out


def do_summary(args):
    path = os.path.expanduser(args.session)
    if not os.path.exists(path):
        print(f"no session at {path} -- record a pass first.", file=sys.stderr)
        return 1
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") == "object_accuracy" and r["class"] == args.cls:
                rows.append(r)
    if not rows:
        print(f"no passes for class {args.cls!r} in {path}", file=sys.stderr)
        return 1
    for line in summarise(rows, args.error_max):
        print(line)
    return 0


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Objective 4, measured from the robot -- no map survey needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cls", help="class label, e.g. chair")
    ap.add_argument("--fwd", type=float,
                    help="tape: metres forward from the drive-axle midpoint")
    ap.add_argument("--left", type=float,
                    help="tape: metres to the robot's left (negative = right)")
    ap.add_argument("--pass-label", default="front",
                    help="front / right / back / left")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--error-max", type=float, default=0.50)
    ap.add_argument("--dup-radius", type=float, default=0.50)
    ap.add_argument("--topic", default=LANDMARK_TOPIC)
    ap.add_argument("--map-frame", default="map")
    ap.add_argument("--base-frame", default="base_footprint")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--session", default=DEFAULT_SESSION)
    ap.add_argument("--summary", action="store_true",
                    help="read the session back; no robot needed")
    args = ap.parse_args()

    if args.summary:
        raise SystemExit(do_summary(args))
    if args.fwd is None or args.left is None:
        ap.error("--fwd and --left are required (or pass --summary)")
    raise SystemExit(do_measure(args))


if __name__ == "__main__":
    main()
