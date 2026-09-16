#!/usr/bin/env python3
"""Measure /detections for the Day 5 gate: rate, latency, track persistence, thermals.

WHAT THIS IS FOR. checklists/day-5-yolo.md's gate has four lines:

    /detections is stable            -> rate AND its jitter, not one hz sample
    track IDs persist across frames  -> per-id lifetime over the window, for a
                                        STATIONARY object in view
    no thermal throttling after 5 min -> GPU clock vs its max, and the
                                        temperatures, sampled through the run
    versions recorded                -> printed here so they can be pasted

`ros2 topic hz` gives the first number and nothing else, and "the ids look
stable" is the kind of eyeballed claim this project has already been burned
by (Day 2's 90 degrees). Run this for the full five minutes with something
the detector recognises -- a chair, a person sitting still, a bottle on the
desk -- held still in frame, and read the verdicts.

HOW TO READ IT.

  rate       mean Hz over the window plus the inter-arrival sd. The camera is
             15 Hz; anything at or near 15 with sd of a few ms means the node
             keeps up and every frame is processed. Well under 15 with a small
             sd means the node is the bottleneck and dropping evenly (fine,
             the depth-1 queue is doing its job). A large sd means stalls.
  age        header.stamp is the CAPTURE time (P2), so now - stamp at receipt
             is the whole camera -> inference -> publish -> transport latency.
             The semantic node fuses at the pose for that stamp, so age is not
             an error, but a growing age means the queue is not depth 1.
  ids        every track id seen, its class, how many frames it was present in,
             and the span first..last as a fraction of the window. A
             stationary object should produce ONE id that lives ~100% of the
             window. Many short-lived ids of the same class at the same place
             is the tracker losing and re-acquiring it: raise conf, or hold
             the object more squarely in frame. Ids with span >= --persist
             (default 0.8) count as persistent for the verdict.
  thermal    sampled every second from sysfs: GPU devfreq cur/max, GPU load,
             and the gpu/cpu/tj thermal zones. The verdict is TEMPERATURE:
             tj must stay under --tj-max (default 90 C; the Orin's own
             throttle trip is ~97 C) and must not still be climbing steeply
             over the last minute. The GPU clock is printed but is NOT the
             verdict: the devfreq governor (nvhost_podgov) only raises the
             clock with GPU LOAD, and this pipeline is CPU/launch-bound at
             the nano model size -- measured 14 Sep sitting at the 306 MHz
             floor while holding 15 Hz. A low clock with a low load is
             headroom, not throttling. A clock that FALLS while load and
             temperature rise is; the per-10 s lines show all three.

LISTEN ONLY. Needs `make yolo` (or the detector some other way) running.

    ros2 run my_bot detection_report.py --seconds 300
    ros2 run my_bot detection_report.py --seconds 60 --persist 0.9
"""

import argparse
import glob
import os
import statistics
import sys
import threading
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from vision_msgs.msg import Detection2DArray


GPU_DEVFREQ = glob.glob("/sys/class/devfreq/*.gpu")
GPU_LOAD = glob.glob("/sys/devices/platform/*.gpu/load")   # 0..1000 on Orin
THERMAL_ZONES = {"gpu": None, "cpu": None, "tj": None}


def _find_thermal_zones():
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            t = open(os.path.join(z, "type")).read().strip()
        except OSError:
            continue
        for k in THERMAL_ZONES:
            if t == f"{k}-thermal":
                THERMAL_ZONES[k] = os.path.join(z, "temp")


def _read_int(path):
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return None


def _sample_thermal():
    s = {"t": time.monotonic()}
    if GPU_DEVFREQ:
        s["gpu_cur"] = _read_int(os.path.join(GPU_DEVFREQ[0], "cur_freq"))
        s["gpu_max"] = _read_int(os.path.join(GPU_DEVFREQ[0], "max_freq"))
    if GPU_LOAD:
        v = _read_int(GPU_LOAD[0])
        s["gpu_load"] = v / 10.0 if v is not None else None
    for k, p in THERMAL_ZONES.items():
        v = _read_int(p) if p else None
        s[k] = v / 1000.0 if v is not None else None
    return s


class Report(Node):

    def __init__(self, topic, reliable):
        super().__init__("detection_report")
        qos = QoSProfile(
            reliability=(ReliabilityPolicy.RELIABLE if reliable
                         else ReliabilityPolicy.BEST_EFFORT),
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=50)
        self.lock = threading.Lock()
        self.arrivals = []        # monotonic receipt times
        self.ages_ms = []         # now - header.stamp
        self.ndet = []
        self.frames_with_id = 0
        self.frames_without_id = 0   # detections present but no track id
        self.ids = {}             # id -> dict(cls, n, first, last, conf)
        self.classes = defaultdict(int)
        self.frame_idx = 0
        self.create_subscription(Detection2DArray, topic, self._cb, qos)

    def _cb(self, msg):
        t = time.monotonic()
        now = self.get_clock().now().nanoseconds
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        with self.lock:
            self.frame_idx += 1
            self.arrivals.append(t)
            self.ages_ms.append((now - stamp) / 1e6)
            self.ndet.append(len(msg.detections))
            any_id = False
            for d in msg.detections:
                cls = d.results[0].hypothesis.class_id if d.results else "?"
                conf = d.results[0].hypothesis.score if d.results else 0.0
                self.classes[cls] += 1
                if d.id:
                    any_id = True
                    r = self.ids.setdefault(d.id, {"cls": cls, "n": 0, "first": self.frame_idx,
                                                   "last": self.frame_idx, "conf": []})
                    r["n"] += 1
                    r["last"] = self.frame_idx
                    r["conf"].append(conf)
            if msg.detections:
                if any_id:
                    self.frames_with_id += 1
                else:
                    self.frames_without_id += 1


def _teardown(node, spinner):
    """Stop the executor BEFORE the interpreter starts tearing down.

    rclpy.spin() runs in a daemon thread here. Calling sys.exit() straight after
    rclpy.shutdown() races it: shutdown makes spin return, but if the process
    exits first the C++ executor thread is still live at static-destructor time
    and the run ends in `terminate called without an active exception` /
    `[ros2run]: Aborted`. Seen 16 Sep after a clean 301 s gate run -- the report
    had already printed, so it cost nothing but it looks exactly like a crash.
    Join the spinner, then destroy the node, then exit.
    """
    rclpy.shutdown()
    spinner.join(timeout=2.0)
    try:
        node.destroy_node()
    except Exception:  # already torn down; nothing useful to do here
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=300.0,
                    help="window length; the gate says five minutes (default 300)")
    ap.add_argument("--topic", default="/detections")
    ap.add_argument("--best-effort", action="store_true",
                    help="subscribe best-effort instead of reliable")
    ap.add_argument("--persist", type=float, default=0.8,
                    help="span fraction of the window above which an id counts as persistent")
    ap.add_argument("--tj-max", type=float, default=90.0,
                    help="tj-thermal ceiling in C for the thermal verdict")
    ap.add_argument("--expect-hz", type=float, default=15.0,
                    help="camera rate; the rate verdict wants >= 0.8 x this")
    args = ap.parse_args()

    _find_thermal_zones()
    rclpy.init()
    node = Report(args.topic, not args.best_effort)
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    t0 = time.monotonic()
    samples = []
    last_print = t0
    last_n = 0
    print(f"listening on {args.topic} for {args.seconds:.0f} s "
          f"(gpu devfreq: {'yes' if GPU_DEVFREQ else 'NO'}; zones: "
          f"{', '.join(k for k, v in THERMAL_ZONES.items() if v) or 'none'})")
    try:
        while time.monotonic() - t0 < args.seconds and rclpy.ok():
            time.sleep(1.0)
            s = _sample_thermal()
            with node.lock:
                s["n"] = len(node.arrivals)
            samples.append(s)
            if time.monotonic() - last_print >= 10.0:
                dn = s["n"] - last_n
                last_n = s["n"]
                last_print = time.monotonic()
                gpu = (f"gpu {s['gpu_cur']/1e6:.0f}/{s['gpu_max']/1e6:.0f} MHz"
                       if s.get("gpu_cur") else "gpu ?")
                if s.get("gpu_load") is not None:
                    gpu += f" load {s['gpu_load']:3.0f}%"
                print(f"  {time.monotonic()-t0:5.0f} s  {dn/10.0:5.1f} Hz  {gpu}  "
                      f"tj {s['tj']:.1f} C  gpu {s['gpu']:.1f} C  "
                      f"ids so far {len(node.ids)}", flush=True)
    except KeyboardInterrupt:
        pass
    window = time.monotonic() - t0

    with node.lock:
        arr = list(node.arrivals)
        ages = list(node.ages_ms)
        ndet = list(node.ndet)
        ids = dict(node.ids)
        classes = dict(node.classes)
        fw, fwo = node.frames_with_id, node.frames_without_id
        nframes = node.frame_idx

    print()
    print(f"=== /detections over {window:.0f} s ===")
    if len(arr) < 2:
        print("FAIL  fewer than 2 messages received. Is the detector running, and "
              "does its QoS match (try --best-effort)?")
        _teardown(node, spinner)
        sys.exit(1)

    dts = [(b - a) * 1000 for a, b in zip(arr, arr[1:])]
    hz = (len(arr) - 1) / (arr[-1] - arr[0])
    print(f"rate      {hz:6.2f} Hz   ({len(arr)} msgs; inter-arrival mean "
          f"{statistics.mean(dts):.1f} ms, sd {statistics.pstdev(dts):.1f} ms, "
          f"max {max(dts):.0f} ms)")
    print(f"age       p50 {statistics.median(ages):5.0f} ms  p95 "
          f"{sorted(ages)[int(0.95*(len(ages)-1))]:5.0f} ms  max {max(ages):5.0f} ms   "
          f"(capture -> here; header.stamp is capture time)")
    print(f"dets      {statistics.mean(ndet):.2f} per frame; frames with >=1 det: "
          f"{sum(1 for n in ndet if n)}/{len(ndet)}; "
          f"frames with dets but NO track id: {fwo}")
    if classes:
        top = sorted(classes.items(), key=lambda kv: -kv[1])[:8]
        print("classes   " + ", ".join(f"{c} x{n}" for c, n in top))

    print()
    print(f"track ids ({len(ids)} distinct)   frames  span      class   conf")
    persistent = []
    for tid, r in sorted(ids.items(), key=lambda kv: -kv[1]["n"])[:20]:
        span = (r["last"] - r["first"] + 1) / max(nframes, 1)
        flag = ""
        if span >= args.persist:
            persistent.append(tid)
            flag = "  <- persistent"
        print(f"  id {tid:>5}   {r['n']:6d}  {span*100:5.1f}%  {r['cls']:>8}  "
              f"{statistics.mean(r['conf']):.2f}{flag}")
    if len(ids) > 20:
        print(f"  ... and {len(ids)-20} more")

    print()
    print("=== thermal ===")
    busy = [s for s in samples if s.get("gpu_cur")]
    verdict_thermal = True
    why = ""
    tjs = [s["tj"] for s in samples if s.get("tj") is not None]
    gpus = [s["gpu"] for s in samples if s.get("gpu") is not None]
    if busy:
        loads = [s["gpu_load"] for s in busy if s.get("gpu_load") is not None]
        print(f"gpu clock  start {busy[0]['gpu_cur']/1e6:.0f}  end "
              f"{busy[-1]['gpu_cur']/1e6:.0f}  max seen "
              f"{max(s['gpu_cur'] for s in busy)/1e6:.0f}  / ceiling "
              f"{busy[-1]['gpu_max']/1e6:.0f} MHz"
              + (f";  load mean {statistics.mean(loads):.0f}%  max {max(loads):.0f}%"
                 if loads else ""))
    else:
        print("no GPU devfreq node found; clock not sampled")
    if tjs:
        print(f"tj         start {tjs[0]:.1f} C  end {tjs[-1]:.1f} C  max {max(tjs):.1f} C")
    if gpus:
        print(f"gpu temp   start {gpus[0]:.1f} C  end {gpus[-1]:.1f} C  max {max(gpus):.1f} C")
    if tjs:
        if max(tjs) >= args.tj_max:
            verdict_thermal, why = False, f"tj reached {max(tjs):.1f} C"
        elif len(tjs) >= 120 and (tjs[-1] - tjs[-60]) > 3.0:
            # Still climbing > 3 C in the final minute: not settled yet.
            verdict_thermal, why = False, (f"tj still rising {tjs[-1]-tjs[-60]:+.1f} C "
                                           f"over the last minute -- run longer")
    else:
        print("no tj thermal zone; thermal verdict skipped")

    print()
    print("=== verdicts ===")
    ok_rate = hz >= 0.8 * args.expect_hz and statistics.pstdev(dts) < 0.5 * statistics.mean(dts)
    ok_ids = len(persistent) >= 1
    print(f"[{'PASS' if ok_rate else 'FAIL'}] rate stable: {hz:.1f} Hz vs "
          f">= {0.8*args.expect_hz:.1f}, jitter sd {statistics.pstdev(dts):.0f} ms")
    print(f"[{'PASS' if ok_ids else 'FAIL'}] track ids persist: {len(persistent)} id(s) "
          f"spanning >= {args.persist*100:.0f}% of the window"
          + ("" if ok_ids else "   (was something the model knows held still in frame?)"))
    print(f"[{'PASS' if verdict_thermal else 'FAIL'}] not throttling"
          + (f": {why}" if why else (f": tj max {max(tjs):.1f} C < {args.tj_max:.0f}" if tjs else ""))
          + ("" if window >= 290 else f"   (only {window:.0f} s -- the gate wants 300)"))

    _teardown(node, spinner)
    sys.exit(0 if (ok_rate and ok_ids and verdict_thermal) else 1)


if __name__ == "__main__":
    main()
