#!/usr/bin/env python3
"""Sample CPU / GPU / RAM / thermals through a run. Objective 4.

WHAT THIS IS FOR. The report's objective 4 asks whether the Orin can carry SLAM
and object detection AT THE SAME TIME, and sets the criterion as average load
on the main processor not exceeding 80% for the whole working period. That is a
time series, not a glance at `htop` -- so this runs `tegrastats` for a fixed
window, parses every line, and prints the numbers with a verdict.

RUN IT DURING A REAL RUN, not on an idle board. The whole claim is about
concurrency, so the stack that must be up is `make real` + `make slam` +
`make yolo` + `make semantic`, and the robot must be DRIVING for part of the
window -- the motors, the controller loop and the scan matcher are all load.
Five minutes is a reasonable window; the demo lap is better still.

    ros2 run my_bot resource_report.py --seconds 300 --label "slam+yolo, driving"

It needs no ROS graph of its own (it does not even import rclpy), so it can be
started from any terminal at any time without disturbing anything.

WHICH NUMBER IS "THE CPU". This board has 6 cores and the report's criterion is
an average, so the headline is the mean across all 6, averaged over the window.
That is the number the objective asks for and the one to put in the table. But
the BUSIEST CORE is printed next to it and matters more for engineering: a ROS
executor is single-threaded, so one core pinned at 100% while the mean sits at
35% means the system is saturated even though it passes. If that is what the
run shows, report the mean and say the sentence about the busy core -- it reads
as understanding rather than as a failure.

WHAT ELSE IS PRINTED, and why each is in the report.
  GR3D    the GPU. Detection is the only thing using it; a low number with a
          healthy frame rate means the ONNX Runtime session is doing its job.
  RAM     peak, against the 7.6 GB this board has. The margin is the answer to
          "would a bigger model fit".
  tj      junction temperature. The Orin throttles near 97 C. A flat tj under
          that for the whole window is the evidence that the frame rate quoted
          elsewhere is sustainable and not a first-minute figure.
  power   VDD_IN, which is board input power. Useful for a battery-life
          sentence and for nothing else.

Each run appends to --session (default ~/maps/resource_session.jsonl) so that
several windows -- idle, mapping only, mapping plus detection -- can be quoted
side by side. That comparison is worth more than any single number: it shows
what the detector actually costs.
"""

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time

CPU_RE = re.compile(r"CPU \[([^\]]+)\]")
CORE_RE = re.compile(r"(\d+)%@(\d+)")
GR3D_RE = re.compile(r"GR3D_FREQ (\d+)%")
RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
TJ_RE = re.compile(r"tj@([\d.]+)C")
VDD_RE = re.compile(r"VDD_IN (\d+)mW")


def parse(line):
    """One tegrastats line -> dict, or None if it is not one."""
    m = CPU_RE.search(line)
    if not m:
        return None
    cores = [int(c) for c, _ in CORE_RE.findall(m.group(1))]
    if not cores:
        return None
    out = {"cores": cores}
    if (g := GR3D_RE.search(line)):
        out["gpu"] = int(g.group(1))
    if (r := RAM_RE.search(line)):
        out["ram_mb"], out["ram_total_mb"] = int(r.group(1)), int(r.group(2))
    if (t := TJ_RE.search(line)):
        out["tj"] = float(t.group(1))
    if (v := VDD_RE.search(line)):
        out["mw"] = int(v.group(1))
    return out


def collect(seconds, interval_ms, quiet):
    proc = subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    samples = []
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            s = parse(line)
            if s is None:
                continue
            samples.append(s)
            if not quiet and len(samples) % 10 == 0:
                left = deadline - time.monotonic()
                mean = statistics.fmean(s["cores"])
                print(f"  {len(samples):4d} samples, {left:5.0f} s left   "
                      f"cpu {mean:5.1f}%  gpu {s.get('gpu', 0):3d}%  "
                      f"tj {s.get('tj', 0):.1f} C", flush=True)
    except KeyboardInterrupt:
        print("\n(interrupted -- reporting what was collected)")
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return samples


def report(samples, args):
    if len(samples) < 5:
        print("too few samples. Is tegrastats present and runnable?")
        return 1

    per_sample_mean = [statistics.fmean(s["cores"]) for s in samples]
    ncore = len(samples[0]["cores"])
    per_core = [[s["cores"][i] for s in samples if len(s["cores"]) > i]
                for i in range(ncore)]
    gpus = [s["gpu"] for s in samples if "gpu" in s]
    rams = [s["ram_mb"] for s in samples if "ram_mb" in s]
    tjs = [s["tj"] for s in samples if "tj" in s]
    mws = [s["mw"] for s in samples if "mw" in s]

    cpu_mean = statistics.fmean(per_sample_mean)
    cpu_max = max(per_sample_mean)
    busiest = max(range(ncore), key=lambda i: statistics.fmean(per_core[i]))

    dur = len(samples) * args.interval / 1000.0
    print(f"\n=== resources over {dur:.0f} s, {len(samples)} samples "
          f"@ {args.interval} ms ===")
    if args.label:
        print(f"    {args.label}")
    print()
    print(f"CPU mean (all {ncore} cores)   {cpu_mean:5.1f} %   "
          f"peak of the {ncore}-core mean {cpu_max:5.1f} %")
    for i in range(ncore):
        print(f"   core {i}                  {statistics.fmean(per_core[i]):5.1f} %"
              f"   max {max(per_core[i]):3d} %"
              + ("   <- busiest" if i == busiest else ""))
    if gpus:
        print(f"GPU (GR3D)                {statistics.fmean(gpus):5.1f} %   "
              f"max {max(gpus):3d} %")
    if rams:
        total = samples[0].get("ram_total_mb", 0)
        print(f"RAM                       {statistics.fmean(rams):5.0f} MB  "
              f"peak {max(rams)} / {total} MB")
    if tjs:
        print(f"tj                        {statistics.fmean(tjs):5.1f} C   "
              f"max {max(tjs):.1f} C"
              + ("   <- THROTTLING RANGE" if max(tjs) >= 90 else ""))
    if mws:
        print(f"board power (VDD_IN)      {statistics.fmean(mws)/1000:5.2f} W  "
              f"peak {max(mws)/1000:.2f} W")

    print()
    verdict = "PASS" if cpu_mean <= args.cpu_max else "FAIL"
    print(f"OBJECTIVE 4   mean CPU {cpu_mean:.1f} % against "
          f"{args.cpu_max:.0f} %  -> {verdict}")
    busy_mean = statistics.fmean(per_core[busiest])
    if busy_mean > 80 and cpu_mean <= args.cpu_max:
        print(f"  ! core {busiest} averaged {busy_mean:.0f} %. The criterion "
              "passes on the mean, but a\n    single-threaded executor is "
              "the real ceiling -- say so in the report.")
    if tjs and max(tjs) < 90:
        print(f"  thermals stable: tj peaked at {max(tjs):.1f} C, no throttle "
              "in this window.")

    row = {
        "t": time.time(), "label": args.label, "seconds": dur,
        "samples": len(samples),
        "cpu_mean": cpu_mean, "cpu_peak": cpu_max,
        "cpu_per_core_mean": [statistics.fmean(c) for c in per_core],
        "gpu_mean": statistics.fmean(gpus) if gpus else None,
        "gpu_max": max(gpus) if gpus else None,
        "ram_peak_mb": max(rams) if rams else None,
        "tj_max": max(tjs) if tjs else None,
        "watt_mean": statistics.fmean(mws) / 1000.0 if mws else None,
    }
    path = os.path.expanduser(args.session)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"\nappended to {path}")
    return 0


def do_summary(args):
    path = os.path.expanduser(args.session)
    if not os.path.exists(path):
        print(f"no session file at {path}.")
        return 1
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(f"=== {len(rows)} window(s) from {path} ===\n")
    print(f"{'label':<34}{'s':>5}{'cpu%':>7}{'gpu%':>6}{'RAM MB':>8}{'tj C':>7}")
    for r in rows:
        print(f"{(r.get('label') or '-')[:33]:<34}{r['seconds']:5.0f}"
              f"{r['cpu_mean']:7.1f}"
              f"{(r['gpu_mean'] or 0):6.1f}{(r['ram_peak_mb'] or 0):8.0f}"
              f"{(r['tj_max'] or 0):7.1f}")
    print("\nThe difference between windows is the cost of what you added.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=300.0,
                    help="length of the window (default 300)")
    ap.add_argument("--interval", type=int, default=1000,
                    help="tegrastats sampling interval, ms (default 1000)")
    ap.add_argument("--cpu-max", type=float, default=80.0,
                    help="objective 4's criterion, percent (default 80)")
    ap.add_argument("--label", default="",
                    help="what was running, e.g. 'slam+yolo, driving'")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--session", default="~/maps/resource_session.jsonl")
    ap.add_argument("--summary", action="store_true",
                    help="print every window recorded so far and stop")
    args = ap.parse_args()

    if args.summary:
        return do_summary(args)

    if not args.label:
        print("!! no --label. Record WHAT WAS RUNNING or the number is "
              "unusable later.\n")
    print(f"sampling for {args.seconds:.0f} s -- Ctrl-C stops early and still "
          "reports.")
    samples = collect(args.seconds, args.interval, args.quiet)
    return report(samples, args)


if __name__ == "__main__":
    sys.exit(main())
