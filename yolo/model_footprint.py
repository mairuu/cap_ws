#!/usr/bin/env python3
"""What a detector model costs on this board: RAM, GPU, CPU, time per frame.

    ~/yolo/venv/bin/python yolo/model_footprint.py MODEL [MODEL ...] \
        [--frames ~/eval/insitu2_stationary/frames] [--n 150] [--hz 15]

Each model runs in its own child process, so one model's allocations cannot
colour the next one's numbers. Per model, two passes over real frames with
`model.track()` -- the call the node makes:

  flat out   frames back to back: track() wall time -> the ceiling in FPS
  paced      at --hz (the camera's 15 Hz): GPU load %, the process's CPU %
             and whether it keeps up -- what the robot actually pays

RAM is the board's MemAvailable drop from before the model loads to the
lowest point during the run. On the Orin GPU memory IS system memory, so this
counts CUDA/TensorRT allocations too -- per-process RSS would not.

Run with nothing else on the GPU; a TensorRT build or a live `make yolo`
beside it skews every number. Results are JSON lines on stdout's last line
per model, for records/calibration.md.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

GPU_LOAD = "/sys/devices/platform/bus@0/17000000.gpu/load"


def mem_available_mb():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def proc_cpu_seconds():
    t = os.times()
    return t.user + t.system


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = False
        self.gpu, self.mem = [], []

    def run(self):
        while not self.stop:
            try:
                with open(GPU_LOAD) as fh:
                    self.gpu.append(int(fh.read().strip()) / 10.0)
            except OSError:
                pass
            self.mem.append(mem_available_mb())
            time.sleep(0.02)


def child(model_path, frames_dir, n, hz):
    os.environ["YOLO_OFFLINE"] = "1"
    # Same as yolo.launch.py sets for the node. Without it OpenBLAS starts a
    # thread per core that busy-waits between frames: ~500 % CPU here instead
    # of ~65 %, which is a number about this script, not about the robot.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import cv2
    import numpy as np
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    imgs = [cv2.imread(os.path.join(frames_dir, f)) for f in files]
    imgs = (imgs * (n // len(imgs) + 1))[:n]

    base = mem_available_mb()
    s = Sampler()
    s.start()
    from ultralytics import YOLO
    t0 = time.perf_counter()
    m = YOLO(model_path, task="detect")
    for im in imgs[:10]:
        m.track(im, persist=True, conf=0.5, verbose=False)
    load_s = time.perf_counter() - t0

    # flat out
    wall, infer = [], []
    for im in imgs:
        a = time.perf_counter()
        r = m.track(im, persist=True, conf=0.5, verbose=False)[0]
        wall.append((time.perf_counter() - a) * 1000)
        infer.append(r.speed["inference"])

    # paced at the camera rate
    period = 1.0 / hz
    g0 = len(s.gpu)
    c0, w0 = proc_cpu_seconds(), time.perf_counter()
    late = 0
    nxt = time.perf_counter()
    for im in imgs:
        m.track(im, persist=True, conf=0.5, verbose=False)
        nxt += period
        d = nxt - time.perf_counter()
        if d > 0:
            time.sleep(d)
        else:
            late += 1
    paced_s = time.perf_counter() - w0
    cpu_pct = (proc_cpu_seconds() - c0) / paced_s * 100
    gpu_paced = s.gpu[g0:]
    s.stop = True
    s.join()

    wall = np.array(wall)
    print(json.dumps({
        "model": os.path.basename(model_path),
        "imgsz": list(getattr(m.predictor, "imgsz", [])),
        "load_and_warmup_s": round(load_s, 1),
        "ram_used_mb": round(base - min(s.mem)),
        "track_ms_mean": round(float(wall.mean()), 1),
        "track_ms_p90": round(float(np.percentile(wall, 90)), 1),
        "inference_ms_mean": round(float(np.mean(infer)), 1),
        "max_fps": round(1000.0 / float(wall.mean()), 1),
        "paced_hz": hz,
        "paced_kept_up_pct": round(100.0 * (len(imgs) - late) / len(imgs), 1),
        "paced_gpu_load_mean": round(float(np.mean(gpu_paced)), 1) if gpu_paced else None,
        "paced_proc_cpu_pct": round(cpu_pct, 1),
        "frames": len(imgs),
    }))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="+")
    ap.add_argument("--frames", default=os.path.expanduser("~/eval/insitu2_stationary/frames"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        return child(os.path.expanduser(a.models[0]), os.path.expanduser(a.frames), a.n, a.hz)
    print("MemAvailable before: %.0f MB" % mem_available_mb(), file=sys.stderr)
    for mp in a.models:
        out = subprocess.run([sys.executable, __file__, "--child", mp, "--frames", a.frames,
                              "--n", str(a.n), "--hz", str(a.hz)],
                             capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("{")]
        print(line[-1] if line else json.dumps({"model": mp, "error": out.stderr[-400:]}))
        sys.stdout.flush()
        time.sleep(3)   # let the driver hand the memory back before the next one


if __name__ == "__main__":
    main()
