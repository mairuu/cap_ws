#!/usr/bin/env python3
"""YOLO backend bench (23 Sep). One config per process:
    ~/yolo/venv/bin/python yolo/bench_yolo.py <model.pt|.onnx> gpu|cpu <half 0|1>
Splits raw model forward from the ultralytics pipeline (predict speed split,
track() wall), and records the evidence that the GPU did the work: torch
parameter device, ORT providers actually bound, GPU load and clock sampled.
Results: records/calibration.md, "YOLO bench re-run, 23 Sep"."""
import json, os, sys, threading, time
import numpy as np

os.environ["YOLO_OFFLINE"] = "1"
path, device, half = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
N, WARM = int(os.environ.get("N", 100)), 20
GPU = "/sys/devices/platform/bus@0/17000000.gpu"
FREQ = "/sys/class/devfreq/17000000.gpu/cur_freq"


def rd(p):
    try:
        return int(open(p).read().strip())
    except Exception:
        return -1


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.stop = False; self.load = []; self.freq = []
    def run(self):
        while not self.stop:
            self.load.append(rd(GPU + "/load") / 10.0); self.freq.append(rd(FREQ) / 1e6); time.sleep(0.02)


def stats(xs):
    a = np.array(xs)
    return {"mean": round(a.mean(), 2), "p50": round(float(np.median(a)), 2), "sd": round(a.std(), 2)}


import torch, cv2
from ultralytics import YOLO

out = {"model": os.path.basename(path), "dir": os.path.basename(os.path.dirname(path)), "device": device, "half": half}
is_onnx = path.endswith(".onnx")

# ---- A. raw forward, no ultralytics pre/post ----
x = np.random.rand(1, 3, 640, 640).astype(np.float16 if (is_onnx and half) else np.float32)
if is_onnx:
    import onnxruntime as ort
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(path, providers=prov)
    out["ort_providers_used"] = sess.get_providers()
    name = sess.get_inputs()[0].name
    fwd = lambda: sess.run(None, {name: x})
else:
    m = YOLO(path).model.eval()
    dev = torch.device("cuda:0" if device != "cpu" else "cpu")
    m = m.to(dev); m = m.half() if half else m.float()
    xt = torch.from_numpy(x).to(dev); xt = xt.half() if half else xt
    out["torch_param_device"] = str(next(m.parameters()).device)
    out["torch_param_dtype"] = str(next(m.parameters()).dtype)
    def fwd():
        with torch.inference_mode():
            m(xt)
        if dev.type == "cuda":
            torch.cuda.synchronize()
n = N if device != "cpu" else 20
for _ in range(WARM if device != "cpu" else 3):
    fwd()
s = Sampler(); s.start(); t = []
for _ in range(n):
    t0 = time.perf_counter(); fwd(); t.append((time.perf_counter() - t0) * 1e3)
s.stop = True; s.join()
out["raw_forward_ms"] = stats(t)
out["raw_gpu_load_pct_mean"] = round(float(np.mean(s.load)), 1)
out["raw_gpu_freq_mhz_max"] = max(s.freq)
del fwd

# ---- B. the pipeline the node runs: predict() and track() on a real image ----
img = cv2.resize(cv2.imread(os.path.join(os.path.dirname(__import__("ultralytics").__file__), "assets/bus.jpg")), (640, 480))
y = YOLO(path, task="detect")
kw = dict(imgsz=640, device=(0 if device != "cpu" else "cpu"), verbose=False, conf=0.5)
if not is_onnx:
    kw["half"] = half
for _ in range(WARM if device != "cpu" else 3):
    y.predict(img, **kw)
if is_onnx:
    b = y.predictor.model
    sess2 = getattr(b, "session", None) or getattr(getattr(b, "backend", None), "session", None)
    out["ultralytics_ort_providers"] = sess2.get_providers() if sess2 else "n/a"
else:
    b = y.predictor.model
    out["ultralytics_param_device"] = f"{b.device} / {next(getattr(b, 'model', b).parameters()).device} {next(getattr(b, 'model', b).parameters()).dtype}"
pre, inf, post, wall = [], [], [], []
s = Sampler(); s.start()
for _ in range(n):
    t0 = time.perf_counter(); r = y.predict(img, **kw)[0]; wall.append((time.perf_counter() - t0) * 1e3)
    pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
s.stop = True; s.join()
out["predict"] = {"pre": stats(pre)["mean"], "inf": stats(inf)["mean"], "post": stats(post)["mean"], "wall": stats(wall)}
out["predict_ndet"] = len(r.boxes)
out["predict_gpu_load_pct_mean"] = round(float(np.mean(s.load)), 1)
tw = []
for i in range(WARM + n if device != "cpu" else 10):
    t0 = time.perf_counter(); y.track(img, persist=True, **kw); d = (time.perf_counter() - t0) * 1e3
    if i >= (WARM if device != "cpu" else 3):
        tw.append(d)
out["track_wall"] = stats(tw)
print("RESULT " + json.dumps(out))
