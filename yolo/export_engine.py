#!/usr/bin/env python3
"""Build the TensorRT engine `make yolo` runs, if it is missing (D-30).

    ~/yolo/venv/bin/python yolo/export_engine.py \
        --model ~/yolo/yolo26l.pt --out ~/yolo/yolo26l_480x640.engine --imgsz 480 640

An engine only loads on the TensorRT version that built it, so after a JetPack
change the file has to be rebuilt ON THIS BOARD -- the .pt is the source, the
engine is a cache. This does nothing when --out already exists (pass --force to
rebuild), so `make yolo` pays for it once.

The build is the expensive thing on this robot: ~13 min and ~3.1 GB of the
7.4 GB that CPU and GPU share, with no disk swap behind it. It therefore
refuses to start with less than --min-free-mb available -- i.e. with the full
stack up -- instead of pushing the stack into the OOM killer mid-demo.

imgsz is (480, 640), the camera's own shape: a square 640x640 export pads the
frame and measured ~2 F1 points worse (records/calibration.md, 24 Sep).
"""
import argparse
import os
import shutil
import sys
import tempfile
import time


def mem_available_mb():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return -1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="source .pt")
    ap.add_argument("--out", required=True, help="engine file to create")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[480, 640], metavar=("H", "W"))
    ap.add_argument("--min-free-mb", type=int, default=4500)
    ap.add_argument("--force", action="store_true", help="rebuild even if --out exists")
    a = ap.parse_args()

    src = os.path.expanduser(a.model)
    out = os.path.expanduser(a.out)
    if os.path.exists(out) and not a.force:
        print(f"{out} exists -- skipping the engine build (FORCE=true to rebuild).")
        return 0
    if not os.path.exists(src):
        print(f"!! {src} not found. Fetch it once from ~/yolo:\n"
              f"   cd ~/yolo && YOLO_OFFLINE=0 {sys.executable} -c "
              f"\"from ultralytics.utils.downloads import attempt_download_asset as a; "
              f"a('{os.path.basename(src)}')\"", file=sys.stderr)
        return 2
    free = mem_available_mb()
    if free < a.min_free_mb:
        print(f"!! only {free} MB available; the engine build needs ~3.1 GB and the "
              f"board has no disk swap. Stop the stack first (or --min-free-mb).",
              file=sys.stderr)
        return 3

    os.environ["YOLO_OFFLINE"] = "1"
    from ultralytics import YOLO
    print(f"building {out} from {src} at imgsz {a.imgsz}, fp16 -- ~13 min, "
          f"{free} MB available. Do not start the stack meanwhile.", flush=True)
    t0 = time.time()
    # Export in a scratch dir: ultralytics writes <stem>.onnx and <stem>.engine
    # beside the .pt, and a stray 480x640 yolo26l.onnx next to the real .pt
    # would look like a deployable model.
    with tempfile.TemporaryDirectory(dir=os.path.dirname(out)) as tmp:
        pt = os.path.join(tmp, os.path.basename(src))
        shutil.copy2(src, pt)
        built = YOLO(pt).export(format="engine", half=True, imgsz=tuple(a.imgsz),
                                device=0, batch=1, dynamic=False)
        os.replace(built, out)
    print(f"built {out} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
