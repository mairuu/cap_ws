#!/usr/bin/env python3
"""Export an ultralytics .pt to .onnx on THIS board, idempotently.

WHY THIS EXISTS. `make yolo` used to hand the detector a .pt and let torch run
it. A .onnx runs under onnxruntime-gpu instead, which needs the graph built
once, ahead of time. The old board's equivalent was /home/jetson/yolo/compile.py
(lost with the board, see reference/nvme-recovery-audit.md); this is not that
file -- that one built TensorRT .engine plans, which are tied to the TensorRT
version that built them. An .onnx is not: it is a portable graph, so unlike the
recovered engines it survives a JetPack change.

IDEMPOTENT ON PURPOSE. `make yolo` depends on this, and re-exporting on every
bring-up would add ~30 s to a demo-day start. The export is skipped when the
.onnx already exists, is newer than the .pt, and its EMBEDDED metadata matches
what was asked for. ultralytics writes `imgsz`, `task` and `names` into the
ONNX metadata_props at export, so the file itself says what it was built for --
there is no sidecar to fall out of date. --force re-exports regardless.

THE imgsz TRAP. A static ONNX graph has its input resolution baked in. Exporting
at 640 and then running `make yolo IMGSZ=480` would feed a 480 letterbox to a
640 input: onnxruntime raises a shape error, or worse, ultralytics letterboxes
to 640 anyway and IMGSZ silently does nothing. That is exactly what the metadata
check above catches -- change IMGSZ and this re-exports.

RUN IT WITH THE VENV'S PYTHON, not the system one: torch and ultralytics live
in the hand-built venv (~/yolo/venv, see setup_yolo_venv.sh). No rclpy is
needed here, so there is no PYTHONPATH trick -- just the venv interpreter.

    ~/yolo/venv/bin/python yolo/export_onnx.py --model ~/yolo/yolo26s.pt
    make yolo-onnx                      # the same thing, with the Makefile's vars
    make yolo-onnx FORCE=true           # re-export even if up to date
"""

import argparse
import os
import sys
import time


def _fail_import(err):
    print(
        f"\nCannot import torch/ultralytics: {err}\n"
        f"Run this with the hand-built venv's interpreter, not the system one:\n"
        f"    ~/yolo/venv/bin/python yolo/export_onnx.py ...\n"
        f"The venv is rebuilt by cap_ws/yolo/setup_yolo_venv.sh. Do NOT run\n"
        f"`uv sync` against it -- see reference/nvme-recovery-audit.md.\n",
        file=sys.stderr,
    )
    sys.exit(2)


try:
    import torch
    from ultralytics import YOLO
except ImportError as e:  # pragma: no cover
    _fail_import(e)


def onnx_metadata(path):
    """The imgsz/task ultralytics baked into the file, or None if unreadable.

    Unreadable is not an error here: a truncated or half-written .onnx from an
    interrupted export should simply be rebuilt, not crash the bring-up.
    """
    try:
        import onnx
        meta = {p.key: p.value for p in onnx.load(path, load_external_data=False).metadata_props}
    except Exception:
        return None
    return meta


def up_to_date(pt, out, imgsz, half):
    if not os.path.exists(out):
        return False, "no .onnx yet"
    if os.path.getmtime(out) < os.path.getmtime(pt):
        return False, f"{os.path.basename(pt)} is newer than the .onnx"
    meta = onnx_metadata(out)
    if meta is None:
        return False, "existing .onnx has no readable metadata (truncated export?)"
    # ultralytics writes imgsz as the string form of a [h, w] list.
    want = str([imgsz, imgsz])
    if meta.get("imgsz") != want:
        return False, f"built for imgsz {meta.get('imgsz')}, want {want}"
    # `half` is not in the metadata, so read the graph's own input dtype:
    # 1 is FLOAT, 10 is FLOAT16 in the ONNX TensorProto enum.
    try:
        import onnx
        elem = onnx.load(out, load_external_data=False).graph.input[0].type.tensor_type.elem_type
    except Exception:
        return False, "cannot read the .onnx input dtype"
    is_half = elem == 10
    if is_half != half:
        return False, f"built {'fp16' if is_half else 'fp32'}, want {'fp16' if half else 'fp32'}"
    return True, f"imgsz {imgsz}, {'fp16' if half else 'fp32'}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser("~/yolo/yolo26s.pt"),
                    help="source .pt weights (default: ~/yolo/yolo26s.pt)")
    ap.add_argument("--out", default=None,
                    help="destination .onnx (default: the .pt path with .onnx)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference size baked into the graph (default: 640)")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset (default: 17)")
    ap.add_argument("--half", action="store_true",
                    help="export fp16 weights; needs a GPU and --device cuda")
    ap.add_argument("--device", default="cuda:0",
                    help="device to trace on (default: cuda:0)")
    ap.add_argument("--no-simplify", dest="simplify", action="store_false",
                    help="skip the onnxslim graph simplification pass")
    ap.add_argument("--force", action="store_true",
                    help="re-export even if the .onnx is already up to date")
    args = ap.parse_args()

    pt = os.path.expanduser(args.model)
    out = os.path.expanduser(args.out) if args.out else os.path.splitext(pt)[0] + ".onnx"

    if not os.path.exists(pt):
        print(f"\nweights not found: {pt}\n"
              f"Fetch them once, on a machine with a network:\n"
              f"    curl -L -o {pt} \\\n"
              f"      https://github.com/ultralytics/assets/releases/download/v8.4.0/"
              f"{os.path.basename(pt)}\n"
              f"`make yolo` sets YOLO_OFFLINE=1 deliberately, so ultralytics will\n"
              f"NOT download this for you at run time.\n", file=sys.stderr)
        return 2

    if not args.force:
        ok, why = up_to_date(pt, out, args.imgsz, args.half)
        if ok:
            print(f"{out} is up to date ({why}) -- skipping export. --force to rebuild.")
            return 0
        print(f"exporting: {why}")

    if args.half and not args.device.startswith("cuda"):
        print("--half needs a CUDA device; fp16 export on cpu is not supported "
              "by ultralytics.", file=sys.stderr)
        return 2
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("device is cuda but torch.cuda.is_available() is False -- the venv "
              "is wrong (PyPI torch instead of the JetPack wheel?). Re-run "
              "cap_ws/yolo/setup_yolo_venv.sh.", file=sys.stderr)
        return 2

    print(f"{pt} -> {out}   imgsz {args.imgsz}, opset {args.opset}, "
          f"{'fp16' if args.half else 'fp32'}, device {args.device}")

    # ultralytics ALWAYS writes beside the weights, at <stem>.onnx, with no way
    # to redirect it. So an --out pointing anywhere else would silently destroy
    # the .onnx that `make yolo` runs -- export clobbers it, and then we move
    # the result away. Cost two accidental deletions on 16 Sep while measuring
    # fp16 against fp32. Set it aside first and put it back afterwards.
    default_out = os.path.splitext(pt)[0] + ".onnx"
    stash = None
    if os.path.abspath(default_out) != os.path.abspath(out) and os.path.exists(default_out):
        stash = default_out + ".stashed"
        os.replace(default_out, stash)

    t0 = time.perf_counter()
    try:
        produced = YOLO(pt, task="detect").export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            half=args.half,
            device=args.device,
            simplify=args.simplify,
            dynamic=False,   # static shapes: see THE imgsz TRAP above
            nms=False,       # ultralytics does its own NMS in postprocess
            verbose=False,
        )
        dt = time.perf_counter() - t0
        produced = os.path.abspath(str(produced))
        if produced != os.path.abspath(out):
            os.replace(produced, out)
    finally:
        if stash is not None:
            os.replace(stash, default_out)

    size_mb = os.path.getsize(out) / 1e6
    print(f"wrote {out}  ({size_mb:.1f} MB, {dt:.1f} s)")
    ok, why = up_to_date(pt, out, args.imgsz, args.half)
    if not ok:
        print(f"WARNING: the exported file does not verify: {why}", file=sys.stderr)
        return 1
    print(f"verified: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
