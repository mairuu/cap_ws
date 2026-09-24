#!/usr/bin/env python3
"""YOLO detection + tracking off /image, published as vision_msgs/Detection2DArray.

WHAT THIS IS. The Day 5 detector, decision D-11 option B: one custom node,
ultralytics `model.track(persist=True)`, `vision_msgs` on `/detections`. Not
`yolo_ros`. The recovered stack ran `yolo_ros` + `yolo_msgs` + TensorRT engines
and measured 15 Hz, but two of its four prerequisites died with the board (a
one-line `yolo_ros` patch, and the engines themselves, which JetPack 6.1's
TensorRT will not load). This node has none of those dependencies and the June
`semantic_objects` tree already parses exactly what it publishes.

WHAT IT PUBLISHES. One Detection2DArray per processed frame, on /detections:

    header          COPIED FROM THE IMAGE. header.stamp is the CAPTURE time, not
                    the publish time. The semantic node looks up TF at this
                    stamp (design note P2); a publish-time stamp would put every
                    landmark where the robot is now, not where it was when the
                    camera saw the object. Frame id is whatever cam2image set
                    (camera_link, per the Makefile).
    detections[i]
      .id           the tracker's track id, as a string ("" if the tracker gave
                    none this frame). This is what satisfies P5: the semantic
                    layer can tell "the same chair, again" from "a second chair".
                    Detection2D.id is a string in vision_msgs, hence str().
      .bbox         centre + size, in pixels of the ORIGINAL image, whatever
                    imgsz the network ran at. ultralytics scales boxes back.
      .results[0]   one hypothesis: class_id is the CLASS NAME ("chair", not
                    "56"), score is the confidence. The June node reads
                    hypothesis.class_id straight into its class_label.

WHY THE INPUT QUEUE IS ONE DEEP. cam2image publishes at 15 Hz whether or not
we keep up. A deep queue would let the node fall steadily behind and publish
detections for frames from seconds ago, with stamps to match -- which is worse
than dropping frames, because the semantic node would fuse them at the right
pose for the wrong image content. depth=1 KEEP_LAST means the executor always
hands us the newest frame and silently discards the ones we were too slow for.
The per-frame age printed in the stats line is how you see whether that is
happening: it should sit near one frame period, not grow.

QoS. cam2image publishes RELIABLE by default and a best-effort subscriber will
never match it, so `image_reliability` defaults to reliable. If the camera
driver ever changes to best-effort, set the parameter -- the node logs which it
chose. /detections is published RELIABLE, which matches either kind of
subscriber (the June semantic node subscribes best-effort; fine).

THE INTERPRETER. This file runs under /usr/bin/python3 like every other node,
with the hand-built venv's site-packages PREPENDED to PYTHONPATH by
launch/yolo.launch.py. That is the whole venv trick: same interpreter, so
rclpy/cv_bridge come from the system and torch/ultralytics from the venv. There
is no `activate`. Running it bare (`ros2 run my_bot yolo_detector.py`) fails
at `import ultralytics` -- use `make yolo`, or export PYTHONPATH yourself.

MODELS. Three formats load here, and `precision` applies to only one of them.

  .onnx    THE DEFAULT since 16 Sep: yolo26s.onnx, built from the .pt by
           yolo/export_onnx.py, which `make yolo` runs for you. Runs under
           onnxruntime-gpu's CUDA execution provider. Resolution and precision
           are BAKED INTO THE GRAPH at export, so `imgsz` and `precision` here
           must match what it was built with -- the Makefile keeps them in step
           and re-exports when IMGSZ changes. An .onnx is portable: unlike an
           engine it does not die with a JetPack change.
  .pt      Loaded by torch, runs on CUDA directly. The Day 5 path, kept as the
           fallback: `make yolo MODEL=~/yolo/yolo26s.pt` skips the export.
  .engine  A TensorRT plan, and it only loads on the TensorRT version that
           built it -- JetPack 6.1 is TensorRT 10.3, so the recovered engines
           are dead and a new one must be exported on this board.

    make yolo
    make yolo MODEL=~/yolo/yolo26s.pt                    # torch fallback
    make yolo IMGSZ=480                                  # re-exports the .onnx
    ros2 run my_bot detection_report.py --seconds 60     # the gate check

Ctrl-C to stop.
"""

import argparse
import os
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from sensor_msgs.msg import Image
from vision_msgs.msg import (BoundingBox2D, Detection2D, Detection2DArray,
                             ObjectHypothesisWithPose)

from cv_bridge import CvBridge


def _fail_import(what, err):
    print(
        f"\nCannot import {what}: {err}\n"
        f"This node needs the hand-built venv's site-packages on PYTHONPATH.\n"
        f"Use `make yolo`, which sets it, or:\n"
        f"    export PYTHONPATH=$HOME/yolo/venv/lib/python3.10/site-packages:$PYTHONPATH\n"
        f"The venv itself is rebuilt by cap_ws/yolo/setup_yolo_venv.sh. Do NOT\n"
        f"run `uv sync` against it -- see reference/nvme-recovery-audit.md.\n",
        file=sys.stderr,
    )
    sys.exit(2)


try:
    import numpy as np
    import torch
    from ultralytics import YOLO
except ImportError as e:  # pragma: no cover
    _fail_import("torch/ultralytics", e)


class YoloDetector(Node):

    def __init__(self):
        super().__init__("yolo_detector")

        self.declare_parameter("model", os.path.expanduser("~/yolo/yolo26s.onnx"))
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.5)
        self.declare_parameter("iou", 0.7)
        self.declare_parameter("precision", "fp16")
        self.declare_parameter("tracker", "bytetrack.yaml")
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("image_reliability", "reliable")
        self.declare_parameter("detections_topic", "/detections")
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("debug_topic", "/detections/image")
        self.declare_parameter("report_period", 5.0)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self._model_path = os.path.expanduser(p("model"))
        self._device = p("device")
        self._imgsz = int(p("imgsz"))
        self._conf = float(p("conf"))
        self._iou = float(p("iou"))
        # ultralytics 8.4: `half` is deprecated for `quantize` (16 or None).
        self._quantize = 16 if str(p("precision")).lower() in ("fp16", "16", "half") else None
        self._tracker = p("tracker")
        self._publish_debug = bool(p("publish_debug"))
        self._report_period = float(p("report_period"))

        if not os.path.exists(self._model_path):
            hint = ("Build it: `make yolo-onnx`, which exports the .pt beside it."
                    if self._model_path.endswith(".onnx") else
                    "ultralytics downloads a bare name like 'yolo26s.pt' into the "
                    "cwd on first use; do that once from ~/yolo, then point here.")
            self.get_logger().fatal(f"model not found: {self._model_path}. {hint}")
            raise SystemExit(2)

        # Precision is baked in at export for both compiled formats, so the
        # runtime flag is meaningless for them -- and for ONNX it is worse than
        # meaningless: quantize=16 against an fp32 graph feeds onnxruntime an
        # fp16 tensor its input does not accept. Keep the log honest and the
        # dtype right by clearing it.
        ext = os.path.splitext(self._model_path)[1]
        fmt = {".engine": "TensorRT engine", ".onnx": "ONNX (onnxruntime)"}.get(ext, "torch")
        if ext in (".engine", ".onnx"):
            self._quantize = None
        # The input resolution is baked in too. Do NOT pass imgsz for them:
        # ultralytics then takes the shape from the model's own metadata. For
        # a .onnx it does that even when given imgsz, but a TensorRT engine
        # asserts instead -- a 480x640 engine (the camera's shape, which
        # scores ~2 F1 points better than a padded 640x640, 24 Sep) dies on
        # `imgsz=640` with "input size ... not equal to max model size".
        self._size_kw = {} if ext in (".engine", ".onnx") else {"imgsz": self._imgsz}

        if self._device.startswith("cuda") and not torch.cuda.is_available():
            self.get_logger().fatal(
                "device is cuda but torch.cuda.is_available() is False. The "
                "venv is wrong (PyPI torch instead of the JetPack wheel?) -- "
                "re-run cap_ws/yolo/setup_yolo_venv.sh. For a demo-day "
                "fallback: device:=cpu, expect ~5 Hz.")
            raise SystemExit(2)

        t0 = time.perf_counter()
        self._model = YOLO(self._model_path, task="detect")
        self._names = self._model.names
        load_ms = (time.perf_counter() - t0) * 1000

        # The first inference pays for CUDA context + cuDNN autotune + (for
        # fp16) weight conversion: seconds, not milliseconds. Do it now on a
        # blank frame so the first real frame is not a multi-second stall that
        # the depth-1 queue then turns into a burst of dropped frames.
        warm = np.zeros((480, 640, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        self._model.predict(warm, device=self._device,
                            quantize=self._quantize, verbose=False, **self._size_kw)
        warm_ms = (time.perf_counter() - t0) * 1000

        dev_name = (torch.cuda.get_device_name(0)
                    if self._device.startswith("cuda") else "cpu")
        self.get_logger().info(
            f"model {self._model_path} ({fmt}) "
            f"on {dev_name}; imgsz {list(self._model.predictor.imgsz)}"
            f"{' (from the model)' if not self._size_kw else ''}, "
            # For a compiled format the node does not choose the precision --
            # it is in the graph. Printing "fp32/native" because quantize is
            # cleared would misreport an fp16 .onnx as fp32 (seen 16 Sep).
            f"precision {'baked in at export' if ext in ('.engine', '.onnx') else ('fp16' if self._quantize == 16 else 'fp32/native')}, "
            f"conf {self._conf}, iou {self._iou}, tracker {self._tracker}; "
            f"load {load_ms:.0f} ms, warm-up {warm_ms:.0f} ms; "
            f"torch {torch.__version__}")

        self._bridge = CvBridge()

        rel = str(p("image_reliability")).lower()
        image_qos = QoSProfile(
            reliability=(ReliabilityPolicy.BEST_EFFORT if rel == "best_effort"
                         else ReliabilityPolicy.RELIABLE),
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        det_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._pub = self.create_publisher(Detection2DArray, p("detections_topic"), det_qos)
        self._dbg_pub = (self.create_publisher(Image, p("debug_topic"), det_qos)
                         if self._publish_debug else None)
        self._sub = self.create_subscription(Image, p("image_topic"),
                                             self._on_image, image_qos)
        self.get_logger().info(
            f"subscribed {p('image_topic')} ({rel}, depth 1) -> "
            f"{p('detections_topic')}"
            + (f" + {p('debug_topic')}" if self._publish_debug else ""))

        # Rolling stats, reported every report_period seconds.
        self._n = 0
        self._infer_ms = []
        self._age_ms = []
        self._ndet = []
        self._ids = set()
        self._last_report = time.monotonic()
        self.create_timer(self._report_period, self._report)

    # ------------------------------------------------------------------

    def _on_image(self, msg: Image):
        t_cb = time.perf_counter()
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"cv_bridge failed on {msg.encoding}: {e}",
                                   throttle_duration_sec=5.0)
            return

        results = self._model.track(
            frame,
            persist=True,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            quantize=self._quantize,
            tracker=self._tracker,
            verbose=False,
            **self._size_kw,
        )
        infer_ms = (time.perf_counter() - t_cb) * 1000

        r = results[0]
        out = Detection2DArray()
        out.header = msg.header          # capture time, camera frame (P2)

        boxes = r.boxes
        if boxes is not None and len(boxes):
            xywh = boxes.xywh.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            ids = (boxes.id.cpu().numpy().astype(int)
                   if boxes.id is not None else [None] * len(boxes))
            for (cx, cy, w, h), c, k, tid in zip(xywh, confs, clss, ids):
                d = Detection2D()
                d.header = msg.header
                d.id = "" if tid is None else str(int(tid))
                d.bbox = BoundingBox2D()
                d.bbox.center.position.x = float(cx)
                d.bbox.center.position.y = float(cy)
                d.bbox.center.theta = 0.0
                d.bbox.size_x = float(w)
                d.bbox.size_y = float(h)
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(self._names.get(int(k), int(k)))
                hyp.hypothesis.score = float(c)
                d.results.append(hyp)
                out.detections.append(d)
                if tid is not None:
                    self._ids.add(int(tid))

        self._pub.publish(out)

        if self._dbg_pub is not None and self._dbg_pub.get_subscription_count() > 0:
            # plot() draws boxes, labels and track ids; skipped when nobody is
            # looking, since it is a few ms of CPU per frame.
            dbg = self._bridge.cv2_to_imgmsg(r.plot(), encoding="bgr8")
            dbg.header = msg.header
            self._dbg_pub.publish(dbg)

        now = self.get_clock().now().nanoseconds
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self._n += 1
        self._infer_ms.append(infer_ms)
        self._age_ms.append((now - stamp) / 1e6)
        self._ndet.append(len(out.detections))

    def _report(self):
        now = time.monotonic()
        dt = now - self._last_report
        self._last_report = now
        if not self._infer_ms:
            self.get_logger().warn(
                f"no frames in {dt:.0f} s -- is cam2image up, and is its "
                f"reliability the same as image_reliability?")
            return
        ms = self._infer_ms
        age = self._age_ms
        self.get_logger().info(
            f"{len(ms)/dt:5.1f} Hz | infer+track p50 {statistics.median(ms):5.1f} "
            f"max {max(ms):5.1f} ms | age@publish p50 {statistics.median(age):5.0f} "
            f"max {max(age):4.0f} ms | dets/frame {statistics.mean(self._ndet):.1f} "
            f"| track ids so far {len(self._ids)}")
        self._infer_ms.clear()
        self._age_ms.clear()
        self._ndet.clear()


def main():
    # --help without starting ROS.
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter
                                ).print_help()
        return
    rclpy.init(args=sys.argv)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
