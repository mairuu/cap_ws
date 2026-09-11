#!/usr/bin/env python3
"""Pin fx to a tape measure. The one check a checkerboard calibration cannot do alone.

WHY THIS EXISTS. A chessboard calibration has no absolute length in it. Scale
the squares by any factor k and the solver scales every board distance by k and
returns *exactly the same* K and D -- verified on this board 11 Sep: fx, fy, cx
and cy agreed to seven significant figures across a 2.5x change in --square,
while the reported board distance went 0.42 m -> 1.05 m.

Two consequences, and the second is why this script exists:

  GOOD: a mis-measured printout cannot corrupt fx, so it cannot corrupt any
  BEARING. atan((u - cx) / fx) has no length in it either. The semantic layer
  is safe from a badly printed board.

  BAD: nothing inside the calibration can tell you fx is wrong. Reprojection
  error cannot -- it is computed in pixels against the same self-consistent fit.
  A degenerate capture (every board at roughly the same depth) lets fx trade
  against board distance almost freely, and the result fits its own images
  beautifully while being 15-20% out. That is not hypothetical: the 11 Sep
  48-image run split into 819.74 (first 29, all at 0.70-1.23 m, cx running away
  to 391.7) against 676.30 (last 19, spanning 0.18-1.01 m, cx a healthy 322.1).

The fix is to introduce a length the calibration does not control: a tape
measure. Put the board at a known distance and see whether the model agrees.

HOW IT WORKS. solvePnP the board with the installed K and D. The reported Z
scales with fx -- an fx that is 10% high puts the board 10% too far away. So

    Z_reported  =  (fx_config / fx_true) * (D_tape + e)

where `e` is the unknown offset between whatever you measured from and the
lens's entrance pupil, which is somewhere inside the glass. Measure at several
distances and regress Z on D: the SLOPE is fx_config / fx_true and the
intercept absorbs `e` entirely. That is the whole point of using more than one
distance -- it removes the one quantity you cannot measure.

    fx_true = fx_config / slope        slope 1.000 means the config is right

A single distance still works, but `e` then goes straight into the answer, so
use a long one and treat the result as approximate.

USAGE

    ros2 run my_bot camera_check_scale.py --distances 0.4,0.7,1.0

with `make camera` already running in another terminal. It prompts before each
distance. Hold the board FLAT-ON and roughly centred -- tilt is handled by
solvePnP, but a board filling the frame edge-to-edge is measured through the
worst of the distortion.

MEASURE FROM THE FRONT OF THE LENS, consistently, and do not try to guess where
the pupil is -- that is what the intercept is for. Measure to the board's
PRINTED SURFACE, not to whatever it is taped to.

WHAT TO DO WITH THE ANSWER. Slope within ~2% of 1.0: the calibration is sound,
record it and move on. Slope well off 1.0: the capture was degenerate, or the
focus moved. Recapture with the board swept through a WIDE RANGE OF DEPTHS --
that is what conditions fx -- and with autofocus locked (`make camera` does
this; see the Makefile).
"""

import argparse
import sys

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class Grabber(Node):
    """Collects frames off /image. RELIABLE to match cam2image, which is RELIABLE."""

    def __init__(self, topic):
        super().__init__("camera_check_scale")
        self.frames = []
        self.want = 0
        # cam2image publishes RELIABLE. A BEST_EFFORT subscriber is compatible,
        # but keep it RELIABLE so a dropped frame is a real fault, not silence.
        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.sub = self.create_subscription(Image, topic, self._cb, qos)

    def _cb(self, msg):
        if len(self.frames) >= self.want:
            return
        if msg.encoding not in ("bgr8", "rgb8", "mono8"):
            self.get_logger().warn("unhandled encoding %s" % msg.encoding)
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding == "mono8":
            img = buf.reshape(msg.height, msg.width)
        else:
            img = buf.reshape(msg.height, msg.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if msg.encoding == "bgr8"
                               else cv2.COLOR_RGB2GRAY)
        self.frames.append(img)

    def grab(self, n, timeout=20.0):
        self.frames, self.want = [], n
        deadline = self.get_clock().now().nanoseconds + timeout * 1e9
        while len(self.frames) < n:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.get_clock().now().nanoseconds > deadline:
                break
        return list(self.frames)


def load_config(path):
    with open(path) as f:
        info = yaml.safe_load(f)
    m = info["camera_matrix"]
    K = np.array(m["data"], dtype=np.float64).reshape(m["rows"], m["cols"])
    d = info["distortion_coefficients"]
    D = np.array(d["data"], dtype=np.float64).reshape(-1, 1)
    return info, K, D


def board_depth(frames, K, D, cols, rows, square):
    """Median solvePnP Z over the frames in which the board is found."""
    objp = np.zeros((rows * cols, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    zs = []
    for gray in frames:
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, D)
        if ok:
            zs.append(float(tvec[2]))
    return zs


def main():
    ap = argparse.ArgumentParser(
        description="Check the installed fx against a tape measure.")
    ap.add_argument("--distances", default="0.4,0.7,1.0",
                    help="comma-separated tape distances in METRES. Two or more "
                         "cancels the entrance-pupil offset; one does not.")
    ap.add_argument("--config", default=None,
                    help="camera_info yaml (default: the installed c615_640x480.yaml)")
    ap.add_argument("--size", default="9x6", help="interior corners, COLSxROWS")
    ap.add_argument("--square", type=float, default=0.020, help="square size, METRES")
    ap.add_argument("--topic", default="/image")
    ap.add_argument("--frames", type=int, default=12,
                    help="frames to average per distance (default 12)")
    args = ap.parse_args()

    cols, rows = (int(v) for v in args.size.lower().split("x"))
    cfg = args.config
    if cfg is None:
        import os
        here = os.path.realpath(__file__)
        cfg = os.path.join(os.path.dirname(os.path.dirname(here)),
                           "config", "c615_640x480.yaml")
    try:
        info, K, D = load_config(cfg)
    except FileNotFoundError:
        sys.exit("no camera config at %s -- run `make calib-report` first" % cfg)

    fx_cfg = K[0, 0]
    print("config     %s" % cfg)
    print("fx %.4f  fy %.4f  cx %.4f  cy %.4f  @ %dx%d"
          % (fx_cfg, K[1, 1], K[0, 2], K[1, 2], info["image_width"], info["image_height"]))
    print("board      %dx%d, %.1f mm\n" % (cols, rows, args.square * 1000.0))

    dists = [float(v) for v in args.distances.split(",")]
    rclpy.init()
    node = Grabber(args.topic)

    measured = []
    try:
        for d in dists:
            input("place the board FLAT-ON and centred at %.3f m, "
                  "measured from the front of the lens to the printed face, "
                  "then press Enter... " % d)
            frames = node.grab(args.frames)
            if not frames:
                sys.exit("no frames on %s -- is `make camera` running, and is "
                         "ROS_DOMAIN_ID 42 in this shell?" % args.topic)
            zs = board_depth(frames, K, D, cols, rows, args.square)
            if len(zs) < 3:
                print("  board found in only %d of %d frames -- skipping this "
                      "distance. Move it into better light or square it up."
                      % (len(zs), len(frames)))
                continue
            z = float(np.median(zs))
            spread = float(np.max(zs) - np.min(zs))
            measured.append((d, z))
            print("  tape %.3f m  ->  model says %.4f m   (ratio %.4f, "
                  "spread over %d frames %.1f mm)"
                  % (d, z, z / d, len(zs), spread * 1000.0))
            if spread > 0.02:
                print("  WARN: %.0f mm of spread across frames at one fixed "
                      "distance. Autofocus hunting, or the board moved."
                      % (spread * 1000.0))
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not measured:
        sys.exit("\nno usable distances -- nothing measured")

    print()
    if len(measured) >= 2:
        d = np.array([m[0] for m in measured])
        z = np.array([m[1] for m in measured])
        slope, intercept = np.polyfit(d, z, 1)
        fx_true = fx_cfg / slope
        resid = z - (slope * d + intercept)
        print("regression over %d distances:  Z = %.4f * D + %.4f" %
              (len(measured), slope, intercept))
        print("  residuals   %s mm" % np.round(resid * 1000.0, 1).tolist())
        print("  entrance-pupil offset absorbed by the intercept: %.1f mm"
              % (intercept * 1000.0))
        if abs(intercept) > 0.05:
            print("  WARN: an intercept over 50 mm is not a lens offset. Suspect "
                  "an inconsistent measuring datum between distances.")
    else:
        d, z = measured[0]
        slope = z / d
        fx_true = fx_cfg / slope
        print("ONE distance only -- the entrance-pupil offset goes straight into "
              "this number.\nTreat it as approximate; pass two or more distances "
              "to remove it.")

    hf_cfg = 2.0 * np.degrees(np.arctan2(info["image_width"] / 2.0, fx_cfg))
    hf_true = 2.0 * np.degrees(np.arctan2(info["image_width"] / 2.0, fx_true))
    err = 100.0 * (fx_cfg - fx_true) / fx_true

    print()
    print("  fx in config     %8.2f   (HFOV %.1f deg)" % (fx_cfg, hf_cfg))
    print("  fx from the tape %8.2f   (HFOV %.1f deg)" % (fx_true, hf_true))
    print("  the config is %+.1f%% off" % err)
    print()

    if abs(err) < 2.0:
        print("  PASS. The calibration agrees with the tape. Record fx and the")
        print("  slope in records/calibration.md and move on.")
        return 0

    print("  FAIL. A %+.1f%% fx error is a %+.1f%% bearing error at the frame"
          % (err, err))
    print("  edge, which is what the semantic layer is built on.")
    print()
    print("  This is almost always a DEGENERATE CAPTURE, not a bad board: if every")
    print("  view was at roughly the same depth, fx and board distance trade off")
    print("  against each other and the fit is happy while being badly wrong.")
    print("  Recapture with the board swept through a wide range of depths --")
    print("  close enough to fill the frame, far enough to be a small patch --")
    print("  and check cx, cy come out near the frame centre. Do NOT simply")
    print("  scale fx by this factor; recalibrate.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
