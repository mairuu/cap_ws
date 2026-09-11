#!/usr/bin/env python3
"""Score a cameracalibrator run and install the result. Gives the number the GUI hides.

WHY THIS EXISTS. The Day 4 gate says "reprojection error < 0.5 px". You cannot
read that off `cameracalibrator`. It is computed and then dropped on the floor:

    /opt/ros/humble/lib/python3.10/site-packages/camera_calibration/calibrator.py:797
        reproj_err, self.intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(...)

`reproj_err` is never stored and never printed. What the GUI shows next to the
CALIBRATE button is the *linear* error -- how straight an undistorted row of
corners comes out -- which is a different measurement with a different scale and
is not the gate. The tarball it saves contains `ost.yaml` and every image it
used, so the number is recoverable after the fact. That is all this does.

It also gives you the thing that actually improves a bad run: the error PER
IMAGE. A run that lands at 0.8 px is usually four good frames and one where the
board was tilted past 60 degrees or caught motion blur. Drop those frames and
re-run; do not re-shoot the whole set blind.

WHAT IT CHECKS

  * RMS reprojection error of the SHIPPED intrinsics, overall and per image.
    Per image it solvePnPs the board with K and D fixed, then reprojects -- so
    this scores ost.yaml as published, not a fresh fit that would flatter it.
  * the resolution in ost.yaml is the one we actually run (640x480). cam2image
    DEFAULTS TO 320x240, and a calibration captured at the default is silently
    wrong for the real pipeline: fx, fy, cx, cy all scale with resolution.
  * DEPTH SPREAD of the board across the images. This is the one that matters
    and the one that caught the 11 Sep run. fx and board distance are nearly
    degenerate when every view is at the same depth: the solver can trade one
    against the other and still fit its own images beautifully. Splitting that
    run by capture order gave fx 819.74 from the 29 frames at 0.70-1.23 m
    against 676.30 from the 19 spanning 0.18-1.01 m -- 17.5% apart, with the
    narrow-depth group also throwing cx out to 391.7. Reprojection error saw
    none of it: the narrow group scored 0.15 px, the wide one 0.53.
  * horizontal FOV implied by fx, against the C615's ~62 deg spec -- as a weak
    smell test only. NOTE, because an earlier version of this file said the
    opposite: HFOV does NOT detect a mis-scaled printout. fx is exactly
    invariant to square size (verified: seven significant figures across a 2.5x
    change in --square), so a badly printed board cannot move it. That also
    means a badly printed board cannot corrupt any BEARING, since
    atan((u - cx)/fx) carries no length either. Square size sets only the
    board-distance scale. To test fx against the physical world you need an
    independent length: camera_check_scale.py, a tape measure.
  * cx, cy near the frame centre. Far off is the fingerprint of the degenerate
    fit above -- the solver pays for a wrong fx by sliding the principal point.

USAGE

    ./camera_calib_report.py                          # score /tmp/calibrationdata.tar.gz
    ./camera_calib_report.py --write                  # ...and install the config
    ./camera_calib_report.py --tarball /tmp/other.tar.gz --size 9x6 --square 0.020

`--write` copies the intrinsics to my_bot/config/c615_640x480.yaml with the
camera name set. It REFUSES if the error is over the gate unless you pass
--force, because an unrecorded bad calibration is worse than none: every bearing
in the semantic layer derives from these four numbers.

Then paste the printed block into records/calibration.md. Numbers go in that
file the moment they are measured -- it is the file whose loss cost the most.
"""

import argparse
import io
import os
import sys
import tarfile

import cv2
import numpy as np
import yaml

GATE_PX = 0.5
C615_HFOV_DEG = 62.0          # vendor spec, reference/hardware-inventory.md


def load_tarball(path, prefix="left"):
    """Return (ost.yaml as dict, [(name, gray image), ...]) from a calibrator tarball."""
    if not os.path.exists(path):
        sys.exit("no such tarball: %s\n"
                 "cameracalibrator writes it when you press SAVE, after CALIBRATE.\n"
                 "If you pressed COMMIT instead, that tries the set_camera_info\n"
                 "service -- which cam2image does not offer -- and nothing is saved."
                 % path)

    info, images = None, []
    with tarfile.open(path, "r:*") as tf:
        for member in tf.getmembers():
            name = os.path.basename(member.name)
            if name == "ost.yaml":
                info = yaml.safe_load(tf.extractfile(member).read())
            elif name.startswith(prefix) and name.endswith((".png", ".pgm")):
                buf = np.frombuffer(tf.extractfile(member).read(), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append((name, img))

    if info is None:
        sys.exit("%s has no ost.yaml -- it is not a cameracalibrator tarball" % path)
    images.sort()
    return info, images


def matrix_from(info, key):
    m = info[key]
    return np.array(m["data"], dtype=np.float64).reshape(m["rows"], m["cols"])


def score(images, K, D, cols, rows, square_m):
    """Per-image RMS reprojection error with K and D held fixed."""
    objp = np.zeros((rows * cols, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    per_image, missed, sq_sum, n_pts, depths = [], [], 0.0, 0, []
    for name, gray in images:
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
        if not found:
            missed.append(name)
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)

        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, D)
        if not ok:
            missed.append(name)
            continue

        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        err = (proj.reshape(-1, 2) - corners.reshape(-1, 2))
        sq = float(np.sum(err ** 2))
        per_image.append((name, float(np.sqrt(sq / len(err))), float(np.max(
            np.linalg.norm(err, axis=1)))))
        depths.append(float(tvec[2]))
        sq_sum += sq
        n_pts += len(err)

    overall = float(np.sqrt(sq_sum / n_pts)) if n_pts else float("nan")
    return overall, per_image, missed, depths


def default_dest():
    """my_bot/config/c615_640x480.yaml in the SOURCE tree.

    realpath, not abspath: run through `ros2 run my_bot`, __file__ is the copy
    under install/my_bot/lib/my_bot/. `colcon build --symlink-install` makes
    that a symlink back to src/, so resolving it lands in the source tree where
    the config belongs and where git can see it. Writing into install/ would
    look like it worked and vanish on the next `colcon build`.
    """
    here = os.path.realpath(__file__)
    pkg = os.path.dirname(os.path.dirname(here))       # .../my_bot/scripts -> my_bot
    dest = os.path.join(pkg, "config", "c615_640x480.yaml")
    if not os.path.isdir(os.path.join(pkg, "config")) or os.sep + "install" + os.sep in dest:
        sys.exit("cannot locate the my_bot source tree from %s.\n"
                 "That happens when the package was built WITHOUT --symlink-install.\n"
                 "Pass the path explicitly:  --dest src/my_bot/config/c615_640x480.yaml"
                 % here)
    return dest


def write_config(info, K, dest, camera_name):
    """Write a camera_info yaml the way camera_info_manager wants to read it."""
    doc = {
        "image_width": int(info["image_width"]),
        "image_height": int(info["image_height"]),
        "camera_name": camera_name,
        "camera_matrix": info["camera_matrix"],
        "distortion_model": info.get("distortion_model", "plumb_bob"),
        "distortion_coefficients": info["distortion_coefficients"],
        "rectification_matrix": info["rectification_matrix"],
        "projection_matrix": info["projection_matrix"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "w") as f:
        f.write("# Logitech C615 intrinsics at %dx%d.\n"
                "# Produced by camera_calibration/cameracalibrator, scored and\n"
                "# installed by my_bot/scripts/camera_calib_report.py.\n"
                "# The method, the date and the reprojection error are in\n"
                "# records/calibration.md. Do not hand-edit these numbers.\n"
                % (doc["image_width"], doc["image_height"]))
        yaml.safe_dump(doc, f, default_flow_style=None, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(
        description="Score a cameracalibrator tarball and install the intrinsics.")
    ap.add_argument("--tarball", default="/tmp/calibrationdata.tar.gz")
    ap.add_argument("--size", default="9x6",
                    help="interior corners, COLSxROWS (default 9x6)")
    ap.add_argument("--square", type=float, default=0.020,
                    help="square size in METRES (default 0.020)")
    ap.add_argument("--write", action="store_true",
                    help="install the intrinsics into --dest")
    ap.add_argument("--dest", default=None,
                    help="default: the my_bot SOURCE tree's config/c615_640x480.yaml")
    ap.add_argument("--camera-name", default="c615")
    ap.add_argument("--force", action="store_true",
                    help="write even when the error is over the %.1f px gate" % GATE_PX)
    args = ap.parse_args()

    cols, rows = (int(v) for v in args.size.lower().split("x"))
    info, images = load_tarball(args.tarball)

    K = matrix_from(info, "camera_matrix")
    D = matrix_from(info, "distortion_coefficients").reshape(-1, 1)
    w, h = int(info["image_width"]), int(info["image_height"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    if not images:
        sys.exit("tarball has no images -- cannot score the reprojection error")

    overall, per_image, missed, depths = score(images, K, D, cols, rows, args.square)
    hfov = 2.0 * np.degrees(np.arctan2(w / 2.0, fx))
    vfov = 2.0 * np.degrees(np.arctan2(h / 2.0, fy))

    print("tarball        %s" % args.tarball)
    print("board          %dx%d interior corners, %.1f mm squares"
          % (cols, rows, args.square * 1000.0))
    print("images         %d used, %d rejected" % (len(per_image), len(missed)))
    print("resolution     %d x %d" % (w, h))
    print()
    print("  fx %10.4f    cx %10.4f" % (fx, cx))
    print("  fy %10.4f    cy %10.4f" % (fy, cy))
    print("  D  %s" % np.ravel(D).tolist())
    print()
    print("  reprojection RMS   %.4f px   %s"
          % (overall, "PASS" if overall < GATE_PX else "FAIL (gate is < %.1f)" % GATE_PX))
    print("  implied FOV        %.1f deg horizontal, %.1f deg vertical" % (hfov, vfov))
    if depths:
        print("  board depth range  %.2f - %.2f m  (ratio %.1fx -- want 2.5x or more)"
              % (min(depths), max(depths), max(depths) / min(depths)))
    print()

    print("per image, worst first:")
    for name, rms, worst in sorted(per_image, key=lambda r: -r[1]):
        flag = "  <-- drop this one and re-run" if rms > 2.0 * overall else ""
        print("  %-16s rms %6.4f px   worst corner %6.4f px%s"
              % (name, rms, worst, flag))
    if missed:
        print("\nno board found in: %s" % ", ".join(missed))
        print("  (harmless here -- these frames were also excluded from the fit)")

    # The checks reprojection error cannot make.
    problems = []
    if (w, h) != (640, 480):
        problems.append(
            "resolution is %dx%d, but the robot runs cam2image at 640x480.\n"
            "     cam2image DEFAULTS to 320x240 -- pass -p width:=640 -p height:=480.\n"
            "     Intrinsics do not transfer across resolutions. Recapture." % (w, h))
    if depths and max(depths) / min(depths) < 2.5:
        problems.append(
            "DEGENERATE CAPTURE: every board sits between %.2f m and %.2f m.\n"
            "     fx and board distance are nearly interchangeable over a narrow depth\n"
            "     range, so the solver can be badly wrong about fx and still fit these\n"
            "     images perfectly -- reprojection error cannot see this. Recapture with\n"
            "     the board swept from filling the frame to a small patch."
            % (min(depths), max(depths)))
    if abs(hfov - C615_HFOV_DEG) > 8.0:
        problems.append(
            "implied HFOV %.1f deg is far from the C615's ~%.0f deg spec.\n"
            "     This does NOT mean the printout is mis-scaled -- fx is invariant to\n"
            "     square size. It means fx itself may be wrong: suspect the depth spread\n"
            "     above, or the focus moving mid-capture. Settle it against a tape\n"
            "     measure:  make calib-scale"
            % (hfov, C615_HFOV_DEG))
    if abs(cx - w / 2.0) > 0.10 * w or abs(cy - h / 2.0) > 0.10 * h:
        problems.append(
            "principal point (%.1f, %.1f) is more than 10%% off the frame centre\n"
            "     (%.1f, %.1f). Usually too few samples near the frame edges."
            % (cx, cy, w / 2.0, h / 2.0))
    if len(per_image) < 8:
        problems.append(
            "only %d usable images. Ten-plus, spread over the whole frame and\n"
            "     through a range of tilts, is what makes fx and the distortion\n"
            "     terms separable." % len(per_image))
    for p in problems:
        print("\n  WARN: %s" % p)

    print("\n--- paste into records/calibration.md -------------------------------")
    print("| | Value |")
    print("|---|---|")
    print("| Resolution | %d x %d |" % (w, h))
    print("| `fx` | **%.4f** |" % fx)
    print("| `fy` | **%.4f** |" % fy)
    print("| `cx` | **%.4f** |" % cx)
    print("| `cy` | **%.4f** |" % cy)
    print("| Distortion coefficients | `%s` |" % np.ravel(D).tolist())
    print("| **Reprojection error** | **%.4f px** (target < %.1f) |" % (overall, GATE_PX))
    print("| Implied HFOV | %.1f deg (C615 spec ~%.0f deg) |" % (hfov, C615_HFOV_DEG))
    print("| Checkerboard | **%dx%d, %.0f mm** |" % (cols, rows, args.square * 1000.0))
    print("| Images used | %d |" % len(per_image))
    if depths:
        print("| Board depth range | %.2f – %.2f m (%.1f×) |"
              % (min(depths), max(depths), max(depths) / min(depths)))
    print("| Method | `cam2image` 640x480 RELIABLE + `cameracalibrator "
          "--no-service-check`, scored by `camera_calib_report.py` |")
    print("--------------------------------------------------------------------")

    if not args.write:
        print("\n(not written -- pass --write to install into config/)")
        return 0 if overall < GATE_PX else 1

    dest = args.dest or default_dest()
    if overall >= GATE_PX and not args.force:
        sys.exit("\nreprojection RMS %.4f px is over the %.1f px gate -- NOT written.\n"
                 "Drop the worst frames and re-run, or recapture. --force overrides."
                 % (overall, GATE_PX))

    write_config(info, K, dest, args.camera_name)
    print("\nwrote %s" % dest)
    print("Now: record the numbers above, delete the semantic node's 554.0 defaults,")
    print("and commit. A missing params file must fail loudly, not guess a focal length.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
