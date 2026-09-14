"""
lidar_range_extractor.py
------------------------
Given a bounding box (pixel coords) and a lidar LaserScan,
return the estimated range to the object.

Standalone: no ROS2 imports. Takes plain Python data structures
so it can be unit-tested without a running ROS2 system.

Coordinate convention
---------------------
Camera:
    x-right, y-down, z-forward (standard OpenCV)
    (u, v) = (column, row), origin top-left

Lidar:
    Angles measured CCW from the robot's forward axis (x-forward).
    LaserScan angles: angle_min → angle_max, step = angle_increment.
    Index i → angle = angle_min + i * angle_increment.

Camera ↔ Lidar alignment assumption:
    Both are mounted facing the same direction (forward).
    The camera's horizontal centre ray aligns with lidar angle = 0.
    A static TF offset (x, y, z) is handled upstream in the ROS node;
    here we only deal with angular geometry.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures (mirror ROS2 types so we can test without ROS)
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    """Minimal pinhole camera model parameters."""
    fx: float          # focal length in pixels (horizontal)
    fy: float          # focal length in pixels (vertical)
    cx: float          # principal point x (pixels)
    cy: float          # principal point y (pixels)
    width: int         # image width  (pixels)
    height: int        # image height (pixels)

    @property
    def hfov(self) -> float:
        """Horizontal field of view in radians."""
        return 2.0 * math.atan2(self.width / 2.0, self.fx)

    def pixel_to_azimuth(self, u: float) -> float:
        """
        Convert a horizontal pixel coordinate to an azimuth angle (radians).

        u = cx  → 0.0   (straight ahead)
        u < cx  → negative (left)
        u > cx  → positive (right)

        Uses the pinhole model: azimuth = atan2(u - cx, fx)
        This is more accurate than the linear approximation (u/width * hfov)
        especially near the image edges.
        """
        return math.atan2(u - self.cx, self.fx)


@dataclass
class BoundingBox:
    """
    Axis-aligned bounding box in pixel space.
    (x1, y1) = top-left corner
    (x2, y2) = bottom-right corner
    """
    x1: float
    y1: float
    x2: float
    y2: float
    class_label: str = ""
    confidence: float = 1.0

    @property
    def center_u(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_v(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width_px(self) -> float:
        return self.x2 - self.x1

    @property
    def height_px(self) -> float:
        return self.y2 - self.y1


@dataclass
class LaserScanData:
    """
    Plain-Python mirror of sensor_msgs/LaserScan (relevant fields only).
    All angles in radians, ranges in metres.
    """
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: list[float]          # len = (angle_max - angle_min) / angle_increment

    def angle_at(self, index: int) -> float:
        return self.angle_min + index * self.angle_increment

    def index_at(self, angle: float) -> Optional[int]:
        """Return the scan index closest to `angle`, or None if out of range."""
        if angle < self.angle_min or angle > self.angle_max:
            return None
        raw = (angle - self.angle_min) / self.angle_increment
        return int(round(raw))


@dataclass
class RangeEstimate:
    """Result returned by extract_range()."""
    range_m: float              # estimated distance to object, metres
    azimuth_center: float       # azimuth of bounding box centre, radians
    azimuth_left: float         # azimuth of left edge of bounding box
    azimuth_right: float        # azimuth of right edge of bounding box
    n_returns: int              # number of valid lidar returns used
    method: str                 # "median" | "min" | "none"
    valid: bool                 # False if no usable returns found


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class LidarRangeExtractor:
    """
    Projects a 2D bounding box onto a 2D lidar scan and returns
    a range estimate to the detected object.

    Parameters
    ----------
    camera : CameraIntrinsics
        Calibrated intrinsics for the camera providing detections.
    range_method : str
        How to aggregate multiple lidar returns within the angular window.
        "median"  – robust to outliers (default, recommended)
        "min"     – closest return; useful if object is partially occluded
        "mean"    – simple average; sensitive to outliers
    angular_padding : float
        Extra angular margin (radians) added to each side of the
        bounding box window. Compensates for small extrinsic misalignments.
        Default 0.0 (no padding).
    min_returns : int
        Minimum valid returns required for a confident estimate.
        If fewer are found, RangeEstimate.valid is still True but
        n_returns will be low — caller can decide how to handle.
    """

    def __init__(
        self,
        camera: CameraIntrinsics,
        range_method: str = "median",
        angular_padding: float = 0.0,
        min_returns: int = 1,
    ):
        if range_method not in ("median", "min", "mean"):
            raise ValueError(f"Unknown range_method: {range_method!r}")
        self.camera = camera
        self.range_method = range_method
        self.angular_padding = angular_padding
        self.min_returns = min_returns

    def extract_range(
        self,
        bbox: BoundingBox,
        scan: LaserScanData,
    ) -> RangeEstimate:
        """
        Main entry point.

        Steps
        -----
        1. Convert bbox left/center/right pixel columns → azimuths.
        2. Collect lidar returns whose angle falls within [left, right] window.
        3. Filter out-of-range returns (inf, nan, below range_min, above range_max).
        4. Aggregate with chosen method.
        5. Return RangeEstimate.
        """
        az_center = self.camera.pixel_to_azimuth(bbox.center_u)
        az_left   = self.camera.pixel_to_azimuth(bbox.x1) - self.angular_padding
        az_right  = self.camera.pixel_to_azimuth(bbox.x2) + self.angular_padding

        # Ensure left < right (should always hold for valid bboxes)
        if az_left > az_right:
            az_left, az_right = az_right, az_left

        # Collect valid returns within the angular window
        valid_ranges = self._collect_returns(scan, az_left, az_right)

        if not valid_ranges:
            return RangeEstimate(
                range_m=float("inf"),
                azimuth_center=az_center,
                azimuth_left=az_left,
                azimuth_right=az_right,
                n_returns=0,
                method="none",
                valid=False,
            )

        range_m = self._aggregate(valid_ranges)

        return RangeEstimate(
            range_m=range_m,
            azimuth_center=az_center,
            azimuth_left=az_left,
            azimuth_right=az_right,
            n_returns=len(valid_ranges),
            method=self.range_method,
            valid=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_returns(
        self,
        scan: LaserScanData,
        az_left: float,
        az_right: float,
    ) -> list[float]:
        """
        Walk through every scan ray and collect those whose angle
        falls within [az_left, az_right].

        Returns a list of valid (finite, in-range) distances.
        """
        valid: list[float] = []

        # Clamp window to the scan's angular coverage
        search_left  = max(az_left,  scan.angle_min)
        search_right = min(az_right, scan.angle_max)

        if search_left > search_right:
            # Window entirely outside scan field of view
            return valid

        # Convert window bounds to scan indices
        idx_start = max(0, self._angle_to_index(scan, search_left))
        idx_end   = min(len(scan.ranges) - 1, self._angle_to_index(scan, search_right))

        for i in range(idx_start, idx_end + 1):
            r = scan.ranges[i]
            if self._is_valid_range(r, scan):
                valid.append(r)

        return valid

    @staticmethod
    def _angle_to_index(scan: LaserScanData, angle: float) -> int:
        """Floor index for a given angle (does not clamp to valid range)."""
        return int((angle - scan.angle_min) / scan.angle_increment)

    @staticmethod
    def _is_valid_range(r: float, scan: LaserScanData) -> bool:
        """True if r is a finite, in-range lidar return."""
        return (
            math.isfinite(r)
            and r >= scan.range_min
            and r <= scan.range_max
        )

    def _aggregate(self, ranges: list[float]) -> float:
        if self.range_method == "median":
            return statistics.median(ranges)
        elif self.range_method == "min":
            return min(ranges)
        elif self.range_method == "mean":
            return statistics.mean(ranges)
        raise RuntimeError("Unreachable")
