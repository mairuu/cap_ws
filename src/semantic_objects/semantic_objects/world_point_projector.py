"""
world_point_projector.py
------------------------
Step 3: Convert a RangeEstimate into a world-frame (x, y) position.

No ROS2 imports — pure Python geometry so it can be unit-tested standalone.
The ROS2 node will call this after pulling the pose from TF.

Coordinate frames
-----------------

  map frame (world)
  ┌─────────────────────────────────┐
  │  x →                           │
  │  y ↑                           │
  │                                 │
  │        robot                    │
  │        ● ──── heading (yaw)     │
  │                                 │
  └─────────────────────────────────┘

  base_link frame (robot body)
  ┌──────────────┐
  │  x → forward │
  │  y ← left    │
  └──────────────┘

  camera frame offset (static TF, measured physically)
  camera is mounted at some (dx, dy) offset from base_link origin,
  and may have a yaw offset (pan angle) relative to the robot's forward axis.

Pipeline
--------
  RangeEstimate (azimuth in camera frame, range in metres)
        ↓
  rotate by camera yaw offset  →  azimuth in base_link frame
        ↓
  polar → Cartesian in base_link frame
        ↓
  translate by camera position offset  →  point in base_link frame
        ↓
  rotate + translate by robot pose  →  point in map frame
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .lidar_range_extractor import RangeEstimate


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Pose2D:
    """
    Robot pose in the map frame.
    Sourced from TF: map → base_link transform.
    """
    x: float        # metres
    y: float        # metres
    yaw: float      # radians, CCW from map +x axis


@dataclass
class CameraExtrinsics:
    """
    Static transform: where the camera sits relative to base_link.
    Measure physically from your robot's CAD or with a ruler.

    dx, dy  – camera origin offset from base_link origin (metres)
               dx > 0 → camera is in front of robot centre
               dy > 0 → camera is to the left

    yaw     – camera pan angle relative to robot forward axis (radians)
               0.0 = camera points straight ahead (most common)
               positive = camera panned left

    Note: we ignore z (height) because the lidar is 2D and we produce
    a 2D floor-plan map. If you add a depth camera later, add dz here.
    """
    dx: float = 0.0
    dy: float = 0.0
    yaw: float = 0.0


@dataclass
class WorldPoint:
    """
    A detected object's estimated position in the map frame.
    """
    x: float                    # map frame, metres
    y: float                    # map frame, metres
    range_m: float              # range from camera (for quality assessment)
    azimuth_rad: float          # azimuth in camera frame (for debugging)
    n_lidar_returns: int        # how many lidar rays backed this estimate
    valid: bool                 # False if range was invalid


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------

class WorldPointProjector:
    """
    Projects a RangeEstimate into a 2D world coordinate.

    Parameters
    ----------
    extrinsics : CameraExtrinsics
        Physical offset of the camera from base_link.
        If camera is centred and pointing forward, use the default (all zeros).
    max_range : float
        Discard projections beyond this range (metres).
        Objects far away have high angular uncertainty.
        Default: 5.0 m (reasonable for indoor lidar + webcam).
    min_range : float
        Discard projections closer than this (metres).
        Very close detections are often partially visible / noisy.
        Default: 0.15 m.
    """

    def __init__(
        self,
        extrinsics: Optional[CameraExtrinsics] = None,
        max_range: float = 5.0,
        min_range: float = 0.15,
    ):
        self.extrinsics = extrinsics or CameraExtrinsics()
        self.max_range = max_range
        self.min_range = min_range

    def project(self, estimate: RangeEstimate, robot_pose: Pose2D) -> WorldPoint:
        """
        Convert a RangeEstimate + robot pose → WorldPoint in map frame.

        Returns WorldPoint(valid=False) if:
        - estimate.valid is False
        - range is outside [min_range, max_range]
        """
        if not estimate.valid:
            return WorldPoint(
                x=float("nan"), y=float("nan"),
                range_m=estimate.range_m,
                azimuth_rad=estimate.azimuth_center,
                n_lidar_returns=estimate.n_returns,
                valid=False,
            )

        r = estimate.range_m

        if r < self.min_range or r > self.max_range:
            return WorldPoint(
                x=float("nan"), y=float("nan"),
                range_m=r,
                azimuth_rad=estimate.azimuth_center,
                n_lidar_returns=estimate.n_returns,
                valid=False,
            )

        # ------------------------------------------------------------------
        # Step 1+2: azimuth in camera frame → az in base_link (ROS) frame,
        # then polar → Cartesian.
        #
        # Two conventions:
        #   az_camera  — CW from camera forward (right is positive, image convention)
        #   camera_yaw — CCW rotation of camera body in base_link (ROS TF convention)
        #                +π/6 = camera panned 30° to the LEFT
        #
        # az in ROS base_link (CCW from robot forward, left = +y):
        #   az_base_ros = -az_camera + camera_yaw
        #
        # Checks:
        #   yaw=0, az=0    → dead ahead in both frames: 0  ✓
        #   yaw=+π/6, az=0 → camera points left, object is left: +π/6  ✓
        #   yaw=0, az=+π/4 → object is right in image → -π/4 (right = -y in ROS)  ✓
        # ------------------------------------------------------------------
        az_base_ros = -estimate.azimuth_center + self.extrinsics.yaw

        x_cam_in_base = r * math.cos(az_base_ros)
        y_cam_in_base = r * math.sin(az_base_ros)

        # ------------------------------------------------------------------
        # Step 3: apply camera position offset in base_link frame
        # ------------------------------------------------------------------
        x_in_base = x_cam_in_base + self.extrinsics.dx
        y_in_base = y_cam_in_base + self.extrinsics.dy

        # ------------------------------------------------------------------
        # Step 4: transform base_link → map frame using robot pose
        # Rotation by yaw, then translation.
        #
        #   [x_map]   [cos(yaw)  -sin(yaw)] [x_base]   [robot_x]
        #   [y_map] = [sin(yaw)   cos(yaw)] [y_base] + [robot_y]
        # ------------------------------------------------------------------
        cos_yaw = math.cos(robot_pose.yaw)
        sin_yaw = math.sin(robot_pose.yaw)

        x_map = cos_yaw * x_in_base - sin_yaw * y_in_base + robot_pose.x
        y_map = sin_yaw * x_in_base + cos_yaw * y_in_base + robot_pose.y

        return WorldPoint(
            x=x_map,
            y=y_map,
            range_m=r,
            azimuth_rad=estimate.azimuth_center,
            n_lidar_returns=estimate.n_returns,
            valid=True,
        )

    def project_point_in_base(
        self, estimate: RangeEstimate
    ) -> tuple[float, float] | None:
        """
        Convenience: return (x, y) in base_link frame only (no robot pose).
        Useful for debugging the extrinsics in isolation.
        Returns None if estimate is invalid.
        """
        dummy_pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
        wp = self.project(estimate, dummy_pose)
        if not wp.valid:
            return None
        return (wp.x, wp.y)
