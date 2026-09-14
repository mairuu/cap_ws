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

  sensor offsets (static TF, from the URDF via tf2 -- decision D-10)
  The BEARING is the camera's: a ray from the camera mount's origin at
  azimuth (about its x-axis) plus the mount's yaw in base_link. The RANGE
  is the lidar's: a circle of radius r around laser_frame's origin. The
  object is where the ray meets the circle. CameraExtrinsics carries both
  origins: (dx, dy, yaw) for the camera and (lidar_dx, lidar_dy) for the
  lidar.

  Why not just "camera origin + r along the ray" (the June code): the
  range is not measured from the camera. With the camera at (0.05, -0.03)
  and the lidar at (-0.034, 0) that put every landmark ~9 cm off.
  Why not "lidar origin + r along the ray": the camera sits 3 cm beside
  the lidar's axis, so an object dead ahead of the camera is at 1.7 deg
  from the lidar at 1 m -- more than one lidar ray (1.03 deg). The
  ray/circle intersection is exact for both offsets and costs one sqrt.

Pipeline
--------
  RangeEstimate (azimuth in camera frame, range in metres)
        ↓
  mirror + rotate by camera mount yaw  →  bearing in base_link frame
        ↓
  camera ray ∩ lidar range circle      →  point in base_link frame
        ↓
  rotate + translate by robot pose     →  point in map frame

2D model: z, the camera's -3 deg pitch and lens distortion are dropped.
At 3 m the pitch biases the bearing by <= 2 cm at the frame corners and
distortion by <= 1.3 cm at the edge -- both under one lidar ray (1.03 deg).
Documented limitation, not modelled.
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
    Static sensor geometry in base_link. Filled from TF by the node (D-10),
    never from a params file.

    dx, dy  – the CAMERA mount's origin in base_link (metres):
               dx > 0 forward, dy > 0 left (REP-103; a camera mounted on
               the robot's right has dy < 0).
    yaw     – the camera MOUNT's pan relative to robot forward (radians)
               0.0 = camera points straight ahead
               positive = camera panned left (CCW, REP-103)
               Take it from camera_link, the x-forward mount frame. The
               optical frame's yaw is -90 deg and would rotate every landmark
               by a right angle.
    lidar_dx, lidar_dy – the LIDAR's origin in base_link (metres), where the
               range is measured from. None = same as the camera (the June
               behaviour, kept for old callers and tests).

    Note: we ignore z (height) because the lidar is 2D and we produce
    a 2D floor-plan map.
    """
    dx: float = 0.0
    dy: float = 0.0
    yaw: float = 0.0
    lidar_dx: Optional[float] = None
    lidar_dy: Optional[float] = None

    @property
    def range_origin(self) -> tuple[float, float]:
        return (
            self.dx if self.lidar_dx is None else self.lidar_dx,
            self.dy if self.lidar_dy is None else self.lidar_dy,
        )


@dataclass
class WorldPoint:
    """
    A detected object's estimated position in the map frame.
    """
    x: float                    # map frame, metres
    y: float                    # map frame, metres
    range_m: float              # lidar range used (for quality assessment)
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

        # ------------------------------------------------------------------
        # Step 3: the object is where the CAMERA's ray meets the LIDAR's
        # range circle, in base_link.
        #   ray:    P = C + t * d,  d = (cos az, sin az),  t >= 0
        #   circle: |P - L| = r
        #   => t^2 + 2 t (m . d) + |m|^2 - r^2 = 0,  m = C - L
        # Take the far root: the near one is the ray's entry into the circle
        # behind/beside the camera. No real root means the range is shorter
        # than the sensor offset geometry allows -- reject.
        # With C == L this reduces exactly to C + r * d.
        # ------------------------------------------------------------------
        dx_ray = math.cos(az_base_ros)
        dy_ray = math.sin(az_base_ros)
        cx, cy = self.extrinsics.dx, self.extrinsics.dy
        lx, ly = self.extrinsics.range_origin
        mx, my = cx - lx, cy - ly
        m_dot_d = mx * dx_ray + my * dy_ray
        disc = m_dot_d * m_dot_d - (mx * mx + my * my) + r * r
        if disc < 0.0:
            return WorldPoint(
                x=float("nan"), y=float("nan"),
                range_m=r,
                azimuth_rad=estimate.azimuth_center,
                n_lidar_returns=estimate.n_returns,
                valid=False,
            )
        t = -m_dot_d + math.sqrt(disc)
        x_in_base = cx + t * dx_ray
        y_in_base = cy + t * dy_ray

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
