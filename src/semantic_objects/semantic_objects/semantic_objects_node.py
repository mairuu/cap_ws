"""
semantic_objects_node.py
------------------------
ROS2 node: semantic_objects

Subscribes to:
  /scan              (sensor_msgs/LaserScan)
  /detections        (vision_msgs/Detection2DArray)

Publishes:
  /semantic_landmarks  (std_msgs/String — JSON for web UI)
  /semantic_markers    (visualization_msgs/MarkerArray — RViz overlay)

TF lookups:
  map → base_link    (robot pose in world)
  base_link → camera (camera position on robot — static)

Parameters (set via ROS2 params or a YAML file):
  camera/fx, fy, cx, cy, width, height  — camera intrinsics
  camera/dx, dy, yaw                    — camera extrinsics (offset from base_link)
  detection/min_confidence              — YOLO confidence gate
  detection/range_method                — "min" | "median" | "mean"
  detection/angular_padding             — extra angular margin per bbox (rad)
  landmark/merge_radius                 — association distance threshold (m)
  landmark/ema_alpha                    — position smoothing factor
  landmark/min_seen_to_publish          — observations before landmark is confirmed
  landmark/stale_timeout                — seconds before landmark is marked stale
  landmark/persist_path                 — path to JSON file (empty = no persistence)
  publish/rate_hz                       — how often to publish landmarks (Hz)
  tf/map_frame                          — name of map TF frame
  tf/base_frame                         — name of robot body TF frame
  tf/camera_frame                       — name of camera TF frame
  tf/lookup_timeout                     — max wait for TF transform (seconds)
  sync/slop                             — ApproximateTimeSynchronizer tolerance (s)
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Duration

import tf2_ros
from tf2_ros import TransformException

from message_filters import ApproximateTimeSynchronizer, Subscriber

from sensor_msgs.msg import LaserScan
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from .lidar_range_extractor import CameraIntrinsics, LidarRangeExtractor
from .world_point_projector import CameraExtrinsics, WorldPointProjector
from .landmark_store import LandmarkStore
from .ros_bridge import (
    laserscan_to_data,
    detections_to_bboxes,
    transform_to_pose2d,
    landmarks_to_marker_array,
    landmarks_to_json,
)


# ---------------------------------------------------------------------------
# QoS profiles
# ---------------------------------------------------------------------------

# Sensor data: best-effort, volatile (drop old messages, don't queue)
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    depth=5,
)

# Latched-style for landmark output: transient local so new subscribers
# immediately receive the last published state
LANDMARK_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SemanticObjectsNode(Node):

    def __init__(self):
        super().__init__("semantic_objects")
        self._declare_parameters()

        # --- Build pipeline components ---
        self._extractor  = self._build_extractor()
        self._projector  = self._build_projector()
        self._store      = self._build_store()

        # --- TF ---
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # --- Subscribers with time-sync ---
        self._setup_subscribers()

        # --- Publishers ---
        self._pub_json    = self.create_publisher(String,      "/semantic_landmarks", LANDMARK_QOS)
        self._pub_markers = self.create_publisher(MarkerArray, "/semantic_markers",   LANDMARK_QOS)

        # --- Periodic timers ---
        rate_hz       = self.get_parameter("publish/rate_hz").value
        stale_check_s = max(10.0, self.get_parameter("landmark/stale_timeout").value / 10)

        self._publish_timer = self.create_timer(1.0 / rate_hz,   self._publish_landmarks)
        self._stale_timer   = self.create_timer(stale_check_s,   self._check_stale)

        # --- Stats ---
        self._total_detections = 0
        self._total_fused      = 0

        self.get_logger().info(
            f"semantic_objects ready  "
            f"merge_radius={self.get_parameter('landmark/merge_radius').value}m  "
            f"ema_alpha={self.get_parameter('landmark/ema_alpha').value}  "
            f"persist={self.get_parameter('landmark/persist_path').value or 'off'}"
        )

    # ------------------------------------------------------------------
    # Parameter declaration
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        self.declare_parameters(
            namespace="",
            parameters=[
                # Camera intrinsics (must be calibrated!)
                ("camera.fx",     554.0),
                ("camera.fy",     554.0),
                ("camera.cx",     320.0),
                ("camera.cy",     240.0),
                ("camera.width",  640),
                ("camera.height", 480),

                # Camera extrinsics (measure physically from robot)
                ("camera.dx",  0.0),   # metres forward from base_link
                ("camera.dy",  0.0),   # metres left from base_link
                ("camera.yaw", 0.0),   # radians CCW from robot forward

                # Detection pipeline
                ("detection.min_confidence", 0.5),
                ("detection.range_method",   "min"),
                ("detection.angular_padding", 0.0),
                ("detection.max_range",       5.0),
                ("detection.min_range",       0.15),

                # Landmark store
                ("landmark.merge_radius",         0.5),
                ("landmark.ema_alpha",            0.3),
                ("landmark.min_seen_to_publish",  2),
                ("landmark.stale_timeout",       300.0),
                ("landmark.persist_path",         ""),

                # Publishing
                ("publish.rate_hz", 2.0),

                # TF frames
                ("tf.map_frame",    "map"),
                ("tf.base_frame",   "base_link"),
                ("tf.camera_frame", "camera_link"),
                ("tf.lookup_timeout", 0.1),

                # Time sync
                ("sync.slop", 0.1),
            ],
        )

    # ------------------------------------------------------------------
    # Component builders
    # ------------------------------------------------------------------

    def _build_extractor(self) -> LidarRangeExtractor:
        p = self.get_parameters_by_prefix("camera")
        intrinsics = CameraIntrinsics(
            fx=self.get_parameter("camera.fx").value,
            fy=self.get_parameter("camera.fy").value,
            cx=self.get_parameter("camera.cx").value,
            cy=self.get_parameter("camera.cy").value,
            width=self.get_parameter("camera.width").value,
            height=self.get_parameter("camera.height").value,
        )
        return LidarRangeExtractor(
            camera=intrinsics,
            range_method=self.get_parameter("detection.range_method").value,
            angular_padding=self.get_parameter("detection.angular_padding").value,
        )

    def _build_projector(self) -> WorldPointProjector:
        extrinsics = CameraExtrinsics(
            dx=self.get_parameter("camera.dx").value,
            dy=self.get_parameter("camera.dy").value,
            yaw=self.get_parameter("camera.yaw").value,
        )
        return WorldPointProjector(
            extrinsics=extrinsics,
            max_range=self.get_parameter("detection.max_range").value,
            min_range=self.get_parameter("detection.min_range").value,
        )

    def _build_store(self) -> LandmarkStore:
        path_str = self.get_parameter("landmark.persist_path").value
        return LandmarkStore(
            merge_radius=self.get_parameter("landmark.merge_radius").value,
            ema_alpha=self.get_parameter("landmark.ema_alpha").value,
            min_seen_to_publish=self.get_parameter("landmark.min_seen_to_publish").value,
            stale_timeout=self.get_parameter("landmark.stale_timeout").value,
            persist_path=Path(path_str) if path_str else None,
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _setup_subscribers(self):
        """
        ApproximateTimeSynchronizer pairs each LaserScan with the nearest
        Detection2DArray within `slop` seconds.

        Without sync, we'd project bounding boxes against a lidar scan
        that was acquired at a different robot position — causing position
        errors proportional to robot speed × time delta.
        """
        slop = self.get_parameter("sync.slop").value

        self._scan_sub = Subscriber(self, LaserScan, "/scan", qos_profile=SENSOR_QOS)
        self._det_sub  = Subscriber(self, Detection2DArray, "/detections", qos_profile=SENSOR_QOS)

        self._sync = ApproximateTimeSynchronizer(
            [self._scan_sub, self._det_sub],
            queue_size=10,
            slop=slop,
        )
        self._sync.registerCallback(self._on_synced)

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def _on_synced(self, scan_msg: LaserScan, det_msg: Detection2DArray):
        """
        Called when a LaserScan and Detection2DArray arrive within `slop` seconds.

        Pipeline:
          1. Get robot pose from TF.
          2. Convert ROS messages → pure-Python types.
          3. For each bounding box:
             a. Extract lidar range.
             b. Project to world point.
             c. Fuse into landmark store.
          4. Log summary.
        """
        # Step 1: TF lookup — robot pose in map frame
        robot_pose = self._lookup_pose()
        if robot_pose is None:
            return   # TF not yet available; skip this frame

        # Step 2: convert messages
        scan  = laserscan_to_data(scan_msg)
        boxes = detections_to_bboxes(
            det_msg,
            min_confidence=self.get_parameter("detection.min_confidence").value,
        )

        if not boxes:
            return

        # Step 3: process each detection
        fused_this_frame = 0

        for bbox in boxes:
            # a. Lidar range extraction
            estimate = self._extractor.extract_range(bbox, scan)
            if not estimate.valid:
                self.get_logger().debug(
                    f"No lidar returns for {bbox.class_label!r} "
                    f"bbox=({bbox.x1:.0f},{bbox.y1:.0f},{bbox.x2:.0f},{bbox.y2:.0f})"
                )
                continue

            # b. Project to world coordinates
            world_pt = self._projector.project(estimate, robot_pose)
            if not world_pt.valid:
                self.get_logger().debug(
                    f"Range {estimate.range_m:.2f}m out of gate for {bbox.class_label!r}"
                )
                continue

            # c. Fuse into landmark store
            result = self._store.observe_with_confidence(
                world_pt,
                bbox.class_label,
                bbox.confidence,
            )

            action = "NEW" if result.created else f"UPD d={result.distance:.2f}m"
            self.get_logger().debug(
                f"{action}  {bbox.class_label!r}  "
                f"world=({world_pt.x:.2f},{world_pt.y:.2f})  "
                f"range={estimate.range_m:.2f}m  n={estimate.n_returns}"
            )
            fused_this_frame += 1

        self._total_detections += len(boxes)
        self._total_fused      += fused_this_frame

        if fused_this_frame:
            confirmed = len(self._store.confirmed_landmarks())
            self.get_logger().info(
                f"Fused {fused_this_frame}/{len(boxes)} detections  "
                f"landmarks={len(self._store)} confirmed={confirmed}  "
                f"total_fused={self._total_fused}"
            )

    # ------------------------------------------------------------------
    # TF helper
    # ------------------------------------------------------------------

    def _lookup_pose(self):
        """
        Look up the current robot pose (map → base_link transform).
        Returns Pose2D or None if TF is unavailable.
        """
        map_frame    = self.get_parameter("tf.map_frame").value
        base_frame   = self.get_parameter("tf.base_frame").value
        timeout_s    = self.get_parameter("tf.lookup_timeout").value

        try:
            tf = self._tf_buffer.lookup_transform(
                map_frame,
                base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout_s),
            )
            return transform_to_pose2d(tf)

        except TransformException as e:
            self.get_logger().warn(
                f"TF lookup {map_frame}→{base_frame} failed: {e}",
                throttle_duration_sec=5.0,
            )
            return None

    # ------------------------------------------------------------------
    # Periodic publishers
    # ------------------------------------------------------------------

    def _publish_landmarks(self):
        """Publish confirmed landmarks as JSON + RViz markers."""
        confirmed = self._store.confirmed_landmarks()

        if not confirmed:
            return

        # JSON for web UI
        self._pub_json.publish(landmarks_to_json(confirmed))

        # RViz MarkerArray
        map_frame = self.get_parameter("tf.map_frame").value
        self._pub_markers.publish(
            landmarks_to_marker_array(confirmed, frame_id=map_frame)
        )

    def _check_stale(self):
        """Mark landmarks not seen recently as stale."""
        newly_stale = self._store.mark_stale()
        for lid in newly_stale:
            lm = self._store.get(lid)
            if lm:
                self.get_logger().info(
                    f"Landmark stale: {lm.class_label!r} at "
                    f"({lm.x:.2f},{lm.y:.2f})  id={lid[:8]}"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = SemanticObjectsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
