"""
ros_bridge.py
-------------
Converts between ROS2 message types and our pure-Python data structures.

Kept in a separate file so the core logic (extractor, projector, store)
stays ROS2-free and testable. The node imports this; tests do not.

Supported conversions
---------------------
  sensor_msgs/LaserScan         → LaserScanData
  vision_msgs/Detection2DArray  → list[BoundingBox]
  geometry_msgs/TransformStamped → Pose2D          (yaw extracted from quaternion)
  list[SemanticLandmark]        → visualization_msgs/MarkerArray  (RViz overlay)
  list[SemanticLandmark]        → std_msgs/String  (JSON for web UI)
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

# ROS2 message types — imported lazily so this module can still be read
# by non-ROS tooling (linters, docs) without a ROS2 install.
try:
    from sensor_msgs.msg import LaserScan
    from vision_msgs.msg import Detection2DArray
    from geometry_msgs.msg import TransformStamped
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import String, ColorRGBA
    from builtin_interfaces.msg import Duration
    import rclpy.time
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False

from .lidar_range_extractor import BoundingBox, LaserScanData
from .landmark_store import SemanticLandmark
from .world_point_projector import Pose2D


# ---------------------------------------------------------------------------
# LaserScan
# ---------------------------------------------------------------------------

def laserscan_to_data(msg: "LaserScan") -> LaserScanData:
    """Convert sensor_msgs/LaserScan → LaserScanData."""
    return LaserScanData(
        angle_min=msg.angle_min,
        angle_max=msg.angle_max,
        angle_increment=msg.angle_increment,
        range_min=msg.range_min,
        range_max=msg.range_max,
        ranges=list(msg.ranges),
    )


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------

def detections_to_bboxes(
    msg: "Detection2DArray",
    min_confidence: float = 0.5,
) -> list[BoundingBox]:
    """
    Convert vision_msgs/Detection2DArray → list[BoundingBox].

    Filters out detections below min_confidence.
    Assumes the bounding box centre + size encoding (standard for vision_msgs).

    vision_msgs/BoundingBox2D:
        center.position.x, center.position.y  (centre pixel)
        size_x, size_y                         (width, height in pixels)

    The best hypothesis (highest score) in each detection is used for
    class_label and confidence.
    """
    boxes: list[BoundingBox] = []

    for det in msg.detections:
        # Pick the highest-scoring hypothesis
        if not det.results:
            continue
        best = max(det.results, key=lambda r: r.hypothesis.score)
        confidence = best.hypothesis.score

        if confidence < min_confidence:
            continue

        # vision_msgs uses centre + size
        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        hw = det.bbox.size_x / 2.0
        hh = det.bbox.size_y / 2.0

        boxes.append(BoundingBox(
            x1=cx - hw,
            y1=cy - hh,
            x2=cx + hw,
            y2=cy + hh,
            class_label=best.hypothesis.class_id,
            confidence=confidence,
        ))

    return boxes


# ---------------------------------------------------------------------------
# TF pose extraction
# ---------------------------------------------------------------------------

def transform_to_pose2d(tf: "TransformStamped") -> Pose2D:
    """
    Extract a Pose2D from a geometry_msgs/TransformStamped.

    Converts the quaternion rotation to a yaw angle (rotation around Z).
    Ignores pitch and roll — valid for a ground robot on a flat floor.

    Quaternion → yaw:
        yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    """
    t = tf.transform.translation
    q = tf.transform.rotation

    # Standard quaternion-to-yaw formula
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )

    return Pose2D(x=t.x, y=t.y, yaw=yaw)


# ---------------------------------------------------------------------------
# Landmark → RViz MarkerArray
# ---------------------------------------------------------------------------

# Colour palette by class — extend as needed
_CLASS_COLOURS: dict[str, tuple[float, float, float]] = {
    "chair":   (0.2, 0.6, 1.0),   # blue
    "bottle":  (0.2, 0.9, 0.3),   # green
    "couch":   (0.9, 0.5, 0.1),   # orange
    "person":  (1.0, 0.2, 0.2),   # red
    "cup":     (0.8, 0.8, 0.2),   # yellow
    "laptop":  (0.6, 0.2, 0.9),   # purple
    "default": (0.7, 0.7, 0.7),   # grey
}

# Physical size hints per class (metres) for marker sphere radius
_CLASS_RADIUS: dict[str, float] = {
    "chair":  0.25,
    "couch":  0.45,
    "person": 0.25,
    "bottle": 0.08,
    "cup":    0.06,
    "default": 0.15,
}


def landmarks_to_marker_array(
    landmarks: list[SemanticLandmark],
    frame_id: str = "map",
    z_height: float = 0.5,
    stale_alpha: float = 0.3,
) -> "MarkerArray":
    """
    Convert a list of SemanticLandmark → visualization_msgs/MarkerArray.

    Each landmark gets two markers:
      - A sphere at its (x, y) position.
      - A text label floating above it.

    Stale landmarks are rendered semi-transparent.

    Parameters
    ----------
    frame_id   : TF frame the positions are in (should be "map").
    z_height   : Height at which to render markers above the floor (metres).
    stale_alpha: Alpha of stale landmark markers (0 = invisible, 1 = solid).
    """
    array = MarkerArray()
    now_stamp = rclpy.time.Time().to_msg()

    for i, lm in enumerate(landmarks):
        rgb = _CLASS_COLOURS.get(lm.class_label, _CLASS_COLOURS["default"])
        radius = _CLASS_RADIUS.get(lm.class_label, _CLASS_RADIUS["default"])
        alpha = stale_alpha if lm.stale else 1.0

        # --- Sphere marker ---
        sphere = Marker()
        sphere.header.frame_id = frame_id
        sphere.header.stamp = now_stamp
        sphere.ns = "semantic_landmarks"
        sphere.id = i * 2          # even IDs for spheres
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD

        sphere.pose.position.x = lm.x
        sphere.pose.position.y = lm.y
        sphere.pose.position.z = z_height
        sphere.pose.orientation.w = 1.0

        sphere.scale.x = radius * 2
        sphere.scale.y = radius * 2
        sphere.scale.z = radius * 2

        sphere.color.r = rgb[0]
        sphere.color.g = rgb[1]
        sphere.color.b = rgb[2]
        sphere.color.a = alpha

        # Markers persist until explicitly deleted (lifetime = 0)
        sphere.lifetime = Duration(sec=0, nanosec=0)

        array.markers.append(sphere)

        # --- Text label marker ---
        label = Marker()
        label.header.frame_id = frame_id
        label.header.stamp = now_stamp
        label.ns = "semantic_labels"
        label.id = i * 2 + 1      # odd IDs for labels
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD

        label.pose.position.x = lm.x
        label.pose.position.y = lm.y
        label.pose.position.z = z_height + radius + 0.1  # float above sphere

        label.scale.z = 0.18       # text height in metres

        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = alpha

        stale_tag = " [stale]" if lm.stale else ""
        label.text = f"{lm.class_label}{stale_tag}\n×{lm.seen_count}"

        label.lifetime = Duration(sec=0, nanosec=0)

        array.markers.append(label)

    return array


# ---------------------------------------------------------------------------
# Landmark → JSON string (for web UI)
# ---------------------------------------------------------------------------

def landmarks_to_json_str(landmarks: list[SemanticLandmark]) -> str:
    """
    Serialise confirmed landmarks as a plain JSON string.
    Pure Python — no ROS2 dependency.

    Output schema:
    {
      "landmarks": [
        { "id": "...", "class_label": "chair", "x": 1.23, "y": 4.56,
          "confidence": 0.87, "seen_count": 5, "stale": false },
        ...
      ]
    }
    """
    payload = {
        "landmarks": [
            {
                "id": lm.id,
                "class_label": lm.class_label,
                "x": round(lm.x, 3),
                "y": round(lm.y, 3),
                "confidence": round(lm.confidence, 3),
                "seen_count": lm.seen_count,
                "stale": lm.stale,
            }
            for lm in landmarks
        ]
    }
    return json.dumps(payload)


def landmarks_to_json(landmarks: list[SemanticLandmark]) -> "String":
    """
    Wrap landmarks_to_json_str in a std_msgs/String message.
    Called by the ROS2 node; requires ROS2 to be installed.
    """
    msg = String()
    msg.data = landmarks_to_json_str(landmarks)
    return msg
