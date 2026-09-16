"""
semantic_objects_node.py
------------------------
ROS 2 node: semantic_objects. Camera + 2D lidar semantic landmark fusion.

Subscribes to:
  /scan                (sensor_msgs/LaserScan, best-effort -- matches the driver)
  /detections          (vision_msgs/Detection2DArray from my_bot's yolo_detector.py;
                        header.stamp is the image CAPTURE time, Detection2D.id the
                        tracker id, hypothesis.class_id the class NAME)
  /diff_cont/odom      (nav_msgs/Odometry -- the rotation gate, P3. NOT /odom.)

Publishes:
  /semantic_landmarks  (std_msgs/String, JSON for the browser bridge; latched)
  /semantic_markers    (visualization_msgs/MarkerArray for RViz; latched)

Services:
  clear_landmarks      (std_srvs/Empty) -- called by semantic_bridge's /api/clear

TF:
  map → base_link      at the DETECTION's stamp (P2), interpolated by tf2
  base_link → camera_link   once at startup: the camera's origin and YAW
  base_link → laser_frame   once at startup: the lidar's origin (range origin)
  Both static lookups are decision D-10: the URDF is the single source of the
  sensor geometry; nothing here is read from a params file. camera_link is the
  x-forward MOUNT frame; camera_optical_link would give yaw = -90 deg and
  rotate every landmark by a right angle, and the node refuses it.

Parameters -- DOTTED names, under `semantic_objects: ros__parameters:` in
config/robot_params.yaml. (The June-era code read five of these with slashes
and died in __init__ with ParameterNotDeclaredError.)

  camera.calibration_file   camera_info YAML (my_bot/config/c615_640x480.yaml).
                            fx/fy/cx/cy are read from camera_matrix. NO defaults:
                            a missing file is fatal, not a silent 20 % error.
  camera.width, .height     must match the file's image_width/height
  detection.min_confidence  YOLO confidence gate
  detection.range_method    "min" | "median" | "mean"
  detection.angular_padding extra angular margin per bbox (rad). The scan
                            window is computed at the camera; the lidar sits
                            3 cm beside it, so allow ~2 deg at 1 m.
  detection.max_range, .min_range
  detection.min_returns     P4: reject windows backed by fewer live rays
  detection.max_spread      P4: reject windows whose returns spread more (m)
  motion.max_omega          P3: drop frames while |yaw rate| exceeds this (rad/s)
  motion.odom_max_age       P3: drop frames when odom is older than this (s);
                            fail-CLOSED, with a warning that says so
  odom_topic                /diff_cont/odom
  track.timeout, .max_jump  P5 track-id binding lifetime (s) and sanity jump (m)
  landmark.merge_radius, .ema_alpha, .min_seen_to_publish, .stale_timeout
  landmark.persist_path     JSON written on the publish timer when dirty
  landmark.restore_on_start false: one-session SLAM (D-05) means a new map
                            frame every run; restored landmarks would be wrong
  publish.rate_hz
  tf.map_frame, .base_frame, .camera_frame, .laser_frame
  tf.lookup_timeout         wait for map→base_link at the detection stamp (s)
  tf.static_timeout         wait for the two static frames at startup (s)
  sync.slop                 ApproximateTimeSynchronizer tolerance (s). Both
                            stamps are capture times; scan 86 ms / detections
                            66 ms -> the nearest pair is <= 33 ms apart. 0.1.
  report_period             seconds between the "fused n/m" summary lines
"""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import yaml

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time

import tf2_ros
from tf2_ros import TransformException

from message_filters import ApproximateTimeSynchronizer, Subscriber

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty
from vision_msgs.msg import Detection2DArray
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

# Sensor data: best-effort, volatile. Matches the ydlidar driver's SensorDataQoS
# exactly, and is compatible with yolo_detector.py's RELIABLE /detections.
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


class ConfigError(RuntimeError):
    """A parameter or file the node cannot run without."""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SemanticObjectsNode(Node):

    def __init__(self):
        super().__init__("semantic_objects")
        self._declare_parameters()
        P = self._p

        # --- Camera intrinsics: from the calibration file, no defaults ---
        self._intrinsics = self._load_intrinsics()

        # --- TF. spin_thread=True is not optional: with the node's own
        # single-threaded executor, a lookup_transform(timeout=...) inside a
        # callback sleeps in a loop during which no /tf can arrive, so every
        # wait times out and merely stalls the pipeline. A listener thread
        # lets the buffer fill while we wait.
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self, spin_thread=True)

        # --- Sensor geometry from TF (D-10) ---
        cam, lidar = self._lookup_sensor_geometry()

        # --- Pipeline components ---
        self._extractor = LidarRangeExtractor(
            camera=self._intrinsics,
            range_method=P("detection.range_method"),
            angular_padding=P("detection.angular_padding"),
            min_returns=P("detection.min_returns"),
            max_spread=P("detection.max_spread"),
            camera_yaw=cam.yaw,
        )
        self._projector = WorldPointProjector(
            extrinsics=CameraExtrinsics(dx=cam.x, dy=cam.y, yaw=cam.yaw,
                                        lidar_dx=lidar.x, lidar_dy=lidar.y),
            max_range=P("detection.max_range"),
            min_range=P("detection.min_range"),
        )
        self._store = self._build_store()

        # --- Odometry gate state (P3) ---
        self._omega: Optional[float] = None
        self._odom_rx_time: Optional[float] = None
        self._odom_sub = self.create_subscription(
            Odometry, P("odom_topic"), self._on_odom, SENSOR_QOS)

        # --- Stats, reset every report ---
        # MUST be initialised BEFORE _setup_subscribers(). TransformListener is
        # constructed with spin_thread=True, so a background executor exists and
        # can dispatch _on_synced the instant the synchroniser is registered --
        # before __init__ has finished. If /detections is ALREADY flowing when
        # this node starts, that happens immediately, _on_synced raises
        # AttributeError on self._stats, and the exception kills the executor:
        # the process stays alive and the node stays registered while nothing
        # is ever processed again. Seen 16 Sep starting `make semantic` after
        # `make yolo` was already publishing at 15 Hz. Starting them the other
        # way round hides it, which is why it survived until now.
        self._stats = Counter()
        self._total_detections = 0
        self._total_fused = 0

        # --- Subscribers with time-sync ---
        self._setup_subscribers()

        # --- Publishers ---
        self._pub_json = self.create_publisher(String, "/semantic_landmarks", LANDMARK_QOS)
        self._pub_markers = self.create_publisher(MarkerArray, "/semantic_markers", LANDMARK_QOS)

        # --- Service ---
        self._clear_srv = self.create_service(Empty, "clear_landmarks", self._on_clear)

        # --- Periodic timers ---
        rate_hz = P("publish.rate_hz")
        stale_check_s = max(10.0, P("landmark.stale_timeout") / 10)
        self._publish_timer = self.create_timer(1.0 / rate_hz, self._publish_landmarks)
        self._stale_timer = self.create_timer(stale_check_s, self._check_stale)
        self._report_timer = self.create_timer(P("report_period"), self._report)

        # Publish once so RViz/bridge see an (empty) latched payload immediately.
        self._publish_landmarks()

        self.get_logger().info(
            f"semantic_objects ready  "
            f"fx={self._intrinsics.fx:.3f} fy={self._intrinsics.fy:.3f} "
            f"cx={self._intrinsics.cx:.3f} cy={self._intrinsics.cy:.3f}  "
            f"camera=({cam.x:+.3f},{cam.y:+.3f}) yaw {math.degrees(cam.yaw):+.2f}deg  "
            f"lidar=({lidar.x:+.3f},{lidar.y:+.3f})  "
            f"merge_radius={P('landmark.merge_radius')}m  "
            f"ema_alpha={P('landmark.ema_alpha')}  "
            f"min_returns={P('detection.min_returns')} max_spread={P('detection.max_spread')}m  "
            f"max_omega={P('motion.max_omega')}rad/s  "
            f"persist={P('landmark.persist_path') or 'off'}"
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _p(self, name: str):
        return self.get_parameter(name).value

    def _declare_parameters(self):
        self.declare_parameters(
            namespace="",
            parameters=[
                # Camera intrinsics come from the calibration file. There is
                # deliberately no fx/fy/cx/cy parameter and no default: the
                # June node's built-in 554.0 was 17 % off the real 667.9 and
                # would have mapped the room 20 % wrong without a word.
                ("camera.calibration_file", ""),
                ("camera.width", 640),
                ("camera.height", 480),

                # Detection pipeline
                ("detection.min_confidence", 0.5),
                ("detection.range_method", "min"),
                ("detection.angular_padding", 0.035),
                ("detection.max_range", 5.0),
                ("detection.min_range", 0.15),
                ("detection.min_returns", 3),      # P4
                ("detection.max_spread", 0.5),     # P4

                # Motion gate (P3)
                ("motion.max_omega", 0.3),
                ("motion.odom_max_age", 0.5),
                ("odom_topic", "/diff_cont/odom"),

                # Track association (P5)
                ("track.timeout", 2.0),
                ("track.max_jump", 1.0),

                # Landmark store
                ("landmark.merge_radius", 0.5),
                ("landmark.ema_alpha", 0.3),
                ("landmark.min_seen_to_publish", 2),
                ("landmark.stale_timeout", 300.0),
                ("landmark.persist_path", ""),
                ("landmark.restore_on_start", False),

                # Publishing
                ("publish.rate_hz", 2.0),
                ("report_period", 5.0),

                # TF frames
                ("tf.map_frame", "map"),
                ("tf.base_frame", "base_link"),
                ("tf.camera_frame", "camera_link"),
                ("tf.laser_frame", "laser_frame"),
                ("tf.lookup_timeout", 0.05),
                ("tf.static_timeout", 10.0),

                # Topics and time sync
                ("scan_topic", "/scan"),
                ("detections_topic", "/detections"),
                ("sync.slop", 0.1),
            ],
        )

    # ------------------------------------------------------------------
    # Startup: intrinsics and sensor geometry
    # ------------------------------------------------------------------

    def _load_intrinsics(self) -> CameraIntrinsics:
        path_str = self._p("camera.calibration_file")
        if not path_str:
            raise ConfigError(
                "camera.calibration_file is not set. The node has NO built-in "
                "intrinsics; point it at my_bot/config/c615_640x480.yaml "
                "(launch/semantic.launch.py does this).")
        path = Path(path_str).expanduser()
        if not path.is_file():
            raise ConfigError(f"camera.calibration_file not found: {path}")
        try:
            data = yaml.safe_load(path.read_text())
            k = data["camera_matrix"]["data"]
            fx, cx, fy, cy = float(k[0]), float(k[2]), float(k[4]), float(k[5])
            w, h = int(data["image_width"]), int(data["image_height"])
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise ConfigError(f"{path} is not a camera_info YAML (camera_matrix/data, "
                              f"image_width, image_height): {e}") from e
        if fx <= 0 or fy <= 0:
            raise ConfigError(f"{path}: fx/fy must be positive, got {fx}/{fy}")
        if (w, h) != (self._p("camera.width"), self._p("camera.height")):
            raise ConfigError(
                f"{path} is calibrated at {w}x{h} but camera.width/height say "
                f"{self._p('camera.width')}x{self._p('camera.height')}. Intrinsics "
                f"are only valid at the size they were calibrated at.")
        self.get_logger().info(f"intrinsics from {path}: fx {fx:.3f} fy {fy:.3f} "
                               f"cx {cx:.3f} cy {cy:.3f} @ {w}x{h}")
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)

    def _lookup_static(self, parent: str, child: str):
        """Wait up to tf.static_timeout for a static transform; the listener
        thread fills the buffer while we wait."""
        timeout = float(self._p("tf.static_timeout"))
        deadline = time.monotonic() + timeout
        last_err = None
        while time.monotonic() < deadline:
            try:
                return self._tf_buffer.lookup_transform(
                    parent, child, Time(), timeout=Duration(seconds=0.5))
            except TransformException as e:
                last_err = e
        raise ConfigError(
            f"TF {parent} -> {child} not available after {timeout:.0f} s: {last_err}. "
            f"Is robot_state_publisher up (make real)?")

    @staticmethod
    def _roll_pitch(q) -> tuple[float, float]:
        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr, cosr)
        sinp = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
        pitch = math.asin(sinp)
        return roll, pitch

    def _lookup_sensor_geometry(self):
        """Camera origin + mount yaw from camera_link; range origin from laser_frame.
        Returns two Pose2D in base_link: (camera, lidar)."""
        base = self._p("tf.base_frame")
        cam_frame = self._p("tf.camera_frame")
        laser_frame = self._p("tf.laser_frame")

        cam_tf = self._lookup_static(base, cam_frame)
        roll, pitch = self._roll_pitch(cam_tf.transform.rotation)
        if abs(roll) > math.radians(20) or abs(pitch) > math.radians(20):
            raise ConfigError(
                f"tf.camera_frame={cam_frame!r} has roll {math.degrees(roll):.0f} deg / "
                f"pitch {math.degrees(pitch):.0f} deg. That is the OPTICAL frame "
                f"(z-forward); this node's 2D projector needs the x-forward MOUNT "
                f"frame, camera_link. Using the optical frame rotates every "
                f"landmark by 90 deg.")
        cam_pose = transform_to_pose2d(cam_tf)

        laser_tf = self._lookup_static(base, laser_frame)
        laser_pose = transform_to_pose2d(laser_tf)

        self.get_logger().info(
            f"{base}->{cam_frame}: ({cam_pose.x:+.3f},{cam_pose.y:+.3f}) yaw "
            f"{math.degrees(cam_pose.yaw):+.2f} deg, pitch {math.degrees(pitch):+.1f} deg "
            f"(ignored, 2D);  {base}->{laser_frame}: ({laser_pose.x:+.3f},{laser_pose.y:+.3f}) "
            f"yaw {math.degrees(laser_pose.yaw):+.2f} deg")
        if abs(laser_pose.yaw) > math.radians(1.0):
            self.get_logger().warn(
                f"{laser_frame} is yawed {math.degrees(laser_pose.yaw):+.1f} deg in "
                f"{base}. The scan's angles are assumed to be in base_link "
                f"heading; a yawed laser_joint would bias every bearing. "
                f"(ydlidar.yaml's reversion/inverted are the right place for "
                f"scan orientation, not the joint.)")
        return cam_pose, laser_pose

    # ------------------------------------------------------------------
    # Component builders
    # ------------------------------------------------------------------

    def _build_store(self) -> LandmarkStore:
        path_str = self._p("landmark.persist_path")
        path = Path(path_str).expanduser() if path_str else None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        return LandmarkStore(
            merge_radius=self._p("landmark.merge_radius"),
            ema_alpha=self._p("landmark.ema_alpha"),
            min_seen_to_publish=self._p("landmark.min_seen_to_publish"),
            stale_timeout=self._p("landmark.stale_timeout"),
            persist_path=path,
            autosave=False,                       # saved from the publish timer
            restore_on_start=bool(self._p("landmark.restore_on_start")),
            track_timeout=self._p("track.timeout"),
            track_max_jump=self._p("track.max_jump"),
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _setup_subscribers(self):
        """
        ApproximateTimeSynchronizer pairs each LaserScan with the nearest
        Detection2DArray within `slop` seconds. Both stamps are capture times.
        """
        slop = self._p("sync.slop")
        self._scan_sub = Subscriber(self, LaserScan, self._p("scan_topic"), qos_profile=SENSOR_QOS)
        self._det_sub = Subscriber(self, Detection2DArray, self._p("detections_topic"),
                                   qos_profile=SENSOR_QOS)
        self._sync = ApproximateTimeSynchronizer(
            [self._scan_sub, self._det_sub], queue_size=10, slop=slop)
        self._sync.registerCallback(self._on_synced)

    def _on_odom(self, msg: Odometry):
        self._omega = msg.twist.twist.angular.z
        self._odom_rx_time = time.monotonic()

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def _on_synced(self, scan_msg: LaserScan, det_msg: Detection2DArray):
        """
        Pipeline per paired (scan, detections):
          0. Motion gate (P3): skip the frame while turning, or without odom.
          1. Robot pose from TF at the DETECTION's stamp (P2).
          2. Convert messages.
          3. Per bbox: lidar range (P4 rejection) -> world point -> store (P5).
        """
        self._stats["frames"] += 1

        # Step 0: motion gate, fail-closed
        if self._odom_rx_time is None or \
                time.monotonic() - self._odom_rx_time > self._p("motion.odom_max_age"):
            self._stats["gate_no_odom"] += 1
            self.get_logger().warn(
                f"no odometry on {self._p('odom_topic')} -- motion gate CLOSED, "
                f"nothing is fused. Is make real up?",
                throttle_duration_sec=5.0)
            return
        if abs(self._omega) > self._p("motion.max_omega"):
            self._stats["gate_turning"] += 1
            return

        # Step 1: TF at the detection's capture time
        robot_pose = self._lookup_pose(det_msg.header.stamp)
        if robot_pose is None:
            self._stats["tf_miss"] += 1
            return

        # Step 2: convert messages
        scan = laserscan_to_data(scan_msg)
        boxes = detections_to_bboxes(
            det_msg, min_confidence=self._p("detection.min_confidence"))
        if not boxes:
            return

        # Step 3: process each detection
        fused_this_frame = 0
        for bbox in boxes:
            estimate = self._extractor.extract_range(bbox, scan)
            if not estimate.valid:
                self._stats[f"reject_{estimate.reason}"] += 1
                self.get_logger().debug(
                    f"reject {estimate.reason} for {bbox.class_label!r}: "
                    f"n={estimate.n_returns} spread={estimate.spread_m:.2f}m "
                    f"bbox=({bbox.x1:.0f},{bbox.y1:.0f},{bbox.x2:.0f},{bbox.y2:.0f})")
                continue

            world_pt = self._projector.project(estimate, robot_pose)
            if not world_pt.valid:
                self._stats["reject_range_gate"] += 1
                self.get_logger().debug(
                    f"range {estimate.range_m:.2f}m out of gate for {bbox.class_label!r}")
                continue

            result = self._store.observe_with_confidence(
                world_pt, bbox.class_label, bbox.confidence, track_id=bbox.track_id)

            how = "NEW" if result.created else ("TRK" if result.by_track else f"NN d={result.distance:.2f}m")
            self.get_logger().debug(
                f"{how}  {bbox.class_label!r} id={bbox.track_id or '-'}  "
                f"world=({world_pt.x:.2f},{world_pt.y:.2f})  "
                f"range={estimate.range_m:.2f}m n={estimate.n_returns} "
                f"bearing={math.degrees(-estimate.azimuth_center):+.1f}deg")
            fused_this_frame += 1

        self._stats["detections"] += len(boxes)
        self._stats["fused"] += fused_this_frame
        self._total_detections += len(boxes)
        self._total_fused += fused_this_frame

    # ------------------------------------------------------------------
    # TF helper
    # ------------------------------------------------------------------

    def _lookup_pose(self, stamp):
        """map → base_link at `stamp` (the detection's capture time), or None.
        Skips the frame on any failure -- tf2 refusing to extrapolate is the
        right answer, not a reason to fall back to 'latest'."""
        map_frame = self._p("tf.map_frame")
        base_frame = self._p("tf.base_frame")
        try:
            tf = self._tf_buffer.lookup_transform(
                map_frame, base_frame, Time.from_msg(stamp),
                timeout=Duration(seconds=float(self._p("tf.lookup_timeout"))))
            return transform_to_pose2d(tf)
        except TransformException as e:
            self.get_logger().warn(
                f"TF {map_frame}->{base_frame} at detection stamp failed: {e}",
                throttle_duration_sec=5.0)
            return None

    # ------------------------------------------------------------------
    # Service
    # ------------------------------------------------------------------

    def _on_clear(self, request, response):
        n = len(self._store)
        self._store.clear()
        self._store.save()
        self._publish_landmarks()
        self.get_logger().info(f"clear_landmarks: dropped {n} landmark(s)")
        return response

    # ------------------------------------------------------------------
    # Periodic
    # ------------------------------------------------------------------

    def _publish_landmarks(self):
        """Publish confirmed landmarks as JSON + RViz markers -- ALWAYS, empty
        included, so a clear reaches the UI and the latched payload is never
        stale. Save when dirty."""
        confirmed = self._store.confirmed_landmarks()
        self._pub_json.publish(landmarks_to_json(confirmed))
        self._pub_markers.publish(
            landmarks_to_marker_array(confirmed, frame_id=self._p("tf.map_frame")))
        if self._store.dirty and self._store.persist_path:
            try:
                self._store.save()
            except OSError as e:
                self.get_logger().error(f"could not save {self._store.persist_path}: {e}",
                                        throttle_duration_sec=30.0)

    def _check_stale(self):
        for lid in self._store.mark_stale():
            lm = self._store.get(lid)
            if lm:
                self.get_logger().info(
                    f"landmark stale: {lm.class_label!r} at ({lm.x:.2f},{lm.y:.2f}) id={lid[:8]}")

    def _report(self):
        s = self._stats
        if not s["frames"]:
            self.get_logger().warn(
                f"no paired scan+detections in {self._p('report_period'):.0f} s -- "
                f"are {self._p('scan_topic')} and {self._p('detections_topic')} both "
                f"publishing, within sync.slop={self._p('sync.slop')} s of each other?")
            return
        rejects = ", ".join(f"{k[7:]} {v}" for k, v in sorted(s.items()) if k.startswith("reject_"))
        self.get_logger().info(
            f"fused {s['fused']}/{s['detections']} detections in {s['frames']} frames | "
            f"gate: no-odom {s['gate_no_odom']} turning {s['gate_turning']} | "
            f"tf miss {s['tf_miss']} | rejected: {rejects or 'none'} | "
            f"landmarks {len(self._store)} confirmed {len(self._store.confirmed_landmarks())} "
            f"| total fused {self._total_fused}/{self._total_detections}")
        s.clear()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SemanticObjectsNode()
        rclpy.spin(node)
    except ConfigError as e:
        rclpy.logging.get_logger("semantic_objects").fatal(str(e))
        raise SystemExit(2)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
