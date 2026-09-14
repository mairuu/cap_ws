"""
test_ros_bridge.py
-------------------
Tests for ros_bridge.py conversions.

We don't need a running ROS2 system. We build lightweight mock message
objects that mimic the fields we read — just enough structure for the
conversion functions to work.

Real ROS2 messages have the same field layout; these mocks are drop-in
substitutes for the purpose of unit testing.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import pytest

# ── the functions under test ────────────────────────────────────────────────
from semantic_objects.ros_bridge import (
    landmarks_to_json_str,
    laserscan_to_data,
    detections_to_bboxes,
    transform_to_pose2d,
    landmarks_to_json,
)
from semantic_objects.landmark_store import SemanticLandmark


# ---------------------------------------------------------------------------
# Minimal ROS2 message mocks
# ---------------------------------------------------------------------------

@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

@dataclass
class Transform:
    translation: Vec3 = field(default_factory=Vec3)
    rotation: Quaternion = field(default_factory=Quaternion)

@dataclass
class TransformStamped:
    transform: Transform = field(default_factory=Transform)

@dataclass
class MockLaserScan:
    angle_min: float = -math.pi / 2
    angle_max: float =  math.pi / 2
    angle_increment: float = math.pi / 180  # 1° per ray
    range_min: float = 0.1
    range_max: float = 12.0
    ranges: list = field(default_factory=lambda: [2.0] * 181)

@dataclass
class BBoxCenter:
    position: Vec3 = field(default_factory=Vec3)

@dataclass
class BBox2D:
    center: BBoxCenter = field(default_factory=BBoxCenter)
    size_x: float = 100.0
    size_y: float = 100.0

@dataclass
class Hypothesis:
    class_id: str = "chair"
    score: float = 0.9

@dataclass
class ObjectHypothesisWithPose:
    hypothesis: Hypothesis = field(default_factory=Hypothesis)

@dataclass
class Detection2D:
    bbox: BBox2D = field(default_factory=BBox2D)
    results: list = field(default_factory=lambda: [ObjectHypothesisWithPose()])

@dataclass
class Detection2DArray:
    detections: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_detection(cx: float, cy: float, w: float, h: float,
                   class_id: str = "chair", score: float = 0.9) -> Detection2D:
    det = Detection2D()
    det.bbox.center.position.x = cx
    det.bbox.center.position.y = cy
    det.bbox.size_x = w
    det.bbox.size_y = h
    det.results = [ObjectHypothesisWithPose(Hypothesis(class_id=class_id, score=score))]
    return det


def make_landmark(x: float, y: float, label: str = "chair",
                  seen: int = 3, confidence: float = 0.8,
                  stale: bool = False) -> SemanticLandmark:
    now = time.time()
    return SemanticLandmark(
        id="test-id-" + label,
        class_label=label,
        x=x, y=y,
        confidence=confidence,
        seen_count=seen,
        first_seen=now,
        last_seen=now,
        stale=stale,
    )


def identity_quaternion() -> Quaternion:
    return Quaternion(x=0, y=0, z=0, w=1)


def yaw_quaternion(yaw: float) -> Quaternion:
    """Quaternion for a pure Z-axis rotation."""
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2),
        w=math.cos(yaw / 2),
    )


# ---------------------------------------------------------------------------
# LaserScan conversion
# ---------------------------------------------------------------------------

class TestLaserScanConversion:

    def test_fields_copied_correctly(self):
        msg = MockLaserScan(
            angle_min=-1.0, angle_max=1.0, angle_increment=0.01,
            range_min=0.2, range_max=10.0,
            ranges=[3.0, 1.5, float("inf")]
        )
        data = laserscan_to_data(msg)

        assert data.angle_min == -1.0
        assert data.angle_max == 1.0
        assert data.angle_increment == 0.01
        assert data.range_min == 0.2
        assert data.range_max == 10.0
        assert data.ranges == [3.0, 1.5, float("inf")]

    def test_ranges_are_copied_not_aliased(self):
        """Mutating the original list must not affect the converted data."""
        original = [1.0, 2.0, 3.0]
        msg = MockLaserScan(ranges=original)
        data = laserscan_to_data(msg)

        original[0] = 99.0
        assert data.ranges[0] == 1.0   # unchanged

    def test_empty_ranges_converts(self):
        msg = MockLaserScan(ranges=[])
        data = laserscan_to_data(msg)
        assert data.ranges == []


# ---------------------------------------------------------------------------
# Detection conversion
# ---------------------------------------------------------------------------

class TestDetectionConversion:

    def test_single_detection_converted(self):
        msg = Detection2DArray(detections=[
            make_detection(320, 240, 100, 80, "chair", 0.9)
        ])
        boxes = detections_to_bboxes(msg, min_confidence=0.5)

        assert len(boxes) == 1
        b = boxes[0]
        assert b.class_label == "chair"
        assert abs(b.confidence - 0.9) < 1e-6

    def test_bounding_box_corners_computed_correctly(self):
        """centre=(320,240) size=(100,80) → x1=270,y1=200,x2=370,y2=280."""
        msg = Detection2DArray(detections=[
            make_detection(cx=320, cy=240, w=100, h=80)
        ])
        boxes = detections_to_bboxes(msg)
        b = boxes[0]
        assert abs(b.x1 - 270.0) < 1e-6
        assert abs(b.y1 - 200.0) < 1e-6
        assert abs(b.x2 - 370.0) < 1e-6
        assert abs(b.y2 - 280.0) < 1e-6

    def test_low_confidence_filtered(self):
        msg = Detection2DArray(detections=[
            make_detection(320, 240, 100, 80, "chair", score=0.3)
        ])
        boxes = detections_to_bboxes(msg, min_confidence=0.5)
        assert len(boxes) == 0

    def test_threshold_is_exclusive(self):
        """Score == threshold is kept."""
        msg = Detection2DArray(detections=[
            make_detection(320, 240, 100, 80, "chair", score=0.5)
        ])
        boxes = detections_to_bboxes(msg, min_confidence=0.5)
        assert len(boxes) == 1

    def test_multiple_detections(self):
        msg = Detection2DArray(detections=[
            make_detection(200, 200, 80,  80,  "chair",  0.9),
            make_detection(400, 200, 60,  60,  "bottle", 0.7),
            make_detection(300, 300, 120, 100, "couch",  0.3),  # below threshold
        ])
        boxes = detections_to_bboxes(msg, min_confidence=0.5)
        assert len(boxes) == 2
        labels = {b.class_label for b in boxes}
        assert labels == {"chair", "bottle"}

    def test_best_hypothesis_chosen(self):
        """
        Detection with two hypotheses: bottle@0.4 and chair@0.8.
        Should pick chair.
        """
        det = Detection2D()
        det.bbox.center.position.x = 320
        det.bbox.center.position.y = 240
        det.bbox.size_x = 100
        det.bbox.size_y = 80
        det.results = [
            ObjectHypothesisWithPose(Hypothesis("bottle", 0.4)),
            ObjectHypothesisWithPose(Hypothesis("chair",  0.8)),
        ]
        msg = Detection2DArray(detections=[det])
        boxes = detections_to_bboxes(msg)

        assert len(boxes) == 1
        assert boxes[0].class_label == "chair"
        assert abs(boxes[0].confidence - 0.8) < 1e-6

    def test_detection_with_no_results_skipped(self):
        det = Detection2D(results=[])
        msg = Detection2DArray(detections=[det])
        boxes = detections_to_bboxes(msg)
        assert len(boxes) == 0

    def test_empty_array_returns_empty_list(self):
        msg = Detection2DArray(detections=[])
        boxes = detections_to_bboxes(msg)
        assert boxes == []


# ---------------------------------------------------------------------------
# TF / pose extraction
# ---------------------------------------------------------------------------

class TestTransformToPose2D:

    def test_identity_transform_gives_origin(self):
        tf = TransformStamped()
        tf.transform.translation = Vec3(0, 0, 0)
        tf.transform.rotation = identity_quaternion()

        pose = transform_to_pose2d(tf)
        assert abs(pose.x)   < 1e-9
        assert abs(pose.y)   < 1e-9
        assert abs(pose.yaw) < 1e-9

    def test_translation_preserved(self):
        tf = TransformStamped()
        tf.transform.translation = Vec3(3.0, 4.0, 0.0)
        tf.transform.rotation = identity_quaternion()

        pose = transform_to_pose2d(tf)
        assert abs(pose.x - 3.0) < 1e-9
        assert abs(pose.y - 4.0) < 1e-9

    def test_yaw_zero_from_identity_quaternion(self):
        tf = TransformStamped()
        tf.transform.rotation = identity_quaternion()
        pose = transform_to_pose2d(tf)
        assert abs(pose.yaw) < 1e-9

    def test_yaw_90_degrees(self):
        tf = TransformStamped()
        tf.transform.translation = Vec3(0, 0, 0)
        tf.transform.rotation = yaw_quaternion(math.pi / 2)

        pose = transform_to_pose2d(tf)
        assert abs(pose.yaw - math.pi / 2) < 1e-6

    def test_yaw_180_degrees(self):
        tf = TransformStamped()
        tf.transform.rotation = yaw_quaternion(math.pi)
        pose = transform_to_pose2d(tf)
        # atan2 returns values in (-π, π], so π is valid
        assert abs(abs(pose.yaw) - math.pi) < 1e-6

    def test_yaw_negative(self):
        tf = TransformStamped()
        tf.transform.rotation = yaw_quaternion(-math.pi / 4)
        pose = transform_to_pose2d(tf)
        assert abs(pose.yaw - (-math.pi / 4)) < 1e-6

    def test_full_transform(self):
        tf = TransformStamped()
        tf.transform.translation = Vec3(1.5, -2.3, 0.0)
        tf.transform.rotation = yaw_quaternion(0.8)

        pose = transform_to_pose2d(tf)
        assert abs(pose.x   -  1.5) < 1e-6
        assert abs(pose.y   - -2.3) < 1e-6
        assert abs(pose.yaw -  0.8) < 1e-6

    def test_z_translation_ignored(self):
        """Z (height) does not affect the 2D pose."""
        tf = TransformStamped()
        tf.transform.translation = Vec3(1.0, 2.0, 5.0)   # 5m height
        tf.transform.rotation = identity_quaternion()

        pose = transform_to_pose2d(tf)
        assert abs(pose.x - 1.0) < 1e-9
        assert abs(pose.y - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

class TestLandmarksToJson:

    def test_output_is_valid_json(self):
        lms = [make_landmark(1.0, 2.0, "chair")]
        raw = landmarks_to_json_str(lms)
        # should not raise
        parsed = json.loads(raw)
        assert "landmarks" in parsed

    def test_landmark_fields_present(self):
        lm = make_landmark(1.23, 4.56, "bottle", seen=5, confidence=0.88)
        raw = landmarks_to_json_str([lm])
        parsed = json.loads(raw)
        entry = parsed["landmarks"][0]

        assert entry["class_label"] == "bottle"
        assert abs(entry["x"] - 1.23) < 0.001
        assert abs(entry["y"] - 4.56) < 0.001
        assert abs(entry["confidence"] - 0.88) < 0.001
        assert entry["seen_count"] == 5
        assert entry["stale"] is False
        assert "id" in entry

    def test_stale_flag_serialised(self):
        lm = make_landmark(0.0, 0.0, "chair", stale=True)
        raw = landmarks_to_json_str([lm])
        parsed = json.loads(raw)
        assert parsed["landmarks"][0]["stale"] is True

    def test_multiple_landmarks_serialised(self):
        lms = [
            make_landmark(1.0, 0.0, "chair"),
            make_landmark(2.0, 0.0, "bottle"),
            make_landmark(3.0, 0.0, "couch"),
        ]
        raw = landmarks_to_json_str(lms)
        parsed = json.loads(raw)
        assert len(parsed["landmarks"]) == 3

    def test_empty_list_gives_empty_landmarks(self):
        raw = landmarks_to_json_str([])
        parsed = json.loads(raw)
        assert parsed["landmarks"] == []

    def test_coordinates_rounded_to_3dp(self):
        lm = make_landmark(1.23456789, 4.56789012, "chair")
        raw = landmarks_to_json_str([lm])
        parsed = json.loads(raw)
        entry = parsed["landmarks"][0]
        # JSON floats should be rounded — 3 decimal places max
        assert entry["x"] == round(entry["x"], 3)
        assert entry["y"] == round(entry["y"], 3)


# ---------------------------------------------------------------------------
# Integration: conversion chain
# ---------------------------------------------------------------------------

class TestConversionChain:

    def test_scan_and_detection_to_bbox_and_data(self):
        """
        Verify that converting a scan and detection together gives
        coherent types that can be passed directly into the pipeline.
        """
        from semantic_objects.lidar_range_extractor import (
            CameraIntrinsics, LidarRangeExtractor
        )

        scan_msg = MockLaserScan(
            angle_min=-math.pi / 2,
            angle_max= math.pi / 2,
            angle_increment=math.pi / 180,
            range_min=0.1, range_max=12.0,
            ranges=[2.0] * 181,
        )
        det_msg = Detection2DArray(detections=[
            make_detection(320, 240, 100, 80, "chair", 0.9)
        ])

        scan  = laserscan_to_data(scan_msg)
        boxes = detections_to_bboxes(det_msg)

        camera = CameraIntrinsics(fx=554, fy=554, cx=320, cy=240,
                                  width=640, height=480)
        extractor = LidarRangeExtractor(camera)
        estimate = extractor.extract_range(boxes[0], scan)

        assert estimate.valid
        assert abs(estimate.range_m - 2.0) < 0.1
