"""
test_world_point_projector.py
------------------------------
Unit tests for WorldPointProjector.

Strategy: test each transformation step in isolation, then compose.

Naming convention for test cases:
  robot at origin facing east (+x) = yaw 0.0 — simplest case
  robot at origin facing north (+y) = yaw π/2
  robot at (3, 4) facing various directions
"""

import math
import pytest

from semantic_objects.lidar_range_extractor import RangeEstimate
from semantic_objects.world_point_projector import (
    CameraExtrinsics,
    Pose2D,
    WorldPoint,
    WorldPointProjector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_estimate(
    range_m: float = 2.0,
    azimuth_center: float = 0.0,
    n_returns: int = 5,
    valid: bool = True,
) -> RangeEstimate:
    return RangeEstimate(
        range_m=range_m,
        azimuth_center=azimuth_center,
        azimuth_left=azimuth_center - 0.05,
        azimuth_right=azimuth_center + 0.05,
        n_returns=n_returns,
        method="median",
        valid=valid,
    )


def assert_point_close(wp: WorldPoint, x: float, y: float, tol: float = 1e-6):
    assert wp.valid, f"Expected valid WorldPoint, got invalid. range={wp.range_m}"
    assert abs(wp.x - x) < tol, f"x: expected {x:.4f}, got {wp.x:.4f}"
    assert abs(wp.y - y) < tol, f"y: expected {y:.4f}, got {wp.y:.4f}"


# ---------------------------------------------------------------------------
# Basic projection — no extrinsics offset, robot at origin
# ---------------------------------------------------------------------------

class TestProjectionAtOrigin:
    """Robot at (0,0), yaw=0 (facing +x / east). No camera offset."""

    def setup_method(self):
        self.proj = WorldPointProjector(max_range=10.0)
        self.pose = Pose2D(x=0.0, y=0.0, yaw=0.0)

    def test_straight_ahead_lands_on_positive_x(self):
        """Object dead ahead (azimuth=0) → world point at (range, 0)."""
        est = make_estimate(range_m=2.0, azimuth_center=0.0)
        wp = self.proj.project(est, self.pose)
        assert_point_close(wp, x=2.0, y=0.0)

    def test_object_to_the_right(self):
        """
        Object at +45° azimuth (camera right).
        Camera azimuth is CW, so right = positive.
        In ROS base_link: right = -y.
        → world point at (r*cos45, -r*sin45).
        """
        az = math.pi / 4          # 45° right
        r  = math.sqrt(2.0)       # so x=1, y=-1
        est = make_estimate(range_m=r, azimuth_center=az)
        wp = self.proj.project(est, self.pose)
        assert_point_close(wp, x=1.0, y=-1.0, tol=1e-5)

    def test_object_to_the_left(self):
        """Object at -45° azimuth (camera left) → world point at (1, +1)."""
        az = -math.pi / 4
        r  = math.sqrt(2.0)
        est = make_estimate(range_m=r, azimuth_center=az)
        wp = self.proj.project(est, self.pose)
        assert_point_close(wp, x=1.0, y=1.0, tol=1e-5)

    def test_range_and_azimuth_preserved_in_result(self):
        est = make_estimate(range_m=3.5, azimuth_center=0.1)
        wp = self.proj.project(est, self.pose)
        assert abs(wp.range_m - 3.5) < 1e-9
        assert abs(wp.azimuth_rad - 0.1) < 1e-9
        assert wp.n_lidar_returns == 5


# ---------------------------------------------------------------------------
# Robot rotation
# ---------------------------------------------------------------------------

class TestRobotRotation:

    def test_robot_facing_north_object_ahead_lands_on_positive_y(self):
        """
        Robot at origin, yaw=π/2 (facing +y / north).
        Object straight ahead → world point at (0, range).
        """
        proj = WorldPointProjector(max_range=10.0)
        pose = Pose2D(x=0.0, y=0.0, yaw=math.pi / 2)
        est  = make_estimate(range_m=2.0, azimuth_center=0.0)

        wp = proj.project(est, pose)
        assert_point_close(wp, x=0.0, y=2.0, tol=1e-5)

    def test_robot_facing_west_object_ahead_lands_on_negative_x(self):
        """Robot at origin, yaw=π (facing west/-x). Object ahead → (-range, 0)."""
        proj = WorldPointProjector(max_range=10.0)
        pose = Pose2D(x=0.0, y=0.0, yaw=math.pi)
        est  = make_estimate(range_m=3.0, azimuth_center=0.0)

        wp = proj.project(est, pose)
        assert_point_close(wp, x=-3.0, y=0.0, tol=1e-5)

    def test_robot_facing_south_object_ahead_lands_on_negative_y(self):
        """Robot at origin, yaw=-π/2. Object ahead → (0, -range)."""
        proj = WorldPointProjector(max_range=10.0)
        pose = Pose2D(x=0.0, y=0.0, yaw=-math.pi / 2)
        est  = make_estimate(range_m=1.5, azimuth_center=0.0)

        wp = proj.project(est, pose)
        assert_point_close(wp, x=0.0, y=-1.5, tol=1e-5)

    def test_360_rotation_returns_to_start(self):
        """
        Same object seen from robot rotated 0° and 360° → same world point.
        """
        proj = WorldPointProjector(max_range=10.0)
        est  = make_estimate(range_m=2.0, azimuth_center=0.3)

        pose_0   = Pose2D(x=1.0, y=2.0, yaw=0.5)
        pose_360 = Pose2D(x=1.0, y=2.0, yaw=0.5 + 2 * math.pi)

        wp_0   = proj.project(est, pose_0)
        wp_360 = proj.project(est, pose_360)

        assert abs(wp_0.x - wp_360.x) < 1e-5
        assert abs(wp_0.y - wp_360.y) < 1e-5


# ---------------------------------------------------------------------------
# Robot translation
# ---------------------------------------------------------------------------

class TestRobotTranslation:

    def test_robot_offset_adds_to_world_point(self):
        """
        Object at (2, 0) in base_link. Robot at (3, 4) facing east.
        → World point at (5, 4).
        """
        proj = WorldPointProjector(max_range=10.0)
        pose = Pose2D(x=3.0, y=4.0, yaw=0.0)
        est  = make_estimate(range_m=2.0, azimuth_center=0.0)

        wp = proj.project(est, pose)
        assert_point_close(wp, x=5.0, y=4.0)

    def test_robot_at_arbitrary_position_and_yaw(self):
        """
        Robot at (1, 1), yaw=π/2 (north). Object straight ahead at 2m.
        Object should land at (1, 3) in world frame.
        """
        proj = WorldPointProjector(max_range=10.0)
        pose = Pose2D(x=1.0, y=1.0, yaw=math.pi / 2)
        est  = make_estimate(range_m=2.0, azimuth_center=0.0)

        wp = proj.project(est, pose)
        assert_point_close(wp, x=1.0, y=3.0, tol=1e-5)


# ---------------------------------------------------------------------------
# Camera extrinsics
# ---------------------------------------------------------------------------

class TestCameraExtrinsics:

    def test_forward_offset_shifts_point_along_robot_forward(self):
        """
        Camera mounted 0.1 m in front of base_link. Robot facing east.
        Object at 2 m → effective distance from base_link = 2.1 m.
        """
        extr = CameraExtrinsics(dx=0.1, dy=0.0, yaw=0.0)
        proj = WorldPointProjector(extrinsics=extr, max_range=10.0)
        pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
        est  = make_estimate(range_m=2.0, azimuth_center=0.0)

        wp = proj.project(est, pose)
        # x should be 2.0 (range) + 0.1 (camera offset) = 2.1
        assert_point_close(wp, x=2.1, y=0.0)

    def test_lateral_offset_shifts_point_perpendicular(self):
        """
        Camera mounted 0.05 m to the left (dy=+0.05). Robot facing east, yaw=0.
        Object straight ahead at 2m.
        → world point should be 2.0 ahead and 0.05 left.
        """
        extr = CameraExtrinsics(dx=0.0, dy=0.05, yaw=0.0)
        proj = WorldPointProjector(extrinsics=extr, max_range=10.0)
        pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
        est  = make_estimate(range_m=2.0, azimuth_center=0.0)

        wp = proj.project(est, pose)
        assert_point_close(wp, x=2.0, y=0.05)

    def test_camera_yaw_offset_rotates_azimuth(self):
        """
        Camera panned 30° left (yaw=+π/6). Object appears at azimuth 0
        in camera frame but is actually 30° to the left in base_link frame.
        Robot facing east at origin.
        → Object at (r*cos30°, r*sin30°) in world (left = +y).
        """
        camera_yaw = math.pi / 6   # 30° left pan
        extr = CameraExtrinsics(dx=0.0, dy=0.0, yaw=camera_yaw)
        proj = WorldPointProjector(extrinsics=extr, max_range=10.0)
        pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
        est  = make_estimate(range_m=2.0, azimuth_center=0.0)

        wp = proj.project(est, pose)
        # az_base = 0 + π/6, az_base_ros = -π/6
        # x = 2 * cos(-π/6) = 2 * cos(30°) = √3 ≈ 1.732
        # y = 2 * sin(-π/6) = 2 * (-0.5) = -1.0
        # Wait: camera yaw +π/6 means the camera points LEFT (CCW).
        # An object at az=0 in camera is actually to the left of forward.
        # az_base = 0.0 + π/6 = π/6 (in base_link CCW before flip)
        # az_base_ros = -π/6
        # x = 2*cos(-π/6) = √3, y = 2*sin(-π/6) = -1.0
        # Hmm: that's to the right? Let me reason again.
        #
        # camera yaw = +π/6 means the camera is rotated CCW (panned left).
        # An object at az=0 in the camera is in the camera's forward direction.
        # Camera forward is π/6 to the LEFT of the robot forward.
        # So the object is to the LEFT of robot forward.
        # In ROS base_link: left = +y.
        # Expected: x = 2*cos(π/6) = √3, y = 2*sin(π/6) = 1.0
        #
        # The sign flip: az_base_ros = -(az_base) = -(0 + π/6) = -π/6
        # That gives y = sin(-π/6) = -0.5 → -1.0  (right, not left!)
        #
        # Root cause: camera yaw is defined as CCW but our azimuth is CW.
        # camera_yaw = +π/6 means camera rotated CCW in base_link.
        # Object at camera az=0 → angle π/6 from robot forward in CCW direction (left).
        # In ROS: az_ros = π/6 → y = sin(π/6) > 0 (left). Correct.
        # But az_base_ros = -(az_camera + camera_yaw) = -(0 + π/6) = -π/6.
        # That gives y < 0. Wrong.
        #
        # Fix understanding: camera_yaw adds in the SAME space as az_base_ros.
        # camera_yaw CCW = consistent with ROS. az_camera is CW.
        # So: az_base_ros = -az_camera + camera_yaw_ros
        #                 = -0.0 + π/6 = π/6
        # → x = 2*cos(π/6) = √3, y = 2*sin(π/6) = 1.0.
        #
        # This reveals a sign convention issue in the projector that this test will catch.
        expected_x = 2.0 * math.cos(math.pi / 6)   # ≈ 1.732
        expected_y = 2.0 * math.sin(math.pi / 6)   # = 1.0  (left of forward)
        assert_point_close(wp, x=expected_x, y=expected_y, tol=1e-5)


# ---------------------------------------------------------------------------
# Range gating
# ---------------------------------------------------------------------------

class TestRangeGating:

    def test_invalid_estimate_returns_invalid_worldpoint(self):
        proj = WorldPointProjector()
        pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
        est  = make_estimate(valid=False)

        wp = proj.project(est, pose)
        assert not wp.valid

    def test_range_beyond_max_is_rejected(self):
        proj = WorldPointProjector(max_range=5.0)
        pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
        est  = make_estimate(range_m=6.0)

        wp = proj.project(est, pose)
        assert not wp.valid

    def test_range_at_exactly_max_is_accepted(self):
        proj = WorldPointProjector(max_range=5.0)
        pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
        est  = make_estimate(range_m=5.0)

        wp = proj.project(est, pose)
        assert wp.valid

    def test_range_below_min_is_rejected(self):
        proj = WorldPointProjector(min_range=0.15)
        pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
        est  = make_estimate(range_m=0.10)

        wp = proj.project(est, pose)
        assert not wp.valid

    def test_range_at_exactly_min_is_accepted(self):
        proj = WorldPointProjector(min_range=0.15)
        pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
        est  = make_estimate(range_m=0.15)

        wp = proj.project(est, pose)
        assert wp.valid


# ---------------------------------------------------------------------------
# Round-trip / consistency
# ---------------------------------------------------------------------------

class TestConsistency:

    def test_distance_to_robot_matches_range(self):
        """
        Euclidean distance from robot to projected world point
        should equal the range (when no extrinsics offset).
        """
        proj = WorldPointProjector(max_range=10.0)
        pose = Pose2D(x=3.0, y=7.0, yaw=1.2)
        est  = make_estimate(range_m=2.5, azimuth_center=0.4)

        wp = proj.project(est, pose)
        assert wp.valid

        dist = math.sqrt((wp.x - pose.x) ** 2 + (wp.y - pose.y) ** 2)
        # dist ≈ range (not exact because camera may have extrinsics offset,
        # but here extrinsics = 0 so it should be exact)
        assert abs(dist - est.range_m) < 1e-5

    def test_two_robots_same_object_same_world_point(self):
        """
        If two robot poses both look at the same physical object,
        they should project to (approximately) the same world point.
        """
        proj = WorldPointProjector(max_range=10.0)

        # Object at world (4.0, 2.0)
        # Robot A at (2.0, 2.0) facing east → object is 2m ahead, az=0
        pose_a  = Pose2D(x=2.0, y=2.0, yaw=0.0)
        est_a   = make_estimate(range_m=2.0, azimuth_center=0.0)

        # Robot B at (4.0, 0.0) facing north → object is 2m ahead (north), az=0
        pose_b  = Pose2D(x=4.0, y=0.0, yaw=math.pi / 2)
        est_b   = make_estimate(range_m=2.0, azimuth_center=0.0)

        wp_a = proj.project(est_a, pose_a)
        wp_b = proj.project(est_b, pose_b)

        assert wp_a.valid and wp_b.valid
        assert abs(wp_a.x - wp_b.x) < 1e-5
        assert abs(wp_a.y - wp_b.y) < 1e-5

    def test_project_point_in_base_convenience(self):
        """project_point_in_base should match project() with identity pose."""
        proj = WorldPointProjector(max_range=10.0)
        est  = make_estimate(range_m=2.0, azimuth_center=math.pi / 6)

        result_base = proj.project_point_in_base(est)
        assert result_base is not None

        identity_pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
        wp = proj.project(est, identity_pose)

        assert abs(result_base[0] - wp.x) < 1e-9
        assert abs(result_base[1] - wp.y) < 1e-9

    def test_project_point_in_base_invalid_returns_none(self):
        proj = WorldPointProjector()
        est  = make_estimate(valid=False)
        assert proj.project_point_in_base(est) is None
