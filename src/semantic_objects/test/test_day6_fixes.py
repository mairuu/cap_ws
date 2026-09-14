"""
Tests for the Day 6 (14 Sep 2026) fixes to the June-era modules.

Each test names the defect it guards. The June test suite passed both before
and after the mirror fix, which is exactly why these exist: every window it
used was symmetric about the optical axis.
"""

import math

import pytest

from semantic_objects.lidar_range_extractor import (
    BoundingBox,
    CameraIntrinsics,
    LaserScanData,
    LidarRangeExtractor,
    RangeEstimate,
)
from semantic_objects.world_point_projector import (
    CameraExtrinsics,
    Pose2D,
    WorldPointProjector,
)
from semantic_objects.landmark_store import LandmarkStore
from semantic_objects.world_point_projector import WorldPoint


# ---------------------------------------------------------------------------
# Helpers -- the real C615 numbers, not the 554.0 placeholder
# ---------------------------------------------------------------------------

@pytest.fixture
def c615() -> CameraIntrinsics:
    return CameraIntrinsics(fx=667.874, fy=669.846, cx=321.569, cy=234.502,
                            width=640, height=480)


def full_scan(background: float = 5.0, n_rays: int = 350,
              range_min: float = 0.12, range_max: float = 8.0) -> LaserScanData:
    """An X3-Pro-shaped scan: 350 rays over 360 deg, CCW-positive, REP-103."""
    inc = 2 * math.pi / n_rays
    return LaserScanData(
        angle_min=-math.pi, angle_max=-math.pi + inc * (n_rays - 1),
        angle_increment=inc, range_min=range_min, range_max=range_max,
        ranges=[background] * n_rays,
    )


def put_object(scan: LaserScanData, bearing_deg: float, rng: float,
               half_width_deg: float = 3.0) -> None:
    """Paint an object of the given angular half-width at a REP-103 bearing
    (positive = robot's LEFT)."""
    for i in range(len(scan.ranges)):
        a = math.degrees(scan.angle_at(i))
        if abs(a - bearing_deg) <= half_width_deg:
            scan.ranges[i] = rng


def bbox_at_bearing(cam: CameraIntrinsics, bearing_deg: float,
                    half_width_px: float = 40.0) -> BoundingBox:
    """A bbox whose centre column corresponds to a REP-103 bearing (left
    positive). Image u grows to the RIGHT, so left bearings are u < cx."""
    u = cam.cx - cam.fx * math.tan(math.radians(bearing_deg))
    return BoundingBox(x1=u - half_width_px, y1=200, x2=u + half_width_px, y2=300,
                       class_label="chair", confidence=0.9)


def wp(x, y):
    return WorldPoint(x=x, y=y, range_m=1.0, azimuth_rad=0.0, n_lidar_returns=5, valid=True)


# ---------------------------------------------------------------------------
# 1. The mirrored scan window
# ---------------------------------------------------------------------------

class TestMirror:
    """The June code searched the scan at image azimuths (right-positive);
    the lidar is left-positive. A box on the image's right read rays on the
    robot's left. Found 14 Sep by review; no symmetric test can see it."""

    def test_object_on_the_left_is_ranged_by_a_bbox_on_the_left(self, c615):
        scan = full_scan(background=5.0)
        put_object(scan, bearing_deg=+20.0, rng=1.5)      # robot's LEFT
        ex = LidarRangeExtractor(c615, range_method="min", min_returns=1)
        est = ex.extract_range(bbox_at_bearing(c615, +20.0), scan)
        assert est.valid
        assert est.range_m == pytest.approx(1.5)

    def test_bbox_on_the_right_does_not_see_the_left_object(self, c615):
        scan = full_scan(background=5.0)
        put_object(scan, bearing_deg=+20.0, rng=1.5)      # robot's LEFT
        ex = LidarRangeExtractor(c615, range_method="min", min_returns=1)
        est = ex.extract_range(bbox_at_bearing(c615, -20.0), scan)   # image RIGHT
        assert est.valid
        assert est.range_m == pytest.approx(5.0)          # background, not 1.5

    def test_object_on_the_right_is_ranged_by_a_bbox_on_the_right(self, c615):
        scan = full_scan(background=5.0)
        put_object(scan, bearing_deg=-20.0, rng=1.2)      # robot's RIGHT
        ex = LidarRangeExtractor(c615, range_method="min", min_returns=1)
        assert ex.extract_range(bbox_at_bearing(c615, -20.0), scan).range_m == pytest.approx(1.2)
        assert ex.extract_range(bbox_at_bearing(c615, +20.0), scan).range_m == pytest.approx(5.0)

    def test_camera_yaw_shifts_the_window(self, c615):
        # Camera panned 15 deg LEFT: a bbox on the optical axis looks at +15.
        scan = full_scan(background=5.0)
        put_object(scan, bearing_deg=+15.0, rng=2.0)
        ex = LidarRangeExtractor(c615, range_method="min", min_returns=1,
                                 camera_yaw=math.radians(15.0))
        est = ex.extract_range(bbox_at_bearing(c615, 0.0), scan)
        assert est.range_m == pytest.approx(2.0)
        ex0 = LidarRangeExtractor(c615, range_method="min", min_returns=1)
        assert ex0.extract_range(bbox_at_bearing(c615, 0.0), scan).range_m == pytest.approx(5.0)

    def test_bearing_round_trip_through_projector(self, c615):
        """Extractor + projector together: an object on the robot's left ends
        up at positive y in base_link. The sign flips cancel only if both
        modules agree on the convention."""
        scan = full_scan(background=5.0)
        put_object(scan, bearing_deg=+30.0, rng=2.0)
        ex = LidarRangeExtractor(c615, range_method="min", min_returns=1)
        est = ex.extract_range(bbox_at_bearing(c615, +30.0), scan)
        pt = WorldPointProjector().project(est, Pose2D(0.0, 0.0, 0.0))
        assert pt.valid
        assert pt.x == pytest.approx(2.0 * math.cos(math.radians(30)), abs=0.02)
        assert pt.y == pytest.approx(+2.0 * math.sin(math.radians(30)), abs=0.02)


# ---------------------------------------------------------------------------
# 2. P4 rejection: min_returns and max_spread
# ---------------------------------------------------------------------------

class TestRejection:

    def test_min_returns_rejects_a_thin_window(self, c615):
        scan = full_scan(background=5.0)
        # Only one live ray in the window: everything else is dropout (0.0).
        for i in range(len(scan.ranges)):
            scan.ranges[i] = 0.0
        idx = scan.index_at(0.0)
        scan.ranges[idx] = 2.0
        ex = LidarRangeExtractor(c615, range_method="min", min_returns=3)
        est = ex.extract_range(bbox_at_bearing(c615, 0.0), scan)
        assert not est.valid
        assert est.reason == "min_returns"
        assert est.n_returns == 1

    def test_min_returns_passes_with_enough(self, c615):
        scan = full_scan(background=2.0)
        ex = LidarRangeExtractor(c615, range_method="min", min_returns=3)
        est = ex.extract_range(bbox_at_bearing(c615, 0.0), scan)
        assert est.valid and est.n_returns >= 3

    def test_max_spread_rejects_edge_straddle(self, c615):
        """Design note P4: the window covers a chair edge at 1.5 m AND the
        wall behind at 5 m. 'min' would confidently say 1.5 -- or 5, from the
        next frame. Reject."""
        scan = full_scan(background=5.0)
        put_object(scan, bearing_deg=+3.0, rng=1.5, half_width_deg=1.5)   # half the window
        ex = LidarRangeExtractor(c615, range_method="min", min_returns=1, max_spread=0.5)
        est = ex.extract_range(bbox_at_bearing(c615, 0.0, half_width_px=40), scan)
        assert not est.valid
        assert est.reason == "max_spread"
        assert est.spread_m == pytest.approx(3.5)

    def test_max_spread_passes_a_tight_window(self, c615):
        scan = full_scan(background=5.0)
        put_object(scan, bearing_deg=0.0, rng=1.5, half_width_deg=6.0)     # covers the whole window
        ex = LidarRangeExtractor(c615, range_method="min", min_returns=1, max_spread=0.5)
        est = ex.extract_range(bbox_at_bearing(c615, 0.0), scan)
        assert est.valid
        assert est.spread_m == pytest.approx(0.0)

    def test_no_returns_reason(self, c615):
        scan = full_scan(background=float("inf"))
        ex = LidarRangeExtractor(c615)
        est = ex.extract_range(bbox_at_bearing(c615, 0.0), scan)
        assert not est.valid and est.reason == "no_returns"

    def test_bad_thresholds_rejected(self, c615):
        with pytest.raises(ValueError):
            LidarRangeExtractor(c615, min_returns=0)
        with pytest.raises(ValueError):
            LidarRangeExtractor(c615, max_spread=0.0)


# ---------------------------------------------------------------------------
# 3. Range origin: the camera's ray meets the lidar's circle
# ---------------------------------------------------------------------------

class TestRangeOrigin:

    def make(self, cam=(0.05, -0.03), lidar=(-0.034, 0.0), yaw=0.0):
        return WorldPointProjector(CameraExtrinsics(
            dx=cam[0], dy=cam[1], yaw=yaw, lidar_dx=lidar[0], lidar_dy=lidar[1]))

    @staticmethod
    def est(azimuth_cam_rad: float, r: float) -> RangeEstimate:
        return RangeEstimate(range_m=r, azimuth_center=azimuth_cam_rad,
                             azimuth_left=azimuth_cam_rad, azimuth_right=azimuth_cam_rad,
                             n_returns=5, method="min", valid=True)

    def test_point_is_at_lidar_range_from_the_lidar(self):
        """Whatever the offsets, |P - L| must equal the measured range."""
        proj = self.make()
        pt = proj.project(self.est(math.radians(-20.0), 2.0), Pose2D(0, 0, 0))
        assert pt.valid
        assert math.hypot(pt.x + 0.034, pt.y - 0.0) == pytest.approx(2.0, abs=1e-9)

    def test_point_is_on_the_camera_ray(self):
        proj = self.make()
        az_cam = math.radians(-20.0)          # image-left => robot-left (+20)
        pt = proj.project(self.est(az_cam, 2.0), Pose2D(0, 0, 0))
        bearing = math.atan2(pt.y - (-0.03), pt.x - 0.05)
        assert bearing == pytest.approx(math.radians(20.0), abs=1e-9)

    def test_june_bias_is_gone(self):
        """June: P = camera + r*d, i.e. |P - L| != r by ~9 cm."""
        proj = self.make()
        pt = proj.project(self.est(0.0, 1.0), Pose2D(0, 0, 0))
        june_x = 0.05 + 1.0
        assert abs(pt.x - june_x) > 0.05

    def test_same_origin_reduces_to_june_formula(self):
        proj = WorldPointProjector(CameraExtrinsics(dx=0.1, dy=0.2, yaw=0.0))
        pt = proj.project(self.est(math.radians(-30.0), 2.0), Pose2D(0, 0, 0))
        assert pt.x == pytest.approx(0.1 + 2.0 * math.cos(math.radians(30)))
        assert pt.y == pytest.approx(0.2 + 2.0 * math.sin(math.radians(30)))

    def test_range_shorter_than_offset_is_invalid(self):
        proj = self.make(cam=(0.0, -0.5), lidar=(0.0, 0.5))
        pt = proj.project(self.est(0.0, 0.2), Pose2D(0, 0, 0))
        assert not pt.valid

    def test_lateral_offset_bearing_matters_at_one_metre(self):
        """A camera 3 cm right of the lidar sees an object dead ahead; from
        the lidar that object is 1.7 deg right. The point must land on the
        camera's ray (y = -0.03), not on the lidar's axis (y = 0)."""
        proj = self.make(cam=(0.0, -0.03), lidar=(0.0, 0.0))
        pt = proj.project(self.est(0.0, 1.0), Pose2D(0, 0, 0))
        assert pt.y == pytest.approx(-0.03, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. P5: association on track id
# ---------------------------------------------------------------------------

class TestTrackAssociation:

    def test_same_track_updates_across_a_radius_jump(self):
        store = LandmarkStore(merge_radius=0.5, persist_path=None)
        a = store.observe(wp(1.0, 0.0), "chair", track_id="7", now=100.0)
        b = store.observe(wp(1.8, 0.0), "chair", track_id="7", now=100.1)   # 0.8 m > merge_radius
        assert a.created and not b.created and b.by_track
        assert len(store) == 1

    def test_geometry_alone_would_have_split(self):
        store = LandmarkStore(merge_radius=0.5, persist_path=None)
        store.observe(wp(1.0, 0.0), "chair", now=100.0)
        r = store.observe(wp(1.8, 0.0), "chair", now=100.1)
        assert r.created and len(store) == 2

    def test_class_mismatch_ignores_the_track(self):
        store = LandmarkStore(persist_path=None)
        store.observe(wp(1.0, 0.0), "chair", track_id="1", now=100.0)
        r = store.observe(wp(1.0, 0.0), "cup", track_id="1", now=100.1)
        assert r.created and not r.by_track and len(store) == 2

    def test_restarted_detector_reusing_id_across_the_room(self):
        store = LandmarkStore(track_max_jump=1.0, persist_path=None)
        store.observe(wp(0.0, 0.0), "chair", track_id="1", now=100.0)
        r = store.observe(wp(4.0, 0.0), "chair", track_id="1", now=100.1)
        assert r.created and not r.by_track and len(store) == 2

    def test_binding_expires(self):
        store = LandmarkStore(track_timeout=2.0, merge_radius=0.5, persist_path=None)
        store.observe(wp(1.0, 0.0), "chair", track_id="1", now=100.0)
        r = store.observe(wp(1.8, 0.0), "chair", track_id="1", now=103.0)   # 3 s later
        assert r.created and not r.by_track

    def test_empty_id_uses_geometry(self):
        store = LandmarkStore(merge_radius=0.5, persist_path=None)
        store.observe(wp(1.0, 0.0), "chair", track_id="", now=100.0)
        r = store.observe(wp(1.2, 0.0), "chair", track_id="", now=100.1)
        assert not r.created and not r.by_track

    def test_clear_drops_bindings(self):
        store = LandmarkStore(persist_path=None)
        store.observe(wp(1.0, 0.0), "chair", track_id="1", now=100.0)
        store.clear()
        r = store.observe(wp(1.0, 0.0), "chair", track_id="1", now=100.1)
        assert r.created and not r.by_track

    def test_observe_with_confidence_passes_track(self):
        store = LandmarkStore(merge_radius=0.5, persist_path=None)
        store.observe_with_confidence(wp(1.0, 0.0), "chair", 0.6, track_id="2", now=100.0)
        r = store.observe_with_confidence(wp(1.8, 0.0), "chair", 0.7, track_id="2", now=100.1)
        assert r.by_track and r.landmark.confidence == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 5. Persistence behaviour the node relies on
# ---------------------------------------------------------------------------

class TestPersistenceFlags:

    def test_no_restore_on_start(self, tmp_path):
        path = tmp_path / "landmarks.json"
        s1 = LandmarkStore(persist_path=path)
        s1.observe(wp(1.0, 0.0), "chair")
        assert path.exists()
        s2 = LandmarkStore(persist_path=path, restore_on_start=False)
        assert len(s2) == 0
        s3 = LandmarkStore(persist_path=path, restore_on_start=True)
        assert len(s3) == 1

    def test_autosave_off_writes_only_on_save(self, tmp_path):
        path = tmp_path / "sub" / "landmarks.json"      # parent does not exist
        s = LandmarkStore(persist_path=path, autosave=False)
        s.observe(wp(1.0, 0.0), "chair")
        assert not path.exists() and s.dirty
        s.save()
        assert path.exists() and not s.dirty
        assert not (tmp_path / "sub" / "landmarks.tmp").exists()
        assert not (tmp_path / "sub" / "landmarks.json.tmp").exists()

    def test_expands_tilde(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        s = LandmarkStore(persist_path="~/x/landmarks.json", autosave=False)
        assert str(s.persist_path).startswith(str(tmp_path))
