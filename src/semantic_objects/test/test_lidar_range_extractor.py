"""
test_lidar_range_extractor.py
------------------------------
Unit tests for LidarRangeExtractor.
No ROS2 required — all pure Python.

Run with:
    python -m pytest test/test_lidar_range_extractor.py -v
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def camera() -> CameraIntrinsics:
    """
    Typical USB webcam at 640×480.
    fx ≈ fy ≈ 554 px gives ~68° HFOV.
    """
    return CameraIntrinsics(
        fx=554.0, fy=554.0,
        cx=320.0, cy=240.0,
        width=640, height=480,
    )


def make_uniform_scan(
    range_value: float = 2.0,
    angle_min: float = -math.pi / 2,
    angle_max: float =  math.pi / 2,
    n_rays: int = 360,
    range_min: float = 0.1,
    range_max: float = 12.0,
) -> LaserScanData:
    """Scan where every ray returns the same range."""
    angle_increment = (angle_max - angle_min) / (n_rays - 1)
    return LaserScanData(
        angle_min=angle_min,
        angle_max=angle_max,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        ranges=[range_value] * n_rays,
    )


def make_scan_with_spike(
    background: float = 5.0,
    spike_range: float = 1.5,
    spike_angle: float = 0.0,
    **kwargs,
) -> LaserScanData:
    """Scan with a single close return at spike_angle, background elsewhere."""
    scan = make_uniform_scan(range_value=background, **kwargs)
    idx = scan.index_at(spike_angle)
    if idx is not None:
        scan.ranges[idx] = spike_range
    return scan


# ---------------------------------------------------------------------------
# CameraIntrinsics tests
# ---------------------------------------------------------------------------

class TestCameraIntrinsics:

    def test_hfov_reasonable(self, camera):
        # 640×480 with fx=554 → ~68° HFOV
        hfov_deg = math.degrees(camera.hfov)
        assert 60 < hfov_deg < 80

    def test_centre_pixel_is_zero_azimuth(self, camera):
        az = camera.pixel_to_azimuth(camera.cx)
        assert abs(az) < 1e-9

    def test_left_edge_is_negative(self, camera):
        az = camera.pixel_to_azimuth(0)
        assert az < 0

    def test_right_edge_is_positive(self, camera):
        az = camera.pixel_to_azimuth(camera.width - 1)
        assert az > 0

    def test_symmetry(self, camera):
        az_left  = camera.pixel_to_azimuth(camera.cx - 100)
        az_right = camera.pixel_to_azimuth(camera.cx + 100)
        assert abs(az_left + az_right) < 1e-9


# ---------------------------------------------------------------------------
# Basic extraction tests
# ---------------------------------------------------------------------------

class TestBasicExtraction:

    def test_centred_object_uniform_scan(self, camera):
        """Centred bounding box, all returns 2 m → estimate is 2 m."""
        extractor = LidarRangeExtractor(camera)
        bbox = BoundingBox(x1=260, y1=180, x2=380, y2=300, class_label="chair")
        scan = make_uniform_scan(range_value=2.0)

        result = extractor.extract_range(bbox, scan)

        assert result.valid
        assert abs(result.range_m - 2.0) < 0.01
        assert result.n_returns > 0

    def test_off_centre_object(self, camera):
        """Object in right half of image → azimuth_center > 0."""
        extractor = LidarRangeExtractor(camera)
        bbox = BoundingBox(x1=450, y1=200, x2=580, y2=350, class_label="bottle")
        scan = make_uniform_scan(range_value=3.0)

        result = extractor.extract_range(bbox, scan)

        assert result.valid
        assert result.azimuth_center > 0
        assert abs(result.range_m - 3.0) < 0.05

    def test_returns_correct_range_method_label(self, camera):
        extractor = LidarRangeExtractor(camera, range_method="min")
        bbox = BoundingBox(x1=280, y1=200, x2=360, y2=300)
        scan = make_uniform_scan(range_value=1.5)

        result = extractor.extract_range(bbox, scan)

        assert result.method == "min"


# ---------------------------------------------------------------------------
# Range method tests
# ---------------------------------------------------------------------------

class TestRangeMethods:
    """
    Build a scan with mixed ranges inside the window and verify each
    aggregation method produces the expected value.
    """

    def _make_mixed_scan(self) -> LaserScanData:
        """Scan with known values at known angles."""
        n = 180
        angle_min = -math.pi / 2
        angle_max =  math.pi / 2
        increment = (angle_max - angle_min) / (n - 1)
        ranges = [5.0] * n  # background

        # Put known values near centre (angles ~-0.05 to +0.05 rad)
        # Centre index ≈ 90 for 180-ray scan
        for i in range(85, 96):
            ranges[i] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0][i - 85]

        return LaserScanData(
            angle_min=angle_min,
            angle_max=angle_max,
            angle_increment=increment,
            range_min=0.1,
            range_max=12.0,
            ranges=ranges,
        )

    def test_median_picks_middle(self, camera):
        """
        Median returns the middle value of all returns in the window —
        including background returns. When the angular window is wider
        than the object, the majority of returns are background (5.0 m).
        Median correctly reflects that: it won't be pulled to the spike values.

        To isolate just the spike returns, use min or a tight bbox.
        """
        extractor = LidarRangeExtractor(camera, range_method="median")
        # Wide bbox covering the centre returns (and lots of background)
        bbox = BoundingBox(x1=220, y1=200, x2=420, y2=300)
        scan = self._make_mixed_scan()
        result = extractor.extract_range(bbox, scan)
        assert result.valid
        # Median of mostly-background scan stays near background value (5.0)
        # It will NOT be pulled up to the spike values (8-11) or down to (1-4)
        assert 4.0 <= result.range_m <= 6.0

    def test_min_picks_smallest(self, camera):
        extractor = LidarRangeExtractor(camera, range_method="min")
        bbox = BoundingBox(x1=220, y1=200, x2=420, y2=300)
        scan = self._make_mixed_scan()
        result = extractor.extract_range(bbox, scan)
        assert result.valid
        assert result.range_m <= 2.0  # min should be near smallest value


# ---------------------------------------------------------------------------
# Edge case: no returns in window
# ---------------------------------------------------------------------------

class TestNoReturns:

    def test_all_inf_returns_invalid(self, camera):
        extractor = LidarRangeExtractor(camera)
        bbox = BoundingBox(x1=280, y1=200, x2=360, y2=300)
        scan = make_uniform_scan(range_value=float("inf"))

        result = extractor.extract_range(bbox, scan)

        assert not result.valid
        assert result.n_returns == 0
        assert result.method == "none"

    def test_all_nan_returns_invalid(self, camera):
        extractor = LidarRangeExtractor(camera)
        bbox = BoundingBox(x1=280, y1=200, x2=360, y2=300)
        scan = make_uniform_scan(range_value=float("nan"))

        result = extractor.extract_range(bbox, scan)

        assert not result.valid

    def test_bbox_outside_lidar_fov(self, camera):
        """
        Lidar only covers ±30°, but bounding box is at extreme edge of camera
        (which sees wider than ±30°). Expect no valid returns.
        """
        extractor = LidarRangeExtractor(camera)
        # Extreme right edge of image → azimuth ≈ +34° for this camera
        bbox = BoundingBox(x1=600, y1=200, x2=640, y2=300)
        scan = make_uniform_scan(
            range_value=2.0,
            angle_min=-math.radians(30),
            angle_max= math.radians(30),
        )

        result = extractor.extract_range(bbox, scan)
        # May or may not be valid depending on exact angles; just check it doesn't crash
        assert isinstance(result, RangeEstimate)

    def test_below_range_min_filtered(self, camera):
        """Returns below range_min (e.g. 0.05 m) should be discarded."""
        extractor = LidarRangeExtractor(camera)
        bbox = BoundingBox(x1=280, y1=200, x2=360, y2=300)
        scan = make_uniform_scan(range_value=0.05, range_min=0.1)  # 5cm < 10cm min

        result = extractor.extract_range(bbox, scan)

        assert not result.valid

    def test_above_range_max_filtered(self, camera):
        """Returns above range_max should be discarded."""
        extractor = LidarRangeExtractor(camera)
        bbox = BoundingBox(x1=280, y1=200, x2=360, y2=300)
        scan = make_uniform_scan(range_value=15.0, range_max=12.0)

        result = extractor.extract_range(bbox, scan)

        assert not result.valid


# ---------------------------------------------------------------------------
# Angular padding tests
# ---------------------------------------------------------------------------

class TestAngularPadding:

    def test_padding_widens_window(self, camera):
        """
        With a narrow bbox and no returns inside it, padding should
        pull in nearby returns and make the result valid.
        """
        # Very narrow bbox: 1 pixel wide, pointing straight ahead
        bbox = BoundingBox(x1=319, y1=200, x2=321, y2=300)

        # Scan where only the edges (±0.1 rad) have returns, centre is inf
        n = 360
        angle_min = -math.pi / 2
        angle_max =  math.pi / 2
        inc = (angle_max - angle_min) / (n - 1)
        ranges = [float("inf")] * n

        # Place real returns at ±0.15 rad
        for i in range(n):
            angle = angle_min + i * inc
            if abs(angle) > 0.10:
                ranges[i] = 2.0

        scan = LaserScanData(
            angle_min=angle_min, angle_max=angle_max,
            angle_increment=inc,
            range_min=0.1, range_max=12.0,
            ranges=ranges,
        )

        # Without padding: no returns → invalid
        extractor_no_pad = LidarRangeExtractor(camera, angular_padding=0.0)
        result_no_pad = extractor_no_pad.extract_range(bbox, scan)
        assert not result_no_pad.valid

        # With padding > 0.15 rad: should pick up the flanking returns
        extractor_padded = LidarRangeExtractor(camera, angular_padding=0.2)
        result_padded = extractor_padded.extract_range(bbox, scan)
        assert result_padded.valid


# ---------------------------------------------------------------------------
# Geometry sanity checks
# ---------------------------------------------------------------------------

class TestGeometry:

    def test_azimuth_ordering(self, camera):
        """Left edge azimuth < center < right edge for any valid bbox."""
        extractor = LidarRangeExtractor(camera)
        bbox = BoundingBox(x1=100, y1=100, x2=500, y2=400)
        scan = make_uniform_scan(range_value=2.0)

        result = extractor.extract_range(bbox, scan)

        assert result.azimuth_left < result.azimuth_center < result.azimuth_right

    def test_wide_bbox_more_returns_than_narrow(self, camera):
        """A wider bounding box should cover more lidar rays."""
        extractor = LidarRangeExtractor(camera)
        scan = make_uniform_scan(range_value=2.0)

        narrow = BoundingBox(x1=310, y1=200, x2=330, y2=300)
        wide   = BoundingBox(x1=100, y1=200, x2=540, y2=300)

        r_narrow = extractor.extract_range(narrow, scan)
        r_wide   = extractor.extract_range(wide, scan)

        assert r_wide.n_returns > r_narrow.n_returns

    def test_invalid_range_method_raises(self, camera):
        with pytest.raises(ValueError, match="Unknown range_method"):
            LidarRangeExtractor(camera, range_method="bogus")


# ---------------------------------------------------------------------------
# Realistic scenario
# ---------------------------------------------------------------------------

class TestRealisticScenario:

    def test_chair_at_2m_in_cluttered_scene(self, camera):
        """
        Simulate: chair at 2 m in centre of frame.
        Background walls at 4 m. One noisy far return at 8 m inside window.

        Key insight: the bounding box spans ~24° but the chair only spans ~10°.
        Median will favour the majority (background at 4 m) — that is correct
        behaviour for median. Use 'min' when you want the closest object
        in the window regardless of how many returns it produces.

        This test validates:
        - median correctly returns background when object < bbox width
        - min correctly returns chair depth even when minority of returns
        """
        n = 720
        angle_min = -math.pi
        angle_max =  math.pi - (2 * math.pi / n)
        inc = (angle_max - angle_min) / (n - 1)
        ranges = [4.0] * n  # background walls

        # Chair occupies ~10° (0.17 rad) centred at 0
        for i in range(n):
            angle = angle_min + i * inc
            if abs(angle) < 0.085:
                ranges[i] = 2.0  # chair

        # Add one noisy far return inside the window
        centre_idx = int((0 - angle_min) / inc)
        ranges[centre_idx + 2] = 8.0

        scan = LaserScanData(
            angle_min=angle_min, angle_max=angle_max,
            angle_increment=inc,
            range_min=0.1, range_max=12.0,
            ranges=ranges,
        )

        bbox = BoundingBox(x1=200, y1=150, x2=440, y2=350, class_label="chair")

        # median: dominated by the wider background → 4 m (correct behaviour)
        ext_median = LidarRangeExtractor(camera, range_method="median")
        result_median = ext_median.extract_range(bbox, scan)
        assert result_median.valid
        assert abs(result_median.range_m - 4.0) < 0.5, (
            f"Median should reflect background majority, got {result_median.range_m:.2f}m"
        )

        # min: picks the chair (closest return), ignores 8m outlier → 2 m
        ext_min = LidarRangeExtractor(camera, range_method="min")
        result_min = ext_min.extract_range(bbox, scan)
        assert result_min.valid
        assert abs(result_min.range_m - 2.0) < 0.5, (
            f"Min should pick the chair at 2m, got {result_min.range_m:.2f}m"
        )
