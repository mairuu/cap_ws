"""
test_landmark_store.py
-----------------------
Unit tests for LandmarkStore.

Tests are grouped by concern:
  - Creation: first observation makes a landmark
  - Association: class-gating and distance threshold
  - EMA update: position converges correctly
  - Staleness: timeout marking
  - Filtering: confirmed_landmarks() gate
  - Persistence: save / load round-trip
  - Edge cases: empty store, duplicate class at same location, etc.
"""

import json
import math
import time
import uuid
from pathlib import Path

import pytest

from semantic_objects.landmark_store import LandmarkStore, SemanticLandmark
from semantic_objects.world_point_projector import WorldPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_wp(x: float, y: float, r: float = 2.0, n: int = 5) -> WorldPoint:
    return WorldPoint(x=x, y=y, range_m=r, azimuth_rad=0.0, n_lidar_returns=n, valid=True)


def make_store(**kwargs) -> LandmarkStore:
    """Store with no persistence path unless overridden."""
    kwargs.setdefault("persist_path", None)
    return LandmarkStore(**kwargs)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

class TestCreation:

    def test_first_observation_creates_landmark(self):
        store = make_store()
        result = store.observe(make_wp(1.0, 2.0), "chair")

        assert result.created
        assert len(store) == 1
        lm = result.landmark
        assert lm.class_label == "chair"
        assert abs(lm.x - 1.0) < 1e-9
        assert abs(lm.y - 2.0) < 1e-9
        assert lm.seen_count == 1

    def test_new_landmark_gets_unique_id(self):
        store = make_store()
        r1 = store.observe(make_wp(0.0, 0.0), "chair")
        r2 = store.observe(make_wp(5.0, 5.0), "chair")

        assert r1.landmark.id != r2.landmark.id

    def test_id_is_valid_uuid(self):
        store = make_store()
        result = store.observe(make_wp(1.0, 1.0), "bottle")
        # Should not raise
        uuid.UUID(result.landmark.id)

    def test_first_seen_and_last_seen_set_on_creation(self):
        before = time.time()
        store = make_store()
        result = store.observe(make_wp(1.0, 1.0), "chair")
        after = time.time()

        lm = result.landmark
        assert before <= lm.first_seen <= after
        assert before <= lm.last_seen <= after
        assert lm.first_seen == lm.last_seen

    def test_new_landmark_not_stale(self):
        store = make_store()
        result = store.observe(make_wp(1.0, 1.0), "chair")
        assert not result.landmark.stale


# ---------------------------------------------------------------------------
# Association: class gating
# ---------------------------------------------------------------------------

class TestClassGating:

    def test_same_class_same_location_updates_not_creates(self):
        store = make_store(merge_radius=1.0)
        store.observe(make_wp(1.0, 1.0), "chair")
        result = store.observe(make_wp(1.0, 1.0), "chair")

        assert not result.created
        assert len(store) == 1

    def test_different_class_same_location_creates_new(self):
        """Chair and bottle at the same spot are two separate landmarks."""
        store = make_store(merge_radius=1.0)
        store.observe(make_wp(1.0, 1.0), "chair")
        result = store.observe(make_wp(1.0, 1.0), "bottle")

        assert result.created
        assert len(store) == 2

    def test_three_classes_same_location_three_landmarks(self):
        store = make_store(merge_radius=2.0)
        for label in ("chair", "bottle", "couch"):
            store.observe(make_wp(0.0, 0.0), label)
        assert len(store) == 3

    def test_nearest_of_class_ignores_other_classes(self):
        """
        Bottle at (0,0). Chair at (0.1, 0). New chair observation at (0.2, 0).
        The new chair should match the existing chair, not the bottle.
        """
        store = make_store(merge_radius=1.0)
        store.observe(make_wp(0.0, 0.0), "bottle")
        store.observe(make_wp(0.1, 0.0), "chair")
        result = store.observe(make_wp(0.2, 0.0), "chair")

        assert not result.created
        assert len(store) == 2  # bottle + chair, not a third


# ---------------------------------------------------------------------------
# Association: distance threshold
# ---------------------------------------------------------------------------

class TestDistanceThreshold:

    def test_within_merge_radius_updates(self):
        store = make_store(merge_radius=0.5)
        store.observe(make_wp(0.0, 0.0), "chair")
        result = store.observe(make_wp(0.3, 0.0), "chair")  # 0.3m < 0.5m

        assert not result.created
        assert len(store) == 1

    def test_beyond_merge_radius_creates(self):
        store = make_store(merge_radius=0.5)
        store.observe(make_wp(0.0, 0.0), "chair")
        result = store.observe(make_wp(1.0, 0.0), "chair")  # 1.0m > 0.5m

        assert result.created
        assert len(store) == 2

    def test_exactly_at_merge_radius_updates(self):
        """Boundary: distance == merge_radius should update (<=, not <)."""
        store = make_store(merge_radius=0.5)
        store.observe(make_wp(0.0, 0.0), "chair")
        result = store.observe(make_wp(0.5, 0.0), "chair")

        assert not result.created

    def test_distance_returned_in_result(self):
        store = make_store(merge_radius=1.0)
        store.observe(make_wp(0.0, 0.0), "chair")
        result = store.observe(make_wp(0.3, 0.4), "chair")  # dist = 0.5

        assert not result.created
        assert abs(result.distance - 0.5) < 1e-6

    def test_new_landmark_distance_is_zero(self):
        store = make_store()
        result = store.observe(make_wp(5.0, 5.0), "chair")
        assert result.created
        assert result.distance == 0.0

    def test_always_matches_nearest_not_first(self):
        """
        Two chairs: one at (0,0), one at (3,0).
        New observation at (2.8, 0) → should match (3,0), not (0,0).
        """
        store = make_store(merge_radius=1.0)
        r1 = store.observe(make_wp(0.0, 0.0), "chair")
        r2 = store.observe(make_wp(3.0, 0.0), "chair")
        result = store.observe(make_wp(2.8, 0.0), "chair")

        assert not result.created
        assert result.landmark.id == r2.landmark.id


# ---------------------------------------------------------------------------
# EMA position update
# ---------------------------------------------------------------------------

class TestEMAUpdate:

    def test_position_moves_toward_new_observation(self):
        """After update, position should be between old and new."""
        store = make_store(ema_alpha=0.3, merge_radius=5.0)
        store.observe(make_wp(0.0, 0.0), "chair")
        result = store.observe(make_wp(1.0, 0.0), "chair")

        lm = result.landmark
        # new_x = 0.3 * 1.0 + 0.7 * 0.0 = 0.3
        assert abs(lm.x - 0.3) < 1e-9
        assert abs(lm.y - 0.0) < 1e-9

    def test_seen_count_increments(self):
        store = make_store(merge_radius=5.0)
        store.observe(make_wp(0.0, 0.0), "chair")
        store.observe(make_wp(0.1, 0.0), "chair")
        store.observe(make_wp(0.1, 0.0), "chair")

        lm = store.all_landmarks()[0]
        assert lm.seen_count == 3

    def test_repeated_same_observation_converges(self):
        """
        Repeated observation at (2.0, 0.0) from initial (0.0, 0.0).
        With α=0.3, after N steps: x = 2*(1 - 0.7^N).
        After many steps this converges to 2.0.
        """
        store = make_store(ema_alpha=0.3, merge_radius=10.0)
        store.observe(make_wp(0.0, 0.0), "chair")

        for _ in range(30):
            store.observe(make_wp(2.0, 0.0), "chair")

        lm = store.all_landmarks()[0]
        assert abs(lm.x - 2.0) < 0.01  # converged to within 1 cm

    def test_ema_alpha_controls_smoothing(self):
        """Higher alpha → position changes faster."""
        store_fast = make_store(ema_alpha=0.9, merge_radius=10.0)
        store_slow = make_store(ema_alpha=0.1, merge_radius=10.0)

        for store in (store_fast, store_slow):
            store.observe(make_wp(0.0, 0.0), "chair")
            store.observe(make_wp(10.0, 0.0), "chair")

        fast_x = store_fast.all_landmarks()[0].x
        slow_x = store_slow.all_landmarks()[0].x

        # Fast (α=0.9): x = 0.9*10 + 0.1*0 = 9.0
        # Slow (α=0.1): x = 0.1*10 + 0.9*0 = 1.0
        assert fast_x > slow_x
        assert abs(fast_x - 9.0) < 1e-9
        assert abs(slow_x - 1.0) < 1e-9

    def test_last_seen_updates_on_each_observation(self):
        store = make_store(merge_radius=5.0)
        store.observe(make_wp(0.0, 0.0), "chair")
        first_time = store.all_landmarks()[0].last_seen

        time.sleep(0.01)
        store.observe(make_wp(0.1, 0.0), "chair")
        second_time = store.all_landmarks()[0].last_seen

        assert second_time > first_time

    def test_first_seen_does_not_change_on_update(self):
        store = make_store(merge_radius=5.0)
        store.observe(make_wp(0.0, 0.0), "chair")
        t_first = store.all_landmarks()[0].first_seen

        time.sleep(0.01)
        store.observe(make_wp(0.0, 0.0), "chair")
        t_first_after = store.all_landmarks()[0].first_seen

        assert t_first == t_first_after


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

class TestStaleness:

    def test_fresh_landmark_not_stale(self):
        store = make_store(stale_timeout=60.0)
        store.observe(make_wp(1.0, 1.0), "chair")
        store.mark_stale(now=time.time())

        assert not store.all_landmarks()[0].stale

    def test_old_landmark_marked_stale(self):
        store = make_store(stale_timeout=60.0)
        store.observe(make_wp(1.0, 1.0), "chair")

        # Simulate time passing: mark_stale called 120s in the future
        far_future = time.time() + 120.0
        newly_stale = store.mark_stale(now=far_future)

        assert store.all_landmarks()[0].stale
        assert len(newly_stale) == 1

    def test_re_observation_clears_stale(self):
        store = make_store(stale_timeout=60.0, merge_radius=5.0)
        store.observe(make_wp(1.0, 1.0), "chair")
        store.mark_stale(now=time.time() + 120.0)
        assert store.all_landmarks()[0].stale

        store.observe(make_wp(1.0, 1.0), "chair")
        assert not store.all_landmarks()[0].stale

    def test_mark_stale_returns_only_newly_stale(self):
        store = make_store(stale_timeout=60.0)
        store.observe(make_wp(0.0, 0.0), "chair")
        store.observe(make_wp(5.0, 0.0), "bottle")

        future = time.time() + 120.0
        newly = store.mark_stale(now=future)
        assert len(newly) == 2

        # Second call — already stale, should not appear again
        newly2 = store.mark_stale(now=future + 10)
        assert len(newly2) == 0


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFiltering:

    def test_confirmed_landmarks_requires_min_seen(self):
        store = make_store(min_seen_to_publish=2, merge_radius=5.0)
        store.observe(make_wp(0.0, 0.0), "chair")

        assert len(store.confirmed_landmarks()) == 0  # only seen once

        store.observe(make_wp(0.0, 0.0), "chair")
        assert len(store.confirmed_landmarks()) == 1  # now seen twice

    def test_all_landmarks_includes_unconfirmed(self):
        store = make_store(min_seen_to_publish=3, merge_radius=5.0)
        store.observe(make_wp(0.0, 0.0), "chair")

        assert len(store.all_landmarks()) == 1
        assert len(store.confirmed_landmarks()) == 0

    def test_all_landmarks_and_confirmed_agree_when_all_confirmed(self):
        store = make_store(min_seen_to_publish=1)
        store.observe(make_wp(0.0, 0.0), "chair")
        store.observe(make_wp(5.0, 0.0), "bottle")

        assert len(store.all_landmarks()) == len(store.confirmed_landmarks()) == 2


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

class TestCRUD:

    def test_get_returns_landmark_by_id(self):
        store = make_store()
        result = store.observe(make_wp(1.0, 1.0), "chair")
        lm = store.get(result.landmark.id)
        assert lm is not None
        assert lm.id == result.landmark.id

    def test_get_returns_none_for_unknown_id(self):
        store = make_store()
        assert store.get("nonexistent-id") is None

    def test_remove_deletes_landmark(self):
        store = make_store()
        result = store.observe(make_wp(1.0, 1.0), "chair")
        removed = store.remove(result.landmark.id)

        assert removed
        assert len(store) == 0

    def test_remove_returns_false_for_unknown_id(self):
        store = make_store()
        assert not store.remove("no-such-id")

    def test_clear_empties_store(self):
        store = make_store()
        store.observe(make_wp(0.0, 0.0), "chair")
        store.observe(make_wp(1.0, 0.0), "bottle")
        store.clear()

        assert len(store) == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "landmarks.json"
        store = LandmarkStore(persist_path=path)
        store.observe(make_wp(1.0, 2.0), "chair")

        assert path.exists()

    def test_load_restores_landmarks(self, tmp_path):
        path = tmp_path / "landmarks.json"

        # Session 1: observe and persist
        store1 = LandmarkStore(persist_path=path, merge_radius=5.0)
        store1.observe(make_wp(1.0, 2.0), "chair")
        store1.observe(make_wp(3.0, 4.0), "bottle")

        # Session 2: load from file
        store2 = LandmarkStore(persist_path=path)
        assert len(store2) == 2

        labels = {lm.class_label for lm in store2.all_landmarks()}
        assert labels == {"chair", "bottle"}

    def test_positions_survive_round_trip(self, tmp_path):
        path = tmp_path / "landmarks.json"

        store1 = LandmarkStore(persist_path=path)
        store1.observe(make_wp(1.23, 4.56), "chair")

        store2 = LandmarkStore(persist_path=path)
        lm = store2.all_landmarks()[0]
        assert abs(lm.x - 1.23) < 1e-6
        assert abs(lm.y - 4.56) < 1e-6

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / "landmarks.json"
        path.write_text("{ not valid json !!!")

        # Should not raise, should start with empty store
        store = LandmarkStore(persist_path=path)
        assert len(store) == 0

    def test_seen_count_survives_round_trip(self, tmp_path):
        path = tmp_path / "landmarks.json"

        store1 = LandmarkStore(persist_path=path, merge_radius=5.0)
        for _ in range(5):
            store1.observe(make_wp(1.0, 1.0), "chair")

        store2 = LandmarkStore(persist_path=path)
        assert store2.all_landmarks()[0].seen_count == 5

    def test_save_is_atomic(self, tmp_path):
        """
        The .tmp file should not remain after save.
        """
        path = tmp_path / "landmarks.json"
        store = LandmarkStore(persist_path=path)
        store.observe(make_wp(1.0, 1.0), "chair")

        tmp = path.with_suffix(".tmp")
        assert not tmp.exists()


# ---------------------------------------------------------------------------
# Confidence tracking
# ---------------------------------------------------------------------------

class TestConfidence:

    def test_observe_with_confidence_sets_initial(self):
        store = make_store()
        result = store.observe_with_confidence(make_wp(1.0, 1.0), "chair", 0.85)

        assert abs(result.landmark.confidence - 0.85) < 1e-9

    def test_confidence_takes_max_on_update(self):
        store = make_store(merge_radius=5.0)
        store.observe_with_confidence(make_wp(0.0, 0.0), "chair", 0.6)
        store.observe_with_confidence(make_wp(0.1, 0.0), "chair", 0.9)
        store.observe_with_confidence(make_wp(0.1, 0.0), "chair", 0.7)

        lm = store.all_landmarks()[0]
        assert abs(lm.confidence - 0.9) < 1e-9  # max of 0.6, 0.9, 0.7


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_invalid_ema_alpha_raises(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            LandmarkStore(ema_alpha=0.0)
        with pytest.raises(ValueError, match="ema_alpha"):
            LandmarkStore(ema_alpha=1.0)
        with pytest.raises(ValueError, match="ema_alpha"):
            LandmarkStore(ema_alpha=1.5)

    def test_invalid_merge_radius_raises(self):
        with pytest.raises(ValueError, match="merge_radius"):
            LandmarkStore(merge_radius=0.0)
        with pytest.raises(ValueError, match="merge_radius"):
            LandmarkStore(merge_radius=-1.0)
