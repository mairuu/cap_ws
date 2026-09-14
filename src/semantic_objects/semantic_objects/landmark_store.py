"""
landmark_store.py
-----------------
Step 4: Semantic memory.

Maintains a collection of SemanticLandmark objects. For each new
WorldPoint observation, decides whether to create a new landmark or
update an existing one, then applies an EMA position fusion.

Responsibilities
----------------
  • Associate incoming observations with existing landmarks
    (class-gated nearest-neighbour)
  • Create new landmarks when no match is found
  • Update existing landmarks via EMA position smoothing
  • Age / mark stale landmarks
  • Persist to / restore from JSON

No ROS2 imports. Pure Python.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .world_point_projector import WorldPoint


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SemanticLandmark:
    """
    A persistent semantic landmark in the map frame.

    Position is the EMA-smoothed estimate of where the object is.
    Confidence reflects detection quality and observation count.
    """
    id: str                   # stable UUID across sessions
    class_label: str          # e.g. "chair", "bottle"
    x: float                  # map frame, metres (EMA-smoothed)
    y: float                  # map frame, metres (EMA-smoothed)
    confidence: float         # 0.0 – 1.0
    seen_count: int           # total number of fused observations
    first_seen: float         # Unix timestamp
    last_seen: float          # Unix timestamp
    stale: bool = False       # True if not seen for > stale_timeout seconds

    def distance_to(self, x: float, y: float) -> float:
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticLandmark":
        return cls(**d)


# ---------------------------------------------------------------------------
# Association result
# ---------------------------------------------------------------------------

@dataclass
class AssociationResult:
    landmark: SemanticLandmark
    created: bool              # True = new landmark, False = existing updated
    distance: float            # distance from observation to landmark (0 if new)


# ---------------------------------------------------------------------------
# Landmark store
# ---------------------------------------------------------------------------

class LandmarkStore:
    """
    In-memory semantic landmark store with JSON persistence.

    Parameters
    ----------
    merge_radius : float
        Max distance (metres) to associate an observation with an existing
        landmark of the same class. Beyond this → new landmark.
        Default: 0.5 m (good for indoor objects).
    ema_alpha : float
        EMA weight for new observations (0 < α < 1).
        Higher α = new observations dominate (faster to update, noisier).
        Lower α = old position dominates (slower to update, smoother).
        Default: 0.3 — lets the position converge over ~5–10 observations.
    min_seen_to_publish : int
        Landmarks with fewer observations than this are not returned by
        confirmed_landmarks(). Filters one-shot false positives.
        Default: 2.
    stale_timeout : float
        Seconds after last_seen before a landmark is marked stale.
        Stale landmarks stay in memory but are flagged for the UI.
        Default: 300.0 (5 minutes).
    persist_path : str | Path | None
        If set, load from this JSON file on init and save on every update.
        None = in-memory only (useful for tests).
    """

    def __init__(
        self,
        merge_radius: float = 0.5,
        ema_alpha: float = 0.3,
        min_seen_to_publish: float = 2,
        stale_timeout: float = 300.0,
        persist_path: Optional[str | Path] = None,
    ):
        if not (0.0 < ema_alpha < 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1), got {ema_alpha}")
        if merge_radius <= 0:
            raise ValueError(f"merge_radius must be > 0, got {merge_radius}")

        self.merge_radius = merge_radius
        self.ema_alpha = ema_alpha
        self.min_seen_to_publish = min_seen_to_publish
        self.stale_timeout = stale_timeout
        self.persist_path = Path(persist_path) if persist_path else None

        # Primary store: id → landmark
        self._landmarks: dict[str, SemanticLandmark] = {}

        if self.persist_path and self.persist_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, world_point: WorldPoint, class_label: str) -> AssociationResult:
        """
        Main entry point. Called once per valid detection.

        1. Find nearest existing landmark of the same class.
        2. If within merge_radius → update (EMA position, confidence, staleness).
        3. Otherwise → create new landmark.
        4. Persist if a path is configured.

        Returns an AssociationResult with the landmark and whether it was created.
        """
        now = time.time()
        nearest, dist = self._nearest_of_class(class_label, world_point.x, world_point.y)

        if nearest is not None and dist <= self.merge_radius:
            self._update(nearest, world_point, now)
            result = AssociationResult(landmark=nearest, created=False, distance=dist)
        else:
            landmark = self._create(class_label, world_point, now)
            result = AssociationResult(landmark=landmark, created=True, distance=0.0)

        if self.persist_path:
            self._save()

        return result

    def all_landmarks(self) -> list[SemanticLandmark]:
        """All landmarks, including unconfirmed and stale ones."""
        return list(self._landmarks.values())

    def confirmed_landmarks(self) -> list[SemanticLandmark]:
        """
        Landmarks seen at least min_seen_to_publish times.
        These are the ones worth showing on the map.
        """
        return [
            lm for lm in self._landmarks.values()
            if lm.seen_count >= self.min_seen_to_publish
        ]

    def get(self, landmark_id: str) -> Optional[SemanticLandmark]:
        return self._landmarks.get(landmark_id)

    def remove(self, landmark_id: str) -> bool:
        """Remove a landmark by ID. Returns True if it existed."""
        if landmark_id in self._landmarks:
            del self._landmarks[landmark_id]
            if self.persist_path:
                self._save()
            return True
        return False

    def clear(self) -> None:
        """Wipe all landmarks (useful for debug / re-mapping)."""
        self._landmarks.clear()
        if self.persist_path:
            self._save()

    def mark_stale(self, now: Optional[float] = None) -> list[str]:
        """
        Mark any landmark not seen within stale_timeout as stale.
        Returns list of IDs that were newly marked.
        Call periodically (e.g. every 30 s) from the ROS node.
        """
        if now is None:
            now = time.time()
        newly_stale = []
        for lm in self._landmarks.values():
            was_stale = lm.stale
            lm.stale = (now - lm.last_seen) > self.stale_timeout
            if lm.stale and not was_stale:
                newly_stale.append(lm.id)
        return newly_stale

    def __len__(self) -> int:
        return len(self._landmarks)

    # ------------------------------------------------------------------
    # Internal: association
    # ------------------------------------------------------------------

    def _nearest_of_class(
        self, class_label: str, x: float, y: float
    ) -> tuple[Optional[SemanticLandmark], float]:
        """
        Find the nearest landmark with the given class_label.
        Returns (landmark, distance) or (None, inf) if none exist.

        Class-gating is intentional: a chair and a bottle at the same
        position are distinct objects and must not be merged.
        """
        best: Optional[SemanticLandmark] = None
        best_dist = float("inf")

        for lm in self._landmarks.values():
            if lm.class_label != class_label:
                continue
            d = lm.distance_to(x, y)
            if d < best_dist:
                best_dist = d
                best = lm

        return best, best_dist

    # ------------------------------------------------------------------
    # Internal: create / update
    # ------------------------------------------------------------------

    def _create(
        self, class_label: str, wp: WorldPoint, now: float
    ) -> SemanticLandmark:
        lm = SemanticLandmark(
            id=str(uuid.uuid4()),
            class_label=class_label,
            x=wp.x,
            y=wp.y,
            confidence=wp.range_m,   # placeholder; caller can set real confidence
            seen_count=1,
            first_seen=now,
            last_seen=now,
            stale=False,
        )
        # Store confidence properly — WorldPoint doesn't carry detection
        # confidence, so initialise to a sensible default.
        # The ROS node will pass in the YOLO confidence via observe_with_confidence().
        lm.confidence = 0.5
        self._landmarks[lm.id] = lm
        return lm

    def _update(
        self, lm: SemanticLandmark, wp: WorldPoint, now: float
    ) -> None:
        """
        EMA position update + metadata refresh.

        Position: x_new = α * obs + (1-α) * x_old
        This weights recent observations less than the accumulated history,
        letting the estimate converge toward the true position over time.

        Why EMA and not a simple mean?
        - Simple mean would require storing the sum, breaking persistence.
        - EMA is O(1) and gives more weight to earlier (closer) observations
          naturally as seen_count grows and new observations get averaged in.
        - α = 0.3 means each new observation contributes 30%, so after
          ~5 observations the estimate is stable (0.7^5 ≈ 2% residual).
        """
        α = self.ema_alpha
        lm.x = α * wp.x + (1 - α) * lm.x
        lm.y = α * wp.y + (1 - α) * lm.y
        lm.seen_count += 1
        lm.last_seen = now
        lm.stale = False   # un-stale on re-observation

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        assert self.persist_path is not None
        data = {
            "version": 1,
            "landmarks": [lm.to_dict() for lm in self._landmarks.values()],
        }
        # Write to temp file then rename for atomicity
        tmp = self.persist_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.persist_path)

    def _load(self) -> None:
        assert self.persist_path is not None
        try:
            data = json.loads(self.persist_path.read_text())
            for d in data.get("landmarks", []):
                lm = SemanticLandmark.from_dict(d)
                self._landmarks[lm.id] = lm
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Corrupt file — start fresh rather than crash the robot
            print(f"[LandmarkStore] WARNING: could not load {self.persist_path}: {e}")
            self._landmarks = {}

    # ------------------------------------------------------------------
    # Extended observe with explicit confidence
    # ------------------------------------------------------------------

    def observe_with_confidence(
        self,
        world_point: WorldPoint,
        class_label: str,
        detection_confidence: float,
    ) -> AssociationResult:
        """
        Like observe(), but also tracks YOLO detection confidence.
        Confidence is updated as max(existing, new) — we keep the
        best evidence seen, not the average.
        """
        result = self.observe(world_point, class_label)
        lm = result.landmark
        if result.created:
            lm.confidence = detection_confidence
        else:
            lm.confidence = max(lm.confidence, detection_confidence)
        return result
