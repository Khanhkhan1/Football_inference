from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from .config import TrackingConfig
from .detection import Detection


@dataclass
class TrackedDetection(Detection):
    track_id: int


class Tracker:
    """Wraps supervision's ByteTrack to keep a stable id per player across frames.

    `supervision.ByteTrack` is deprecated upstream (slated for removal in
    supervision v0.30) in favor of the separate `trackers` package, but that
    package's PyPI release currently ships only a CLI with no importable
    library code -- revisit this wrapper once it has a usable Python API.
    """

    def __init__(self, config: TrackingConfig):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=config.frame_rate,
        )

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        if not detections:
            self.tracker.update_with_detections(sv.Detections.empty())
            return []

        sv_detections = sv.Detections(
            xyxy=np.array([d.bbox for d in detections], dtype=np.float32),
            confidence=np.array([d.confidence for d in detections], dtype=np.float32),
            class_id=np.array([d.class_id for d in detections], dtype=int),
        )
        class_names = {d.class_id: d.class_name for d in detections}

        tracked = self.tracker.update_with_detections(sv_detections)

        return [
            TrackedDetection(
                bbox=tuple(xyxy),
                class_id=int(class_id),
                class_name=class_names.get(int(class_id), str(class_id)),
                confidence=float(confidence),
                track_id=int(track_id),
            )
            for xyxy, confidence, class_id, track_id in zip(
                tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id
            )
        ]
