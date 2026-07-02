from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clustering import TeamAssigner, extract_jersey_color
from .config import PipelineConfig
from .detection import Detector
from .minimap import ViewTransformer
from .tracking import Tracker, TrackedDetection


@dataclass
class PlayerResult:
    track_id: int
    bbox: tuple[float, float, float, float]
    team: int | None
    minimap_position: tuple[float, float]


@dataclass
class FrameResult:
    players: list[PlayerResult]
    ball_detections: list[TrackedDetection]


class FootballPipeline:
    """Ties detection -> tracking -> clustering -> minimap together."""

    def __init__(self, config: PipelineConfig, device: str = "cpu"):
        self.config = config
        self.detector = Detector(config.detection, device=device)
        self.tracker = Tracker(config.tracking)
        self.team_assigner = TeamAssigner(config.clustering)
        self.view_transformer = ViewTransformer(config.minimap)

        self._frame_count = 0

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        detections = self.detector.detect(frame)
        tracked = self.tracker.update(detections)

        players = [d for d in tracked if d.class_name == "player"]
        balls = [d for d in tracked if d.class_name == "ball"]

        for player in players:
            color = extract_jersey_color(frame, player.bbox)
            self.team_assigner.observe(player.track_id, color)

        self._frame_count += 1
        if self._frame_count % 10 == 0:
            self.team_assigner.fit()

        results = []
        for player in players:
            x1, y1, x2, y2 = player.bbox

            foot_point = ((x1 + x2) / 2, y2)
            minimap_position = self.view_transformer.transform_point(foot_point)

            results.append(
                PlayerResult(
                    track_id=player.track_id,
                    bbox=player.bbox,
                    team=self.team_assigner.team_for(player.track_id),
                    minimap_position=minimap_position,
                )
            )

        return FrameResult(players=results, ball_detections=balls)
