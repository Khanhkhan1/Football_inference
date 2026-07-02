from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DetectionConfig:
    weights_path: str
    confidence_threshold: float
    slice_height: int
    slice_width: int
    overlap_height_ratio: float
    overlap_width_ratio: float
    class_names: dict[int, str]
    roi_polygon: list[list[int]] | None = None


@dataclass
class TrackingConfig:
    track_activation_threshold: float
    lost_track_buffer: int
    minimum_matching_threshold: float
    frame_rate: int


@dataclass
class ClusteringConfig:
    algorithm: str
    eps: float
    min_samples: int
    num_expected_groups: int


@dataclass
class MinimapConfig:
    pitch_length_m: float
    pitch_width_m: float
    source_keypoints: list[tuple[float, float]]
    position: str = "bottom_right"  # one of: top_left, top_right, bottom_left, bottom_right


@dataclass
class PipelineConfig:
    detection: DetectionConfig
    tracking: TrackingConfig
    clustering: ClusteringConfig
    minimap: MinimapConfig


def load_config(path: str | Path = "config.yaml") -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return PipelineConfig(
        detection=DetectionConfig(**raw["detection"]),
        tracking=TrackingConfig(**raw["tracking"]),
        clustering=ClusteringConfig(**raw["clustering"]),
        minimap=MinimapConfig(
            pitch_length_m=raw["minimap"]["pitch_length_m"],
            pitch_width_m=raw["minimap"]["pitch_width_m"],
            source_keypoints=[tuple(p) for p in raw["minimap"]["source_keypoints"]],
            position=raw["minimap"].get("position", "bottom_right"),
        ),
    )
