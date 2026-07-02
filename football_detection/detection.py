from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from .config import DetectionConfig


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float


class Detector:
    """YOLO detector run through SAHI's sliced inference for small/far objects."""

    def __init__(self, config: DetectionConfig, device: str = "cpu"):
        self.config = config
        self.roi_polygon = (
            np.array(config.roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
            if config.roi_polygon
            else None
        )
        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=config.weights_path,
            confidence_threshold=config.confidence_threshold,
            device=device,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = get_sliced_prediction(
            frame,
            self.model,
            slice_height=self.config.slice_height,
            slice_width=self.config.slice_width,
            overlap_height_ratio=self.config.overlap_height_ratio,
            overlap_width_ratio=self.config.overlap_width_ratio,
            verbose=0,
        )

        detections = []
        for pred in result.object_prediction_list:
            class_id = pred.category.id
            class_name = self.config.class_names.get(class_id, pred.category.name)
            x1, y1, x2, y2 = pred.bbox.to_xyxy()
            if self.roi_polygon is not None:
                foot = ((x1 + x2) / 2, y2)
                if cv2.pointPolygonTest(self.roi_polygon, foot, False) < 0:
                    continue
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=pred.score.value,
                )
            )
        return detections
