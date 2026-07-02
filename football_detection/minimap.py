from __future__ import annotations

import cv2
import numpy as np

from .config import MinimapConfig


class ViewTransformer:
    """Projects image-space points onto a top-down pitch (bird's-eye) view.

    `config.source_keypoints` are the four pitch corners as seen by the camera
    (pixels); they map onto the real-world pitch corners (meters). These
    keypoints must be calibrated per camera/video — placeholders until then.
    """

    def __init__(self, config: MinimapConfig):
        target_keypoints = np.array(
            [
                [0, 0],
                [config.pitch_length_m, 0],
                [config.pitch_length_m, config.pitch_width_m],
                [0, config.pitch_width_m],
            ],
            dtype=np.float32,
        )
        source_keypoints = np.array(config.source_keypoints, dtype=np.float32)

        homography, _ = cv2.findHomography(source_keypoints, target_keypoints)
        if homography is None:
            raise ValueError("Could not compute homography from source_keypoints")
        self.homography = homography

    def transform_point(self, point: tuple[float, float]) -> tuple[float, float]:
        src = np.array([[point]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.homography)
        x, y = dst[0, 0]
        return float(x), float(y)
