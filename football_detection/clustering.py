from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from .config import ClusteringConfig


def extract_jersey_color(
    frame: np.ndarray, bbox: tuple[float, float, float, float]
) -> np.ndarray:
    """Mean BGR color of a player's torso, with grass-green pixels masked out."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    torso = frame[y1 : y1 + (y2 - y1) // 2, x1:x2]
    if torso.size == 0:
        return np.zeros(3, dtype=np.float32)

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, (25, 40, 40), (44, 255, 255))
    jersey_pixels = torso[cv2.bitwise_not(grass_mask) > 0]
    if jersey_pixels.size == 0:
        jersey_pixels = torso.reshape(-1, 3)

    return jersey_pixels.mean(axis=0).astype(np.float32)


class TeamAssigner:
    """Groups players into teams by jersey color.

    Color features are smoothed across frames via `observe()`. Call `fit()`
    periodically (e.g. once enough players have been seen) to (re)compute
    cluster assignments, then use `team_for()` to look up a stable group id
    per tracker id.
    """

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self._features: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}
        self._labels: dict[int, int] = {}
        self._centers: np.ndarray | None = None

    def observe(self, track_id: int, color: np.ndarray) -> None:
        color = color.astype(np.float32)
        count = self._counts.get(track_id, 0)
        if count == 0:
            self._features[track_id] = color
        else:
            previous = self._features[track_id]
            self._features[track_id] = previous + (color - previous) / (count + 1)
        self._counts[track_id] = count + 1

    def fit(self) -> None:
        track_ids = list(self._features.keys())
        if not track_ids:
            return

        features = np.stack([self._features[tid] for tid in track_ids])
        algorithm = self.config.algorithm.lower()

        if algorithm == "kmeans":
            labels = self._fit_kmeans(features)
        elif algorithm == "dbscan":
            if len(features) < self.config.min_samples:
                return
            dbscan = DBSCAN(eps=self.config.eps, min_samples=self.config.min_samples)
            labels = dbscan.fit_predict(features)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.config.algorithm}")

        self._labels = dict(zip(track_ids, (int(label) for label in labels)))

    def _fit_kmeans(self, features: np.ndarray) -> np.ndarray:
        n_clusters = self.config.num_expected_groups
        if len(features) < n_clusters:
            return np.full(len(features), -1, dtype=int)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_

        if self._centers is not None and self._centers.shape == centers.shape:
            label_map = self._match_centers(centers)
            labels = np.array([label_map[int(label)] for label in labels], dtype=int)
            ordered_centers = np.empty_like(centers)
            for new_label, old_label in label_map.items():
                ordered_centers[old_label] = centers[new_label]
            centers = ordered_centers

        self._centers = centers
        return labels

    def _match_centers(self, centers: np.ndarray) -> dict[int, int]:
        """Map new KMeans labels to previous labels by nearest centroid."""
        distances = np.linalg.norm(self._centers[:, None, :] - centers[None, :, :], axis=2)
        remaining_old = set(range(len(centers)))
        remaining_new = set(range(len(centers)))
        label_map: dict[int, int] = {}

        while remaining_old and remaining_new:
            old_label, new_label = min(
                (
                    (old_label, new_label)
                    for old_label in remaining_old
                    for new_label in remaining_new
                ),
                key=lambda pair: distances[pair[0], pair[1]],
            )
            label_map[new_label] = old_label
            remaining_old.remove(old_label)
            remaining_new.remove(new_label)

        return label_map

    def team_for(self, track_id: int) -> int | None:
        return self._labels.get(track_id)
