from __future__ import annotations

import cv2
import numpy as np

PLAYER_COLORS = [(255, 80, 80), (80, 80, 255)]  # BGR
BALL_COLOR = (0, 220, 0)
UNKNOWN_COLOR = (200, 200, 200)


def draw_ellipse(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: str | None = None,
) -> None:
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    y2_int = int(y2)
    width = int(x2 - x1)

    cv2.ellipse(
        frame,
        (center_x, y2_int),
        (width // 2, max(int(0.35 * width), 6)),
        0,
        -45,
        235,
        color,
        2,
    )

    if label:
        cv2.putText(
            frame,
            label,
            (center_x - 10, y2_int + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def draw_triangle(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> None:
    x1, y1, x2, y2 = bbox
    apex_x = int((x1 + x2) / 2)
    apex_y = int(y1)
    points = np.array(
        [[apex_x, apex_y], [apex_x - 10, apex_y - 18], [apex_x + 10, apex_y - 18]]
    )
    cv2.drawContours(frame, [points], 0, BALL_COLOR, cv2.FILLED)
    cv2.drawContours(frame, [points], 0, (0, 0, 0), 1)


MINIMAP_LINE = (225, 225, 225)
GRASS_DARK = (35, 95, 35)
GRASS_LIGHT = (48, 118, 48)


def _draw_pitch(
    overlay: np.ndarray,
    map_w: int,
    map_h: int,
    pad: int,
    pitch_length_m: float,
    pitch_width_m: float,
) -> None:
    """Draws grass stripes and standard pitch markings into the minimap panel."""
    pitch_w_px = map_w - 2 * pad
    pitch_h_px = map_h - 2 * pad
    length, width = pitch_length_m, pitch_width_m
    cx, cy = length / 2, width / 2

    def to_px(x_m: float, y_m: float) -> tuple[int, int]:
        return (
            pad + int(x_m / length * pitch_w_px),
            pad + int(y_m / width * pitch_h_px),
        )

    def line(p1: tuple[float, float], p2: tuple[float, float]) -> None:
        cv2.line(overlay, to_px(*p1), to_px(*p2), MINIMAP_LINE, 1, cv2.LINE_AA)

    def rect(p1: tuple[float, float], p2: tuple[float, float]) -> None:
        cv2.rectangle(overlay, to_px(*p1), to_px(*p2), MINIMAP_LINE, 1)

    # Mowing stripes.
    stripes = 10
    for i in range(stripes):
        x0 = pad + int(i / stripes * pitch_w_px)
        x1 = pad + int((i + 1) / stripes * pitch_w_px)
        shade = GRASS_LIGHT if i % 2 == 0 else GRASS_DARK
        cv2.rectangle(overlay, (x0, pad), (x1, pad + pitch_h_px), shade, -1)

    # Circle radius for the center circle / penalty arcs (9.15 m).
    rx = int(9.15 / length * pitch_w_px)
    ry = int(9.15 / width * pitch_h_px)

    rect((0, 0), (length, width))                       # boundary
    line((cx, 0), (cx, width))                          # halfway line
    cv2.ellipse(overlay, to_px(cx, cy), (rx, ry), 0, 0, 360, MINIMAP_LINE, 1, cv2.LINE_AA)
    cv2.circle(overlay, to_px(cx, cy), 2, MINIMAP_LINE, -1)  # center spot

    # Penalty areas (16.5 x 40.32 m) and goal areas (5.5 x 18.32 m), both ends.
    rect((0, cy - 20.16), (16.5, cy + 20.16))
    rect((length - 16.5, cy - 20.16), (length, cy + 20.16))
    rect((0, cy - 9.16), (5.5, cy + 9.16))
    rect((length - 5.5, cy - 9.16), (length, cy + 9.16))

    # Penalty spots (11 m) and arcs.
    cv2.circle(overlay, to_px(11, cy), 2, MINIMAP_LINE, -1)
    cv2.circle(overlay, to_px(length - 11, cy), 2, MINIMAP_LINE, -1)
    cv2.ellipse(overlay, to_px(11, cy), (rx, ry), 0, -53, 53, MINIMAP_LINE, 1, cv2.LINE_AA)
    cv2.ellipse(
        overlay, to_px(length - 11, cy), (rx, ry), 0, 127, 233, MINIMAP_LINE, 1, cv2.LINE_AA
    )

    # Goals.
    rect((-1.5, cy - 3.66), (0, cy + 3.66))
    rect((length, cy - 3.66), (length + 1.5, cy + 3.66))


def _put_label(overlay: np.ndarray, text: str, org: tuple[int, int]) -> None:
    """White id text with a dark outline so it stays readable over the pitch."""
    cv2.putText(overlay, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(
        overlay, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA
    )


def _minimap_anchor(
    frame_w: int,
    frame_h: int,
    map_w: int,
    map_h: int,
    margin: int,
    position: str,
) -> tuple[int, int]:
    """Top-left pixel of the minimap panel for a named frame corner."""
    x = margin if position.endswith("left") else frame_w - map_w - margin
    y = margin if position.startswith("top") else frame_h - map_h - margin
    return x, y


def draw_minimap(
    frame: np.ndarray,
    player_points: list[tuple[tuple[float, float], tuple[int, int, int], int]],
    pitch_length_m: float,
    pitch_width_m: float,
    size: tuple[int, int] = (400, 260),
    margin: int = 20,
    position: str = "bottom_right",
) -> None:
    """Draws a top-down pitch with player dots, anchored at the given frame corner."""
    map_w, map_h = size
    frame_h, frame_w = frame.shape[:2]
    ox, oy = _minimap_anchor(frame_w, frame_h, map_w, map_h, margin, position)
    pad = 8

    overlay = frame[oy : oy + map_h, ox : ox + map_w]
    _draw_pitch(overlay, map_w, map_h, pad, pitch_length_m, pitch_width_m)
    cv2.rectangle(overlay, (0, 0), (map_w - 1, map_h - 1), (255, 255, 255), 2)

    pitch_w_px = map_w - 2 * pad
    pitch_h_px = map_h - 2 * pad
    for (x_m, y_m), color, track_id in player_points:
        px = pad + int(x_m / pitch_length_m * pitch_w_px)
        py = pad + int(y_m / pitch_width_m * pitch_h_px)
        if 0 <= px < map_w and 0 <= py < map_h:
            cv2.circle(overlay, (px, py), 5, color, -1)
            cv2.circle(overlay, (px, py), 5, (20, 20, 20), 1, cv2.LINE_AA)
            _put_label(overlay, str(track_id), (px + 6, py - 5))

    frame[oy : oy + map_h, ox : ox + map_w] = overlay
