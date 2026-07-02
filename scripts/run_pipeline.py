"""Process a video through the full football analysis pipeline.

Usage:
    python scripts/run_pipeline.py --source input.mp4 --output output.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_detection.config import load_config
from football_detection.pipeline import FootballPipeline
from football_detection.visualize import (
    PLAYER_COLORS,
    UNKNOWN_COLOR,
    draw_ellipse,
    draw_minimap,
    draw_triangle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the football analysis pipeline on a video"
    )
    parser.add_argument("--source", required=True, help="Path to input video")
    parser.add_argument("--output", default="output.mp4", help="Path to write annotated video")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu, cuda, mps")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = FootballPipeline(config, device=args.device)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = pipeline.process_frame(frame)

        minimap_points = []
        for player in result.players:
            has_team = player.team is not None and player.team >= 0
            color = (
                PLAYER_COLORS[player.team % len(PLAYER_COLORS)]
                if has_team
                else UNKNOWN_COLOR
            )
            label = (
                f"Team {player.team + 1} #{player.track_id}"
                if has_team
                else f"#{player.track_id}"
            )
            draw_ellipse(frame, player.bbox, color, label)
            minimap_points.append((player.minimap_position, color, player.track_id))

        for ball in result.ball_detections:
            draw_triangle(frame, ball.bbox)

        draw_minimap(
            frame,
            minimap_points,
            config.minimap.pitch_length_m,
            config.minimap.pitch_width_m,
            position=config.minimap.position,
        )

        writer.write(frame)
        frame_index += 1
        if frame_index % 50 == 0:
            print(f"Processed {frame_index} frames")

    cap.release()
    writer.release()
    print(f"Done. Wrote {frame_index} frames to {args.output}")


if __name__ == "__main__":
    main()
