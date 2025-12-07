#!/usr/bin/env python
# replay_2d_court_from_csv_no_args.py

import cv2
import numpy as np
import pandas as pd
from typing import Tuple


# ================== CONFIG YOU CAN CHANGE ==================
CSV_PATH = r"H:\TemprorayCodes\Dr. Klaus\Ball Tracking\TennisProject\output\test3_2_30nov.csv"
FPS = 30.0          # playback fps
START_INDEX = 0     # row index in CSV to start from
END_INDEX = None    # row index to stop (None = until end)
MARGIN = 40         # border margin in pixels
WINDOW_NAME = "2D Court Replay (from CSV)"
# ==========================================================


def detect_num_kps(df: pd.DataFrame) -> int:
    """Detect how many court_map_kp_*_x/y pairs exist."""
    count = 0
    while True:
        x_col = f"court_map_kp_{count}_x"
        y_col = f"court_map_kp_{count}_y"
        if x_col in df.columns and y_col in df.columns:
            count += 1
        else:
            break
    return count


def collect_coord_ranges(df: pd.DataFrame, n_kps: int) -> Tuple[float, float, float, float]:
    """Collect global min/max for all coordinates used on canvas (only translation, no scaling)."""
    xs = []
    ys = []

    # minimap entities (ball + players)
    minimap_cols = [
        ("minimap_ball_x", True),
        ("minimap_ball_y", False),
        ("minimap_top_x", True),
        ("minimap_top_y", False),
        ("minimap_bot_x", True),
        ("minimap_bot_y", False),
    ]

    for col, is_x in minimap_cols:
        if col in df.columns:
            vals = df[col].astype(float).to_numpy()
            vals = vals[~np.isnan(vals)]
            if vals.size > 0:
                (xs if is_x else ys).append(vals)

    # projected court keypoints in court/map space
    for i in range(n_kps):
        x_col = f"court_map_kp_{i}_x"
        y_col = f"court_map_kp_{i}_y"
        if x_col in df.columns:
            vals = df[x_col].astype(float).to_numpy()
            vals = vals[~np.isnan(vals)]
            if vals.size > 0:
                xs.append(vals)
        if y_col in df.columns:
            vals = df[y_col].astype(float).to_numpy()
            vals = vals[~np.isnan(vals)]
            if vals.size > 0:
                ys.append(vals)

    if not xs or not ys:
        raise ValueError("No numeric coordinate columns found to determine ranges.")

    xs_all = np.concatenate(xs)
    ys_all = np.concatenate(ys)

    min_x = float(xs_all.min())
    max_x = float(xs_all.max())
    min_y = float(ys_all.min())
    max_y = float(ys_all.max())

    return min_x, max_x, min_y, max_y


def prepare_canvas(min_x: float, max_x: float, min_y: float, max_y: float, margin: int) -> Tuple[int, int, float, float]:
    """Compute canvas size + translation offsets (no scaling)."""
    width = int(np.ceil(max_x - min_x)) + 2 * margin
    height = int(np.ceil(max_y - min_y)) + 2 * margin
    width = max(width, 200)
    height = max(height, 200)


    offset_x = -min_x + margin
    offset_y = -min_y + margin

    return width, height, offset_x, offset_y


def draw_court_kps(frame_img, row: pd.Series, n_kps: int, offset_x: float, offset_y: float) -> None:
    """Draw projected court keypoints (court_map_kp_*)."""
    for i in range(n_kps):
        x_col = f"court_map_kp_{i}_x"
        y_col = f"court_map_kp_{i}_y"
        if x_col not in row or y_col not in row:
            continue

        x = row[x_col]
        y = row[y_col]
        if pd.isna(x) or pd.isna(y):
            continue

        px = int(round(x + offset_x))
        py = int(round(y + offset_y))

        cv2.circle(frame_img, (px, py), 3, (255, 255, 255), -1)
        cv2.putText(
            frame_img,
            str(i),
            (px + 3, py - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
        )


def draw_entity(frame_img, row: pd.Series, prefix: str, color, offset_x: float, offset_y: float) -> None:
    """Draw one entity from minimap_* coords."""
    x_col = f"minimap_{prefix}_x"
    y_col = f"minimap_{prefix}_y"

    if x_col not in row or y_col not in row:
        return

    x = row[x_col]
    y = row[y_col]
    if pd.isna(x) or pd.isna(y):
        return

    px = int(round(x + offset_x))
    py = int(round(y + offset_y))

    cv2.circle(frame_img, (px, py), 5, color, -1)
    cv2.putText(
        frame_img,
        prefix,
        (px + 6, py),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
    )


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    n_kps = detect_num_kps(df)
    if n_kps == 0:
        print("Warning: no 'court_map_kp_*_x/y' columns found; will show only minimap entities.")

    min_x, max_x, min_y, max_y = collect_coord_ranges(df, n_kps)
    width, height, offset_x, offset_y = prepare_canvas(min_x, max_x, min_y, max_y, MARGIN)

    start_idx = max(0, START_INDEX)
    end_idx = len(df) if END_INDEX is None else min(len(df), END_INDEX)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)

    delay_ms = int(1000 / FPS) if FPS > 0 else 1

    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        frame_img = np.zeros((height, width, 3), dtype=np.uint8)
        frame_img[:] = (30, 30, 30)

        if n_kps > 0:
            draw_court_kps(frame_img, row, n_kps, offset_x, offset_y)

        draw_entity(frame_img, row, "ball", (0, 255, 255), offset_x, offset_y)
        draw_entity(frame_img, row, "top", (0, 0, 255), offset_x, offset_y)
        draw_entity(frame_img, row, "bot", (255, 0, 0), offset_x, offset_y)

        frame_id = row["frame_id"] if "frame_id" in row else idx
        cv2.putText(
            frame_img,
            f"frame: {frame_id}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            2,
        )

        cv2.imshow(WINDOW_NAME, frame_img)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
