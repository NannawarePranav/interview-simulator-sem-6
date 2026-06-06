"""Utility helpers for logging, timing, and file IO."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Tuple

import cv2


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def save_violation_screenshot(frame, output_dir: str, tag: str) -> str:
    ensure_dir(output_dir)
    filename = f"{tag}_{now_str()}.jpg"
    full_path = os.path.join(output_dir, filename)
    cv2.imwrite(full_path, frame)
    return full_path


def update_fps(prev_time: float) -> Tuple[float, float]:
    current = time.time()
    elapsed = current - prev_time
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    return current, fps
