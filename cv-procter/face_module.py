"""Face, gaze, and head-orientation checks using MediaPipe."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceResult:
    face_present: bool
    look_direction: str
    looking_away: bool
    different_person: bool
    face_signature: Optional[np.ndarray]
    face_bbox: Optional[Tuple[int, int, int, int]]


class FaceMonitor:
    """
    Lightweight face and gaze monitor.

    - Detects face presence.
    - Uses eye-center offsets and nose/head geometry to estimate look direction.
    - Builds a simple embedding-like signature from landmarks to track identity drift.
    """

    LEFT_EYE_IDS = [33, 133, 159, 145]
    RIGHT_EYE_IDS = [362, 263, 386, 374]
    LEFT_IRIS_IDS = [468, 469, 470, 471, 472]
    RIGHT_IRIS_IDS = [473, 474, 475, 476, 477]
    NOSE_TIP_ID = 1
    FACE_LEFT_ID = 234
    FACE_RIGHT_ID = 454
    CHIN_ID = 152
    FOREHEAD_ID = 10

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        look_away_seconds: float = 3.0,
        gaze_x_threshold: float = 0.22,
        gaze_y_threshold: float = 0.22,
        yaw_threshold: float = 0.20,
        pitch_threshold: float = 0.20,
        identity_distance_threshold: float = 0.30,
        smoothing_window: int = 7,
        min_away_votes: int = 5,
    ) -> None:
        self.look_away_seconds = look_away_seconds
        self.gaze_x_threshold = gaze_x_threshold
        self.gaze_y_threshold = gaze_y_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.identity_distance_threshold = identity_distance_threshold
        self.min_away_votes = min_away_votes

        self._mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.reference_signature: Optional[np.ndarray] = None
        self.away_start_time: Optional[float] = None
        self.recent_away_flags: deque[bool] = deque(maxlen=max(3, smoothing_window))

    def process_frame(self, frame_bgr: np.ndarray) -> FaceResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            self.away_start_time = None
            return FaceResult(
                face_present=False,
                look_direction="no_face",
                looking_away=False,
                different_person=False,
                face_signature=None,
                face_bbox=None,
            )

        face_landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]
        coords = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks], dtype=np.float32)
        face_bbox = self._compute_face_bbox(coords, w, h)

        look_direction, is_away = self._estimate_gaze_and_pose(coords)
        looking_away = self._update_away_state(is_away)

        signature = self._build_face_signature(coords)
        different_person = self._check_identity_drift(signature)

        return FaceResult(
            face_present=True,
            look_direction=look_direction,
            looking_away=looking_away,
            different_person=different_person,
            face_signature=signature,
            face_bbox=face_bbox,
        )

    def _estimate_gaze_and_pose(self, coords: np.ndarray) -> Tuple[str, bool]:
        left_eye = coords[self.LEFT_EYE_IDS]
        right_eye = coords[self.RIGHT_EYE_IDS]
        left_iris = coords[self.LEFT_IRIS_IDS]
        right_iris = coords[self.RIGHT_IRIS_IDS]
        nose = coords[self.NOSE_TIP_ID]
        face_left = coords[self.FACE_LEFT_ID]
        face_right = coords[self.FACE_RIGHT_ID]
        chin = coords[self.CHIN_ID]
        forehead = coords[self.FOREHEAD_ID]

        face_width = np.linalg.norm(face_right - face_left) + 1e-6
        face_height = np.linalg.norm(chin - forehead) + 1e-6

        # Head orientation (yaw/pitch) from stable face geometry.
        yaw = ((nose[0] - face_left[0]) / face_width) - 0.5
        pitch = ((nose[1] - forehead[1]) / face_height) - 0.58

        # Eye-gaze orientation from iris position inside eye corners/lids.
        left_iris_center = left_iris.mean(axis=0)
        right_iris_center = right_iris.mean(axis=0)

        left_eye_width = abs(coords[133][0] - coords[33][0]) + 1e-6
        right_eye_width = abs(coords[263][0] - coords[362][0]) + 1e-6
        left_eye_height = abs(coords[145][1] - coords[159][1]) + 1e-6
        right_eye_height = abs(coords[374][1] - coords[386][1]) + 1e-6

        left_gaze_x = (left_iris_center[0] - ((coords[33][0] + coords[133][0]) / 2.0)) / left_eye_width
        right_gaze_x = (right_iris_center[0] - ((coords[362][0] + coords[263][0]) / 2.0)) / right_eye_width
        gaze_x = (left_gaze_x + right_gaze_x) / 2.0

        left_gaze_y = (left_iris_center[1] - ((coords[159][1] + coords[145][1]) / 2.0)) / left_eye_height
        right_gaze_y = (right_iris_center[1] - ((coords[386][1] + coords[374][1]) / 2.0)) / right_eye_height
        gaze_y = (left_gaze_y + right_gaze_y) / 2.0

        direction = "center"
        horizontal_signal = abs(gaze_x) + abs(yaw)
        vertical_signal = abs(gaze_y) + abs(pitch)
        if horizontal_signal > vertical_signal:
            if gaze_x > self.gaze_x_threshold or yaw > self.yaw_threshold:
                direction = "right"
            elif gaze_x < -self.gaze_x_threshold or yaw < -self.yaw_threshold:
                direction = "left"
        else:
            if gaze_y > self.gaze_y_threshold or pitch > self.pitch_threshold:
                direction = "down"
            elif gaze_y < -self.gaze_y_threshold or pitch < -self.pitch_threshold:
                direction = "up"

        is_away = direction != "center"
        return direction, is_away

    def _update_away_state(self, currently_away: bool) -> bool:
        self.recent_away_flags.append(currently_away)
        away_votes = sum(self.recent_away_flags)
        smoothed_away = away_votes >= self.min_away_votes

        now = time.time()
        if smoothed_away:
            if self.away_start_time is None:
                self.away_start_time = now
            return (now - self.away_start_time) >= self.look_away_seconds
        self.away_start_time = None
        return False

    def _compute_face_bbox(
        self, coords: np.ndarray, frame_w: int, frame_h: int
    ) -> Tuple[int, int, int, int]:
        min_xy = coords.min(axis=0)
        max_xy = coords.max(axis=0)
        pad_x = int((max_xy[0] - min_xy[0]) * 0.12)
        pad_y = int((max_xy[1] - min_xy[1]) * 0.18)

        x1 = max(0, int(min_xy[0]) - pad_x)
        y1 = max(0, int(min_xy[1]) - pad_y)
        x2 = min(frame_w - 1, int(max_xy[0]) + pad_x)
        y2 = min(frame_h - 1, int(max_xy[1]) + pad_y)
        return x1, y1, x2, y2

    def _build_face_signature(self, coords: np.ndarray) -> np.ndarray:
        key_ids = [
            self.NOSE_TIP_ID,
            self.FACE_LEFT_ID,
            self.FACE_RIGHT_ID,
            self.CHIN_ID,
            self.FOREHEAD_ID,
            *self.LEFT_EYE_IDS,
            *self.RIGHT_EYE_IDS,
        ]
        selected = coords[key_ids].copy()
        center = selected.mean(axis=0, keepdims=True)
        selected -= center
        scale = np.linalg.norm(selected[1] - selected[2]) + 1e-6
        selected /= scale
        return selected.flatten()

    def _check_identity_drift(self, signature: np.ndarray) -> bool:
        if self.reference_signature is None:
            self.reference_signature = signature
            return False

        distance = float(np.linalg.norm(self.reference_signature - signature))
        self.reference_signature = 0.995 * self.reference_signature + 0.005 * signature
        return distance > self.identity_distance_threshold
