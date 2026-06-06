"""Real-time AI-based interview monitoring (console-first)."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict

import cv2

from face_module import FaceMonitor
from object_module import ObjectMonitor
from scoring import ScoringManager
from utils import save_violation_screenshot, update_fps


VIOLATION_DIR = "violations"

# Easy-to-tune settings
PENALTIES = {
    "no_face": 2,
    "look_away": 1,
    "multiple_people": 1,
    "phone_detected": 1,
    "different_person": 2,
}
ALERT_COOLDOWN_SECONDS = {
    "no_face": 2.0,
    "look_away": 4.0,
    "multiple_people": 3.0,
    "phone_detected": 2.5,
    "different_person": 8.0,
}

CONSECUTIVE_FRAMES_REQUIRED = {
    "look_away": 15,         # ~0.5s at ~30 FPS before timer check.
    "multiple_people": 8,    # Debounce person false positives.
    "phone_detected": 5,     # Debounce short detector spikes.
    "different_person": 20,  # Stronger confidence before firing.
}


def should_trigger_violation(
    name: str, current_time: float, last_trigger_times: Dict[str, float]
) -> bool:
    cooldown = ALERT_COOLDOWN_SECONDS.get(name, 2.0)
    last = last_trigger_times.get(name, 0.0)
    if current_time - last >= cooldown:
        last_trigger_times[name] = current_time
        return True
    return False


def register_violation(
    frame,
    scoring: ScoringManager,
    violation_type: str,
    message: str,
    level: str = "WARNING",
) -> None:
    screenshot = save_violation_screenshot(frame, VIOLATION_DIR, violation_type)
    scoring.add_violation(violation_type, message, screenshot)
    print(f"[{level}] {message}")


def run() -> None:
    face_monitor = FaceMonitor(
        look_away_seconds=3.0,
        gaze_x_threshold=0.22,
        gaze_y_threshold=0.22,
        yaw_threshold=0.20,
        pitch_threshold=0.20,
        identity_distance_threshold=0.32,
    )
    object_monitor = ObjectMonitor(model_name="yolov8n.pt", yolo_every_n_frames=3)
    scoring = ScoringManager(initial_score=10, penalties=PENALTIES)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("=" * 60)
    print("Real-Time AI-Based Interview Monitoring")
    print("Press 'q' in the video window to quit.")
    print("=" * 60)

    last_violation_time: Dict[str, float] = defaultdict(float)
    consecutive_flags: Dict[str, int] = defaultdict(int)
    prev_fps_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame from webcam.")
            break

        now = time.time()
        face_result = face_monitor.process_frame(frame)
        object_result = object_monitor.process_frame(frame)

        if not face_result.face_present and should_trigger_violation(
            "no_face", now, last_violation_time
        ):
            register_violation(frame, scoring, "no_face", "No face detected", "WARNING")

        if face_result.face_present and face_result.looking_away:
            consecutive_flags["look_away"] += 1
        else:
            consecutive_flags["look_away"] = 0
        if (
            consecutive_flags["look_away"] >= CONSECUTIVE_FRAMES_REQUIRED["look_away"]
            and should_trigger_violation("look_away", now, last_violation_time)
        ):
            direction = face_result.look_direction
            register_violation(
                frame,
                scoring,
                "look_away",
                f"Candidate looking away ({direction}) for too long",
                "WARNING",
            )
            consecutive_flags["look_away"] = 0

        if object_result.person_count > 1:
            consecutive_flags["multiple_people"] += 1
        else:
            consecutive_flags["multiple_people"] = 0
        if (
            consecutive_flags["multiple_people"]
            >= CONSECUTIVE_FRAMES_REQUIRED["multiple_people"]
            and should_trigger_violation("multiple_people", now, last_violation_time)
        ):
            register_violation(
                frame,
                scoring,
                "multiple_people",
                f"Multiple people detected ({object_result.person_count})",
                "ALERT",
            )
            consecutive_flags["multiple_people"] = 0

        if object_result.phone_detected:
            consecutive_flags["phone_detected"] += 1
        else:
            consecutive_flags["phone_detected"] = 0
        if (
            consecutive_flags["phone_detected"]
            >= CONSECUTIVE_FRAMES_REQUIRED["phone_detected"]
            and should_trigger_violation("phone_detected", now, last_violation_time)
        ):
            register_violation(
                frame,
                scoring,
                "phone_detected",
                "Mobile phone detected",
                "ALERT",
            )
            consecutive_flags["phone_detected"] = 0

        if face_result.face_present and face_result.different_person:
            consecutive_flags["different_person"] += 1
        else:
            consecutive_flags["different_person"] = 0
        if (
            consecutive_flags["different_person"]
            >= CONSECUTIVE_FRAMES_REQUIRED["different_person"]
            and should_trigger_violation("different_person", now, last_violation_time)
        ):
            register_violation(
                frame,
                scoring,
                "different_person",
                "Possible different person detected",
                "ALERT",
            )
            consecutive_flags["different_person"] = 0

        prev_fps_time, fps = update_fps(prev_fps_time)
        overlay = frame.copy()
        if face_result.face_present and face_result.face_bbox is not None:
            x1, y1, x2, y2 = face_result.face_bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                overlay,
                "Candidate",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 0),
                2,
            )
        cv2.putText(
            overlay,
            f"Score: {scoring.score}/10",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            overlay,
            f"Look: {face_result.look_direction}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            overlay,
            f"Persons: {object_result.person_count} | Phone: {object_result.phone_detected}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            f"FPS: {fps:.1f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 255),
            2,
        )

        cv2.imshow("Interview Monitoring (Press 'q' to exit)", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print()
    print(scoring.build_final_report())


if __name__ == "__main__":
    run()
