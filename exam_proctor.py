"""
exam_proctor.py - CLI Exam Proctor
Detects eye gaze direction using MediaPipe Face Mesh.
Flags when the user looks away from the screen.

Usage:
    python exam_proctor.py [--camera 0] [--threshold 2.0] [--log session.log]

Install deps:
    pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import sys
from datetime import datetime
from collections import deque


# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe landmark indices  (Face Mesh with refine_landmarks=True)
# ──────────────────────────────────────────────────────────────────────────────
LEFT_IRIS          = [474, 475, 476, 477]
RIGHT_IRIS         = [469, 470, 471, 472]
LEFT_EYE_CORNERS   = [33,  133]          # inner, outer
RIGHT_EYE_CORNERS  = [362, 263]
LEFT_EYE_TB        = [159, 145]          # top, bottom lids
RIGHT_EYE_TB       = [386, 374]


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def lm_px(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def iris_center(landmarks, indices, w, h):
    return np.mean([lm_px(landmarks, i, w, h) for i in indices], axis=0)


def iris_ratio(iris_c, corner_a, corner_b):
    """
    Normalised position of iris between two eye corners.
    0.0 = fully toward corner_a, 0.5 = center, 1.0 = corner_b.
    """
    vec  = corner_b - corner_a
    proj = np.dot(iris_c - corner_a, vec) / (np.linalg.norm(vec) ** 2 + 1e-9)
    return float(np.clip(proj, 0.0, 1.0))


def eye_aspect_ratio(landmarks, top_idx, bot_idx, l_idx, r_idx, w, h):
    top  = lm_px(landmarks, top_idx, w, h)
    bot  = lm_px(landmarks, bot_idx, w, h)
    left = lm_px(landmarks, l_idx,   w, h)
    rgt  = lm_px(landmarks, r_idx,   w, h)
    return np.linalg.norm(top - bot) / (np.linalg.norm(left - rgt) + 1e-6)


def classify_gaze(h_ratio, v_ratio, h_tol=0.18, v_tol=0.22):
    """
    Returns (looking_at_screen: bool, direction_label: str).
    Center zone: h ∈ [0.5-h_tol, 0.5+h_tol], v ∈ [0.5-v_tol, 0.5+v_tol]
    """
    hd = h_ratio - 0.5
    vd = v_ratio - 0.5
    h_ok = abs(hd) <= h_tol
    v_ok = abs(vd) <= v_tol
    if h_ok and v_ok:
        return True, "CENTER"
    parts = []
    if not v_ok:
        parts.append("UP" if vd < 0 else "DOWN")
    if not h_ok:
        parts.append("LEFT" if hd < 0 else "RIGHT")
    return False, "-".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Session statistics
# ──────────────────────────────────────────────────────────────────────────────

class SessionStats:
    def __init__(self):
        self.start_time        = time.time()
        self.frames            = 0
        self.on_frames         = 0
        self.off_frames        = 0
        self.no_face_frames    = 0
        self.blink_count       = 0
        self._in_blink         = False
        self.violations        = []        # (timestamp_str, duration_s, direction)
        self._viol_start       = None
        self._viol_dir         = None

    # called every frame
    def record(self, face: bool, on: bool, direction: str):
        self.frames += 1
        if not face:
            self.no_face_frames += 1
            self._close_violation()
            return
        if on:
            self.on_frames += 1
            self._close_violation()
        else:
            self.off_frames += 1
            if self._viol_start is None:
                self._viol_start = time.time()
            self._viol_dir = direction

    def _close_violation(self):
        if self._viol_start is not None:
            dur = time.time() - self._viol_start
            self.violations.append((
                datetime.fromtimestamp(self._viol_start).strftime("%H:%M:%S"),
                round(dur, 2),
                self._viol_dir or "UNKNOWN"
            ))
            self._viol_start = None
            self._viol_dir   = None

    def record_blink(self):
        self.blink_count += 1

    def attention_pct(self):
        scored = self.on_frames + self.off_frames
        return 100.0 if scored == 0 else round(100 * self.on_frames / scored, 1)

    def elapsed(self):
        return time.time() - self.start_time

    def summary(self):
        self._close_violation()
        e = self.elapsed()
        lines = [
            "",
            "=" * 56,
            "   EXAM PROCTOR - SESSION SUMMARY",
            "=" * 56,
            f"   Duration        : {int(e // 60)}m {int(e % 60)}s",
            f"   Total frames    : {self.frames}",
            f"   Attention score : {self.attention_pct()}%",
            f"   Blinks detected : {self.blink_count}",
            f"   No-face frames  : {self.no_face_frames}",
            f"   Violations      : {len(self.violations)}",
        ]
        if self.violations:
            lines += [
                "",
                "   Violation log:",
                "   {:>8}  {:>7}  {}".format("Time", "Dur(s)", "Direction"),
                "   " + "-" * 32,
            ]
            for ts, dur, d in self.violations:
                lines.append(f"   {ts}  {dur:>6.2f}s  {d}")
        lines.append("=" * 56)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def run_proctor(camera_id=0, away_threshold=2.0, log_path=None,
                h_tol=0.18, v_tol=0.22, blink_ear=0.18, show_preview=True):

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_id}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    stats        = SessionStats()
    away_since   = None
    alert_active = False
    log_file     = open(log_path, "w") if log_path else None

    if log_file:
        log_file.write(f"# Exam Proctor log - {datetime.now()}\n")
        log_file.write("timestamp,event,details\n")

    def emit(event, detail=""):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"  [{ts}] {event:<22} {detail}")
        if log_file:
            log_file.write(f"{ts},{event},{detail}\n")
            log_file.flush()

    print()
    print("+----------------------------------------------+")
    print("|         EXAM PROCTOR  -  ACTIVE              |")
    print("+----------------------------------------------+")
    print(f"|  Camera      : {camera_id:<30}|")
    print(f"|  Alert after : {away_threshold}s off-screen{' ' * 19}|")
    print(f"|  H tolerance : +/-{int(h_tol*100)}% of eye width{' ' * 15}|")
    print(f"|  V tolerance : +/-{int(v_tol*100)}% of eye height{' ' * 14}|")
    print(f"|  Preview     : {'ON (press Q to quit)' if show_preview else 'OFF (Ctrl-C to quit)':<30}|")
    print("+----------------------------------------------+")
    print()
    emit("SESSION_START")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame   = cv2.flip(frame, 1)
            h, w    = frame.shape[:2]
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            face_found = False
            on_screen  = False
            direction  = "NO_FACE"

            if results.multi_face_landmarks:
                face_found = True
                lms = results.multi_face_landmarks[0].landmark

                # ── compute horizontal iris ratio ──────────────────────────
                l_iris = iris_center(lms, LEFT_IRIS,  w, h)
                r_iris = iris_center(lms, RIGHT_IRIS, w, h)

                l_lc = lm_px(lms, LEFT_EYE_CORNERS[0],  w, h)
                l_rc = lm_px(lms, LEFT_EYE_CORNERS[1],  w, h)
                r_lc = lm_px(lms, RIGHT_EYE_CORNERS[0], w, h)
                r_rc = lm_px(lms, RIGHT_EYE_CORNERS[1], w, h)

                l_top = lm_px(lms, LEFT_EYE_TB[0],  w, h)
                l_bot = lm_px(lms, LEFT_EYE_TB[1],  w, h)
                r_top = lm_px(lms, RIGHT_EYE_TB[0], w, h)
                r_bot = lm_px(lms, RIGHT_EYE_TB[1], w, h)

                lh = iris_ratio(l_iris, l_lc, l_rc)
                rh = iris_ratio(r_iris, r_lc, r_rc)
                h_avg = (lh + rh) / 2

                # ── vertical iris ratio ────────────────────────────────────
                l_eye_h = np.linalg.norm(l_bot - l_top) + 1e-6
                r_eye_h = np.linalg.norm(r_bot - r_top) + 1e-6
                lv = (l_iris[1] - l_top[1]) / l_eye_h
                rv = (r_iris[1] - r_top[1]) / r_eye_h
                v_avg = (lv + rv) / 2

                on_screen, direction = classify_gaze(h_avg, v_avg, h_tol, v_tol)

                # ── blink detection ────────────────────────────────────────
                l_ear = eye_aspect_ratio(lms, LEFT_EYE_TB[0],  LEFT_EYE_TB[1],
                                         LEFT_EYE_CORNERS[0],  LEFT_EYE_CORNERS[1], w, h)
                r_ear = eye_aspect_ratio(lms, RIGHT_EYE_TB[0], RIGHT_EYE_TB[1],
                                         RIGHT_EYE_CORNERS[0], RIGHT_EYE_CORNERS[1], w, h)
                ear = (l_ear + r_ear) / 2
                if ear < blink_ear:
                    stats._in_blink = True
                elif stats._in_blink:
                    stats.record_blink()
                    stats._in_blink = False

                # ── preview annotations ────────────────────────────────────
                if show_preview:
                    color = (0, 210, 0) if on_screen else (30, 30, 220)
                    for pt in [l_iris, r_iris]:
                        cv2.circle(frame, tuple(pt.astype(int)), 4, color, -1)
                    for pt in [l_lc, l_rc, r_lc, r_rc]:
                        cv2.circle(frame, tuple(pt.astype(int)), 2, (180, 180, 0), -1)
                    cv2.putText(frame, f"GAZE: {direction}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, f"H:{h_avg:.2f}  V:{v_avg:.2f}", (10, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                    cv2.putText(frame, f"Blinks: {stats.blink_count}  Attention: {stats.attention_pct()}%",
                                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 210, 0) if stats.attention_pct() >= 80 else (0, 140, 255), 1)

            else:
                if show_preview:
                    cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 220), 2)

            stats.record(face_found, on_screen, direction)

            # ── alert logic ────────────────────────────────────────────────
            if not face_found or not on_screen:
                if away_since is None:
                    away_since = time.time()
                elapsed_away = time.time() - away_since
                if elapsed_away >= away_threshold and not alert_active:
                    alert_active = True
                    emit("!  VIOLATION", f"looking {direction} for > {away_threshold}s")
                if show_preview:
                    # red progress bar toward the threshold
                    bar = int(min(elapsed_away / away_threshold, 1.0) * (w - 20))
                    cv2.rectangle(frame, (10, h - 7), (10 + bar, h - 2), (30, 30, 220), -1)
            else:
                if alert_active:
                    emit("OK RETURNED",
                         f"was away {round(time.time() - away_since, 2)}s")
                    alert_active = False
                away_since = None

            # ── display ────────────────────────────────────────────────────
            if show_preview:
                if alert_active:
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (30, 30, 220), 5)
                cv2.imshow("Exam Proctor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        cap.release()
        face_mesh.close()
        if show_preview:
            cv2.destroyAllWindows()
        if log_file:
            log_file.close()
        emit("SESSION_END")
        print(stats.summary())


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CLI Exam Proctor - gaze tracking")
    p.add_argument("--camera",     type=int,   default=0,
                   help="Camera device index (default: 0)")
    p.add_argument("--threshold",  type=float, default=2.0,
                   help="Seconds off-screen before violation alert (default: 2.0)")
    p.add_argument("--log",        type=str,   default=None,
                   help="Path for CSV event log (optional)")
    p.add_argument("--h-tol",      type=float, default=0.18,
                   help="Horizontal gaze tolerance 0–0.5 (default: 0.18)")
    p.add_argument("--v-tol",      type=float, default=0.22,
                   help="Vertical gaze tolerance 0–0.5 (default: 0.22)")
    p.add_argument("--blink-ear",  type=float, default=0.18,
                   help="Eye Aspect Ratio threshold for blink (default: 0.18)")
    p.add_argument("--no-preview", action="store_true",
                   help="Disable the OpenCV preview window (headless / pure CLI)")
    args = p.parse_args()

    run_proctor(
        camera_id      = args.camera,
        away_threshold = args.threshold,
        log_path       = args.log,
        h_tol          = args.h_tol,
        v_tol          = args.v_tol,
        blink_ear      = args.blink_ear,
        show_preview   = not args.no_preview,
    )
