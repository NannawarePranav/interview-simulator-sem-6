# Real-Time AI-Based Interview Monitoring System

Lightweight, modular Python application for real-time interview and exam monitoring. It combines **OpenCV** (webcam capture and display), **MediaPipe Face Mesh** (face presence, gaze/head pose, identity drift), and **YOLOv8 Nano** (person and phone detection) into a single console-first monitoring loop with on-screen overlays and a session report.

**Version:** CV-V1.2  
**Python:** 3.10 recommended (tested with `.venv310`)

---

## Features

| Capability | Description |
|------------|-------------|
| Live webcam feed | OpenCV window with score, gaze direction, person/phone status, and FPS |
| Face & gaze | MediaPipe Face Mesh (1 face); left/right/up/down/center + sustained look-away |
| Object detection | YOLOv8n on CPU every N frames; counts persons and flags phones |
| Identity drift | Landmark-based signature vs. rolling reference (possible swap) |
| Scoring | Starts at **10**, penalties per violation, floor at **0** |
| Evidence | Timestamped JPG screenshots saved under `violations/` |
| Debouncing | Per-violation cooldowns + consecutive-frame thresholds to reduce false positives |
| Session report | Printed to console when you press **`q`** |

### Violation types and default penalties

| Type | Default penalty | Trigger summary |
|------|-----------------|-----------------|
| `no_face` | −2 | No face in frame (cooldown 2.0 s) |
| `look_away` | −1 | Non-center gaze/head pose sustained (~3 s + frame debounce) |
| `multiple_people` | −1 | More than one person box above size/conf thresholds |
| `phone_detected` | −1 | Cell phone class above confidence threshold |
| `different_person` | −2 | Face landmark signature drifts from session reference |

---

## Project structure

```
CV-V1.2/
├── main.py              # Main loop, violation logic, overlay UI, session entry
├── face_module.py       # MediaPipe face mesh, gaze/pose, identity signature
├── object_module.py     # YOLOv8 inference (person / phone)
├── scoring.py           # Score tracking and final report builder
├── utils.py             # FPS helper, violation screenshot I/O
├── requirements.txt     # Python dependencies
├── yolov8n.pt           # YOLOv8 Nano weights (~6.3 MB, bundled)
├── violations/          # Auto-created; violation screenshots (git-ignore recommended)
├── .venv310/            # Local virtual environment (create locally, do not commit)
└── README.md
```

**Approximate source size:** ~22 KB Python (excluding model, venv, and screenshots).

---

## Requirements

- **OS:** Windows 10/11 (paths below use PowerShell; Linux/macOS work with equivalent venv commands)
- **Python:** 3.10.x
- **Hardware:** Webcam; CPU-only inference (no GPU required)
- **Disk:** ~500 MB+ for venv after `pip install` (PyTorch, MediaPipe, OpenCV)

### Dependencies (`requirements.txt`)

| Package | Role |
|---------|------|
| `opencv-python` | Capture, display, image I/O |
| `mediapipe==0.10.11` | Face mesh, iris landmarks, gaze heuristics |
| `ultralytics` | YOLOv8 wrapper |
| `numpy` | Array math for landmarks and signatures |

Installing `requirements.txt` also pulls **PyTorch**, **torchvision**, and related packages via Ultralytics.

---

## Setup

### 1. Create and activate virtual environment

```powershell
cd E:\RIFAZ\CV-V1.2
py -3.10 -m venv .venv310
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv310\Scripts\Activate.ps1
```

You should see `(.venv310)` in your prompt.

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

> **Important:** Run this after creating the venv. A common error is `ModuleNotFoundError: No module named 'mediapipe'` if only OpenCV/numpy were installed manually.

### 3. Run the system

```powershell
python main.py
```

- Allow webcam access when prompted.
- Use the **“Interview Monitoring (Press 'q' to exit)”** window.
- Press **`q`** to end the session and print the final report.

---

## How it works

### Data flow

```
Webcam → main.py loop
           ├─→ FaceMonitor.process_frame()   → face, gaze, identity
           ├─→ ObjectMonitor.process_frame() → person count, phone flag
           ├─→ Violation rules (cooldown + consecutive frames)
           ├─→ ScoringManager.add_violation() + screenshot
           └─→ OpenCV overlay + imshow
```

### `main.py`

- Opens default camera (`cv2.VideoCapture(0)`).
- Instantiates `FaceMonitor`, `ObjectMonitor`, and `ScoringManager`.
- Applies **cooldowns** (`ALERT_COOLDOWN_SECONDS`) so the same violation type is not logged repeatedly.
- Uses **consecutive frame counters** (`CONSECUTIVE_FRAMES_REQUIRED`) before firing look-away, multi-person, phone, and identity violations.
- Draws overlay: candidate bbox, score, look direction, person/phone status, FPS.
- On exit, prints `ScoringManager.build_final_report()`.

### `face_module.py`

- MediaPipe Face Mesh: `max_num_faces=1`, `refine_landmarks=True` (iris points for gaze).
- **Gaze:** iris position relative to eye corners/lids; combined with yaw/pitch from nose and face geometry.
- **Look-away:** smoothed vote window (`min_away_votes`) then timer (`look_away_seconds`, default 3.0 s in `main.py`).
- **Identity:** normalized landmark signature; exponential moving average reference; flags drift above `identity_distance_threshold`.

### `object_module.py`

- Loads `yolov8n.pt`, runs on **CPU** at `imgsz=640`.
- Skips inference on most frames (`yolo_every_n_frames=3`) and reuses last result for speed.
- **Person:** confidence ≥ 0.55 and bounding-box area ≥ 3% of frame (reduces distant false positives).
- **Phone:** `cell phone` / `mobile phone` / `phone` labels above confidence threshold.

### `scoring.py`

- Dataclass-based `ViolationEvent` log with timestamps and optional screenshot paths.
- `build_final_report()`: score, counts by type, chronological detailed log.

### `utils.py`

- `save_violation_screenshot()` → `violations/{type}_{timestamp}.jpg`
- `update_fps()` for overlay display

---

## Configuration reference

### `main.py` — penalties and timing

```python
PENALTIES = {
    "no_face": 2,
    "look_away": 1,
    "multiple_people": 1,
    "phone_detected": 1,
    "different_person": 2,
}

ALERT_COOLDOWN_SECONDS = { ... }      # Min seconds between same violation type
CONSECUTIVE_FRAMES_REQUIRED = { ... }  # Frames condition must hold before trigger
```

`FaceMonitor` constructor in `run()`:

| Parameter | Default in `main.py` | Purpose |
|-----------|----------------------|---------|
| `look_away_seconds` | 3.0 | Sustained away gaze before violation |
| `gaze_x_threshold` / `gaze_y_threshold` | 0.22 | Eye-gaze sensitivity |
| `yaw_threshold` / `pitch_threshold` | 0.20 | Head pose sensitivity |
| `identity_distance_threshold` | 0.32 | Identity drift sensitivity |

### `object_module.py`

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `conf_threshold` | 0.5 | Base YOLO confidence |
| `yolo_every_n_frames` | 3 | Inference stride |
| `person_conf_threshold` | 0.55 | Stricter person filter |
| `person_min_area_ratio` | 0.03 | Ignore small/distant detections |
| `phone_conf_threshold` | 0.55 | Phone detection confidence |

---

## Troubleshooting

| Issue | Likely cause | Fix |
|-------|----------------|-----|
| `ModuleNotFoundError: mediapipe` | Dependencies not installed in active venv | Activate `.venv310`, run `pip install -r requirements.txt` |
| `Could not open webcam` | Camera in use or missing | Close other apps; try index `1` in `VideoCapture(1)` |
| Script won’t activate venv | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process Bypass` then `Activate.ps1` |
| Too many false look-away alerts | Lighting / thresholds | Increase `look_away_seconds` or gaze/yaw thresholds in `main.py` |
| Missed phone / person events | Debounce or confidence | Lower `CONSECUTIVE_FRAMES_REQUIRED` or thresholds in `object_module.py` |
| Low FPS | CPU load | Increase `yolo_every_n_frames`; ensure `device="cpu"` is acceptable |

---

## Output artifacts

- **Console:** Real-time `[WARNING]` / `[ALERT]` lines; final structured report.
- **`violations/`:** One JPG per logged violation (`{type}_{timestamp}.jpg`).

Consider adding `violations/` and `.venv310/` to `.gitignore` if using version control.

---

## Limitations and disclaimer

- Heuristic gaze and identity checks are **not** biometrically rigorous; lighting, glasses, and movement affect accuracy.
- YOLO may miss partially visible phones or count posters/reflections as persons.
- Identity “different person” is a **drift warning**, not proof of impersonation.
- Intended as an **academic monitoring aid**, not a legal or compliance-grade proctoring system.
- Processing is local; no network upload is implemented in this codebase.

---

## Quick command reference

```powershell
cd E:\RIFAZ\CV-V1.2
.\.venv310\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
# Press 'q' in the video window to quit and view report
```

---

## License

Not specified in repository. Add a `LICENSE` file if you plan to distribute this project.
