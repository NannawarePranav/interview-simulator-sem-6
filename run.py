import os
import threading
import time
import sys
from flask import Flask, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
from interview.controller import InterviewController
from interview.report import generate_report
from data.resume_parser import extract_from_resume
from data.session_db import get_recent_sessions
from config import PROCTOR_VIOLATION_LIMIT
import numpy as np
from collections import defaultdict
import cv2

# ── Dynamic Import of cv-procter Package ──────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cv-procter'))
try:
    from face_module import FaceMonitor
    from object_module import ObjectMonitor
    from scoring import ScoringManager
    _cv_proctor_available = True
    print("[V2 CV-Proctor] Successfully imported proctoring modules!")
except Exception as e:
    print(f"[V2 CV-Proctor] Skipping python-based proctoring: {e}")
    _cv_proctor_available = False

class ServerProctor:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.cap = None
        self.current_frame = None
        self.current_violation_msg = None
        
        self.scoring = None
        self.face_monitor = None
        self.object_monitor = None
        self.last_violation_time = defaultdict(float)
        self.consecutive_flags = defaultdict(int)

    def preload_models(self):
        if not _cv_proctor_available:
            print("[ServerProctor] Skipping preload - CV Proctor modules not available.")
            return
        
        with self.lock:
            if self.face_monitor is not None and self.object_monitor is not None:
                return  # already loaded
                
            print("[ServerProctor] Warmloading MediaPipe and YOLOv8 models into memory on server start...")
            try:
                self.face_monitor = FaceMonitor(
                    look_away_seconds=3.0,
                    gaze_x_threshold=0.22,
                    gaze_y_threshold=0.22,
                    yaw_threshold=0.20,
                    pitch_threshold=0.20,
                    identity_distance_threshold=0.32,
                )
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cv-procter', 'yolov8n.pt')
                self.object_monitor = ObjectMonitor(
                    model_name=model_path,
                    yolo_every_n_frames=3,
                    conf_threshold=0.28,
                    phone_conf_threshold=0.30,
                    person_conf_threshold=0.55
                )
                print("[ServerProctor] CV Models preloaded successfully and warmed up!")
            except Exception as e:
                print(f"[ServerProctor] Warning during model preloading: {e}")

    def start(self):
        if not _cv_proctor_available:
            print("[ServerProctor] Skipping start - CV Proctor modules not available.")
            return
        with self.lock:
            if self.running:
                return
            self.running = True
            
            # Reset proctor structures
            self.scoring = ScoringManager(initial_score=10)
            
            if self.face_monitor is None:
                self.face_monitor = FaceMonitor(
                    look_away_seconds=3.0,
                    gaze_x_threshold=0.22,
                    gaze_y_threshold=0.22,
                    yaw_threshold=0.20,
                    pitch_threshold=0.20,
                    identity_distance_threshold=0.32,
                )
            else:
                # Reset stateful attributes on face monitor
                self.face_monitor.away_start_time = None
                self.face_monitor.reference_embedding = None
                self.face_monitor.identity_drift_detected = False
                
            if self.object_monitor is None:
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cv-procter', 'yolov8n.pt')
                self.object_monitor = ObjectMonitor(
                    model_name=model_path,
                    yolo_every_n_frames=3,
                    conf_threshold=0.28,
                    phone_conf_threshold=0.30,
                    person_conf_threshold=0.55
                )
            
            self.last_violation_time = defaultdict(float)
            self.consecutive_flags = defaultdict(int)
            self.current_frame = None
            self.current_violation_msg = None
            
            # Start thread
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            print("[ServerProctor] Daemon thread started.")

    def stop(self):
        with self.lock:
            if not self.running:
                return
            self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[ServerProctor] Daemon thread stopped.")

    def _run_loop(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ServerProctor] ERROR: Could not open local webcam. Streaming placeholder.")
            while True:
                with self.lock:
                    if not self.running:
                        break
                # Create a black dummy frame
                dummy = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(dummy, "Webcam Unavailable", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(dummy, "Please check connection", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                with self.lock:
                    self.current_frame = dummy
                time.sleep(0.1)
            return

        cooldowns = {
            "no_face": 2.0,
            "look_away": 4.0,
            "multiple_people": 3.0,
            "phone_detected": 2.5,
            "different_person": 8.0,
        }
        consecutive_reqs = {
            "look_away": 15,
            "multiple_people": 8,
            "phone_detected": 2,
            "different_person": 20,
        }

        violation_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'violations')
        os.makedirs(violation_dir, exist_ok=True)

        prev_fps_time = time.time()
        fps = 0.0

        while True:
            with self.lock:
                if not self.running:
                    break

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            now = time.time()
            face_result = self.face_monitor.process_frame(frame)
            object_result = self.object_monitor.process_frame(frame)
            
            # Real-time transient warning cleared by default every frame
            current_warning = None

            # ── Check no face ────────────────────────────────────────────────
            if not face_result.face_present:
                current_warning = "No face detected"
                last = self.last_violation_time.get("no_face", 0.0)
                if now - last >= cooldowns["no_face"]:
                    self.last_violation_time["no_face"] = now
                    filename = f"no_face_{int(now)}.jpg"
                    screenshot_path = os.path.join(violation_dir, filename)
                    cv2.imwrite(screenshot_path, frame)
                    self.scoring.add_violation("no_face", "No face detected", f"violations/{filename}")
                    print("[PROCTOR WARNING] No face detected")

            # ── Check look away ──────────────────────────────────────────────
            if face_result.face_present and face_result.looking_away:
                self.consecutive_flags["look_away"] += 1
                if self.consecutive_flags["look_away"] >= 4:
                    current_warning = "Please look at the screen!"
            else:
                self.consecutive_flags["look_away"] = 0

            if self.consecutive_flags["look_away"] >= consecutive_reqs["look_away"]:
                last = self.last_violation_time.get("look_away", 0.0)
                if now - last >= cooldowns["look_away"]:
                    self.last_violation_time["look_away"] = now
                    direction = face_result.look_direction
                    filename = f"look_away_{int(now)}.jpg"
                    screenshot_path = os.path.join(violation_dir, filename)
                    cv2.imwrite(screenshot_path, frame)
                    self.scoring.add_violation("look_away", f"Candidate looking away ({direction}) for too long", f"violations/{filename}")
                    print(f"[PROCTOR WARNING] Candidate looking away ({direction})")
                    self.consecutive_flags["look_away"] = 0

            # ── Check multiple people ─────────────────────────────────────────
            if object_result.person_count > 1:
                self.consecutive_flags["multiple_people"] += 1
                current_warning = "Multiple people detected"
            else:
                self.consecutive_flags["multiple_people"] = 0

            if self.consecutive_flags["multiple_people"] >= consecutive_reqs["multiple_people"]:
                last = self.last_violation_time.get("multiple_people", 0.0)
                if now - last >= cooldowns["multiple_people"]:
                    self.last_violation_time["multiple_people"] = now
                    filename = f"multiple_people_{int(now)}.jpg"
                    screenshot_path = os.path.join(violation_dir, filename)
                    cv2.imwrite(screenshot_path, frame)
                    self.scoring.add_violation("multiple_people", f"Multiple people detected ({object_result.person_count})", f"violations/{filename}")
                    print(f"[PROCTOR ALERT] Multiple people detected ({object_result.person_count})")
                    self.consecutive_flags["multiple_people"] = 0

            # ── Check phone detected ──────────────────────────────────────────
            if object_result.phone_detected:
                self.consecutive_flags["phone_detected"] += 1
                current_warning = "Mobile phone detected"
            else:
                self.consecutive_flags["phone_detected"] = 0

            if self.consecutive_flags["phone_detected"] >= consecutive_reqs["phone_detected"]:
                last = self.last_violation_time.get("phone_detected", 0.0)
                if now - last >= cooldowns["phone_detected"]:
                    self.last_violation_time["phone_detected"] = now
                    filename = f"phone_detected_{int(now)}.jpg"
                    screenshot_path = os.path.join(violation_dir, filename)
                    cv2.imwrite(screenshot_path, frame)
                    self.scoring.add_violation("phone_detected", "Mobile phone detected", f"violations/{filename}")
                    print("[PROCTOR ALERT] Mobile phone detected")
                    self.consecutive_flags["phone_detected"] = 0

            # ── Check identity drift (different person) ────────────────────────
            if face_result.face_present and face_result.different_person:
                self.consecutive_flags["different_person"] += 1
                current_warning = "Possible different person detected"
            else:
                self.consecutive_flags["different_person"] = 0

            if self.consecutive_flags["different_person"] >= consecutive_reqs["different_person"]:
                last = self.last_violation_time.get("different_person", 0.0)
                if now - last >= cooldowns["different_person"]:
                    self.last_violation_time["different_person"] = now
                    filename = f"different_person_{int(now)}.jpg"
                    screenshot_path = os.path.join(violation_dir, filename)
                    cv2.imwrite(screenshot_path, frame)
                    self.scoring.add_violation("different_person", "Possible different person detected", f"violations/{filename}")
                    print("[PROCTOR ALERT] Possible different person detected")
                    self.consecutive_flags["different_person"] = 0

            # Compute FPS and draw overlay
            current_time = time.time()
            elapsed = current_time - prev_fps_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            prev_fps_time = current_time

            overlay = frame.copy()
            if face_result.face_present and face_result.face_bbox is not None:
                x1, y1, x2, y2 = face_result.face_bbox
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(overlay, "Candidate", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            
            cv2.putText(overlay, f"Score: {self.scoring.score}/10", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(overlay, f"Look: {face_result.look_direction}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(overlay, f"Persons: {object_result.person_count} | Phone: {object_result.phone_detected}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(overlay, f"FPS: {fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

            with self.lock:
                self.current_frame = overlay
                self.current_violation_msg = current_warning

            time.sleep(0.03)

    def get_current_frame(self):
        with self.lock:
            return self.current_frame

server_proctor = ServerProctor()

app = Flask(__name__, static_folder='frontend')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')

# ── Global session state ──────────────────────────────────────────────────────
controller = None

# ── Proctor state — updated by browser via POST /api/proctor/report ───────────
_proctor_lock       = threading.Lock()
_proctor_violations = 0


# ── Frontend routes ───────────────────────────────────────────────────────────
@app.route('/')
def serve_frontend_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_frontend_files(path):
    return send_from_directory(app.static_folder, path)


# ── Interview API ─────────────────────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
def api_upload_resume():
    global controller
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file provided'}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    extra_skills = request.form.get('skills', '').strip()

    filename = secure_filename(file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_resume.pdf')
    file.save(pdf_path)

    resume_txt_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resume.txt')
    skills_txt_path = os.path.join(app.config['UPLOAD_FOLDER'], 'skills.txt')

    extract_from_resume(pdf_path, resume_txt_path, skills_txt_path)

    if extra_skills:
        with open(skills_txt_path, 'a', encoding='utf-8') as f:
            for skill in extra_skills.split(','):
                s = skill.strip()
                if s:
                    f.write(f"{s}\n")

    controller = InterviewController()
    controller.start_session(resume_txt_path, skills_txt_path)

    # Start the server-side AI proctoring background thread!
    server_proctor.start()

    return jsonify({
        'status': 'success',
        'topics_to_cover': controller.state.topics_to_cover
    })


@app.route('/api/state', methods=['GET'])
def api_state():
    if not controller:
        return jsonify({'error': 'Session not started'}), 400
    return jsonify({
        'topics_covered':   controller.state.topics_covered,
        'current_topic':    controller.state.current_topic,
        'topics_to_cover':  controller.state.topics_to_cover
    })


@app.route('/api/next', methods=['GET'])
def api_next():
    if not controller:
        return jsonify({'error': 'Session not started'}), 400
    q = controller.next_question()
    if not q:
        return jsonify({'finished': True})
    return jsonify({
        'finished': False,
        'question': q,
        'topic':    controller.state.current_topic
    })


@app.route('/api/answer', methods=['POST'])
def api_answer():
    if not controller:
        return jsonify({'error': 'Session not started'}), 400
    data   = request.json
    answer = data.get('answer', '')

    if data.get('skip', False):
        score = 0.1
        topic = controller.current_q_data['topic']
        controller.state.scores[topic].append(score)
        controller.state.consecutive_weak_answers += 1
    else:
        score = controller.process_answer(answer)

    return jsonify({'score': score, 'topic': controller.state.current_topic})


@app.route('/api/report', methods=['GET'])
def api_report():
    if not controller:
        return jsonify({'error': 'Session not started'}), 400

    # Stop the server proctoring background thread!
    server_proctor.stop()

    # Safely attach proctor metrics if server proctor was active
    if server_proctor.scoring:
        controller.state.proctor_score = server_proctor.scoring.score
        controller.state.proctor_violations = len(server_proctor.scoring.events)
        controller.state.proctor_violations_breakdown = server_proctor.scoring.get_violation_counts()
        
        logs = []
        for event in server_proctor.scoring.events:
            logs.append({
                'timestamp': event.timestamp,
                'type': event.violation_type,
                'message': event.message,
                'screenshot_path': event.screenshot_path
            })
        controller.state.proctor_violations_log = logs
    else:
        controller.state.proctor_violations = _proctor_violations

    path, content = generate_report(controller.state)

    # Finalize session in DB
    controller.finalize_session()

    return jsonify({'report_content': content})


# ── Session History ───────────────────────────────────────────────────────────
@app.route('/api/history', methods=['GET'])
def api_history():
    sessions = get_recent_sessions(limit=10)
    return jsonify({'sessions': sessions})


# ── Server-Side AI Proctoring Routes ──────────────────────────────────────────

@app.route('/api/proctor/video_feed')
def proctor_video_feed():
    """Streams the live webcam feed with AI annotation overlays as MJPEG."""
    def generate():
        while True:
            frame = server_proctor.get_current_frame()
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.04)  # ~25 FPS
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/proctor/status', methods=['GET'])
def api_proctor_status():
    """Polls real-time violation count, scoring metrics, and latest alerts."""
    limit = getattr(__import__('config'), 'PROCTOR_VIOLATION_LIMIT', 5)
    
    if server_proctor and server_proctor.running and server_proctor.scoring:
        violations_count = len(server_proctor.scoring.events)
        score = server_proctor.scoring.score
        last_violation = server_proctor.current_violation_msg
        return jsonify({
            'violations': violations_count,
            'score': score,
            'last_violation': last_violation,
            'active': True,
            'limit': limit
        })
        
    return jsonify({
        'violations': 0,
        'score': 10,
        'last_violation': None,
        'active': False,
        'limit': limit
    })


@app.route('/api/proctor/report', methods=['POST'])
def api_proctor_report():
    """Browser posts violation count (kept for backward compatibility)."""
    global _proctor_violations
    data = request.json or {}
    with _proctor_lock:
        _proctor_violations = int(data.get('violations', _proctor_violations))
    return jsonify({'status': 'ok', 'violations': _proctor_violations})


@app.route('/api/proctor/start', methods=['GET'])
def api_proctor_start():
    """Triggers server proctor start manually."""
    server_proctor.start()
    return jsonify({'status': 'ok'})


@app.route('/api/proctor/stop', methods=['GET'])
def api_proctor_stop():
    """Triggers server proctor stop manually."""
    server_proctor.stop()
    return jsonify({'status': 'ok'})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Starting AI Mock Interviewer server on http://localhost:5000")
    server_proctor.preload_models()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
