from __future__ import annotations

import hmac
import os
import threading
import time
import uuid
from functools import wraps
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from ensemble_engine import SafetyEnsembleEngine
from incident_store import IncidentStore
from zone_store import ZoneStore


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "results"
TEMPLATE_DIR = BASE_DIR / "templates"
for folder in (UPLOAD_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(BASE_DIR / "static"))
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY") or uuid.uuid4().hex + uuid.uuid4().hex
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

SUPERVISOR_USER = os.environ.get("SUPERVISOR_USER", "supervisor")
SUPERVISOR_PASSWORD = os.environ.get("SUPERVISOR_PASSWORD", "Safety@2026!")
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "0")
CAMERA_ID = os.environ.get("CAMERA_ID", "default")
LIVE_SAMPLE_EVERY = max(1, int(os.environ.get("LIVE_SAMPLE_EVERY", "3")))
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

engine = SafetyEnsembleEngine(str(BASE_DIR / "models" / "ensemble_config.json"))
store = IncidentStore(str(BASE_DIR / "data" / "incidents.db"))
zones = ZoneStore(str(BASE_DIR / "data" / "incidents.db"))
inference_lock = threading.RLock()


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("authenticated"):
            if request.path.startswith("/api/") or request.path.startswith("/analyze") or request.path.startswith("/live-feed"):
                return jsonify({"error": "authentication_required"}), 401
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper


def extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def unique_name(filename: str) -> str:
    suffix = Path(secure_filename(filename)).suffix.lower()
    return f"{uuid.uuid4().hex}{suffix}"


def camera_source_value():
    return int(CAMERA_SOURCE) if CAMERA_SOURCE.strip().isdigit() else CAMERA_SOURCE


def serialize_frame_result(result):
    return {
        "frame_index": result["frame_index"],
        "workers": result["workers"],
        "equipment": result["equipment"],
        "danger_zones": result["danger_zones"],
        "restricted_zones": result.get("restricted_zones", []),
        "incidents": result["incidents"],
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if hmac.compare_digest(username, SUPERVISOR_USER) and hmac.compare_digest(password, SUPERVISOR_PASSWORD):
            session.clear()
            session["authenticated"] = True
            session["username"] = SUPERVISOR_USER
            return redirect(url_for("index"))
        error = "اسم المستخدم أو كلمة المرور غير صحيحة"
    return render_template("login.html", error=error)


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/")
@login_required
def index():
    return render_template("index.html", username=session.get("username", "supervisor"), camera_id=CAMERA_ID)


@app.get("/api/status")
@login_required
def status():
    return jsonify({
        "status": "ready",
        "inference": "REAL_TWO_MODEL_ENSEMBLE",
        "models": engine.model_status(),
        "incident_store": str(store.db_path),
        "camera_id": CAMERA_ID,
        "camera_source_configured": bool(CAMERA_SOURCE),
        "live_sample_every": LIVE_SAMPLE_EVERY,
    })


@app.post("/analyze")
@login_required
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    uploaded = request.files["image"]
    if not uploaded.filename or extension(uploaded.filename) not in IMAGE_EXTENSIONS:
        return jsonify({"error": "Invalid image type"}), 400
    camera_id = request.form.get("camera_id", CAMERA_ID)
    filename = unique_name(uploaded.filename)
    input_path = UPLOAD_DIR / filename
    output_name = f"annotated_{Path(filename).stem}.jpg"
    output_path = OUTPUT_DIR / output_name
    uploaded.save(input_path)
    try:
        frame = cv2.imread(str(input_path))
        if frame is None:
            return jsonify({"error": "Unable to decode image"}), 400
        with inference_lock:
            engine.reset_tracking()
            result = engine.analyze_frame(frame, frame_index=0, fixed_zones=zones.list(camera_id, enabled_only=True))
        cv2.imwrite(str(output_path), engine.draw_overlay(frame, result))
        saved = store.add_many(filename, result["incidents"])
        response = serialize_frame_result(result)
        response.update({"annotated_image": f"/static/results/{output_name}", "saved_incidents": saved, "inference": "REAL_TWO_MODEL_ENSEMBLE"})
        return jsonify(response)
    finally:
        input_path.unlink(missing_ok=True)


@app.post("/analyze-video")
@login_required
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400
    uploaded = request.files["video"]
    if not uploaded.filename or extension(uploaded.filename) not in VIDEO_EXTENSIONS:
        return jsonify({"error": "Invalid video type"}), 400
    camera_id = request.form.get("camera_id", CAMERA_ID)
    filename = unique_name(uploaded.filename)
    input_path = UPLOAD_DIR / filename
    output_name = f"safety_{Path(filename).stem}.mp4"
    output_path = OUTPUT_DIR / output_name
    uploaded.save(input_path)
    try:
        sample_every = max(1, int(request.form.get("sample_every_n_frames", 3)))
        with inference_lock:
            report = engine.process_video(
                str(input_path), str(output_path), sample_every_n_frames=sample_every,
                fixed_zones=zones.list(camera_id, enabled_only=True),
            )
        saved = store.add_many(filename, report["incidents"])
        return jsonify({
            "status": "completed", "inference": "REAL_TWO_MODEL_ENSEMBLE",
            "video": f"/static/results/{output_name}", "frames": report["frames"],
            "analyzed_frames": report["analyzed_frames"], "incident_count": len(report["incidents"]),
            "saved_incidents": saved, "incidents": report["incidents"],
        })
    finally:
        input_path.unlink(missing_ok=True)


def live_stream_generator():
    source = camera_source_value()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        # Yield an HTTP-valid JPEG-like stream is not useful; generator simply ends.
        return
    engine.reset_tracking()
    frame_index = 0
    last_result = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % LIVE_SAMPLE_EVERY == 0:
                with inference_lock:
                    last_result = engine.analyze_frame(frame, frame_index, fixed_zones=zones.list(CAMERA_ID, enabled_only=True))
                if last_result["incidents"]:
                    store.add_many(f"live:{CAMERA_ID}", last_result["incidents"])
            if last_result:
                frame = engine.draw_overlay(frame, last_result)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            frame_index += 1
    finally:
        cap.release()


@app.get("/live-feed")
@login_required
def live_feed():
    return Response(live_stream_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/zones")
@login_required
def list_zones():
    camera_id = request.args.get("camera_id", CAMERA_ID)
    return jsonify({"camera_id": camera_id, "zones": zones.list(camera_id)})


@app.post("/api/zones")
@login_required
def save_zone():
    data = request.get_json(silent=True) or {}
    try:
        zone = zones.upsert(
            camera_id=data.get("camera_id", CAMERA_ID),
            name=data.get("name", "Restricted Zone"),
            points=data.get("points", []),
            zone_type=data.get("zone_type", "RESTRICTED"),
            enabled=data.get("enabled", True),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "saved", "zone": zone}), 201


@app.delete("/api/zones/<int:zone_id>")
@login_required
def delete_zone(zone_id: int):
    return jsonify({"deleted": zones.delete(zone_id), "zone_id": zone_id})


@app.get("/api/incidents")
@login_required
def incidents():
    limit = min(max(int(request.args.get("limit", 100)), 1), 500)
    return jsonify({"incidents": store.recent(limit)})


@app.get("/api/reports/summary")
@login_required
def report_summary():
    return jsonify(store.summary())


@app.post("/api/emergency")
@login_required
def emergency_response():
    payload = request.get_json(silent=True) or {}
    return jsonify({
        "status": "EMERGENCY_RESPONSE_TRIGGERED",
        "mode": "APPLICATION_SIMULATION",
        "camera": payload.get("camera", CAMERA_ID),
        "incident_id": payload.get("incident_id"),
        "actions": ["raise_critical_alert", "notify_supervisor", "mark_incident_emergency"],
        "timestamp": time.time(),
    })


if __name__ == "__main__":
    print("Smart Workplace Safety — REAL TWO-MODEL ENSEMBLE")
    print(engine.model_status())
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
