from __future__ import annotations

import hmac
import os
import uuid
from functools import wraps
from pathlib import Path

import cv2
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from ensemble_engine import SafetyEnsembleEngine
from incident_store import IncidentStore


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
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

engine = SafetyEnsembleEngine(str(BASE_DIR / "models" / "ensemble_config.json"))
store = IncidentStore(str(BASE_DIR / "data" / "incidents.db"))


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("authenticated"):
            if request.path.startswith("/api/") or request.path.startswith("/analyze"):
                return jsonify({"error": "authentication_required"}), 401
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper


def extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def unique_name(filename: str) -> str:
    suffix = Path(secure_filename(filename)).suffix.lower()
    return f"{uuid.uuid4().hex}{suffix}"


def serialize_frame_result(result):
    return {
        "frame_index": result["frame_index"],
        "workers": result["workers"],
        "equipment": result["equipment"],
        "danger_zones": result["danger_zones"],
        "incidents": result["incidents"],
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        valid_user = hmac.compare_digest(username, SUPERVISOR_USER)
        valid_password = hmac.compare_digest(password, SUPERVISOR_PASSWORD)
        if valid_user and valid_password:
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
    return render_template("index.html", username=session.get("username", "supervisor"))


@app.get("/api/status")
@login_required
def status():
    return jsonify(
        {
            "status": "ready",
            "inference": "REAL_TWO_MODEL_ENSEMBLE",
            "models": engine.model_status(),
            "incident_store": str(store.db_path),
        }
    )


@app.post("/analyze")
@login_required
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    uploaded = request.files["image"]
    if not uploaded.filename or extension(uploaded.filename) not in IMAGE_EXTENSIONS:
        return jsonify({"error": "Invalid image type"}), 400

    filename = unique_name(uploaded.filename)
    input_path = UPLOAD_DIR / filename
    output_name = f"annotated_{Path(filename).stem}.jpg"
    output_path = OUTPUT_DIR / output_name
    uploaded.save(input_path)
    try:
        frame = cv2.imread(str(input_path))
        if frame is None:
            return jsonify({"error": "Unable to decode image"}), 400
        result = engine.analyze_frame(frame, frame_index=0)
        annotated = engine.draw_overlay(frame, result)
        cv2.imwrite(str(output_path), annotated)
        saved = store.add_many(filename, result["incidents"])
        response = serialize_frame_result(result)
        response.update(
            {
                "annotated_image": f"/static/results/{output_name}",
                "saved_incidents": saved,
                "inference": "REAL_TWO_MODEL_ENSEMBLE",
            }
        )
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

    filename = unique_name(uploaded.filename)
    input_path = UPLOAD_DIR / filename
    output_name = f"safety_{Path(filename).stem}.mp4"
    output_path = OUTPUT_DIR / output_name
    uploaded.save(input_path)
    try:
        sample_every = max(1, int(request.form.get("sample_every_n_frames", 3)))
        report = engine.process_video(str(input_path), str(output_path), sample_every_n_frames=sample_every)
        saved = store.add_many(filename, report["incidents"])
        return jsonify(
            {
                "status": "completed",
                "inference": "REAL_TWO_MODEL_ENSEMBLE",
                "video": f"/static/results/{output_name}",
                "frames": report["frames"],
                "analyzed_frames": report["analyzed_frames"],
                "incident_count": len(report["incidents"]),
                "saved_incidents": saved,
                "incidents": report["incidents"],
            }
        )
    finally:
        input_path.unlink(missing_ok=True)


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
    return jsonify(
        {
            "status": "EMERGENCY_RESPONSE_TRIGGERED",
            "mode": "APPLICATION_SIMULATION",
            "camera": payload.get("camera"),
            "incident_id": payload.get("incident_id"),
            "actions": ["raise_critical_alert", "notify_supervisor", "mark_incident_emergency"],
        }
    )


if __name__ == "__main__":
    print("Smart Workplace Safety — REAL TWO-MODEL ENSEMBLE")
    print(engine.model_status())
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
