from __future__ import annotations

import os
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory
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

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

engine = SafetyEnsembleEngine(str(BASE_DIR / "models" / "ensemble_config.json"))
store = IncidentStore(str(BASE_DIR / "data" / "incidents.db"))


def extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def unique_name(filename: str) -> str:
    suffix = Path(secure_filename(filename)).suffix.lower()
    return f"{uuid.uuid4().hex}{suffix}"


def serialize_frame_result(result):
    # Remove large raw detection lists from the public response while preserving
    # the real worker/equipment inference values used by the hazard engine.
    return {
        "frame_index": result["frame_index"],
        "workers": result["workers"],
        "equipment": result["equipment"],
        "danger_zones": result["danger_zones"],
        "incidents": result["incidents"],
    }


@app.get("/")
def index():
    try:
        return render_template("index.html")
    except Exception:
        return jsonify(
            {
                "name": "Smart Workplace Safety",
                "status": "running",
                "model_status": engine.model_status(),
                "routes": ["/api/status", "/analyze", "/analyze-video", "/api/incidents", "/api/reports/summary"],
            }
        )


@app.get("/api/status")
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
def incidents():
    limit = min(max(int(request.args.get("limit", 100)), 1), 500)
    return jsonify({"incidents": store.recent(limit)})


@app.get("/api/reports/summary")
def report_summary():
    return jsonify(store.summary())


@app.post("/api/emergency")
def emergency_response():
    payload = request.get_json(silent=True) or {}
    # Application-level emergency response proof. Hardware/PLC activation is
    # deliberately not claimed unless an external controller is configured.
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
