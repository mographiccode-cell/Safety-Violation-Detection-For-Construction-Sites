from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2

from track_action_manager import TrackActionManager


class CombinedSafetyPipeline:
    """Composes the stable safety detector/reasoner with real per-track TSSTG Action AI."""

    def __init__(self, safety_engine):
        self.safety = safety_engine
        cfg = safety_engine.config
        selected = cfg["selected_models"]
        thresholds = cfg["thresholds"]
        self.action_fall_min_confidence = float(thresholds["action_fall_min_confidence"])
        self.action_min_pose_coverage = float(thresholds["action_min_pose_coverage"])
        self.action = TrackActionManager(
            pose_model_path=str(safety_engine.config_path.parent.parent / selected["action_pose"]["path"]),
            action_weight_path=str(safety_engine.config_path.parent.parent / selected["action_sequence"]["path"]),
            sequence_length=int(thresholds["action_sequence_frames"]),
            inference_step=int(thresholds["action_inference_step"]),
            min_pose_coverage=self.action_min_pose_coverage,
        )
        self.last_action_incident_frame: Dict[int, int] = {}
        self.action_incident_cooldown = int(thresholds.get("incident_cooldown_frames", 30))

    def reset(self) -> None:
        self.safety.reset_tracking()
        self.action.reset()
        self.last_action_incident_frame.clear()

    def status(self) -> Dict[str, Any]:
        return {
            "safety": self.safety.model_status(),
            "action": self.action.status(),
            "fall_down_hazard_gate": {
                "min_confidence": self.action_fall_min_confidence,
                "min_pose_coverage": self.action_min_pose_coverage,
            },
        }

    def _action_incident_allowed(self, track_id: int, frame_index: int) -> bool:
        last = self.last_action_incident_frame.get(track_id)
        if last is None or frame_index - last >= self.action_incident_cooldown:
            self.last_action_incident_frame[track_id] = frame_index
            return True
        return False

    def analyze_frame(self, frame, frame_index: int, fixed_zones: Optional[List[Dict[str, Any]]] = None):
        result = self.safety.analyze_frame(frame, frame_index, fixed_zones=fixed_zones)
        action_states = self.action.update(frame, result["workers"], frame_index)
        rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

        for worker in result["workers"]:
            track_id = int(worker["track_id"])
            state = action_states.get(track_id, {
                "status": "no_pose_evidence",
                "source": "YOLO_POSE_TSSTG_SEQUENCE",
                "collected_frames": 0,
                "required_frames": 30,
                "pose_coverage": 0.0,
            })
            worker["action"] = state

            confirmed_fall = (
                state.get("status") == "classified"
                and state.get("action") == "Fall Down"
                and float(state.get("confidence", 0.0)) >= self.action_fall_min_confidence
                and float(state.get("pose_coverage", 0.0)) >= self.action_min_pose_coverage
            )
            if confirmed_fall:
                hazard = {
                    "type": "FALL_DOWN_ACTION",
                    "severity": "CRITICAL",
                    "confidence": float(state["confidence"]),
                    "source": "TSSTG_SEQUENCE_30_FRAME",
                    "pose_coverage": float(state["pose_coverage"]),
                }
                worker["hazards"].append(hazard)
                worker["severity"] = max(
                    [worker["severity"], hazard["severity"]],
                    key=lambda s: rank[s],
                )
                worker["alert"] = True
                if self._action_incident_allowed(track_id, frame_index):
                    result["incidents"].append({
                        "track_id": track_id,
                        **hazard,
                        "frame_index": frame_index,
                    })
        return result

    def draw_overlay(self, frame, result):
        out = self.safety.draw_overlay(frame, result)
        for worker in result.get("workers", []):
            action = worker.get("action") or {}
            x1, y1, x2, y2 = map(int, worker["box"])
            if action.get("status") == "classified":
                text = f"Action AI: {action.get('action')} {float(action.get('confidence', 0.0)):.0%} | Pose {float(action.get('pose_coverage', 0.0)):.0%}"
                color = (0, 0, 255) if action.get("action") == "Fall Down" else (255, 190, 0)
            else:
                collected = int(action.get("collected_frames", 0))
                required = int(action.get("required_frames", 30))
                text = f"Action AI: collecting temporal pose {collected}/{required}"
                color = (180, 180, 180)
            ty = max(24, y1 - 32)
            cv2.putText(out, text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
        return out

    def process_video(self, input_path: str, output_path: str, sample_every_n_frames: int = 3, fixed_zones=None):
        self.reset()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        frame_index = 0
        analyzed_frames = 0
        incidents = []
        last_result = None
        action_summary: Dict[int, Dict[str, Any]] = {}
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % max(1, int(sample_every_n_frames)) == 0:
                    last_result = self.analyze_frame(frame, frame_index, fixed_zones=fixed_zones)
                    incidents.extend(last_result["incidents"])
                    analyzed_frames += 1
                    for worker in last_result["workers"]:
                        if worker.get("action", {}).get("status") == "classified":
                            action_summary[int(worker["track_id"])] = worker["action"]
                if last_result is not None:
                    frame = self.draw_overlay(frame, last_result)
                writer.write(frame)
                frame_index += 1
        finally:
            cap.release()
            writer.release()
        return {
            "input": input_path,
            "output": output_path,
            "frames": frame_index,
            "analyzed_frames": analyzed_frames,
            "incidents": incidents,
            "latest_actions_by_track": action_summary,
        }
