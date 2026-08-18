from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from combined_pipeline import CombinedSafetyPipeline
from ensemble_engine import box_iou, center, expand_box, point_in_box, point_in_polygon


class TiledEquipmentCombinedPipeline(CombinedSafetyPipeline):
    """Production wrapper: precision-gated workers + tiled machinery + TSSTG.

    Tiled machinery candidates are real outputs from the construction model. The
    lower tile threshold is never accepted from one frame: machinery must be seen
    in at least 3 of the last 5 analyzed frames before it creates dynamic zones.
    """

    def __init__(self, safety_engine, tile_min_confidence=None, tile_required=None, tile_window=None, tile_rois=None):
        super().__init__(safety_engine)
        t = safety_engine.config.get("thresholds", {})
        self.single_model_person_min = float(t.get("single_model_person_min_confidence", 0.70))
        self.tile_min_confidence = float(tile_min_confidence if tile_min_confidence is not None else t.get("tiled_equipment_candidate_min_confidence", 0.12))
        self.tile_required = int(tile_required if tile_required is not None else t.get("tiled_equipment_required_frames", 3))
        window = int(tile_window if tile_window is not None else t.get("tiled_equipment_window_frames", 5))
        self.tile_history = deque(maxlen=window)
        self.tile_rois = tile_rois or [(0.0, 0.70, "tile_left"), (0.15, 0.85, "tile_center"), (0.30, 1.0, "tile_right")]
        self.last_tiled_equipment: List[Dict[str, Any]] = []

    def reset(self):
        super().reset()
        self.tile_history.clear()
        self.last_tiled_equipment = []

    def _precision_gate_workers(self, frame, result, frame_index):
        """Reject weak single-model persons and clamp foot geometry to image."""
        retained = []
        kept_ids = set()
        h, w = frame.shape[:2]
        for worker in result.get("workers", []):
            sources = worker.get("person_sources", [])
            conf = float(worker.get("person_confidence", 0.0))
            if len(sources) < 2 and conf < self.single_model_person_min:
                continue
            track_id = int(worker["track_id"])
            kept_ids.add(track_id)
            fx, fy = worker.get("foot_point", (0.0, 0.0))
            fp = (min(max(float(fx), 0.0), float(w - 1)), min(max(float(fy), 0.0), float(h - 1)))
            worker["foot_point"] = fp

            hits = []
            for zone in result.get("restricted_zones", []):
                if str(zone.get("zone_type", "RESTRICTED")).upper() != "RESTRICTED":
                    continue
                polygon = np.asarray(zone.get("polygon_px", []), dtype=np.int32)
                if len(polygon) >= 3 and point_in_polygon(fp, polygon):
                    hits.append({"id": zone.get("id"), "name": zone.get("name"), "zone_type": zone.get("zone_type", "RESTRICTED")})
            worker["restricted_zone_hits"] = hits
            if hits and not any(x.get("type") == "RESTRICTED_ZONE_ENTRY" for x in worker.get("hazards", [])):
                hazard = {"type": "RESTRICTED_ZONE_ENTRY", "severity": "HIGH", "confidence": 1.0, "source": "tracked_worker_geometry_clamped", "zone_id": hits[0].get("id"), "zone_name": hits[0].get("name")}
                worker["hazards"].append(hazard)
                worker["severity"] = "HIGH" if worker.get("severity") != "CRITICAL" else "CRITICAL"
                worker["alert"] = True
                track = self.safety.tracker.tracks.get(track_id)
                if track is not None and self.safety._incident_allowed(track, hazard, frame_index):
                    result["incidents"].append({"track_id": track_id, **hazard, "frame_index": frame_index})
            retained.append(worker)
        result["workers"] = retained
        result["incidents"] = [x for x in result.get("incidents", []) if int(x.get("track_id", -1)) in kept_ids]

    def _tiled_candidates(self, frame):
        h, w = frame.shape[:2]
        rois = [(int(xa * w), 0, int(xb * w), h, source) for xa, xb, source in self.tile_rois]
        candidates = []
        for x1, y1, x2, y2, source in rois:
            crop = frame[y1:y2, x1:x2]
            prediction = self.safety.secondary.predict(crop, imgsz=640, conf=max(0.04, self.tile_min_confidence / 3), iou=0.45, verbose=False)[0]
            if prediction.boxes is None:
                continue
            for box in prediction.boxes:
                name = self.safety.secondary.names[int(box.cls.item())]
                conf = float(box.conf.item())
                if name != "machinery" or conf < self.tile_min_confidence:
                    continue
                bx1, by1, bx2, by2 = [float(v) for v in box.xyxy[0].tolist()]
                candidates.append({"class_name": name, "confidence": conf, "xyxy": (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1), "source": f"secondary_{source}_temporal_candidate"})
        candidates.sort(key=lambda d: d["confidence"], reverse=True)
        merged = []
        for det in candidates:
            if any(box_iou(det["xyxy"], old["xyxy"]) > 0.45 for old in merged):
                continue
            merged.append(det)
        return merged[:3]

    def _enhance_equipment(self, frame, result, frame_index):
        if result.get("equipment"):
            self.tile_history.append(True)
            self.last_tiled_equipment = result["equipment"]
            return
        current = self._tiled_candidates(frame)
        self.tile_history.append(bool(current))
        if current:
            self.last_tiled_equipment = current
        if sum(bool(x) for x in self.tile_history) < self.tile_required or not self.last_tiled_equipment:
            return

        equipment = [dict(x) for x in (current or self.last_tiled_equipment)]
        if not current:
            for det in equipment:
                det["source"] = det["source"].replace("_candidate", "_temporal_hold")
        danger_zones = [{"equipment_class": d["class_name"], "confidence": d["confidence"], "equipment_box": d["xyxy"], "equipment_source": d["source"], "zone_box": expand_box(d["xyxy"], frame.shape[:2])} for d in equipment]
        result["equipment"] = equipment
        result["danger_zones"] = danger_zones

        for worker in result.get("workers", []):
            fp = tuple(worker["foot_point"])
            relationships = []
            for zone in danger_zones:
                equipment_point = center(zone["equipment_box"])
                if point_in_box(fp, zone["zone_box"]):
                    dx, dy = fp[0] - equipment_point[0], fp[1] - equipment_point[1]
                    relationships.append({"type": "CLOSE_TO", "equipment_class": zone["equipment_class"], "equipment_confidence": zone["confidence"], "equipment_source": zone["equipment_source"], "distance_px": (dx * dx + dy * dy) ** 0.5, "distance_unit": "pixels_uncalibrated", "worker_point": fp, "equipment_point": equipment_point})
            worker["relationships"] = relationships
            worker["inside_dynamic_danger_zone"] = bool(relationships)
            track = self.safety.tracker.tracks.get(int(worker["track_id"]))
            if track is not None:
                if track.proximity_history:
                    track.proximity_history[-1] = bool(relationships)
                else:
                    track.proximity_history.append(bool(relationships))
                confirmed = track.confirmed(track.proximity_history, self.safety.temporal_required)
            else:
                confirmed = False
            worker["proximity_temporal_confirmed"] = confirmed
            if confirmed and not any(h.get("type") == "DANGEROUS_MACHINE_PROXIMITY" for h in worker.get("hazards", [])):
                hazard = {"type": "DANGEROUS_MACHINE_PROXIMITY", "severity": "CRITICAL", "confidence": max((r["equipment_confidence"] for r in relationships), default=0.0), "source": "tiled_real_equipment_model_plus_tracked_geometry_3of5"}
                worker["hazards"].append(hazard)
                worker["severity"] = "CRITICAL"
                worker["alert"] = True
                if track is not None and self.safety._incident_allowed(track, hazard, frame_index):
                    result["incidents"].append({"track_id": int(worker["track_id"]), **hazard, "frame_index": frame_index})

    def analyze_frame(self, frame, frame_index: int, fixed_zones: Optional[List[Dict[str, Any]]] = None):
        result = self.safety.analyze_frame(frame, frame_index, fixed_zones=fixed_zones)
        self._precision_gate_workers(frame, result, frame_index)
        self._enhance_equipment(frame, result, frame_index)
        action_states = self.action.update(frame, result["workers"], frame_index)
        for worker in result["workers"]:
            track_id = int(worker["track_id"])
            state = action_states.get(track_id, {"status": "no_pose_evidence", "source": "YOLO_POSE_TSSTG_SEQUENCE", "collected_frames": 0, "required_frames": 30, "pose_coverage": 0.0})
            worker["action"] = state
            confirmed_fall = state.get("status") == "classified" and state.get("action") == "Fall Down" and float(state.get("confidence", 0.0)) >= self.action_fall_min_confidence and float(state.get("pose_coverage", 0.0)) >= self.action_min_pose_coverage
            if confirmed_fall and not any(h.get("type") == "FALL_DOWN_ACTION" for h in worker.get("hazards", [])):
                hazard = {"type": "FALL_DOWN_ACTION", "severity": "CRITICAL", "confidence": float(state["confidence"]), "source": "TSSTG_SEQUENCE_30_FRAME", "pose_coverage": float(state["pose_coverage"])}
                worker["hazards"].append(hazard)
                worker["severity"] = "CRITICAL"
                worker["alert"] = True
                if self._action_incident_allowed(track_id, frame_index):
                    result["incidents"].append({"track_id": track_id, **hazard, "frame_index": frame_index})
        return result
