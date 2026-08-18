from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


Box = Tuple[float, float, float, float]


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def center(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def foot_point(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def point_in_box(point: Tuple[float, float], box: Box) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def expand_box(box: Box, frame_shape: Tuple[int, int], x_ratio: float = 0.35, y_ratio: float = 0.18) -> Box:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return (
        max(0.0, x1 - bw * x_ratio),
        max(0.0, y1 - bh * y_ratio),
        min(float(w - 1), x2 + bw * x_ratio),
        min(float(h - 1), y2 + bh * y_ratio),
    )


def normalized_polygon_to_pixels(points: List[List[float]], frame_shape: Tuple[int, int]) -> np.ndarray:
    h, w = frame_shape[:2]
    return np.array(
        [[int(round(float(x) * (w - 1))), int(round(float(y) * (h - 1)))] for x, y in points],
        dtype=np.int32,
    )


def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    if polygon is None or len(polygon) < 3:
        return False
    return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0


def detection_dict(box, names: Dict[int, str]) -> Dict[str, Any]:
    cls_id = int(box.cls.item())
    return {
        "class_id": cls_id,
        "class_name": names[cls_id],
        "confidence": float(box.conf.item()),
        "xyxy": tuple(float(v) for v in box.xyxy[0].tolist()),
    }


def det_center_in_person(det: Dict[str, Any], person_box: Box) -> bool:
    return point_in_box(center(det["xyxy"]), person_box)


@dataclass
class TrackState:
    track_id: int
    box: Box
    last_frame: int
    helmet_history: deque = field(default_factory=lambda: deque(maxlen=5))
    vest_history: deque = field(default_factory=lambda: deque(maxlen=5))
    proximity_history: deque = field(default_factory=lambda: deque(maxlen=5))

    def update_box(self, box: Box, frame_index: int) -> None:
        self.box = box
        self.last_frame = frame_index

    @staticmethod
    def confirmed(history: deque, required: int) -> bool:
        return sum(bool(v) for v in history) >= required


class IoUTracker:
    """Associates real detector boxes across frames; it creates no detections."""

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 15, history_len: int = 5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.history_len = history_len
        self.next_id = 1
        self.tracks: Dict[int, TrackState] = {}

    def update(self, person_boxes: List[Box], frame_index: int) -> List[TrackState]:
        candidates = [(tid, tr) for tid, tr in self.tracks.items() if frame_index - tr.last_frame <= self.max_age]
        unused_tracks = {tid for tid, _ in candidates}
        output: List[TrackState] = []
        for pbox in person_boxes:
            best_tid, best_iou = None, 0.0
            for tid, tr in candidates:
                if tid not in unused_tracks:
                    continue
                score = box_iou(pbox, tr.box)
                if score > best_iou:
                    best_tid, best_iou = tid, score
            if best_tid is not None and best_iou >= self.iou_threshold:
                tr = self.tracks[best_tid]
                tr.update_box(pbox, frame_index)
                unused_tracks.remove(best_tid)
            else:
                tr = TrackState(
                    track_id=self.next_id,
                    box=pbox,
                    last_frame=frame_index,
                    helmet_history=deque(maxlen=self.history_len),
                    vest_history=deque(maxlen=self.history_len),
                    proximity_history=deque(maxlen=self.history_len),
                )
                self.tracks[self.next_id] = tr
                self.next_id += 1
            output.append(tr)
        stale = [tid for tid, tr in self.tracks.items() if frame_index - tr.last_frame > self.max_age]
        for tid in stale:
            self.tracks.pop(tid, None)
        return output


class SafetyEnsembleEngine:
    """Two real safety models + temporal consensus + spatial hazard reasoning."""

    def __init__(self, config_path: str = "models/ensemble_config.json"):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        root = self.config_path.parent.parent
        selected = self.config["selected_models"]
        primary_path = Path(selected["ppe_primary"]["path"])
        secondary_path = Path(selected["construction_secondary"]["path"])
        if not primary_path.is_absolute():
            primary_path = root / primary_path
        if not secondary_path.is_absolute():
            secondary_path = root / secondary_path
        self.primary_path = primary_path
        self.secondary_path = secondary_path
        self.primary = YOLO(str(primary_path))
        self.secondary = YOLO(str(secondary_path))
        t = self.config["thresholds"]
        self.person_min = float(t["person_min_confidence"])
        self.ppe_min = float(t["ppe_positive_min_confidence"])
        self.machinery_min = float(t["machinery_min_confidence"])
        self.vehicle_min = float(t["vehicle_min_confidence"])
        self.agreement_iou = float(t["iou_model_agreement"])
        self.temporal_window = int(t["temporal_window_frames"])
        self.temporal_required = int(t["temporal_required_frames"])
        self.tracker = IoUTracker(history_len=self.temporal_window)

    def reset_tracking(self) -> None:
        self.tracker = IoUTracker(history_len=self.temporal_window)

    def model_status(self) -> Dict[str, Any]:
        return {
            "primary": {"path": str(self.primary_path), "classes": self.primary.names},
            "secondary": {"path": str(self.secondary_path), "classes": self.secondary.names},
            "temporal_rule": f"{self.temporal_required}-of-{self.temporal_window}",
        }

    def _predict(self, model: YOLO, frame: np.ndarray, conf: float = 0.15) -> List[Dict[str, Any]]:
        result = model.predict(frame, imgsz=640, conf=conf, iou=0.45, verbose=False)[0]
        if result.boxes is None:
            return []
        return [detection_dict(b, model.names) for b in result.boxes]

    @staticmethod
    def _by_class(dets: Iterable[Dict[str, Any]], names: Iterable[str], min_conf: float = 0.0) -> List[Dict[str, Any]]:
        wanted = set(names)
        return [d for d in dets if d["class_name"] in wanted and d["confidence"] >= min_conf]

    def _person_detections(self, primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates = []
        for source, dets in (("primary", primary), ("secondary", secondary)):
            for d in self._by_class(dets, ["Person"], self.person_min):
                candidates.append({"box": d["xyxy"], "confidence": d["confidence"], "sources": [source]})
        candidates.sort(key=lambda d: d["confidence"], reverse=True)
        merged: List[Dict[str, Any]] = []
        for det in candidates:
            match = next((x for x in merged if box_iou(det["box"], x["box"]) > 0.45), None)
            if match:
                if det["confidence"] > match["confidence"]:
                    match["box"] = det["box"]
                    match["confidence"] = det["confidence"]
                for src in det["sources"]:
                    if src not in match["sources"]:
                        match["sources"].append(src)
            else:
                merged.append(det.copy())
        return merged

    def _positive_agreement(self, person_box: Box, primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]], primary_class: str, secondary_class: str) -> Dict[str, Any]:
        pa = [d for d in self._by_class(primary, [primary_class], self.ppe_min) if det_center_in_person(d, person_box)]
        sb = [d for d in self._by_class(secondary, [secondary_class], self.ppe_min) if det_center_in_person(d, person_box)]
        best = None
        for a in pa:
            for b in sb:
                iou = box_iou(a["xyxy"], b["xyxy"])
                if iou >= self.agreement_iou:
                    score = min(a["confidence"], b["confidence"])
                    if best is None or score > best["confidence"]:
                        best = {
                            "confirmed": True,
                            "confidence": score,
                            "primary_confidence": a["confidence"],
                            "secondary_confidence": b["confidence"],
                            "iou": iou,
                            "primary_box": a["xyxy"],
                            "secondary_box": b["xyxy"],
                        }
        if best:
            return best
        return {
            "confirmed": False,
            "confidence": 0.0,
            "primary_seen": max([d["confidence"] for d in pa], default=0.0),
            "secondary_seen": max([d["confidence"] for d in sb], default=0.0),
        }

    def _explicit_negative(self, person_box: Box, secondary: List[Dict[str, Any]], class_name: str) -> Optional[Dict[str, Any]]:
        items = [d for d in self._by_class(secondary, [class_name], self.ppe_min) if det_center_in_person(d, person_box)]
        return max(items, key=lambda d: d["confidence"], default=None)

    @staticmethod
    def _fixed_zone_runtime(fixed_zones: Optional[List[Dict[str, Any]]], frame_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        runtime = []
        for zone in fixed_zones or []:
            if not zone.get("enabled", True):
                continue
            polygon = normalized_polygon_to_pixels(zone.get("points", []), frame_shape)
            if len(polygon) >= 3:
                runtime.append({**zone, "polygon_px": polygon})
        return runtime

    @staticmethod
    def _match_person_detection(track_box: Box, people: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not people:
            return {"confidence": 0.0, "sources": []}
        best = max(people, key=lambda p: box_iou(track_box, p["box"]))
        return best if box_iou(track_box, best["box"]) > 0.20 else {"confidence": 0.0, "sources": []}

    def analyze_frame(self, frame: np.ndarray, frame_index: int = 0, fixed_zones: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        primary = self._predict(self.primary, frame)
        secondary = self._predict(self.secondary, frame)
        people = self._person_detections(primary, secondary)
        tracks = self.tracker.update([p["box"] for p in people], frame_index)

        equipment = self._by_class(secondary, ["machinery"], self.machinery_min)
        equipment += self._by_class(secondary, ["vehicle"], self.vehicle_min)
        danger_zones = [
            {
                "equipment_class": d["class_name"],
                "confidence": d["confidence"],
                "equipment_box": d["xyxy"],
                "zone_box": expand_box(d["xyxy"], frame.shape[:2]),
            }
            for d in equipment
        ]
        restricted_zones = self._fixed_zone_runtime(fixed_zones, frame.shape[:2])

        workers, incidents = [], []
        for tr in tracks:
            person_det = self._match_person_detection(tr.box, people)
            helmet = self._positive_agreement(tr.box, primary, secondary, "Helmet", "Hardhat")
            vest = self._positive_agreement(tr.box, primary, secondary, "Vest", "Safety Vest")
            no_helmet = self._explicit_negative(tr.box, secondary, "NO-Hardhat")
            no_vest = self._explicit_negative(tr.box, secondary, "NO-Safety Vest")
            fp = foot_point(tr.box)

            relationships = []
            for z in danger_zones:
                equipment_center = center(z["equipment_box"])
                distance_px = math.hypot(fp[0] - equipment_center[0], fp[1] - equipment_center[1])
                inside = point_in_box(fp, z["zone_box"])
                if inside:
                    relationships.append({
                        "type": "CLOSE_TO",
                        "equipment_class": z["equipment_class"],
                        "equipment_confidence": z["confidence"],
                        "distance_px": distance_px,
                        "worker_point": fp,
                        "equipment_point": equipment_center,
                    })
            proximity_now = bool(relationships)
            restricted_hits = [
                {"id": z.get("id"), "name": z.get("name"), "zone_type": z.get("zone_type", "RESTRICTED")}
                for z in restricted_zones
                if str(z.get("zone_type", "RESTRICTED")).upper() == "RESTRICTED" and point_in_polygon(fp, z["polygon_px"])
            ]

            tr.helmet_history.append(bool(helmet["confirmed"]))
            tr.vest_history.append(bool(vest["confirmed"]))
            tr.proximity_history.append(proximity_now)
            helmet_temporal = tr.confirmed(tr.helmet_history, self.temporal_required)
            vest_temporal = tr.confirmed(tr.vest_history, self.temporal_required)
            proximity_temporal = tr.confirmed(tr.proximity_history, self.temporal_required)

            hazards: List[Dict[str, Any]] = []
            if no_helmet:
                hazards.append({"type": "NO_HARDHAT", "severity": "HIGH", "confidence": no_helmet["confidence"]})
            elif len(tr.helmet_history) >= self.temporal_required and not helmet_temporal:
                hazards.append({"type": "HELMET_NOT_CONFIRMED", "severity": "MEDIUM", "confidence": 1.0})
            if no_vest:
                hazards.append({"type": "NO_SAFETY_VEST", "severity": "HIGH", "confidence": no_vest["confidence"]})
            elif len(tr.vest_history) >= self.temporal_required and not vest_temporal:
                hazards.append({"type": "VEST_NOT_CONFIRMED", "severity": "MEDIUM", "confidence": 1.0})
            for zone in restricted_hits:
                hazards.append({"type": "RESTRICTED_ZONE_ENTRY", "severity": "HIGH", "confidence": 1.0, "zone_id": zone["id"], "zone_name": zone["name"]})
            if proximity_temporal:
                hazards.append({
                    "type": "DANGEROUS_MACHINE_PROXIMITY",
                    "severity": "CRITICAL",
                    "confidence": max([r["equipment_confidence"] for r in relationships], default=1.0),
                })

            rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
            severity = max((h["severity"] for h in hazards), key=lambda s: rank[s], default="LOW")
            worker = {
                "track_id": tr.track_id,
                "box": tr.box,
                "person_confidence": float(person_det.get("confidence", 0.0)),
                "person_sources": person_det.get("sources", []),
                "foot_point": fp,
                "helmet": {"frame": helmet, "temporal_confirmed": helmet_temporal, "history": list(tr.helmet_history)},
                "vest": {"frame": vest, "temporal_confirmed": vest_temporal, "history": list(tr.vest_history)},
                "relationships": relationships,
                "inside_dynamic_danger_zone": proximity_now,
                "proximity_temporal_confirmed": proximity_temporal,
                "restricted_zone_hits": restricted_hits,
                "hazards": hazards,
                "severity": severity,
                "alert": severity in {"HIGH", "CRITICAL"},
            }
            workers.append(worker)
            for hazard in hazards:
                if hazard["severity"] in {"HIGH", "CRITICAL"}:
                    incidents.append({"track_id": tr.track_id, **hazard, "frame_index": frame_index})

        serialized_restricted = []
        for z in restricted_zones:
            item = {k: v for k, v in z.items() if k != "polygon_px"}
            item["polygon_px"] = z["polygon_px"].tolist()
            serialized_restricted.append(item)
        return {
            "frame_index": frame_index,
            "workers": workers,
            "equipment": equipment,
            "danger_zones": danger_zones,
            "restricted_zones": serialized_restricted,
            "incidents": incidents,
            "raw": {"primary": primary, "secondary": secondary},
        }

    @staticmethod
    def _ppe_text(label: str, ppe: Dict[str, Any]) -> str:
        current = ppe["frame"]
        history_count = sum(bool(v) for v in ppe["history"])
        if ppe["temporal_confirmed"]:
            return f"{label} SAFE {history_count}/{len(ppe['history'])}"
        if current.get("confirmed"):
            return f"{label} AI {current['confidence']:.0%} (pending temporal)"
        primary_seen = current.get("primary_seen", 0.0)
        secondary_seen = current.get("secondary_seen", 0.0)
        if primary_seen or secondary_seen:
            return f"{label} NOT CONFIRMED P:{primary_seen:.0%} S:{secondary_seen:.0%}"
        return f"{label} NOT CONFIRMED"

    def draw_overlay(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        out = frame.copy()
        for z in result.get("restricted_zones", []):
            polygon = np.array(z["polygon_px"], dtype=np.int32)
            overlay = out.copy()
            cv2.fillPoly(overlay, [polygon], (45, 45, 210))
            out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)
            cv2.polylines(out, [polygon], True, (55, 55, 255), 3)
            x, y = map(int, polygon[0])
            cv2.putText(out, f"RESTRICTED: {z.get('name','Zone')}", (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (55, 55, 255), 2)

        for z in result["danger_zones"]:
            x1, y1, x2, y2 = map(int, z["zone_box"])
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            out = cv2.addWeighted(overlay, 0.14, out, 0.86, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            ex1, ey1, ex2, ey2 = map(int, z["equipment_box"])
            cv2.rectangle(out, (ex1, ey1), (ex2, ey2), (0, 165, 255), 3)
            cv2.putText(out, f"{z['equipment_class']} {z['confidence']:.0%}", (ex1, max(20, ey1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

        for w in result["workers"]:
            x1, y1, x2, y2 = map(int, w["box"])
            sev = w["severity"]
            color = {"LOW": (0, 180, 0), "MEDIUM": (0, 215, 255), "HIGH": (0, 80, 255), "CRITICAL": (0, 0, 255)}[sev]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            cv2.putText(out, f"Worker #{w['track_id']} | Person {w['person_confidence']:.0%} | {sev}", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

            line_y = min(out.shape[0] - 12, y2 + 22)
            cv2.putText(out, self._ppe_text("Helmet", w["helmet"]), (x1, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
            line_y = min(out.shape[0] - 12, line_y + 20)
            cv2.putText(out, self._ppe_text("Vest", w["vest"]), (x1, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

            for relation in w.get("relationships", []):
                wp = tuple(map(int, relation["worker_point"]))
                ep = tuple(map(int, relation["equipment_point"]))
                cv2.line(out, wp, ep, (0, 0, 255), 3)
                mx, my = (wp[0] + ep[0]) // 2, (wp[1] + ep[1]) // 2
                cv2.putText(
                    out,
                    f"CLOSE_TO {relation['equipment_class']} {relation['equipment_confidence']:.0%} | {relation['distance_px']:.0f}px",
                    (mx, max(22, my - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 255), 2,
                )

            if w.get("hazards"):
                hazards = ", ".join(h["type"] for h in w["hazards"])
                cv2.putText(out, hazards[:90], (x1, min(out.shape[0] - 34, line_y + 23)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 2)
            if w["alert"]:
                cv2.putText(out, "ALERT ACTIVE", (x1, min(out.shape[0] - 10, line_y + 46)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2)
        return out

    def process_video(self, input_path: str, output_path: str, sample_every_n_frames: int = 3, fixed_zones: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        self.reset_tracking()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        frame_index, analyzed_frames = 0, 0
        incidents: List[Dict[str, Any]] = []
        last_result: Optional[Dict[str, Any]] = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % max(1, sample_every_n_frames) == 0:
                    last_result = self.analyze_frame(frame, frame_index, fixed_zones=fixed_zones)
                    incidents.extend(last_result["incidents"])
                    analyzed_frames += 1
                if last_result:
                    frame = self.draw_overlay(frame, last_result)
                writer.write(frame)
                frame_index += 1
        finally:
            cap.release()
            writer.release()
        return {"input": input_path, "output": output_path, "frames": frame_index, "analyzed_frames": analyzed_frames, "incidents": incidents}
