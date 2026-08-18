from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = BASE_DIR / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from stgcn.Models import TwoStreamSpatialTemporalGraph  # noqa: E402
from stgcn.pose_utils import normalize_points_with_size, scale_pose  # noqa: E402


ACTION_NAMES = [
    "Standing",
    "Walking",
    "Sitting",
    "Lying Down",
    "Stand up",
    "Sit down",
    "Fall Down",
]


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


class TSSTGSequenceModel:
    """Real two-stream ST-GCN inference using the Git-pulled TSSTG weight."""

    def __init__(self, weight_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_path = Path(weight_path)
        self.model = TwoStreamSpatialTemporalGraph({"strategy": "spatial"}, len(ACTION_NAMES)).to(self.device)
        try:
            checkpoint = torch.load(str(self.weight_path), map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(str(self.weight_path), map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and checkpoint and all(str(k).startswith("module.") for k in checkpoint):
            checkpoint = {str(k)[7:]: v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, points: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        """points shape: (T, 17, 3), channels x/y/keypoint confidence."""
        if points.ndim != 3 or points.shape[1:] != (17, 3):
            raise ValueError(f"Expected pose sequence (T,17,3), got {points.shape}")
        if len(points) < 2:
            raise ValueError("TSSTG requires at least 2 temporal pose frames")

        pts = points.astype(np.float32, copy=True)
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        # TSSTG graph expects the original 17 COCO nodes plus the shoulder-center node.
        shoulder_center = np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2.0, 1)
        pts = np.concatenate((pts, shoulder_center), axis=1)

        pts_t = torch.tensor(pts, dtype=torch.float32, device=self.device).permute(2, 0, 1)[None, :]
        mot = pts_t[:, :2, 1:, :] - pts_t[:, :2, :-1, :]
        probabilities = self.model((pts_t, mot))[0].detach().cpu().numpy().astype(float)
        return probabilities


class ActionRecognitionEngine:
    """YOLOv8 Pose -> 17 COCO keypoints -> real TSSTG action recognition."""

    def __init__(
        self,
        pose_model_path: str = "models/action_candidates/yolov8n-pose.pt",
        action_weight_path: str = "models/action_candidates/bigtuo__tsstg-model.pth",
        sequence_length: int = 30,
        pose_confidence: float = 0.20,
    ):
        self.pose_model_path = self._resolve(pose_model_path)
        self.action_weight_path = self._resolve(action_weight_path)
        self.pose = YOLO(str(self.pose_model_path))
        self.action = TSSTGSequenceModel(str(self.action_weight_path))
        self.sequence_length = int(sequence_length)
        self.pose_confidence = float(pose_confidence)

    @staticmethod
    def _resolve(path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else BASE_DIR / p

    def status(self) -> Dict[str, Any]:
        return {
            "pose_model": str(self.pose_model_path),
            "action_model": str(self.action_weight_path),
            "action_classes": ACTION_NAMES,
            "sequence_length": self.sequence_length,
            "device": self.action.device,
        }

    def extract_pose(self, frame: np.ndarray, target_box: Optional[Sequence[float]] = None) -> Optional[Dict[str, Any]]:
        result = self.pose.predict(frame, imgsz=640, conf=self.pose_confidence, verbose=False)[0]
        if result.boxes is None or result.keypoints is None or result.keypoints.data is None:
            return None
        if len(result.boxes) == 0 or int(result.keypoints.data.shape[0]) == 0:
            return None

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        box_conf = result.boxes.conf.detach().cpu().numpy()
        keypoints = result.keypoints.data.detach().cpu().numpy()
        count = min(len(boxes), len(keypoints))
        if count == 0:
            return None

        if target_box is not None:
            scores = [_box_iou(boxes[i], target_box) for i in range(count)]
            idx = int(np.argmax(scores))
            if scores[idx] <= 0.0:
                idx = int(np.argmax(box_conf[:count]))
        else:
            idx = int(np.argmax(box_conf[:count]))

        pts = keypoints[idx]
        if pts.shape != (17, 3):
            return None
        return {
            "box": boxes[idx].astype(float).tolist(),
            "box_confidence": float(box_conf[idx]),
            "keypoints": pts.astype(np.float32),
            "visible_keypoints": int(np.sum(pts[:, 2] >= 0.20)),
        }

    @staticmethod
    def _fill_missing(sequence: List[Optional[np.ndarray]]) -> Tuple[np.ndarray, int]:
        valid = [i for i, p in enumerate(sequence) if p is not None]
        if not valid:
            raise ValueError("No pose was detected in this action sequence")
        filled: List[np.ndarray] = [None] * len(sequence)  # type: ignore[list-item]
        first = valid[0]
        for i in range(0, first + 1):
            filled[i] = sequence[first].copy()  # type: ignore[union-attr]
            if i != first:
                filled[i][:, 2] *= 0.25
        last_valid = first
        for i in range(first + 1, len(sequence)):
            if sequence[i] is not None:
                filled[i] = sequence[i].copy()  # type: ignore[union-attr]
                last_valid = i
            else:
                filled[i] = filled[last_valid].copy()
                filled[i][:, 2] *= 0.25
        return np.stack(filled).astype(np.float32), len(valid)

    def classify_pose_sequence(self, pose_sequence: List[Optional[np.ndarray]], image_size: Tuple[int, int]) -> Dict[str, Any]:
        points, detected_frames = self._fill_missing(pose_sequence)
        probabilities = self.action.predict(points, image_size)
        order = np.argsort(probabilities)[::-1]
        top = int(order[0])
        return {
            "action": ACTION_NAMES[top],
            "confidence": float(probabilities[top]),
            "probabilities": {ACTION_NAMES[i]: float(probabilities[i]) for i in range(len(ACTION_NAMES))},
            "detected_pose_frames": detected_frames,
            "sequence_frames": len(pose_sequence),
            "pose_coverage": detected_frames / max(1, len(pose_sequence)),
        }

    def analyze_video(self, video_path: str, stride: int = 5) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open action video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        poses: List[Optional[np.ndarray]] = []
        pose_meta: List[Dict[str, Any]] = []
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                det = self.extract_pose(frame)
                poses.append(None if det is None else det["keypoints"])
                pose_meta.append({
                    "frame_index": frame_index,
                    "time_s": frame_index / fps,
                    "pose_detected": det is not None,
                    "person_confidence": 0.0 if det is None else det["box_confidence"],
                    "visible_keypoints": 0 if det is None else det["visible_keypoints"],
                })
                frame_index += 1
        finally:
            cap.release()

        if len(poses) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} frames, video has {len(poses)}")

        windows = []
        for start in range(0, len(poses) - self.sequence_length + 1, max(1, int(stride))):
            end = start + self.sequence_length
            result = self.classify_pose_sequence(poses[start:end], (width, height))
            result.update({
                "start_frame": start,
                "end_frame": end - 1,
                "start_time_s": start / fps,
                "end_time_s": (end - 1) / fps,
            })
            windows.append(result)

        mean_probs = {
            name: float(np.mean([w["probabilities"][name] for w in windows]))
            for name in ACTION_NAMES
        }
        dominant = max(mean_probs, key=mean_probs.get)
        return {
            "status": "ok",
            "video": video_path,
            "fps": fps,
            "frame_count": len(poses),
            "image_size": [width, height],
            "pose_frames_detected": sum(1 for p in poses if p is not None),
            "pose_coverage": sum(1 for p in poses if p is not None) / max(1, len(poses)),
            "dominant_action": dominant,
            "dominant_confidence": mean_probs[dominant],
            "mean_probabilities": mean_probs,
            "windows": windows,
            "pose_meta": pose_meta,
            "model_status": self.status(),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--output", default="action_result.json")
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    engine = ActionRecognitionEngine()
    result = engine.analyze_video(args.video, stride=args.stride)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "dominant_action": result["dominant_action"],
        "dominant_confidence": result["dominant_confidence"],
        "pose_coverage": result["pose_coverage"],
        "windows": len(result["windows"]),
    }, indent=2))
