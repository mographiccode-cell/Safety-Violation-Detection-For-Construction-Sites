from collections import defaultdict, deque

import numpy as np

from action_engine import ActionRecognitionEngine, _box_iou


class TrackActionManager:
    def __init__(self, pose_model_path, action_weight_path, sequence_length=30, inference_step=5, min_pose_coverage=0.50):
        self.engine = ActionRecognitionEngine(
            pose_model_path=pose_model_path,
            action_weight_path=action_weight_path,
            sequence_length=sequence_length,
        )
        self.sequence_length = int(sequence_length)
        self.inference_step = max(1, int(inference_step))
        self.min_pose_coverage = float(min_pose_coverage)
        self.buffers = defaultdict(lambda: deque(maxlen=self.sequence_length))
        self.sample_counts = defaultdict(int)
        self.latest = {}

    def reset(self):
        self.buffers.clear()
        self.sample_counts.clear()
        self.latest.clear()

    def status(self):
        data = self.engine.status()
        data.update({
            "per_track_buffer": self.sequence_length,
            "inference_step": self.inference_step,
            "min_pose_coverage": self.min_pose_coverage,
        })
        return data

    def _pose_candidates(self, frame):
        result = self.engine.pose.predict(
            frame, imgsz=640, conf=self.engine.pose_confidence, verbose=False
        )[0]
        if result.boxes is None or result.keypoints is None or result.keypoints.data is None:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        keypoints = result.keypoints.data.detach().cpu().numpy()
        count = min(len(boxes), len(confs), len(keypoints))
        items = []
        for i in range(count):
            pts = keypoints[i]
            if pts.shape != (17, 3):
                continue
            items.append({
                "box": boxes[i].astype(float).tolist(),
                "confidence": float(confs[i]),
                "keypoints": pts.astype(np.float32),
                "visible_keypoints": int(np.sum(pts[:, 2] >= 0.20)),
            })
        return items

    @staticmethod
    def _match_pose(worker_box, candidates):
        if not candidates:
            return None
        best = max(candidates, key=lambda p: _box_iou(worker_box, p["box"]))
        return best if _box_iou(worker_box, best["box"]) >= 0.10 else None

    def update(self, frame, workers, frame_index):
        candidates = self._pose_candidates(frame) if workers else []
        output = {}
        for worker in workers:
            track_id = int(worker["track_id"])
            pose = self._match_pose(worker["box"], candidates)
            buffer = self.buffers[track_id]
            buffer.append(None if pose is None else pose["keypoints"])
            self.sample_counts[track_id] += 1
            detected = sum(1 for p in buffer if p is not None)
            state = {
                "status": "collecting" if len(buffer) < self.sequence_length else "ready_for_sequence",
                "source": "YOLO_POSE_TSSTG_SEQUENCE",
                "collected_frames": len(buffer),
                "required_frames": self.sequence_length,
                "pose_frames_detected": detected,
                "pose_coverage": detected / max(1, len(buffer)),
                "pose_person_confidence": 0.0 if pose is None else pose["confidence"],
                "visible_keypoints": 0 if pose is None else pose["visible_keypoints"],
                "frame_index": frame_index,
            }
            if len(buffer) == self.sequence_length and self.sample_counts[track_id] % self.inference_step == 0:
                result = self.engine.classify_pose_sequence(
                    list(buffer), (frame.shape[1], frame.shape[0])
                )
                result.update({
                    "status": "classified" if result["pose_coverage"] >= self.min_pose_coverage else "low_pose_coverage",
                    "source": "TSSTG_SEQUENCE",
                    "frame_index": frame_index,
                })
                self.latest[track_id] = result
            output[track_id] = dict(self.latest.get(track_id, state))
        return output
