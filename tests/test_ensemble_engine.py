from collections import deque

from ensemble_engine import IoUTracker, box_iou, expand_box, foot_point, point_in_box


def test_box_iou_overlap():
    assert 0.1 < box_iou((0, 0, 100, 100), (50, 50, 150, 150)) < 0.2


def test_tracker_keeps_same_worker_id():
    tracker = IoUTracker(iou_threshold=0.2, max_age=5, history_len=5)
    first = tracker.update([(10, 10, 110, 210)], 0)[0]
    second = tracker.update([(15, 12, 115, 212)], 1)[0]
    assert first.track_id == second.track_id


def test_tracker_creates_new_id_for_far_worker():
    tracker = IoUTracker(iou_threshold=0.2, max_age=5, history_len=5)
    first = tracker.update([(10, 10, 110, 210)], 0)[0]
    second = tracker.update([(300, 10, 400, 210)], 1)[0]
    assert first.track_id != second.track_id


def test_three_of_five_temporal_confirmation():
    tracker = IoUTracker(history_len=5)
    tr = tracker.update([(10, 10, 110, 210)], 0)[0]
    tr.helmet_history.extend([True, False, True, True, False])
    assert tr.confirmed(tr.helmet_history, 3) is True
    tr.vest_history.extend([True, False, False, True, False])
    assert tr.confirmed(tr.vest_history, 3) is False


def test_dynamic_danger_zone_contains_near_worker_footpoint():
    frame_shape = (720, 1280)
    equipment = (400, 200, 800, 650)
    zone = expand_box(equipment, frame_shape)
    worker = (300, 250, 430, 620)
    assert point_in_box(foot_point(worker), zone) is True


def test_dynamic_danger_zone_excludes_far_worker():
    frame_shape = (720, 1280)
    equipment = (500, 200, 800, 650)
    zone = expand_box(equipment, frame_shape)
    worker = (10, 250, 100, 620)
    assert point_in_box(foot_point(worker), zone) is False
