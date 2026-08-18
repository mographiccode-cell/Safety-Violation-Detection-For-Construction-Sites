from pathlib import Path

from ensemble_engine import (
    IoUTracker,
    box_iou,
    expand_box,
    foot_point,
    normalized_polygon_to_pixels,
    point_in_box,
    point_in_polygon,
)
from zone_store import ZoneStore


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


def test_normalized_restricted_polygon_hit():
    polygon = normalized_polygon_to_pixels(
        [[0.50, 0.20], [0.95, 0.20], [0.95, 0.95], [0.50, 0.95]],
        (720, 1280),
    )
    assert point_in_polygon((900.0, 600.0), polygon) is True
    assert point_in_polygon((100.0, 600.0), polygon) is False


def test_zone_store_persists_normalized_polygon(tmp_path):
    store = ZoneStore(str(tmp_path / "zones.db"))
    zone = store.upsert(
        "cam-1",
        "Forklift Lane",
        [[0.4, 0.2], [0.9, 0.2], [0.9, 0.9], [0.4, 0.9]],
    )
    assert zone["enabled"] is True
    assert zone["zone_type"] == "RESTRICTED"
    assert len(store.list("cam-1", enabled_only=True)) == 1
    assert store.delete(zone["id"]) is True
    assert store.list("cam-1") == []


def test_zone_store_rejects_pixel_coordinates(tmp_path):
    store = ZoneStore(str(tmp_path / "zones.db"))
    try:
        store.upsert("cam-1", "Bad", [[10, 20], [30, 20], [30, 40]])
    except ValueError as exc:
        assert "normalized" in str(exc)
    else:
        raise AssertionError("Expected normalized coordinate validation to fail")
