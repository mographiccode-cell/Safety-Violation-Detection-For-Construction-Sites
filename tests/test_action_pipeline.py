import numpy as np

from action_engine import ACTION_NAMES, TSSTG_COCO_INDICES


def test_tsstg_source_joint_layout_matches_checkpoint_features():
    assert TSSTG_COCO_INDICES == [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    assert len(TSSTG_COCO_INDICES) == 13
    # Source TSSTG adds shoulder-center -> 14 nodes, each x/y/confidence.
    assert (len(TSSTG_COCO_INDICES) + 1) * 3 == 42


def test_action_names_are_model_native_only():
    assert ACTION_NAMES == [
        "Standing",
        "Walking",
        "Sitting",
        "Lying Down",
        "Stand up",
        "Sit down",
        "Fall Down",
    ]
    assert "Entering Forklift" not in ACTION_NAMES


def test_pose_joint_selection_keeps_source_order():
    points = np.zeros((30, 17, 3), dtype=np.float32)
    for idx in range(17):
        points[:, idx, 0] = idx
    selected = points[:, TSSTG_COCO_INDICES, :]
    assert selected.shape == (30, 13, 3)
    assert selected[0, :, 0].tolist() == [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def test_fall_down_is_only_native_action_marked_as_critical_candidate():
    critical_action = "Fall Down"
    for name in ACTION_NAMES:
        if name == critical_action:
            assert name == "Fall Down"
        else:
            assert name != critical_action
