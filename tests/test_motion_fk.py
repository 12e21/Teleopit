from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from train_mimic.data.motion_fk import (
    MotionFkExtractor,
    compute_npz_fk_consistency,
    normalize_quaternion,
    quat_rotate_inverse,
    quat_wxyz_to_xyzw,
)
from train_mimic.scripts.convert_pkl_to_npz import (
    _MJLAB_G1_BODY_NAMES,
    convert_pkl_to_npz,
)


def _synthetic_motion_payload() -> dict[str, object]:
    extractor = MotionFkExtractor()
    body_names = list(_MJLAB_G1_BODY_NAMES)

    root_pos = np.asarray(
        [
            [0.0, 0.0, 0.76],
            [0.03, -0.01, 0.77],
            [0.07, -0.02, 0.775],
            [0.10, -0.015, 0.78],
        ],
        dtype=np.float32,
    )
    root_quat_wxyz = normalize_quaternion(
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9950042, 0.0, 0.0, 0.0998334],
                [0.9800666, 0.0, 0.0, 0.1986693],
                [0.9553365, 0.0, 0.0, 0.2955202],
            ],
            dtype=np.float32,
        )
    )
    dof_pos = np.zeros((4, extractor.num_actions), dtype=np.float32)
    dof_pos[:, 3] = np.asarray([0.05, 0.15, 0.30, 0.20], dtype=np.float32)
    dof_pos[:, 9] = np.asarray([-0.05, -0.10, -0.20, -0.12], dtype=np.float32)
    dof_pos[:, 17] = np.asarray([0.10, 0.30, 0.45, 0.25], dtype=np.float32)
    dof_pos[:, 23] = np.asarray([-0.08, -0.20, -0.35, -0.18], dtype=np.float32)

    body_pos_w, _ = extractor.extract(root_pos, root_quat_wxyz, dof_pos, body_names)
    local_body_pos = quat_rotate_inverse(
        root_quat_wxyz[:, None, :],
        body_pos_w - root_pos[:, None, :],
    ).astype(np.float32)

    return {
        "fps": 30,
        "root_pos": root_pos,
        "root_rot": quat_wxyz_to_xyzw(root_quat_wxyz).astype(np.float32),
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos,
        "link_body_list": body_names,
    }


def test_convert_pkl_to_npz_uses_fk_consistent_body_targets(tmp_path: Path) -> None:
    payload = _synthetic_motion_payload()
    pkl_path = tmp_path / "clip.pkl"
    npz_path = tmp_path / "clip.npz"
    with pkl_path.open("wb") as f:
        pickle.dump(payload, f)

    convert_pkl_to_npz(str(pkl_path), str(npz_path))

    stats = compute_npz_fk_consistency(npz_path, sample_count=0)
    assert stats.pos_max < 1e-5
    assert stats.quat_max < 1e-5

    data = np.load(npz_path, allow_pickle=True)
    body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
    body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
    quat_spread = np.max(np.linalg.norm(body_quat_w - body_quat_w[:, :1, :], axis=-1))
    ang_vel_spread = np.max(np.linalg.norm(body_ang_vel_w - body_ang_vel_w[:, :1, :], axis=-1))
    assert quat_spread > 1e-3
    assert ang_vel_spread > 1e-3


def test_compute_npz_fk_consistency_detects_corrupted_body_orientation(tmp_path: Path) -> None:
    payload = _synthetic_motion_payload()
    pkl_path = tmp_path / "clip.pkl"
    npz_path = tmp_path / "clip.npz"
    with pkl_path.open("wb") as f:
        pickle.dump(payload, f)
    convert_pkl_to_npz(str(pkl_path), str(npz_path))

    data = dict(np.load(npz_path, allow_pickle=True))
    data["body_quat_w"] = np.tile(data["body_quat_w"][:, :1, :], (1, len(_MJLAB_G1_BODY_NAMES), 1))
    bad_npz_path = tmp_path / "clip_bad.npz"
    np.savez(bad_npz_path, **data)

    stats = compute_npz_fk_consistency(bad_npz_path, sample_count=0)
    assert stats.quat_mean > 0.05

