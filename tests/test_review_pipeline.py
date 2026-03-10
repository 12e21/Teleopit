from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from scripts.data import build_dataset_from_review
from scripts.data import export_reviewed_manifest
from scripts.data import init_review_manifest
from train_mimic.data.review_lib import ReviewRow, load_review_state, save_review_state


BODY_NAMES = np.array(["pelvis", "torso"], dtype=str)


def _write_manifest(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "clip_id",
                "source",
                "file_rel",
                "num_frames",
                "fps",
                "resolved_split",
                "resolved_npz_path",
                "weight",
            ]
        )
        writer.writerow(
            [
                "src:clip_train",
                "src",
                "cache/clip_train.npz",
                4,
                24,
                "train",
                "/tmp/placeholder_train.npz",
                2.5,
            ]
        )
        writer.writerow(
            [
                "src:clip_val",
                "src",
                "cache/clip_val.npz",
                5,
                30,
                "val",
                "/tmp/placeholder_val.npz",
                0.75,
            ]
        )


def _write_npz(path: Path, *, num_frames: int, fps: int) -> None:
    joint_pos = np.linspace(0.0, 0.2, num_frames * 29, dtype=np.float32).reshape(num_frames, 29)
    joint_vel = np.gradient(joint_pos, axis=0).astype(np.float32)
    body_pos_w = np.zeros((num_frames, len(BODY_NAMES), 3), dtype=np.float32)
    body_pos_w[:, 0, 2] = np.linspace(0.75, 0.8, num_frames, dtype=np.float32)
    body_pos_w[:, 1, 2] = body_pos_w[:, 0, 2] + 0.3
    body_quat_w = np.zeros((num_frames, len(BODY_NAMES), 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = np.zeros((num_frames, len(BODY_NAMES), 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, len(BODY_NAMES), 3), dtype=np.float32)
    np.savez(
        path,
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        body_names=BODY_NAMES,
    )


def test_init_review_manifest_preserves_weight(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest_resolved.csv"
    review_path = tmp_path / "review_state.csv"
    _write_manifest(manifest_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "init_review_manifest.py",
            "--dataset",
            "demo",
            "--manifest",
            str(manifest_path),
            "--output",
            str(review_path),
        ],
    )

    init_review_manifest.main()

    rows = load_review_state(review_path)
    assert [row.weight for row in rows] == [2.5, 0.75]
    assert [row.decision for row in rows] == ["", ""]


def test_export_reviewed_manifest_preserves_weight_and_filters_keep(
    tmp_path: Path,
    monkeypatch,
) -> None:
    review_path = tmp_path / "review_state.csv"
    output_path = tmp_path / "filtered_manifest.csv"
    summary_path = tmp_path / "review_summary.json"
    save_review_state(
        [
            ReviewRow(
                clip_id="src:clip_train",
                source="src",
                file_rel="cache/clip_train.npz",
                resolved_split="train",
                num_frames=4,
                fps=24,
                duration_s=4 / 24,
                weight=2.5,
                decision="keep",
            ),
            ReviewRow(
                clip_id="src:clip_val",
                source="src",
                file_rel="cache/clip_val.npz",
                resolved_split="val",
                num_frames=5,
                fps=30,
                duration_s=5 / 30,
                weight=0.75,
                decision="drop",
            ),
        ],
        review_path,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_reviewed_manifest.py",
            "--review",
            str(review_path),
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
        ],
    )

    export_reviewed_manifest.main()

    with output_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["clip_id"] == "src:clip_train"
    assert float(rows[0]["weight"]) == 2.5


def test_build_dataset_from_review_resamples_mixed_fps_and_preserves_weights(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    train_npz = cache_dir / "clip_train.npz"
    val_npz = cache_dir / "clip_val.npz"
    _write_npz(train_npz, num_frames=4, fps=24)
    _write_npz(val_npz, num_frames=5, fps=30)

    filtered_manifest = tmp_path / "filtered_manifest.csv"
    with filtered_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "clip_id",
                "source",
                "file_rel",
                "num_frames",
                "fps",
                "resolved_split",
                "resolved_npz_path",
                "weight",
            ]
        )
        writer.writerow(
            [
                "src:clip_train",
                "src",
                str(train_npz),
                4,
                24,
                "train",
                str(train_npz),
                2.5,
            ]
        )
        writer.writerow(
            [
                "src:clip_val",
                "src",
                str(val_npz),
                5,
                30,
                "val",
                str(val_npz),
                0.75,
            ]
        )

    output_dir = tmp_path / "twist2_full_cleaned"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dataset_from_review.py",
            "--filtered_manifest",
            str(filtered_manifest),
            "--output_dir",
            str(output_dir),
        ],
    )

    build_dataset_from_review.main()

    train = np.load(output_dir / "train.npz", allow_pickle=True)
    val = np.load(output_dir / "val.npz", allow_pickle=True)
    assert int(train["fps"]) == 30
    assert int(val["fps"]) == 30
    assert train["clip_weights"].tolist() == [2.5]
    assert val["clip_weights"].tolist() == [0.75]
