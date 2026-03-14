from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from train_mimic.data.dataset_builder import (
    DatasetClipRow,
    DatasetSourceSpec,
    DatasetSpec,
    assign_splits,
    build_dataset_from_spec,
    convert_source_to_npz_clips,
    load_dataset_spec,
)
from train_mimic.data.dataset_lib import inspect_npz
from train_mimic.data.motion_fk import (
    MotionFkExtractor,
    normalize_quaternion,
    quat_rotate_inverse,
    quat_wxyz_to_xyzw,
)
from train_mimic.scripts.convert_pkl_to_npz import _MJLAB_G1_BODY_NAMES, convert_pkl_to_npz


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


def _write_pkl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(_synthetic_motion_payload(), handle)


def _write_npz_from_pkl(path: Path) -> None:
    pkl_path = path.with_suffix(".pkl")
    _write_pkl(pkl_path)
    convert_pkl_to_npz(str(pkl_path), str(path))


def test_load_dataset_spec_parses_typed_sources(tmp_path: Path) -> None:
    spec_path = tmp_path / "demo.yaml"
    spec_path.write_text(
        f"""name: demo
target_fps: 30
val_percent: 5
hash_salt: ""
sources:
  - name: clips
    type: npz
    input: {tmp_path / 'npz_source'}
    weight: 2.5
  - name: lafan1
    type: bvh
    input: {tmp_path / 'lafan1'}
    bvh_format: lafan1
    robot_name: unitree_g1
""",
        encoding="utf-8",
    )

    spec = load_dataset_spec(spec_path)
    assert spec.name == "demo"
    assert spec.target_fps == 30
    assert spec.sources[0].type == "npz"
    assert spec.sources[0].weight == 2.5
    assert spec.sources[1].type == "bvh"
    assert spec.sources[1].bvh_format == "lafan1"


def test_load_dataset_spec_rejects_bvh_without_format(tmp_path: Path) -> None:
    spec_path = tmp_path / "bad.yaml"
    spec_path.write_text(
        """name: demo
target_fps: 30
val_percent: 5
hash_salt: ""
sources:
  - name: broken
    type: bvh
    input: data/lafan1_bvh
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires bvh_format"):
        load_dataset_spec(spec_path)


def test_assign_splits_guarantees_non_empty_train_and_val() -> None:
    rows = [
        DatasetClipRow("clip_a", "src", "a.npz", 10, 30, "", "/tmp/a.npz"),
        DatasetClipRow("clip_b", "src", "b.npz", 10, 30, "", "/tmp/b.npz"),
    ]
    resolved = assign_splits(rows, 1, "")
    splits = {row.clip_id: row.resolved_split for row in resolved}
    assert set(splits.values()) == {"train", "val"}


def test_assign_splits_rejects_single_clip_dataset() -> None:
    rows = [DatasetClipRow("clip_a", "src", "a.npz", 10, 30, "", "/tmp/a.npz")]
    with pytest.raises(ValueError, match="at least 2 clips"):
        assign_splits(rows, 5, "")


def test_convert_source_to_npz_clips_handles_pkl_source(tmp_path: Path) -> None:
    input_dir = tmp_path / "pkl_source"
    _write_pkl(input_dir / "clip.pkl")

    source = DatasetSourceSpec(name="pkl_src", type="pkl", input=str(input_dir))
    output_dir = tmp_path / "dataset" / "clips" / "pkl_src"
    report = convert_source_to_npz_clips(source, output_dir, jobs=1)

    assert report["clips"] == 1
    clip_path = output_dir / "clip.npz"
    assert clip_path.is_file()
    assert inspect_npz(clip_path).fps == 30


def test_convert_source_to_npz_clips_handles_bvh_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir = tmp_path / "bvh_source"
    input_dir.mkdir()
    (input_dir / "clip.bvh").write_text("HIERARCHY\n", encoding="utf-8")
    fake_xml = tmp_path / "fake.xml"
    fake_xml.write_text("<xml />", encoding="utf-8")
    payload = _synthetic_motion_payload()

    def fake_convert_bvh_to_retarget_pkl(
        bvh_path: Path,
        output_pkl: Path,
        bvh_format: str,
        robot_name: str,
        max_frames: int,
        model: object,
        ) -> None:
            assert bvh_path.name == "clip.bvh"
            assert bvh_format == "lafan1"
            output_pkl.parent.mkdir(parents=True, exist_ok=True)
            with output_pkl.open("wb") as handle:
                pickle.dump(payload, handle)

    monkeypatch.setattr(
        "train_mimic.data.dataset_builder.mocap_xml_path",
        lambda _, robot_name="unitree_g1": fake_xml if robot_name == "unitree_g1" else None,
    )
    monkeypatch.setattr("train_mimic.data.dataset_builder.convert_bvh_to_retarget_pkl", fake_convert_bvh_to_retarget_pkl)
    monkeypatch.setattr(
        "train_mimic.data.dataset_builder.mujoco.MjModel.from_xml_path",
        lambda _: object(),
    )

    source = DatasetSourceSpec(
        name="bvh_src",
        type="bvh",
        input=str(input_dir),
        bvh_format="lafan1",
        robot_name="unitree_g1",
    )
    output_dir = tmp_path / "dataset" / "clips" / "bvh_src"
    report = convert_source_to_npz_clips(source, output_dir, jobs=1)

    assert report["clips"] == 1
    assert (output_dir / "clip.npz").is_file()


def test_convert_source_to_npz_clips_rejects_non_g1_bvh_robot(tmp_path: Path) -> None:
    input_dir = tmp_path / "bvh_source"
    input_dir.mkdir()
    (input_dir / "clip.bvh").write_text("HIERARCHY\n", encoding="utf-8")

    source = DatasetSourceSpec(
        name="bvh_src",
        type="bvh",
        input=str(input_dir),
        bvh_format="lafan1",
        robot_name="fourier_n1",
    )

    with pytest.raises(ValueError, match="currently supports only 'unitree_g1'"):
        convert_source_to_npz_clips(source, tmp_path / "dataset" / "clips" / "bvh_src", jobs=1)


def test_convert_source_to_npz_clips_rejects_dataset_root_for_npz_source(tmp_path: Path) -> None:
    dataset_root = tmp_path / "old_dataset"
    clips_dir = dataset_root / "clips" / "src"
    clips_dir.mkdir(parents=True)
    _write_npz_from_pkl(clips_dir / "clip_a.npz")
    _write_npz_from_pkl(dataset_root / "train.npz")
    (dataset_root / "build_info.json").write_text("{}", encoding="utf-8")

    source = DatasetSourceSpec(name="npz_src", type="npz", input=str(dataset_root))
    with pytest.raises(ValueError, match="points at dataset root"):
        convert_source_to_npz_clips(source, tmp_path / "dataset" / "clips" / "npz_src", jobs=1)


def test_build_dataset_from_spec_writes_single_directory_outputs(tmp_path: Path) -> None:
    npz_input = tmp_path / "npz_source"
    _write_npz_from_pkl(npz_input / "clip_a.npz")
    _write_npz_from_pkl(npz_input / "clip_b.npz")

    spec = DatasetSpec(
        name="demo_dataset",
        target_fps=30,
        val_percent=5,
        hash_salt="",
        sources=[DatasetSourceSpec(name="npz_src", type="npz", input=str(npz_input))],
    )

    output_root = tmp_path / "datasets"
    report = build_dataset_from_spec(spec, jobs=2, skip_fk_check=True, output_root=output_root)

    dataset_dir = output_root / "demo_dataset"
    assert report["dataset_dir"] == str(dataset_dir)
    assert report["build_dir"] == str(dataset_dir)
    assert (dataset_dir / "clips" / "npz_src" / "clip_a.npz").is_file()
    assert (dataset_dir / "clips" / "npz_src" / "clip_b.npz").is_file()
    assert (dataset_dir / "train.npz").is_file()
    assert (dataset_dir / "val.npz").is_file()
    assert (dataset_dir / "manifest_resolved.csv").is_file()
    assert (dataset_dir / "build_info.json").is_file()
    assert report["clip_counts"]["total"] == 2
