from __future__ import annotations

from pathlib import Path

import pytest

from train_mimic.data.dataset_v2 import (
    DatasetClipRow,
    assign_splits,
    build_source_convert_command,
    load_dataset_spec,
)


def _write_spec(path: Path) -> None:
    path.write_text(
        """name: demo

target_fps: 30
val_percent: 5
hash_salt: ""
sources:
  - name: a
    input: data/a
""",
        encoding="utf-8",
    )


def test_load_dataset_spec_parses_yaml(tmp_path: Path) -> None:
    spec_path = tmp_path / "demo.yaml"
    _write_spec(spec_path)
    spec = load_dataset_spec(spec_path)
    assert spec.name == "demo"
    assert spec.target_fps == 30
    assert spec.val_percent == 5
    assert len(spec.sources) == 1
    assert spec.sources[0].name == "a"


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


def test_build_source_convert_command_points_to_batch_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "demo.yaml"
    _write_spec(spec_path)
    spec = load_dataset_spec(spec_path)
    monkeypatch.setattr(
        "train_mimic.data.dataset_v2.resolve_source_input_dir",
        lambda source: Path("/tmp/pkl_source"),
    )
    command = build_source_convert_command(spec.sources[0], Path("/tmp/out"))
    assert command[0]
    assert command[1].endswith("train_mimic/scripts/convert_pkl_to_npz.py")
    assert command[-4:] == ["--input", "/tmp/pkl_source", "--output", "/tmp/out"]
