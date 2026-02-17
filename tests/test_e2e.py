from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from teleopit.bus.topics import TOPIC_ACTION, TOPIC_MIMIC_OBS, TOPIC_ROBOT_STATE


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


requires_mujoco = pytest.mark.skipif(not _has_module("mujoco"), reason="mujoco not installed")
requires_onnxruntime = pytest.mark.skipif(not _has_module("onnxruntime"), reason="onnxruntime not installed")
requires_h5py = pytest.mark.skipif(not _has_module("h5py"), reason="h5py not installed")
requires_mink = pytest.mark.skipif(not _has_module("mink"), reason="mink not installed")


def _asset_paths(project_root: Path) -> tuple[Path, Path, Path]:
    policy = project_root.parent / "TWIST2" / "assets" / "ckpts" / "twist2_1017_20k.onnx"
    bvh = project_root / "teleopit" / "retargeting" / "gmr" / "assets" / "xsens_bvh_test" / "251021_04_boxing_120Hz_cm_3DsMax.bvh"
    xml = project_root / "teleopit" / "retargeting" / "gmr" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    return policy, bvh, xml


@requires_mujoco
@requires_onnxruntime
@requires_mink
@requires_h5py
def test_bvh_to_mujoco_pipeline_stands_and_records(project_root: Path, tmp_dir: Path) -> None:
    import h5py
    from omegaconf import OmegaConf

    from teleopit.pipeline import TeleopPipeline

    policy_path, bvh_path, xml_path = _asset_paths(project_root)
    if not policy_path.exists() or not bvh_path.exists() or not xml_path.exists():
        pytest.skip("required e2e assets are missing")

    robot_cfg = OmegaConf.load(project_root / "teleopit" / "configs" / "robot" / "g1.yaml")
    controller_cfg = OmegaConf.load(project_root / "teleopit" / "configs" / "controller" / "rl_policy.yaml")
    input_cfg = OmegaConf.load(project_root / "teleopit" / "configs" / "input" / "bvh.yaml")

    robot_cfg.xml_path = str(xml_path)
    controller_cfg.policy_path = str(policy_path)
    controller_cfg.default_dof_pos = list(robot_cfg.default_angles)
    controller_cfg.action_scale = 0.0
    controller_cfg.clip_range = [-10.0, 10.0]

    input_cfg.bvh_file = str(bvh_path)
    input_cfg.provider = "bvh"
    input_cfg.bvh_format = "lafan1"
    input_cfg.human_format = "bvh_xsens"
    input_cfg.robot_name = "unitree_g1"

    recording_path = tmp_dir / "e2e.h5"
    cfg = OmegaConf.create(
        {
            "robot": robot_cfg,
            "controller": controller_cfg,
            "input": input_cfg,
                "policy_hz": 50,
                "pd_hz": 50,
            "recording": {"output_path": str(recording_path)},
        }
    )

    pipeline = TeleopPipeline(cfg)

    action_count: list[int] = []
    mimic_count: list[int] = []
    state_count: list[int] = []

    pipeline.bus.subscribe(TOPIC_ACTION, lambda _: action_count.append(1))
    pipeline.bus.subscribe(TOPIC_MIMIC_OBS, lambda _: mimic_count.append(1))
    pipeline.bus.subscribe(TOPIC_ROBOT_STATE, lambda _: state_count.append(1))

    result = pipeline.run(num_steps=100, record=True)

    assert float(result["root_height"]) > 0.3
    assert int(result["steps"]) == 100

    assert len(action_count) == 100
    assert len(mimic_count) == 100
    assert len(state_count) == 100

    latest_action = pipeline.bus.get_latest(TOPIC_ACTION)
    latest_mimic = pipeline.bus.get_latest(TOPIC_MIMIC_OBS)
    latest_state = pipeline.bus.get_latest(TOPIC_ROBOT_STATE)

    assert isinstance(latest_action, np.ndarray)
    assert latest_action.shape == (29,)
    assert isinstance(latest_mimic, np.ndarray)
    assert latest_mimic.shape == (35,)
    assert latest_state is not None

    record_path = Path(str(result["record_path"]))
    assert record_path.exists()

    with h5py.File(record_path, "r") as f:
        total_frames = int(np.asarray(f.attrs["total_frames"]).item())
        assert total_frames == 100

        joint_pos = cast(h5py.Dataset, f["joint_pos"])
        joint_vel = cast(h5py.Dataset, f["joint_vel"])
        mimic_obs = cast(h5py.Dataset, f["mimic_obs"])
        action = cast(h5py.Dataset, f["action"])

        assert joint_pos.shape[0] == 100
        assert joint_vel.shape[0] == 100
        assert mimic_obs.shape == (100, 35)
        assert action.shape == (100, 29)
