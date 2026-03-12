from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from teleopit.pipeline import TeleopPipeline

from conftest import requires_mujoco


def test_pipeline_inherits_robot_action_decode_config(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyRobot:
        def __init__(self, cfg: object) -> None:
            captured["robot_cfg"] = cfg

    class DummyController:
        def __init__(self, cfg: object) -> None:
            captured["controller_cfg"] = cfg

    class DummyObsBuilder:
        def __init__(self, cfg: object) -> None:
            captured["obs_cfg"] = cfg

    class DummyInputProvider:
        human_format = "lafan1"

        def __init__(self, **kwargs: object) -> None:
            captured["input_kwargs"] = kwargs

    class DummyRetargeter:
        def __init__(self, **kwargs: object) -> None:
            captured["retarget_kwargs"] = kwargs

    class DummyLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["loop_kwargs"] = kwargs

    policy_path = tmp_path / "policy.onnx"
    policy_path.write_bytes(b"dummy")
    bvh_path = tmp_path / "input.bvh"
    bvh_path.write_text("HIERARCHY\n", encoding="utf-8")
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text("<mujoco model='dummy'/>", encoding="utf-8")

    robot_action_scale = [0.1, 0.2, 0.3]
    robot_default_angles = [-0.4, 0.5, -0.6]
    cfg = OmegaConf.create(
        {
            "robot": {
                "num_actions": 3,
                "xml_path": str(xml_path),
                "default_angles": robot_default_angles,
                "action_scale": robot_action_scale,
            },
            "controller": {
                "policy_path": str(policy_path),
                "action_scale": None,
                "default_dof_pos": None,
            },
            "input": {
                "provider": "bvh",
                "bvh_file": str(bvh_path),
                "bvh_format": "lafan1",
                "robot_name": "unitree_g1",
            },
            "policy_hz": 50,
            "pd_hz": 1000,
        }
    )

    monkeypatch.setattr("teleopit.pipeline.MuJoCoRobot", DummyRobot)
    monkeypatch.setattr("teleopit.pipeline.RLPolicyController", DummyController)
    monkeypatch.setattr("teleopit.pipeline.MjlabObservationBuilder", DummyObsBuilder)
    monkeypatch.setattr("teleopit.pipeline.BVHInputProvider", DummyInputProvider)
    monkeypatch.setattr("teleopit.pipeline.RetargetingModule", DummyRetargeter)
    monkeypatch.setattr("teleopit.pipeline.SimulationLoop", DummyLoop)

    TeleopPipeline(cfg)

    controller_cfg = captured["controller_cfg"]
    assert list(controller_cfg.default_dof_pos) == robot_default_angles
    assert list(controller_cfg.action_scale) == robot_action_scale


# =====================================================================
# Startup dim-validation integration tests
# These use the REAL MjlabObservationBuilder (not a dummy) so the
# dim-check branch in _build_obs_builder actually executes.
# =====================================================================


def _find_xml_path() -> str | None:
    candidates = [
        Path(__file__).parent.parent / "GMR" / "assets" / "unitree_g1" / "g1_sim2sim_29dof.xml",
        Path(__file__).parent.parent / "teleopit" / "retargeting" / "gmr" / "assets" / "unitree_g1" / "g1_sim2sim_29dof.xml",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


_XML_PATH = _find_xml_path()
_skip_no_xml = pytest.mark.skipif(_XML_PATH is None, reason="Robot XML not found")

_DEFAULT_ANGLES_29 = [
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
]


def _pipeline_cfg(tmp_path: Path, has_state_estimation: bool | None = None) -> dict:
    """Build a pipeline cfg dict. has_state_estimation=None means omit the key."""
    policy_path = tmp_path / "policy.onnx"
    policy_path.write_bytes(b"dummy")
    bvh_path = tmp_path / "input.bvh"
    bvh_path.write_text("HIERARCHY\n", encoding="utf-8")

    robot = {
        "num_actions": 29,
        "xml_path": _XML_PATH or "",
        "default_angles": _DEFAULT_ANGLES_29,
        "action_scale": [0.5] * 29,
    }
    if has_state_estimation is not None:
        robot["has_state_estimation"] = has_state_estimation
    return {
        "robot": robot,
        "controller": {
            "policy_path": str(policy_path),
            "action_scale": None,
            "default_dof_pos": None,
        },
        "input": {
            "provider": "bvh",
            "bvh_file": str(bvh_path),
            "bvh_format": "lafan1",
            "robot_name": "unitree_g1",
        },
        "policy_hz": 50,
        "pd_hz": 1000,
    }


@requires_mujoco
@_skip_no_xml
def test_pipeline_dim_mismatch_raises(monkeypatch, tmp_path: Path) -> None:
    """154D obs_builder + 160D policy → startup ValueError from _build_obs_builder."""

    class DummyController160:
        _expected_obs_dim = 160
        def __init__(self, cfg: object) -> None:
            pass

    class DummyRobot:
        def __init__(self, cfg: object) -> None:
            pass

    class DummyInputProvider:
        human_format = "lafan1"
        def __init__(self, **kw: object) -> None:
            pass

    class DummyRetargeter:
        def __init__(self, **kw: object) -> None:
            pass

    class DummyLoop:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

    monkeypatch.setattr("teleopit.pipeline.MuJoCoRobot", DummyRobot)
    monkeypatch.setattr("teleopit.pipeline.RLPolicyController", DummyController160)
    monkeypatch.setattr("teleopit.pipeline.BVHInputProvider", DummyInputProvider)
    monkeypatch.setattr("teleopit.pipeline.RetargetingModule", DummyRetargeter)
    monkeypatch.setattr("teleopit.pipeline.SimulationLoop", DummyLoop)
    # NOTE: MjlabObservationBuilder is NOT patched — real builder runs

    cfg = OmegaConf.create(_pipeline_cfg(tmp_path, has_state_estimation=False))

    with pytest.raises(ValueError, match="mismatch"):
        TeleopPipeline(cfg)


@requires_mujoco
@_skip_no_xml
def test_pipeline_dim_match_passes(monkeypatch, tmp_path: Path) -> None:
    """154D obs_builder + 154D policy → no error."""

    class DummyController154:
        _expected_obs_dim = 154
        def __init__(self, cfg: object) -> None:
            pass

    class DummyRobot:
        def __init__(self, cfg: object) -> None:
            pass

    class DummyInputProvider:
        human_format = "lafan1"
        def __init__(self, **kw: object) -> None:
            pass

    class DummyRetargeter:
        def __init__(self, **kw: object) -> None:
            pass

    class DummyLoop:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

    monkeypatch.setattr("teleopit.pipeline.MuJoCoRobot", DummyRobot)
    monkeypatch.setattr("teleopit.pipeline.RLPolicyController", DummyController154)
    monkeypatch.setattr("teleopit.pipeline.BVHInputProvider", DummyInputProvider)
    monkeypatch.setattr("teleopit.pipeline.RetargetingModule", DummyRetargeter)
    monkeypatch.setattr("teleopit.pipeline.SimulationLoop", DummyLoop)

    cfg = OmegaConf.create(_pipeline_cfg(tmp_path, has_state_estimation=False))

    pipeline = TeleopPipeline(cfg)
    assert pipeline.obs_builder.total_obs_size == 154


@requires_mujoco
@_skip_no_xml
def test_pipeline_160d_default_when_key_omitted(monkeypatch, tmp_path: Path) -> None:
    """When has_state_estimation is omitted from robot cfg, pipeline defaults to True (160D)."""

    class DummyController160:
        _expected_obs_dim = 160
        def __init__(self, cfg: object) -> None:
            pass

    class DummyRobot:
        def __init__(self, cfg: object) -> None:
            pass

    class DummyInputProvider:
        human_format = "lafan1"
        def __init__(self, **kw: object) -> None:
            pass

    class DummyRetargeter:
        def __init__(self, **kw: object) -> None:
            pass

    class DummyLoop:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

    monkeypatch.setattr("teleopit.pipeline.MuJoCoRobot", DummyRobot)
    monkeypatch.setattr("teleopit.pipeline.RLPolicyController", DummyController160)
    monkeypatch.setattr("teleopit.pipeline.BVHInputProvider", DummyInputProvider)
    monkeypatch.setattr("teleopit.pipeline.RetargetingModule", DummyRetargeter)
    monkeypatch.setattr("teleopit.pipeline.SimulationLoop", DummyLoop)

    cfg = OmegaConf.create(_pipeline_cfg(tmp_path, has_state_estimation=None))

    pipeline = TeleopPipeline(cfg)
    assert pipeline.obs_builder.total_obs_size == 160
