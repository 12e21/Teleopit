from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from teleopit.pipeline import TeleopPipeline


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
