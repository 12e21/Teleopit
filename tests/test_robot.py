"""Tests for teleopit.robots.mujoco_robot — MuJoCoRobot init, state extraction, reset.

MuJoCoRobot requires a valid XML file and omegaconf DictConfig, so tests
are guarded with skipif markers. We find the actual XML path from config.
"""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from conftest import requires_mujoco


def _find_xml_path():
    """Try to locate the g1 XML file relative to the project."""
    candidates = [
        Path(__file__).parent.parent / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
        Path(__file__).parent.parent / "teleopit" / "retargeting" / "gmr" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _make_cfg(xml_path: str):
    """Build a DictConfig-like object for MuJoCoRobot."""
    try:
        from omegaconf import OmegaConf
        return OmegaConf.create({
            "xml_path": xml_path,
            "num_actions": 29,
            "kps": [100]*6 + [100]*6 + [150]*3 + [40]*4 + [20]*3 + [40]*4 + [20]*3,
            "kds": [2]*6 + [2]*6 + [4]*3 + [5]*4 + [1]*3 + [5]*4 + [1]*3,
            "default_angles": [
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
                0, 0, 0,
                0, 0.4, 0, 1.2, 0.0, 0.0, 0.0,
                0, -0.4, 0, 1.2, 0.0, 0.0, 0.0,
            ],
            "action_scale": 0.5,
            "torque_limits": [100]*6 + [100]*6 + [150]*3 + [40]*4 + [4.0]*3 + [40]*4 + [4.0]*3,
            "sim_dt": 0.001,
            "mujoco_default_qpos": [
                0, 0, 0.793, 1, 0, 0, 0,
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
                0, 0, 0,
                0, 0.2, 0, 1.2, 0.0, 0.0, 0.0,
                0, -0.2, 0, 1.2, 0.0, 0.0, 0.0,
            ],
        })
    except ImportError:
        return None


_XML_PATH = _find_xml_path()
_skip_no_xml = pytest.mark.skipif(_XML_PATH is None, reason="Robot XML not found")


@requires_mujoco
@_skip_no_xml
class TestMuJoCoRobotInit:
    """Test MuJoCoRobot initialization."""

    def test_init_creates_model_and_data(self):
        from teleopit.robots.mujoco_robot import MuJoCoRobot
        cfg = _make_cfg(_XML_PATH)
        if cfg is None:
            pytest.skip("omegaconf not available")
        robot = MuJoCoRobot(cfg)
        assert robot.model is not None
        assert robot.data is not None
        assert robot.num_actions == 29

    def test_init_missing_xml_raises(self):
        from teleopit.robots.mujoco_robot import MuJoCoRobot
        cfg = _make_cfg("/nonexistent/path/robot.xml")
        if cfg is None:
            pytest.skip("omegaconf not available")
        with pytest.raises(FileNotFoundError):
            MuJoCoRobot(cfg)


@requires_mujoco
@_skip_no_xml
class TestMuJoCoRobotState:
    """Test state extraction."""

    def test_get_state_returns_robot_state(self):
        from teleopit.robots.mujoco_robot import MuJoCoRobot
        from teleopit.interfaces import RobotState
        cfg = _make_cfg(_XML_PATH)
        if cfg is None:
            pytest.skip("omegaconf not available")
        robot = MuJoCoRobot(cfg)
        state = robot.get_state()
        assert isinstance(state, RobotState)
        assert state.qpos.shape == (29,)
        assert state.qvel.shape == (29,)
        assert state.quat.shape == (4,)
        assert state.ang_vel.shape == (3,)
        assert isinstance(state.timestamp, float)

    def test_reset_restores_default_pose(self):
        from teleopit.robots.mujoco_robot import MuJoCoRobot
        cfg = _make_cfg(_XML_PATH)
        if cfg is None:
            pytest.skip("omegaconf not available")
        robot = MuJoCoRobot(cfg)

        # Perturb state
        robot.data.qpos[:] = 999.0
        robot.reset()

        # After reset, joint qpos should match mujoco_default_qpos[7:]
        state = robot.get_state()
        expected = np.array(cfg.mujoco_default_qpos, dtype=np.float64)[7:]
        np.testing.assert_allclose(state.qpos, expected, atol=1e-6)


@requires_mujoco
@_skip_no_xml
class TestMuJoCoRobotStep:
    """Test step and action."""

    def test_step_advances_time(self):
        from teleopit.robots.mujoco_robot import MuJoCoRobot
        cfg = _make_cfg(_XML_PATH)
        if cfg is None:
            pytest.skip("omegaconf not available")
        robot = MuJoCoRobot(cfg)
        t0 = robot.data.time
        robot.step()
        assert robot.data.time > t0

    def test_set_action_applies_torque(self):
        from teleopit.robots.mujoco_robot import MuJoCoRobot
        cfg = _make_cfg(_XML_PATH)
        if cfg is None:
            pytest.skip("omegaconf not available")
        robot = MuJoCoRobot(cfg)
        action = np.ones(29, dtype=np.float64) * 5.0
        robot.set_action(action)
        np.testing.assert_allclose(robot.data.ctrl[:29], 5.0)
