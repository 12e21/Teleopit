"""Tests for observation builders.

Covers both the legacy `TWIST2ObservationBuilder` and the primary
`MjlabObservationBuilder` (160D / 154D modes).
"""
from pathlib import Path

import numpy as np
import pytest

from teleopit.controllers.observation import TWIST2ObservationBuilder, MjlabObservationBuilder, quatToEuler
from teleopit.interfaces import RobotState
from conftest import NUM_ACTIONS, DEFAULT_ANGLES, ANKLE_IDX, requires_mujoco


class TestQuatToEuler:
    """Unit tests for quaternion-to-euler conversion."""

    def test_identity_quaternion(self):
        quat = np.array([1, 0, 0, 0], dtype=np.float32)
        euler = quatToEuler(quat)
        assert euler.shape == (3,)
        np.testing.assert_allclose(euler, [0, 0, 0], atol=1e-6)

    def test_90deg_rotation(self):
        # 90° around Z: quat = [cos(45°), 0, 0, sin(45°)]
        angle = np.pi / 2
        quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)], dtype=np.float32)
        euler = quatToEuler(quat)
        assert euler.shape == (3,)
        # yaw (euler[2]) should be ~pi/2
        np.testing.assert_allclose(euler[2], np.pi / 2, atol=1e-5)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="4D"):
            quatToEuler(np.zeros(3, dtype=np.float32))


class TestTWIST2ObservationBuilder:
    """Tests for the legacy TWIST2 observation builder."""

    def _make_builder(self, cfg=None):
        if cfg is None:
            cfg = {
                "num_actions": NUM_ACTIONS,
                "ang_vel_scale": 0.25,
                "dof_pos_scale": 1.0,
                "dof_vel_scale": 0.05,
                "ankle_idx": ANKLE_IDX.tolist(),
                "default_dof_pos": DEFAULT_ANGLES.tolist(),
            }
        return TWIST2ObservationBuilder(cfg)

    def _make_state(self):
        return RobotState(
            qpos=np.zeros(NUM_ACTIONS, dtype=np.float32),
            qvel=np.zeros(NUM_ACTIONS, dtype=np.float32),
            quat=np.array([1, 0, 0, 0], dtype=np.float32),
            ang_vel=np.zeros(3, dtype=np.float32),
            timestamp=0.0,
        )

    def test_output_dimension_is_1402(self):
        builder = self._make_builder()
        state = self._make_state()
        # mimic_obs size: total_obs_size - n_proprio - history contribution
        # We need to figure out a valid mimic_obs size. Let's compute:
        # n_proprio = 3 + 2 + 3*29 = 92
        # obs_single_dim = mimic_obs_dim + 92
        # total = obs_single_dim + hist_dim + mimic_obs_dim = 1402
        # hist_dim = 1402 - obs_single_dim - mimic_obs_dim
        # hist_dim = 1402 - (M+92) - M = 1402 - 2M - 92 = 1310 - 2M
        # hist_dim must be >= 0 → M <= 655
        # Let's use a reasonable mimic_obs size
        mimic_obs = np.zeros(100, dtype=np.float32)
        last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

        obs = builder.build(state, mimic_obs, last_action)
        assert obs.shape == (1402,)
        assert obs.dtype == np.float32

    def test_build_observation_interface(self):
        """build_observation() is the Protocol-compatible entry point."""
        builder = self._make_builder()
        state = self._make_state()
        mimic_obs = np.zeros(100, dtype=np.float32)

        obs = builder.build_observation(state, history=[], action_mimic=mimic_obs)
        assert obs.shape == (1402,)

    def test_build_observation_with_history(self):
        """build_observation() uses last element of history as last_action."""
        builder = self._make_builder()
        state = self._make_state()
        mimic_obs = np.zeros(100, dtype=np.float32)
        history = [np.ones(NUM_ACTIONS, dtype=np.float32)]

        obs = builder.build_observation(state, history=history, action_mimic=mimic_obs)
        assert obs.shape == (1402,)

    def test_reset_clears_history(self):
        builder = self._make_builder()
        state = self._make_state()
        mimic_obs = np.zeros(100, dtype=np.float32)
        last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # Build once to populate history
        builder.build(state, mimic_obs, last_action)
        assert len(builder.proprio_history_buf) > 0

        builder.reset()
        # After reset with known dim, buffer should be refilled with zeros
        assert len(builder.proprio_history_buf) == builder.history_len

    def test_mismatched_num_actions_raises(self):
        builder = self._make_builder()
        state = RobotState(
            qpos=np.zeros(10, dtype=np.float32),  # wrong size
            qvel=np.zeros(10, dtype=np.float32),
            quat=np.array([1, 0, 0, 0], dtype=np.float32),
            ang_vel=np.zeros(3, dtype=np.float32),
            timestamp=0.0,
        )
        mimic_obs = np.zeros(100, dtype=np.float32)
        last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

        with pytest.raises(ValueError):
            builder.build(state, mimic_obs, last_action)

    def test_config_validation_default_dof_pos_mismatch(self):
        """default_dof_pos length must match num_actions."""
        cfg = {
            "num_actions": 5,
            "ang_vel_scale": 0.25,
            "dof_pos_scale": 1.0,
            "dof_vel_scale": 0.05,
            "ankle_idx": [0, 1],
            "default_dof_pos": [0.0, 0.0, 0.0],  # length 3 != 5
        }
        with pytest.raises(ValueError, match="default_dof_pos"):
            TWIST2ObservationBuilder(cfg)


# =====================================================================
# MjlabObservationBuilder — 160D / 154D tests
# =====================================================================

def _find_xml_path() -> str | None:
    from conftest import find_g1_xml_path

    return find_g1_xml_path()


_XML_PATH = _find_xml_path()
_skip_no_xml = pytest.mark.skipif(_XML_PATH is None, reason="Robot XML not found")


def _mjlab_cfg(has_state_estimation: bool = True) -> dict:
    return {
        "num_actions": NUM_ACTIONS,
        "default_dof_pos": DEFAULT_ANGLES.tolist(),
        "xml_path": _XML_PATH or "",
        "anchor_body_name": "torso_link",
        "has_state_estimation": has_state_estimation,
    }


def _make_motion_qpos() -> np.ndarray:
    """Dummy motion qpos: 7D root + 29D joints."""
    qpos = np.zeros(36, dtype=np.float32)
    qpos[3] = 1.0  # identity quaternion w
    return qpos


def _make_state_with_base() -> RobotState:
    return RobotState(
        qpos=np.zeros(NUM_ACTIONS, dtype=np.float32),
        qvel=np.zeros(NUM_ACTIONS, dtype=np.float32),
        quat=np.array([1, 0, 0, 0], dtype=np.float32),
        ang_vel=np.zeros(3, dtype=np.float32),
        timestamp=0.0,
        base_pos=np.zeros(3, dtype=np.float32),
        base_lin_vel=np.zeros(3, dtype=np.float32),
    )


def _make_state_no_base() -> RobotState:
    return RobotState(
        qpos=np.zeros(NUM_ACTIONS, dtype=np.float32),
        qvel=np.zeros(NUM_ACTIONS, dtype=np.float32),
        quat=np.array([1, 0, 0, 0], dtype=np.float32),
        ang_vel=np.zeros(3, dtype=np.float32),
        timestamp=0.0,
    )


@requires_mujoco
@_skip_no_xml
class TestMjlabObservationBuilder160D:
    """Regression tests for the 160D (has_state_estimation=True) path."""

    def test_160d_output_dimension(self):
        builder = MjlabObservationBuilder(_mjlab_cfg(has_state_estimation=True))
        assert builder.total_obs_size == 160
        obs = builder.build(
            _make_state_with_base(), _make_motion_qpos(),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
        )
        assert obs.shape == (160,)
        assert obs.dtype == np.float32

    def test_160d_requires_base_pos(self):
        builder = MjlabObservationBuilder(_mjlab_cfg(has_state_estimation=True))
        with pytest.raises(ValueError, match="base_pos"):
            builder.build(
                _make_state_no_base(), _make_motion_qpos(),
                np.zeros(NUM_ACTIONS, dtype=np.float32),
                np.zeros(NUM_ACTIONS, dtype=np.float32),
            )


@requires_mujoco
@_skip_no_xml
class TestMjlabObservationBuilder154D:
    """Tests for the 154D (has_state_estimation=False) no-state-estimation path."""

    def test_154d_output_dimension(self):
        builder = MjlabObservationBuilder(_mjlab_cfg(has_state_estimation=False))
        assert builder.total_obs_size == 154
        obs = builder.build(
            _make_state_no_base(), _make_motion_qpos(),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
        )
        assert obs.shape == (154,)
        assert obs.dtype == np.float32

    def test_154d_no_base_pos_required(self):
        """base_pos=None should NOT raise when has_state_estimation=False."""
        builder = MjlabObservationBuilder(_mjlab_cfg(has_state_estimation=False))
        obs = builder.build(
            _make_state_no_base(), _make_motion_qpos(),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
        )
        assert obs.shape == (154,)

    def test_154d_field_order(self):
        """Verify 154D concatenation order matches mjlab No-State-Estimation task.

        Expected: command(58) + motion_anchor_ori_b(6) + base_ang_vel(3) +
                  joint_pos_rel(29) + joint_vel(29) + last_action(29)
        """
        builder = MjlabObservationBuilder(_mjlab_cfg(has_state_estimation=False))
        # Use distinguishable last_action to verify placement
        last_action = np.ones(NUM_ACTIONS, dtype=np.float32) * 0.42
        obs = builder.build(
            _make_state_no_base(), _make_motion_qpos(),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
            last_action,
        )
        # last_action should be the final 29 elements
        np.testing.assert_allclose(obs[-NUM_ACTIONS:], last_action, atol=1e-7)
        # command is the first 58 elements (2 * 29)
        assert obs[:58].shape == (58,)

    def test_154d_also_works_with_base_pos_provided(self):
        """Even if base_pos is provided, 154D mode ignores it."""
        builder = MjlabObservationBuilder(_mjlab_cfg(has_state_estimation=False))
        obs = builder.build(
            _make_state_with_base(), _make_motion_qpos(),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
        )
        assert obs.shape == (154,)


@requires_mujoco
@_skip_no_xml
class TestMjlabObsDimBasics:
    """Verify total_obs_size values for use by startup dim-validation.

    Integration tests that exercise the actual pipeline/sim2real dim-check
    code paths are in test_pipeline.py (test_pipeline_dim_mismatch_raises etc.).
    """

    def test_154d_total_obs_size(self):
        builder = MjlabObservationBuilder(_mjlab_cfg(has_state_estimation=False))
        assert builder.total_obs_size == 154

    def test_160d_total_obs_size(self):
        builder = MjlabObservationBuilder(_mjlab_cfg(has_state_estimation=True))
        assert builder.total_obs_size == 160
