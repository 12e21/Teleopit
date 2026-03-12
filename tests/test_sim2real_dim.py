"""Integration tests for Sim2RealController startup dim validation.

These tests exercise the actual Sim2RealController.__init__ code path
(with SDK/UDP/retarget dependencies mocked out) to verify:
- Default has_state_estimation=False produces a 154D obs_builder
- Dim mismatch between obs_builder and policy raises at startup
- Matching dims pass without error
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from conftest import find_g1_xml_path, requires_mujoco


_XML_PATH = find_g1_xml_path()
_skip_no_xml = pytest.mark.skipif(_XML_PATH is None, reason="Robot XML not found")

_DEFAULT_ANGLES_29 = [
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
]


def _make_dummy_policy(expected_obs_dim: int) -> MagicMock:
    """Create a mock RLPolicyController with a specific _expected_obs_dim."""
    policy = MagicMock()
    policy._expected_obs_dim = expected_obs_dim
    return policy


def _make_sim2real_cfg(
    tmp_path: Path,
    has_state_estimation: bool | None = None,
) -> dict:
    """Build a minimal Sim2RealController cfg dict.

    has_state_estimation=None means omit the key from robot config,
    so the controller.py default (False) applies.
    """
    policy_path = tmp_path / "policy.onnx"
    policy_path.write_bytes(b"dummy")
    ref_bvh = tmp_path / "ref.bvh"
    ref_bvh.write_text("HIERARCHY\n", encoding="utf-8")

    robot = {
        "num_actions": 29,
        "xml_path": _XML_PATH or "",
        "default_angles": _DEFAULT_ANGLES_29,
        "action_scale": [0.5] * 29,
        "obs_builder": "mjlab",
        "anchor_body_name": "torso_link",
    }
    if has_state_estimation is not None:
        robot["has_state_estimation"] = has_state_estimation

    return {
        "policy_hz": 50.0,
        "real_robot": {"network_interface": "lo"},
        "gamepad": {},
        "mocap_switch": {},
        "input": {
            "reference_bvh": str(ref_bvh),
            "udp_host": "",
            "udp_port": 1118,
            "bvh_format": "hc_mocap",
            "udp_timeout": 0.1,
            "robot_name": "unitree_g1",
        },
        "controller": {
            "policy_path": str(policy_path),
            "action_scale": [0.5] * 29,
            "default_dof_pos": list(_DEFAULT_ANGLES_29),
        },
        "robot": robot,
    }


def _apply_sim2real_mocks(monkeypatch, policy_mock: MagicMock) -> None:
    """Monkeypatch all heavy SDK/IO dependencies so Sim2RealController.__init__ runs.

    Leaves MjlabObservationBuilder UN-patched so the real obs_builder + dim check runs.
    """
    # SDK robot + remote
    monkeypatch.setattr(
        "teleopit.sim2real.controller.UnitreeG1Robot",
        lambda cfg: MagicMock(check_mode=MagicMock(return_value={"name": "mock"})),
    )
    monkeypatch.setattr(
        "teleopit.sim2real.controller.UnitreeRemote",
        MagicMock,
    )

    # UDP provider
    dummy_provider = MagicMock()
    dummy_provider.human_format = "hc_mocap"
    monkeypatch.setattr(
        "teleopit.sim2real.controller.UDPBVHInputProvider",
        lambda **kw: dummy_provider,
    )

    # Retargeting
    monkeypatch.setattr(
        "teleopit.sim2real.controller.RetargetingModule",
        lambda **kw: MagicMock(),
    )

    # Policy controller — return the provided mock
    monkeypatch.setattr(
        "teleopit.sim2real.controller.RLPolicyController",
        lambda cfg: policy_mock,
    )


@requires_mujoco
@_skip_no_xml
class TestSim2RealStartupDim:
    """Integration tests for dim validation in Sim2RealController.__init__."""

    def test_default_no_state_estimation_produces_154d(self, monkeypatch, tmp_path: Path) -> None:
        """When has_state_estimation is omitted, sim2real defaults to False → 154D."""
        from teleopit.sim2real.controller import Sim2RealController

        policy_mock = _make_dummy_policy(154)
        _apply_sim2real_mocks(monkeypatch, policy_mock)
        cfg = _make_sim2real_cfg(tmp_path, has_state_estimation=None)

        ctrl = Sim2RealController(cfg)
        assert ctrl.obs_builder.total_obs_size == 154
        assert ctrl.obs_builder.has_state_estimation is False

    def test_explicit_false_produces_154d(self, monkeypatch, tmp_path: Path) -> None:
        """Explicit has_state_estimation=False → 154D."""
        from teleopit.sim2real.controller import Sim2RealController

        policy_mock = _make_dummy_policy(154)
        _apply_sim2real_mocks(monkeypatch, policy_mock)
        cfg = _make_sim2real_cfg(tmp_path, has_state_estimation=False)

        ctrl = Sim2RealController(cfg)
        assert ctrl.obs_builder.total_obs_size == 154

    def test_explicit_true_is_rejected(self, monkeypatch, tmp_path: Path) -> None:
        """Sim2real must reject has_state_estimation=True."""
        from teleopit.sim2real.controller import Sim2RealController

        policy_mock = _make_dummy_policy(154)
        _apply_sim2real_mocks(monkeypatch, policy_mock)
        cfg = _make_sim2real_cfg(tmp_path, has_state_estimation=True)

        with pytest.raises(ValueError, match="has_state_estimation=false"):
            Sim2RealController(cfg)

    def test_160d_policy_is_rejected(self, monkeypatch, tmp_path: Path) -> None:
        """Sim2real must reject 160D ONNX policies even if has_state_estimation=False."""
        from teleopit.sim2real.controller import Sim2RealController

        policy_mock = _make_dummy_policy(160)
        _apply_sim2real_mocks(monkeypatch, policy_mock)
        cfg = _make_sim2real_cfg(tmp_path, has_state_estimation=False)

        with pytest.raises(ValueError, match="only supports 154D"):
            Sim2RealController(cfg)

    def test_dim_match_passes(self, monkeypatch, tmp_path: Path) -> None:
        """154D obs_builder + 154D policy → no error."""
        from teleopit.sim2real.controller import Sim2RealController

        policy_mock = _make_dummy_policy(154)
        _apply_sim2real_mocks(monkeypatch, policy_mock)
        cfg = _make_sim2real_cfg(tmp_path, has_state_estimation=False)

        ctrl = Sim2RealController(cfg)
        assert ctrl.obs_builder.total_obs_size == 154
