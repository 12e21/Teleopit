"""Tests for teleopit.retargeting — RetargetingModule, extract_mimic_obs.

GMR requires MuJoCo XML assets and mink, so tests that instantiate the full
pipeline are guarded with skipif markers. We test the module-level import,
the RetargetingModule.retarget output contract via a mock GMR, and the
extract_mimic_obs helper.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from conftest import requires_mink, requires_mujoco


class TestRetargetingModuleImport:
    """Verify the module can be imported."""

    def test_import_retargeting_core(self):
        from teleopit.retargeting.core import RetargetingModule
        assert RetargetingModule is not None

    def test_import_from_package(self):
        from teleopit.retargeting import RetargetingModule
        assert callable(RetargetingModule)


class TestRetargetingModuleRetargetOutput:
    """Test retarget() output shape using a mocked GMR backend."""

    def _make_module_with_mock_gmr(self, fake_qpos):
        """Create a RetargetingModule with a mocked GMR that returns fake_qpos."""
        from teleopit.retargeting.core import RetargetingModule

        module = object.__new__(RetargetingModule)
        mock_gmr = MagicMock()
        mock_gmr.retarget.return_value = fake_qpos
        module._gmr = mock_gmr
        return module

    def test_retarget_returns_float64_array(self):
        """retarget() should return a float64 ndarray (full qpos)."""
        n_total = 3 + 4 + 29  # root_pos + root_quat + joints
        fake_qpos = np.arange(n_total, dtype=np.float64)

        module = self._make_module_with_mock_gmr(fake_qpos)
        result = module.retarget({})

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.shape == (n_total,)
        np.testing.assert_array_equal(result, fake_qpos)

    def test_retarget_preserves_qpos_structure(self):
        """Returned qpos should have 7D root + 29D joints = 36D."""
        fake_qpos = np.ones(36, dtype=np.float64)

        module = self._make_module_with_mock_gmr(fake_qpos)
        result = module.retarget({})

        # Verify we can extract root and joints
        root_pos = result[:3]
        root_quat = result[3:7]
        joints = result[7:]
        assert root_pos.shape == (3,)
        assert root_quat.shape == (4,)
        assert joints.shape == (29,)


class TestExtractMimicObs:
    """Test the extract_mimic_obs helper function."""

    def test_output_dimension_is_35(self):
        from teleopit.retargeting.core import extract_mimic_obs

        qpos = np.zeros(36, dtype=np.float64)
        qpos[3] = 1.0  # valid quaternion w component
        result = extract_mimic_obs(qpos)
        assert result.shape == (35,)
        assert result.dtype == np.float32

    def test_with_last_qpos(self):
        from teleopit.retargeting.core import extract_mimic_obs

        qpos = np.zeros(36, dtype=np.float64)
        qpos[3] = 1.0
        last_qpos = np.zeros(36, dtype=np.float64)
        last_qpos[3] = 1.0
        result = extract_mimic_obs(qpos, last_qpos=last_qpos)
        assert result.shape == (35,)

    def test_invalid_qpos_shape_raises(self):
        from teleopit.retargeting.core import extract_mimic_obs

        with pytest.raises(ValueError, match="1D"):
            extract_mimic_obs(np.zeros((2, 36), dtype=np.float64))

    def test_short_qpos_raises(self):
        from teleopit.retargeting.core import extract_mimic_obs

        with pytest.raises(ValueError, match="7D root"):
            extract_mimic_obs(np.zeros(10, dtype=np.float64))

    def test_negative_dt_raises(self):
        from teleopit.retargeting.core import extract_mimic_obs

        qpos = np.zeros(36, dtype=np.float64)
        qpos[3] = 1.0
        with pytest.raises(ValueError, match="dt must be positive"):
            extract_mimic_obs(qpos, dt=-1.0)


@requires_mujoco
@requires_mink
class TestRetargetingModuleInit:
    """Integration-level init test (only runs if mujoco+mink available)."""

    def test_init_with_invalid_robot_raises(self):
        from teleopit.retargeting.core import RetargetingModule

        with pytest.raises((KeyError, FileNotFoundError)):
            RetargetingModule(
                robot_name="nonexistent_robot_xyz",
                human_format="nonexistent_format",
            )
