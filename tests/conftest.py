"""Shared fixtures for Teleopit test suite."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from teleopit.interfaces import RobotState


# ── Skip markers ────────────────────────────────────────────────

def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


requires_mujoco = pytest.mark.skipif(
    not _has_module("mujoco"), reason="mujoco not installed"
)
requires_onnxruntime = pytest.mark.skipif(
    not _has_module("onnxruntime"), reason="onnxruntime not installed"
)
requires_h5py = pytest.mark.skipif(
    not _has_module("h5py"), reason="h5py not installed"
)
requires_mink = pytest.mark.skipif(
    not _has_module("mink"), reason="mink not installed"
)


# ── Paths ───────────────────────────────────────────────────────

@pytest.fixture
def config_path():
    """Return path to config directory."""
    return os.path.join(os.path.dirname(__file__), "..", "teleopit", "configs")


@pytest.fixture
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


# ── Robot config values (from g1.yaml) ──────────────────────────

NUM_ACTIONS = 29

DEFAULT_ANGLES = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    0, 0, 0,
    0, 0.4, 0, 1.2, 0.0, 0.0, 0.0,
    0, -0.4, 0, 1.2, 0.0, 0.0, 0.0,
], dtype=np.float32)

ANKLE_IDX = np.array([4, 5, 10, 11], dtype=np.int64)


@pytest.fixture
def num_actions():
    return NUM_ACTIONS


@pytest.fixture
def default_angles():
    return DEFAULT_ANGLES.copy()


# ── Observation builder config ──────────────────────────────────

@pytest.fixture
def obs_builder_cfg():
    """Config dict for the legacy TWIST2ObservationBuilder."""
    return {
        "num_actions": NUM_ACTIONS,
        "ang_vel_scale": 0.25,
        "dof_pos_scale": 1.0,
        "dof_vel_scale": 0.05,
        "ankle_idx": ANKLE_IDX.tolist(),
        "default_dof_pos": DEFAULT_ANGLES.tolist(),
    }


# ── Fake RobotState ────────────────────────────────────────────

@pytest.fixture
def fake_robot_state():
    """Create a plausible RobotState for testing."""
    return RobotState(
        qpos=np.zeros(NUM_ACTIONS, dtype=np.float32),
        qvel=np.zeros(NUM_ACTIONS, dtype=np.float32),
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ang_vel=np.zeros(3, dtype=np.float32),
        timestamp=0.0,
    )


# ── Temp directory ──────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
