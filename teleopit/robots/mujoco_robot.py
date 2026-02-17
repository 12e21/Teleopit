"""MuJoCo simulation backend for robot control."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from omegaconf import DictConfig

from teleopit.interfaces import RobotState


class MuJoCoRobot:
    """MuJoCo-based robot simulation. Implements the Robot Protocol.

    All robot-specific parameters (gains, limits, default poses) are loaded
    from a Hydra DictConfig — zero hardcoded constants.
    """

    def __init__(self, cfg: DictConfig) -> None:
        # Resolve XML path to absolute
        xml_path = Path(cfg.xml_path).expanduser()
        if not xml_path.is_absolute():
            xml_path = Path.cwd() / xml_path
        xml_path = xml_path.resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # Simulation timestep from config (default 0.001)
        self.model.opt.timestep = float(cfg.get("sim_dt", 0.001))

        # Robot parameters — all from config
        self._num_actions = int(cfg.num_actions)
        self._kps = np.array(cfg.kps, dtype=np.float64)
        self._kds = np.array(cfg.kds, dtype=np.float64)
        self._default_dof_pos = np.array(cfg.default_angles, dtype=np.float64)
        self._torque_limits = np.array(cfg.torque_limits, dtype=np.float64)

        # action_scale can be scalar or per-joint array
        action_scale = cfg.action_scale
        if isinstance(action_scale, (int, float)):
            self._action_scale = np.full(self._num_actions, float(action_scale), dtype=np.float64)
        else:
            self._action_scale = np.array(action_scale, dtype=np.float64)

        # Full MuJoCo qpos for reset (7D root + joint DOFs)
        self._mujoco_default_qpos = np.array(cfg.mujoco_default_qpos, dtype=np.float64)

        # Initialize to default pose
        self.reset()

    # ── Properties ──────────────────────────────────────────────

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def default_dof_pos(self) -> np.ndarray:
        return self._default_dof_pos.copy()

    @property
    def kps(self) -> np.ndarray:
        return self._kps

    @property
    def kds(self) -> np.ndarray:
        return self._kds

    @property
    def action_scale(self) -> np.ndarray:
        return self._action_scale

    @property
    def torque_limits(self) -> np.ndarray:
        return self._torque_limits

    # ── Robot Protocol methods ──────────────────────────────────

    def get_state(self) -> RobotState:
        """Extract current robot state from MuJoCo data."""
        n = self._num_actions
        dof_pos = self.data.qpos[7 : 7 + n].copy()
        dof_vel = self.data.qvel[6 : 6 + n].copy()
        quat = self.data.qpos[3:7].copy()
        ang_vel = self.data.qvel[3:6].copy()
        return RobotState(
            qpos=dof_pos,
            qvel=dof_vel,
            quat=quat,
            ang_vel=ang_vel,
            timestamp=float(self.data.time),
        )

    def apply_torque(self, torque: np.ndarray) -> None:
        """Set joint torques via data.ctrl."""
        self.data.ctrl[: len(torque)] = torque

    def set_action(self, action: np.ndarray) -> None:
        """Alias for apply_torque — satisfies Robot Protocol."""
        self.apply_torque(action)

    def step(self) -> None:
        """Advance simulation by one timestep."""
        mujoco.mj_step(self.model, self.data)

    def reset(self, qpos: np.ndarray | None = None) -> None:
        """Reset simulation to default or specified qpos."""
        mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            self.data.qpos[: len(qpos)] = qpos
        else:
            self.data.qpos[: len(self._mujoco_default_qpos)] = self._mujoco_default_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
