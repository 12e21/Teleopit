from __future__ import annotations

import math
from collections import deque
from typing import cast, final

import numpy as np

from teleopit.interfaces import RobotState

FloatVec = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
IntVec = np.ndarray[tuple[int, ...], np.dtype[np.int64]]
ConfigType = dict[str, object] | object


def _as_float_vec(value: object, name: str) -> FloatVec:
    out = np.asarray(value, dtype=np.float32)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return out


def _as_int_vec(value: object, name: str) -> IntVec:
    out = np.asarray(value, dtype=np.int64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return out


def _cfg_get(cfg: ConfigType, key: str) -> object:
    if isinstance(cfg, dict):
        if key not in cfg:
            raise KeyError(key)
        return cast(object, cfg[key])
    if not hasattr(cfg, key):
        raise KeyError(key)
    return cast(object, object.__getattribute__(cfg, key))


def _as_int_scalar(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    return value


def _as_float_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a float")
    return float(value)


def quatToEuler(quat: FloatVec) -> FloatVec:
    if quat.shape[0] != 4:
        raise ValueError("quat must be 4D")

    euler = np.zeros((3,), dtype=np.float32)
    qw = float(cast(np.float32, quat[0]))
    qx = float(cast(np.float32, quat[1]))
    qy = float(cast(np.float32, quat[2]))
    qz = float(cast(np.float32, quat[3]))

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    euler[0] = np.float32(math.atan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1.0:
        euler[1] = np.float32(math.copysign(math.pi / 2.0, sinp))
    else:
        euler[1] = np.float32(math.asin(sinp))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    euler[2] = np.float32(math.atan2(siny_cosp, cosy_cosp))
    return euler


@final
class TWIST2ObservationBuilder:
    def __init__(self, cfg: ConfigType):
        self.num_actions: int = _as_int_scalar(_cfg_get(cfg, "num_actions"), "num_actions")
        self.ang_vel_scale: float = _as_float_scalar(_cfg_get(cfg, "ang_vel_scale"), "ang_vel_scale")
        self.dof_pos_scale: float = _as_float_scalar(_cfg_get(cfg, "dof_pos_scale"), "dof_pos_scale")
        self.dof_vel_scale: float = _as_float_scalar(_cfg_get(cfg, "dof_vel_scale"), "dof_vel_scale")
        self.ankle_idx: IntVec = _as_int_vec(_cfg_get(cfg, "ankle_idx"), "ankle_idx")
        self.default_dof_pos: FloatVec = _as_float_vec(_cfg_get(cfg, "default_dof_pos"), "default_dof_pos")

        if self.default_dof_pos.shape[0] != self.num_actions:
            raise ValueError("default_dof_pos length must match num_actions")

        self.history_len: int = 10
        self.n_proprio: int = 3 + 2 + 3 * self.num_actions
        self.total_obs_size: int = 1402
        self.proprio_history_buf: deque[FloatVec] = deque(maxlen=self.history_len)
        self._last_obs_single_dim: int | None = None
        self.reset()

    def reset(self) -> None:
        self.proprio_history_buf.clear()
        if self._last_obs_single_dim is None:
            return
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros((self._last_obs_single_dim,), dtype=np.float32))

    def build(self, robot_state: RobotState, mimic_obs: FloatVec, last_action: FloatVec) -> FloatVec:
        dof_pos = _as_float_vec(cast(object, robot_state.qpos), "robot_state.qpos")[: self.num_actions]
        dof_vel = _as_float_vec(cast(object, robot_state.qvel), "robot_state.qvel")[: self.num_actions]
        quat = _as_float_vec(cast(object, robot_state.quat), "robot_state.quat")
        ang_vel = _as_float_vec(cast(object, robot_state.ang_vel), "robot_state.ang_vel")
        mimic_obs = _as_float_vec(mimic_obs, "mimic_obs")
        last_action = _as_float_vec(last_action, "last_action")

        if dof_pos.shape[0] != self.num_actions or dof_vel.shape[0] != self.num_actions:
            raise ValueError("robot_state qpos/qvel must contain num_actions entries")
        if ang_vel.shape[0] != 3:
            raise ValueError("robot_state ang_vel must be 3D")
        if last_action.shape[0] != self.num_actions:
            raise ValueError("last_action length must match num_actions")

        rpy = quatToEuler(quat)
        obs_dof_vel = dof_vel.copy()
        obs_dof_vel[self.ankle_idx] = 0.0

        obs_proprio = np.concatenate(
            (
                ang_vel * self.ang_vel_scale,
                rpy[:2],
                (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                obs_dof_vel * self.dof_vel_scale,
                last_action,
            ),
            dtype=np.float32,
        )
        if obs_proprio.shape[0] != self.n_proprio:
            raise ValueError(f"Expected proprio dim {self.n_proprio}, got {obs_proprio.shape[0]}")

        obs_full = np.concatenate((mimic_obs, obs_proprio), dtype=np.float32)
        obs_single_dim = int(obs_full.shape[0])

        if self._last_obs_single_dim != obs_single_dim:
            self._last_obs_single_dim = obs_single_dim
            self.proprio_history_buf.clear()
            for _ in range(self.history_len):
                self.proprio_history_buf.append(np.zeros((obs_single_dim,), dtype=np.float32))

        obs_hist = np.asarray(self.proprio_history_buf, dtype=np.float32).reshape(-1)
        expected_hist_dim = self.total_obs_size - obs_single_dim - int(mimic_obs.shape[0])
        if expected_hist_dim < 0:
            raise ValueError("Configured total_obs_size is too small for current and future observation")
        if obs_hist.shape[0] >= expected_hist_dim:
            obs_hist = obs_hist[:expected_hist_dim]
        else:
            obs_hist = np.pad(obs_hist, (0, expected_hist_dim - int(obs_hist.shape[0])))

        self.proprio_history_buf.append(obs_full)
        obs_buf = np.concatenate((obs_full, obs_hist, mimic_obs.copy()), dtype=np.float32)

        if obs_buf.shape[0] != self.total_obs_size:
            raise ValueError(f"Expected {self.total_obs_size} obs, got {obs_buf.shape[0]}")
        return obs_buf

    def build_observation(
        self,
        state: RobotState,
        history: list[FloatVec],
        action_mimic: FloatVec,
    ) -> FloatVec:
        if history:
            last_action = _as_float_vec(history[-1], "history[-1]")
        else:
            last_action = np.zeros((self.num_actions,), dtype=np.float32)
        return self.build(state, action_mimic, last_action)
