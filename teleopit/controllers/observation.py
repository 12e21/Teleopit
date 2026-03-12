from __future__ import annotations

import math
from collections import deque
from pathlib import Path
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


# =========================================================================
# MjlabObservationBuilder -- matches mjlab tracking environment (160D sim / 154D real)
# =========================================================================


def _quat_inv_np(q: FloatVec) -> FloatVec:
    """Inverse of wxyz quaternion (assumes unit quaternion)."""
    inv = q.copy()
    inv[..., 1:] = -inv[..., 1:]
    return inv


def _quat_mul_np(q1: FloatVec, q2: FloatVec) -> FloatVec:
    """Multiply two wxyz quaternions."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1).astype(np.float32)


def _quat_rotate_np(q: FloatVec, v: FloatVec) -> FloatVec:
    """Rotate vector v by wxyz quaternion q."""
    v_quat = np.zeros((*v.shape[:-1], 4), dtype=np.float32)
    v_quat[..., 1:4] = v
    result = _quat_mul_np(_quat_mul_np(q, v_quat), _quat_inv_np(q))
    return result[..., 1:4]


def _quat_to_rot6d_np(q: FloatVec) -> FloatVec:
    """Convert wxyz quaternion to 6D rotation matching PyTorch matrix[:, :2].reshape(-1).

    Training uses matrix[:, :2].reshape(-1) which gives [r00, r01, r10, r11, r20, r21]
    (first two elements of each row), NOT column-major order.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    return np.array([r00, r01, r10, r11, r20, r21], dtype=np.float32)


@final
class MjlabObservationBuilder:
    """Observation builder matching the mjlab tracking environment.

    Produces 160D or 154D policy observations depending on ``has_state_estimation``.

    160D (has_state_estimation=True, sim2sim default):
        command(58) + motion_anchor_pos_b(3) + motion_anchor_ori_b(6) +
        base_lin_vel(3) + base_ang_vel(3) + joint_pos_rel(29) +
        joint_vel(29) + last_action(29).

    154D (has_state_estimation=False, real-robot default):
        command(58) + motion_anchor_ori_b(6) +
        base_ang_vel(3) + joint_pos_rel(29) +
        joint_vel(29) + last_action(29).
    """

    def __init__(self, cfg: ConfigType) -> None:
        self.num_actions: int = _as_int_scalar(_cfg_get(cfg, "num_actions"), "num_actions")
        self.default_dof_pos: FloatVec = _as_float_vec(
            _cfg_get(cfg, "default_dof_pos"), "default_dof_pos"
        )

        # Whether the robot provides base_pos and base_lin_vel (sim only).
        try:
            raw = _cfg_get(cfg, "has_state_estimation")
        except KeyError:
            raw = True
        self.has_state_estimation: bool = bool(raw)

        # MuJoCo model for FK computation
        xml_path = str(_cfg_get(cfg, "xml_path"))
        if not Path(xml_path).is_file():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        import mujoco
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_data = mujoco.MjData(self._mj_model)

        # Build body name -> index mapping
        self._body_name_to_idx: dict[str, int] = {}
        for i in range(self._mj_model.nbody):
            name = self._mj_model.body(i).name
            self._body_name_to_idx[name] = i

        # Anchor body
        anchor_name = str(_cfg_get(cfg, "anchor_body_name")) if _cfg_get(cfg, "anchor_body_name") is not None else "torso_link"
        if anchor_name not in self._body_name_to_idx:
            raise ValueError(f"Anchor body '{anchor_name}' not found in model")
        self._anchor_body_id = self._body_name_to_idx[anchor_name]

        # 160D: command(2*29) + 3 + 6 + 3 + 3 + joint_pos_rel(29) + joint_vel(29) + last_action(29)
        # 154D: command(2*29)     + 6     + 3 + joint_pos_rel(29) + joint_vel(29) + last_action(29)
        state_est_dims = (3 + 3) if self.has_state_estimation else 0  # motion_anchor_pos_b + base_lin_vel
        self.total_obs_size: int = self.num_actions * 2 + state_est_dims + 6 + 3 + self.num_actions * 3

    def reset(self) -> None:
        """Reset internal state."""
        pass

    def _run_fk(self, base_pos: FloatVec, base_quat: FloatVec, joint_pos: FloatVec) -> None:
        """Run forward kinematics with full floating-base qpos."""
        import mujoco
        self._mj_data.qpos[:] = 0.0
        self._mj_data.qpos[0:3] = np.asarray(base_pos, dtype=np.float64).reshape(3)
        quat = np.asarray(base_quat, dtype=np.float64).reshape(4)
        quat = quat / max(np.linalg.norm(quat), 1e-8)
        self._mj_data.qpos[3:7] = quat
        n = min(len(joint_pos), self._mj_model.nq - 7)
        self._mj_data.qpos[7:7 + n] = np.asarray(joint_pos, dtype=np.float64)[:n]
        mujoco.mj_kinematics(self._mj_model, self._mj_data)

    def _get_body_pos(self, body_id: int) -> FloatVec:
        """Get body position from FK result."""
        return np.asarray(self._mj_data.xpos[body_id], dtype=np.float32).copy()

    def _get_body_quat(self, body_id: int) -> FloatVec:
        """Get body quaternion (wxyz) from FK result."""
        # MuJoCo xquat is already wxyz
        return np.asarray(self._mj_data.xquat[body_id], dtype=np.float32).copy()

    def build(
        self,
        robot_state: RobotState,
        motion_qpos: FloatVec,
        motion_joint_vel: FloatVec,
        last_action: FloatVec,
    ) -> FloatVec:
        """Build policy observation aligned with mjlab training config.

        Returns 160D when has_state_estimation=True, 154D when False.

        Args:
            robot_state: Current robot state.
            motion_qpos: Retargeted motion qpos (7D root + 29D joints).
            motion_joint_vel: Motion target joint velocities (29D).
            last_action: Previous action (29D)
        """
        qpos = np.asarray(robot_state.qpos, dtype=np.float32).reshape(-1)[:self.num_actions]
        qvel = np.asarray(robot_state.qvel, dtype=np.float32).reshape(-1)[:self.num_actions]
        robot_quat = np.asarray(robot_state.quat, dtype=np.float32).reshape(-1)
        ang_vel = np.asarray(robot_state.ang_vel, dtype=np.float32).reshape(-1)
        if robot_quat.shape[0] != 4:
            raise ValueError(f"robot_state.quat must be 4D (wxyz), got {robot_quat.shape[0]}")
        if ang_vel.shape[0] != 3:
            raise ValueError(f"robot_state.ang_vel must be 3D, got {ang_vel.shape[0]}")

        if self.has_state_estimation:
            if robot_state.base_pos is None or robot_state.base_lin_vel is None:
                raise ValueError(
                    "MjlabObservationBuilder with has_state_estimation=True requires "
                    "robot_state.base_pos and robot_state.base_lin_vel."
                )
            robot_base_pos = np.asarray(robot_state.base_pos, dtype=np.float32).reshape(-1)
            base_lin_vel = np.asarray(robot_state.base_lin_vel, dtype=np.float32).reshape(-1)
            if robot_base_pos.shape[0] != 3 or base_lin_vel.shape[0] != 3:
                raise ValueError(
                    f"robot_state.base_pos/base_lin_vel must be 3D, got "
                    f"{robot_base_pos.shape[0]}/{base_lin_vel.shape[0]}"
                )
        else:
            # No state estimation: use zeros for FK base_pos (orientation doesn't depend on it)
            robot_base_pos = np.zeros(3, dtype=np.float32)

        motion = np.asarray(motion_qpos, dtype=np.float32).reshape(-1)
        if motion.shape[0] < 7 + self.num_actions:
            raise ValueError(
                f"motion_qpos must contain 7D root + {self.num_actions}D joints, got {motion.shape[0]}"
            )
        motion_base_pos = motion[0:3]
        motion_base_quat = motion[3:7]
        motion_joint_pos = motion[7:7 + self.num_actions]
        motion_joint_vel_vec = np.asarray(motion_joint_vel, dtype=np.float32).reshape(-1)[:self.num_actions]
        if motion_joint_vel_vec.shape[0] != self.num_actions:
            raise ValueError(
                f"motion_joint_vel must be {self.num_actions}D, got {motion_joint_vel_vec.shape[0]}"
            )
        last_act = np.asarray(last_action, dtype=np.float32)
        if last_act.shape[0] != self.num_actions:
            raise ValueError(f"last_action length must be {self.num_actions}, got {last_act.shape[0]}")

        # Robot anchor state from robot current qpos.
        self._run_fk(robot_base_pos, robot_quat, qpos)
        robot_anchor_pos = self._get_body_pos(self._anchor_body_id)
        robot_anchor_quat = self._get_body_quat(self._anchor_body_id)

        # Motion anchor state from retargeted motion qpos.
        self._run_fk(motion_base_pos, motion_base_quat, motion_joint_pos)
        motion_anchor_pos = self._get_body_pos(self._anchor_body_id)
        motion_anchor_quat = self._get_body_quat(self._anchor_body_id)

        # command = target joint pos + target joint vel.
        command = np.concatenate((motion_joint_pos, motion_joint_vel_vec), dtype=np.float32)

        # motion anchor in robot-anchor frame.
        diff = motion_anchor_pos - robot_anchor_pos
        motion_anchor_pos_b = _quat_rotate_np(_quat_inv_np(robot_anchor_quat), diff)
        rel_quat = _quat_mul_np(_quat_inv_np(robot_anchor_quat), motion_anchor_quat)
        motion_anchor_ori_b = _quat_to_rot6d_np(rel_quat)

        # robot proprio.
        joint_pos_rel = qpos - self.default_dof_pos
        joint_vel_obs = qvel
        base_ang_vel = ang_vel

        if self.has_state_estimation:
            obs = np.concatenate([
                command,              # 58
                motion_anchor_pos_b,  # 3
                motion_anchor_ori_b,  # 6
                base_lin_vel,         # 3
                base_ang_vel,         # 3
                joint_pos_rel,        # 29
                joint_vel_obs,        # 29
                last_act,             # 29
            ], dtype=np.float32)
        else:
            obs = np.concatenate([
                command,              # 58
                motion_anchor_ori_b,  # 6
                base_ang_vel,         # 3
                joint_pos_rel,        # 29
                joint_vel_obs,        # 29
                last_act,             # 29
            ], dtype=np.float32)
        if obs.shape[0] != self.total_obs_size:
            raise ValueError(f"Expected {self.total_obs_size}D mjlab observation, got {obs.shape[0]}")
        return obs

    def build_observation(
        self,
        state: RobotState,
        history: list[FloatVec],
        action_mimic: FloatVec,
    ) -> FloatVec:
        raise ValueError(
            "MjlabObservationBuilder requires full motion qpos and joint velocities. "
            "Use build(robot_state, motion_qpos, motion_joint_vel, last_action)."
        )
