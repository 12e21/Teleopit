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
# MjlabObservationBuilder -- matches mjlab tracking environment (~189D)
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
    """Convert wxyz quaternion to 6D rotation (first 2 columns of rotation matrix)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + w * z)
    r20 = 2 * (x * z - w * y)
    r01 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r21 = 2 * (y * z + w * x)
    return np.array([r00, r10, r20, r01, r11, r21], dtype=np.float32)


@final
class MjlabObservationBuilder:
    """Observation builder matching the mjlab tracking environment.

    Produces ~189D observations from robot state and motion reference,
    exactly matching the training-side observation structure.

    Observation layout:
        motion_anchor_pos_b  (3D) : motion target position in robot body frame
        motion_anchor_ori_b  (6D) : motion target orientation in robot body frame
        robot_anchor_ori_w   (6D) : robot orientation in world frame
        robot_anchor_lin_vel_w (3D) : robot linear velocity
        robot_anchor_ang_vel_w (3D) : robot angular velocity
        robot_body_pos_b     (nb*3) : tracked body positions in anchor frame
        robot_body_ori_b     (nb*6) : tracked body orientations in anchor frame
        joint_pos_rel        (29D) : joint positions relative to default
        joint_vel            (29D) : joint velocities
        last_action          (29D) : previous action
    """

    # Default tracked body names (9 bodies * 3 = 27D pos, 9 * 6 = 54D ori)
    # Uses bodies available in g1_sim2sim_29dof.xml (inference model).
    # head_mocap -> imu_in_torso (closest available in sim2sim model)
    DEFAULT_TRACKING_BODIES: list[str] = [
        "left_rubber_hand",
        "right_rubber_hand",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_elbow_link",
        "right_elbow_link",
        "imu_in_torso",
    ]

    # Fallback mapping: if a body is missing, try these alternatives
    _BODY_FALLBACKS: dict[str, str] = {
        "head_mocap": "imu_in_torso",
        "left_toe_link": "left_ankle_roll_link",
        "right_toe_link": "right_ankle_roll_link",
    }

    def __init__(self, cfg: ConfigType) -> None:
        self.num_actions: int = _as_int_scalar(_cfg_get(cfg, "num_actions"), "num_actions")
        self.default_dof_pos: FloatVec = _as_float_vec(
            _cfg_get(cfg, "default_dof_pos"), "default_dof_pos"
        )

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

        # Tracked body indices
        tracking_bodies = list(_cfg_get(cfg, "tracking_bodies")) if _cfg_get(cfg, "tracking_bodies") is not None else self.DEFAULT_TRACKING_BODIES
        self._tracking_body_names = [str(b) for b in tracking_bodies]
        self._tracking_body_ids: list[int] = []
        for name in self._tracking_body_names:
            if name in self._body_name_to_idx:
                self._tracking_body_ids.append(self._body_name_to_idx[name])
            elif name in self._BODY_FALLBACKS and self._BODY_FALLBACKS[name] in self._body_name_to_idx:
                self._tracking_body_ids.append(self._body_name_to_idx[self._BODY_FALLBACKS[name]])
            else:
                raise ValueError(f"Body '{name}' not found in MuJoCo model. Available: {list(self._body_name_to_idx.keys())}")

        # Anchor body
        anchor_name = str(_cfg_get(cfg, "anchor_body_name")) if _cfg_get(cfg, "anchor_body_name") is not None else "torso_link"
        if anchor_name not in self._body_name_to_idx:
            raise ValueError(f"Anchor body '{anchor_name}' not found in model")
        self._anchor_body_id = self._body_name_to_idx[anchor_name]

        nb = len(self._tracking_body_ids)
        # 3 + 6 + 6 + 3 + 3 + nb*3 + nb*6 + 29 + 29 + 29
        self.total_obs_size: int = 3 + 6 + 6 + 3 + 3 + nb * 3 + nb * 6 + self.num_actions * 3

    def reset(self) -> None:
        """Reset internal state."""
        pass

    def _run_fk(self, joint_pos: FloatVec) -> None:
        """Run forward kinematics with given joint positions."""
        import mujoco
        # Reset qpos, set joint values, run FK
        self._mj_data.qpos[:] = 0.0
        # Floating base: qpos[0:3] = pos, qpos[3:7] = quat (wxyz in MuJoCo)
        self._mj_data.qpos[2] = 0.76  # Default standing height
        self._mj_data.qpos[3] = 1.0   # Identity quaternion w
        n = min(len(joint_pos), self._mj_model.nq - 7)
        self._mj_data.qpos[7:7 + n] = joint_pos[:n]
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
        motion_anchor_pos_w: FloatVec,
        motion_anchor_quat_w: FloatVec,
        last_action: FloatVec,
    ) -> FloatVec:
        """Build observation vector (~189D).

        Args:
            robot_state: Current robot state (qpos, qvel, quat wxyz, ang_vel)
            motion_anchor_pos_w: Motion reference anchor position (3D)
            motion_anchor_quat_w: Motion reference anchor quaternion (4D, wxyz)
            last_action: Previous action (29D)

        Returns:
            Observation vector (~189D)
        """
        qpos = np.asarray(robot_state.qpos, dtype=np.float32)[:self.num_actions]
        qvel = np.asarray(robot_state.qvel, dtype=np.float32)[:self.num_actions]
        robot_quat = np.asarray(robot_state.quat, dtype=np.float32)  # wxyz
        ang_vel = np.asarray(robot_state.ang_vel, dtype=np.float32)
        motion_pos = np.asarray(motion_anchor_pos_w, dtype=np.float32)
        motion_quat = np.asarray(motion_anchor_quat_w, dtype=np.float32)
        last_act = np.asarray(last_action, dtype=np.float32)

        # Run FK to get body positions/orientations
        self._run_fk(qpos)

        # Robot anchor state from FK
        anchor_pos = self._get_body_pos(self._anchor_body_id)
        anchor_quat = self._get_body_quat(self._anchor_body_id)

        # Use robot_state quat/ang_vel for anchor (more accurate with IMU)
        anchor_quat = robot_quat
        lin_vel = np.zeros(3, dtype=np.float32)  # Not available from robot_state directly

        # 1. motion_anchor_pos_b: motion target in robot body frame (3D)
        diff = motion_pos - anchor_pos
        motion_anchor_pos_b = _quat_rotate_np(_quat_inv_np(anchor_quat), diff)

        # 2. motion_anchor_ori_b: relative orientation (6D)
        rel_quat = _quat_mul_np(_quat_inv_np(anchor_quat), motion_quat)
        motion_anchor_ori_b = _quat_to_rot6d_np(rel_quat)

        # 3. robot_anchor_ori_w (6D)
        robot_anchor_ori_w = _quat_to_rot6d_np(anchor_quat)

        # 4. robot_anchor_lin_vel_w (3D)
        robot_anchor_lin_vel_w = lin_vel

        # 5. robot_anchor_ang_vel_w (3D)
        robot_anchor_ang_vel_w = ang_vel

        # 6. robot_body_pos_b: tracked bodies in anchor frame (nb*3)
        body_pos_parts = []
        for bid in self._tracking_body_ids:
            bpos = self._get_body_pos(bid)
            rel_pos = _quat_rotate_np(_quat_inv_np(anchor_quat), bpos - anchor_pos)
            body_pos_parts.append(rel_pos)
        robot_body_pos_b = np.concatenate(body_pos_parts, dtype=np.float32)

        # 7. robot_body_ori_b: tracked bodies orientation in anchor frame (nb*6)
        body_ori_parts = []
        for bid in self._tracking_body_ids:
            bquat = self._get_body_quat(bid)
            rel_q = _quat_mul_np(_quat_inv_np(anchor_quat), bquat)
            body_ori_parts.append(_quat_to_rot6d_np(rel_q))
        robot_body_ori_b = np.concatenate(body_ori_parts, dtype=np.float32)

        # 8. joint_pos_rel (29D)
        joint_pos_rel = qpos - self.default_dof_pos

        # 9. joint_vel (29D)
        joint_vel_obs = qvel

        # 10. last_action (29D)
        last_action_obs = last_act

        obs = np.concatenate([
            motion_anchor_pos_b,      # 3
            motion_anchor_ori_b,      # 6
            robot_anchor_ori_w,       # 6
            robot_anchor_lin_vel_w,   # 3
            robot_anchor_ang_vel_w,   # 3
            robot_body_pos_b,         # nb*3
            robot_body_ori_b,         # nb*6
            joint_pos_rel,            # 29
            joint_vel_obs,            # 29
            last_action_obs,          # 29
        ], dtype=np.float32)

        return obs

    def build_observation(
        self,
        state: RobotState,
        history: list[FloatVec],
        action_mimic: FloatVec,
    ) -> FloatVec:
        """Build observation compatible with ObservationBuilder protocol.

        For MjlabObservationBuilder, action_mimic is interpreted as
        [motion_anchor_pos_w(3), motion_anchor_quat_w(4)] = 7D.
        """
        if history:
            last_action = _as_float_vec(history[-1], "history[-1]")
        else:
            last_action = np.zeros((self.num_actions,), dtype=np.float32)

        mimic = _as_float_vec(action_mimic, "action_mimic")
        motion_pos = mimic[:3]
        motion_quat = mimic[3:7] if len(mimic) >= 7 else np.array([1, 0, 0, 0], dtype=np.float32)

        return self.build(state, motion_pos, motion_quat, last_action)
