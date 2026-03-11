"""MuJoCo simulation backend for robot control."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from omegaconf import DictConfig

from teleopit.interfaces import RobotState


def _quat_conjugate(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_multiply(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a_wxyz, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b_wxyz, dtype=np.float64)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _quat_rotate_inverse(quat_wxyz: np.ndarray, vec_xyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec_xyz, dtype=np.float64)
    qn = q / max(np.linalg.norm(q), 1e-8)
    q_inv = _quat_conjugate(qn)
    v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return _quat_multiply(_quat_multiply(q_inv, v_quat), qn)[1:]


def _quat_rotate(quat_wxyz: np.ndarray, vec_xyz: np.ndarray) -> np.ndarray:
    """Rotate vector by quaternion (forward rotation)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec_xyz, dtype=np.float64)
    qn = q / max(np.linalg.norm(q), 1e-8)
    v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return _quat_multiply(_quat_multiply(qn, v_quat), _quat_conjugate(qn))[1:]


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

        self.xml_path: str = str(xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Simulation timestep from config (default 0.005 to match mjlab training)
        self.model.opt.timestep = float(cfg.get("sim_dt", 0.005))

        # Match mjlab training solver settings
        solver = str(cfg.get("solver", "newton")).lower()
        solver_map = {"newton": mujoco.mjtSolver.mjSOL_NEWTON, "pgs": mujoco.mjtSolver.mjSOL_PGS, "cg": mujoco.mjtSolver.mjSOL_CG}
        self.model.opt.solver = solver_map.get(solver, mujoco.mjtSolver.mjSOL_NEWTON)
        integrator = str(cfg.get("integrator", "implicitfast")).lower()
        integrator_map = {"euler": mujoco.mjtIntegrator.mjINT_EULER, "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT, "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST}
        self.model.opt.integrator = integrator_map.get(integrator, mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        self.model.opt.iterations = int(cfg.get("solver_iterations", 10))
        self.model.opt.ls_iterations = int(cfg.get("ls_iterations", 20))

        # Look up IMU sensor addresses in sensordata for gyro and velocimeter.
        # These match the training's MuJoCo sensors at imu_in_pelvis site.
        self._gyro_adr: int | None = None
        self._velocimeter_adr: int | None = None
        for i in range(self.model.nsensor):
            name = self.model.sensor(i).name
            stype = int(self.model.sensor_type[i])
            if stype == mujoco.mjtSensor.mjSENS_GYRO:
                self._gyro_adr = int(self.model.sensor_adr[i])
            elif stype == mujoco.mjtSensor.mjSENS_VELOCIMETER:
                self._velocimeter_adr = int(self.model.sensor_adr[i])

        # Robot parameters — all from config
        self._num_actions = int(cfg.num_actions)
        self._kps = np.array(cfg.kps, dtype=np.float64)
        self._kds = np.array(cfg.kds, dtype=np.float64)
        self._default_dof_pos = np.array(cfg.default_angles, dtype=np.float64)
        self._torque_limits = np.array(cfg.torque_limits, dtype=np.float64)

        # Convert <motor> actuators to built-in PD (<general> biastype=affine)
        # to match mjlab training. With built-in PD, ctrl = position target and
        # MuJoCo computes force = kp*(ctrl - qpos) - kd*qvel internally during
        # the implicit integration step, giving much more stable dynamics.
        self._builtin_pd = bool(cfg.get("builtin_pd", True))
        if self._builtin_pd:
            n_act = min(self._num_actions, self.model.nu)
            for i in range(n_act):
                kp = float(self._kps[i])
                kd = float(self._kds[i])
                self.model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
                self.model.actuator_gainprm[i, 0] = kp
                self.model.actuator_biasprm[i, 0] = 0.0
                self.model.actuator_biasprm[i, 1] = -kp
                self.model.actuator_biasprm[i, 2] = -kd
                # Set force limits matching torque_limits
                self.model.actuator_forcelimited[i] = 1
                self.model.actuator_forcerange[i, 0] = -float(self._torque_limits[i])
                self.model.actuator_forcerange[i, 1] = float(self._torque_limits[i])

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
        base_pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()

        # Read IMU sensors directly from MuJoCo sensordata.
        # This exactly matches training which uses gyro/velocimeter sensors
        # at the imu_in_pelvis site. Using sensordata avoids errors from
        # the cvel-based emulation (cvel is COM-based, not origin-based).
        if self._velocimeter_adr is not None:
            base_lin_vel_b = self.data.sensordata[self._velocimeter_adr:self._velocimeter_adr + 3].copy()
        else:
            # Fallback: rotate qvel to body frame
            base_lin_vel_w = self.data.qvel[0:3].copy()
            base_lin_vel_b = _quat_rotate_inverse(quat, base_lin_vel_w)

        if self._gyro_adr is not None:
            ang_vel = self.data.sensordata[self._gyro_adr:self._gyro_adr + 3].copy()
        else:
            # Fallback: rotate world angular velocity to body frame
            ang_vel_world = self.data.qvel[3:6].copy()
            ang_vel = _quat_rotate_inverse(quat, ang_vel_world)

        return RobotState(
            qpos=dof_pos,
            qvel=dof_vel,
            quat=quat,
            ang_vel=np.asarray(ang_vel, dtype=np.float64),
            timestamp=float(self.data.time),
            base_pos=base_pos,
            base_lin_vel=np.asarray(base_lin_vel_b, dtype=np.float64),
        )

    def apply_torque(self, torque: np.ndarray) -> None:
        """Set joint torques via data.ctrl (only for external PD mode)."""
        self.data.ctrl[: len(torque)] = torque

    def set_position_target(self, target_pos: np.ndarray) -> None:
        """Set position targets for built-in PD actuators."""
        self.data.ctrl[: len(target_pos)] = target_pos

    def set_action(self, action: np.ndarray) -> None:
        """Set control signal — position targets for built-in PD, torques otherwise."""
        if self._builtin_pd:
            self.set_position_target(action)
        else:
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
