from __future__ import annotations

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, cast

import numpy as np
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

from teleopit.constants import FULL_QPOS_DIM
from teleopit.bus.topics import TOPIC_ACTION, TOPIC_MIMIC_OBS, TOPIC_ROBOT_STATE
from teleopit.controllers.observation import VelCmdObservationBuilder
from teleopit.controllers import reference_processing as ref_proc
from teleopit.interfaces import MessageBus, ObservationBuilder, Robot, RobotState
from teleopit.retargeting.core import extract_mimic_obs
from teleopit.sim.reference_timeline import ReferenceWindow
from teleopit.sim.reference_utils import obs_builder_requires_reference_window
from teleopit.sim.realtime_utils import ExponentialVecSmoother

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]


class _SupportsGetTarget(Protocol):
    def get_target_dof_pos(self, raw_action: Float32Array) -> Float32Array: ...




@dataclass(frozen=True)
class MotionPreparation:
    qpos: Float64Array
    retarget_viewer_qpos: Float64Array
    mimic_obs: Float32Array
    motion_joint_vel: Float32Array
    raw_motion_joint_vel: Float32Array
    motion_anchor_lin_vel_w: Float32Array | None = None
    motion_anchor_ang_vel_w: Float32Array | None = None
    raw_motion_anchor_lin_vel_w: Float32Array | None = None
    raw_motion_anchor_ang_vel_w: Float32Array | None = None


class RuntimePublisher:
    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus

    def publish(self, mimic_obs: Float32Array, action: Float32Array, robot_state: object) -> None:
        self._bus.publish(TOPIC_MIMIC_OBS, mimic_obs)
        self._bus.publish(TOPIC_ACTION, action)
        self._bus.publish(TOPIC_ROBOT_STATE, robot_state)


class PolicyStepRunner:
    def __init__(
        self,
        *,
        robot: Robot,
        controller: object,
        obs_builder: ObservationBuilder,
        policy_hz: float,
        decimation: int,
        num_actions: int,
        kps: Float32Array,
        kds: Float32Array,
        torque_limits: Float32Array,
        default_dof_pos: Float32Array,
        reference_velocity_smoothing_alpha: float = 1.0,
        reference_anchor_velocity_smoothing_alpha: float = 1.0,
    ) -> None:
        self.robot = robot
        self.controller = controller
        self.obs_builder = obs_builder
        self.policy_hz = policy_hz
        self.decimation = decimation
        self.num_actions = num_actions
        self.kps = kps
        self.kds = kds
        self.torque_limits = torque_limits
        self.default_dof_pos = default_dof_pos
        self._motion_joint_vel_smoother = ExponentialVecSmoother(reference_velocity_smoothing_alpha)
        self._motion_anchor_lin_vel_smoother = ExponentialVecSmoother(reference_anchor_velocity_smoothing_alpha)
        self._motion_anchor_ang_vel_smoother = ExponentialVecSmoother(reference_anchor_velocity_smoothing_alpha)
        self.last_action: Float32Array = np.zeros((self.num_actions,), dtype=np.float32)
        self.last_retarget_qpos: Float64Array | None = None
        self.last_reference_qpos: Float64Array | None = None
        self._pending_reference_qpos: Float64Array | None = None
        self._fixed_reference_yaw_quat: Float32Array | None = None
        self._fixed_reference_pivot_pos_w: Float32Array | None = None
        self._fixed_reference_xy_offset_w: Float32Array | None = None
        self._reference_alignment_target_xy_w: Float32Array | None = None
        self._joint_limit_lower: Float32Array | None = None
        self._joint_limit_upper: Float32Array | None = None
        self._joint_limit_limited: NDArray[np.bool_] | None = None
        self._joint_limit_joint_names: tuple[str, ...] | None = None
        self._last_original_target: Float32Array | None = None
        self._last_guarded_target: Float32Array | None = None
        self._last_joint_limit_clip_mask: NDArray[np.bool_] | None = None
        self._last_joint_limit_correction: Float32Array | None = None
        self._last_joint_limit_max_correction = 0.0
        self._initialize_model_joint_limits()

    @property
    def last_original_target(self) -> Float32Array | None:
        return None if self._last_original_target is None else self._last_original_target.copy()

    @property
    def last_guarded_target(self) -> Float32Array | None:
        return None if self._last_guarded_target is None else self._last_guarded_target.copy()

    @property
    def last_joint_limit_clip_mask(self) -> NDArray[np.bool_] | None:
        if self._last_joint_limit_clip_mask is None:
            return None
        return self._last_joint_limit_clip_mask.copy()

    @property
    def last_joint_limit_correction(self) -> Float32Array | None:
        if self._last_joint_limit_correction is None:
            return None
        return self._last_joint_limit_correction.copy()

    @property
    def last_joint_limit_max_correction(self) -> float:
        return self._last_joint_limit_max_correction

    @property
    def joint_limit_joint_names(self) -> tuple[str, ...] | None:
        return self._joint_limit_joint_names

    def _initialize_model_joint_limits(self) -> None:
        model = getattr(self.robot, "model", None)
        if model is None or not hasattr(model, "actuator_trnid"):
            return

        required = ("nu", "njnt", "nq", "jnt_qposadr", "jnt_limited", "jnt_range")
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise ValueError(
                "MuJoCo model is missing joint-limit mapping fields: " + ", ".join(missing)
            )

        nu = int(model.nu)
        njnt = int(model.njnt)
        nq = int(model.nq)
        if nu < self.num_actions:
            raise ValueError(
                f"MuJoCo model has {nu} actuators, expected at least {self.num_actions}"
            )

        actuator_trnid = np.asarray(model.actuator_trnid)
        jnt_qposadr = np.asarray(model.jnt_qposadr).reshape(-1)
        jnt_limited = np.asarray(model.jnt_limited).reshape(-1)
        jnt_range = np.asarray(model.jnt_range)
        if actuator_trnid.ndim != 2 or actuator_trnid.shape[0] < self.num_actions:
            raise ValueError(
                "MuJoCo actuator_trnid does not contain the required actuator mappings"
            )
        if jnt_qposadr.shape[0] < njnt or jnt_limited.shape[0] < njnt:
            raise ValueError("MuJoCo joint metadata is incomplete")
        if jnt_range.ndim != 2 or jnt_range.shape[0] < njnt or jnt_range.shape[1] < 2:
            raise ValueError("MuJoCo jnt_range metadata is incomplete")

        joint_ids: list[int] = []
        qpos_addresses: list[int] = []
        joint_names: list[str] = []
        lower = np.zeros((self.num_actions,), dtype=np.float32)
        upper = np.zeros((self.num_actions,), dtype=np.float32)
        limited = np.zeros((self.num_actions,), dtype=np.bool_)

        joint_accessor = getattr(model, "joint", None)
        if not callable(joint_accessor):
            raise ValueError("MuJoCo model does not provide joint names through model.joint()")

        for actuator_index in range(self.num_actions):
            joint_id = int(actuator_trnid[actuator_index, 0])
            if joint_id < 0 or joint_id >= njnt:
                raise ValueError(
                    f"Actuator {actuator_index} has invalid joint id {joint_id}"
                )
            qpos_address = int(jnt_qposadr[joint_id])
            if qpos_address < 0 or qpos_address >= nq:
                raise ValueError(
                    f"Actuator {actuator_index} joint {joint_id} has invalid qpos address "
                    f"{qpos_address}"
                )
            if joint_id in joint_ids:
                raise ValueError(f"Actuator {actuator_index} repeats joint id {joint_id}")
            if qpos_address in qpos_addresses:
                raise ValueError(
                    f"Actuator {actuator_index} repeats qpos address {qpos_address}"
                )

            joint_name = str(getattr(joint_accessor(joint_id), "name", ""))
            if not joint_name:
                raise ValueError(f"Joint {joint_id} has no valid name")

            is_limited = bool(jnt_limited[joint_id])
            joint_lower = float(jnt_range[joint_id, 0])
            joint_upper = float(jnt_range[joint_id, 1])
            if is_limited and (
                not np.isfinite(joint_lower)
                or not np.isfinite(joint_upper)
                or joint_lower > joint_upper
            ):
                raise ValueError(
                    f"Joint '{joint_name}' has invalid range [{joint_lower}, {joint_upper}]"
                )

            joint_ids.append(joint_id)
            qpos_addresses.append(qpos_address)
            joint_names.append(joint_name)
            limited[actuator_index] = is_limited
            lower[actuator_index] = np.float32(joint_lower)
            upper[actuator_index] = np.float32(joint_upper)

        self._joint_limit_lower = lower.copy()
        self._joint_limit_upper = upper.copy()
        self._joint_limit_limited = limited.copy()
        self._joint_limit_joint_names = tuple(joint_names)

    def reset(self) -> None:
        self.last_action = np.zeros((self.num_actions,), dtype=np.float32)
        self.last_retarget_qpos = None
        self.last_reference_qpos = None
        self._pending_reference_qpos = None
        self._fixed_reference_yaw_quat = None
        self._fixed_reference_pivot_pos_w = None
        self._fixed_reference_xy_offset_w = None
        self._reference_alignment_target_xy_w = None
        self._last_original_target = None
        self._last_guarded_target = None
        self._last_joint_limit_clip_mask = None
        self._last_joint_limit_correction = None
        self._last_joint_limit_max_correction = 0.0
        self._motion_joint_vel_smoother.reset()
        self._motion_anchor_lin_vel_smoother.reset()
        self._motion_anchor_ang_vel_smoother.reset()

    def soft_reset_reference_state(self, *, reset_alignment: bool = True) -> None:
        self.last_reference_qpos = None
        self._pending_reference_qpos = None
        if reset_alignment:
            self._fixed_reference_yaw_quat = None
            self._fixed_reference_pivot_pos_w = None
            self._fixed_reference_xy_offset_w = None
            self._reference_alignment_target_xy_w = None
        self._motion_joint_vel_smoother.reset()
        self._motion_anchor_lin_vel_smoother.reset()
        self._motion_anchor_ang_vel_smoother.reset()

    def reset_reference_alignment(self, target_qpos: Float64Array | None = None) -> None:
        self._fixed_reference_yaw_quat = None
        self._fixed_reference_pivot_pos_w = None
        self._fixed_reference_xy_offset_w = None
        self._reference_alignment_target_xy_w = (
            None
            if target_qpos is None
            else np.asarray(target_qpos[0:2], dtype=np.float32).reshape(2).copy()
        )

    def prepare_static_motion_command(self, qpos: Float64Array) -> MotionPreparation:
        hold_qpos = self._retarget_to_qpos(qpos)
        mimic_obs = extract_mimic_obs(
            qpos=hold_qpos,
            last_qpos=hold_qpos,
            dt=1.0 / self.policy_hz,
        )
        zeros_joint_vel = np.zeros((self.num_actions,), dtype=np.float32)
        zeros_anchor_vel = np.zeros(3, dtype=np.float32)
        self._pending_reference_qpos = hold_qpos.copy()
        return MotionPreparation(
            qpos=hold_qpos.copy(),
            retarget_viewer_qpos=hold_qpos.copy(),
            mimic_obs=np.asarray(mimic_obs, dtype=np.float32),
            motion_joint_vel=zeros_joint_vel,
            raw_motion_joint_vel=zeros_joint_vel.copy(),
            motion_anchor_lin_vel_w=zeros_anchor_vel,
            motion_anchor_ang_vel_w=zeros_anchor_vel.copy(),
            raw_motion_anchor_lin_vel_w=zeros_anchor_vel.copy(),
            raw_motion_anchor_ang_vel_w=zeros_anchor_vel.copy(),
        )

    def prepare_motion_command(self, retargeted: object, state: object) -> MotionPreparation:
        reference_qpos = self._retarget_to_qpos(retargeted)
        reference_qpos = self._align_velcmd_reference_yaw(reference_qpos, state)
        self._pending_reference_qpos = reference_qpos.copy()
        qpos = reference_qpos.copy()

        mimic_obs = extract_mimic_obs(qpos=qpos, last_qpos=self.last_retarget_qpos, dt=1.0 / self.policy_hz)
        retarget_viewer_qpos = qpos.copy()
        raw_motion_joint_vel = self._compute_motion_joint_vel(qpos)
        motion_joint_vel = self._motion_joint_vel_smoother.apply(raw_motion_joint_vel)

        raw_motion_anchor_lin_vel_w: Float32Array | None = None
        raw_motion_anchor_ang_vel_w: Float32Array | None = None
        motion_anchor_lin_vel_w: Float32Array | None
        motion_anchor_ang_vel_w: Float32Array | None
        if obs_builder_requires_reference_window(self.obs_builder):
            motion_anchor_lin_vel_w = None
            motion_anchor_ang_vel_w = None
        else:
            raw_motion_anchor_lin_vel_w, raw_motion_anchor_ang_vel_w = self._compute_anchor_velocities(reference_qpos)
            motion_anchor_lin_vel_w = self._motion_anchor_lin_vel_smoother.apply(raw_motion_anchor_lin_vel_w)
            motion_anchor_ang_vel_w = self._motion_anchor_ang_vel_smoother.apply(raw_motion_anchor_ang_vel_w)

        return MotionPreparation(
            qpos=qpos,
            retarget_viewer_qpos=retarget_viewer_qpos,
            mimic_obs=np.asarray(mimic_obs, dtype=np.float32),
            motion_joint_vel=motion_joint_vel,
            raw_motion_joint_vel=raw_motion_joint_vel,
            motion_anchor_lin_vel_w=motion_anchor_lin_vel_w,
            motion_anchor_ang_vel_w=motion_anchor_ang_vel_w,
            raw_motion_anchor_lin_vel_w=raw_motion_anchor_lin_vel_w,
            raw_motion_anchor_ang_vel_w=raw_motion_anchor_ang_vel_w,
        )

    def _align_velcmd_reference_yaw(self, reference_qpos: Float64Array, state: RobotState) -> Float64Array:
        robot_quat = np.asarray(state.quat, dtype=np.float32)
        (
            aligned,
            self._fixed_reference_yaw_quat,
            self._fixed_reference_pivot_pos_w,
            self._fixed_reference_xy_offset_w,
        ) = ref_proc.align_reference_yaw(
            reference_qpos,
            robot_quat,
            self._fixed_reference_yaw_quat,
            self._fixed_reference_pivot_pos_w,
            self._fixed_reference_xy_offset_w,
            self._reference_alignment_target_xy_w,
        )
        return aligned

    def _compute_anchor_velocities(
        self, qpos: Float64Array
    ) -> tuple[Float32Array, Float32Array]:
        """Compute motion anchor linear/angular velocity in world frame via finite diff."""
        return ref_proc.compute_anchor_velocities(
            self.obs_builder._base, qpos, self.last_reference_qpos,
            self.num_actions, self.policy_hz,
        )

    def compute_target_dof_pos(self, action: Float32Array) -> Float32Array:
        get_target = getattr(self.controller, "get_target_dof_pos", None)
        if callable(get_target):
            target = np.asarray(
                cast(_SupportsGetTarget, cast(object, self.controller)).get_target_dof_pos(action),
                dtype=np.float32,
            ).reshape(-1)
        else:
            target = action + self.default_dof_pos

        if target.shape[0] != self.num_actions:
            raise ValueError(f"Target dof pos has {target.shape[0]} entries, expected {self.num_actions}")

        original_target = target.copy()
        guarded_target = original_target.copy()
        if (
            self._joint_limit_limited is not None
            and self._joint_limit_lower is not None
            and self._joint_limit_upper is not None
        ):
            limited = self._joint_limit_limited
            guarded_target[limited] = np.clip(
                guarded_target[limited],
                self._joint_limit_lower[limited],
                self._joint_limit_upper[limited],
            )

        correction = np.asarray(guarded_target - original_target, dtype=np.float32)
        clip_mask = np.asarray(correction != 0.0, dtype=np.bool_)
        self._last_original_target = original_target.copy()
        self._last_guarded_target = guarded_target.copy()
        self._last_joint_limit_clip_mask = clip_mask.copy()
        self._last_joint_limit_correction = correction.copy()
        self._last_joint_limit_max_correction = float(
            np.max(np.abs(correction), initial=np.float32(0.0))
        )
        return guarded_target.copy()

    def _align_reference_window(
        self,
        reference_window: ReferenceWindow | None,
        state: RobotState,
    ) -> ReferenceWindow | None:
        robot_quat = np.asarray(state.quat, dtype=np.float32)
        (
            aligned,
            self._fixed_reference_yaw_quat,
            self._fixed_reference_pivot_pos_w,
            self._fixed_reference_xy_offset_w,
        ) = ref_proc.align_reference_window(
            reference_window,
            robot_quat,
            self._fixed_reference_yaw_quat,
            self._fixed_reference_pivot_pos_w,
            self._fixed_reference_xy_offset_w,
            self._reference_alignment_target_xy_w,
        )
        return aligned

    def build_observation(
        self,
        state: RobotState,
        motion_prep: MotionPreparation,
        last_action: Float32Array,
        reference_window: ReferenceWindow | None = None,
    ) -> Float32Array:
        motion_qpos = np.asarray(motion_prep.qpos[:7 + self.num_actions], dtype=np.float32)
        motion_joint_vel = np.asarray(motion_prep.motion_joint_vel, dtype=np.float32)
        aligned_reference_window = self._align_reference_window(reference_window, state)
        return ref_proc.dispatch_build_observation(
            self.obs_builder, state, reference_window, aligned_reference_window,
            motion_qpos, motion_joint_vel, last_action,
            motion_prep.motion_anchor_lin_vel_w, motion_prep.motion_anchor_ang_vel_w,
        )

    def _compute_motion_joint_vel(self, retarget_qpos: Float64Array) -> Float32Array:
        if retarget_qpos.shape[0] < 7 + self.num_actions:
            raise ValueError(
                f"Retargeted qpos too short: {retarget_qpos.shape[0]} "
                f"(need >= {7 + self.num_actions})"
            )
        motion_joint_pos = np.asarray(retarget_qpos[7:7 + self.num_actions], dtype=np.float32)
        if self.last_retarget_qpos is None:
            return np.zeros((self.num_actions,), dtype=np.float32)
        prev_joint_pos = np.asarray(self.last_retarget_qpos[7:7 + self.num_actions], dtype=np.float32)
        vel = np.asarray((motion_joint_pos - prev_joint_pos) * np.float32(self.policy_hz), dtype=np.float32)
        if not np.all(np.isfinite(vel)):
            _logger.warning("NaN/inf in motion_joint_vel, damping to zero")
            return np.zeros((self.num_actions,), dtype=np.float32)
        return vel

    def _extract_motion_joint_data(
        self, retarget_qpos: Float64Array
    ) -> tuple[Float32Array, Float32Array]:
        """Extract motion qpos and joint velocities from retarget qpos."""
        motion_qpos = np.asarray(retarget_qpos[:7 + self.num_actions], dtype=np.float32)
        return motion_qpos, self._compute_motion_joint_vel(retarget_qpos)

    def validate_observation_for_policy(self, obs: Float32Array) -> Float32Array:
        expected_raw = getattr(self.controller, "_expected_obs_dim", None)
        if not isinstance(expected_raw, int) or expected_raw <= 0:
            return obs
        if obs.shape[0] != expected_raw:
            raise ValueError(
                f"Observation dimension mismatch: obs_builder produced {obs.shape[0]}, "
                f"but policy expects {expected_raw}. "
                "Use a matching observation builder and ONNX policy; automatic pad/trim is disabled."
            )
        return obs

    def apply_control(self, target_dof_pos: Float32Array) -> tuple[Float32Array, object]:
        builtin_pd = getattr(self.robot, "_builtin_pd", False)
        torque: Float32Array = np.zeros((self.num_actions,), dtype=np.float32)

        if builtin_pd:
            self.robot.set_action(target_dof_pos)
            for _ in range(self.decimation):
                self.robot.step()
        else:
            for _ in range(self.decimation):
                pd_state = self.robot.get_state()
                dof_pos = np.asarray(pd_state.qpos, dtype=np.float32)[: self.num_actions]
                dof_vel = np.asarray(pd_state.qvel, dtype=np.float32)[: self.num_actions]
                torque = np.asarray(
                    (target_dof_pos - dof_pos) * self.kps - dof_vel * self.kds,
                    dtype=np.float32,
                )
                torque = np.asarray(np.clip(torque, -self.torque_limits, self.torque_limits), dtype=np.float32)
                self.robot.set_action(torque)
                self.robot.step()

        return torque, self.robot.get_state()

    def finish_step(self, action: Float32Array, qpos: Float64Array) -> None:
        self.last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self.last_retarget_qpos = qpos.copy()
        if self._pending_reference_qpos is not None:
            self.last_reference_qpos = self._pending_reference_qpos.copy()
            self._pending_reference_qpos = None

    @staticmethod
    def _retarget_to_qpos(retargeted: object) -> Float64Array:
        return ref_proc.retarget_to_qpos(retargeted)


class ViewerManager:
    def __init__(
        self,
        *,
        robot: Robot,
        viewers: set[str],
        start_robot_viewer: Callable[[str, int, bool, str, int, int], tuple[mp.Process, mp.Array, mp.Value, mp.Event]],
        start_camera_viewer: Callable[[str, int, str, int, int, str], tuple[mp.Process, mp.Array, mp.Value, mp.Event]],
        mocap_viewer_proc: Callable[[list[int], mp.Array, int, mp.Event, mp.Value, int, int], None],
    ) -> None:
        self._robot = robot
        self._viewers = set(viewers)
        self._start_robot_viewer = start_robot_viewer
        self._start_camera_viewer = start_camera_viewer
        self._mocap_viewer_proc = mocap_viewer_proc
        self._sub_viewers: dict[str, tuple[mp.Process, mp.Array, mp.Value, mp.Event]] = {}
        self._mocap_pos_arr: mp.Array | None = None
        self._mocap_n_bones = 0

        xml_path = getattr(self._robot, "xml_path", None)
        model = getattr(self._robot, "model", None)
        win_positions = {"mocap": (50, 50), "retarget": (900, 50), "sim2sim": (1750, 50)}

        if xml_path is None or model is None:
            return

        nq = model.nq
        if "sim2sim" in self._viewers:
            wx, wy = win_positions["sim2sim"]
            self._sub_viewers["sim2sim"] = self._start_robot_viewer(
                xml_path,
                nq,
                False,
                "Sim2Sim",
                wx,
                wy,
            )

        if "retarget" in self._viewers:
            wx, wy = win_positions["retarget"]
            self._sub_viewers["retarget"] = self._start_robot_viewer(
                xml_path,
                nq,
                True,
                "Retarget",
                wx,
                wy,
            )

        if "camera" in self._viewers:
            import mujoco

            camera_name = "d435i_rgb"
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id < 0:
                raise ValueError(
                    f"viewers includes 'camera', but named camera '{camera_name}' was not found in {xml_path}. "
                    "Use a G1 MJCF that defines this camera."
                )
            self._sub_viewers["camera"] = self._start_camera_viewer(
                xml_path,
                nq,
                camera_name,
                640,
                480,
                "D435i RGB",
            )

    def has_viewers(self) -> bool:
        return bool(self._viewers)

    def any_active(self) -> bool:
        for _, _, alive, _ in self._sub_viewers.values():
            if alive.value:
                return True
        return False

    def wait_until_ready(self, timeout_s: float = 10.0) -> None:
        if not self._sub_viewers:
            return
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if all(alive.value for _, _, alive, _ in self._sub_viewers.values()):
                break
            time.sleep(0.1)

    def ensure_mocap_viewer(self, input_provider: object) -> None:
        if "mocap" not in self._viewers or "mocap" in self._sub_viewers:
            return

        bone_names: list[str] | None = getattr(input_provider, "bone_names", None)
        bone_parents: np.ndarray | None = getattr(input_provider, "bone_parents", None)
        if bone_names is None or bone_parents is None or len(bone_names) == 0:
            return

        n_bones = len(bone_names)
        pos_arr = mp.Array("d", n_bones * 3)
        shutdown = mp.Event()
        alive = mp.Value("i", 0)
        proc = mp.Process(
            target=self._mocap_viewer_proc,
            args=(list(bone_parents.astype(int)), pos_arr, n_bones, shutdown, alive, 50, 50),
            daemon=True,
        )
        proc.start()
        self._sub_viewers["mocap"] = (proc, pos_arr, alive, shutdown)
        self._mocap_pos_arr = pos_arr
        self._mocap_n_bones = n_bones

    def write_sim2sim(self, robot: Robot) -> None:
        sim2sim_entry = self._sub_viewers.get("sim2sim")
        if sim2sim_entry is None or not sim2sim_entry[2].value:
            return
        robot_data = getattr(robot, "data", None)
        if robot_data is None:
            return
        sim_qpos = np.asarray(robot_data.qpos, dtype=np.float64)
        arr = sim2sim_entry[1]
        with arr.get_lock():
            arr[:len(sim_qpos)] = sim_qpos.tolist()

    def write_camera(self, robot: Robot) -> None:
        camera_entry = self._sub_viewers.get("camera")
        if camera_entry is None or not camera_entry[2].value:
            return
        robot_data = getattr(robot, "data", None)
        if robot_data is None:
            return
        sim_qpos = np.asarray(robot_data.qpos, dtype=np.float64)
        arr = camera_entry[1]
        with arr.get_lock():
            arr[:len(sim_qpos)] = sim_qpos.tolist()

    def write_retarget(self, retarget_viewer_qpos: Float64Array) -> None:
        retarget_entry = self._sub_viewers.get("retarget")
        if retarget_entry is None or not retarget_entry[2].value:
            return
        arr = retarget_entry[1]
        with arr.get_lock():
            arr[:len(retarget_viewer_qpos)] = retarget_viewer_qpos.tolist()

    def write_mocap(self, input_provider: object, human_frame: dict[str, tuple[Any, Any]]) -> None:
        mocap_entry = self._sub_viewers.get("mocap")
        if mocap_entry is None or not mocap_entry[2].value or self._mocap_pos_arr is None:
            return

        bone_names_attr: list[str] | None = getattr(input_provider, "bone_names", None)
        if bone_names_attr is None:
            return

        n = self._mocap_n_bones
        pos_flat = np.zeros(n * 3, dtype=np.float64)
        for i, bname in enumerate(bone_names_attr):
            if bname in human_frame:
                pos_flat[i * 3:(i + 1) * 3] = human_frame[bname][0]
        with self._mocap_pos_arr.get_lock():
            self._mocap_pos_arr[:n * 3] = pos_flat.tolist()

    def shutdown(self) -> None:
        for _, _, _, shutdown in self._sub_viewers.values():
            shutdown.set()
        for proc, _, _, _ in self._sub_viewers.values():
            proc.join(timeout=3)
            if proc.is_alive():
                proc.terminate()
        self._sub_viewers.clear()
