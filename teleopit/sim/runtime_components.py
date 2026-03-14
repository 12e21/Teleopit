from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from teleopit.bus.topics import TOPIC_ACTION, TOPIC_MIMIC_OBS, TOPIC_ROBOT_STATE
from teleopit.controllers.observation import MjlabObservationBuilder, align_motion_qpos_yaw
from teleopit.controllers.qpos_interpolator import QposInterpolator
from teleopit.interfaces import MessageBus, ObservationBuilder, Recorder, Robot
from teleopit.retargeting.core import extract_mimic_obs

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]


class _SupportsGetTarget(Protocol):
    def get_target_dof_pos(self, raw_action: Float32Array) -> Float32Array: ...


class _SupportsBuild(Protocol):
    def build(self, state: object, mimic_obs: Float32Array, last_action: Float32Array) -> Float32Array: ...


class _SupportsBuildObservation(Protocol):
    def build_observation(
        self,
        state: object,
        history: list[Float32Array],
        action_mimic: Float32Array,
    ) -> Float32Array: ...


class _SupportsAddFrame(Protocol):
    def add_frame(self, data: dict[str, object]) -> None: ...


class _SupportsRecordStep(Protocol):
    def record_step(self, data: dict[str, object]) -> None: ...


@dataclass(frozen=True)
class MotionPreparation:
    qpos: Float64Array
    retarget_viewer_qpos: Float64Array
    mimic_obs: Float32Array


class RuntimePublisher:
    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus

    def publish(self, mimic_obs: Float32Array, action: Float32Array, robot_state: object) -> None:
        self._bus.publish(TOPIC_MIMIC_OBS, mimic_obs)
        self._bus.publish(TOPIC_ACTION, action)
        self._bus.publish(TOPIC_ROBOT_STATE, robot_state)


class RunRecorder:
    def record(
        self,
        recorder: Recorder | None,
        state: object,
        mimic_obs: Float32Array,
        action: Float32Array,
        target_dof_pos: Float32Array,
        torque: Float32Array,
    ) -> None:
        if recorder is None:
            return

        state_qpos = np.asarray(getattr(state, "qpos"), dtype=np.float32)
        state_qvel = np.asarray(getattr(state, "qvel"), dtype=np.float32)
        state_timestamp = np.asarray(float(getattr(state, "timestamp")), dtype=np.float64)

        payload: dict[str, object] = {
            "joint_pos": state_qpos,
            "joint_vel": state_qvel,
            "mimic_obs": mimic_obs.astype(np.float32, copy=False),
            "action": action.astype(np.float32, copy=False),
            "target_dof_pos": target_dof_pos.astype(np.float32, copy=False),
            "torque": torque.astype(np.float32, copy=False),
            "timestamp": state_timestamp,
        }
        add_frame = getattr(recorder, "add_frame", None)
        if callable(add_frame):
            cast(_SupportsAddFrame, cast(object, recorder)).add_frame(payload)
            return
        record_step = getattr(recorder, "record_step", None)
        if callable(record_step):
            cast(_SupportsRecordStep, cast(object, recorder)).record_step(payload)
            return
        raise TypeError("Recorder does not provide add_frame() or record_step()")


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
        qpos_interpolator: QposInterpolator,
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
        self.qpos_interpolator = qpos_interpolator
        self.last_action: Float32Array = np.zeros((self.num_actions,), dtype=np.float32)
        self.last_retarget_qpos: Float64Array | None = None

    def prepare_motion_command(self, retargeted: object, state: object) -> MotionPreparation:
        qpos = self._retarget_to_qpos(retargeted)

        if self.last_retarget_qpos is None and self.qpos_interpolator.duration > 0:
            start_qpos = np.zeros(36, dtype=np.float64)
            start_qpos[0:3] = np.asarray(state.base_pos[:3], dtype=np.float64)
            start_qpos[3:7] = np.asarray(state.quat[:4], dtype=np.float64)
            start_qpos[7:36] = np.asarray(state.qpos[:29], dtype=np.float64)
            self.qpos_interpolator.start(start_qpos)
        qpos = self.qpos_interpolator.apply(qpos)

        mimic_obs = extract_mimic_obs(qpos=qpos, last_qpos=self.last_retarget_qpos, dt=1.0 / self.policy_hz)
        retarget_viewer_qpos = qpos.copy()

        robot_xy = np.asarray(state.base_pos[:2], dtype=np.float64)
        qpos[0:2] = robot_xy
        align_motion_qpos_yaw(np.asarray(state.quat, dtype=np.float32), qpos)

        return MotionPreparation(
            qpos=qpos,
            retarget_viewer_qpos=retarget_viewer_qpos,
            mimic_obs=np.asarray(mimic_obs, dtype=np.float32),
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
        return target

    def build_observation(
        self,
        state: object,
        mimic_obs: Float32Array,
        last_action: Float32Array,
        retarget_qpos: Float64Array,
    ) -> Float32Array:
        if isinstance(self.obs_builder, MjlabObservationBuilder):
            if retarget_qpos.shape[0] < 7 + self.num_actions:
                raise ValueError(
                    f"Retargeted qpos too short for MjlabObservationBuilder: {retarget_qpos.shape[0]} "
                    f"(need >= {7 + self.num_actions})"
                )
            motion_joint_pos = np.asarray(retarget_qpos[7:7 + self.num_actions], dtype=np.float32)
            if self.last_retarget_qpos is None:
                motion_joint_vel = np.zeros((self.num_actions,), dtype=np.float32)
            else:
                prev_joint_pos = np.asarray(
                    self.last_retarget_qpos[7:7 + self.num_actions],
                    dtype=np.float32,
                )
                motion_joint_vel = (motion_joint_pos - prev_joint_pos) * np.float32(self.policy_hz)
            obs = self.obs_builder.build(
                cast(object, state),
                np.asarray(retarget_qpos[:7 + self.num_actions], dtype=np.float32),
                motion_joint_vel,
                last_action,
            )
        elif hasattr(self.obs_builder, "build_observation"):
            obs = cast(_SupportsBuildObservation, self.obs_builder).build_observation(state, [last_action], mimic_obs)
        elif hasattr(self.obs_builder, "build"):
            obs = cast(_SupportsBuild, cast(object, self.obs_builder)).build(state, mimic_obs, last_action)
        else:
            raise TypeError("ObservationBuilder must provide build_observation() or build().")
        return np.asarray(obs, dtype=np.float32)

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

    @staticmethod
    def _retarget_to_qpos(retargeted: object) -> Float64Array:
        if isinstance(retargeted, tuple) and len(retargeted) == 3:
            base_pos = np.asarray(retargeted[0], dtype=np.float64).reshape(-1)
            base_rot = np.asarray(retargeted[1], dtype=np.float64).reshape(-1)
            joint_pos = np.asarray(retargeted[2], dtype=np.float64).reshape(-1)
            qpos = np.concatenate((base_pos, base_rot, joint_pos))
        else:
            qpos = np.array(retargeted, dtype=np.float64).reshape(-1)
        if qpos.shape[0] < 36:
            raise ValueError(f"Retargeted qpos too short: {qpos.shape[0]} (need >= 36)")
        return qpos


class ViewerManager:
    def __init__(
        self,
        *,
        robot: Robot,
        viewers: set[str],
        start_robot_viewer: Callable[[str, int, bool, str, int, int], tuple[mp.Process, mp.Array, mp.Value, mp.Event]],
        bvh_viewer_proc: Callable[[list[int], mp.Array, int, mp.Event, mp.Value, int, int], None],
    ) -> None:
        self._robot = robot
        self._viewers = set(viewers)
        self._start_robot_viewer = start_robot_viewer
        self._bvh_viewer_proc = bvh_viewer_proc
        self._sub_viewers: dict[str, tuple[mp.Process, mp.Array, mp.Value, mp.Event]] = {}
        self._bvh_pos_arr: mp.Array | None = None
        self._bvh_n_bones = 0

        xml_path = getattr(self._robot, "xml_path", None)
        model = getattr(self._robot, "model", None)
        win_positions = {"bvh": (50, 50), "retarget": (900, 50), "sim2sim": (1750, 50)}

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

    def ensure_bvh_viewer(self, input_provider: object) -> None:
        if "bvh" not in self._viewers or "bvh" in self._sub_viewers:
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
            target=self._bvh_viewer_proc,
            args=(list(bone_parents.astype(int)), pos_arr, n_bones, shutdown, alive, 50, 50),
            daemon=True,
        )
        proc.start()
        self._sub_viewers["bvh"] = (proc, pos_arr, alive, shutdown)
        self._bvh_pos_arr = pos_arr
        self._bvh_n_bones = n_bones

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

    def write_retarget(self, retarget_viewer_qpos: Float64Array) -> None:
        retarget_entry = self._sub_viewers.get("retarget")
        if retarget_entry is None or not retarget_entry[2].value:
            return
        arr = retarget_entry[1]
        with arr.get_lock():
            arr[:len(retarget_viewer_qpos)] = retarget_viewer_qpos.tolist()

    def write_bvh(self, input_provider: object, human_frame: dict[str, tuple[Any, Any]]) -> None:
        bvh_entry = self._sub_viewers.get("bvh")
        if bvh_entry is None or not bvh_entry[2].value or self._bvh_pos_arr is None:
            return

        bone_names_attr: list[str] | None = getattr(input_provider, "bone_names", None)
        if bone_names_attr is None:
            return

        n = self._bvh_n_bones
        pos_flat = np.zeros(n * 3, dtype=np.float64)
        for i, bname in enumerate(bone_names_attr):
            if bname in human_frame:
                pos_flat[i * 3:(i + 1) * 3] = human_frame[bname][0]
        with self._bvh_pos_arr.get_lock():
            self._bvh_pos_arr[:n * 3] = pos_flat.tolist()

    def shutdown(self) -> None:
        for _, _, _, shutdown in self._sub_viewers.values():
            shutdown.set()
        for proc, _, _, _ in self._sub_viewers.values():
            proc.join(timeout=3)
            if proc.is_alive():
                proc.terminate()
        self._sub_viewers.clear()
