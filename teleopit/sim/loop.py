from __future__ import annotations

import multiprocessing as mp
import time
from typing import Protocol, cast, final
import mujoco
import numpy as np
from numpy.typing import NDArray

from teleopit.bus.topics import TOPIC_ACTION, TOPIC_MIMIC_OBS, TOPIC_ROBOT_STATE
from teleopit.interfaces import Controller, InputProvider, MessageBus, ObservationBuilder, Recorder, Retargeter, Robot
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


class _SupportsGet(Protocol):
    def get(self, key: str) -> object | None: ...


# ---------------------------------------------------------------------------
# Subprocess viewer functions (each runs in its own process with GLFW context)
# ---------------------------------------------------------------------------

def _robot_viewer_proc(
    xml_path: str,
    qpos_arr: mp.Array,
    qpos_len: int,
    shutdown: mp.Event,
    alive: mp.Value,
    foot_z_correction: bool,
    left_foot_name: str,
    right_foot_name: str,
    title: str = "",
    win_x: int = -1,
    win_y: int = -1,
) -> None:
    """Subprocess: robot model viewer — displays qpos with optional foot Z fix.

    Used for both sim2sim (physics result) and retarget (kinematic result).
    """
    import mujoco
    import mujoco.viewer
    import numpy as np
    import os
    import re

    # Set window title via model name and position via GLFW hints
    if title:
        with open(xml_path) as f:
            xml_str = f.read()
        xml_str = re.sub(r'<mujoco\s+model="[^"]*"', f'<mujoco model="{title}"', xml_str)
        os.chdir(os.path.dirname(os.path.abspath(xml_path)))
        model = mujoco.MjModel.from_xml_string(xml_str)
    else:
        model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    left_foot_id = -1
    right_foot_id = -1
    if foot_z_correction:
        left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_foot_name)
        right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_foot_name)

    pelvis_id = -1
    try:
        pelvis_id = model.body("pelvis").id
    except Exception:
        pass

    # Set initial window position via GLFW hints (GLFW 3.4+)
    if win_x >= 0 and win_y >= 0:
        try:
            import glfw
            glfw.init()
            glfw.window_hint(glfw.POSITION_X, win_x)
            glfw.window_hint(glfw.POSITION_Y, win_y)
        except Exception:
            pass

    v = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
    v.cam.distance = 2.0
    alive.value = 1

    try:
        while v.is_running() and not shutdown.is_set():
            with qpos_arr.get_lock():
                qpos = np.array(qpos_arr[:qpos_len], dtype=np.float64)

            data.qpos[:qpos_len] = qpos
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)

            if foot_z_correction and left_foot_id >= 0 and right_foot_id >= 0:
                lowest_z = min(data.xpos[left_foot_id][2], data.xpos[right_foot_id][2])
                if lowest_z < 0.0:
                    data.qpos[2] -= lowest_z
                    mujoco.mj_forward(model, data)

            if pelvis_id >= 0:
                v.cam.lookat[:] = data.xpos[pelvis_id]
            else:
                v.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.8]
            v.sync()
            time.sleep(0.02)
    finally:
        alive.value = 0
        try:
            v.close()
        except Exception:
            pass


def _bvh_viewer_proc(
    parents_list: list[int],
    pos_arr: mp.Array,
    n_bones: int,
    shutdown: mp.Event,
    alive: mp.Value,
    win_x: int = -1,
    win_y: int = -1,
) -> None:
    """Subprocess: BVH skeleton viewer using matplotlib 3D (matches render_sim.py)."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import numpy as np

    parents = parents_list

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.manager.set_window_title("BVH Skeleton")

    # Set window position via Tk geometry
    if win_x >= 0 and win_y >= 0:
        try:
            fig.canvas.manager.window.wm_geometry(f"+{win_x}+{win_y}")
        except Exception:
            pass

    plt.ion()
    plt.show(block=False)
    alive.value = 1

    try:
        while not shutdown.is_set():
            # Check if window was closed
            if not plt.fignum_exists(fig.number):
                break

            with pos_arr.get_lock():
                pos = np.array(pos_arr[:n_bones * 3], dtype=np.float64).reshape(n_bones, 3)

            ax.cla()
            root = pos[0]
            ax.set_xlim(root[0] - 1.0, root[0] + 1.0)
            ax.set_ylim(root[1] - 1.0, root[1] + 1.0)
            ax.set_zlim(0.0, 2.0)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("BVH Skeleton")
            ax.view_init(elev=20, azim=135)

            for j in range(n_bones):
                p = parents[j]
                if p < 0:
                    continue
                ax.plot(
                    [pos[p, 0], pos[j, 0]],
                    [pos[p, 1], pos[j, 1]],
                    [pos[p, 2], pos[j, 2]],
                    "b-", linewidth=2,
                )
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="red", s=15, depthshade=True)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.03)
    finally:
        alive.value = 0
        try:
            plt.close(fig)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helper: create a robot-model viewer subprocess
# ---------------------------------------------------------------------------

def _start_robot_viewer(
    xml_path: str, nq: int, foot_z_correction: bool,
    title: str = "", win_x: int = -1, win_y: int = -1,
) -> tuple[mp.Process, mp.Array, mp.Value, mp.Event]:
    """Launch a subprocess viewer for a robot model.

    Returns (process, qpos_shared_array, alive_flag, shutdown_event).
    """
    arr = mp.Array("d", nq)
    shutdown = mp.Event()
    alive = mp.Value("i", 0)
    proc = mp.Process(
        target=_robot_viewer_proc,
        args=(xml_path, arr, nq, shutdown, alive,
              foot_z_correction, "left_ankle_roll_link", "right_ankle_roll_link",
              title, win_x, win_y),
        daemon=True,
    )
    proc.start()
    return proc, arr, alive, shutdown


@final
class SimulationLoop:
    def __init__(
        self,
        robot: Robot,
        controller: Controller,
        obs_builder: ObservationBuilder,
        bus: MessageBus,
        cfg: object,
        viewers: set[str] | None = None,
        viewer: bool = False,
    ) -> None:
        self.robot: Robot = robot
        self.controller: Controller = controller
        self.obs_builder: ObservationBuilder = obs_builder
        self.bus: MessageBus = bus
        self.cfg: object = cfg

        self.policy_hz: float = self._to_float(self._get_cfg("policy_hz", "sim.policy_hz", "control.policy_hz", "policy_frequency"))
        self.pd_hz: float = self._to_float(self._get_cfg("pd_hz", "sim.pd_hz", "control.pd_hz", "pd_frequency"))
        if self.policy_hz <= 0.0 or self.pd_hz <= 0.0:
            raise ValueError("policy_hz and pd_hz must be positive")
        ratio = self.pd_hz / self.policy_hz
        if ratio < 1.0:
            raise ValueError("pd_hz must be >= policy_hz")

        self.decimation: int = int(round(ratio))
        if abs(ratio - self.decimation) > 1e-6:
            raise ValueError(f"pd_hz/policy_hz must be an integer ratio, got {ratio}")

        self._num_actions: int = int(getattr(self.robot, "num_actions"))
        self._kps: Float32Array = np.asarray(getattr(self.robot, "kps"), dtype=np.float32)
        self._kds: Float32Array = np.asarray(getattr(self.robot, "kds"), dtype=np.float32)
        self._torque_limits: Float32Array = np.asarray(getattr(self.robot, "torque_limits"), dtype=np.float32)
        self._default_dof_pos: Float32Array = np.asarray(getattr(self.robot, "default_dof_pos"), dtype=np.float32)

        self._last_action: Float32Array = np.zeros((self._num_actions,), dtype=np.float32)
        self._last_retarget_qpos: Float64Array | None = None

        # Resolve viewer set (support legacy viewer=bool kwarg)
        if viewers is not None:
            self._viewers: set[str] = set(viewers)
        elif viewer:
            self._viewers = {"sim2sim"}
        else:
            self._viewers = set()

        # All viewers run in subprocesses to avoid GLFW/GLX single-context limit.
        # Each entry: (process, shared_array, alive_flag, shutdown_event)
        self._sub_viewers: dict[str, tuple[mp.Process, mp.Array, mp.Value, mp.Event]] = {}

        xml_path = getattr(self.robot, "xml_path", None)
        model = getattr(self.robot, "model", None)

        # Window layout: spread horizontally, 850px apart
        # Order: BVH (left) | Retarget (middle) | Sim2Sim (right)
        _win_positions = {"bvh": (50, 50), "retarget": (900, 50), "sim2sim": (1750, 50)}

        if xml_path is not None and model is not None:
            nq = model.nq
            # --- sim2sim viewer (shows physics result, no foot Z fix) ---
            if "sim2sim" in self._viewers:
                wx, wy = _win_positions["sim2sim"]
                proc, arr, alive, shutdown = _start_robot_viewer(
                    xml_path, nq, foot_z_correction=False,
                    title="Sim2Sim", win_x=wx, win_y=wy,
                )
                self._sub_viewers["sim2sim"] = (proc, arr, alive, shutdown)

            # --- retarget viewer (shows kinematic target, with foot Z fix) ---
            if "retarget" in self._viewers:
                wx, wy = _win_positions["retarget"]
                proc, arr, alive, shutdown = _start_robot_viewer(
                    xml_path, nq, foot_z_correction=True,
                    title="Retarget", win_x=wx, win_y=wy,
                )
                self._sub_viewers["retarget"] = (proc, arr, alive, shutdown)

        # BVH viewer subprocess (created lazily in run() — needs bone topology)
        self._bvh_pos_arr: mp.Array | None = None
        self._bvh_n_bones: int = 0

    def _any_viewer_active(self) -> bool:
        """Return True if at least one viewer window is still open."""
        for _, _, alive, _ in self._sub_viewers.values():
            if alive.value:
                return True
        return False

    def _has_any_viewer(self) -> bool:
        """Return True if any viewer was requested."""
        return bool(self._viewers)

    def run(
        self,
        input_provider: InputProvider,
        retargeter: Retargeter,
        num_steps: int,
        recorder: Recorder | None = None,
    ) -> dict[str, float | int]:
        # --- Lazily create BVH viewer subprocess (matplotlib 3D) ---
        if "bvh" in self._viewers and "bvh" not in self._sub_viewers:
            bone_names: list[str] | None = getattr(input_provider, "bone_names", None)
            bone_parents: np.ndarray | None = getattr(input_provider, "bone_parents", None)
            if bone_names is not None and bone_parents is not None and len(bone_names) > 0:
                n_bones = len(bone_names)
                pos_arr = mp.Array("d", n_bones * 3)
                shutdown = mp.Event()
                alive = mp.Value("i", 0)
                bvh_wx, bvh_wy = 50, 50  # leftmost position
                proc = mp.Process(
                    target=_bvh_viewer_proc,
                    args=(list(bone_parents.astype(int)), pos_arr, n_bones, shutdown, alive,
                          bvh_wx, bvh_wy),
                    daemon=True,
                )
                proc.start()
                self._sub_viewers["bvh"] = (proc, pos_arr, alive, shutdown)
                self._bvh_pos_arr = pos_arr
                self._bvh_n_bones = n_bones

        steps_done = 0
        has_viewers = self._has_any_viewer()
        policy_dt = 1.0 / self.policy_hz
        wall_start = time.monotonic() if has_viewers else 0.0

        # Wait for subprocess viewers to signal alive (up to 10s)
        if self._sub_viewers:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if all(alive.value for _, _, alive, _ in self._sub_viewers.values()):
                    break
                time.sleep(0.1)

        # Frame-rate alignment: BVH fps may differ from policy Hz.
        input_fps: float = float(getattr(input_provider, "fps", self.policy_hz))
        last_bvh_idx = -1
        cached_human_frame: dict | None = None
        cached_retargeted: object = None

        try:
            for _ in range(num_steps):
                if has_viewers and not self._any_viewer_active():
                    break

                policy_time = steps_done * policy_dt
                bvh_idx = int(policy_time * input_fps)

                new_bvh_frame = bvh_idx != last_bvh_idx
                if new_bvh_frame:
                    if not input_provider.is_available():
                        break
                    cached_human_frame = input_provider.get_frame()
                    cached_retargeted = retargeter.retarget(cached_human_frame)
                    last_bvh_idx = bvh_idx

                retargeted = cached_retargeted
                qpos = self._retarget_to_qpos(retargeted)
                mimic_obs = extract_mimic_obs(qpos=qpos, last_qpos=self._last_retarget_qpos, dt=1.0 / self.policy_hz)

                state = self.robot.get_state()
                obs = self._build_observation(state=state, mimic_obs=mimic_obs, last_action=self._last_action)
                policy_obs = self._adapt_observation_for_policy(obs)
                action: Float32Array = np.asarray(self.controller.compute_action(policy_obs), dtype=np.float32).reshape(-1)
                if action.shape[0] != self._num_actions:
                    raise ValueError(f"Controller returned {action.shape[0]} actions, expected {self._num_actions}")

                target_dof_pos = self._compute_target_dof_pos(action)

                torque: Float32Array = np.zeros((self._num_actions,), dtype=np.float32)
                for _ in range(self.decimation):
                    pd_state = self.robot.get_state()
                    dof_pos = np.asarray(pd_state.qpos, dtype=np.float32)[: self._num_actions]
                    dof_vel = np.asarray(pd_state.qvel, dtype=np.float32)[: self._num_actions]
                    torque = np.asarray((target_dof_pos - dof_pos) * self._kps - dof_vel * self._kds, dtype=np.float32)
                    torque = np.asarray(np.clip(torque, -self._torque_limits, self._torque_limits), dtype=np.float32)
                    self.robot.set_action(torque)
                    self.robot.step()

                final_state = self.robot.get_state()
                self._publish(mimic_obs, action, final_state)
                self._record(recorder, final_state, mimic_obs, action, target_dof_pos, torque)

                # --- Write sim2sim qpos (post-physics) to shared memory ---
                sim2sim_entry = self._sub_viewers.get("sim2sim")
                if sim2sim_entry is not None and sim2sim_entry[2].value:
                    robot_data = getattr(self.robot, "data", None)
                    if robot_data is not None:
                        sim_qpos = np.asarray(robot_data.qpos, dtype=np.float64)
                        arr = sim2sim_entry[1]
                        with arr.get_lock():
                            arr[:len(sim_qpos)] = sim_qpos.tolist()

                # --- Write retarget qpos to shared memory ---
                retarget_entry = self._sub_viewers.get("retarget")
                if retarget_entry is not None and retarget_entry[2].value:
                    arr = retarget_entry[1]
                    with arr.get_lock():
                        arr[:len(qpos)] = qpos.tolist()

                # --- Write BVH positions to shared memory ---
                if new_bvh_frame and cached_human_frame is not None:
                    bvh_entry = self._sub_viewers.get("bvh")
                    if bvh_entry is not None and bvh_entry[2].value and self._bvh_pos_arr is not None:
                        bone_names_attr: list[str] | None = getattr(input_provider, "bone_names", None)
                        if bone_names_attr is not None:
                            n = self._bvh_n_bones
                            pos_flat = np.zeros(n * 3, dtype=np.float64)
                            for i, bname in enumerate(bone_names_attr):
                                if bname in cached_human_frame:
                                    pos_flat[i * 3:(i + 1) * 3] = cached_human_frame[bname][0]
                            with self._bvh_pos_arr.get_lock():
                                self._bvh_pos_arr[:n * 3] = pos_flat.tolist()

                # Real-time pacing
                if has_viewers:
                    sim_time = (steps_done + 1) * policy_dt
                    wall_elapsed = time.monotonic() - wall_start
                    sleep_time = sim_time - wall_elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                self._last_action = action
                self._last_retarget_qpos = qpos.copy()
                steps_done += 1
        finally:
            # Shutdown all subprocess viewers
            for name, (proc, _, _, shutdown) in self._sub_viewers.items():
                shutdown.set()
            for name, (proc, _, _, _) in self._sub_viewers.items():
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.terminate()
            self._sub_viewers.clear()

        final_state = self.robot.get_state()
        return {
            "steps": steps_done,
            "root_height": self._get_root_height(final_state),
            "policy_hz": self.policy_hz,
            "pd_hz": self.pd_hz,
            "decimation": self.decimation,
        }

    def run_headless(
        self,
        input_provider: InputProvider,
        retargeter: Retargeter,
        num_steps: int,
        recorder: Recorder | None = None,
    ) -> dict[str, float | int]:
        return self.run(input_provider=input_provider, retargeter=retargeter, num_steps=num_steps, recorder=recorder)

    def _compute_target_dof_pos(self, action: Float32Array) -> Float32Array:
        get_target = getattr(self.controller, "get_target_dof_pos", None)
        if callable(get_target):
            target = np.asarray(cast(_SupportsGetTarget, cast(object, self.controller)).get_target_dof_pos(action), dtype=np.float32).reshape(-1)
        else:
            target = action + self._default_dof_pos

        if target.shape[0] != self._num_actions:
            raise ValueError(f"Target dof pos has {target.shape[0]} entries, expected {self._num_actions}")
        return target

    def _build_observation(self, state: object, mimic_obs: Float32Array, last_action: Float32Array) -> Float32Array:
        if hasattr(self.obs_builder, "build"):
            obs = cast(_SupportsBuild, cast(object, self.obs_builder)).build(state, mimic_obs, last_action)
        else:
            obs = cast(_SupportsBuildObservation, self.obs_builder).build_observation(state, [last_action], mimic_obs)
        return np.asarray(obs, dtype=np.float32)

    def _publish(self, mimic_obs: Float32Array, action: Float32Array, robot_state: object) -> None:
        self.bus.publish(TOPIC_MIMIC_OBS, mimic_obs)
        self.bus.publish(TOPIC_ACTION, action)
        self.bus.publish(TOPIC_ROBOT_STATE, robot_state)

    def _record(
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

    def _retarget_to_qpos(self, retargeted: object) -> Float64Array:
        if isinstance(retargeted, tuple) and len(retargeted) == 3:
            base_pos = np.asarray(retargeted[0], dtype=np.float64).reshape(-1)
            base_rot = np.asarray(retargeted[1], dtype=np.float64).reshape(-1)
            joint_pos = np.asarray(retargeted[2], dtype=np.float64).reshape(-1)
            qpos = np.concatenate((base_pos, base_rot, joint_pos))
        else:
            qpos = np.asarray(retargeted, dtype=np.float64).reshape(-1)
        if qpos.shape[0] < 36:
            raise ValueError(f"Retargeted qpos too short: {qpos.shape[0]} (need >= 36)")
        return qpos

    def _get_cfg(self, *keys: str) -> object:
        for key in keys:
            value = self._try_get_cfg(key)
            if value is not None:
                return value
        raise KeyError(f"Missing required config value. Tried keys: {keys}")

    def _try_get_cfg(self, key: str) -> object | None:
        if "." in key:
            cur: object | None = self.cfg
            for part in key.split("."):
                cur = self._get_single(cur, part)
                if cur is None:
                    return None
            return cur
        return self._get_single(self.cfg, key)

    @staticmethod
    def _get_single(obj: object | None, key: str) -> object | None:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return cast(dict[str, object], obj).get(key)
        if hasattr(obj, "get"):
            try:
                value = cast(_SupportsGet, cast(object, obj)).get(key)
                if value is not None:
                    return value
            except Exception:
                pass
        return getattr(obj, key, None)

    @staticmethod
    def _to_float(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Expected numeric config value, got {value}")
        return float(value)

    def _adapt_observation_for_policy(self, obs: Float32Array) -> Float32Array:
        expected_raw = getattr(self.controller, "_expected_obs_dim", None)
        if not isinstance(expected_raw, int) or expected_raw <= 0:
            return obs
        if obs.shape[0] == expected_raw:
            return obs
        if obs.shape[0] > expected_raw:
            return obs[:expected_raw]
        pad = np.zeros((expected_raw - obs.shape[0],), dtype=np.float32)
        return np.concatenate((obs, pad), dtype=np.float32)

    def _get_root_height(self, state: object) -> float:
        robot_data = getattr(self.robot, "data", None)
        if robot_data is not None:
            qpos = np.asarray(getattr(robot_data, "qpos"), dtype=np.float64)
            if qpos.shape[0] >= 3:
                return float(qpos[2])
        qpos_state = np.asarray(getattr(state, "qpos"), dtype=np.float64)
        if qpos_state.shape[0] >= 3:
            return float(qpos_state[2])
        raise ValueError("Unable to infer root height from robot state")
