from __future__ import annotations

import multiprocessing as mp
import time
from typing import Protocol, cast, final

import mujoco
import numpy as np
from numpy.typing import NDArray

from teleopit.controllers.observation import MjlabObservationBuilder
from teleopit.controllers.qpos_interpolator import QposInterpolator
from teleopit.interfaces import Controller, InputProvider, MessageBus, ObservationBuilder, Recorder, Retargeter, Robot
from teleopit.sim.runtime_components import PolicyStepRunner, RunRecorder, RuntimePublisher, ViewerManager

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]


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
        self._realtime: bool = bool(self._try_get_cfg("realtime") or False)

        # Motion command transition smoothing
        transition_dur = float(self._try_get_cfg("transition_duration") or 0.0)
        self._qpos_interpolator = QposInterpolator(transition_dur, self.policy_hz)

        self._viewers: set[str] = set(viewers or set())
        self._step_runner = PolicyStepRunner(
            robot=self.robot,
            controller=cast(object, self.controller),
            obs_builder=self.obs_builder,
            policy_hz=self.policy_hz,
            decimation=self.decimation,
            num_actions=self._num_actions,
            kps=self._kps,
            kds=self._kds,
            torque_limits=self._torque_limits,
            default_dof_pos=self._default_dof_pos,
            qpos_interpolator=self._qpos_interpolator,
        )
        self._publisher = RuntimePublisher(self.bus)
        self._recorder_helper = RunRecorder()
        self._viewer_manager = ViewerManager(
            robot=self.robot,
            viewers=self._viewers,
            start_robot_viewer=_start_robot_viewer,
            bvh_viewer_proc=_bvh_viewer_proc,
        )

    def run(
        self,
        input_provider: InputProvider,
        retargeter: Retargeter,
        num_steps: int,
        recorder: Recorder | None = None,
    ) -> dict[str, float | int]:
        self._viewer_manager.ensure_bvh_viewer(cast(object, input_provider))

        steps_done = 0
        has_viewers = self._viewer_manager.has_viewers()
        needs_pacing = has_viewers or self._realtime
        policy_dt = 1.0 / self.policy_hz
        wall_start = time.monotonic() if needs_pacing else 0.0
        max_steps = num_steps if num_steps > 0 else 2**63

        self._viewer_manager.wait_until_ready(timeout_s=10.0)

        # Frame-rate alignment: BVH fps may differ from policy Hz.
        input_fps: float = float(getattr(input_provider, "fps", self.policy_hz))
        last_bvh_idx = -1
        cached_human_frame: dict | None = None
        cached_retargeted: object = None

        try:
            for _ in range(max_steps):
                if has_viewers and not self._viewer_manager.any_active():
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

                state = self.robot.get_state()
                preparation = self._step_runner.prepare_motion_command(cached_retargeted, state)

                obs = self._build_observation(
                    state=state,
                    mimic_obs=preparation.mimic_obs,
                    last_action=self._step_runner.last_action,
                    retarget_qpos=preparation.qpos,
                )
                policy_obs = self._validate_observation_for_policy(obs)
                action: Float32Array = np.asarray(self.controller.compute_action(policy_obs), dtype=np.float32).reshape(-1)
                if action.shape[0] != self._num_actions:
                    raise ValueError(f"Controller returned {action.shape[0]} actions, expected {self._num_actions}")

                target_dof_pos = self._compute_target_dof_pos(action)
                torque, final_state = self._step_runner.apply_control(target_dof_pos)
                self._publish(preparation.mimic_obs, action, final_state)
                self._record(recorder, final_state, preparation.mimic_obs, action, target_dof_pos, torque)
                self._viewer_manager.write_sim2sim(self.robot)
                self._viewer_manager.write_retarget(preparation.retarget_viewer_qpos)
                if new_bvh_frame and cached_human_frame is not None:
                    self._viewer_manager.write_bvh(cast(object, input_provider), cached_human_frame)

                # Real-time pacing
                if needs_pacing:
                    sim_time = (steps_done + 1) * policy_dt
                    wall_elapsed = time.monotonic() - wall_start
                    sleep_time = sim_time - wall_elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                self._step_runner.finish_step(action, preparation.qpos)
                steps_done += 1
        except KeyboardInterrupt:
            pass
        finally:
            self._viewer_manager.shutdown()

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
        return self._step_runner.compute_target_dof_pos(action)

    def _build_observation(
        self,
        state: object,
        mimic_obs: Float32Array,
        last_action: Float32Array,
        retarget_qpos: Float64Array,
    ) -> Float32Array:
        return self._step_runner.build_observation(state, mimic_obs, last_action, retarget_qpos)

    def _publish(self, mimic_obs: Float32Array, action: Float32Array, robot_state: object) -> None:
        self._publisher.publish(mimic_obs, action, robot_state)

    def _record(
        self,
        recorder: Recorder | None,
        state: object,
        mimic_obs: Float32Array,
        action: Float32Array,
        target_dof_pos: Float32Array,
        torque: Float32Array,
    ) -> None:
        self._recorder_helper.record(recorder, state, mimic_obs, action, target_dof_pos, torque)

    def _retarget_to_qpos(self, retargeted: object) -> Float64Array:
        return self._step_runner._retarget_to_qpos(retargeted)

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

    def _validate_observation_for_policy(self, obs: Float32Array) -> Float32Array:
        return self._step_runner.validate_observation_for_policy(obs)

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
