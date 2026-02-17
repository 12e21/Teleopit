from __future__ import annotations

from typing import Protocol, cast, final

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


@final
class SimulationLoop:
    def __init__(
        self,
        robot: Robot,
        controller: Controller,
        obs_builder: ObservationBuilder,
        bus: MessageBus,
        cfg: object,
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

    def run(
        self,
        input_provider: InputProvider,
        retargeter: Retargeter,
        num_steps: int,
        recorder: Recorder | None = None,
    ) -> dict[str, float | int]:
        steps_done = 0
        for _ in range(num_steps):
            human_frame = input_provider.get_frame()
            retargeted = retargeter.retarget(human_frame)
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

            self._last_action = action
            self._last_retarget_qpos = qpos.copy()
            steps_done += 1

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
